"""Persistent, isolated OmniVoice speech adapter; the engine never imports its Transformers runtime.

Validated male/female design profiles condition Telugu speech; unresolved automatic mode remains explicit.
Only the subprocess loads OmniVoice. Raw PCM takes remain local and are watermarked after timing changes.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

import numpy as np

from .base import Cancelled
from ..timing.stretch import wsola

MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
SOURCE_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
VOICE_POLICY = "telugu-gender-profiles-v1"
GENERATION_SEED = 20261010
PROTOCOL_VERSION = 2
VOICE_PROFILES = {"automatic": None, "male": "male", "female": "female"}
SAMPLE_RATE = 24_000
MAX_AUDIO_SECONDS = 60.0


class OmniVoiceError(RuntimeError):
    """A failed local worker or invalid configuration; never silently use another voice/model."""


@dataclass(frozen=True, slots=True)
class OmniVoiceVoice:
    identity: str
    profile: str = "automatic"


@dataclass(slots=True)
class WaveformTake:
    samples: np.ndarray  # unwatermarked natural speech; watermark only after its final rate is known
    sample_rate: int = SAMPLE_RATE
    capped: bool = False
    over_budget: bool = False  # complete natural audio longer than guidance; this is not a cutoff

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def _audio(value: object) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or float(np.max(np.abs(x))) < 1e-6:
        raise OmniVoiceError("OmniVoice returned empty, nonfinite or silent audio.")
    if len(x) > int((MAX_AUDIO_SECONDS + 1) * SAMPLE_RATE):
        raise OmniVoiceError("OmniVoice returned audio beyond the bounded 60-second generation limit.")
    return x


class OmniVoiceTTS:
    """One lazy reusable subprocess, serialized requests, and interruptible inference.

    Profiles use the globally fixed audition seed, without reference cloning or a promise of stable identity.
    ``seed_reference`` is an explicit experiment-only override; production callers
    omit it. No source-English audio is used. Constructor work is model-free.
    ``cancel`` signals the process without acquiring the request lock; the request thread reaps it.
    """

    native_voice = True
    sample_rate = SAMPLE_RATE
    model_revision = MODEL_REVISION
    source_revision = SOURCE_REVISION
    voice_policy = VOICE_POLICY

    def __init__(self, model_dir: Path, runtime_python: Path | None = None, voice_dir: Path | None = None,
                 device: str = "mps", seed_reference: tuple[Path, str] | None = None,
                 reference_sha256: str | None = None, request_timeout: float = 120.0,
                 load_timeout: float = 180.0, stop_grace: float = 1.0) -> None:
        if device not in ("mps", "cuda", "cpu"):
            raise ValueError(f"Unsupported OmniVoice device: {device}")
        self.model_dir = Path(model_dir).expanduser().resolve()
        config = self.model_dir.parent.parent
        self.runtime_python = Path(runtime_python or config / "runtimes/omnivoice/bin/python").expanduser().absolute()
        self.voice_dir = Path(voice_dir or config / "voices/omnivoice").expanduser().resolve()
        self.device, self._device = device, device
        self.steps, self.dtype = 32, "float16" if device != "cpu" else "float32"
        self.request_timeout, self.load_timeout, self.stop_grace = request_timeout, load_timeout, stop_grace
        if min(request_timeout, load_timeout, stop_grace) <= 0:
            raise ValueError("Worker timeouts must be positive.")
        self.mode = "reference-experiment" if seed_reference is not None else "automatic"
        self.seed = GENERATION_SEED
        self.reference_path = Path(seed_reference[0]).expanduser().resolve() if seed_reference else None
        self.reference_text = seed_reference[1] if seed_reference else ""
        self._expected_reference = reference_sha256
        self._reference_hash = None
        if self.reference_path is not None and self.reference_path.is_file():
            self._reference_hash = hashlib.sha256(self.reference_path.read_bytes()).hexdigest()
        self._identity = {
            "model": MODEL_REVISION, "source": SOURCE_REVISION, "steps": self.steps,
            "precision": self.dtype, "device": device, "voice_policy": VOICE_POLICY,
            "mode": self.mode, "seed": self.seed, "seed_policy": "fixed-before-every-generation",
            "profiles": dict(VOICE_PROFILES),
            "reference_audio_sha256": self._reference_hash,
            "reference_text_sha256": hashlib.sha256(self.reference_text.encode()).hexdigest(),
            "preprocess_prompt": False, "normalize_text": False, "duration": "natural",
            "watermark": "PerthImplicitWatermarker", "watermark_version": self._watermark_version(),
            "protocol": PROTOCOL_VERSION,
        }
        self._voices = {profile: OmniVoiceVoice(hashlib.sha256(json.dumps(
            self.profile_cache_identity(profile), sort_keys=True).encode()).hexdigest(), profile)
            for profile in (VOICE_PROFILES if self.mode == "automatic" else ("automatic",))}
        self._voice = self._voices["automatic"]
        self._lock, self._state_lock = threading.Lock(), threading.Lock()
        self._process: subprocess.Popen | None = None
        self._temp: tempfile.TemporaryDirectory | None = None
        self._responses: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr = None
        self._counter, self._cancel_epoch = 0, 0
        self._cancel_event: threading.Event | None = None
        self._prepared = False
        self._watermarker = None
        self._verified_signature = None
        atexit.register(self.release)

    @staticmethod
    def _watermark_version() -> str:
        try:
            return importlib.metadata.version("resemble-perth")
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    @property
    def cache_identity(self) -> dict:
        return {**self._identity, "profiles": dict(VOICE_PROFILES)}

    @staticmethod
    def _profile(name: str) -> str:
        profile = "automatic" if name in ("native", "native-telugu") else name
        if profile not in VOICE_PROFILES:
            raise OmniVoiceError("Unknown OmniVoice profile; choose male, female or automatic.")
        return profile

    def profile_cache_identity(self, name: str) -> dict:
        """Cheap per-profile provenance for voice keys and calibration; no inference or source audio."""
        profile = self._profile(name)
        if self.mode == "reference-experiment" and profile != "automatic":
            raise OmniVoiceError("Gender profiles cannot use experimental reference conditioning.")
        return {**self.cache_identity, "profile": profile, "instruct": VOICE_PROFILES[profile]}

    def missing(self) -> list[str]:
        missing = [str(p) for p in (self.runtime_python, self.model_dir / "model.safetensors",
                   self.model_dir / "config.json", self.model_dir / "tokenizer.json",
                   self.model_dir / "audio_tokenizer/model.safetensors",
                   self.model_dir / "audio_tokenizer/config.json") if not p.is_file()]
        if self.mode == "reference-experiment" and (self.reference_path is None or not self.reference_path.is_file()
                                                   or not self.reference_text.strip()):
            missing.append("experimental reference audio and exact Telugu transcript")
        if self.mode == "reference-experiment" and self._expected_reference is not None \
                and self._reference_hash != self._expected_reference:
            missing.append("Voice B reference checksum mismatch")
        return missing

    def _verify_models(self) -> None:
        """Hash pinned weights once per unchanged file signature, never in the cheap cache identity path."""
        from ..models import _spec, load_lock, verify

        row = next((m for m in load_lock()["models"] if m["id"] == "omnivoice"), None)
        if row is None or row.get("revision") != MODEL_REVISION:
            raise OmniVoiceError("The OmniVoice model manifest does not match this adapter's pinned revision.")
        try:
            paths = [(self.model_dir / f["path"], _spec(f)) for f in row["files"]]
            signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p, _ in paths)
            if signature == self._verified_signature:
                return
            invalid = [p.name for p, spec in paths if not verify(p, spec)]
        except (OSError, ValueError, KeyError) as exc:
            raise OmniVoiceError("The pinned OmniVoice model files are missing or unreadable.") from exc
        if invalid:
            raise OmniVoiceError("Pinned OmniVoice model checksum mismatch: " + ", ".join(invalid))
        self._verified_signature = signature

    def _command(self, work_dir: Path) -> list[str]:
        # Isolated mode excludes this script's directory (which also contains our adapter named omnivoice.py)
        # and user site/config, so imports resolve to the pinned runtime's upstream package.
        command = [str(self.runtime_python), "-I", "-u", str(Path(__file__).with_name("_omnivoice_worker.py")),
                   "--work-dir", str(work_dir)]
        if sys.platform == "darwin":
            # Audio inference must have no network access, independent of Hugging Face offline settings.
            sandbox = Path("/usr/bin/sandbox-exec")
            if not sandbox.is_file():
                raise OmniVoiceError("The local network-denied OmniVoice worker requires sandbox-exec.")
            command = [str(sandbox), "-p", "(version 1)(allow default)(deny network*)", *command]
        return command

    def _start(self) -> None:
        absent = self.missing()
        if absent:
            raise OmniVoiceError("OmniVoice is not provisioned: " + "; ".join(absent))
        # Check again immediately before loading: replacing the provisioned voice cannot change a live cache identity.
        if self.reference_path is not None and hashlib.sha256(self.reference_path.read_bytes()).hexdigest() != self._reference_hash:
            raise OmniVoiceError("Voice B reference changed after the adapter was configured.")
        self._verify_models()
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="maata-omnivoice-")
        work = Path(self._temp.name)
        self._stderr = (work / "stderr.log").open("wb")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
               "HF_HOME": str(work / "hf-offline")}
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        try:
            proc = subprocess.Popen(self._command(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=self._stderr, env=env, cwd=work, start_new_session=True)
        except Exception:
            self._dispose()
            raise
        with self._state_lock:
            self._process = proc
        responses = self._responses = queue.Queue()

        def read() -> None:
            try:
                while True:
                    line = proc.stdout.readline(65537)
                    if not line:
                        break
                    if len(line) > 65536 or not line.endswith(b"\n"):
                        responses.put(OmniVoiceError("Oversized or incomplete OmniVoice protocol response."))
                        return
                    responses.put(line)
            except (OSError, ValueError) as exc:
                responses.put(exc)
            finally:
                responses.put(None)

        self._reader = threading.Thread(target=read, name="omnivoice-protocol", daemon=True)
        self._reader.start()
        self._prepared = False

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, sig)
                else:
                    proc.terminate() if sig == signal.SIGTERM else proc.kill()
            except ProcessLookupError:
                pass
            except PermissionError:
                # A concurrent reaper can remove the last member between poll() and killpg() on macOS,
                # which sometimes reports EPERM rather than ESRCH. Popen's direct signal path rechecks exit.
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    pass

    def cancel(self) -> None:
        """Nonblocking cancellation, including a request currently waiting for the request lock."""
        with self._state_lock:
            self._cancel_epoch += 1
            proc = self._process
        if proc is not None:
            self._signal(proc, signal.SIGTERM)
            # Also reap an idle worker: cancellation may happen between model requests, with no request thread
            # waiting in _dispose. Keep the event loop nonblocking; the process lock in Popen serializes waiters.
            threading.Thread(target=self._reap, args=(proc,), name="omnivoice-reap", daemon=True).start()

    def bind_cancel_event(self, event: threading.Event) -> None:
        """Bind the admitted render's lifetime; a queued call must not restart inference after its job paused."""
        self._cancel_event = event

    def _cancelled(self, epoch: int) -> bool:
        return epoch != self._cancel_epoch or (self._cancel_event is not None and self._cancel_event.is_set())

    def _reap(self, proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=self.stop_grace)
        except subprocess.TimeoutExpired:
            self._signal(proc, signal.SIGKILL)
            proc.wait(timeout=self.stop_grace + 1)

    def _exit_detail(self) -> str:
        """Keep only bounded dependency/loader errors, never upstream transcript or prompt log messages."""
        if self._temp is None:
            return ""
        try:
            with (Path(self._temp.name) / "stderr.log").open("rb") as source:
                source.seek(max(0, source.seek(0, os.SEEK_END) - 4096))
                tail = source.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        prefixes = ("ModuleNotFoundError:", "ImportError:", "OSError:", "RuntimeError:")
        lines = [line.strip() for line in tail.splitlines() if line.startswith(prefixes)]
        return " " + lines[-1][:400] if lines else ""

    def _dispose(self) -> None:
        with self._state_lock:
            proc, self._process = self._process, None
        if proc is not None:
            self._signal(proc, signal.SIGTERM)
            self._reap(proc)
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    stream.close()
        if self._reader is not None:
            self._reader.join(timeout=self.stop_grace + 1)
            self._reader = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._prepared = False

    def release(self) -> None:
        self.cancel()
        with self._lock:
            self._dispose()

    def _request(self, operation: str, **payload) -> dict:
        epoch = self._cancel_epoch
        with self._lock:
            if self._cancelled(epoch):
                raise Cancelled("OmniVoice request cancelled before starting.")
            try:
                if self._process is None or self._process.poll() is not None:
                    self._dispose()
                    self._start()
                if self._cancelled(epoch):
                    raise Cancelled("OmniVoice request cancelled during startup.")
                self._counter += 1
                rid = self._counter
                message = {"version": PROTOCOL_VERSION, "id": rid, "op": operation,
                           "config": {"model_dir": str(self.model_dir), "voice_dir": str(self.voice_dir),
                                      "device": self.device, "steps": self.steps, "dtype": self.dtype,
                                      "mode": self.mode, "seed": self.seed,
                                      "reference_path": str(self.reference_path) if self.reference_path is not None else None,
                                      "reference_text": self.reference_text,
                                      "reference_sha256": self._reference_hash, "voice_identity": self._voice.identity},
                           **payload}
                deadline = time.monotonic() + (self.request_timeout if self._prepared else self.load_timeout)
                self._process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                self._process.stdin.flush()
                while True:
                    if self._cancelled(epoch):
                        raise Cancelled("OmniVoice synthesis cancelled.")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OmniVoiceError("OmniVoice worker exceeded its inference timeout.")
                    try:
                        raw = self._responses.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        continue
                    if raw is None:
                        raise OmniVoiceError("OmniVoice worker exited before completing its response." + self._exit_detail())
                    if isinstance(raw, Exception):
                        raise OmniVoiceError("OmniVoice worker protocol failed.") from raw
                    try:
                        reply = json.loads(raw)
                    except (ValueError, UnicodeError) as exc:
                        raise OmniVoiceError("OmniVoice returned invalid protocol JSON.") from exc
                    if not isinstance(reply, dict) or reply.get("id") != rid or reply.get("version") != PROTOCOL_VERSION:
                        raise OmniVoiceError("OmniVoice response identity or protocol version did not match.")
                    if not reply.get("ok"):
                        raise OmniVoiceError(str(reply.get("error") or "OmniVoice local inference failed."))
                    if reply.get("sample_rate") != SAMPLE_RATE:
                        raise OmniVoiceError("OmniVoice sample rate does not match the pinned codec.")
                    self._prepared = True
                    if operation == "generate":
                        if any(reply.get(k) != payload[k] for k in ("profile", "instruct", "voice_identity")):
                            raise OmniVoiceError("OmniVoice response voice profile did not match its request.")
                        # Output is always our exact request's private file, never a path supplied by the worker.
                        path = Path(self._temp.name) / f"audio-{rid}.npy"
                        if path.is_symlink() or not path.is_file() or path.stat().st_size > SAMPLE_RATE * 61 * 4 + 4096:
                            raise OmniVoiceError("OmniVoice output file is missing or exceeds the audio bound.")
                        try:
                            reply["samples"] = _audio(np.load(path, allow_pickle=False))
                        finally:
                            path.unlink(missing_ok=True)
                    return reply
            except BaseException:
                self._dispose()
                if self._cancelled(epoch):
                    raise Cancelled("OmniVoice synthesis cancelled.") from None
                raise

    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> OmniVoiceVoice:
        # The input source audio is deliberately not sent to the worker or used to condition Telugu pronunciation.
        return self.preset_voice("native-telugu")

    def preset_voice(self, name: str) -> OmniVoiceVoice:
        profile = self._profile(name)
        self.profile_cache_identity(profile)  # reject profile/reference mixing before starting any worker
        if not self._prepared:
            self._request("prepare")
        return self._voices[profile]

    def synthesize_mel(self, text: str, voice: OmniVoiceVoice, language: str = "te",
                       max_seconds: float | None = None) -> WaveformTake:
        """Compatibility take API: already decoded PCM, with natural duration for existing calibration/planning."""
        if (language != "te" or not isinstance(voice, OmniVoiceVoice)
                or voice != self._voices.get(voice.profile)):
            raise OmniVoiceError("OmniVoice requires Telugu and its configured automatic/reference mode.")
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise OmniVoiceError("OmniVoice text is empty or exceeds its bounded input size.")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite.")
        result = self._request("generate", text=text, max_seconds=max_seconds,
                               profile=voice.profile, instruct=VOICE_PROFILES[voice.profile],
                               voice_identity=voice.identity)
        samples = result["samples"]
        over_budget = max_seconds is not None and len(samples) / SAMPLE_RATE >= max_seconds
        # Completed natural fixed-frame output did not hit an autoregressive stop-token cap. Existing QA interprets
        # capped as a cutoff and excludes it from pace learning, so retain complete slow audio for the planner.
        return WaveformTake(samples, over_budget=over_budget)

    def vocode(self, take: WaveformTake, rate: float = 1.0) -> np.ndarray:
        if not isinstance(take, WaveformTake) or take.sample_rate != SAMPLE_RATE:
            raise OmniVoiceError("Invalid OmniVoice waveform take.")
        if not math.isfinite(rate) or not 1.0 <= rate <= 1.5:
            raise ValueError("OmniVoice timing rate must be between 1 and 1.5.")
        raw = _audio(take.samples)
        adjusted = wsola(raw, rate, SAMPLE_RATE) if abs(rate - 1.0) > 1e-3 else raw
        if self._watermarker is None:
            import perth
            self._watermarker = perth.PerthImplicitWatermarker()
        return _audio(self._watermarker.apply_watermark(adjusted, sample_rate=SAMPLE_RATE))

    def synthesize(self, text: str, voice: OmniVoiceVoice, language: str = "te",
                   max_seconds: float | None = None) -> np.ndarray:
        return self.vocode(self.synthesize_mel(text, voice, language, max_seconds))

    def pack_take(self, take: WaveformTake) -> dict[str, np.ndarray]:
        return {"omnivoice_raw": _audio(take.samples), "omnivoice_capped": np.bool_(take.capped),
                "omnivoice_over_budget": np.bool_(take.over_budget),
                "omnivoice_sample_rate": np.int64(take.sample_rate)}

    def unpack_take(self, arrays: dict[str, np.ndarray]) -> WaveformTake:
        if int(arrays["omnivoice_sample_rate"]) != SAMPLE_RATE:
            raise OmniVoiceError("Stored OmniVoice take uses a different sample rate.")
        return WaveformTake(_audio(arrays["omnivoice_raw"]), capped=bool(arrays["omnivoice_capped"]),
                            over_budget=bool(arrays.get("omnivoice_over_budget", False)))
