"""Standalone local worker launched by omnivoice.py in its pinned, isolated Python runtime.

No engine imports or sockets. JSON control arrives on stdin; raw PCM leaves via a private local .npy file.
The production parent applies PerTh after timing adjustments in its existing runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import queue
import threading

PROTOCOL_VERSION = 2
SAMPLE_RATE = 24_000
MAX_SECONDS = 60.0
SOURCE_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
GENERATION_SEED = 20261010
VOICE_PROFILES = {"automatic": None, "male": "male", "female": "female"}


def checked_profile(profile, instruct, identity, mode):
    """Only predeclared categories; free-form instructions cannot enter production requests."""
    if not isinstance(profile, str) or profile not in VOICE_PROFILES or instruct != VOICE_PROFILES[profile]:
        raise ValueError("Invalid OmniVoice profile or instruction.")
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("Invalid OmniVoice profile identity.")
    if mode == "reference-experiment" and profile != "automatic":
        raise ValueError("Gender profiles cannot use reference conditioning.")
    return profile


def only_local(value) -> str:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError("OmniVoice automatic remote model resolution is disabled.")
    return str(path.resolve())


def no_asr(*args, **kwargs):
    raise RuntimeError("OmniVoice automatic ASR is disabled; exact reference text is required.")


def verify_runtime() -> None:
    for name, expected in {"torch": "2.8.0", "torchaudio": "2.8.0", "transformers": "5.3.0"}.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Isolated OmniVoice requires pinned {name} {expected}.")
    source = json.loads(importlib.metadata.distribution("omnivoice").read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != SOURCE_REVISION:
        raise RuntimeError("Isolated OmniVoice source revision does not match its pin.")


class SpeechWorker:
    def __init__(self, config: dict):
        verify_runtime()  # fail before importing a mismatched model/Transformers implementation
        import numpy as np
        import torch
        import omnivoice.models.omnivoice as impl

        self.np, self.torch = np, torch
        self.config = config
        root = Path(only_local(config["model_dir"]))
        only_local(root / "audio_tokenizer")
        impl._resolve_model_path = only_local
        impl.OmniVoice.load_asr_model = no_asr
        mode = config.get("mode")
        if mode not in ("automatic", "reference-experiment") or config.get("seed") != GENERATION_SEED:
            raise RuntimeError("Unsupported OmniVoice generation mode or fixed seed.")
        reference = None
        if mode == "reference-experiment":
            reference = Path(config["reference_path"])
            if not reference.is_absolute() or not reference.is_file() or not config["reference_text"].strip():
                raise RuntimeError("The experimental reference and exact transcript are required.")
            if hashlib.sha256(reference.read_bytes()).hexdigest() != config["reference_sha256"]:
                raise RuntimeError("The experimental reference checksum does not match.")
        elif config.get("reference_path") or config.get("reference_text") or config.get("reference_sha256"):
            raise RuntimeError("Automatic OmniVoice must not receive reference conditioning.")
        identity = config["voice_identity"]
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise RuntimeError("Invalid provisioned voice identity.")
        if config["steps"] != 32 or config["device"] not in ("mps", "cuda", "cpu"):
            raise RuntimeError("Unsupported OmniVoice generation configuration.")
        expected_dtype = "float32" if config["device"] == "cpu" else "float16"
        if config["dtype"] != expected_dtype:
            raise RuntimeError("Unsupported OmniVoice precision.")
        self.model = impl.OmniVoice.from_pretrained(root, device_map=config["device"],
                         dtype=getattr(torch, expected_dtype), local_files_only=True,
                         trust_remote_code=False, load_asr=False)
        if int(self.model.sampling_rate) != SAMPLE_RATE:
            raise RuntimeError("Pinned OmniVoice codec sample rate changed.")
        self.prompt = None
        if reference is None:
            return
        store = Path(config["voice_dir"])
        if not store.is_absolute():
            raise RuntimeError("Voice prompt cache must be an absolute local directory.")
        store.mkdir(parents=True, exist_ok=True)
        path = store / f"{identity}.pt"
        if path.is_file() and not path.is_symlink():
            self.prompt = impl.VoiceClonePrompt.load(str(path), map_location="cpu")
            if self.prompt.ref_text != config["reference_text"]:
                raise RuntimeError("Cached voice prompt transcript does not match its identity.")
        else:
            self.prompt = self.model.create_voice_clone_prompt(ref_audio=str(reference),
                                ref_text=config["reference_text"], preprocess_prompt=False)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            try:
                self.prompt.save(str(temporary))
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def generate(self, text: str, profile: str = "automatic"):
        if profile not in VOICE_PROFILES or (self.prompt is not None and profile != "automatic"):
            raise RuntimeError("Invalid OmniVoice generation profile.")
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise RuntimeError("Empty or oversized OmniVoice text.")
        # Unlike autoregressive speech decoding, OmniVoice fixes its frame count before its 32 steps.
        # Bound pathological allocations, but never force a shorter duration or crop the natural speech.
        tokens = self.model._estimate_target_tokens(text, self.prompt.ref_text if self.prompt is not None else None,
                                                    self.prompt.ref_audio_tokens.size(-1) if self.prompt is not None else None)
        predicted = tokens / float(self.model.audio_tokenizer.config.frame_rate)
        if not math.isfinite(predicted) or predicted > MAX_SECONDS:
            raise RuntimeError(f"Natural OmniVoice duration estimate exceeds the {MAX_SECONDS:g}-second limit.")
        # One predeclared seed from the original user-selected auto audition, reset for every call. No search,
        # text-derived seeds or retry seeds, and no dependence on which lines happened to run before this one.
        self.torch.manual_seed(GENERATION_SEED)
        self.np.random.seed(GENERATION_SEED)
        options = {"voice_clone_prompt": self.prompt} if self.prompt is not None else {}
        if VOICE_PROFILES[profile] is not None:
            options["instruct"] = VOICE_PROFILES[profile]
        with self.torch.inference_mode():
            audio = self.model.generate(text=text, language="te", num_step=32, normalize_text=False, **options)[0]
        x = self.np.asarray(audio, dtype=self.np.float32)
        if (x.ndim != 1 or not len(x) or not self.np.isfinite(x).all()
                or float(self.np.max(self.np.abs(x))) < 1e-6):
            raise RuntimeError("OmniVoice returned empty, nonfinite or silent audio.")
        if len(x) > (MAX_SECONDS + 1) * SAMPLE_RATE:
            raise RuntimeError("OmniVoice output exceeds its bounded generation limit.")
        return x, predicted


def serve(work_dir: Path, factory=SpeechWorker, *, stop_on_eof: bool = True) -> None:
    worker = None
    config = None
    work_dir = work_dir.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    incoming = queue.Queue()

    def read_parent() -> None:
        while True:
            raw = sys.stdin.buffer.readline(65537)
            if not raw:
                if stop_on_eof:
                    # The parent owns the stdin pipe. Its disappearance must stop even a model currently loading
                    # or generating, so a restarted engine never competes with an orphaned GPU worker.
                    os._exit(0)
                incoming.put(None)
                return
            incoming.put(raw)
            if len(raw) > 65536:
                return

    threading.Thread(target=read_parent, name="parent-lifeline", daemon=True).start()
    while (raw := incoming.get()) is not None:
        reply = {"version": PROTOCOL_VERSION, "id": None, "ok": False, "sample_rate": SAMPLE_RATE}
        try:
            if len(raw) > 65536:
                raise ValueError("OmniVoice request exceeds its protocol bound.")
            message = json.loads(raw)
            rid = message.get("id")
            reply["id"] = rid
            if (message.get("version") != PROTOCOL_VERSION or type(rid) is not int or rid < 1
                    or message.get("op") not in ("prepare", "generate")):
                raise ValueError("Invalid OmniVoice protocol request.")
            if message["op"] == "generate":
                checked_profile(message.get("profile"), message.get("instruct"), message.get("voice_identity"),
                                message.get("config", {}).get("mode"))
            with contextlib.redirect_stdout(sys.stderr):
                if worker is None:
                    config = message["config"]
                    worker = factory(config)
                elif config != message["config"]:
                    raise ValueError("A persistent worker cannot change model or voice configuration.")
                if message["op"] == "generate":
                    audio, predicted = worker.generate(message["text"], message["profile"])
                    # The parent controls rid and work_dir; no arbitrary output path is accepted.
                    path = work_dir / f"audio-{rid}.npy"
                    with path.open("wb") as output:
                        worker.np.save(output, audio, allow_pickle=False)
                    reply.update(frames=len(audio), estimated_seconds=predicted,
                                 **{k: message[k] for k in ("profile", "instruct", "voice_identity")})
            reply["ok"] = True
        except Exception as exc:
            reply["error"] = f"{type(exc).__name__}: {exc}"
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.work_dir.is_dir():
        parser.error("work directory must already exist")
    serve(arguments.work_dir)
