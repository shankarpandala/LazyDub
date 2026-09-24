"""SessionController (§6.8): one per video. Processes look-ahead windows in playhead order and
streams fitted dub units to the UI.

GPU stages are serialised through one lock (ADR-004); CPU work runs in threads.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from .backends.base import SR_ANALYSIS, Backend, TranslationRequest
from .resolve import ResolveError, Resolver, ResolvedVideo
from .segment import segment
from .text.akshara import count_units
from .text.normalize_te import normalize_telugu
from .timing.duration import DurationEstimator
from .timing.isochrony import AddedTimeLedger, TimingSettings, Verdict, fit_unit, needs_condense, slot_budget, target_units
from .timing.stretch import wsola
from .timing.timeline import DubTimeline
from .types import SourceUnit, SpeakerTurn, VoiceKind
from .urls import URLError, parse_youtube_url

log = logging.getLogger("maata.session")

WINDOW = 60.0
OVERLAP = 5.0
LOOKAHEAD_CAP = 300.0  # §6.8: don't work more than 5 min ahead
REF_MIN, REF_MAX = 6.0, 15.0

SendJSON = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


@dataclass
class SpeakerState:
    id: str
    label: str
    ref: list[np.ndarray] = field(default_factory=list)
    ref_seconds: float = 0.0
    voice: object | None = None
    kind: VoiceKind = VoiceKind.PRESET
    use_preset: bool = False
    switched: bool = False


class Session:
    def __init__(self, backend: Backend, resolver: Resolver, cache_dir: Path, send_json: SendJSON, send_bytes: SendBytes,
                 settings: TimingSettings | None = None, gpu_lock: asyncio.Lock | None = None) -> None:
        self.b, self.resolver, self.cache_dir = backend, resolver, cache_dir
        self.send_json, self.send_bytes = send_json, send_bytes
        self.settings = settings or TimingSettings()
        self.gpu = gpu_lock or asyncio.Lock()
        self.timeline = DubTimeline()
        self.ledger = AddedTimeLedger(self.settings)
        self.estimator = DurationEstimator()
        self.speakers: dict[str, SpeakerState] = {}
        self.video: ResolvedVideo | None = None
        self.audio = np.zeros(0, np.float32)
        self.language: str | None = None
        self.playhead = 0.0
        self.done_windows: set[int] = set()
        self.units: list[SourceUnit] = []
        self.context: list[tuple[str, str]] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False
        self.ready_until = 0.0
        self.user_speed = 1.0
        self.raw: dict[int, tuple[np.ndarray, int, float]] = {}  # unit id → (samples at 1.0×, sr, audio rate)

    # ---- control -------------------------------------------------------------------------------
    async def open(self, url: str) -> None:
        try:
            ref = parse_youtube_url(url)
        except URLError as e:
            await self.send_json({"type": "error", "message": str(e), "retryable": False})
            return
        await self._status("resolving", "Finding the video…")
        await self.send_json({"type": "video", "videoId": ref.video_id, "start": ref.start, "title": "", "duration": 0})
        try:
            self.video = await asyncio.to_thread(self.resolver.resolve, ref, self.cache_dir)
            await self.send_json({"type": "video", "videoId": ref.video_id, "start": ref.start, "title": self.video.title,
                                  "channel": self.video.channel, "duration": self.video.duration})
            await self._status("fetching", "Fetching audio…")
            self.audio = await asyncio.to_thread(self.resolver.load_audio, self.video)
        except ResolveError as e:
            await self.send_json({"type": "error", "message": str(e), "retryable": True})
            return
        except Exception:
            log.exception("resolve failed")
            await self.send_json({"type": "error", "message": "Couldn't load this video.", "retryable": True})
            return
        self.playhead = ref.start
        self._task = asyncio.create_task(self._run())

    def seek(self, t: float) -> None:
        self.playhead = max(t, 0.0)
        self._wake.set()

    def set_playhead(self, t: float) -> None:
        self.playhead = max(t, 0.0)
        self._wake.set()

    async def set_speed(self, speed: float) -> None:
        """User playback speed (0.75–1.5×): re-render units that haven't played yet (§6.7)."""
        self.user_speed = min(max(speed, 0.75), 1.5)
        for uid, (samples, sr, rate) in list(self.raw.items()):
            p = self.timeline.placements.get(uid)
            if p and p.start + p.audio_wall >= self.playhead:
                await self._send_audio(uid, samples, sr, rate)

    async def _send_audio(self, uid: int, samples: np.ndarray, sr: int, rate: float) -> None:
        eff = rate * self.user_speed
        out = await asyncio.to_thread(wsola, samples, eff, sr) if abs(eff - 1.0) > 1e-3 else np.asarray(samples, np.float32)
        await self.send_bytes(struct.pack("<IIII", uid, sr, len(out), 0) + out.astype(np.float32).tobytes())

    async def set_speaker_preset(self, speaker: str, use_preset: bool) -> None:
        st = self.speakers.get(speaker)
        if st:
            st.use_preset = use_preset
            await self._send_speakers()

    async def close(self) -> None:
        self._closed = True
        if self._task:
            self._task.cancel()

    # ---- pipeline ------------------------------------------------------------------------------
    def _duration(self) -> float:
        return len(self.audio) / SR_ANALYSIS

    def _next_window(self) -> int | None:
        dur = self._duration()
        first = int(self.playhead // WINDOW)
        last = int(min(self.playhead + LOOKAHEAD_CAP, dur) // WINDOW)
        for w in range(first, last + 1):
            if w not in self.done_windows and w * WINDOW < dur:
                return w
        return None

    async def _run(self) -> None:
        try:
            while not self._closed:
                w = self._next_window()
                if w is None:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                await self._process_window(w)
                self.done_windows.add(w)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("pipeline failed")
            await self.send_json({"type": "error", "message": "Dubbing stopped unexpectedly.", "retryable": True})

    async def _process_window(self, w: int) -> None:
        t0 = w * WINDOW
        a0, a1 = max(t0 - OVERLAP, 0.0), min(t0 + WINDOW + OVERLAP, self._duration())
        chunk = self.audio[int(a0 * SR_ANALYSIS): int(a1 * SR_ANALYSIS)]
        await self._status("listening", "Listening…", t0)
        started = time.perf_counter()
        async with self.gpu:
            tr = await asyncio.to_thread(self.b.transcriber.transcribe, chunk, self.language)
            turns = await asyncio.to_thread(self.b.diarizer.diarize, chunk)
        self.language = self.language or tr.language
        words = [type(x)(x.text, x.start + a0, x.end + a0, x.confidence) for x in tr.words]
        words = [x for x in words if t0 <= x.start < t0 + WINDOW]  # own only the window's core
        turns = self._reconcile([SpeakerTurn(t.speaker, t.start + a0, t.end + a0) for t in turns])
        self._collect_references(turns)
        units = segment(words, turns)
        base = len(self.units)
        for k, u in enumerate(units):
            u.id = base + k
        self.units.extend(units)
        await self._send_speakers()
        for i, u in enumerate(units):
            if self._closed:
                return
            nxt = units[i + 1].start if i + 1 < len(units) else None
            await self._dub_unit(u, nxt)
        self.ready_until = max(self.ready_until, t0 + WINDOW)
        await self.send_json({"type": "stage_time", "stage": "window", "seconds": time.perf_counter() - started, "window": w})

    def _reconcile(self, turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        """Map window-local labels to session speakers by overlap with already-known turns.

        Phase 3 replaces this with embedding-based reconciliation (spec §6.2).
        """
        known = getattr(self, "_turns", [])
        mapping: dict[str, str] = {}
        for label in {t.speaker for t in turns}:
            best, best_ov = None, 0.0
            for t in (t for t in turns if t.speaker == label):
                for k in known:
                    ov = min(t.end, k.end) - max(t.start, k.start)
                    if ov > best_ov:
                        best, best_ov = k.speaker, ov
            mapping[label] = best or f"S{len(self.speakers) + len(mapping) + 1}"
        out = [SpeakerTurn(mapping[t.speaker], t.start, t.end) for t in turns]
        self._turns = (known + out)[-400:]
        for sid in {t.speaker for t in out}:
            self.speakers.setdefault(sid, SpeakerState(sid, f"Speaker {len(self.speakers) + 1}"))
        return out

    def _collect_references(self, turns: list[SpeakerTurn]) -> None:
        for t in turns:
            st = self.speakers[t.speaker]
            clean = t.end - t.start
            overlapped = any(o.speaker != t.speaker and o.start < t.end and o.end > t.start for o in turns)
            if clean < 1.5 or overlapped or st.ref_seconds >= REF_MAX:
                continue
            take = min(clean, REF_MAX - st.ref_seconds)
            st.ref.append(self.audio[int(t.start * SR_ANALYSIS): int((t.start + take) * SR_ANALYSIS)])
            st.ref_seconds += take

    async def _voice_for(self, speaker: str) -> tuple[object, VoiceKind]:
        st = self.speakers[speaker]
        if not st.use_preset and st.voice is not None and st.kind is VoiceKind.CLONED:
            return st.voice, st.kind
        if not st.use_preset and st.ref_seconds >= REF_MIN and not st.switched:
            ref = np.concatenate(st.ref)
            st.voice = await asyncio.to_thread(self.b.tts.prepare_voice, ref, SR_ANALYSIS)
            st.kind, st.switched = VoiceKind.CLONED, True  # switch once, at a unit boundary
            await self._send_speakers()
            return st.voice, st.kind
        preset = "preset_m" if int(speaker.lstrip("S") or 1) % 2 else "preset_f"
        return await asyncio.to_thread(self.b.tts.preset_voice, preset), VoiceKind.PRESET

    async def _dub_unit(self, u: SourceUnit, next_start: float | None) -> None:
        s = self.settings
        budget = slot_budget(u.start, u.end, next_start, s)
        target = target_units(budget, self.estimator.rate(u.speaker), s)
        req = TranslationRequest(u.text, self.language or "en", context=self.context[-3:], target_units=target)
        try:
            async with self.gpu:
                telugu = normalize_telugu(await asyncio.to_thread(self.b.translator.translate, req))
                voice, kind = await self._voice_for(u.speaker)
                samples = await asyncio.to_thread(self.b.tts.synthesize, telugu, voice)
                sr = self.b.tts.sample_rate
                dur = len(samples) / sr
                self.estimator.observe(telugu, u.speaker, dur)
                rounds = 0
                while needs_condense(dur, budget, s) and rounds < s.max_condense_rounds:
                    rounds += 1
                    limit = budget * self.estimator.rate(u.speaker) * (s.target_fill - 0.1 * rounds)
                    cands = await asyncio.to_thread(self.b.translator.condense, telugu, limit)
                    alt = normalize_telugu(cands[0])
                    alt_samples = await asyncio.to_thread(self.b.tts.synthesize, alt, voice)
                    if len(alt_samples) < len(samples):
                        telugu, samples, dur = alt, alt_samples, len(alt_samples) / sr
                        self.estimator.observe(telugu, u.speaker, dur)
            fit = fit_unit(u.id, u.start, u.end, next_start, dur, s, self.ledger)
            if fit.verdict is Verdict.CONDENSE:  # budget used up: place at the cap and trim the tail
                keep = int(min(dur, budget * s.speed_cap) * sr)
                samples = samples[:keep]
                fit = fit_unit(u.id, u.start, u.end, next_start, keep / sr, s, self.ledger)
        except Exception:
            log.exception("unit %s failed", u.id)
            await self.send_json({"type": "unit_skipped", "id": u.id, "start": u.start, "end": u.end})
            return
        p = fit.placement
        assert p is not None
        self.timeline.add(p)
        self.context.append((u.text, telugu))
        await self.send_json({
            "type": "unit", "id": u.id, "speaker": u.speaker, "start": p.start, "end": u.end, "budget": p.budget,
            "audioRate": p.audio_rate, "audioWall": p.audio_wall, "voice": kind.value,
            "source": u.text, "telugu": telugu, "units": count_units(telugu),
            "edits": [{"kind": e.kind.value, "at": e.at, "span": e.video_span, "rate": e.rate, "added": e.added} for e in p.edits],
        })
        self.raw[u.id] = (np.asarray(samples, np.float32), sr, p.audio_rate)
        await self._send_audio(u.id, self.raw[u.id][0], sr, p.audio_rate)
        self.ready_until = max(self.ready_until, u.end)
        await self.send_json({"type": "ready", "until": self.ready_until})

    async def _send_speakers(self) -> None:
        await self.send_json({"type": "speakers", "speakers": [
            {"id": s.id, "label": s.label, "voice": s.kind.value, "referenceSeconds": round(s.ref_seconds, 1), "usePreset": s.use_preset}
            for s in self.speakers.values()
        ]})

    async def _status(self, stage: str, message: str, at: float | None = None) -> None:
        await self.send_json({"type": "status", "stage": stage, "message": message, "at": at})
