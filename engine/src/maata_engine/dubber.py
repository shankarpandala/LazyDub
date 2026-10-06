"""The per-line dubbing machinery (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §6) that the render job
(`render.RenderJob`) is built on: speakers' voices (built from their clean speech and calibrated), each line's sizes and
the wording that fits its speech time (ARCHITECTURE §4.3-§4.5), scene calls with their coverage review (§4), takes at
the voice's natural pace (§3.8, §3.13), their place on the video's clock (§3.10) and units.jsonl.

What the job does differently goes through its hooks: `_gpu_priority`, `_skip`, `notify` and `_on_cancel`. Every model
call goes through `_on_gpu`, which keeps the GPU until the call has returned, even when the task awaiting it is
cancelled (OFFLINE-RENDER §2.1).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np

from .backends.base import (SR_ANALYSIS, Backend, LineResult, LineSpec, SceneRequest, SceneTranslator, Transcript,
                            Wording)
from .claude_cli import ClaudeCLIError
from .gpu import BACKGROUND, VOICE, GpuScheduler
from .qa import take as take_qa
from .qa.coverage import CLASSES, approved_full, voiced_class
from .segment import sentence_end
from .speakers import SpeakerRegistry
from .text import asr_guard, tenglish
from .text.akshara import count_units, mixed_units
from .timing.duration import MAX_OVERHEAD, MIN_RATE, DurationEstimator, VoiceKey
from .timing import pauses as pz
from .timing.metrics import line_metrics
from .timing.planner import (EPS, INF, V1, LineSlot, Plan, PlannerSettings, TimelinePlanner, anchor_errors,
                             rate_ceiling, voiced)
from .timing.stretch import wsola
from .types import SourceUnit, VoiceKind

log = logging.getLogger("maata.dubber")

CHUNK = 60.0             # s of audio per ASR pass
CHUNK_PAD = 5.0          # s of extra audio after the chunk so the last sentence can finish
REF_TARGET = 10.0        # s of reference audio per cloned voice (S3Gen uses 10 s, T3 the first 6 s)
REF_MIN = 4.0            # s: below this a speaker keeps a preset voice until more clean speech appears
MAX_LINE_SECONDS = 30.0
SINGLE_CLIP_MIN = 8.0   # s: a speaker's best single clean clip must be this long to be the timbre reference (ADR-017)
# Bounded translation scenes: at most 30 s and 6 whole lines, cut at a speaker turn or a pause >= 1 s.
SCENE_MAX_S, SCENE_MAX_UNITS, SCENE_PAUSE = 30.0, 6, 1.0
CONTEXT_BEFORE, CONTEXT_AFTER, CONTEXT_SPAN = 3, 2, 60.0  # context lines each side of a scene, from within this many s
# Aksharas of `full` per English syllable (§4.3): a weak prior until a speaker has K_MIN_LINES validated lines, then
# the running median of theirs (lines of at least K_MIN_SYLLABLES syllables, the last K_WINDOW).
K_PRIOR, K_MIN_LINES, K_MIN_SYLLABLES, K_WINDOW = 1.4, 3, 4, 400
BAND = (0.85, 1.10)     # §4.5: a wording fits when its predicted duration is within this share of the line's speech time
DENSE = 0.8             # a slot is speech-dense when speech fills this share of its span (only then is `fuller` asked)
TIERS_BY_FULLNESS = ("fuller", "full", "concise", "very_concise")
CLAUDE_BACKOFF, CLAUDE_BACKOFF_MAX = 30.0, 300.0  # s before the next scene call after a failure, doubling
MARKS = re.compile(r"[,;:.!?।॥…—]+")  # where a wording breaks inside: a Telugu pause there is plausible (§3.10 step 3)
BRIEF_TRIES = 3
# Calibration sentences (original; ARCHITECTURE §3.7 step 5), voiced once per cloned voice to set its pace: three lengths,
# so rate and overhead separate, written to the script contract like every dub line (English loans in Telugu script,
# with their English map for the Latin TTS input).
CALIBRATION_TE = (
    Wording("నిజం చెప్పాలంటే, ఈ టాపిక్ చాలా ఇంట్రెస్టింగ్.", ((3, "topic"), (5, "interesting"))),
    Wording("మన ఛానెల్లో ఇంతకు ముందు ఇలాంటి వీడియో ఒకటి చేశాం, లింక్ కింద ఇస్తాను.", ((1, "channel"), (5, "video"), (8, "link"))),
    Wording("ఆఫీస్ నుంచి ఇంటికి వచ్చాక కూడా మెయిల్స్ చెక్ చేస్తూ ఉంటే, మన మైండ్కి అసలు రెస్ట్ దొరకదు, అందుకే ఫోన్ పక్కన "
            "పెట్టేయాలి.", ((0, "office"), (5, "mails"), (6, "check"), (10, "mind"), (12, "rest"), (15, "phone"))),
)


@dataclass
class VoiceState:
    voice: object | None = None
    kind: VoiceKind = VoiceKind.PRESET
    ref_seconds: float = 0.0
    use_preset: bool = False
    status: str = "found"  # found | cloning | cloned | preset
    key: VoiceKey | None = None  # the clone's key in the duration estimator, taken (and calibrated) when it is built


@dataclass
class UnitState:
    unit: SourceUnit
    next_start: float | None
    speech_s: float = 0.0                    # diarized speech inside the span, what lengths are measured against (§4.3)
    music: bool = False                      # looks sung (asr_guard.music_like): reported, still dubbed (§3.4)
    telugu: str | None = None                # the chosen wording as the TTS says it; None until translated
    voicing: bool = False                    # the dub loop has started on it: its wording no longer changes
    voiced: bool = False
    scene: int | None = None                 # the scene call it is in, or was translated in
    line: LineResult | None = None           # its validated translation: the tiers, English map and delivery
    tier: str | None = None                  # the tier chosen (§4.5)
    pred_full: float | None = None           # `full` aksharas predicted when its scene was asked for (units.jsonl)
    plan: Plan | None = None                 # where its take was placed (on the working plan), which a fix-up keeps
    take_s: float | None = None              # natural duration of the take voiced
    take: dict | None = None                 # the render's takes.jsonl row of the take voiced: no audio (OFFLINE-RENDER §2.9)
    # How the text was made (units.jsonl, ARCHITECTURE §4.7).
    translate_ready_at: float | None = None  # epoch s the line's text was ready
    translate_s: float | None = None         # wall time of its scene call
    model: str | None = None                 # the text model that answered
    prompt_hash: str | None = None
    cache_hit: bool | None = None            # served from the per-video line cache


@dataclass
class VoiceCost:
    """What voicing one unit cost, for units.jsonl (ARCHITECTURE §7 step 0): waits for the GPU, every synthesis take (T3
    summed over takes, and S3Gen over the takes vocoded, when the TTS reports them; None when it doesn't) and rendering
    at the planned rate; with the takes asked for, the retakes, why any take failed, and the GPU priority of its
    synthesis. `synth_s` is T3 plus the S3Gen flow of the take voiced (which runs when it is vocoded) and `render_s` the
    vocoder alone, as before takes were batched. `t3_ms_per_token` is per decode step, which samples a token for every
    take of a batch: it stays comparable whatever number of takes a line asks for."""

    lock_wait_s: float = 0.0
    takes: int = 0
    synth_s: float = 0.0
    render_s: float = 0.0
    t3_s: float | None = None
    t3_tokens: int | None = None
    t3_steps: int | None = None
    flow_s: float | None = None
    cfm_steps: int | None = None
    takes_n: int | None = None          # takes asked for in the line's first batch (the render's TAKES_N, §2.9)
    retakes: int = 0                    # takes made because every take of a batch failed
    failures: list[str] = field(default_factory=list)  # why takes failed (qa.take: "cap", "short"), over all batches
    priority: int | None = None         # the GPU priority of its first take

    @asynccontextmanager
    async def hold(self, gpu: GpuScheduler, priority: int) -> AsyncIterator[None]:
        t0 = time.perf_counter()
        async with gpu.hold(priority):
            self.lock_wait_s += time.perf_counter() - t0
            yield

    def add_takes(self, takes: list, seconds: float) -> None:
        """One synthesis call's takes (a batch shares one decode, counted once in the steps) and its wall time."""
        self.takes += len(takes)
        self.synth_s += seconds
        mel = [t for t in takes if hasattr(t, "t3_s")]  # MelTakes
        for take in mel:
            self.t3_s = (self.t3_s or 0.0) + take.t3_s
            self.t3_tokens = (self.t3_tokens or 0) + take.t3_tokens
            self.flow_s = (self.flow_s or 0.0) + take.flow_s
            self.cfm_steps = take.cfm_steps
        if mel:
            self.t3_steps = (self.t3_steps or 0) + max(t.t3_steps for t in mel)

    def add_take(self, take: object, seconds: float) -> None:
        self.add_takes([take], seconds)

    def add_flow(self, before: float, take: object) -> float:
        """S3Gen time a take picked up while it was vocoded (the flow runs then: torch_common.MelTake), counted as
        synthesis. Returns it."""
        if hasattr(take, "flow_s") and take.flow_s > before:
            self.flow_s = (self.flow_s or 0.0) + take.flow_s - before
            self.synth_s += take.flow_s - before
            return take.flow_s - before
        return 0.0

    def fields(self) -> dict:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 3)

        per_step = 1000 * self.t3_s / self.t3_steps if self.t3_s is not None and self.t3_steps else None
        return {"lock_wait_s": r(self.lock_wait_s), "takes": self.takes, "synth_s": r(self.synth_s), "t3_s": r(self.t3_s),
                "t3_tokens": self.t3_tokens, "t3_steps": self.t3_steps, "t3_ms_per_token": r(per_step),
                "flow_s": r(self.flow_s), "cfm_steps": self.cfm_steps, "render_s": r(self.render_s),
                "takes_n": self.takes_n, "retakes": self.retakes, "take_failures": list(self.failures),
                "gpu_priority": self.priority}


@dataclass
class Said:
    """A line's takes as the dub loop made them (ARCHITECTURE §3.10): one, or one per piece at hard breaks, each with its
    natural duration, its audio at the voice's natural pace (where its pauses are found, and what plays when it isn't
    sped up), its pauses (`pauses.pauses`) and the pace the estimator learns from (`_take`)."""

    wordings: list[Wording] = field(default_factory=list)
    takes: list[object] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)
    pauses: list[tuple[tuple[float, float], ...]] = field(default_factory=list)
    paces: list[float | None] = field(default_factory=list)
    failed: str | None = None  # why a take voiced failed (the first that did), or None
    failures: list[str | None] = field(default_factory=list)  # why each take failed, or None

    @property
    def total(self) -> float:
        return sum(self.seconds)

    def timed(self) -> list[tuple[float, tuple[tuple[float, float], ...]]]:
        """Each take as the planner takes it: (natural seconds, pauses)."""
        return list(zip(self.seconds, self.pauses))


class Dubber:
    """One video's lines, from their English to their Telugu audio placed on the video's clock. A job fills `audio`,
    `registry`, `units`, `voices` and `tr`, and calls the per-line methods; its hooks say what it does differently."""

    def __init__(self, backend: Backend, gpu: GpuScheduler | None = None, style: str = "colloquial",
                 speed_cap: float = 1.2, allow_freeze: bool = True, tts_script: str = "telugu",
                 timing: str = "v2") -> None:
        self.b = backend
        self.gpu = gpu or GpuScheduler()  # the engine's: one GPU owner at a time
        self.style = "formal" if style == "formal" else "colloquial"
        self.speed_cap = min(max(float(speed_cap), 1.0), 1.25)  # the UI's "Maximum speed-up"
        self.allow_freeze = bool(allow_freeze)
        # What the TTS reads (decision D6): the Telugu-script wording, or for the maintainer's A/B its English words in
        # Latin script, rebuilt from the English-word map.
        self.tts_script = "latin" if tts_script == "latin" else "telugu"
        # How lines are timed (ADR-017, §3.10), for measurements: "v2"; "v2-whole", every sentence said whole (§7.1's
        # unit-and-contract arm A); "v1", the timing before v2 (the baseline run its targets are measured against).
        self.timing = timing if timing in ("v2-whole", "v1") else "v2"
        self.planner = TimelinePlanner(PlannerSettings(speed_cap=self.speed_cap, max_freeze=0.6 if self.allow_freeze else 0.0,
                                                       freeze_budget=1.0 if self.allow_freeze else 0.0,
                                                       **(V1 if self.timing == "v1" else {})))
        self._resynths = 0
        self._retakes = 0
        self.estimator = DurationEstimator()
        self.calibrate_s = 0.0  # s spent calibrating voices (three takes each)
        self.registry = SpeakerRegistry()
        self.voices: dict[str, VoiceState] = {}
        self.audio = np.zeros(0, np.float32)  # the video's audio, mono at SR_ANALYSIS
        self.start = 0.0
        self.units: dict[int, UnitState] = {}
        self._wake = _Wake()
        self._dir: Path | None = None  # the video's cache directory (units.jsonl)
        self._run_id: str | None = None  # distinguishes fresh work from earlier attempts and cache replay
        self.tr: SceneTranslator | None = None
        self._side_tasks: set[asyncio.Task] = set()   # fits, a scene's fix-ups, saves: never awaited by the dub loop
        self._ratios: dict[str, list[float]] = {}   # per speaker: `full` aksharas / English syllables of each line
        self._claude_hold = 0.0               # monotonic s before which no new scene call starts
        self._claude_fails = 0
        self._claude_kind: str | None = None
        self._claude_failed_at = float("-inf")  # monotonic s of the last failure
        self._claude_said_ok = False            # claude_ok sent by this job
        self._presets_made: set[str] = set()    # preset voices made (`_preset`)

    # ---- hooks: what a job does differently ---------------------------------------------------------------------
    def _gpu_priority(self) -> int:
        """The GPU priority of a model call that names none (gpu.py): voicing a line, building a voice."""
        return VOICE

    async def _skip(self, st: UnitState, why: str) -> None:
        """A line that won't be voiced: it counts as done, and why is logged and traced."""
        st.voiced = True
        u = st.unit
        log.warning("skipped unit %d (%s): %s", u.id, why, u.text)
        self._trace({"event": "skipped", "id": u.id, "start": u.start, "end": u.end, "source": u.text, "why": why,
                     "music": st.music, "speech": [list(x) for x in self._speech_spans(u)]})

    async def notify(self, msg: dict) -> None:
        """A message for the UI (`claude_error`, `claude_ok`)."""

    def _on_cancel(self) -> None:
        """The task awaiting a model call was cancelled, and `_on_gpu` is about to wait the call out: a job whose long
        calls watch a cancel event sets it here, so the call stops early instead of running to its end."""

    # ---- the GPU ------------------------------------------------------------------------------------------------
    async def _on_gpu(self, fn: Callable[..., Any], *args: Any, cost: VoiceCost | None = None,
                      priority: int | None = None, **kwargs: Any) -> tuple[Any, float]:
        """Run the model call `fn(*args, **kwargs)` in a worker thread, holding the GPU at `priority` (by default
        `_gpu_priority`); through `cost` when it is given, so its wait for the GPU counts in the line's `lock_wait_s`.
        Returns its result and the seconds from getting the GPU to its return. If the awaiting task is cancelled, the
        GPU is kept until the call has returned, however many more cancels arrive meanwhile, and only then let go: a
        worker thread can't be stopped, and the next holder must not run beside it (a second pyannote run, or a second
        `synthesize_takes` that swaps the model's conditionals under the first; OFFLINE-RENDER §2.1). `_on_cancel` runs
        first, so a call that watches a cancel event returns early."""
        priority = self._gpu_priority() if priority is None else priority
        async with (self.gpu.hold(priority) if cost is None else cost.hold(self.gpu, priority)):
            t0 = time.perf_counter()
            fut = asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))
            try:
                out = await asyncio.shield(fut)
            except asyncio.CancelledError:
                self._on_cancel()
                while not fut.done():
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(fut)
                if not fut.cancelled():
                    fut.exception()  # retrieved: the call's own error isn't news once it was cancelled
                raise
            return out, time.perf_counter() - t0

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _duration(self) -> float:
        return len(self.audio) / SR_ANALYSIS

    def _trace(self, rec: dict) -> None:
        """One line of units.jsonl. Events: stage (per stage of a run: its seconds, GPU seconds, `cached` when served
        from disk), asr (per chunk: `run_s` holding the GPU, `lock_wait_s` to get it, the GPU `priority` asked with, the
        ASR guards' findings), clone (`build_s`, `calibrate_s`, `lock_wait_s`), scene (per scene: its lines' coverage
        classes), fixups (per scene settled), unit (per final line, `RenderJob._unit_event`: timings from
        `VoiceCost.fields`, the text's `UnitState` fields, its coverage class and that class as the metrics count it for
        the wording voiced, speech fill and required rate, the timing fields), skipped, and from the translator's `trace`
        (may arrive from a worker thread): claude (per CLI call), translate (per request), review (per scene review) and
        brief."""
        if self._dir is None:
            return
        try:
            with (self._dir / "units.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "run_id": self._run_id, **rec},
                                   ensure_ascii=False) + "\n")
        except OSError:
            log.warning("could not write units.jsonl", exc_info=True)

    # ---- voices -------------------------------------------------------------------------------------------------
    def _cut(self, a: float, b: float) -> np.ndarray:
        return self.audio[int(a * SR_ANALYSIS): int(b * SR_ANALYSIS)]

    def _stitched(self, clips: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
        """The stitched reference: `clips` joined with short gaps (used when no single clip is long enough)."""
        parts, secs = [], 0.0
        gap = np.zeros(int(0.15 * SR_ANALYSIS), np.float32)
        for a, b in clips:
            parts += [self._cut(a, b), gap]
            secs += b - a
        return (np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)), secs

    def _voice_spans(self, sid: str) -> tuple[str, tuple[float, float] | None, list[tuple[float, float]]]:
        """Where a speaker's voice is built from: ("single-clip", the timbre span, the identity clips) when their best
        single clean clip is long enough and the TTS takes parts, else ("stitched", None, the stitched reference's clips:
        their best clips up to REF_TARGET s). ADR-017, measured on the maintainer's podcast: timbre from the speaker's best
        single clean clip plus identity averaged over up to 60 s of their clean speech made the guest's clone closer
        (similarity 0.40 -> 0.47) and fixed its pace (3.4 -> 5.1 aksharas/s). With under ~8 s in one clip, the stitched
        reference was better, so it falls back to that."""
        # Skip the first minute (often a cold-open montage over music) when enough later speech has been diarized.
        avoid = self.start + 60.0 if self.registry.covered_until(self.start) >= self.start + 180.0 else None
        best = self.registry.best_span(sid, length=10.0, min_len=SINGLE_CLIP_MIN, avoid_before=avoid)
        if best is not None and hasattr(self.b.tts, "prepare_voice_parts"):
            return "single-clip", best, self.registry.clean_clips(sid, max_total=60.0, avoid_before=avoid)
        return "stitched", None, self.registry.reference_clips(sid, target=REF_TARGET, avoid_before=self.start + 60.0)

    async def _voice_from(self, how: str, best: tuple[float, float] | None, clips: list[tuple[float, float]],
                          cost: VoiceCost, priority: int = BACKGROUND) -> tuple[object | None, float, str, str]:
        """The voice built from the spans `_voice_spans` chose (or a render stored). Returns (voice, seconds of
        reference, how it was built, a hash of the reference audio); no voice under REF_MIN s of stitched reference."""
        tts = self.b.tts
        if how == "single-clip":
            timbre = self._cut(*best)
            pool = [self._cut(a, b) for a, b in clips] or [timbre]
            voice, _ = await self._on_gpu(tts.prepare_voice_parts, timbre, pool, timbre, SR_ANALYSIS, 0.5, cost=cost,
                                          priority=priority)
            return voice, sum(b - a for a, b in clips) or best[1] - best[0], "single-clip", _audio_hash(timbre, *pool)
        ref, secs = self._stitched(clips)
        if secs < REF_MIN:
            return None, secs, "none", ""
        voice, _ = await self._on_gpu(tts.prepare_voice, ref, SR_ANALYSIS, cost=cost, priority=priority)
        return voice, secs, "stitched", _audio_hash(ref)

    def _voice_key(self, name: str, voice: object | None, reference: str = "") -> VoiceKey:
        """How the duration estimator knows a voice: its name, its guidance weight (its own, else the TTS's), the TTS's
        exaggeration (None where the TTS has neither) and the hash of its reference audio."""
        tts = self.b.tts
        cfg = getattr(voice, "cfg_weight", None)
        return VoiceKey(name, getattr(tts, "cfg_weight", None) if cfg is None else cfg, getattr(tts, "exaggeration", None),
                        reference)

    def _key(self, sid: str) -> VoiceKey:
        """The estimator's key for the voice a speaker's lines are voiced with now, as `_voice_for` picks it: their clone,
        else their preset (keyed by its name, so speakers who share a preset share its pace)."""
        v = self.voices.get(sid)
        if v is not None and v.kind is VoiceKind.CLONED and v.voice is not None and v.key is not None and not v.use_preset:
            return v.key
        return self._voice_key(_preset_name(sid), None)

    async def _calibrate(self, sid: str, key: VoiceKey, voice: object, cost: VoiceCost, priority: int = BACKGROUND,
                         keep: list[tuple[str, float, object]] | None = None) -> int:
        """Set a new voice's pace, rate and overhead together, from the three calibration sentences (ARCHITECTURE §3.7), so
        length targets fit it from the first line. Each is said as lines are (`tts_script`) and counted on its Telugu
        script; a take that says next to nothing or runs to its cap is left out. The cap is the slowest take the estimator
        can represent (MIN_RATE after MAX_OVERHEAD), so a slow voice is still measured and only a runaway is dropped.
        The takes run at the priority of the voice they calibrate, which isn't published until they are done.
        `keep`, when given, gets each take used as (spoken, seconds, take). Returns the number of takes used."""
        tts = self.b.tts
        if not hasattr(tts, "synthesize_mel"):
            return 0
        takes: list[tuple[str, float]] = []
        for w in CALIBRATION_TE:
            cap = min(MAX_LINE_SECONDS, MAX_OVERHEAD + count_units(w.spoken) / MIN_RATE)
            take, _ = await self._on_gpu(tts.synthesize_mel, self._tts_text(w), voice, "te", cap, cost=cost, priority=priority)
            if 1.0 < take.seconds < cap and not getattr(take, "capped", False):
                takes.append((w.spoken, take.seconds))
                if keep is not None:
                    keep.append((w.spoken, take.seconds, take))
        if not takes:
            log.warning("%s: no usable calibration take; its pace is learned from its lines", sid)
            return 0
        fitted = self.estimator.calibrate(key, takes)
        log.info("%s speaks at %.1f aksharas/s with %.2f s overhead (%d calibration takes%s)", sid, self.estimator.rate(key),
                 self.estimator.overhead(key), len(takes), "" if fitted else ", one length: an update, not a fit")
        return len(takes)

    async def _preset(self, name: str, cost: VoiceCost) -> object:
        """A preset voice. The first time a name is asked for, it is made holding the GPU like any model call: it may be
        the TTS's first use, which loads its weights (torch_common.ChatterboxTeluguTTS loads on first use), and a preset
        read from a wav is conditioned on the GPU, which swaps the model's conditionals. After that it is the TTS's
        cached copy, and the dub loop doesn't queue for the GPU twice to voice one line."""
        if name in self._presets_made:
            return await asyncio.to_thread(self.b.tts.preset_voice, name)
        voice, _ = await self._on_gpu(self.b.tts.preset_voice, name, cost=cost)
        self._presets_made.add(name)
        return voice

    # ---- listening ----------------------------------------------------------------------------------------------
    def _asr_checks(self, tr: Transcript, lo: float, a: float, b: float, words: list) -> dict:
        """What the ASR guards found in the chunk [a, b] (ARCHITECTURE §3.4), logged and returned for its asr event (video
        times): the diarized speech the transcriber decoded again, the words that brought back and those it didn't
        trust, the punctuation pass's window-cut guesses left out, the speech still without words, clusters of
        low-confidence words and phrases repeated back to back."""
        def spans(xs: list[tuple[float, float]], shift: float = 0.0) -> list[list[float]]:
            return [[round(x + shift, 2), round(y + shift, 2)] for x, y in xs]

        speech = [(max(t.start, a), min(t.end, b)) for t in self.registry.turns_in(a, b, exclusive=False)]
        found = {"redecoded": spans(tr.gaps, lo), "recovered": tr.recovered, "rejected": tr.rejected,
                 "punctuated": tr.punctuated, "edge_guesses": tr.edge_guesses,
                 "uncovered": spans(asr_guard.uncovered(words, speech)),
                 "low_confidence": spans(asr_guard.low_confidence(words)), "repeats": spans(asr_guard.repeats(words))}
        if found["redecoded"]:
            log.info("heard %.0f-%.0f s: decoded speech with no words again at %s: %d words added, %d not trusted", a, b,
                     found["redecoded"], tr.recovered, tr.rejected)
        if found["uncovered"]:
            log.info("heard %.0f-%.0f s: diarized speech still without words at %s", a, b, found["uncovered"])
        if found["low_confidence"] or found["repeats"]:
            log.info("heard %.0f-%.0f s: low-confidence words at %s, repeated phrases at %s", a, b,
                     found["low_confidence"], found["repeats"])
        return found

    @staticmethod
    def _sentence_cut(words: list, b: float, last: bool) -> float:
        """Where this chunk ends: after the last sentence-final word ending by b; else after the first one ending in the
        pad after b that another word follows (the audio didn't stop mid-sentence there), so a sentence running over
        b isn't cut (ARCHITECTURE §3.4); else at the longest pause; else b. Sentence ends are the segmenter's, which
        reads the next word ("at 9 a.m. with" runs on)."""
        if not words:
            return b
        if last:
            return words[-1].end
        nexts = [w.text for w in words[1:]] + [None]
        for k in reversed(range(len(words))):
            if words[k].end <= b and sentence_end(words[k].text, nexts[k]):
                return words[k].end
        for w, nxt in zip(words, words[1:]):  # another word follows it: the audio didn't stop mid-sentence there
            if w.end > b and sentence_end(w.text, nxt.text):
                return w.end
        inside = [w for w in words if w.end <= b]
        best, gap = None, 0.4
        for w0, w1 in zip(inside, inside[1:]):
            if w1.start - w0.end > gap:
                best, gap = w0.end, w1.start - w0.end
        return best if best is not None else (inside[-1].end if inside else b)

    # ---- translating (the Claude CLI; ARCHITECTURE §4) ----------------------------------------------------------
    def _speech(self, u: SourceUnit) -> float:
        """Speech time inside the line's span (§4.3): its speaker's diarized turns there, or its words' own durations when
        those say more; the span when neither says anything."""
        span = max(u.end - u.start, 0.0)
        turns = sum(t.end - t.start for t in self.registry.turns_in(u.start, u.end) if t.speaker == u.speaker)
        words = sum(w.end - w.start for w in u.words)
        return min(max(turns, words), span) or span

    def _k(self, speaker: str) -> float:
        got = self._ratios.get(speaker, ())
        return statistics.median(got) if len(got) >= K_MIN_LINES else K_PRIOR

    def _target(self, st: UnitState) -> float:
        """The line's length target in aksharas: what the voice says in its speech time at its calibrated pace (§4.3) after
        its overhead, so a wording written to it is predicted at the speech time by the band rule (§4.5). (§4.3 leaves the
        calibrated overhead out, which would put such a `full` above the band on any line under ten times the overhead.)"""
        key = self._key(st.unit.speaker)
        return max(st.speech_s - self.estimator.overhead(key), 0.0) * self.estimator.rate(key)

    def _line_spec(self, st: UnitState, want: tuple[str, ...], **call: object) -> LineSpec:
        """The line as a Claude call sees it: its sizes, the tiers asked for, its breaks (where each pause of 1 s or more
        in it starts) and whether the speaker was cut off (§3.4, §4.4). `call`: the fields of one call type."""
        u = st.unit
        return LineSpec(u.id, u.speaker, u.text, u.start, u.end, round(st.speech_s, 2), round(self._target(st), 1), want,
                        breaks=tuple(round(u.words[k - 1].end, 2) for k in u.breaks), cut_off=u.cut_off, **call)

    def _spec(self, st: UnitState) -> LineSpec:
        """The line as a scene call sees it, with the tiers its predicted length asks for, set here before the call
        (§4.3): `full` always; `concise` and `very_concise` when the predicted `full` runs past 1.10 x the target;
        `fuller` when it falls under 0.85 x a speech-dense slot."""
        u, target = st.unit, self._target(st)
        st.pred_full = self._k(u.speaker) * mixed_units(u.text)
        want: tuple[str, ...] = ("full",)
        if st.pred_full > BAND[1] * target:
            want = ("full", "concise", "very_concise")
        elif st.pred_full < BAND[0] * target and st.speech_s >= DENSE * (u.end - u.start):
            want = ("full", "fuller")
        return self._line_spec(st, want)

    def _tts_text(self, w: Wording) -> str:
        return tenglish.latin_spoken(w.spoken, w.english) if self.tts_script == "latin" else w.spoken

    def _pick(self, st: UnitState, line: LineResult, key: VoiceKey | None = None) -> tuple[str, bool]:
        """The band rule (§4.5) over `line`'s wordings: the most complete one whose predicted duration is within
        [0.85, 1.10] x the line's speech time; else the closest from above, if the planner absorbs it within the speed
        cap and the lag limit; else the closest from below (from above when nothing is below). Durations are predicted
        from the Telugu script, whichever script the TTS reads, for the voice `key` (by default the speaker's voice now).
        Returns (the tier, whether it fits: in the band, or absorbed)."""
        key = key or self._key(st.unit.speaker)
        if approved_full(line):
            pred = self.estimator.estimate(line.full.spoken, key)
            fits = BAND[0] * st.speech_s <= pred <= BAND[1] * st.speech_s or self._absorbs(st, pred)
            return "full", fits  # timing cannot choose the shorter wording that the same review found incomplete
        tiers = [t for t in TIERS_BY_FULLNESS if t in line.tiers]
        pred = {t: self.estimator.estimate(line.tiers[t].spoken, key) for t in tiers}
        lo, hi = BAND[0] * st.speech_s, BAND[1] * st.speech_s
        pick = next((t for t in tiers if lo <= pred[t] <= hi), None)
        if pick is not None:
            return pick, True
        above = min((t for t in tiers if pred[t] > hi), key=pred.__getitem__, default=None)
        below = max((t for t in tiers if pred[t] < lo), key=pred.__getitem__, default=None)
        fits = above is not None and self._absorbs(st, pred[above])
        return (above if fits or below is None else below), fits

    def _choose(self, st: UnitState, key: VoiceKey | None = None) -> bool:
        """Make the band rule's pick (`_pick`) the line's wording, the one the TTS says. Returns whether it fits."""
        st.tier, fits = self._pick(st, st.line, key)
        st.telugu = self._tts_text(st.line.tiers[st.tier])
        return fits

    def _absorbs(self, st: UnitState, duration: float) -> bool:
        """Whether the planner would place `duration` s for the line within the speed cap and the lag limit, with no
        freeze: priced as `_plan` would place it now, with the same lookahead (the next line's slot decides its lag
        limit)."""
        try:
            _, plan = self.planner.evaluate(self._slot(st), duration, ahead=self._ahead(st))
        except ValueError:  # placed already
            return False
        return plan.freeze <= 0.0 and plan.overdraft <= 1e-9

    def _fit_spec(self, st: UnitState) -> LineSpec | None:
        """A fit for a line whose pick is outside the band (§4.2), for wordings on the side the band is. Over it: the
        shorter tiers the line lacks, cut from the pick. Under it, with longer tiers over it (the band falls between two
        of its wordings): the tiers between those two that it lacks, else the pick again, each cut from the longer one,
        which the fit gets as its `current` (it copies that into `full`, and a tier it gives must be shorter). When the
        pick is `full`, which a fit never rewrites, a `fuller` closer to the slot instead, from the pick. Under it with
        nothing longer: `fuller`, for a speech-dense slot."""
        u, line = st.unit, st.line
        if approved_full(line):
            return None  # preserve the complete wording; a new unreviewed fit cannot reverse that decision
        start = line.tiers[st.tier]  # the fit's `current`: the wording its tiers are made from
        if self.estimator.estimate(start.spoken, self._key(u.speaker)) > BAND[1] * st.speech_s:
            want = tuple(t for t in ("concise", "very_concise") if t not in line.tiers)
        elif longer := [t for t in TIERS_BY_FULLNESS[:TIERS_BY_FULLNESS.index(st.tier)] if t in line.tiers]:
            if st.tier == "full":
                want = ("fuller",)
            else:
                k = TIERS_BY_FULLNESS.index(longer[-1])
                want = TIERS_BY_FULLNESS[k + 1:TIERS_BY_FULLNESS.index(st.tier)] or (st.tier,)
                start = line.tiers[longer[-1]]
        elif "fuller" not in line.tiers and st.speech_s >= DENSE * (u.end - u.start):
            want = ("fuller",)
        else:
            want = ()
        if not want:
            return None
        return self._line_spec(st, want, current=start.spoken,
                               overflow=round(count_units(start.spoken) - self._target(st), 1))

    def _learn_k(self, st: UnitState) -> None:
        """k (§4.3): the line's `full` in aksharas, counted as every length is (`count_units`), per English syllable."""
        syllables = mixed_units(st.unit.text)
        if syllables >= K_MIN_SYLLABLES:
            got = self._ratios.setdefault(st.unit.speaker, [])
            got.append(count_units(st.line.full.spoken) / syllables)
            del got[:-K_WINDOW]

    async def _scene(self, req: SceneRequest) -> None:
        """One scene call and what comes of it: its lines' coverage review, after which they count as translated. The
        dub loop never waits on this (§4.9)."""
        t0, asked = time.perf_counter(), time.monotonic()
        try:
            res = await self.tr.submit(req)
        except asyncio.CancelledError:
            self._release(req)
            if _cancelling():
                raise
            log.info("scene %d dropped: its call was cancelled", req.scene)
            self._trace({"event": "scene", "scene": req.scene, "dropped": True})
            return
        except ClaudeCLIError as err:
            self._release(req)
            await self._claude_failed(err)
            return
        except Exception as exc:  # a bug, or the cache unwritable: the lines are asked for again after a back-off
            log.exception("scene %d failed", req.scene)
            self._release(req)
            await self._claude_failed(ClaudeCLIError("failed", f"Translation failed: {exc}"))
            return
        if res.calls:
            await self._claude_ok(asked)
        seconds = time.perf_counter() - t0
        got: dict[int, LineResult] = {}
        for spec in req.lines:
            st = self.units.get(spec.id)
            if st is None or st.voiced:
                continue
            line = res.lines.get(spec.id)
            if line is None:  # the translator asked again, then line by line: never padded, never English (§4.9)
                await self._skip(st, f"not translated: {res.skipped.get(spec.id, 'no answer')}")
                continue
            got[spec.id] = line
        reviewed = await self._review(req, got)
        if reviewed is None:
            self._release(req)
            log.info("scene %d dropped during its review", req.scene)
            self._trace({"event": "scene", "scene": req.scene, "dropped": True})
            return
        done: list[UnitState] = []
        fits: list[LineSpec] = []
        for lid, line in reviewed.items():
            st = self.units.get(lid)
            if st is None or st.voiced:
                continue
            st.line, st.model, st.cache_hit, st.prompt_hash = line, line.model, line.cached, self.tr.prompt_hash
            st.translate_s, st.translate_ready_at = round(seconds, 3), round(time.time(), 3)
            self._learn_k(st)
            if not self._choose(st) and (fit := self._fit_spec(st)) is not None:
                fits.append(fit)
            done.append(st)
        if fits:
            self._side(self._fit(SceneRequest(req.scene, tuple(fits), "fit")))
        mix = tenglish.cmi((s.line.tiers[s.tier].spoken, s.line.tiers[s.tier].english) for s in done)
        # Legacy trace field: now it counts reviewed Latin substitutions only, not all Telugu-spelled English loans.
        # There is no preferred mixing quota, and declined/unreviewed substitutions must not trigger a quality warning.
        self._trace({"event": "scene", "scene": req.scene, "a": round(req.lines[0].start, 2),
                     "b": round(req.lines[-1].end, 2), "lines": len(req.lines), "translated": len(done),
                     "cached": sum(1 for s in done if s.cache_hit), "fits": len(fits), "cmi": round(mix, 1),
                     "context_te": sum(1 for _, te in req.context_before if te), "brief": self.tr.brief.version,
                     **self._classes(done),
                     "wall_s": round(seconds, 2), "ready_s": round(time.perf_counter() - t0, 2)})
        self._wake.set()

    async def _review(self, req: SceneRequest, lines: dict[int, LineResult]) -> dict[int, LineResult] | None:
        """The coverage review of a scene's lines before they count as translated (§4.6): of the wording the band rule
        picks for each now, with one re-translation of any it finds a phrase missing from or a meaning error in. None
        when its call was cancelled (or the job paused): the lines are asked for again (from the line cache). A review
        that fails leaves the lines unreviewed, never untranslated, so the dub loop never waits on Claude for them; a
        failure the user must fix is shown and holds new scene calls like any other; one that goes through clears it,
        as a scene call does."""
        if not lines:
            return lines
        def safe_wordings(answer: dict[int, LineResult]) -> dict[int, LineResult]:
            # The Latin spelling is an optional pronunciation aid. Only a semantic review of that exact tier may
            # authorize it; unexpected reviewer failures must still preserve the full Telugu wording.
            return {i: replace(line, tiers={tier: w if line.coverage is not None
                                            and line.coverage.by == "review" and line.coverage.cls in ("C", "m")
                                            and line.coverage.tier == tier else Wording(w.spoken)
                                            for tier, w in line.tiers.items()}) for i, line in answer.items()}
        chosen = {i: self._pick(self.units[i], line)[0] for i, line in lines.items()}
        def fallback_ok(lid: int) -> bool:
            st, line = self.units.get(lid), lines.get(lid)
            return (st is not None and line is not None and not (st.voicing or st.voiced)
                    and chosen.get(lid) != "full" and line.full != line.tiers.get(chosen.get(lid))
                    and self._absorbs(st, self.estimator.estimate(line.full.spoken, self._key(st.unit.speaker))))
        fallback_kw = ({"fallbacks": {i for i in lines if fallback_ok(i)}, "fallback_check": fallback_ok}
                       if getattr(self.tr, "supports_full_fallback", False) else {})
        asked = time.monotonic()
        try:
            res = await self.tr.review(req, lines, chosen, **fallback_kw)
        except asyncio.CancelledError:
            if _cancelling():
                raise
            return None
        except Exception:
            log.exception("the review of scene %d failed; its lines go on unreviewed", req.scene)
            return safe_wordings(lines)
        if res.error is not None:
            await self._claude_failed(res.error)
        elif res.calls:
            await self._claude_ok(asked)
        return safe_wordings(res.lines)

    def _release(self, req: SceneRequest) -> None:
        """A scene call that ended without an answer: its lines go back to be asked for again."""
        for spec in req.lines:
            st = self.units.get(spec.id)
            if st is not None and st.line is None:
                st.scene = None
        self._wake.set()

    async def _fit(self, req: SceneRequest) -> None:
        """New tiers for lines outside the band (§4.2), folded in if the dub loop hasn't started on their line (whose
        tier must stay the wording it voices: the next scene's context, its take row and the manifest all read it)."""
        try:
            res = await self.tr.submit(req)
        except asyncio.CancelledError:
            if _cancelling():
                raise
            return
        except ClaudeCLIError as err:
            log.warning("fit for scene %d failed (%s): %s", req.scene, err.kind, err)
            return
        for lid, line in res.lines.items():
            st = self.units.get(lid)
            if st is not None and st.line is not None and not (st.voicing or st.voiced):
                if approved_full(st.line):
                    continue  # a fit submitted before the complete fallback was approved may arrive afterwards
                st.line = line
                self._choose(st)
        self._wake.set()

    async def _claude_failed(self, err: ClaudeCLIError) -> None:
        """A failure the translator leaves to the user (not signed in, a usage limit, the CLI missing or too old, a
        stall or throttle that outlasted its retries): new scene calls wait, the UI says what to do, and translation
        tries again later. Lines already translated keep being voiced; rephrases under way are dropped (their lines
        settle as voiced, flagged long) rather than hold the translator's slots through the hold."""
        self._claude_failed_at = time.monotonic()
        if self.tr is not None:
            self.tr.cancel(lambda r: r.call == "rephrase")
        if time.monotonic() < self._claude_hold and err.kind == self._claude_kind:
            return  # the same failure from another call already under way
        self._claude_fails += 1
        self._claude_kind = err.kind
        wait = min(CLAUDE_BACKOFF * 2 ** (self._claude_fails - 1), CLAUDE_BACKOFF_MAX)
        if err.kind == "usage_limit" and err.resets_at:
            wait = max(wait, err.resets_at - time.time() + 5.0)
        self._claude_hold = time.monotonic() + wait
        log.warning("Translation %s: %s; the next scene call in %.0f s", err.kind, err, wait)
        await self.notify({"type": "claude_error", "kind": err.kind, "message": str(err), "limit": err.limit,
                              "resetsAt": err.resets_at, "retryIn": round(wait)})

    async def _claude_ok(self, asked: float) -> None:
        """A scene call asked at `asked` (monotonic s) went through Claude. The UI's Claude banner (a problem from hello,
        the job before or a failure here) is cleared on the first such call of the job and on the first after a
        failure. A call asked before the last failure says nothing about it: a usage limit hit meanwhile still holds."""
        if asked < self._claude_failed_at or (self._claude_said_ok and not self._claude_fails):
            return
        self._claude_fails, self._claude_kind, self._claude_said_ok = 0, None, True
        await self.notify({"type": "claude_ok"})

    def _side(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._side_tasks.add(task)
        task.add_done_callback(_side_done)
        task.add_done_callback(self._side_tasks.discard)
        return task

    def _context_for(self, st: UnitState) -> list[tuple[str, str]]:
        """The two lines before, as (English, the wording chosen in Telugu script): what `tenglish.lint` compares."""
        before = sorted((s for s in self.units.values() if s.line and s.tier and s.unit.start < st.unit.start),
                        key=lambda s: s.unit.start)
        return [(s.unit.text, s.line.tiers[s.tier].spoken) for s in before[-2:]]

    def _slot(self, st: UnitState) -> LineSlot:
        """The line as the planner places it (§3.10): its span; the latest end of the source before it, of the lines
        before it and of any diarized speech, so an early start goes only into real silence (timing v1: of the lines
        alone); the akshara ceiling of its voice; its hard breaks that no other line starts inside, and the other
        speech inside them; its soft anchors, and its speaker's speech."""
        u = st.unit
        prev_end = max((s.unit.end for s in self.units.values() if s.unit.start < u.start), default=None)
        spoken = None if self.timing == "v1" else self._heard_before(u)
        before = max((t for t in (prev_end, spoken) if t is not None), default=None)
        inside = [s.unit.start for s in self.units.values() if u.start < s.unit.start < u.end]
        breaks = tuple((a, b) for a, b in self._pauses_at(u, u.breaks) if not any(a <= t <= b for t in inside))
        return LineSlot(u.id, u.speaker, u.start, u.end, st.next_start, before,
                        overlaps_prev=prev_end is not None and prev_end > u.start + 0.05,
                        rate_cap=rate_ceiling(self.estimator.rate(self._key(u.speaker)), self.planner.s),
                        breaks=breaks, heard=self._heard_in(st, breaks), anchors=self._pauses_at(u, u.anchors),
                        speech=self._speech_spans(u))

    def _heard_before(self, u: SourceUnit) -> float | None:
        """Where diarized speech just before the line ends (any speaker; the line's own speaker only in a turn that
        ended before it, since a turn running into the line is the line's own start, heard before its first word)."""
        s = self.planner.s
        turns = self.registry.turns_in(u.start - s.lead_max - s.guard - 0.5, u.start, exclusive=False)
        return max((t.end for t in turns if t.speaker != u.speaker or t.end < u.start - 1e-6), default=None)

    def _heard_in(self, st: UnitState, breaks: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        """Other speech inside the line's hard breaks, clipped to them: another speaker's diarized turn, or another line
        running into the pause. A piece after a break starts early only into the silence left (as `_heard_before`)."""
        u, out = st.unit, set()
        for a, b in breaks:
            spans = [(t.start, t.end) for t in self.registry.turns_in(a, b, exclusive=False) if t.speaker != u.speaker]
            spans += [(s.unit.start, s.unit.end) for s in self.units.values() if s is not st]
            out |= {(round(max(x, a), 3), round(min(y, b), 3)) for x, y in spans if x < b and y > a}
        return tuple(sorted(out))

    @staticmethod
    def _pauses_at(u: SourceUnit, at: list[int]) -> tuple[tuple[float, float], ...]:
        """The English pauses before words[k] for each k in `at` (a unit's `breaks` or `anchors`), as (start, end)."""
        return tuple((u.words[k - 1].end, u.words[k].start) for k in at
                     if 0 < k < len(u.words) and u.words[k].start > u.words[k - 1].end)

    def _speech_spans(self, u: SourceUnit) -> tuple[tuple[float, float], ...]:
        """The line's speaker's diarized speech turns inside its span (the speech-level metrics' yardstick, §3.10), else
        the span. Overlapped speech counts for every speaker in it: a speaker whose last words run under an interjection
        ends where they stop, not where the interjection starts."""
        turns = tuple((round(t.start, 3), round(t.end, 3))
                      for t in self.registry.turns_in(u.start, u.end, exclusive=False) if t.speaker == u.speaker)
        return turns or ((u.start, u.end),)

    # ---- voicing ------------------------------------------------------------------------------------------------
    async def _synth(self, text: str, voice: object, cap: float, n: int, cost: VoiceCost) -> list[tuple[object, float, bool]]:
        """One synthesis call at the voice's natural pace: `n` takes in one batched decode where the TTS can
        (`synthesize_takes`), else one mel take (vocoded later at the planned rate), else plain samples. Returns each
        take with its natural duration and whether it ran to its cap."""
        tts = self.b.tts
        priority = self._gpu_priority()
        cost.priority = priority if cost.priority is None else cost.priority
        if hasattr(tts, "synthesize_takes"):
            takes, seconds = await self._on_gpu(tts.synthesize_takes, text, voice, "te", cap, n, cost=cost, priority=priority)
            cost.add_takes(takes, seconds)
            return [(t, t.seconds, bool(t.capped)) for t in takes]
        if hasattr(tts, "synthesize_mel"):
            take, seconds = await self._on_gpu(tts.synthesize_mel, text, voice, "te", cap, cost=cost, priority=priority)
            cost.add_take(take, seconds)
            return [(take, take.seconds, bool(getattr(take, "capped", False)))]
        samples, seconds = await self._on_gpu(tts.synthesize, text, voice, "te", cap, cost=cost, priority=priority)
        cost.add_take(samples, seconds)
        return [(samples, len(samples) / tts.sample_rate, False)]

    async def _take(self, st: UnitState, w: Wording, voice: object, key: VoiceKey, seconds: float, cost: VoiceCost,
                    n: int = 1, speech: float | None = None) -> tuple[object, float, float | None, str | None]:
        """Voice wording `w` of the line (ARCHITECTURE §3.8, §3.13): `n` takes in one batched decode where the TTS batches
        them, and one more take if every one of them failed; the one voiced is picked by `qa.take` (cap
        hits and takes far shorter than the estimate out, then the closest to the line's speech time). A TTS that can't
        batch gives one take and no retake: without a seed it would say the line the same way again. Returns (the take,
        its natural duration, the mean duration of the takes that passed for the estimator (of those that didn't run to
        their cap when none passed: a voice faster than its estimate is still learned; None if all ran away), and why
        the take voiced failed, or None). `speech`: the speech time the take is picked against, for a piece of the line;
        by default the line's."""
        got, failed = await self._take_candidates(w, voice, key, seconds, cost, n)
        k = take_qa.pick([secs for _, secs, _ in got], failed, st.speech_s if speech is None else speech)
        return got[k][0], got[k][1], self._take_pace(got, failed), failed[k]

    async def _take_candidates(self, w: Wording, voice: object, key: VoiceKey, seconds: float, cost: VoiceCost,
                               n: int) -> tuple[list[tuple[object, float, bool]], list[str | None]]:
        """Keep a batch's candidates until the chosen waveform passes QA; flow still runs only for candidates tried."""
        batched = hasattr(self.b.tts, "synthesize_takes")
        n = max(n, 1) if batched else 1
        cost.takes_n = n if cost.takes_n is None else cost.takes_n
        text = self._tts_text(w)
        cap = min(MAX_LINE_SECONDS, max(4.0, seconds * self.speed_cap * 2.0))
        estimate = self.estimator.estimate(w.spoken, key)
        got = await self._synth(text, voice, cap, n, cost)
        failed = [take_qa.failure(secs, capped, estimate) for _, secs, capped in got]
        if batched and all(failed):  # the retake ladder's second rung (§3.13)
            self._retakes += 1
            cost.retakes += 1
            more = await self._synth(text, voice, cap, 1, cost)
            got += more
            failed += [take_qa.failure(secs, capped, estimate) for _, secs, capped in more]
        cost.failures += [f for f in failed if f]
        return got, failed

    @staticmethod
    def _take_pace(got: list[tuple[object, float, bool]], failed: list[str | None]) -> float | None:
        # Duration-only "short" failures can teach a newly calibrated voice's true pace. Broken waveforms cannot.
        usable = ([secs for (_, secs, _), f in zip(got, failed) if f is None]
                  or [secs for (_, secs, capped), f in zip(got, failed) if not capped and f == "short"])
        return sum(usable) / len(usable) if usable else None

    async def _render(self, take: object, rate: float, cost: VoiceCost) -> np.ndarray:
        """The take voiced, rendered at the planned rate: vocoded (after its S3Gen flow, which counts as synthesis), or
        time-stretched."""
        tts = self.b.tts
        if hasattr(tts, "vocode"):
            flowed = getattr(take, "flow_s", 0.0)
            audio, seconds = await self._on_gpu(tts.vocode, take, rate, cost=cost)
            cost.render_s += seconds - cost.add_flow(flowed, take)
        else:
            t0 = time.perf_counter()
            samples = np.asarray(take, np.float32)
            audio = await asyncio.to_thread(wsola, samples, rate, tts.sample_rate) if abs(rate - 1.0) > 1e-3 else samples
            cost.render_s += time.perf_counter() - t0
        return audio

    async def _say(self, st: UnitState, words: list[Wording], voice: object, key: VoiceKey, seconds: float,
                   cost: VoiceCost, n: int = 1) -> Said:
        """Voice the line as `words`, one take each (`_take`): its wording, or its pieces at hard breaks (each picked
        against its share of the speech time). Each take is rendered at its natural pace, where its pauses are found."""
        said = Said()
        pred = [self.estimator.estimate(w.spoken, key) for w in words]
        for w, p in zip(words, pred):
            share = p / sum(pred) if len(words) > 1 and sum(pred) > 0 else 1.0
            take, dur, natural, pauses, pace, failed = await self._said_take(st, w, voice, key, seconds, cost, n,
                                                                             st.speech_s * share)
            said.wordings.append(w)
            said.takes.append(take)
            said.seconds.append(dur)
            said.audio.append(natural)
            said.pauses.append(pauses)
            said.paces.append(pace)
            said.failures.append(failed)
            said.failed = said.failed or failed
        return said

    async def _said_take(self, st: UnitState, w: Wording, voice: object, key: VoiceKey, seconds: float,
                         cost: VoiceCost, n: int, speech: float
                         ) -> tuple[object, float, np.ndarray | None, tuple[tuple[float, float], ...], float | None,
                                    str | None]:
        """One wording of `_say`: select against `speech` s, render at its natural pace, and check its waveform before
        finding pauses. Only broken audio tries other candidates, then at most one fresh batched take. Returns (take,
        natural seconds, audio, pauses, the pace learned, duration failure); no usable audio raises UnusableAudioError.
        """
        got, failed = await self._take_candidates(w, voice, key, seconds, cost, n)
        pending = list(range(len(got)))
        retried = False
        while pending:
            k = pending.pop(take_qa.pick([got[i][1] for i in pending], [failed[i] for i in pending], speech))
            take, dur, _ = got[k]
            natural = await self._render(take, 1.0, cost)
            reason = take_qa.audio_failure(natural)
            if reason is None:
                return (take, dur, natural, pz.pauses(natural, self.b.tts.sample_rate),
                        self._take_pace(got, failed), failed[k])
            failed[k] = reason
            cost.failures.append(reason)
            log.warning("unit %d: rejected a take (%s)", st.unit.id, reason)
            # Exhaust the existing candidates first. Only stochastic/batched engines get one fresh retry;
            # deterministic engines would return the same broken output. Never loop indefinitely on a bad model.
            if not pending and not retried and hasattr(self.b.tts, "synthesize_takes"):
                retried = True
                self._retakes += 1
                cost.retakes += 1
                cap = min(MAX_LINE_SECONDS, max(4.0, seconds * self.speed_cap * 2.0))
                more = await self._synth(self._tts_text(w), voice, cap, 1, cost)
                pending.extend(range(len(got), len(got) + len(more)))
                got += more
                reasons = [take_qa.failure(secs, capped, self.estimator.estimate(w.spoken, key))
                           for _, secs, capped in more]
                failed += reasons
                cost.failures += [f for f in reasons if f]
        raise take_qa.UnusableAudioError(
            f"Could not generate usable audio for line {st.unit.id + 1}. Resume to retry this line. "
            f"Audio checks: {', '.join(sorted(set(f for f in failed if f)))}.")

    def _learn(self, said: Said, key: VoiceKey) -> None:
        for w, pace in zip(said.wordings, said.paces):
            if pace is not None:
                self.estimator.observe(w.spoken, key, pace)

    def _pieces(self, st: UnitState, key: VoiceKey) -> list[Wording]:
        """The line's pieces (§3.10 step 2), each with its part of the English map, to be voiced one take each: when its
        wording is `full`, Claude split it where Telugu naturally pauses (§4.4), and placed piece by piece at their
        predicted lengths each piece meets its hard break. Else [] (the line is said whole, drifting over its pauses), and
        always under timing v1 and v2-whole."""
        line = st.line
        if self.timing != "v2" or st.tier != "full" or len(line.pieces) < 2:
            return []
        slot = self._slot(st)
        if len(slot.breaks) < len(line.pieces) - 1:
            return []
        words = self._piece_words(line)
        pred = [(self.estimator.estimate(w.spoken, key), ()) for w in words]
        try:
            _, plan = self.planner.evaluate(slot, sum(d for d, _ in pred), ahead=self._ahead(st), pieces=pred)
        except ValueError:  # placed already
            return []
        return words if plan.said == "pieces" else []

    @staticmethod
    def _piece_words(line: LineResult) -> list[Wording]:
        """The line's pieces as wordings, each with its part of `full`'s English map."""
        words, at = [], 0
        for piece in line.pieces:
            k = len(piece.split())
            words.append(Wording(piece, tuple((i - at, en) for i, en in line.full.english if at <= i < at + k)))
            at += k
        return words

    def _shorter_tier(self, st: UnitState, key: VoiceKey, seconds: float) -> str | None:
        """The wording to say instead when a take runs long, from those the line already has (no Claude call: §4.9):
        the most complete shorter one predicted within the band of `seconds`, else the shortest; None when it has none
        shorter than its tier."""
        size = lambda t: count_units(st.line.tiers[t].spoken)  # noqa: E731
        if approved_full(st.line):
            return None
        shorter = [t for t in st.line.tiers if size(t) < size(st.tier)]
        if not shorter:
            return None
        fit = [t for t in shorter if self.estimator.estimate(st.line.tiers[t].spoken, key) <= BAND[1] * seconds]
        return max(fit, key=size, default=min(shorter, key=size))

    def _rephrase_tier(self, line: LineResult, key: VoiceKey, take_s: float, plan: Plan) -> str | None:
        """The wording of rephrase `line` to voice in place of a take of `take_s` s placed as `plan` (§3.13): the most
        complete one predicted to end inside the line's window, else the shortest one predicted shorter than the take;
        None when there is none."""
        pred = {t: self.estimator.estimate(line.tiers[t].spoken, key)
                for t in ("full", "concise", "very_concise") if t in line.tiers}
        room = take_s - plan.excess * plan.rate  # natural seconds that would end inside the window
        return next((t for t in pred if pred[t] <= room),
                    min((t for t in pred if pred[t] < take_s), key=pred.__getitem__, default=None))

    def _shortened(self, plan: Plan, duration: float, pauses: tuple[tuple[float, float], ...] = (),
                   next_start: float | None = None) -> Plan:
        """The plan of another take of the line in the place of the one placed (a rephrase, shorter; the render's
        re-translation, of any length): the same start and rate, so the lines placed after it keep their places. What a
        shorter one saves comes off the freeze first; a hold left under the planner's `min_freeze` would stutter on the
        player, so it is that long (never longer than before). A longer one gets no more freeze than the one placed (a
        freeze can't be priced here against the rolling budget): what it adds runs over. It is said whole, even where
        the one placed was said in parts. `pauses`: the take's (where it is voiced). `next_start`: the next line's onset,
        which bounds the window of a take that ended inside it (for a longer take).
        Still long (`needs_shorter`): a shorter take of a line that was, while it needs a freeze or an overdraft or runs
        past its window (ending inside it, it delays the next line by at most `min_gap`, under `shorter_tol`); a longer
        one, when the one placed was, or it runs over the next line's lag limit or `shorter_tol` past its window."""
        s = self.planner.s
        wall = duration / plan.rate
        saved = plan.wall - wall
        freeze = max(plan.freeze - saved, 0.0) if saved >= 0 else plan.freeze
        if 0.0 < freeze < s.min_freeze:
            freeze = s.min_freeze
        over = plan.freeze + plan.overdraft  # how far the take placed ran past the next line's lag limit, where it did
        overdraft = max(over - saved - freeze, 0.0) if over > EPS else 0.0
        if plan.excess > EPS:  # where its window ends: known for a take that ran past it
            soft = plan.start + plan.wall - plan.excess
        else:
            soft = INF if next_start is None else max(plan.start + plan.wall, next_start - s.guard)
        excess = max(plan.start + wall - soft, 0.0)
        if saved >= 0:
            long = plan.needs_shorter and (freeze > 0.0 or overdraft > EPS or excess > EPS)
        else:
            long = plan.needs_shorter or overdraft > EPS or excess > s.shorter_tol
        got = replace(plan, wall=wall, freeze=freeze, freeze_at=plan.freeze_at if freeze else None, overdraft=overdraft,
                      needs_shorter=long, excess=excess, parts=(), said="whole")
        return replace(got, voiced=voiced(got, [(duration, pauses)], s.keep_pause))

    async def _squeezed(self, said: Said, k: int, rate: float, cost: VoiceCost) -> np.ndarray:
        """Take `k` played `rate` times faster (§3.10 step 7): its inner pauses shorten first, each to no less than
        `keep_pause`, and only what they can't give speeds its speech up (`pauses.squeeze`)."""
        sr = self.b.tts.sample_rate
        r, cuts = pz.squeeze(said.pauses[k], said.seconds[k], rate, self.planner.s.keep_pause)
        # Its natural audio where it is in memory; a take read back from its file (OFFLINE-RENDER §2.12) is vocoded again.
        natural = said.audio[k] if k < len(said.audio) else None
        audio = natural if abs(r - 1.0) <= 1e-3 and natural is not None else await self._render(said.takes[k], r, cost)
        return pz.cut(audio, sr, r, cuts, round(said.seconds[k] / rate * sr)) if cuts else audio

    async def _play(self, said: Said, plan: Plan, cost: VoiceCost) -> np.ndarray:
        """The line's audio as `plan` places it: its take squeezed to the planned rate, or for a line said in parts
        (pieces at hard breaks, a take split at soft anchors) each part at its own time, with silence between."""
        if not plan.parts:
            return await self._squeezed(said, 0, plan.rate, cost)
        sr = self.b.tts.sample_rate
        out = np.zeros(round(plan.wall * sr), np.float32)
        rendered: dict[tuple[int, float], np.ndarray] = {}
        for p in plan.parts:
            if (p.take, p.rate) not in rendered:
                rendered[p.take, p.rate] = await self._squeezed(said, p.take, p.rate, cost)
            r, cuts = pz.squeeze(said.pauses[p.take], said.seconds[p.take], p.rate, self.planner.s.keep_pause)
            a = round(pz.out_time(p.a, r, cuts) * sr)
            seg = rendered[p.take, p.rate][a:a + round(p.wall * sr)]
            at = min(round((p.start - plan.start) * sr), len(out))
            out[at:at + len(seg)] = seg[:len(out) - at]
        return out

    def _predicted(self, st: UnitState) -> float | None:
        """A line's duration: its take's once it has one (what the render's final plan prices, OFFLINE-RENDER §2.11),
        else predicted once it has a wording."""
        if st.take_s is not None:
            return st.take_s
        return (self.estimator.estimate(st.line.tiers[st.tier].spoken, self._key(st.unit.speaker))
                if st.line and st.tier else None)

    def _window(self, st: UnitState) -> list[UnitState]:
        """The lines the planner looks at after `st` (§3.10 step 5): the next one, and up to `window_lines` in all that
        start within `window_s` of it."""
        s, u = self.planner.s, st.unit
        later = sorted((x for x in self.units.values() if x.unit.start > u.start), key=lambda x: (x.unit.start, x.unit.id))
        return [x for k, x in enumerate(later[:s.window_lines]) if k == 0 or x.unit.start <= u.start + s.window_s]

    def _ahead(self, st: UnitState, planner: TimelinePlanner | None = None) -> tuple[tuple[LineSlot, float | None], ...]:
        """What `planner` (by default the one lines are voiced on) knows of the lines after `st`: each line of its
        window (`_window`) with its slot and its duration (`_predicted`; None until it has a wording)."""
        return tuple((self._slot(x), self._predicted(x)) for x in self._window(st))

    def _plan(self, st: UnitState, said: Said, planner: TimelinePlanner | None = None) -> Plan:
        """Place the line's takes on `planner` (by default the one lines are voiced on): pieces each at its break, else
        one take, which may wait at a soft anchor near one of its text's breaks."""
        planner = planner or self.planner
        slot, ahead = self._slot(st), self._ahead(st, planner)
        if len(said.takes) > 1:
            return planner.place(slot, said.total, ahead=ahead, pieces=said.timed())
        return planner.place(slot, said.total, ahead=ahead, pauses=said.pauses[0], marks=_marks(said.wordings[0].spoken))

    def _timing(self, slot: LineSlot, plan: Plan) -> dict:
        """units.jsonl's timing fields for a voiced line (§3.10 metrics): how it was said and its ceiling, its speaker's
        speech and where the dub is voiced (video time), and per line the speech-level end error, overlap and the onset
        error of each part timed against an English onset."""
        row = {"speech": [list(x) for x in slot.speech], "voiced": [list(x) for x in plan.voiced],
               "anchor_errors": anchor_errors(plan)}
        return {"said": plan.said, "parts": len(plan.parts), "rate_cap": round(slot.rate_cap, 3),
                "hard_breaks": len(slot.breaks), **row, **line_metrics(row)}

    @staticmethod
    def _classes(lines: list[UnitState]) -> dict:
        """A scene's lines by coverage class, each counted only where its class is of the wording chosen for it
        (`voiced_class`): the class counts, then the lines classed on another wording and the unreviewed ones."""
        got = [voiced_class(st.line.coverage, st.tier) for st in lines]
        return {"coverage": {k: got.count(k) for k in CLASSES}, "other_tier": got.count("other_tier"),
                "unreviewed": got.count("unreviewed")}

    @staticmethod
    def _fill(st: UnitState) -> dict:
        """How the line's take fills its speech time (§4.5, §7 step 4): the seconds it plays for over the speech seconds,
        and the audio rate it would need to say it in exactly the speech time."""
        if not st.speech_s or st.plan is None or st.take_s is None:
            return {"speech_fill": None, "required_rate": None}
        return {"speech_fill": round(st.plan.played / st.speech_s, 3), "required_rate": round(st.take_s / st.speech_s, 3)}


class _Wake:
    """Wakes the loops idling in `RenderJob._idle` (the dub loop, the final plan). Each waits on an event of its own, so
    one going idle can't clear news meant for another that is still busy (which would then sleep out its wait)."""

    def __init__(self) -> None:
        self.events: dict[object, asyncio.Event] = {}

    def set(self) -> None:
        for ev in self.events.values():
            ev.set()


def _cancelling() -> bool:
    """Whether the running task itself is being cancelled (the engine stopping), as opposed to a translator request it
    awaits being cancelled under it (a rephrase dropped after a Claude failure)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _side_done(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("background task failed", exc_info=task.exception())


def _marks(spoken: str) -> tuple[float, ...]:
    """Where a wording breaks inside (a comma, a sentence end: not its final mark), as shares of its aksharas."""
    total = count_units(spoken)
    end = len(spoken.rstrip())
    return tuple(round(count_units(spoken[:m.end()]) / total, 3) for m in MARKS.finditer(spoken)
                 if m.end() < end) if total > 0 else ()


def _preset_name(sid: str) -> str:
    n = int(sid.lstrip("S") or 1) if sid.lstrip("S").isdigit() else 1
    return "preset_m" if n % 2 else "preset_f"


def _save_pcm(path: Path, audio: np.ndarray) -> None:
    """Write a line's PCM (float16) whole, then put it in place: a reader never sees a file half written, whether the
    line is new or a replacement."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, audio.astype(np.float16))
    os.replace(tmp, path)


def _audio_hash(*clips: np.ndarray) -> str:
    h = hashlib.sha256()
    for clip in clips:
        h.update(np.ascontiguousarray(clip, np.float32).tobytes())
    return h.hexdigest()[:16]


def scene_cut(run: list[SourceUnit], *, max_seconds: float | None = None, max_units: int | None = None) -> int:
    """How many lines of `run` (lines in time order, from where the next scene starts to the video's end, or at least
    one past its line limit) the next scene takes (ARCHITECTURE §4.1, OFFLINE-RENDER §2.8): cut at the last speaker turn
    or pause of SCENE_PAUSE s or more in its second half, else at the limit. One line longer than a scene stays whole,
    and the rest of the video under the limits is one. Explicit caps support controlled latency comparisons; omitted
    caps keep the production SCENE_MAX_S/SCENE_MAX_UNITS defaults."""
    max_seconds = SCENE_MAX_S if max_seconds is None else max_seconds
    max_units = SCENE_MAX_UNITS if max_units is None else max_units
    if not 0 < max_seconds < float("inf") or not isinstance(max_units, int) or max_units < 1:
        raise ValueError("scene limits must be a positive finite duration and a positive integer line count")
    s0 = run[0].start
    fit = 0
    while fit < min(len(run), max_units) and run[fit].end - s0 <= max_seconds:
        fit += 1
    if fit == 0:
        return 1  # one sentence longer than a scene
    if fit == len(run) and fit < max_units:
        return fit  # the rest of the video
    for i in range(fit - 1, 0, -1):
        a, b = run[i], run[i + 1] if i + 1 < len(run) else None
        if a.end - s0 < max_seconds / 2:
            break
        if b is None or b.speaker != a.speaker or b.start - a.end >= SCENE_PAUSE:
            return i + 1
    return fit
