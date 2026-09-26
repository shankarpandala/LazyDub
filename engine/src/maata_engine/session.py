"""SessionController (§6.8, ADR-016, ADR-017, ADR-019): one per video.

Four stages run as tasks and stay within `lookahead` seconds of the playhead (listening also covers the whole pre-pass):
- diarizer: speaker pre-pass over the first PREPASS seconds (in blocks), then keeps diarizing ahead of ASR.
  Every speaker found in the pre-pass is cloned (and their pace calibrated) before the first line is voiced.
- frontend: ASR over sentence-aligned chunks from a cursor, with the transcriber's punctuation and coverage guards
  (ARCHITECTURE §3.4); words get speakers from the global registry and become sentence-complete units, one per
  sentence. It hears the rest of the pre-pass window in the background for brief v1.
- translator (the Claude CLI, no GPU; docs/research/dubbing-2026-09/ARCHITECTURE.md §4): cuts the heard lines into
  scenes (a short first scene after the start or a seek, then about 60 s, then up to 150 s at a speaker turn or a
  pause) and translates them up to the lookahead horizon, up to three calls at once. Scene 1 starts from a brief made
  of the video's metadata (v0); a brief made from the pre-pass transcript (v1) is swapped in at a scene boundary. Each
  line is asked for the wordings its predicted length needs, and the one that fits its speech time is chosen locally.
  A scene's lines count as translated once the coverage review (another Claude model) has classed the chosen wordings;
  a line missing a phrase or carrying a meaning error gets one re-translation from its English first.
- voicer (MPS): synthesizes each translated line at its natural pace, as many takes in one batched decode as the
  throughput governor allows (ARCHITECTURE §5.2), and voices the one that fits its speech time best (§3.13); lets the
  timeline planner place it on the video's own clock (§3.10: a small early start into silence, a gentle steady speed-up
  that shortens the take's own pauses first, slight lag, the lines ahead weighed too; a sentence with a pause of 1 s or
  more inside is said as Claude's pieces, each at its own English onset; freezes only as a metered last resort),
  re-synthesizes once with a shorter wording it already has if it still doesn't fit, and streams it to the UI. A line
  that still runs long goes out as a provisional take while Claude rephrases it in the background; a rephrase that
  comes back in time replaces it. The voicer never waits on Claude. Audio is never cut.
MLX and MPS work take turns on the GPU through a priority scheduler shared by every session (gpu.py, ARCHITECTURE §5.3):
the voicer first while its lead is under a minute, then ASR and diarization the translator waits on, then the voicer,
then the rest. The pipeline works up to a horizon past the playhead (the lookahead, or more when the engine is too slow
for it, or while the video is held; §3.12); dub PCM is kept in memory, and sent to the UI, only near the playhead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import statistics
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from .backends.base import (SR_ANALYSIS, Backend, Brief, LineResult, LineSpec, SceneRequest, SceneTranslator, Transcript,
                            VideoMeta, Wording)
from .backends.claude_translator import LADDER, brief_v0
from .claude_cli import ClaudeCLIError
from .gpu import BACKGROUND, FRONTIER, URGENT, VOICE, GpuScheduler
from .pacing import SLOW_R, ReadyRanges, Throughput, takes_for, target_lead
from .qa import take as take_qa
from .qa.coverage import CLASSES, coverage_json, voiced_class
from .resolve import ResolveError, Resolver, ResolvedVideo
from .segment import merge_fragments, segment, sentence_end
from .speakers import SpeakerRegistry
from .text import asr_guard, tenglish
from .text.asr_fix import fix_words
from .text.akshara import count_units, mixed_units
from .timing.duration import MAX_OVERHEAD, MIN_RATE, DurationEstimator, VoiceKey
from .timing.isochrony import TimingSettings
from .timing import pauses as pz
from .timing.metrics import line_metrics
from .timing.planner import (V1, LineSlot, Plan, PlannerSettings, TimelinePlanner, anchor_errors, rate_ceiling,
                             target_seconds, voiced)
from .timing.stretch import wsola
from .types import SourceUnit, VoiceKind
from .urls import URLError, parse_youtube_url

log = logging.getLogger("maata.session")

PREPASS = 600.0          # s of video diarized (and every speaker in it cloned) before the first line is voiced
DIAR_BLOCK = 180.0       # s per diarization block
DIAR_OVERLAP = 15.0      # s shared with the previous block, for linking speakers across blocks
DIAR_AHEAD = 240.0       # keep diarization this far ahead of the ASR cursor
CHUNK = 60.0             # s of audio per ASR pass
CHUNK_PAD = 5.0          # s of extra audio after the chunk so the last sentence can finish
DEFAULT_LOOKAHEAD = 600.0
REF_TARGET = 10.0        # s of reference audio per cloned voice (S3Gen uses 10 s, T3 the first 6 s)
REF_MIN = 4.0            # s: below this a speaker keeps a preset voice until more clean speech appears
MAX_LINE_SECONDS = 30.0
RESYNTH_SHARE = 0.10   # at most this share of lines is synthesized a second time (a shorter wording, or a rephrase)
RETAKE_SHARE = 0.15    # retakes of failed batches while the governor asks one take a line (§5.2: about 15% QA retakes)
SINGLE_CLIP_MIN = 8.0   # s: a speaker's best single clean clip must be this long to be the timbre reference (ADR-017)
# Scenes (ARCHITECTURE §4.1): the first after the start or a seek ends at the first sentence end at or after 20 s (by
# 30 s), the second is about 60 s, later ones at most 150 s and 30 lines, cut at a speaker turn or a pause >= 1 s.
LEAD_SCENE = (20.0, 30.0)
SECOND_SCENE = 60.0
SCENE_MAX_S, SCENE_MAX_UNITS, SCENE_PAUSE = 150.0, 30, 1.0
SCENE_URGENT = 120.0    # s: a scene whose first line is this close to the playhead is cut from what is heard so far
BANK_SHARE = 0.8        # the app's prepare-ahead setting banks at most this share of the lookahead (buffer.ts LOOKAHEAD_USE)
SCENES_IN_FLIGHT = 3    # scene calls at once, with no GPU lock (the translator's own semaphore also holds it to 3)
CONTEXT_BEFORE, CONTEXT_AFTER, CONTEXT_SPAN = 3, 2, 60.0  # context lines each side of a scene, from within this many s
# Aksharas of `full` per English syllable (§4.3): a weak prior until a speaker has K_MIN_LINES validated lines, then
# the running median of theirs (lines of at least K_MIN_SYLLABLES syllables, the last K_WINDOW).
K_PRIOR, K_MIN_LINES, K_MIN_SYLLABLES, K_WINDOW = 1.4, 3, 4, 400
BAND = (0.85, 1.10)     # §4.5: a wording fits when its predicted duration is within this share of the line's speech time
DENSE = 0.8             # a slot is speech-dense when speech fills this share of its span (only then is `fuller` asked)
TIERS_BY_FULLNESS = ("fuller", "full", "concise", "very_concise")
# %: a scene's code-mixing index is logged, and warned about only well outside CoSTA's ~20-40 (it tops out at 50, where
# half the words are English)
CMI_WARN = (5.0, 45.0)
REPHRASE_LEAD = 60.0    # s of video: a rephrased take must be voiced at least this long before the playhead reaches it
URGENT_LEAD = 60.0      # s: with the dub less than this far ahead of the playhead, the voicer goes first on the GPU (§5.3)
HELD_R = 1.25           # below this throughput a held video lets the pipeline work on past the lookahead (§3.12)
KEEP_BEHIND = 60.0      # s of dub PCM kept in memory behind the playhead; ahead, up to the lookahead (§3.12, §5.1)
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

SendJSON = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


@dataclass
class VoiceState:
    voice: object | None = None
    kind: VoiceKind = VoiceKind.PRESET
    ref_seconds: float = 0.0
    use_preset: bool = False
    status: str = "found"  # found | cloning | cloned | preset
    key: VoiceKey | None = None  # the clone's key in the duration estimator, taken (and calibrated) when it is built


@dataclass
class Chunk:
    """A stretch of video time the frontend has turned into units; it is dubbed once all its units are. After a seek
    into it, the lines that end before the seek point are passed over (the voicer never reaches them): it is then dubbed
    from the seek point on once the rest are."""

    a: float
    b: float
    pending: set[int] = field(default_factory=set)
    ready: bool = False
    passed: set[int] = field(default_factory=set)  # lines before a seek point in it, not waited on
    since: float = 0.0                             # that seek point, while any are passed over
    counted: float | None = None                   # where the stretch counted as dubbed (in the throughput) starts

    def dubbed_from(self) -> float:
        return max(self.a, self.since) if self.passed else self.a


@dataclass
class UnitState:
    unit: SourceUnit
    next_start: float | None
    chunk: Chunk
    speech_s: float = 0.0                    # diarized speech inside the span, what lengths are measured against (§4.3)
    music: bool = False                      # looks sung (asr_guard.music_like): reported, still dubbed (§3.4)
    telugu: str | None = None                # the chosen wording as the TTS says it; None until translated
    voicing: bool = False                    # the voicer has started on it: its wording no longer changes
    voiced: bool = False
    scene: int | None = None                 # the scene call it is in, or was translated in
    line: LineResult | None = None           # its validated translation: the tiers, English map and delivery
    tier: str | None = None                  # the tier chosen (§4.5)
    pred_full: float | None = None           # `full` aksharas predicted when its scene was asked for (units.jsonl)
    plan: Plan | None = None                 # where its take was placed, which a replacement must fit
    take_s: float | None = None              # natural duration of the take voiced
    provisional: bool = False                # voiced with a take that runs long while a rephrase is on its way
    replacement: LineResult | None = None    # a rephrase that came back in time, waiting for the voicer
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
    at the planned rate; with the takes the governor asked for, the retakes, why any take failed, and the voicer's GPU
    priority. `synth_s` is T3 plus the S3Gen flow of the take voiced (which runs when it is vocoded) and `render_s` the
    vocoder alone, as before takes were batched. `t3_ms_per_token` is per decode step, which samples a token for every
    take of a batch: it stays comparable whatever number of takes the governor asks for."""

    lock_wait_s: float = 0.0
    takes: int = 0
    synth_s: float = 0.0
    render_s: float = 0.0
    t3_s: float | None = None
    t3_tokens: int | None = None
    t3_steps: int | None = None
    flow_s: float | None = None
    cfm_steps: int | None = None
    takes_n: int | None = None          # takes asked for in the line's first batch (the throughput governor's N)
    retakes: int = 0                    # takes made because every take of a batch failed
    failures: list[str] = field(default_factory=list)  # why takes failed (qa.take: "cap", "short"), over all batches
    priority: int | None = None         # the voicer's GPU priority for its first take

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
    """A line's takes as the voicer made them (ARCHITECTURE §3.10): one, or one per piece at hard breaks, each with its
    natural duration, its audio at the voice's natural pace (where its pauses are found, and what plays when it isn't
    sped up), its pauses (`pauses.pauses`) and the pace the estimator learns from (`_take`)."""

    wordings: list[Wording] = field(default_factory=list)
    takes: list[object] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)
    pauses: list[tuple[tuple[float, float], ...]] = field(default_factory=list)
    paces: list[float | None] = field(default_factory=list)
    failed: str | None = None  # why a take voiced failed (the first that did), or None

    @property
    def total(self) -> float:
        return sum(self.seconds)

    def timed(self) -> list[tuple[float, tuple[tuple[float, float], ...]]]:
        """Each take as the planner takes it: (natural seconds, pauses)."""
        return list(zip(self.seconds, self.pauses))


class Session:
    def __init__(self, backend: Backend, resolver: Resolver, cache_dir: Path, send_json: SendJSON, send_bytes: SendBytes,
                 settings: TimingSettings | None = None, gpu: GpuScheduler | None = None,
                 lookahead: float = DEFAULT_LOOKAHEAD, style: str = "colloquial", prepass: float = PREPASS,
                 speed_cap: float = 1.2, allow_freeze: bool = True, clone_strength: str = "closest",
                 tts_script: str = "telugu", timing: str = "v2") -> None:
        self.b, self.resolver, self.cache_dir = backend, resolver, cache_dir
        self.send_json, self.send_bytes = send_json, send_bytes
        self.settings = settings or TimingSettings()
        self.gpu = gpu or GpuScheduler()  # the engine's, shared by every session: one GPU owner at a time
        self.lookahead = min(max(float(lookahead), 60.0), 3600.0)
        self.style = "formal" if style == "formal" else "colloquial"
        self.prepass = prepass
        self.speed_cap = min(max(float(speed_cap), 1.0), 1.25)  # the UI's "Maximum speed-up"
        self.allow_freeze = bool(allow_freeze)
        self.clone_strength = clone_strength if clone_strength in ("closest", "balanced", "natural") else "closest"
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
        self._voiced = 0
        self.estimator = DurationEstimator()
        self.calibrate_s = 0.0  # s spent calibrating voices (three takes each, on the GPU path to first audio)
        self.registry = SpeakerRegistry()
        self.voices: dict[str, VoiceState] = {}
        self.video: ResolvedVideo | None = None
        self.audio = np.zeros(0, np.float32)
        self.language: str | None = None
        self.start = 0.0
        self.playhead = 0.0
        self.held = False          # the UI holds the video (paused, or waiting for the dub): time the engine may use (§3.12)
        self.prepare_all = False   # "prepare the whole video": work to the end, whatever the lookahead (§3.12)
        self._recovering = False   # after a seek into video not dubbed yet: one take per line until the lead is back (§5.2)
        self.user_speed = 1.0
        self.units: dict[int, UnitState] = {}
        self.chunks: list[Chunk] = []
        self.ready = ReadyRanges()
        self.throughput = Throughput()
        self.raw: dict[int, tuple[np.ndarray, int, float]] = {}  # unit id -> (samples at 1.0x, sr, audio rate), near the playhead
        self._saving: set[int] = set()  # units whose PCM file is being written: kept in `raw` until it is
        self._next_id = 0
        self._asr_cursor = 0.0
        self._diar_cursor = 0.0
        self._prepass_heard = asyncio.Event()  # the pre-pass is diarized: talk shares for brief v0
        self._prepass_done = asyncio.Event()   # ...and its speakers cloned
        self._clone_task: asyncio.Task | None = None  # cloning speakers found later (`_clone_later`)
        self._clone_more = False
        self._wake = _Wake()
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._dir: Path | None = None
        self._last_ready_sent = 0.0
        # The translator stage (ARCHITECTURE §4).
        self.tr: SceneTranslator | None = None
        self._scene_no = 0
        self._scene_tasks: set[asyncio.Task] = set()
        self._side_tasks: set[asyncio.Task] = set()   # fits, rephrases, the brief, re-sends: never awaited by the voicer
        self._run_next: tuple[int | None, int] = (None, 0)  # a scene right after this unit id is this far into its run
        self._fresh_at: float | None = 0.0    # the start or a seek point: a scene starting near it starts a run
        self._ratios: dict[str, list[float]] = {}   # per speaker: `full` aksharas / English syllables of each line
        self._brief_next: Brief | None = None  # made, waiting for the next scene boundary
        self._brief_task: asyncio.Task | None = None
        self._brief_tries = 0
        self._claude_hold = 0.0               # monotonic s before which no new scene call starts
        self._claude_fails = 0
        self._claude_kind: str | None = None
        self._claude_failed_at = float("-inf")  # monotonic s of the last failure
        self._claude_said_ok = False            # claude_ok sent this session

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
        self._dir = self.cache_dir / ref.video_id
        (self._dir / "pcm").mkdir(parents=True, exist_ok=True)
        self.start = self.playhead = self._fresh_at = min(max(ref.start, 0.0), self._duration())
        self._asr_cursor = self._diar_cursor = self.start
        log.info("open %s %r (%s, %.0f s) style=%s lookahead=%.0f s", ref.video_id, self.video.title, self.video.channel,
                 self._duration(), self.style, self.lookahead)
        self._trace({"event": "open", "video": ref.video_id, "title": self.video.title, "duration": self._duration(),
                     "start": self.start, "style": self.style, "backend": self.b.name})
        for coro in (self._diarizer(), self._frontend(), self._translator(), self._voicer()):
            self._tasks.append(asyncio.create_task(self._guard(coro)))

    def seek(self, t: float) -> None:
        t = min(max(t, 0.0), self._duration())
        self.playhead = t
        undubbed = not self.ready.covered(t, min(t + 1.0, self._duration()))
        if undubbed:
            self._pass_over(t)
        if undubbed and not self._frontend_covers(t):
            # Jumped into unprocessed video: restart ASR (and diarization if needed) at the new position.
            self._asr_cursor = t
            if not self.registry.is_covered(t, min(t + 1.0, self._duration())):
                self._diar_cursor = t
            self.throughput.reset()
            self.planner.reset(t)
        if undubbed:
            self._fresh_at = t  # a scene starting here is the first of its run: a short one (ARCHITECTURE §3.12)
            self._recovering = True  # one take per line until the lead is back (§3.12, §5.2)
        if self.tr is not None:
            # Claude work outside the new window stops (queued calls never start; running ones get SIGINT). Lines
            # already validated stay in the line cache, so seeking back costs nothing. The window is the one the video
            # plays with, even while it is held (a seek while paused is common, and the far calls would keep the slots
            # the new frontier's first scene needs); "prepare the whole video" keeps them all. Into video not dubbed
            # yet, the scene call holding the seek point stops too when it runs on well past a short scene from there:
            # first audio then waits for 20-30 s of lines, not up to 150 s. Its lines are asked for again from the seek
            # point.
            at = min((s for s in self.units.values() if s.unit.end >= t), key=lambda s: (s.unit.start, s.unit.id),
                     default=None)
            held = at.scene if undubbed and at is not None and at.line is None else None
            lo, hi = t, self._duration() if self.prepare_all else self._play_horizon()

            def drop(r: SceneRequest) -> bool:
                a, b = min(x.start for x in r.lines), max(x.end for x in r.lines)
                return b < lo or a >= hi or (r.call == "scene" and r.scene == held and b - t > LEAD_SCENE[1])

            if n := self.tr.cancel(drop):
                log.info("seek to %.1f s: cancelled %d Claude calls", t, n)
        self._side(self._resend_near(t))
        self._wake.set()

    def set_playhead(self, t: float, playing: bool | None = None) -> None:
        """Where the UI's playhead is, and whether the video is playing (None: not said). A video held (paused, or
        waiting for the dub) is time the pipeline may use past the lookahead (§3.12)."""
        self.playhead = max(t, 0.0)
        if playing is not None:
            self.held = not playing
        self._evict_raw()
        self._wake.set()

    def set_prepare_all(self, on: bool) -> None:
        """"Prepare the whole video" (§3.12): dub to the end whatever the lookahead; memory stays windowed."""
        self.prepare_all = bool(on)
        self._wake.set()

    def ask_audio(self, ids: list[int]) -> None:
        """The UI asks for lines' audio it doesn't hold: held back outside the window, or evicted (§3.12)."""
        self._side(self._send_audio_for(ids))

    def set_player_rates(self, rates: list[float]) -> None:
        """The IFrame player's discrete rates: slow-downs must use one of them (§6.7)."""
        slow = tuple(sorted(float(r) for r in rates if 0 < float(r) < 1.0))
        if slow:
            self.settings = replace(self.settings, available_video_rates=slow, min_video_rate=max(min(slow), 0.75))

    async def set_speed(self, speed: float) -> None:
        """User playback speed (0.75–1.5x): re-render units that haven't played yet (§6.7)."""
        self.user_speed = min(max(speed, 0.75), 1.5)
        for uid, (samples, sr, rate) in list(self.raw.items()):
            st = self.units.get(uid)
            if st and st.unit.end >= self.playhead:
                await self._send_audio(uid, samples, sr, rate)

    async def set_speaker_preset(self, speaker: str, use_preset: bool) -> None:
        v = self.voices.get(speaker)
        if v:
            v.use_preset = use_preset
            await self._send_speakers()

    async def close(self) -> None:
        self._closed = True
        for t in [*self._tasks, *self._scene_tasks, *self._side_tasks]:
            t.cancel()
        if self.tr is not None:
            self.tr.cancel()  # every queued or running Claude call; running ones get SIGINT

    # ---- helpers -------------------------------------------------------------------------------
    def _duration(self) -> float:
        return len(self.audio) / SR_ANALYSIS

    def _horizon(self) -> float:
        """Where the pipeline stops until the playhead moves (ARCHITECTURE §3.12): `_play_horizon`, or the video's end in
        "prepare the whole video" mode and while the video is held and r is under 1.25 (or not measured yet): a paused
        video is time a barely-fast-enough engine should use."""
        if self.prepare_all or (self.held and self.throughput.rate() < HELD_R):
            return self._duration()
        return self._play_horizon()

    def _play_horizon(self) -> float:
        """Where the pipeline stops while the video plays: the lookahead past the playhead, or the lead that plays to the
        end without a stall at the measured throughput r when that is longer."""
        dur = self._duration()
        return min(self.playhead + max(self.lookahead, target_lead(dur - self.playhead, self.throughput.rate())), dur)

    def _lead(self) -> float:
        """Video seconds dubbed without a gap past the playhead: the ready ranges, then the voiced (or skipped) lines of
        the heard stretch after them, and on through the next ready range when that stretch is all voiced, up to where
        the first line not voiced yet starts."""
        end = self.playhead
        while True:
            end = self.ready.contiguous_end(end)
            chunk = next((c for c in self.chunks if c.a - 0.25 <= end < c.b), None)
            if chunk is None:
                break
            for st in sorted((s for s in self.units.values() if s.chunk is chunk and s.unit.end > end),
                             key=lambda s: (s.unit.start, s.unit.id)):
                if not st.voiced:
                    end = max(st.unit.start, self.playhead)  # up to it: before it, the stretch is dubbed or silent
                    return max(end - self.playhead, 0.0)
                end = st.unit.end
            end = chunk.b  # each pass ends past a chunk, so this ends
        return max(end - self.playhead, 0.0)

    def _pass_over(self, t: float) -> None:
        """A seek to `t` into a heard stretch not dubbed yet (§3.12): its lines that end before t - 1 s, which the voicer
        passes over (`_next_to_voice`), no longer hold it back; it is dubbed from `t` once the rest are, and may be so
        now. Lines passed over at an earlier seek and not before this one are waited on again."""
        for c in self.chunks:
            if not c.a - 0.5 <= t < c.b:
                continue
            waiting = {i for i in c.pending | c.passed if (st := self.units.get(i)) is not None and not st.voiced}
            behind = {i for i in waiting if self.units[i].unit.end < t - 1.0}
            if waiting - behind - c.pending:  # a seek back: lines passed over before are waited on again
                c.ready = False
            c.pending, c.passed, c.since = waiting - behind, behind, t
            if not c.pending:
                self._mark_ready(c)

    def _lead_target(self) -> float:
        """The lead the throughput governor holds takes back for (§5.2): what plays to the end without a stall at the
        measured throughput, and at least a minute."""
        return max(URGENT_LEAD, target_lead(self._duration() - self.playhead, self.throughput.rate()))

    def _frontier(self) -> float:
        """The scene frontier: how far past the playhead the heard lines have all gone to Claude (in a scene call,
        translated or voiced)."""
        end = self.playhead
        for st in sorted((s for s in self.units.values() if s.unit.end > self.playhead),
                         key=lambda s: (s.unit.start, s.unit.id)):
            if st.scene is None and st.line is None and not st.voiced:
                return max(end, st.unit.start)
            end = max(end, st.unit.end)
        return end

    def _listen_priority(self, a: float) -> int:
        """GPU priority of ASR from video time `a` (§5.3): the frontier's while the translator waits on it (it starts
        within a scene's length past the frontier, before the horizon), else background. Diarization asks for the
        ASR chunk its block lets through."""
        return FRONTIER if a < min(self._frontier() + SCENE_MAX_S, self._horizon()) else BACKGROUND

    def _voice_priority(self) -> int:
        """The voicer's GPU priority (§5.3): first while the dub is under a minute ahead of the playhead (after the start
        or a seek, once its first line is translated), else after the frontier's ASR and diarization."""
        return URGENT if self._lead() < URGENT_LEAD else VOICE

    def _caught_up(self) -> bool:
        """Everything up to the horizon is heard and dubbed: the pipeline waits for the playhead to move, not for work."""
        h = self._horizon()
        return self._asr_cursor >= min(h, self._duration() - 0.2) and not any(
            not s.voiced and s.unit.end >= self.playhead - 1.0 and s.unit.start < h for s in self.units.values())

    def _listen_until(self) -> float:
        """Where ASR stops until the playhead moves: the lookahead horizon, or the end of the pre-pass window when that is
        further (its transcript makes brief v1: ARCHITECTURE §5.3 step 6)."""
        return min(max(self._horizon(), self.start + self.prepass), self._duration())

    def _frontend_covers(self, t: float) -> bool:
        return any(c.a - 0.5 <= t < c.b for c in self.chunks)

    async def _guard(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("pipeline stage failed")
            await self.send_json({"type": "error", "message": "Dubbing stopped unexpectedly.", "retryable": True})

    async def _idle(self) -> None:
        """Wait up to a second for news; news that came while this stage was busy counts (`_Wake`)."""
        ev = self._wake.events.setdefault(asyncio.current_task(), asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        ev.clear()

    def _trace(self, rec: dict) -> None:
        """One line of units.jsonl. Events: open, diar and asr (per block: `run_s` holding the GPU, `lock_wait_s` to get
        it, the GPU `priority` asked with), clone (`build_s`, `calibrate_s`, `lock_wait_s`), scene (per scene: its
        lines' coverage classes), unit (per voiced line: timings from `VoiceCost.fields` (takes asked for, retakes and
        why any take failed, GPU priority), the lead and throughput it was voiced at, the text's `UnitState` fields, its coverage
        class and that class as the metrics count it for the wording voiced, speech fill and required rate), skipped,
        rephrase, and from the translator's `trace` (may arrive from a worker thread): claude (per CLI call), translate
        (per request), review (per scene review) and brief."""
        if self._dir is None:
            return
        try:
            with (self._dir / "units.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 3), **rec}, ensure_ascii=False) + "\n")
        except OSError:
            log.warning("could not write units.jsonl", exc_info=True)

    # ---- stage 1: speakers ---------------------------------------------------------------------
    async def _diarizer(self) -> None:
        dur = self._duration()
        prepass_end = min(self.start + self.prepass, dur)
        await self.send_json({"type": "speaker_scan", "phase": "scanning", "until": self.start, "duration": prepass_end, "found": 0})
        while not self._closed:
            if not self._prepass_done.is_set() and self.registry.is_covered(self.start, prepass_end):
                self._prepass_heard.set()  # scene 1 can go to Claude while the voices are cloned (ARCHITECTURE §5.3)
                self._wake.set()
                await self._clone_all(prepass_end)  # at the voicer's priority: the first line waits on it
                self._prepass_done.set()
                self._wake.set()
            # The whole pre-pass window whatever the lookahead; past it, ahead of ASR and of where ASR stops.
            want = min(dur, max(prepass_end, min(self._asr_cursor, self._listen_until()) + DIAR_AHEAD))
            a = self._diar_cursor
            if a >= want - 1.0:
                await self._idle()
                continue
            b = min(a + DIAR_BLOCK, dur)
            a0 = max(0.0, a - DIAR_OVERLAP) if a > self.start else a
            priority = self._listen_priority(a - CHUNK - CHUNK_PAD)  # the ASR chunk that waits for this block
            asked = time.perf_counter()
            async with self.gpu.hold(priority):
                t0 = time.perf_counter()
                block = await asyncio.to_thread(self.b.diarizer.diarize_block, self.audio[int(a0 * SR_ANALYSIS): int(b * SR_ANALYSIS)], a0)
                dt = time.perf_counter() - t0
            mapping = self.registry.add_block(block)
            self._diar_cursor = b
            for sid in set(mapping.values()):
                self.voices.setdefault(sid, VoiceState())
            log.info("diarized %.0f-%.0f s in %.1f s: %s", a0, b, dt, mapping)
            self._trace({"event": "diar", "a": round(a0, 2), "b": round(b, 2), "run_s": round(dt, 3),
                         "lock_wait_s": round(t0 - asked, 3), "priority": priority, "turns": len(block.turns),
                         "speakers": len(set(mapping.values()))})
            if not self._prepass_done.is_set():
                await self.send_json({"type": "speaker_scan", "phase": "scanning", "until": b, "duration": prepass_end,
                                      "found": len(self.registry.speakers)})
            else:
                self._clone_later()
            await self._send_speakers()
            self._wake.set()

    def _cut(self, a: float, b: float) -> np.ndarray:
        return self.audio[int(a * SR_ANALYSIS): int(b * SR_ANALYSIS)]

    def _reference(self, sid: str) -> tuple[np.ndarray, float]:
        """The stitched reference: the speaker's best clips up to REF_TARGET s (used when no single clip is long enough)."""
        clips = self.registry.reference_clips(sid, target=REF_TARGET, avoid_before=self.start + 60.0)
        parts, secs = [], 0.0
        gap = np.zeros(int(0.15 * SR_ANALYSIS), np.float32)
        for a, b in clips:
            parts += [self._cut(a, b), gap]
            secs += b - a
        return (np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)), secs

    async def _build_voice(self, sid: str, cost: VoiceCost, priority: int = BACKGROUND) -> tuple[object | None, float, str, str]:
        """ADR-017, measured on the maintainer's podcast: timbre from the speaker's best single clean clip plus identity
        averaged over up to 60 s of their clean speech made the guest's clone closer (similarity 0.40 -> 0.47) and fixed its
        pace (3.4 -> 5.1 aksharas/s). With under ~8 s in one clip, the stitched reference was better, so fall back to it.
        Returns (voice, seconds of reference, how it was built, a hash of the reference audio)."""
        tts = self.b.tts
        # Skip the first minute (often a cold-open montage over music) when enough later speech has been diarized.
        avoid = self.start + 60.0 if self.registry.covered_until(self.start) >= self.start + 180.0 else None
        best = self.registry.best_span(sid, length=10.0, min_len=SINGLE_CLIP_MIN, avoid_before=avoid)
        if best is not None and hasattr(tts, "prepare_voice_parts"):
            timbre = self._cut(*best)
            clips = self.registry.clean_clips(sid, max_total=60.0, avoid_before=avoid)
            pool = [self._cut(a, b) for a, b in clips] or [timbre]
            async with cost.hold(self.gpu, priority):
                voice = await asyncio.to_thread(tts.prepare_voice_parts, timbre, pool, timbre, SR_ANALYSIS, 0.5)
            return voice, sum(b - a for a, b in clips) or best[1] - best[0], "single-clip", _audio_hash(timbre, *pool)
        ref, secs = self._reference(sid)
        if secs < REF_MIN:
            return None, secs, "none", ""
        async with cost.hold(self.gpu, priority):
            voice = await asyncio.to_thread(tts.prepare_voice, ref, SR_ANALYSIS)
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

    async def _calibrate(self, sid: str, key: VoiceKey, voice: object, cost: VoiceCost, priority: int = BACKGROUND) -> int:
        """Set a new voice's pace, rate and overhead together, from the three calibration sentences (ARCHITECTURE §3.7), so
        length targets fit it from the first line. Each is said as lines are (`tts_script`) and counted on its Telugu
        script; a take that says next to nothing or runs to its cap is left out. The cap is the slowest take the estimator
        can represent (MIN_RATE after MAX_OVERHEAD), so a slow voice is still measured and only a runaway is dropped.
        The takes run at the priority of the voice they calibrate, which isn't published until they are done.
        Returns the number of takes used."""
        tts = self.b.tts
        if not hasattr(tts, "synthesize_mel"):
            return 0
        takes: list[tuple[str, float]] = []
        for w in CALIBRATION_TE:
            cap = min(MAX_LINE_SECONDS, MAX_OVERHEAD + count_units(w.spoken) / MIN_RATE)
            async with cost.hold(self.gpu, priority):
                take = await asyncio.to_thread(tts.synthesize_mel, self._tts_text(w), voice, "te", cap)
            if 1.0 < take.seconds < cap and not getattr(take, "capped", False):
                takes.append((w.spoken, take.seconds))
        if not takes:
            log.warning("%s: no usable calibration take; its pace is learned from its lines", sid)
            return 0
        fitted = self.estimator.calibrate(key, takes)
        log.info("%s speaks at %.1f aksharas/s with %.2f s overhead (%d calibration takes%s)", sid, self.estimator.rate(key),
                 self.estimator.overhead(key), len(takes), "" if fitted else ", one length: an update, not a fit")
        return len(takes)

    async def _clone(self, sid: str, cost: VoiceCost | None = None, priority: int = BACKGROUND) -> None:
        """Build a speaker's voice and calibrate it; their lines keep the voice they had until both are done. (The voicer
        can take the GPU between calibration takes: a line voiced then with the new voice would be sized at the prior's
        pace, and the calibration would overwrite what its takes taught the estimator.) `cost` is the unit being voiced
        when the clone happens on its path: the clone's waits for the GPU then count in that unit's `lock_wait_s` (and
        its GPU time only in the unit's `wall_s`). `priority`: the GPU priority of its holds (§5.3), the voicer's when a
        line waits on it, else background."""
        v = self.voices.setdefault(sid, VoiceState())
        v.status = "cloning"
        await self._send_speakers()
        cost = VoiceCost() if cost is None else cost  # only its lock waits are touched
        waited = cost.lock_wait_s
        t0 = time.perf_counter()
        voice, secs, how, reference = await self._build_voice(sid, cost, priority)
        if voice is None:
            v.status = "preset" if v.voice is None else v.status
            log.warning("%s has only %.1f s of clean speech so far; using a preset voice until more is heard", sid, secs)
            return
        key = self._voice_key(sid, voice, reference)
        log.info("cloned %s from %.1f s of reference (%s)", sid, secs, how)
        built = time.perf_counter()
        try:
            used = await self._calibrate(sid, key, voice, cost, priority)
        finally:
            v.voice, v.kind, v.ref_seconds, v.status, v.key = voice, VoiceKind.CLONED, secs, "cloned", key
        calibrate_s = time.perf_counter() - built
        self.calibrate_s += calibrate_s
        self._trace({"event": "clone", "speaker": sid, "ref_seconds": round(secs, 1), "method": how,
                     "build_s": round(built - t0, 2), "calibrate_s": round(calibrate_s, 2),
                     "lock_wait_s": round(cost.lock_wait_s - waited, 3), "calibration_takes": used,
                     "pace": round(self.estimator.rate(key), 2), "overhead_s": round(self.estimator.overhead(key), 3)})

    async def _clone_all(self, until: float) -> None:
        await self.send_json({"type": "speaker_scan", "phase": "cloning", "until": until, "duration": until,
                              "found": len(self.registry.speakers)})
        t0, calibrated = time.perf_counter(), self.calibrate_s
        for sid in sorted(self.registry.speakers, key=lambda s: -self.registry.speakers[s].talk_seconds):
            await self._clone(sid, priority=self._voice_priority())
        log.info("pre-pass voices ready in %.1f s, %.1f s of it calibrating", time.perf_counter() - t0,
                 self.calibrate_s - calibrated)
        await self.send_json({"type": "speaker_scan", "phase": "ready", "until": until, "duration": until,
                              "found": len(self.registry.speakers)})
        await self._send_speakers()

    def _clone_later(self) -> None:
        """Clone the speakers first heard after the pre-pass, off the diarizer's loop: their GPU holds are background
        work (§5.3), and the diarizer must not wait on them while the next block may be what frontier ASR needs. One
        pass at a time, and one more after it when a block found speakers meanwhile."""
        self._clone_more = True
        if self._clone_task is None or self._clone_task.done():
            self._clone_task = self._side(self._clone_new())

    async def _clone_new(self) -> None:
        """Speakers first heard after the pre-pass: clone once enough clean speech exists, before their lines are voiced
        (until then, and while a clone waits for the GPU, their lines keep a preset voice)."""
        while self._clone_more:
            self._clone_more = False
            for sid in list(self.registry.speakers):
                v = self.voices.setdefault(sid, VoiceState())
                if v.kind is not VoiceKind.CLONED and v.status != "cloning":
                    await self._clone(sid)
                    await self._send_speakers()

    async def _voice_for(self, sid: str, cost: VoiceCost) -> tuple[object, VoiceKind, VoiceKey]:
        """The voice to say a speaker's line with, and its key in the duration estimator."""
        v = self.voices.setdefault(sid, VoiceState())
        if v.kind is VoiceKind.CLONED and v.voice is not None and not v.use_preset:
            return v.voice, VoiceKind.CLONED, self._key(sid)
        if v.status != "cloning" and not v.use_preset:
            await self._clone(sid, cost, self._voice_priority())  # takes the GPU itself
            if v.kind is VoiceKind.CLONED and v.voice is not None:
                return v.voice, VoiceKind.CLONED, self._key(sid)
        preset = _preset_name(sid)
        return await asyncio.to_thread(self.b.tts.preset_voice, preset), VoiceKind.PRESET, self._voice_key(preset, None)

    # ---- stage 2: listening --------------------------------------------------------------------
    async def _frontend(self) -> None:
        dur = self._duration()
        while not self._closed:
            a = self._asr_cursor
            if a >= dur - 0.2 or a >= self._listen_until():
                await self._idle()
                continue
            b = min(a + CHUNK, dur)
            if not self.registry.is_covered(a, min(b + CHUNK_PAD, dur)):
                await self._idle()  # wait for the diarizer to get ahead of us
                continue
            await self._status("listening", "Listening…", a)
            lo, hi = max(0.0, a - 0.3), min(dur, b + CHUNK_PAD)
            # The diarized speech from the cursor on, for the transcriber's coverage guard (on the audio's clock).
            speech = [(max(t.start, a) - lo, min(t.end, hi) - lo)
                      for t in self.registry.turns_in(a, hi, exclusive=False)]
            priority = self._listen_priority(a)
            if a < self._horizon():  # hearing what the dub waits on is work, whatever the voicer saw when it last looked
                self.throughput.resume(time.monotonic())
            asked = time.perf_counter()
            async with self.gpu.hold(priority):
                t0 = time.perf_counter()
                tr = await asyncio.to_thread(self.b.transcriber.transcribe, self._cut(lo, hi), self.language, speech=speech)
                dt = time.perf_counter() - t0
            self.language = self.language or tr.language
            words = fix_words([type(w)(w.text, w.start + lo, w.end + lo, w.confidence) for w in tr.words])
            words = [w for w in words if w.start >= a - 0.05]  # the previous chunk owns words that started before the cursor
            cut = self._sentence_cut(words, b, last=(hi >= dur - 0.01))
            owned = [w for w in words if w.end <= cut + 1e-6]
            nxt = cut if owned else b
            self._trace({"event": "asr", "a": round(lo, 2), "b": round(hi, 2), "run_s": round(dt, 3),
                         "lock_wait_s": round(t0 - asked, 3), "priority": priority, "words": len(tr.words),
                         **self._asr_checks(tr, lo, a, nxt, owned)})
            if self._asr_cursor != a:  # a seek moved the cursor while we were transcribing
                continue
            turns = self.registry.turns_in(a, nxt, exclusive=True)
            before = [st for st in self.units.values() if st.unit.end <= a + 0.05]
            prev_text = max(before, key=lambda st: st.unit.end).unit.text if before else ("" if a <= 0.05 else None)
            units = merge_fragments(segment(owned, turns, prev_text=prev_text))
            chunk = Chunk(a, nxt)
            self.chunks.append(chunk)
            for k, u in enumerate(units):
                u.id = self._next_id
                self._next_id += 1
                nstart = units[k + 1].start if k + 1 < len(units) else None
                st = self.units[u.id] = UnitState(u, nstart, chunk, speech_s=self._speech(u),
                                                  music=asr_guard.music_like(u, turns))
                if st.music:
                    log.warning("unit %d %.1f-%.1f s looks sung (long, unsure words, a repeated phrase, one voice): "
                                "flagged for the report, still dubbed", u.id, u.start, u.end)
                chunk.pending.add(u.id)
            log.info("heard %.0f-%.0f s: %d words, %d units (language %s)", a, nxt, len(owned), len(units), self.language)
            self._asr_cursor = max(nxt, a + 1.0)
            if not chunk.pending:
                await self._chunk_ready(chunk)
            self._wake.set()

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

    # ---- stage 3: translating (the Claude CLI; ARCHITECTURE §4) --------------------------------------------------
    async def _translator(self) -> None:
        await self._prepass_heard.wait()
        v = self.video
        meta = VideoMeta(v.title, v.channel, v.description, v.chapters, v.tags,
                         talk_shares=tuple(sorted(self.registry.talk_share().items())))
        self.tr = self.b.translator(self.cache_dir, v.video_id, brief_v0(meta), style=self.style, trace=self._trace)
        while not self._closed:
            self._maybe_brief()
            got = None
            if len(self._scene_tasks) < SCENES_IN_FLIGHT and time.monotonic() >= self._claude_hold:
                got = self._next_scene()
            if got is None:
                await self._idle()
                continue
            req, nth = got
            if self._brief_next is not None:  # a new brief takes effect at a scene boundary: from this scene on
                log.info("brief v%d in use from scene %d", self._brief_next.version, req.scene)
                self.tr.use_brief(self._brief_next)
                self._brief_next = None
            task = asyncio.create_task(self._scene(req, nth))
            self._scene_tasks.add(task)
            task.add_done_callback(self._scene_done)

    def _scene_done(self, task: asyncio.Task) -> None:
        self._scene_tasks.discard(task)
        self._wake.set()  # a slot is free: only now, or the loop could wake and still count this task

    def _next_scene(self) -> tuple[SceneRequest, int] | None:
        """The next scene to translate: the earliest untranslated lines not behind the playhead, from before the lookahead
        horizon, cut by `scene_cut`; None when there is nothing to do yet."""
        ordered = sorted(self.units.values(), key=lambda s: (s.unit.start, s.unit.id))
        free = [s.line is None and s.scene is None and not s.voiced for s in ordered]
        k = next((i for i, s in enumerate(ordered) if free[i] and s.unit.end >= self.playhead - 2.0), None)
        if k is None or ordered[k].unit.start >= self._horizon():
            return None
        end = next((i for i in range(k, len(ordered)) if not free[i]), len(ordered))
        run = ordered[k:end]
        prev = ordered[k - 1].unit.id if k else None
        fresh = self._fresh_at is not None and run[0].unit.start <= self._fresh_at + LEAD_SCENE[1]
        nth = 0 if fresh or prev is None or prev != self._run_next[0] else self._run_next[1]
        # Cut from what is heard when nothing more joins the run soon: a translated line or the end follows it, its first
        # line is due soon, or ASR has stopped until the playhead moves while the run starts inside the lead the app
        # banks before it plays (with the playhead held there, waiting for the limit would never end).
        urgent = SCENE_URGENT
        if self._asr_cursor >= self._listen_until():
            urgent = max(urgent, BANK_SHARE * self.lookahead)
        final = (end < len(ordered) or self._asr_cursor >= self._duration() - 0.2
                 or run[0].unit.start - self.playhead < urgent)
        n = scene_cut([s.unit for s in run], nth, final)
        if n is None:
            return None
        lines = run[:n]
        self._scene_no += 1
        self._fresh_at, self._run_next = None, (lines[-1].unit.id, nth + 1)
        for st in lines:
            st.scene = self._scene_no
        a, b = lines[0].unit.start, lines[-1].unit.end
        before = [s for s in ordered[max(k - CONTEXT_BEFORE, 0):k] if s.unit.end >= a - CONTEXT_SPAN]
        after = [s for s in ordered[k + n:k + n + CONTEXT_AFTER] if s.unit.start <= b + CONTEXT_SPAN]
        # The lines before, with the Telugu chosen for them; English only when they aren't translated (after a seek).
        req = SceneRequest(self._scene_no, tuple(self._spec(s) for s in lines),
                           context_before=tuple((s.unit.text, s.line.tiers[s.tier].spoken if s.line and s.tier else None)
                                                for s in before),
                           context_after_en=tuple(s.unit.text for s in after))
        log.info("scene %d: %.1f-%.1f s, %d lines (%s of its run)", req.scene, a, b, n, nth + 1)
        return req, nth

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

    async def _scene(self, req: SceneRequest, nth: int) -> None:
        """One scene call and what comes of it: its lines' coverage review, after which they count as translated. The
        voicer never waits on this (§4.9)."""
        t0, asked = time.perf_counter(), time.monotonic()
        try:
            res = await self.tr.submit(req)
        except asyncio.CancelledError:
            self._release(req, nth)
            if _cancelling():
                raise
            log.info("scene %d dropped: outside the window after a seek", req.scene)
            self._trace({"event": "scene", "scene": req.scene, "dropped": True})
            return
        except ClaudeCLIError as err:
            self._release(req, nth)
            await self._claude_failed(err)
            return
        except Exception as exc:  # a bug, or the cache unwritable: the lines are asked for again after a back-off
            log.exception("scene %d failed", req.scene)
            self._release(req, nth)
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
            self._release(req, nth)
            log.info("scene %d dropped during its review: outside the window after a seek", req.scene)
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
        if done and not CMI_WARN[0] <= mix <= CMI_WARN[1]:
            log.warning("scene %d: code-mixing index %.0f %% is well outside the usual 20-40 %%", req.scene, mix)
        self._trace({"event": "scene", "scene": req.scene, "nth": nth, "a": round(req.lines[0].start, 2),
                     "b": round(req.lines[-1].end, 2), "lines": len(req.lines), "translated": len(done),
                     "cached": sum(1 for s in done if s.cache_hit), "fits": len(fits), "cmi": round(mix, 1),
                     "context_te": sum(1 for _, te in req.context_before if te), "brief": self.tr.brief.version,
                     **self._classes(done),
                     "wall_s": round(seconds, 2), "ready_s": round(time.perf_counter() - t0, 2)})
        self._wake.set()

    async def _review(self, req: SceneRequest, lines: dict[int, LineResult]) -> dict[int, LineResult] | None:
        """The coverage review of a scene's lines before they count as translated (§4.6): of the wording the band rule
        picks for each now, with one re-translation of any it finds a phrase missing from or a meaning error in. None
        when a seek dropped it: the lines are asked for again (from the line cache). A review that fails leaves the
        lines unreviewed, never untranslated, so the voicer never waits on Claude for them; a failure the user must fix
        is shown and holds new scene calls like any other; one that goes through clears it, as a scene call does."""
        if not lines:
            return lines
        chosen = {i: self._pick(self.units[i], line)[0] for i, line in lines.items()}
        asked = time.monotonic()
        try:
            res = await self.tr.review(req, lines, chosen)
        except asyncio.CancelledError:
            if _cancelling():
                raise
            return None
        except Exception:
            log.exception("the review of scene %d failed; its lines go on unreviewed", req.scene)
            return lines
        if res.error is not None:
            await self._claude_failed(res.error)
        elif res.calls:
            await self._claude_ok(asked)
        return res.lines

    def _release(self, req: SceneRequest, nth: int) -> None:
        """A scene call that ended without an answer: its lines go back to be asked for again, in the same place of
        their run of scenes if nothing was cut after them."""
        for spec in req.lines:
            st = self.units.get(spec.id)
            if st is not None and st.line is None:
                st.scene = None
        if self._run_next[0] == req.lines[-1].id:
            first = self.units.get(req.lines[0].id)
            prev = max((s for s in self.units.values() if first and s.unit.start < first.unit.start),
                       key=lambda s: (s.unit.start, s.unit.id), default=None)
            self._run_next = (prev.unit.id if prev and nth else None, nth)
        self._wake.set()

    async def _fit(self, req: SceneRequest) -> None:
        """New tiers for lines outside the band (§4.2), folded in if the voicer hasn't started on their line (whose tier
        must stay the wording it voices: the next scene's context, a rephrase and units.jsonl all read it)."""
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
                st.line = line
                self._choose(st)
        self._wake.set()

    def _maybe_brief(self) -> None:
        """Brief v1 once the pre-pass is heard (§4.2): one call in the background, swapped in at a scene boundary."""
        if (self._brief_task is not None or self._brief_tries >= BRIEF_TRIES or time.monotonic() < self._claude_hold
                or sum(c.b - c.a for c in self.chunks) < min(self.prepass, self._duration() - self.start) - 1.0):
            return
        self._brief_tries += 1
        heard = sorted(self.units.values(), key=lambda s: (s.unit.start, s.unit.id))
        self._brief_task = self._side(self._make_brief([(s.unit.speaker, s.unit.text) for s in heard]))

    async def _make_brief(self, transcript: list[tuple[str, str]]) -> None:
        try:
            brief = await self.tr.make_brief(self.tr.brief.meta, transcript)
        except ClaudeCLIError as err:
            log.warning("brief v1 failed (%s); scenes keep the metadata brief for now", err.kind)
            self._brief_task = None  # asked again (once Claude is back), up to BRIEF_TRIES times
            if err.kind not in LADDER:  # a bad or slow reply is only retried: the brief is optional, v0 stays
                await self._claude_failed(err)
            return
        self._brief_next = brief
        log.info("brief v%d ready: %d speakers, %d glossary terms", brief.version, len(brief.speakers),
                 len(brief.glossary))
        self._wake.set()

    async def _claude_failed(self, err: ClaudeCLIError) -> None:
        """A failure the translator leaves to the user (not signed in, a usage limit, the CLI missing or too old, a
        stall or throttle that outlasted its retries): new scene calls wait, the UI says what to do, and translation
        tries again later. Lines already translated keep being voiced; rephrases under way are dropped (their lines keep
        their provisional takes) rather than hold the translator's slots through the pause."""
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
        log.warning("Claude %s: %s; the next scene call in %.0f s", err.kind, err, wait)
        await self.send_json({"type": "claude_error", "kind": err.kind, "message": str(err), "limit": err.limit,
                              "resetsAt": err.resets_at, "retryIn": round(wait)})

    async def _claude_ok(self, asked: float) -> None:
        """A scene call asked at `asked` (monotonic s) went through Claude. The UI's Claude banner (a problem from hello,
        the last video or a failure here) is cleared on the first such call of the session and on the first after a
        failure. A call asked before the last failure says nothing about it: a usage limit hit meanwhile still holds."""
        if asked < self._claude_failed_at or (self._claude_said_ok and not self._claude_fails):
            return
        self._claude_fails, self._claude_kind, self._claude_said_ok = 0, None, True
        await self.send_json({"type": "claude_ok"})

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

    def _after(self, st: UnitState) -> UnitState | None:
        later = [s for s in self.units.values() if s.unit.start > st.unit.start]
        return min(later, key=lambda s: s.unit.start) if later else None

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

    # ---- stage 4: voicing ----------------------------------------------------------------------
    def _next_to_voice(self) -> UnitState | None:
        """The earliest line not voiced yet, once it is translated. Lines go out in onset order even when a later scene
        comes back from Claude first: the planner places them on one timeline, and a line placed ahead of an earlier
        one would push that one past it. Waiting here is idling, never awaiting Claude (§4.9)."""
        horizon = self._horizon()
        cands = [s for s in self.units.values() if not s.voiced and s.unit.end >= self.playhead - 1.0 and s.unit.start < horizon]
        first = min(cands, key=lambda s: (s.unit.start, s.unit.id)) if cands else None
        return first if first is not None and first.telugu else None

    async def _voicer(self) -> None:
        await self._prepass_done.wait()
        while not self._closed:
            swap = min((s for s in self.units.values() if s.replacement is not None), key=lambda s: s.unit.start,
                       default=None)
            if swap is not None:
                try:
                    await self._replace(swap)
                except Exception:
                    log.exception("replacing unit %s failed; its provisional take stays", swap.unit.id)
                continue
            st = self._next_to_voice()
            if st is None and self._caught_up():  # waiting for the playhead, not for work: not the engine's speed
                self.throughput.pause(time.monotonic())
            else:  # voicing, or waiting on the stages before it: that is the engine's speed
                self.throughput.resume(time.monotonic())
            if st is None:
                await self._idle()
                continue
            try:
                await self._dub(st)
            except Exception:
                log.exception("unit %s failed", st.unit.id)
                await self._skip(st, "synthesis failed")

    async def _synth(self, text: str, voice: object, cap: float, n: int, cost: VoiceCost) -> list[tuple[object, float, bool]]:
        """One synthesis call at the voice's natural pace: `n` takes in one batched decode where the TTS can
        (`synthesize_takes`), else one mel take (vocoded later at the planned rate), else plain samples. Returns each
        take with its natural duration and whether it ran to its cap."""
        tts = self.b.tts
        priority = self._voice_priority()
        cost.priority = priority if cost.priority is None else cost.priority
        async with cost.hold(self.gpu, priority):
            t0 = time.perf_counter()
            if hasattr(tts, "synthesize_takes"):
                takes = await asyncio.to_thread(tts.synthesize_takes, text, voice, "te", cap, n)
                cost.add_takes(takes, time.perf_counter() - t0)
                return [(t, t.seconds, bool(t.capped)) for t in takes]
            if hasattr(tts, "synthesize_mel"):
                take = await asyncio.to_thread(tts.synthesize_mel, text, voice, "te", cap)
                cost.add_take(take, time.perf_counter() - t0)
                return [(take, take.seconds, bool(getattr(take, "capped", False)))]
            samples = await asyncio.to_thread(tts.synthesize, text, voice, "te", cap)
            cost.add_take(samples, time.perf_counter() - t0)
        return [(samples, len(samples) / tts.sample_rate, False)]

    async def _take(self, st: UnitState, w: Wording, voice: object, key: VoiceKey, seconds: float, cost: VoiceCost,
                    n: int = 1, speech: float | None = None) -> tuple[object, float, float | None, str | None]:
        """Voice wording `w` of the line (ARCHITECTURE §3.8, §3.13): `n` takes in one batched decode where the TTS batches
        them, and one more take if every one of them failed (`_retake_ok`); the one voiced is picked by `qa.take` (cap
        hits and takes far shorter than the estimate out, then the closest to the line's speech time). A TTS that can't
        batch gives one take and no retake: without a seed it would say the line the same way again. Returns (the take,
        its natural duration, the mean duration of the takes that passed for the estimator (of those that didn't run to
        their cap when none passed: a voice faster than its estimate is still learned; None if all ran away), and why
        the take voiced failed, or None). `speech`: the speech time the take is picked against, for a piece of the line;
        by default the line's."""
        batched = hasattr(self.b.tts, "synthesize_takes")
        n = max(n, 1) if batched else 1
        cost.takes_n = n if cost.takes_n is None else cost.takes_n
        text = self._tts_text(w)
        cap = min(MAX_LINE_SECONDS, max(4.0, seconds * self.speed_cap * 2.0))
        estimate = self.estimator.estimate(w.spoken, key)
        got = await self._synth(text, voice, cap, n, cost)
        failed = [take_qa.failure(secs, capped, estimate) for _, secs, capped in got]
        if batched and all(failed) and self._retake_ok():  # the retake ladder's second rung (§3.13); step 6 adds QA's
            self._retakes += 1                               # reasons to `failed`
            cost.retakes += 1
            more = await self._synth(text, voice, cap, 1, cost)
            got += more
            failed += [take_qa.failure(secs, capped, estimate) for _, secs, capped in more]
        cost.failures += [f for f in failed if f]
        k = take_qa.pick([secs for _, secs, _ in got], failed, st.speech_s if speech is None else speech)
        usable = ([secs for (_, secs, _), f in zip(got, failed) if f is None]
                  or [secs for _, secs, capped in got if not capped])
        return got[k][0], got[k][1], (sum(usable) / len(usable) if usable else None), failed[k]

    def _retake_ok(self) -> bool:
        """Whether a batch whose every take failed gets one more take. Always, unless the lead is under its target and the
        engine barely outruns playback (r < 1.1), where the governor asks one take a line (§5.2): then while retakes
        stay within RETAKE_SHARE of the lines voiced (§3.13's QA budget, governed by throughput)."""
        if self._lead() >= self._lead_target() or self.throughput.rate() >= SLOW_R:
            return True
        return self._retakes < max(3, RETAKE_SHARE * self._voiced)

    def _takes_n(self, st: UnitState) -> int:
        """The throughput governor's takes for the line (§5.2), from the lead and r now; a seek's hold on one take ends
        once the lead is back at target."""
        lead, target = self._lead(), self._lead_target()
        if self._recovering and lead >= target:
            self._recovering = False
        return takes_for(lead, target, self.throughput.rate(), st.speech_s, self._recovering)

    async def _render(self, take: object, rate: float, cost: VoiceCost) -> np.ndarray:
        """The take voiced, rendered at the planned rate: vocoded (after its S3Gen flow, which counts as synthesis), or
        time-stretched."""
        tts = self.b.tts
        if hasattr(tts, "vocode"):
            flowed = getattr(take, "flow_s", 0.0)
            async with cost.hold(self.gpu, self._voice_priority()):
                t0 = time.perf_counter()
                audio = await asyncio.to_thread(tts.vocode, take, rate)
            cost.render_s += time.perf_counter() - t0 - cost.add_flow(flowed, take)
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
            take, dur, pace, failed = await self._take(st, w, voice, key, seconds, cost, n, speech=st.speech_s * share)
            natural = await self._render(take, 1.0, cost)
            said.wordings.append(w)
            said.takes.append(take)
            said.seconds.append(dur)
            said.audio.append(natural)
            said.pauses.append(pz.pauses(natural, self.b.tts.sample_rate))
            said.paces.append(pace)
            said.failed = said.failed or failed
        return said

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
        words, at = [], 0
        for piece in line.pieces:
            k = len(piece.split())
            words.append(Wording(piece, tuple((i - at, en) for i, en in line.full.english if at <= i < at + k)))
            at += k
        pred = [(self.estimator.estimate(w.spoken, key), ()) for w in words]
        try:
            _, plan = self.planner.evaluate(slot, sum(d for d, _ in pred), ahead=self._ahead(st), pieces=pred)
        except ValueError:  # placed already
            return []
        return words if plan.said == "pieces" else []

    async def _squeezed(self, said: Said, k: int, rate: float, cost: VoiceCost) -> np.ndarray:
        """Take `k` played `rate` times faster (§3.10 step 7): its inner pauses shorten first, each to no less than
        `keep_pause`, and only what they can't give speeds its speech up (`pauses.squeeze`)."""
        sr = self.b.tts.sample_rate
        r, cuts = pz.squeeze(said.pauses[k], said.seconds[k], rate, self.planner.s.keep_pause)
        audio = said.audio[k] if abs(r - 1.0) <= 1e-3 else await self._render(said.takes[k], r, cost)
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
        """A line's predicted duration once it has a wording."""
        return (self.estimator.estimate(st.line.tiers[st.tier].spoken, self._key(st.unit.speaker))
                if st.line and st.tier else None)

    def _ahead(self, st: UnitState) -> tuple[tuple[LineSlot, float | None], ...]:
        """What the planner knows of the lines after `st` (§3.10 step 5): the next one, and up to `window_lines` in all
        that start within `window_s` of it, each with its slot and its predicted duration (None until it has a
        wording)."""
        s, u = self.planner.s, st.unit
        later = sorted((x for x in self.units.values() if x.unit.start > u.start), key=lambda x: (x.unit.start, x.unit.id))
        near = [x for k, x in enumerate(later[:s.window_lines]) if k == 0 or x.unit.start <= u.start + s.window_s]
        return tuple((self._slot(x), self._predicted(x)) for x in near)

    def _plan(self, st: UnitState, said: Said) -> Plan:
        """Place the line's takes: pieces each at its break, else one take, which may wait at a soft anchor near one of
        its text's breaks."""
        slot, ahead = self._slot(st), self._ahead(st)
        if len(said.takes) > 1:
            return self.planner.place(slot, said.total, ahead=ahead, pieces=said.timed())
        return self.planner.place(slot, said.total, ahead=ahead, pauses=said.pauses[0], marks=_marks(said.wordings[0].spoken))

    def _budget_left(self) -> bool:
        return self._resynths < max(3, RESYNTH_SHARE * self._voiced)

    async def _dub(self, st: UnitState) -> None:
        u = st.unit
        t0 = time.perf_counter()
        cost = VoiceCost()
        seconds = target_seconds(self._slot(st), self.planner.s)
        st.voicing = True  # a fit that comes back from here on is too late for this line
        voice, kind, key = await self._voice_for(u.speaker, cost)
        self._choose(st, key)  # again, now that the voice's pace may be better known
        first_tier, telugu = st.tier, st.telugu
        lead, r = self._lead(), self.throughput.rate()
        n = self._takes_n(st)
        said = await self._say(st, self._pieces(st, key) or [st.line.tiers[st.tier]], voice, key, seconds, cost, n)
        self._learn(said, key)
        plan = self._plan(st, said)
        resynth = False
        if plan.needs_shorter and self._budget_left():
            size = lambda t: count_units(st.line.tiers[t].spoken)  # noqa: E731
            shorter = [t for t in st.line.tiers if size(t) < size(st.tier)]
            if shorter:  # the most complete shorter wording that fits, else the shortest (no Claude call: §4.9)
                fit = [t for t in shorter if self.estimator.estimate(st.line.tiers[t].spoken, key) <= BAND[1] * seconds]
                alt = max(fit, key=size, default=min(shorter, key=size))
                alt_said = await self._say(st, [st.line.tiers[alt]], voice, key, seconds, cost, n)
                if 0 < alt_said.total < said.total and alt_said.failed is None:  # a failed take is shorter for the wrong reason
                    st.tier, telugu, said, resynth = alt, self._tts_text(st.line.tiers[alt]), alt_said, True
                    self._learn(alt_said, key)
                    plan = self._plan(st, said)  # replaces the plan just placed for this line
                self._resynths += 1
        audio = await self._play(said, plan, cost)
        # Still too long: ship this take now, flagged, and ask Claude for a rephrase that may replace it (§3.13). Asked
        # once it is rendered, so the rephrase's deadline allows for a synthesis as long as this one, flow included.
        st.provisional = plan.needs_shorter and self._ask_rephrase(st, plan, cost)
        st.voiced, st.telugu, st.plan, st.take_s = True, telugu, plan, said.total
        self._voiced += 1
        await self._ship(st, kind, audio, seconds)
        took = time.perf_counter() - t0
        w = st.line.tiers[st.tier]
        flags = tenglish.lint(w.spoken, w.english, u.text, self._context_for(st))
        slot = self._slot(st)
        log.info("unit %d %s %.1f-%.1f s | %s | %s | %.1f s audio at %.2fx, lag %+.2f s%s%s, %s%s%s, %d takes (%d a batch, "
                 "%d retakes)%s, lead %.0f s, %.1f s", u.id, u.speaker, u.start, u.end, u.text, telugu, said.total,
                 plan.rate, plan.lag, f", freeze {plan.freeze:.2f} s" if plan.freeze else "",
                 f", said in {len(plan.parts)} parts ({plan.said})" if plan.parts else "", kind.value,
                 " re-synthesized shorter" if resynth else "", " provisional" if st.provisional else "", cost.takes,
                 cost.takes_n or n, cost.retakes, f", {' and '.join(cost.failures)} failed" if cost.failures else "",
                 lead, took)
        self._trace({"event": "unit", "id": u.id, "speaker": u.speaker, "start": u.start, "end": u.end, "target_s": round(seconds, 2),
                     "speech_s": round(st.speech_s, 2), "breaks": len(u.breaks), "anchors": len(u.anchors),
                     "cut_off": u.cut_off, "music": st.music, "scene": st.scene, "source": u.text, "tier_first": first_tier,
                     "tier": st.tier, "tiers": sorted(st.line.tiers), "telugu": telugu,
                     "latin_ratio": round(tenglish.latin_ratio(telugu), 3), "english_words": len(w.english),
                     "full_aksharas": round(count_units(st.line.full.spoken), 1),
                     "full_pred_aksharas": None if st.pred_full is None else round(st.pred_full, 1),
                     "flags": list(st.line.flags), "lint": flags, "lint_version": tenglish.LINT_VERSION,
                     "coverage": coverage_json(st.line.coverage),
                     "coverage_voiced": voiced_class(st.line.coverage, st.tier), **self._fill(st),
                     "audio_s": round(said.total, 2), "dub_start": round(plan.start, 3),
                     "lag": round(plan.lag, 3), "audio_rate": round(plan.rate, 3), "freeze": round(plan.freeze, 3),
                     **self._timing(slot, plan),
                     "overdraft": round(plan.overdraft, 2), "needs_shorter": plan.needs_shorter, "resynth": resynth,
                     "provisional": st.provisional, "voice": kind.value, "wall_s": round(took, 2),
                     "lead_s": round(lead, 1), "r": round(r, 3),
                     **cost.fields(), "translate_ready_at": st.translate_ready_at, "translate_s": st.translate_s,
                     "model": st.model, "prompt_hash": st.prompt_hash, "cache_hit": st.cache_hit})
        await self._unit_done(st)

    def _timing(self, slot: LineSlot, plan: Plan) -> dict:
        """units.jsonl's timing fields for a voiced line (§3.10 metrics): how it was said and its ceiling, its speaker's
        speech and where the dub is voiced (video time), and per line the speech-level end error, overlap and the onset
        error of each part timed against an English onset."""
        row = {"speech": [list(x) for x in slot.speech], "voiced": [list(x) for x in plan.voiced],
               "anchor_errors": anchor_errors(plan)}
        return {"said": plan.said, "parts": len(plan.parts), "rate_cap": round(slot.rate_cap, 3),
                "hard_breaks": len(slot.breaks), **row, **line_metrics(row)}

    async def _ship(self, st: UnitState, kind: VoiceKind, audio: np.ndarray, seconds: float) -> None:
        """Send a voiced line (the unit, then its audio) and cache its PCM on disk for re-sending. Audio for a line outside
        the window near the playhead is held back (`audio` false): the UI asks for it as the playhead nears (§3.12)."""
        u, plan = st.unit, st.plan
        sr = self.b.tts.sample_rate
        edits = [{"kind": "freeze", "at": plan.end, "span": 0.0, "rate": 0.0, "added": plan.freeze}] if plan.freeze > 0 else []
        near = self._in_window(st)
        # In memory until its file is written (an ask for it meanwhile is served from here), then only in the window.
        self.raw[u.id] = (audio, sr, 1.0)  # already rendered at the planned rate; only the user's speed is applied on top
        self._saving.add(u.id)
        try:
            await self.send_json({
                "type": "unit", "id": u.id, "speaker": u.speaker, "start": plan.start, "end": u.end,
                "budget": round(seconds, 3), "audioRate": plan.rate, "audioWall": plan.wall, "lag": round(plan.lag, 3),
                "freeze": round(plan.freeze, 3), "voice": kind.value, "source": u.text, "telugu": st.telugu,
                "units": count_units(st.line.tiers[st.tier].spoken), "edits": edits,
                "provisional": st.provisional, "audio": near,
            })
            if near:
                await self._send_audio(u.id, audio, sr, 1.0)
            if self._dir is not None:
                await asyncio.to_thread(_save_pcm, self._dir / "pcm" / f"{u.id}.npy", audio)
        finally:
            self._saving.discard(u.id)
        self._evict_raw()

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

    def _video_ahead(self, st: UnitState) -> float:
        """Wall seconds until the playhead reaches the line, playing at the user's speed."""
        return (st.unit.start - self.playhead) / self.user_speed

    def _ask_rephrase(self, st: UnitState, plan: Plan, cost: VoiceCost) -> bool:
        """Queue a Claude rephrase for a line whose take still runs long (§3.13, §4.9), in the background. Its answer is
        wanted by when the new take can still be voiced REPHRASE_LEAD s before the playhead reaches the line. None while
        scene calls are paused after a Claude failure: the take ships as it is."""
        if self.tr is None or not self._budget_left() or time.monotonic() < self._claude_hold:
            return False
        ahead = self._video_ahead(st) - REPHRASE_LEAD - cost.synth_s
        if ahead <= 0:
            return False
        u, w = st.unit, st.line.tiers[st.tier]
        over = plan.excess * plan.rate  # seconds of speech past the line's window
        spec = self._line_spec(st, ("full", "concise", "very_concise"), current=w.spoken,
                               finding={"overrun_s": round(over, 2),
                                        "overrun_aksharas": round(over * self.estimator.rate(self._key(u.speaker)))})
        self._side(self._rephrase(st, SceneRequest(st.scene or 0, (spec,), "rephrase", deadline=time.time() + ahead)))
        return True

    async def _rephrase(self, st: UnitState, req: SceneRequest) -> None:
        u = st.unit
        try:
            res = await self.tr.submit(req)
            line = res.lines.get(u.id)
            outcome = ("late" if res.dropped or (line and self._video_ahead(st) < REPHRASE_LEAD)
                       else "failed" if line is None else "arrived")
        except asyncio.CancelledError:
            if _cancelling():
                raise
            outcome, line = "cancelled", None
        except ClaudeCLIError as err:
            outcome, line = err.kind, None
        if outcome == "arrived":
            st.replacement = line
            self._wake.set()
        log.info("rephrase of unit %d: %s", u.id, outcome)
        self._trace({"event": "rephrase", "id": u.id, "outcome": outcome})

    async def _replace(self, st: UnitState) -> None:
        """Voice a rephrase that came back in time in place of the provisional take, if its take fits the span planned
        for that one: shorter than it, played at the same rate from the same start, with the freeze it no longer needs
        taken off (§3.13). The wording is the most complete one predicted to end inside the line's window, else the
        shortest one predicted shorter than the take. Otherwise the provisional take stays."""
        line, st.replacement = st.replacement, None
        u, outcome = st.unit, "late"
        if st.plan is not None and st.take_s is not None and self._video_ahead(st) >= REPHRASE_LEAD:
            outcome = "too long"
            key = self._key(u.speaker)
            pred = {t: self.estimator.estimate(line.tiers[t].spoken, key)
                    for t in ("full", "concise", "very_concise") if t in line.tiers}
            room = st.take_s - st.plan.excess * st.plan.rate  # natural seconds that would end inside the window
            pick = next((t for t in pred if pred[t] <= room),
                        min((t for t in pred if pred[t] < st.take_s), key=pred.__getitem__, default=None))
            if pick is not None:
                cost = VoiceCost()
                seconds = target_seconds(self._slot(st), self.planner.s)
                voice, kind, key = await self._voice_for(u.speaker, cost)
                said = await self._say(st, [line.tiers[pick]], voice, key, seconds, cost)
                self._learn(said, key)
                self._resynths += 1
                if 0 < said.total < st.take_s and said.failed is None:  # its take must pass (§3.13), as well as fit
                    plan = self._shortened(st.plan, said.total, said.pauses[0])
                    audio = await self._play(said, plan, cost)
                    text = self._tts_text(line.tiers[pick])
                    st.line, st.tier, st.telugu, st.take_s, st.provisional = line, pick, text, said.total, False
                    st.plan = plan
                    await self._ship(st, kind, audio, seconds)
                    outcome = "replaced"
        log.info("rephrase of unit %d: %s", u.id, outcome)
        self._trace({"event": "rephrase", "id": u.id, "outcome": outcome, "tier": st.tier,
                     **({**self._fill(st), "freeze": round(st.plan.freeze, 3), **self._timing(self._slot(st), st.plan)}
                        if outcome == "replaced" else {})})

    def _shortened(self, plan: Plan, duration: float, pauses: tuple[tuple[float, float], ...] = ()) -> Plan:
        """The plan of a shorter take in the place of a longer one: the same start and rate, so it never ends later than
        that one did (lines placed after it keep their places). What it saves comes off the freeze first; a hold left
        under the planner's `min_freeze` would stutter on the player, so it is that long (never longer than before). It
        is said whole, even where the longer one was said in parts. `pauses`: the take's (where it is voiced)."""
        wall = duration / plan.rate
        saved = plan.wall - wall
        freeze = max(plan.freeze - saved, 0.0)
        if 0.0 < freeze < self.planner.s.min_freeze:
            freeze = self.planner.s.min_freeze
        got = replace(plan, wall=wall, freeze=freeze, freeze_at=plan.freeze_at if freeze else None,
                      excess=max(plan.excess - saved, 0.0), parts=(), said="whole")
        return replace(got, voiced=voiced(got, [(duration, pauses)], self.planner.s.keep_pause))

    async def _skip(self, st: UnitState, why: str) -> None:
        st.voiced = True
        u = st.unit
        log.warning("skipped unit %d (%s): %s", u.id, why, u.text)
        self._trace({"event": "skipped", "id": u.id, "start": u.start, "end": u.end, "source": u.text, "why": why,
                     "music": st.music, "speech": [list(x) for x in self._speech_spans(u)]})
        await self.send_json({"type": "unit_skipped", "id": u.id, "start": u.start, "end": u.end})
        await self._unit_done(st)

    async def _unit_done(self, st: UnitState) -> None:
        st.chunk.pending.discard(st.unit.id)
        if not st.chunk.pending and not st.chunk.ready:
            await self._chunk_ready(st.chunk)

    async def _chunk_ready(self, chunk: Chunk) -> None:
        self._mark_ready(chunk)
        await self._send_ready(force=True)

    def _mark_ready(self, chunk: Chunk) -> None:
        """The chunk is dubbed, from where `dubbed_from` says: that stretch joins the ready ranges, and what of it wasn't
        counted before joins the throughput if it starts before the horizon. Past the horizon only ASR for brief v1
        runs (a stretch with nothing to say is dubbed as soon as it is heard): that isn't the dub's pace, and it runs
        while the voicer may have stopped the throughput clock."""
        chunk.ready = True
        a = chunk.dubbed_from()
        self.ready.add(a, chunk.b)
        new = (chunk.b if chunk.counted is None else chunk.counted) - a
        if new > 0:
            chunk.counted = a
            if a < self._horizon():
                self.throughput.record(time.monotonic(), new)

    async def _send_ready(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_ready_sent < 1.0:
            return
        self._last_ready_sent = now
        x = self.throughput.rate()
        remaining = max(self._duration() - self.playhead, 0.0)
        # The lead that plays to the end without a stall (banking it takes lead / x of wall time). It isn't capped by
        # the lookahead: the pipeline works past the lookahead to bank it (`_horizon`).
        await self.send_json({"type": "ready", "until": self.ready.contiguous_end(self.playhead),
                              "ranges": [[round(a, 2), round(b, 2)] for a, b in self.ready.ranges],
                              "throughput": round(x, 3), "targetLead": round(target_lead(remaining, x), 1)})

    async def _send_audio(self, uid: int, samples: np.ndarray, sr: int, rate: float) -> None:
        eff = rate * self.user_speed
        out = await asyncio.to_thread(wsola, samples, eff, sr) if abs(eff - 1.0) > 1e-3 else np.asarray(samples, np.float32)
        await self.send_bytes(struct.pack("<IIII", uid, sr, len(out), 0) + out.astype(np.float32).tobytes())

    async def _pcm(self, st: UnitState) -> tuple[np.ndarray, int, float] | None:
        """A voiced line's PCM: from memory, else from the cache on disk (kept in memory again if it is in the window)."""
        uid = st.unit.id
        raw = self.raw.get(uid)
        path = self._dir / "pcm" / f"{uid}.npy" if self._dir is not None else None
        if raw is None and path is not None and path.is_file():
            try:
                pcm = (await asyncio.to_thread(np.load, path)).astype(np.float32)
            except (OSError, ValueError):
                log.warning("could not read the cached audio of unit %d", uid, exc_info=True)
                return None
            raw = (pcm, self.b.tts.sample_rate, 1.0)
            if self._in_window(st):
                self.raw[uid] = raw
        return raw

    async def _resend_near(self, t: float) -> None:
        """After a seek the UI may have evicted audio; re-send what is already dubbed near the new playhead."""
        lo, hi = t - 5.0, t + self.lookahead
        for st in sorted((s for s in self.units.values() if s.voiced and s.telugu and lo <= s.unit.start < hi), key=lambda s: s.unit.start):
            if (raw := await self._pcm(st)) is not None:
                await self._send_audio(st.unit.id, *raw)
        await self._send_ready(force=True)

    async def _send_audio_for(self, ids: list[int]) -> None:
        """Send the audio of the voiced lines the UI asked for, in time order."""
        for st in sorted((s for i in ids if (s := self.units.get(i)) is not None and s.voiced and s.telugu),
                         key=lambda s: s.unit.start):
            if (raw := await self._pcm(st)) is not None:
                await self._send_audio(st.unit.id, *raw)

    def _in_window(self, st: UnitState) -> bool:
        """Whether a voiced line's audio sounds within [playhead - KEEP_BEHIND, playhead + lookahead]: the dub PCM kept in
        memory and in the UI (§3.12, §5.1); the rest waits on disk until asked for."""
        end = max(st.unit.end, st.plan.start + st.plan.wall) if st.plan is not None else st.unit.end
        return end >= self.playhead - KEEP_BEHIND and st.unit.start <= self.playhead + self.lookahead

    def _evict_raw(self) -> None:
        for uid in [k for k in self.raw if k not in self._saving
                    and ((st := self.units.get(k)) is None or not self._in_window(st))]:
            self.raw.pop(uid, None)

    async def _send_speakers(self) -> None:
        share = self.registry.talk_share()
        await self.send_json({"type": "speakers", "speakers": [
            {"id": sp.id, "label": sp.label, "voice": self.voices.get(sp.id, VoiceState()).kind.value,
             "status": self.voices.get(sp.id, VoiceState()).status,
             "referenceSeconds": round(self.voices.get(sp.id, VoiceState()).ref_seconds, 1),
             "talkSeconds": round(sp.talk_seconds, 1), "share": round(share.get(sp.id, 0.0), 3),
             "usePreset": self.voices.get(sp.id, VoiceState()).use_preset}
            for sp in sorted(self.registry.speakers.values(), key=lambda x: x.first_at)
        ]})

    async def _status(self, stage: str, message: str, at: float | None = None) -> None:
        await self.send_json({"type": "status", "stage": stage, "message": message, "at": at})
        await self._send_ready()


class _Wake:
    """Wakes the stages idling in `Session._idle`. Each stage waits on an event of its own, so a stage going idle can't
    clear news meant for another that is still busy (which would then sleep out its whole second)."""

    def __init__(self) -> None:
        self.events: dict[object, asyncio.Event] = {}

    def set(self) -> None:
        for ev in self.events.values():
            ev.set()


def _cancelling() -> bool:
    """Whether the running task itself is being cancelled (the session closing), as opposed to a translator request it
    awaits being cancelled under it (a seek)."""
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


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'”’)").endswith((".", "?", "!", "…"))


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


def scene_cut(run: list[SourceUnit], nth: int, final: bool) -> int | None:
    """How many lines of `run` (untranslated lines in time order, with nothing between them) the next scene takes
    (ARCHITECTURE §4.1), or None to wait until more is heard. `nth` is the scene's place in its run of scenes:
    - 0, the first after the start or a seek: up to the first sentence end at or after 20 s, and no later than 30 s;
    - 1: about 60 s; later ones at most 150 s and 30 lines;
    both cut at the last speaker turn or pause of 1 s or more in their second half, else at the limit.
    `final`: nothing more will join `run` soon (a translated line or the video's end follows it, or its first line is
    due soon), so cut from what is there instead of waiting for the limit."""
    s0 = run[0].start
    if nth == 0:
        for i, u in enumerate(run):
            if u.end - s0 > LEAD_SCENE[1]:
                return max(i, 1)
            if u.end - s0 >= LEAD_SCENE[0] and _ends_sentence(u.text):
                return i + 1
        return len(run) if final else None
    limit = SECOND_SCENE if nth == 1 else SCENE_MAX_S
    fit = 0
    while fit < min(len(run), SCENE_MAX_UNITS) and run[fit].end - s0 <= limit:
        fit += 1
    if fit == 0:
        return 1  # one sentence longer than a scene
    if fit == len(run) and fit < SCENE_MAX_UNITS:
        return fit if final else None  # the limit isn't reached yet
    for i in range(fit - 1, 0, -1):
        a, b = run[i], run[i + 1] if i + 1 < len(run) else None
        if a.end - s0 < limit / 2:
            break
        if b is None or b.speaker != a.speaker or b.start - a.end >= SCENE_PAUSE:
            return i + 1
    return fit
