"""The offline whole-video render (docs/research/dubbing-2026-09/OFFLINE-RENDER.md, the maintainer's request of
2026-09-25): one `RenderJob` per video works on the whole video, one GPU stage at a time, and keeps what each stage makes
on disk as it goes, so a pause, a quit or a crash continues from there (§2.1). The stages so far (§2.2-§2.8):
- fetch: the audio (yt-dlp), decoded once to render/audio16k.f32 and memory-mapped;
- speakers: pyannote on the whole file, settled, with the ids of any earlier run (render/diarization.json);
- transcript: Whisper over the whole file in 60 s chunks, one row per chunk (render/transcript.jsonl);
- units: the sentence units of the whole transcript, recomputed every run and never stored;
- voices, on the GPU, beside the brief, on Claude: each speaker's voice built from their best speech anywhere in the
  video and calibrated (render/voices.json, and render/voices/<id>.npy, the Hear voice sample); the video brief from the
  whole transcript, in parts (the translator's briefs.jsonl);
- translate: every scene of the range in three lanes, each scene with the Telugu of the one before it as its context,
  and reviewed (the translator's lines.jsonl; the lines it skipped in render/skipped.jsonl);
- voice_lines, beside the translation: the dub loop (§2.9, §2.10), which voices the range's lines in onset order on a
  working plan, keeps each chosen take (render/takes/<take key>.npz and a row of render/takes.jsonl), and once a scene is
  voiced asks for its fix-ups: one batched rephrase and one review of the wordings voiced (render/fixups.jsonl);
- finish, trailing the dub loop (§2.11-§2.13): the final plan F places each line once every line of its lookahead window
  has its take in a settled scene, with no freezes (a file keeps the video's timing), the dub loop writes its PCM
  (render/pcm/<pcm key>.npy), and once every line is final the manifest (render/manifest.json), the export's input, is
  written, once;
- separate, on the GPU before the dub loop, beside the translation: the original music and effects, the source audio
  less its speech (Mel-Band RoFormer, `separate.py`), in 60 s blocks (render/bed/<k>.npy) with the speech's level per
  0.1 s (render/bed_vocals.npy) (§2.14);
- video, beside them, with the network and no GPU: the video-only stream the MP4 copies (§2.2);
- export: the mix (the voices over the ducked bed), the subtitles and the MP4 in the output folder (§2.15-§2.17,
  `export.py`); after the whole video's, the sources the job downloaded and the bed go.
A stage's file starts with the inputs it was made from; the stage runs again only when the inputs it would use now
differ, and is served from disk otherwise. What the dub loop keeps is content-keyed (§2.1), so a resume voices only the
lines without a take that still matches, and finds each line's PCM already there when its final plan is unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Coroutine, Iterable, Iterator

import numpy as np

from .backends.apple import STYLE_PROMPT
from . import export as mp4
from . import mix, separate, subtitles
from .backends.base import (COARSE_STEP, SR_ANALYSIS, Backend, Cancelled, Coverage, LineResult, LineSpec,
                            SceneRequest, VideoMeta, Wording)
from .backends.claude_translator import LADDER, brief_v0, line_json, line_key
from .backends.omnivoice import OmniVoiceError
from .claude_cli import ClaudeCLIError
from .dubber import (BAND, BRIEF_TRIES, CALIBRATION_TE, CHUNK, CHUNK_PAD, CONTEXT_AFTER, CONTEXT_BEFORE, CONTEXT_SPAN,
                     REF_MIN, SCENE_MAX_UNITS, Dubber, Said, UnitState, VoiceCost, VoiceState, _cancelling,
                     _preset_name, _save_pcm, scene_cut)
from .gpu import GpuScheduler
from .models import load_lock
from .qa import take as take_qa
from .qa.coverage import APPROVAL_POLICY, CLASSES, approved_full, coverage_from, coverage_json, voiced_class
from .qa.validators import check_line
from .resolve import VIDEO_FORMAT, ResolveError, ResolvedVideo, Resolver, audio_start, decode_audio_to, decode_stereo
from .settings import default_output_dir
from .segment import merge_fragments, segment
from .speakers import FLOOR_CAP, MIN_SHARE, MIN_TALK, SAME_COS, SpeakerRegistry, activity, match_ids, settle
from .text import asr_guard, tenglish
from .text.akshara import count_units
from .text.asr_fix import fix_words
from .text.scene_prompt import PROMPT_HASH, REVIEW_HASH
from .timing.duration import DurationEstimator, VoiceKey
from .timing.planner import LineSlot, Plan, TimelinePlanner, rate_ceiling, target_seconds
from .types import SourceUnit, TimedWord, VoiceKind
from .urls import VideoRef, parse_youtube_url

log = logging.getLogger("maata.render")

# The stages in order: key, label and the unit of their progress (§4).
STAGES = (("fetch", "Download", "MB"), ("speakers", "Speakers", "fraction"), ("transcript", "Transcript", "video s"),
          ("units", "Sentences", "lines"), ("voices", "Voices", "speakers"), ("brief", "Video brief", "parts"),
          ("separate", "Background sound", "video s"), ("translate", "Translation", "lines"),
          ("voice_lines", "Telugu speech", "speech s"),
          ("finish", "Finishing", "lines"), ("video", "Video download", "MB"), ("export", "Saving the video", "video s"))
# How they run: one group after another, the stages of a group side by side (the voices on the GPU while the brief is
# on Claude; the background sound on the GPU, then the dub loop, behind the translation, the final plan behind the dub
# loop, and the video download beside them, on the network, §1, §2.14); the export once they are all done.
ORDER = (("fetch",), ("speakers",), ("transcript",), ("units",), ("voices", "brief"),
         ("separate", "translate", "voice_lines", "finish", "video"), ("export",))
SERIES = ("separate", ("voice_lines", "finish"))  # in that group the GPU runs `separate`, then the dub loop (§2.14)
# Bumped whenever how a stage's output is made changes, so a resumed render never mixes old and new (§12).
DIAR_VERSION = 1
ASR_VERSION = 1
VOICE_VERSION = "v1"     # in every voice key's reference, f"{VOICE_VERSION}:{hash of the reference audio}" (§2.1)
TAKES_VERSION = 2        # synthesis identity and selected-waveform QA: old takes must be checked again (§2.1)
MIX_VERSION = 1          # in every PCM key: how a line's audio is made from its takes and its final plan (§2.12)
BRIEF_PART_WORDS = 8000  # words of transcript per brief call, about 45 min of talk (the CLI's 240 s timeout, §2.7)
LANES = 3                # translation lanes: the translator's concurrency (§2.8)
LANE_WEIGHTS = (1.0, 1.0, 1.0)  # balanced production baseline; explicit weights let a benchmark front-load the cursor
PAST_STOP = 30.0         # s: a preview translates the scenes that start before its stop point plus this (§3)
KEEP_VOICE = 0.9         # a stored voice is kept while this share of the speech in its spans is still its speaker's
HOLD_POLL = 0.2          # s between looks at the clock (and at a pause) while a Claude hold is waited out
AUTO_SPEAKERS = (1, 6)   # the speaker count's bounds in auto mode (the UI's range)
DISK_PER_HOUR = 0.5e9    # bytes of free disk a render needs per hour of video, plus DISK_HEADROOM (estimate, §7)
DISK_HEADROOM = 1e9
VIDEO_DISK_PER_HOUR = 0.9e9  # bytes of the cache's volume per hour of video beside the video itself: the bed and the
                             # export's audio (estimate, §2.2, §7)
PROGRESS_EVERY = 0.5     # s: progress alone rewrites job.json (and sends a render event) at most this often
SLEEP_GAP = 60.0         # s: wall-clock time past the monotonic clock's between two progress ticks: the Mac slept (§4)
# The dub loop's takes (§2.9, M4): TAKES_N in one batched decode, TAKES_N_SHORT for a line under SHORT_SPEECH s of
# speech; fix-up syntheses (a shorter tier, a rephrase) at most FIXUP_SHARE of the range's lines.
TAKES_N, TAKES_N_SHORT, SHORT_SPEECH = 2, 3, 3.0
FIXUP_SHARE = 0.15
FIXUP_WORDINGS = ("rephrase", "retranslate")  # take rows whose wording is a fixups.jsonl answer, not the line's own
ETA_AFTER = 30.0         # s of a stage's own work in this run before its measured rate gives its ETA (§4)
# The static priors of the ETA and the estimate (§7, estimates for the M5 Pro): wall seconds per second of video in
# each stage's range, (low, high). Translation runs beside the dub loop, and the final plan trails it.
PRIORS = {"fetch": (0.003, 0.01), "speakers": (0.04, 0.086), "transcript": (0.046, 0.143), "units": (0.0002, 0.001),
          "voices": (0.011, 0.023), "brief": (0.0034, 0.026), "translate": (0.23, 0.34), "voice_lines": (0.75, 1.6),
          "finish": (0.0, 0.0), "video": (0.012, 0.06), "export": (0.02, 0.05), "separate": (0.035, 0.11)}
LINES_PER_S = 0.18                  # lines per second of video (his run, §7)
WAITING_BED = "Waiting for the background sound"  # the dub loop's row while `separate` runs (§2.14)
NO_SEPARATOR = "This backend has no separator: the dubbed video will have the Telugu voices only."
CLAUDE_PER_LINE = (0.16, 0.19)      # Claude calls per line from scratch (§7: 310-360 calls for about 1,900 lines)

OnEvent = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class RenderSettings:
    """What the user chose for the render (job.json `settings`, §2.1)."""

    speakers: int | None = None    # the speaker count, 1-6; None: auto
    style: str = "colloquial"
    stop_at: float | None = None   # the preview's stop point (§3); None: the whole video
    speed_cap: float = 1.2
    tts_script: str = "telugu"
    presets: tuple[str, ...] = ()  # speakers voiced with their stock voice instead of their clone (`set_voice`, §4)

    def __post_init__(self) -> None:
        if self.speakers is not None and not AUTO_SPEAKERS[0] <= self.speakers <= AUTO_SPEAKERS[1]:
            raise ValueError(f"speakers must be {AUTO_SPEAKERS[0]}-{AUTO_SPEAKERS[1]} or None, not {self.speakers}")
        object.__setattr__(self, "presets", tuple(sorted({str(s) for s in self.presets})))

    def to_json(self) -> dict:
        return {"speakers": self.speakers, "style": self.style, "stopAt": self.stop_at, "speedCap": self.speed_cap,
                "ttsScript": self.tts_script, "presets": list(self.presets)}

    @classmethod
    def from_json(cls, doc: object) -> RenderSettings:
        """What `to_json` wrote; the defaults when there is nothing (a new job) or it doesn't read. An older job.json's
        `allowFreeze` is ignored: a file has no freezes (§2.11)."""
        if not isinstance(doc, dict):
            return cls()
        names = {"speakers": "speakers", "style": "style", "stopAt": "stop_at", "speedCap": "speed_cap",
                 "ttsScript": "tts_script", "presets": "presets"}
        try:
            return cls(**{f: doc[k] for k, f in names.items() if k in doc})
        except (TypeError, ValueError):
            return cls()


class RenderError(RuntimeError):
    """A render stopped for a reason the user can act on; the message says what."""


class UnusableSpeechError(RenderError):
    """Synthesis exhausted its candidates: required speech stops the job, optional alternatives keep the original."""


class _Paused(Exception):
    """`pause()` was called: the stage stopped at a safe point."""


@dataclass
class _Acct:
    """A running stage's bookkeeping. Stages of a group run side by side, each in a task of its own, so each finds its
    own through `_STAGE`."""

    key: str
    gpu_s: float = 0.0     # GPU seconds of its model calls
    worked: bool = False   # it had work to do (`_begin`); else it was served from disk
    extra: dict = field(default_factory=dict)  # more fields for its trace row


@dataclass
class Scene:
    """A scene of the render's translation (§2.8), and how ready its lines are for the dub loop."""

    no: int                        # 1, 2, ... in onset order over the whole video
    lines: list[UnitState]
    k: int                         # its first line's place in the onset order
    reviewed: bool = False         # translated and reviewed (a line the review had no usable answer for stays unclassed)
    held: bool = False             # its review failed while Claude was held: translated, unreviewed until the retry
    fit: asyncio.Task | None = None  # its fit call, when it has one
    failed: ClaudeCLIError | None = None  # the failure that released it or ended its review, this attempt
    released: bool = False         # this attempt ended without its lines translated
    settled: bool = False          # voiced, with its fix-ups done (or given up while Claude is held): §2.10

    @property
    def fitted(self) -> bool:
        return self.fit is None or self.fit.done()

    @property
    def ready(self) -> bool:
        """Its lines can be voiced: reviewed (or its review failed while Claude is held), with its fit back."""
        return (self.reviewed or self.held) and self.fitted


@dataclass
class _Fix:
    """A fix-up take for the dub loop to voice ahead of new lines (§2.10): the line, the wording to say (`tier` of
    `line`), what asked for it ("rephrase", or "retranslate": the voiced-wording review's re-translation) with its
    fixups.jsonl row, and whether it was kept, once it is voiced."""

    st: UnitState
    line: LineResult
    tier: str
    kind: str
    key: str
    done: asyncio.Future


_STAGE: contextvars.ContextVar[_Acct | None] = contextvars.ContextVar("maata_render_stage", default=None)
# The scene a translation lane is asking for: what `_release`, `_claude_failed` and `_side` (its fit) are about.
_SCENE: contextvars.ContextVar[Scene | None] = contextvars.ContextVar("maata_render_scene", default=None)


class RenderJob(Dubber):
    """One video's render. `run()` runs the stages in turn and returns the job's status; `pause()` stops it at the next
    safe point, and `run()` again continues from disk. job.json (in <cache>/<video id>/render/) says where it is."""

    def __init__(self, backend: Backend, resolver: Resolver, cache_dir: Path, ref: str,
                 settings: RenderSettings | None = None, gpu: GpuScheduler | None = None,
                 on_event: OnEvent | None = None, output_dir: Path | None = None) -> None:
        video_ref = _video_ref(ref)
        saved = _read_json(cache_dir / video_ref.video_id / "render" / "job.json") or {}
        # Without settings (a job resumed by its video id) it keeps the ones it was started with.
        self.settings = s = settings or RenderSettings.from_json(saved.get("settings"))
        # A file keeps the video's timing: both planners run with no freezes (§2.11), and an overrun drifts.
        super().__init__(backend, gpu, style=s.style, speed_cap=s.speed_cap, allow_freeze=False,
                         tts_script=s.tts_script)
        self.resolver, self.cache_dir, self.on_event = resolver, cache_dir, on_event
        self.output_dir = Path(output_dir) if output_dir is not None else default_output_dir()  # where the MP4 goes
        self.ref = video_ref
        self.video_id = self.ref.video_id
        self._dir = cache_dir / self.video_id
        self.render_dir = self._dir / "render"
        self.video: ResolvedVideo | None = None
        self.language: str | None = None
        # settle's merges: [the merged speaker's id (above every speaker's), id merged into, why, their talk seconds]
        self.merged: list[list] = []
        self._stop = threading.Event()     # pause(): stages stop at their next safe point, pyannote's hook raises
        self._running = asyncio.Lock()     # one `run()` at a time
        self._pauses = 0                   # `pause()` calls so far
        self._fp: dict | None = None       # the audio's fingerprint (size and sha256 of its source)
        self._rows: list[dict] = []        # transcript.jsonl's chunk rows
        self._active: list[str] = []       # the stages running now
        self._waiting = 0                  # stages and lanes waiting out a Claude hold: the job is `waiting`
        self._saved_at = float("-inf")
        self.scenes: list[Scene] = []      # the whole video's scenes (§2.8)
        self._scenes_cut = asyncio.Event()  # this run's scenes are cut: the dub loop starts from them
        self._separated = asyncio.Event()  # this run's background sound is done (or there is none): the dub loop's GPU
        self.skipped: dict[int, str] = {}  # unit id -> why it won't be voiced (the translator's ladder gave up)
        self._learned: set[int] = set()    # units whose k this run has learned (`_learn_k`)
        # The units in onset order and their index (§2.5): per-line queries in O(log n + k).
        self._order: list[UnitState] = []
        self._onsets: list[float] = []
        self._ended: list[float | None] = []  # _ended[k]: the latest end of the units before _order[k]
        self._max_len = 0.0
        # The dub loop (§2.9, §2.10).
        self._take_rows: dict[str, list[dict]] = {}  # takes.jsonl's rows by line key, in the order they were kept
        self._fixup_rows: dict[str, dict] = {}       # fixups.jsonl's rows by key
        self._made: set[str] = set()       # take keys the render has made (each teaches the estimator once)
        self._new: list[list] = []         # [take key, spoken, pace learned or None] of each take made since the voices
        self._said: dict[int, Said] = {}   # unit id -> its takes voiced this run, natural audio in memory until its PCM
        self._voiced_now: set[int] = set()  # unit ids voiced this run (not restored from their rows)
        self._long: set[int] = set()       # units whose take still runs long: their scene's rephrase, else flagged
        self._restored: set[int] = set()   # units the dub loop restored from their take rows this run
        self._fixes: list[_Fix] = []       # fix-up takes the loop voices ahead of new lines
        self._budget = 0.0                 # fix-up syntheses this range may have (FIXUP_SHARE)
        self._dubbing = False              # the loop is running: it can voice a fix-up
        self.flags: dict[int, list[str]] = {}  # unit id -> why its line plays as it is ("long", "unreviewed")
        # Take files the dub loop is using (written, or read from the take cache) for the line or fix-up it is on: the
        # garbage collection leaves them, and `_take_lock` makes its check and a take-cache read one step each (§2.13).
        self._pending_takes: set[str] = set()
        self._touched: set[str] | None = None  # while a sweep runs: the take files the loop has claimed since it began
        self._take_lock = threading.Lock()
        # The final plan (§2.11-§2.13).
        self._final: TimelinePlanner | None = None  # F: this run's, placed from the first line
        self._scene_of: dict[int, Scene] = {}      # unit id -> its scene (this run's)
        self.final: dict[int, Plan] = {}           # unit id -> F's plan
        self._pcm_todo: list[tuple[UnitState, Plan, str]] = []  # placed by F, waiting for their PCM (the dub loop)
        self.pcm: dict[int, tuple[str, int]] = {}  # unit id -> its PCM's key and frames, once written (or found)
        self._pcm_keys: dict[int, str] = {}        # unit id -> its PCM key, for every line F placed
        self._placed_all = False                   # F has placed every line of the range it finalizes
        self._final_lines: list[UnitState] = []    # the lines this run finalizes (§3)
        self._events: dict[int, dict] = {}         # unit id -> the `unit` trace fields of its final plan (the stats)
        self._out: dict[int, dict] = {}            # unit id -> its line in the manifest
        self._stages_at: dict[str, tuple[float, float]] = {}  # stage -> (monotonic s, done) when it started this run
        self._started = time.monotonic()
        self._ticked = (time.time(), time.monotonic())  # the last progress tick: wall clock and monotonic (`_tick`)
        self._quit: str | None = None      # Maata is quitting: the status a cancelled run writes (`quit()`, §4)
        self._final_done = False           # the final plan's stage ended this run: the TTS goes once the dub loop has
        self._extra: dict[str, str] = {}   # stage -> what it is doing, for its row ("Finishing the file…")
        self.doc = self._load(saved)

    # ---- the job --------------------------------------------------------------------------------------------------
    @property
    def stopping(self) -> bool:
        """A pause was asked for: the run is on its way to a safe point."""
        return self._stop.is_set()

    def quit(self, paused: bool = False) -> None:
        """Maata is quitting (§4): the engine cancels the run's task next. The state that graceful stop leaves is written
        now, before the run unwinds, since a model call in a worker thread may outlast the engine's exit fallback: each
        running stage back to `todo` with its start not counted, and the job `interrupted`, to continue at the next
        launch, or `paused` when a pause of the user's was still finishing (`paused`), so it stays paused. `_stage` and
        `run()` then find their stages and the job so."""
        self._quit = "paused" if paused else "interrupted"
        for key in self._active:
            st = self.doc["stages"][key]
            if st["state"] == "running":
                st["state"] = "todo"
                st["attempts"] -= 1
        now = time.monotonic()
        self.doc.update(status=self._quit, elapsed=round(self.doc["elapsed"] + now - self._started, 1),
                        updatedAt=time.time())
        self._started = now  # (run() adds the time from here to its return)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.render_dir / "job.json", self.doc)

    async def hold(self, status: str, queued_at: float | None = None) -> None:
        """The job isn't running, and the engine's queue says where it is (§4): `queued` (`queued_at`, which orders the
        queue after a restart; None keeps the one it has) or `paused` (taken out of the queue). Writes job.json and
        sends the render event."""
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.doc["status"] = status
        if queued_at is not None:
            self.doc["queuedAt"] = queued_at
        await self._save(force=True)

    def pause(self) -> None:
        """Stop at the next safe point: between ASR chunks, within a batch inside pyannote, between download chunks.
        What is done stays on disk; `run()` continues from it. Safe from any thread."""
        self._pauses += 1
        self._stop.set()
        cancel = getattr(self.b.tts, "cancel", None)
        if cancel is not None:
            cancel()

    async def run(self) -> str:
        """Run every stage in turn, each from disk where its inputs are unchanged. Returns the job's status: done,
        paused or failed. A run asked for while one is going (a resume right after a pause) waits for that one to stop,
        so two never work at once; it doesn't start at all if `pause()` came after it was asked for. Cancelling the task
        that awaits it writes the job as `quit()` said (`interrupted`, or `paused`) when Maata is quitting, and otherwise
        leaves it `running` on disk, as a crash does, which the engine takes for an interrupted render when it next
        starts (§4)."""
        asked = self._pauses
        async with self._running:
            if self._pauses != asked:
                return self.doc["status"]
            self._stop.clear()
            self._scenes_cut = asyncio.Event()
            self._separated = asyncio.Event()
            self._wake.events.pop("dub", None)
            self._wake.events.pop("finish", None)
            self._final, self._final_lines = None, []
            self._final_done = False
            self._extra.clear()
            self._started = time.monotonic()
            self._run_id = uuid.uuid4().hex
            self._ticked = (time.time(), self._started)
            self._stages_at.clear()
            self.render_dir.mkdir(parents=True, exist_ok=True)
            self.doc.update(status="running", error=None, expired=False, slept=None)
            self._trace({"event": "run_start", "backend": self.b.name, "device": self.b.device,
                         "elapsed_before_s": self.doc["elapsed"]})
            await self._save(force=True)
            try:
                if self._whole_output() is None:  # (a whole video's MP4 needs it again only if the lines change)
                    self._separator_ready()
                    self._tts_ready()
                for group in ORDER:
                    await self._all(self._stage(key) for key in group)
            except _Paused:
                self.doc["status"] = "paused"
                log.info("render %s paused in %s", self.video_id, self.doc["stage"])
            except (RenderError, ResolveError, OmniVoiceError) as e:
                log.warning("render %s failed in %s: %s", self.video_id, self.doc["stage"], e)
                self.doc.update(status="failed", error=str(e))
            except asyncio.CancelledError:
                if self._quit:  # a graceful stop (as `quit()` wrote it): its stages' starts not counted
                    self.doc.update(status=self._quit,
                                    elapsed=round(self.doc["elapsed"] + time.monotonic() - self._started, 1))
                    log.info("render %s %s in %s: Maata is quitting", self.video_id, self._quit, self.doc["stage"])
                _write_json(self.render_dir / "job.json", self.doc)
                self._trace({"event": "run_end", "status": self._quit or "cancelled",
                             "wall_s": round(time.monotonic() - self._started, 3)})
                raise
            except Exception:
                log.exception("render %s failed in %s", self.video_id, self.doc["stage"])
                self.doc.update(status="failed", error="Rendering stopped unexpectedly.")
            else:
                self.doc["status"] = "done"
            if self.doc["status"] == "done":
                await asyncio.to_thread(self._let_go_of_sources)
            self.doc["elapsed"] = round(self.doc["elapsed"] + time.monotonic() - self._started, 1)
            await self._save(force=True)
            self._trace({"event": "run_end", "status": self.doc["status"],
                         "wall_s": round(time.monotonic() - self._started, 3)})
            return self.doc["status"]

    def _let_go_of_sources(self) -> None:
        """Once the whole video's MP4 is written (§2.17, M6), the large sources go: the analysis audio (672 MB for 2.9 h;
        a later re-run decodes it again from the source audio), the video the job downloaded (`video.*` in its own
        folder, when `source.video.owned`) and the bed (render/bed/, bed.json, bed_vocals.npy: 1.86 GB for 2.9 h). Never
        a file a resolver found outside the job's folder (a bench's input), nor the demo's demo.mp4. A preview keeps
        them: continuing needs them."""
        out, source = self.doc.get("output"), self.doc["source"] if isinstance(self.doc["source"], dict) else {}
        if not isinstance(out, dict) or out.get("kind") != "whole":
            return
        if source.get("file") and (self._dir / source["file"]).is_file():  # (else the analysis audio is the source)
            self.audio = np.zeros(0, np.float32)  # the memory map, closed (Windows can't delete a mapped file)
            with contextlib.suppress(OSError):
                (self.render_dir / "audio16k.f32").unlink(missing_ok=True)
        self._drop_own_video(source.get("video"))
        shutil.rmtree(self.render_dir / "bed", ignore_errors=True)  # the bed: made again only if the lines change
        for name in ("bed.json", "bed_vocals.npy"):
            with contextlib.suppress(OSError):
                (self.render_dir / name).unlink()

    def _drop_own_video(self, rec: object) -> None:
        """Delete the video the job downloaded (`video.*` in its own folder), when job.json's `source.video` (`rec`)
        says the job owns it and its file lies directly in that folder; never a file a resolver found elsewhere."""
        rec = rec if isinstance(rec, dict) else {}
        path = self._dir / str(rec.get("file") or "")
        if rec.get("owned") and path.name.startswith("video.") and path.resolve().parent == self._dir.resolve():
            for f in self._dir.glob("video.*"):
                with contextlib.suppress(OSError):
                    f.unlink()

    def _on_cancel(self) -> None:
        self._stop.set()  # the model call being waited out (pyannote) stops at its next check
        cancel = getattr(self.b.tts, "cancel", None)
        if cancel is not None:
            cancel()

    async def _all(self, stages: Iterable[Coroutine[Any, Any, None]]) -> None:
        """Run `stages` (stages, or translation lanes) side by side and return once every one has stopped. A pause stops
        each at its next safe point; an error in one cancels the others, since it fails the job. Raises the first error,
        else `_Paused` if one paused."""
        coros = list(stages)
        if len(coros) == 1:
            return await coros[0]
        tasks = [asyncio.create_task(c) for c in coros]
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                if any(not t.cancelled() and not isinstance(t.exception(), (_Paused, type(None))) for t in done):
                    for t in pending:
                        t.cancel()
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.wait(tasks)
            raise
        errors = [e for t in tasks if not t.cancelled() and (e := t.exception()) is not None]
        errors.sort(key=lambda e: isinstance(e, _Paused))
        if errors:
            raise errors[0]

    async def _stage(self, key: str) -> None:
        st = self.doc["stages"][key]
        acct = _Acct(key)
        token = _STAGE.set(acct)
        self._active.append(key)
        self.doc["stage"] = self._first_active()
        t0 = time.perf_counter()
        outcome = "done"
        self._trace({"event": "stage_start", "key": key})
        try:
            await getattr(self, "_" + key)()
        except asyncio.CancelledError:
            outcome = "cancelled"
            # The engine is stopping, which is not a kill: a download in its worker thread stops at its next chunk, and
            # the stage's start doesn't count as an interrupted one (the coarse step, the crash-loop guard).
            self._stop.set()
            if st["state"] == "running":
                st["state"] = "todo"
                st["attempts"] -= 1
            raise
        except Exception as e:
            paused = isinstance(e, (_Paused, Cancelled)) or self._stop.is_set()
            outcome = "paused" if paused else "failed"
            if st["state"] == "running":
                st["state"] = "todo" if paused else "failed"
                st["attempts"] -= paused  # a pause is not an interrupted start (the crash-loop guard, §4)
            if paused:
                raise _Paused from None
            self.doc["stage"] = key  # where the job failed
            raise
        finally:
            _STAGE.reset(token)
            self._active.remove(key)
            elapsed = time.perf_counter() - t0
            st["seconds"] = round(st["seconds"] + elapsed, 2)
            st["gpu_s"] = round(st["gpu_s"] + acct.gpu_s, 2)
            self._trace({"event": "stage_timing", "key": key, "outcome": outcome,
                         "wall_s": round(elapsed, 3), "gpu_s": round(acct.gpu_s, 3),
                         "cached": not acct.worked})
        st["state"] = "done"
        st["span"] = round(_span(key, float(self.doc["duration"] or 0.0), self.settings.stop_at), 1)  # (`_prior_left`)
        # (A stage finishing while a pause stops the others, the video download, leaves the job where they stopped.)
        self.doc["stage"] = self._first_active() or (self.doc["stage"] if self._stop.is_set() else key)
        self._trace({**acct.extra, "event": "stage", "key": key, "cached": not acct.worked, "seconds": st["seconds"],
                     "gpu_s": st["gpu_s"], "done": st["done"], "total": st["total"]})
        await self._save(force=True)

    def _first_active(self) -> str | None:
        """The stage job.json shows: the first in order of those running."""
        keys = [key for key, _, _ in STAGES]
        return min(self._active, key=keys.index, default=None)

    async def _begin(self, key: str) -> None:
        """The stage has work to do (it isn't served from disk): count the start in `attempts`, the starts since it last
        finished, and in `interrupted` those that found an earlier start still `running`: the engine died during it
        with no exception (an out-of-memory kill; a pause, a failure or the engine stopping all leave another state)."""
        st = self.doc["stages"][key]
        if st["state"] == "done":
            st["attempts"], st["interrupted"] = 1, 0
        else:
            st["attempts"] += 1
            st["interrupted"] += st["state"] == "running"
        st["state"] = "running"
        self._stages_at.pop(key, None)  # its rate is measured from its first progress in this run (the ETA, §4)
        if (acct := _STAGE.get()) is not None:
            acct.worked = True
        await self._save(force=True)

    def _load(self, old: dict) -> dict:
        """job.json as it was on disk (`old`), or a new one; the settings are this job's."""
        now = time.time()
        doc = {"version": 1, "videoId": self.video_id, "title": "", "channel": "", "duration": 0.0, "source": None,
               "settings": None, "status": "paused", "stage": None, "finalUntil": None, "queuedAt": None, "stages": {},
               "brief": None, "coverage": None, "report": None, "slept": None, "output": None, "part": None, "bed": None,
               "error": None, "createdAt": now, "updatedAt": now, "expired": False, "elapsed": 0.0}
        doc.update({k: v for k, v in old.items() if k in doc})
        doc["settings"] = self.settings.to_json()
        stages = doc["stages"] if isinstance(doc["stages"], dict) else {}
        doc["stages"] = {key: {"state": "todo", "done": 0, "total": 0, "unit": unit, "seconds": 0.0, "gpu_s": 0.0,
                               "attempts": 0, "interrupted": 0, "span": None, **(stages.get(key) or {})}
                         for key, _, unit in STAGES}
        return doc

    async def _save(self, force: bool = False) -> None:
        """Write job.json (atomically) and send the render event; progress alone does so at most every PROGRESS_EVERY s."""
        if not force and time.monotonic() - self._saved_at < PROGRESS_EVERY:
            return
        self._saved_at = time.monotonic()
        self.doc["updatedAt"] = time.time()
        _write_json(self.render_dir / "job.json", self.doc)
        if self.on_event is not None:
            await self.on_event(self.snapshot())

    def _progress(self, key: str, done: float, total: float) -> None:
        """A stage's progress (on the event loop; a worker thread goes through `call_soon_threadsafe`)."""
        st = self.doc["stages"][key]
        st["done"], st["total"] = done, total
        if st["state"] == "running":
            self._stages_at.setdefault(key, (time.monotonic(), done))
        slept = self._tick()
        if slept or time.monotonic() - self._saved_at >= PROGRESS_EVERY:
            self._side(self._save(force=slept))

    def _tick(self) -> bool:
        """A progress tick notices a sleep (§4): `time.monotonic()` (mach_absolute_time on macOS) stops while the Mac
        sleeps and the wall clock doesn't, so a wall-clock gap since the last tick more than SLEEP_GAP s past the
        monotonic one is a sleep, kept as job.json's `slept`: the last tick before it and the first after it (wall
        clock). Returns whether this tick noticed one."""
        wall, mono = time.time(), time.monotonic()
        (w0, m0), self._ticked = self._ticked, (wall, mono)
        if (wall - w0) - (mono - m0) <= SLEEP_GAP:
            return False
        self.doc["slept"] = {"at": round(w0, 1), "resumed": round(wall, 1)}
        log.info("render %s: the Mac slept for about %.0f s", self.video_id, (wall - w0) - (mono - m0))
        return True

    def snapshot(self) -> dict:
        """The job as the UI's `render` message shows it (§4), with each stage's ETA and the job's."""
        d, now = self.doc, time.monotonic()
        running = d["status"] in ("running", "waiting")
        etas = {key: self._eta(key, now) for key, _, _ in STAGES}
        eta = sum(_group_seconds(group, etas.__getitem__) for group in ORDER)
        if d["status"] == "waiting":  # held by a usage limit: at least until it resets, then the work left
            eta += max(self._claude_hold - now, 0.0)
        duration = float(d["duration"] or 0.0)
        return {"type": "render", "videoId": self.video_id, "status": d["status"], "stage": d["stage"],
                "stages": [{"key": key, "label": label, **{f: d["stages"][key][f] for f in
                                                          ("state", "done", "total", "unit", "seconds")},
                            "state": shown_state(key, d["stages"][key], duration, self.settings.stop_at),
                            "eta": round(etas[key]) if running else None,
                            **({"extra": self._extra[key]} if key in self._extra and running else {})}
                           for key, label, _ in STAGES],
                "stopAt": self.settings.stop_at, "finalUntil": d["finalUntil"],
                "eta": round(eta) if running else None,
                "elapsed": round(d["elapsed"] + (now - self._started if running else 0.0), 1),
                "output": output_view(d), "slept": d["slept"], "coverage": d["coverage"], "report": d["report"],
                "error": d["error"]}

    def _eta(self, key: str, now: float) -> float:
        """Seconds a stage has left (§4): its remaining work at its own rate once it has ETA_AFTER s of work behind it in
        this run, else its share left of the static prior (§7) for its range. With no separator the background sound
        is served at once: none."""
        st = self.doc["stages"][key]
        if key == "separate" and self.b.separator is None:
            return 0.0
        if st["state"] == "running" and (at := self._stages_at.get(key)) is not None:
            t0, d0 = at
            if now - t0 >= ETA_AFTER and st["done"] > d0:
                return max(st["total"] - st["done"], 0) * (now - t0) / (st["done"] - d0)
        return _prior_left(key, st, float(self.doc["duration"] or 0.0), self.settings.stop_at)

    async def notify(self, msg: dict) -> None:
        if self.on_event is not None:
            await self.on_event({**msg, "videoId": self.video_id})

    async def _on_gpu(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
        if self._stop.is_set():
            raise _Paused  # a pause stops GPU work at the next call (§2.1)
        @functools.wraps(fn)
        def admitted(*a: Any, **kw: Any) -> Any:
            # Admission may have waited behind another stage. A pause during that wait must not launch a worker.
            if self._stop.is_set():
                raise _Paused
            bind = getattr(self.b.tts, "bind_cancel_event", None)
            if bind is not None:
                bind(self._stop)
            return fn(*a, **kw)
        queued_at = time.perf_counter()
        out, seconds = await super()._on_gpu(admitted, *args, **kwargs)
        self._gpu_work(fn, seconds, queued_at)
        return out, seconds

    def _gpu_work(self, fn: Callable[..., Any], seconds: float, queued_at: float) -> None:
        """Completed scheduled work in this run, including restored-take reads; never historical take costs.
        These are serialized operation durations, not a measurement of hardware utilization."""
        acct = _STAGE.get()
        if acct is not None:
            acct.gpu_s += seconds
        self._trace({"event": "gpu_work", "stage": acct.key if acct else None,
                     "operation": getattr(fn, "__name__", type(fn).__name__),
                     "run_s": round(seconds, 4),
                     "queue_wait_s": round(max(0.0, time.perf_counter() - queued_at - seconds), 4)})

    def _cut(self, a: float, b: float) -> np.ndarray:
        return np.array(super()._cut(a, b))  # out of the memory map: a model call gets a plain array

    # ---- fetch (§2.2) ---------------------------------------------------------------------------------------------
    async def _fetch(self) -> None:
        """The analysis audio. When fetch finished before and its source and audio16k.f32 are still there, unchanged, it
        is served from them and job.json's metadata, with no call to the resolver, so a render resumes offline or while
        YouTube asks for a sign-in check. Otherwise the resolver's metadata, the download if the cache hasn't the
        audio, and one decode to audio16k.f32."""
        dest = self.render_dir / "audio16k.f32"
        st = self.doc["stages"]["fetch"]
        got = await asyncio.to_thread(self._fetched, dest)
        video, src, fp = got if got is not None else await self._resolve(dest)
        if not dest.is_file():  # a finished render let go of it (§2.1): decoded again from its source, offline
            await self._begin("fetch")
            await asyncio.to_thread(decode_audio_to, src, dest)
        if dest.stat().st_size < 4 * SR_ANALYSIS:  # under a second
            raise RenderError("This video has no audio to dub.")
        if src is not None and self.doc["source"].get("audioStart") is None:  # sample 0's time in its stream (§2.2)
            self.doc["source"]["audioStart"] = round(await asyncio.to_thread(audio_start, src), 6)
        self.video, self._fp = video, fp
        self.audio = np.memmap(dest, np.float32, mode="r")
        self.doc["duration"] = round(self._duration(), 2)
        st["done"] = st["total"] = round((src or dest).stat().st_size / 1e6, 1)

    def _fetched(self, dest: Path) -> tuple[ResolvedVideo, Path | None, dict] | None:
        """What an earlier fetch left, if it still matches its inputs: the video from job.json, its source and the
        fingerprint. None when fetch has to run."""
        st, source = self.doc["stages"]["fetch"], self.doc["source"]
        if st["state"] != "done" or not isinstance(source, dict):
            return None
        src = self._dir / source["file"] if source.get("file") else None
        if (src is None and not dest.is_file()) or (src is not None and not src.is_file()):
            return None  # (the demo's analysis audio is its source)
        fp = _fingerprint(src or dest)
        if st.get("inputs") != {"audio": fp, "sr": SR_ANALYSIS}:
            return None
        d = self.doc
        chapters = tuple((float(a), str(t)) for a, t in source.get("chapters") or ())
        video = ResolvedVideo(self.video_id, d["title"], float(d["duration"]), d["channel"], src,
                              source.get("description") or "", chapters, tuple(source.get("tags") or ()))
        return video, src, fp

    async def _resolve(self, dest: Path) -> tuple[ResolvedVideo, Path | None, dict | None]:
        """Fetch through the resolver: metadata, the audio (downloaded unless the cache has it) and, unless audio16k.f32
        is already the decode of that audio, the decode."""
        st = self.doc["stages"]["fetch"]
        video = await asyncio.to_thread(self.resolver.resolve, self.ref, self.cache_dir, download=False)
        self._describe(video)
        src = video.audio_path
        fp = await asyncio.to_thread(_fingerprint, src) if src is not None else None
        if fp is None or not dest.is_file() or st.get("inputs") != {"audio": fp, "sr": SR_ANALYSIS}:
            await self._begin("fetch")
            need = DISK_PER_HOUR * video.duration / 3600 + DISK_HEADROOM
            free = shutil.disk_usage(self.render_dir).free
            if free < need:
                raise RenderError(f"Rendering this video needs about {need / 1e9:.1f} GB of free disk space, and "
                                  f"{free / 1e9:.1f} GB is free.")
            if src is None:
                loop = asyncio.get_running_loop()

                def progress(done: int, total: int) -> None:
                    if self._stop.is_set():
                        raise Cancelled("download paused")
                    loop.call_soon_threadsafe(self._progress, "fetch", round(done / 1e6, 1), round(total / 1e6, 1))

                video = await asyncio.to_thread(self.resolver.resolve, self.ref, self.cache_dir, download=True,
                                                progress=progress)
                self._describe(video)
                src = video.audio_path
                fp = await asyncio.to_thread(_fingerprint, src) if src is not None else None
            if self._stop.is_set():
                raise _Paused
            if src is not None:
                await asyncio.to_thread(decode_audio_to, src, dest)
                self.doc["source"]["audioStart"] = round(await asyncio.to_thread(audio_start, src), 6)
            else:  # a resolver with no audio file: its samples are the analysis audio
                audio = await asyncio.to_thread(self.resolver.load_audio, video)
                await asyncio.to_thread(_save_f32, audio, dest)
                fp = await asyncio.to_thread(_fingerprint, dest)
            st["inputs"] = {"audio": fp, "sr": SR_ANALYSIS}
        return video, src, fp

    def _describe(self, video: ResolvedVideo) -> None:
        """job.json's copy of the video's metadata, its audio file (by name in the job's folder; a file elsewhere, a
        bench's, by its absolute path, so a resume finds it) and its thumbnail: what `_fetched` serves it from. What the
        video stage recorded stays, and the audio's start while the audio is the same file."""
        old = self.doc["source"] if isinstance(self.doc["source"], dict) else {}
        src = video.audio_path
        file = None if src is None else src.name if src.parent.resolve() == self._dir.resolve() else str(src.resolve())
        thumb = video.thumb_path
        self.doc.update(title=video.title, channel=video.channel, duration=video.duration,
                        source={"file": file, "audioStart": old.get("audioStart") if old.get("file") == file else None,
                                "thumb": thumb.name if thumb is not None else old.get("thumb"),
                                "video": old.get("video"), "description": video.description,
                                "chapters": [list(c) for c in video.chapters], "tags": list(video.tags)})

    # ---- speakers (§2.3) ------------------------------------------------------------------------------------------
    async def _speakers(self) -> None:
        """pyannote over the whole file (auto 1-6, or the user's count), settled, with the ids an earlier run on the same
        audio gave the same people; served from diarization.json when its inputs are unchanged. The registry is always
        rebuilt from that file, so a resumed render has exactly the speakers a fresh one has."""
        path = self.render_dir / "diarization.json"
        k = self.settings.speakers
        inputs = {"audio": self._fp, "diarizer": type(self.b.diarizer).__name__,
                  "model": _model_rev(self.b.name, "diarization"), "speakers": k or "auto",
                  "bounds": None if k else list(AUTO_SPEAKERS),
                  "settle": {"minTalk": MIN_TALK, "minShare": MIN_SHARE, "sameCos": SAME_COS, "floorCap": FLOOR_CAP},
                  "version": DIAR_VERSION}
        prev = _read_json(path)
        fresh = prev is None or prev.get("inputs") != inputs
        if fresh:
            doc = await self._diarize(inputs, k, prev)
            _write_json(path, doc)
        else:
            doc = prev
        self.registry = SpeakerRegistry.restore(doc, 0.0, self._duration())
        self.merged = doc.get("merged", [])
        self.voices = {sid: VoiceState() for sid in self.registry.speakers}
        st = self.doc["stages"]["speakers"]
        st["done"] = st["total"] = 1
        await self._speakers_found(doc, fresh)

    async def _diarize(self, inputs: dict, k: int | None, prev: dict | None) -> dict:
        await self._begin("speakers")
        # A start since it last finished was killed (for memory, likely: that leaves no exception): the coarse step. Not
        # after a pause or an ordinary error, which would lower the whole video's resolution for nothing.
        step = COARSE_STEP if self.doc["stages"]["speakers"]["interrupted"] else None
        count = {"num_speakers": k} if k else {"min_speakers": AUTO_SPEAKERS[0], "max_speakers": AUTO_SPEAKERS[1]}
        loop = asyncio.get_running_loop()

        def progress(share: float) -> None:
            loop.call_soon_threadsafe(self._progress, "speakers", round(share, 3), 1)

        block, seconds = await self._on_gpu(self.b.diarizer.diarize, self.audio, **count, step=step, progress=progress,
                                            cancel=self._stop)
        reg = SpeakerRegistry()
        first = reg.add_block(block)
        before = dict(reg.speakers)
        merged = settle(reg, hinted=k is not None)
        if prev is not None and (prev.get("inputs") or {}).get("audio") == self._fp:
            reg.rename(match_ids([tuple(x) for x in prev.get("exclusive", [])], reg))  # the same people keep their ids
        into = {a: b for a, b, _ in merged}

        def final(sid: str) -> str:
            while sid in into:
                sid = into[sid]
            return before[sid].id

        # A merged speaker's id from before the renumbering may now be a live speaker's: name the merged ones above
        # every live id instead, in order of first speech, so "S3 merged into S1" never names someone still there.
        top = max((int(sid[1:]) for sid in reg.speakers), default=0)
        gone = sorted((a for a, _, _ in merged), key=lambda a: (before[a].first_at, a))
        name = {a: f"S{top + n}" for n, a in enumerate(gone, 1)}
        total = sum(g.talk_seconds for g in reg.speakers.values())
        speakers = [{"id": sid, "talkSeconds": round(g.talk_seconds, 2),
                     "share": round(g.talk_seconds / total, 4) if total > 0 else 0.0, "firstAt": round(g.first_at, 2),
                     "turns": sum(1 for t in reg.turns_in(0.0, block.end) if t.speaker == sid)}
                    for sid, g in sorted(reg.speakers.items(), key=lambda x: x[1].first_at)]
        rows = [[name[a], final(b), why, round(before[a].talk_seconds, 2)] for a, b, why in merged]
        rss = _peak_rss_bytes()  # the live check of §7: the engine's peak so far, and what the clustering took
        log.info("speakers: %d in %.0f s (step %s, %s embeddings clustered; %d merged: %s); peak RSS %s",
                 len(reg.speakers), seconds, block.step, block.embeddings, len(merged), rows,
                 "unknown" if rss is None else f"{rss / 1e9:.1f} GB")
        _STAGE.get().extra = {"step": block.step, "embeddings": block.embeddings, "speakers": len(reg.speakers),
                              "merged": len(merged), "peak_rss_bytes": rss}
        return {"inputs": inputs, "step": block.step, "labels": {lab: final(sid) for lab, sid in sorted(first.items())},
                **reg.dump(), "merged": rows, "speakers": speakers}

    async def _speakers_found(self, doc: dict, fresh: bool = False) -> None:
        if self.on_event is not None:
            await self.on_event(found_message(self.video_id, doc, self.registry, self.doc, self._duration(), fresh))

    # ---- transcript (§2.4) ----------------------------------------------------------------------------------------
    async def _transcript(self) -> None:
        """ASR from 0 to the end in chunks, whatever the range (the brief and the scenes need the whole transcript), each
        chunk a row of transcript.jsonl as soon as it is heard; a resumed render continues after the last row, in the
        language the first chunk was heard in. Whisper is let go afterwards."""
        path = self.render_dir / "transcript.jsonl"
        inputs = {"audio": self._fp, "transcriber": type(self.b.transcriber).__name__, "model": _model_rev(self.b.name, "asr"),
                  "chunk": CHUNK, "pad": CHUNK_PAD, "style": hashlib.sha256(STYLE_PROMPT.encode()).hexdigest()[:12],
                  "version": ASR_VERSION}
        rows = _read_rows(path)
        if rows and rows[0].get("inputs") == inputs:  # not the diarization: another speaker count moves no speech
            self._rows = [r for r in rows[1:] if {"a", "b", "next", "words"} <= r.keys()]
        else:
            _write_json_rows(path, [{"inputs": inputs}])
            self._rows = []
        self.language = self._rows[0].get("language") if self._rows else None
        dur = self._duration()
        a = self._rows[-1]["next"] if self._rows else 0.0
        self._progress("transcript", round(min(a, dur), 1), round(dur, 1))
        if a < dur - 0.2:
            await self._begin("transcript")
            while a < dur - 0.2:
                if self._stop.is_set():
                    raise _Paused
                row = await self._chunk(a, dur)
                _append_row(path, row)
                self._rows.append(row)
                a = row["next"]
                self._progress("transcript", round(min(a, dur), 1), round(dur, 1))
        release = getattr(self.b.transcriber, "release", None)
        if release is not None:
            await self._on_gpu(release)
        self.doc["stages"]["transcript"]["done"] = round(dur, 1)

    async def _chunk(self, a: float, dur: float) -> dict:
        """The ASR chunk from `a`: CHUNK s and CHUNK_PAD more so its last sentence can finish, the diarized speech for
        the coverage guard, `fix_words`, cut at a sentence end, and its guards' findings. Returns its transcript.jsonl row:
        `next` is where the next chunk starts."""
        b = min(a + CHUNK, dur)
        lo, hi = max(0.0, a - 0.3), min(dur, b + CHUNK_PAD)
        speech = [(max(t.start, a) - lo, min(t.end, hi) - lo) for t in self.registry.turns_in(a, hi, exclusive=False)]
        asked = time.perf_counter()
        tr, dt = await self._on_gpu(self.b.transcriber.transcribe, self._cut(lo, hi), self.language, speech=speech)
        waited = time.perf_counter() - asked - dt
        self.language = self.language or tr.language
        words = fix_words([type(w)(w.text, w.start + lo, w.end + lo, w.confidence) for w in tr.words])
        words = [w for w in words if w.start >= a - 0.05]  # the previous chunk owns words that started before the cursor
        cut = self._sentence_cut(words, b, last=(hi >= dur - 0.01))
        owned = [w for w in words if w.end <= cut + 1e-6]
        nxt = cut if owned else b
        checks = self._asr_checks(tr, lo, a, nxt, owned)
        self._trace({"event": "asr", "a": round(lo, 2), "b": round(hi, 2), "run_s": round(dt, 3),
                     "lock_wait_s": round(waited, 3), "priority": self._gpu_priority(), "words": len(tr.words), **checks})
        log.info("heard %.0f-%.0f s: %d words (language %s)", a, nxt, len(owned), self.language)
        return {"a": a, "b": nxt, "next": max(nxt, a + 1.0), "language": self.language,
                "words": [[w.text, w.start, w.end, w.confidence] for w in owned], "checks": checks}

    # ---- units (§2.5) ---------------------------------------------------------------------------------------------
    async def _units(self) -> None:
        await self._begin("units")
        states = await asyncio.to_thread(self._segment, list(self._rows))  # ~28k words for 2.9 h: off the event loop
        self._index(states)
        st = self.doc["stages"]["units"]
        st["done"] = st["total"] = len(states)

    def _index(self, states: list[UnitState]) -> None:
        """The job's units, `states` in onset order, and their index for the per-line queries."""
        self.units = {st.unit.id: st for st in states}
        self._order = states
        self._onsets = [st.unit.start for st in states]
        self._ended = [None]
        for st in states:
            last = self._ended[-1]
            self._ended.append(st.unit.end if last is None else max(last, st.unit.end))
        self._max_len = max((st.unit.end - st.unit.start for st in states), default=0.0)

    def _segment(self, rows: list[dict]) -> list[UnitState]:
        """The sentence units of the whole transcript: each chunk segmented on its own (with the exclusive turns of its
        stretch and the text before it), then one `merge_fragments` pass over them all, so a sentence split at a chunk cut
        rejoins. Ids are onset order. Built from the rows alone, so a resumed render gets the units a fresh one does."""
        units: list[SourceUnit] = []
        for row in rows:
            a, b = row["a"], row["b"]
            words = [TimedWord(text, start, end, conf) for text, start, end, conf in row["words"]]
            before = [u for u in units[-8:] if u.end <= a + 0.05]
            prev_text = max(before, key=lambda u: u.end).text if before else ("" if a <= 0.05 else None)
            units += segment(words, self.registry.turns_in(a, b, exclusive=True), prev_text=prev_text)
        units = sorted(merge_fragments(units), key=lambda u: (u.start, u.end))
        out = []
        for k, u in enumerate(units):
            u.id = k
            st = UnitState(u, units[k + 1].start if k + 1 < len(units) else None, speech_s=self._speech(u),
                           music=asr_guard.music_like(u, self.registry.turns_in(u.start, u.end)))
            if st.music:
                log.warning("unit %d %.1f-%.1f s looks sung: flagged for the report, still dubbed", u.id, u.start, u.end)
            out.append(st)
        return out

    # ---- voices (§2.6) --------------------------------------------------------------------------------------------
    async def _voices(self) -> None:
        """Each speaker's voice, most talk first, by today's build path (`_voice_spans`, `_voice_from`, then `_calibrate`:
        ADR-017), which now chooses from the whole video; a speaker with too little clean speech gets a
        preset. What each voice was built from and its calibration go to voices.json as soon as it is made. A resumed
        render rebuilds a stored voice from its spans and re-fits its pace from its stored calibration, with no
        synthesis (`_restore_voice`), and the paces of the takes voiced since are replayed into the estimator
        (`_load_takes`)."""
        path = self.render_dir / "voices.json"
        # From the calibrations alone, then the takes' paces: a run after a pause on the same job counts none twice.
        self.estimator = DurationEstimator()
        tts = self.b.tts
        inputs = {"audio": self._fp, "tts": type(tts).__name__, "model": self._tts_revision(),
                  "cfg": getattr(tts, "cfg_weight", None), "exaggeration": getattr(tts, "exaggeration", None),
                  "cfmSteps": getattr(tts, "cfm_steps", None), "ttsScript": self.tts_script,
                  "calibration": _digest([[w.spoken, [list(e) for e in w.english]] for w in CALIBRATION_TE]),
                  "calibrationTts": _digest([self._tts_text(w) for w in CALIBRATION_TE]),
                  "version": VOICE_VERSION}
        if hasattr(tts, "cache_identity"):
            inputs["synthesis"] = tts.cache_identity
        prev = _read_json(path)
        prior = prev.get("inputs") if prev else None
        prior = prior if isinstance(prior, dict) else {}
        same_voice = {k: v for k, v in prior.items() if k != "calibrationTts"} == \
                     {k: v for k, v in inputs.items() if k != "calibrationTts"}
        entries = dict(prev.get("speakers") or {}) if prev and same_voice else {}
        # A calibration-only text change needs new pace samples, not new reference selection. Legacy Telugu input
        # is exactly its stored spoken text; legacy Latin reconstruction cannot be proved unchanged from those pairs.
        legacy_tts = prior.get("calibrationTts", inputs["calibrationTts"] if self.tts_script != "latin" else None)
        for entry in entries.values():
            entry.setdefault("calibrationTts", legacy_tts)  # per voice: a pause can leave only some recalibrated
        reg = self.registry
        order = sorted(reg.speakers, key=lambda sid: (-reg.speakers[sid].talk_seconds, sid))
        if getattr(tts, "native_voice", False):
            await self._native_voices(order, entries, inputs, path)
            return
        began = False
        for n, sid in enumerate(order):
            self._progress("voices", n, len(order))
            if self._stop.is_set():
                raise _Paused
            if sid in self.settings.presets:  # voiced with their stock voice (`set_voice`); a clone they have is kept
                await self._preset(_preset_name(sid), VoiceCost())
                self.voices.setdefault(sid, VoiceState()).use_preset = True
                continue
            if sid in entries:
                recalibrate = entries[sid]["calibrationTts"] != inputs["calibrationTts"] \
                              and entries[sid].get("how") != "preset"
                if recalibrate and not began:
                    await self._begin("voices")
                    began = True
                if await self._restore_voice(sid, entries[sid], recalibrate=recalibrate):
                    entries[sid]["calibrationTts"] = inputs["calibrationTts"]
                    if recalibrate:
                        _write_json(path, {"inputs": inputs, "speakers": entries})
                    continue
            if not began:
                await self._begin("voices")
                began = True
            entries[sid] = await self._make_voice(sid)
            entries[sid]["calibrationTts"] = inputs["calibrationTts"]
            _write_json(path, {"inputs": inputs, "speakers": entries})
        _write_json(path, {"inputs": inputs, "speakers": {sid: entries[sid] for sid in order if sid in entries}})
        for f in (self.render_dir / "voices").glob("*.npy"):  # speakers merged away, or gone with another count
            if f.stem not in reg.speakers:
                f.unlink(missing_ok=True)
        self._load_takes(*await asyncio.to_thread(lambda: (_read_rows(self.render_dir / "takes.jsonl"),
                                                           _read_rows(self.render_dir / "fixups.jsonl"))))
        st = self.doc["stages"]["voices"]
        st["done"] = st["total"] = len(order)

    async def _native_voices(self, order: list[str], entries: dict, inputs: dict, path: Path) -> None:
        """Automatic Telugu speech without source-speaker conditioning (ADR-028).

        English references are deliberately unused. Calibration, take validation and the timing planner are shared
        with the clone path; provenance states `native`, never `cloned`. Reuse calibration only for the exact model,
        reference and actual TTS text. The shared automatic policy needs one pace calibration, not one per detected speaker.
        """
        if self._stop.is_set():
            raise _Paused
        cost = VoiceCost()
        voice = await self._preset("native", cost)
        key = self._voice_key("native", voice, "native:" + _digest(self.b.tts.cache_identity))
        previous = next((e for e in entries.values() if e.get("how") == "native"
                         and e.get("calibrationTts") == inputs["calibrationTts"]
                         and e.get("calibration")), None)
        kept: list[tuple[str, float, object]] = []
        audio = None
        if previous is None and order:
            await self._begin("voices")
            t0 = time.perf_counter()
            await self._calibrate("native", key, voice, cost, self._gpu_priority(), keep=kept)
            self.calibrate_s += time.perf_counter() - t0
            if not kept:
                raise RenderError("The Telugu voice could not produce usable calibration speech. Check the voice setup before retrying.")
            audio, _ = await self._on_gpu(self.b.tts.vocode, kept[0][2], 1.0, cost=cost)
            pairs = [[spoken, seconds] for spoken, seconds, _ in kept]
        else:
            pairs = previous["calibration"] if previous else []
            if pairs:
                self.estimator.calibrate(key, [(spoken, seconds) for spoken, seconds in pairs])
        if order and audio is None and any(not (self.render_dir / "voices" / f"{sid}.npy").is_file() for sid in order):
            audio, _ = await self._on_gpu(self.b.tts.synthesize, self._tts_text(CALIBRATION_TE[0]), voice, "te", cost=cost)
        saved = {}
        for n, sid in enumerate(order):
            if self._stop.is_set():
                raise _Paused
            v = self.voices.setdefault(sid, VoiceState())
            v.voice, v.kind, v.ref_seconds, v.status, v.key = voice, VoiceKind.NATIVE, 0.0, "native", key
            v.use_preset = False  # legacy clone/stock switches cannot change this mode
            sample = self.render_dir / "voices" / f"{sid}.npy"
            if audio is not None:
                sample.parent.mkdir(exist_ok=True)
                _save_pcm(sample, np.asarray(audio, np.float32))
            saved[sid] = {"how": "native", "refSeconds": 0.0, "calibration": pairs,
                          "calibrationTts": inputs["calibrationTts"], "sample": sample.is_file(),
                          "key": {"voice": key.voice, "cfg": key.cfg, "exaggeration": key.exaggeration,
                                  "reference": key.reference},
                          "pace": round(self.estimator.rate(key), 3), "overhead": round(self.estimator.overhead(key), 3)}
            self._progress("voices", n + 1, len(order))
        _write_json(path, {"inputs": inputs, "speakers": saved})
        await self.notify({"type": "voice_ready", "videoId": self.video_id})
        self._load_takes(*await asyncio.to_thread(lambda: (_read_rows(self.render_dir / "takes.jsonl"),
                                                         _read_rows(self.render_dir / "fixups.jsonl"))))
        self.doc["stages"]["voices"]["done"] = self.doc["stages"]["voices"]["total"] = len(order)

    async def _make_voice(self, sid: str) -> dict:
        """Build a speaker's voice (`_voice_spans`, `_voice_from`) and calibrate it, with the voice key's reference
        versioned, and their Hear voice sample (§5): the first usable calibration take, vocoded, so watermarked, like
        every take: synthetic Telugu of a fixed sentence, never the source audio (a TTS without mel takes says that
        sentence). Returns the speaker's voices.json entry."""
        cost, priority = VoiceCost(), self._gpu_priority()
        v = self.voices.setdefault(sid, VoiceState())
        sample = self.render_dir / "voices" / f"{sid}.npy"
        t0 = time.perf_counter()
        how, best, clips = self._voice_spans(sid)
        voice, secs, how, reference = await self._voice_from(how, best, clips, cost, priority)
        if voice is None:
            name = _preset_name(sid)
            await self._preset(name, cost)  # made now, holding the GPU, rather than on a line's path
            v.voice, v.kind, v.ref_seconds, v.status, v.key = None, VoiceKind.PRESET, secs, "preset", None
            sample.unlink(missing_ok=True)
            log.warning("%s has only %.1f s of clean speech: voiced with a preset", sid, secs)
            return {"how": "preset", "preset": name, "refSeconds": secs, "sample": False}
        key = self._voice_key(sid, voice, f"{VOICE_VERSION}:{reference}")
        log.info("cloned %s from %.1f s of reference (%s)", sid, secs, how)
        built = time.perf_counter()
        kept: list[tuple[str, float, object]] = []
        used = await self._calibrate(sid, key, voice, cost, priority, keep=kept)
        calibrate_s = time.perf_counter() - built
        self.calibrate_s += calibrate_s
        v.voice, v.kind, v.ref_seconds, v.status, v.key = voice, VoiceKind.CLONED, secs, "cloned", key
        audio = None
        if kept and hasattr(self.b.tts, "vocode"):
            audio, _ = await self._on_gpu(self.b.tts.vocode, kept[0][2], 1.0, cost=cost, priority=priority)
        elif not hasattr(self.b.tts, "synthesize_mel"):  # no mel takes, so no calibration (the mock): the sentence said
            audio, _ = await self._on_gpu(self.b.tts.synthesize, self._tts_text(CALIBRATION_TE[0]), voice, "te",
                                          cost=cost, priority=priority)
        if audio is not None:
            sample.parent.mkdir(exist_ok=True)
            _save_pcm(sample, np.asarray(audio, np.float32))
        else:
            sample.unlink(missing_ok=True)
        self._trace({"event": "clone", "speaker": sid, "ref_seconds": round(secs, 1), "method": how,
                     "build_s": round(built - t0, 2), "calibrate_s": round(calibrate_s, 2),
                     "lock_wait_s": round(cost.lock_wait_s, 3), "calibration_takes": used,
                     "pace": round(self.estimator.rate(key), 2), "overhead_s": round(self.estimator.overhead(key), 3)})
        # Spans unrounded: a rebuild must cut the very samples the reference hash was taken of.
        return {"how": how, "timbre": list(best) if best else None, "clips": [list(c) for c in clips],
                "speech": self._own_speech(sid, ([best] if best else []) + clips), "refSeconds": secs,
                "hash": reference,
                "key": {"voice": key.voice, "cfg": key.cfg, "exaggeration": key.exaggeration, "reference": key.reference},
                "calibration": [[spoken, seconds] for spoken, seconds, _ in kept],
                "pace": round(self.estimator.rate(key), 3), "overhead": round(self.estimator.overhead(key), 3),
                "sample": sample.is_file()}

    async def _restore_voice(self, sid: str, entry: dict, *, recalibrate: bool = False) -> bool:
        """A stored voice back, with no synthesis. A preset while the speaker still has too little clean speech for a
        clone. A clone while at least KEEP_VOICE of the speaker's speech in its spans when it was built is still theirs
        (a speakers re-run can give it to someone else): rebuilt from those spans (4-8 s on the GPU), with its pace
        re-fitted from its stored calibration pairs. If only the actual calibration text changed, repeat calibration
        on this restored reference. False when the voice must be built again."""
        v = self.voices.setdefault(sid, VoiceState())
        if entry.get("how") == "preset":
            how, _, clips = self._voice_spans(sid)
            if how != "stitched" or sum(b - a for a, b in clips) >= REF_MIN:
                return False  # enough clean speech for a clone now
            await self._preset(_preset_name(sid), VoiceCost())
            v.voice, v.kind, v.ref_seconds, v.status, v.key = None, VoiceKind.PRESET, entry["refSeconds"], "preset", None
            return True
        best = tuple(entry["timbre"]) if entry.get("timbre") else None
        clips = [tuple(c) for c in entry["clips"]]
        own = self._own_speech(sid, ([best] if best else []) + clips)
        if own < KEEP_VOICE * entry["speech"]:
            log.info("%s: %.0f %% of the speech %s's voice was built from is still theirs; built again", sid,
                     100 * own / entry["speech"], sid)
            return False
        voice, secs, how, reference = await self._voice_from(entry["how"], best, clips, VoiceCost(), self._gpu_priority())
        if voice is None or reference != entry["hash"]:
            return False  # the audio under its spans isn't what it was built from
        key = self._voice_key(sid, voice, f"{VOICE_VERSION}:{reference}")
        if recalibrate:
            cost, kept, t0 = VoiceCost(), [], time.perf_counter()
            used = await self._calibrate(sid, key, voice, cost, self._gpu_priority(), keep=kept)
            elapsed = time.perf_counter() - t0
            self.calibrate_s += elapsed
            entry["calibration"] = [[spoken, seconds] for spoken, seconds, _ in kept]
            entry["pace"], entry["overhead"] = round(self.estimator.rate(key), 3), round(self.estimator.overhead(key), 3)
            sample, audio = self.render_dir / "voices" / f"{sid}.npy", None
            if kept and hasattr(self.b.tts, "vocode"):
                audio, _ = await self._on_gpu(self.b.tts.vocode, kept[0][2], 1.0, cost=cost)
            elif not hasattr(self.b.tts, "synthesize_mel"):
                audio, _ = await self._on_gpu(self.b.tts.synthesize, self._tts_text(CALIBRATION_TE[0]), voice, "te",
                                             cost=cost)
            if audio is not None:
                sample.parent.mkdir(exist_ok=True)
                _save_pcm(sample, np.asarray(audio, np.float32))
            else:
                sample.unlink(missing_ok=True)
            entry["sample"] = sample.is_file()
            self._trace({"event": "recalibrate", "speaker": sid, "calibrate_s": round(elapsed, 3),
                         "calibration_takes": used, "pace": entry["pace"], "overhead_s": entry["overhead"]})
        pairs = [(spoken, seconds) for spoken, seconds in entry["calibration"]]
        if pairs and not recalibrate:
            self.estimator.calibrate(key, pairs)
        v.voice, v.kind, v.ref_seconds, v.status, v.key = voice, VoiceKind.CLONED, secs, "cloned", key
        entry["sample"] = entry["sample"] and (self.render_dir / "voices" / f"{sid}.npy").is_file()
        log.info("%s: voice restored from %.1f s of reference (%s), %.1f aksharas/s", sid, secs, how,
                 self.estimator.rate(key))
        return True

    def _own_speech(self, sid: str, spans: list[tuple[float, float]]) -> float:
        """Seconds of `sid`'s exclusive speech inside `spans` (overlaps counted once)."""
        return sum(t.end - t.start for a, b in _union(spans) for t in self.registry.turns_in(a, b) if t.speaker == sid)

    # ---- brief (§2.7) ---------------------------------------------------------------------------------------------
    async def _brief(self) -> None:
        """The video brief from the whole transcript, complete before any scene is translated: v1 from the first part,
        then each later part's diff on the brief so far. Each reply is cached by its message (briefs.jsonl), so a resumed
        render asks only for the parts it has no reply to. A Claude hold is waited out, with the job `waiting`; after
        BRIEF_TRIES unusable replies to a part, the render goes on with the brief made so far (v0, the metadata alone,
        when that is part 1), and job.json says so. The brief is put in use once, here."""
        await self._begin("brief")
        v = self.video
        meta = VideoMeta(v.title, v.channel, v.description, v.chapters, v.tags,
                         talk_shares=tuple(sorted(self.registry.talk_share().items())))
        self.tr = self.b.translator(self.cache_dir, self.video_id, brief_v0(meta), style=self.style, trace=self._trace)
        if hasattr(self.tr, "call_priority"):
            self.tr.call_priority = self._translation_priority
        parts = self._brief_parts()
        brief, made, gave_up = self.tr.brief, 0, None
        for k, part in enumerate(parts):
            self._progress("brief", k, len(parts))
            tries, again = 0, False
            while gave_up is None:
                await self._before_call()
                asked = time.monotonic()
                try:
                    brief = await self.tr.make_brief(meta, part, brief if k else None)
                except ClaudeCLIError as err:
                    again = True
                    if err.kind not in LADDER:
                        await self._claude_failed(err)  # held: asked again once the hold is over
                        continue
                    tries += 1
                    log.warning("brief part %d of %d: no usable reply (%s: %s), try %d of %d", k + 1, len(parts),
                                err.kind, err, tries, BRIEF_TRIES)
                    if tries >= BRIEF_TRIES:
                        gave_up = {"part": k + 1, "why": f"{err.kind}: {err}"}
                    continue
                made += 1
                if again:  # a failed call stores no reply, so this one went through Claude: the banner and back-off clear
                    await self._claude_ok(asked)
                break
            if gave_up is not None:
                log.warning("the brief stops at v%d: part %d of %d had no usable reply", brief.version, k + 1, len(parts))
                break
        self.tr.use_brief(brief)
        self.doc["brief"] = {"version": brief.version, "parts": len(parts), "made": made, "gaveUp": gave_up}
        st = self.doc["stages"]["brief"]
        st["done"], st["total"] = made, len(parts)

    def _brief_parts(self) -> list[list[tuple[str, str]]]:
        """The transcript as the brief reads it, (speaker, English) per unit, in parts of whole transcript rows (a unit
        goes with the row it starts in) of at most BRIEF_PART_WORDS words, so each part's message depends only on the
        stored rows and the diarization, and a resumed render finds its reply in briefs.jsonl."""
        starts = [row["a"] for row in self._rows]
        rows: list[list[UnitState]] = [[] for _ in starts]
        for st in self._order:
            rows[max(bisect_right(starts, st.unit.start) - 1, 0)].append(st)
        parts: list[list[tuple[str, str]]] = []
        part: list[tuple[str, str]] = []
        words = 0
        for units in rows:
            n = sum(len(st.unit.text.split()) for st in units)
            if part and words + n > BRIEF_PART_WORDS:
                parts.append(part)
                part, words = [], 0
            part += [(st.unit.speaker, st.unit.text) for st in units]
            words += n
        return parts + [part] if part else parts

    # ---- translate (§2.8) -----------------------------------------------------------------------------------------
    def _translation_priority(self, call: str, scene: int | None) -> float:
        """Queued Claude work closest to the dub's cursor goes first. Completing the next scene's review or fit
        feeds the GPU sooner than starting distant scenes. Earlier scenes' fix-ups keep the final plan moving too.
        The translator re-evaluates this at each free slot; running calls keep their slot and the three lanes retain
        their existing context boundaries. Unknown/no-scene work keeps a neutral priority."""
        if scene is None or not self.scenes:
            return 0.0
        front = next((sc.no for sc in self.scenes if any(not st.voiced and self._voiced_here(st) for st in sc.lines)),
                     self.scenes[-1].no + 1)
        phase = {"review": 0, "retranslate": 1, "fit": 2, "rephrase": 3, "scene": 4}.get(call, 5)
        return 10.0 * max(scene - front, 0) + phase

    async def _translate(self) -> None:
        """Every scene of the range, once the brief and every voice are ready, so each line is sized at its voice's
        calibrated pace. The whole video's units are cut into scenes once. The scenes of the range the line cache serves
        whole (after a preview, or a resume) are served first, in order, with no call; the rest are split into LANES
        contiguous blocks, each translated in order by a lane of its own (`_lane`). Lines skipped before stay skipped. A
        pause lets the calls in flight, fits included, finish into the line cache. The coverage classes of the wordings
        chosen go to job.json."""
        await self._begin("translate")
        # A run after a pause starts again from the transcript's units: what an earlier run of this job learned from
        # its lines would count them twice.
        self.skipped.clear()
        self._ratios.clear()
        self._learned.clear()
        self.scenes = self._cut_scenes()
        self._scene_of = {st.unit.id: sc for sc in self.scenes for st in sc.lines}
        self._scenes_cut.set()
        todo = self._range()
        total = sum(len(sc.lines) for sc in todo)
        self._skipped_before([st for sc in todo for st in sc.lines])
        self._progress("translate", 0, total)
        rest: list[Scene] = []
        try:
            for sc in todo:
                if self._served(sc):
                    await self._lane([sc], total, whole=True)  # no call: its Telugu is the next scene's context
                else:
                    rest.append(sc)
            n = len(rest)
            await self._all(self._lane(lane, total) for lane in self._translation_lanes(rest))
        except asyncio.CancelledError:
            for sc in todo:
                if sc.fit is not None:
                    sc.fit.cancel()
            raise
        finally:
            fits = [sc.fit for sc in todo if not sc.fitted]
            if fits:
                await asyncio.wait(fits)
        self.doc["coverage"] = self._coverage([st for sc in todo for st in sc.lines])
        log.info("translated %d scenes (%d from the line cache alone), %d lines: %s", len(todo), len(todo) - n, total,
                 self.doc["coverage"])
        st = self.doc["stages"]["translate"]
        st["done"] = st["total"] = total

    def _range(self) -> list[Scene]:
        """The scenes of the range: every one, or for a preview those that start before its stop point plus PAST_STOP."""
        stop = self.settings.stop_at
        return [sc for sc in self.scenes if stop is None or sc.lines[0].unit.start < stop + PAST_STOP]

    def _voiced_here(self, st: UnitState) -> bool:
        """Whether this run voices the line (§3): every line, or for a preview those that start before its stop point
        plus PAST_STOP, so the last lines it finalizes have their whole lookahead window."""
        stop = self.settings.stop_at
        return stop is None or st.unit.start < stop + PAST_STOP

    def _finalized_here(self, st: UnitState) -> bool:
        """Whether this run makes the line final (§3): every line, or for a preview those that start before its stop
        point."""
        stop = self.settings.stop_at
        return stop is None or st.unit.start < stop

    def _coverage(self, lines: list[UnitState]) -> dict:
        """job.json's coverage report of `lines` (§2.8): the classes of the wordings chosen (once voiced, of the wordings
        voiced), those classed on another tier, the unreviewed and the skipped."""
        got = self._classes([st for st in lines if st.line is not None and st.tier is not None])
        return {**got["coverage"], "otherTier": got["other_tier"], "unreviewed": got["unreviewed"],
                "skipped": sum(1 for st in lines if st.unit.id in self.skipped), "lines": len(lines)}

    def _cut_scenes(self, *, max_seconds: float | None = None, max_units: int | None = None) -> list[Scene]:
        """The whole video's scenes (§2.8), cut once from the first line to the last with `scene_cut`'s full scenes (up to
        30 s and 6 lines, at a speaker turn or a pause): the same list for the same transcript and diarization.
        Explicit caps are an internal benchmark hook; preview range and cache contents never change the boundaries."""
        scenes: list[Scene] = []
        k, limit = 0, SCENE_MAX_UNITS if max_units is None else max_units
        while k < len(self._order):
            # scene_cut reads at most one line past its limit of lines: this slice cuts as the whole rest would
            n = scene_cut([st.unit for st in self._order[k:k + limit + 1]],
                          max_seconds=max_seconds, max_units=max_units)
            scenes.append(Scene(len(scenes) + 1, self._order[k:k + n], k))
            k += n
        return scenes

    def _translation_lanes(self, scenes: list[Scene], *, weights: tuple[float, ...] | None = None
                           ) -> list[list[Scene]]:
        """At most three contiguous lanes. Weighting their sizes changes only their two context boundaries; each
        lane still awaits a scene's meaning review before requesting the next scene with its preceding Telugu.
        Explicit weights are an internal benchmark hook, with balanced production defaults until compared."""
        return [scenes[a:b] for a, b in _lane_bounds(len(scenes), LANE_WEIGHTS if weights is None else weights)]

    async def _lane(self, scenes: list[Scene], total: int, whole: bool = False) -> None:
        """One lane: its scenes in order, one at a time, each with `Dubber._scene` (the scene call, its review and one
        re-translation), asked for once the scene before has its Telugu, which is its context. A scene released after a
        Claude failure, or whose review failed with one, is asked for again once the hold is over, its lines reset
        (meanwhile a scene whose review failed is `held`: translated, unreviewed, until its lines are reset). Lines the
        translator skipped stay skipped. A pause stops the lane before its next call. `whole`: the scenes are served
        whole from the line cache (`_served`)."""
        for sc in scenes:
            while not sc.reviewed:
                if not sc.fitted:  # a late fit from the attempt before would fold into the lines asked for again
                    await asyncio.wait([sc.fit])
                await self._before_call()  # after that wait, so a pause or a hold that came during it is seen
                sc.fit, sc.failed, sc.released, sc.held = None, None, False, False  # its lines lose their wording
                lines = self._reset(sc)
                if lines:
                    token = _SCENE.set(sc)
                    try:
                        await self._scene(self._scene_request(sc, lines, whole))
                    finally:
                        _SCENE.reset(token)
                if sc.released:
                    continue
                if sc.failed is not None:
                    sc.held = True
                    self._wake.set()  # voiceable while the hold lasts
                    continue
                sc.reviewed, sc.held = True, False
                self._wake.set()
            self._progress("translate", sum(len(s.lines) for s in self.scenes if s.reviewed), total)

    def _reset(self, sc: Scene) -> list[UnitState]:
        """The scene's lines to ask for, with what an earlier attempt left on them cleared. A line being voiced or voiced
        keeps its wording, and a skipped line is done."""
        lines = [st for st in sc.lines if not (st.voicing or st.voiced)]
        for st in lines:
            st.line = st.tier = st.telugu = None
            st.scene = sc.no
        return lines

    def _served(self, sc: Scene) -> bool:
        """Whether the line cache holds every line of `sc` still to ask for, reviewed: served whole (`_scene_request`),
        the scene makes no scene call and no review."""
        for st in sc.lines:
            if not (st.voicing or st.voiced):
                line = self.tr.cached(self._line_spec(st, ("full",)))
                if line is None or line.coverage is None:
                    return False
        return True

    def _skipped_before(self, lines: list[UnitState]) -> None:
        """Lines the translator skipped in an earlier run (render/skipped.jsonl, by line key, under this prompt and
        model) are skipped again with no call (§2.8), unless the line cache now serves them (a line said the same way
        and answered elsewhere in the video)."""
        rows = _read_rows(self.render_dir / "skipped.jsonl")
        why = {r["key"]: r.get("why", "") for r in rows
               if "key" in r and r.get("prompt_hash") == self.tr.prompt_hash and r.get("model") == self.tr.model}
        if not why:
            return
        for st in lines:
            spec = self._line_spec(st, ("full",))
            if not st.voiced and (w := why.get(line_key(spec, self.style))) is not None and self.tr.cached(spec) is None:
                st.voiced = True
                self.skipped[st.unit.id] = w
        log.info("%d lines skipped in an earlier run stay skipped", len(self.skipped))

    def _scene_request(self, sc: Scene, lines: list[UnitState], whole: bool = False) -> SceneRequest:
        """The scene call for `lines` of `sc`, with its context: up to CONTEXT_BEFORE lines before it (from within
        CONTEXT_SPAN s) with the Telugu chosen for them, English only where they have none, and the English of up to
        CONTEXT_AFTER lines after it. `whole`: every line is served from the line cache, and
        asks for no tier up front that its cached line lacks. Such a tier is a guess from the predicted `full` (from k,
        which depends on the order lines were learned in), while the cached `full`'s length is known: the fit after the
        band rule's pick (`_fit_spec`) asks for what the line needs, and a resume asks for nothing its first run
        didn't."""
        specs = [self._spec(s) for s in lines]
        if whole:
            specs = [replace(p, want=tuple(t for t in p.want if t in line.tiers)) if (line := self.tr.cached(p)) else p
                     for p in specs]
        before, after = self._context(sc)
        return SceneRequest(sc.no, tuple(specs), context_before=before, context_after_en=after)

    def _context(self, sc: Scene) -> tuple[tuple[tuple[str, str | None], ...], tuple[str, ...]]:
        """A scene's context: up to CONTEXT_BEFORE lines before it (from within CONTEXT_SPAN s), each with the Telugu
        chosen for it (None where it has none), and the English of up to CONTEXT_AFTER lines after it."""
        k, n = sc.k, len(sc.lines)
        a, b = sc.lines[0].unit.start, sc.lines[-1].unit.end
        before = [s for s in self._order[max(k - CONTEXT_BEFORE, 0):k] if s.unit.end >= a - CONTEXT_SPAN]
        after = [s for s in self._order[k + n:k + n + CONTEXT_AFTER] if s.unit.start <= b + CONTEXT_SPAN]
        return (tuple((s.unit.text, s.line.tiers[s.tier].spoken if s.line and s.tier else None) for s in before),
                tuple(s.unit.text for s in after))

    async def _before_call(self) -> None:
        """Where a Claude call may start: never after a pause (calls in flight finish, no new one starts), and not before
        a hold (a back-off, a usage limit) is over. The hold is waited out with the job `waiting`, while GPU work goes on."""
        if self._stop.is_set():
            raise _Paused
        if time.monotonic() >= self._claude_hold:
            return
        self._waiting += 1
        back = False
        try:
            if self.doc["status"] == "running":
                self.doc["status"] = "waiting"
                await self._save(force=True)
            while (left := self._claude_hold - time.monotonic()) > 0:
                if self._stop.is_set():
                    raise _Paused
                await asyncio.sleep(min(HOLD_POLL, left))
        finally:
            self._waiting -= 1
            if not self._waiting and self.doc["status"] == "waiting":
                self.doc["status"], back = "running", True
        if back:
            await self._save(force=True)

    # ---- the dub loop (§2.9, §2.10) -------------------------------------------------------------------------------
    async def _voice_lines(self) -> None:
        """The dub loop: one task that owns the GPU and voices the range's lines in onset order, behind the translation.
        A line is voiced once its scene is ready (reviewed, or its review failed while Claude is held, and its fit
        back): the wording reviewed unless it has left the band, TAKES_N takes in one batched decode, placed on the
        working planner W (`self.planner`, on which every line voiced is placed), a shorter tier it has if it runs long
        (within FIXUP_SHARE), and its chosen take kept (`_keep`). A line whose take row still matches is restored from
        it, with no synthesis. Once a scene's last line is voiced its fix-ups run beside the loop (`_settle`), and the
        loop voices the takes they ask for ahead of new lines. The PCM of the lines the final plan has placed (§2.12,
        `_final_pcm`) comes next, then the next line to voice. A preview voices the lines that start before its stop point
        plus PAST_STOP (§3). Done once every line of the range is voiced (or skipped), every scene settled, and every
        line the final plan places has its PCM. Its first GPU call waits for the background sound (§2.14)."""
        await self._begin("voice_lines")
        self.planner = TimelinePlanner(self.planner.s)  # W from the first line: a run after a pause places them again
        self._resynths = self._retakes = 0
        self._long.clear()
        self._restored.clear()
        self._said.clear()
        self._voiced_now.clear()
        self.flags.clear()
        self._dubbing = True
        try:
            while not self._scenes_cut.is_set():  # the translate stage cuts this run's scenes
                if self._stop.is_set():
                    raise _Paused
                await self._idle()
            await self._wait_for_bed()  # the GPU runs the separator first (§2.14)
            todo = self._range()
            lines = [st for sc in todo for st in sc.lines if self._voiced_here(st)]
            self._budget = max(3.0, FIXUP_SHARE * len(lines))
            speech = np.cumsum([0.0] + [st.speech_s for st in lines]).tolist()
            self._progress("voice_lines", 0.0, round(speech[-1], 1))
            settles: list[asyncio.Task] = []
            k = restored = 0
            try:
                while True:
                    if self._stop.is_set():
                        raise _Paused
                    for t in settles:
                        if t.done() and not t.cancelled() and (err := t.exception()) is not None:
                            raise err  # a scene's fix-ups couldn't write to disk (`_settle` keeps every other failure)
                    self._pending_takes.clear()  # what the last line or fix-up used is in its row now, or garbage
                    if self._fixes:
                        await self._voice_fix(self._fixes.pop(0))
                        continue
                    if self._pcm_todo:
                        await self._final_pcm(*self._pcm_todo.pop(0))
                        continue
                    while k < len(lines) and lines[k].voiced:
                        k += 1
                    while len(settles) < len(todo) and all(st.voiced for st in todo[len(settles)].lines
                                                           if self._voiced_here(st)):
                        task = self._side(self._settle(todo[len(settles)]))
                        task.add_done_callback(lambda _: self._wake.set())
                        settles.append(task)
                    if k == len(lines):
                        if len(settles) == len(todo) and all(t.done() for t in settles) and not self._fixes \
                                and self._placed_all and not self._pcm_todo:
                            break
                        await self._idle()
                        continue
                    st = lines[k]
                    if not self._scene_of[st.unit.id].ready:
                        await self._idle()
                        continue
                    try:
                        restored += await self._voice_line(st)
                    except (_Paused, Cancelled, RenderError, OmniVoiceError):
                        # Worker failures must remain retryable; a pause or an unusable take must not skip speech.
                        raise
                    except Exception:
                        log.exception("unit %d failed", st.unit.id)
                        await self._skip(st, "synthesis failed")
                    self._wake.set()  # the final plan may place more
                    self._progress("voice_lines", round(speech[k + 1], 1), round(speech[-1], 1))
            except BaseException as e:
                self._dubbing = False
                for fix in self._fixes:
                    fix.done.set_result(False)
                self._fixes.clear()
                self._pcm_todo.clear()
                pending = [t for t in settles if not t.done()]
                if pending:
                    if not isinstance(e, _Paused):  # failing, or the engine stopping: now
                        for t in pending:
                            t.cancel()
                    await asyncio.wait(pending)  # paused: their Claude calls in flight finish into fixups.jsonl
                raise
        finally:
            self._dubbing = False
        self.doc["coverage"] = self._coverage(lines)  # of the wordings voiced
        voiced = sum(1 for st in lines if st.take is not None)
        _STAGE.get().extra = {"lines": len(lines), "voiced": voiced, "restored": restored, "fixups": self._resynths,
                              "retakes": self._retakes, "flagged": len(self.flags)}
        log.info("voiced %d lines (%d restored from their takes), %d fix-up takes, %d lines flagged: %s", voiced,
                 restored, self._resynths, len(self.flags), self.doc["coverage"])
        st = self.doc["stages"]["voice_lines"]
        st["done"] = st["total"] = round(speech[-1], 1)
        await self._let_go_of_tts()

    async def _idle(self, key: str = "dub") -> None:
        """Wait for news for the dub loop, or with `key` "finish" the final plan (a scene ready, a fit back, a fix-up
        asked for, a scene settled, a line placed or voiced), at most HOLD_POLL s, so a pause is seen too."""
        ev = self._wake.events.setdefault(key, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout=HOLD_POLL)
        except asyncio.TimeoutError:
            pass
        ev.clear()

    async def _voice_line(self, st: UnitState) -> bool:
        """Voice one line (§2.9), or restore it from its take row. Returns whether it was restored."""
        st.voicing = True  # a fit or a lane asking for its scene again no longer touches its wording
        got = await self._restorable(st)
        if got is not None:
            self._restore(st, *got)
            return True
        u, cost = st.unit, VoiceCost()
        seconds = target_seconds(self._slot(st), self.planner.s)
        voice, kind, key = await self._voice_for(u.speaker, cost)
        self._wording(st, key)
        n = _takes_n(st)
        words = self._pieces(st, key) or [st.line.tiers[st.tier]]
        mark = len(self._new)
        said = await self._say(st, words, voice, key, seconds, cost, n)
        self._learn(said, key)
        plan = self._plan(st, said)
        wording, first, fixups = (f"piece:{len(words)}" if len(words) > 1 else st.tier), st.tier, 0
        # Too long: the most complete shorter wording the line has that fits, else its shortest (no Claude call).
        if plan.needs_shorter and self._budget_left() and (alt := self._shorter_tier(st, key, seconds)) is not None:
            alt_mark = len(self._new)
            try:
                alt_said = await self._say(st, [st.line.tiers[alt]], voice, key, seconds, cost, n)
            except UnusableSpeechError:
                # The original already has usable speech. The failed alternative wrote no take, but spent a
                # synthesis attempt (including its bounded waveform retry); charge it so failures exhaust the budget.
                fixups = 1
                log.warning("unit %d: the %s take had no usable speech; keeping %s", u.id, alt, first)
            else:
                if 0 < alt_said.total < said.total and alt_said.failed is None:
                    st.tier, said, wording = alt, alt_said, alt
                    self._learn(alt_said, key)
                    plan = self._plan(st, said)  # replaces the plan just placed for this line
                else:
                    for m in self._new[alt_mark:]:
                        m[2] = None  # made, but not learned from: a failed take is shorter for the wrong reason
                fixups = int(len(self._new) > alt_mark)  # a fix-up synthesis: a take no line of the render had made
            self._resynths += fixups
        self._keep(st, said, plan, wording, key, n, cost, fixups=fixups, made=self._new[mark:])
        if plan.needs_shorter:
            self._long.add(u.id)  # still too long: its scene's rephrase (§2.10), within the budget; else flagged
        log.info("unit %d %s %.1f-%.1f s: %s%s, %.1f s of audio (%s), %d takes, lag %+.2f s%s", u.id, u.speaker,
                 u.start, u.end, wording, f" (after {first})" if st.tier != first else "", said.total, kind.value,
                 cost.takes, plan.lag, ", still long" if u.id in self._long else "")
        return False

    def _wording(self, st: UnitState, key: VoiceKey) -> None:
        """The wording to voice (§2.9): the tier the review classed (the one chosen, while unreviewed), unless at the
        voice's pace now it has left the band and another tier fits (`_pick`, priced on W)."""
        c = st.line.coverage
        if approved_full(st.line):
            st.tier, st.telugu = "full", self._tts_text(st.line.full)
            return
        tier = c.tier if c is not None and c.tier in st.line.tiers else st.tier
        pred = self.estimator.estimate(st.line.tiers[tier].spoken, key)
        if not BAND[0] * st.speech_s <= pred <= BAND[1] * st.speech_s:
            pick, fits = self._pick(st, st.line, key)
            if fits:
                tier = pick
        st.tier, st.telugu = tier, self._tts_text(st.line.tiers[tier])

    def _keep(self, st: UnitState, said: Said, plan: Plan, wording: str, key: VoiceKey, n: int, cost: VoiceCost,
              **extra: object) -> None:
        """Keep a line's chosen take (§2.9): its row appended to takes.jsonl (each of its take files is on disk already:
        `_said_take` writes one as its take is made), and the line's state set from it. Its natural audio stays in
        memory (`_said`) until the line's PCM is written. `extra`: the fix-up syntheses spent on the line so far (which a
        resume counts in the budget again), the takes this voicing made (`made`: what the estimator learned from them,
        which a resume replays), and a fix-up's wording and fixups.jsonl row."""
        voice = self._voice_row(key)
        texts = [self._tts_text(w) for w in said.wordings]
        c = st.line.coverage
        self._put_row(st, {
            "line": self._line_key(st), "at": round(st.unit.start, 3), "wording": wording, **extra,
            # the class of the wording voiced, where it has one: the line cache's is the last occurrence's to be classed
            "coverage": coverage_json(c) if voiced_class(c, st.tier) in CLASSES else None,
            "tts": _tts_hash(texts), "voice": voice, "synthesis": self._take_inputs(n),
            "durationPrompt": PROMPT_HASH,
            "approvalPolicy": self._approval_policy(),
            "takes": [{"key": self._take_key(text, voice, n), "spoken": w.spoken, "seconds": secs,
                       "pauses": [list(p) for p in pauses], "pace": pace, "failed": failed}
                      for text, w, secs, pauses, pace, failed in zip(texts, said.wordings, said.seconds, said.pauses,
                                                                     said.paces, said.failures)],
            "unit": st.unit.id, "cost": cost.fields()})
        st.telugu = self._tts_text(st.line.tiers[st.tier])
        st.plan, st.take_s, st.voiced = plan, said.total, True
        self._said[st.unit.id] = said
        self._voiced_now.add(st.unit.id)

    def _put_row(self, st: UnitState, row: dict) -> None:
        """A line's take row, appended to takes.jsonl and now the line's."""
        with _disk("the list of takes"):
            _append_row(self.render_dir / "takes.jsonl", row)
        self._take_rows.setdefault(row["line"], []).append(row)
        st.take = row

    async def _restorable(self, st: UnitState) -> tuple[dict, LineResult, str, list[Wording]] | None:
        """The take row a line is restored from (§2.9): the latest of its occurrence's rows (line key and onset) for its
        speaker's voice now whose wording still says what it said then and whose take files all read whole; None when
        there is none."""
        for got in self._rows_for(st):
            keys = [t["key"] for t in got[0]["takes"]]
            if await asyncio.to_thread(lambda: all(self._claim_take(k) is not None for k in keys)):
                return got
        return None

    def _rows_for(self, st: UnitState) -> Iterator[tuple[dict, LineResult, str, list[Wording]]]:
        """The line's take rows that still match it, latest first, each with the line, tier and wordings it says: rows
        of its line key (never of another line: unit ids are display-only) and its onset (the same English said again by
        the speaker is another occurrence, with a slot of its own), for its speaker's voice now, whose wording resolves
        to the same TTS text."""
        if st.line is None:
            return
        voice, at = self._voice_row(self._key(st.unit.speaker)), round(st.unit.start, 3)
        for row in reversed(self._take_rows.get(self._line_key(st), ())):
            n = _takes_n(st)
            if row["at"] != at or row["voice"] != voice or row.get("synthesis") != self._take_inputs(n):
                continue
            got = self._row_wording(st, row)
            if got is None:
                continue
            if approved_full(st.line):
                # A matching older short take or wording adjustment must not undo the current meaning review.
                # An identical full take (whole or in pieces) remains reusable with today's approved coverage.
                if row["wording"] in FIXUP_WORDINGS or got[1] != "full" or got[0].full != st.line.full:
                    continue
            texts = [self._tts_text(w) for w in got[2]]
            if (_tts_hash(texts) == row["tts"]
                    and [t["key"] for t in row["takes"]] == [self._take_key(text, voice, n) for text in texts]):
                yield (row, *got)

    def _row_wording(self, st: UnitState, row: dict) -> tuple[LineResult, str, list[Wording]] | None:
        """What a take row said, from the line's translation now: a tier of it, its `full` in pieces ("piece:<n>"), or
        a fix-up's wording, from its fixups.jsonl row (a rephrase, or a re-translation of the voiced-wording review)."""
        w = row["wording"]
        if w in FIXUP_WORDINGS:
            line, tier = self._fixup_line(st, self._fixup_rows.get(row.get("fixup"))), row.get("tier")
            return (line, tier, [line.tiers[tier]]) if line is not None and tier in line.tiers else None
        line = st.line
        if w.startswith("piece:"):
            words = self._piece_words(line) if len(line.pieces) >= 2 else []
            return (line, "full", words) if words and w == f"piece:{len(words)}" else None
        return (line, w, [line.tiers[w]]) if w in line.tiers else None

    def _restore(self, st: UnitState, row: dict, line: LineResult, tier: str, words: list[Wording]) -> None:
        """A line back from its take row, with no synthesis and no `_choose`: the wording and take it was voiced with,
        placed on W again in onset order (CPU only). Its natural audio isn't in memory: its PCM vocodes the take again
        from its file. Still too long, it is flagged, and (in a wording of its own) queued for its scene's rephrase
        again, whose answer fixups.jsonl has."""
        if (voiced_class(line.coverage, tier) not in CLASSES and row.get("durationPrompt") == PROMPT_HASH
                and row.get("approvalPolicy") == self._approval_policy()
                and isinstance(row.get("coverage"), dict)):
            line = replace(line, coverage=coverage_from(row["coverage"]))  # the line cache's is another occurrence's
        # Identical audio from an older text/review policy is reusable, but its semantic approval is not. Preserve
        # today's absent/other-tier class so the voiced-wording review checks the restored wording under this policy.
        said = _said_of(row, words)
        st.line, st.tier, st.telugu = line, tier, self._tts_text(line.tiers[tier])
        st.plan, st.take, st.take_s, st.voiced = self._plan(st, said), row, said.total, True
        self._resynths += row["fixups"]  # spent on it before: the budget is the range's, resumed or not
        self._restored.add(st.unit.id)
        if st.plan.needs_shorter:
            self._long.add(st.unit.id)

    async def _voice_fix(self, fix: _Fix) -> None:
        """Voice a fix-up take (§2.10) and keep it if it passes (a rephrase: and is shorter than the take it would
        replace), placed where that take was (`_shortened`: the lines after it keep their places on W), and flagged long
        while it still runs long. A rephrase that makes a take (one no line of the render had made), or exhausts its
        candidates without usable audio, counts in the budget. An attempted replacement not kept restates the line's
        row with what it cost and taught, so a resume spends and learns as this run did."""
        st, kept = fix.st, False
        try:
            # Different scenes can queue rephrases while a budget slot is still free. The dub loop owns spending,
            # so check again when the queued work actually executes. Meaning corrections keep their separate path.
            if fix.kind == "rephrase" and (approved_full(st.line) or not self._budget_left()):
                return
            cost = VoiceCost()
            seconds = target_seconds(self._slot(st), self.planner.s)
            voice, _, key = await self._voice_for(st.unit.speaker, cost)
            n = _takes_n(st)
            w = fix.line.tiers[fix.tier]
            mark = len(self._new)
            said = await self._say(st, [w], voice, key, seconds, cost, n)
            self._learn(said, key)
            made = self._new[mark:]
            charged = int(fix.kind == "rephrase" and bool(made))
            self._resynths += charged
            spent = st.take["fixups"] + charged
            if fix.kind == "rephrase":
                kept = 0 < said.total < st.take_s and said.failed is None
            else:  # the review's re-translation: the right words, whatever their length
                kept = said.total > 0 and said.failed is None
            if kept:
                st.line, st.tier = fix.line, fix.tier
                plan = self._shortened(st.plan, said.total, said.pauses[0], st.next_start)
                self._keep(st, said, plan, fix.kind, key, n, cost, fixups=spent, made=made, tier=fix.tier,
                           fixup=fix.key)
                (self._long.add if plan.needs_shorter else self._long.discard)(st.unit.id)
            elif made:
                self._put_row(st, {**st.take, "fixups": spent, "made": made, "cost": cost.fields()})
            log.info("unit %d: %s voiced, %s", st.unit.id, fix.kind, "kept" if kept else "not kept")
        except UnusableSpeechError:
            # Keep the current take, its timing/coverage and flags. There is no new take to learn from or cache,
            # but persist the failed attempt's cost and budget charge so a resume does not get free retries.
            charged = int(fix.kind == "rephrase")
            self._resynths += charged
            self._put_row(st, {**st.take, "fixups": st.take["fixups"] + charged,
                               "made": self._new[mark:], "cost": cost.fields()})
            log.warning("unit %d: the %s take had no usable speech; keeping its take", st.unit.id, fix.kind)
        except (_Paused, RenderError):
            raise
        except Exception:
            log.exception("unit %d: the %s take failed; the line keeps its take", st.unit.id, fix.kind)
        finally:
            if not fix.done.done():
                fix.done.set_result(kept)

    async def _fixup(self, st: UnitState, line: LineResult, tier: str, kind: str, key: str) -> bool:
        """Hand a fix-up take to the dub loop, which voices it ahead of new lines. Returns whether it was kept; False at
        once when the loop has stopped (a pause)."""
        if not self._dubbing or (kind == "rephrase" and approved_full(st.line)):
            return False
        fix = _Fix(st, line, tier, kind, key, asyncio.get_running_loop().create_future())
        self._fixes.append(fix)
        self._wake.set()
        return await fix.done

    async def _settle(self, sc: Scene) -> None:
        """A scene's fix-ups (§2.10), beside the dub loop once its last line is voiced: one batched rephrase of its lines
        still running long, each voiced by the loop in the most complete wording predicted to fit (`_rephrase_tier`)
        and kept if it passes and is shorter; then one review of the wordings voiced that have no class, where a P or
        E that a re-translation replaces is voiced once more. Their answers go to fixups.jsonl, never to lines.jsonl,
        and a resume takes them from there. While Claude is held (or once paused) no call starts: the scene settles as
        it is, with its lines flagged. A line voiced in a fix-up's wording (restored from its row) isn't rephrased again,
        nor is a line restored in a scene an earlier run settled (a `settled` row in fixups.jsonl): its rephrase was
        decided then, against that run's estimator and plan, and a resume doesn't decide it again."""
        before = self._settled_key(sc) in self._fixup_rows
        long = [st for st in sc.lines if st.unit.id in self._long and not approved_full(st.line)
                and st.take["wording"] not in FIXUP_WORDINGS
                and not (before and st.unit.id in self._restored)]
        kept, held, failed = 0, False, False
        try:
            if long and self._budget_left():
                answers, held = await self._rephrased(sc, long)
                for st in long:
                    if st.unit.id not in answers or not self._budget_left():
                        continue
                    line, key = answers[st.unit.id]
                    tier = self._rephrase_tier(line, self._key(st.unit.speaker), st.take_s, st.plan)
                    if tier is not None and await self._fixup(st, line, tier, "rephrase", key):
                        kept += 1
            held = await self._review_voiced(sc) or held
        except RenderError:
            raise
        except Exception:
            log.exception("the fix-ups of scene %d failed; its lines play as voiced", sc.no)
            failed = True
        if not (before or held or failed or self._stop.is_set()):  # fix-ups given up (a pause) are decided again
            row = {"key": self._settled_key(sc), "kind": "settled", "at": round(time.time(), 3)}
            with _disk("the fix-ups"):
                _append_row(self.render_dir / "fixups.jsonl", row)
            self._fixup_rows[row["key"]] = row
        self._flag(sc)
        sc.settled = True
        self._trace({"event": "fixups", "scene": sc.no, "long": len(long), "rephrased": kept, "held": held,
                     "flagged": sum(1 for st in sc.lines if st.unit.id in self.flags)})

    async def _rephrased(self, sc: Scene, lines: list[UnitState]) -> tuple[dict[int, tuple[LineResult, str]], bool]:
        """Rephrases of a scene's lines that run long (§2.10): from fixups.jsonl, else all in one `rephrase` request, with
        no deadline, each line with its overrun as its finding. Returns {unit id: (the rephrase, its fixups.jsonl
        key)} and whether Claude held back the call it needed."""
        got: dict[int, tuple[LineResult, str]] = {}
        ask: list[tuple[UnitState, str]] = []
        for st in lines:
            key = self._fixup_key("rephrase", self._line_key(st), st.line.tiers[st.tier].spoken)
            line = self._fixup_line(st, self._fixup_rows.get(key))
            if line is not None:
                got[st.unit.id] = (line, key)
            else:
                ask.append((st, key))
        if not ask:
            return got, False
        if self._stop.is_set() or self._held():
            return got, True
        specs = []
        for st, _ in ask:
            over = st.plan.excess * st.plan.rate  # seconds of speech past the line's window
            specs.append(self._line_spec(st, ("full", "concise", "very_concise"), current=st.line.tiers[st.tier].spoken,
                                         finding={"overrun_s": round(over, 2), "overrun_aksharas": round(
                                             over * self.estimator.rate(self._key(st.unit.speaker)))}))
        asked = time.monotonic()
        try:
            res = await self.tr.submit(SceneRequest(sc.no, tuple(specs), "rephrase"))
        except asyncio.CancelledError:
            if _cancelling():
                raise
            return got, True  # dropped by another call's Claude failure (`_claude_failed`)
        except ClaudeCLIError as err:
            await self._claude_failed(err)
            return got, True
        if res.calls:
            await self._claude_ok(asked)
        for st, key in ask:
            if (line := res.lines.get(st.unit.id)) is not None:
                self._put_fixup(st, key, "rephrase", line)
                got[st.unit.id] = (line, key)
        return got, False

    async def _review_voiced(self, sc: Scene) -> bool:
        """The voiced-wording review (§2.10): a scene's voiced lines whose wording has no class (unreviewed, a tier the
        review didn't class, a rephrase) go to one review call, each on the wording voiced; one a re-translation
        replaces (a P or E) is voiced once more and not reviewed again (`_reviewed`). Classes from fixups.jsonl need no
        call. A line's own wording also goes to the line cache with its class (never the re-translation, which only a
        take row that kept it and fixups.jsonl say), a rephrase's never. Returns whether Claude held back the call it
        needed."""
        ask: list[tuple[UnitState, str]] = []
        for st in sc.lines:
            if st.take is None or voiced_class(st.line.coverage, st.tier) in CLASSES:
                continue
            key = self._fixup_key("review", self._line_key(st), st.line.tiers[st.tier].spoken)
            line = self._fixup_line(st, self._fixup_rows.get(key))
            if line is not None and line.coverage is not None:
                await self._reviewed(st, line, key)
            else:
                ask.append((st, key))
        if not ask:
            return False
        if self._stop.is_set() or self._held():
            return True
        before, after = self._context(sc)
        req = SceneRequest(sc.no, tuple(self._line_spec(st, tuple(dict.fromkeys(("full", st.tier)))) for st, _ in ask),
                           context_before=before, context_after_en=after)
        own = {st.unit.id for st, _ in ask if st.take["wording"] not in FIXUP_WORDINGS}
        asked = time.monotonic()
        try:
            res = await self.tr.review(req, {st.unit.id: replace(st.line, coverage=None) for st, _ in ask},
                                       {st.unit.id: st.tier for st, _ in ask}, cache=own)
        except asyncio.CancelledError:
            if _cancelling():
                raise
            return True
        if res.error is not None:
            await self._claude_failed(res.error)
        elif res.calls:
            await self._claude_ok(asked)
        for st, key in ask:
            line = res.lines.get(st.unit.id)
            if line is not None and line.coverage is not None:
                if st.unit.id not in res.review_pending:
                    self._put_fixup(st, key, "review", line)
                await self._reviewed(st, line, key)
        return res.error is not None or bool(res.review_pending)

    async def _reviewed(self, st: UnitState, line: LineResult, key: str) -> None:
        """A voiced wording's class. When a reviewed re-translation replaced the wording (a P or E), it
        is voiced once more and, if its take passes, voiced in its place; else the wording voiced keeps the review's
        class."""
        c = line.coverage
        correction = c.by == "validators" or (c.by == "review" and c.first in ("P", "E"))
        if correction and c.tier in line.tiers:
            if await self._fixup(st, line, c.tier, "retranslate", key):
                return
            c = Coverage(c.first or c.cls, tier=st.tier)
        st.line = replace(st.line, coverage=c)

    def _flag(self, sc: Scene) -> None:
        """Why each voiced line of a settled scene plays as it is: still too long, or its wording unreviewed."""
        for st in sc.lines:
            if st.take is None:
                continue
            self._flag_line(st, st.unit.id in self._long)

    def _flag_line(self, st: UnitState, long: bool) -> None:
        why = (["long"] if long else []) + \
              (["unreviewed"] if voiced_class(st.line.coverage, st.tier) not in CLASSES else [])
        if why:
            self.flags[st.unit.id] = why
        else:
            self.flags.pop(st.unit.id, None)

    def _held(self) -> bool:
        return time.monotonic() < self._claude_hold

    # ---- the final plan, line PCM and the manifest (§2.11-§2.13) --------------------------------------------------
    async def _finish(self) -> None:
        """The final plan F (§2.11), trailing the dub loop: a planner of its own, with W's settings, that places the lines
        this run finalizes (§3) in onset order, from the first line on every run. Line i is placed once it and every line
        of its lookahead window (`_window`) has its take in a settled scene or is skipped (a line past what a preview
        voices counts as not there): so F prices only takes, and places each line as a plan made after the last take
        would. It asks for no fix-up. Lines are placed in batches in a worker thread; the dub loop writes each placed
        line's PCM (`_final_pcm`), and job.json's `finalUntil` follows the finished prefix. Done once every line is final
        (its PCM written, or skipped): then the manifest is written, once, and the take and PCM files nothing names any
        more are collected."""
        await self._begin("finish")
        final = self._final = TimelinePlanner(self.planner.s)
        self.final.clear()
        self.pcm.clear()
        self._events.clear()
        self._out.clear()
        self._pcm_todo.clear()
        self._pcm_keys.clear()
        self._placed_all = False
        self._final_lines = []
        while not self._scenes_cut.is_set():  # the translate stage cuts this run's scenes
            if self._stop.is_set():
                raise _Paused
            await self._idle("finish")
        lines = self._final_lines = [st for st in self._order if self._finalized_here(st)]
        k = 0                                     # F's cursor: the lines before it are placed (or skipped)
        while True:
            if self._stop.is_set():
                raise _Paused
            n = k
            while n < len(lines) and self._final_ready(lines[n]):
                n += 1
            if n > k:
                for st, plan in await asyncio.to_thread(self._place_final, final, lines[k:n]):
                    key = self._pcm_key(st, plan)
                    self.final[st.unit.id], self._pcm_keys[st.unit.id] = plan, key
                    self._pcm_todo.append((st, plan, key))
                k = n
                self._placed_all = k == len(lines)
                self._wake.set()  # the dub loop writes their PCM
            done = self._finished(lines)
            self.doc["finalUntil"] = self._final_until(lines, done)  # "Telugu speech final up to 42:10" (§2.13)
            self._progress("finish", done, len(lines))
            if done == len(lines):
                break
            if n == k:
                await self._idle("finish")
        doc = await asyncio.to_thread(self._manifest)
        with _disk("the manifest"):
            await asyncio.to_thread(_write_json, self.render_dir / "manifest.json", doc)
        self.doc["report"] = _report(doc)
        await self._collect()
        _STAGE.get().extra = {"lines": len(lines), "pcm": len(self.pcm), "skipped": len(lines) - len(self.pcm)}
        st = self.doc["stages"]["finish"]
        st["done"] = st["total"] = len(lines)
        self._final_done = True
        await self._let_go_of_tts()

    def _final_ready(self, st: UnitState) -> bool:
        """Whether F can place the line (§2.11): it is skipped (then it is final as it is), or it and every line of its
        lookahead window has its take in a settled scene, or is skipped, or is past what this run voices."""
        if st.unit.id in self.skipped:
            return True
        return self._settled(st) and all(self._settled(x) or x.unit.id in self.skipped or not self._voiced_here(x)
                                         for x in self._window(st))

    def _settled(self, st: UnitState) -> bool:
        sc = self._scene_of.get(st.unit.id)
        return st.take is not None and sc is not None and sc.settled

    def _place_final(self, planner: TimelinePlanner, lines: list[UnitState]) -> list[tuple[UnitState, Plan]]:
        """Place `lines` on F, in onset order, each from its take row (`_said_of`: no audio), with the takes of the lines
        of its window (`_ahead`). Pure CPU: run in a worker thread. A skipped line isn't placed."""
        return [(st, self._plan(st, _said_of(st.take), planner)) for st in lines if st.unit.id not in self.skipped]

    def _finished(self, lines: list[UnitState]) -> int:
        """How many of `lines`, from the first, are final: with their PCM, or skipped."""
        n = 0
        while n < len(lines) and (lines[n].unit.id in self.pcm or lines[n].unit.id in self.skipped):
            n += 1
        return n

    def _final_until(self, lines: list[UnitState], done: int) -> float:
        """Where the finished prefix ends (§2.13): the start of the first line not final, else the stop point or the end."""
        if done < len(lines):
            return round(lines[done].unit.start, 2)
        duration = float(self.doc["duration"])
        stop = self.settings.stop_at
        return round(duration if stop is None else min(stop, duration), 2)

    def _pcm_key(self, st: UnitState, plan: Plan) -> str:
        """A line's PCM key (§2.1, §2.12): its takes and what of its final plan shapes its audio (the rate, and each
        part's take, stretch, place and rate), rounded, with how takes are squeezed and MIX_VERSION."""
        r = functools.partial(round, ndigits=3)
        parts = [[p.take, r(p.a), r(p.b), r(p.start - plan.start), r(p.rate), r(p.wall)] for p in plan.parts]
        return _key20([[t["key"] for t in st.take["takes"]], [r(plan.rate), r(plan.wall), parts],
                       self.planner.s.keep_pause, self.b.tts.sample_rate, MIX_VERSION])

    async def _final_pcm(self, st: UnitState, plan: Plan, key: str) -> None:
        """A line F placed, made final (§2.12), in the dub loop: its audio as the final plan places it (`_play`: squeezed
        and cut, parts at their times; a rate other than 1 vocodes the take's mel again, watermarked as ever, ADR-014),
        saved as render/pcm/<pcm key>.npy (float16), unless that file is there already (a resume, a line said the same
        way at the same rate). Its takes come from memory while the natural audio of its voicing is still there, else
        from their files. The `unit` trace event is written with the PCM: only then is the line's plan final."""
        u = st.unit
        path = self.render_dir / "pcm" / f"{key}.npy"
        frames = await asyncio.to_thread(_pcm_frames, path)
        # W chose fix-ups with the lookahead available then; F now knows every take in the window. Display the final
        # plan's result so fresh and resumed renders of identical PCM cannot disagree about a line still running long.
        # Keep W and _long unchanged: this reports the finished audio and never schedules more synthesis.
        self._flag_line(st, plan.needs_shorter)
        event = self._unit_event(st, plan)
        cost, made = VoiceCost(), frames is None
        if made:
            said = self._said.get(u.id)
            if said is None:
                if not await asyncio.to_thread(lambda: all(self._claim_take(t["key"]) is not None
                                                           for t in st.take["takes"])):
                    raise RenderError(f"A take of line {u.id} couldn't be read back. Resume to voice it again.")
                said = await self._load_said(st.take)
            audio = np.asarray(await self._play(said, plan, cost), np.float32)
            with _disk("a line's audio"):
                await asyncio.to_thread(_save_pcm_file, path, audio)
            frames = len(audio)
        # Every line this run makes final, its PCM made now or found (a resume's run traces its whole prefix too).
        self._trace({**event, "pcm": key, "samples": frames, "pcm_made": made, "pcm_render_s": round(cost.render_s, 3)})
        self._said.pop(u.id, None)  # its natural audio isn't needed any more
        self._events[u.id] = event
        self._out[u.id] = self._manifest_line(st, plan, key, frames, event)
        self.pcm[u.id] = (key, frames)
        self._wake.set()  # the final plan counts it

    def _unit_event(self, st: UnitState, plan: Plan) -> dict:
        """A final line's `unit` trace event (`Dubber._trace`'s fields, from its final plan and its take row): the
        manifest's stats are computed from these."""
        u, row = st.unit, st.take
        w = st.line.tiers[st.tier]
        slot = self._slot(st)
        fill = {"speech_fill": round(plan.played / st.speech_s, 3) if st.speech_s else None,
                "required_rate": round(st.take_s / st.speech_s, 3) if st.speech_s else None}
        return {"event": "unit", "id": u.id, "speaker": u.speaker, "start": u.start, "end": u.end,
                "target_s": round(target_seconds(slot, self.planner.s), 2), "speech_s": round(st.speech_s, 2),
                "breaks": len(u.breaks), "anchors": len(u.anchors), "cut_off": u.cut_off, "music": st.music,
                "scene": st.scene, "source": u.text, "wording": row["wording"], "tier": st.tier,
                "tiers": sorted(st.line.tiers), "telugu": st.telugu,
                "latin_ratio": round(tenglish.latin_ratio(st.telugu), 3), "english_words": len(w.english),
                "full_aksharas": round(count_units(st.line.full.spoken), 1),
                "full_pred_aksharas": None if st.pred_full is None else round(st.pred_full, 1),
                "flags": list(st.line.flags), "lint": tenglish.lint(w.spoken, w.english, u.text, self._context_for(st)),
                "lint_version": tenglish.LINT_VERSION, "coverage": coverage_json(st.line.coverage),
                "coverage_voiced": voiced_class(st.line.coverage, st.tier), **fill,
                "audio_s": round(st.take_s, 2), "dub_start": round(plan.start, 3), "lag": round(plan.lag, 3),
                "audio_rate": round(plan.rate, 3), "freeze": round(plan.freeze, 3), **self._timing(slot, plan),
                "overdraft": round(plan.overdraft, 2), "needs_shorter": plan.needs_shorter,
                "render_flags": list(self.flags.get(u.id, ())), "voice": _voice_kind(row),
                "takes": [t["key"] for t in row["takes"]], "fixups": row["fixups"], **row["cost"],
                "translate_ready_at": st.translate_ready_at, "translate_s": st.translate_s, "model": st.model,
                "prompt_hash": st.prompt_hash, "cache_hit": st.cache_hit}

    def _manifest_line(self, st: UnitState, plan: Plan, key: str, frames: int, event: dict) -> dict:
        """A final line as the manifest gives it (§2.13): where the dub starts and how it is played, where the source
        said it, what is said (`telugu`: the wording voiced, in Telugu script also when the TTS read its Latin form,
        `ttsScript`; the Telugu subtitles show it, §2.16), and its PCM."""
        u = st.unit
        return {"id": u.id, "speaker": u.speaker, "start": round(plan.start, 3), "srcStart": round(u.start, 3),
                "srcEnd": round(u.end, 3), "audioRate": round(plan.rate, 4), "audioWall": round(plan.wall, 4),
                "lag": round(plan.lag, 3), "said": plan.said, "tier": st.tier, "voice": event["voice"], "source": u.text,
                "telugu": st.line.tiers[st.tier].spoken, "coverage": event["coverage_voiced"],
                "flags": event["render_flags"], "pcm": key, "samples": frames}

    def _manifest(self) -> dict:
        """render/manifest.json (§2.13), the export's input, once every line of the range is final: the lines and the
        skips, the speakers, and the stats of the lines (bench's `timing_metrics` and `dub_metrics`, and the overdraft
        the planner carried: §2.11)."""
        from .bench import dub_metrics, timing_metrics  # (bench's own imports stay out of the engine's start)

        out, skipped, events = [], [], []
        voiced = self.final_voiced()
        for st in self._final_lines:
            u = st.unit
            if u.id in self.pcm:
                out.append(self._out[u.id])
                events.append(self._events[u.id])
            else:
                skipped.append({"id": u.id, "start": round(u.start, 3), "end": round(u.end, 3),
                                "why": self.skipped.get(u.id, "")})
                events.append({"event": "skipped", "id": u.id, "start": u.start, "end": u.end,
                               "speech": [list(x) for x in self._speech_spans(u)]})
        timing, dub = timing_metrics(events), dub_metrics(voiced)
        s, reg = self.settings, self.registry
        speakers = []
        for sid, sp in sorted(reg.speakers.items(), key=lambda x: x[1].first_at):
            v = self.voices.get(sid, VoiceState())
            cloned = v.kind is VoiceKind.CLONED and v.voice is not None and not v.use_preset
            native = v.kind is VoiceKind.NATIVE and v.voice is not None
            speakers.append({"id": sid, "label": sp.label, "talkSeconds": round(sp.talk_seconds, 2),
                             "voice": "native" if native else "cloned" if cloned else "preset", "referenceSeconds": round(v.ref_seconds, 1),
                             "pace": round(self.estimator.rate(self._key(sid)), 2),
                             "sample": (cloned or native) and (self.render_dir / "voices" / f"{sid}.npy").is_file()})
        plans = [self.final[st.unit.id] for st in self._final_lines if st.unit.id in self.pcm]
        d = self.doc
        return {"version": 2, "videoId": self.video_id, "title": d["title"], "channel": d["channel"],
                "duration": d["duration"], "stopAt": s.stop_at, "complete": True,  # (written once every line is final)
                "sampleRate": self.b.tts.sample_rate, "settings": self._manifest_settings(), "speakers": speakers,
                "lines": out, "skipped": skipped,
                "stats": {"lines": len(out), "coverage": dub["coverage"], "lag_p95": timing["lag_p95"],
                          "rate_p90": timing["rate_p90"],  # (each line its own overdraft, as the planner's stats)
                          "overdraft_s": round(sum(p.own_overdraft for p in plans), 2),
                          "overdraft_carried_s": round(sum(p.carried for p in plans), 2), "fixups": self._resynths}}

    def final_voiced(self) -> list[SimpleNamespace]:
        """The final lines with their PCM, as bench's `dub_metrics` reads them (its `UnitState`'s fields)."""
        return [SimpleNamespace(voiced=True, plan=self.final[st.unit.id], take_s=st.take_s, speech_s=st.speech_s,
                                line=st.line, tier=st.tier) for st in self._final_lines if st.unit.id in self.pcm]

    async def _collect(self) -> None:
        """The garbage collection (§2.13), once the manifest is written: take files that neither the take row each line
        uses (the latest of its occurrence's) nor the dub loop names, and PCM files that no line F placed names, are
        deleted: what fix-ups, re-runs and `set_voice` left. (In a preview the dub loop may still be settling the scenes
        past its stop point: a take it claims meanwhile stays, `_claim_take`.)"""
        using = {(r["line"], r["at"]): r for st in self.units.values() if (r := st.take) is not None}
        rows = {(r["line"], r["at"]): r for got in self._take_rows.values() for r in got}
        rows.update(using)
        takes = {t["key"] for r in rows.values() for t in r["takes"]} | set(self._pending_takes)
        pcm = set(self._pcm_keys.values())
        self._touched = set()
        try:
            gone = await asyncio.to_thread(self._sweep, takes, pcm)
        finally:
            self._touched = None
        if gone:
            log.info("render %s: %d take and PCM files nothing names any more deleted", self.video_id, gone)

    def _sweep(self, takes: set[str], pcm: set[str]) -> int:
        n = 0
        for f in (self.render_dir / "takes").glob("*.npz"):
            with self._take_lock:  # a take the loop claims meanwhile stays (`_claim_take`)
                if f.stem in takes or f.stem in self._pending_takes or f.stem in (self._touched or ()):
                    continue
                with contextlib.suppress(OSError):
                    f.unlink()
                    n += 1
        for f in (self.render_dir / "pcm").glob("*.npy"):
            if f.stem not in pcm:
                with contextlib.suppress(OSError):
                    f.unlink()
                    n += 1
        return n

    def _manifest_settings(self) -> dict:
        """The settings a manifest was rendered with (§2.13)."""
        s = self.settings
        return {"style": s.style, "speedCap": s.speed_cap, "ttsScript": s.tts_script, "speakers": s.speakers or "auto",
                "presets": list(s.presets)}

    async def _let_go_of_tts(self) -> None:
        """The TTS model goes once the final plan's stage and the dub loop have both ended this run (§2.13): the export
        needs no GPU, and Chatterbox would otherwise stay resident through it. (In a preview the dub loop can outlive the
        final plan, voicing the lines past the stop point.)"""
        release = getattr(self.b.tts, "release", None)
        if release is not None and self._final_done and not self._dubbing:
            await self._on_gpu(release)

    # ---- the background sound (§2.14) ----------------------------------------------------------------------------
    def _bed_inputs(self) -> dict | None:
        """What the bed is made from (§2.1): the source audio, the separator and its pinned model, the rate, the chunk
        and its overlap, the block, the compute dtype and SEP_VERSION; not the speakers or the text (what depends on
        them is computed at the export from bed_vocals.npy). None when there is no bed: no separator on the backend, or
        no source audio file to separate."""
        sep = self.b.separator
        if sep is None or self.video is None or self.video.audio_path is None:
            return None
        return {"audio": self._fp, "separator": type(sep).__name__, "model": _model_rev(self.b.name, "separation"),
                "sampleRate": sep.sample_rate, "chunk": sep.chunk, "overlap": 0.5, "block": separate.SEP_BLOCK,
                "dtype": separate.SEP_DTYPE, "version": separate.SEP_VERSION}

    async def _separate(self) -> None:
        """The background sound (§2.14), on the GPU before the dub loop, beside the translation: the bed's blocks of the
        range (`_bed_end`) that render/bed/ lacks. Served from disk when they are all there under the inputs they would
        be made from now, and from the output while the whole video's MP4 they went into still matches (the bed was
        deleted then, §2.1: the export makes the blocks again only if it must mix again). With no separator there is no
        bed, and job.json's `bed` says so. The dub loop's GPU waits for this stage (`_separated`); the separator lets go
        of its model when it ends."""
        st, sep = self.doc["stages"]["separate"], self.b.separator
        try:
            inputs = self._bed_inputs()
            end = self._bed_end()
            st["total"] = round(end, 1)
            if inputs is None:
                note = NO_SEPARATOR if sep is None else "This video's audio has no file to separate: the dubbed " \
                                                         "video will have the Telugu voices only."
                self.doc["bed"] = {"separator": None, "until": None, "note": note}
                st["done"] = st["total"] = 0
                _STAGE.get().extra = {"bed": "none"}
            elif self._bed_in_output(inputs):
                log.info("render %s: the background sound went into the MP4 at %s, which still matches", self.video_id,
                         self.doc["output"]["path"])
                self.doc["bed"] = {"separator": inputs["separator"], "until": round(end, 1), "note": None}
                st["done"] = st["total"]
                _STAGE.get().extra = {"bed": "output"}
            else:
                made, found = await self._make_bed(inputs, end)
                self.doc["bed"] = {"separator": inputs["separator"], "until": round(end, 1), "note": None}
                st["done"] = st["total"]
                _STAGE.get().extra = {"blocks": made, "found": found}  # (the trace row's `cached` is the stage's own)
        finally:
            if sep is not None:
                sep.release()  # (quick: the model's arrays and MLX's cache; no separator call is running now)
        self._separated.set()

    def _tts_ready(self) -> None:
        missing = getattr(self.b.tts, "missing", None)
        if missing is not None and (absent := missing()):
            raise RenderError("OmniVoice setup is incomplete. Run scripts/setup-omnivoice.sh before dubbing. "
                              + "; ".join(absent))

    def _separator_ready(self) -> None:
        """Fail saying how to fetch them when the separator's model files aren't in the models folder (one filled
        before the separator was pinned lacks them): at the run's start, so a job that will separate fails in seconds,
        not after the GPU stages, and again before any block is made (the export's re-make)."""
        sep = self.b.separator
        if sep is not None and (missing := sep.missing()):
            log.warning("render %s: the separator's model lacks %s", self.video_id, ", ".join(missing))
            self.doc["stage"] = "separate"  # where the job failed (inside a stage, `_stage` says which)
            raise RenderError(f"The background-sound model isn't downloaded. Run `uv run maata-bench fetch --backend "
                              f"{self.b.name}` in Maata's engine folder, then resume.")

    def _bed_end(self) -> float:
        """Where the bed's range ends (§2.14): the video's end, or a preview's stop point plus twice PAST_STOP."""
        return _span("separate", float(self.doc["duration"] or 0.0), self.settings.stop_at)

    def _bed_in_output(self, inputs: dict) -> bool:
        """Whether the whole video's MP4 is still there, mixed over a bed made from `inputs` (§2.1)."""
        out = self._whole_output()
        return out is not None and (out.get("inputs") or {}).get("bed") == inputs

    def _whole_output(self) -> dict | None:
        """job.json's `output` when it is the whole video's MP4 and the file at its path still has its bytes."""
        out = self.doc.get("output")
        if not isinstance(out, dict) or out.get("kind") != "whole":
            return None
        path = Path(str(out.get("path") or ""))
        return out if path.is_file() and path.stat().st_size == out.get("bytes") else None

    async def _make_bed(self, inputs: dict, end: float, count: bool = True) -> tuple[int, int]:
        """Make the bed's blocks of [0, `end`) s that render/bed/ lacks under `inputs` (`_bed_doc`), in order. The source
        audio is decoded from its start, never sought (§2.14: a resumed block then equals a whole-file run's), into a
        sliding buffer that keeps only what the next block's chunks read; each block's chunks go to the separator
        SEP_BATCH at a time through `_on_gpu` (a pause stops between batches), and the block is kept whole
        (`_keep_block`). Progress: video seconds separated. `count`: the separate stage's start; without it (the export's
        re-make, whose start is the export's) the stage's row shows the work, and is left `todo` if it stops (the next
        run's separate stage makes what is missing). Returns the blocks made and those found."""
        sep = self.b.separator
        sr, chunk = sep.sample_rate, sep.chunk
        doc = await asyncio.to_thread(self._bed_doc, inputs)
        dur = float(self.doc["duration"] or 0.0)
        n_est = int(round(dur * sr))  # the decode's length, as the analysis audio says (the decode tells exactly)
        want = range(separate.block_count(min(int(math.ceil(end * sr)), n_est), sr))
        done = set(doc["blocks"]) & set(want)
        todo = [k for k in want if k not in done]
        total = round(min(end, dur), 1)

        def separated() -> float:
            return round(min(total, sum(min(separate.block_samples(sr), n_est - k * separate.block_samples(sr))
                                        for k in done) / sr), 1)

        self._progress("separate", separated(), total)
        if not todo:
            return 0, len(done)
        self._separator_ready()
        st = self.doc["stages"]["separate"]
        if count:
            await self._begin("separate")
        else:
            st["state"] = "running"
        frames = decode_stereo(self.video.audio_path, sr)
        buf = separate.Buffer()
        n: int | None = None  # the decode's length, once it has ended
        made = 0
        try:
            for k in todo:
                lo, hi = separate.block_span(k, n if n is not None else 1 << 62, sr=sr, chunk=chunk)
                buf.drop(lo)  # a resume discards what comes before its first missing block, counting it
                if n is None:
                    n = await asyncio.to_thread(_fill, frames, buf, hi)
                if n is not None and k * separate.block_samples(sr) >= n:
                    break  # the analysis audio's length promised a block the decode hasn't
                blk = separate.Block(k, n if n is not None else buf.end, buf.get(), buf.at, sr=sr, chunk=chunk)
                for i in range(len(blk.groups)):
                    batch = await asyncio.to_thread(blk.batch, i)
                    vocals, _ = await self._on_gpu(sep.vocals, batch)
                    await asyncio.to_thread(blk.add, i, vocals)
                bed, ms = await asyncio.to_thread(blk.result)
                with _disk("the background sound"):
                    await asyncio.to_thread(self._keep_block, doc, k, bed, ms)
                done.add(k)
                made += 1
                self._progress("separate", separated(), total)
        except BaseException:
            if not count:
                st["state"] = "todo"
            raise
        finally:
            with contextlib.suppress(ValueError):  # (still running in a worker thread when the engine stops: let it be)
                frames.close()
        if not count:
            st["state"] = "done"
        return made, len(done) - made

    def _bed_doc(self, inputs: dict) -> dict:
        """bed.json for `inputs`: as on disk, less the blocks whose file or vocals envelope doesn't read whole (a torn
        file is made again); or, when it was made from other inputs or isn't there, a new one, with every block and the
        envelope of the old one deleted."""
        path, folder = self.render_dir / "bed.json", self.render_dir / "bed"
        doc = _read_json(path)
        if doc is None or doc.get("inputs") != inputs or not isinstance(doc.get("blocks"), list):
            shutil.rmtree(folder, ignore_errors=True)
            with contextlib.suppress(OSError):
                (self.render_dir / "bed_vocals.npy").unlink()
            doc = {"inputs": inputs, "blocks": []}
            _write_json(path, doc)
            return doc
        frames = _read_npy(self.render_dir / "bed_vocals.npy")
        sr = int(inputs["sampleRate"])
        doc["blocks"] = sorted(k for k in doc["blocks"] if isinstance(k, int) and _block_ok(folder, k, frames, sr))
        return doc

    def _keep_block(self, doc: dict, k: int, bed: np.ndarray, ms: np.ndarray) -> None:
        """Keep block k: its bed (render/bed/<k>.npy, float16, written whole, then put in place), its vocals envelope
        into render/bed_vocals.npy (NaN where no block is made yet), then bed.json listing it."""
        folder = self.render_dir / "bed"
        folder.mkdir(parents=True, exist_ok=True)
        _save_npy(folder / f"{k:05d}.npy", bed.astype(np.float16))
        path = self.render_dir / "bed_vocals.npy"
        frames = _read_npy(path)
        at = k * round(separate.SEP_BLOCK / separate.FRAME)
        if frames is None or frames.ndim != 1:
            frames = np.zeros(0, np.float32)
        if len(frames) < at + len(ms):
            frames = np.concatenate([frames, np.full(at + len(ms) - len(frames), np.nan, np.float32)])
        frames = frames.astype(np.float32)
        frames[at:at + len(ms)] = ms
        _save_npy(path, frames)
        doc["blocks"] = sorted(set(doc["blocks"]) | {k})
        _write_json(self.render_dir / "bed.json", doc)

    async def _wait_for_bed(self) -> None:
        """The dub loop's GPU waits for the background sound (§2.14), its row saying so; a pause stops the wait."""
        if self._separated.is_set():
            return
        self._extra["voice_lines"] = WAITING_BED
        await self._save(force=True)
        try:
            while not self._separated.is_set():
                if self._stop.is_set():
                    raise _Paused
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._separated.wait(), timeout=HOLD_POLL)
        finally:
            self._extra.pop("voice_lines", None)

    # ---- the video download (§2.2) --------------------------------------------------------------------------------
    def _video_inputs(self) -> dict:
        return {"id": self.video_id, "format": VIDEO_FORMAT}

    async def _video(self) -> None:
        """The video stream the MP4 copies, beside the dub (it needs the network and no GPU). Served when the file it
        recorded is still there with its fingerprint, or from the output while the whole video's MP4 it went into still
        matches (the file itself was deleted then, §2.1); else fetched through the resolver (`_fetch_video`), the job's
        own download it recorded deleted first: a resolver hands back a video.* it finds, which is no longer the one
        asked for (another VIDEO_FORMAT) or not the one recorded (its fingerprint)."""
        st, source = self.doc["stages"]["video"], self.doc["source"]
        rec = source.get("video") if isinstance(source.get("video"), dict) else None
        if rec is not None and st.get("inputs") == self._video_inputs():
            if self._video_in_output(rec):
                log.info("render %s: the video went into the MP4 at %s, which still matches", self.video_id,
                         self.doc["output"]["path"])
                return
            path = self._video_file(rec)
            if path.is_file() and await asyncio.to_thread(_fingerprint, path) == rec.get("fingerprint"):
                st["done"] = st["total"] = round(path.stat().st_size / 1e6, 1)
                return
        if rec is not None:
            await asyncio.to_thread(self._drop_own_video, rec)
        await self._fetch_video()

    def _video_file(self, rec: dict) -> Path:
        return self._dir / str(rec.get("file") or "")  # (a file outside the job's folder is recorded whole)

    def _video_in_output(self, rec: dict) -> bool:
        """Whether the whole video's MP4 at `output.path` is still there, with its bytes, made from the video this stage
        would use now (§2.1)."""
        out = self._whole_output()
        return out is not None and (out.get("inputs") or {}).get("video") == {**self._video_inputs(),
                                                                                "fingerprint": rec["fingerprint"]}

    async def _fetch_video(self, count: bool = True) -> None:
        """Fetch the video through the resolver: before the download, the disk checks (the cache's volume and the output
        folder's); a pause stops the download at its next chunk. job.json's `source.video` records the file, whether the
        job owns it, its codec, size, frame rate and start, and its fingerprint (size and sha256), which stay after the
        file is deleted. `count`: the video stage's start (the export's own download counts as the export's)."""
        if count:
            await self._begin("video")
        st = self.doc["stages"]["video"]
        loop = asyncio.get_running_loop()
        checked = threading.Event()

        def progress(done: int, total: int) -> None:
            if self._stop.is_set():
                raise Cancelled("download paused")
            if total and not checked.is_set():
                checked.set()
                self._video_disk(total, downloading=True)
            loop.call_soon_threadsafe(self._progress, "video", round(done / 1e6, 1), round(total / 1e6, 1))

        got = await asyncio.to_thread(self.resolver.fetch_video, self.video, self.cache_dir, progress)
        if self._stop.is_set():
            raise _Paused
        size = got.path.stat().st_size
        if not checked.is_set():  # nothing downloaded (the file was there): the room for the rest
            self._video_disk(size, downloading=False)
        try:
            info = await asyncio.to_thread(mp4.probe, got.path)
        except (ValueError, OSError) as e:
            raise RenderError(f"The video can't be saved with the dub: {e}") from e
        fp = await asyncio.to_thread(_fingerprint, got.path)
        inside = got.path.parent.resolve() == self._dir.resolve()
        self.doc["source"]["video"] = {
            "file": got.path.name if inside else str(got.path.resolve()), "owned": got.owned and inside,
            **{k: info[k] for k in ("codec", "width", "height", "fps", "start")}, "fingerprint": fp}
        st["inputs"] = self._video_inputs()
        st["done"] = st["total"] = round(size / 1e6, 1)

    def _video_disk(self, size: int, downloading: bool) -> None:
        """The free space the video and the export need (§2.2): on the cache's volume, the video (`downloading`) and
        VIDEO_DISK_PER_HOUR an hour of video plus DISK_HEADROOM; on the output folder's, the video's size and
        OUTPUT_PER_HOUR an hour; both on one volume when they share it. Raises RenderError with the numbers."""
        hours = float(self.doc["duration"] or 0.0) / 3600
        cache = (size if downloading else 0) + VIDEO_DISK_PER_HOUR * hours + DISK_HEADROOM
        out = size + mp4.OUTPUT_PER_HOUR * hours
        here, there = self._dir, _existing(self.output_dir)
        if os.stat(here).st_dev == os.stat(there).st_dev:
            _need(here, cache + out, "Dubbing this video")
        else:
            _need(here, cache, "Dubbing this video")
            _need(there, out, f"Saving the dubbed video in {self.output_dir}")

    # ---- the export (§2.15-§2.17) ---------------------------------------------------------------------------------
    async def _export(self) -> None:
        """The MP4: served when the file at `output.path` is there with its recorded size and the inputs it was made
        from are the ones it would use now (§2.1); else made again from the start (a partial file of an earlier attempt
        deleted first), with the video downloaded again and the bed's blocks made again if the whole video's export
        deleted them. The output folder is checked first, that it can be written to and its volume's free space (hours
        may have passed since the video stage), before any download or re-made bed, and its free space again after.
        job.json's `output` says where it went, its size, kind, loudness and any warning."""
        manifest = await asyncio.to_thread(_read_json, self.render_dir / "manifest.json")
        if manifest is None or not isinstance(manifest.get("lines"), list):
            raise RenderError("The dub's final lines are missing. Resume to make them again.")
        english = self._english_cues()
        out = self.doc.get("output") if isinstance(self.doc.get("output"), dict) else None
        own = Path(out["path"]) if out and out.get("path") else None
        if own is not None and not (own.is_file() and own.stat().st_size == out.get("bytes")):
            own = None  # moved, deleted, or another file under its name now (another job's, the user's): not the job's
        dest = mp4.output_path(self.output_dir, mp4.output_name(self.doc["title"], self.video_id, self.settings.stop_at),
                               own)
        inputs = self._export_inputs(manifest, english, dest)
        if own is not None and out.get("inputs") == inputs:
            return
        await self._begin("export")
        for stale in {Path(self.doc["part"])} if self.doc.get("part") else set():
            with contextlib.suppress(OSError):
                stale.unlink()
        try:  # before a download or a re-made bed (minutes of GPU) that a folder it can't use would waste
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(self.output_dir, os.W_OK):
                raise PermissionError(errno.EACCES, "not writable")
        except OSError as e:
            raise RenderError(f"Maata can't save videos in {self.output_dir} ({e.strerror or e}). Choose another folder "
                              f"in Settings, then resume.") from e
        rec = self.doc["source"].get("video")
        path = self._video_file(rec) if isinstance(rec, dict) else None
        hours = float(self.doc["duration"] or 0.0) / 3600
        if path is not None and path.is_file():
            size = path.stat().st_size
        else:  # a download to come: the size the video stage recorded
            size = int(((rec if isinstance(rec, dict) else {}).get("fingerprint") or {}).get("size") or 0)
        _need(self.output_dir, size + mp4.OUTPUT_PER_HOUR * hours, f"Saving the dubbed video in {self.output_dir}")
        if path is None or not path.is_file():  # the whole video's export deleted it: downloaded again, by the same code
            vst = self.doc["stages"]["video"]
            vst["state"] = "running"  # its row shows the download (the start counted is the export's)
            try:
                await self._fetch_video(count=False)
            except BaseException:
                vst["state"] = "todo"  # the next run's video stage downloads it
                raise
            vst["state"] = "done"
            rec = self.doc["source"]["video"]
            path = self._video_file(rec)
            inputs = self._export_inputs(manifest, english, dest)
        bed = await self._background(manifest)
        _need(self.output_dir, path.stat().st_size + mp4.OUTPUT_PER_HOUR * hours, f"Saving the dubbed video in "
              f"{self.output_dir}")  # (again: the video as downloaded, and what the bed's blocks took meanwhile)
        self.doc["part"] = str(mp4.part_path(dest))
        await self._save(force=True)
        loop = asyncio.get_running_loop()

        def progress(done: float, total: float, extra: str | None) -> None:
            loop.call_soon_threadsafe(self._export_progress, done, total, extra)

        sr = int(manifest.get("sampleRate") or self.b.tts.sample_rate)
        lines = [mix.Line(float(x["start"]), str(x["speaker"]), self.render_dir / "pcm" / f"{x['pcm']}.npy",
                          int(x["samples"])) for x in manifest["lines"]]
        try:
            got = await asyncio.to_thread(
                mp4.export, dest, video=path, lines=lines, voice_sr=sr,
                telugu=subtitles.telugu_cues(manifest["lines"], sr), english=english,
                audio_start=float(self.doc["source"].get("audioStart") or 0.0), stop=self.settings.stop_at,
                metadata=self._metadata(), bed=bed, cancel=self._stop, progress=progress)
        except OSError as e:
            fix = ("Free some disk space" if e.errno in (errno.ENOSPC, errno.EDQUOT)
                   else "Check that the output folder can be written to")
            raise RenderError(f"The dubbed video couldn't be saved ({e.strerror or e}). {fix}, then resume.") from e
        finally:
            self._extra.pop("export", None)
        self.doc["part"] = None
        self.doc["output"] = {"path": str(dest), "bytes": got["bytes"], "kind": "whole" if self.settings.stop_at is None
                              else "preview", "at": round(time.time(), 1), "inputs": inputs,
                              "loudness": got["loudness"], "warning": got["warning"], "bed": bed is not None}
        st = self.doc["stages"]["export"]
        st["done"] = st["total"] = round(got["seconds"], 1)
        _STAGE.get().extra = {"bytes": got["bytes"], "media_seconds": got["seconds"], "bedGain": got["bedGain"],
                              **got["loudness"]}
        log.info("render %s saved %s (%.1f MB; %s)", self.video_id, dest, got["bytes"] / 1e6, got["loudness"])

    def _export_progress(self, done: float, total: float, extra: str | None) -> None:
        if extra:
            self._extra["export"] = extra
        self._progress("export", done, total)
        if extra:
            self._side(self._save(force=True))  # "Finishing the file…" shows at once

    def _english_cues(self) -> list[subtitles.Cue]:
        """The English track's cues on the dub's clock: every transcript unit, voiced, skipped or untranslated alike,
        over its words' times (§2.16)."""
        return subtitles.english_cues((st.unit.speaker, [(w.text, w.start, w.end) for w in st.unit.words])
                                      for st in self._order)

    def _export_inputs(self, manifest: dict, english: list[subtitles.Cue], dest: Path) -> dict:
        """What the MP4 is made from (§2.1): the final lines (PCM keys, starts, texts, speakers), digests of the English
        cue list, of the diarized turns (the balance's) and of the duck spans, the bed's inputs (None: no bed), the
        video, the audio's start, the metadata, the mix and subtitle constants, EXPORT_VERSION and the output path."""
        rec = self.doc["source"].get("video") or {}
        turns = self._speech_turns()
        return {"lines": _digest([[x["pcm"], x["start"], x["telugu"], x["speaker"]] for x in manifest["lines"]]),
                "english": _digest([[c.start, c.end, c.speaker, c.text] for c in english]),
                "turns": _digest([[t.speaker, t.start, t.end] for t in turns]), "bed": self._bed_inputs(),
                "duck": _digest(mix.duck_spans(*self._duck_inputs(manifest, turns))),
                "video": {**self._video_inputs(), "fingerprint": rec.get("fingerprint")},
                "audioStart": self.doc["source"].get("audioStart"), "stopAt": self.settings.stop_at,
                "metadata": self._metadata(),
                "mix": [mix.MIX_SR, mix.BLOCK, mix.VOICE_REF, mix.VOICE_CLAMP, mix.MIX_LUFS, mix.LIMITER,
                        mp4.AUDIO_BITRATE, mp4.aac(), mix.DUCK_DB, mix.DUCK_BARE_DB, mix.BRIDGE, mix.BARE_GAP,
                        mix.BARE_MIN, mix.ATTACK, mix.RELEASE, mix.BED_CLAMP],
                "subtitles": [subtitles.SUB_ROWS, subtitles.SUB_ROW, subtitles.SUB_MIN, subtitles.DASH],
                "version": mp4.EXPORT_VERSION, "path": str(dest)}

    def _speech_turns(self) -> list:
        """The current diarized speech turns of the whole video, overlaps kept (`turns_in(..., exclusive=False)`)."""
        return self.registry.turns_in(0.0, float(self.doc["duration"] or 0.0) + 1.0, exclusive=False)

    def _duck_inputs(self, manifest: dict, turns: list) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """The Telugu line spans (the manifest's: start and PCM length) and the English speech spans, on the dub's
        clock: what the ducking's spans are made of (§2.15)."""
        sr = int(manifest.get("sampleRate") or self.b.tts.sample_rate)
        return ([(float(x["start"]), float(x["start"]) + int(x["samples"]) / sr) for x in manifest["lines"]],
                [(t.start, t.end) for t in turns])

    async def _background(self, manifest: dict) -> mix.Background | None:
        """The bed the export mixes (§2.15), or None when there is none. Blocks of its range the bed lacks (the whole
        video's export deleted them, §2.17) are made again first, by the separate stage's code, on its row (the start
        counted is the export's). The balance's vocals level is the separator's vocals envelope over the current
        speech turns, so a speakers re-run never needs the separator."""
        inputs = self._bed_inputs()
        if inputs is None:
            return None
        try:
            await self._make_bed(inputs, self._bed_end(), count=False)
        finally:
            self.b.separator.release()
        doc = await asyncio.to_thread(self._bed_doc, inputs)
        frames = await asyncio.to_thread(_read_npy, self.render_dir / "bed_vocals.npy")
        turns = self._speech_turns()
        blocks = 0
        while blocks in doc["blocks"]:
            blocks += 1
        duck, bare = mix.duck_spans(*self._duck_inputs(manifest, turns))
        return mix.Background(self.render_dir / "bed", separate.block_samples(inputs["sampleRate"]), blocks,
                              separate.mean_square(frames if frames is not None else np.zeros(0, np.float32),
                                                   [(t.start, t.end) for t in turns]),
                              tuple(duck), tuple(bare))

    def _metadata(self) -> dict[str, str]:
        """The MP4's metadata (§2.17); `date` is the day the job was made, so an export made again is the same file."""
        d = self.doc
        return {"title": f"{d['title']} (Telugu)", "artist": d["channel"],
                "comment": f"Telugu dub by Maata (AI voices) of https://youtu.be/{self.video_id}",
                "date": time.strftime("%Y-%m-%d", time.localtime(float(d["createdAt"])))}

    # ---- take files and rows --------------------------------------------------------------------------------------
    def _load_takes(self, rows: list[dict], fixups: list[dict]) -> None:
        """takes.jsonl's `rows` by line key and fixups.jsonl's by key (a torn row, or one that doesn't read as one, is
        left out), then what the takes made before taught the estimator (§2.6) back into it, in the order they were
        made (each row's `made`: a take a shorter tier or a fix-up replaced, or one not kept, too), for each voice a
        speaker has now: a take of a voice since built again says nothing about the new one."""
        # A row written before the synthesis identity existed, or under another model/quality setting, cannot
        # restore a take or teach this run's duration estimator. Looking up the current occurrence also applies its
        # current short-line batch policy; merely accepting either batch size would keep old short/long choices.
        current = {(self._line_key(st), round(st.unit.start, 3)): _takes_n(st) for st in self.units.values()}
        rows = [r for r in rows if _take_row(r)
                and (n := current.get((r["line"], r["at"]))) is not None
                and r.get("synthesis") == self._take_inputs(n)]
        self._take_rows = {}
        for r in rows:
            self._take_rows.setdefault(r["line"], []).append(r)
        self._fixup_rows = {r["key"]: r for r in fixups if isinstance(r.get("key"), str)}
        self._made, self._new = set(), []
        keys = {tuple(self._voice_row(k)): k for k in (self._key(sid) for sid in self.registry.speakers)}
        replayed: set[str] = set()
        for r in rows:
            if (key := keys.get(tuple(r["voice"]))) is not None:
                for tk, spoken, pace in r["made"]:
                    self._made.add(tk)
                    # A changed text contract can change what an English map makes the TTS say even when the
                    # displayed Telugu is unchanged. Keep identical audio reusable, but do not teach its old
                    # Telugu-count/duration pairing to the current estimator (including legacy rows).
                    if r.get("durationPrompt") != PROMPT_HASH or tk in replayed:
                        continue
                    replayed.add(tk)
                    if pace is not None:
                        self.estimator.observe(spoken, key, pace)

    @staticmethod
    def _voice_row(key: VoiceKey) -> list:
        """A voice key as a take row keeps it: a clone's reference carries VOICE_VERSION (§2.1), and a preset is named
        with it, so a change to how voices are made changes every take key."""
        return [key.voice, key.cfg, key.exaggeration, key.reference or f"{VOICE_VERSION}:preset"]

    def _batch(self, n: int) -> int:
        """The takes a synthesis gives: `n` where the TTS batches them, else one."""
        return n if hasattr(self.b.tts, "synthesize_takes") else 1

    def _tts_revision(self) -> str | None:
        return getattr(self.b.tts, "model_revision", None) or _model_rev(self.b.name, "tts")

    def _take_inputs(self, n: int) -> dict:
        """Cheap synthesis identity for both files and rows: no model loading or weight hashing. The model revision
        comes from the process-cached lock manifest. The storage kind separates mel takes from sample-only backends;
        wrapper subclasses with the same backend/model and protocol remain compatible. Script is included so the
        estimator never learns Latin-input timings when this run synthesizes Telugu-script text, or vice versa."""
        tts = self.b.tts
        inputs = {"backend": self.b.name, "model": self._tts_revision(), "device": self.b.device,
                "t3Dtype": str(getattr(tts, "_t3_dtype", None)), "sampleRate": tts.sample_rate,
                "format": "mel" if hasattr(tts, "synthesize_mel") else "pcm", "batch": self._batch(n),
                "cfmSteps": getattr(tts, "cfm_steps", None), "ttsScript": self.tts_script, "version": TAKES_VERSION}
        if hasattr(tts, "cache_identity"):
            inputs["adapter"] = tts.cache_identity
        return inputs

    def _take_key(self, text: str, voice: list, n: int) -> str:
        """A take file's key (§2.1): text, voice and the complete synthesis identity. The same wording said with the
        same model, voice and settings shares one take wherever it occurs; changing them misses both file and row."""
        return _key20([text, voice, self._take_inputs(n)])

    def _line_key(self, st: UnitState) -> str:
        return line_key(self._line_spec(st, ("full",)), self.style)

    def _read_take(self, key: str) -> dict[str, np.ndarray] | None:
        """A take file's arrays (the take's, as `_write_take` keeps it, and what W needs of it); None when it is missing
        or doesn't read whole."""
        try:
            with np.load(self.render_dir / "takes" / f"{key}.npz", allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            return None
        if not {"seconds", "pauses", "failed", "spoken", "pace"} < d.keys():
            return None
        if "samples" in d and take_qa.audio_failure(d["samples"]) is not None:
            return None
        return d

    def _claim_take(self, key: str) -> dict[str, np.ndarray] | None:
        """`_read_take` for the dub loop, which is about to use that take (restore a row with it, take it from the take
        cache, or write it): the garbage collection, which runs beside the loop, leaves it (`_sweep`)."""
        with self._take_lock:
            self._pending_takes.add(key)
            if self._touched is not None:
                self._touched.add(key)
            return self._read_take(key)

    async def _write_take(self, key: str, take: object, seconds: float, pauses: tuple[tuple[float, float], ...],
                          failed: str | None, spoken: str, pace: float | None) -> None:
        """A take's file (§2.9): its mel as the TTS's `pack_take` keeps it (samples, for a TTS without mel takes), with
        what W and the estimator need of it, written whole and then put in place, so a file is never half there."""
        if isinstance(take, np.ndarray):
            d = {"samples": np.asarray(take, np.float32)}
        else:
            d, _ = await self._gpu_now(self.b.tts.pack_take, take)  # the flow, if it hasn't run: the take was vocoded
        d.update(seconds=np.float64(seconds), pauses=np.asarray(pauses, np.float64).reshape(-1, 2),
                 failed=np.str_(failed or ""), spoken=np.str_(spoken), pace=np.float64(np.nan if pace is None else pace))
        with _disk("a take"):
            await asyncio.to_thread(_save_npz, self.render_dir / "takes" / f"{key}.npz", d)

    async def _take_from(self, d: dict[str, np.ndarray]) -> object:
        """A take back from its file's arrays: its samples, or its mel on the TTS's device (`unpack_take`)."""
        if "samples" in d:
            return np.asarray(d["samples"], np.float32)
        take, _ = await self._gpu_now(self.b.tts.unpack_take, d)
        return take

    async def _load_said(self, row: dict) -> Said:
        """A take row's takes back from their files, as `_say` made them, but for their natural audio: samples where
        the take is samples, else None (its PCM vocodes the mel again)."""
        takes = [await self._take_from(await asyncio.to_thread(self._read_take, t["key"])) for t in row["takes"]]
        return Said(wordings=[Wording(t["spoken"]) for t in row["takes"]], takes=takes,
                    seconds=[float(t["seconds"]) for t in row["takes"]],
                    audio=[x if isinstance(x, np.ndarray) else None for x in takes],
                    pauses=[_spans(t["pauses"]) for t in row["takes"]], paces=[t.get("pace") for t in row["takes"]],
                    failed=next((t["failed"] for t in row["takes"] if t.get("failed")), None),
                    failures=[t.get("failed") for t in row["takes"]])

    def _settled_key(self, sc: Scene) -> str:
        """A scene's `settled` row in fixups.jsonl: its lines (line key and onset) and what answered its fix-ups."""
        return _key20(["settled", [[self._line_key(st), round(st.unit.start, 3)] for st in sc.lines],
                       self.tr.prompt_hash, self.tr.model, REVIEW_HASH, self._approval_policy()])

    def _approval_policy(self) -> str:
        """Approval provenance is independent of wording and PCM compatibility."""
        return getattr(self.tr, "approval_policy", APPROVAL_POLICY)

    def _fixup_key(self, kind: str, line: str, spoken: str) -> str:
        """A fixups.jsonl row's key (§2.10): what it is, the line key, the wording it is about and what answered it (the
        translator's prompt and model; the review's prompt too)."""
        return _key20([kind, line, spoken, self.tr.prompt_hash, self.tr.model,
                       REVIEW_HASH if kind == "review" else None, self._approval_policy()])

    def _put_fixup(self, st: UnitState, key: str, kind: str, line: LineResult) -> None:
        row = {"key": key, "kind": kind, "line": self._line_key(st), "answer": line_json(line),
               "model": self.tr.model, "prompt_hash": self.tr.prompt_hash,
               "approval_policy": self._approval_policy(),
               "coverage": coverage_json(line.coverage), "at": round(time.time(), 3)}
        with _disk("the fix-ups"):
            _append_row(self.render_dir / "fixups.jsonl", row)
        self._fixup_rows[key] = row

    def _fixup_line(self, st: UnitState, row: dict | None) -> LineResult | None:
        """A fixups.jsonl answer back as the line it was for, with its class; None when there is none (or it no longer
        validates)."""
        # A take refers to its fix-up by the stored key, bypassing a fresh _fixup_key lookup. Check the answer's
        # provenance here too, so an older model/prompt cannot replace wording newly translated on a resume.
        if (row is None or row.get("model") != self.tr.model or row.get("prompt_hash") != self.tr.prompt_hash
                or not isinstance(row.get("answer"), dict)):
            return None
        line, _ = check_line(row["answer"], self._line_spec(st, ()))
        if line is not None:
            if row.get("approval_policy") == self._approval_policy():
                line.coverage = coverage_from(row.get("coverage"))
            else:
                # Reuse checked wording, but neither heuristic completeness nor unreviewed Latin substitutions.
                # Identical audio may still restore; _review_voiced must then approve its actual wording afresh.
                line.coverage = None
                line.tiers = {tier: Wording(wording.spoken) for tier, wording in line.tiers.items()}
        return line

    async def _gpu_now(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """`_on_gpu` for a call that finishes work already done (keeping a take just made, or reading one back), which a
        pause doesn't stop."""
        queued_at = time.perf_counter()
        out, seconds = await super()._on_gpu(fn, *args, **kwargs)
        self._gpu_work(fn, seconds, queued_at)
        return out, seconds

    # ---- Dubber's hooks, for the lanes ----------------------------------------------------------------------------
    def _release(self, req: SceneRequest) -> None:
        super()._release(req)
        if (sc := _SCENE.get()) is not None:
            sc.released = True

    async def _claude_failed(self, err: ClaudeCLIError) -> None:
        if (sc := _SCENE.get()) is not None:
            sc.failed = err
        await super()._claude_failed(err)

    def _side(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = super()._side(coro)
        if (sc := _SCENE.get()) is not None:
            sc.fit = task  # the one side task a scene call starts: its fit
            task.add_done_callback(lambda _: self._wake.set())  # the dub loop waits for it
        return task

    def _fit_spec(self, st: UnitState) -> LineSpec | None:
        if next(self._rows_for(st), None) is not None:
            return None  # the dub loop restores it in the wording its take row says: no tier it lacks is wanted
        return super()._fit_spec(st)

    async def _review(self, req: SceneRequest, lines: dict[int, LineResult]) -> dict[int, LineResult] | None:
        if self._stop.is_set():
            return None  # paused: no new call. The scene is released, and its lines come from the line cache on resume
        return await super()._review(req, lines)

    async def _skip(self, st: UnitState, why: str) -> None:
        self.skipped[st.unit.id] = why
        if _SCENE.get() is not None:  # the translator's ladder gave up on it: never asked again (§2.8)
            _append_row(self.render_dir / "skipped.jsonl",
                        {"key": line_key(self._line_spec(st, ("full",)), self.style), "prompt_hash": self.tr.prompt_hash,
                         "model": self.tr.model, "unit": st.unit.id, "why": why})
        await super()._skip(st, why)

    def _learn_k(self, st: UnitState) -> None:
        if st.unit.id not in self._learned:  # once a run: a scene asked again after its review failed isn't counted twice
            self._learned.add(st.unit.id)
            super()._learn_k(st)

    # ---- Dubber's hooks, for the dub loop --------------------------------------------------------------------------
    async def _voice_for(self, sid: str, cost: VoiceCost) -> tuple[object, VoiceKind, VoiceKey]:
        """The voice the voices stage made for the speaker (§2.6): their clone, else their preset. Never a clone made on
        a line's path."""
        v = self.voices.setdefault(sid, VoiceState())
        if v.kind in (VoiceKind.CLONED, VoiceKind.NATIVE) and v.voice is not None and not v.use_preset:
            return v.voice, v.kind, self._key(sid)
        name = _preset_name(sid)
        return await self._preset(name, cost), VoiceKind.PRESET, self._voice_key(name, None)

    def _budget_left(self) -> bool:
        return self._resynths < self._budget  # shorter tiers and rephrases together (M4)

    async def _said_take(self, st: UnitState, w: Wording, voice: object, key: VoiceKey, seconds: float,
                         cost: VoiceCost, n: int, speech: float
                         ) -> tuple[object, float, np.ndarray | None, tuple[tuple[float, float], ...], float | None,
                                    str | None]:
        """The content-keyed take cache (§2.1): a wording the voice has said before (with as many takes to pick from)
        comes from its file, with no synthesis; a file is never written over, so every row that names it keeps meaning
        the same take. A take from a file brings no natural audio into memory where it isn't samples (its PCM vocodes
        its mel then). Else it is synthesized, and its file written before any row names it.
        Each take key teaches the estimator once a render, where it is first made (`_made`): one a row of the render
        has already made teaches nothing, while a take from a file that no row has (made before a pause, its line's row
        never written) teaches as its synthesis did, with the pace kept in its file. Each take made goes to `_new`, for
        the row that keeps what it taught."""
        tk = self._take_key(self._tts_text(w), self._voice_row(key), n)
        d = await asyncio.to_thread(self._claim_take, tk)
        if d is not None:
            take = await self._take_from(d)
            out = (take, float(d["seconds"]), take if isinstance(take, np.ndarray) else None, _spans(d["pauses"]),
                   float(d["pace"]) if np.isfinite(d["pace"]) else None, str(d["failed"]) or None)
        else:
            try:
                out = await super()._said_take(st, w, voice, key, seconds, cost, n, speech)
            except take_qa.UnusableAudioError as err:
                raise UnusableSpeechError(f"Line {st.unit.id + 1} produced no usable speech. Resume to try it again.") from err
            take, dur, _, pauses, pace, failed = out
            await self._write_take(tk, take, dur, pauses, failed, w.spoken, pace)
        if tk in self._made:
            return (*out[:4], None, out[5])
        self._made.add(tk)
        self._new.append([tk, w.spoken, out[4]])
        return out

    # ---- per-line queries over the onset index (Dubber's scan every unit) ------------------------------------------
    def _context_for(self, st: UnitState) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        k = bisect_left(self._onsets, st.unit.start) - 1
        while k >= 0 and len(out) < 2:
            s = self._order[k]
            if s.line and s.tier:
                out.append((s.unit.text, s.line.tiers[s.tier].spoken))
            k -= 1
        return out[::-1]

    def _slot(self, st: UnitState) -> LineSlot:
        u = st.unit
        prev_end = self._ended[bisect_left(self._onsets, u.start)]
        spoken = None if self.timing == "v1" else self._heard_before(u)
        before = max((t for t in (prev_end, spoken) if t is not None), default=None)
        inside = self._onsets[bisect_right(self._onsets, u.start):bisect_left(self._onsets, u.end)]
        breaks = tuple((a, b) for a, b in self._pauses_at(u, u.breaks) if not any(a <= t <= b for t in inside))
        return LineSlot(u.id, u.speaker, u.start, u.end, st.next_start, before,
                        overlaps_prev=prev_end is not None and prev_end > u.start + 0.05,
                        rate_cap=rate_ceiling(self.estimator.rate(self._key(u.speaker)), self.planner.s),
                        breaks=breaks, heard=self._heard_in(st, breaks), anchors=self._pauses_at(u, u.anchors),
                        speech=self._speech_spans(u))

    def _heard_in(self, st: UnitState, breaks: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        u, out = st.unit, set()
        for a, b in breaks:
            spans = [(t.start, t.end) for t in self.registry.turns_in(a, b, exclusive=False) if t.speaker != u.speaker]
            near = self._order[bisect_left(self._onsets, a - self._max_len - 1e-6):bisect_left(self._onsets, b)]
            spans += [(s.unit.start, s.unit.end) for s in near if s is not st]
            out |= {(round(max(x, a), 3), round(min(y, b), 3)) for x, y in spans if x < b and y > a}
        return tuple(sorted(out))

    def _window(self, st: UnitState) -> list[UnitState]:
        s, u = self.planner.s, st.unit
        k = bisect_right(self._onsets, u.start)
        later = self._order[k:k + s.window_lines]
        return [x for j, x in enumerate(later) if j == 0 or x.unit.start <= u.start + s.window_s]

    def _ahead(self, st: UnitState, planner: TimelinePlanner | None = None) -> tuple[tuple[LineSlot, float | None], ...]:
        if planner is None or planner is self.planner:
            return super()._ahead(st, planner)
        # The final plan F prices only takes (§2.11): a line skipped, or past the lines this run voices, is None.
        return tuple((self._slot(x), x.take_s if x.take is not None else None) for x in self._window(st))


def _takes_n(st: UnitState) -> int:
    """The takes a line is voiced from (§2.9): TAKES_N, or TAKES_N_SHORT under SHORT_SPEECH s of speech."""
    return TAKES_N_SHORT if st.speech_s < SHORT_SPEECH else TAKES_N


def _tts_hash(texts: list[str]) -> str:
    """A take row's hash of what the TTS said: its wording, or its pieces in order."""
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]


def _key20(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()[:20]


def _spans(raw: object) -> tuple[tuple[float, float], ...]:
    return tuple((float(a), float(b)) for a, b in np.asarray(raw, np.float64).reshape(-1, 2))


def _said_of(row: dict, words: list[Wording] | None = None) -> Said:
    """A take row's takes as the planner takes them (their keys, natural seconds and pauses, with no audio): what a
    restored line is placed on W with, and every line on F (§2.11). `words`: its wordings, else what each take said."""
    takes = row["takes"]
    return Said(wordings=list(words) if words is not None else [Wording(t["spoken"]) for t in takes],
                takes=[t["key"] for t in takes], seconds=[float(t["seconds"]) for t in takes],
                pauses=[_spans(t["pauses"]) for t in takes], paces=[t.get("pace") for t in takes],
                failed=next((t["failed"] for t in takes if t.get("failed")), None),
                failures=[t.get("failed") for t in takes])


def _group_seconds(group: tuple[str, ...], each: Callable[[str], float]) -> float:
    """A group's seconds from its stages' (`each`): side by side, its longest; but where the GPU runs `separate` and then
    the dub loop (SERIES, §2.14), those count in series: max(translate, video, separate + max(voice_lines, finish))."""
    first, then = SERIES
    if first not in group:
        return max(each(key) for key in group)
    return max([each(first) + max(each(key) for key in then)]
               + [each(key) for key in group if key != first and key not in then])


def _span(key: str, duration: float, stop_at: float | None) -> float:
    """Seconds of video in a stage's range (§3): the whole video, but for translation, the dub loop, the final plan and
    the export of a preview, up to its stop point plus PAST_STOP, and for the background sound up to its stop point plus
    twice that (the preview's cut, §2.14) (the video download is the whole stream either way)."""
    if stop_at is None:
        return duration
    if key == "separate":
        return min(duration, stop_at + 2 * PAST_STOP)
    if key not in ("translate", "voice_lines", "finish", "export"):
        return duration
    return min(duration, stop_at + PAST_STOP)


def shown_state(key: str, st: dict, duration: float, stop_at: float | None) -> str | None:
    """A stage's state as the job view shows it (§5): `todo` for one done for a shorter range than the job's now (a
    preview's, continued to the whole video) until it runs for that range, so the job view doesn't tick "Saving the
    video" for the preview's file while the whole video is being dubbed. job.json keeps `done` and its `span`: the ETA
    counts what the longer range adds (`_prior_left`)."""
    if st.get("state") == "done" and st.get("span") is not None and st["span"] < _span(key, duration, stop_at) - 0.05:
        return "todo"
    return st.get("state")


def _prior_left(key: str, st: dict, duration: float, stop_at: float | None) -> float:
    """Seconds a stage has left by the static priors (§7): its share left of its range's. A stage done for a shorter
    range (`span`: a preview's, continued to the whole video) has the video seconds the new range adds left: what it
    did is served from its caches."""
    span = _span(key, duration, stop_at)
    if st["state"] == "done":
        return sum(PRIORS[key]) / 2 * max(span - st["span"], 0.0) if st.get("span") is not None else 0.0
    left = 1.0 - st["done"] / st["total"] if st["total"] else 1.0
    return sum(PRIORS[key]) / 2 * span * min(max(left, 0.0), 1.0)


def left(doc: dict, separates: bool = True) -> float:
    """Seconds a job that isn't running has left, from its job.json (the stages of a group side by side), by the static
    priors: a queued job's share of the time ahead of a new one (§4). `separates`: the backend has a separator (else
    the background sound is served at once)."""
    stages, stop = doc.get("stages") or {}, (doc.get("settings") or {}).get("stopAt")
    duration = float(doc.get("duration") or 0.0)
    todo = {"state": "todo", "done": 0, "total": 0}
    return sum(_group_seconds(group, lambda key: 0.0 if key == "separate" and not separates
                              else _prior_left(key, stages.get(key) or todo, duration, stop))
               for group in ORDER)


def estimate(duration: float, stop_at: float | None = None, separates: bool = True) -> dict:
    """What a render of a video `duration` s long should take on the M5 Pro, from the static priors of §7, (low, high):
    wall seconds (the stages of a group side by side) and Claude calls (the scenes' about CLAUDE_PER_LINE a line, and
    the brief's parts). For the prepare view's estimate line (§5). `separates`: the backend has a separator (else the
    background sound is served at once, and counts nothing)."""
    seconds = [sum(_group_seconds(group, lambda k: 0.0 if k == "separate" and not separates
                                  else PRIORS[k][i] * _span(k, duration, stop_at)) for group in ORDER)
               for i in (0, 1)]
    lines = LINES_PER_S * _span("translate", duration, stop_at)
    parts = max(1, math.ceil(duration * 2.9 / BRIEF_PART_WORDS))  # about 2.9 words a second of talk (his video)
    return {"seconds": [round(s) for s in seconds],
            "claudeCalls": [round(CLAUDE_PER_LINE[i] * lines) + parts for i in (0, 1)]}


def _report(manifest: dict) -> dict:
    """job.json's `report` of the final lines, for the job view's Done card (§5): how many there are, how many play
    flagged long (still over their time after the fix-ups) or unreviewed, and the overdraft seconds (each line's own,
    and what the plan carried to the lines after it, §2.11)."""
    flags = [line.get("flags") or [] for line in manifest["lines"]]
    stats = manifest["stats"]
    return {"lines": len(flags), "long": sum("long" in f for f in flags),
            "unreviewed": sum("unreviewed" in f for f in flags), "overdraft": stats["overdraft_s"],
            "carried": stats["overdraft_carried_s"]}


def found_message(video_id: str, diar: dict, reg: SpeakerRegistry, job: dict, duration: float,
                  fresh: bool = False) -> dict:
    """The `speakers_found` message (§4) from diarization.json (`diar`), its registry and job.json (`job`): who was
    found, how much each talks and where, and what was merged; `fresh` when this run diarized (not served from disk),
    and `freeFor`, the seconds (by the priors) until the dub loop starts: until then a correction costs no speech
    synthesis (§2.3)."""
    settings, stages = job.get("settings") or {}, job.get("stages") or {}
    k, stop = settings.get("speakers"), settings.get("stopAt")
    todo = {"state": "todo", "done": 0, "total": 0}
    free = sum(max(_prior_left(key, stages.get(key) or todo, duration, stop) for key in group)
               for group in ORDER[ORDER.index(("transcript",)):ORDER.index(("voices", "brief")) + 1])
    return {"type": "speakers_found", "videoId": video_id, "mode": "hint" if k else "auto", "fresh": fresh,
            "freeFor": round(free), "bounds": None if k else list(AUTO_SPEAKERS),
            "speakers": [{**row, "label": reg.speakers[row["id"]].label, "activity": activity(reg, row["id"], duration)}
                         for row in diar.get("speakers", []) if row["id"] in reg.speakers],
            "merged": [{"from": a, "into": b, "why": why, "talkSeconds": talk}
                       for a, b, why, talk in diar.get("merged", [])]}


def found_on_disk(video_dir: Path, job: dict | None = None) -> dict | None:
    """A job's `speakers_found` rebuilt from its diarization.json, for a window that connects after the job's speakers
    stage ran (a reload, an engine restarted, a relaunch: §5) or for a job that isn't running. `job`: its job.json (read
    when None; the running job's own doc otherwise). None until that stage is done for the job's speaker count (a
    `set_speakers` waiting to run, or running it, has none yet), or when the file doesn't read."""
    job = job if job is not None else _read_json(video_dir / "render" / "job.json")
    diar = _read_json(video_dir / "render" / "diarization.json")
    if job is None or diar is None:
        return None
    k = (job.get("settings") or {}).get("speakers")
    if ((job.get("stages") or {}).get("speakers") or {}).get("state") != "done" \
            or (diar.get("inputs") or {}).get("speakers") != (k or "auto"):
        return None
    duration = float(job.get("duration") or 0.0)
    try:
        reg = SpeakerRegistry.restore(diar, 0.0, duration)
        return found_message(video_dir.name, diar, reg, job, duration)
    except (KeyError, TypeError, ValueError, IndexError):
        log.warning("couldn't read the speakers of %s", video_dir.name, exc_info=True)
        return None


def output_view(doc: dict) -> dict | None:
    """job.json's `output` as the UI shows it (§4): where the MP4 went, its size, kind, loudness and warning, and
    whether it is still there ("File moved or deleted" when not)."""
    out = doc.get("output")
    if not isinstance(out, dict):
        return None
    missing = not Path(str(out.get("path") or "")).is_file()
    return {**{k: v for k, v in out.items() if k != "inputs"}, "missing": missing,
            **({"note": "File moved or deleted"} if missing else {})}


def _existing(path: Path) -> Path:
    """`path`, or the nearest folder above it that exists (where a folder made on first use will be)."""
    path = path.expanduser().absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _need(path: Path, need: float, what: str) -> None:
    """Fail saying how much is needed when `path`'s volume has less than `need` bytes free."""
    free = shutil.disk_usage(_existing(path)).free
    if free < need:
        raise RenderError(f"{what} needs about {need / 1e9:.1f} GB of free disk space, and {free / 1e9:.1f} GB is free.")


def _lane_bounds(n: int, weights: tuple[float, ...]) -> list[tuple[int, int]]:
    """Partition ordered scenes without gaps or duplicate work, retaining a scene per lane when possible.
    Equal weights reproduce the original integer-division partition; fewer than three scenes get their own lanes."""
    if n < 0 or len(weights) != LANES or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("lane sizes require a nonnegative scene count and three finite positive weights")
    total = sum(weights)
    if not math.isfinite(total):
        raise ValueError("lane weights must have a finite sum")
    if n < LANES:
        return [(i, i + 1) for i in range(n)]
    if len(set(weights)) == 1:
        return [(i * n // LANES, (i + 1) * n // LANES) for i in range(LANES)]
    edges = [0]
    for i in range(1, LANES):
        edge = int(n * (sum(weights[:i]) / total))
        edges.append(max(edges[-1] + 1, min(edge, n - (LANES - i))))
    edges.append(n)
    return list(zip(edges, edges[1:]))


def _take_row(r: dict) -> bool:
    """Whether a takes.jsonl row reads as one (§2.9)."""
    takes, made = r.get("takes"), r.get("made")
    return (isinstance(r.get("line"), str) and isinstance(r.get("at"), (int, float))
            and isinstance(r.get("wording"), str) and isinstance(r.get("tts"), str)
            and isinstance(r.get("voice"), list) and len(r["voice"]) == 4
            and all(v is None or isinstance(v, (str, int, float)) for v in r["voice"])
            and isinstance(r.get("fixups"), int)
            and isinstance(takes, list) and bool(takes)
            and all(isinstance(t, dict) and isinstance(t.get("key"), str) and isinstance(t.get("spoken"), str)
                    and isinstance(t.get("seconds"), (int, float)) and isinstance(t.get("pauses"), list) for t in takes)
            and isinstance(made, list)
            and all(isinstance(m, list) and len(m) == 3 and isinstance(m[0], str) and isinstance(m[1], str)
                    and (m[2] is None or isinstance(m[2], (int, float))) for m in made))


@contextlib.contextmanager
def _disk(what: str) -> Iterator[None]:
    """A write of the dub loop's: when the disk refuses it (full, or no permission), the render stops saying so, and a
    resume goes on from what is on disk (rather than every line after it being skipped)."""
    try:
        yield
    except OSError as e:
        fix = ("Free some disk space" if e.errno in (errno.ENOSPC, errno.EDQUOT)
               else "Check that the cache folder can be written to")
        raise RenderError(f"The dub couldn't save {what} ({e.strerror or e}). {fix}, then resume.") from e


def _voice_kind(row: dict) -> str:
    """How a take row's line was voiced: "preset" or "cloned" (`_voice_row`)."""
    ref = str(row["voice"][3])
    return "native" if ref.startswith("native:") else "preset" if ref.endswith(":preset") else "cloned"


def _pcm_frames(path: Path) -> int | None:
    """The frames of a line's PCM file (from its header), or None when it is missing or doesn't read."""
    try:
        return int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
    except (OSError, ValueError, IndexError):
        return None


def _save_pcm_file(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_pcm(path, audio)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a take file whole, then put it in place: a reader never sees one half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _fill(frames: Iterator[np.ndarray], buf: separate.Buffer, until: int) -> int | None:
    """Decode into `buf` until it has the file's samples up to `until`. Returns the file's length in samples when the
    decode ends first, else None."""
    while buf.end < until:
        x = next(frames, None)
        if x is None:
            return buf.end
        buf.put(x)
    return None


def _read_npy(path: Path) -> np.ndarray | None:
    """An .npy file's array, or None when it is missing or doesn't read whole."""
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Write an .npy file whole, then put it in place: a reader never sees one half written."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def _block_ok(folder: Path, k: int, frames: np.ndarray | None, sr: int) -> bool:
    """Whether bed block k reads whole (float16 [2, m], at most a block's samples) and its vocals envelope is there."""
    try:
        x = np.load(folder / f"{k:05d}.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return False
    m = x.shape[1] if x.ndim == 2 and x.shape[0] == 2 else 0
    if x.dtype != np.float16 or not 0 < m <= separate.block_samples(sr):
        return False
    at, f = k * round(separate.SEP_BLOCK / separate.FRAME), separate.frames_in(m, sr)
    return frames is not None and frames.ndim == 1 and len(frames) >= at + f and bool(np.isfinite(frames[at:at + f]).all())


def _video_ref(text: str) -> VideoRef:
    """A render's video: a YouTube link, or a bare video id (a job resumed from its cache directory)."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text.strip()):
        return VideoRef(text.strip())
    return parse_youtube_url(text)


@functools.lru_cache(maxsize=None)
def _model_rev(backend: str, role: str) -> str | None:
    """The pinned commit of the model `backend` uses for `role` (models.lock.json); None for the mock backend."""
    try:
        models = load_lock().get("models", [])
    except (OSError, ValueError):
        return None
    return next((m.get("revision") for m in models if m.get("role") == role and backend in m.get("backends", ())), None)


def _fingerprint(path: Path) -> dict:
    """The audio's identity: its size and the sha256 of its bytes, with no mtime, so a copy or a re-download of the same
    bytes is the same audio (§2.1)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()[:12]


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _save_f32(audio: np.ndarray, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    np.asarray(audio, np.float32).tofile(part)
    os.replace(part, dest)


def _read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _write_json(path: Path, doc: dict) -> None:
    """Write the whole file, then put it in place: a reader never sees it half written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write_json_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict]:
    """A JSONL file's rows in order, every one of them (not `_Jsonl`'s last row per key); a line that doesn't parse, the
    torn last line of a crash, is left out, even one cut inside a UTF-8 character. Lines are split on bytes, at \\n and
    \\r only, which JSON always escapes (str.splitlines would also split at a raw U+2028 inside a string)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    rows = []
    for line in data.splitlines():
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_row(path: Path, row: dict) -> None:
    """Append a row; after a torn last line it starts a line of its own."""
    with path.open("a+b") as f:
        f.seek(0, os.SEEK_END)
        torn = False
        if f.tell():
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        f.write(b"\n" * torn + (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def _peak_rss_bytes() -> int | None:
    """The engine's peak resident memory so far; None where there is no `resource` module (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
