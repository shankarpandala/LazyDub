"""Engine-stage protocols (spec §5). Every model sits behind one of these so engines can be swapped."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Container, Mapping, Protocol, Sequence

import numpy as np

from ..speakers import DiarBlock
from ..types import TimedWord

SR_ANALYSIS = 16_000
COARSE_STEP = 0.2  # whole-file diarization's fallback segmentation step, as a share of its window (OFFLINE-RENDER §2.3)


class Cancelled(RuntimeError):
    """A long model call stopped early because its `cancel` event was set (a render paused: OFFLINE-RENDER §2.1)."""


@dataclass(slots=True)
class Transcript:
    words: list[TimedWord]
    language: str | None
    # What the ASR guards did (ARCHITECTURE §3.4), on the audio's clock: diarized speech that had no words and was
    # decoded again, how many words that brought back and how many it found but didn't trust, whether a punctuation
    # pass was transferred onto the words, and how many of that pass's guesses at its window cuts were left out.
    gaps: list[tuple[float, float]] = field(default_factory=list)
    recovered: int = 0
    rejected: int = 0
    punctuated: bool = False
    edge_guesses: int = 0


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   speech: Sequence[tuple[float, float]] = ()) -> Transcript:
        """Timed words for 16 kHz `audio`. `speech`: the diarized speech spans on the audio's clock, for the coverage
        guard (a transcriber may ignore them)."""
        ...


class Diarizer(Protocol):
    def diarize(self, audio: np.ndarray, *, num_speakers: int | None = None, min_speakers: int | None = None,
                max_speakers: int | None = None, step: float | None = None,
                progress: Callable[[float], None] | None = None, cancel: threading.Event | None = None) -> DiarBlock:
        """Diarize a whole file of 16 kHz audio at offset 0, in one clustering (OFFLINE-RENDER §2.3): `num_speakers`
        exactly, or between `min_speakers` and `max_speakers`. `step`: the segmentation step as a share of the window
        (None: the diarizer's own, with its fallbacks). `progress(fraction)` may be called from the worker thread; a set
        `cancel` raises `Cancelled` at the diarizer's next check. The block's `step` says which step was used, and its
        `embeddings` how many embeddings the clustering took, where the diarizer knows."""
        ...


# ---- scene translation (docs/research/dubbing-2026-09/ARCHITECTURE.md §4) -------------------------------------------
# Wordings of one line, shortest first. `full` is the natural complete line and always present; the others exist only
# when asked for, and are strictly ordered by aksharas.
TIERS = ("very_concise", "concise", "full", "fuller")
EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised", "serious")
ENERGIES = ("low", "mid", "high")
CALLS = ("scene", "fit", "retranslate", "rephrase")


@dataclass(frozen=True, slots=True)
class Wording:
    """One way to say a line. `spoken` is Telugu script only: the TTS input, the QA reference and what aksharas are
    counted on. `english` maps the index of each English word in `spoken` (split on whitespace) to its Latin form, so
    captions and the English-word lint are rebuilt from these two fields alone."""

    spoken: str
    english: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Delivery:
    emotion: str = "neutral"
    energy: str = "mid"
    question: bool = False
    emphasis: tuple[int, ...] = ()  # word indices in `full.spoken`


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    term: str                  # as the English says it
    spoken: str                # its Telugu-script spelling (keep_english) or its fixed Telugu rendering
    keep_english: bool = True
    note: str = ""             # a pronunciation note, from the brief


@dataclass(frozen=True, slots=True)
class LineSpec:
    """One line to translate, as the job sizes it. The last five fields belong to one call type each."""

    id: int
    speaker: str
    en: str
    start: float
    end: float
    speech_s: float                        # diarized speech time inside the span, not the raw span
    target_aksharas: float                 # (speech time - the voice's overhead) x its calibrated rate
    want: tuple[str, ...] = ("full",)      # tiers to ask for, set locally before the call (§4.3)
    to: str | None = None                  # the speaker addressed, when known
    breaks: tuple[float, ...] = ()         # video times of pauses >= 1 s inside the sentence
    cut_off: bool = False                  # the speaker was interrupted
    delivery_hint: dict[str, float] | None = None  # energy_rel, rate_rel, pitch_rel_st
    current: str | None = None             # fit: the chosen wording; rephrase: the wording TTS failed on
    overflow: float | None = None          # fit: aksharas over (+) or under (-) the slot
    missing: tuple[str, ...] = ()          # retranslate: English words the review found missing
    finding: dict | None = None            # rephrase: the QA finding (skipped words, CER, overrun)
    problems: tuple[str, ...] = ()         # set by the translator when it asks again: what was wrong last time


@dataclass(frozen=True, slots=True)
class SceneRequest:
    scene: int
    lines: tuple[LineSpec, ...]
    call: str = "scene"                                        # one of CALLS
    context_before: tuple[tuple[str, str | None], ...] = ()    # (English, final Telugu or None when not translated)
    context_after_en: tuple[str, ...] = ()                     # context only, never translated
    deadline: float | None = None                              # rephrase: epoch s after which the answer is useless


@dataclass(slots=True)
class LineResult:
    id: int
    tiers: dict[str, Wording]              # `full` always
    delivery: Delivery = field(default_factory=Delivery)
    pieces: tuple[str, ...] = ()           # split at natural Telugu breaks; join back to full.spoken
    moved: tuple[str, ...] = ()            # content that had to move between pieces
    unfinished: bool = False
    flags: tuple[str, ...] = ()            # what the validators repaired or suspect; never blocking
    cached: bool = False
    model: str | None = None               # the model that answered
    brief_version: int | None = None
    coverage: Coverage | None = None       # the coverage review's class once reviewed (§4.6)

    @property
    def full(self) -> Wording:
        return self.tiers["full"]


@dataclass(slots=True)
class SceneResult:
    scene: int
    call: str
    lines: dict[int, LineResult] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)      # id -> why; never padded, never English
    glossary_additions: list[GlossaryEntry] = field(default_factory=list)
    dropped: bool = False                  # rephrase: arrived (or would start) after its deadline
    calls: int = 0                         # CLI calls made for this request
    seconds: float = 0.0
    error: Exception | None = None         # review: a failure the user must fix, which left lines unreviewed


@dataclass(frozen=True, slots=True)
class Coverage:
    """A line's coverage class (§4.6; research gap-4 E5): C complete, m a minor drop, P a phrase or clause missing, E a
    meaning error. `by` says who classed it: "review" (the semantic reviewer, on wording `tier`) or "validators" (a
    re-translation checked deterministically). `first` keeps the rejected wording's review class when a correction
    replaces it, or when the same review explicitly approves an eligible existing full wording instead."""

    cls: str
    missing: tuple[str, ...] = ()          # English words whose meaning the Telugu lacks
    added: tuple[str, ...] = ()            # Telugu content not in the English, said in English
    error: str | None = None               # E only: negation | number | name | question | addition | other
    tier: str | None = None
    by: str = "review"
    first: str | None = None


@dataclass(frozen=True, slots=True)
class VideoMeta:
    title: str
    channel: str = ""
    description: str = ""
    chapters: tuple[tuple[float, str], ...] = ()
    tags: tuple[str, ...] = ()
    talk_shares: tuple[tuple[str, float], ...] = ()  # (speaker id, share of talk time) from the pre-pass


@dataclass(frozen=True, slots=True)
class SpeakerNote:
    id: str
    name: str = ""
    gender: str = "unknown"                          # male | female | unknown: Telugu verb agreement
    role: str = ""
    audience: str = "polite"                         # polite (మీరు) | familiar (నువ్వు)
    address: tuple[tuple[str, str], ...] = ()        # (speaker addressed, polite | familiar)


@dataclass(frozen=True, slots=True)
class Brief:
    """What every scene call knows about the video. Version 0 is the metadata alone (no call); 1 and 2 are made by
    Claude from the transcript. It lives in the cached system prompt, so it changes only at a swap."""

    version: int
    meta: VideoMeta
    topic: str = ""
    register: str = ""
    speakers: tuple[SpeakerNote, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()
    entities: tuple[str, ...] = ()
    idioms: tuple[str, ...] = ()
    numbers: str = ""
    asr_fixes: tuple[tuple[str, str], ...] = ()      # (heard, meant), high-confidence only


class SceneTranslator(Protocol):
    """Scene-batched translation (§4). Calls may run concurrently; cancelling a task that awaits `translate` drops a
    queued call before it starts and stops one in flight. Failures the user must act on (not signed in, usage limit,
    CLI missing or too old) raise; bad output is retried and ends in `SceneResult.skipped`, never in English."""

    prompt_hash: str
    model: str
    brief: Brief

    def use_brief(self, brief: Brief) -> None:
        """Swap the brief in for the calls that start from now on."""
        ...

    async def make_brief(self, meta: VideoMeta, transcript: Sequence[tuple[str, str]],
                         previous: Brief | None = None) -> Brief: ...

    def cached(self, spec: LineSpec) -> LineResult | None:
        """The line a scene call would serve from the line cache (with its coverage class), or None."""
        ...

    async def translate(self, req: SceneRequest) -> SceneResult: ...

    def submit(self, req: SceneRequest) -> asyncio.Task[SceneResult]:
        """`translate` as a task the caller can cancel (a scene fix-up's rephrase, which a Claude failure drops)."""
        ...

    def cancel(self, match: Callable[[SceneRequest], bool] | None = None) -> int:
        """Cancel submitted requests (all, or those `match` picks); returns how many."""
        ...

    async def review(self, req: SceneRequest, lines: Mapping[int, LineResult], chosen: Mapping[int, str],
                     cache: Container[int] | None = None) -> SceneResult:
        """The coverage review of a scene's `lines` on the tiers `chosen` for them, with one re-translation of each line
        classed P or E (§4.6): the lines to use, each with its class, stored in the line cache (with `cache`, only
        those of the ids in it, each as the wording reviewed with its class). `cancel` reaches it as it reaches `req`."""
        ...

    async def aclose(self) -> None:
        """Cancel every submitted request and return once each has ended."""
        ...


class TranslatorFactory(Protocol):
    """Makes a video's scene translator: text goes through the Claude CLI on every backend (ADR-019); the mock backend's
    answers without Claude. `trace` receives its usage and per-request records (units.jsonl)."""

    def __call__(self, cache_dir: Path, video_id: str, brief: Brief, *, style: str = "colloquial",
                 trace: Callable[[dict], None] | None = None) -> SceneTranslator: ...


class Voice(Protocol):
    """Opaque, cacheable voice conditioning."""


class VoiceCloningTTS(Protocol):
    sample_rate: int
    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> Voice: ...
    def preset_voice(self, name: str) -> Voice: ...
    def synthesize(self, text: str, voice: Voice, language: str = "te", max_seconds: float | None = None) -> np.ndarray:
        """`max_seconds` caps generation (a runaway line can't burn a minute of GPU); None = the model's own limit."""
        ...


class Separator(Protocol):
    """The vocal separator under the background sound (OFFLINE-RENDER §2.14): `vocals` of a batch of `chunk`-sample
    stereo chunks at `sample_rate`; the bed is the mixture minus them. Loaded on first use; `release()` lets go of the
    model when the separate stage ends."""

    sample_rate: int  # 44,100
    chunk: int        # samples a chunk (352,800: 8 s)

    def vocals(self, batch: np.ndarray) -> np.ndarray:
        """[b, 2, chunk] float32 → [b, 2, chunk] float32, the vocals stem of each chunk."""
        ...

    def missing(self) -> list[str]:
        """The model files its folder lacks (empty: it can load). A models folder filled before the separator was
        pinned lacks them all, and a job that needs it fails at once saying how to fetch them."""
        ...

    def release(self) -> None: ...


@dataclass(slots=True)
class Backend:
    name: str
    device: str
    transcriber: Transcriber
    diarizer: Diarizer
    translator: TranslatorFactory
    tts: VoiceCloningTTS
    models_dir: Path | None = None
    separator: Separator | None = None  # None: the export has the Telugu voices only, and says so (§2.15)
