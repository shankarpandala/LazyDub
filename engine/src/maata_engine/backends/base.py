"""Engine-stage protocols (spec §5). Every model sits behind one of these so engines can be swapped."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..types import SpeakerTurn, TimedWord

SR_ANALYSIS = 16_000


@dataclass(slots=True)
class Transcript:
    words: list[TimedWord]
    language: str | None


@dataclass(slots=True)
class TranslationRequest:
    text: str
    source_lang: str
    target_lang: str = "te"
    context: list[tuple[str, str]] = field(default_factory=list)  # (source, telugu) of previous units
    glossary: dict[str, str] = field(default_factory=dict)
    target_units: float | None = None
    style: str = "colloquial"


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript: ...


class Diarizer(Protocol):
    def diarize(self, audio: np.ndarray) -> list[SpeakerTurn]: ...


class Translator(Protocol):
    def translate(self, req: TranslationRequest) -> str: ...
    def condense(self, telugu: str, max_units: float, n: int = 3) -> list[str]: ...


class Voice(Protocol):
    """Opaque, cacheable voice conditioning."""


class VoiceCloningTTS(Protocol):
    sample_rate: int
    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> Voice: ...
    def preset_voice(self, name: str) -> Voice: ...
    def synthesize(self, text: str, voice: Voice, language: str = "te") -> np.ndarray: ...


@dataclass(slots=True)
class Backend:
    name: str
    device: str
    transcriber: Transcriber
    diarizer: Diarizer
    translator: Translator
    tts: VoiceCloningTTS
    models_dir: Path | None = None
