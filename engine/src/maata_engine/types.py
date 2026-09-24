"""Shared value types. Times are video-time seconds (float) unless noted."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float


@dataclass(slots=True)
class SourceUnit:
    """A dubbing unit built from source-language words (spec §6.2 Segmenter)."""

    id: int
    speaker: str
    start: float
    end: float
    text: str
    words: list[TimedWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceKind(str, Enum):
    CLONED = "cloned"
    PRESET = "preset"


@dataclass(slots=True)
class Speaker:
    id: str
    label: str
    voice: VoiceKind = VoiceKind.PRESET
    reference_seconds: float = 0.0
    use_preset: bool = False


@dataclass(slots=True)
class DubUnit:
    """A synthesized Telugu line ready to be placed on the timeline."""

    source: SourceUnit
    telugu: str
    samples: "object"  # numpy float32 array, mono
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)  # type: ignore[arg-type]
