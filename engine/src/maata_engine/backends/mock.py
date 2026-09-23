"""A model-free backend. It exercises the whole pipeline and UI without weights (dev, CI, demos).

It never pretends to be real: the UI shows "Demo engine" whenever this backend is active.
"""

from __future__ import annotations

import numpy as np

from ..text.akshara import count_units
from ..types import SpeakerTurn, TimedWord
from .base import Backend, Transcript, TranslationRequest

_LINES = [
    ("Welcome back to the channel.", "మళ్ళీ ఛానల్ కి స్వాగతం."),
    ("Today we're going to build something really fun.", "ఈ రోజు మనం ఒక సరదా project build చేద్దాం."),
    ("First, let's look at how the timing works.", "ముందుగా timing ఎలా పని చేస్తుందో చూద్దాం."),
    ("Every line starts exactly when the speaker starts.", "ప్రతి మాట speaker మొదలుపెట్టిన క్షణంలోనే మొదలవుతుంది."),
    ("That's what makes a dub feel natural.", "అదే dub ని సహజంగా అనిపించేలా చేస్తుంది."),
    ("Let me know what you think in the comments.", "మీ అభిప్రాయం comments లో చెప్పండి."),
]


class MockTranscriber:
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        seconds = len(audio) / 16_000
        words: list[TimedWord] = []
        t, i = 0.6, 0
        while t < seconds - 2:
            src = _LINES[i % len(_LINES)][0].split()
            for wd in src:
                d = 0.18 + 0.04 * len(wd)
                words.append(TimedWord(wd, round(t, 3), round(t + d, 3)))
                t += d + 0.06
            t += 0.9
            i += 1
        return Transcript(words, language or "en")


class MockDiarizer:
    def diarize(self, audio: np.ndarray) -> list[SpeakerTurn]:
        seconds = len(audio) / 16_000
        turns, t, spk = [], 0.0, 0
        while t < seconds:
            turns.append(SpeakerTurn(f"S{spk + 1}", t, min(t + 14.0, seconds)))
            t += 14.0
            spk ^= 1
        return turns


class MockTranslator:
    def translate(self, req: TranslationRequest) -> str:
        for en, te in _LINES:
            if req.text.strip().startswith(en.split()[0]):
                return te
        return _LINES[hash(req.text) % len(_LINES)][1]

    def condense(self, telugu: str, max_units: float, n: int = 3) -> list[str]:
        words = telugu.split()
        out = []
        for k in range(n):
            cut = words[: max(1, len(words) - (k + 1))]
            out.append(" ".join(cut))
        return [c for c in out if count_units(c) <= max_units] or out[-1:]


class MockTTS:
    sample_rate = 24_000

    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> dict:
        f0 = 110 + 40 * (float(np.abs(reference).mean()) * 50 % 1) if reference.size else 140
        return {"f0": f0}

    def preset_voice(self, name: str) -> dict:
        return {"f0": 170.0 if name.endswith("f") else 120.0}

    def synthesize(self, text: str, voice: dict, language: str = "te") -> np.ndarray:
        seconds = 0.15 + count_units(text) / 6.0
        n = int(seconds * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        f0 = voice.get("f0", 140.0)
        # A soft, speech-like buzz with syllable-rate amplitude modulation.
        wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        env = 0.5 * (1 - np.cos(2 * np.pi * 5.0 * t)) * np.minimum(1, np.minimum(t, seconds - t) / 0.04)
        return (0.12 * wave * env).astype(np.float32)


def make_mock_backend() -> Backend:
    return Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockTranslator(), MockTTS())
