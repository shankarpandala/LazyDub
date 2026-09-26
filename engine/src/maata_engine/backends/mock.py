"""A model-free backend. It exercises the whole pipeline and UI without weights (dev, CI, demos).

It never pretends to be real: the UI shows "Demo engine" whenever this backend is active.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..claude_cli import ClaudeCLIError, ClaudeReply, validate
from ..speakers import DiarBlock
from ..text.akshara import mixed_units
from ..types import SpeakerTurn, TimedWord
from .base import Backend, Brief, Transcript
from .claude_translator import ClaudeTranslator

# The demo's English lines, and how the mock translator says them: Telugu script, with the English words listed (the
# scene contract, ARCHITECTURE §4.3).
_LINES: list[tuple[str, str, list[tuple[int, str]]]] = [
    ("Welcome back to the channel.", "మళ్ళీ ఛానల్ కి స్వాగతం.", [(1, "channel")]),
    ("Today we're going to build something really fun.", "ఈ రోజు మనం ఒక సరదా ప్రాజెక్ట్ బిల్డ్ చేద్దాం.",
     [(5, "project"), (6, "build")]),
    ("First, let's look at how the timing works.", "ముందుగా టైమింగ్ ఎలా పని చేస్తుందో చూద్దాం.", [(1, "timing")]),
    ("Every line starts exactly when the speaker starts.", "ప్రతి మాట స్పీకర్ మొదలుపెట్టిన క్షణంలోనే మొదలవుతుంది.",
     [(2, "speaker")]),
    ("That's what makes a dub feel natural.", "అదే డబ్ ని సహజంగా అనిపించేలా చేస్తుంది.", [(1, "dub")]),
    ("Let me know what you think in the comments.", "మీ అభిప్రాయం కామెంట్స్ లో చెప్పండి.", [(2, "comments")]),
]


class MockTranscriber:
    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   speech: Sequence[tuple[float, float]] = ()) -> Transcript:
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
    """Two speakers taking 14 s turns on a fixed video-time grid, with stable fake embeddings."""

    _EMB = {"A": np.eye(8, dtype=np.float32)[0], "B": np.eye(8, dtype=np.float32)[1]}

    def diarize_block(self, audio: np.ndarray, offset: float) -> DiarBlock:
        end = offset + len(audio) / 16_000
        turns: list[SpeakerTurn] = []
        t = offset
        while t < end:
            k = int(t // 14.0)
            nxt = min((k + 1) * 14.0, end)
            turns.append(SpeakerTurn("A" if k % 2 == 0 else "B", t, nxt))
            t = nxt
        labels = {x.speaker for x in turns}
        return DiarBlock(offset, end, turns, list(turns), {k: v for k, v in self._EMB.items() if k in labels})


_SPOKEN = {en: (spoken, english) for en, spoken, english in _LINES}
_FAKE_WORDS = ("మాట", "కథ", "పని", "దారి", "నీళ్ళు", "ఆట", "పాట", "ఇల్లు", "చెట్టు", "వాన", "గాలి", "రోజు")


def _fake_spoken(en: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Telugu-script stand-in for any other line: one fixed word per English word, the same every run."""
    if en in _SPOKEN:
        spoken, english = _SPOKEN[en]
        return spoken.split(), english
    words = [_FAKE_WORDS[int(hashlib.sha256(w.lower().encode()).hexdigest(), 16) % len(_FAKE_WORDS)] for w in en.split()]
    return (words or [_FAKE_WORDS[0]]), []


def _wording(words: list[str], english: list[tuple[int, str]]) -> dict:
    return {"spoken": " ".join(words), "english": [{"i": i, "en": e} for i, e in english if i < len(words)]}


def _fake_line(line: dict, call: str = "scene") -> dict:
    """A reply line for one requested line: `full` (in a fit, `current` copied, as the prompt asks), the tiers it wants
    (shorter ones by dropping trailing words, `fuller` with a filler in front), pieces at its breaks, and unfinished
    when cut off."""
    words, english = _fake_spoken(line["en"])
    out: dict = {"id": line["id"], "full": _wording(words, english)}
    if call == "fit" and line.get("current"):
        out["full"] = {"spoken": line["current"], "english": []}
    n = len(words)
    for tier, share in (("concise", 0.75), ("very_concise", 0.5)):
        k = max(1, math.ceil(len(words) * share))
        if tier in line.get("want", []) and k < n:
            out[tier], n = _wording(words[:k], english), k
    if "fuller" in line.get("want", []):
        out["fuller"] = _wording(["అంటే"] + words, [(i + 1, e) for i, e in english])
    if line.get("breaks") and len(words) > 1:
        half = len(words) // 2
        out["pieces"] = [" ".join(words[:half]), " ".join(words[half:])]
    if line.get("cut_off"):
        out["unfinished"] = True
    out["delivery"] = {"emotion": "neutral", "energy": "mid", "question": "?" in line["en"]}
    return out


class MockClaude:
    """Stands in for `ClaudeCLI.ask` without Claude: answers scene, fit, re-translate, rephrase, review and brief calls
    from the message alone, deterministically (a review finds every line complete), and checks its own answer against
    the schema like the CLI does."""

    model = "mock"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[dict] = []

    def ask(self, system: str, prompt: str, schema: dict | None = None, call: str = "text", *,
            effort: str | None = None, cancel: threading.Event | None = None, tags: dict | None = None) -> ClaudeReply:
        msg = json.loads(prompt)
        self.calls.append({"call": call, "effort": effort, "message": msg, "system": system})
        if self.delay and cancel is not None and cancel.wait(self.delay):
            raise ClaudeCLIError("cancelled", "The Claude call was cancelled")
        if self.delay and cancel is None:
            time.sleep(self.delay)
        if msg.get("call") == "brief":
            speakers = [{"id": s["id"], "gender": "unknown", "audience": "polite"}
                        for s in msg.get("video", {}).get("speakers", [])]
            data: dict = {"topic": msg["video"].get("title", ""), "register": "casual", "speakers": speakers,
                          "glossary": []}
        elif msg.get("call") == "review":
            data = {"lines": [{"id": line["id"], "class": "C", "missing": [], "added": [], "error": "none"}
                              for line in msg["lines"]]}
        else:
            data = {"lines": [_fake_line(line, msg.get("call", "scene")) for line in msg["lines"]], "glossary_additions": []}
        if schema is not None and validate(data, schema):
            raise ClaudeCLIError("bad_output", "the mock's answer didn't match the schema")
        return ClaudeReply(text="", data=data, seconds=self.delay, model=self.model)


class MockSceneTranslator(ClaudeTranslator):
    """The real scene translator (validators, line cache, concurrency, cancellation) over `MockClaude`: Telugu-script
    fake output for the demo and tests, and no Claude."""

    def __init__(self, cache_dir: Path, video_id: str, brief: Brief, *, delay: float = 0.0, **kw) -> None:
        super().__init__(cache_dir, video_id, brief, cli=MockClaude(delay), **kw)


class MockTTS:
    sample_rate = 24_000

    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> dict:
        f0 = 110 + 40 * (float(np.abs(reference).mean()) * 50 % 1) if reference.size else 140
        return {"f0": f0}

    def preset_voice(self, name: str) -> dict:
        return {"f0": 170.0 if name.endswith("f") else 120.0}

    def synthesize(self, text: str, voice: dict, language: str = "te", max_seconds: float | None = None) -> np.ndarray:
        seconds = 0.15 + mixed_units(text) / 6.0  # English said in Latin script too
        if max_seconds:
            seconds = min(seconds, max_seconds)
        n = int(seconds * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        f0 = voice.get("f0", 140.0)
        # A soft, speech-like buzz with syllable-rate amplitude modulation.
        wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        env = 0.5 * (1 - np.cos(2 * np.pi * 5.0 * t)) * np.minimum(1, np.minimum(t, seconds - t) / 0.04)
        return (0.12 * wave * env).astype(np.float32)


def make_mock_backend() -> Backend:
    return Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, MockTTS())
