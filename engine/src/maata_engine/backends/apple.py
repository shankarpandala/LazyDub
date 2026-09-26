"""Apple Silicon backend (primary, ADR-003): MLX for Whisper, PyTorch MPS for diarization and TTS. Text goes through the
Claude CLI (ADR-019): no local LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..text import asr_guard
from ..text.punct_transfer import transfer, unguess
from ..types import TimedWord
from .base import SR_ANALYSIS, Backend, Transcript
from .claude_translator import ClaudeTranslator
from .torch_common import ChatterboxTeluguTTS, make_diarizer

# The punctuation pass (ARCHITECTURE §3.4): Whisper copies a prompt's style, so a pass prompted with this (original)
# punctuated passage punctuates and capitalises. Only that style is kept; its words never are (a prompted pass drops
# whole passages of speech: docs/research/dubbing-2026-09/gap-3.md E2).
STYLE_PROMPT = ("Okay, so here's the plan for today. First, we'll look at the numbers; then, we'll talk about what "
                "they mean. Sound good? Great. Let's begin.")
STYLE_WINDOW = 30.0          # s: with condition_on_previous_text off, a prompt reaches only a call's first 30 s window
STYLE_EDGE = 1.0             # s: a window cut mid-speech ends (and the next begins) with Whisper's guess at a sentence
                             # edge: the last word's marks and the first word's capital this close to a cut lend nothing
HALLUCINATION_SILENCE = 2.0  # s: skip silence before a possible hallucination (report 10 R4; needs word timestamps)
REDECODE_PAD = 0.3           # s of audio each side of a stretch of speech that is decoded again


class MLXWhisper:
    def __init__(self, model_dir: Path) -> None:
        import mlx_whisper

        self._mw = mlx_whisper
        self.model_dir = str(model_dir)

    def _run(self, audio: np.ndarray, language: str | None, offset: float = 0.0,
             **kw: object) -> tuple[list[TimedWord], str | None, list[float]]:
        """One mlx-whisper call: its words (times shifted by `offset` s), the language, and for each word its segment's
        no-speech probability."""
        out = self._mw.transcribe(
            audio, path_or_hf_repo=self.model_dir, word_timestamps=True, language=language,
            condition_on_previous_text=False, verbose=None, hallucination_silence_threshold=HALLUCINATION_SILENCE, **kw,
        )
        got = [
            (TimedWord(w["word"].strip(), float(w["start"]) + offset, float(w["end"]) + offset,
                       float(w.get("probability", 1.0))), float(seg.get("no_speech_prob", 0.0)))
            for seg in out.get("segments", []) for w in seg.get("words", []) if w["word"].strip()
        ]
        return [w for w, _ in got], out.get("language"), [q for _, q in got]

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   speech: Sequence[tuple[float, float]] = ()) -> Transcript:
        """Whisper's words and timestamps, with the guards of ARCHITECTURE §3.4:
        - diarized `speech` that got no words for more than 0.8 s is decoded again on its own, and the words found
          inside it are added if they pass asr_guard.vet_redecode (one call per stretch: mlx-whisper 0.4.3 doesn't
          restart at each of several clips);
        - for English, a second pass per 30 s window, prompted with STYLE_PROMPT, lends its casing and trailing marks to
          these words by alignment (text/punct_transfer.py), except its guesses at the cuts between windows. No word is
          taken from it."""
        audio = audio.astype(np.float32)
        dur = len(audio) / SR_ANALYSIS
        words, language, _ = self._run(audio, language)
        gaps = asr_guard.uncovered(words, list(speech))
        found: list[TimedWord] = []
        rejected = 0
        for a, b in gaps:
            clip = [max(a - REDECODE_PAD, 0.0), min(b + REDECODE_PAD, dur)]
            again, _, quiet = self._run(audio, language, clip_timestamps=clip)
            inside = [k for k, w in enumerate(again) if a <= (w.start + w.end) / 2 <= b]
            kept, dropped = asr_guard.vet_redecode([again[k] for k in inside], [quiet[k] for k in inside], words,
                                                   (a, b), REDECODE_PAD)
            found += kept
            rejected += len(dropped)
        words = sorted(words + found, key=lambda w: (w.start, w.end))
        punctuated = language == "en" and bool(words)
        guesses = 0
        if punctuated:
            step = int(STYLE_WINDOW * SR_ANALYSIS)
            styled: list[TimedWord] = []
            for k in range(0, len(audio), step):
                window = audio[k:k + step]
                if len(window) < SR_ANALYSIS:  # a last sliver under a second has nothing to lend
                    continue
                t0, t1 = k / SR_ANALYSIS, (k + len(window)) / SR_ANALYSIS
                got, _, _ = self._run(window, language, t0, initial_prompt=STYLE_PROMPT)
                # Only cuts inside the audio are guesses; the audio's own start and end are the unprompted pass's too.
                got, n = unguess(got, t0 if k > 0 else None, t1 if k + step < len(audio) else None, STYLE_EDGE)
                styled += got
                guesses += n
            words = transfer(words, styled)
        return Transcript(words, language, gaps=gaps, recovered=len(found), rejected=rejected, punctuated=punctuated,
                          edge_guesses=guesses)


def make_apple_backend(models_dir: Path) -> Backend:
    m = models_dir
    # MAATA_CFM_STEPS: a dev override for the S3Gen flow-matching steps (ADR-014).
    tts = ChatterboxTeluguTTS(m / "chatterbox-telugu", "mps", m / "preset-voices", cfm_steps=int(os.environ.get("MAATA_CFM_STEPS") or 10))
    return Backend(
        name="apple",
        device="mps",
        transcriber=MLXWhisper(m / "whisper-large-v3-turbo-mlx"),
        diarizer=make_diarizer(m / "pyannote-community-1", "mps"),
        translator=ClaudeTranslator,
        tts=tts,
        models_dir=m,
    )
