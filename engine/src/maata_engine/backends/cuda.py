"""NVIDIA backend (secondary): faster-whisper, PyTorch CUDA for diarization and TTS. Text goes through the Claude CLI
(ADR-019): no local LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import TimedWord
from .base import Backend, Transcript
from .claude_translator import ClaudeTranslator
from .torch_common import ChatterboxTeluguTTS, make_diarizer


class FasterWhisper:
    def __init__(self, model_dir: Path) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(model_dir), device="cuda", compute_type="float16")

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   speech: Sequence[tuple[float, float]] = ()) -> Transcript:
        # The punctuation pass and the coverage re-decode are Apple-only so far (ARCHITECTURE §3.4); `speech` is unused.
        segments, info = self.model.transcribe(audio.astype(np.float32), language=language, word_timestamps=True, condition_on_previous_text=False)
        words = [TimedWord(w.word.strip(), float(w.start), float(w.end), float(w.probability)) for s in segments for w in (s.words or []) if w.word.strip()]
        return Transcript(words, info.language)


def make_cuda_backend(models_dir: Path) -> Backend:
    m = models_dir
    return Backend(
        name="cuda",
        device="cuda",
        transcriber=FasterWhisper(m / "whisper-large-v3-turbo-ct2"),
        diarizer=make_diarizer(m / "pyannote-community-1", "cuda"),
        translator=ClaudeTranslator,
        tts=ChatterboxTeluguTTS(m / "chatterbox-telugu", "cuda", m / "preset-voices"),
        models_dir=m,
    )
