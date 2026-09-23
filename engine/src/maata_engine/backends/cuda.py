"""NVIDIA backend (secondary): faster-whisper, llama.cpp, PyTorch CUDA for diarization and TTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import TimedWord
from .base import Backend, Transcript, TranslationRequest
from .torch_common import ChatterboxTeluguTTS, PyannoteDiarizer


class FasterWhisper:
    def __init__(self, model_dir: Path) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(model_dir), device="cuda", compute_type="float16")

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        segments, info = self.model.transcribe(audio.astype(np.float32), language=language, word_timestamps=True, condition_on_previous_text=False)
        words = [TimedWord(w.word.strip(), float(w.start), float(w.end), float(w.probability)) for s in segments for w in (s.words or []) if w.word.strip()]
        return Transcript(words, info.language)


class LlamaCppTranslator:
    def __init__(self, gguf: Path, condense_gguf: Path | None = None) -> None:
        from llama_cpp import Llama

        self._Llama = Llama
        self.llm = Llama(model_path=str(gguf), n_gpu_layers=-1, n_ctx=2048, verbose=False)
        self.condense_gguf = condense_gguf
        self._condenser = None

    def translate(self, req: TranslationRequest) -> str:
        # TranslateGemma's prompt, written out (GGUF chat templates don't carry its language codes).
        prompt = (
            "<start_of_turn>user\n"
            f"You are a professional {req.source_lang} to {req.target_lang} translator. Produce only the translation.\n\n"
            f"{req.text}<end_of_turn>\n<start_of_turn>model\n"
        )
        out = self.llm(prompt, max_tokens=256, stop=["<end_of_turn>"])
        return out["choices"][0]["text"].strip()

    def condense(self, telugu: str, max_units: float, n: int = 3) -> list[str]:
        return [telugu]  # CUDA condenser lands after S3 picks the instruct model


def make_cuda_backend(models_dir: Path) -> Backend:
    m = models_dir
    return Backend(
        name="cuda",
        device="cuda",
        transcriber=FasterWhisper(m / "whisper-large-v3-turbo-ct2"),
        diarizer=PyannoteDiarizer(m / "pyannote-community-1", "cuda"),
        translator=LlamaCppTranslator(m / "translategemma-4b-it-q4_k_m.gguf"),
        tts=ChatterboxTeluguTTS(m / "chatterbox-telugu", "cuda", m / "preset-voices"),
        models_dir=m,
    )
