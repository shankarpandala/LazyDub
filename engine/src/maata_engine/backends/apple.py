"""Apple Silicon backend (primary, ADR-003): MLX for Whisper and LLMs, PyTorch MPS for diarization and TTS."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..text.akshara import count_units
from ..types import TimedWord
from .base import Backend, Transcript, TranslationRequest
from .torch_common import ChatterboxTeluguTTS, PyannoteDiarizer


class MLXWhisper:
    def __init__(self, model_dir: Path) -> None:
        import mlx_whisper

        self._mw = mlx_whisper
        self.model_dir = str(model_dir)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        out = self._mw.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.model_dir, word_timestamps=True,
            language=language, condition_on_previous_text=False, verbose=None,
        )
        words = [
            TimedWord(w["word"].strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
            for seg in out.get("segments", []) for w in seg.get("words", []) if w["word"].strip()
        ]
        return Transcript(words, out.get("language"))


_JSON = re.compile(r"\{.*\}", re.S)


class MLXTranslator:
    """TranslateGemma for translation (its strict template), a small instruct model for condensing."""

    def __init__(self, translate_dir: Path, condense_dir: Path | None = None) -> None:
        from mlx_lm import generate, load

        self._generate = generate
        self._load = load
        self.model, self.tok = load(str(translate_dir))
        self._fix_translategemma(self.model)
        self.condense_dir = condense_dir
        self._condenser = None

    @staticmethod
    def _fix_translategemma(model) -> None:
        # TranslateGemma stores its ×8 linear RoPE scaling in `rope_parameters`, which loaders that
        # read `rope_scaling` miss (research survey §3). Verified on the reference Mac in S3.
        args = getattr(model, "args", None)
        if args is not None and getattr(args, "rope_scaling", None) is None:
            rp = getattr(args, "rope_parameters", None) or {}
            fa = rp.get("full_attention") if isinstance(rp, dict) else None
            if fa and fa.get("rope_type") == "linear":
                args.rope_scaling = {"type": "linear", "factor": fa.get("factor", 8.0)}

    def translate(self, req: TranslationRequest) -> str:
        context = "".join(f"{s}\n" for s, _ in req.context[-3:])
        text = (context + req.text) if context else req.text
        messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": req.source_lang, "target_lang_code": req.target_lang, "text": text}]}]
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = self._generate(self.model, self.tok, prompt=prompt, max_tokens=256, verbose=False)
        out = out.split("<end_of_turn>")[0].strip()
        return out.splitlines()[-1].strip() if context and "\n" in out else out

    def condense(self, telugu: str, max_units: float, n: int = 3) -> list[str]:
        if self._condenser is None:
            if not self.condense_dir:
                return [telugu]
            self._condenser = self._load(str(self.condense_dir))
        model, tok = self._condenser
        instruction = (
            "Rewrite this spoken Telugu line shorter, keeping its meaning and any English technical terms. "
            f"Each rewrite must be at most {int(max_units)} aksharas. Reply with JSON only: "
            f'{{"candidates": [{", ".join(["\"...\""] * n)}]}}\n\nLine: {telugu}'
        )
        prompt = tok.apply_chat_template([{"role": "user", "content": instruction}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        raw = self._generate(model, tok, prompt=prompt, max_tokens=400, verbose=False)
        try:
            cands = json.loads(_JSON.search(raw).group(0))["candidates"]  # type: ignore[union-attr]
        except Exception:
            return [telugu]
        cands = [c.strip() for c in cands if isinstance(c, str) and c.strip()]
        return sorted(cands, key=lambda c: abs(count_units(c) - max_units)) or [telugu]


def make_apple_backend(models_dir: Path) -> Backend:
    m = models_dir
    return Backend(
        name="apple",
        device="mps",
        transcriber=MLXWhisper(m / "whisper-large-v3-turbo-mlx"),
        diarizer=PyannoteDiarizer(m / "pyannote-community-1", "mps"),
        translator=MLXTranslator(m / "translategemma-4b-it-4bit-mlx", m / "qwen3-4b-instruct-4bit-mlx"),
        tts=ChatterboxTeluguTTS(m / "chatterbox-telugu", "mps", m / "preset-voices"),
        models_dir=m,
    )
