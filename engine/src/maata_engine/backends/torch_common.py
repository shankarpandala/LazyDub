"""PyTorch stages shared by the Apple (MPS) and CUDA backends: diarization and Chatterbox-Telugu TTS.

Imports are lazy so the engine starts (and tests run) without torch installed.
Models load only from local directories; the engine sets HF_HUB_OFFLINE=1 before import.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import SpeakerTurn
from .base import SR_ANALYSIS


class PyannoteDiarizer:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self._torch = torch
        self.pipeline = Pipeline.from_pretrained(str(model_dir))
        self.pipeline.to(torch.device(device))

    def diarize(self, audio: np.ndarray) -> list[SpeakerTurn]:
        wav = self._torch.from_numpy(audio).unsqueeze(0)
        out = self.pipeline({"waveform": wav, "sample_rate": SR_ANALYSIS})
        ann = getattr(out, "speaker_diarization", out)  # pyannote 4 returns DiarizeOutput
        return [SpeakerTurn(str(spk), float(seg.start), float(seg.end)) for seg, _, spk in ann.itertracks(yield_label=True)]


@dataclass
class ChatterboxVoice:
    conds: object


class ChatterboxTeluguTTS:
    """chatterbox-telugu through the maintainer's patched Chatterbox-Multilingual pipeline.

    Voice conditioning (`prepare_conditionals`) is computed once per speaker and cached;
    the PerTh watermark the Python pipeline applies is kept.
    """

    def __init__(self, ckpt_dir: Path, device: str, presets_dir: Path | None = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._torch = torch
        self.model = ChatterboxMultilingualTTS.from_local(str(ckpt_dir), device)
        self.sample_rate = int(self.model.sr)
        self.presets_dir = presets_dir
        self.exaggeration, self.cfg_weight = exaggeration, cfg_weight

    def _conds_from_wav(self, audio: np.ndarray, sr: int) -> ChatterboxVoice:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio.astype(np.float32), sr)
            self.model.prepare_conditionals(f.name, exaggeration=self.exaggeration)
        return ChatterboxVoice(self.model.conds)

    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> ChatterboxVoice:
        return self._conds_from_wav(reference, sample_rate)

    def preset_voice(self, name: str) -> ChatterboxVoice:
        if not self.presets_dir:
            raise FileNotFoundError("No preset voices installed")
        import soundfile as sf

        audio, sr = sf.read(str(self.presets_dir / f"{name}.wav"), dtype="float32")
        return self._conds_from_wav(audio, sr)

    def synthesize(self, text: str, voice: ChatterboxVoice, language: str = "te") -> np.ndarray:
        self.model.conds = voice.conds
        with self._torch.inference_mode():
            wav = self.model.generate(text, language_id=language, exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
        return wav.squeeze(0).float().cpu().numpy()
