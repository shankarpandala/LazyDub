"""PyTorch stages shared by the Apple (MPS) and CUDA backends: diarization and Chatterbox-Telugu TTS.

Imports are lazy so the engine starts (and tests run) without torch installed.
Models load only from local directories; the engine sets HF_HUB_OFFLINE=1 before import.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..speakers import DiarBlock
from ..types import SpeakerTurn
from .base import SR_ANALYSIS, Diarizer

log = logging.getLogger("maata.backends")

S3_TOKEN_RATE = 25  # Chatterbox speech tokens per second of audio
EOS_CHECK = 8       # T3 decode steps between checks that every take has ended: each check waits for the GPU (§3.8)


class SingleSpeakerDiarizer:
    """Everyone is speaker S1, one voice for the whole video (spec §12 Phase 1: no diarization).

    Used while pyannote's gated model isn't installed. One turn spans the block, so every word keeps
    the same voice; with no centroid the registry links blocks by their overlap.
    """

    def diarize_block(self, audio: np.ndarray, offset: float) -> DiarBlock:
        end = offset + len(audio) / SR_ANALYSIS
        turns = [SpeakerTurn("S1", offset, end)] if len(audio) else []
        return DiarBlock(offset, end, turns, list(turns), {})


def make_diarizer(model_dir: Path, device: str) -> Diarizer:
    if all((model_dir / f).is_file() for f in ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin")):
        return PyannoteDiarizer(model_dir, device)
    log.warning("Speaker diarization model not installed (%s); dubbing everyone with one voice.", model_dir.name)
    return SingleSpeakerDiarizer()


class PyannoteDiarizer:
    """pyannote community-1. Keeps what the speaker registry needs: exclusive turns (one speaker per
    instant, for assigning words) and per-speaker centroid embeddings (for linking blocks)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self._torch = torch
        self.pipeline = Pipeline.from_pretrained(str(model_dir))
        self.pipeline.to(torch.device(device))

    def diarize_block(self, audio: np.ndarray, offset: float) -> DiarBlock:
        end = offset + len(audio) / SR_ANALYSIS
        if len(audio) < SR_ANALYSIS:  # under a second: nothing to diarize
            return DiarBlock(offset, end, [], [], {})
        out = self.pipeline({"waveform": self._torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0), "sample_rate": SR_ANALYSIS})
        ann = out.speaker_diarization
        excl = getattr(out, "exclusive_speaker_diarization", None) or ann

        def shifted(a) -> list[SpeakerTurn]:
            return [SpeakerTurn(str(spk), offset + float(seg.start), offset + float(seg.end)) for seg, _, spk in a.itertracks(yield_label=True)]

        # speaker_embeddings[i] belongs to ann.labels()[i]; speakers without a cluster get all-zero rows.
        centroids: dict[str, np.ndarray] = {}
        emb = getattr(out, "speaker_embeddings", None)
        if emb is not None:
            for label, row in zip(ann.labels(), np.asarray(emb, dtype=np.float32)):
                if np.isfinite(row).all() and np.linalg.norm(row) > 1e-6:
                    centroids[str(label)] = row
        return DiarBlock(offset, end, shifted(ann), shifted(excl), centroids)


@dataclass
class ChatterboxVoice:
    conds: object
    cfg_weight: float | None = None  # per-voice guidance; None = the model default


@dataclass
class MelTake:
    """A synthesized line before vocoding: its natural duration is known from its speech tokens, and the planner picks the
    rate. S3Gen's flow runs when it is vocoded, so of a line's takes only the one voiced pays for it. The timings go into
    units.jsonl (ARCHITECTURE §7 step 0)."""

    mel: object      # torch (1, 80, frames) at 50 frames/s, once flowed
    n_tokens: int
    t3_s: float = 0.0     # T3: prefill plus the autoregressive decode (a batched decode's time, shared out over its takes)
    t3_tokens: int = 0    # speech tokens sampled for this take, before invalid ones are dropped
    t3_steps: int = 0     # decode steps of its batch, shared by every take in it (a step samples a token for each)
    flow_s: float = 0.0   # S3Gen flow matching, once vocoded
    cfm_steps: int = 0
    capped: bool = False  # ran to its token cap without an end token: a failed take (ARCHITECTURE §3.8)
    speech: object = None  # its speech tokens, until flowed
    gen: object = None     # the voice's S3Gen reference, for the flow

    @property
    def seconds(self) -> float:
        return max(self.n_tokens - 1, 0) / S3_TOKEN_RATE  # the fork drops the final token's audio


def _loudness_normalise(wav: np.ndarray, sr: int, target_dbfs: float = -20.0) -> np.ndarray:
    """Trim leading/trailing silence and bring a reference clip to a common loudness."""
    import librosa

    wav = np.asarray(wav, np.float32)
    trimmed, _ = librosa.effects.trim(wav, top_db=35)
    if trimmed.size > sr // 2:
        wav = trimmed
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    if rms > 1e-5:
        wav = wav * (10 ** (target_dbfs / 20) / rms)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return wav * (0.95 / peak) if peak > 0.95 else wav


class ChatterboxTeluguTTS:
    """chatterbox-telugu through the maintainer's patched Chatterbox-Multilingual pipeline.

    Voice conditioning (`prepare_conditionals`) is computed once per speaker and cached; the PerTh
    watermark the Python pipeline applies is kept. `synthesize` mirrors the fork's `generate()` with
    three speed changes measured on the M5 Pro (docs/DECISIONS.md ADR-014): the T3 transformer runs in
    bf16 (sampling stays fp32), generation is capped by the line's time budget instead of a fixed
    1000 tokens (40 s), and the S3Gen flow-matching step count is configurable.
    """

    def __init__(self, ckpt_dir: Path, device: str, presets_dir: Path | None = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5, t3_bf16: bool = True, cfm_steps: int = 10) -> None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._torch = torch
        # The checkpoint names its fine-tuned T3 weights (t3_mtl_te.safetensors); the loader's
        # default is the base multilingual model's file, which this checkpoint doesn't ship.
        cfg = ckpt_dir / "config.json"
        t3_model = json.loads(cfg.read_text()).get("t3_model") if cfg.is_file() else None
        self.model = ChatterboxMultilingualTTS.from_local(str(ckpt_dir), device, t3_model=t3_model)
        self.sample_rate = int(self.model.sr)
        self.builtin = self.model.conds  # conds.pt: the checkpoint's own voice, the preset of last resort
        self.presets_dir = presets_dir
        self._presets: dict[str, ChatterboxVoice] = {}
        self.exaggeration, self.cfg_weight = exaggeration, cfg_weight
        self.cfm_steps = cfm_steps
        self._t3_dtype = torch.bfloat16 if t3_bf16 and device in ("mps", "cuda") else None
        if self._t3_dtype is not None:
            self.model.t3.tfmr.to(self._t3_dtype)
        self._backend = None

    def _conds_from_wav(self, audio: np.ndarray, sr: int) -> ChatterboxVoice:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio.astype(np.float32), sr)
            self.model.prepare_conditionals(f.name, exaggeration=self.exaggeration)
        return ChatterboxVoice(self.model.conds)

    def prepare_voice(self, reference: np.ndarray, sample_rate: int) -> ChatterboxVoice:
        return self._conds_from_wav(reference, sample_rate)

    def prepare_voice_parts(self, timbre: np.ndarray, embed_clips: list[np.ndarray], prompt: np.ndarray | None,
                            sample_rate: int = SR_ANALYSIS, cfg_weight: float | None = None) -> ChatterboxVoice:
        """Build a voice from separate parts (ADR-017), instead of one stitched reference:

        - `timbre`: one contiguous clean span of the speaker (<= 10 s) conditions S3Gen, which sets the timbre;
        - `embed_clips`: many clean clips of the speaker, averaged into T3's speaker embedding (identity);
        - `prompt`: the 6 s of speech T3 is prompted with, which carries pace, prosody and accent. The speaker's own
          English sounds closest to them; a native Telugu clip gives more natural Telugu delivery.
        All inputs are mono float audio at `sample_rate` (16 kHz analysis audio in the engine).
        """
        import librosa
        import torch
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.mtl_tts import Conditionals

        m = self.model
        sr16 = 16_000

        def to16(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, np.float32)
            return x if sample_rate == sr16 else librosa.resample(x, orig_sr=sample_rate, target_sr=sr16)

        timbre16 = _loudness_normalise(to16(timbre), sr16)[: 10 * sr16]
        prompt16 = _loudness_normalise(to16(prompt), sr16) if prompt is not None and len(prompt) else timbre16
        clips16 = [_loudness_normalise(to16(c), sr16) for c in embed_clips if len(c) >= sr16] or [timbre16]
        with torch.inference_mode():
            gen = m.s3gen.embed_ref(torch.from_numpy(timbre16), sr16, device=m.device)
            plen = m.t3.hp.speech_cond_prompt_len
            tokens, _ = m.s3gen.tokenizer.forward([prompt16[: 6 * sr16]], max_len=plen) if plen else (None, None)
            tokens = torch.atleast_2d(tokens).to(m.device) if tokens is not None else None
            emb = torch.from_numpy(m.ve.embeds_from_wavs(clips16, sample_rate=sr16)).mean(axis=0, keepdim=True).to(m.device)
            t3 = T3Cond(speaker_emb=emb, cond_prompt_speech_tokens=tokens,
                        emotion_adv=self.exaggeration * torch.ones(1, 1, 1)).to(device=m.device)
        return ChatterboxVoice(Conditionals(t3, gen), cfg_weight)

    def preset_voice(self, name: str) -> ChatterboxVoice:
        if name not in self._presets:  # computed once: the session asks for a preset every unit
            wav = self.presets_dir / f"{name}.wav" if self.presets_dir else None
            if wav is not None and wav.is_file():
                import soundfile as sf

                audio, sr = sf.read(str(wav), dtype="float32")
                self._presets[name] = self._conds_from_wav(audio, sr)
            elif self.builtin is not None:
                self._presets[name] = ChatterboxVoice(self.builtin)
            else:
                raise FileNotFoundError(f"No preset voice '{name}' and the checkpoint has no built-in voice")
        return self._presets[name]

    def _t3_tokens(self, text_tokens, max_new_tokens: int, cfg_weight: float, n: int = 1, seeds: list[int] | None = None,
                   temperature: float = 0.8, repetition_penalty: float = 1.2, min_p: float = 0.05):
        """The fork's T3 sampling loop (t3.py inference) with a bf16 transformer and a token cap, for `n` takes of one line
        in one batched decode (ARCHITECTURE §3.8): 2n rows, the n conditional ones then their n unconditional twins, so n
        takes cost about as much as one (the decode is bound by per-step overhead, not by the batch). Whether every take
        has sampled its end token is read back every EOS_CHECK steps, not every step (each read waits for the GPU); each
        take is cut at its first one. Returns, per take, its tokens and whether it ran to the cap without ending (a
        failed take), and the decode steps run (up to the check after the last end, so a few past it: what the decode's
        time is spent on). `seeds`: one per take, each sampled from its own generator, so a take comes out the same
        batched or alone; without them the batch samples from torch's global generator."""
        import torch
        from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor

        t3 = self.model.t3
        if self._backend is None:  # built once, not per line as in the fork
            from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

            self._backend = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
        cast = (lambda x: x.to(self._t3_dtype)) if self._t3_dtype is not None else (lambda x: x)
        start = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=self.model.conds.t3, text_tokens=text_tokens, speech_tokens=start, cfg_weight=cfg_weight)
        bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=embeds.device)
        bos_e = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
        prompt = torch.cat([embeds, torch.cat([bos_e, bos_e])], dim=1).repeat_interleave(n, dim=0)  # c * n, then u * n
        out = self._backend(inputs_embeds=cast(prompt), past_key_values=None, use_cache=True, output_hidden_states=True,
                            return_dict=True)
        min_p_warper, rep = MinPLogitsWarper(min_p=min_p), RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        eos = t3.hp.stop_speech_token
        gens = [torch.Generator(device=embeds.device).manual_seed(int(s)) for s in seeds] if seeds else None
        generated = bos.expand(n, 1).clone()
        ended = torch.zeros(n, dtype=torch.bool, device=embeds.device)
        for i in range(max_new_tokens):
            logits = out.logits[:, -1, :].float()
            cond, uncond = logits[:n], logits[n:]
            logits = cond + cfg_weight * (cond - uncond)  # classifier-free guidance
            probs = torch.softmax(min_p_warper(generated, rep(generated, logits) / temperature), dim=-1)
            nxt = (torch.cat([torch.multinomial(probs[k:k + 1], 1, generator=g) for k, g in enumerate(gens)])
                   if gens else torch.multinomial(probs, num_samples=1))
            generated = torch.cat([generated, nxt], dim=1)
            ended |= nxt[:, 0] == eos
            if (i + 1) % EOS_CHECK == 0 and bool(ended.all()):
                break
            if i + 1 == max_new_tokens:
                break  # no forward pass for a token that won't be sampled
            e = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            out = self._backend(inputs_embeds=cast(torch.cat([e, e])), past_key_values=out.past_key_values,
                                output_hidden_states=True, return_dict=True)
        takes = []
        for row, done in zip(generated[:, 1:], ended.tolist()):
            if done:
                takes.append((row[: int((row == eos).nonzero()[0])], False))
            else:
                log.info("a TTS take hit its %d-token cap", max_new_tokens)
                takes.append((row, True))
        return takes, generated.shape[1] - 1

    def synthesize_takes(self, text: str, voice: ChatterboxVoice, language: str = "te", max_seconds: float | None = None,
                         n: int = 1, seeds: list[int] | None = None) -> list[MelTake]:
        """`n` takes of a line from one batched T3 decode (ARCHITECTURE §3.8), each at the voice's natural pace. The one
        voiced is flowed and vocoded by `vocode`, at the rate the planner picks."""
        import torch
        import torch.nn.functional as F
        from chatterbox.models.s3tokenizer import drop_invalid_tokens
        from chatterbox.mtl_tts import punc_norm

        m = self.model
        m.conds = voice.conds
        cfg = self.cfg_weight if voice.cfg_weight is None else voice.cfg_weight
        max_new = min(1000, math.ceil(S3_TOKEN_RATE * max_seconds) + 10) if max_seconds else 1000
        with torch.inference_mode():
            tt = m.tokenizer.text_to_tokens(punc_norm(text), language_id=language).to(m.device)
            tt = torch.cat([tt, tt], dim=0)  # two sequences for CFG
            tt = F.pad(F.pad(tt, (1, 0), value=m.t3.hp.start_text_token), (0, 1), value=m.t3.hp.stop_text_token)
            t0 = time.perf_counter()
            sampled, steps = self._t3_tokens(tt, max_new, cfg, n, seeds)  # already synced: the ends were read back
            t3_s = (time.perf_counter() - t0) / len(sampled)
            takes = []
            for tokens, capped in sampled:
                speech = drop_invalid_tokens(tokens).to(m.device)
                takes.append(MelTake(None, int(speech.shape[-1]), t3_s, int(tokens.numel()), steps, 0.0, self.cfm_steps,
                                     capped, speech if speech.numel() else None, voice.conds.gen))
        return takes

    def synthesize_mel(self, text: str, voice: ChatterboxVoice, language: str = "te", max_seconds: float | None = None) -> MelTake:
        """One take of the line at its natural pace (T3 only). Vocode it with `vocode` at the rate the planner picks."""
        return self.synthesize_takes(text, voice, language, max_seconds)[0]

    def _flow(self, take: MelTake) -> None:
        """S3Gen flow matching: the take's speech tokens -> its mel, timed."""
        import torch

        if take.mel is not None or take.speech is None:
            return
        with torch.inference_mode():
            t0 = time.perf_counter()
            take.mel = self.model.s3gen.flow_inference(take.speech, ref_dict=take.gen, n_cfm_timesteps=self.cfm_steps,
                                                       finalize=True)
            self._sync()
            take.flow_s = time.perf_counter() - t0
        take.speech = None

    def _sync(self) -> None:
        """Wait for queued GPU work, so a stage's time is its own (MPS and CUDA run asynchronously). The vocoder would wait
        for it anyway, so this costs nothing."""
        dev = str(self.model.device)
        if dev.startswith("mps"):
            self._torch.mps.synchronize()
        elif dev.startswith("cuda"):
            self._torch.cuda.synchronize()

    def vocode(self, take: MelTake, rate: float = 1.0) -> np.ndarray:
        """Mel -> audio. `rate` > 1 plays the line faster without changing pitch: the mel is resampled in time before the
        vocoder (CosyVoice's speed control; the fork leaves it as a TODO in s3gen.py). Kept gentle (<= ~1.2x) by the planner."""
        import torch
        import torch.nn.functional as F

        self._flow(take)
        if take.mel is None or take.n_tokens == 0:
            return np.zeros(0, np.float32)
        s3 = self.model.s3gen
        with torch.inference_mode():
            mel = take.mel
            if abs(rate - 1.0) > 1e-3:
                frames = max(1, round(mel.shape[-1] / rate))
                mel = F.interpolate(mel.float(), size=frames, mode="linear", align_corners=False)
            wav, _ = s3.hift_inference(mel.to(dtype=s3.dtype), None)
            wav[:, : len(s3.trim_fade)] *= s3.trim_fade  # as the fork does: soften spill-over from the reference
            wav = wav.squeeze(0).float().cpu().numpy()
        wav = wav[: round(max(1, take.n_tokens - 1) * (self.sample_rate // S3_TOKEN_RATE) / rate)]
        return np.asarray(self.model.watermarker.apply_watermark(wav, sample_rate=self.sample_rate), np.float32)

    def synthesize(self, text: str, voice: ChatterboxVoice, language: str = "te", max_seconds: float | None = None) -> np.ndarray:
        return self.vocode(self.synthesize_mel(text, voice, language, max_seconds), 1.0)
