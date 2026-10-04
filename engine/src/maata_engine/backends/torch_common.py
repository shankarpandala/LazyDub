"""PyTorch stages shared by the Apple (MPS) and CUDA backends: diarization and Chatterbox-Telugu TTS.

Imports are lazy so the engine starts (and tests run) without torch installed.
Models load only from local directories; the engine sets HF_HUB_OFFLINE=1 before import.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..speakers import DiarBlock
from ..types import SpeakerTurn
from .base import COARSE_STEP, SR_ANALYSIS, Cancelled, Diarizer

log = logging.getLogger("maata.backends")

S3_TOKEN_RATE = 25  # Chatterbox speech tokens per second of audio
EOS_CHECK = 8       # T3 decode steps between checks that every take has ended: each check waits for the GPU (§3.8)
# Whole-file diarization's memory guard (OFFLINE-RENDER §2.3, §7): pyannote's VBx runs scipy's centroid linkage over every
# embedding it keeps, which peaks at about 2.1x the condensed distance matrix (measured on scipy 1.18.1): 8.4 n^2 bytes.
DIAR_CLUSTER_BUDGET = 3e9  # bytes the clustering may take
LINKAGE_BYTES = 8.4
MIN_ACTIVE = 0.2           # share of a window's frames a speaker must talk alone in for VBx to keep its embedding


class SingleSpeakerDiarizer:
    """Everyone is speaker S1, one voice for the whole video (spec §12 Phase 1: no diarization).

    Used while pyannote's gated model isn't installed. One turn spans the whole file, so every word keeps
    the same voice.
    """

    def diarize(self, audio: np.ndarray, **_: object) -> DiarBlock:
        end = len(audio) / SR_ANALYSIS
        turns = [SpeakerTurn("S1", 0.0, end)] if len(audio) else []
        return DiarBlock(0.0, end, turns, list(turns), {})


def make_diarizer(model_dir: Path, device: str) -> Diarizer:
    if all((model_dir / f).is_file() for f in ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin")):
        return PyannoteDiarizer(model_dir, device)
    log.warning("Speaker diarization model not installed (%s); dubbing everyone with one voice.", model_dir.name)
    return SingleSpeakerDiarizer()


class PyannoteDiarizer:
    """pyannote community-1. Keeps what the speaker registry needs: exclusive turns (one speaker per
    instant, for assigning words) and per-speaker centroid embeddings (for settling the speakers found)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self._torch = torch
        self.pipeline = Pipeline.from_pretrained(str(model_dir))
        self.pipeline.to(torch.device(device))

    def diarize(self, audio: np.ndarray, *, num_speakers: int | None = None, min_speakers: int | None = None,
                max_speakers: int | None = None, step: float | None = None,
                progress: Callable[[float], None] | None = None, cancel: threading.Event | None = None) -> DiarBlock:
        """The whole file in one pipeline call, so one clustering sees every speaker (OFFLINE-RENDER §2.3). Its hook
        reports segmentation as 0-40 % and embeddings as 40-90 % (clustering has no hook), raises `Cancelled` once
        `cancel` is set (until clustering starts: a pause during it lets the run finish), and runs the memory guard: if the clustering of the embeddings VBx would keep is predicted
        over DIAR_CLUSTER_BUDGET, the run stops before any embedding is computed and starts again at COARSE_STEP (half
        the windows, a quarter of the matrix). `step` forces a step from the start. The pipeline's own step is put
        back afterwards."""
        end = len(audio) / SR_ANALYSIS
        seg = self.pipeline._segmentation
        default = seg.step
        ratio = default / seg.duration if step is None else step
        if len(audio) < SR_ANALYSIS:
            return DiarBlock(0.0, end, [], [], {}, step=round(ratio, 3))
        # A copy: the job's audio is memory-mapped read-only, which torch won't take.
        wave = {"waveform": self._torch.from_numpy(np.array(audio, np.float32)).unsqueeze(0), "sample_rate": SR_ANALYSIS}
        count = {"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}
        kept: list[int] = []  # the guard's count of the embeddings each run clusters
        try:
            while True:
                seg.step = ratio * seg.duration
                try:
                    out = self.pipeline(wave, hook=_hook(progress, cancel, guard=ratio < COARSE_STEP, kept=kept), **count)
                    break
                except _TooMany as e:
                    log.warning("diarization would cluster %d embeddings (about %.1f GB); again at a %.1f s step", e.n,
                                LINKAGE_BYTES * e.n ** 2 / 1e9, COARSE_STEP * seg.duration)
                    ratio = COARSE_STEP
        finally:
            seg.step = default
        return replace(_block(out, 0.0, end), step=round(ratio, 3), embeddings=kept[-1] if kept else None)


class _TooMany(Exception):
    """The memory guard's stop: `n` embeddings would be clustered."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.n = n


def _retained(data: np.ndarray) -> int:
    """How many (window, local speaker) embeddings VBx will cluster: pyannote's `filter_embeddings` keeps a pair whose
    speaker talks alone for at least MIN_ACTIVE of the window's frames (NaN embeddings aside, so this is an upper
    bound). community-1's segmentation is powerset, so `data` is already 0/1."""
    active = np.asarray(data) > 0.5
    alone = active & (active.sum(axis=2, keepdims=True) == 1)
    return int((alone.sum(axis=1) >= MIN_ACTIVE * active.shape[1]).sum())


# The steps whose hook calls a set cancel event stops at: up to the start of clustering. pyannote's clustering has no
# hook, and its next call ("discrete_diarization") comes after it: a run that far returns its result, not to be thrown
# away and redone on resume.
_CANCELLABLE = ("segmentation", "speaker_counting", "embeddings")


def _hook(progress: Callable[[float], None] | None, cancel: threading.Event | None, guard: bool,
          kept: list[int]) -> Callable[..., None]:
    """pyannote's `hook(step, artifact, file=, completed=, total=)`: progress, cancellation and the memory guard (see
    `PyannoteDiarizer.diarize`), which appends its count to `kept`."""
    def hook(name: str, artifact: object = None, *, completed: int | None = None, total: int | None = None,
             **_: object) -> None:
        if cancel is not None and cancel.is_set() and name in _CANCELLABLE:
            raise Cancelled("diarization cancelled")
        if name == "segmentation" and artifact is not None:
            n = _retained(artifact.data)
            kept.append(n)
            log.info("diarization: %d embeddings to cluster (about %.1f GB at its peak)", n, LINKAGE_BYTES * n ** 2 / 1e9)
            if guard and LINKAGE_BYTES * n ** 2 > DIAR_CLUSTER_BUDGET:
                raise _TooMany(n)
        if progress is not None and completed is not None and total:
            share = min(completed / total, 1.0)
            if name == "segmentation":
                progress(0.4 * share)
            elif name == "embeddings":
                progress(0.4 + 0.5 * share)
    return hook


def _block(out, offset: float, end: float) -> DiarBlock:
    """A pipeline output as a block: its turns and exclusive turns shifted by `offset`, and its centroids."""
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

    The weights load on first use, not at engine start, so they aren't resident while a render
    diarizes the whole file (OFFLINE-RENDER §2.3, §7).
    """

    def __init__(self, ckpt_dir: Path, device: str, presets_dir: Path | None = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5, t3_bf16: bool = True, cfm_steps: int = 10) -> None:
        import torch
        from chatterbox.models.s3gen import S3GEN_SR

        self._torch = torch
        self._ckpt_dir, self._device = ckpt_dir, device
        self._model = None
        self._builtin = None
        self._load_lock = threading.Lock()
        self.sample_rate = int(S3GEN_SR)
        self.presets_dir = presets_dir
        self._presets: dict[str, ChatterboxVoice] = {}
        self.exaggeration, self.cfg_weight = exaggeration, cfg_weight
        self.cfm_steps = cfm_steps
        self._t3_dtype = torch.bfloat16 if t3_bf16 and device in ("mps", "cuda") else None
        self._backend = None

    @property
    def model(self):
        """The Chatterbox pipeline, loaded on first use (from whichever worker thread asks first)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load()
        return self._model

    @model.setter
    def model(self, model) -> None:
        self._model = model

    @property
    def builtin(self):
        """conds.pt: the checkpoint's own voice, the preset of last resort."""
        self.model  # noqa: B018  (loads it)
        return self._builtin

    def _load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        # The checkpoint names its fine-tuned T3 weights (t3_mtl_te.safetensors); the loader's
        # default is the base multilingual model's file, which this checkpoint doesn't ship.
        cfg = self._ckpt_dir / "config.json"
        t3_model = json.loads(cfg.read_text()).get("t3_model") if cfg.is_file() else None
        t0 = time.perf_counter()
        model = ChatterboxMultilingualTTS.from_local(str(self._ckpt_dir), self._device, t3_model=t3_model)
        self.sample_rate = int(model.sr)
        self._builtin = model.conds
        if self._t3_dtype is not None:
            model.t3.tfmr.to(self._t3_dtype)
        self._model = model
        log.info("Chatterbox loaded in %.1f s", time.perf_counter() - t0)

    def release(self) -> None:
        """Let go of the model, its built-in voice, the presets made from it and its T3 backend, and empty the MPS (or
        CUDA) cache: a render calls this once its dub is final, so the export runs without Chatterbox's 2.2-3.2 GB
        resident (OFFLINE-RENDER §2.13, §7). The next synthesis loads it again."""
        import gc

        with self._load_lock:
            self._model = self._builtin = self._backend = None
            self._presets.clear()
        gc.collect()
        if self._device == "mps" and self._torch.backends.mps.is_available():
            self._torch.mps.empty_cache()
        elif self._device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

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
        if name not in self._presets:  # computed once: the dub loop asks for a preset every line
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

    def pack_take(self, take: MelTake) -> dict[str, np.ndarray]:
        """A take as the render keeps it on disk (OFFLINE-RENDER §2.9): its mel, flowed first if it wasn't yet (S3Gen's
        flow: hold the GPU), as float16, and its speech-token count, which sets its length."""
        self._flow(take)
        mel = (np.zeros((1, 80, 0), np.float16) if take.mel is None
               else take.mel.detach().to("cpu", self._torch.float32).numpy().astype(np.float16))
        return {"mel": mel, "n_tokens": np.array(take.n_tokens, np.int64)}

    def unpack_take(self, d: dict[str, np.ndarray]) -> MelTake:
        """A take `pack_take` kept, back as a flowed take with its mel on the model's device, for `vocode`."""
        mel = np.asarray(d["mel"])
        mel = self._torch.from_numpy(mel.astype(np.float32)).to(self.model.device) if mel.shape[-1] else None
        return MelTake(mel, int(d["n_tokens"]), cfm_steps=self.cfm_steps)

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
