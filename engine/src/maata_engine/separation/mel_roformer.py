"""Mel-Band RoFormer on MLX: the vocal separator under the dub's background sound (OFFLINE-RENDER §2.14, ADR-020).

Vendored from Blaizzy/mlx-audio at commit a8546e64ac5fdc7ace03e4b75dd2daad444a614e (2026-04-27):
`mlx_audio/sts/models/mel_roformer/model.py` and `mlx_audio/sts/models/mel_roformer/config.py`. The pinned weights
(`mlx-community/mel-roformer-kim-vocal-2-mlx`, models.lock.json) were converted by mlx-audio commit 8380ab8 in bf16 with
mlx 0.31.0; the commits since changed only the STFT code, not the key layout `sanitize` maps.

Changes from upstream:
- the STFT, inverse STFT and mel filterbank come from the vendored `.dsp` (upstream imports `mlx_audio.dsp`);
- of the config's presets only `kim_vocal_2` is kept;
- `from_pretrained` loads only `<dir>/model.safetensors` and `<dir>/config.json` from a local folder: no hub lookup
  (`get_model_path` is gone), no other file name;
- weights load with `strict=True` (upstream's `strict=False` would silently leave a key `sanitize` missed random);
- precision (§2.14): the STFT, the mask's application and the inverse STFT run in float32, and the band-split input is
  cast to the model's `dtype` (bf16 by default), so the transformer and the mask estimator run in it, with the RoPE
  tables cast to it too (a float32 table would promote every attention to float32);
- `MelRoFormerResult` (unused) is left out.

mlx-audio is MIT-licensed: Copyright (c) 2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio).
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
"AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Architecture (Kim Vocal 2, about 228M parameters): [B, 2, samples] → STFT → CaC interleave → BandSplit → 6× dual-axis
transformer (time, then frequency) → MaskEstimate → complex multiply → iSTFT → [B, 2, samples] (the vocals). Reference
implementations: lucidrains/BS-RoFormer and ZFTurbo/Music-Source-Separation-Training.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .dsp import istft as _istft
from .dsp import mel_filters
from .dsp import stft as _stft

WEIGHTS = "model.safetensors"
CONFIG = "config.json"


@dataclass
class MelRoFormerConfig:
    """Mel-Band RoFormer's hyperparameters. Architecture: STFT → CaC interleave → BandSplit → N× DualAxisTransformer →
    MaskEstimate → complex multiply → iSTFT."""

    # Model architecture
    dim: int = 384
    depth: int = 6  # dual-axis transformer levels
    heads: int = 8
    dim_head: int = 64
    num_bands: int = 60
    num_stems: int = 1  # 1 = vocals only
    ff_mult: int = 4
    mlp_expansion_factor: int = 4
    mask_estimator_depth: int = 2

    # STFT
    n_fft: int = 2048
    hop_length: int = 441
    win_length: int = 2048
    sample_rate: int = 44100

    # Processing
    chunk_size: int = 352800  # 8 s at 44.1 kHz
    num_overlap: int = 2      # 50 % overlap for chunked processing

    checkpoint_family: Optional[str] = None

    @property
    def dim_inner(self) -> int:
        return self.heads * self.dim_head

    @property
    def ff_dim(self) -> int:
        return self.dim * self.ff_mult

    @property
    def mlp_hidden(self) -> int:
        return self.dim * self.mlp_expansion_factor

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1

    @classmethod
    def kim_vocal_2(cls) -> MelRoFormerConfig:
        """KimberleyJSN/melbandroformer (depth 6, 60 bands, 44.1 kHz): ZFTurbo/Music-Source-Separation-Training's
        configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml."""
        return cls(depth=6, checkpoint_family="kim_vocal_2")

    @classmethod
    def from_dict(cls, data: dict) -> MelRoFormerConfig:
        """A config from a config.json's fields (the converter's own keys, `_dtype` and the like, are left out)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


class RMSNorm(nn.Module):
    """RMSNorm matching ZFTurbo's `F.normalize(x, dim=-1) * sqrt(dim) * gamma` (an `eps` of 1e-12 on the norm, not an
    additive 1e-5 as `mlx.nn.RMSNorm` has, which diverges at the quiet high-frequency bins)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return (x / mx.maximum(norm, 1e-12)) * self.scale * self.weight


# ---------- Mel filterbank ----------


class MelFilterbank:
    """Mel-scale band boundaries and the gather indices of each band (Slaney mel scale, binarised triangular filters,
    stereo-interleaved)."""

    def __init__(self, config: MelRoFormerConfig):
        self.config = config
        fb = self._build_mel_filterbank()
        self.freq_indices, self.band_dims, self.num_bands_per_freq = self._compute_band_info(fb)

    def _build_mel_filterbank(self) -> np.ndarray:
        """ZFTurbo's binarised mel filterbank: the Slaney triangular filters, with the DC bin forced into the first band
        and the Nyquist bin into the last (the filters leave them out, and every bin must be covered)."""
        c = self.config
        fb = np.asarray(mel_filters(sample_rate=c.sample_rate, n_fft=c.n_fft, n_mels=c.num_bands,
                                    mel_scale="slaney")).astype(np.float32)
        fb[0, 0] = 1.0
        fb[-1, -1] = 1.0
        return (fb > 0).astype(np.float32)

    def _compute_band_info(self, fb: np.ndarray):
        """Per band: the gather indices into the CaC spectrum (each bin twice, left and right), its input size (x2 for
        real and imaginary), and per CaC entry how many bands cover it (to average where bands overlap)."""
        freq_bins = self.config.freq_bins
        freq_indices = []
        num_bands_per_freq = np.zeros(freq_bins * 2, dtype=np.float32)
        for i in range(self.config.num_bands):
            bin_indices = np.where(fb[i] > 0)[0]
            if len(bin_indices) == 0:
                bin_indices = np.array([i])
            cac_indices = [j for b in bin_indices for j in (b * 2, b * 2 + 1)]
            freq_indices.append(np.array(cac_indices, dtype=np.int32))
            for idx in cac_indices:
                if idx < len(num_bands_per_freq):
                    num_bands_per_freq[idx] += 1
        band_dims = [len(idx) * 2 for idx in freq_indices]
        num_bands_per_freq = np.maximum(num_bands_per_freq, 1.0)
        return [mx.array(idx) for idx in freq_indices], band_dims, num_bands_per_freq


# ---------- RoPE ----------


class RotaryEmbedding:
    """Interleaved-pair RoPE, as `rotary_embedding_torch` (which ZFTurbo uses): dimensions (x[2i], x[2i+1]) rotate
    together."""

    def __init__(self, dim_head: int, base: float = 10000.0):
        half = dim_head // 2
        self.freqs_mx = mx.array(1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half)))

    def get_cos_sin(self, seq_len: int) -> tuple[mx.array, mx.array]:
        """(cos, sin), each [seq_len, dim_head], each base frequency twice along the last axis."""
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.repeat(mx.outer(t, self.freqs_mx), 2, axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate each pair (x[2i], x[2i+1]) by its angle."""
    shape = x.shape
    pairs = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(shape)
    return x * cos + rotated * sin


# ---------- the transformer ----------


class RoFormerAttention(nn.Module):
    """Multi-head attention with RoPE and per-head sigmoid gates."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_gates = nn.Linear(dim, heads, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.rotary_embed = RotaryEmbedding(dim_head)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        h = self.norm(x)
        q = self.to_q(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        cos, sin = self.rotary_embed.get_cos_sin(T)
        cos, sin = cos.astype(q.dtype), sin.astype(q.dtype)  # the compute dtype's, not float32 (§2.14)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.dim_head))
        gates = mx.sigmoid(self.to_gates(h)).transpose(0, 2, 1)[..., None]  # [B, heads, T, 1]
        attn = (attn * gates).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(attn)


class RoFormerFFN(nn.Module):
    """RMSNorm → Linear → GELU → Linear, its keys as PyTorch's Sequential indices (net.0, net.1, net.4)."""

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        ff_dim = dim * ff_mult
        self.net = [RMSNorm(dim), nn.Linear(dim, ff_dim), None, None, nn.Linear(ff_dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[4](nn.gelu(self.net[1](self.net[0](x))))


class Transformer(nn.Module):
    """One axis's transformer: RoPE attention + FFN per layer, then an RMSNorm."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, ff_mult: int):
        super().__init__()
        self.layers = [[RoFormerAttention(dim, heads, dim_head), RoFormerFFN(dim, ff_mult)] for _ in range(depth)]
        self.norm = RMSNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------- band split and mask estimation ----------


class BandSplit(nn.Module):
    """The spectrogram split into mel bands, each projected to the model's dimension."""

    def __init__(self, config: MelRoFormerConfig):
        super().__init__()
        self.filterbank = MelFilterbank(config)
        self.to_features = [[RMSNorm(bd), nn.Linear(bd, config.dim)] for bd in self.filterbank.band_dims]

    def split(self, x: mx.array) -> mx.array:
        """[B, freq_bins*2, T, 2] (CaC, real/imaginary) → [B, T, num_bands, dim]."""
        B, _, T, _ = x.shape
        bands = []
        for indices, (norm, proj) in zip(self.filterbank.freq_indices, self.to_features):
            gathered = x[:, indices, :, :]
            band_input = gathered.transpose(0, 2, 1, 3).reshape(B, T, gathered.shape[1] * 2)
            bands.append(proj(norm(band_input)))
        return mx.stack(bands, axis=2)

    def merge(self, band_masks: list, freq_bins_times_two: int) -> mx.array:
        """The bands' masks ([B, T, band_dim] each) scattered back to [B, freq_bins*2, T, 2] in float32, averaged where
        bands overlap."""
        B, T = band_masks[0].shape[:2]
        output = mx.zeros((B, freq_bins_times_two, T, 2), dtype=mx.float32)
        for indices, mask in zip(self.filterbank.freq_indices, band_masks):
            n_freqs = len(indices)
            mask = mask.reshape(B, T, n_freqs, 2).transpose(0, 2, 1, 3)
            for j, idx in enumerate(indices):
                output = output.at[:, idx, :, :].add(mask[:, j, :, :])
        return output / mx.array(self.filterbank.num_bands_per_freq).reshape(1, -1, 1, 1)


class MaskEstimator(nn.Module):
    """Per band, an MLP (ZFTurbo's: Linear → Tanh, (Linear → Tanh) × (depth - 1), Linear) and a GLU: the band's complex
    mask."""

    def __init__(self, config: MelRoFormerConfig, band_dims: list):
        super().__init__()
        dim, hidden, depth = config.dim, config.mlp_hidden, config.mask_estimator_depth
        self.to_freqs = []
        for bd in band_dims:
            layers = [[nn.Linear(dim, hidden)]]
            layers += [[nn.Linear(hidden, hidden)] for _ in range(depth - 1)]
            layers.append([nn.Linear(hidden, bd * 2)])
            self.to_freqs.append(layers)

    def __call__(self, x: mx.array) -> list:
        """[B, T, num_bands, dim] → one [B, T, band_dim] mask per band."""
        masks = []
        for i in range(x.shape[2]):
            h = x[:, :, i, :]
            layers = self.to_freqs[i]
            for layer in layers[:-1]:
                h = mx.tanh(layer[0](h))
            h = layers[-1][0](h)
            half = h.shape[-1] // 2
            masks.append(h[..., :half] * mx.sigmoid(h[..., half:]))  # GLU
        return masks


# ---------- STFT / iSTFT over a batch of stereo signals ----------


def stft(audio: mx.array, n_fft: int, hop_length: int, window: mx.array) -> tuple[mx.array, mx.array]:
    """[B, channels, samples] → (real, imag), each [B, channels, freq_bins, frames], one signal at a time."""
    B, C, _ = audio.shape
    spectra = [_stft(audio[b, c], n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=True,
                     pad_mode="reflect") for b in range(B) for c in range(C)]
    stacked = mx.stack(spectra, axis=0)  # [B*C, frames, freq_bins] complex
    frames = stacked.shape[1]
    real = stacked.real.reshape(B, C, frames, -1).transpose(0, 1, 3, 2)
    imag = stacked.imag.reshape(B, C, frames, -1).transpose(0, 1, 3, 2)
    return real, imag


def istft(real: mx.array, imag: mx.array, n_fft: int, hop_length: int, window: mx.array, length: int) -> mx.array:
    """(real, imag), each [B, channels, freq_bins, frames] → [B, channels, length]. The dsp's istft is called with no
    `length` (that branch strips the centre padding) and the result cut or zero-padded to `length` here; its
    normalisation is the squared window's (COLA), as PyTorch's."""
    B, C, _, _ = real.shape
    spectra = real + 1j * imag
    waves = []
    for b in range(B):
        for c in range(C):
            recon = _istft(spectra[b, c], hop_length=hop_length, win_length=n_fft, window=window, center=True,
                           length=None, normalized=True)
            if recon.shape[0] < length:
                recon = mx.concatenate([recon, mx.zeros((length - recon.shape[0],), dtype=recon.dtype)])
            waves.append(recon[:length])
    return mx.stack(waves, axis=0).reshape(B, C, length)


# ---------- the model ----------


class MelRoFormer(nn.Module):
    """Mel-Band RoFormer vocal separation: STFT → CaC → BandSplit → dual-axis transformer → MaskEstimate → iSTFT.
    `dtype`: the transformer's and the mask estimator's compute dtype (the weights are cast to it by the caller)."""

    def __init__(self, config: MelRoFormerConfig = MelRoFormerConfig(), dtype: mx.Dtype = mx.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.band_split = BandSplit(config)
        # Per level: [time transformer, frequency transformer].
        self.layers = [[Transformer(config.dim, 1, config.heads, config.dim_head, config.ff_mult),
                        Transformer(config.dim, 1, config.heads, config.dim_head, config.ff_mult)]
                       for _ in range(config.depth)]
        self.mask_estimators = [MaskEstimator(config, self.band_split.filterbank.band_dims)]
        self.set_dtype(dtype)  # the parameters in the compute dtype, as `from_pretrained` loads them

    def __call__(self, audio: mx.array) -> mx.array:
        """[B, 2, samples] stereo at 44.1 kHz → [B, 2, samples], the vocals (float32)."""
        length = audio.shape[2]
        c = self.config
        audio = audio.astype(mx.float32)
        window = mx.array(np.hanning(c.n_fft + 1)[:-1].astype(np.float32))

        # STFT in float32 → [B, 2, freq_bins, T].
        stft_real, stft_imag = stft(audio, c.n_fft, c.hop_length, window)
        B, _, freq_bins, T = stft_real.shape

        # CaC interleave → [B, freq_bins*2, T, 2] (real, imaginary).
        real = stft_real.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
        imag = stft_imag.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
        stft_repr = mx.stack([real, imag], axis=-1)

        # Band split, the transformer and the mask estimator in the compute dtype.
        x = self.band_split.split(stft_repr.astype(self.dtype))  # [B, T, num_bands, dim]
        Nb, D = x.shape[2], x.shape[3]
        for time_tf, freq_tf in self.layers:
            x = time_tf(x.transpose(0, 2, 1, 3).reshape(B * Nb, T, D)).reshape(B, Nb, T, D).transpose(0, 2, 1, 3)
            x = freq_tf(x.reshape(B * T, Nb, D)).reshape(B, T, Nb, D)
        masks = self.mask_estimators[0](x)

        # The mask (merged in float32) applied to the float32 spectrum: a complex multiply.
        mask = self.band_split.merge(masks, freq_bins * 2)
        mr, mi = mask[..., 0], mask[..., 1]
        out_real = real * mr - imag * mi
        out_imag = real * mi + imag * mr

        # De-interleave → [B, 2, freq_bins, T], then the iSTFT in float32.
        out_real = out_real.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)
        out_imag = out_imag.reshape(B, freq_bins, 2, T).transpose(0, 2, 1, 3)
        return istft(out_real, out_imag, c.n_fft, c.hop_length, window, length)

    def sanitize(self, weights: dict) -> dict:
        """The converted checkpoint's keys as this model names them, per key in turn:
        1. a packed `to_qkv` projection split into `to_q`, `to_k` and `to_v`;
        2. `rotary_embed.freqs` dropped (computed here);
        3. the mask estimator's PyTorch Sequential indices (linears at 0, 2, 4) mapped to list indices (0, 1, 2);
        4. `to_out.0.weight` (PyTorch's Sequential(Linear, Dropout)) unwrapped to `to_out.weight`;
        5. a trailing `.gamma` (PyTorch RMSNorm's scale) renamed `.weight`."""
        mask_mlp_re = re.compile(r"^(mask_estimators\.\d+\.to_freqs\.\d+)\.0\.(\d+)\.(weight|bias)$")
        out = {}
        for key, value in weights.items():
            if "to_qkv.weight" in key:
                prefix = key.replace("to_qkv.weight", "")
                third = value.shape[0] // 3
                out[f"{prefix}to_q.weight"] = value[:third]
                out[f"{prefix}to_k.weight"] = value[third:2 * third]
                out[f"{prefix}to_v.weight"] = value[2 * third:]
                continue
            if key.endswith("rotary_embed.freqs"):
                continue
            if m := mask_mlp_re.match(key):
                prefix, seq_idx, kind = m.groups()
                key = f"{prefix}.{int(seq_idx) // 2}.0.{kind}"
            if key.endswith("to_out.0.weight"):
                key = key[:-len(".0.weight")] + ".weight"
            if key.endswith(".gamma"):
                key = key[:-len(".gamma")] + ".weight"
            out[key] = value
        return out

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, dtype: mx.Dtype = mx.bfloat16) -> MelRoFormer:
        """The model in the local folder `model_dir`: its `config.json` and `model.safetensors`, nothing else, never a
        download. Every parameter must be in the weights and every weight a parameter (`strict`), with its shape; the
        weights are cast to `dtype`."""
        path = Path(model_dir)
        config = MelRoFormerConfig.from_dict(json.loads((path / CONFIG).read_text(encoding="utf-8")))
        model = cls(config, dtype)
        weights = model.sanitize(dict(mx.load(str(path / WEIGHTS))))
        model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
        model.eval()
        return model
