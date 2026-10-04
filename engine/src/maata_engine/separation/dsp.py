"""The STFT, inverse STFT and mel filterbank the separator uses (OFFLINE-RENDER §2.14, ADR-020).

Vendored from Blaizzy/mlx-audio, `mlx_audio/dsp.py` at commit a8546e64ac5fdc7ace03e4b75dd2daad444a614e (2026-04-27):
only `stft`, `istft`, `mel_filters` and the Hann window they fall back to, which import nothing but mlx and numpy. The
rest of that module (loudness, Kaldi features, `ISTFTCache`) is left out. Changes: the window table keeps only "hann",
the one window Maata names (the separator passes its own window array anyway).

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
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx


@lru_cache(maxsize=None)
def hanning(size: int, periodic: bool = False) -> mx.array:
    """Hann window of `size` samples (`periodic` for spectral analysis)."""
    denom = size if periodic else size - 1
    return mx.array([0.5 * (1 - math.cos(2 * math.pi * n / denom)) for n in range(size)])


STR_TO_WINDOW_FN = {"hann": hanning, "hanning": hanning}


def stft(x: mx.array, n_fft: int = 800, hop_length: int | None = None, win_length: int | None = None,
         window: mx.array | str = "hann", center: bool = True, pad_mode: str = "reflect") -> mx.array:
    """The STFT of the 1-D signal `x`: (frames, n_fft // 2 + 1) complex."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length)
    else:
        w = window

    if w.shape[0] < n_fft:
        w = mx.concatenate([w, mx.zeros((n_fft - w.shape[0],))], axis=0)

    def _pad(x: mx.array, padding: int, pad_mode: str = "reflect") -> mx.array:
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        if pad_mode == "reflect":
            prefix = x[1:padding + 1][::-1]
            suffix = x[-(padding + 1):-1][::-1]
            return mx.concatenate([prefix, x, suffix])
        raise ValueError(f"Invalid pad_mode {pad_mode}")

    if center:
        x = _pad(x, n_fft // 2, pad_mode)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with hop_length={hop_length} "
                         f"and center={center}.")

    frames = mx.as_strided(x, shape=(num_frames, n_fft), strides=(hop_length, 1))
    return mx.fft.rfft(frames * w)


def istft(x: mx.array, hop_length: int | None = None, win_length: int | None = None, window: mx.array | str = "hann",
          center: bool = True, length: int | None = None, normalized: bool = False) -> mx.array:
    """The inverse STFT of `x`, (n_fft // 2 + 1, frames) complex, by overlap-add normalised by the window sum (with
    `normalized`, the squared window's: COLA, as PyTorch's istft). With `center` and no `length`, the centre padding is
    removed; with `length`, the signal is cut to it (and the centre padding is not removed)."""
    if win_length is None:
        win_length = (x.shape[1] - 1) * 2
    if hop_length is None:
        hop_length = win_length // 4

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length + 1)[:-1]
    else:
        w = window

    if w.shape[0] < win_length:
        w = mx.concatenate([w, mx.zeros((win_length - w.shape[0],))], axis=0)

    num_frames = x.shape[1]
    t = (num_frames - 1) * hop_length + win_length

    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)  # each frame's inverse FFT

    # Where each frame's samples go in the signal.
    frame_offsets = mx.arange(num_frames) * hop_length
    indices_flat = (frame_offsets[:, None] + mx.arange(win_length)).flatten()

    updates_reconstructed = (frames_time * w).flatten()
    window_norm = (w * w) if normalized else w
    updates_window = mx.tile(window_norm, (num_frames,)).flatten()

    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)
    reconstructed = mx.where(window_sum > 1e-10, reconstructed / window_sum, reconstructed)

    if center and length is None:
        reconstructed = reconstructed[win_length // 2:-win_length // 2]
    if length is not None:
        reconstructed = reconstructed[:length]
    return reconstructed


@lru_cache(maxsize=None)
def mel_filters(sample_rate: int, n_fft: int, n_mels: int, f_min: float = 0, f_max: Optional[float] = None,
                norm: Optional[str] = None, mel_scale: str = "htk") -> mx.array:
    """Triangular mel filters, (n_mels, n_fft // 2 + 1), on the HTK or the Slaney mel scale (librosa's orientation)."""

    def hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)
        f_min, f_sp = 0.0, 200.0 / 3  # Slaney: linear below 1 kHz, logarithmic above
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: mx.array, mel_scale: str = "htk") -> mx.array:
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        return mx.where(mels >= min_log_mel, min_log_hz * mx.exp(logstep * (mels - min_log_mel)), freqs)

    f_max = f_max or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    m_pts = mx.linspace(hz_to_mel(f_min, mel_scale), hz_to_mel(f_max, mel_scale), n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)
    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes))

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    return filterbank.moveaxis(0, 1)
