"""Pitch-preserving time-stretch (WSOLA) for the §6.6 speed-up and user playback speed.

`rate` > 1 makes the audio shorter. Output length is round(len / rate) samples exactly, so
placements computed from `duration / rate` hold.
"""

from __future__ import annotations

import numpy as np


def wsola(x: np.ndarray, rate: float, sr: int, frame_ms: float = 30.0, search_ms: float = 12.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    target_len = int(round(len(x) / rate))
    if abs(rate - 1.0) < 1e-3 or len(x) < 4:
        return x.copy() if len(x) == target_len else np.resize(x, target_len).astype(np.float32)
    n = max(int(sr * frame_ms / 1000) // 2 * 2, 64)
    hop = n // 2
    search = int(sr * search_ms / 1000)
    win = np.hanning(n).astype(np.float32)
    pad = np.concatenate([np.zeros(n, np.float32), x, np.zeros(n + search + hop, np.float32)])
    out = np.zeros(target_len + 2 * n, np.float32)
    norm = np.zeros_like(out)
    prev = pad[n: n + n].copy()  # natural continuation of the last copied frame
    k = 0
    while k * hop < target_len + n:
        ideal = n + int(round(k * hop * rate))
        lo, hi = max(ideal - search, 0), min(ideal + search, len(pad) - n)
        if hi <= lo:
            break
        # Pick the segment in the search region that best matches the natural continuation.
        region = pad[lo: hi + n]
        corr = np.correlate(region, prev[:n], mode="valid")
        best = lo + int(np.argmax(corr)) if corr.size else ideal
        seg = pad[best: best + n]
        o = k * hop
        out[o: o + n] += seg * win
        norm[o: o + n] += win
        prev = pad[best + hop: best + hop + n]
        k += 1
    norm[norm < 1e-3] = 1.0
    y = out / norm
    y = y[:target_len]
    return y.astype(np.float32) if len(y) == target_len else np.pad(y, (0, target_len - len(y))).astype(np.float32)
