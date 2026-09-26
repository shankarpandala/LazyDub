"""Pauses inside a take, and how a take is compressed into less time (ARCHITECTURE §3.10 steps 3 and 7).

Pure numpy. A take's pauses are found on its audio at the voice's natural pace (the level of 10 ms frames against the
take's own loud frames), so the same rule works for every TTS. A line played faster first shortens its internal pauses,
each to no less than `keep` (Amazon found non-uniform scaling beats uniform at 1.1-1.4x, research 04 N2), and only what
they can't give speeds up the speech. Leading and trailing silence are left as they are.
"""

from __future__ import annotations

import numpy as np

FRAME = 0.01        # s per level frame
FLOOR_DB = -40.0    # a frame this far under the take's loud frames (their 95th percentile) is silent
MIN_PAUSE = 0.15    # s: a shorter quiet run is a stop closure or a gap between syllables, not a pause
FADE = 0.0025       # s of fade out and in at each cut (it is made in silence; the fade only keeps it click-free)

Span = tuple[float, float]


def pauses(audio: np.ndarray, sr: int, min_len: float = MIN_PAUSE, floor_db: float = FLOOR_DB) -> tuple[Span, ...]:
    """Silent runs of at least `min_len` s in `audio`, in seconds, and its leading and trailing silence however short
    (it isn't voiced: the speech-level metrics start and end the take where its voice does)."""
    x = np.asarray(audio, np.float32)
    hop = max(int(round(sr * FRAME)), 1)
    n = len(x) // hop
    if n == 0:
        return ()
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    loud = float(np.percentile(rms, 95))
    if loud <= 1e-6:  # all silent
        return ((0.0, len(x) / sr),)
    quiet = rms < loud * 10 ** (floor_db / 20)
    out: list[Span] = []
    k = 0
    while k < n:
        if not quiet[k]:
            k += 1
            continue
        j = k
        while j < n and quiet[j]:
            j += 1
        a, b = k * hop / sr, (len(x) if j == n else j * hop) / sr
        if b - a >= min_len - 1e-9 or k == 0 or j == n:
            out.append((round(a, 4), round(b, 4)))
        k = j
    return tuple(out)


def internal(spans: tuple[Span, ...], total: float) -> tuple[Span, ...]:
    """The pauses inside the take: not its leading or trailing silence."""
    return tuple((a, b) for a, b in spans if a > FRAME / 2 and b < total - FRAME / 2)


def voiced(spans: tuple[Span, ...], total: float) -> tuple[Span, ...]:
    """Where the take is voiced: [0, total] less its pauses."""
    out, t = [], 0.0
    for a, b in sorted(spans):
        if a > t + 1e-9:
            out.append((t, min(a, total)))
        t = max(t, b)
    if total > t + 1e-9:
        out.append((t, total))
    return tuple(out)


def squeeze(spans: tuple[Span, ...], total: float, rate: float, keep: float) -> tuple[float, tuple[Span, ...]]:
    """How a take of `total` natural seconds plays in total / rate: its internal pauses shorten first, each to no less
    than `keep` s, and only what they can't give is taken by speeding up. Returns the rate the speech plays at (every
    pause too, before it is cut) and, per pause shortened, (its middle in the take, seconds cut from it at that rate)."""
    if rate <= 1.0 + 1e-9 or total <= 0.0:
        return 1.0, ()
    inner = internal(spans, total)
    target = total / rate
    spare = [max(b - a - keep, 0.0) for a, b in inner]
    if total - sum(spare) <= target + 1e-12:  # the pauses alone give it: cut each in proportion, speech at 1.0
        need = total - target
        cuts = tuple(((a + b) / 2, need * s / sum(spare)) for (a, b), s in zip(inner, spare) if s > 0)
        return 1.0, cuts

    def length(r: float) -> float:
        return total / r - sum(max((b - a) / r - keep, 0.0) for a, b in inner)

    lo, hi = 1.0, rate  # length() falls as r rises; length(rate) <= target
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if length(mid) > target else (lo, mid)
    r = hi
    return r, tuple(((a + b) / 2, (b - a) / r - keep) for a, b in inner if (b - a) / r > keep)


def out_time(t: float, speech_rate: float, cuts: tuple[Span, ...]) -> float:
    """Where natural second `t` of a take plays once it is squeezed (`squeeze`'s rate and cuts)."""
    return t / speech_rate - sum(q for m, q in cuts if m < t)


def cut(audio: np.ndarray, sr: int, speech_rate: float, cuts: tuple[Span, ...], length: int) -> np.ndarray:
    """Take `cuts` (natural middles, seconds at `speech_rate`) out of `audio`, a take rendered at `speech_rate`, and
    return exactly `length` samples (a rounding sample or two is padded or trimmed at the end)."""
    x = np.asarray(audio, np.float32)
    keep: list[np.ndarray] = []
    at = 0
    for m, q in sorted(cuts):
        c = int(round(m / speech_rate * sr))
        a = min(max(c - int(round(q * sr / 2)), at), len(x))
        b = min(max(a + int(round(q * sr)), a), len(x))
        keep.append(x[at:a])
        at = b
    keep.append(x[at:])
    f = max(int(round(FADE * sr)), 1)
    for k in range(len(keep) - 1):  # fade each side of a join
        head, tail = keep[k], keep[k + 1]
        if len(head):
            head = head.copy()
            n = min(f, len(head))
            head[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
            keep[k] = head
        if len(tail):
            tail = tail.copy()
            n = min(f, len(tail))
            tail[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            keep[k + 1] = tail
    y = np.concatenate(keep) if keep else np.zeros(0, np.float32)
    if len(y) >= length:
        return y[:length].astype(np.float32)
    return np.pad(y, (0, length - len(y))).astype(np.float32)
