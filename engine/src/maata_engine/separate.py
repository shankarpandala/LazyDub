"""The background sound's chunking (OFFLINE-RENDER §2.14), numpy only: one global chunk grid over the whole file, the
separator's output overlap-added block by block, so a 60 s block equals a whole-file run's samples (up to float
rounding) and a resume, a preview or its continuation computes only the blocks it lacks.

The grid: half a chunk of padding at each end of the file (reflected when the file is longer than twice that, zero
otherwise, so a 3 s file works); chunks of `chunk` samples every hop (half a chunk) from the padded start; each chunk's
vocals weighted by a periodic Hann window and overlap-added, divided by the window sum. Every sample of the file lies
where two chunks' windows sum to one, and nothing past the padded end is read (zeros there). A block computes exactly
the chunks that cover it (`block_chunks`): one more than a block's share, about 6 %.

Sample i is the i-th sample of the 44.1 kHz decode, counted from the first decoded sample as `decode_audio_to` counts
(§2.2), so a dub time t is the same instant in the analysis audio, the bed and the output."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

SR = 44_100           # the separator's rate (the bed's, the mix's)
CHUNK = 352_800       # samples a chunk: 8 s
SEP_BLOCK = 60.0      # s a block of render/bed/
SEP_BATCH = 2         # chunks per separator call: a pause stops within a few seconds (4 to be measured on the M5 Pro)
SEP_DTYPE = "bfloat16"  # the separator's compute dtype (§2.14; float32 if bf16 measures over 0.1 dB SI-SDR worse)
SEP_VERSION = 1       # bumped whenever how a block is made changes (§12)
FRAME = 0.1           # s a value of the vocals envelope (render/bed_vocals.npy: 10 a second)

Vocals = Callable[[np.ndarray], np.ndarray]  # [b, 2, chunk] -> [b, 2, chunk], the vocals stem of each chunk


def block_samples(sr: int = SR) -> int:
    return int(round(SEP_BLOCK * sr))


def frame_samples(sr: int = SR) -> int:
    return int(round(FRAME * sr))


def block_count(n: int, sr: int = SR) -> int:
    """Blocks in a file of `n` samples."""
    return -(-n // block_samples(sr))


def block_range(k: int, n: int, sr: int = SR) -> tuple[int, int]:
    """Block k's samples [a, b) of a file of `n` samples."""
    a = k * block_samples(sr)
    return a, min(a + block_samples(sr), n)


def chunk_count(n: int, chunk: int = CHUNK) -> int:
    """Chunks on the grid of a file of `n` samples: the last starts at or after the file's end (in the file's own
    samples), so every sample lies where two windows sum to one."""
    return -(-n // (chunk // 2)) + 1 if n > 0 else 0


def chunk_start(j: int, chunk: int = CHUNK) -> int:
    """Where chunk j starts, in the file's own samples (negative in the padding)."""
    return j * (chunk // 2) - chunk // 2


def block_chunks(k: int, n: int, *, sr: int = SR, chunk: int = CHUNK) -> range:
    """The chunks covering block k of a file of `n` samples: those whose span [start, start + chunk) meets it."""
    a, b = block_range(k, n, sr)
    hop = chunk // 2
    return range(a // hop, min(chunk_count(n, chunk), -(-b // hop) + 1))


def block_span(k: int, n: int, *, sr: int = SR, chunk: int = CHUNK) -> tuple[int, int]:
    """The file's samples [lo, hi) block k's chunks read (their reflections included), within [0, n)."""
    js = block_chunks(k, n, sr=sr, chunk=chunk)
    return max(0, chunk_start(js.start, chunk)), min(n, chunk_start(js.stop - 1, chunk) + chunk)


def window(chunk: int = CHUNK) -> np.ndarray:
    """The periodic Hann window: at a hop of half its length, any two neighbours sum to one."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(chunk) / chunk)).astype(np.float32)


class Block:
    """Block k of a file of `n` samples, made from `mix` ([2, m] float32: the file's samples [at, at + m), covering
    `block_span`): its chunks in batches of `batch` (`batch(i)`), each batch's vocals overlap-added (`add(i, vocals)`),
    then `result()`. A caller that doesn't know `n` yet (the decode hasn't reached the end) may pass any `n` at or past
    the span's end as if the file went on: the chunks, padding and samples are then the same."""

    def __init__(self, k: int, n: int, mix: np.ndarray, at: int, *, sr: int = SR, chunk: int = CHUNK,
                 batch: int = SEP_BATCH) -> None:
        self.k, self.n, self.sr, self.chunk = k, n, sr, chunk
        self.a, self.b = block_range(k, n, sr)
        if self.b <= self.a:
            raise ValueError(f"block {k} is past the end of a file of {n} samples")
        lo, hi = block_span(k, n, sr=sr, chunk=chunk)
        if at > lo or at + mix.shape[1] < hi:
            raise ValueError(f"block {k} needs samples [{lo}, {hi}), not [{at}, {at + mix.shape[1]})")
        self.mix, self.at = mix, at
        js = list(block_chunks(k, n, sr=sr, chunk=chunk))
        self.groups = [js[i:i + batch] for i in range(0, len(js), batch)]
        self.reflect = n > chunk  # the padding (half a chunk each side) reflects only in a file over twice its length
        self.win = window(chunk)
        self.out = np.zeros((2, self.b - self.a), np.float32)
        self.wsum = np.zeros(self.b - self.a, np.float32)

    def _chunk(self, j: int) -> np.ndarray:
        """Chunk j's samples [2, chunk]: the file's, reflected (or zero) in the padding, zero past the padded end."""
        i = np.arange(chunk_start(j, self.chunk), chunk_start(j, self.chunk) + self.chunk)
        pad = self.chunk // 2
        if self.reflect:
            i = np.where(i < 0, -i, i)
            i = np.where((i >= self.n) & (i < self.n + pad), 2 * (self.n - 1) - i, i)
        ok = (i >= 0) & (i < self.n)
        out = np.zeros((2, self.chunk), np.float32)
        out[:, ok] = self.mix[:, i[ok] - self.at]
        return out

    def batch(self, i: int) -> np.ndarray:
        """Batch i's chunks, [len, 2, chunk] float32."""
        return np.stack([self._chunk(j) for j in self.groups[i]])

    def add(self, i: int, vocals: np.ndarray) -> None:
        """Overlap-add batch i's vocals ([len, 2, chunk]) into the block, each chunk windowed."""
        for j, y in zip(self.groups[i], np.asarray(vocals, np.float32)):
            s = chunk_start(j, self.chunk)
            lo, hi = max(self.a, s), min(self.b, s + self.chunk)
            if hi > lo:
                w = self.win[lo - s:hi - s]
                self.out[:, lo - self.a:hi - self.a] += y[:, lo - s:hi - s] * w
                self.wsum[lo - self.a:hi - self.a] += w

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """The bed (the mixture minus the vocals, [2, b - a] float32) and the vocals' mean square over both channels in
        FRAME s frames from the block's start (the last frame over the samples it has)."""
        vocals = self.out / np.where(self.wsum > 0, self.wsum, 1.0)
        bed = self.mix[:, self.a - self.at:self.b - self.at] - vocals
        f = frame_samples(self.sr)
        frames = -(-(self.b - self.a) // f)
        sq = np.zeros(frames * f, np.float64)
        sq[:self.b - self.a] = (vocals.astype(np.float64) ** 2).mean(axis=0)
        counts = np.full(frames, f, np.float64)
        counts[-1] = (self.b - self.a) - (frames - 1) * f
        return bed.astype(np.float32), (sq.reshape(frames, f).sum(axis=1) / counts).astype(np.float32)


def block_bed(k: int, n: int, mix: np.ndarray, at: int, vocals: Vocals, *, sr: int = SR, chunk: int = CHUNK,
              batch: int = SEP_BATCH) -> tuple[np.ndarray, np.ndarray]:
    """Block k's bed and its vocals' mean square per FRAME (`Block.result`), with `vocals` called on each batch of up to
    `batch` chunks. The render does the same steps itself, each batch on the GPU through `_on_gpu`."""
    blk = Block(k, n, mix, at, sr=sr, chunk=chunk, batch=batch)
    for i in range(len(blk.groups)):
        blk.add(i, vocals(blk.batch(i)))
    return blk.result()


class Buffer:
    """The decoded file's samples [at, end), stereo float32, as a sliding window: `put` appends what the decode gives,
    keeping nothing before `keep` (samples before it are counted and dropped as they come)."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.at = 0    # the file's sample of the first sample held
        self.end = 0   # the file's samples decoded so far
        self.keep = 0  # nothing before this is kept

    def put(self, x: np.ndarray) -> None:
        if not x.shape[1]:
            return
        self.end += x.shape[1]
        if self.end <= self.keep:  # all before the window: dropped as it comes
            self.at = self.end
            return
        self.parts.append(x)
        self.drop(self.keep)

    def drop(self, before: int) -> None:
        """Keep nothing before sample `before` from now on."""
        self.keep = max(self.keep, before)
        while self.parts and self.at + self.parts[0].shape[1] <= self.keep:
            self.at += self.parts.pop(0).shape[1]
        if self.parts and self.at < self.keep:
            self.parts[0] = self.parts[0][:, self.keep - self.at:]
            self.at = self.keep

    def get(self) -> np.ndarray:
        """What it holds, [2, end - at], as one array."""
        if len(self.parts) != 1:
            self.parts = [np.concatenate(self.parts, axis=1) if self.parts else np.zeros((2, 0), np.float32)]
        return self.parts[0]


def frames_in(n: int, sr: int = SR) -> int:
    """FRAME s frames in a file of `n` samples (the last over what it has)."""
    return -(-n // frame_samples(sr))


def mean_square(frames: np.ndarray, spans: list[tuple[float, float]]) -> float | None:
    """The mean of the FRAME s values `frames` (from 0 s; NaN where unknown) whose middle lies in one of `spans` (s);
    None when no known frame does."""
    sel = np.zeros(len(frames), bool)
    for a, b in spans:  # frame i's middle, (i + 0.5) FRAME, in [a, b)
        sel[max(0, math.ceil(a / FRAME - 0.5)):max(0, math.ceil(b / FRAME - 0.5))] = True
    vals = np.asarray(frames, np.float64)[sel]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else None
