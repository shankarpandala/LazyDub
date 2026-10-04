"""The dubbed MP4's audio (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §2.15), on the CPU, numpy blocks and
libavfilter through PyAV: the Telugu track (each final line's PCM added at its start, times its speaker's gain) at 24 kHz
mono, resampled to 44.1 kHz and copied to both channels, with the background bed (the original music and effects,
§2.14) added under it when there is one: at the original's voice/bed balance and ducked under speech by a fixed
envelope; then a static gain to MIX_LUFS and a limiter with no delay. Loudness is BS.1770 (FFmpeg's ebur128 filter).

Times are seconds on the output's clock: the caller places each line where it plays in the file. A `Background` is on
the dub's clock and is moved onto the output's by `Background.on`."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

MIX_SR = 44_100          # the mix's rate: the separator's and YouTube's m4a rate, so a bed is never resampled
BLOCK = 10.0             # s a block
VOICE_REF = -20.0        # LUFS every speaker's voice is set to
VOICE_CLAMP = 6.0        # dB: no speaker's gain beyond this, so a near-silent clone is never pumped into noise
MIX_LUFS = -16.0         # the mix's integrated loudness (Apple's level for spoken programmes)
LIMIT_DBFS = -2.0        # the limiter's sample-peak ceiling, meant to keep the true peak under TP_CEIL after AAC
TP_CEIL = -1.0           # dBTP: a measured true peak above this is flagged
SILENT = -69.0           # LUFS: ebur128 reads -70 when nothing passes its absolute gate
LIMITER = f"limit={10 ** (LIMIT_DBFS / 20):.6f}:attack=5:release=50:level=0:latency=1"  # latency=1: no 5 ms delay
# The bed's ducking (§2.15, M11): DUCK_DB inside the union of the Telugu lines and the English speech (gaps under BRIDGE
# s bridged), DUCK_BARE_DB inside English speech with no Telugu line within BARE_GAP s that lasts BARE_MIN s or more,
# raised-cosine ramps of ATTACK s before a span and RELEASE s after it; 0 dB elsewhere.
DUCK_DB = -6.0
DUCK_BARE_DB = -12.0
BRIDGE = 0.5
BARE_GAP = 1.0
BARE_MIN = 1.5
ATTACK = 0.15
RELEASE = 0.4
BED_CLAMP = 12.0         # dB: the voice/bed balance never moves the bed more than this (a separator that left the
                         # speech turns near-silent in its vocals would otherwise raise the music over the voices)

Bed = Callable[[int, int], np.ndarray]  # the bed's samples [a, b) of the output at MIX_SR, (2, b - a) float32
Spans = Sequence[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Line:
    """A final line as the mix places it: from `start` s on the output's clock, its speaker, and its PCM file
    (render/pcm/<key>.npy, mono float16 at the voice's rate, watermarked)."""

    start: float
    speaker: str
    pcm: Path
    samples: int


class Loudness:
    """BS.1770 loudness through an ebur128 filter graph, fed block by block: integrated loudness (gated), loudness range,
    and with `peak` the true peak (4x oversampled). Mono is measured as the dual mono it becomes in the stereo mix."""

    def __init__(self, sample_rate: int, channels: int, peak: bool = False) -> None:
        from av.filter import Graph

        self.sr, self.channels = sample_rate, channels
        self.graph = Graph()
        src = self.graph.add_abuffer(format="fltp", sample_rate=sample_rate, layout="mono" if channels == 1 else "stereo",
                                     time_base=Fraction(1, sample_rate))
        opts = "metadata=1" + (":peak=true" if peak else "") + (":dualmono=true" if channels == 1 else "")
        ebur, sink = self.graph.add("ebur128", opts), self.graph.add("abuffersink")
        src.link_to(ebur)
        ebur.link_to(sink)
        self.graph.configure()
        self.n = 0
        self.meta: dict = {}

    def add(self, x: np.ndarray) -> None:
        """`x`: (channels, n) float, or (n,) for mono."""
        import av

        x = np.ascontiguousarray(np.reshape(x, (self.channels, -1)), np.float32)
        if not x.shape[1]:
            return
        frame = av.AudioFrame.from_ndarray(x, format="fltp", layout="mono" if self.channels == 1 else "stereo")
        frame.sample_rate, frame.pts, frame.time_base = self.sr, self.n, Fraction(1, self.sr)
        self.n += x.shape[1]
        self.graph.push(frame)
        self._pull()

    def _pull(self) -> None:
        import av

        while True:
            try:
                out = self.graph.pull()
            except (av.BlockingIOError, av.EOFError):
                return
            self.meta = dict(out.metadata) or self.meta

    def result(self) -> dict:
        """{"I": LUFS, "LRA": LU, "TP": dBTP or None}, once everything is added."""
        self.graph.push(None)
        self._pull()
        m = self.meta
        tp = float(m["lavfi.r128.true_peak"]) if "lavfi.r128.true_peak" in m else None
        return {"I": float(m.get("lavfi.r128.I", -70.0)), "LRA": float(m.get("lavfi.r128.LRA", 0.0)),
                "TP": round(20 * math.log10(tp), 2) if tp else None}


def _read(line: Line, a: int, b: int) -> np.ndarray:
    """Samples [a, b) of a line's PCM, as float32."""
    return np.asarray(np.load(line.pcm, mmap_mode="r", allow_pickle=False)[a:b], np.float32)


def speaker_gains(lines: Iterable[Line], sample_rate: int) -> dict[str, float]:
    """Each speaker's gain in dB: VOICE_REF minus the integrated loudness of their lines' PCM one after another, clamped
    to VOICE_CLAMP; 0 for a speaker with nothing loud enough to measure."""
    by: dict[str, list[Line]] = {}
    for line in lines:
        by.setdefault(line.speaker, []).append(line)
    gains = {}
    for sid, got in sorted(by.items()):
        meter = Loudness(sample_rate, 1)
        for line in got:
            meter.add(_read(line, 0, line.samples))
        level = meter.result()["I"]
        gains[sid] = 0.0 if level <= SILENT else float(np.clip(VOICE_REF - level, -VOICE_CLAMP, VOICE_CLAMP))
    return gains


@functools.lru_cache(maxsize=None)
def resampler_delay(rate_in: int, rate_out: int = MIX_SR) -> int:
    """Output samples by which a streaming PyAV resampler from `rate_in` to `rate_out` puts an impulse late, measured
    once: what `Mix` drops from the front so a click lands on its planned sample (0 with PyAV 18.1.0's aresample)."""
    x = np.zeros(2 * rate_in, np.float32)
    x[rate_in] = 1.0
    y = np.concatenate(list(_resample([x], rate_in, rate_out)))
    return int(np.argmax(np.abs(y))) - rate_out


def _resample(blocks: Iterable[np.ndarray], rate_in: int, rate_out: int) -> Iterator[np.ndarray]:
    """Mono float32 blocks at `rate_in` through one streaming resampler: its output as it comes, then its flush."""
    import av

    r = av.AudioResampler(format="flt", layout="mono", rate=rate_out)
    n = 0
    for x in blocks:
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(x, np.float32)[None], format="flt", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = rate_in, n, Fraction(1, rate_in)
        n += len(x)
        for out in r.resample(frame):
            yield out.to_ndarray().reshape(-1)
    for out in r.resample(None):
        yield out.to_ndarray().reshape(-1)


class _Fifo:
    """Samples in, blocks of exact sizes out."""

    def __init__(self, channels: int) -> None:
        self.parts: list[np.ndarray] = []
        self.n = 0
        self.channels = channels

    def put(self, x: np.ndarray) -> None:
        if x.shape[-1]:
            self.parts.append(x)
            self.n += x.shape[-1]

    def take(self, n: int) -> np.ndarray:
        """`n` samples, padded with silence when it holds fewer."""
        x = np.concatenate(self.parts, axis=-1) if self.parts else np.zeros((self.channels, 0), np.float32)
        out, rest = x[..., :n], x[..., n:]
        self.parts, self.n = ([rest] if rest.shape[-1] else []), rest.shape[-1]
        if out.shape[-1] < n:
            out = np.concatenate([out, np.zeros((*out.shape[:-1], n - out.shape[-1]), np.float32)], axis=-1)
        return out


class Mix:
    """The mix of `lines` (sorted or not) over [0, `end`) s of the output, in BLOCK s blocks of MIX_SR stereo: each line's
    PCM ADDED at its start (lines of different speakers may overlap), times its speaker's gain (`gains`, dB), resampled
    from `voice_sr` by one streaming resampler (its delay compensated), on both channels, plus `bed` when there is one.
    `blocks()` can run again (the two passes)."""

    def __init__(self, lines: Sequence[Line], end: float, voice_sr: int, gains: dict[str, float] | None = None,
                 bed: Bed | None = None) -> None:
        self.lines = sorted(lines, key=lambda x: x.start)
        self.end, self.voice_sr, self.bed = end, voice_sr, bed
        self.gains = {sid: 10 ** (g / 20) for sid, g in (gains or {}).items()}
        self.total = int(round(end * MIX_SR))  # output samples
        self.block = int(round(BLOCK * MIX_SR))

    def _voice(self) -> Iterator[np.ndarray]:
        """The Telugu track at the voices' rate, in BLOCK s blocks: only the lines that touch a block are read."""
        step = int(round(BLOCK * self.voice_sr))
        total = int(math.ceil(self.end * self.voice_sr))
        k = 0  # the first line that may touch the block
        for a in range(0, total, step):
            b = min(a + step, total)
            out = np.zeros(b - a, np.float32)
            while k < len(self.lines) and int(round(self.lines[k].start * self.voice_sr)) + self.lines[k].samples <= a:
                k += 1
            for line in self.lines[k:]:
                s = int(round(line.start * self.voice_sr))
                if s >= b:
                    break
                lo, hi = max(a, s), min(b, s + line.samples)
                if hi > lo:
                    np.add(out[lo - a:hi - a], self.gains.get(line.speaker, 1.0) * _read(line, lo - s, hi - s),
                           out=out[lo - a:hi - a])
            yield out

    def blocks(self) -> Iterator[np.ndarray]:
        """The mix in blocks of BLOCK s (the last shorter), (2, n) float32 at MIX_SR, before the master gain."""
        fifo = _Fifo(1)
        skip = resampler_delay(self.voice_sr)
        done = 0
        for y in _resample(self._voice(), self.voice_sr, MIX_SR):
            if skip > 0:
                y, skip = y[skip:], max(skip - len(y), 0)
            fifo.put(y[None])
            while fifo.n >= self.block and done < self.total:
                yield self._out(done, fifo.take(min(self.block, self.total - done)))
                done += min(self.block, self.total - done)
        while done < self.total:
            n = min(self.block, self.total - done)
            yield self._out(done, fifo.take(n))
            done += n

    def _out(self, a: int, voice: np.ndarray) -> np.ndarray:
        out = np.repeat(voice, 2, axis=0)
        if self.bed is not None:
            out += self.bed(a, a + out.shape[1])
        return out


def master(blocks: Iterable[np.ndarray], gain_db: float) -> Iterator[np.ndarray]:
    """The mix's master: one static gain, then FFmpeg's alimiter (LIMITER: a LIMIT_DBFS ceiling, no make-up gain, no
    delay). Yields blocks of the sizes it is given, each once the limiter has let it out."""
    import av
    from av.filter import Graph

    g = Graph()
    src = g.add_abuffer(format="fltp", sample_rate=MIX_SR, layout="stereo", time_base=Fraction(1, MIX_SR))
    lim, fmt, sink = g.add("alimiter", LIMITER), g.add("aformat", "sample_fmts=fltp"), g.add("abuffersink")
    src.link_to(lim)
    lim.link_to(fmt)
    fmt.link_to(sink)
    g.configure()
    fifo, sizes, scale, n = _Fifo(2), [], 10 ** (gain_db / 20), 0

    def pull() -> None:
        while True:
            try:
                out = g.pull()
            except (av.BlockingIOError, av.EOFError):
                return
            fifo.put(out.to_ndarray().astype(np.float32, copy=False))

    for x in blocks:
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(x * scale, np.float32), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = MIX_SR, n, Fraction(1, MIX_SR)
        n += x.shape[1]
        sizes.append(x.shape[1])
        g.push(frame)
        pull()
        while sizes and fifo.n >= sizes[0]:
            yield fifo.take(sizes.pop(0))
    g.push(None)
    pull()
    while sizes:
        yield fifo.take(sizes.pop(0))


# ---- the bed (§2.15) -------------------------------------------------------------------------------------------------
def _union(spans: Iterable[tuple[float, float]], bridge: float = 0.0) -> list[tuple[float, float]]:
    """The union of `spans`, sorted, with gaps under `bridge` s closed."""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a < out[-1][1] + bridge:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _minus(spans: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of `spans` outside `cut` (both sorted unions)."""
    out = []
    for a, b in spans:
        for c, d in cut:
            if d <= a or c >= b:
                continue
            if c > a:
                out.append((a, c))
            a = max(a, d)
        if b > a:
            out.append((a, b))
    return out


def duck_spans(lines: Spans, speech: Spans) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The ducking's spans (s, on the clock of `lines` and `speech`): DUCK_DB's, the union of the Telugu line spans and
    the English speech with gaps under BRIDGE bridged; DUCK_BARE_DB's, the English speech (bridged likewise) with no
    Telugu line within BARE_GAP, in pieces of at least BARE_MIN s (skipped or untranslated lines: nothing masks what
    the separator left of the English there). A shorter tail after a Telugu line stays at DUCK_DB."""
    duck = _union(list(lines) + list(speech), BRIDGE)
    near = _union((a - BARE_GAP, b + BARE_GAP) for a, b in lines)
    bare = [(a, b) for a, b in _minus(_union(speech, BRIDGE), near) if b - a >= BARE_MIN]
    return duck, bare


def _ramps(spans: list[tuple[float, float]], a: int, n: int, sr: int) -> np.ndarray:
    """How far into its spans each sample of [a, a + n) (at `sr`) is: 1 inside one, a raised cosine from 0 to 1 over the
    ATTACK s before it and from 1 to 0 over the RELEASE s after it, 0 elsewhere (the highest where they meet)."""
    import bisect

    r = np.zeros(n, np.float64)
    t0, t1 = a / sr, (a + n) / sr
    starts = [x for x, _ in spans]
    for s, e in spans[max(0, bisect.bisect_left([y for _, y in spans], t0 - RELEASE) - 1):
                      bisect.bisect_right(starts, t1 + ATTACK)]:
        for lo, hi, f in ((s - ATTACK, s, lambda t: 0.5 - 0.5 * np.cos(np.pi * (t - (s - ATTACK)) / ATTACK)),
                          (s, e, lambda t: np.ones_like(t)),
                          (e, e + RELEASE, lambda t: 0.5 + 0.5 * np.cos(np.pi * (t - e) / RELEASE))):
            i, j = max(0, math.ceil(lo * sr) - a), min(n, math.ceil(hi * sr) - a)
            if j > i:
                np.maximum(r[i:j], f((a + np.arange(i, j)) / sr), out=r[i:j])
    return r


def envelope(duck: list[tuple[float, float]], bare: list[tuple[float, float]], a: int, n: int,
             sr: int = MIX_SR) -> np.ndarray:
    """The bed's gain (linear) for samples [a, a + n) at `sr`: DUCK_DB times how far into a `duck` span, plus
    DUCK_BARE_DB - DUCK_DB times how far into a `bare` one (bare spans lie inside duck spans, so each ramp is exact)."""
    db = DUCK_DB * _ramps(duck, a, n, sr) + (DUCK_BARE_DB - DUCK_DB) * _ramps(bare, a, n, sr)
    return (10 ** (db / 20)).astype(np.float32)


def bed_gain(lines: Iterable[Line], gains: dict[str, float], vocals_ms: float | None) -> float:
    """The bed's gain in dB that puts the Telugu voices over it as the English speech sat over it (§2.15, M11): the
    voices' mean square over their line spans (each at its speaker's gain, dB) against the original vocals' mean square
    over the speech turns (`vocals_ms`, from the separator's vocals envelope), within BED_CLAMP; 0 when either is
    unknown or silent."""
    energy = count = 0.0
    for line in lines:
        x = _read(line, 0, line.samples).astype(np.float64)
        energy += 10 ** (gains.get(line.speaker, 0.0) / 10) * float(np.dot(x, x))
        count += line.samples
    if not count or not energy or not vocals_ms or vocals_ms <= 0:
        return 0.0
    return float(np.clip(10 * math.log10(energy / count / vocals_ms), -BED_CLAMP, BED_CLAMP))


@dataclass(frozen=True)
class Background:
    """The bed under the voices (§2.14, §2.15) on the dub's clock: `blocks` files `<folder>/<k:05d>.npy` (float16
    [2, m] at MIX_SR, `block` samples each from sample 0, the last maybe shorter), silence past them; the original
    vocals' mean square over the speech turns (`vocals_ms`, None when unknown); and the ducking's spans (s)."""

    folder: Path
    block: int
    blocks: int
    vocals_ms: float | None
    duck: tuple[tuple[float, float], ...]
    bare: tuple[tuple[float, float], ...]

    def read(self, a: int, b: int) -> np.ndarray:
        """The bed's samples [a, b) on the dub's clock, (2, b - a) float32: zeros before 0 and past the last block."""
        out = np.zeros((2, b - a), np.float32)
        for k in range(max(a, 0) // self.block, min(self.blocks, -(-b // self.block)) if b > 0 else 0):
            x = np.load(self.folder / f"{k:05d}.npy", mmap_mode="r", allow_pickle=False)
            s = k * self.block
            lo, hi = max(a, s), min(b, s + x.shape[1])
            if hi > lo:
                out[:, lo - a:hi - a] = x[:, lo - s:hi - s]
        return out

    def on(self, off: float, gain_db: float) -> Bed:
        """The bed as the mix reads it, on the output's clock (dub time t plays at t + `off`), times `gain_db` (the
        balance) and the ducking envelope."""
        shift, g = int(round(off * MIX_SR)), 10 ** (gain_db / 20)
        duck, bare = list(self.duck), list(self.bare)

        def bed(a: int, b: int) -> np.ndarray:
            return self.read(a - shift, b - shift) * (g * envelope(duck, bare, a - shift, b - a))

        return bed


def mix_gain(level: float) -> float:
    """The static gain in dB that takes a mix measured at `level` LUFS to MIX_LUFS; 0 for a silent mix."""
    return 0.0 if level <= SILENT else MIX_LUFS - level
