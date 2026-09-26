"""Playback pacing (spec §6.8, ADR-016; ARCHITECTURE §3.12, §5.2): which video time is dubbed, how fast the engine is
going, how much lead the UI should bank before playing so it never has to stop, and how many takes a line can afford.

Pure logic. With throughput x (video seconds dubbed per wall second) and R seconds left to play, playing
at 1.0x drains the lead at (1 - x) per second, so a lead of R * (1 - x) plays to the end without a stall. Banking it
while the video waits takes that lead / x of wall time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

EPS = 1e-6


@dataclass
class ReadyRanges:
    """Merged, sorted [start, end) ranges of video time whose dub is complete."""

    ranges: list[list[float]] = field(default_factory=list)

    def add(self, a: float, b: float) -> None:
        if b <= a + EPS:
            return
        out: list[list[float]] = []
        placed = False
        for s, e in self.ranges:
            if e < a - EPS or s > b + EPS:  # disjoint (touching ranges merge)
                if not placed and s > b + EPS:
                    out.append([a, b])
                    placed = True
                out.append([s, e])
            else:
                a, b = min(a, s), max(b, e)
        if not placed:
            out.append([a, b])
        self.ranges = sorted(out)

    def contiguous_end(self, t: float, tolerance: float = 0.25) -> float:
        """End of the dubbed range containing t (or starting within `tolerance` after it); t if none."""
        for s, e in self.ranges:
            if s - tolerance <= t < e:
                return e
        return t

    def lead(self, t: float) -> float:
        return max(0.0, self.contiguous_end(t) - t)

    def covered(self, a: float, b: float) -> bool:
        return any(s - EPS <= a and b <= e + EPS for s, e in self.ranges)


@dataclass
class Throughput:
    """Video seconds dubbed per wall second of work, over the recent `horizon` of it. Wall time the pipeline spends
    caught up (everything up to where it stops is dubbed, and it waits for the playhead to move) is left out: it says
    nothing of the engine's speed, and counting it would make a paused video look like a slow engine."""

    horizon: float = 180.0
    min_span: float = 20.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)  # (working time, cumulative video seconds)
    total: float = 0.0
    idle: float = 0.0                  # wall seconds left out so far
    idle_since: float | None = None    # wall time the pipeline caught up, while it is

    def reset(self) -> None:
        self.samples.clear()

    def pause(self, wall: float) -> None:
        """The pipeline has caught up: wall time from now on isn't work, until `resume`."""
        if self.idle_since is None:
            self.idle_since = wall

    def resume(self, wall: float) -> None:
        if self.idle_since is not None:
            self.idle += max(wall - self.idle_since, 0.0)
            self.idle_since = None

    def working(self, wall: float) -> float:
        """`wall` on the clock that runs only while the pipeline works."""
        return wall - self.idle - (max(wall - self.idle_since, 0.0) if self.idle_since is not None else 0.0)

    def record(self, wall: float, video_seconds: float) -> None:
        self.total += max(video_seconds, 0.0)
        w = self.working(wall)
        self.samples.append((w, self.total))
        while len(self.samples) > 2 and w - self.samples[1][0] > self.horizon:
            self.samples.popleft()

    def rate(self) -> float:
        """0.0 until at least `min_span` seconds of work have been observed."""
        if len(self.samples) < 2:
            return 0.0
        (w0, v0), (w1, v1) = self.samples[0], self.samples[-1]
        return (v1 - v0) / (w1 - w0) if w1 - w0 >= self.min_span else 0.0


def target_lead(remaining: float, throughput: float, margin: float = 30.0, floor: float = 60.0) -> float:
    """Lead (video seconds) to bank before playing so playback reaches the end without stopping.

    Unknown throughput (0) returns 0: the UI then falls back to the user's "prepare ahead" setting.
    """
    remaining = max(remaining, 0.0)
    if throughput <= 0:
        return 0.0
    if throughput >= 1.0:
        need = floor  # the engine outruns playback; keep a small cushion for slow lines
    else:
        need = remaining * (1.0 - throughput) + margin
    return min(remaining, need)


# The throughput governor (ARCHITECTURE §5.2): takes per line, from r and the lead.
SLOW_R = 1.1        # below this r, with the lead under target, a line gets one take (a second only if it fails)
SHORT_LINE = 3.0    # s of speech: with the lead comfortable, a line this short gets three takes (they cost little)


def takes_for(lead: float, target: float, throughput: float, speech_s: float, recovering: bool = False) -> int:
    """How many takes of a line to synthesize in one batched decode. One while the lead is under `target` and the engine
    is barely faster than playback (r < 1.1, or not measured yet), and after a seek into video not processed yet until
    the lead is back at target (`recovering`); two otherwise, and three for a short line once the lead is comfortable.
    Never more: three takes one after another fall behind playback (research gap-1 §4, scenario C)."""
    if recovering or (lead < target and throughput < SLOW_R):
        return 1
    if lead < target:
        return 2
    return 3 if speech_s < SHORT_LINE else 2
