"""DubTimeline: all audio placements and video edits, planned ahead of the playhead (§6.6).

`wall_time(v)` maps video time to "presentation" seconds from video t=0, accounting for
every slow-down and freeze planned so far. The UI uses the edits; tests use the mapping to
check that no two dub lines overlap in what the viewer actually hears.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .isochrony import EditKind, Placement, VideoEdit


@dataclass
class DubTimeline:
    placements: dict[int, Placement] = field(default_factory=dict)
    _starts: list[float] = field(default_factory=list)
    _ids: list[int] = field(default_factory=list)

    def add(self, p: Placement) -> None:
        if p.unit_id in self.placements:
            self.remove(p.unit_id)
        i = bisect.bisect_left(self._starts, p.start)
        self._starts.insert(i, p.start)
        self._ids.insert(i, p.unit_id)
        self.placements[p.unit_id] = p

    def remove(self, unit_id: int) -> None:
        p = self.placements.pop(unit_id)
        i = self._ids.index(unit_id)
        del self._ids[i], self._starts[i]
        assert p is not None

    def ordered(self) -> list[Placement]:
        return [self.placements[i] for i in self._ids]

    def edits(self) -> list[VideoEdit]:
        return sorted((e for p in self.ordered() for e in p.edits), key=lambda e: (e.at, e.kind != EditKind.SLOW))

    def wall_time(self, v: float) -> float:
        """Presentation time at which video time `v` is shown (a freeze at `v` counts as before `v`)."""
        t = v
        for e in self.edits():
            if e.kind is EditKind.SLOW:
                if v <= e.at:
                    continue
                span = min(v, e.at + e.video_span) - e.at
                t += span / e.rate - span
            elif v >= e.at:  # a freeze at v holds the frame before v's content is shown
                t += e.added
        return t

    def audible_intervals(self) -> list[tuple[int, float, float]]:
        """(unit id, start, end) of each line in presentation time."""
        return [(p.unit_id, self.wall_time(p.start), self.wall_time(p.start) + p.audio_wall) for p in self.ordered()]

    def added_between(self, v0: float, v1: float) -> float:
        return self.wall_time(v1) - self.wall_time(v0) - (v1 - v0)
