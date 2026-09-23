"""The §6.6 isochrony cascade: fit a synthesized Telugu line into the original line's slot.

Pure logic. The orchestrator runs steps 1–2 (length-targeted translation, condense) with the
models; this module decides the budget, and steps 3–4 (speed-up, video slow-down, freeze).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

EPS = 1e-6


@dataclass(frozen=True, slots=True)
class TimingSettings:
    borrow_guard: float = 0.15       # s of silence always left before the next line
    borrow_max: float = 0.8          # s of following silence a line may borrow
    condense_threshold: float = 0.05  # condense only if overflow > 5 %
    max_condense_rounds: int = 2
    speed_cap: float = 1.2
    min_audio_rate: float = 0.95     # never slow a line below this (we pad instead)
    allow_slowdown: bool = True
    min_video_rate: float = 0.85
    allow_freeze: bool = True
    max_freeze: float = 1.5          # s per event
    added_budget: float = 4.0        # s of added video time per rolling window
    budget_window: float = 60.0      # s of video time
    target_fill: float = 0.95        # target_aksharas = B × rate × 0.95
    available_video_rates: tuple[float, ...] | None = None  # IFrame player rates, if restricted


def slot_budget(start: float, end: float, next_start: float | None, s: TimingSettings) -> float:
    """B_u = (e_u − s_u) + borrow_u, borrow_u = clamp(gap_u − guard, 0, borrow_max)."""
    gap = float("inf") if next_start is None else max(next_start - end, 0.0)
    borrow = min(max(gap - s.borrow_guard, 0.0), s.borrow_max)
    return (end - start) + borrow


def target_units(budget: float, voice_rate: float, s: TimingSettings) -> float:
    """Step 1: target length for length-aware translation (aksharas)."""
    return budget * voice_rate * s.target_fill


def needs_condense(duration: float, budget: float, s: TimingSettings) -> bool:
    """Step 2 gate: condense only when the overflow exceeds the threshold."""
    return duration > budget * (1 + s.condense_threshold) + EPS


class EditKind(str, Enum):
    SLOW = "slow"
    FREEZE = "freeze"


@dataclass(frozen=True, slots=True)
class VideoEdit:
    kind: EditKind
    at: float             # video time where the edit starts
    video_span: float     # SLOW: video seconds played at `rate`; FREEZE: 0
    rate: float           # SLOW: video rate; FREEZE: 0
    added: float          # wall-clock seconds added versus playing at 1.0×


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a dub line goes. `start` is exactly s_u (invariant)."""

    unit_id: int
    start: float
    budget: float
    audio_rate: float
    audio_wall: float          # wall seconds the (sped-up) line plays for
    pad_after: float           # silence after the line inside its slot
    edits: tuple[VideoEdit, ...] = ()

    @property
    def added(self) -> float:
        return sum(e.added for e in self.edits)


class Verdict(str, Enum):
    PLACED = "placed"
    CONDENSE = "condense"  # overflow can't be absorbed; go back to step 2 with a harder target


@dataclass(frozen=True, slots=True)
class FitResult:
    verdict: Verdict
    placement: Placement | None = None
    condense_target_ratio: float = 1.0  # shrink the line to this fraction of its current length


@dataclass
class AddedTimeLedger:
    """Tracks video time added by edits, to enforce the rolling-window budget."""

    settings: TimingSettings
    events: deque[tuple[float, float]] = field(default_factory=deque)  # (video time, added)

    def added_in_window(self, at: float) -> float:
        lo = at - self.settings.budget_window
        return sum(a for t, a in self.events if lo < t <= at + EPS)

    def can_add(self, at: float, amount: float) -> bool:
        return self.added_in_window(at) + amount <= self.settings.added_budget + EPS

    def record(self, at: float, amount: float) -> None:
        if amount > 0:
            self.events.append((at, amount))


def _pick_video_rate(needed: float, s: TimingSettings) -> float | None:
    """Slowest allowed video rate ≥ needed rate (IFrame players offer discrete rates)."""
    floor = max(needed, s.min_video_rate)
    if s.available_video_rates is None:
        return floor
    candidates = sorted(r for r in s.available_video_rates if floor - EPS <= r < 1.0)
    return candidates[0] if candidates else None


def fit_unit(
    unit_id: int,
    start: float,
    end: float,
    next_start: float | None,
    duration: float,
    settings: TimingSettings,
    ledger: AddedTimeLedger,
    commit: bool = True,
) -> FitResult:
    """Steps 3–4 of the cascade for a line whose synthesized duration is `duration`."""
    s = settings
    budget = slot_budget(start, end, next_start, s)

    if duration <= budget + EPS:  # fits: play at 1.0×, pad the rest; never slow the audio
        return FitResult(Verdict.PLACED, Placement(unit_id, start, budget, 1.0, duration, budget - duration))

    rate = min(duration / budget, s.speed_cap)  # step 3
    wall = duration / rate
    overflow = wall - budget
    if overflow <= EPS:
        return FitResult(Verdict.PLACED, Placement(unit_id, start, budget, rate, wall, 0.0))

    edits: list[VideoEdit] = []
    remaining = overflow
    if s.allow_slowdown:  # step 4a: stretch the unit's video span to cover the audio
        vr = _pick_video_rate(budget / wall, s)
        if vr is not None and vr < 1.0 - EPS:
            # vr ≥ budget/wall, so the gain never exceeds the overflow.
            gained = budget / vr - budget
            edits.append(VideoEdit(EditKind.SLOW, start, budget, vr, gained))
            remaining -= gained
    remaining = max(remaining, 0.0)
    if remaining > EPS and s.allow_freeze:  # step 4b
        freeze = min(remaining, s.max_freeze)
        edits.append(VideoEdit(EditKind.FREEZE, start + budget, 0.0, 0.0, freeze))
        remaining -= freeze

    added = sum(e.added for e in edits)
    if remaining > EPS or not ledger.can_add(start, added):
        return FitResult(Verdict.CONDENSE, condense_target_ratio=budget * s.speed_cap / duration * 0.97)

    # Slack from a coarse video rate: the slot may end after the audio does; pad it.
    pad = max(budget + added - wall, 0.0)
    placement = Placement(unit_id, start, budget, rate, wall, pad, tuple(edits))
    if commit:
        ledger.record(start, added)
    return FitResult(Verdict.PLACED, placement)
