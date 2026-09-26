"""Fixed-timeline planner (architect §3; ARCHITECTURE §3.10): place each Telugu line on the video's own clock.

Pure logic (ADR-005). It replaces the ADR-016 cascade (speed-up, video slow-down, freeze) as the
normal path. The video plays at 1.0×. Each dub line starts at its English onset and is fitted by:
- a small in-vocoder speed-up, never above its ceiling (`speed_cap`, and the voice's akshara ceiling) and never faster
  than its window needs; a take's internal pauses shorten before its speech speeds up (`pauses.squeeze`);
- an early start into silence before it, at most `lead_max`; starting early costs twice what starting late does;
- letting its end, and so the next line's onset, slip a little.
Human dubbers keep their rate steady and let timing slip (Brannon et al., TACL 2023), so speeding
a line up beyond the speaker's last rate costs extra. Nothing speeds a line up to match its
neighbours: the rate is a time compression on top of each take's own pace, and a line that fits
plays at 1.0. A freeze is a metered last resort. Whatever the limits can't absorb is reported as
overdraft: audio is never cut.

A sentence is placed whole over its span and drifts across the English pauses inside it (§3.10 step 1), with two
exceptions. At a hard break (a pause of 1 s or more, `LineSlot.breaks`) a line said as the translator's pieces, one
take each, is placed piece by piece, each at its own English onset: the Telugu pause falls where Telugu breaks, and the
silence there meets the English pause. At a soft anchor (a 0.3 s breath at a clause mark) a line that fits at rate 1
waits in a pause of its own take for the English to resume, when that pause falls near the anchor (duration match ×
break plausibility) and there is room before the next line.

`place()` is a rolling horizon. It commits one line given the committed chain and a lookahead: the next onset, the next
line's slot and predicted duration, and (windowed, §3.10 step 5) the lines after it, about 8 or 30 s, so slack flows
across gaps. Lines normally come in onset order; one placed behind lines already placed further on (a seek back into
video dubbed only in part) is planned against the lines placed before it, and the line after a run of such lines
against those lines too. `evaluate()` prices a duration the same way without committing it, to choose between text
candidates.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from . import pauses as pz
from .metrics import length, pct, speech_metrics, union

EPS = 1e-9
INF = math.inf
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
Span = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    speed_cap: float = 1.2        # max audio rate (in-vocoder time compression); never exceeded
    # aksharas/s no line is sped past: a voice's ceiling is this ÷ its natural pace (§3.10). Off until measured on Telugu:
    # §3.10's 7.5 came from Kannada read speech, and on the M5 Pro (2026-09-25) a calibrated voice's pace of 8.45 (its fit
    # held at the 1.0 s overhead limit) put the ceiling below 1.0, so no line could speed up and onset lag p95 hit 0.63 s.
    akshara_ceiling: float = INF
    lead_max: float = 0.3         # a line may start up to this early when there is silence before it
    guard: float = 0.15           # silence kept before the next line / after the previous dub
    max_lag: float = 0.6          # normal max onset delay
    max_lag_resync: float = 1.0   # allowed if a gap >= resync_gap follows within resync_window seconds
    resync_gap: float = 1.0
    resync_window: float = 5.0
    max_freeze: float = 0.6       # last-resort freeze length per event
    min_freeze: float = 0.15      # a shorter hold stutters on the IFrame player: it stays as overdraft instead
    freeze_budget: float = 1.0    # seconds of freeze per 60 s of video (rolling)
    smooth: float = 0.10          # a speaker's rate may rise this much above their last line's before it costs extra
    smooth_window: float = 10.0   # a speaker's pace carries over pauses up to this long (s)
    min_gap: float = 0.1          # hard silence between a dub and the next one (unless the source overlapped)
    target_borrow: float = 0.5    # target_seconds() borrows at most this much of the following silence
    shorter_tol: float = 0.3      # needs_shorter when the dub still delays a later line by more than this (s)
    budget_window: float = 60.0   # s of video the freeze budget rolls over
    late_after: float = 0.5       # stats: a line starting later than this counts as late
    # Onset error per s²: early audio is noticed at about half the offset of late audio (same-language lip-sync studies,
    # research 08 F11; inferred for dubs), so starting early costs twice as much.
    w_early: float = 2.0
    w_late: float = 1.0           # this line's lateness, and the next line's when this one pushes it
    w_rate: float = 4.0           # cost of speed-up, per (rate − 1)²: rate 1.1 costs as much as 0.2 s lag
    w_smooth: float = 4.0         # cost of a same-speaker rate rise beyond `smooth`
    w_over: float = 10.0          # evaluate(): cost per s² of running past the hard limit (freeze or overdraft)
    window_lines: int = 8         # the lookahead prices up to this many lines after the one placed...
    window_s: float = 30.0        # ...that start within this many seconds of it
    keep_pause: float = 0.12      # s each internal pause of a take keeps when the take is compressed
    piece_gap: float = 0.3        # natural s between pieces said as one line when they can't meet their breaks
    anchor_max: float = 0.8       # s: a soft anchor's duration match falls to 0 at this much silence to insert
    anchor_score: float = 0.5     # duration match × break plausibility an anchor needs to be used
    anchor_min: float = 0.05      # s: less silence to insert isn't worth splitting a take for


# The planner as it was before timing v2 (ADR-017's amendment of 2026-09-25), for the baseline run that v2's targets are
# measured against (`maata-bench pipeline --timing v1`): early and late onsets cost the same, a one-line lookahead, no
# akshara ceiling, uniform compression (no pause shortens first) and no soft anchors. The session also leaves out the
# pieces at hard breaks and the diarized speech before a line's onset.
V1 = {"w_early": 1.0, "window_lines": 1, "akshara_ceiling": INF, "keep_pause": INF, "anchor_score": INF}


@dataclass(frozen=True, slots=True)
class LineSlot:
    id: int
    speaker: str
    start: float                   # source speech span (video seconds)
    end: float
    next_start: float | None       # next line's onset (any speaker)
    prev_end: float | None         # latest source end among earlier lines (any speaker), or of diarized speech
    overlaps_prev: bool = False    # source speech overlapped the previous speaker's line
    resume_start: float | None = None  # first onset at or after `end` (any speaker); inf if none; None if unknown
    rate_cap: float | None = None  # this line's speed ceiling (`rate_ceiling` of its voice); None: speed_cap
    breaks: tuple[Span, ...] = ()  # hard breaks inside it: (where an English pause of 1 s or more starts, where it ends)
    heard: tuple[Span, ...] = ()   # other speech inside its hard breaks: a piece after one starts early only into silence
    anchors: tuple[Span, ...] = ()  # soft anchors inside it: the same, for breaths of 0.3 s or more at a clause mark
    speech: tuple[Span, ...] = ()  # the source speaker's diarized speech inside the line (speech-level metrics)


@dataclass(frozen=True, slots=True)
class Part:
    """A stretch of a line said on its own: a piece (its own take) at a hard break, or a take split at a soft anchor."""

    take: int              # which take: the piece's index, or 0 for a take split at soft anchors
    a: float               # the stretch of that take it says (natural seconds)
    b: float
    start: float           # video time it starts
    rate: float
    wall: float            # seconds it plays for
    src: float | None      # the English onset it is timed against; None when it only follows the part before it
    src_end: float         # where its English ends


@dataclass(frozen=True, slots=True)
class Plan:
    id: int
    start: float                   # when the dub starts (video time)
    rate: float                    # audio time-compression factor in [1.0, speed_cap] (over its parts, their mean)
    wall: float                    # seconds the dub plays for (duration / rate, and any silence set between parts)
    lag: float                     # start - source start (negative = early)
    freeze: float = 0.0            # video hold while the dub plays (last resort), seconds
    overdraft: float = 0.0         # seconds the next line is pushed past its lag limit after all limits (never trimmed)
    needs_shorter: bool = False    # a shorter text would help: it needed a freeze or overdraft, or delays the next line
    freeze_at: float | None = None # video time the frame is held at: the pause before the next line, within the dub
    carried: float = 0.0           # part of `overdraft` that only passes on lateness from earlier lines' overdraft
    excess: float = 0.0            # seconds the dub runs past its window (next onset − guard) before any freeze
    parts: tuple[Part, ...] = ()   # a line said in parts (at hard breaks or soft anchors); () when said in one go
    voiced: tuple[Span, ...] = ()  # video time the dub is voiced (its takes' pauses left out)
    # How it is said: "whole" (one take over its span), "pieces" (one take per piece, each at its own English onset),
    # "joined" (the pieces said as one line) or "anchored" (one take that waits at soft anchors).
    said: str = "whole"

    @property
    def end(self) -> float:
        """Video time at which the dub ends (a freeze lets the video wait for it)."""
        return self.start + self.wall - self.freeze

    @property
    def own_overdraft(self) -> float:
        """Overdraft this line's own audio caused."""
        return self.overdraft - self.carried

    @property
    def played(self) -> float:
        """Seconds of take played, leaving out any silence set between parts."""
        return sum(p.wall for p in self.parts) if self.parts else self.wall


Take = tuple[float, tuple[Span, ...]]  # a take's natural seconds and its pauses (`pauses.pauses`)


def target_seconds(slot: LineSlot, s: PlannerSettings = PlannerSettings()) -> float:
    """Duration translation and TTS aim for: the source span plus a small borrow of the silence after it."""
    gap = INF if slot.next_start is None else slot.next_start - slot.end
    return (slot.end - slot.start) + min(max(gap - s.guard, 0.0), s.target_borrow)


def lag_limit(slot: LineSlot, s: PlannerSettings = PlannerSettings()) -> float:
    """How late this line may start: `max_lag`, or `max_lag_resync` if a long gap soon lets it re-sync."""
    gap = INF if slot.next_start is None else slot.next_start - slot.end
    if gap >= s.resync_gap - EPS and slot.end - slot.start <= s.resync_window + EPS:
        return s.max_lag_resync
    return s.max_lag


def rate_ceiling(voice_rate: float | None, s: PlannerSettings = PlannerSettings()) -> float:
    """A voice's speed ceiling: `speed_cap`, and no faster than `akshara_ceiling` aksharas/s for a voice whose natural
    pace is `voice_rate` aksharas/s (§3.10 step 6); never under 1.0 (the planner never slows a line)."""
    if not voice_rate or voice_rate <= 0:
        return s.speed_cap
    return max(1.0, min(s.speed_cap, s.akshara_ceiling / voice_rate))


def _pos(v: float) -> float:
    return v if v > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class _Dub:
    id: int
    speaker: str
    src_end: float   # where the source speech ended
    end: float       # where the dub ends, in video time after any freeze


@dataclass
class _Chain:
    """What the committed dubs constrain."""

    dubs: list[_Dub] = field(default_factory=list)   # committed dubs that may still bound a later start
    last_id: int | None = None
    key: tuple[float, int] | None = None             # (onset, id) of the last line committed; None since a reset
    end_by_speaker: dict[str, float] = field(default_factory=dict)
    rate_by_speaker: dict[str, float] = field(default_factory=dict)
    freezes: list[tuple[float, float]] = field(default_factory=list)  # (video time of the hold, seconds)

    def copy(self) -> _Chain:
        return replace(self, dubs=list(self.dubs), end_by_speaker=dict(self.end_by_speaker),
                       rate_by_speaker=dict(self.rate_by_speaker), freezes=list(self.freezes))


@dataclass(frozen=True, slots=True)
class _Horizon:
    soft: float      # the dub should end by here: next onset − guard (or the source end, if the next line overlaps it)
    hard: float      # past here a later line starts later than its lag limit
    free: float      # ending past here delays a later line at all
    slack: float     # how far past `soft` the next line can absorb for free
    w_knock: float   # cost per s² of pushing the next line past its slack


def _argmin_pl(h, a: float, b: float, breaks: Iterable[float]) -> float:
    """Minimiser over [a, b] of a convex function whose derivative `h` is piecewise linear
    and increasing, with kinks only at `breaks`."""
    if b <= a or h(a) >= 0.0:
        return a
    if h(b) <= 0.0:
        return b
    left, right = a, b
    for p in sorted(q for q in breaks if a < q < b):
        if h(p) >= 0.0:
            right = p
            break
        left = p
    hl, hr = h(left), h(right)
    return left + (right - left) * (-hl) / (hr - hl)


class TimelinePlanner:
    """Commits one line at a time on a fixed timeline. Re-placing the last line replaces it."""

    def __init__(self, s: PlannerSettings = PlannerSettings()) -> None:
        self.s = s
        self._chain = _Chain()
        self._undo: tuple[int, _Chain] | None = None
        self._placed: dict[int, tuple[LineSlot, Plan]] = {}
        self._keys: list[tuple[float, int]] = []           # (onset, id) of every placed line, sorted

    # --- public -----------------------------------------------------------------------------

    def place(self, slot: LineSlot, duration: float, next_predicted: float | None = None,
              next_slot: LineSlot | None = None, *, ahead: Sequence[tuple[LineSlot, float | None]] | None = None,
              pauses: tuple[Span, ...] = (), marks: tuple[float, ...] = (), pieces: Sequence[Take] = ()) -> Plan:
        """Plan and commit `slot`, whose synthesized audio lasts `duration` s at rate 1.0.

        `next_predicted` is the next line's predicted duration. `next_slot` is the next line's slot:
        with it, the lookahead knows the next line's own window and whether it may re-sync late.
        `ahead` replaces both with the lines after this one, in onset order, each with its predicted duration (None
        while it has no wording): the windowed lookahead (`window`).
        `pauses` are the take's pauses (natural seconds, `pauses.pauses`): where it is voiced, and where it may wait at a
        soft anchor. `marks` are the take's text breaks (a comma or a sentence end inside it), as shares of its
        aksharas: a pause near one is a plausible Telugu break. `pieces`: a line with hard breaks said as the
        translator's pieces, each piece's take as (seconds, pauses); `duration` is then their total.
        Calling again with the id just placed (say, after a shorter re-synthesis) replaces that plan.
        Any other line already placed can't be re-placed until `reset()` drops it.
        """
        base = self._base(slot)
        _, plan = self._evaluate(base, slot, duration, _ahead(next_predicted, next_slot, ahead), pauses, marks, pieces)
        self._chain = base.copy()
        self._commit(slot, plan)
        self._undo = (slot.id, base)
        return plan

    def evaluate(self, slot: LineSlot, duration: float, next_predicted: float | None = None,
                 next_slot: LineSlot | None = None, *, ahead: Sequence[tuple[LineSlot, float | None]] | None = None,
                 pauses: tuple[Span, ...] = (), marks: tuple[float, ...] = (), pieces: Sequence[Take] = ()
                 ) -> tuple[float, Plan]:
        """The plan `place()` would commit for this duration, and its cost, without committing it.

        Lower is better timing: onset error, speed-up, a speaker's rate jump, delay to the next line (and, windowed,
        the lines after it), and running past the hard limit (freeze or overdraft). It says nothing about how complete
        a text is. For the line just placed, it prices a replacement, as `place()` would.
        """
        return self._evaluate(self._base(slot), slot, duration, _ahead(next_predicted, next_slot, ahead), pauses, marks,
                              pieces)

    def reset(self, t: float) -> None:
        """After a seek to video time `t`: forget the chain; lines from `t` on will be placed again."""
        self._chain = _Chain(freezes=[(at, f) for at, f in self._chain.freezes if at < t])
        self._undo = None
        self._placed = {i: sp for i, sp in self._placed.items() if sp[0].start < t}
        self._keys = [k for k in self._keys if k[1] in self._placed]

    def stats(self) -> dict:
        """Onset lag, rate, freeze and overdraft figures over every line placed so far, and the speech-level ones
        (`metrics.speech_metrics`: a line with no speech or pauses given counts its span and all its audio)."""
        rows = sorted(self._placed.values(), key=lambda sp: (sp[0].start, sp[0].id))
        if not rows:
            return {"lines": 0}
        s = self.s
        plans = [p for _, p in rows]
        lags = [p.lag for p in plans]
        rates = [p.rate for p in plans]
        steps: list[float] = []
        last: dict[str, float] = {}
        for sl, p in rows:
            if sl.speaker in last:
                steps.append(abs(p.rate - last[sl.speaker]))
            last[sl.speaker] = p.rate
        # A line pushed late by its predecessor's overdraft is flagged; the rest must meet the limits.
        unflagged = [p.lag for prev, p in zip([None, *plans], plans) if prev is None or prev.overdraft <= EPS]
        minutes = max((max(sl.end for sl, _ in rows) - min(sl.start for sl, _ in rows)) / 60.0, 1e-9)
        freezes = [p.freeze for p in plans if p.freeze > EPS]
        n = len(plans)
        return {
            "lines": n,
            "minutes": _r(minutes),
            "lag_p50": _r(pct(lags, 0.50)),
            "lag_p95": _r(pct(lags, 0.95)),
            "lag_max": _r(max(lags)),
            "lag_min": _r(min(lags)),
            "lag_max_unflagged": _r(max(unflagged)) if unflagged else 0.0,
            "late_share": _r(sum(v > s.late_after + EPS for v in lags) / n),
            "rate_mean": _r(sum(rates) / n),
            "rate_p50": _r(pct(rates, 0.50)),
            "rate_p90": _r(pct(rates, 0.90)),
            "rate_max": _r(max(rates)),
            "at_cap_share": _r(sum(p.rate >= self._cap(sl) - 1e-6 for sl, p in rows) / n),
            "rate_step_p90": _r(pct(steps, 0.90)) if steps else 0.0,
            "rate_step_ok_share": _r(sum(d <= s.smooth + 1e-6 for d in steps) / len(steps)) if steps else 1.0,
            "video_slowdowns": 0,
            "freezes": len(freezes),
            "freeze_s": _r(sum(freezes)),
            "freezes_per_min": _r(len(freezes) / minutes),
            "freeze_s_per_min": _r(sum(freezes) / minutes),
            "freeze_s_per_10min": _r(10.0 * sum(freezes) / minutes),
            "freeze_line_share": _r(len(freezes) / n),
            # Each line counts only the overdraft its own audio caused, not lateness passed down the chain.
            "overdraft_s": _r(sum(p.own_overdraft for p in plans)),
            "overdraft_lines": sum(p.own_overdraft > EPS for p in plans),
            "overdraft_carried_s": _r(sum(p.carried for p in plans)),
            "needs_shorter": sum(p.needs_shorter for p in plans),
            "iou_mean": _r(sum(_iou(sl, p) for sl, p in rows) / n),
            "pieced_lines": sum(p.said == "pieces" for p in plans),
            "joined_lines": sum(p.said == "joined" for p in plans),
            "anchored_lines": sum(p.said == "anchored" for p in plans),
            **speech_metrics(_metric_row(sl, p) for sl, p in rows),
        }

    # --- planning ---------------------------------------------------------------------------

    def _base(self, slot: LineSlot) -> _Chain:
        """The chain to plan `slot` against: the current one, or for the line just placed, the one before it. When the
        current chain doesn't end with the line placed just before `slot` in onset order (`slot` comes behind lines
        placed further on, or follows a run placed behind them), it is rebuilt from the lines placed before `slot`."""
        if self._undo is not None and self._undo[0] == slot.id:
            return self._undo[1]
        if slot.id in self._placed:
            raise ValueError(f"line {slot.id} was placed before the last line; reset({slot.start}) before placing it again")
        key = (slot.start, slot.id)
        i = bisect.bisect_left(self._keys, key)
        if self._chain.key is not None and self._chain.key != (self._keys[i - 1] if i else None):
            return self._rebuild(key)
        return self._chain

    def _rebuild(self, key: tuple[float, int]) -> _Chain:
        """The chain of the lines placed before `key` (onset, id), committed again in onset order. Every placed line's
        freeze stays in it: the video holds there whichever order the lines were placed in."""
        rows = [self._placed[k[1]] for k in self._keys]
        c = _Chain(freezes=[(p.freeze_at, p.freeze) for sl, p in rows
                            if p.freeze_at is not None and (sl.start, sl.id) >= key])
        for sl, p in rows:
            if (sl.start, sl.id) < key:
                self._append(c, sl, p)
        return c

    def _evaluate(self, c: _Chain, slot: LineSlot, duration: float, ahead: tuple, pauses: tuple[Span, ...],
                  marks: tuple[float, ...], pieces: Sequence[Take]) -> tuple[float, Plan]:
        for d in [duration, *(p[0] for p in pieces)]:
            if not math.isfinite(d) or d < 0.0:
                raise ValueError(f"duration must be a finite number >= 0, got {d!r}")
        if len(pieces) >= 2:
            got = self._pieced(c, slot, pieces, ahead) if len(slot.breaks) >= len(pieces) - 1 else None
            return got if got is not None else self._joined(c, slot, pieces, ahead)
        own, knock, plan, hz = self._line(c, slot, duration, ahead)
        plan = self._anchored(slot, plan, duration, pauses, marks, hz)
        return own + knock, replace(plan, voiced=voiced(plan, [(duration, pauses)], self.s.keep_pause))

    def _line(self, c: _Chain, slot: LineSlot, duration: float, ahead: tuple, window: bool = True
              ) -> tuple[float, float, Plan, _Horizon]:
        """One take over the line's span: its own cost (onset, rate, running past the hard limit), the knock-on cost to
        the next line, the plan and its horizon. With a window, a line that doesn't fit is priced against the lines
        after it (`_window`)."""
        s, src = self.s, slot.start
        cap = self._cap(slot)
        nxt_slot, nxt_pred = ahead[0] if ahead else (None, None)
        gap_before = src - slot.prev_end if slot.prev_end is not None else src
        lead = min(s.lead_max, max(gap_before - s.guard, 0.0))
        lo = max(src - lead, self._floor(c, slot))   # earliest start
        x_top = max(lo, src)                         # starting after the onset never helps
        hz = self._horizon(slot, nxt_pred, nxt_slot)
        prev = self._prev_rate(c, slot)

        if duration <= EPS:
            x, r = x_top, 1.0
        elif lo + duration / cap > hz.hard + EPS:
            x, r = lo, cap                           # can't fit even flat out: shrink the overshoot first
        else:
            x, r, fits = self._optimise(src, duration, lo, x_top, hz, prev, cap)
            if window and not fits and len(ahead) > 1 and nxt_slot is not None and nxt_pred is not None:
                x, r = self._window(c, slot, duration, lo, x_top, hz, prev, cap, ahead, x, r)
        own, knock, plan = self._price(c, slot, duration, x, r, hz, prev)
        return own, knock, plan, hz

    def _price(self, c: _Chain, slot: LineSlot, duration: float, x: float, r: float, hz: _Horizon,
               prev: float | None) -> tuple[float, float, Plan]:
        s, src = self.s, slot.start
        wall = duration / r
        end = x + wall
        overshoot = end - hz.hard
        freeze = overdraft = 0.0
        at = None
        if overshoot > EPS:
            f0 = min(overshoot, s.max_freeze, wall)
            # Hold on the pause before the next line, not on the next speaker's first words.
            hold = slot.end if slot.next_start is None else slot.next_start - s.guard
            at = min(max(hold, x), end - f0)
            f = min(f0, s.freeze_budget - self._window_load(c, at, at + s.budget_window))
            if f >= s.min_freeze - EPS and f > EPS:
                freeze = f
            else:
                at = None
            overdraft = max(overshoot - freeze, 0.0)
        # Lateness beyond this line's own limit can only come from an earlier line's overdraft.
        late = x - src - lag_limit(slot, s)
        carried = min(overdraft, late) if late > EPS else 0.0
        delay = _pos(end - freeze - hz.free)          # how late the dub makes a later line start
        needs_shorter = freeze > 0.0 or overdraft > EPS or delay > s.shorter_tol + EPS
        own = self._onset(x - src) + self._rate_cost(r, prev) + s.w_over * _pos(overshoot) ** 2
        return own, self._knock(end, hz), Plan(slot.id, x, r, wall, x - src, freeze, overdraft, needs_shorter, at,
                                               carried, _pos(end - hz.soft))

    def _cap(self, slot: LineSlot) -> float:
        return max(1.0, min(self.s.speed_cap, slot.rate_cap if slot.rate_cap is not None else INF))

    def _floor(self, c: _Chain, slot: LineSlot) -> float:
        """Earliest start the committed dubs allow: `min_gap` after each, except that another
        speaker's dub may be overlapped by as much as the source speech overlapped it. For the
        previous line that is `overlaps_prev`; for older ones (a long line with an interjection
        inside it) it is the source spans. A speaker never talks over itself."""
        floor = -INF
        for d in c.dubs:
            overlap = d.speaker != slot.speaker and (
                slot.overlaps_prev if d.id == c.last_id else d.src_end > slot.start + EPS)
            floor = max(floor, d.end - _pos(d.src_end - slot.start) if overlap else d.end + self.s.min_gap)
        return floor

    def _horizon(self, slot: LineSlot, next_predicted: float | None, next_slot: LineSlot | None) -> _Horizon:
        s, n = self.s, slot.next_start
        if n is None:
            return _Horizon(INF, INF, INF, INF, 0.0)
        limit = lag_limit(next_slot, s) if next_slot is not None else s.max_lag
        inside = n < slot.end - EPS and (
            next_slot is None or (next_slot.overlaps_prev and next_slot.speaker != slot.speaker))
        if inside:
            # Lines starting inside this one may overlap its dub by as much as their source overlapped it,
            # so each starts as late as the dub ends past `slot.end`. The next line's limit covers them all
            # only when no other line starts inside; otherwise use `max_lag`.
            alone = next_slot is not None and (next_slot.next_start is None or next_slot.next_start >= slot.end - EPS)
            inner = limit if alone else min(limit, s.max_lag)
            # The first line from `slot.end` on waits `min_gap` after the dub; its limit is unknown, so `max_lag`.
            if slot.resume_start is not None:
                resume = max(slot.resume_start, slot.end)
            elif alone:
                resume = INF if next_slot.next_start is None else next_slot.next_start
            else:
                resume = None
            if resume is None:  # unknown: it may start right at this line's end
                soft, hard, free = slot.end, slot.end + min(inner, s.max_lag) - s.min_gap, slot.end - s.min_gap
            else:
                soft = min(slot.end, resume - s.guard)
                hard = min(slot.end + inner, resume + s.max_lag - s.min_gap)
                free = min(slot.end, resume - s.min_gap)
        else:
            soft, hard, free = n - s.guard, n + limit - s.min_gap, n - s.min_gap
        if next_predicted is None:
            return _Horizon(soft, hard, free, INF, 0.0)
        d = max(next_predicted, 0.0)
        # Unknown window: assume the next line was sized to fill its slot (no slack at rate 1).
        slack = 0.0
        if next_slot is not None and next_slot.next_start is not None:
            slack = _pos(next_slot.next_start - s.guard - n - d)
        elif next_slot is not None:
            slack = INF
        # Pushing a short next line costs more: it can absorb less by speeding up.
        return _Horizon(soft, hard, free, slack, s.w_rate / max(d, 0.5) ** 2)

    def _prev_rate(self, c: _Chain, slot: LineSlot) -> float | None:
        """The speaker's last rate, unless a long pause has reset their pace."""
        rp = c.rate_by_speaker.get(slot.speaker)
        if rp is not None and slot.start - c.end_by_speaker[slot.speaker] > self.s.smooth_window:
            return None
        return rp

    def _onset(self, dx: float) -> float:
        """Cost of starting `dx` s after the onset (negative: early)."""
        return (self.s.w_early if dx < 0.0 else self.s.w_late) * dx * dx

    def _rate_cost(self, r: float, prev: float | None) -> float:
        """Speed-up costs, and more so past the speaker's last rate + `smooth` (Brannon et al.: let timing
        slip rather than jump the rate). Both terms only resist speeding up; neither ever causes it."""
        s = self.s
        c = s.w_rate * (r - 1.0) ** 2
        if prev is not None:
            c += s.w_smooth * _pos(r - prev - s.smooth) ** 2
        return c

    def _knock(self, e: float, hz: _Horizon) -> float:
        """Cost of ending at `e`: the next line's onset error past `soft`, and the knock-on past its slack."""
        return self.s.w_late * _pos(e - hz.soft) ** 2 + hz.w_knock * _pos(e - hz.soft - hz.slack) ** 2

    def _optimise(self, src: float, duration: float, lo: float, x_top: float, hz: _Horizon,
                  prev: float | None, cap: float) -> tuple[float, float, bool]:
        """min onset(x−src) + rate terms(r) + K(end), end = x + duration/r ≤ hard, r in [1, cap].

        K charges ending past `soft` (the next line's onset error) and past its slack (the knock-on).
        The problem is jointly convex, so a golden-section search over r with the exact best x per r
        finds the optimum. The flag says the line fits at its onset: there was nothing to trade.
        """
        r_min = self._r_min(duration, lo, hz, cap)
        if x_top + duration / r_min <= min(hz.soft, hz.hard):
            return x_top, r_min, True                 # fits at its onset: nothing to trade

        def best(r: float) -> tuple[float, float]:
            span = duration / r
            x = self._best_x(src, span, lo, max(min(x_top, hz.hard - span), lo), hz)
            return self._onset(x - src) + self._rate_cost(r, prev) + self._knock(x + span, hz), x

        a, b = r_min, cap
        c1, c2 = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        f1, f2 = best(c1)[0], best(c2)[0]
        for _ in range(40):
            if f1 <= f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - _GOLDEN * (b - a)
                f1 = best(c1)[0]
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + _GOLDEN * (b - a)
                f2 = best(c2)[0]
        _, r = min((best(r)[0], r) for r in {r_min, (a + b) / 2.0, cap})  # ties: the lower rate
        return best(r)[1], r, False

    def _r_min(self, duration: float, lo: float, hz: _Horizon, cap: float) -> float:
        room = hz.hard - lo
        return 1.0 if room >= duration else (min(duration / room, cap) if room > EPS else cap)

    def _best_x(self, src: float, span: float, xa: float, xb: float, hz: _Horizon) -> float:
        """Exact minimiser over [xa, xb] of onset(x−src) + K(x + span)."""
        s = self.s

        def h(x: float) -> float:  # half the derivative
            e = x + span
            return ((s.w_early if x < src else s.w_late) * (x - src) + s.w_late * _pos(e - hz.soft)
                    + hz.w_knock * _pos(e - hz.soft - hz.slack))

        return _argmin_pl(h, xa, xb, (src, hz.soft - span, hz.soft + hz.slack - span))

    def _window(self, c: _Chain, slot: LineSlot, duration: float, lo: float, x_top: float, hz: _Horizon,
                prev: float | None, cap: float, ahead: tuple, x0: float, r0: float) -> tuple[float, float]:
        """The windowed lookahead (§3.10 step 5) for a line that doesn't fit: besides the one-line optimum (x0, r0),
        rates across [r_min, cap] each with its best start and its latest, are priced with the lines of the window
        placed after them, one by one as `place()` would (`_rollout`), so slack flows across gaps and the rates that
        follow are priced. The cheapest wins; ties go to the one-line optimum, then the lower rate."""
        r_min = self._r_min(duration, lo, hz, cap)
        cands = [(x0, r0)]
        for k in range(5):
            r = r_min + (cap - r_min) * k / 4
            span = duration / r
            xb = max(min(x_top, hz.hard - span), lo)
            cands += [(self._best_x(slot.start, span, lo, xb, hz), r), (xb, r)]
        best, pick = INF, (x0, r0)
        for x, r in dict.fromkeys(cands):
            own, _, plan = self._price(c, slot, duration, x, r, hz, prev)
            cost = own + self._rollout(c, slot, plan, ahead)
            if cost < best - 1e-12:
                best, pick = cost, (x, r)
        return pick

    def _rollout(self, c: _Chain, slot: LineSlot, plan: Plan, ahead: tuple) -> float:
        """What the lines of the window cost once `plan` is committed: each placed in turn with a one-line lookahead,
        its own cost counted, and the last one's knock-on to the line after it. It stops at a line with no slot or
        no predicted duration yet."""
        c = c.copy()
        self._append(c, slot, plan)
        total, knock = 0.0, 0.0
        for j, (sl, d) in enumerate(ahead):
            if sl is None or d is None:
                break
            own, knock, p, _ = self._line(c, sl, d, ahead[j + 1:j + 2], window=False)
            total += own
            self._append(c, sl, p)
        return total + knock

    # --- parts: pieces at hard breaks, pauses at soft anchors --------------------------------

    def _pieced(self, c: _Chain, slot: LineSlot, pieces: Sequence[Take], ahead: tuple) -> tuple[float, Plan] | None:
        """The pieces placed one by one, each over its own stretch of English between the breaks the Telugu breaks
        meet (`_breaks_for`), with the next piece as its lookahead; the last with the line's. A piece after a break
        starts early only into the silence left there by other speakers (`LineSlot.heard`). None when a piece before
        the last runs past its lag limit (it would need a freeze or overdraft in the middle of the sentence): then the
        pieces are said as one line (`_joined`)."""
        n = len(pieces)
        chosen = self._breaks_for(slot, [d for d, _ in pieces])
        spans = [(slot.start if k == 0 else chosen[k - 1][1], slot.end if k == n - 1 else chosen[k][0]) for k in range(n)]
        before = [slot.prev_end, *(max([spans[k - 1][1], *(e for h, e in slot.heard if h < spans[k][0])])
                                   for k in range(1, n))]
        subs = [LineSlot(slot.id, slot.speaker, a, b,
                         next_start=spans[k + 1][0] if k + 1 < n else slot.next_start,
                         prev_end=before[k],
                         overlaps_prev=slot.overlaps_prev if k == 0 else False,
                         resume_start=spans[k + 1][0] if k + 1 < n else slot.resume_start,
                         rate_cap=slot.rate_cap)
                for k, (a, b) in enumerate(spans)]
        cw, plans, cost = c.copy(), [], 0.0
        for k, (d, _) in enumerate(pieces):
            last = k + 1 == n
            own, knock, p, _ = self._line(cw, subs[k], d, ahead if last else ((subs[k + 1], pieces[k + 1][0]),), last)
            if not last and (p.freeze > 0.0 or p.overdraft > EPS):
                return None
            cost += own + (knock if last else 0.0)  # a piece's knock-on is the next piece's own onset cost
            self._append(cw, subs[k], p)
            plans.append(p)
        parts = tuple(Part(k, 0.0, d, p.start, p.rate, p.wall, spans[k][0], spans[k][1])
                      for k, ((d, _), p) in enumerate(zip(pieces, plans)))
        first, end = plans[0], plans[-1]
        played = sum(p.wall for p in plans)
        # Only the last piece can delay the next line: a piece before it that runs into the next piece's pause is
        # priced by that piece's own onset.
        plan = Plan(slot.id, first.start, sum(d for d, _ in pieces) / played if played > EPS else 1.0,
                    end.start + end.wall - first.start, first.lag, end.freeze, end.overdraft, end.needs_shorter,
                    end.freeze_at, end.carried, end.excess, parts, said="pieces")
        return cost, replace(plan, voiced=voiced(plan, pieces, self.s.keep_pause))

    def _joined(self, c: _Chain, slot: LineSlot, pieces: Sequence[Take], ahead: tuple) -> tuple[float, Plan]:
        """The pieces said as one line drifting over its span, `piece_gap` natural seconds apart."""
        g = self.s.piece_gap
        total = sum(d for d, _ in pieces) + g * (len(pieces) - 1)
        own, knock, plan, _ = self._line(c, slot, total, ahead)
        parts, at = [], 0.0
        for k, (d, _) in enumerate(pieces):
            parts.append(Part(k, 0.0, d, plan.start + at / plan.rate, plan.rate, d / plan.rate, None, slot.end))
            at += d + g
        plan = replace(plan, parts=tuple(parts), said="joined")
        return own + knock, replace(plan, voiced=voiced(plan, pieces, self.s.keep_pause))

    def _breaks_for(self, slot: LineSlot, durations: list[float]) -> tuple[Span, ...]:
        """The hard breaks the pieces' joins meet: all of them when there is one piece more, else those whose share of
        the English speech before them best matches the share of the Telugu said before each join (duration match;
        where Telugu breaks is the translator's, by fluency)."""
        need = len(durations) - 1
        if len(slot.breaks) == need or sum(durations) <= EPS:
            return slot.breaks[:need]
        speech = union(slot.speech) or [(slot.start, slot.end)]
        total = length(speech)
        en = [length(union((a, min(b, p)) for a, b in speech)) / total for p, _ in slot.breaks]
        te = list(itertools.accumulate(durations[:-1]))
        te = [t / sum(durations) for t in te]
        pick = min(itertools.combinations(range(len(slot.breaks)), need),
                   key=lambda ix: sum(abs(te[j] - en[i]) for j, i in enumerate(ix)))
        return tuple(slot.breaks[i] for i in pick)

    def _anchored(self, slot: LineSlot, plan: Plan, duration: float, pauses: tuple[Span, ...],
                  marks: tuple[float, ...], hz: _Horizon) -> Plan:
        """Soft anchors (§3.10 step 3): a line placed at rate 1 with no freeze that ends before `soft` may wait at a
        pause of its own take for the English to resume after an anchor. For each anchor, the take's pauses whose end
        comes before that English onset are scored: duration match (1 − the silence to insert ÷ anchor_max) × break
        plausibility (the pause's length, up to 0.25 s, and 1.0 near a text break in `marks` or else 0.5). The best
        with a score of at least `anchor_score` gets that silence at its middle, while the dub still ends by `soft`.
        Nothing moves the dub's start or its rate, so its cost is unchanged."""
        s = self.s
        inner = pz.internal(pauses, duration)
        room = hz.soft - plan.end
        if (not slot.anchors or not inner or plan.parts or plan.rate > 1.0 + EPS or plan.freeze > 0.0
                or plan.overdraft > EPS or room < s.anchor_min):
            return plan
        head = max((b for a, b in pauses if a <= pz.FRAME / 2), default=0.0)
        tail = duration - min((a for a, b in pauses if b >= duration - pz.FRAME / 2), default=duration)
        said = max(duration - head - tail, 0.0)
        near = max(0.3, 0.1 * duration)
        texts = [head + m * said for m in marks]
        splits: list[tuple[float, float, float, float]] = []  # (middle in the take, silence set there, onset, anchor start)
        shift, after = 0.0, -INF
        for p0, q in sorted(slot.anchors, key=lambda a: a[1]):
            best = None
            for a, b in inner:
                delta = q - (plan.start + b + shift)
                if a <= after or delta <= 0.0:
                    continue
                plausible = min(1.0, (b - a) / 0.25) * (1.0 if any(a - near <= t <= b + near for t in texts) else 0.5)
                score = max(0.0, 1.0 - delta / s.anchor_max) * plausible
                if score >= s.anchor_score - EPS and (best is None or score > best[0]):
                    best = (score, a, b, delta)
            if best is None:
                continue
            _, a, b, delta = best
            delta = min(delta, room - shift)
            if delta < s.anchor_min:
                break
            splits.append(((a + b) / 2, delta, q, p0))
            shift += delta
            after = b
        if not splits:
            return plan
        cuts = [0.0, *(m for m, _, _, _ in splits), duration]
        onsets = [slot.start, *(q for _, _, q, _ in splits)]
        ends = [*(p0 for _, _, _, p0 in splits), slot.end]
        parts, moved = [], 0.0
        for k in range(len(cuts) - 1):
            parts.append(Part(0, cuts[k], cuts[k + 1], plan.start + cuts[k] + moved, 1.0, cuts[k + 1] - cuts[k],
                              onsets[k], ends[k]))
            if k < len(splits):
                moved += splits[k][1]
        return replace(plan, wall=plan.wall + shift, parts=tuple(parts), said="anchored")

    def _window_load(self, c: _Chain, a: float, b: float) -> float:
        """Largest freeze total in any rolling window (w − budget_window, w] with w in [a, b)."""
        bw, ev = self.s.budget_window, c.freezes
        ends = [a, *(u for u, _ in ev if a <= u < b)]
        return max(sum(f for t, f in ev if w - bw < t <= w + EPS) for w in ends)

    def _commit(self, slot: LineSlot, plan: Plan) -> None:
        self._append(self._chain, slot, plan)
        if slot.id not in self._placed:
            bisect.insort(self._keys, (slot.start, slot.id))
        self._placed[slot.id] = (slot, plan)

    def _append(self, c: _Chain, slot: LineSlot, plan: Plan) -> None:
        end = plan.end
        # A chain grows in onset order, so a dub that ended well before this onset can't bound a later one.
        horizon = slot.start - self.s.lead_max - self.s.min_gap - 1.0
        c.dubs = [d for d in c.dubs if d.end > horizon or d.src_end > horizon]
        # A line said in parts bounds later lines part by part: an interjection after one part isn't held by the next.
        for p in plan.parts[:-1]:
            c.dubs.append(_Dub(slot.id, slot.speaker, p.src_end, p.start + p.wall))
        c.dubs.append(_Dub(slot.id, slot.speaker, slot.end, end))
        c.last_id = slot.id
        c.key = (slot.start, slot.id)
        c.end_by_speaker[slot.speaker] = max(c.end_by_speaker.get(slot.speaker, -INF), end)
        c.rate_by_speaker[slot.speaker] = plan.parts[-1].rate if plan.parts else plan.rate
        if plan.freeze_at is not None:
            c.freezes.append((plan.freeze_at, plan.freeze))


def voiced(plan: Plan, takes: Sequence[Take], keep: float = PlannerSettings.keep_pause) -> tuple[Span, ...]:
    """Where the dub is voiced, in video time: each part's take (`takes[part.take]`: its natural seconds and pauses) less
    its pauses, as it is rendered (squeezed at its rate, `pauses.squeeze`); after a freeze the audio plays while the
    video holds."""
    parts = plan.parts or (Part(0, 0.0, takes[0][0], plan.start, plan.rate, plan.wall, None, 0.0),)
    out: list[Span] = []
    for p in parts:
        d, quiet = takes[p.take]
        r, cuts = pz.squeeze(quiet, d, p.rate, keep)
        base = p.start - pz.out_time(p.a, r, cuts)
        for a, b in pz.voiced(quiet, d):
            a, b = max(a, p.a), min(b, p.b)
            if b > a + EPS:
                out.append((base + pz.out_time(a, r, cuts), base + pz.out_time(b, r, cuts)))

    def video(t: float) -> float:
        if plan.freeze <= 0.0 or plan.freeze_at is None or t <= plan.freeze_at:
            return t
        return max(plan.freeze_at, t - plan.freeze)

    return tuple((round(video(a), 3), round(video(b), 3)) for a, b in out if video(b) > video(a) + EPS)


def _ahead(next_predicted: float | None, next_slot: LineSlot | None,
           ahead: Sequence[tuple[LineSlot, float | None]] | None) -> tuple:
    if ahead is not None:
        return tuple(ahead)
    return () if next_slot is None and next_predicted is None else ((next_slot, next_predicted),)


def anchor_errors(plan: Plan) -> list[float]:
    """For each part after the first timed against an English onset (a piece at a hard break, or a take resumed at a
    soft anchor): its voiced onset minus that onset."""
    out = []
    for p in plan.parts[1:]:
        if p.src is None:
            continue
        onset = next((a for a, b in plan.voiced if a >= p.start - 1e-6), None)
        if onset is not None:
            out.append(round(onset - p.src, 3))
    return out


def _metric_row(slot: LineSlot, plan: Plan) -> dict:
    return {"start": slot.start, "end": slot.end, "speech": slot.speech or ((slot.start, slot.end),),
            "voiced": plan.voiced or ((plan.start, plan.end),), "anchor_errors": anchor_errors(plan)}


def line_slots(records: Iterable[Mapping]) -> list[LineSlot]:
    """Slots for lines given as dicts with id, speaker, start and end, in onset order.

    A line overlaps the previous one when it starts before that line's source end and the
    speaker differs; a speaker's own lines never overlap, so they are never flagged. `prev_end`
    is the latest end among all earlier lines, so a long line still running counts after an
    interjection inside it."""
    rows = sorted(records, key=lambda u: (float(u["start"]), int(u["id"])))
    starts = [float(u["start"]) for u in rows]
    slots: list[LineSlot] = []
    latest: float | None = None
    for i, u in enumerate(rows):
        prev = rows[i - 1] if i else None
        start, end = starts[i], float(u["end"])
        j = bisect.bisect_left(starts, end - EPS, i + 1)
        slots.append(LineSlot(
            id=int(u["id"]), speaker=str(u["speaker"]), start=start, end=end,
            next_start=starts[i + 1] if i + 1 < len(rows) else None,
            prev_end=latest,
            overlaps_prev=prev is not None and str(prev["speaker"]) != str(u["speaker"]) and start < float(prev["end"]) - EPS,
            resume_start=starts[j] if j < len(rows) else INF,
        ))
        latest = end if latest is None else max(latest, end)
    return slots


def window(slots: Sequence[LineSlot], i: int, predicted: Sequence[float | None],
           s: PlannerSettings = PlannerSettings()) -> tuple[tuple[LineSlot, float | None], ...]:
    """The lookahead window after slot `i`: up to `window_lines` lines starting within `window_s` of it, each with its
    predicted duration."""
    t = slots[i].start + s.window_s
    return tuple((slots[j], predicted[j]) for j in range(i + 1, min(i + 1 + s.window_lines, len(slots)))
                 if slots[j].start <= t)


def replay(records: Iterable[Mapping], scale: float = 1.0, s: PlannerSettings = PlannerSettings(),
           windowed: bool = False) -> dict:
    """Replay traced lines (dicts with id, speaker, start, end, audio_s, like the engine's units.jsonl
    'unit' events) through the planner with every audio duration scaled by `scale`. The lookahead
    gets the next line's slot and its true scaled duration (with `windowed`, the whole window's). Returns `stats()`
    plus `scale`."""
    rows = sorted(records, key=lambda u: (float(u["start"]), int(u["id"])))
    slots = line_slots(rows)
    durations = [float(u["audio_s"]) * scale for u in rows]
    planner = TimelinePlanner(s)
    for i, sl in enumerate(slots):
        if windowed:
            planner.place(sl, durations[i], ahead=window(slots, i, durations, s))
            continue
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        predicted = durations[i + 1] if nxt is not None else None
        planner.place(sl, durations[i], predicted, nxt)
    return {"scale": scale, **planner.stats()}


def _iou(slot: LineSlot, p: Plan) -> float:
    """Overlap of the dub's video span with the source speech span."""
    a0, a1, b0, b1 = p.start, max(p.end, p.start), slot.start, slot.end
    union_ = max(a1, b1) - min(a0, b0)
    return _pos(min(a1, b1) - max(a0, b0)) / union_ if union_ > EPS else 1.0


def _r(v: float) -> float:
    return round(v, 4)
