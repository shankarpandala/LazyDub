import math
from dataclasses import replace

import pytest
from hypothesis import given, settings, strategies as st

from maata_engine.timing.planner import (
    V1, LineSlot, Plan, PlannerSettings, TimelinePlanner, anchor_errors, lag_limit, line_slots, rate_ceiling, replay,
    target_seconds, voiced, window,
)

S = PlannerSettings()
TOL = 1e-6


def slot(i, start, end, nxt=None, prev=None, spk="a", overlaps=False, resume=None):
    return LineSlot(i, spk, start, end, nxt, prev, overlaps, resume)


def place_all(slots, durations, lookahead=True):
    pl, plans = TimelinePlanner(), []
    for i, (sl, d) in enumerate(zip(slots, durations)):
        nxt = slots[i + 1] if lookahead and i + 1 < len(slots) else None
        plans.append(pl.place(sl, d, durations[i + 1] if nxt is not None else None, nxt))
    return pl, plans


def on_time(p):
    return p.rate == 1.0 and p.freeze == p.overdraft == 0.0 and not p.needs_shorter


# --- per-line quantities ----------------------------------------------------------------------

def test_target_borrows_a_little_of_the_following_silence():
    assert target_seconds(slot(0, 10, 12, 12.4)) == pytest.approx(2 + 0.25)
    assert target_seconds(slot(0, 10, 12, 20)) == pytest.approx(2 + 0.5)       # borrow capped
    assert target_seconds(slot(0, 10, 12, 12.1)) == pytest.approx(2)           # gap under the guard
    assert target_seconds(slot(0, 10, 12, 11.5)) == pytest.approx(2)           # overlapped by the next line
    assert target_seconds(slot(0, 10, 12)) == pytest.approx(2.5)               # last line


def test_lag_limit_relaxes_only_when_a_long_gap_follows_soon():
    assert lag_limit(slot(0, 0, 3, 4.5)) == S.max_lag_resync
    assert lag_limit(slot(0, 0, 3, 3.5)) == S.max_lag
    assert lag_limit(slot(0, 0, 7, 9.0)) == S.max_lag                          # the gap comes too late
    assert lag_limit(slot(0, 0, 3)) == S.max_lag_resync                        # end of the video


# --- placement --------------------------------------------------------------------------------

def test_a_line_that_fits_starts_on_its_onset_at_rate_1():
    p = TimelinePlanner().place(slot(0, 10, 12, 13, 8), 1.8)
    assert (p.start, p.rate, p.lag, p.freeze, p.overdraft, p.needs_shorter) == (10, 1.0, 0.0, 0.0, 0.0, False)
    assert p.wall == 1.8


def test_small_overflow_is_shared_between_early_start_speed_up_and_slip():
    # Window 10 → 11.85 after the guard, with silence before; a 2.2 s line needs 0.35 s more.
    p = TimelinePlanner().place(slot(0, 10, 11.5, 12.0, 8), 2.2)
    slip = p.start + p.wall - (12.0 - S.guard)
    assert -S.lead_max < p.lag < 0 and 1.0 < p.rate < 1.1 and 0 < slip < 0.2
    assert p.freeze == p.overdraft == 0.0 and not p.needs_shorter


def test_no_early_start_without_silence_before():
    p = TimelinePlanner().place(slot(0, 10, 11.5, 12.0, 9.95), 2.2)
    assert p.lag == 0.0 and p.rate > 1.0


def test_early_start_never_crosses_the_previous_dub():
    pl = TimelinePlanner()
    a = pl.place(slot(0, 0, 3, 5, None), 3.0)
    b = pl.place(slot(1, 5, 6, 6.3, 3), 1.6)
    assert b.start >= a.start + a.wall + S.min_gap - TOL


def test_timing_slips_before_rate_jumps():
    # A line a bit too long for its window: it may end past the next onset rather than hit the cap.
    p = TimelinePlanner().place(slot(0, 0, 3, 3.2, None), 3.9, next_predicted=1.0,
                                next_slot=slot(1, 3.2, 4.5, 8.0, 3))
    assert p.rate < S.speed_cap and p.start + p.wall > 3.2 - S.guard
    assert p.start + p.wall <= 3.2 + lag_limit(slot(1, 3.2, 4.5, 8.0, 3)) - S.min_gap + TOL


def test_hopeless_line_signals_shorter_then_freezes_then_overdrafts():
    p = TimelinePlanner().place(slot(0, 0, 2, 2.3, None), 6.0)
    assert p.rate == S.speed_cap and p.wall == pytest.approx(5.0) and p.needs_shorter
    hard = 2.3 + S.max_lag - S.min_gap
    assert p.freeze == pytest.approx(S.max_freeze)
    assert p.overdraft == pytest.approx(p.start + p.wall - hard - p.freeze) and p.carried == 0.0
    assert p.end == pytest.approx(p.start + p.wall - p.freeze)
    assert p.excess == pytest.approx(p.start + p.wall - (2.3 - S.guard))


def test_a_freeze_holds_the_pause_before_the_next_line_not_its_first_words():
    pl = TimelinePlanner()
    a = pl.place(slot(0, 0, 2, 2.3, None), 6.0)
    assert a.freeze_at == pytest.approx(2.3 - S.guard) and a.start <= a.freeze_at <= a.end
    # A dub that only starts after that pause (pushed by the overdraft) holds at its own start.
    b = pl.place(slot(1, 2.3, 3, 3.3, 2), 2.0)
    assert b.freeze > 0 and b.freeze_at == pytest.approx(b.start)
    assert pl.place(slot(2, 9, 10, 20, 3.3), 1.0).freeze_at is None


def test_a_tiny_overshoot_is_left_as_lag_rather_than_a_stuttering_hold():
    hard = 2.2 + S.max_lag - S.min_gap          # no lead: the line starts the video
    tiny = TimelinePlanner().place(slot(0, 0, 2, 2.2, None), (hard + 0.05) * S.speed_cap)
    assert tiny.freeze == 0.0 and tiny.freeze_at is None
    assert tiny.overdraft == pytest.approx(0.05) and tiny.needs_shorter
    held = TimelinePlanner().place(slot(0, 0, 2, 2.2, None), (hard + 0.2) * S.speed_cap)
    assert held.freeze == pytest.approx(0.2) and held.overdraft == pytest.approx(0.0, abs=1e-9)


def test_needs_shorter_only_when_the_next_line_is_really_delayed():
    nxt = slot(1, 2.2, 3, 5, 2)                  # may re-sync late: a long gap follows it
    mild = TimelinePlanner().place(slot(0, 0, 2, 2.2, None), 2.6, 0.8, nxt)
    firm = TimelinePlanner().place(slot(0, 0, 2, 2.2, None), 2.9, 0.8, nxt)
    for p in (mild, firm):
        assert p.freeze == p.overdraft == 0.0 and p.excess == pytest.approx(p.start + p.wall - (2.2 - S.guard))
    assert 0.0 < mild.end - (2.2 - S.min_gap) <= S.shorter_tol and not mild.needs_shorter
    assert firm.end - (2.2 - S.min_gap) > S.shorter_tol and firm.needs_shorter
    assert firm.excess > mild.excess > 0.0


def test_overdraft_passed_down_the_chain_is_counted_once():
    pl = TimelinePlanner()
    p0 = pl.place(slot(0, 0, 2, 2.2, None), 6.0)
    p1 = pl.place(slot(1, 2.2, 4.2, 4.4, 2), 2.0)   # would fit on time at rate 1.2
    p2 = pl.place(slot(2, 4.4, 6.4, 6.6, 4.2), 2.0)
    assert p0.overdraft > 0 and p0.carried == 0.0
    for p in (p1, p2):
        assert p.overdraft > 0 and p.carried == pytest.approx(p.overdraft) and p.own_overdraft == pytest.approx(0.0)
    st_ = pl.stats()
    assert st_["overdraft_s"] == pytest.approx(p0.overdraft, abs=1e-3) and st_["overdraft_lines"] == 1
    assert st_["overdraft_carried_s"] == pytest.approx(p1.overdraft + p2.overdraft, abs=1e-3)


def test_freeze_budget_rolls_over_a_minute_of_video():
    pl, freezes = TimelinePlanner(), []
    for i in range(8):                    # a hopeless line every 5 s
        t = 5.0 * i
        freezes.append(pl.place(slot(i, t, t + 1, t + 1.3, t - 4 if i else None), 3.5).freeze)
    assert sum(freezes) == pytest.approx(S.freeze_budget)
    assert freezes[:2] == [pytest.approx(S.max_freeze), pytest.approx(S.freeze_budget - S.max_freeze)]
    assert all(f == 0.0 for f in freezes[2:])
    late = pl.place(slot(99, 70, 71, 71.3, 66), 3.5)
    assert late.freeze == pytest.approx(S.max_freeze)  # the first freezes have left the window


@pytest.mark.parametrize("lookahead", [False, True])
def test_overlapping_speakers_overlap_only_as_much_as_the_source(lookahead):
    # b starts while a is still speaking; a's line fits its own span and needs nothing.
    slots = [slot(0, 0, 4, 3.5, None, "a"), slot(1, 3.5, 5, 7, 4, "b", overlaps=True), slot(2, 7, 8, None, 5, "a")]
    _, (a, b, c) = place_all(slots, (4.0, 1.5, 1.0), lookahead)
    assert on_time(a) and a.lag == 0.0
    assert b.start == pytest.approx(max(3.5, a.end - 0.5))
    assert c.start >= b.end + S.min_gap - TOL


@pytest.mark.parametrize("lookahead", [False, True])
def test_an_interjection_inside_a_long_line_need_not_wait_for_it(lookahead):
    # a speaks 0–10; b interjects at 2 and again at 5 (not overlapping b's own first line).
    rows = [dict(id=0, speaker="a", start=0, end=10), dict(id=1, speaker="b", start=2, end=2.6),
            dict(id=2, speaker="b", start=5, end=5.6), dict(id=3, speaker="a", start=11, end=12)]
    slots = line_slots(rows)
    assert [s.overlaps_prev for s in slots] == [False, True, False, False]
    assert slots[0].resume_start == 11 and slots[3].resume_start == math.inf
    _, plans = place_all(slots, (9.5, 0.6, 0.6, 1.0), lookahead)
    assert on_time(plans[0]) and plans[0].lag == 0.0   # 9.5 s fits a's own 10 s span
    assert plans[1].lag == 0.0 and plans[2].lag == 0.0
    assert plans[3].start >= plans[0].end + S.min_gap - TOL


def test_the_line_after_a_long_one_keeps_its_own_lag_limit_despite_an_interjection():
    # b's interjection may re-sync late (1.0 s), but a's next line (gap 0.2 s after it) may not.
    rows = [dict(id=0, speaker="a", start=0, end=10), dict(id=1, speaker="b", start=2, end=2.6),
            dict(id=2, speaker="a", start=10.3, end=12), dict(id=3, speaker="b", start=12.2, end=14)]
    slots = line_slots(rows)
    assert lag_limit(slots[1]) == S.max_lag_resync and lag_limit(slots[2]) == S.max_lag
    _, plans = place_all(slots, (12.9, 0.5, 1.5, 1.0))
    assert all(p.overdraft == 0.0 for p in plans)
    assert all(p.lag <= lag_limit(sl) + TOL for sl, p in zip(slots, plans))
    # Longer still: the overshoot is reported as overdraft where it arises, and only then is a line late.
    _, plans = place_all(slots, (13.08, 0.5, 1.5, 1.0))
    assert plans[0].overdraft == pytest.approx(13.08 / S.speed_cap - (10.3 + S.max_lag - S.min_gap))
    assert plans[2].lag == pytest.approx(S.max_lag + plans[0].overdraft)


def test_a_line_is_never_sped_up_beyond_what_its_window_needs():
    pl = TimelinePlanner()
    fast = pl.place(slot(0, 0, 2, 2.2, None, "a"), 2.6)
    assert fast.rate > 1.1
    roomy = pl.place(slot(1, 2.2, 4, 9, 2, "a"), 1.0)   # needs no speed-up at all
    assert roomy.rate == 1.0 and not roomy.needs_shorter
    assert roomy.start == pytest.approx(max(2.2, fast.end + S.min_gap))
    # Nor to ease toward the speaker's next line, however fast that one has to be.
    p = TimelinePlanner().place(slot(0, 0, 3, 3.2), 1.0, next_predicted=3.6, next_slot=slot(1, 3.2, 5, 5.2, 3))
    assert on_time(p) and p.lag == 0.0


def test_a_speakers_rate_resists_jumping_up():
    tight = slot(1, 10, 12, 12.3, 9.95, "a")               # 2.9 s of audio for 2.15 s
    fresh = TimelinePlanner().place(tight, 2.9)
    runs = {}
    for spk, t0 in (("a", 6), ("b", 6), ("a", -5)):         # a unhurried line just before; another speaker's; a long pause
        pl = TimelinePlanner()
        pl.place(slot(0, t0, t0 + 3.9, 10, None, spk), 3.0)
        runs[spk, t0] = pl.place(tight, 2.9)
    assert fresh.rate > 1.0 + S.smooth
    assert runs["a", 6].rate < fresh.rate                  # slips a little rather than jump from 1.0
    assert runs["b", 6] == fresh and runs["a", -5] == fresh


def test_replacing_the_last_line_undoes_it():
    first, pl = TimelinePlanner(), TimelinePlanner()
    for p in (first, pl):
        p.place(slot(0, 0, 2, 2.3, None), 2.0)
    long = pl.place(slot(1, 2.3, 3, 3.5, 2), 4.0)
    assert long.needs_shorter and long.freeze > 0
    again = pl.place(slot(1, 2.3, 3, 3.5, 2), 1.0)
    assert again == first.place(slot(1, 2.3, 3, 3.5, 2), 1.0)
    assert pl.stats()["freezes"] == 0 and pl.stats()["lines"] == 2


def test_an_earlier_line_cannot_be_replaced_behind_later_ones():
    pl = TimelinePlanner()
    pl.place(slot(0, 0, 2, 2.3, None), 4.0)
    b = pl.place(slot(1, 2.3, 3, 8, 2, "b"), 1.0)
    with pytest.raises(ValueError):
        pl.place(slot(0, 0, 2, 2.3, None), 1.5)             # a shorter take of line 0 arriving late
    with pytest.raises(ValueError):
        pl.evaluate(slot(0, 0, 2, 2.3, None), 1.5)
    assert pl.stats()["lines"] == 2 and pl.place(slot(1, 2.3, 3, 8, 2, "b"), 1.0) == b
    pl.reset(0.0)                                           # the way to re-place it: from its onset on
    assert on_time(pl.place(slot(0, 0, 2, 2.3, None), 1.5))
    assert pl.stats()["freezes"] == 0 and pl.stats()["lines"] == 1


def test_evaluate_prices_a_duration_without_committing_it():
    sl, pl = slot(0, 10, 11.5, 12.0, 8), TimelinePlanner()
    priced = [pl.evaluate(sl, d) for d in (1.5, 2.2, 2.6, 3.0, 4.0)]
    assert pl.stats() == {"lines": 0}
    costs = [c for c, _ in priced]
    assert costs[0] == 0.0 and costs == sorted(costs) and len(set(costs)) == len(costs)
    assert priced[-1][1].freeze > 0                          # the longest needs a hold, and costs most
    assert pl.place(sl, 2.6) == priced[2][1]
    # For the line just placed, a candidate is priced against the chain before it, as `place()` would.
    first = pl.place(slot(1, 12.0, 13, 13.4, 11.5), 3.0)
    cost, alt = pl.evaluate(slot(1, 12.0, 13, 13.4, 11.5), 1.0)
    assert first.needs_shorter and cost < pl.evaluate(slot(1, 12.0, 13, 13.4, 11.5), 3.0)[0]
    assert alt == pl.place(slot(1, 12.0, 13, 13.4, 11.5), 1.0) and pl.stats()["lines"] == 2


def test_reset_after_a_seek_frees_the_chain():
    pl = TimelinePlanner()
    pl.place(slot(0, 0, 2, 2.3, None), 8.0)
    pl.place(slot(1, 2.3, 4, 30, 2), 1.0)
    pl.reset(2.0)
    p = pl.place(slot(1, 2.3, 4, 30, 2), 1.0)
    assert p.lag <= 0.0 and pl.stats()["lines"] == 2


def test_a_line_placed_behind_later_ones_is_planned_against_the_lines_before_it():
    """A seek forward, then back into video dubbed only up to line 0: line 1 follows line 0, not the far lines placed
    meanwhile, and line 6 then follows line 5 again (it runs late), not line 1."""
    l0, l1 = slot(0, 0, 2, 2.3, None), slot(1, 2.3, 4, 60, 2)
    l5, l6 = slot(5, 60, 62, 62.3, 4), slot(6, 62.3, 64, None, 62)
    near, far = TimelinePlanner(), TimelinePlanner()
    near.place(l0, 2.6)
    in_order = (near.place(l1, 1.5), far.place(l5, 3.0), far.place(l6, 1.0))
    pl = TimelinePlanner()
    pl.place(l0, 2.6)
    pl.reset(60.0)                                          # the seek forward
    assert pl.place(l5, 3.0) == in_order[1]
    assert pl.evaluate(l1, 1.5)[1] == in_order[0]           # priced the same way, and nothing committed
    p1 = pl.place(l1, 1.5)                                  # the seek back: voiced behind line 5
    assert p1 == in_order[0] and p1.lag > 0.0 and p1.lag < S.max_lag
    p6 = pl.place(l6, 1.0)
    assert p6 == in_order[2] and p6.lag > 0.0
    assert pl.stats()["lines"] == 4


def test_rejects_bad_durations():
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            TimelinePlanner().place(slot(0, 0, 1), bad)


def test_zero_length_audio_is_placed():
    p = TimelinePlanner().place(slot(0, 3, 4, 5, 1), 0.0)
    assert p.start == 3 and p.wall == 0.0 and p.rate == 1.0


# --- replay -----------------------------------------------------------------------------------

def _dense_talk(n=40):
    """A made-up two-person conversation: short gaps, the odd overlap, lines too long for their slots."""
    rows, t = [], 1.0
    for i in range(n):
        length = 1.2 + (i * 7 % 5) * 0.6
        gap = (-0.3, 0.1, 0.25, 0.6, 1.4)[i % 5]
        spk = "a" if i % 3 else "b"
        if rows and rows[-1]["speaker"] == spk:
            gap = max(gap, 0.05)
        start = max(t + gap, rows[-1]["start"] + 0.1) if rows else t
        rows.append(dict(id=i, speaker=spk, start=start, end=start + length, audio_s=length * 1.3))
        t = start + length
    return rows


def test_replay_reports_the_x0_metrics_and_shorter_audio_helps():
    rows = _dense_talk()
    full, short = replay(rows, 1.0), replay(rows, 0.75)
    for key in ("lag_p50", "lag_p95", "lag_max", "rate_mean", "rate_p90", "freezes_per_min",
                "freeze_s_per_min", "overdraft_s", "needs_shorter", "video_slowdowns"):
        assert key in full
    assert full["lines"] == short["lines"] == 40 and full["video_slowdowns"] == 0
    assert short["lag_p95"] <= full["lag_p95"] and short["overdraft_s"] <= full["overdraft_s"]
    assert short["needs_shorter"] <= full["needs_shorter"] and short["rate_mean"] <= full["rate_mean"]
    assert replay(list(reversed(rows)), 0.9) == replay(rows, 0.9)


def test_stats_of_an_empty_planner():
    assert TimelinePlanner().stats() == {"lines": 0}


def test_line_slots_measure_the_gap_before_from_the_latest_earlier_end():
    # c starts 0.05 s after a's long line ends; b's interjection inside it ended long before.
    rows = [dict(id=0, speaker="a", start=0, end=10), dict(id=1, speaker="b", start=2, end=2.6),
            dict(id=2, speaker="c", start=10.05, end=11.5), dict(id=3, speaker="a", start=11.6, end=13)]
    slots = line_slots(rows)
    assert [s.prev_end for s in slots] == [None, 10, 10, 11.5]
    assert [s.resume_start for s in slots] == [10.05, 10.05, 11.6, math.inf]
    _, plans = place_all(slots, (9.0, 0.5, 1.4, 1.0))
    assert plans[2].start >= 10.0 - TOL                      # no early start over a's English


# --- properties -------------------------------------------------------------------------------

@st.composite
def talks(draw):
    """Lines in onset order. Different speakers may overlap (nested too); a speaker never overlaps itself."""
    n = draw(st.integers(1, 24))
    rows, speaker_end = [], {}
    start, end = draw(st.floats(0, 3)), 0.0
    for i in range(n):
        spk = draw(st.sampled_from("abc"))
        gap = draw(st.floats(-2.0, 3.0))
        length = draw(st.floats(0.1, 9.0))
        start = max(start, end + gap if rows else start, speaker_end.get(spk, -math.inf) + 0.01)
        end = start + length
        speaker_end[spk] = end
        rows.append(dict(id=i, speaker=spk, start=start, end=end,
                         audio_s=length * draw(st.floats(0.0, 2.5))))
    return rows


def _latest_end_before(rows, sl):
    earlier = [r["end"] for r in rows if (r["start"], r["id"]) < (sl.start, sl.id)]
    return max(earlier) if earlier else None


def _run(rows, s, lookahead, noise):
    slots, pl, plans = line_slots(rows), TimelinePlanner(s), []
    by_id = {r["id"]: r for r in rows}
    for i, sl in enumerate(slots):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        pred = by_id[nxt.id]["audio_s"] * noise if nxt is not None else None
        plans.append(pl.place(sl, by_id[sl.id]["audio_s"], pred, nxt if lookahead else None))
    return slots, plans, pl


settings_st = st.builds(PlannerSettings, speed_cap=st.sampled_from([1.1, 1.2, 1.25]),
                        lead_max=st.sampled_from([0.0, 0.3]), max_freeze=st.sampled_from([0.3, 0.6]),
                        freeze_budget=st.sampled_from([0.5, 1.0]), smooth=st.sampled_from([0.05, 0.1]))


@settings(max_examples=300, deadline=None)
@given(talks(), settings_st, st.booleans(), st.sampled_from([0.7, 1.0, 1.4]))
def test_planner_invariants(rows, s, lookahead, noise):
    slots, plans, pl = _run(rows, s, lookahead, noise)
    by_id = {r["id"]: r for r in rows}
    for k, (sl, p) in enumerate(zip(slots, plans)):
        d = by_id[sl.id]["audio_s"]
        assert isinstance(p, Plan) and p.id == sl.id
        assert p.wall * p.rate == pytest.approx(d, rel=1e-9, abs=1e-12)          # never trimmed
        assert 1.0 - TOL <= p.rate <= s.speed_cap + TOL
        assert p.start >= sl.start - s.lead_max - TOL and p.lag == pytest.approx(p.start - sl.start)
        prev_end = _latest_end_before(rows, sl)
        gap_before = sl.start - prev_end if prev_end is not None else sl.start
        assert p.start >= sl.start - max(min(s.lead_max, gap_before - s.guard), 0.0) - TOL
        assert p.freeze == 0.0 or s.min_freeze - TOL <= p.freeze <= min(s.max_freeze, p.wall) + TOL
        assert (p.freeze_at is None) == (p.freeze == 0.0)
        assert p.freeze_at is None or p.start - TOL <= p.freeze_at <= p.end + TOL
        assert p.overdraft >= 0.0 and 0.0 <= p.carried <= p.overdraft + TOL and p.excess >= 0.0
        # Later than its own lag limit only when an earlier dub already reported overdraft that reaches here.
        limit = lag_limit(sl, s)
        blocked = any(q.overdraft > 0 and q.end + s.min_gap > sl.start + limit - TOL for q in plans[:k])
        assert p.lag <= limit + TOL or blocked
        assert p.carried == 0.0 or blocked
        for sk, q in zip(slots[:k], plans[:k]):
            if sk.speaker != sl.speaker and sk.end > sl.start + 1e-9:
                # Another speaker's dub is overlapped by no more than the source overlapped it.
                assert p.start >= q.end - (sk.end - sl.start) - TOL
            else:
                assert p.start >= q.end + s.min_gap - TOL                       # no overlap, gap kept
    held = [(p.freeze_at, p.freeze) for p in plans if p.freeze > 0]
    for w, _ in held:                                                           # rolling freeze budget
        assert sum(f for t, f in held if w - 60.0 < t <= w + TOL) <= s.freeze_budget + TOL
    st_ = pl.stats()
    assert st_["lines"] == len(rows) and st_["needs_shorter"] == sum(p.needs_shorter for p in plans)
    assert st_["freeze_s"] == pytest.approx(sum(p.freeze for p in plans), abs=1e-3)
    assert st_["overdraft_s"] == pytest.approx(sum(p.overdraft - p.carried for p in plans), abs=1e-3)


@settings(max_examples=100, deadline=None)
@given(talks(), st.booleans())
def test_planner_is_deterministic(rows, lookahead):
    assert _run(rows, S, lookahead, 1.0)[1] == _run(rows, S, lookahead, 1.0)[1]
    scaled = [{**r, "audio_s": r["audio_s"] * 0.85} for r in rows]
    assert replay(rows, 0.85) == replay(rows, 0.85) == {"scale": 0.85, **_run(scaled, S, True, 1.0)[2].stats()}


@st.composite
def roomy_talks(draw):
    """Lines separated by more than guard + min_gap, each with audio that fits its window at rate 1."""
    rows, t = [], draw(st.floats(0, 3))
    for i in range(draw(st.integers(1, 20))):
        length = draw(st.floats(0.2, 6.0))
        rows.append(dict(id=i, speaker=draw(st.sampled_from("ab")), start=t, end=t + length,
                         audio_s=length * draw(st.floats(0.0, 1.0))))
        t += length + draw(st.floats(S.guard + S.min_gap + 0.01, 3.0))
    return rows


@settings(max_examples=150, deadline=None)
@given(roomy_talks())
def test_lines_that_fit_are_placed_on_their_onsets_at_rate_1(rows):
    _, plans, pl = _run(rows, S, True, 1.0)
    for p in plans:
        assert p.lag == 0.0 and p.rate == 1.0 and p.freeze == p.overdraft == 0.0 and not p.needs_shorter
    assert pl.stats()["iou_mean"] <= 1.0


# --- timing v2 (ARCHITECTURE §3.10) ---------------------------------------------------------------

def test_starting_early_costs_twice_what_starting_late_does():
    # The same squeeze as above: with early starts priced like late ones the line leads more, and slips less.
    sl = slot(0, 10, 11.5, 12.0, 8)
    even = TimelinePlanner(PlannerSettings(w_early=1.0)).place(sl, 2.2)
    p = TimelinePlanner().place(sl, 2.2)
    assert even.lag < p.lag < 0.0
    assert p.start + p.wall > even.start + even.wall                       # the rest goes into the slip
    assert p.freeze == p.overdraft == 0.0 and not p.needs_shorter


def test_the_rate_ceiling_is_the_speed_cap_or_the_akshara_ceiling_never_under_1():
    assert rate_ceiling(9.0) == S.speed_cap                                  # off by default until measured on Telugu
    c = PlannerSettings(akshara_ceiling=7.5)
    assert rate_ceiling(None, c) == rate_ceiling(0.0, c) == c.speed_cap
    assert rate_ceiling(6.1, c) == c.speed_cap                               # 7.5 / 6.1 = 1.23: the cap binds
    assert rate_ceiling(6.8, c) == pytest.approx(7.5 / 6.8)
    assert rate_ceiling(9.0, c) == 1.0                                       # a fast voice is never slowed down
    assert rate_ceiling(5.0, PlannerSettings(speed_cap=1.1, akshara_ceiling=7.5)) == 1.1
    # A hopeless line goes no faster than its own ceiling.
    p = TimelinePlanner().place(LineSlot(0, "a", 0, 3, 3.2, None, rate_cap=1.1), 6.0)
    assert p.rate == pytest.approx(1.1) and p.needs_shorter
    assert TimelinePlanner().place(LineSlot(0, "a", 0, 3, 3.2, None, rate_cap=0.9), 6.0).rate == 1.0


CHAIN = [dict(id=0, speaker="a", start=2.0, end=4.7, audio_s=2.3), dict(id=1, speaker="b", start=5.2, end=8.1, audio_s=3.6),
         dict(id=2, speaker="a", start=8.4, end=10.9, audio_s=3.4), dict(id=3, speaker="b", start=11.4, end=15.0, audio_s=4.2),
         dict(id=4, speaker="a", start=15.3, end=19.1, audio_s=5.1), dict(id=5, speaker="b", start=19.3, end=23.2, audio_s=4.6)]


def test_the_windowed_lookahead_lets_slack_flow_along_a_chain_of_long_lines():
    one, win = replay(CHAIN), replay(CHAIN, windowed=True)
    assert win["lag_max"] < one["lag_max"] - 0.1 and win["lag_p95"] < one["lag_p95"]
    assert win["needs_shorter"] < one["needs_shorter"] and win["rate_max"] <= one["rate_max"]
    assert win["overdraft_s"] == one["overdraft_s"] == 0.0 and win["freeze_s"] == 0.0


def test_a_window_of_one_line_is_the_one_line_lookahead():
    slots = line_slots(CHAIN)
    d = [r["audio_s"] for r in CHAIN]
    a, b = TimelinePlanner(), TimelinePlanner()
    for i, sl in enumerate(slots):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        pa = a.place(sl, d[i], d[i + 1] if nxt else None, nxt)
        pb = b.place(sl, d[i], ahead=((nxt, d[i + 1]),) if nxt else ())
        assert pa == pb


def test_the_window_is_about_8_lines_or_30_s():
    rows = [dict(id=i, speaker="a", start=4.0 * i, end=4.0 * i + 3.0) for i in range(20)]
    slots = line_slots(rows)
    w = window(slots, 0, [3.0] * 20)
    assert [sl.id for sl, _ in w] == list(range(1, 8))                  # 8 would start 32 s after line 0
    assert [sl.id for sl, _ in window(slots, 0, [3.0] * 20, PlannerSettings(window_lines=3))] == [1, 2, 3]
    assert window(slots, 19, [3.0] * 20) == ()
    assert [d for _, d in window(slots, 17, [3.0] * 18 + [None, 2.0])] == [None, 2.0]


def test_a_line_said_in_pieces_meets_each_hard_break():
    # 10-18 s with a 1.5 s pause at 13.0; the Telugu comes as two pieces, each its own take.
    sl = LineSlot(0, "a", 10, 18, 19, 8, breaks=((13.0, 14.5),))
    p = TimelinePlanner().place(sl, 5.4, pieces=[(2.4, ()), (3.0, ())])
    assert p.said == "pieces" and [pt.take for pt in p.parts] == [0, 1]
    assert p.parts[0].start == 10 and p.parts[1].start == 14.5           # each at its own English onset
    assert p.wall == pytest.approx(7.5) and p.rate == 1.0 and p.end == pytest.approx(17.5)
    assert p.voiced == ((10.0, 12.4), (14.5, 17.5)) and anchor_errors(p) == [0.0]
    # The next line sees the dub as placed: a line inside the pause (another speaker) isn't held back by the second piece.
    pl = TimelinePlanner()
    pl.place(LineSlot(0, "a", 10, 18, 13.4, 8, breaks=((13.0, 14.5),)), 5.4, pieces=[(2.4, ()), (3.0, ())])
    q = pl.place(LineSlot(1, "b", 13.4, 13.9, 19, 18, overlaps_prev=True), 0.5)
    assert q.start == 13.4


def test_a_piece_starts_early_only_into_the_silence_left_in_its_pause():
    # 10-17 s with a pause at 12.8-14.2; the second piece is long, so it starts early into the pause...
    sl = LineSlot(0, "a", 10, 17, 17.2, None, breaks=((12.8, 14.2),))
    pieces = [(2.4, ()), (3.4, ())]
    assert TimelinePlanner().place(sl, 5.8, pieces=pieces).parts[1].start < 14.2 - 0.05
    # ...unless someone else talks there: until 14.1 s (no early start at all), or until 13.6 s (as far as its guard).
    p = TimelinePlanner().place(replace(sl, heard=((12.8, 14.1),)), 5.8, pieces=pieces)
    assert p.said == "pieces" and p.parts[1].start >= 14.2 - TOL
    q = TimelinePlanner().place(replace(sl, heard=((12.8, 13.9),)), 5.8, pieces=pieces)
    assert q.said == "pieces" and 13.9 + S.guard - TOL <= q.parts[1].start < 14.2
    # Speech in the pause before another piece's break doesn't hold this one.
    assert TimelinePlanner().place(replace(sl, heard=((10.5, 11.0),)), 5.8, pieces=pieces) == \
        TimelinePlanner().place(sl, 5.8, pieces=pieces)


def test_v1_settings_are_the_planner_before_timing_v2():
    s = PlannerSettings(**V1)
    # Early starts cost what late ones do.
    sl = slot(0, 10, 11.5, 12.0, 8)
    assert TimelinePlanner(s).place(sl, 2.2) == TimelinePlanner(PlannerSettings(w_early=1.0)).place(sl, 2.2)
    # A one-line lookahead: the window is the next line, placed as the one-line planner placed it.
    slots, d = line_slots(CHAIN), [r["audio_s"] for r in CHAIN]
    a, b = TimelinePlanner(s), TimelinePlanner(s)
    for i, sl in enumerate(slots):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        assert window(slots, i, d, s) == (((nxt, d[i + 1]),) if nxt else ())
        assert a.place(sl, d[i], ahead=window(slots, i, d, s)) == b.place(sl, d[i], d[i + 1] if nxt else None, nxt)
    assert replay(CHAIN, s=s, windowed=True) == replay(CHAIN, s=s)
    # No akshara ceiling, no soft anchors, and a sped-up take is compressed evenly (its pause as much as its speech).
    assert rate_ceiling(9.0, s) == s.speed_cap
    anchored = LineSlot(0, "a", 10, 16, 18, 8, anchors=((12.3, 12.6),))
    assert TimelinePlanner(s).place(anchored, 4.5, pauses=((2.0, 2.3),), marks=(0.45,)).said == "whole"
    got = voiced(Plan(0, 10.0, 1.05, 4.0, 0.0), [(4.2, ((1.0, 1.6),))], s.keep_pause)
    assert got[0] == pytest.approx((10.0, 10.0 + 1.0 / 1.05), abs=1e-3)
    assert got[1] == pytest.approx((10.0 + 1.6 / 1.05, 14.0), abs=1e-3)


def test_a_piece_that_cant_meet_its_break_turns_the_pieces_into_one_line():
    # The first piece is far too long for its stretch: even at 1.2x from 9.7 s it would push the second piece more than
    # 1.0 s late (a freeze in mid-sentence). Said as one line, the pieces 0.3 s apart (`piece_gap`).
    sl = LineSlot(0, "a", 10, 18, 19, 8, breaks=((13.0, 14.5),))
    p = TimelinePlanner().place(sl, 9.0, pieces=[(7.0, ()), (2.0, ())])
    assert p.said == "joined" and all(pt.src is None for pt in p.parts)
    assert p.wall * p.rate == pytest.approx(9.0 + S.piece_gap)
    assert p.parts[1].start - p.parts[0].start == pytest.approx((7.0 + S.piece_gap) / p.rate)
    # 6 s for the first piece still fits: at 1.2x it ends before 15.4 (the second onset plus its 1.0 s resync limit).
    assert TimelinePlanner().place(sl, 8.0, pieces=[(6.0, ()), (2.0, ())]).said == "pieces"
    # More pieces than breaks + 1 can't be placed piece by piece either.
    assert TimelinePlanner().place(sl, 3.0, pieces=[(1.0, ()), (1.0, ()), (1.0, ())]).said == "joined"


def test_fewer_pieces_than_breaks_meet_the_breaks_that_match_their_share_of_the_telugu():
    # Two breaks, one join: the Telugu's first piece is 70 % of it, the English before the second break 72 %.
    sl = LineSlot(0, "a", 0, 14, None, None, breaks=((3.0, 4.2), (9.0, 10.2)),
                  speech=((0.0, 3.0), (4.2, 9.0), (10.2, 14.0)))
    p = TimelinePlanner().place(sl, 10.0, pieces=[(7.0, ()), (3.0, ())])
    assert p.said == "pieces" and p.parts[1].src == 10.2
    q = TimelinePlanner().place(sl, 10.0, pieces=[(2.5, ()), (7.5, ())])
    assert q.said == "pieces" and q.parts[1].src == 4.2


def test_a_line_that_fits_waits_at_a_telugu_pause_near_a_soft_anchor():
    # English 10-16 s with a breath at 12.3-12.6; the 4.5 s take pauses at 2.0-2.3 s, where its comma falls.
    sl = LineSlot(0, "a", 10, 16, 18, 8, anchors=((12.3, 12.6),))
    p = TimelinePlanner().place(sl, 4.5, pauses=((2.0, 2.3),), marks=(0.45,))
    assert p.said == "anchored" and p.rate == 1.0 and len(p.parts) == 2
    assert p.parts[1].src == 12.6 and p.voiced == ((10.0, 12.0), (12.6, 14.8)) and anchor_errors(p) == [0.0]
    assert p.wall == pytest.approx(4.8) and p.played == pytest.approx(4.5)
    # Not when the Telugu pause is nowhere near its text break, too far from the anchor, or the line had to speed up.
    assert TimelinePlanner().place(sl, 4.5, pauses=((2.0, 2.3),), marks=(0.9,)).said == "whole"
    far = LineSlot(0, "a", 10, 16, 18, 8, anchors=((12.5, 12.9),))
    assert TimelinePlanner().place(far, 4.5, pauses=((2.0, 2.3),), marks=(0.45,)).said == "whole"
    tight = LineSlot(0, "a", 10, 13.5, 13.9, 8, anchors=((11.8, 12.1),))
    q = TimelinePlanner().place(tight, 4.5, pauses=((1.4, 1.7),), marks=(0.33,))
    assert q.rate > 1.0 and q.said == "whole"


def test_a_soft_anchor_never_pushes_the_dub_past_its_window():
    # The take ends at 17.65: room for only 0.2 s of the 0.3 s of silence the anchor wants before 17.85 (next onset - guard).
    sl = LineSlot(0, "a", 10, 17.5, 18.0, 8, anchors=((12.3, 12.6),))
    p = TimelinePlanner().place(sl, 7.65, pauses=((2.0, 2.3),), marks=(0.3,))
    assert p.said == "anchored" and p.end == pytest.approx(18.0 - S.guard)
    assert anchor_errors(p) == [pytest.approx(-0.1, abs=1e-3)]           # it resumes a little before the English


def test_voiced_time_leaves_the_takes_pauses_out_and_squeezes_them_first():
    # A 4.2 s take with a 0.6 s pause, placed at 1.05x: the pause gives the 0.2 s, the speech plays at 1.0.
    plan = Plan(0, 10.0, 1.05, 4.0, 0.0)
    got = voiced(plan, [(4.2, ((1.0, 1.6),))], 0.12)
    assert got[0] == pytest.approx((10.0, 11.0)) and got[1] == pytest.approx((11.4, 14.0))
    # A freeze: the audio after the hold plays while the video waits.
    held = Plan(0, 10.0, 1.0, 3.0, 0.0, freeze=0.5, freeze_at=12.0)
    assert voiced(held, [(3.0, ())]) == ((10.0, 12.5),)


def test_stats_report_the_speech_level_metrics():
    pl = TimelinePlanner()
    pl.place(LineSlot(0, "a", 0, 4, 6, None, speech=((0.0, 3.0),)), 3.2, pauses=((1.0, 1.4),))
    pl.place(LineSlot(1, "b", 6, 9, None, 4, breaks=((7.0, 8.2),)), 1.8, pieces=[(0.9, ()), (0.9, ())])
    st_ = pl.stats()
    # line 0: voiced to 3.2 against speech to 3.0; line 1 (its span as its speech): the second piece ends at 9.1
    assert st_["speech_lines"] == 2 and st_["end_error_p50"] == pytest.approx((0.2 + 0.1) / 2, abs=1e-3)
    assert st_["pieced_lines"] == 1 and st_["anchored_lines"] == st_["joined_lines"] == 0
    assert st_["anchors"] == 1 and st_["freeze_s_per_10min"] == 0.0
    for key in ("overlap_speech_mean", "silent_s_per_min", "anchor_err_p95", "long_end_error_p50", "end_early_1s_share"):
        assert key in st_


def _run_windowed(rows, s):
    slots, pl = line_slots(rows), TimelinePlanner(s)
    d = [r["audio_s"] for r in sorted(rows, key=lambda r: (r["start"], r["id"]))]
    return slots, [pl.place(sl, d[i], ahead=window(slots, i, d, s)) for i, sl in enumerate(slots)], pl


@settings(max_examples=60, deadline=None)
@given(talks(), settings_st)
def test_windowed_placement_keeps_every_invariant(rows, s):
    slots, plans, pl = _run_windowed(rows, s)
    by_id = {r["id"]: r for r in rows}
    for k, (sl, p) in enumerate(zip(slots, plans)):
        assert p.wall * p.rate == pytest.approx(by_id[sl.id]["audio_s"], rel=1e-9, abs=1e-12)
        assert 1.0 - TOL <= p.rate <= s.speed_cap + TOL
        assert p.start >= sl.start - s.lead_max - TOL
        limit = lag_limit(sl, s)
        blocked = any(q.overdraft > 0 and q.end + s.min_gap > sl.start + limit - TOL for q in plans[:k])
        assert p.lag <= limit + TOL or blocked
        for sk, q in zip(slots[:k], plans[:k]):
            if sk.speaker != sl.speaker and sk.end > sl.start + 1e-9:
                assert p.start >= q.end - (sk.end - sl.start) - TOL
            else:
                assert p.start >= q.end + s.min_gap - TOL
    held = [(p.freeze_at, p.freeze) for p in plans if p.freeze > 0]
    for w, _ in held:
        assert sum(f for t, f in held if w - 60.0 < t <= w + TOL) <= s.freeze_budget + TOL
    assert _run_windowed(rows, s)[1] == plans                             # deterministic


@st.composite
def pieced_talks(draw):
    """Lines with a hard break inside, said in two pieces; another speaker may talk into the break."""
    rows, t = [], draw(st.floats(0, 2))
    for i in range(draw(st.integers(1, 8))):
        a = draw(st.floats(0.5, 4.0))
        pause = draw(st.floats(1.0, 2.5))
        b = draw(st.floats(0.5, 4.0))
        heard = draw(st.none() | st.floats(0.05, 1.0))  # another speaker in the pause, up to this share of it
        rows.append(dict(id=i, speaker="a" if i % 2 else "b", start=t, end=t + a + pause + b, brk=(t + a, t + a + pause),
                         pieces=(a * draw(st.floats(0.4, 1.6)), b * draw(st.floats(0.4, 1.6))),
                         heard=((t + a, t + a + heard * pause),) if heard else ()))
        t += a + pause + b + draw(st.floats(0.1, 2.0))
    return rows


@settings(max_examples=80, deadline=None)
@given(pieced_talks())
def test_pieces_are_never_trimmed_and_each_stays_within_its_lag_limit(rows):
    slots = [replace(sl, breaks=(r["brk"],), heard=r["heard"]) for sl, r in zip(line_slots(rows), rows)]
    pl = TimelinePlanner()
    for i, (sl, r) in enumerate(zip(slots, rows)):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        p = pl.place(sl, sum(r["pieces"]), sum(rows[i + 1]["pieces"]) if nxt else None, nxt,
                     pieces=[(d, ()) for d in r["pieces"]])
        assert p.said in ("pieces", "joined") and len(p.parts) == 2
        assert sum(pt.wall * pt.rate for pt in p.parts) == pytest.approx(sum(r["pieces"]) + (S.piece_gap if p.said == "joined" else 0))
        assert p.parts[1].start >= p.parts[0].start + p.parts[0].wall - TOL
        assert p.end == pytest.approx(p.parts[1].start + p.parts[1].wall - p.freeze)
        if p.said == "pieces":                                            # the first piece met its break
            assert p.parts[1].start - r["brk"][1] <= S.max_lag_resync + TOL and p.parts[1].src == r["brk"][1]
            assert p.parts[1].start >= r["brk"][1] - S.lead_max - TOL
            for _, h in r["heard"]:                                       # an early start only into silence
                assert p.parts[1].start >= min(h + S.guard, r["brk"][1]) - TOL
    assert pl.stats()["lines"] == len(rows)
