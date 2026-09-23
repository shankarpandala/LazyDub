from hypothesis import given, settings, strategies as st

from maata_engine.timing.duration import DurationEstimator
from maata_engine.timing.isochrony import (
    AddedTimeLedger, EditKind, TimingSettings, Verdict, fit_unit, slot_budget,
)
from maata_engine.timing.timeline import DubTimeline

S = TimingSettings()


def test_budget_borrows_silence_with_guard_and_cap():
    assert slot_budget(0, 2, 2.5, S) == 2 + 0.35
    assert slot_budget(0, 2, 5.0, S) == 2 + 0.8
    assert slot_budget(0, 2, 2.1, S) == 2


def test_short_line_is_padded_not_slowed():
    r = fit_unit(0, 10, 12, 14, 1.0, S, AddedTimeLedger(S))
    assert r.verdict is Verdict.PLACED and r.placement.audio_rate == 1.0 and r.placement.pad_after > 0


def test_speedup_capped_then_slow_then_freeze():
    led = AddedTimeLedger(S)
    r = fit_unit(0, 0, 2, 2.1, 2.2, S, led)          # 1.1× fits
    assert r.placement.audio_rate == 1.1 and not r.placement.edits
    r = fit_unit(1, 10, 12, 12.1, 2.8, S, led)        # needs 1.4×: cap 1.2 then slow video
    kinds = [e.kind for e in r.placement.edits]
    assert r.placement.audio_rate == 1.2 and kinds == [EditKind.SLOW]
    r = fit_unit(2, 20, 22, 22.1, 3.4, S, led)        # beyond 0.85× → freeze too
    assert [e.kind for e in r.placement.edits] == [EditKind.SLOW, EditKind.FREEZE]


def test_huge_overflow_asks_for_condense():
    r = fit_unit(0, 0, 2, 2.1, 9.0, S, AddedTimeLedger(S))
    assert r.verdict is Verdict.CONDENSE and 0 < r.condense_target_ratio < 1


def test_rolling_budget_forces_condense():
    led = AddedTimeLedger(S)
    t = 0.0
    verdicts = []
    for i in range(10):
        verdicts.append(fit_unit(i, t, t + 2, t + 2.1, 3.3, S, led).verdict)
        t += 3
    assert Verdict.CONDENSE in verdicts
    assert led.added_in_window(t) <= S.added_budget + 1e-6


def test_discrete_player_rates():
    s = TimingSettings(available_video_rates=(0.25, 0.5, 0.75, 1.0, 1.25))
    r = fit_unit(0, 0, 2, 2.1, 2.8, s, AddedTimeLedger(s))
    # 0.75 is below the 0.85 floor, so no slow-down: freeze only
    assert [e.kind for e in r.placement.edits] == [EditKind.FREEZE]


units = st.lists(
    st.tuples(st.floats(0.5, 12), st.floats(0.0, 3.0), st.floats(0.3, 18)),  # (length, gap after, dub duration)
    min_size=1, max_size=40,
)


@settings(max_examples=300, deadline=None)
@given(units)
def test_invariants(spec):
    """§6.6 invariants: exact starts, no overlap, speed cap, no slow audio, rolling budget."""
    tl, led = DubTimeline(), AddedTimeLedger(S)
    t, rows = 0.0, []
    for i, (length, gap, _) in enumerate(spec):
        rows.append((i, t, t + length))
        t += length + gap
    for k, (i, s0, e0) in enumerate(rows):
        nxt = rows[k + 1][1] if k + 1 < len(rows) else None
        dur = spec[k][2]
        r = fit_unit(i, s0, e0, nxt, dur, S, led)
        while r.verdict is Verdict.CONDENSE:  # the orchestrator condenses; simulate a rewrite
            dur *= r.condense_target_ratio
            r = fit_unit(i, s0, e0, nxt, dur, S, led)
        p = r.placement
        assert p.start == s0                          # planned to start exactly at s_u
        assert 1.0 <= p.audio_rate <= S.speed_cap + 1e-9  # never slowed, never above cap
        for e in p.edits:
            if e.kind is EditKind.SLOW:
                assert e.rate >= S.min_video_rate - 1e-9
            else:
                assert e.added <= S.max_freeze + 1e-9
        tl.add(p)
    iv = tl.audible_intervals()
    for (_, _, end), (_, start, _) in zip(iv, iv[1:]):
        assert end <= start + 1e-6                    # no two dub lines overlap
    for (_, s0, _) in rows:
        assert tl.added_between(max(s0 - S.budget_window, 0), s0 + 1e-9) <= S.added_budget + 1e-6


def test_duration_estimator_learns_rate():
    est = DurationEstimator()
    text = "నమస్కారం మీరు ఎలా ఉన్నారు"
    for _ in range(30):
        est.observe(text, "v", 0.2 + 12 / 8.0)  # true rate 8 aksharas/s, 0.2 s overhead
    assert abs(est.estimate(text, "v") - (0.2 + 12 / 8.0)) < 0.1
