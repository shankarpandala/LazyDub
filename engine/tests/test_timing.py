from hypothesis import example, given, settings, strategies as st

import pytest

from maata_engine.timing.duration import DEFAULT_RATE, DurationEstimator, VoiceKey
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
    r = fit_unit(1, 10, 12, 12.1, 2.8, S, led)        # needs 1.4×: cap 1.25 then slow video
    kinds = [e.kind for e in r.placement.edits]
    assert r.placement.audio_rate == 1.25 and kinds == [EditKind.SLOW]
    r = fit_unit(2, 20, 22, 22.1, 3.4, S, led)        # beyond 0.75× → freeze too
    assert [e.kind for e in r.placement.edits] == [EditKind.SLOW, EditKind.FREEZE]


def test_huge_overflow_asks_for_condense():
    r = fit_unit(0, 0, 2, 2.1, 9.0, S, AddedTimeLedger(S))
    assert r.verdict is Verdict.CONDENSE and 0 < r.condense_target_ratio < 1


def test_overdraft_places_huge_overflow_with_one_long_freeze():
    led = AddedTimeLedger(S)
    r = fit_unit(0, 0, 2, 2.1, 9.0, S, led, overdraft=True)
    p = r.placement
    assert r.verdict is Verdict.PLACED and p.audio_rate == S.speed_cap and p.audio_wall == 9.0 / S.speed_cap
    assert [e.kind for e in p.edits] == [EditKind.SLOW, EditKind.FREEZE]
    assert abs(p.added - (p.audio_wall - p.budget)) < 1e-9 and p.pad_after == 0.0
    assert abs(p.overdraft - (p.edits[1].added - S.max_freeze)) < 1e-9  # only the freeze beyond max_freeze
    assert abs(led.added_in_window(p.budget) - (p.added - p.overdraft)) < 1e-9  # the budget counts only the in-limit part


def test_overdraft_is_a_no_op_when_the_line_fits_the_limits():
    led = AddedTimeLedger(S)
    plain = fit_unit(0, 20, 22, 22.1, 3.4, S, led, commit=False)
    r = fit_unit(0, 20, 22, 22.1, 3.4, S, led, overdraft=True)
    assert r == plain and r.placement.overdraft == 0.0


def test_overdraft_past_the_rolling_budget():
    led = AddedTimeLedger(S)
    led.record(0.0, S.added_budget)                 # this minute's budget is spent
    assert fit_unit(0, 10, 12, 12.1, 3.4, S, led, commit=False).verdict is Verdict.CONDENSE
    r = fit_unit(0, 10, 12, 12.1, 3.4, S, led, overdraft=True)
    assert r.verdict is Verdict.PLACED and abs(r.placement.overdraft - r.placement.added) < 1e-9
    assert led.added_in_window(30) == S.added_budget  # all overdraft: nothing more is counted


def test_one_overdrawn_line_does_not_make_the_next_minute_condense():
    led = AddedTimeLedger(S)
    r = fit_unit(0, 0, 4, 4.1, 30.0, S, led, overdraft=True)  # 20 s added, 15.7 s of it overdraft
    assert r.placement.overdraft > S.added_budget
    for i in range(1, 15):  # then lines needing ~1.4x: a ~0.46 s slow-down each, within every limit
        s0 = 4.1 + (i - 1) * 4.0
        dur = slot_budget(s0, s0 + 3.5, s0 + 4.0, S) * 1.4
        assert fit_unit(i, s0, s0 + 3.5, s0 + 4.0, dur, S, led).verdict is Verdict.PLACED


def test_rolling_budget_forces_condense():
    led = AddedTimeLedger(S)
    t = 0.0
    verdicts = []
    for i in range(12):
        verdicts.append(fit_unit(i, t, t + 2, t + 2.1, 4.4, S, led).verdict)  # 1.52 s added each
        t += 3
    assert Verdict.CONDENSE in verdicts
    assert led.added_in_window(t) <= S.added_budget + 1e-6


def test_discrete_player_rates():
    s = TimingSettings(available_video_rates=(0.25, 0.5, 0.75, 1.0, 1.25))
    r = fit_unit(0, 0, 2, 2.1, 2.8, s, AddedTimeLedger(s))
    # needs ≈ 0.89×; the slowest player rate at or above that is 1.0, so no slow-down: freeze only
    assert [e.kind for e in r.placement.edits] == [EditKind.FREEZE]
    r = fit_unit(1, 10, 12, 12.1, 3.4, s, AddedTimeLedger(s))
    # needs ≈ 0.74×, below the 0.75 floor: slow to 0.75 (not 0.5), freeze the rest, no negative pad
    p = r.placement
    assert [(e.kind, e.rate) for e in p.edits] == [(EditKind.SLOW, 0.75), (EditKind.FREEZE, 0.0)]
    assert p.pad_after >= 0 and abs(p.added - (p.audio_wall - p.budget)) < 1e-9


units = st.lists(
    st.tuples(st.floats(0.5, 12), st.floats(0.0, 3.0), st.floats(0.3, 18)),  # (length, gap after, dub duration)
    min_size=1, max_size=40,
)


@settings(max_examples=300, deadline=None)
@given(units)
# A line just before a window's lower edge whose freeze (at its slot end) falls inside the window.
@example([(2.0, 8.0, 8.4)] + [(2.0, 1.0, 5.0)] * 6 + [(1.0, 31.0, 1.0), (1.0, 0.2, 4.2), (1.0, 1.0, 1.0)])
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


COARSE = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)  # the IFrame player's rates


@settings(max_examples=300, deadline=None)
@given(units, st.booleans(), st.floats(0.0, 12.0))
def test_overdraft_never_loses_audio(spec, coarse, spent):
    """never_trim: every line is placed at s_u, whole, and the video yields all of its overflow."""
    s = TimingSettings(available_video_rates=COARSE if coarse else None)
    tl, led = DubTimeline(), AddedTimeLedger(s)
    led.record(0.0, spent)  # start with part of the first minute's budget already used
    t, rows = 0.0, []
    for i, (length, gap, _) in enumerate(spec):
        rows.append((i, t, t + length))
        t += length + gap
    for k, (i, s0, e0) in enumerate(rows):
        nxt = rows[k + 1][1] if k + 1 < len(rows) else None
        dur = spec[k][2]
        plain = fit_unit(i, s0, e0, nxt, dur, s, led, commit=False)
        r = fit_unit(i, s0, e0, nxt, dur, s, led, overdraft=True)
        p = r.placement
        assert r.verdict is Verdict.PLACED and p is not None
        assert p.start == s0
        assert abs(p.audio_wall * p.audio_rate - dur) < 1e-9          # no audio lost
        assert 1.0 <= p.audio_rate <= s.speed_cap + 1e-9
        assert p.added >= max(p.audio_wall - p.budget, 0.0) - 1e-6    # the video yields all overflow
        assert p.pad_after >= 0 and p.overdraft >= 0
        assert all(e.rate >= s.min_video_rate - 1e-9 for e in p.edits if e.kind is EditKind.SLOW)
        if plain.verdict is Verdict.PLACED:
            assert p == plain.placement and p.overdraft == 0.0
        else:
            assert p.overdraft > 0
        tl.add(p)
    iv = tl.audible_intervals()
    for (_, _, end), (_, start, _) in zip(iv, iv[1:]):
        assert end <= start + 1e-6                    # still no two dub lines overlap


def test_duration_estimator_learns_rate():
    est = DurationEstimator()
    text = "నమస్కారం మీరు ఎలా ఉన్నారు"
    for _ in range(30):
        est.observe(text, "v", 0.2 + 12 / 8.0)  # true rate 8 aksharas/s, 0.2 s overhead
    assert abs(est.estimate(text, "v") - (0.2 + 12 / 8.0)) < 0.1


# ---- calibration and voice keys (ARCHITECTURE §3.7 step 5) -------------------------------------------------------
def takes_at(rate: float, overhead: float, lengths=(15, 27, 45)) -> list[tuple[str, float]]:
    """Calibration takes of a voice that says `rate` aksharas per second after `overhead` s: "ప" is one akshara."""
    return [("ప" * n, overhead + n / rate) for n in lengths]


def test_calibration_sets_rate_and_overhead_directly_from_three_lengths():
    est = DurationEstimator()
    key = VoiceKey("S1", 0.5, 0.5, "ref-a")
    assert est.calibrate(key, takes_at(5.2, 0.3))
    assert est.rate(key) == pytest.approx(5.2) and est.overhead(key) == pytest.approx(0.3)
    assert est.estimate("ప" * 20, key) == pytest.approx(0.3 + 20 / 5.2)


def test_a_voice_is_keyed_by_its_settings_and_its_reference():
    est = DurationEstimator()
    est.calibrate(VoiceKey("S1", 0.5, 0.5, "ref-a"), takes_at(5.2, 0.3))
    for other in (VoiceKey("S1", 0.3, 0.5, "ref-a"), VoiceKey("S1", 0.5, 0.7, "ref-a"), VoiceKey("S1", 0.5, 0.5, "ref-b"),
                  VoiceKey("S2", 0.5, 0.5, "ref-a")):
        assert est.rate(other) == pytest.approx(DEFAULT_RATE)  # another cfg, exaggeration, reference or voice: its own
    assert est.rate(VoiceKey("S1", 0.5, 0.5, "ref-a")) == pytest.approx(5.2)


def test_the_prior_is_5_5_rescaled_for_english_counted_in_telugu_script():
    assert 6.0 <= DEFAULT_RATE <= 6.3 and DEFAULT_RATE == pytest.approx(5.5 * 1.09, abs=0.15)


def test_a_fit_implying_a_negative_overhead_goes_through_zero():
    est = DurationEstimator()
    takes = takes_at(6.0, -0.4)
    assert est.calibrate("v", takes)
    units = [15, 27, 45]
    through_zero = sum(u * s for u, (_, s) in zip(units, takes)) / sum(u * u for u in units)
    assert est.overhead("v") == 0.0 and est.rate("v") == pytest.approx(1 / through_zero)


def test_the_rate_stays_in_range_whatever_the_takes_say():
    est = DurationEstimator()
    assert est.calibrate("v", [("ప" * 15, 0.3), ("ప" * 45, 0.4)])  # 300 aksharas/s between them
    assert est.rate("v") == pytest.approx(20.0)


def test_one_length_is_an_online_update_not_a_fit():
    est = DurationEstimator()
    assert not est.calibrate("v", [("ప" * 30, 0.2 + 30 / 5.0)] * 2)
    assert est.voices["v"].n == 2 and 5.0 < est.rate("v") < DEFAULT_RATE  # moved toward 5.0, not set to it


def test_after_a_calibration_one_bad_take_moves_the_pace_less_than_from_the_prior():
    calibrated, fresh = DurationEstimator(), DurationEstimator()
    calibrated.calibrate("v", takes_at(DEFAULT_RATE, 0.15))  # the same pace as the prior, but measured
    for est in (calibrated, fresh):
        est.observe("ప" * 30, "v", 9.0)  # a take that dragged
    assert fresh.rate("v") < calibrated.rate("v") < DEFAULT_RATE
