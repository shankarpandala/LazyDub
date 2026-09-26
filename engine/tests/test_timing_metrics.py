"""Speech-level timing yardsticks and the §3.10 acceptance guards (timing/metrics.py). Pure."""

import pytest

from maata_engine.timing.metrics import GUARDS, guards, speech_metrics


def row(start, end, speech, voiced, anchors=()):
    return {"start": start, "end": end, "speech": speech, "voiced": voiced, "anchor_errors": list(anchors)}


def test_end_error_is_measured_against_the_last_speech_not_the_span():
    # The English stops talking at 13.0 although the unit runs to 14.0 (the last word's timestamp took in the silence):
    # a dub ending at 13.1 is 0.1 s late, not 0.9 s early.
    m = speech_metrics([row(10.0, 14.0, [[10.0, 11.5], [11.9, 13.0]], [[10.0, 12.2], [12.4, 13.1]])])
    assert m["speech_lines"] == 1 and m["end_error_p50"] == pytest.approx(0.1)
    assert m["end_early_1s_share"] == 0.0 and m["long_lines"] == 0 and m["long_end_error_p50"] is None


def test_early_endings_and_long_lines_are_counted():
    rows = [row(0.0, 3.0, [[0.0, 3.0]], [[0.0, 1.5]]),                 # 1.5 s early
            row(5.0, 14.0, [[5.0, 14.0]], [[5.0, 12.0]]),              # a long line, 2 s early
            row(20.0, 23.0, [[20.0, 23.0]], [[20.1, 23.2]])]           # 0.2 s late
    m = speech_metrics(rows)
    assert m["end_error_p50"] == pytest.approx(-1.5) and m["end_early_1s_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["long_lines"] == 1 and m["long_end_error_p50"] == pytest.approx(-2.0)


def test_overlap_is_intersection_over_union_of_speech_and_voiced_time():
    m = speech_metrics([row(0.0, 4.0, [[0.0, 2.0], [3.0, 4.0]], [[0.5, 3.5]])])
    # shared: 1.5 + 0.5 = 2.0; union: 3.0 + 3.0 - 2.0 = 4.0
    assert m["overlap_speech_mean"] == pytest.approx(0.5)


def test_silent_seconds_per_minute_of_speech_count_skipped_lines_and_gaps_in_the_dub():
    rows = [row(0.0, 20.0, [[0.0, 20.0]], [[0.0, 18.0]]),              # 2 s of speech with no dub over it
            row(30.0, 40.0, [[30.0, 40.0]], []),                       # a skipped line: all 10 s silent
            row(50.0, 80.0, [[50.0, 80.0]], [[49.8, 80.0]])]           # fully covered (the early start counts nothing)
    m = speech_metrics(rows)
    assert m["silent_s_per_min"] == pytest.approx(12.0 / 1.0)          # 12 s over one minute of speech
    assert m["speech_lines"] == 2                                        # the skipped line has no end error


def test_a_dub_over_another_lines_speech_covers_it():
    # The dub of line 0 runs on over line 1's English, and line 1's dub starts late: its first second is still covered.
    rows = [row(0.0, 2.0, [[0.0, 2.0]], [[0.0, 3.0]]), row(2.0, 4.0, [[2.0, 4.0]], [[3.1, 4.0]])]
    assert speech_metrics(rows)["silent_s_per_min"] == pytest.approx(0.1 / (4.0 / 60.0))


def test_anchor_onset_errors_are_absolute():
    m = speech_metrics([row(0.0, 5.0, [[0.0, 5.0]], [[0.0, 5.0]], anchors=[0.1, -0.3, 0.0])])
    assert m["anchors"] == 3 and m["anchor_err_p50"] == pytest.approx(0.1) and m["anchor_err_p95"] == pytest.approx(0.28)


def test_no_rows_or_no_speech_say_none():
    m = speech_metrics([])
    assert m["speech_lines"] == 0 and m["end_error_p50"] is None and m["silent_s_per_min"] is None
    assert m["anchors"] == 0 and m["anchor_err_p50"] is None and m["overlap_speech_mean"] is None


def test_guards_hold_the_regression_limits_and_measure_the_targets_against_the_baseline():
    stats = {"lag_p95": 0.25, "rate_p90": 1.12, "freeze_s_per_10min": 0.5, "rate_step_p90": 0.08,
             "silent_s_per_min": 2.5, "end_error_p50": -0.2, "end_early_1s_share": 0.1, "long_end_error_p50": -0.9}
    base = {"rate_step_p90": 0.1, "end_early_1s_share": 0.25, "long_end_error_p50": -1.98}
    g = guards(stats, base)
    assert set(GUARDS) <= set(g)
    assert g["lag_p95"]["ok"] is True and g["rate_p90"] == {"value": 1.12, "limit": 1.1, "ok": False}
    assert g["freeze_s_per_10min"]["ok"] is True and g["rate_step_p90"]["ok"] is True
    assert g["silent_s_per_min"]["ok"] is True and g["end_error_p50"]["ok"] is True
    assert g["end_early_1s_share"] == {"value": 0.1, "limit": 0.125, "ok": True}       # at least halved
    assert g["long_end_error_p50"] == {"value": -0.9, "limit": 0.99, "ok": True}
    # without a baseline, what is measured against it says nothing
    g = guards(stats)
    assert g["rate_step_p90"]["ok"] is None and g["end_early_1s_share"]["ok"] is None
    assert g["long_end_error_p50"]["ok"] is None and g["lag_p95"]["ok"] is True
