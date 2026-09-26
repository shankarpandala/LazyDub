from hypothesis import given
from hypothesis import strategies as st

from maata_engine.pacing import ReadyRanges, Throughput, takes_for, target_lead


def test_ranges_merge_and_lead():
    r = ReadyRanges()
    r.add(10, 20)
    r.add(30, 40)
    r.add(20, 30)  # touching ranges merge into one
    assert r.ranges == [[10, 40]]
    assert r.lead(15) == 25
    assert r.lead(45) == 0
    r.add(50, 60)
    assert r.ranges == [[10, 40], [50, 60]]
    assert r.contiguous_end(55) == 60
    assert r.covered(12, 38) and not r.covered(35, 55)


def test_lead_counts_a_range_that_starts_just_after_the_playhead():
    r = ReadyRanges()
    r.add(10.1, 30)
    assert r.lead(10.0) == 20.0


@given(st.lists(st.tuples(st.floats(0, 1000), st.floats(0, 50)), max_size=40))
def test_ranges_stay_sorted_and_disjoint(items):
    r = ReadyRanges()
    for a, d in items:
        r.add(a, a + d)
    for (s0, e0), (s1, e1) in zip(r.ranges, r.ranges[1:]):
        assert s0 < e0 and e0 < s1 and s1 < e1


def test_throughput_needs_enough_wall_time_then_reports_the_recent_rate():
    t = Throughput(horizon=100, min_span=10)
    t.record(0, 0)
    t.record(5, 10)
    assert t.rate() == 0.0  # only 5 s observed
    t.record(20, 30)  # 40 video seconds over 20 wall seconds
    assert t.rate() == 2.0
    for w in range(21, 400):
        t.record(w, 0.5)  # slows to 0.5 x
    assert abs(t.rate() - 0.5) < 0.05


def test_target_lead():
    assert target_lead(1000, 0.0) == 0.0          # unknown: UI falls back to its setting
    assert target_lead(1000, 0.5) == 530.0        # half speed: bank half of what is left, plus margin
    assert target_lead(100, 0.5) == 80.0
    assert target_lead(1000, 1.3) == 60.0          # faster than playback: small cushion
    assert target_lead(40, 1.3) == 40.0            # never more than what is left


def test_throughput_leaves_out_the_time_the_pipeline_sat_caught_up():
    t = Throughput(horizon=100, min_span=10)
    t.record(0, 0)
    t.record(20, 30)                 # 1.5 x while working
    t.pause(20)                      # everything up to the horizon is dubbed: it waits for the playhead
    assert t.rate() == 1.5
    t.resume(320)                    # five minutes later the playhead has moved on
    t.record(340, 30)
    assert t.rate() == 1.5           # the wait isn't counted as a slow engine
    t.pause(340)
    assert t.working(400) == 40.0 and t.rate() == 1.5  # no work time passes while caught up (and the session records
    # nothing then: a stretch heard past the horizon, for brief v1, isn't the dub's pace; test_session_realtime)
    t.pause(400)                     # pausing twice keeps the first start
    t.resume(410)
    assert t.working(410) == 40.0


def test_the_governor_takes_one_take_when_slow_and_behind_more_when_comfortable():
    # lead under target with r < 1.1 (or not measured yet): one take, a second only if it fails
    assert takes_for(lead=30.0, target=60.0, throughput=1.05, speech_s=5.0) == 1
    assert takes_for(lead=30.0, target=60.0, throughput=0.0, speech_s=5.0) == 1
    # under target but gaining on playback: two, never three
    assert takes_for(lead=30.0, target=60.0, throughput=1.4, speech_s=1.5) == 2
    # comfortable: two, three for a short line
    assert takes_for(lead=200.0, target=60.0, throughput=0.9, speech_s=5.0) == 2
    assert takes_for(lead=200.0, target=60.0, throughput=1.4, speech_s=2.0) == 3
    # after a seek into video not processed yet: one until the lead is back
    assert takes_for(lead=200.0, target=60.0, throughput=1.4, speech_s=2.0, recovering=True) == 1
