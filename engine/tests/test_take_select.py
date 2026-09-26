"""Take selection without a Telugu recogniser (ARCHITECTURE §3.8, §3.13): drop cap hits and takes far shorter than
their wording, then pick by duration fit to the line's speech time."""

from maata_engine.qa.take import failure, pick


def test_a_take_that_ran_to_its_cap_or_is_far_too_short_failed():
    assert failure(4.0, capped=True, estimate=4.0) == "cap"
    assert failure(2.3, capped=False, estimate=4.0) == "short"       # under 0.6 x 4.0 s
    assert failure(2.4, capped=False, estimate=4.0) is None
    assert failure(9.0, capped=False, estimate=4.0) is None          # long is the band rule's and the planner's business


def test_the_take_closest_to_the_speech_time_wins_among_those_that_passed():
    assert pick([3.0, 4.2, 5.0], [None, None, None], target=4.0) == 1
    assert pick([3.0, 5.0], [None, None], target=4.0) == 1            # by ratio: 5/4 is closer than 3/4
    assert pick([4.0, 4.0], [None, None], target=4.0) == 0            # a tie keeps the first
    assert pick([4.0, 2.0, 3.0], ["cap", None, None], target=4.0) == 2


def test_when_every_take_failed_the_least_bad_is_voiced():
    assert pick([8.0, 1.0], ["cap", "short"], target=4.0) == 1        # a short take beats a runaway
    assert pick([1.0, 1.5], ["short", "short"], target=4.0) == 1
    assert pick([4.0, 1.5], ["cap", "cer"], target=4.0) == 1          # a reason step-6 QA adds ranks with "short"
