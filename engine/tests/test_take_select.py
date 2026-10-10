"""Take selection without a Telugu recogniser (ARCHITECTURE §3.8, §3.13): drop cap hits and takes far shorter than
their wording, then pick by duration fit to the line's speech time."""

import numpy as np
import pytest

from maata_engine.qa.take import audio_failure, failure, pick


@pytest.mark.parametrize(("audio", "reason"), [
    (np.array([], dtype=np.float32), "audio_empty"),
    (np.zeros(100, np.float32), "audio_silent"),
    (np.full(100, 1e-8, np.float32), "audio_silent"),
    (np.array([0.1, np.nan]), "audio_nonfinite"),
    (np.array([0.1, np.inf]), "audio_nonfinite"),
    (np.ones((2, 100), np.float32), "audio_shape"),
])
def test_selected_waveform_rejects_broken_audio(audio, reason):
    assert audio_failure(audio) == reason


def test_quiet_audio_and_long_pauses_are_not_mistaken_for_failed_speech():
    quiet = 1e-4 * np.sin(np.arange(100))
    assert audio_failure(quiet) is None
    assert audio_failure(np.concatenate([np.zeros(10_000), quiet, np.zeros(10_000)])) is None
    assert audio_failure(np.array([1.0, -1.0])) is None  # clipping is not proof of missing speech


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
