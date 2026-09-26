"""Pauses inside a take and uneven compression (ARCHITECTURE §3.10 steps 3 and 7): pure numpy, synthetic audio."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from maata_engine.timing import pauses as pz

SR = 24_000


def burst(seconds: float, f: float = 180.0) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def quiet(seconds: float, level: float = 0.0) -> np.ndarray:
    return np.full(round(seconds * SR), level, np.float32)


def take(*pieces: tuple[str, float]) -> np.ndarray:
    return np.concatenate([burst(s) if kind == "v" else quiet(s) for kind, s in pieces])


def test_pauses_are_quiet_runs_of_at_least_150_ms_and_the_leading_and_trailing_silence():
    x = take(("q", 0.1), ("v", 1.0), ("q", 0.4), ("v", 0.8), ("q", 0.1), ("v", 0.5), ("q", 0.3))
    got = pz.pauses(x, SR)
    # the 0.1 s gap inside is a stop closure, not a pause; the 0.1 s lead-in isn't voiced, so it counts however short
    assert len(got) == 3
    assert got[0] == pytest.approx((0.0, 0.1), abs=0.011)
    assert got[1] == pytest.approx((1.1, 1.5), abs=0.011) and got[2] == pytest.approx((2.9, 3.2), abs=0.011)
    assert pz.internal(got, 3.2) == (got[1],)
    assert pz.pauses(np.zeros(SR, np.float32), SR) == ((0.0, 1.0),)
    assert pz.pauses(np.zeros(0, np.float32), SR) == ()


@pytest.mark.parametrize("edge", [0.05, 0.1])
def test_a_short_lead_in_and_tail_are_not_voiced_and_not_squeezed(edge):
    x = take(("q", edge), ("v", 1.2), ("q", 0.3), ("v", 0.9), ("q", edge))
    total = len(x) / SR
    got = pz.pauses(x, SR)
    said = pz.voiced(got, total)
    assert said[0][0] == pytest.approx(edge, abs=0.011) and said[-1][1] == pytest.approx(total - edge, abs=0.011)
    assert len(pz.internal(got, total)) == 1                            # only the inner pause shortens
    assert len(pz.squeeze(got, total, 1.1, 0.12)[1]) == 1


def test_a_pause_is_judged_against_the_takes_own_loud_frames():
    # A soft room tone 50 dB under the speech is silence; a murmur 20 dB under it is not.
    loud = burst(1.0)
    assert len(pz.pauses(np.concatenate([loud, quiet(0.5, 0.2 * 10 ** (-50 / 20)), loud]), SR)) == 1
    assert pz.pauses(np.concatenate([loud, burst(0.5) * 0.1, loud]), SR) == ()


def test_voiced_is_the_take_less_its_pauses():
    assert pz.voiced(((0.0, 0.2), (1.0, 1.4), (2.8, 3.0)), 3.0) == ((0.2, 1.0), (1.4, 2.8))
    assert pz.voiced((), 2.0) == ((0.0, 2.0),)
    assert pz.voiced(((0.0, 2.0),), 2.0) == ()


def test_squeeze_takes_time_from_the_pauses_before_the_speech():
    spans = ((0.0, 0.1), (1.0, 1.6), (2.6, 3.0), (4.0, 4.2))  # lead-in, two inner pauses, the tail
    # 4.2 s at 1.05x = 4.0 s: the inner pauses (0.48 + 0.28 spare over 0.12 each) give the 0.2 s; speech stays at 1.0
    r, cuts = pz.squeeze(spans, 4.2, 1.05, 0.12)
    assert r == 1.0 and sum(q for _, q in cuts) == pytest.approx(4.2 - 4.0)
    assert [m for m, _ in cuts] == [1.3, 2.8]                          # at each pause's middle
    assert cuts[0][1] / cuts[1][1] == pytest.approx(0.48 / 0.28)       # in proportion to what each can spare
    # 1.25x needs 0.84 s: the pauses give all they can spare (0.76 at 1.0, a little less once the speech is sped up), the
    # speech the rest, and never faster than the plan's rate
    r, cuts = pz.squeeze(spans, 4.2, 1.25, 0.12)
    assert 1.0 < r < 1.25
    assert 4.2 / r - sum(q for _, q in cuts) == pytest.approx(4.2 / 1.25, abs=1e-6)
    assert all((b - a) / r - q == pytest.approx(0.12) for (a, b), (_, q) in zip(spans[1:3], cuts))
    assert pz.squeeze(spans, 4.2, 1.0, 0.12) == (1.0, ())
    assert pz.squeeze((), 4.2, 1.2, 0.12) == (1.2, ())                  # no pauses: all speech


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.floats(0.05, 1.5), st.floats(0.0, 1.2)), min_size=1, max_size=6),
       st.floats(1.0, 1.25), st.floats(0.05, 0.2))
def test_squeeze_always_lands_on_the_planned_length(segments, rate, keep):
    spans, t = [], 0.0
    for speech, pause in segments:
        t += speech
        if pause > 0:
            spans.append((t, t + pause))
            t += pause
    total = t + 0.3
    r, cuts = pz.squeeze(tuple(spans), total, rate, keep)
    assert 1.0 <= r <= rate + 1e-9
    assert total / r - sum(q for _, q in cuts) == pytest.approx(total / rate, abs=1e-6)
    for (m, q) in cuts:                                                  # every pause keeps at least `keep`
        a, b = next((a, b) for a, b in spans if a < m < b)
        assert (b - a) / r - q >= keep - 1e-9 or r == 1.0 and (b - a) - q >= keep - 1e-9
    # out_time is monotone over the take's voiced stretches, and the take ends at total / rate
    ts = sorted({0.0, total, *(x for a, b in pz.voiced(tuple(spans), total) for x in (a, b))})
    outs = [pz.out_time(v, r, cuts) for v in ts]
    assert all(b >= a - 1e-9 for a, b in zip(outs, outs[1:]))
    assert outs[-1] == pytest.approx(total / rate, abs=1e-6)


def test_cut_takes_the_silence_out_and_returns_exactly_the_planned_length():
    x = take(("v", 1.0), ("q", 0.6), ("v", 1.0))
    spans = pz.pauses(x, SR)
    r, cuts = pz.squeeze(spans, len(x) / SR, 1.1, 0.12)
    assert r == 1.0 and len(cuts) == 1
    n = round(len(x) / SR / 1.1 * SR)
    y = pz.cut(x, SR, r, cuts, n)
    assert len(y) == n and y.dtype == np.float32
    # the speech is all there: only the pause shrank
    assert np.abs(y[: round(0.99 * SR)]).max() > 0.1 and np.abs(y[-round(0.99 * SR):]).max() > 0.1
    left = pz.internal(pz.pauses(y, SR), len(y) / SR)
    assert len(left) == 1 and left[0][1] - left[0][0] == pytest.approx(0.6 - cuts[0][1], abs=0.02)
    # a rounding sample short or long is padded or trimmed
    assert len(pz.cut(x, SR, 1.0, (), len(x) + 3)) == len(x) + 3 and len(pz.cut(x, SR, 1.0, (), 10)) == 10
