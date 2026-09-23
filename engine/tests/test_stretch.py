import numpy as np
import pytest

from maata_engine.timing.stretch import wsola

SR = 24000


def dominant_hz(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[np.argmax(spec)]


@pytest.mark.parametrize("rate", [0.8, 1.0, 1.1, 1.2, 1.5])
def test_length_exact_and_pitch_preserved(rate):
    t = np.arange(int(2.0 * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = wsola(x, rate, SR)
    assert len(y) == round(len(x) / rate)
    assert abs(dominant_hz(y[SR // 10: -SR // 10]) - 220) < 5
    rms = lambda a: float(np.sqrt(np.mean(a[SR // 10: -SR // 10] ** 2)))
    assert abs(rms(y) - rms(x)) / rms(x) < 0.15
