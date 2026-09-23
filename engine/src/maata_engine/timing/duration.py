"""DurationEstimator (§6.6): per-voice speaking rate, refined online after every synthesis.

Model: duration ≈ overhead + units / rate. Updated with a Huber-weighted recursive least squares
on (units, seconds) pairs, so one bad synthesis can't drag the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text.akshara import count_units

DEFAULT_RATE = 5.5   # Telugu aksharas per second, conversational (provisional; S4 measures)
DEFAULT_OVERHEAD = 0.15


@dataclass
class _VoiceModel:
    inv_rate: float = 1.0 / DEFAULT_RATE
    overhead: float = DEFAULT_OVERHEAD
    # 2x2 RLS covariance, initialised with a weak prior
    p: list[list[float]] = field(default_factory=lambda: [[0.5, 0.0], [0.0, 0.01]])
    n: int = 0


@dataclass
class DurationEstimator:
    huber_delta: float = 0.35  # seconds
    forgetting: float = 0.98
    voices: dict[str, _VoiceModel] = field(default_factory=dict)

    def _m(self, voice: str) -> _VoiceModel:
        return self.voices.setdefault(voice, _VoiceModel())

    def rate(self, voice: str) -> float:
        return 1.0 / max(self._m(voice).inv_rate, 1e-3)

    def estimate(self, text: str, voice: str) -> float:
        m = self._m(voice)
        return max(m.overhead + count_units(text) * m.inv_rate, 0.05)

    def observe(self, text: str, voice: str, actual: float) -> None:
        m = self._m(voice)
        x = (1.0, count_units(text))
        pred = m.overhead + x[1] * m.inv_rate
        err = actual - pred
        w = 1.0 if abs(err) <= self.huber_delta else self.huber_delta / abs(err)
        p = m.p
        px = (p[0][0] * x[0] + p[0][1] * x[1], p[1][0] * x[0] + p[1][1] * x[1])
        denom = self.forgetting / w + x[0] * px[0] + x[1] * px[1]
        k = (px[0] / denom, px[1] / denom)
        m.overhead += k[0] * err
        m.inv_rate += k[1] * err
        m.inv_rate = min(max(m.inv_rate, 1 / 20.0), 1 / 1.5)
        m.overhead = min(max(m.overhead, 0.0), 1.0)
        m.p = [[(p[i][j] - k[i] * px[j]) / self.forgetting for j in range(2)] for i in range(2)]
        m.n += 1
