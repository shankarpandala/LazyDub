"""DurationEstimator (§6.6): per-voice speaking rate, set by a calibration and refined online after every synthesis.

Model: duration ≈ overhead + units / rate, with units counted on the Telugu-script line (`count_units`). A voice is
keyed by everything its pace depends on (`VoiceKey`): cfg 0.3 speaks 6-9 % slower than 0.5, and the reference audio moves
it more (docs/research/dubbing-2026-09/gap-4.md E2, E3), so changing any of them makes a new voice. Calibration sets rate
and overhead directly from takes of different lengths (one sentence misjudges the rate by -13 % to +18 %); after that a
Huber-weighted recursive least squares on (units, seconds) pairs refines them, so one bad synthesis can't drag the
estimate.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from ..text.akshara import count_units

# Aksharas per second of a Telugu-script line. 5.5 matched the calibrated voices when English was counted in Latin
# syllables; English written in Telugu script counts about 9 % more (research gap-2 C). A calibration replaces it.
DEFAULT_RATE = 6.1
DEFAULT_OVERHEAD = 0.15
MIN_RATE, MAX_RATE, MAX_OVERHEAD = 1.5, 20.0, 1.0
_PRIOR_P = ((0.5, 0.0), (0.0, 0.01))  # the weak prior's 2x2 covariance on (overhead, 1 / rate)


@dataclass(frozen=True, slots=True)
class VoiceKey:
    """What a voice's pace depends on: the voice (a speaker's clone, or a preset's name), its guidance weight (cfg), its
    exaggeration and a hash of the reference audio it was built from. None where the TTS has no such setting."""

    voice: str
    cfg: float | None = None
    exaggeration: float | None = None
    reference: str = ""


@dataclass
class _VoiceModel:
    inv_rate: float = 1.0 / DEFAULT_RATE
    overhead: float = DEFAULT_OVERHEAD
    # 2x2 RLS covariance, initialised with a weak prior
    p: list[list[float]] = field(default_factory=lambda: [list(row) for row in _PRIOR_P])
    n: int = 0


@dataclass
class DurationEstimator:
    huber_delta: float = 0.35  # seconds
    forgetting: float = 0.98
    voices: dict[Hashable, _VoiceModel] = field(default_factory=dict)

    def _m(self, voice: Hashable) -> _VoiceModel:
        return self.voices.setdefault(voice, _VoiceModel())

    def rate(self, voice: Hashable) -> float:
        return 1.0 / max(self._m(voice).inv_rate, 1e-3)

    def overhead(self, voice: Hashable) -> float:
        return self._m(voice).overhead

    def estimate(self, text: str, voice: Hashable) -> float:
        m = self._m(voice)
        return max(m.overhead + count_units(text) * m.inv_rate, 0.05)

    def observe(self, text: str, voice: Hashable, actual: float) -> None:
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
        m.inv_rate = min(max(m.inv_rate, 1 / MAX_RATE), 1 / MIN_RATE)
        m.overhead = min(max(m.overhead, 0.0), MAX_OVERHEAD)
        m.p = [[(p[i][j] - k[i] * px[j]) / self.forgetting for j in range(2)] for i in range(2)]
        m.n += 1

    def calibrate(self, voice: Hashable, takes: Iterable[tuple[str, float]]) -> bool:
        """Set a voice's rate and overhead from calibration takes, (text, natural seconds), by least squares. Its
        covariance becomes the prior's plus what the takes say, so later lines refine the fit as they would have if
        they had come after the takes. With fewer than two lengths to fit, each take is an ordinary online update
        instead. Returns whether the voice was set directly."""
        takes = list(takes)
        pts = [(count_units(text), seconds) for text, seconds in takes]
        if len({u for u, _ in pts}) < 2:
            for text, seconds in takes:
                self.observe(text, voice, seconds)
            return False
        n = len(pts)
        su, ss = sum(u for u, _ in pts), sum(s for _, s in pts)
        suu = sum(u * u for u, _ in pts)
        mu, ms = su / n, ss / n
        inv = sum((u - mu) * (s - ms) for u, s in pts) / sum((u - mu) ** 2 for u, _ in pts)
        overhead = ms - inv * mu
        if not 0.0 <= overhead <= MAX_OVERHEAD:  # refit the rate through the nearest overhead the model allows
            overhead = min(max(overhead, 0.0), MAX_OVERHEAD)
            inv = sum(u * (s - overhead) for u, s in pts) / suu
        inv = min(max(inv, 1 / MAX_RATE), 1 / MIN_RATE)
        # (P0^-1 + X'X)^-1 for rows x = (1, units), with P0 the weak prior
        a, b, d = 1 / _PRIOR_P[0][0] + n, su, 1 / _PRIOR_P[1][1] + suu
        det = a * d - b * b
        self.voices[voice] = _VoiceModel(inv, overhead, [[d / det, -b / det], [-b / det, a / det]], n)
        return True
