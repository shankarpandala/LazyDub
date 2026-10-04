"""scripts/verify_mac.py's separator checks on synthetic stems (no model, no `say`): the English left in the bed reads a
leak at its energy, a filtered one too (a mask keeps a band of the speech, not a scaled copy), and never reads the music
or the artefacts as English; the bf16/float32 comparison reports each dtype's own separator."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("verify_mac", Path(__file__).parents[2] / "scripts" / "verify_mac.py")
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

SR = vm.SR
TURNS = np.array([[1.0, 6.0], [7.0, 12.5], [14.0, 19.0]])


def _stems(seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Speech-like [2, n] (a gliding voice with 30 harmonics in syllables, and fricative noise) over the turns, the
    helper's synthetic music under it at make_video's levels, and the turns' mask."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    on = vm._turn_mask(TURNS, n)
    phase = 2 * np.pi * np.cumsum(140 + 40 * np.sin(2 * np.pi * 0.7 * t)) / SR
    voiced = sum(np.sin(h * phase) / h for h in range(1, 31)) * np.sin(2 * np.pi * 2 * t) ** 2
    hiss = np.diff(np.random.default_rng(3).standard_normal(n + 1)) * (np.sin(2 * np.pi * 1.5 * t) > 0.8)
    speech = vm._rms_to(np.stack([(voiced + 0.3 * hiss) * on] * 2), vm.SPEECH_DB, on)
    return speech, vm._rms_to(vm._music(n), vm.MUSIC_DB), on


def _high_band(x: np.ndarray, share: float) -> np.ndarray:
    """The top of mono `x`'s spectrum that holds `share` of its energy."""
    spec = np.fft.rfft(x)
    e = np.abs(spec) ** 2
    top = np.cumsum(e[::-1])[::-1] / e.sum()
    spec[:int(np.argmax(top < share))] = 0
    return np.fft.irfft(spec, n=len(x))


def _db(x: np.ndarray, ref: np.ndarray, on: np.ndarray) -> float:
    return 10 * math.log10(float(np.sum(x[:, on] ** 2)) / float(np.sum(ref[:, on] ** 2)))


def test_a_clean_bed_reads_no_english_and_a_scaled_copy_reads_its_gain():
    speech, music, on = _stems()
    clean = vm.english_left(music, speech, music, on)
    assert clean["english_left_db"] < -60 and abs(clean["background_gain"] - 1.0) < 1e-3
    assert abs(vm.english_left(music + 0.1 * speech, speech, music, on)["english_left_db"] + 20.0) < 0.2


def test_a_band_of_the_speech_left_in_the_bed_reads_at_its_energy_and_fails():
    speech, music, on = _stems()
    band = np.stack([_high_band(speech[0], 0.09)] * 2)
    true = _db(band, speech, on)
    got = vm.english_left(music + band, speech, music, on)["english_left_db"]
    assert abs(got - true) < 1.0  # (one gain over all the turns read this at half its dB: about -21 dB)
    assert got > vm.ENGLISH_MAX_DB


def test_quieter_music_and_artefacts_are_not_read_as_english():
    speech, music, on = _stems()
    noise = vm._rms_to(np.random.default_rng(5).standard_normal(music.shape), vm.SPEECH_DB - 30.0, on)
    got = vm.english_left(0.8 * music + noise, speech, music, on)
    assert got["english_left_db"] < -35
    assert abs(got["artefacts_db"] + 30.0) < 1.0 and abs(got["background_gain"] - 0.8) < 0.01


class _Sep:
    chunk = 352_800

    def __init__(self, dtype: str, gain: float, noise: float) -> None:
        self.dtype, self.gain, self.noise = dtype, gain, noise
        self.rng = np.random.default_rng(9)

    def vocals(self, batch: np.ndarray) -> np.ndarray:
        return (self.gain * batch + self.noise * self.rng.standard_normal(batch.shape)).astype(np.float32)


def test_the_dtype_check_reports_each_dtypes_own_separator(tmp_path: Path):
    speech, music, on = _stems()
    np.savez(tmp_path / "stems.npz", speech=speech, background=music, turns=TURNS)
    main = _Sep("float32", 0.9, 0.0)  # SEP_DTYPE switched to float32: the pair still compares both dtypes
    bf16 = _Sep("bfloat16", 0.9, 0.03)
    out = vm.separation(tmp_path, main, {"bfloat16": bf16, "float32": main})
    check = out["dtype_check"]
    assert out["dtype"] == "float32"
    assert check["float32"]["si_sdr_db"] == out["si_sdr_db"] > check["bfloat16"]["si_sdr_db"]
    assert check["bf16_loss_db"] > vm.DTYPE_MARGIN and check["sep_dtype"] == "float32"
