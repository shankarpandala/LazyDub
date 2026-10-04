"""The mix (OFFLINE-RENDER §2.15): lines added at their start, a click on its planned 44.1 kHz sample after the
resampler, overlapping lines of different speakers summed, speakers equalised (and clamped), the master's loudness and
true peak after AAC, a limiter with no delay, BS.1770 on the EBU 1 kHz case, the bed as an optional input, and the bed
itself (§2.15): the ducking envelope exactly (both depths, both ramps), no pumping across short tails and gaps, the
voice/bed balance over the current speech turns, and the bed read on the output's clock. Synthetic signals only."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from maata_engine import export, mix, separate  # noqa: E402
from maata_engine.mix import MIX_SR, Line, Loudness, Mix  # noqa: E402

SR = 24_000


def pcm(tmp_path, name: str, audio: np.ndarray):
    path = tmp_path / f"{name}.npy"
    np.save(path, np.asarray(audio, np.float16))
    return path


def line(tmp_path, name: str, start: float, audio: np.ndarray, speaker: str = "S1") -> Line:
    return Line(start, speaker, pcm(tmp_path, name, audio), len(audio))


def tone(seconds: float, hz: float, amp: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def whole(m: Mix) -> np.ndarray:
    return np.concatenate(list(m.blocks()), axis=1)


def test_a_click_lands_on_its_planned_sample_after_the_resampler(tmp_path):
    assert mix.resampler_delay(SR) == 0  # measured (PyAV 18.1.0); compensated whatever it is
    click = np.zeros(SR // 2, np.float32)
    click[1200] = 0.9  # 0.05 s into the line
    out = whole(Mix([line(tmp_path, "c", 13.0, click)], 30.0, SR))
    assert out.shape == (2, 30 * MIX_SR) and np.array_equal(out[0], out[1])  # both channels, the whole span
    assert int(np.argmax(np.abs(out[0]))) == round(13.05 * MIX_SR)
    # blocks of 10 s, and a line that starts before 0 or runs past the end is cut, not shifted
    assert [b.shape[1] for b in Mix([], 25.0, SR).blocks()] == [441_000, 441_000, 220_500]
    early = whole(Mix([line(tmp_path, "e", -0.05, click)], 1.0, SR))
    assert int(np.argmax(np.abs(early[0]))) == 0


def test_two_overlapping_lines_of_different_speakers_are_summed(tmp_path):
    a = line(tmp_path, "a", 2.0, tone(3.0, 300.0, 0.2), "S1")
    b = line(tmp_path, "b", 3.0, tone(3.0, 500.0, 0.2), "S2")
    both, only_a, only_b = (whole(Mix(x, 8.0, SR)) for x in ([a, b], [a], [b]))
    np.testing.assert_allclose(both, only_a + only_b, atol=1e-5)  # added, never assigned
    overlap = both[0, 3 * MIX_SR + 1000:5 * MIX_SR - 1000]
    spectrum = np.abs(np.fft.rfft(overlap))
    freqs = np.fft.rfftfreq(len(overlap), 1 / MIX_SR)
    assert {int(round(f)) for f in freqs[np.argsort(spectrum)[-2:]]} == {300, 500}  # both voices are heard there


def test_speakers_are_equalised_and_the_gain_is_clamped(tmp_path):
    loud = [line(tmp_path, f"l{k}", 4.0 * k, tone(3.0, 220.0, 0.2), "S1") for k in range(4)]
    quiet = [line(tmp_path, f"q{k}", 20.0 + 4.0 * k, tone(3.0, 330.0, 0.1), "S2") for k in range(4)]  # 6 dB under
    gains = mix.speaker_gains(loud + quiet, SR)
    assert gains["S2"] - gains["S1"] == pytest.approx(6.0, abs=0.2)
    levels = []
    for sid, got in (("S1", loud), ("S2", quiet)):
        meter = Loudness(SR, 1)
        for x in got:
            meter.add(np.load(x.pcm).astype(np.float32) * 10 ** (gains[sid] / 20))
        levels.append(meter.result()["I"])
    assert abs(levels[0] - levels[1]) <= 0.5 and levels[0] == pytest.approx(mix.VOICE_REF, abs=0.5)
    faint = [line(tmp_path, f"f{k}", 4.0 * k, tone(3.0, 220.0, 0.01), "S3") for k in range(4)]  # about -40 LUFS
    blaring = [line(tmp_path, f"b{k}", 4.0 * k, tone(3.0, 220.0, 0.9), "S4") for k in range(4)]  # about -7 LUFS
    assert mix.speaker_gains(faint + blaring, SR) == {"S3": mix.VOICE_CLAMP, "S4": -mix.VOICE_CLAMP}
    assert mix.speaker_gains([line(tmp_path, "z", 0.0, np.zeros(SR), "S5")], SR) == {"S5": 0.0}  # silence: none


def test_a_hot_mix_is_set_to_the_target_loudness_with_its_true_peak_under_the_ceiling_after_aac(tmp_path):
    rng = np.random.default_rng(7)
    lines = []
    for k in range(10):  # loud speech-like bursts with near full-scale peaks
        n = 2 * SR
        env = np.abs(np.sin(np.pi * np.arange(n) / n * 6)) ** 0.5
        burst = (np.clip(rng.standard_normal(n) * 0.35, -0.99, 0.99) * env).astype(np.float32)
        burst[n // 2] = 0.99
        lines.append(line(tmp_path, f"h{k}", 0.5 + 2.5 * k, burst, "S1" if k % 2 else "S2"))
    m = Mix(lines, 26.0, SR, mix.speaker_gains(lines, SR))
    meter = Loudness(MIX_SR, 2)
    for b in m.blocks():
        meter.add(b)
    gain = mix.mix_gain(meter.result()["I"])
    path = tmp_path / "hot.m4a"
    with av.open(str(path), "w", format="mp4") as out:
        a = out.add_stream(export.aac(), rate=MIX_SR)
        a.layout, a.bit_rate = "stereo", export.AUDIO_BITRATE
        n = 0
        for b in mix.master(m.blocks(), gain):
            assert np.abs(b).max() <= 10 ** (mix.LIMIT_DBFS / 20) + 1e-4  # the limiter's ceiling, sample peak
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(b), format="fltp", layout="stereo")
            f.sample_rate, f.pts, f.time_base = MIX_SR, n, Fraction(1, MIX_SR)
            n += b.shape[1]
            for p in a.encode(f):
                out.mux(p)
        for p in a.encode(None):
            out.mux(p)
    got = export._measure(path)
    assert got["I"] == pytest.approx(mix.MIX_LUFS, abs=0.5) and got["TP"] <= mix.TP_CEIL


def test_the_limiter_adds_no_delay(tmp_path):
    x = np.zeros((2, 3 * MIX_SR), np.float32)
    x[:, 10_000] = 0.5     # under the ceiling: kept, on its sample
    x[:, 50_000] = 1.5     # over it: limited, on its sample
    blocks = [x[:, :MIX_SR], x[:, MIX_SR:2 * MIX_SR], x[:, 2 * MIX_SR:]]
    y = np.concatenate(list(mix.master(blocks, 0.0)), axis=1)
    assert y.shape == x.shape
    assert int(np.argmax(np.abs(y[0, :20_000]))) == 10_000 and y[0, 10_000] == pytest.approx(0.5, abs=1e-3)
    assert int(np.argmax(np.abs(y[0, 40_000:]))) + 40_000 == 50_000
    assert np.abs(y).max() <= 10 ** (mix.LIMIT_DBFS / 20) + 1e-4


def test_bs1770_reads_the_ebu_reference_case():
    """EBU Tech 3341's 1 kHz stereo sine at -23 dBFS reads -23.0 LUFS; mono reads as the dual mono it becomes."""
    sr = 48_000
    s = tone(20.0, 1000.0, 10 ** (-23 / 20), sr)
    meter = Loudness(sr, 2)
    meter.add(np.stack([s, s]))
    assert meter.result()["I"] == pytest.approx(-23.0, abs=0.05)
    mono = Loudness(sr, 1)
    mono.add(s)
    assert mono.result()["I"] == pytest.approx(-23.0, abs=0.05)


def test_the_bed_is_an_optional_input(tmp_path):
    calls = []

    def bed(a: int, b: int) -> np.ndarray:
        calls.append((a, b))
        return np.full((2, b - a), 0.01, np.float32)

    click = np.zeros(SR // 10, np.float32)
    click[0] = 0.5
    voices = whole(Mix([line(tmp_path, "c", 1.0, click)], 12.0, SR))
    with_bed = whole(Mix([line(tmp_path, "c", 1.0, click)], 12.0, SR, bed=bed))
    np.testing.assert_allclose(with_bed, voices + 0.01, atol=1e-6)
    assert calls == [(0, 441_000), (441_000, 529_200)]


# ---- the bed (§2.15) -------------------------------------------------------------------------------------------------
def expected_db(t: np.ndarray, spans: list[tuple[float, float, float]]) -> np.ndarray:
    """The envelope written out: for each (start, end, depth dB) layer, the depth inside, a 0.15 s raised cosine into it
    before its start and a 0.4 s one out of it after its end; the layers added."""
    db = np.zeros_like(t)
    for a, b, depth in spans:
        r = np.where((t >= a) & (t < b), 1.0, 0.0)
        up = (t >= a - 0.15) & (t < a)
        r[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - (a - 0.15)) / 0.15)
        down = (t >= b) & (t < b + 0.4)
        r[down] = 0.5 + 0.5 * np.cos(np.pi * (t[down] - b) / 0.4)
        db += depth * r
    return db


def test_the_ducking_envelope_is_exact_at_both_depths_with_its_ramps():
    lines = [(10.0, 14.0), (14.3, 18.0), (40.0, 42.0)]  # the first two 0.3 s apart: bridged
    speech = [(9.8, 18.5), (25.0, 30.0), (40.0, 50.0)]  # 25-30 has no Telugu line (skipped); 43-50 none within 1 s
    duck, bare = mix.duck_spans(lines, speech)
    assert duck == [(9.8, 18.5), (25.0, 30.0), (40.0, 50.0)] and bare == [(25.0, 30.0), (43.0, 50.0)]
    a, n = int(5 * MIX_SR), int(50 * MIX_SR)
    got = 20 * np.log10(mix.envelope(duck, bare, a, n).astype(np.float64))
    t = (a + np.arange(n)) / MIX_SR
    want = expected_db(t, [(9.8, 18.5, -6.0), (25.0, 30.0, -6.0), (40.0, 50.0, -6.0),
                           (25.0, 30.0, -6.0), (43.0, 50.0, -6.0)])  # bare: another -6 on top, to -12
    assert np.abs(got - want).max() < 1e-4
    at = lambda s: got[int(round((s - 5) * MIX_SR))]  # noqa: E731
    assert at(8.0) == 0.0 and at(12.0) == pytest.approx(-6.0, abs=1e-5) and at(27.0) == pytest.approx(-12.0, abs=1e-5)
    assert at(9.725) == pytest.approx(-3.0, abs=0.01)    # half way through the attack
    assert at(18.7) == pytest.approx(-3.0, abs=0.01)     # half way through the release
    assert at(24.925) == pytest.approx(-6.0, abs=0.01)   # a bare span from silence: one 0.15 s ramp to -12
    assert at(42.925) == pytest.approx(-9.0, abs=0.01)   # inside the duck: from -6 to -12 over the same 0.15 s
    assert at(50.2) == pytest.approx(-6.0, abs=0.01) and at(50.45) == 0.0  # out of -12 over 0.4 s
    # sample by sample, block by block: the same envelope wherever the blocks are cut
    parts = np.concatenate([mix.envelope(duck, bare, b, min(441_000, a + n - b)) for b in range(a, a + n, 441_000)])
    assert np.array_equal(parts, mix.envelope(duck, bare, a, n))


def test_short_tails_and_short_gaps_never_dip_the_music():
    # a Telugu line shorter than its English: 1.2 s of English left past the 1 s margin, under BARE_MIN: stays -6 dB
    duck, bare = mix.duck_spans([(10.0, 14.0)], [(10.0, 16.2)])
    assert bare == [] and duck == [(10.0, 16.2)]
    db = 20 * np.log10(mix.envelope(duck, bare, 10 * MIX_SR, int(6.2 * MIX_SR)).astype(np.float64))
    assert np.allclose(db, -6.0, atol=1e-5)
    # lines and words with a 0.3 s gap between them (22.0-22.3): bridged, so the music doesn't come up there
    duck, bare = mix.duck_spans([(20.0, 22.0), (22.4, 24.0)], [(20.0, 21.9), (22.3, 24.0)])
    assert duck == [(20.0, 24.0)] and bare == []
    db = 20 * np.log10(mix.envelope(duck, bare, 20 * MIX_SR, 4 * MIX_SR).astype(np.float64))
    assert np.allclose(db, -6.0, atol=1e-5)
    # English speech with no Telugu, 1.4 s: too short for -12
    assert mix.duck_spans([(0.0, 2.0)], [(5.0, 6.4)])[1] == []
    # with nothing to duck, nothing changes
    assert np.array_equal(mix.envelope([], [], 0, 1000), np.ones(1000, np.float32))


def test_the_balance_puts_the_voices_over_the_bed_as_the_english_sat_over_it(tmp_path):
    loud, quiet = tone(2.0, 300, 0.2), tone(2.0, 300, 0.1)
    lines = [line(tmp_path, "a", 1.0, loud, "S1"), line(tmp_path, "b", 4.0, quiet, "S2")]
    gains = {"S1": -3.0, "S2": 3.0}
    voice = (10 ** (-3 / 10) * np.mean(loud.astype(np.float16).astype(np.float64) ** 2)
             + 10 ** (3 / 10) * np.mean(quiet.astype(np.float16).astype(np.float64) ** 2)) / 2
    for vocals in (voice / 4, voice, voice * 2):  # the English 6 dB under, level with, 3 dB over the voices
        assert mix.bed_gain(lines, gains, vocals) == pytest.approx(10 * np.log10(voice / vocals), abs=1e-3)
    assert mix.bed_gain(lines, gains, voice * 1e4) == -mix.BED_CLAMP and mix.bed_gain(lines, gains, voice / 1e4) == \
        mix.BED_CLAMP
    assert mix.bed_gain(lines, gains, None) == 0.0 and mix.bed_gain(lines, gains, 0.0) == 0.0
    assert mix.bed_gain([], gains, voice) == 0.0
    # the original vocals' level: the envelope's frames over the CURRENT speech turns only (NaN: not separated)
    frames = np.full(100, 0.001, np.float32)  # 10 s at 10 a second
    frames[20:40] = 0.04   # 2-4 s: one speaker
    frames[60:70] = 0.01   # 6-7 s: the other
    frames[90:] = np.nan
    assert separate.mean_square(frames, [(2.0, 4.0), (6.0, 7.0)]) == pytest.approx((20 * 0.04 + 10 * 0.01) / 30)
    assert separate.mean_square(frames, [(2.0, 4.0)]) == pytest.approx(0.04)  # turns changed: the level follows
    assert separate.mean_square(frames, [(9.0, 12.0)]) is None and separate.mean_square(frames, []) is None


def test_the_bed_is_read_across_its_blocks_and_moved_onto_the_outputs_clock(tmp_path):
    rng = np.random.default_rng(7)
    whole_bed = (0.1 * rng.standard_normal((2, 2500))).astype(np.float16)
    for k in range(3):  # 1,000-sample blocks, the last shorter
        np.save(tmp_path / f"{k:05d}.npy", whole_bed[:, k * 1000:(k + 1) * 1000])
    bg = mix.Background(tmp_path, 1000, 3, None, (), ())
    ref = whole_bed.astype(np.float32)
    assert np.array_equal(bg.read(950, 2100), ref[:, 950:2100])
    got = bg.read(-50, 2600)  # silence before 0 and past the last block
    assert np.array_equal(got[:, 50:2550], ref) and not got[:, :50].any() and not got[:, 2550:].any()
    assert not mix.Background(tmp_path, 1000, 1, None, (), ()).read(1000, 2000).any()  # only the blocks made
    # on the output's clock: dub time t plays at t + off, times the balance and the envelope
    off = 10 / MIX_SR
    duck = ((0.0, 1.0),)
    bg = mix.Background(tmp_path, 1000, 3, None, duck, ())
    bed = bg.on(off, -6.0)
    out = bed(0, 2600)
    env = mix.envelope(list(duck), [], -10, 2600)
    np.testing.assert_allclose(out[:, 10:2510], ref * 10 ** (-6 / 20) * env[10:2510], rtol=1e-5, atol=1e-7)
    assert not out[:, :10].any()
