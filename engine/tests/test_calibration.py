"""Voice calibration and the duration estimator's voice keys in a render job (ARCHITECTURE §3.7 step 5, §4.5; OFFLINE-RENDER
§2.6): three Telugu-script sentences per new voice set its rate and overhead; every length is predicted from a line's
Telugu script, whichever script the TTS reads, and the lint compares lines in Telugu script too. The calibration
sentences and all test text are original."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fakes import bare_job, put  # noqa: E402

from maata_engine.backends.base import SR_ANALYSIS, LineResult, Wording  # noqa: E402
from maata_engine.backends.mock import MockTTS  # noqa: E402
from maata_engine.dubber import BAND, CALIBRATION_TE, MAX_LINE_SECONDS, UnitState, VoiceCost, VoiceState  # noqa: E402
from maata_engine.qa import validators  # noqa: E402
from maata_engine.render import _read_rows  # noqa: E402
from maata_engine.text import tenglish  # noqa: E402
from maata_engine.text.akshara import count_units, mixed_units  # noqa: E402
from maata_engine.timing.duration import (DEFAULT_RATE, MAX_OVERHEAD, MIN_RATE, DurationEstimator,  # noqa: E402
                                          VoiceKey)
from maata_engine.types import SourceUnit, VoiceKind  # noqa: E402


@dataclass
class Take:
    seconds: float


class PacedTTS(MockTTS):
    """A mel-take voice that says `rate` units per second after `overhead` s (Latin-script English in syllables), with
    the guidance and exaggeration settings of a Chatterbox TTS. `babble` makes takes of that text run to their cap."""

    cfg_weight, exaggeration = 0.5, 0.7

    def __init__(self, rate: float = 5.2, overhead: float = 0.3, babble: str | None = None) -> None:
        self.rate, self.overhead, self.babble = rate, overhead, babble
        self.asked: list[tuple[str, float | None]] = []

    def synthesize_mel(self, text, voice, language="te", max_seconds=None) -> Take:
        self.asked.append((text, max_seconds))
        seconds = 1e9 if text == self.babble else self.overhead + mixed_units(text) / self.rate
        return Take(min(seconds, max_seconds) if max_seconds else seconds)

    def vocode(self, take: Take, rate: float = 1.0) -> np.ndarray:
        # Valid, audible output at the controlled duration; silence is now a synthesis failure, not a timing fixture.
        t = np.arange(int(take.seconds / rate * self.sample_rate), dtype=np.float32) / self.sample_rate
        return 0.1 * np.sin(2 * np.pi * 140 * t)

    def pack_take(self, take: Take) -> dict[str, np.ndarray]:
        return {"mel": np.zeros(1, np.float16)}

    def unpack_take(self, d: dict[str, np.ndarray]) -> Take:
        return Take(float(d["seconds"]))


# ---- the sentences ------------------------------------------------------------------------------------------------
def test_the_calibration_sentences_follow_the_script_contract():
    for w, safe_indices in zip(CALIBRATION_TE, ((3, 5), (5, 8), (0, 5, 6, 12, 15))):
        raw = {"spoken": w.spoken, "english": tenglish.anchored_english(w.spoken, w.english)}
        safe = Wording(w.spoken, tuple((i, en) for i, en in w.english if i in safe_indices))
        assert validators.wording(raw) == (safe, [], [])  # exact anchors; inflected channel/mind stay in Telugu
        assert tenglish.latin_spoken(safe.spoken, safe.english) == tenglish.latin_spoken(w.spoken, w.english)
        assert w.english  # English loans, written in Telugu script
        assert not any(ch.isascii() and ch.isalpha() for ch in w.spoken)
        assert tenglish.lint(w.spoken, w.english, "") == []
    lengths = sorted(count_units(w.spoken) for w in CALIBRATION_TE)
    assert len(lengths) == 3 and lengths[0] >= 12 and lengths[1] >= 1.5 * lengths[0] and lengths[2] >= 1.5 * lengths[1]
    assert max(DurationEstimator().estimate(w.spoken, "prior") for w in CALIBRATION_TE) < 10.0  # a short job


# ---- calibrating a voice ------------------------------------------------------------------------------------------
async def test_a_new_voice_is_calibrated_from_all_three_sentences(tmp_path):
    tts = PacedTTS()
    job = bare_job(tmp_path, tts)
    key = VoiceKey("S1", 0.5, 0.7, "r")
    assert await job._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    assert [text for text, _ in tts.asked] == [w.spoken for w in CALIBRATION_TE]
    assert job.estimator.rate(key) == pytest.approx(5.2) and job.estimator.overhead(key) == pytest.approx(0.3)
    # each take is capped at the slowest take the estimator can represent (a runaway stops there)
    caps = [min(MAX_LINE_SECONDS, MAX_OVERHEAD + count_units(w.spoken) / MIN_RATE) for w in CALIBRATION_TE]
    assert [cap for _, cap in tts.asked] == pytest.approx(caps)


@pytest.mark.parametrize("rate", [2.2, 3.0, 9.0])
async def test_a_slow_or_fast_voice_is_measured_from_all_three_takes(tmp_path, rate):
    """A clone at 2.2 aksharas/s runs about 2.7x what the prior predicts: still a voice, not a runaway."""
    tts = PacedTTS(rate=rate, overhead=0.3)
    job = bare_job(tmp_path, tts)
    key = VoiceKey("S1")
    assert await job._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    assert job.estimator.rate(key) == pytest.approx(rate) and job.estimator.overhead(key) == pytest.approx(0.3)


async def test_for_the_latin_ab_the_sentences_are_said_in_latin_and_counted_in_telugu_script(tmp_path):
    tts = PacedTTS()
    job = bare_job(tmp_path, tts, tts_script="latin")
    key = VoiceKey("S1")
    assert await job._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    said = [text for text, _ in tts.asked]
    assert said == [tenglish.latin_spoken(w.spoken, w.english) for w in CALIBRATION_TE] and "topic" in said[0]
    assert "ఛానెల్లో" in said[1] and "channel" not in said[1]
    assert "మైండ్కి" in said[2] and "mind" not in said[2]  # preserve the attached Telugu endings
    expected = DurationEstimator()
    expected.calibrate(key, [(w.spoken, 0.3 + mixed_units(text) / 5.2) for w, text in zip(CALIBRATION_TE, said)])
    assert job.estimator.rate(key) == pytest.approx(expected.rate(key))


async def test_a_calibration_take_that_runs_to_its_cap_is_left_out(tmp_path):
    tts = PacedTTS(babble=CALIBRATION_TE[2].spoken)
    job = bare_job(tmp_path, tts)
    key = VoiceKey("S1")
    assert await job._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 2
    assert job.estimator.rate(key) == pytest.approx(5.2) and job.estimator.overhead(key) == pytest.approx(0.3)


async def test_a_tts_without_mel_takes_is_not_calibrated(tmp_path):
    job = bare_job(tmp_path, MockTTS())
    assert await job._calibrate("S1", VoiceKey("S1"), {"f0": 140.0}, VoiceCost()) == 0
    assert not job.estimator.voices


# ---- voice keys ---------------------------------------------------------------------------------------------------
def cloned_from(job, monkeypatch, level: float) -> None:
    """The stitched reference of S1: one 6 s clip of audio at `level`."""
    job.audio = np.full(10 * SR_ANALYSIS, level, np.float32)
    monkeypatch.setattr(job.registry, "best_span", lambda *a, **k: None)
    monkeypatch.setattr(job.registry, "reference_clips", lambda *a, **k: [(0.0, 6.0)])


async def test_a_clone_is_keyed_by_its_settings_and_reference_and_calibrated_under_that_key(tmp_path, monkeypatch):
    tts = PacedTTS()
    job = bare_job(tmp_path, tts)
    cloned_from(job, monkeypatch, 0.01)
    entry = await job._make_voice("S1")
    first = job._key("S1")
    assert first == job.voices["S1"].key and (first.voice, first.cfg, first.exaggeration) == ("S1", 0.5, 0.7)
    assert first.reference == f"v1:{entry['hash']}" and job.estimator.rate(first) == pytest.approx(5.2)
    assert job.estimator.rate(job._voice_key("preset_m", None)) == pytest.approx(DEFAULT_RATE)  # the preset's own pace
    # rebuilt from other audio, the voice is a new one: calibrated afresh, the old pace kept under its own key
    tts.rate = 6.4
    cloned_from(job, monkeypatch, 0.02)
    await job._make_voice("S1")
    second = job._key("S1")
    assert second.reference != first.reference and job.estimator.rate(second) == pytest.approx(6.4)
    assert job.estimator.rate(first) == pytest.approx(5.2)
    # both calibrations' time adds up for maata-bench (`calibration_seconds`), as each clone's trace has it
    clones = [e for e in _read_rows(job._dir / "units.jsonl") if e["event"] == "clone"]
    assert [e["calibration_takes"] for e in clones] == [3, 3] and job.calibrate_s > 0
    assert job.calibrate_s == pytest.approx(sum(e["calibrate_s"] for e in clones), abs=0.01)


def test_a_voices_own_guidance_weight_wins_over_the_tts_default(tmp_path):
    job = bare_job(tmp_path, PacedTTS())
    assert job._voice_key("S1", SimpleNamespace(cfg_weight=0.3), "r") == VoiceKey("S1", 0.3, 0.7, "r")
    assert job._voice_key("S1", SimpleNamespace(cfg_weight=None), "r") == VoiceKey("S1", 0.5, 0.7, "r")
    assert bare_job(tmp_path / "2", MockTTS())._voice_key("S1", {"f0": 1.0}, "r") == VoiceKey("S1", None, None, "r")


async def test_a_speaker_on_a_preset_shares_its_pace_with_the_speakers_on_that_preset(tmp_path):
    job = bare_job(tmp_path, PacedTTS())
    assert job._key("S1") == job._key("S3") == job._voice_key("preset_m", None) != job._key("S2")
    _, kind, key = await job._voice_for("S1", VoiceCost())  # no voice built: never cloned on a line's path
    assert kind is VoiceKind.PRESET and key == job._key("S3") and "S1" not in job.estimator.voices
    job.voices["S1"] = VoiceState({"f0": 1.0}, VoiceKind.CLONED, status="cloned", key=VoiceKey("S1", 0.5, 0.7, "r"))
    assert job._key("S1") == VoiceKey("S1", 0.5, 0.7, "r")
    _, kind, key = await job._voice_for("S1", VoiceCost())
    assert kind is VoiceKind.CLONED and key == job._key("S1")
    job.voices["S1"].use_preset = True  # the listener switched them to a preset: their lines are that voice's
    _, kind, key = await job._voice_for("S1", VoiceCost())
    assert kind is VoiceKind.PRESET and key == job._key("S1") == job._key("S3")


# ---- the Telugu script is what is measured and compared ------------------------------------------------------------
def line_with_english(job) -> UnitState:
    u = SourceUnit(0, "S1", 100.0, 102.0, "Restart the router once, then check the settings.")
    st = UnitState(u, 104.0, speech_s=2.0)
    st.line = LineResult(0, {
        "full": Wording("ఒకసారి రౌటర్ని రీస్టార్ట్ చేసి, తర్వాత సెట్టింగ్స్ చెక్ చేయండి.", ((1, "router"), (2, "restart"),
                                                                                  (5, "settings"), (6, "check"))),
        "concise": Wording("రౌటర్ రీస్టార్ట్ చేసి సెట్టింగ్స్ చూడండి.", ((0, "router"), (1, "restart"), (3, "settings")))})
    put(job, st)
    return st


@pytest.mark.parametrize("script", ["telugu", "latin"])
def test_the_band_rule_predicts_from_the_telugu_script_whatever_the_tts_reads(tmp_path, script):
    job = bare_job(tmp_path, PacedTTS(), tts_script=script)
    st = line_with_english(job)
    st.speech_s = 2.25
    st.unit.end = st.unit.start + st.speech_s
    job.estimator.calibrate(job._key("S1"), [("ప" * n, 0.2 + n / 7.0) for n in (15, 27, 45)])
    full, concise = st.line.full, st.line.tiers["concise"]
    assert 0.2 + count_units(full.spoken) / 7.0 > BAND[1] * st.speech_s  # full runs long; concise fits
    # The Latin rebuild still retains the router's case ending in Telugu, but would undercount the other loans.
    latin = tenglish.latin_spoken(full.spoken, full.english)
    assert latin == "ఒకసారి రౌటర్ని restart చేసి, తర్వాత settings check చేయండి." and count_units(latin) == 15.0
    assert BAND[0] * st.speech_s <= 0.2 + count_units(latin) / 7.0 <= BAND[1] * st.speech_s
    assert job._choose(st) and st.tier == "concise" and job._target(st) == pytest.approx((st.speech_s - 0.2) * 7.0)
    assert st.telugu == (tenglish.latin_spoken(concise.spoken, concise.english) if script == "latin" else concise.spoken)


def test_a_full_written_to_its_target_is_predicted_in_the_band_whatever_the_voices_overhead(tmp_path):
    """The target leaves the voice's calibrated overhead out of the speech time, as the band rule's prediction has it in."""
    job = bare_job(tmp_path, PacedTTS())
    st = line_with_english(job)
    st.speech_s = 2.4
    job.estimator.calibrate(job._key("S1"), [("ప" * n, 0.4 + n / 6.0) for n in (15, 27, 45)])  # 6 aksharas/s after 0.4 s
    assert job._target(st) == pytest.approx(12.0)
    st.line = LineResult(0, {"full": Wording("ప" * 12)})
    assert job._choose(st) and job.estimator.estimate(st.line.full.spoken, job._key("S1")) == pytest.approx(2.4)
    # (speech time x rate would ask for 14.4 aksharas, and those are predicted at 2.8 s: above the band)
    assert 0.4 + 14.4 / 6.0 > BAND[1] * 2.4
    st.speech_s = 0.3  # shorter than the overhead: nothing fits, and the target says so
    assert job._target(st) == 0.0


def test_k_counts_full_as_every_length_is_counted(tmp_path):
    job = bare_job(tmp_path, PacedTTS())
    st = line_with_english(job)
    job._learn_k(st)
    assert mixed_units(st.unit.text) == 11  # English syllables
    assert job._ratios["S1"] == [pytest.approx(count_units(st.line.full.spoken) / 11)]
    assert count_units(st.line.full.spoken) == 21.5  # the English words' Telugu-script aksharas included


async def test_a_line_is_voiced_in_the_wording_that_fits_its_voices_measured_pace(tmp_path, monkeypatch):
    """The wording reviewed was chosen at the prior's pace; the speaker's voice, measured when it was built, is slower:
    the dub loop voices the wording that fits that pace instead, in one take."""
    tts = PacedTTS(rate=3.0, overhead=0.2)  # a slow speaker
    job = bare_job(tmp_path, tts)
    u = SourceUnit(0, "S1", 100.0, 103.0, "Some words here.")
    st = UnitState(u, 104.0, speech_s=3.0, scene=1)
    st.line = LineResult(0, {"full": Wording("ప" * 17), "concise": Wording("ప" * 8)})
    put(job, st)
    assert job._choose(st) and st.tier == "full"  # at the prior's pace, full fits its 3 s
    cloned_from(job, monkeypatch, 0.01)
    await job._make_voice("S1")
    assert job.voices["S1"].kind is VoiceKind.CLONED and job.estimator.rate(job._key("S1")) == pytest.approx(3.0)
    await job._voice_line(st)
    assert st.tier == "concise"  # full would run 5.9 s at this voice's pace
    assert [text for text, _ in tts.asked[len(CALIBRATION_TE):]] == ["ప" * 8]  # one take: no re-synthesis needed
    (row,) = _read_rows(job.render_dir / "takes.jsonl")
    assert row["wording"] == "concise" and row["voice"][0] == "S1"


def test_the_lint_sees_an_echo_of_the_line_before_in_telugu_script(tmp_path):
    job = bare_job(tmp_path, PacedTTS())
    before = line_with_english(job)
    before.tier = "full"
    u = SourceUnit(1, "S2", 104.0, 107.0, "Now open the app again.")
    st = UnitState(u, None, speech_s=3.0, tier="full")
    st.line = LineResult(1, {"full": before.line.full})  # the same Telugu for a different English line
    put(job, st)
    w = st.line.full
    assert job._context_for(st) == [(before.unit.text, w.spoken)]
    assert tenglish.ECHO in tenglish.lint(w.spoken, w.english, u.text, job._context_for(st))
