"""Voice calibration and the duration estimator's voice keys in the session (ARCHITECTURE §3.7 step 5, §4.5): three
Telugu-script sentences per new voice set its rate and overhead; every length is predicted from a line's Telugu script,
whichever script the TTS reads, and the lint compares lines in Telugu script too. The calibration sentences and all test
text are original."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from maata_engine import session as sm
from maata_engine.backends.base import SR_ANALYSIS, Backend, LineResult, Wording
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.qa import validators
from maata_engine.resolve import DemoResolver
from maata_engine.session import Session, UnitState, VoiceCost, VoiceState
from maata_engine.text import tenglish
from maata_engine.text.akshara import count_units, mixed_units
from maata_engine.timing.duration import DEFAULT_RATE, MAX_OVERHEAD, MIN_RATE, DurationEstimator, VoiceKey
from maata_engine.types import SourceUnit, VoiceKind


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
        return np.zeros(int(take.seconds / rate * self.sample_rate), np.float32)


def session(tmp_path, tts, **kw) -> Session:
    async def sj(_: dict) -> None: ...

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts)
    return Session(backend, DemoResolver(), tmp_path, sj, sb, **kw)


# ---- the sentences ------------------------------------------------------------------------------------------------
def test_the_calibration_sentences_follow_the_script_contract():
    for w in sm.CALIBRATION_TE:
        raw = {"spoken": w.spoken, "english": [{"i": i, "en": en} for i, en in w.english]}
        assert validators.wording(raw) == (w, [], [])  # Telugu script only; every map entry is a Telugu word
        assert w.english  # English loans, written in Telugu script
        assert not any(ch.isascii() and ch.isalpha() for ch in w.spoken)
        assert tenglish.lint(w.spoken, w.english, "") == []
    lengths = sorted(count_units(w.spoken) for w in sm.CALIBRATION_TE)
    assert len(lengths) == 3 and lengths[0] >= 12 and lengths[1] >= 1.5 * lengths[0] and lengths[2] >= 1.5 * lengths[1]
    assert max(DurationEstimator().estimate(w.spoken, "prior") for w in sm.CALIBRATION_TE) < 10.0  # a short job


# ---- calibrating a voice ------------------------------------------------------------------------------------------
async def test_a_new_voice_is_calibrated_from_all_three_sentences(tmp_path):
    tts = PacedTTS()
    s = session(tmp_path, tts)
    key = VoiceKey("S1", 0.5, 0.7, "r")
    assert await s._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    assert [text for text, _ in tts.asked] == [w.spoken for w in sm.CALIBRATION_TE]
    assert s.estimator.rate(key) == pytest.approx(5.2) and s.estimator.overhead(key) == pytest.approx(0.3)
    # each take is capped at the slowest take the estimator can represent (a runaway stops there)
    caps = [min(sm.MAX_LINE_SECONDS, MAX_OVERHEAD + count_units(w.spoken) / MIN_RATE) for w in sm.CALIBRATION_TE]
    assert [cap for _, cap in tts.asked] == pytest.approx(caps)


@pytest.mark.parametrize("rate", [2.2, 3.0, 9.0])
async def test_a_slow_or_fast_voice_is_measured_from_all_three_takes(tmp_path, rate):
    """A clone at 2.2 aksharas/s runs about 2.7x what the prior predicts: still a voice, not a runaway."""
    tts = PacedTTS(rate=rate, overhead=0.3)
    s = session(tmp_path, tts)
    key = VoiceKey("S1")
    assert await s._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    assert s.estimator.rate(key) == pytest.approx(rate) and s.estimator.overhead(key) == pytest.approx(0.3)


async def test_for_the_latin_ab_the_sentences_are_said_in_latin_and_counted_in_telugu_script(tmp_path):
    tts = PacedTTS()
    s = session(tmp_path, tts, tts_script="latin")
    key = VoiceKey("S1")
    assert await s._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 3
    said = [text for text, _ in tts.asked]
    assert said == [tenglish.latin_spoken(w.spoken, w.english) for w in sm.CALIBRATION_TE] and "topic" in said[0]
    expected = DurationEstimator()
    expected.calibrate(key, [(w.spoken, 0.3 + mixed_units(text) / 5.2) for w, text in zip(sm.CALIBRATION_TE, said)])
    assert s.estimator.rate(key) == pytest.approx(expected.rate(key))


async def test_a_calibration_take_that_runs_to_its_cap_is_left_out(tmp_path):
    tts = PacedTTS(babble=sm.CALIBRATION_TE[2].spoken)
    s = session(tmp_path, tts)
    key = VoiceKey("S1")
    assert await s._calibrate("S1", key, {"f0": 140.0}, VoiceCost()) == 2
    assert s.estimator.rate(key) == pytest.approx(5.2) and s.estimator.overhead(key) == pytest.approx(0.3)


async def test_a_tts_without_mel_takes_is_not_calibrated(tmp_path):
    s = session(tmp_path, MockTTS())
    assert await s._calibrate("S1", VoiceKey("S1"), {"f0": 140.0}, VoiceCost()) == 0
    assert not s.estimator.voices


# ---- voice keys ---------------------------------------------------------------------------------------------------
def cloned_from(s: Session, monkeypatch, level: float) -> None:
    """The stitched reference of S1: one 6 s clip of audio at `level`."""
    s.audio = np.full(10 * SR_ANALYSIS, level, np.float32)
    monkeypatch.setattr(s.registry, "best_span", lambda *a, **k: None)
    monkeypatch.setattr(s.registry, "reference_clips", lambda *a, **k: [(0.0, 6.0)])


async def test_a_clone_is_keyed_by_its_settings_and_reference_and_calibrated_under_that_key(tmp_path, monkeypatch):
    tts = PacedTTS()
    s = session(tmp_path, tts)
    s._dir = tmp_path
    cloned_from(s, monkeypatch, 0.01)
    await s._clone("S1")
    first = s._key("S1")
    assert first == s.voices["S1"].key and (first.voice, first.cfg, first.exaggeration) == ("S1", 0.5, 0.7)
    assert first.reference and s.estimator.rate(first) == pytest.approx(5.2)
    assert s.estimator.rate(s._voice_key("preset_m", None)) == pytest.approx(DEFAULT_RATE)  # the preset's own pace
    # rebuilt from other audio, the voice is a new one: calibrated afresh, the old pace kept under its own key
    tts.rate = 6.4
    cloned_from(s, monkeypatch, 0.02)
    await s._clone("S1")
    second = s._key("S1")
    assert second.reference != first.reference and s.estimator.rate(second) == pytest.approx(6.4)
    assert s.estimator.rate(first) == pytest.approx(5.2)
    # both calibrations' time adds up for maata-bench (`calibration_seconds`), as each clone's trace has it
    clones = [json.loads(line) for line in (tmp_path / "units.jsonl").read_text().splitlines()]
    assert [e["calibration_takes"] for e in clones] == [3, 3] and s.calibrate_s > 0
    assert s.calibrate_s == pytest.approx(sum(e["calibrate_s"] for e in clones), abs=0.01)


def test_a_voices_own_guidance_weight_wins_over_the_tts_default(tmp_path):
    s = session(tmp_path, PacedTTS())
    assert s._voice_key("S1", SimpleNamespace(cfg_weight=0.3), "r") == VoiceKey("S1", 0.3, 0.7, "r")
    assert s._voice_key("S1", SimpleNamespace(cfg_weight=None), "r") == VoiceKey("S1", 0.5, 0.7, "r")
    assert session(tmp_path, MockTTS())._voice_key("S1", {"f0": 1.0}, "r") == VoiceKey("S1", None, None, "r")


async def test_a_speaker_on_a_preset_shares_its_pace_with_the_speakers_on_that_preset(tmp_path):
    s = session(tmp_path, PacedTTS())
    assert s._key("S1") == s._key("S3") == s._voice_key("preset_m", None) != s._key("S2")
    s.voices["S1"] = VoiceState({"f0": 1.0}, VoiceKind.CLONED, status="cloned", key=VoiceKey("S1", 0.5, 0.7, "r"))
    assert s._key("S1") == VoiceKey("S1", 0.5, 0.7, "r")
    _, kind, key = await s._voice_for("S1", VoiceCost())
    assert kind is VoiceKind.CLONED and key == s._key("S1")
    s.voices["S1"].use_preset = True  # the listener switched them to a preset: their lines are that voice's
    _, kind, key = await s._voice_for("S1", VoiceCost())
    assert kind is VoiceKind.PRESET and key == s._key("S1") == s._key("S3")


# ---- the Telugu script is what is measured and compared ------------------------------------------------------------
def line_with_english(s: Session) -> UnitState:
    u = SourceUnit(0, "S1", 100.0, 102.0, "Restart the router once, then check the settings.")
    st = UnitState(u, 104.0, sm.Chunk(100.0, 110.0), speech_s=2.0)
    st.line = LineResult(0, {
        "full": Wording("ఒకసారి రౌటర్ని రీస్టార్ట్ చేసి, తర్వాత సెట్టింగ్స్ చెక్ చేయండి.", ((1, "router"), (2, "restart"),
                                                                                  (5, "settings"), (6, "check"))),
        "concise": Wording("రౌటర్ రీస్టార్ట్ చేసి సెట్టింగ్స్ చూడండి.", ((0, "router"), (1, "restart"), (3, "settings")))})
    s.units[0] = st
    return st


@pytest.mark.parametrize("script", ["telugu", "latin"])
def test_the_band_rule_predicts_from_the_telugu_script_whatever_the_tts_reads(tmp_path, script):
    s = session(tmp_path, PacedTTS(), tts_script=script)
    st = line_with_english(s)
    s.estimator.calibrate(s._key("S1"), [("ప" * n, 0.2 + n / 7.0) for n in (15, 27, 45)])
    full, concise = st.line.full, st.line.tiers["concise"]
    assert 0.2 + count_units(full.spoken) / 7.0 > sm.BAND[1] * 2.0  # full runs long at this pace; concise fits
    # (the Latin rebuild of full, counted as a dub line, would look short enough: Latin letters count nothing)
    assert sm.BAND[0] * 2.0 <= 0.2 + count_units(tenglish.latin_spoken(full.spoken, full.english)) / 7.0 <= sm.BAND[1] * 2.0
    assert s._choose(st) and st.tier == "concise" and s._target(st) == pytest.approx((2.0 - 0.2) * 7.0)
    assert st.telugu == (tenglish.latin_spoken(concise.spoken, concise.english) if script == "latin" else concise.spoken)


def test_a_full_written_to_its_target_is_predicted_in_the_band_whatever_the_voices_overhead(tmp_path):
    """The target leaves the voice's calibrated overhead out of the speech time, as the band rule's prediction has it in."""
    s = session(tmp_path, PacedTTS())
    st = line_with_english(s)
    st.speech_s = 2.4
    s.estimator.calibrate(s._key("S1"), [("ప" * n, 0.4 + n / 6.0) for n in (15, 27, 45)])  # 6 aksharas/s after 0.4 s
    assert s._target(st) == pytest.approx(12.0)
    st.line = LineResult(0, {"full": Wording("ప" * 12)})
    assert s._choose(st) and s.estimator.estimate(st.line.full.spoken, s._key("S1")) == pytest.approx(2.4)
    # (speech time x rate would ask for 14.4 aksharas, and those are predicted at 2.8 s: above the band)
    assert 0.4 + 14.4 / 6.0 > sm.BAND[1] * 2.4
    st.speech_s = 0.3  # shorter than the overhead: nothing fits, and the target says so
    assert s._target(st) == 0.0


def test_k_counts_full_as_every_length_is_counted(tmp_path):
    s = session(tmp_path, PacedTTS())
    st = line_with_english(s)
    s._learn_k(st)
    assert mixed_units(st.unit.text) == 11  # English syllables
    assert s._ratios["S1"] == [pytest.approx(count_units(st.line.full.spoken) / 11)]
    assert count_units(st.line.full.spoken) == 21.5  # the English words' Telugu-script aksharas included


async def test_a_line_whose_voice_is_cloned_on_its_way_is_chosen_at_that_voices_measured_pace(tmp_path, monkeypatch):
    """A speaker first heard after the pre-pass is cloned (and calibrated) when their first line is voiced: the line's
    wording is chosen after that, at the new voice's pace, not at the prior's."""
    tts = PacedTTS(rate=3.0, overhead=0.2)  # a slow speaker
    s = session(tmp_path, tts)
    cloned_from(s, monkeypatch, 0.01)
    u = SourceUnit(0, "S1", 100.0, 103.0, "Some words here.")
    st = UnitState(u, 104.0, sm.Chunk(100.0, 110.0), speech_s=3.0, scene=1)
    st.line = LineResult(0, {"full": Wording("ప" * 17), "concise": Wording("ప" * 8)})
    s.units[0] = st
    assert s._choose(st) and st.tier == "full"  # at the prior's pace, full fits its 3 s
    await s._dub(st)
    assert s.voices["S1"].kind is VoiceKind.CLONED and s.estimator.rate(s._key("S1")) == pytest.approx(3.0)
    assert st.tier == "concise"  # chosen again once the voice was measured: full would run 5.9 s
    assert [text for text, _ in tts.asked[len(sm.CALIBRATION_TE):]] == ["ప" * 8]  # one take: no re-synthesis needed


async def test_a_line_voiced_while_its_speakers_clone_is_calibrated_keeps_the_voice_it_had(tmp_path, monkeypatch):
    """The diarizer clones a speaker first heard after the pre-pass, and the voicer takes the GPU lock between two of the
    calibration takes. The new voice is used only once it is calibrated: the line voiced in between keeps the preset
    (at the preset's pace, teaching the preset's pace), and the clone's calibration is whole."""
    entered, gate = threading.Event(), threading.Event()

    class GatedTTS(PacedTTS):
        def synthesize_mel(self, text, voice, language="te", max_seconds=None) -> Take:
            if text == sm.CALIBRATION_TE[0].spoken:
                entered.set()
                gate.wait(5.0)
            return super().synthesize_mel(text, voice, language, max_seconds)

    tts = GatedTTS(rate=3.0, overhead=0.2)
    s = session(tmp_path, tts)
    cloned_from(s, monkeypatch, 0.01)
    chunk = sm.Chunk(100.0, 110.0)

    def line(i: int, start: float) -> UnitState:
        st = UnitState(SourceUnit(i, "S1", start, start + 3.0, "Some words here."), start + 4.0, chunk, speech_s=3.0, scene=1)
        st.line = LineResult(i, {"full": Wording("ప" * 17), "concise": Wording("ప" * 8)})
        s.units[i] = st
        return st

    first, second = line(0, 100.0), line(1, 104.0)
    clone = asyncio.create_task(s._clone("S1"))
    while not entered.is_set():  # built, and inside its first calibration take (holding the GPU lock)
        await asyncio.sleep(0.001)
    assert s.voices["S1"].status == "cloning"
    dub = asyncio.create_task(s._dub(first))
    while first.tier is None:  # the voicer has picked its voice and wording, and waits for the lock
        await asyncio.sleep(0.001)
    gate.set()
    await asyncio.gather(clone, dub)
    said = [text for text, _ in tts.asked]
    assert said[:2] == [sm.CALIBRATION_TE[0].spoken, "ప" * 17]  # the line was voiced between the calibration takes
    preset, key = s._voice_key("preset_m", None), s._key("S1")
    line_takes = len(said) - len(sm.CALIBRATION_TE)  # full, then a shorter re-synthesis: the preset is as slow here
    assert line_takes == 2 and s.estimator.voices[preset].n == line_takes  # voiced with the preset, teaching its pace
    assert key.voice == "S1" and s.estimator.voices[key].n == len(sm.CALIBRATION_TE)
    assert s.estimator.rate(key) == pytest.approx(3.0)
    await s._dub(second)  # the next line is the clone's, chosen at its measured pace
    assert second.tier == "concise" and s.estimator.voices[key].n == len(sm.CALIBRATION_TE) + 1


def test_the_lint_sees_an_echo_of_the_line_before_in_telugu_script(tmp_path):
    s = session(tmp_path, PacedTTS())
    before = line_with_english(s)
    before.tier = "full"
    u = SourceUnit(1, "S2", 104.0, 107.0, "Now open the app again.")
    st = UnitState(u, None, sm.Chunk(100.0, 110.0), speech_s=3.0, tier="full")
    st.line = LineResult(1, {"full": before.line.full})  # the same Telugu for a different English line
    s.units[1] = st
    w = st.line.full
    assert s._context_for(st) == [(before.unit.text, w.spoken)]
    assert tenglish.ECHO in tenglish.lint(w.spoken, w.english, u.text, s._context_for(st))
