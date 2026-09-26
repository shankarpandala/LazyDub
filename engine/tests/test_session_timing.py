"""Timing v2 in the voicer (ARCHITECTURE §3.10): pieces at hard breaks, soft anchors, uneven compression, early starts
only into real silence, the akshara ceiling, the windowed lookahead and the timing fields in units.jsonl. No models, no
Claude: the mock voice and small stand-ins. All English and Telugu here is original test text."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from maata_engine.backends.base import Backend, LineResult, Wording
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.resolve import DemoResolver
from maata_engine.session import Chunk, Session, UnitState, _marks
from maata_engine.timing import pauses as pz
from maata_engine.timing.planner import V1, PlannerSettings, rate_ceiling
from maata_engine.types import SourceUnit, SpeakerTurn, TimedWord

SR = MockTTS.sample_rate
FULL = Wording("మేము నిన్న ఊరికి వెళ్ళాం ఇవాళ మళ్ళీ తిరిగి వచ్చేశాం")
PIECES = ("మేము నిన్న ఊరికి వెళ్ళాం", "ఇవాళ మళ్ళీ తిరిగి వచ్చేశాం")


class PausingTTS(MockTTS):
    """The mock voice with a real pause at each comma: `gap` s of silence between the stretches it separates."""

    def __init__(self, gap: float = 0.3) -> None:
        self.gap = gap

    def synthesize(self, text, voice, language="te", max_seconds=None):
        parts = [super(PausingTTS, self).synthesize(p, voice, language) for p in text.split(",")]
        out = [parts[0]]
        for p in parts[1:]:
            out += [np.zeros(round(self.gap * SR), np.float32), p]
        return np.concatenate(out)


class Take:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    @property
    def seconds(self) -> float:
        return len(self.samples) / SR


class MelPausingTTS(PausingTTS):
    """PausingTTS split into a take plus vocoding, recording the rate each take is vocoded at."""

    def __init__(self, gap: float = 0.3) -> None:
        super().__init__(gap)
        self.rates: list[float] = []

    def synthesize_mel(self, text, voice, language="te", max_seconds=None):
        return Take(self.synthesize(text, voice, language))

    def vocode(self, take, rate=1.0):
        self.rates.append(round(rate, 4))
        n = len(take.samples)
        return take.samples[np.minimum(np.round(np.arange(0, n, rate)).astype(int), n - 1)][: round(n / rate)]


def bare(tmp_path, tts=None, timing="v2") -> tuple[Session, list]:
    msgs: list = []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts or MockTTS())
    s = Session(backend, DemoResolver(), tmp_path, sj, sb, timing=timing)
    s._duration = lambda: 600.0
    s._dir = tmp_path / "vid"
    (s._dir / "pcm").mkdir(parents=True)
    s._prepass_done.set()
    return s, msgs


def words(*spec: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(w, a, b) for w, a, b in spec]


def line(s: Session, i: int, u: SourceUnit, nxt: float | None, full: Wording, pieces=(), tier="full") -> UnitState:
    st = UnitState(u, nxt, Chunk(u.start, u.end), speech_s=u.end - u.start)
    st.line, st.tier, st.telugu = LineResult(i, {"full": full}, pieces=pieces), tier, full.spoken
    s.units[i] = st
    return st


def units(s: Session) -> list[dict]:
    return [e for e in map(json.loads, (s._dir / "units.jsonl").read_text().splitlines()) if e["event"] == "unit"]


def diarized(monkeypatch, s: Session, ex: list[SpeakerTurn], overlapping: list[SpeakerTurn] | None = None) -> None:
    """Stand in the registry's turns: the exclusive track, and the overlapping one (the same when not given)."""
    def turns_in(a, b, exclusive=True):
        track = ex if exclusive or overlapping is None else overlapping
        return [SpeakerTurn(t.speaker, max(t.start, a), min(t.end, b)) for t in track if t.end > a and t.start < b]

    monkeypatch.setattr(s.registry, "turns_in", turns_in)


def stretch_with_a_break(i: int = 0, start: float = 10.0) -> SourceUnit:
    """A 7 s English sentence with a 1.4 s pause after its third word (a hard break)."""
    ws = words(("We", start, start + 0.4), ("went", start + 0.5, start + 1.2), ("home,", start + 1.3, start + 2.8),
               ("and", start + 4.2, start + 4.6), ("came", start + 4.7, start + 5.5), ("back.", start + 5.6, start + 7.0))
    return SourceUnit(i, "S1", start, start + 7.0, "We went home, and came back.", ws, breaks=[3])


async def test_a_line_with_a_hard_break_is_said_as_its_pieces_each_at_its_own_onset(tmp_path):
    s, msgs = bare(tmp_path)
    st = line(s, 0, stretch_with_a_break(), 25.0, FULL, PIECES)
    await s._dub(st)
    (u,) = units(s)
    assert u["said"] == "pieces" and u["parts"] == 2 and u["hard_breaks"] == 1
    first, second = st.plan.parts
    assert first.start == pytest.approx(10.0) and second.src == pytest.approx(14.2) and second.start >= 14.2 - 0.3
    assert u["anchor_errors"] and abs(u["anchor_errors"][0]) <= 0.3 + 1e-6
    # One buffer from the dub's start: the second piece's take where the plan puts it, silence in the pause.
    audio = np.load(s._dir / "pcm" / "0.npy")
    assert len(audio) == round(st.plan.wall * SR)
    gap = audio[round((first.wall + 0.05) * SR):round((second.start - st.plan.start - 0.05) * SR)]
    assert gap.size > 0.5 * SR and np.abs(gap).max() == 0.0
    assert np.abs(audio[round((second.start - st.plan.start) * SR):]).max() > 0.05
    sent = next(m for m in msgs if m["type"] == "unit")
    assert sent["telugu"] == FULL.spoken and sent["audioWall"] == pytest.approx(st.plan.wall)
    assert st.take_s == pytest.approx(sum(len(MockTTS().synthesize(p, {"f0": 140.0})) for p in PIECES) / SR, abs=0.01)


async def test_pieces_are_not_used_off_full_or_across_another_line_in_the_pause(tmp_path):
    s, _ = bare(tmp_path)
    st = line(s, 0, stretch_with_a_break(), 25.0, FULL, PIECES, tier="concise")
    st.line = LineResult(0, {"full": FULL, "concise": Wording(PIECES[0])}, pieces=PIECES)
    key = s._key("S1")
    assert s._pieces(st, key) == []                                        # the pieces belong to `full`
    st.tier = "full"
    got = s._pieces(st, key)
    assert [w.spoken for w in got] == list(PIECES)
    # Another speaker answers in the pause: the break is not one to meet, and the line drifts over it whole.
    other = UnitState(SourceUnit(1, "S2", 13.2, 13.9, "Right."), 25.0, Chunk(13.2, 13.9), speech_s=0.7)
    s.units[1] = other
    assert s._slot(st).breaks == () and s._pieces(st, key) == []


def test_a_piece_starts_early_only_into_the_silence_left_in_its_pause(tmp_path, monkeypatch):
    s, _ = bare(tmp_path)
    st = line(s, 0, stretch_with_a_break(), 17.2, FULL, PIECES)          # its pause: 12.8-14.2 s
    long_second = [(2.4, ()), (3.4, ())]                                  # so long that it has to start early

    def second() -> float:
        p = s.planner.evaluate(s._slot(st), 5.8, pieces=long_second)[1]
        assert p.said == "pieces"
        return p.parts[1].start

    assert second() < 14.2 - 0.05                                         # into the silent pause
    # Another speaker answers from before the pause and on into it: nobody starts inside it, so the break stays, but
    # the second piece doesn't start over them.
    s.units[1] = UnitState(SourceUnit(1, "S2", 12.3, 14.1, "Sure, go on."), 17.2, Chunk(12.3, 14.1), speech_s=1.8)
    slot = s._slot(st)
    assert slot.breaks == ((12.8, 14.2),) and slot.heard == ((12.8, 14.1),)
    assert second() >= 14.2 - 1e-9
    # A laugh in the pause with no line of its own counts the same (its diarized turn); the speaker's own doesn't.
    del s.units[1]
    diarized(monkeypatch, s, [SpeakerTurn("S1", 10.0, 12.8)], [SpeakerTurn("S1", 10.0, 13.0), SpeakerTurn("S2", 13.2, 13.9)])
    assert s._slot(st).heard == ((13.2, 13.9),)
    assert 13.9 + s.planner.s.guard - 1e-9 <= second() < 14.2


def test_the_speech_yardstick_is_the_speakers_own_turns_overlap_included(tmp_path, monkeypatch):
    s, _ = bare(tmp_path)
    u = SourceUnit(0, "S1", 20.0, 26.0, "And that was the whole trip.")
    # S2 cuts in at 25.2 while S1 talks on to 25.9. Exclusive turns give the overlap to one speaker (here S2), so S1's
    # would end at 25.2 and a dub ending with S1 would count as 0.7 s late.
    diarized(monkeypatch, s, [SpeakerTurn("S1", 20.0, 25.2), SpeakerTurn("S2", 25.2, 26.5)],
             [SpeakerTurn("S1", 20.0, 25.9), SpeakerTurn("S2", 25.2, 26.5)])
    assert s._speech_spans(u) == ((20.0, 25.9),)
    diarized(monkeypatch, s, [])
    assert s._speech_spans(u) == ((20.0, 26.0),)                            # nothing diarized: the span


def test_the_pieces_carry_their_share_of_the_english_map(tmp_path):
    s, _ = bare(tmp_path)
    full = Wording("మనం ఆఫీస్ కి వెళ్ళాలి తర్వాత మీటింగ్ ఉంది", ((1, "office"), (5, "meeting")))
    st = line(s, 0, stretch_with_a_break(), 25.0, full, ("మనం ఆఫీస్ కి వెళ్ళాలి", "తర్వాత మీటింగ్ ఉంది"))
    got = s._pieces(st, s._key("S1"))
    assert got[0].english == ((1, "office"),) and got[1].english == ((1, "meeting"),)


async def test_a_line_that_fits_waits_at_its_own_comma_for_the_english_to_resume(tmp_path):
    tts = PausingTTS(0.3)
    s, _ = bare(tmp_path, tts)
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = tts.synthesize(text, {"f0": 140.0})
    (a, b), = pz.internal(pz.pauses(natural, SR), len(natural) / SR)
    # The English breathes (0.35 s, a soft anchor) and resumes 0.25 s after the Telugu pause would end.
    resume = 20.0 + b + 0.25
    ws = words(("We", 20.0, 20.5), ("went,", 20.6, resume - 0.35), ("and", resume, resume + 0.4),
               ("came", resume + 0.5, resume + 1.2), ("back.", resume + 1.3, resume + 2.0))
    u = SourceUnit(0, "S1", 20.0, resume + 2.0, "We went, and came back.", ws, anchors=[2])
    st = line(s, 0, u, resume + 5.0, Wording(text))
    await s._dub(st)
    (ev,) = units(s)
    assert ev["said"] == "anchored" and ev["audio_rate"] == 1.0 and ev["anchor_errors"] == [0.0]
    audio = np.load(s._dir / "pcm" / "0.npy")
    (a2, b2), = pz.internal(pz.pauses(audio, SR), len(audio) / SR)
    assert (b2 - a2) - (b - a) == pytest.approx(0.25, abs=0.02)            # the silence set in the Telugu pause
    assert ev["voiced"][1][0] == pytest.approx(resume, abs=0.02)


@pytest.mark.parametrize("gap,short", [(0.3, 1.15), (0.6, 1.04)])
async def test_a_sped_up_take_gives_up_its_pause_before_its_speech_speeds_up(tmp_path, gap, short):
    tts = MelPausingTTS(gap)
    s, _ = bare(tmp_path, tts)
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = len(tts.synthesize(text, {"f0": 140.0})) / SR
    # A span shorter than the take, and the next line right after it: it has to go faster.
    u = SourceUnit(0, "S1", 30.0, 30.0 + natural / short, "We went, and came back.")
    st = line(s, 0, u, 30.0 + natural / short + 0.2, Wording(text))
    await s._dub(st)
    rate = st.plan.rate
    assert rate > 1.0 and st.plan.said == "whole" and tts.rates[0] == 1.0  # vocoded at 1.0 first: its pauses found
    audio = np.load(s._dir / "pcm" / "0.npy")
    assert len(audio) == round(natural / rate * SR)                        # exactly the planned length
    (a, b), = pz.internal(pz.pauses(audio, SR, min_len=0.05), len(audio) / SR)  # a kept pause is under 0.15 s
    if natural - natural / rate > gap - s.planner.s.keep_pause:
        # The pause gives what it can (down to about 0.12 s) and the speech the rest, slower than the plan's rate.
        assert len(tts.rates) == 2 and 1.0 < tts.rates[1] < rate
        assert b - a == pytest.approx(s.planner.s.keep_pause, abs=0.02)
    else:
        # The pause gives it all: the speech isn't sped up at all, and is vocoded once.
        assert tts.rates == [1.0]
        assert b - a == pytest.approx(gap - (natural - natural / rate), abs=0.02)


def test_an_early_start_goes_only_into_real_silence(tmp_path, monkeypatch):
    s, _ = bare(tmp_path)
    st = line(s, 0, SourceUnit(0, "S1", 50.0, 53.0, "So here we are."), 60.0, FULL)
    turns: list[SpeakerTurn] = []

    def turns_in(a, b, exclusive=True):
        return [SpeakerTurn(t.speaker, max(t.start, a), min(t.end, b)) for t in turns if t.end > a and t.start < b]

    monkeypatch.setattr(s.registry, "turns_in", turns_in)
    assert s._slot(st).prev_end is None                                     # nothing heard before it
    turns[:] = [SpeakerTurn("S2", 49.0, 49.9)]                              # someone laughs until 49.9 s: no line of theirs
    assert s._slot(st).prev_end == pytest.approx(49.9)
    turns[:] = [SpeakerTurn("S1", 49.8, 53.0)]                              # the speaker's own turn, heard before the words
    assert s._slot(st).prev_end is None
    turns[:] = [SpeakerTurn("S1", 49.0, 49.6), SpeakerTurn("S1", 49.8, 53.0)]  # ...and a breath of theirs before it
    assert s._slot(st).prev_end == pytest.approx(49.6)
    p = s.planner.evaluate(s._slot(st), 3.4)[1]
    assert p.lag >= -(50.0 - 49.6 - s.planner.s.guard) - 1e-9
    turns[:] = [SpeakerTurn("S2", 48.0, 50.5)]                              # crosstalk into the onset: no early start
    assert s._slot(st).prev_end == pytest.approx(50.0) and s.planner.evaluate(s._slot(st), 3.4)[1].lag >= 0.0


def test_each_line_goes_no_faster_than_its_voices_akshara_ceiling(tmp_path, monkeypatch):
    s, _ = bare(tmp_path)
    st = line(s, 0, SourceUnit(0, "S1", 50.0, 53.0, "So here we are."), 53.2, FULL)
    monkeypatch.setattr(s.estimator, "rate", lambda key: 9.0)
    assert s._slot(st).rate_cap == s.planner.s.speed_cap                    # the ceiling is off by default
    s.planner.s = replace(s.planner.s, akshara_ceiling=7.5)                 # as when a Telugu measurement sets it
    monkeypatch.setattr(s.estimator, "rate", lambda key: 6.1)
    assert s._slot(st).rate_cap == s.planner.s.speed_cap                    # the default pace: the speed cap binds
    monkeypatch.setattr(s.estimator, "rate", lambda key: 7.0)              # a fast talker: 7.5 / 7.0
    assert s._slot(st).rate_cap == pytest.approx(rate_ceiling(7.0, s.planner.s)) == pytest.approx(7.5 / 7.0)
    assert s.planner.evaluate(s._slot(st), 5.0)[1].rate == pytest.approx(7.5 / 7.0)


def test_the_lookahead_window_is_the_next_line_then_up_to_8_within_30_s(tmp_path):
    s, _ = bare(tmp_path)
    sts = [line(s, i, SourceUnit(i, "S1", 100.0 + 5 * i, 103.0 + 5 * i, "A line."), 105.0 + 5 * i, FULL)
           for i in range(12)]
    sts[3].line = None                                                      # not translated yet
    ahead = s._ahead(sts[0])
    assert [sl.id for sl, _ in ahead] == [1, 2, 3, 4, 5, 6]                 # line 7 starts 35 s after line 0
    assert ahead[2][1] is None and ahead[0][1] == pytest.approx(s.estimator.estimate(FULL.spoken, s._key("S1")))
    far = line(s, 20, SourceUnit(20, "S1", 400.0, 402.0, "Later."), None, FULL)
    assert [sl.id for sl, _ in s._ahead(sts[11])] == [20]                   # the next line, however far
    assert s._ahead(far) == ()


@pytest.mark.parametrize("mode", ["v1", "v2-whole"])
async def test_timing_v1_and_v2_whole_say_a_sentence_with_a_hard_break_whole(tmp_path, mode):
    s, _ = bare(tmp_path, timing=mode)
    st = line(s, 0, stretch_with_a_break(), 25.0, FULL, PIECES)
    assert s._pieces(st, s._key("S1")) == []
    await s._dub(st)
    (u,) = units(s)
    assert u["said"] == "whole" and u["parts"] == 0 and u["hard_breaks"] == 1 and u["end_error_s"] is not None


def test_timing_v1_is_the_timing_before_v2_and_v2_whole_only_drops_the_pieces(tmp_path, monkeypatch):
    s, _ = bare(tmp_path, timing="v1")
    assert s.timing == "v1" and s.planner.s == PlannerSettings(**V1)
    sts = [line(s, i, SourceUnit(i, "S1", 50.0 + 5 * i, 53.0 + 5 * i, "So here we are."), 55.0 + 5 * i, FULL)
           for i in range(4)]
    diarized(monkeypatch, s, [SpeakerTurn("S2", 49.0, 49.9)])
    monkeypatch.setattr(s.estimator, "rate", lambda key: 7.0)
    slot = s._slot(sts[0])
    assert slot.prev_end is None and slot.rate_cap == s.planner.s.speed_cap  # lines alone bound a lead; no akshara ceiling
    assert [sl.id for sl, _ in s._ahead(sts[0])] == [1]                     # a one-line lookahead
    w, _ = bare(tmp_path / "whole", timing="v2-whole")
    assert w.timing == "v2-whole" and w.planner.s == PlannerSettings()
    sts = [line(w, i, SourceUnit(i, "S1", 50.0 + 5 * i, 53.0 + 5 * i, "So here we are."), 55.0 + 5 * i, FULL)
           for i in range(4)]
    diarized(monkeypatch, w, [SpeakerTurn("S2", 49.0, 49.9)])
    assert w._slot(sts[0]).prev_end == pytest.approx(49.9) and len(w._ahead(sts[0])) == 3
    assert bare(tmp_path / "other", timing="v3")[0].timing == "v2"


async def test_timing_v1_compresses_a_take_evenly(tmp_path):
    tts = MelPausingTTS(0.3)
    s, _ = bare(tmp_path, tts, timing="v1")
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = len(tts.synthesize(text, {"f0": 140.0})) / SR
    u = SourceUnit(0, "S1", 30.0, 30.0 + natural / 1.15, "We went, and came back.")
    st = line(s, 0, u, 30.0 + natural / 1.15 + 0.2, Wording(text))
    await s._dub(st)
    assert st.plan.rate > 1.0 and tts.rates == [1.0, round(st.plan.rate, 4)]  # its pause sped up as much as its speech


def test_marks_are_the_wordings_inner_breaks_as_shares_of_its_aksharas():
    assert _marks("మేము వెళ్ళాం.") == ()
    got = _marks("మేము వెళ్ళాం, తిరిగి వచ్చాం.")
    assert len(got) == 1 and 0.3 < got[0] < 0.6
    assert len(_marks("అవును. మేము వెళ్ళాం? సరే!")) == 2 and _marks("") == ()


async def test_units_jsonl_carries_the_timing_fields_and_a_skipped_line_its_speech(tmp_path):
    s, _ = bare(tmp_path)
    st = line(s, 0, SourceUnit(0, "S1", 10.0, 14.0, "A line of ours."), 20.0, FULL)
    await s._dub(st)
    (ev,) = units(s)
    assert {"said", "parts", "rate_cap", "hard_breaks", "speech", "voiced", "anchor_errors", "end_error_s",
            "overlap_speech"} <= ev.keys()
    assert ev["speech"] == [[10.0, 14.0]] and ev["voiced"][0][0] == pytest.approx(10.0, abs=0.02)  # the take's lead-in
    assert ev["end_error_s"] == pytest.approx(ev["voiced"][-1][1] - 14.0, abs=1e-3) and 0 < ev["overlap_speech"] <= 1
    gone = line(s, 1, SourceUnit(1, "S1", 30.0, 33.0, "Lost."), None, FULL)
    await s._skip(gone, "synthesis failed")
    skipped = [e for e in map(json.loads, (s._dir / "units.jsonl").read_text().splitlines()) if e["event"] == "skipped"]
    assert skipped[0]["speech"] == [[30.0, 33.0]]
    assert s.planner.stats()["speech_lines"] == 1
