"""A render job voicing its lines (ARCHITECTURE §3.8, §3.10, §3.13; OFFLINE-RENDER §2.9, §2.12): the takes of a line and
the one voiced, a shorter wording only when its take passes, timing v2 (pieces at hard breaks, soft anchors, uneven
compression, early starts only into real silence, the akshara ceiling, the windowed lookahead) and the timing fields of
units.jsonl, each line voiced by the dub loop and made final as the final plan places it. Timing v1 and v2-whole on
`Dubber` itself. No models, no Claude: the mock voice and small stand-ins. All English and Telugu here is original test
text. Ported from the streaming session's tests (OFFLINE-RENDER §10 step 6)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fakes import bare_job, pcm, put, trace, voice  # noqa: E402

from maata_engine.backends.base import Backend, LineResult, Wording  # noqa: E402
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS  # noqa: E402
from maata_engine.dubber import Dubber, UnitState, VoiceCost, _marks  # noqa: E402
from maata_engine.render import _read_rows  # noqa: E402
from maata_engine.timing import pauses as pz  # noqa: E402
from maata_engine.timing.duration import VoiceKey  # noqa: E402
from maata_engine.timing.planner import V1, Plan, PlannerSettings, rate_ceiling  # noqa: E402
from maata_engine.types import SourceUnit, SpeakerTurn, TimedWord  # noqa: E402

SR = MockTTS.sample_rate
FULL = Wording("మేము నిన్న ఊరికి వెళ్ళాం ఇవాళ మళ్ళీ తిరిగి వచ్చేశాం")
PIECES = ("మేము నిన్న ఊరికి వెళ్ళాం", "ఇవాళ మళ్ళీ తిరిగి వచ్చేశాం")
LINE = Wording("ఇది ఒక చిన్న మాట అని చెప్పారు.")


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
    """PausingTTS split into a take plus vocoding, recording the rate each take is vocoded at, and keeping a take in a
    take file as Chatterbox does (`pack_take`, `unpack_take`)."""

    def __init__(self, gap: float = 0.3) -> None:
        super().__init__(gap)
        self.rates: list[float] = []

    def synthesize_mel(self, text, voice, language="te", max_seconds=None):
        return Take(self.synthesize(text, voice, language))

    def vocode(self, take, rate=1.0):
        self.rates.append(round(rate, 4))
        n = len(take.samples)
        return take.samples[np.minimum(np.round(np.arange(0, n, rate)).astype(int), n - 1)][: round(n / rate)]

    def pack_take(self, take: Take) -> dict[str, np.ndarray]:
        return {"mel": take.samples}

    def unpack_take(self, d: dict[str, np.ndarray]) -> Take:
        return Take(np.asarray(d["mel"], np.float32))


@dataclass
class BatchTake:
    samples: np.ndarray
    capped: bool = False
    t3_s: float = 0.01
    t3_tokens: int = 20
    t3_steps: int = 24
    flow_s: float = 0.0
    cfm_steps: int = 6

    @property
    def seconds(self) -> float:
        return len(self.samples) / MockTTS.sample_rate


class BatchTTS(MockTTS):
    """A TTS that batches takes: each call's takes are the mock voice's line at the given paces (1.0 = as the mock says
    it), capped where `caps` says. `script`: per call, (paces, caps); after it runs out, every take is at pace 1.0."""

    def __init__(self, script: list[tuple[list[float], list[bool]]] | None = None) -> None:
        self.script, self.calls = list(script or []), []

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1):
        self.calls.append(n)
        paces, caps = self.script.pop(0) if self.script else ([1.0] * n, [False] * n)
        base = self.synthesize(text, voice, language)
        return [BatchTake(np.resize(base, max(1, round(len(base) * p))), c) for p, c in zip(paces[:n], caps[:n])]

    def synthesize_mel(self, text, voice, language="te", max_seconds=None):
        return self.synthesize_takes(text, voice, language, max_seconds)[0]

    def vocode(self, take, rate=1.0):
        return take.samples[: round(len(take.samples) / rate)]

    def pack_take(self, take: BatchTake) -> dict[str, np.ndarray]:
        return {"mel": take.samples, "capped": np.bool_(take.capped)}

    def unpack_take(self, d: dict[str, np.ndarray]) -> BatchTake:
        return BatchTake(np.asarray(d["mel"], np.float32), bool(d["capped"]))


def words(*spec: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(w, a, b) for w, a, b in spec]


def line(job, i: int, u: SourceUnit, nxt: float | None, full: Wording, pieces=(), tier="full") -> UnitState:
    st = UnitState(u, nxt, speech_s=u.end - u.start)
    st.line, st.tier, st.telugu = LineResult(i, {"full": full}, pieces=pieces), tier, full.spoken
    put(job, st)
    return st


def diarized(monkeypatch, job, ex: list[SpeakerTurn], overlapping: list[SpeakerTurn] | None = None) -> None:
    """Stand in the registry's turns: the exclusive track, and the overlapping one (the same when not given)."""
    def turns_in(a, b, exclusive=True):
        track = ex if exclusive or overlapping is None else overlapping
        return [SpeakerTurn(t.speaker, max(t.start, a), min(t.end, b)) for t in track if t.end > a and t.start < b]

    monkeypatch.setattr(job.registry, "turns_in", turns_in)


def stretch_with_a_break(i: int = 0, start: float = 10.0) -> SourceUnit:
    """A 7 s English sentence with a 1.4 s pause after its third word (a hard break)."""
    ws = words(("We", start, start + 0.4), ("went", start + 0.5, start + 1.2), ("home,", start + 1.3, start + 2.8),
               ("and", start + 4.2, start + 4.6), ("came", start + 4.7, start + 5.5), ("back.", start + 5.6, start + 7.0))
    return SourceUnit(i, "S1", start, start + 7.0, "We went home, and came back.", ws, breaks=[3])


# ---- the takes of a line (§3.8, §3.13) ------------------------------------------------------------------------------
def a_line(job, i: int, start: float, end: float) -> UnitState:
    st = UnitState(SourceUnit(i, "S1", start, end, "A short line here."), None, speech_s=end - start)
    st.line, st.tier, st.telugu = LineResult(i, {"full": LINE}), "full", LINE.spoken
    put(job, st)
    return st


async def test_the_take_closest_to_the_speech_time_is_voiced_of_a_batch(tmp_path):
    tts = BatchTTS([([1.0, 0.8, 1.3], [False, False, False])])
    job = bare_job(tmp_path, tts)
    st = a_line(job, 0, 0.0, 0.0)
    key = VoiceKey("S1", 0.5, 0.5, "r")
    natural = len(tts.synthesize(LINE.spoken, {"f0": 140.0})) / tts.sample_rate
    st.speech_s = 0.8 * natural                    # the 0.8-pace take fits it
    cost = VoiceCost()
    take, dur, pace, failed = await job._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=3)
    assert tts.calls == [3] and dur == pytest.approx(take.seconds) and dur == pytest.approx(0.8 * natural, abs=1e-3)
    assert pace == pytest.approx(natural * (1.0 + 0.8 + 1.3) / 3, abs=1e-3)  # the estimator learns from them all
    assert (cost.takes, cost.takes_n, cost.retakes, cost.failures, failed) == (3, 3, 0, [], None)


async def test_a_cap_hit_or_a_far_too_short_take_is_passed_over_and_a_batch_that_all_failed_gets_one_more_take(tmp_path):
    tts = BatchTTS([([1.0, 0.3], [True, False]),   # a runaway, and one far under the estimate: both failed
                    ([1.05], [False])])            # the retake
    job = bare_job(tmp_path, tts)
    st = a_line(job, 0, 0.0, 3.0)
    key = VoiceKey("S1", 0.5, 0.5, "r")
    natural = len(tts.synthesize(LINE.spoken, {"f0": 140.0})) / tts.sample_rate
    job.estimator.estimate = lambda text, k: natural  # the estimate says the line runs its natural length
    cost = VoiceCost()
    take, dur, pace, failed = await job._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=2)
    assert tts.calls == [2, 1] and dur == pytest.approx(1.05 * natural, abs=1e-3) and failed is None
    assert cost.failures == ["cap", "short"] and cost.takes == 3 and cost.takes_n == 2 and cost.retakes == 1
    assert pace == pytest.approx(1.05 * natural, abs=1e-3)  # only the take that passed teaches the estimator

    tts.script = [([1.0], [True]), ([1.0], [True])]  # everything runs away: the least bad is still voiced
    take, _, pace, failed = await job._take(st, LINE, {"f0": 140.0}, key, 5.0, VoiceCost(), n=1)
    assert take.capped and pace is None and failed == "cap"

    tts.calls, tts.script = [], [([1.0, 0.95], [True, False])]  # one failed, one didn't: no retake
    cost = VoiceCost()
    take, dur, _, failed = await job._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=2)
    assert tts.calls == [2] and not take.capped and cost.failures == ["cap"] and cost.retakes == 0 and failed is None

    # Every take far under the estimate: the voice is faster than the estimator thinks, and it learns that from them.
    tts.script = [([0.5, 0.4], [False, False]), ([0.45], [False])]
    take, dur, pace, failed = await job._take(st, LINE, {"f0": 140.0}, key, 5.0, VoiceCost(), n=2)
    assert failed == "short" and pace == pytest.approx(natural * (0.5 + 0.4 + 0.45) / 3, abs=1e-3)


async def test_a_tts_that_cant_batch_says_a_line_once(tmp_path):
    job = bare_job(tmp_path, MockTTS())
    st = a_line(job, 0, 0.0, 3.0)
    job.estimator.estimate = lambda text, k: 10.0   # the take is far shorter than the estimate
    cost = VoiceCost()
    _, dur, pace, failed = await job._take(st, LINE, {"f0": 140.0}, VoiceKey("S1", 0.5, 0.5, "r"), 5.0, cost, n=3)
    assert cost.takes == cost.takes_n == 1 and cost.failures == ["short"] and failed == "short"  # judged, never retaken
    assert pace == dur and cost.retakes == 0


async def test_a_shorter_wording_is_voiced_instead_only_when_its_take_passes(tmp_path):
    tts = BatchTTS([([1.0], [False]), ([0.3], [False]), ([0.3], [False]),  # line 0: its shorter wording's takes fail
                    ([1.0], [False])])  # line 1: its `full` from the take cache (the same words), its shorter one passes
    job = bare_job(tmp_path, tts)
    job._budget = 10.0  # (the range's fix-up budget)
    job.estimator.estimate = lambda text, k: len(tts.synthesize(text, {"f0": 140.0})) / tts.sample_rate
    job._plan = lambda st, said, planner=None: Plan(st.unit.id, st.unit.start, 1.0, said.total, 0.0,
                                                    needs_shorter=True)  # every take runs long
    shorter = [Wording("చిన్న మాట."), Wording("చిన్న పని.")]
    for i in (0, 1):
        st = a_line(job, i, 10.0 + 10 * i, 13.0 + 10 * i)
        st.line = LineResult(i, {"full": LINE, "concise": shorter[i]})
        await job._voice_line(st)
    rows = _read_rows(job.render_dir / "takes.jsonl")
    assert [(r["wording"], r["takes"][0]["spoken"]) for r in rows] == [("full", LINE.spoken),  # a failed take is
                                                                         ("concise", shorter[1].spoken)]  # shorter
    assert [(r["fixups"], r["cost"]["retakes"], r["cost"]["take_failures"]) for r in rows] == [  # for the wrong reason
        (1, 1, ["short", "short"]), (1, 0, [])]
    # TAKES_N asked for each batch (the render's fixed N, §2.9), one take for the retake; line 1's `full` from its file
    assert tts.calls == [2, 2, 1, 2] and job._resynths == 2 and job.units[1].tier == "concise"


# ---- timing v2 (§3.10) ------------------------------------------------------------------------------------------------
async def test_a_line_with_a_hard_break_is_said_as_its_pieces_each_at_its_own_onset(tmp_path):
    job = bare_job(tmp_path)
    st = line(job, 0, stretch_with_a_break(), 25.0, FULL, PIECES)
    plan = await voice(job, st)
    (u,) = trace(job, "unit")
    assert u["said"] == "pieces" and u["parts"] == 2 and u["hard_breaks"] == 1 and u["wording"] == "piece:2"
    first, second = plan.parts
    assert first.start == pytest.approx(10.0) and second.src == pytest.approx(14.2) and second.start >= 14.2 - 0.3
    assert u["anchor_errors"] and abs(u["anchor_errors"][0]) <= 0.3 + 1e-6
    # One buffer from the dub's start: the second piece's take where the plan puts it, silence in the pause.
    audio = pcm(job, st)
    assert len(audio) == round(plan.wall * SR)
    gap = audio[round((first.wall + 0.05) * SR):round((second.start - plan.start - 0.05) * SR)]
    assert gap.size > 0.5 * SR and np.abs(gap).max() == 0.0
    assert np.abs(audio[round((second.start - plan.start) * SR):]).max() > 0.05
    out = job._out[0]
    assert out["telugu"] == FULL.spoken and out["audioWall"] == pytest.approx(plan.wall, abs=1e-4)
    assert st.take_s == pytest.approx(sum(len(MockTTS().synthesize(p, {"f0": 140.0})) for p in PIECES) / SR, abs=0.01)


async def test_pieces_are_not_used_off_full_or_across_another_line_in_the_pause(tmp_path):
    job = bare_job(tmp_path)
    st = line(job, 0, stretch_with_a_break(), 25.0, FULL, PIECES, tier="concise")
    st.line = LineResult(0, {"full": FULL, "concise": Wording(PIECES[0])}, pieces=PIECES)
    key = job._key("S1")
    assert job._pieces(st, key) == []                                        # the pieces belong to `full`
    st.tier = "full"
    got = job._pieces(st, key)
    assert [w.spoken for w in got] == list(PIECES)
    # Another speaker answers in the pause: the break is not one to meet, and the line drifts over it whole.
    put(job, UnitState(SourceUnit(1, "S2", 13.2, 13.9, "Right."), 25.0, speech_s=0.7))
    assert job._slot(st).breaks == () and job._pieces(st, key) == []


def test_a_piece_starts_early_only_into_the_silence_left_in_its_pause(tmp_path, monkeypatch):
    job = bare_job(tmp_path)
    st = line(job, 0, stretch_with_a_break(), 17.2, FULL, PIECES)          # its pause: 12.8-14.2 s
    long_second = [(2.4, ()), (3.4, ())]                                  # so long that it has to start early

    def second() -> float:
        p = job.planner.evaluate(job._slot(st), 5.8, pieces=long_second)[1]
        assert p.said == "pieces"
        return p.parts[1].start

    assert second() < 14.2 - 0.05                                         # into the silent pause
    # Another speaker answers from before the pause and on into it: nobody starts inside it, so the break stays, but
    # the second piece doesn't start over them.
    put(job, UnitState(SourceUnit(1, "S2", 12.3, 14.1, "Sure, go on."), 17.2, speech_s=1.8))
    slot = job._slot(st)
    assert slot.breaks == ((12.8, 14.2),) and slot.heard == ((12.8, 14.1),)
    assert second() >= 14.2 - 1e-9
    # A laugh in the pause with no line of its own counts the same (its diarized turn); the speaker's own doesn't.
    job._index([st])
    diarized(monkeypatch, job, [SpeakerTurn("S1", 10.0, 12.8)],
             [SpeakerTurn("S1", 10.0, 13.0), SpeakerTurn("S2", 13.2, 13.9)])
    assert job._slot(st).heard == ((13.2, 13.9),)
    assert 13.9 + job.planner.s.guard - 1e-9 <= second() < 14.2


def test_the_speech_yardstick_is_the_speakers_own_turns_overlap_included(tmp_path, monkeypatch):
    job = bare_job(tmp_path)
    u = SourceUnit(0, "S1", 20.0, 26.0, "And that was the whole trip.")
    # S2 cuts in at 25.2 while S1 talks on to 25.9. Exclusive turns give the overlap to one speaker (here S2), so S1's
    # would end at 25.2 and a dub ending with S1 would count as 0.7 s late.
    diarized(monkeypatch, job, [SpeakerTurn("S1", 20.0, 25.2), SpeakerTurn("S2", 25.2, 26.5)],
             [SpeakerTurn("S1", 20.0, 25.9), SpeakerTurn("S2", 25.2, 26.5)])
    assert job._speech_spans(u) == ((20.0, 25.9),)
    diarized(monkeypatch, job, [])
    assert job._speech_spans(u) == ((20.0, 26.0),)                            # nothing diarized: the span


def test_the_pieces_carry_their_share_of_the_english_map(tmp_path):
    job = bare_job(tmp_path)
    full = Wording("మనం ఆఫీస్ కి వెళ్ళాలి తర్వాత మీటింగ్ ఉంది", ((1, "office"), (5, "meeting")))
    st = line(job, 0, stretch_with_a_break(), 25.0, full, ("మనం ఆఫీస్ కి వెళ్ళాలి", "తర్వాత మీటింగ్ ఉంది"))
    got = job._pieces(st, job._key("S1"))
    assert got[0].english == ((1, "office"),) and got[1].english == ((1, "meeting"),)


async def test_a_line_that_fits_waits_at_its_own_comma_for_the_english_to_resume(tmp_path):
    tts = PausingTTS(0.3)
    job = bare_job(tmp_path, tts)
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = tts.synthesize(text, {"f0": 140.0})
    (a, b), = pz.internal(pz.pauses(natural, SR), len(natural) / SR)
    # The English breathes (0.35 s, a soft anchor) and resumes 0.25 s after the Telugu pause would end.
    resume = 20.0 + b + 0.25
    ws = words(("We", 20.0, 20.5), ("went,", 20.6, resume - 0.35), ("and", resume, resume + 0.4),
               ("came", resume + 0.5, resume + 1.2), ("back.", resume + 1.3, resume + 2.0))
    u = SourceUnit(0, "S1", 20.0, resume + 2.0, "We went, and came back.", ws, anchors=[2])
    st = line(job, 0, u, resume + 5.0, Wording(text))
    await voice(job, st)
    (ev,) = trace(job, "unit")
    assert ev["said"] == "anchored" and ev["audio_rate"] == 1.0 and ev["anchor_errors"] == [0.0]
    audio = pcm(job, st)
    (a2, b2), = pz.internal(pz.pauses(audio, SR), len(audio) / SR)
    assert (b2 - a2) - (b - a) == pytest.approx(0.25, abs=0.02)            # the silence set in the Telugu pause
    assert ev["voiced"][1][0] == pytest.approx(resume, abs=0.02)


@pytest.mark.parametrize("gap,short", [(0.3, 1.15), (0.6, 1.04)])
async def test_a_sped_up_take_gives_up_its_pause_before_its_speech_speeds_up(tmp_path, gap, short):
    tts = MelPausingTTS(gap)
    job = bare_job(tmp_path, tts)
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = len(tts.synthesize(text, {"f0": 140.0})) / SR
    # A span shorter than the take, and the next line right after it: it has to go faster.
    u = SourceUnit(0, "S1", 30.0, 30.0 + natural / short, "We went, and came back.")
    st = line(job, 0, u, 30.0 + natural / short + 0.2, Wording(text))
    plan = await voice(job, st)
    rate = plan.rate
    assert rate > 1.0 and plan.said == "whole" and tts.rates[0] == 1.0  # vocoded at 1.0 first: its pauses found
    audio = pcm(job, st)
    assert len(audio) == round(natural / rate * SR)                        # exactly the planned length
    (a, b), = pz.internal(pz.pauses(audio, SR, min_len=0.05), len(audio) / SR)  # a kept pause is under 0.15 s
    if natural - natural / rate > gap - job.planner.s.keep_pause:
        # The pause gives what it can (down to about 0.12 s) and the speech the rest, slower than the plan's rate.
        assert len(tts.rates) == 2 and 1.0 < tts.rates[1] < rate
        assert b - a == pytest.approx(job.planner.s.keep_pause, abs=0.02)
    else:
        # The pause gives it all: the speech isn't sped up at all, and is vocoded once.
        assert tts.rates == [1.0]
        assert b - a == pytest.approx(gap - (natural - natural / rate), abs=0.02)


def test_an_early_start_goes_only_into_real_silence(tmp_path, monkeypatch):
    job = bare_job(tmp_path)
    st = line(job, 0, SourceUnit(0, "S1", 50.0, 53.0, "So here we are."), 60.0, FULL)
    turns: list[SpeakerTurn] = []

    def turns_in(a, b, exclusive=True):
        return [SpeakerTurn(t.speaker, max(t.start, a), min(t.end, b)) for t in turns if t.end > a and t.start < b]

    monkeypatch.setattr(job.registry, "turns_in", turns_in)
    assert job._slot(st).prev_end is None                                     # nothing heard before it
    turns[:] = [SpeakerTurn("S2", 49.0, 49.9)]                              # someone laughs until 49.9 s: no line of theirs
    assert job._slot(st).prev_end == pytest.approx(49.9)
    turns[:] = [SpeakerTurn("S1", 49.8, 53.0)]                              # the speaker's own turn, heard before the words
    assert job._slot(st).prev_end is None
    turns[:] = [SpeakerTurn("S1", 49.0, 49.6), SpeakerTurn("S1", 49.8, 53.0)]  # ...and a breath of theirs before it
    assert job._slot(st).prev_end == pytest.approx(49.6)
    p = job.planner.evaluate(job._slot(st), 3.4)[1]
    assert p.lag >= -(50.0 - 49.6 - job.planner.s.guard) - 1e-9
    turns[:] = [SpeakerTurn("S2", 48.0, 50.5)]                              # crosstalk into the onset: no early start
    assert job._slot(st).prev_end == pytest.approx(50.0) and job.planner.evaluate(job._slot(st), 3.4)[1].lag >= 0.0


def test_each_line_goes_no_faster_than_its_voices_akshara_ceiling(tmp_path, monkeypatch):
    job = bare_job(tmp_path)
    st = line(job, 0, SourceUnit(0, "S1", 50.0, 53.0, "So here we are."), 53.2, FULL)
    monkeypatch.setattr(job.estimator, "rate", lambda key: 9.0)
    assert job._slot(st).rate_cap == job.planner.s.speed_cap                    # the ceiling is off by default
    job.planner.s = replace(job.planner.s, akshara_ceiling=7.5)                 # as when a Telugu measurement sets it
    monkeypatch.setattr(job.estimator, "rate", lambda key: 6.1)
    assert job._slot(st).rate_cap == job.planner.s.speed_cap                    # the default pace: the speed cap binds
    monkeypatch.setattr(job.estimator, "rate", lambda key: 7.0)                # a fast talker: 7.5 / 7.0
    assert job._slot(st).rate_cap == pytest.approx(rate_ceiling(7.0, job.planner.s)) == pytest.approx(7.5 / 7.0)
    assert job.planner.evaluate(job._slot(st), 5.0)[1].rate == pytest.approx(7.5 / 7.0)


def test_the_lookahead_window_is_the_next_line_then_up_to_8_within_30_s(tmp_path):
    job = bare_job(tmp_path)
    sts = [line(job, i, SourceUnit(i, "S1", 100.0 + 5 * i, 103.0 + 5 * i, "A line."), 105.0 + 5 * i, FULL)
           for i in range(12)]
    sts[3].line = None                                                      # not translated yet
    ahead = job._ahead(sts[0])
    assert [sl.id for sl, _ in ahead] == [1, 2, 3, 4, 5, 6]                 # line 7 starts 35 s after line 0
    assert ahead[2][1] is None and ahead[0][1] == pytest.approx(job.estimator.estimate(FULL.spoken, job._key("S1")))
    far = line(job, 20, SourceUnit(20, "S1", 400.0, 402.0, "Later."), None, FULL)
    assert [sl.id for sl, _ in job._ahead(sts[11])] == [20]                 # the next line, however far
    assert job._ahead(far) == ()


def test_marks_are_the_wordings_inner_breaks_as_shares_of_its_aksharas():
    assert _marks("మేము వెళ్ళాం.") == ()
    got = _marks("మేము వెళ్ళాం, తిరిగి వచ్చాం.")
    assert len(got) == 1 and 0.3 < got[0] < 0.6
    assert len(_marks("అవును. మేము వెళ్ళాం? సరే!")) == 2 and _marks("") == ()


async def test_units_jsonl_carries_the_timing_fields_and_a_skipped_line_its_speech(tmp_path):
    job = bare_job(tmp_path)
    st = line(job, 0, SourceUnit(0, "S1", 10.0, 14.0, "A line of ours."), 20.0, FULL)
    await voice(job, st)
    (ev,) = trace(job, "unit")
    assert {"said", "parts", "rate_cap", "hard_breaks", "speech", "voiced", "anchor_errors", "end_error_s",
            "overlap_speech"} <= ev.keys()
    assert ev["speech"] == [[10.0, 14.0]] and ev["voiced"][0][0] == pytest.approx(10.0, abs=0.02)  # the take's lead-in
    assert ev["end_error_s"] == pytest.approx(ev["voiced"][-1][1] - 14.0, abs=1e-3) and 0 < ev["overlap_speech"] <= 1
    gone = line(job, 1, SourceUnit(1, "S1", 30.0, 33.0, "Lost."), None, FULL)
    await job._skip(gone, "synthesis failed")
    (skipped,) = trace(job, "skipped")
    assert skipped["speech"] == [[30.0, 33.0]] and job.skipped == {1: "synthesis failed"}
    assert job.planner.stats()["speech_lines"] == 1


# ---- timing v1 and v2-whole, on Dubber (the measurement arms of §7.1; a render job times lines with v2) --------------
def dubber(tmp_path, tts=None, timing="v2") -> Dubber:
    d = Dubber(Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts or MockTTS()),
               timing=timing)
    d._dir = tmp_path
    return d


def dubber_line(d: Dubber, i: int, u: SourceUnit, nxt: float | None, full: Wording, pieces=()) -> UnitState:
    st = UnitState(u, nxt, speech_s=u.end - u.start)
    st.line, st.tier, st.telugu = LineResult(i, {"full": full}, pieces=pieces), "full", full.spoken
    d.units[i] = st
    return st


@pytest.mark.parametrize("mode", ["v1", "v2-whole"])
async def test_timing_v1_and_v2_whole_say_a_sentence_with_a_hard_break_whole(tmp_path, mode):
    d = dubber(tmp_path, timing=mode)
    st = dubber_line(d, 0, stretch_with_a_break(), 25.0, FULL, PIECES)
    key = d._key("S1")
    assert d._pieces(st, key) == []
    said = await d._say(st, [FULL], {"f0": 140.0}, key, 7.0, VoiceCost())
    plan = d._plan(st, said)
    row = d._timing(d._slot(st), plan)
    assert plan.said == "whole" and row["parts"] == 0 and row["hard_breaks"] == 1 and row["end_error_s"] is not None


def test_timing_v1_is_the_timing_before_v2_and_v2_whole_only_drops_the_pieces(tmp_path, monkeypatch):
    d = dubber(tmp_path, timing="v1")
    assert d.timing == "v1" and d.planner.s == PlannerSettings(**V1)
    sts = [dubber_line(d, i, SourceUnit(i, "S1", 50.0 + 5 * i, 53.0 + 5 * i, "So here we are."), 55.0 + 5 * i, FULL)
           for i in range(4)]
    diarized(monkeypatch, d, [SpeakerTurn("S2", 49.0, 49.9)])
    monkeypatch.setattr(d.estimator, "rate", lambda key: 7.0)
    slot = d._slot(sts[0])
    assert slot.prev_end is None and slot.rate_cap == d.planner.s.speed_cap  # lines alone bound a lead; no akshara ceiling
    assert [sl.id for sl, _ in d._ahead(sts[0])] == [1]                     # a one-line lookahead
    w = dubber(tmp_path, timing="v2-whole")
    assert w.timing == "v2-whole" and w.planner.s == PlannerSettings()
    sts = [dubber_line(w, i, SourceUnit(i, "S1", 50.0 + 5 * i, 53.0 + 5 * i, "So here we are."), 55.0 + 5 * i, FULL)
           for i in range(4)]
    diarized(monkeypatch, w, [SpeakerTurn("S2", 49.0, 49.9)])
    assert w._slot(sts[0]).prev_end == pytest.approx(49.9) and len(w._ahead(sts[0])) == 3
    assert dubber(tmp_path, timing="v3").timing == "v2"


async def test_timing_v1_compresses_a_take_evenly(tmp_path):
    tts = MelPausingTTS(0.3)
    d = dubber(tmp_path, tts, timing="v1")
    text = "మేము నిన్న ఊరికి వెళ్ళాం, ఇవాళ తిరిగి వచ్చాం."
    natural = len(tts.synthesize(text, {"f0": 140.0})) / SR
    u = SourceUnit(0, "S1", 30.0, 30.0 + natural / 1.15, "We went, and came back.")
    st = dubber_line(d, 0, u, 30.0 + natural / 1.15 + 0.2, Wording(text))
    cost = VoiceCost()
    said = await d._say(st, [Wording(text)], {"f0": 140.0}, d._key("S1"), 5.0, cost)
    plan = d._plan(st, said)
    await d._play(said, plan, cost)
    assert plan.rate > 1.0 and tts.rates == [1.0, round(plan.rate, 4)]  # its pause sped up as much as its speech
