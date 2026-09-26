"""The session's real-time shape (ARCHITECTURE §3.12, §5.2, §5.3): GPU priorities per stage, the throughput governor and
batched takes, the pacing horizon, and dub PCM kept only near the playhead. No models, no Claude: the demo backend's
mock stages or small stand-ins. All English and Telugu here is original test text."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from maata_engine import session as sm
from maata_engine.backends.base import Backend, LineResult, LineSpec, SceneRequest, Wording
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.gpu import BACKGROUND, FRONTIER, URGENT, VOICE
from maata_engine.resolve import DemoResolver
from maata_engine.server import Engine
from maata_engine.session import Chunk, Session, UnitState, VoiceCost
from maata_engine.timing.duration import VoiceKey
from maata_engine.speakers import DiarBlock
from maata_engine.timing.planner import Plan
from maata_engine.types import SourceUnit, SpeakerTurn

VIDEO = "https://youtu.be/dQw4w9WgXcQ"
LINE = Wording("ఇది ఒక చిన్న మాట అని చెప్పారు.")


@dataclass
class Take:
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
        return [Take(np.resize(base, max(1, round(len(base) * p))), c) for p, c in zip(paces[:n], caps[:n])]

    def synthesize_mel(self, text, voice, language="te", max_seconds=None):
        return self.synthesize_takes(text, voice, language, max_seconds)[0]

    def vocode(self, take, rate=1.0):
        return take.samples[: round(len(take.samples) / rate)]


def bare(tmp_path, tts=None, duration: float = 600.0, **kw) -> tuple[Session, list, list]:
    """A session with no stages running: a `duration`-second video, a cache directory and recorded sends."""
    msgs, frames = [], []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(b: bytes) -> None:
        frames.append(struct.unpack("<IIII", b[:16])[0])

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts or MockTTS())
    s = Session(backend, DemoResolver(), tmp_path, sj, sb, **kw)
    s._duration = lambda: duration
    s._dir = tmp_path / "vid"
    (s._dir / "pcm").mkdir(parents=True)
    return s, msgs, frames


def add(s: Session, i: int, start: float, end: float, voiced=False, scene=None, chunk: Chunk | None = None) -> UnitState:
    """A heard line, in a stretch of its own unless `chunk` says which."""
    if chunk is None:
        chunk = Chunk(start, end)
        s.chunks.append(chunk)
    st = UnitState(SourceUnit(i, "S1", start, end, "A short line here."), None, chunk, speech_s=end - start)
    st.voiced, st.scene = voiced, scene
    s.units[i] = st
    return st


def rate(s: Session, r: float) -> None:
    s.throughput.rate = lambda: r


# ---- the lead and GPU priorities (§5.3) -----------------------------------------------------------------------------
def test_the_lead_runs_through_voiced_lines_to_the_first_one_not_voiced(tmp_path):
    s, _, _ = bare(tmp_path)
    heard = Chunk(0.0, 60.0)
    s.chunks.append(heard)
    for i, (a, b) in enumerate([(0, 8), (9, 18), (20, 28), (40, 50)]):
        add(s, i, a, b, voiced=i < 3, chunk=heard)
    s.playhead = 5.0
    assert s._lead() == 35.0                      # to where line 3 starts: the pause before it is silence, not a hole
    s.units[3].voiced = True
    s.ready.add(0.0, 60.0)                        # the stretch is dubbed
    assert s._lead() == 55.0
    later = Chunk(60.0, 120.0)
    s.chunks.append(later)
    add(s, 4, 61, 70, voiced=True, chunk=later)
    add(s, 5, 72, 80, chunk=later)
    assert s._lead() == 67.0                      # the ready ranges, then the next stretch's voiced lines
    add(s, 6, 900, 910, voiced=True)              # dubbed far ahead after a seek: not joined up with this
    assert s._lead() == 67.0
    s.playhead = 500.0                            # nothing heard here
    assert s._lead() == 0.0


async def test_a_seek_into_heard_video_not_dubbed_yet_is_dubbed_from_the_seek_point_and_the_lead_grows_past_it(tmp_path):
    s, msgs, _ = bare(tmp_path, duration=600.0, lookahead=300.0)
    s._asr_cursor = 300.0                          # ASR is well ahead of the voicer
    heard, after = Chunk(0.0, 60.0), Chunk(60.0, 120.0)
    s.chunks += [heard, after]
    for i, (a, b) in enumerate([(2, 10), (12, 20), (32, 40), (45, 55), (62, 70), (75, 85), (90, 110)]):
        add(s, i, a, b, chunk=heard if b <= 60 else after)
        (heard if b <= 60 else after).pending.add(i)
    s.seek(30.0)                                   # the lines before it are passed over, never voiced
    await asyncio.gather(*s._side_tasks)
    assert s._next_to_voice() is None and {st.unit.id for st in s.units.values() if st.unit.end >= 29.0} == {2, 3, 4, 5, 6}
    for i in (2, 3, 4, 5, 6):
        s.units[i].voiced = True
        await s._unit_done(s.units[i])
    assert s.ready.ranges == [[30.0, 120.0]] and s.ready.lead(30.0) == 90.0  # what the UI plays on
    assert msgs[-1]["type"] == "ready" and msgs[-1]["ranges"] == [[30.0, 120.0]]
    assert s._lead() == 90.0 and s._voice_priority() == VOICE
    assert s._takes_n(s.units[6]) == 2 and not s._recovering       # the lead is back past its minute
    assert s.throughput.total == 90.0                               # the video dubbed, counted once

    s.seek(15.0)                                   # back: the line from 12 s is waited on again, the one before stays passed
    await asyncio.gather(*s._side_tasks)
    assert heard.pending == {1} and heard.passed == {0} and not heard.ready
    assert s._lead() == 0.0                        # it starts under the playhead
    s.units[1].voiced = True
    await s._unit_done(s.units[1])
    assert s.ready.ranges == [[15.0, 120.0]] and s._lead() == 105.0 and s.throughput.total == 105.0

    # A seek into a heard stretch whose lines past the seek point are all voiced: dubbed from there at once.
    late = Chunk(120.0, 180.0)
    s.chunks.append(late)
    add(s, 7, 122, 130, chunk=late)
    add(s, 8, 150, 160, voiced=True, chunk=late)
    late.pending.add(7)
    s.seek(140.0)
    await asyncio.gather(*s._side_tasks)
    assert late.ready and s.ready.covered(140.0, 180.0) and not s.ready.covered(125.0, 130.0)
    assert [m for m in msgs if m["type"] == "ready"][-1]["ranges"] == [[15.0, 120.0], [140.0, 180.0]]


def test_the_lead_goes_on_through_every_stretch_dubbed_after_the_one_being_voiced(tmp_path):
    s, _, _ = bare(tmp_path)
    first, second, third = Chunk(0.0, 60.0), Chunk(60.0, 120.0), Chunk(120.0, 180.0)
    s.chunks += [first, second, third]
    add(s, 0, 5, 20, voiced=True, chunk=first)
    s.ready.add(60.0, 120.0)
    add(s, 1, 125, 140, voiced=True, chunk=third)
    add(s, 2, 150, 160, chunk=third)
    assert s._lead() == 150.0                      # through the first stretch's lines, the ready one, then the third's


def test_the_voicer_goes_first_on_the_gpu_while_its_lead_is_under_a_minute(tmp_path):
    s, _, _ = bare(tmp_path)
    heard = Chunk(0.0, 120.0)
    s.chunks.append(heard)
    add(s, 0, 0, 30, voiced=True, chunk=heard)
    add(s, 1, 50, 58, chunk=heard)
    assert s._voice_priority() == URGENT          # 50 s ahead
    s.units[1].voiced = True
    add(s, 2, 70, 80, chunk=heard)
    assert s._voice_priority() == VOICE           # 70 s ahead


def test_listening_the_translator_waits_on_goes_before_the_voicer_and_the_rest_after(tmp_path):
    s, _, _ = bare(tmp_path, lookahead=300.0)
    for i in range(5):
        add(s, i, 10.0 * i, 10.0 * i + 8, scene=1)  # handed to Claude up to 48 s
    add(s, 5, 50, 58)                               # heard, not handed yet: the frontier is here
    assert s._frontier() == 50.0
    assert s._listen_priority(120.0) == FRONTIER    # within a scene (150 s) past the frontier
    assert s._listen_priority(210.0) == BACKGROUND
    s.lookahead = 60.0                              # the horizon comes first
    assert s._listen_priority(59.0) == FRONTIER and s._listen_priority(61.0) == BACKGROUND
    assert FRONTIER < VOICE                         # ...and before the voicer in normal operation


async def test_the_demo_listens_at_the_frontier_first_and_beyond_the_horizon_in_the_background(tmp_path):
    msgs = []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, MockTTS())
    s = Session(backend, DemoResolver(), tmp_path, sj, sb, lookahead=60.0, prepass=180.0)
    await s.open(VIDEO)
    trace = tmp_path / "dQw4w9WgXcQ" / "units.jsonl"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            events = [json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
            if sum(e["event"] == "unit" for e in events) >= 3 and s._asr_cursor >= 179.0:
                break
    finally:
        await s.close()
    asr = [e for e in events if e["event"] == "asr"]
    assert asr[0]["priority"] == FRONTIER                                   # the first scene's
    assert all(e["priority"] == BACKGROUND for e in asr if e["a"] >= 60.0)  # the pre-pass past the horizon: for brief v1
    units = [e for e in events if e["event"] == "unit"]
    assert units[0]["gpu_priority"] == URGENT and units[0]["lead_s"] < sm.URGENT_LEAD


# ---- the throughput governor and batched takes (§5.2, §3.8, §3.13) ---------------------------------------------------
def test_the_governor_asks_one_take_when_behind_and_slow_more_when_comfortable(tmp_path):
    s, _, _ = bare(tmp_path, duration=3600.0)
    long_line, short = add(s, 0, 0, 6), add(s, 1, 7, 9)
    assert s._takes_n(long_line) == 1              # lead 0, r not measured
    rate(s, 1.4)
    assert s._takes_n(long_line) == 2              # behind but gaining on playback
    s.ready.add(0.0, 200.0)
    assert s._takes_n(long_line) == 2 and s._takes_n(short) == 3  # comfortable: three for a short line
    rate(s, 0.8)                                   # slower than playback: the target is what plays to the end
    assert s._lead_target() == pytest.approx(3600 * 0.2 + 30)
    assert s._takes_n(long_line) == 1


async def test_after_a_seek_into_video_not_dubbed_one_take_until_the_lead_is_back(tmp_path):
    s, _, _ = bare(tmp_path, duration=3600.0)
    rate(s, 1.5)
    s.ready.add(0.0, 300.0)
    st = add(s, 0, 1000, 1002)
    s.seek(1000.0)                                 # nothing dubbed there
    assert s._recovering and s._takes_n(st) == 1
    s.ready.add(1000.0, 1100.0)                    # the lead is back past its target (a minute at r >= 1)
    assert s._takes_n(st) == 3 and not s._recovering
    await asyncio.gather(*s._side_tasks)


def voiced_line(s: Session, i: int, start: float, end: float) -> UnitState:
    st = add(s, i, start, end)
    st.line, st.tier, st.telugu = LineResult(i, {"full": LINE}), "full", LINE.spoken
    return st


async def test_the_take_closest_to_the_speech_time_is_voiced_of_a_batch(tmp_path):
    tts = BatchTTS([([1.0, 0.8, 1.3], [False, False, False])])
    s, _, _ = bare(tmp_path, tts)
    st = voiced_line(s, 0, 0.0, 0.0)
    key = VoiceKey("S1", 0.5, 0.5, "r")
    natural = len(tts.synthesize(LINE.spoken, {"f0": 140.0})) / tts.sample_rate
    st.speech_s = 0.8 * natural                    # the 0.8-pace take fits it
    cost = VoiceCost()
    take, dur, pace, failed = await s._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=3)
    assert tts.calls == [3] and dur == pytest.approx(take.seconds) and dur == pytest.approx(0.8 * natural, abs=1e-3)
    assert pace == pytest.approx(natural * (1.0 + 0.8 + 1.3) / 3, abs=1e-3)  # the estimator learns from them all
    assert (cost.takes, cost.takes_n, cost.retakes, cost.failures, failed) == (3, 3, 0, [], None)


async def test_a_cap_hit_or_a_far_too_short_take_is_passed_over_and_a_batch_that_all_failed_gets_one_more_take(tmp_path):
    tts = BatchTTS([([1.0, 0.3], [True, False]),   # a runaway, and one far under the estimate: both failed
                    ([1.05], [False])])            # the retake
    s, _, _ = bare(tmp_path, tts)
    st = voiced_line(s, 0, 0.0, 3.0)
    key = VoiceKey("S1", 0.5, 0.5, "r")
    natural = len(tts.synthesize(LINE.spoken, {"f0": 140.0})) / tts.sample_rate
    s.estimator.estimate = lambda text, k: natural  # the estimate says the line runs its natural length
    cost = VoiceCost()
    take, dur, pace, failed = await s._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=2)
    assert tts.calls == [2, 1] and dur == pytest.approx(1.05 * natural, abs=1e-3) and failed is None
    assert cost.failures == ["cap", "short"] and cost.takes == 3 and cost.takes_n == 2 and cost.retakes == 1
    assert pace == pytest.approx(1.05 * natural, abs=1e-3)  # only the take that passed teaches the estimator

    tts.script = [([1.0], [True]), ([1.0], [True])]  # everything runs away: the least bad is still voiced
    take, _, pace, failed = await s._take(st, LINE, {"f0": 140.0}, key, 5.0, VoiceCost(), n=1)
    assert take.capped and pace is None and failed == "cap"

    tts.calls, tts.script = [], [([1.0, 0.95], [True, False])]  # one failed, one didn't: no retake
    cost = VoiceCost()
    take, dur, _, failed = await s._take(st, LINE, {"f0": 140.0}, key, 5.0, cost, n=2)
    assert tts.calls == [2] and not take.capped and cost.failures == ["cap"] and cost.retakes == 0 and failed is None

    # Every take far under the estimate: the voice is faster than the estimator thinks, and it learns that from them.
    tts.script = [([0.5, 0.4], [False, False]), ([0.45], [False])]
    take, dur, pace, failed = await s._take(st, LINE, {"f0": 140.0}, key, 5.0, VoiceCost(), n=2)
    assert failed == "short" and pace == pytest.approx(natural * (0.5 + 0.4 + 0.45) / 3, abs=1e-3)


async def test_a_tts_that_cant_batch_says_a_line_once(tmp_path):
    s, _, _ = bare(tmp_path, MockTTS())
    st = voiced_line(s, 0, 0.0, 3.0)
    s.estimator.estimate = lambda text, k: 10.0   # the take is far shorter than the estimate
    cost = VoiceCost()
    _, dur, pace, failed = await s._take(st, LINE, {"f0": 140.0}, VoiceKey("S1", 0.5, 0.5, "r"), 5.0, cost, n=3)
    assert cost.takes == cost.takes_n == 1 and cost.failures == ["short"] and failed == "short"  # judged, never retaken
    assert pace == dur and cost.retakes == 0


async def test_retakes_are_budgeted_only_while_the_lead_is_short_and_the_engine_barely_outruns_playback(tmp_path):
    tts = BatchTTS()
    s, _, _ = bare(tmp_path, tts)
    st = voiced_line(s, 0, 0.0, 3.0)
    key, voice = VoiceKey("S1", 0.5, 0.5, "r"), {"f0": 140.0}
    natural = len(tts.synthesize(LINE.spoken, voice)) / tts.sample_rate
    s.estimator.estimate = lambda text, k: natural
    rate(s, 1.0)                                   # no lead, and r under 1.1: the governor's one take a line
    s._retakes, s._voiced = 3, 10                  # the budget (15% of the lines voiced, at least 3) is spent
    tts.script = [([0.3], [False])]
    cost = VoiceCost()
    _, _, _, failed = await s._take(st, LINE, voice, key, 5.0, cost)
    assert tts.calls == [1] and cost.retakes == 0 and failed == "short"  # voiced as it is

    s._voiced = 30                                 # 3 retakes in 30 lines: under the budget
    tts.calls, tts.script = [], [([0.3], [False]), ([1.0], [False])]
    cost = VoiceCost()
    _, _, _, failed = await s._take(st, LINE, voice, key, 5.0, cost)
    assert tts.calls == [1, 1] and cost.retakes == 1 and s._retakes == 4 and failed is None

    s._voiced = 10
    assert not s._retake_ok()
    rate(s, 1.2)                                   # gaining on playback: no budget
    assert s._retake_ok()
    rate(s, 1.0)
    s.ready.add(0.0, 200.0)                        # the lead comfortable: no budget
    assert s._retake_ok()


async def test_a_shorter_wording_is_voiced_instead_only_when_its_take_passes(tmp_path):
    tts = BatchTTS([([1.0], [False]), ([0.3], [False]), ([0.3], [False]),  # line 0: its shorter wording's takes fail
                    ([1.0], [False]), ([1.0], [False])])                    # line 1: its shorter wording's take passes
    s, msgs, _ = bare(tmp_path, tts)
    s.estimator.estimate = lambda text, k: len(tts.synthesize(text, {"f0": 140.0})) / tts.sample_rate
    s._plan = lambda st, said: Plan(st.unit.id, st.unit.start, 1.0, said.total, 0.0, needs_shorter=True)  # runs long
    short = Wording("చిన్న మాట.")
    for i in (0, 1):
        st = voiced_line(s, i, 10.0 + 10 * i, 13.0 + 10 * i)
        st.line = LineResult(i, {"full": LINE, "concise": short})
        await s._dub(st)
    sent = [m["telugu"] for m in msgs if m["type"] == "unit"]
    assert sent == [LINE.spoken, short.spoken]     # a failed take is shorter for the wrong reason
    units = [json.loads(x) for x in (s._dir / "units.jsonl").read_text().splitlines() if '"event": "unit"' in x]
    assert [(u["tier"], u["resynth"], u["retakes"], u["take_failures"]) for u in units] == [
        ("full", False, 1, ["short", "short"]), ("concise", True, 0, [])]


async def test_the_demo_voices_lines_with_more_takes_once_it_is_ahead(tmp_path):
    msgs = []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(_: bytes) -> None: ...

    tts = BatchTTS()
    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts)
    s = Session(backend, DemoResolver(), tmp_path, sj, sb, prepass=60.0)
    await s.open(VIDEO)
    trace = tmp_path / "dQw4w9WgXcQ" / "units.jsonl"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not s.ready.covered(1.0, 170.0):
            await asyncio.sleep(0.05)
    finally:
        await s.close()
    units = [json.loads(x) for x in trace.read_text().splitlines() if '"event": "unit"' in x]
    first = [u["takes_n"] for u in units if u["lead_s"] < 60.0]
    ahead = [u["takes_n"] for u in units if u["lead_s"] >= 60.0]
    assert first and set(first) == {1}             # behind, speed unknown: one take
    assert ahead and set(ahead) <= {2, 3} and all(u["takes"] >= u["takes_n"] for u in units)
    assert all(u["gpu_priority"] == (URGENT if u["lead_s"] < 60.0 else VOICE) for u in units)
    assert [m for m in msgs if m["type"] in ("error", "unit_skipped")] == []


# ---- pacing (§3.12) -------------------------------------------------------------------------------------------------
def test_the_horizon_is_the_lookahead_unless_the_engine_is_slow_the_video_held_or_the_whole_video_asked_for(tmp_path):
    s, _, _ = bare(tmp_path, duration=3600.0, lookahead=600.0)
    s.playhead = 100.0
    assert s._horizon() == 700.0                               # speed not measured yet, playing
    rate(s, 0.8)
    assert s._horizon() == pytest.approx(100.0 + 3500 * 0.2 + 30)  # banks what plays to the end without a stall
    rate(s, 1.1)
    assert s._horizon() == 700.0
    s.set_playhead(100.0, playing=False)                       # held: a barely-fast-enough engine works on
    assert s.held and s._horizon() == 3600.0
    rate(s, 1.3)
    assert s._horizon() == 700.0                               # comfortably fast: the lookahead is enough
    s.set_playhead(100.0)                                      # not said: stays held
    assert s.held
    s.set_playhead(100.0, playing=True)
    s.set_prepare_all(True)
    assert not s.held and s._horizon() == 3600.0               # "prepare the whole video"


async def test_the_target_lead_is_what_plays_to_the_end_uncapped_by_the_lookahead(tmp_path):
    s, msgs, _ = bare(tmp_path, duration=3600.0, lookahead=600.0)
    rate(s, 0.5)
    await s._send_ready(force=True)
    assert msgs[-1]["targetLead"] == 3600 * 0.5 + 30           # 1830 s, past 0.8 x the lookahead: the engine banks it


async def test_a_seek_keeps_claude_calls_up_to_the_horizon(tmp_path):
    s, _, _ = bare(tmp_path, duration=3600.0, lookahead=600.0)
    asked = []
    s.tr = SimpleNamespace(cancel=lambda match: asked.append(match) or 0)
    far = SceneRequest(9, (LineSpec(1, "S1", "A short line here.", 900.0, 905.0, 5.0, 30.0),))
    s.seek(100.0)
    assert asked[-1](far)                                      # past the lookahead: dropped
    s.set_playhead(100.0, playing=False)                       # held, speed unknown: the pipeline works to the end...
    assert s._horizon() == 3600.0
    s.seek(100.0)
    assert asked[-1](far)                                      # ...but a seek frees the slots for the new frontier
    s.set_prepare_all(True)
    s.seek(100.0)
    assert not asked[-1](far)                                  # the whole video is wanted
    await asyncio.gather(*s._side_tasks)


def test_the_pipeline_is_caught_up_once_everything_to_the_horizon_is_heard_and_dubbed(tmp_path):
    s, _, _ = bare(tmp_path, duration=3600.0, lookahead=120.0)
    add(s, 0, 10, 20, voiced=True)
    s._asr_cursor = 100.0
    assert not s._caught_up()                                  # still listening
    s._asr_cursor = 130.0
    assert s._caught_up()
    add(s, 1, 110, 118)
    assert not s._caught_up()                                  # a line to voice


async def test_a_stretch_with_nothing_to_say_heard_past_the_horizon_isnt_counted_in_the_throughput(tmp_path):
    s, msgs, _ = bare(tmp_path, duration=1200.0, lookahead=300.0, prepass=600.0)
    s.throughput.record(0.0, 0.0)
    s.throughput.record(30.0, 36.0)                            # 1.2x over 30 s of work
    s._asr_cursor = 320.0                                      # past the horizon (300): ASR hears on for brief v1
    assert s._caught_up() and s._listen_until() == 600.0
    s.throughput.pause(30.0)                                   # the voicer stopped the clock
    await s._chunk_ready(Chunk(320.0, 380.0))                  # heard, no words: dubbed as soon as heard
    assert s.ready.covered(320.0, 380.0) and s.throughput.total == 36.0 and s.throughput.rate() == pytest.approx(1.2)
    assert msgs[-1]["throughput"] == 1.2
    s.throughput.resume(30.0)
    await s._chunk_ready(Chunk(200.0, 260.0))                  # before the horizon: the dub's pace
    assert s.throughput.total == 96.0


async def test_the_throughput_clock_stops_only_while_the_pipeline_is_caught_up(tmp_path):
    s, _, _ = bare(tmp_path)
    s._prepass_done.set()
    caught = [True]
    s._next_to_voice = lambda: None
    s._caught_up = lambda: caught[0]

    async def idle() -> None:
        await asyncio.sleep(0.01)

    s._idle = idle
    task = asyncio.create_task(s._voicer())
    try:
        await asyncio.sleep(0.05)
        assert s.throughput.idle_since is not None   # everything to the horizon is dubbed: waiting isn't work
        caught[0] = False                             # heard lines wait on Claude: that is the engine's speed
        await asyncio.sleep(0.05)
        assert s.throughput.idle_since is None and s.throughput.idle >= 0.04
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class OneThenTwo:
    """One voice in the first block; a second joins in the next."""

    def diarize_block(self, audio: np.ndarray, offset: float) -> DiarBlock:
        end = offset + len(audio) / sm.SR_ANALYSIS
        emb = np.eye(8, dtype=np.float32)
        turns = [SpeakerTurn("A", offset, end)] if offset == 0 else [SpeakerTurn("A", offset, offset + 40.0),
                                                                     SpeakerTurn("B", offset + 40.0, end)]
        return DiarBlock(offset, end, turns, list(turns), {t.speaker: emb["AB".index(t.speaker)] for t in turns})


async def test_speakers_found_later_are_cloned_off_the_diarizers_loop(tmp_path):
    """A later speaker's clone is background GPU work that may wait behind the voicer: the diarizer goes on to its next
    block meanwhile (frontier ASR may need it), and a speaker it finds meanwhile is cloned after (§5.3)."""
    s, msgs, _ = bare(tmp_path, duration=400.0, prepass=60.0)
    s.b.diarizer = OneThenTwo()
    s.audio = np.zeros(int(400 * sm.SR_ANALYSIS), np.float32)
    s._prepass_heard.set()
    s._prepass_done.set()
    gate, cloned = asyncio.Event(), []

    async def clone(sid, cost=None, priority=BACKGROUND) -> None:
        cloned.append((sid, priority))
        s.voices[sid].status = "cloning"
        await gate.wait()                          # waiting for the GPU
        v = s.voices[sid]
        v.voice, v.kind, v.status = {"f0": 150.0}, sm.VoiceKind.CLONED, "cloned"

    s._clone = clone
    task = asyncio.create_task(s._diarizer())
    try:
        deadline = time.monotonic() + 10
        while s._diar_cursor < 360.0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert s._diar_cursor == 360.0 and len(cloned) == 1  # two blocks diarized while the first clone waits
        assert len(s.registry.speakers) == 2                 # the second found meanwhile
        gate.set()
        while (len(cloned) < len(s.registry.speakers) or not s._clone_task.done()) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert sorted(sid for sid, _ in cloned) == sorted(s.registry.speakers) and len(s.registry.speakers) == 2
        assert all(p == BACKGROUND for _, p in cloned)
        assert [x["status"] for x in [m for m in msgs if m["type"] == "speakers"][-1]["speakers"]] == ["cloned"] * 2
    finally:
        task.cancel()
        await asyncio.gather(task, *s._side_tasks, return_exceptions=True)


# ---- dub PCM near the playhead only (§3.12, §5.1) ------------------------------------------------------------------
def plan_for(st: UnitState) -> Plan:
    return Plan(st.unit.id, st.unit.start, 1.0, st.unit.end - st.unit.start, 0.0)


async def test_dub_pcm_is_kept_and_sent_only_near_the_playhead_and_reloaded_from_disk_when_asked(tmp_path):
    s, msgs, frames = bare(tmp_path, duration=3600.0, lookahead=120.0)
    near, far = voiced_line(s, 0, 10.0, 14.0), voiced_line(s, 1, 400.0, 404.0)
    for st in (near, far):
        st.plan, st.voiced = plan_for(st), True
        await s._ship(st, sm.VoiceKind.CLONED, np.full(9600, 0.1, np.float32), 4.0)
    units = {m["id"]: m for m in msgs if m["type"] == "unit"}
    assert units[0]["audio"] is True and units[1]["audio"] is False   # the far one's audio is held back...
    assert frames == [0] and set(s.raw) == {0}                        # ...never sent, not kept in memory
    assert all((s._dir / "pcm" / f"{i}.npy").is_file() for i in (0, 1))  # both on disk

    s.set_playhead(330.0)                          # the window is now [270, 450]
    assert set(s.raw) == set()                     # the line behind it goes from memory
    s.ask_audio([1, 0, 7])                         # the UI nears the far line (7: unknown)
    await asyncio.gather(*s._side_tasks)
    assert frames == [0, 0, 1] and set(s.raw) == {1}  # both sent from disk; only the one in the window kept
    assert np.allclose(np.load(s._dir / "pcm" / "1.npy").astype(np.float32), s.raw[1][0])


async def test_a_line_being_saved_stays_in_memory_and_its_file_is_never_read_half_written(tmp_path, monkeypatch):
    s, msgs, frames = bare(tmp_path, duration=3600.0, lookahead=120.0)
    st = voiced_line(s, 0, 400.0, 404.0)
    st.plan, st.voiced = plan_for(st), True
    started, release = asyncio.Event(), __import__("threading").Event()
    loop, save = asyncio.get_running_loop(), sm._save_pcm

    def slow_save(path, audio) -> None:
        loop.call_soon_threadsafe(started.set)
        release.wait(5)
        save(path, audio)

    monkeypatch.setattr(sm, "_save_pcm", slow_save)
    ship = asyncio.create_task(s._ship(st, sm.VoiceKind.CLONED, np.full(9600, 0.1, np.float32), 4.0))
    await started.wait()                           # the unit is sent; its file is being written
    assert msgs[-1]["type"] == "unit" and msgs[-1]["audio"] is False
    s.set_playhead(100.0)                          # the playhead moves meanwhile, the line still outside the window:
    assert 0 in s.raw and not (s._dir / "pcm" / "0.npy").exists()  # evicting it now would lose it
    s.ask_audio([0])                               # the UI asks for it at once
    await asyncio.gather(*s._side_tasks)
    assert frames == [0]                           # served from memory
    release.set()
    await ship
    assert not s.raw and not list((s._dir / "pcm").glob("*.tmp"))  # evicted once written, and written whole
    assert np.load(s._dir / "pcm" / "0.npy").shape == (9600,)

    s.set_playhead(3000.0)
    (s._dir / "pcm" / "5.npy").write_bytes(b"not an array")  # an earlier line's file that can't be read...
    bad = voiced_line(s, 5, 300.0, 302.0)
    bad.plan, bad.voiced = plan_for(bad), True
    s.ask_audio([5, 0])
    await asyncio.gather(*s._side_tasks)
    assert frames == [0, 0]                        # ...doesn't stop the later one being sent


async def test_after_a_seek_lines_reloaded_near_the_new_playhead_stay_in_memory(tmp_path):
    s, _, frames = bare(tmp_path, duration=3600.0, lookahead=120.0)
    st = voiced_line(s, 0, 400.0, 404.0)
    st.plan, st.voiced = plan_for(st), True
    await s._ship(st, sm.VoiceKind.CLONED, np.full(9600, 0.1, np.float32), 4.0)
    assert frames == [] and not s.raw
    s.playhead = 390.0
    await s._resend_near(390.0)
    assert frames == [0] and set(s.raw) == {0}     # a speed change can re-render it now


async def test_whole_video_mode_over_the_websocket_holds_far_audio_back_until_asked(tmp_path):
    eng = Engine("mock", tmp_path / "models", tmp_path / "cache", None, "tok", demo=True)
    async with serve(eng.handler, "127.0.0.1", 0, process_request=eng.process_request) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws?token=tok"
        async with connect(url, max_size=2**24) as ws:
            await ws.recv()  # hello
            await ws.send(json.dumps({"type": "open", "url": VIDEO, "lookahead": 60}))
            await ws.send(json.dumps({"type": "prepare", "whole": True}))
            await ws.send(json.dumps({"type": "playhead", "time": 0.0, "playing": False}))
            units, audio = {}, []

            async def pump(done) -> None:
                while not done():
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        audio.append(struct.unpack("<IIII", m[:16])[0])
                    else:
                        msg = json.loads(m)
                        assert msg["type"] != "error", msg
                        if msg["type"] == "unit":
                            units[msg["id"]] = msg

            await asyncio.wait_for(pump(lambda: any(u["start"] > 150.0 for u in units.values())), 20)
            far = sorted(i for i, u in units.items() if u["start"] > 60.0)
            assert far and all(units[i]["audio"] is False for i in far) and not set(far) & set(audio)
            assert all(units[i]["audio"] is True for i in units if units[i]["start"] < 50.0)
            await ws.send(json.dumps({"type": "audio", "ids": far[:3]}))
            await asyncio.wait_for(pump(lambda: set(far[:3]) <= set(audio)), 10)
