"""The session's translator stage and the voicer's side of it (ARCHITECTURE §4.1-4.5, §4.9, §3.12, §3.13): scene cuts,
`want` tiers from the k prior, the band rule, the Claude CLI off the voicing path, seeks, brief swaps, rephrases and the
TTS script switch. No Claude: the demo backend's mock translator, or a fake `claude` executable. All English and Telugu
here is original test text (the demo lines included).
"""

from __future__ import annotations

import asyncio
import json
import stat
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maata_engine import session as sm
from maata_engine.backends.base import Backend, Coverage, LineResult, LineSpec, SceneRequest, SceneResult, Wording
from maata_engine.backends.claude_translator import ClaudeTranslator
from maata_engine.backends.mock import MockClaude, MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.resolve import DemoResolver, ResolvedVideo, metadata
from maata_engine.session import Session, UnitState, VoiceCost, scene_cut
from maata_engine.timing.duration import DEFAULT_OVERHEAD, DEFAULT_RATE
from maata_engine.timing.planner import Plan
from maata_engine.types import SourceUnit, TimedWord

VIDEO = "https://youtu.be/dQw4w9WgXcQ"

# A fake `claude`: answers scene, fit and rephrase calls from the message (Telugu script, one made-up word per id), brief
# calls with a small brief and review calls with every line complete, after FAKE_DELAY s; FAKE_MODE=not_signed_in fails
# every call as the CLI does. Each call leaves {call, scene, ids, t0, t1} in FAKE_LOG.
FAKE = r"""#!@PYTHON@
import json, os, sys, time
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print("2.1.281 (Claude Code)")
    sys.exit(0)
if argv[:2] == ["auth", "status"]:
    time.sleep(float(os.environ.get("FAKE_AUTH_DELAY", "0")))
    print(json.dumps({"loggedIn": os.environ.get("FAKE_SIGNED_IN", "1") == "1", "authMethod": "claude.ai"}))
    sys.exit(0)
msg = json.loads(sys.stdin.read())
t0 = time.time()
def emit(ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)
emit({"type": "system", "subtype": "init"})
if os.environ.get("FAKE_MODE") == "not_signed_in":
    emit({"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in · Please run /login"})
    sys.exit(1)
time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
if msg.get("call") == "brief":
    data = {"topic": "a demo", "register": "casual", "speakers": [], "glossary": []}
elif msg.get("call") == "review":
    data = {"lines": [{"id": x["id"], "class": "C", "missing": [], "added": [], "error": "none"} for x in msg["lines"]]}
else:
    def w(words):
        return {"spoken": " ".join(words), "english": []}
    lines = []
    for line in msg["lines"]:
        words = ["ఇది", "".join("మకగచజటడతపబ"[int(d)] for d in str(line["id"])) + "ము", "అని", "చెప్పారు."]
        out = {"id": line["id"], "full": w(words), "delivery": {"emotion": "neutral", "energy": "mid"}}
        if "concise" in line["want"]:
            out["concise"] = w(words[1:])
        if "very_concise" in line["want"]:
            out["very_concise"] = w(words[1:2])
        if "fuller" in line["want"]:
            out["fuller"] = w(["అంటే"] + words)
        lines.append(out)
    data = {"lines": lines}
emit({"type": "result", "subtype": "success", "is_error": False, "result": "", "structured_output": data,
      "usage": {"input_tokens": 1, "output_tokens": 1}, "modelUsage": {}})
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"call": msg.get("call"), "scene": msg.get("scene"),
                        "ids": [x["id"] for x in msg.get("lines", [])], "t0": t0, "t1": time.time()}) + "\n")
"""


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.replace("@PYTHON@", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.setenv("FAKE_DELAY", "0")

    def factory(cache_dir, video_id, brief, *, style="colloquial", trace=None):
        tr = ClaudeTranslator(cache_dir, video_id, brief, style=style, trace=trace, binary=str(exe))
        tr.cli.grace, tr.cli.backoff = 1.0, 0.01
        return tr

    def calls() -> list[dict]:
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

    return SimpleNamespace(exe=exe, factory=factory, calls=calls, env=monkeypatch)


class SlowTTS(MockTTS):
    """The mock voice, taking `pause` s of wall time per take and saying lines at `rate` aksharas (or syllables of
    Latin-script English) per second."""

    def __init__(self, pause: float = 0.0, rate: float = 6.0) -> None:
        self.pause, self.rate, self.texts = pause, rate, []

    def synthesize(self, text, voice, language="te", max_seconds=None):
        self.texts.append(text)
        time.sleep(self.pause)
        seconds = 0.15 + sm.mixed_units(text) / self.rate
        n = int(min(seconds, max_seconds or seconds) * self.sample_rate)
        return np.full(n, 0.01, np.float32)


class LongDemo(DemoResolver):
    """The demo as a longer video: the mock transcriber keeps talking."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def resolve(self, ref, cache_dir) -> ResolvedVideo:
        return replace(super().resolve(ref, cache_dir), duration=self.seconds)


def make_session(tmp_path, translator=MockSceneTranslator, tts=None, resolver=None, **kw) -> tuple[Session, list[dict]]:
    msgs: list[dict] = []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), translator, tts or MockTTS())
    return Session(backend, resolver or DemoResolver(), tmp_path / "cache", sj, sb, **{"prepass": 30.0, **kw}), msgs


async def until(pred, timeout: float = 20.0, step: float = 0.05) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(step)


def trace(tmp_path) -> list[dict]:
    path = tmp_path / "cache" / "dQw4w9WgXcQ" / "units.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ---- scene cuts (§4.1) ---------------------------------------------------------------------------------------------
def unit(i: int, start: float, end: float, speaker: str = "S1", text: str = "A line.") -> SourceUnit:
    return SourceUnit(i, speaker, start, end, text)


def lines_every(seconds: float, n: int, length: float = 3.0, text: str = "A line.", speaker: str = "S1"):
    return [unit(i, i * seconds, i * seconds + length, speaker, text) for i in range(n)]


def test_the_first_scene_ends_at_the_first_sentence_end_at_or_after_20_s():
    run = lines_every(4.0, 12)                                      # ends at 3, 7, 11, 15, 19, 23, ...
    assert scene_cut(run, 0, final=False) == 6                        # 20.0-23.0: the first to end past 20 s
    run[5].text = "and then it goes on"                                # no sentence end there: the next one
    assert scene_cut(run, 0, final=False) == 7


def test_the_first_scene_never_runs_past_30_s():
    run = [unit(i, 4.0 * i, 4.0 * i + 3.0, text="still going,") for i in range(12)]
    assert scene_cut(run, 0, final=False) == 7                        # the line ending at 31 s is left for the next
    assert scene_cut([unit(0, 0.0, 34.0)], 0, final=False) == 1       # one line longer than that still goes


def test_a_first_scene_waits_for_20_s_unless_nothing_more_is_coming():
    run = lines_every(4.0, 3)                                         # 11 s heard
    assert scene_cut(run, 0, final=False) is None
    assert scene_cut(run, 0, final=True) == 3


def test_the_second_scene_is_about_a_minute_cut_at_a_turn_or_pause():
    run = lines_every(4.0, 30, length=3.5)                            # one speaker, half-second gaps
    assert scene_cut(run, 1, final=False) == 15                       # the limit: ends at 59.5 s
    run[11] = unit(11, 44.0, 47.5, speaker="S2")                      # turns after 43.5 s and after 47.5 s
    assert scene_cut(run, 1, final=False) == 12                       # the later one


def test_later_scenes_hold_at_most_150_s_and_30_lines():
    run = lines_every(4.0, 60, length=3.5)
    assert scene_cut(run, 2, final=False) == 30                       # 30 lines come first (they end at 119.5 s)
    sparse = lines_every(8.0, 40)                                     # 5 s gaps between lines: every boundary a pause
    assert scene_cut(sparse, 2, final=False) == 19                    # ends at 147 s, the last pause under 150 s
    turns = [unit(i, 10.0 * i, 10.0 * i + 9.5, "S1" if i < 11 else "S2") for i in range(20)]
    assert scene_cut(turns, 2, final=False) == 11                     # the speaker turn at 110 s, not the limit
    early = [unit(i, 10.0 * i, 10.0 * i + 9.5, "S1" if i < 3 else "S2") for i in range(20)]
    assert scene_cut(early, 2, final=False) == 15                     # a turn in the first half doesn't count


def test_a_later_scene_waits_until_its_limit_is_heard():
    run = lines_every(4.0, 20)                                        # 79 s heard
    assert scene_cut(run, 2, final=False) is None
    assert scene_cut(run, 2, final=True) == 20


# ---- want tiers and the k prior (§4.3) --------------------------------------------------------------------------------
def spec_for(s: Session, text: str, span: float, speech: float | None = None, speaker: str = "S1") -> LineSpec:
    u = SourceUnit(len(s.units), speaker, 10.0, 10.0 + span, text)
    st = UnitState(u, None, sm.Chunk(0.0, 60.0), speech_s=span if speech is None else speech)
    s.units[u.id] = st
    return s._spec(st)


def test_want_tiers_come_from_the_predicted_length(tmp_path):
    s, _ = make_session(tmp_path)
    en = "Please remember to bring the tickets."
    pred = 1.4 * sm.mixed_units(en)                                   # the prior: 1.4 aksharas per English syllable
    # the span whose target is pred / ratio: that many aksharas at the prior's pace, after its overhead
    span = lambda ratio: DEFAULT_OVERHEAD + pred / (ratio * DEFAULT_RATE)  # noqa: E731
    assert s._k("S1") == sm.K_PRIOR == 1.4
    long = spec_for(s, en, span=span(1.2))                            # 1.2 x the target
    assert long.want == ("full", "concise", "very_concise") and long.target_aksharas == round(pred / 1.2, 1)
    assert s.units[long.id].pred_full == pytest.approx(pred)
    assert spec_for(s, en, span=span(1.05)).want == ("full",)
    assert spec_for(s, en, span=span(0.9)).want == ("full",)
    assert spec_for(s, en, span=span(0.7)).want == ("full", "fuller")  # under 0.85 x a speech-dense slot
    # 0.67 x the target, but a quarter of the slot is pause: short lines are not padded to fill pauses
    assert spec_for(s, en, span=span(0.5), speech=0.75 * span(0.5)).want == ("full",)


def test_k_becomes_each_speakers_running_median(tmp_path):
    s, _ = make_session(tmp_path)
    en = "Please remember to bring the tickets."
    syllables = sm.mixed_units(en)
    for ratio in (1.6, 2.0, 1.8):
        st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, en), None, sm.Chunk(0.0, 3.0))
        st.line = LineResult(0, {"full": Wording("ప" * round(syllables * ratio))})
        s._learn_k(st)
        assert s._k("S1") == (sm.K_PRIOR if ratio != 1.8 else pytest.approx(round(syllables * 1.8) / syllables))
    assert s._k("S2") == sm.K_PRIOR
    tiny = UnitState(SourceUnit(0, "S2", 0.0, 1.0, "Yes."), None, sm.Chunk(0.0, 1.0))
    tiny.line = LineResult(0, {"full": Wording("అవును.")})
    s._learn_k(tiny)
    assert "S2" not in s._ratios  # too short to say anything about pace


# ---- the band rule (§4.5) ---------------------------------------------------------------------------------------------
def band_case(tmp_path, tiers: dict[str, float], speech: float = 4.0, tts=None):
    """A line whose tiers have the given predicted durations (s), at the default pace (DEFAULT_RATE aksharas/s)."""
    s, _ = make_session(tmp_path, tts=tts)
    u = SourceUnit(0, "S1", 100.0, 100.0 + speech, "Some words here.")
    st = UnitState(u, 100.0 + speech + 2.0, sm.Chunk(100.0, 110.0), speech_s=speech)
    s.units[0] = st
    st.line = LineResult(0, {t: Wording("ప" * round((d - 0.15) * DEFAULT_RATE)) for t, d in tiers.items()})
    return s, st


def test_the_most_complete_wording_inside_the_band_wins(tmp_path):
    s, st = band_case(tmp_path, {"fuller": 4.2, "full": 3.8, "concise": 3.0})
    assert s._choose(st) and st.tier == "fuller" and st.telugu == st.line.tiers["fuller"].spoken


def test_above_the_band_the_closest_is_taken_when_the_planner_absorbs_it(tmp_path):
    s, st = band_case(tmp_path, {"full": 4.8, "concise": 2.6})  # 4.8 s over a 4 s speech time, 2 s of silence after
    assert s._choose(st) and st.tier == "full"


def test_otherwise_the_closest_from_below(tmp_path):
    s, st = band_case(tmp_path, {"full": 12.0, "concise": 9.0, "very_concise": 2.9})  # 9 s won't fit even at 1.2x
    assert not s._choose(st) and st.tier == "very_concise"
    s, st = band_case(tmp_path, {"full": 12.0, "concise": 9.0})  # nothing below: the closest from above
    assert not s._choose(st) and st.tier == "concise"


def test_a_line_short_of_its_slot_asks_a_fit_for_fuller_and_one_too_long_for_shorter_tiers(tmp_path):
    s, st = band_case(tmp_path, {"full": 2.0})
    assert not s._choose(st) and s._fit_spec(st).want == ("fuller",)
    s, st = band_case(tmp_path, {"full": 12.0, "concise": 9.0})
    assert not s._choose(st)
    fit = s._fit_spec(st)
    assert fit.want == ("very_concise",) and fit.current == st.line.tiers["concise"].spoken and fit.overflow > 0


# ---- code-mixing is monitored, never enforced (§4.5) -------------------------------------------------------------------
@pytest.mark.parametrize("english,warns", [(0, True), (1, False), (3, False), (5, True)])
async def test_a_scenes_code_mixing_is_logged_and_warned_about_only_well_outside_20_to_40(tmp_path, caplog, english,
                                                                                          warns):
    """Ten words, `english` of them English: a code-mixing index of 0, 10, 30 or 50 %. It tops out at 50 (as many English
    words as Telugu), so the upper warning must sit below that to ever fire."""
    s, _ = make_session(tmp_path)
    s._dir = tmp_path / "cache" / "dQw4w9WgXcQ"
    s._dir.mkdir(parents=True)
    spoken = "ఒకటి రెండు మూడు నాలుగు ఐదు ఆరు ఏడు ఎనిమిది తొమ్మిది పది."
    line = LineResult(0, {"full": Wording(spoken, tuple((i, "word") for i in range(english)))})
    st = UnitState(SourceUnit(0, "S1", 10.0, 16.0, "One two three four five six seven eight nine ten."), 17.0,
                   sm.Chunk(0.0, 20.0), speech_s=6.0, scene=1)
    s.units[0] = st

    class Answer:
        prompt_hash, brief = "x", SimpleNamespace(version=0)

        def submit(self, req):
            async def reply() -> SceneResult:
                return SceneResult(req.scene, req.call, lines={0: line})
            return asyncio.ensure_future(reply())

        async def review(self, req, lines, chosen):
            return SceneResult(req.scene, "review", lines=dict(lines))

    s.tr = Answer()
    with caplog.at_level("WARNING", logger="maata.session"):
        await s._scene(SceneRequest(1, (s._spec(st),)), 0)
    (scene,) = [e for e in trace(tmp_path) if e["event"] == "scene"]
    assert scene["cmi"] == 10.0 * min(english, 10 - english)                   # logged for every scene
    assert ("code-mixing index" in caplog.text) == warns and st.line is line   # a warning at most: the line is used


# ---- the pipeline, end to end on the mock backend -----------------------------------------------------------------------
async def test_the_demo_is_translated_in_scenes_and_voiced(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "SCENES_IN_FLIGHT", 1)  # one at a time, so every scene has the one before translated
    s, msgs = make_session(tmp_path)
    await s.open(VIDEO)
    try:
        await until(lambda: s.ready.covered(1.0, 170.0))
    finally:
        await s.close()
    scenes = [e for e in trace(tmp_path) if e["event"] == "scene"]
    assert [e["nth"] for e in scenes][:3] == [0, 1, 2]
    first = scenes[0]
    assert 20.0 <= first["b"] - first["a"] <= 30.0
    assert all(e["b"] - e["a"] <= sm.SCENE_MAX_S for e in scenes)
    units = [e for e in trace(tmp_path) if e["event"] == "unit"]
    assert units and all(u["tier"] and u["model"] == "mock" and not u["provisional"] for u in units)
    assert [m for m in msgs if m["type"] in ("error", "claude_error", "unit_skipped")] == []
    assert [m["type"] for m in msgs].count("claude_ok") == 1  # the first call through clears any Claude banner, once
    # scene 1 has no context; later ones get the last 3 lines of the one before, with the Telugu chosen for them
    assert scenes[0]["context_te"] == 0 and all(e["context_te"] == 3 for e in scenes[1:])


async def test_every_claude_call_slow_the_voicer_keeps_voicing_and_never_awaits_claude(tmp_path, fake_cli, monkeypatch):
    """ARCHITECTURE §4.9: with every call delayed, lines already translated are voiced while the next scenes wait on
    Claude, and no call on the translator is ever made from the voicer's task."""
    fake_cli.env.setenv("FAKE_DELAY", "2.0")
    monkeypatch.setattr(sm, "SECOND_SCENE", 30.0)  # short scenes: about seven for the three-minute demo
    monkeypatch.setattr(sm, "SCENE_MAX_S", 30.0)
    callers: list[tuple[str, asyncio.Task | None]] = []

    def factory(*a, **k):
        tr = fake_cli.factory(*a, **k)
        for name in ("translate", "submit", "make_brief", "review", "use_brief", "cancel"):
            def spy(*args, _f=getattr(tr, name), _n=name, **kw):
                callers.append((_n, asyncio.current_task()))
                return _f(*args, **kw)
            setattr(tr, name, spy)
        return tr

    s, msgs = make_session(tmp_path, factory, SlowTTS(pause=0.12))
    await s.open(VIDEO)
    try:
        await until(lambda: len([c for c in fake_cli.calls() if c["call"] == "scene"]) >= 6, timeout=40)
        await until(lambda: sum(1 for m in msgs if m["type"] == "unit") >= 30, timeout=40)
    finally:
        await s.close()
    voicer = s._tasks[3]
    assert callers and voicer not in {task for _, task in callers}
    units = [e for e in trace(tmp_path) if e["event"] == "unit"]
    calls = [c for c in fake_cli.calls() if c["call"] == "scene"]
    # lines of earlier scenes were voiced while a later scene's call was still waiting on Claude
    busy = [u for u in units if any(c["t0"] < u["t"] < c["t1"] and u["id"] not in c["ids"] for c in calls)]
    assert len(busy) >= 5
    assert max(u["lock_wait_s"] for u in units) < 1.0  # nobody holds the GPU lock while Claude thinks
    overlap = max(sum(1 for c in calls if c["t0"] <= x["t0"] < c["t1"]) for x in calls)
    assert 2 <= overlap <= sm.SCENES_IN_FLIGHT  # scene calls run side by side, up to three
    assert [m for m in msgs if m["type"] in ("error", "claude_error")] == []


async def test_a_seek_cancels_calls_outside_the_new_window_and_starts_a_short_scene(tmp_path):
    slow: list[MockClaude] = []

    def factory(*a, **k):
        tr = MockSceneTranslator(*a, delay=30.0, **k)
        slow.append(tr.cli)
        return tr

    s, _ = make_session(tmp_path, factory, lookahead=60.0)
    await s.open(VIDEO)
    try:
        await until(lambda: slow and len([c for c in slow[0].calls if c["call"] == "scene"]) >= 2)
        before = [c for c in slow[0].calls if c["call"] == "scene"]
        assert all(line["start"] < 60.0 for c in before for line in c["message"]["lines"])
        s.seek(150.0)
        old = {c["message"]["scene"] for c in before}
        await until(lambda: old <= {e["scene"] for e in trace(tmp_path) if e.get("dropped")})
        await until(lambda: any(line["start"] >= 150.0 for c in slow[0].calls if c["call"] == "scene"
                                for line in c["message"]["lines"]))
    finally:
        await s.close()
    after = [c for c in slow[0].calls if c["call"] == "scene" and c["message"]["lines"][0]["start"] >= 150.0]
    lines = after[0]["message"]["lines"]
    assert lines[0]["start"] < 152.0 and 20.0 <= lines[-1]["end"] - lines[0]["start"] <= 30.0  # a short first scene
    context = after[0]["message"]["context_before"]
    assert all("te" not in x for x in context)  # heard before the seek but never translated: English only (§3.12)


async def test_a_seek_keeps_calls_inside_the_new_window(tmp_path):
    s, _ = make_session(tmp_path, lambda *a, **k: MockSceneTranslator(*a, delay=30.0, **k), lookahead=60.0)
    await s.open(VIDEO)
    try:
        await until(lambda: s.tr is not None and len(s._scene_tasks) >= 2)
        s.seek(10.0)  # inside the first scenes
        await asyncio.sleep(0.3)
        assert not any(e.get("dropped") for e in trace(tmp_path)) and len(s._scene_tasks) >= 2
    finally:
        await s.close()


async def test_brief_v1_is_swapped_in_at_a_scene_boundary(tmp_path, monkeypatch):
    """The brief call ends while a scene is in flight: that scene keeps brief v0 for all its calls, and v1 is swapped in
    just before the next scene is submitted."""
    monkeypatch.setattr(sm, "SCENES_IN_FLIGHT", 1)
    log: list[tuple] = []

    class Timed(MockClaude):
        def ask(self, system, prompt, schema=None, call="text", **kw):
            time.sleep(1.0 if call == "brief" else 0.6)
            reply = super().ask(system, prompt, schema, call, **kw)
            log.append(("done", call, json.loads(prompt).get("scene")))
            return reply

    def factory(cache_dir, video_id, brief, **kw):
        tr = ClaudeTranslator(cache_dir, video_id, brief, cli=Timed(), **kw)
        use, submit = tr.use_brief, tr.submit
        tr.use_brief = lambda b: (log.append(("use", b.version)), use(b))[1]
        tr.submit = lambda r: (log.append(("submit", r.call, r.scene, tr.brief.version)), submit(r))[1]
        return tr

    s, _ = make_session(tmp_path, factory)
    await s.open(VIDEO)
    try:
        await until(lambda: any(e[0] == "submit" and e[3] == 1 for e in log), timeout=20)
        swapped = next(e[2] for e in log if e[0] == "submit" and e[3] == 1)
        await until(lambda: ("done", "scene", swapped) in log)
    finally:
        await s.close()
    k = log.index(("use", 1))
    assert log[k + 1][:2] == ("submit", "scene") and log[k + 1][3] == 1  # the swap, then the next scene
    brief_done = next(i for i, e in enumerate(log) if e[:2] == ("done", "brief"))
    in_flight = [e[2] for e in log[:brief_done] if e[:2] == ("submit", "scene")][-1]
    # it was in flight (its call, or its lines' review) when the brief came: the swap waited for it
    ends = [i for i, e in enumerate(log) if e[0] == "done" and e[1] in ("scene", "review") and e[2] == in_flight]
    assert brief_done < ends[-1] < k
    systems = {c["message"]["scene"]: c["system"] for c in s.tr.cli.calls if c["call"] == "scene"}
    assert "VIDEO BRIEF (version 0)" in systems[in_flight]
    assert "VIDEO BRIEF (version 1)" in systems[log[k + 1][2]]


@pytest.mark.parametrize("script", ["telugu", "latin"])
async def test_the_tts_reads_telugu_script_or_for_the_ab_the_latin_rebuild(tmp_path, script):
    tts = SlowTTS()
    s, msgs = make_session(tmp_path, tts=tts, tts_script=script)
    measured: list[str] = []
    observe = s.estimator.observe
    s.estimator.observe = lambda text, key, seconds: (measured.append(text), observe(text, key, seconds))
    await s.open(VIDEO)
    try:
        await until(lambda: len(tts.texts) >= 6)
    finally:
        await s.close()
    # the voice's pace is learned on the Telugu script (what lengths are counted on), whatever the TTS read
    assert measured and not any(ch.isascii() and ch.isalpha() for t in measured for ch in t)
    welcome = next(t for t in tts.texts if "స్వాగతం" in t)
    if script == "telugu":
        assert "ఛానల్" in welcome and not any(ch.isascii() and ch.isalpha() for t in tts.texts for ch in t)
    else:
        assert "channel కి" in welcome and "ఛానల్" not in welcome
    sent = next(m for m in msgs if m["type"] == "unit" and "స్వాగతం" in m["telugu"])
    assert sent["telugu"] in tts.texts and ("channel" in sent["telugu"]) == (script == "latin")  # what was said


# ---- a line that still runs long: a provisional take, then a rephrase (§3.13) ------------------------------------------
async def test_a_long_line_ships_provisional_and_a_rephrase_in_time_replaces_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "RESYNTH_SHARE", 10.0)  # no budget in the way
    s, msgs = make_session(tmp_path, tts=SlowTTS(rate=2.0))  # speaks slowly: most takes run long
    await s.open(VIDEO)
    try:
        await until(lambda: any(e["event"] == "rephrase" and e["outcome"] == "replaced" for e in trace(tmp_path)),
                    timeout=30)
    finally:
        await s.close()
    replaced = next(e for e in trace(tmp_path) if e["event"] == "rephrase" and e["outcome"] == "replaced")
    sent = [m for m in msgs if m["type"] == "unit" and m["id"] == replaced["id"]]
    assert [m["provisional"] for m in sent] == [True, False]
    assert sent[1]["start"] == sent[0]["start"] and sent[1]["audioRate"] == sent[0]["audioRate"]  # the same span
    assert sm.count_units(sent[1]["telugu"]) < sm.count_units(sent[0]["telugu"])
    unit_ev = next(e for e in trace(tmp_path) if e["event"] == "unit" and e["id"] == replaced["id"])
    assert unit_ev["provisional"] and unit_ev["needs_shorter"]
    # the replacement's own timing, as maata-bench counts it: said whole, its freeze no longer than the provisional one's
    assert replaced["said"] == "whole" and 0.0 <= replaced["freeze"] <= unit_ev["freeze"] and replaced["voiced"]
    # lines within a minute of the playhead are shipped as they are: no rephrase could come back in time
    early = [e for e in trace(tmp_path) if e["event"] == "unit" and e["start"] < sm.REPHRASE_LEAD]
    assert early and not any(e["provisional"] for e in early)


def voiced_line(s: Session, start: float = 200.0) -> UnitState:
    u = SourceUnit(7, "S1", start, start + 3.0, "Please remember to bring the tickets.")
    st = UnitState(u, None, sm.Chunk(start, start + 3.0), speech_s=3.0, scene=4, voiced=True, tier="full")
    st.line = LineResult(7, {"full": Wording("దయచేసి టికెట్లు తీసుకురావడం మర్చిపోకండి మరి.")})
    st.plan, st.take_s, st.provisional = Plan(7, start, 1.2, 5.0, 0.0, needs_shorter=True, excess=1.0), 6.0, True
    s.units[7] = st
    return st


async def test_a_rephrase_is_dropped_when_it_comes_back_within_a_minute_of_the_playhead(tmp_path):
    s, _ = make_session(tmp_path)
    s.tr = MockSceneTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")))
    st = voiced_line(s)
    spec = LineSpec(7, "S1", st.unit.text, 200.0, 203.0, 3.0, 16.5, ("full", "concise", "very_concise"),
                    current=st.line.full.spoken)
    req = SceneRequest(4, (spec,), "rephrase", deadline=time.time() + 60)
    s.playhead = 150.0  # 50 s left
    await s._rephrase(st, req)
    assert st.replacement is None
    s.playhead = 100.0
    await s._rephrase(st, req)
    assert st.replacement is not None and "concise" in st.replacement.tiers


async def test_a_replacement_that_doesnt_fit_the_planned_span_leaves_the_provisional_take(tmp_path):
    s, msgs = make_session(tmp_path, tts=SlowTTS(rate=2.0))
    st = voiced_line(s)
    longer = LineResult(7, {"full": Wording(st.line.full.spoken + " అంతే కదా మరి సరే.")})
    st.replacement, st.take_s = longer, 1.0
    await s._replace(st)
    assert st.provisional and st.replacement is None and not [m for m in msgs if m["type"] == "unit"]
    st.replacement, st.take_s = LineResult(7, {"full": Wording("టికెట్లు మర్చిపోకండి."),
                                               "concise": Wording("టికెట్లు.")}), 6.0
    st.plan = replace(st.plan, freeze=0.5, freeze_at=203.0)  # the provisional take needed a freeze
    await s._replace(st)
    (sent,) = [m for m in msgs if m["type"] == "unit"]
    assert not st.provisional and sent["telugu"] == "టికెట్లు మర్చిపోకండి." and not sent["provisional"]
    assert sent["audioWall"] == pytest.approx(st.take_s / 1.2) and sent["freeze"] == 0.0 and sent["edits"] == []


def test_a_shorter_take_in_the_same_place_needs_less_freeze(tmp_path):
    """The replacement plays from the same start at the same rate; the time it saves comes off the planned freeze, and
    it never ends later than the provisional take did (the lines after it are placed already)."""
    s, _ = make_session(tmp_path)
    plan = Plan(7, 200.0, 1.2, 5.0, 0.0, freeze=0.5, needs_shorter=True, freeze_at=203.0, excess=1.0)
    gone = s._shortened(plan, 4.2)                    # 1.5 s shorter on the wall: no freeze left
    assert gone.wall == pytest.approx(3.5) and gone.freeze == 0.0 and gone.freeze_at is None and gone.excess == 0.0
    part = s._shortened(plan, 5.76)                   # 0.2 s shorter: 0.3 s of freeze left, the same end
    assert part.freeze == pytest.approx(0.3) and part.freeze_at == 203.0 and part.excess == pytest.approx(0.8)
    assert part.end == pytest.approx(plan.end) and (part.start, part.rate) == (plan.start, plan.rate)
    tiny = s._shortened(plan, 5.52)                   # 0.1 s left: a hold that short stutters; min_freeze instead
    assert tiny.freeze == s.planner.s.min_freeze and tiny.end < plan.end


def test_a_rephrase_is_handed_to_a_background_task(tmp_path):
    """The only Claude work the voicer starts is a queued rephrase, and it goes to a task of its own."""
    s, _ = make_session(tmp_path)
    s.tr = SimpleNamespace(submit=lambda r: pytest.fail("submitted from the voicer"))
    st = voiced_line(s, start=300.0)
    st.voiced = False
    started: list = []
    s._side = lambda coro: (started.append(coro), coro.close())
    assert s._ask_rephrase(st, st.plan, VoiceCost())
    assert len(started) == 1


# ---- Claude failures ----------------------------------------------------------------------------------------------------
async def test_a_claude_failure_pauses_translation_says_what_to_do_and_recovers(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setattr(sm, "CLAUDE_BACKOFF", 0.5)
    fake_cli.env.setenv("FAKE_MODE", "not_signed_in")
    s, msgs = make_session(tmp_path, fake_cli.factory)
    await s.open(VIDEO)
    try:
        await until(lambda: any(m["type"] == "claude_error" for m in msgs))
        err = next(m for m in msgs if m["type"] == "claude_error")
        assert err["kind"] == "not_signed_in" and err["retryIn"] >= 0
        assert "Not logged in" in err["message"]
        assert not [m for m in msgs if m["type"] in ("unit", "unit_skipped", "error")]  # lines wait, never skipped
        fake_cli.env.setenv("FAKE_MODE", "ok")  # the user signs in
        await until(lambda: any(m["type"] == "unit" for m in msgs))
        assert any(m["type"] == "claude_ok" for m in msgs)
    finally:
        await s.close()


async def test_a_usage_limit_holds_translation_until_it_resets(tmp_path):
    s, msgs = make_session(tmp_path)
    err = ClaudeCLIError("usage_limit", "You've hit your session limit", limit="session",
                         resets_at=time.time() + 3600)
    await s._claude_failed(err)
    (sent,) = msgs
    assert sent == {"type": "claude_error", "kind": "usage_limit", "message": "You've hit your session limit",
                    "limit": "session", "resetsAt": err.resets_at, "retryIn": sent["retryIn"]}
    assert 3590 <= sent["retryIn"] <= 3610 and s._claude_hold - time.monotonic() > 3500
    await s._claude_failed(err)  # the same failure from another call: said once
    assert len(msgs) == 1


async def test_the_first_call_through_claude_clears_the_banner_once_per_session(tmp_path):
    """hello (or the last video's session) may have left a Claude problem on screen: the first call of this session
    that goes through clears it. Later ones say nothing until something fails again."""
    s, msgs = make_session(tmp_path)
    await s._claude_ok(time.monotonic())
    await s._claude_ok(time.monotonic())
    assert msgs == [{"type": "claude_ok"}]
    await s._claude_failed(ClaudeCLIError("not_signed_in", "Not logged in"))
    await s._claude_ok(time.monotonic())
    assert [m["type"] for m in msgs] == ["claude_ok", "claude_error", "claude_ok"]


async def test_a_call_asked_before_a_usage_limit_doesnt_clear_it(tmp_path):
    """A scene call under way when another hits the limit comes back fine: the banner and the hold stay."""
    s, msgs = make_session(tmp_path)
    asked = time.monotonic()
    err = ClaudeCLIError("usage_limit", "You've hit your session limit", limit="session",
                         resets_at=time.time() + 3 * 3600)
    await s._claude_failed(err)
    await s._claude_ok(asked)
    assert [m["type"] for m in msgs] == ["claude_error"] and s._claude_hold - time.monotonic() > 3 * 3600 - 60
    await s._claude_failed(err)  # still the same failure: said once
    assert len(msgs) == 1
    await s._claude_ok(time.monotonic())  # a call asked after it (once the hold is over) goes through
    assert [m["type"] for m in msgs] == ["claude_error", "claude_ok"]


async def test_a_scene_served_from_the_line_cache_says_nothing_about_claude(tmp_path):
    s, msgs = make_session(tmp_path)
    s.tr = MockSceneTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")))
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "Please remember to bring the tickets."), None, sm.Chunk(0.0, 3.0),
                   speech_s=3.0, scene=1)
    s.units[0] = st
    req = SceneRequest(1, (s._spec(st),))
    res = await s.tr.submit(req)  # translated and reviewed in an earlier session: in the line cache, with its class
    await s.tr.review(req, res.lines, {0: "full"})
    await s._scene(req, 0)
    await s.close()
    assert st.line is not None and st.cache_hit and st.line.coverage is not None
    assert not [m for m in msgs if m["type"] == "claude_ok"]


async def test_a_review_that_goes_through_clears_the_claude_banner_on_a_scene_served_from_the_cache(tmp_path):
    """A re-watch after a sign-in: every line comes from the line cache, so the review of the lines stored without a
    class is the only call that goes through Claude. It clears the banner as a scene call would."""
    s, msgs = make_session(tmp_path)
    s.tr = MockSceneTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")))
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "Please remember to bring the tickets."), None, sm.Chunk(0.0, 3.0),
                   speech_s=3.0, scene=1)
    s.units[0] = st
    req = SceneRequest(1, (s._spec(st),))
    await s.tr.submit(req)  # in the line cache, never reviewed
    await s._claude_failed(ClaudeCLIError("not_signed_in", "Not logged in"))
    fails = s._claude_fails
    await s._scene(req, 0)
    await s.close()
    assert st.cache_hit and st.line.coverage.cls == "C" and fails == 1 and s._claude_fails == 0
    assert [m["type"] for m in msgs] == ["claude_error", "claude_ok"]


@pytest.mark.parametrize(("kind", "said"), [("bad_output", False), ("timeout", False), ("not_signed_in", True)])
async def test_a_bad_or_slow_brief_reply_is_only_retried(tmp_path, kind, said):
    """Brief v1 is optional (v0 stays in use): a reply that failed its checks or timed out is asked for again, up to
    BRIEF_TRIES times, without a banner or a pause. What the user must fix is said as for a scene."""
    s, msgs = make_session(tmp_path)

    async def fail(meta, transcript):
        raise ClaudeCLIError(kind, "no brief")

    s.tr = SimpleNamespace(brief=SimpleNamespace(meta=sm.VideoMeta("A title")), make_brief=fail,
                           cancel=lambda match=None: 0)
    s._brief_task = object()
    await s._make_brief([("S1", "Hello there.")])
    assert s._brief_task is None and s._brief_next is None
    assert bool(msgs) == said and (s._claude_hold > time.monotonic()) == said


async def test_no_rephrase_is_asked_while_claude_is_paused_and_one_under_way_is_dropped(tmp_path):
    """A stall or a usage limit pauses scene calls; rephrases would otherwise pile up behind it in the translator's
    slots, past their deadlines. The lines keep their provisional takes."""
    s, _ = make_session(tmp_path)
    s.tr = MockSceneTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")), delay=5.0)
    st = voiced_line(s, start=300.0)
    assert s._ask_rephrase(st, st.plan, VoiceCost())
    (task,) = s._side_tasks
    await asyncio.sleep(0.2)  # under way
    await s._claude_failed(ClaudeCLIError("stalled", "Claude Code didn't start"))
    await asyncio.wait_for(task, 2.0)  # dropped at once, its slot free
    assert st.replacement is None and st.provisional
    assert not s._ask_rephrase(st, st.plan, VoiceCost())  # while paused, a long take ships as it is


# ---- ingest --------------------------------------------------------------------------------------------------------------
def test_resolve_keeps_the_metadata_the_brief_starts_from():
    info = {"description": "  How to fold a paper boat.\n", "tags": ["origami", " origami", "", 5, "boats"],
            "chapters": [{"start_time": 0, "title": "Intro"}, {"start_time": 42.5, "title": " Folding "},
                         {"start_time": None, "title": "Broken"}, {"title": "No start"}]}
    assert metadata(info) == ("How to fold a paper boat.", ((0.0, "Intro"), (42.5, "Folding")), ("origami", "boats"))
    assert metadata({}) == ("", (), ())


async def test_brief_v0_carries_the_metadata_and_the_prepass_talk_shares(tmp_path):
    s, _ = make_session(tmp_path)
    await s.open(VIDEO)
    try:
        await until(lambda: s.tr is not None)
    finally:
        await s.close()
    meta = s.tr.brief.meta
    assert (meta.title, meta.channel, meta.tags) == ("Demo video", "Maata demo", ("demo",))
    assert meta.chapters == ((0.0, "Start"),) and meta.description
    assert {sid for sid, _ in meta.talk_shares} == set(s.registry.speakers)
    assert sum(x for _, x in meta.talk_shares) == pytest.approx(1.0)


def test_speech_time_is_the_speakers_turns_or_words_inside_the_span(tmp_path):
    s, _ = make_session(tmp_path)
    words = [TimedWord("Hello", 10.0, 10.5), TimedWord("there.", 12.0, 12.6)]
    u = SourceUnit(0, "S1", 10.0, 12.6, "Hello there.", words)
    assert s._speech(u) == pytest.approx(1.1)  # no turns diarized there: the words themselves
    assert s._speech(SourceUnit(1, "S1", 5.0, 7.0, "Hm.")) == 2.0  # nothing known: the span


# ---- Claude's state for the UI (hello) --------------------------------------------------------------------------------
def test_health_says_installed_version_signed_in_and_models(tmp_path, fake_cli, monkeypatch):
    from maata_engine import claude_cli

    ok = claude_cli.ClaudeCLI(tmp_path, binary=str(fake_cli.exe)).health()
    assert ok == {"installed": True, "version": "2.1.281", "signedIn": True,
                  "models": ["claude-sonnet-5", "claude-opus-5-5"], "model": "claude-opus-5-5", "problem": None,
                  "message": ""}
    fake_cli.env.setenv("FAKE_SIGNED_IN", "0")
    out = claude_cli.ClaudeCLI(tmp_path, binary=str(fake_cli.exe)).health()
    assert out["signedIn"] is False and out["problem"] == "not_signed_in" and "claude auth login" in out["message"]
    monkeypatch.setattr(claude_cli, "find_binary", lambda explicit=None: None)
    gone = claude_cli.ClaudeCLI(tmp_path).health()
    assert gone["installed"] is False and gone["problem"] == "missing" and "Install Claude Code" in gone["message"]


async def test_hello_carries_claude_health_except_on_the_demo_engine(tmp_path, fake_cli):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from maata_engine.server import Engine

    fake_cli.env.setenv("MAATA_CLAUDE_BIN", str(fake_cli.exe))  # never the real CLI in a test
    eng = Engine("mock", tmp_path / "models", tmp_path / "cache", None, "tok", demo=True)
    async with serve(eng.handler, "127.0.0.1", 0, process_request=eng.process_request) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws?token=tok"
        async with connect(url) as ws:
            assert json.loads(await ws.recv())["claude"] is None  # the demo engine never calls Claude
        eng.backend.name = "apple"  # as a real backend: it translates through the CLI
        async with connect(url) as ws:
            hello = json.loads(await ws.recv())
        fake_cli.env.setattr("maata_engine.server.HEALTH_TIMEOUT", 0.5)
        fake_cli.env.setenv("FAKE_AUTH_DELAY", "3")  # a CLI that hangs at start (#91987)
        t0 = time.monotonic()
        async with connect(url) as ws:
            stuck = json.loads(await ws.recv())
    assert hello["claude"]["installed"] and hello["claude"]["version"] == "2.1.281" and hello["claude"]["signedIn"]
    assert time.monotonic() - t0 < 2.5 and stuck["claude"]["problem"] == "stalled"  # hello isn't held up by it


async def test_a_fit_adds_tiers_to_a_line_not_yet_voiced_and_it_is_chosen_again(tmp_path):
    s, _ = make_session(tmp_path)
    s.tr = MockSceneTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")))
    words = "Every line starts exactly when the speaker starts, even the long ones we say quickly."
    u = SourceUnit(3, "S1", 50.0, 52.0, words)
    st = UnitState(u, 52.2, sm.Chunk(50.0, 60.0), speech_s=2.0, scene=2)
    s.units[3] = st
    st.line = LineResult(3, {"full": Wording("ఇది " * 14 + "ఇది.")})
    assert not s._choose(st) and st.tier == "full"
    fit = s._fit_spec(st)
    await s._fit(SceneRequest(2, (fit,), "fit"))
    assert {"concise", "very_concise"} <= set(st.line.tiers) and st.tier != "full"
    voiced = UnitState(SourceUnit(4, "S1", 60.0, 62.0, words), None, sm.Chunk(60.0, 70.0), speech_s=2.0, voiced=True)
    voiced.line, s.units[4] = LineResult(4, {"full": Wording("ఇది " * 14 + "ఇది.")}), voiced
    await s._fit(SceneRequest(2, (replace_id(fit, 4),), "fit"))
    assert set(voiced.line.tiers) == {"full"}  # too late: it was voiced already


async def test_a_fit_that_comes_back_while_its_line_is_being_voiced_is_not_folded_in(tmp_path):
    """The voicer has started on the line: the tier it records stays the wording it voices (the next scene's context,
    a rephrase and units.jsonl all read it), and the fit's new tiers are too late."""
    s, st = band_case(tmp_path, {"full": 2.0}, tts=SlowTTS(pause=0.6))  # short of its 4 s: a fit asks for `fuller`
    sent: list[dict] = []

    async def keep(m: dict) -> None:
        sent.append(m)

    s.send_json = keep
    assert not s._choose(st)
    fit = s._fit_spec(st)
    fuller = LineResult(0, {"full": st.line.full, "fuller": Wording("ప" * round((3.8 - 0.15) * DEFAULT_RATE))})  # in the band

    class Answer:
        def submit(self, req):
            async def reply() -> SceneResult:
                return SceneResult(req.scene, req.call, lines={0: fuller})
            return asyncio.ensure_future(reply())

    s.tr = Answer()
    dub = asyncio.create_task(s._dub(st))
    await asyncio.sleep(0.2)  # the take is being synthesized
    await s._fit(SceneRequest(1, (fit,), "fit"))
    await dub
    (unit_msg,) = [m for m in sent if m["type"] == "unit"]
    assert st.tier == "full" and set(st.line.tiers) == {"full"} and unit_msg["telugu"] == st.line.full.spoken


def replace_id(spec: LineSpec, i: int) -> LineSpec:
    return replace(spec, id=i)


async def test_an_unexpected_translator_error_puts_the_lines_back_and_backs_off(tmp_path):
    s, msgs = make_session(tmp_path)

    class Broken:
        prompt_hash, model = "x", "broken"

        def submit(self, req):
            async def boom():
                raise RuntimeError("disk full")
            return asyncio.ensure_future(boom())

        def cancel(self, match=None):
            return 0

    s.tr = Broken()
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "A line."), None, sm.Chunk(0.0, 3.0), speech_s=3.0, scene=1)
    s.units[0] = st
    s._run_next = (0, 1)
    await s._scene(SceneRequest(1, (s._spec(st),)), 0)
    assert st.scene is None and st.line is None and s._run_next == (None, 0)  # asked for again, from a fresh run
    (err,) = [m for m in msgs if m["type"] == "claude_error"]
    assert err["kind"] == "failed" and "disk full" in err["message"] and s._claude_hold > time.monotonic()


def heard(tmp_path, n: int, every: float = 4.0, translated: int = 0, video: float | None = None, **kw) -> Session:
    """A session that has heard `n` lines of one speaker, one every `every` s, up to the last line's end; the first
    `translated` of them are scene 1, translated. The video lasts `video` s (by default, what is heard)."""
    s, _ = make_session(tmp_path, **kw)
    s.audio = np.zeros(int((video or n * every) * 16_000), np.float32)
    s._asr_cursor = n * every
    for i in range(n):
        u = SourceUnit(i, "S1", every * i, every * (i + 1) - 0.5, "A line.")
        st = UnitState(u, every * (i + 1), sm.Chunk(0.0, n * every), speech_s=every - 0.5)
        if i < translated:
            st.line, st.scene, st.tier, st.telugu = LineResult(i, {"full": Wording("ఒక మాట.")}), 1, "full", "ఒక మాట."
        s.units[i] = st
    s._run_next, s._fresh_at = ((translated - 1, 1) if translated else (None, 0)), None
    return s


async def test_a_seek_into_video_not_dubbed_yet_starts_a_short_scene_even_right_after_the_last(tmp_path):
    s = heard(tmp_path, 45, translated=6)  # scene 1: 0-23.5 s; everything heard
    req, nth = s._next_scene()
    assert nth == 1 and req.lines[0].id == 6  # straight on: the second scene of the run, about a minute
    s = heard(tmp_path, 45, translated=6)
    s.seek(24.0)
    req, nth = s._next_scene()
    assert nth == 0 and req.lines[0].id == 6 and 20.0 <= req.lines[-1].end - req.lines[0].start <= 30.0
    assert [te for _, te in req.context_before] == ["ఒక మాట."] * 3  # the lines before, with their Telugu


async def test_after_a_seek_a_run_starting_well_past_the_seek_point_keeps_its_place_in_its_run(tmp_path):
    """The short scene is for first audio at the seek point. When lines there are already on their way (a call still
    running), the next free run starts further on and carries on its run of scenes."""
    def session() -> Session:
        s = heard(tmp_path, 45, translated=6)
        for i in range(6, 21):  # scene 2, 24-83.5 s, on its way to Claude
            s.units[i].scene = 2
        s._run_next = (20, 2)
        return s

    s = session()
    s.seek(30.0)
    req, nth = s._next_scene()
    assert req.lines[0].id == 21 and nth == 2  # 84 s: not cut short
    s = session()
    s.seek(80.0)
    req, nth = s._next_scene()
    assert req.lines[0].id == 21 and nth == 0 and req.lines[-1].end - req.lines[0].start <= 30.0  # near it: short


# ---- the lookahead horizon, with the playhead held (the app banks its lead before it plays) ----------------------------
def test_a_run_inside_the_banked_lead_is_cut_once_nothing_more_is_heard_until_the_playhead_moves(tmp_path):
    """Lookahead 300 s: the app waits for 240 s dubbed before it plays, and ASR stops at the horizon until the playhead
    moves. A later scene (up to 150 s) starting inside those 240 s can't wait for its limit to be heard."""
    def session(translated: int, cursor: float) -> Session:
        s = heard(tmp_path, 56, every=6.0, translated=translated, video=900.0, lookahead=300.0)  # heard to 335.5 s
        s._run_next, s._asr_cursor = (translated - 1, 3), cursor
        return s

    req, nth = session(36, 336.0)._next_scene()  # the run starts at 216 s, 120 s of it heard
    assert nth == 3 and req.lines[0].start == 216.0 and len(req.lines) == 20
    assert session(36, 290.0)._next_scene() is None  # ASR still hearing: the scene waits for its limit
    assert session(41, 336.0)._next_scene() is None  # starts at 246 s, past the banked lead: the app will play first


@pytest.mark.parametrize("prepass", [30.0, 600.0])
async def test_with_the_playhead_held_the_lead_the_app_waits_for_is_dubbed(tmp_path, prepass):
    """End to end on a 15-minute mock video with lookahead 300 s and the playhead held at 0 until the app has its lead.
    With the default 10-minute pre-pass, longer than the lookahead, the pre-pass is diarized all the same."""
    s, msgs = make_session(tmp_path, resolver=LongDemo(900.0), lookahead=300.0, prepass=prepass)
    await s.open(VIDEO)
    try:
        await until(lambda: s.ready.contiguous_end(0.0) >= sm.BANK_SHARE * s.lookahead, timeout=40)
    finally:
        await s.close()
    assert s.playhead == 0.0 and not [m for m in msgs if m["type"] in ("error", "claude_error")]


async def test_brief_v1_is_made_from_the_whole_prepass_even_past_a_shorter_lookahead(tmp_path):
    """ARCHITECTURE §5.3 step 6: the rest of the pre-pass window is heard in the background, then brief v1 is made,
    without waiting for the playhead to bring it inside the lookahead."""
    s, _ = make_session(tmp_path, resolver=LongDemo(900.0), lookahead=300.0, prepass=600.0)
    await s.open(VIDEO)
    try:
        await until(lambda: any(e["event"] == "brief" and e["version"] == 1 for e in trace(tmp_path)), timeout=40)
    finally:
        await s.close()
    (call,) = [c for c in s.tr.cli.calls if c["call"] == "brief"]
    assert s.playhead == 0.0 and s._asr_cursor >= 599.0 and s._horizon() == 300.0
    within = sum(1 for st in s.units.values() if st.unit.start < s._horizon())
    assert len(call["message"]["transcript"]) > within  # lines heard past the horizon are in it


async def test_a_seek_into_a_long_scene_under_way_cuts_it_back_to_a_short_one_from_there(tmp_path):
    """First audio after a seek waits for 20-30 s of lines, not for a call of up to 150 s that holds the seek point
    (§3.12). Its lines before the seek point are never asked for again."""
    asked: list[SceneRequest] = []

    def factory(*a, **k):
        tr = MockSceneTranslator(*a, delay=30.0, **k)
        submit = tr.submit
        tr.submit = lambda r: (asked.append(r), submit(r))[1]
        return tr

    def long(r: SceneRequest) -> bool:  # runs on well past its second line
        return r.call == "scene" and len(r.lines) > 2 and r.lines[-1].end - r.lines[1].start > sm.LEAD_SCENE[1] + 1.0

    s, _ = make_session(tmp_path, factory)
    await s.open(VIDEO)
    try:
        await until(lambda: any(long(r) for r in asked))
        held = next(r for r in asked if long(r))
        n, t = len(asked), held.lines[1].start + 0.5
        s.seek(t)
        await until(lambda: held.scene in {e["scene"] for e in trace(tmp_path) if e.get("dropped")})
        await until(lambda: any(r.call == "scene" for r in asked[n:]))
    finally:
        await s.close()
    short = next(r for r in asked[n:] if r.call == "scene")
    lines = short.lines
    assert lines[0].start <= t and lines[0].end >= t - 2.0  # from the seek point (a line just ending there included)
    assert 20.0 <= lines[-1].end - lines[0].start <= 30.0 and short.scene > held.scene


# ---- the coverage review (§4.6) and the band rule's audit (§4.5) -------------------------------------------------------
class Reviewing(MockClaude):
    """The mock, with review calls that can be held (`gate`, until set or cancelled), fail (`fail`) or class lines by
    their English (`classes`: the start of the English -> (class, missing words)); a re-translation says one word more
    (సరిగ్గా, "exactly", which no demo line has)."""

    def __init__(self, gate: threading.Event | None = None, fail: str | None = None,
                 classes: dict[str, tuple[str, list[str]]] | None = None) -> None:
        super().__init__()
        self.gate, self.fail, self.classes = gate, fail, classes or {}

    def ask(self, system, prompt, schema=None, call="text", *, effort=None, cancel=None, tags=None):
        if call == "review":
            if self.gate is not None:
                while not self.gate.wait(0.05):
                    if cancel is not None and cancel.is_set():
                        self.calls.append({"call": "review", "cancelled": True, "message": json.loads(prompt)})
                        raise ClaudeCLIError("cancelled", "The Claude call was cancelled")
            if self.fail:
                self.calls.append({"call": "review", "message": json.loads(prompt)})
                raise ClaudeCLIError(self.fail, "Not logged in · Please run /login")
        reply = super().ask(system, prompt, schema, call, effort=effort, cancel=cancel, tags=tags)
        if call == "review":
            en = {x["id"]: x["en"] for x in json.loads(prompt)["lines"]}
            for x in reply.data["lines"]:
                cls, missing = next((v for k, v in self.classes.items() if en[x["id"]].startswith(k)), ("C", []))
                x.update({"class": cls, "missing": missing})
        elif call == "retranslate":
            for x in reply.data["lines"]:
                for tier in ("full", "fuller", "concise", "very_concise"):
                    if tier in x:
                        w = x[tier]
                        x[tier] = {"spoken": "సరిగ్గా " + w["spoken"],
                                   "english": [{"i": e["i"] + 1, "en": e["en"]} for e in w["english"]]}
        return reply


def reviewing(**kw):
    clis: list[Reviewing] = []

    def factory(cache_dir, video_id, brief, **k):
        clis.append(Reviewing(**kw))
        return ClaudeTranslator(cache_dir, video_id, brief, cli=clis[-1], **k)

    return factory, clis


def reviews(cli: Reviewing) -> list[dict]:
    return [c for c in cli.calls if c["call"] == "review"]


async def test_a_scenes_lines_count_as_translated_only_once_their_review_is_back(tmp_path):
    gate = threading.Event()
    factory, clis = reviewing(gate=gate)
    s, msgs = make_session(tmp_path, factory)
    await s.open(VIDEO)
    try:
        await until(lambda: s.tr is not None and any(c["call"] == "scene" for c in clis[0].calls)
                    and any(st.scene == 1 for st in s.units.values()))
        await asyncio.sleep(0.5)  # the scene call is back; its review is held
        first = [st for st in s.units.values() if st.scene == 1]
        assert first and all(st.line is None and st.telugu is None for st in first)
        assert not [m for m in msgs if m["type"] == "unit"]  # the voicer waits: nothing counts as translated yet
        gate.set()
        await until(lambda: sum(1 for m in msgs if m["type"] == "unit") >= 3)
    finally:
        await s.close()
    units = [e for e in trace(tmp_path) if e["event"] == "unit"]
    assert units and all(u["coverage"]["class"] == "C" and u["coverage"]["by"] == "review" for u in units)
    assert all(u["coverage"]["tier"] in u["tiers"] for u in units)
    review = next(c for c in reviews(clis[0]))
    assert review["effort"] == "high"
    (scene,) = [e for e in trace(tmp_path) if e["event"] == "scene" and e.get("scene") == 1]
    assert scene["coverage"]["C"] == scene["translated"] and scene["unreviewed"] == 0


async def test_a_line_the_review_finds_a_phrase_missing_from_is_retranslated_and_the_better_one_voiced(tmp_path):
    factory, clis = reviewing(classes={"Every line starts": ("P", ["exactly"])})
    s, msgs = make_session(tmp_path, factory)
    await s.open(VIDEO)
    try:
        await until(lambda: any(e["event"] == "unit" and e["source"].startswith("Every line") for e in trace(tmp_path)))
    finally:
        await s.close()
    redo = [c for c in clis[0].calls if c["call"] == "retranslate"]
    assert redo and redo[0]["message"]["lines"][0]["missing"] == ["exactly"] and redo[0]["effort"] == "medium"
    unit = next(e for e in trace(tmp_path) if e["event"] == "unit" and e["source"].startswith("Every line"))
    assert "సరిగ్గా" in unit["telugu"].split()  # the re-translation says more: classed C by the checks, and voiced
    assert unit["coverage"] == {"class": "C", "by": "validators", "tier": "full", "first": "P", "missing": [],
                                "added": [], "error": None}
    others = [e for e in trace(tmp_path) if e["event"] == "unit" and not e["source"].startswith("Every line")]
    assert others and all(e["coverage"]["class"] == "C" and "సరిగ్గా" not in e["telugu"].split() for e in others)


async def test_a_review_that_fails_leaves_its_lines_unreviewed_but_voiced(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CLAUDE_BACKOFF", 0.2)
    factory, clis = reviewing(fail="not_signed_in")
    s, msgs = make_session(tmp_path, factory)
    await s.open(VIDEO)
    try:
        await until(lambda: sum(1 for m in msgs if m["type"] == "unit") >= 3)
    finally:
        await s.close()
    assert any(m["type"] == "claude_error" and m["kind"] == "not_signed_in" for m in msgs)  # said, and held
    units = [e for e in trace(tmp_path) if e["event"] == "unit"]
    assert units and all(u["coverage"] is None for u in units)
    assert [e["error"] for e in trace(tmp_path) if e["event"] == "review"][0] == "not_signed_in"


async def test_a_seek_that_drops_a_scene_under_review_puts_its_lines_back_to_be_translated_again(tmp_path):
    gate = threading.Event()
    factory, clis = reviewing(gate=gate)
    s, msgs = make_session(tmp_path, factory, lookahead=60.0)
    await s.open(VIDEO)
    try:
        await until(lambda: s.tr is not None and len(reviews(clis[0])) == 0 and any(
            st.scene == 1 for st in s.units.values()) and any(c["call"] == "scene" for c in clis[0].calls))
        ids = sorted(st.unit.id for st in s.units.values() if st.scene == 1)
        await asyncio.sleep(0.3)  # scene 1 is back, and under review
        s.seek(150.0)
        await until(lambda: any(e.get("dropped") and e["scene"] == 1 for e in trace(tmp_path)))
        assert all(s.units[i].scene != 1 and s.units[i].line is None for i in ids)  # asked for again later
        gate.set()
        s.seek(0.0)
        await until(lambda: all(s.units[i].voiced for i in ids), timeout=30)
    finally:
        await s.close()
    assert any(c.get("cancelled") for c in reviews(clis[0]))  # the review under way was stopped
    units = {e["id"]: e for e in trace(tmp_path) if e["event"] == "unit"}
    assert all(units[i]["coverage"]["class"] == "C" for i in ids)


async def test_a_rewatch_serves_the_stored_classes_and_reviews_nothing_again(tmp_path):
    first, clis = reviewing()
    s, _ = make_session(tmp_path, first)
    await s.open(VIDEO)
    try:
        await until(lambda: s.ready.covered(1.0, 60.0))
    finally:
        await s.close()
    seen = {x["id"] for c in reviews(clis[0]) for x in c["message"]["lines"]}
    assert seen
    again, clis2 = reviewing()
    s, _ = make_session(tmp_path, again)
    await s.open(VIDEO)
    try:
        await until(lambda: all(s.units.get(i) is not None and s.units[i].voiced for i in seen), timeout=30)
    finally:
        await s.close()
    assert not seen & {x["id"] for c in reviews(clis2[0]) for x in c["message"]["lines"]}
    assert all(s.units[i].line.coverage.cls == "C" and s.units[i].cache_hit for i in seen)


async def test_units_jsonl_has_each_lines_class_speech_fill_and_required_rate(tmp_path):
    s, st = band_case(tmp_path, {"full": 4.6}, speech=4.0)
    st.next_start = 104.2  # the next line follows closely: the take is sped up to fit
    s._dir = tmp_path / "cache" / "dQw4w9WgXcQ"
    (s._dir / "pcm").mkdir(parents=True)
    st.line.coverage = Coverage("m", ("really",), tier="full")
    await s._dub(st)
    (unit,) = [e for e in trace(tmp_path) if e["event"] == "unit"]
    assert unit["coverage"] == {"class": "m", "by": "review", "tier": "full", "first": None, "missing": ["really"],
                                "added": [], "error": None}
    assert unit["tier"] == "full" and unit["coverage_voiced"] == "m"  # the class of the wording voiced
    assert st.plan.rate > 1.0
    assert unit["required_rate"] == round(st.take_s / 4.0, 3) > 1.1  # what it needs to fill the speech time exactly
    assert unit["speech_fill"] == round(st.take_s / st.plan.rate / 4.0, 3) < unit["required_rate"]  # as played


def test_the_band_rule_prices_a_line_from_above_with_the_next_lines_own_lag_limit(tmp_path):
    """The next line is short and a long gap follows it, so it may start up to 1.0 s late, not 0.6 (the planner's
    resync limit): a wording over the band that only fits with that is absorbed, as `_plan` would place it."""
    s, _ = make_session(tmp_path)
    nxt = UnitState(SourceUnit(1, "S1", 104.2, 106.0, "Then more."), 110.0, sm.Chunk(100.0, 120.0), speech_s=1.8)
    later = UnitState(SourceUnit(2, "S1", 110.0, 112.0, "And the end."), None, sm.Chunk(100.0, 120.0), speech_s=2.0)
    st = UnitState(SourceUnit(0, "S1", 100.0, 104.0, "Some words here."), 104.2, sm.Chunk(100.0, 120.0), speech_s=4.0)
    s.units.update({0: st, 1: nxt, 2: later})
    st.line = LineResult(0, {t: Wording("ప" * round((d - 0.15) * DEFAULT_RATE)) for t, d in
                             {"full": 6.2, "concise": 2.0}.items()})
    # 6.2 s at 1.2x from 99.7 s ends at 104.87 s: past 104.7 (0.6 s late, less the gap), inside 105.1 (1.0 s)
    assert s._choose(st) and st.tier == "full"
    nxt.next_start = 106.5  # no long gap after it: 0.6 s, and the wording can't be absorbed
    assert not s._choose(st) and st.tier == "concise"


@pytest.mark.parametrize("tiers,want,start", [
    ({"full": 12.0, "very_concise": 2.9}, ("concise",), "full"),     # the band lies between: the tier between it lacks,
    ({"full": 12.0, "concise": 2.9}, ("concise",), "full"),          # else the pick again, each cut from the longer one
    ({"full": 12.0, "concise": 9.0, "very_concise": 2.9}, ("very_concise",), "concise"),
    ({"fuller": 12.0, "full": 2.9}, ("fuller",), "full"),            # ... but never `full`: a `fuller`, from the pick
])
def test_a_fit_under_the_band_asks_for_the_side_the_band_is_on_never_fuller_past_a_long_tier(tmp_path, tiers, want,
                                                                                            start):
    s, st = band_case(tmp_path, tiers)
    assert not s._choose(st)
    assert s.estimator.estimate(st.line.tiers[st.tier].spoken, s._key("S1")) < sm.BAND[0] * st.speech_s
    fit = s._fit_spec(st)
    assert fit.want == want and fit.current == st.line.tiers[start].spoken
    assert fit.overflow == round(sm.count_units(fit.current) - s._target(st), 1)
    assert (fit.overflow > 0) == (start != st.tier)  # a wording over the slot to cut from, or the pick to add to


class Refit(MockClaude):
    """The mock with set wordings: a scene call answers `scene` (tier -> Telugu); a fit copies "current" into "full",
    as the prompt asks, and answers each tier it wants from `fit`."""

    def __init__(self, scene: dict[str, str], fit: dict[str, str]) -> None:
        super().__init__()
        self.scene, self.fit = scene, fit

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call in ("scene", "fit"):
            for x, line in zip(reply.data["lines"], json.loads(prompt)["lines"]):
                for k in ("fuller", "concise", "very_concise", "pieces"):
                    x.pop(k, None)
                given = self.scene if call == "scene" else {t: self.fit[t] for t in line["want"]}
                x.update({t: {"spoken": te, "english": []} for t, te in given.items()})
        return reply


async def test_a_fit_under_the_band_comes_back_with_the_tier_it_asked_for_and_that_is_chosen(tmp_path):
    """End to end through the translator: `full` runs far over the band and `concise` far under it. The fit asks for
    `concise` again, cut from `full`; its answer is in the band, is folded in, and is the wording chosen now."""
    s, st = band_case(tmp_path, {"full": 12.0, "concise": 2.9})
    words = {t: w.spoken for t, w in st.line.tiers.items()}
    in_band = "ప" * round((3.8 - 0.15) * DEFAULT_RATE)
    s.tr = ClaudeTranslator(tmp_path / "c", "v", sm.brief_v0(sm.VideoMeta("A title")),
                            cli=Refit(words, {"concise": in_band}))
    res = await s.tr.translate(SceneRequest(1, (s._line_spec(st, ("full", "concise")),)))
    st.line = res.lines[0]
    assert not s._choose(st) and st.tier == "concise"
    await s._fit(SceneRequest(1, (s._fit_spec(st),), "fit"))
    assert st.line.tiers["concise"].spoken == in_band and st.line.full.spoken == words["full"]
    assert st.tier == "concise" and st.telugu == in_band and s._choose(st)


def test_a_scenes_classes_count_only_where_the_class_is_of_the_wording_chosen(tmp_path):
    s, st = band_case(tmp_path, {"full": 3.8, "concise": 3.0})
    st.tier = "full"
    lines = [st, replace(st, tier="concise"), replace(st, line=replace(st.line, coverage=None))]
    st.line.coverage = Coverage("P", ("over the hill",), tier="full")
    lines[1].line = st.line  # the class is of `full`; `concise` is voiced
    assert s._classes(lines) == {"coverage": {"C": 0, "m": 0, "P": 1, "E": 0}, "other_tier": 1, "unreviewed": 1}
