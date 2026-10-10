"""A render job's translation of its lines and the dub loop's side of it (ARCHITECTURE §4.1-4.6, §4.9; OFFLINE-RENDER
§2.8-§2.10): the whole video's scene cuts, `want` tiers from the k prior, the band rule and its fits, the coverage
review, the Claude CLI off the dub loop, Claude failures and the banner, and the TTS script switch. No Claude: the mock
translator, small stand-ins, or a fake `codex` executable. All English and Telugu here is original test text (the
demo lines included). Ported from the streaming session's tests (OFFLINE-RENDER §10 step 6)."""

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

sys.path.insert(0, str(Path(__file__).parent))
from fakes import bare_job, put, trace, voice  # noqa: E402

from maata_engine import dubber, render  # noqa: E402
from maata_engine.backends.base import (Coverage, LineResult, LineSpec, SceneRequest, SceneResult, VideoMeta,  # noqa: E402
                                        Wording)
from maata_engine.backends.claude_translator import ClaudeTranslator, brief_v0  # noqa: E402
from maata_engine.backends.mock import MockClaude, MockSceneTranslator, MockTTS  # noqa: E402
from maata_engine.claude_cli import ClaudeCLIError  # noqa: E402
from maata_engine.dubber import BAND, K_PRIOR, SCENE_MAX_S, SCENE_MAX_UNITS, UnitState, scene_cut  # noqa: E402
from maata_engine.render import Scene, _read_rows  # noqa: E402
from maata_engine.text.akshara import count_units, mixed_units  # noqa: E402
from maata_engine.timing.duration import DEFAULT_OVERHEAD, DEFAULT_RATE, DurationEstimator  # noqa: E402
from maata_engine.timing.planner import Plan  # noqa: E402
from maata_engine.types import SourceUnit, TimedWord  # noqa: E402

# A fake `codex`: answers scene, fit and rephrase calls from the message (Telugu script, one made-up word per id), brief
# calls with a small brief and review calls with every line complete, after FAKE_DELAY s; FAKE_MODE=not_signed_in fails
# every call as the CLI does. Each call leaves {call, scene, ids, t0, t1} in FAKE_LOG.
FAKE = r"""#!@PYTHON@
import json, os, sys, time
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print("codex-cli 0.160.0")
    sys.exit(0)
if argv[:2] == ["login", "status"]:
    time.sleep(float(os.environ.get("FAKE_AUTH_DELAY", "0")))
    signed_in = os.environ.get("FAKE_SIGNED_IN", "1") == "1"
    print("Logged in using ChatGPT" if signed_in else "Not logged in")
    sys.exit(0 if signed_in else 1)
msg = json.loads(json.loads(sys.stdin.read())["input"])
t0 = time.time()
def emit(ev):
    print(json.dumps(ev, ensure_ascii=False), flush=True)
emit({"type": "thread.started", "thread_id": "fake"})
emit({"type": "turn.started"})
if os.environ.get("FAKE_MODE") == "not_signed_in":
    emit({"type": "turn.failed", "error": {"message": "Not logged in"}})
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
emit({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(data, ensure_ascii=False)}})
emit({"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}})
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"call": msg.get("call"), "scene": msg.get("scene"),
                        "ids": [x["id"] for x in msg.get("lines", [])], "t0": t0, "t1": time.time()}) + "\n")
"""


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
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
        seconds = 0.15 + mixed_units(text) / self.rate
        n = int(min(seconds, max_seconds or seconds) * self.sample_rate)
        return np.full(n, 0.01, np.float32)


async def until(pred, timeout: float = 20.0, step: float = 0.05) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(step)


def a_brief() -> object:
    return brief_v0(VideoMeta("A title"))


def kinds(events: list[dict], *types: str) -> list[str]:
    return [e["type"] for e in events if e["type"] in types]


# ---- scene cuts (§4.1, OFFLINE-RENDER §2.8) --------------------------------------------------------------------------
def unit(i: int, start: float, end: float, speaker: str = "S1", text: str = "A line.") -> SourceUnit:
    return SourceUnit(i, speaker, start, end, text)


def lines_every(seconds: float, n: int, length: float = 3.0, text: str = "A line.", speaker: str = "S1"):
    return [unit(i, i * seconds, i * seconds + length, speaker, text) for i in range(n)]


def test_legacy_scene_caps_remain_available_for_controlled_comparisons():
    def legacy(run):
        return scene_cut(run, max_seconds=150.0, max_units=30)

    run = lines_every(4.0, 60, length=3.5)
    assert legacy(run) == 30                                         # 30 lines come first (they end at 119.5 s)
    sparse = lines_every(8.0, 40)                                     # 5 s gaps between lines: every boundary a pause
    assert legacy(sparse) == 19                                      # ends at 147 s, the last pause under 150 s
    turns = [unit(i, 10.0 * i, 10.0 * i + 9.5, "S1" if i < 11 else "S2") for i in range(20)]
    assert legacy(turns) == 11                                       # the speaker turn at 110 s, not the limit
    early = [unit(i, 10.0 * i, 10.0 * i + 9.5, "S1" if i < 3 else "S2") for i in range(20)]
    assert legacy(early) == 15                                       # a turn in the first half doesn't count
    assert legacy([unit(0, 0.0, 170.0)]) == 1                          # one sentence longer than a scene still goes


def test_production_scenes_hold_at_most_30_seconds_and_six_whole_lines():
    assert (SCENE_MAX_S, SCENE_MAX_UNITS) == (30.0, 6)
    assert scene_cut(lines_every(4.0, 20, length=3.5)) == 6
    assert scene_cut(lines_every(8.0, 20)) == 4
    turns = [unit(i, 5.0 * i, 5.0 * i + 4.5, "S1" if i < 4 else "S2") for i in range(10)]
    assert scene_cut(turns) == 4
    assert scene_cut([unit(0, 0.0, 45.0), unit(1, 46.0, 48.0)]) == 1


def test_the_rest_of_the_video_under_the_limit_is_one_scene():
    assert scene_cut(lines_every(4.0, 5)) == 5                       # 19 s and 5 lines: all of it
    assert scene_cut(lines_every(4.0, 20), max_seconds=150.0, max_units=30) == 20


@pytest.mark.parametrize("seconds,limit", [(7.5, 1), (30.0, 6), (40.0, 6), (45.0, 8), (150.0, 30), (240.0, 40)])
def test_explicit_scene_caps_cover_every_whole_sentence_and_match_a_bounded_window(seconds, limit):
    run, start = [], 0.0
    for i in range(83):
        end = start + (170.0 if i == 12 else 2.0 + i % 4)
        run.append(unit(i, start, end, f"S{1 + i // 7 % 2}"))
        start = end + (1.2 if i % 11 == 0 else 0.2)
    remaining, seen = run, []
    while remaining:
        n = scene_cut(remaining, max_seconds=seconds, max_units=limit)
        assert n == scene_cut(remaining[:limit + 1], max_seconds=seconds, max_units=limit)
        assert 1 <= n <= limit
        assert n == 1 or remaining[n - 1].end - remaining[0].start <= seconds
        seen.extend(u.id for u in remaining[:n])
        remaining = remaining[n:]
    assert seen == list(range(len(run)))


def test_explicit_scene_cap_uses_its_own_second_half_for_turns_and_pauses():
    turns = [unit(i, 5.0 * i, 5.0 * i + 4.6, "S1" if i < 4 else "S2") for i in range(10)]
    assert scene_cut(turns, max_seconds=30.0, max_units=10) == 4  # the turn at 20 s is in this cap's second half
    early = [replace(u, speaker="S1" if i < 2 else "S2") for i, u in enumerate(turns)]
    assert scene_cut(early, max_seconds=30.0, max_units=10) == 6  # the turn at 10 s is too early
    pauses = [replace(u, speaker="S1", end=u.end - (0.7 if i == 3 else 0.0)) for i, u in enumerate(turns)]
    assert scene_cut(pauses, max_seconds=30.0, max_units=10) == 4  # the 1.1 s pause wins
    assert scene_cut([unit(0, 0.0, 170.0)], max_seconds=30.0, max_units=6) == 1


@pytest.mark.parametrize("seconds,limit", [(0.0, 6), (-1.0, 6), (float("inf"), 6), (float("nan"), 6),
                                          (30.0, 0), (30.0, -1), (30.0, 1.5)])
def test_invalid_scene_caps_are_rejected(seconds, limit):
    with pytest.raises(ValueError, match="scene limits"):
        scene_cut(lines_every(4.0, 4), max_seconds=seconds, max_units=limit)


def test_explicit_production_caps_match_defaults_and_omitted_caps_follow_the_default_constants(monkeypatch):
    for run in (lines_every(4.0, 60, length=3.5), lines_every(8.0, 40), [unit(0, 0.0, 170.0)]):
        assert scene_cut(run) == scene_cut(run, max_seconds=30.0, max_units=6)
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 40.0)
    monkeypatch.setattr(dubber, "SCENE_MAX_UNITS", 8)
    run = lines_every(4.0, 60, length=3.5)
    assert scene_cut(run) == scene_cut(run, max_seconds=40.0, max_units=8) == 8


@pytest.mark.parametrize("seconds,limit", [(30.0, 6), (240.0, 40)])
def test_render_scene_cap_hook_is_deterministic_across_preview_and_cached_state(tmp_path, seconds, limit):
    job = bare_job(tmp_path)
    put(job, *(UnitState(u, None, speech_s=3.0) for u in lines_every(4.0, 85)))
    before = job._cut_scenes(max_seconds=seconds, max_units=limit)
    assert len(before[0].lines) == limit  # a larger test cap is not silently constrained by the old 30-line slice
    job.settings = replace(job.settings, stop_at=20.0)
    for st in job._order[:15]:
        st.voiced = True
    after = job._cut_scenes(max_seconds=seconds, max_units=limit)
    assert [(sc.no, sc.k, [st.unit.id for st in sc.lines]) for sc in before] == \
           [(sc.no, sc.k, [st.unit.id for st in sc.lines]) for sc in after]


def test_the_whole_videos_scenes_are_cut_once_from_the_first_line_to_the_last(tmp_path):
    job = bare_job(tmp_path)
    turns = [UnitState(u, None, speech_s=3.0) for u in
             (unit(i, 6.0 * i, 6.0 * i + 5.0, "S1" if i % 13 < 9 else "S2") for i in range(70))]
    put(job, *turns)
    scenes = job._cut_scenes()
    assert [st.unit.id for sc in scenes for st in sc.lines] == list(range(70))  # every line in one scene, in order
    for sc in scenes:
        rest = [st.unit for st in job._order[sc.k:]]
        assert len(sc.lines) == scene_cut(rest) and sc.lines[0] is job._order[sc.k]
        assert sc.lines[-1].unit.end - sc.lines[0].unit.start <= SCENE_MAX_S and len(sc.lines) <= SCENE_MAX_UNITS
    assert [sc.no for sc in scenes] == list(range(1, len(scenes) + 1)) and job._cut_scenes()[2].lines == scenes[2].lines


# ---- want tiers and the k prior (§4.3) --------------------------------------------------------------------------------
def spec_for(job, text: str, span: float, speech: float | None = None, speaker: str = "S1") -> LineSpec:
    u = SourceUnit(len(job.units), speaker, 10.0, 10.0 + span, text)
    st = UnitState(u, None, speech_s=span if speech is None else speech)
    put(job, st)
    return job._spec(st)


def test_want_tiers_come_from_the_predicted_length(tmp_path):
    job = bare_job(tmp_path)
    en = "Please remember to bring the tickets."
    pred = 1.4 * mixed_units(en)                                      # the prior: 1.4 aksharas per English syllable
    # the span whose target is pred / ratio: that many aksharas at the prior's pace, after its overhead
    span = lambda ratio: DEFAULT_OVERHEAD + pred / (ratio * DEFAULT_RATE)  # noqa: E731
    assert job._k("S1") == K_PRIOR == 1.4
    long = spec_for(job, en, span=span(1.2))                          # 1.2 x the target
    assert long.want == ("full", "concise", "very_concise") and long.target_aksharas == round(pred / 1.2, 1)
    assert job.units[long.id].pred_full == pytest.approx(pred)
    assert spec_for(job, en, span=span(1.05)).want == ("full",)
    assert spec_for(job, en, span=span(0.9)).want == ("full",)
    assert spec_for(job, en, span=span(0.7)).want == ("full", "fuller")  # under 0.85 x a speech-dense slot
    # 0.67 x the target, but a quarter of the slot is pause: short lines are not padded to fill pauses
    assert spec_for(job, en, span=span(0.5), speech=0.75 * span(0.5)).want == ("full",)


def test_k_becomes_each_speakers_running_median(tmp_path):
    job = bare_job(tmp_path)
    en = "Please remember to bring the tickets."
    syllables = mixed_units(en)
    for i, ratio in enumerate((1.6, 2.0, 1.8)):
        st = UnitState(SourceUnit(i, "S1", 4.0 * i, 4.0 * i + 3.0, en), None)
        st.line = LineResult(i, {"full": Wording("ప" * round(syllables * ratio))})
        job._learn_k(st)
        assert job._k("S1") == (K_PRIOR if ratio != 1.8 else pytest.approx(round(syllables * 1.8) / syllables))
    assert job._k("S2") == K_PRIOR
    tiny = UnitState(SourceUnit(3, "S2", 20.0, 21.0, "Yes."), None)
    tiny.line = LineResult(3, {"full": Wording("అవును.")})
    job._learn_k(tiny)
    assert "S2" not in job._ratios  # too short to say anything about pace


# ---- the band rule (§4.5) ---------------------------------------------------------------------------------------------
def band_case(tmp_path, tiers: dict[str, float], speech: float = 4.0, tts=None):
    """A line whose tiers have the given predicted durations (s), at the default pace (DEFAULT_RATE aksharas/s)."""
    job = bare_job(tmp_path, tts)
    u = SourceUnit(0, "S1", 100.0, 100.0 + speech, "Some words here.")
    st = UnitState(u, 100.0 + speech + 2.0, speech_s=speech)
    put(job, st)
    st.line = LineResult(0, {t: Wording("ప" * round((d - 0.15) * DEFAULT_RATE)) for t, d in tiers.items()})
    return job, st


def test_the_most_complete_wording_inside_the_band_wins(tmp_path):
    job, st = band_case(tmp_path, {"fuller": 4.2, "full": 3.8, "concise": 3.0})
    assert job._choose(st) and st.tier == "fuller" and st.telugu == st.line.tiers["fuller"].spoken


def test_above_the_band_the_closest_is_taken_when_the_planner_absorbs_it(tmp_path):
    job, st = band_case(tmp_path, {"full": 4.8, "concise": 2.6})  # 4.8 s over a 4 s speech time, 2 s of silence after
    assert job._choose(st) and st.tier == "full"


def test_otherwise_the_closest_from_below(tmp_path):
    job, st = band_case(tmp_path, {"full": 12.0, "concise": 9.0, "very_concise": 2.9})  # 9 s won't fit even at 1.2x
    assert not job._choose(st) and st.tier == "very_concise"
    job, st = band_case(tmp_path / "2", {"full": 12.0, "concise": 9.0})  # nothing below: the closest from above
    assert not job._choose(st) and st.tier == "concise"


@pytest.mark.parametrize("change,eligible", [("none", True), ("too_long", False), ("voicing", False),
                                              ("voiced", False), ("full_selected", False), ("same_text", False)])
async def test_initial_full_fallback_requires_unvoiced_distinct_text_and_current_planner_room(
        tmp_path, monkeypatch, change, eligible):
    job, st = band_case(tmp_path, {"full": 4.8, "concise": 4.0})
    st.unit = replace(st.unit, text="Bring three copies by Friday, but do not send the original.")
    full = Wording("శుక్రవారంలోగా మూడు కాపీలు తీసుకురండి, కానీ అసలు పత్రాన్ని పంపకండి.")
    short = Wording("కాపీలు తీసుకురండి.")
    st.line = LineResult(0, {"full": full, "concise": short})
    full_seconds = 12.0 if change == "too_long" else 4.0 if change == "full_selected" else 4.8
    if change == "same_text":
        st.line.tiers["concise"] = full
    st.voicing, st.voiced = change == "voicing", change == "voiced"
    monkeypatch.setattr(job.estimator, "estimate", lambda text, key: full_seconds if text == full.spoken else 4.0)

    class Review:
        supports_full_fallback = True

        async def review(self, req, lines, chosen, *, fallbacks, fallback_check):
            assert fallbacks == ({0} if eligible else set())
            assert fallback_check(0) is eligible
            if eligible:
                # The callback is evaluated at the eventual review result, not just when the request is queued.
                st.voicing = True
                assert not fallback_check(0)
                st.voicing = False
                monkeypatch.setattr(job.estimator, "estimate", lambda *_: 30.0)
                assert not fallback_check(0)
            return SceneResult(req.scene, "review", lines=lines)

    job.tr = Review()
    await job._review(SceneRequest(1, (job._spec(st),)), {0: st.line})


@pytest.mark.parametrize("first", ["P", "E"])
def test_semantically_approved_full_is_kept_when_estimated_pace_changes(tmp_path, first):
    job, st = band_case(tmp_path, {"fuller": 3.9, "full": 12.0, "concise": 4.0, "very_concise": 2.0})
    st.line.coverage = Coverage("C", tier="full", by="review", first=first)
    assert not job._choose(st) and st.tier == "full"  # neither a fitting fuller nor known-bad concise can replace it
    job._wording(st, job._key("S1"))
    assert st.tier == "full" and st.telugu == st.line.full.spoken
    assert job._fit_spec(st) is None and job._shorter_tier(st, job._key("S1"), 4.0) is None
    assert set(st.line.tiers) == {"fuller", "full", "concise", "very_concise"}  # retain diagnostic alternatives


async def test_fit_already_in_flight_cannot_replace_newly_approved_full(tmp_path):
    job, st = band_case(tmp_path, {"full": 12.0, "concise": 4.0})
    submitted, release = asyncio.Event(), asyncio.Event()
    replacement = LineResult(0, {"full": Wording("కొత్త చిన్న వాక్యం.")})

    class Fit:
        async def submit(self, req):
            submitted.set()
            await release.wait()
            return SceneResult(req.scene, req.call, lines={0: replacement})

    job.tr = Fit()
    pending = asyncio.create_task(job._fit(SceneRequest(1, (job._line_spec(st, ("concise",)),), "fit")))
    await submitted.wait()
    approved = replace(st.line, coverage=Coverage("C", tier="full", first="P"))
    st.line = approved
    job._choose(st)
    release.set()
    await pending
    assert st.line is approved and st.tier == "full" and st.telugu == approved.full.spoken


def test_a_line_short_of_its_slot_asks_a_fit_for_fuller_and_one_too_long_for_shorter_tiers(tmp_path):
    job, st = band_case(tmp_path, {"full": 2.0})
    assert not job._choose(st) and job._fit_spec(st).want == ("fuller",)
    job, st = band_case(tmp_path / "2", {"full": 12.0, "concise": 9.0})
    assert not job._choose(st)
    fit = job._fit_spec(st)
    assert fit.want == ("very_concise",) and fit.current == st.line.tiers["concise"].spoken and fit.overflow > 0


def test_the_band_rule_prices_a_line_from_above_with_the_next_lines_own_lag_limit(tmp_path):
    """The next line is short and a long gap follows it, so it may start up to 1.0 s late, not 0.6 (the planner's
    resync limit): a wording over the band that only fits with that is absorbed, as `_plan` would place it on W."""
    job = bare_job(tmp_path)
    nxt = UnitState(SourceUnit(1, "S1", 104.2, 106.0, "Then more."), 110.0, speech_s=1.8)
    later = UnitState(SourceUnit(2, "S1", 110.0, 112.0, "And the end."), None, speech_s=2.0)
    st = UnitState(SourceUnit(0, "S1", 100.0, 104.0, "Some words here."), 104.2, speech_s=4.0)
    put(job, st, nxt, later)
    st.line = LineResult(0, {t: Wording("ప" * round((d - 0.15) * DEFAULT_RATE)) for t, d in
                             {"full": 6.2, "concise": 2.0}.items()})
    # 6.2 s at 1.2x from 99.7 s ends at 104.87 s: past 104.7 (0.6 s late, less the gap), inside 105.1 (1.0 s)
    assert job._choose(st) and st.tier == "full"
    nxt.next_start = 106.5  # no long gap after it: 0.6 s, and the wording can't be absorbed
    assert not job._choose(st) and st.tier == "concise"


@pytest.mark.parametrize("tiers,want,start", [
    ({"full": 12.0, "very_concise": 2.9}, ("concise",), "full"),     # the band lies between: the tier between it lacks,
    ({"full": 12.0, "concise": 2.9}, ("concise",), "full"),          # else the pick again, each cut from the longer one
    ({"full": 12.0, "concise": 9.0, "very_concise": 2.9}, ("very_concise",), "concise"),
    ({"fuller": 12.0, "full": 2.9}, ("fuller",), "full"),            # ... but never `full`: a `fuller`, from the pick
])
def test_a_fit_under_the_band_asks_for_the_side_the_band_is_on_never_fuller_past_a_long_tier(tmp_path, tiers, want,
                                                                                            start):
    job, st = band_case(tmp_path, tiers)
    assert not job._choose(st)
    assert job.estimator.estimate(st.line.tiers[st.tier].spoken, job._key("S1")) < BAND[0] * st.speech_s
    fit = job._fit_spec(st)
    assert fit.want == want and fit.current == st.line.tiers[start].spoken
    assert fit.overflow == round(count_units(fit.current) - job._target(st), 1)
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
    job, st = band_case(tmp_path, {"full": 12.0, "concise": 2.9})
    words = {t: w.spoken for t, w in st.line.tiers.items()}
    in_band = "ప" * round((3.8 - 0.15) * DEFAULT_RATE)
    job.tr = ClaudeTranslator(tmp_path / "c", "v", a_brief(), cli=Refit(words, {"concise": in_band}))
    res = await job.tr.translate(SceneRequest(1, (job._line_spec(st, ("full", "concise")),)))
    st.line = res.lines[0]
    assert not job._choose(st) and st.tier == "concise"
    await job._fit(SceneRequest(1, (job._fit_spec(st),), "fit"))
    assert st.line.tiers["concise"].spoken == in_band and st.line.full.spoken == words["full"]
    assert st.tier == "concise" and st.telugu == in_band and job._choose(st)


async def test_a_fit_adds_tiers_to_a_line_not_yet_voiced_and_it_is_chosen_again(tmp_path):
    job = bare_job(tmp_path)
    job.tr = MockSceneTranslator(tmp_path / "c", "v", a_brief())
    words = "Every line starts exactly when the speaker starts, even the long ones we say quickly."
    st = UnitState(SourceUnit(3, "S1", 50.0, 52.0, words), 52.2, speech_s=2.0, scene=2)
    voiced = UnitState(SourceUnit(4, "S1", 60.0, 62.0, words), None, speech_s=2.0, voiced=True)
    put(job, st, voiced)
    st.line = LineResult(3, {"full": Wording("ఇది " * 14 + "ఇది.")})
    assert not job._choose(st) and st.tier == "full"
    fit = job._fit_spec(st)
    await job._fit(SceneRequest(2, (fit,), "fit"))
    assert {"concise", "very_concise"} <= set(st.line.tiers) and st.tier != "full"
    voiced.line = LineResult(4, {"full": Wording("ఇది " * 14 + "ఇది.")})
    await job._fit(SceneRequest(2, (replace(fit, id=4),), "fit"))
    assert set(voiced.line.tiers) == {"full"}  # too late: it was voiced already


async def test_a_fit_that_comes_back_while_its_line_is_being_voiced_is_not_folded_in(tmp_path):
    """The dub loop has started on the line: the tier it keeps stays the wording it voices (its take row, the next
    scene's context and the manifest all read it), and the fit's new tiers are too late."""
    job, st = band_case(tmp_path, {"full": 2.0}, tts=SlowTTS(pause=0.6))  # short of its 4 s: a fit asks for `fuller`
    assert not job._choose(st)
    fit = job._fit_spec(st)
    fuller = LineResult(0, {"full": st.line.full, "fuller": Wording("ప" * round((3.8 - 0.15) * DEFAULT_RATE))})  # in band

    class Answer:
        def submit(self, req):
            async def reply() -> SceneResult:
                return SceneResult(req.scene, req.call, lines={0: fuller})
            return asyncio.ensure_future(reply())

    job.tr = Answer()
    dub = asyncio.create_task(job._voice_line(st))
    await asyncio.sleep(0.2)  # the take is being synthesized
    await job._fit(SceneRequest(1, (fit,), "fit"))
    await dub
    (row,) = _read_rows(job.render_dir / "takes.jsonl")
    assert st.tier == "full" and set(st.line.tiers) == {"full"} and row["wording"] == "full"
    assert st.telugu == st.line.full.spoken and [t["spoken"] for t in row["takes"]] == [st.line.full.spoken]


# ---- a scene call and its review (§4.6) -------------------------------------------------------------------------------
@pytest.mark.parametrize("english", [0, 1, 3, 5])
async def test_reviewed_latin_substitution_index_is_logged_without_a_mixing_quota_warning(tmp_path, caplog, english):
    """The compatible trace value counts approved substitutions; unmapped loans still have Telugu spellings.
    Neither zero substitutions nor many imply a quality problem or an English quota."""
    job = bare_job(tmp_path)
    spoken = "ఒకటి రెండు మూడు నాలుగు ఐదు ఆరు ఏడు ఎనిమిది తొమ్మిది పది."
    line = LineResult(0, {"full": Wording(spoken, tuple((i, "word") for i in range(english)))},
                      coverage=Coverage("C", tier="full"))
    st = UnitState(SourceUnit(0, "S1", 10.0, 16.0, "One two three four five six seven eight nine ten."), 17.0,
                   speech_s=6.0, scene=1)
    put(job, st)

    class Answer:
        prompt_hash, brief = "x", SimpleNamespace(version=1)

        def submit(self, req):
            async def reply() -> SceneResult:
                return SceneResult(req.scene, req.call, lines={0: line})
            return asyncio.ensure_future(reply())

        async def review(self, req, lines, chosen):
            return SceneResult(req.scene, "review", lines=dict(lines))

    job.tr = Answer()
    with caplog.at_level("WARNING", logger="maata.dubber"):
        await job._scene(SceneRequest(1, (job._spec(st),)))
    (scene,) = trace(job, "scene")
    assert scene["cmi"] == 10.0 * min(english, 10 - english)                   # logged for every scene
    assert "code-mixing index" not in caplog.text and st.line == line


def test_a_scenes_classes_count_only_where_the_class_is_of_the_wording_chosen(tmp_path):
    job, st = band_case(tmp_path, {"full": 3.8, "concise": 3.0})
    st.tier = "full"
    lines = [st, replace(st, tier="concise"), replace(st, line=replace(st.line, coverage=None))]
    st.line.coverage = Coverage("P", ("over the hill",), tier="full")
    lines[1].line = st.line  # the class is of `full`; `concise` is voiced
    assert job._classes(lines) == {"coverage": {"C": 0, "m": 0, "P": 1, "E": 0}, "other_tier": 1, "unreviewed": 1}


async def test_a_scene_served_from_the_line_cache_says_nothing_about_claude(tmp_path):
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    job.tr = MockSceneTranslator(tmp_path / "c", "v", a_brief())
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "Please remember to bring the tickets."), None, speech_s=3.0, scene=1)
    put(job, st)
    req = SceneRequest(1, (job._spec(st),))
    res = await job.tr.submit(req)  # translated and reviewed by an earlier run: in the line cache, with its class
    await job.tr.review(req, res.lines, {0: "full"})
    await job._scene(req)
    assert st.line is not None and st.cache_hit and st.line.coverage is not None
    assert not kinds(events, "claude_ok")


async def test_a_review_that_goes_through_clears_the_claude_banner_on_a_scene_served_from_the_cache(tmp_path):
    """A resume after a sign-in: every line comes from the line cache, so the review of the lines stored without a
    class is the only call that goes through Claude. It clears the banner as a scene call would."""
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    job.tr = MockSceneTranslator(tmp_path / "c", "v", a_brief())
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "Please remember to bring the tickets."), None, speech_s=3.0, scene=1)
    put(job, st)
    req = SceneRequest(1, (job._spec(st),))
    await job.tr.submit(req)  # in the line cache, never reviewed
    await job._claude_failed(ClaudeCLIError("not_signed_in", "Not logged in"))
    fails = job._claude_fails
    await job._scene(req)
    assert st.cache_hit and st.line.coverage.cls == "C" and fails == 1 and job._claude_fails == 0
    assert kinds(events, "claude_error", "claude_ok") == ["claude_error", "claude_ok"]
    assert all(e["videoId"] == job.video_id for e in events)


async def test_an_unexpected_translator_error_puts_the_lines_back_and_backs_off(tmp_path):
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)

    class Broken:
        prompt_hash, model = "x", "broken"

        def submit(self, req):
            async def boom():
                raise RuntimeError("disk full")
            return asyncio.ensure_future(boom())

        def cancel(self, match=None):
            return 0

    job.tr = Broken()
    st = UnitState(SourceUnit(0, "S1", 0.0, 3.0, "A line."), None, speech_s=3.0, scene=1)
    put(job, st)
    sc = Scene(1, [st], 0)
    token = render._SCENE.set(sc)  # as its lane asks for it
    try:
        await job._scene(SceneRequest(1, (job._spec(st),)))
    finally:
        render._SCENE.reset(token)
    assert st.scene is None and st.line is None and sc.released  # asked for again once the hold is over
    assert sc.failed is not None and sc.failed.kind == "failed"
    (err,) = [e for e in events if e["type"] == "claude_error"]
    assert err["kind"] == "failed" and "disk full" in err["message"] and job._claude_hold > time.monotonic()


def test_speech_time_is_the_speakers_turns_or_words_inside_the_span(tmp_path):
    job = bare_job(tmp_path)
    words = [TimedWord("Hello", 10.0, 10.5), TimedWord("there.", 12.0, 12.6)]
    u = SourceUnit(0, "S1", 10.0, 12.6, "Hello there.", words)
    assert job._speech(u) == pytest.approx(1.1)  # no turns diarized there: the words themselves
    assert job._speech(SourceUnit(1, "S1", 5.0, 7.0, "Hm.")) == 2.0  # nothing known: the span


# ---- a take placed in another one's place (a fix-up, §2.10) -----------------------------------------------------------
def test_a_shorter_take_in_the_same_place_needs_less_freeze(tmp_path):
    """The fix-up plays from the same start at the same rate; the time it saves comes off any freeze planned, and it
    never ends later than the take it replaces (the lines after it are placed already). (A render's plans have no
    freeze, §2.11; `_shortened` still prices one.)"""
    job = bare_job(tmp_path)
    plan = Plan(7, 200.0, 1.2, 5.0, 0.0, freeze=0.5, needs_shorter=True, freeze_at=203.0, excess=1.0)
    gone = job._shortened(plan, 4.2)                    # 1.5 s shorter on the wall: no freeze left
    assert gone.wall == pytest.approx(3.5) and gone.freeze == 0.0 and gone.freeze_at is None and gone.excess == 0.0
    part = job._shortened(plan, 5.76)                   # 0.2 s shorter: 0.3 s of freeze left, the same end
    assert part.freeze == pytest.approx(0.3) and part.freeze_at == 203.0 and part.excess == pytest.approx(0.8)
    assert part.end == pytest.approx(plan.end) and (part.start, part.rate) == (plan.start, plan.rate)
    tiny = job._shortened(plan, 5.52)                   # 0.1 s left: a hold that short stutters; min_freeze instead
    assert tiny.freeze == job.planner.s.min_freeze and tiny.end < plan.end
    # Still long while it needs a freeze or ends past its window; inside it, not.
    assert not gone.needs_shorter and part.needs_shorter and tiny.needs_shorter
    unfrozen = Plan(7, 200.0, 1.2, 5.0, 0.0, needs_shorter=True, excess=1.0)  # as a render places one
    inside = job._shortened(unfrozen, 4.6)
    assert (inside.start, inside.rate, inside.freeze, inside.excess, inside.needs_shorter) == (200.0, 1.2, 0.0, 0.0, False)


def test_a_take_in_another_ones_place_is_long_while_it_still_runs_long(tmp_path):
    """The overdraft the time saved leaves, and whether the take still runs long, are recomputed; a longer take (the
    voiced-wording review's re-translation) gets no more freeze than planned, and is long only once it runs well past
    its window."""
    job = bare_job(tmp_path)
    over = Plan(7, 200.0, 1.0, 5.0, 0.0, freeze=0.6, overdraft=0.4, needs_shorter=True, freeze_at=204.0, excess=1.6)
    less = job._shortened(over, 4.5)                    # 0.5 s shorter: 0.1 s of freeze (min_freeze), the rest overdraft
    assert less.freeze == job.planner.s.min_freeze and less.overdraft == pytest.approx(0.35) and less.needs_shorter
    inside = job._shortened(over, 3.3)                  # 1.7 s shorter: inside its window, not long
    assert (inside.freeze, inside.overdraft, inside.excess, inside.needs_shorter) == (0.0, 0.0, 0.0, False)
    fits = Plan(8, 200.0, 1.0, 2.0, 0.0)               # ends at 202.0, the next line at 203.0
    a_bit = job._shortened(fits, 2.5, next_start=203.0)  # ends at 202.5: still inside (203.0 - guard)
    assert (a_bit.wall, a_bit.freeze, a_bit.excess, a_bit.needs_shorter) == (2.5, 0.0, 0.0, False)
    much = job._shortened(fits, 3.5, next_start=203.0)   # ends at 203.5: 0.65 s past it
    assert much.freeze == 0.0 and much.excess == pytest.approx(0.65) and much.needs_shorter
    frozen = job._shortened(over, 6.0)                  # longer than a take that needed a freeze: no more freeze
    assert frozen.freeze == over.freeze and frozen.overdraft == pytest.approx(1.4) and frozen.needs_shorter


# ---- Claude failures ----------------------------------------------------------------------------------------------------
async def test_a_claude_failure_holds_translation_says_what_to_do_and_recovers(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.5)
    fake_cli.env.setenv("FAKE_MODE", "not_signed_in")
    events: list[dict] = []
    job = bare_job(tmp_path, translator=fake_cli.factory, events=events)
    run = asyncio.create_task(job.run())
    try:
        await until(lambda: any(e["type"] == "claude_error" for e in events))
        err = next(e for e in events if e["type"] == "claude_error")
        assert err["kind"] == "not_signed_in" and err["retryIn"] >= 0 and "codex login" in err["message"]
        await until(lambda: job.doc["status"] == "waiting")  # the job waits out the hold
        assert not (job.render_dir / "takes.jsonl").exists() and not job.skipped  # lines wait, never skipped
        fake_cli.env.setenv("FAKE_MODE", "ok")  # the user signs in
        assert await asyncio.wait_for(run, 60) == "done"
    finally:
        run.cancel()
    assert kinds(events, "claude_ok") == ["claude_ok"] and not job.skipped
    assert job.doc["brief"]["version"] == 1 and job.doc["coverage"]["C"] == job.doc["coverage"]["lines"]


async def test_a_usage_limit_holds_translation_until_it_resets(tmp_path):
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    err = ClaudeCLIError("usage_limit", "You've hit your session limit", limit="session",
                         resets_at=time.time() + 3600)
    await job._claude_failed(err)
    (sent,) = events
    assert sent == {"type": "claude_error", "kind": "usage_limit", "message": "You've hit your session limit",
                    "limit": "session", "resetsAt": err.resets_at, "retryIn": sent["retryIn"], "videoId": job.video_id}
    assert 3590 <= sent["retryIn"] <= 3610 and job._claude_hold - time.monotonic() > 3500 and job._held()
    await job._claude_failed(err)  # the same failure from another call: said once
    assert len(events) == 1


async def test_the_first_call_through_claude_clears_the_banner_once_per_job(tmp_path):
    """hello (or the job before) may have left a Claude problem on screen: the first call of this job that goes
    through clears it. Later ones say nothing until something fails again."""
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    await job._claude_ok(time.monotonic())
    await job._claude_ok(time.monotonic())
    assert kinds(events, "claude_ok") == ["claude_ok"] and len(events) == 1
    await job._claude_failed(ClaudeCLIError("not_signed_in", "Not logged in"))
    await job._claude_ok(time.monotonic())
    assert [e["type"] for e in events] == ["claude_ok", "claude_error", "claude_ok"]


async def test_a_call_asked_before_a_usage_limit_doesnt_clear_it(tmp_path):
    """A scene call under way when another hits the limit comes back fine: the banner and the hold stay."""
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    asked = time.monotonic()
    err = ClaudeCLIError("usage_limit", "You've hit your session limit", limit="session",
                         resets_at=time.time() + 3 * 3600)
    await job._claude_failed(err)
    await job._claude_ok(asked)
    assert [e["type"] for e in events] == ["claude_error"] and job._claude_hold - time.monotonic() > 3 * 3600 - 60
    await job._claude_failed(err)  # still the same failure: said once
    assert len(events) == 1
    await job._claude_ok(time.monotonic())  # a call asked after it (once the hold is over) goes through
    assert [e["type"] for e in events] == ["claude_error", "claude_ok"]


class BriefFailsOnce(MockClaude):
    """The mock, but the first brief call fails with `kind`."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "brief" and not any(c["call"] == "brief" for c in self.calls):
            self.calls.append({"call": call, "message": json.loads(prompt), "failed": True})
            raise ClaudeCLIError(self.kind, "no brief")
        return super().ask(system, prompt, schema, call, **kw)


@pytest.mark.parametrize(("kind", "said"), [("bad_output", False), ("timeout", False), ("not_signed_in", True)])
async def test_a_bad_or_slow_brief_reply_is_only_retried(tmp_path, monkeypatch, kind, said):
    """A brief reply that failed its checks or timed out is asked for again (up to BRIEF_TRIES times), without a banner
    or a hold. What the user must fix is said, and held, as for a scene."""
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.05)
    cli = BriefFailsOnce(kind)
    events: list[dict] = []

    def make(cache_dir, video_id, brief, **kw) -> ClaudeTranslator:
        return ClaudeTranslator(cache_dir, video_id, brief, cli=cli, **kw)

    job = bare_job(tmp_path, translator=make, events=events)
    assert await job.run() == "done"
    briefs = [c.get("failed", False) for c in cli.calls if c["call"] == "brief"]
    assert briefs == [True, False] and job.doc["brief"]["version"] == 1 and job.doc["brief"]["gaveUp"] is None
    assert ("claude_error" in kinds(events, "claude_error")) == said
    assert any(e["type"] == "render" and e["status"] == "waiting" for e in events) == said


async def test_a_rephrase_under_way_is_dropped_when_claude_fails_and_none_is_asked_while_held(tmp_path):
    """A stall or a usage limit holds new calls; a scene's rephrase under way would otherwise sit in the translator's
    slots through the hold. Its lines settle as voiced, flagged long."""
    job = bare_job(tmp_path)
    job.tr = MockSceneTranslator(tmp_path / "c", "v", a_brief(), delay=5.0)
    st = UnitState(SourceUnit(7, "S1", 300.0, 303.0, "Please remember to bring the tickets."), None, speech_s=3.0,
                   scene=4, voiced=True, tier="full")
    st.line = LineResult(7, {"full": Wording("దయచేసి టికెట్లు తీసుకురావడం మర్చిపోకండి మరి.")})
    st.plan, st.take_s = Plan(7, 300.0, 1.2, 5.0, 0.0, needs_shorter=True, excess=1.0), 6.0
    put(job, st)
    sc = Scene(4, [st], 0)
    asked = asyncio.create_task(job._rephrased(sc, [st]))
    await until(lambda: any(c["call"] == "rephrase" for c in job.tr.cli.calls))  # under way
    await job._claude_failed(ClaudeCLIError("stalled", "Claude Code didn't start"))
    assert await asyncio.wait_for(asked, 2.0) == ({}, True)  # dropped at once, its slot free
    n = len(job.tr.cli.calls)
    assert await job._rephrased(sc, [st]) == ({}, True) and len(job.tr.cli.calls) == n  # while held, no call


# ---- the whole video on the mock backend ------------------------------------------------------------------------------
async def test_the_demo_is_translated_in_scenes_and_voiced(tmp_path):
    events: list[dict] = []
    job = bare_job(tmp_path, events=events)
    assert await job.run() == "done"
    scenes = trace(job, "scene")
    assert scenes and all(e["b"] - e["a"] <= SCENE_MAX_S and e["lines"] <= SCENE_MAX_UNITS for e in scenes)
    assert scenes[0]["scene"] == 1 and scenes[0]["context_te"] == 0  # the first scene has nothing before it
    units = trace(job, "unit")
    assert len(units) == len(job.units) and all(u["tier"] and u["model"] == "mock" for u in units)
    assert not job.skipped and not kinds(events, "error", "claude_error")
    assert kinds(events, "claude_ok") == ["claude_ok"]  # the first call through clears any Claude banner, once


async def test_with_every_claude_call_slow_the_dub_loop_voices_on_and_never_calls_the_translator(tmp_path, fake_cli,
                                                                                                 monkeypatch):
    """ARCHITECTURE §4.9, OFFLINE-RENDER §2.9: with every call delayed, lines already translated are voiced while later
    scenes wait on Claude, and no call on the translator is ever made from the dub loop's task."""
    fake_cli.env.setenv("FAKE_DELAY", "1.0")
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 30.0)  # short scenes: about seven for the three-minute demo
    callers: list[tuple[str, asyncio.Task | None]] = []

    def factory(*a, **k):
        tr = fake_cli.factory(*a, **k)
        for name in ("translate", "submit", "make_brief", "review", "use_brief", "cancel"):
            def spy(*args, _f=getattr(tr, name), _n=name, **kw):
                callers.append((_n, asyncio.current_task()))
                return _f(*args, **kw)
            setattr(tr, name, spy)
        return tr

    events: list[dict] = []
    job = bare_job(tmp_path, SlowTTS(pause=0.05), translator=factory, events=events)
    voiced: list[tuple[float, int, asyncio.Task | None]] = []
    voice_line = job._voice_line

    async def watched(st):
        voiced.append((time.time(), st.unit.id, asyncio.current_task()))
        return await voice_line(st)

    job._voice_line = watched
    assert await job.run() == "done"
    loop = {task for _, _, task in voiced}
    assert callers and len(loop) == 1 and not loop & {task for _, task in callers}
    calls = [c for c in fake_cli.calls() if c["call"] == "scene"]
    # lines of earlier scenes were voiced while a later scene's call was still waiting on Claude
    busy = [i for t, i, _ in voiced if any(c["t0"] < t < c["t1"] and i not in c["ids"] for c in calls)]
    assert len(busy) >= 3
    rows = _read_rows(job.render_dir / "takes.jsonl")
    assert rows and max(r["cost"]["lock_wait_s"] for r in rows) < 1.0  # nobody holds the GPU while Claude thinks
    overlap = max(sum(1 for c in calls if c["t0"] <= x["t0"] < c["t1"]) for x in calls)
    assert 2 <= overlap <= render.LANES  # scene calls run side by side, one a lane
    assert not kinds(events, "error", "claude_error")


@pytest.mark.parametrize("script", ["telugu", "latin"])
async def test_the_tts_reads_telugu_script_or_for_the_ab_the_latin_rebuild(tmp_path, monkeypatch, script):
    measured: list[str] = []
    observe = DurationEstimator.observe
    monkeypatch.setattr(DurationEstimator, "observe",
                        lambda self, text, key, seconds: (measured.append(text), observe(self, text, key, seconds))[1])
    tts = SlowTTS()
    job = bare_job(tmp_path, tts, tts_script=script)
    assert await job.run() == "done"
    # the voice's pace is learned on the Telugu script (what lengths are counted on), whatever the TTS read
    assert measured and not any(ch.isascii() and ch.isalpha() for t in measured for ch in t)
    welcome = next(t for t in tts.texts if "స్వాగతం" in t)
    if script == "telugu":
        assert "ఛానల్" in welcome and not any(ch.isascii() and ch.isalpha() for t in tts.texts for ch in t)
    else:
        assert "channel కి" in welcome and "ఛానల్" not in welcome
    sent = next(e for e in trace(job, "unit") if "స్వాగతం" in e["telugu"])
    assert sent["telugu"] in tts.texts and ("channel" in sent["telugu"]) == (script == "latin")  # what was said
    manifest = json.loads((job.render_dir / "manifest.json").read_text())
    assert manifest["settings"]["ttsScript"] == script  # the subtitles say it in Telugu script either way
    assert not any(ch.isascii() and ch.isalpha() for x in manifest["lines"] for ch in x["telugu"])


async def test_the_brief_starts_from_the_metadata_and_the_whole_videos_talk_shares(tmp_path):
    job = bare_job(tmp_path)
    assert await job.run() == "done"
    meta = job.tr.brief.meta
    assert (meta.title, meta.channel, meta.tags) == ("Demo video", "Maata demo", ("demo",))
    assert meta.chapters == ((0.0, "Start"),) and meta.description
    assert {sid for sid, _ in meta.talk_shares} == set(job.registry.speakers)
    assert sum(x for _, x in meta.talk_shares) == pytest.approx(1.0)


# ---- the coverage review (§4.6) ---------------------------------------------------------------------------------------
class Reviewing(MockClaude):
    """The mock, with review calls that can be held (`gate`, until set or cancelled), fail the first `fails` times with
    `fail`, or class lines by their English (`classes`: the start of the English -> (class, missing words)); a
    re-translation says one word more (సరిగ్గా, "exactly", which no demo line has)."""

    def __init__(self, gate: threading.Event | None = None, fail: str | None = None, fails: int = 1,
                 classes: dict[str, tuple[str, list[str]]] | None = None) -> None:
        super().__init__()
        self.gate, self.fail, self.fails, self.classes = gate, fail, fails, classes or {}

    def ask(self, system, prompt, schema=None, call="text", *, effort=None, cancel=None, tags=None):
        if call == "review":
            if self.gate is not None:
                while not self.gate.wait(0.05):
                    if cancel is not None and cancel.is_set():
                        self.calls.append({"call": "review", "cancelled": True, "message": json.loads(prompt)})
                        raise ClaudeCLIError("cancelled", "The Claude call was cancelled")
            if self.fail and self.fails > 0:
                self.fails -= 1
                self.calls.append({"call": "review", "message": json.loads(prompt), "failed": True})
                raise ClaudeCLIError(self.fail, "Not logged in · Please run /login")
        reply = super().ask(system, prompt, schema, call, effort=effort, cancel=cancel, tags=tags)
        if call == "review":
            en = {x["id"]: x["en"] for x in json.loads(prompt)["lines"]}
            te = {x["id"]: x["te"] for x in json.loads(prompt)["lines"]}
            for x in reply.data["lines"]:
                cls, missing = next((v for k, v in self.classes.items() if en[x["id"]].startswith(k)), ("C", []))
                if te[x["id"]].startswith("సరిగ్గా "):
                    cls, missing = "C", []  # the explicit reviewer recognizes this fixture's exact correction
                x.update({"class": cls, "missing": missing})
        elif call == "retranslate":
            for x in reply.data["lines"]:
                for tier in ("full", "fuller", "concise", "very_concise"):
                    if tier in x:
                        w = x[tier]
                        x[tier] = {"spoken": "సరిగ్గా " + w["spoken"],
                                   "english": w["english"]}
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
    job = bare_job(tmp_path, translator=factory)
    run = asyncio.create_task(job.run())
    try:
        await until(lambda: clis and job.scenes and any(c["call"] == "scene" and c["message"]["scene"] == 1
                                                        for c in clis[0].calls))
        await asyncio.sleep(0.5)  # the scene call is back; its review is held
        first = job.scenes[0].lines
        assert first and all(st.line is None and st.telugu is None for st in first)
        assert not job.scenes[0].ready and not (job.render_dir / "takes.jsonl").exists()  # the dub loop waits
        gate.set()
        assert await asyncio.wait_for(run, 60) == "done"
    finally:
        gate.set()
        run.cancel()
    units = trace(job, "unit")
    assert units and all(u["coverage"]["class"] == "C" and u["coverage"]["by"] == "review" for u in units)
    assert all(u["coverage"]["tier"] in u["tiers"] for u in units)
    assert all(c["effort"] == "low" for c in reviews(clis[0]))
    (scene,) = [e for e in trace(job, "scene") if e["scene"] == 1]
    assert scene["coverage"]["C"] == scene["translated"] and scene["unreviewed"] == 0


async def test_a_line_the_review_finds_a_phrase_missing_from_is_retranslated_and_the_better_one_voiced(tmp_path):
    factory, clis = reviewing(classes={"Every line starts": ("P", ["exactly"])})
    job = bare_job(tmp_path, translator=factory)
    assert await job.run() == "done"
    redo = [c for c in clis[0].calls if c["call"] == "retranslate"]
    assert redo and redo[0]["message"]["lines"][0]["missing"] == ["exactly"] and redo[0]["effort"] == "low"
    unit = next(e for e in trace(job, "unit") if e["source"].startswith("Every line"))
    assert "సరిగ్గా" in unit["telugu"].split()  # explicitly re-reviewed and actually voiced
    assert unit["coverage"] == {"class": "C", "by": "review", "tier": "full", "first": "P", "missing": [],
                                "added": [], "error": None}
    others = [e for e in trace(job, "unit") if not e["source"].startswith("Every line")]
    assert others and all(e["coverage"]["class"] == "C" and "సరిగ్గా" not in e["telugu"].split() for e in others)


async def test_a_review_that_fails_while_claude_is_held_leaves_its_lines_voiced_unreviewed(tmp_path, monkeypatch):
    """The review fails with a problem the user must fix: it is said, the job holds new calls, and the scene's lines
    are voiced unreviewed meanwhile (flagged); the review is asked again once the hold is over."""
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 2.0)
    factory, clis = reviewing(fail="not_signed_in")
    events: list[dict] = []
    job = bare_job(tmp_path, translator=factory, events=events)
    rows: list[dict] = []
    keep = job._keep

    def kept(st, said, plan, wording, key, n, cost, **extra):
        keep(st, said, plan, wording, key, n, cost, **extra)
        rows.append({"unit": st.unit.id, "class": dubber.voiced_class(st.line.coverage, st.tier),
                     "held": job._held()})

    job._keep = kept
    assert await job.run() == "done"
    assert any(e["type"] == "claude_error" and e["kind"] == "not_signed_in" for e in events)  # said, and held
    failed = next(c for c in reviews(clis[0]) if c.get("failed"))
    ids = {x["id"] for x in failed["message"]["lines"]}
    voiced = [r for r in rows if r["unit"] in ids]
    assert voiced and all(r["class"] == "unreviewed" and r["held"] for r in voiced)  # voiced while held, unreviewed
    assert [e["error"] for e in trace(job, "review")][0] == "not_signed_in"


async def test_a_resume_serves_the_stored_classes_and_reviews_nothing_again(tmp_path):
    first, clis = reviewing()
    job = bare_job(tmp_path, translator=first)
    assert await job.run() == "done"
    seen = {x["id"] for c in reviews(clis[0]) for x in c["message"]["lines"]}
    assert seen
    again, clis2 = reviewing()
    job = bare_job(tmp_path, translator=again)
    assert await job.run() == "done"
    assert not reviews(clis2[0])
    assert all(job.units[i].line.coverage.cls == "C" and job.units[i].cache_hit for i in seen)


async def test_units_jsonl_has_each_lines_class_speech_fill_and_required_rate(tmp_path):
    job, st = band_case(tmp_path, {"full": 4.6}, speech=4.0)
    st.next_start = 104.2  # the next line follows closely: the take is sped up to fit
    st.line.coverage = Coverage("m", ("really",), tier="full")
    plan = await voice(job, st)
    (unit,) = trace(job, "unit")
    assert unit["coverage"] == {"class": "m", "by": "review", "tier": "full", "first": None, "missing": ["really"],
                                "added": [], "error": None}
    assert unit["tier"] == "full" and unit["coverage_voiced"] == "m"  # the class of the wording voiced
    assert plan.rate > 1.0
    assert unit["required_rate"] == round(st.take_s / 4.0, 3) > 1.1  # what it needs to fill the speech time exactly
    assert unit["speech_fill"] == round(st.take_s / plan.rate / 4.0, 3) < unit["required_rate"]  # as played


# ---- Claude's state for the UI (hello) --------------------------------------------------------------------------------
def test_health_says_installed_version_signed_in_and_models(tmp_path, fake_cli, monkeypatch):
    from maata_engine import codex_cli

    ok = codex_cli.CodexCLI(tmp_path, binary=str(fake_cli.exe)).health()
    assert ok["models"] == ["gpt-6-luna"]
    assert {k: v for k, v in ok.items() if k != "models"} == {
        "installed": True, "version": "0.160.0", "signedIn": True, "model": "gpt-6-luna",
        "provider": "codex", "problem": None, "message": ""}
    fake_cli.env.setenv("FAKE_SIGNED_IN", "0")
    out = codex_cli.CodexCLI(tmp_path, binary=str(fake_cli.exe)).health()
    assert out["signedIn"] is False and out["problem"] == "not_signed_in" and "codex login" in out["message"]
    monkeypatch.setattr(codex_cli, "find_binary", lambda explicit=None: None)
    gone = codex_cli.CodexCLI(tmp_path).health()
    assert gone["installed"] is False and gone["problem"] == "missing" and "Codex" in gone["message"]


async def test_hello_carries_claude_health_except_on_the_demo_engine(tmp_path, fake_cli):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from maata_engine.server import Engine

    fake_cli.env.setenv("MAATA_CODEX_BIN", str(fake_cli.exe))  # never the real CLI in a test
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
    assert hello["claude"]["installed"] and hello["claude"]["version"] == "0.160.0" and hello["claude"]["signedIn"]
    assert time.monotonic() - t0 < 2.5 and stuck["claude"]["problem"] == "stalled"  # hello isn't held up by it
