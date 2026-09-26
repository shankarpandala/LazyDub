"""The Claude scene translator (ARCHITECTURE §4) against a fake `claude` that answers from the message it is given.

FAKE_MODE is a comma-separated script: the n-th `-p` call plays the n-th mode (the last one repeats). A review call
finds every line complete unless FAKE_REVIEW (JSON: id -> [class, missing, error]) says otherwise. All English and
Telugu here is original test text.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import stat
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from maata_engine.backends import claude_translator as ct
from maata_engine.backends.base import (Brief, Coverage, GlossaryEntry, LineResult, LineSpec, SceneRequest, SpeakerNote,
                                        VideoMeta, Wording)
from maata_engine.backends.claude_translator import ClaudeTranslator, brief_v0, parse_brief
from maata_engine.backends.mock import MockClaude, MockSceneTranslator
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.qa.validators import script_problems
from maata_engine.text.akshara import count_telugu
from maata_engine.text import scene_prompt as sp

FAKE = r"""#!@PYTHON@
import fcntl, json, os, signal, sys, time
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "2.1.281 (Claude Code)"))
    sys.exit(0)
def on_int(sig, _):
    open(os.environ["FAKE_SIGNALS"], "a").write("SIGINT\n")
    time.sleep(float(os.environ.get("FAKE_STOP_DELAY", "0")))  # a CLI that takes a while to stop
    sys.exit(130)
signal.signal(signal.SIGINT, on_int)
prompt = sys.stdin.read()
path = os.environ["FAKE_RECORD"]
with open(path, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    n = sum(1 for _ in f)
    f.write(json.dumps({"argv": argv, "stdin": prompt, "pid": os.getpid(), "t0": time.time()}) + "\n")
modes = os.environ.get("FAKE_MODE", "ok").split(",")
mode = modes[min(n, len(modes) - 1)]
msg = json.loads(prompt)
model = argv[argv.index("--model") + 1]

def finish(data=None, text="", error=None):
    print(json.dumps({"type": "system", "subtype": "init", "model": model}), flush=True)
    res = {"type": "result", "subtype": "success", "is_error": bool(error), "result": error or text,
           "usage": {"input_tokens": 10, "output_tokens": 20}, "modelUsage": {model: {"outputTokens": 20}}}
    if data is not None:
        res["structured_output"] = data
    print(json.dumps(res, ensure_ascii=False), flush=True)
    open(os.environ["FAKE_ENDS"], "a").write(json.dumps({"pid": os.getpid(), "t1": time.time()}) + "\n")
    sys.exit(1 if error else 0)

if mode == "slow":
    time.sleep(30)
if mode == "pause":
    time.sleep(float(os.environ.get("FAKE_PAUSE", "0.5")))
if mode == "not_signed_in":
    finish(error="Not logged in · Please run /login")
if mode == "session_limit":
    finish(error="You've hit your session limit · resets 3pm (Europe/London)")
if mode == "garbage_json":
    finish(text="Here you go: no JSON today")

if msg.get("call") == "review":
    verdicts = json.loads(os.environ.get("FAKE_REVIEW", "{}"))
    out = []
    for line in msg["lines"]:
        cls, missing, error = verdicts.get(str(line["id"]), ["C", [], "none"])
        out.append({"id": line["id"], "class": cls, "missing": missing, "added": [], "error": error})
    finish({"lines": out[:-1] if mode == "drop_last" else out})

if msg.get("call") == "brief":
    ids = [s["id"] for s in msg["video"].get("speakers", [])]
    finish({"topic": "kites", "register": "casual", "speakers": [{"id": i, "gender": "female"} for i in ids],
            "glossary": [{"term": "kite", "spoken": "కైట్", "keep_english": True},
                         {"term": "Bad", "spoken": "bad", "keep_english": True}],
            "idioms": ["high as a kite"] if "previous" in msg else []})

def words_for(line):
    x = "".join("మకగచజటడతపబ"[int(d)] for d in str(line["id"])) + "ము"
    words, english = ["వాక్యం", x, "ఇది."], []
    if "kite" in line["en"].lower():
        words, english = ["కైట్"] + words, [(0, "kite")]
    if msg["call"] in ("retranslate", "rephrase") and mode != "no_longer":
        words, english = ["మళ్ళీ"] + words, [(i + 1, e) for i, e in english]
    return words, english

def wording(words, english):
    return {"spoken": " ".join(words), "english": [{"i": i, "en": e} for i, e in english if i < len(words)]}

def answer(line):
    words, english = words_for(line)
    out = {"id": line["id"], "full": wording(words, english), "delivery": {"emotion": "neutral", "energy": "mid"}}
    want = line.get("want", [])
    if "concise" in want:
        out["concise"] = wording(words[:-1], english)
    if "very_concise" in want:
        out["very_concise"] = wording(words[:-2], english)
    if "fuller" in want:
        out["fuller"] = wording(["అంటే"] + words, [(i + 1, e) for i, e in english])
    return out

lines = [answer(line) for line in msg["lines"]]
if mode == "drop_last":
    lines = lines[:-1]
if mode == "latin_first":
    lines[0]["full"]["spoken"] = "ఇది test వాక్యం."
if mode == "cyrillic_first":
    lines[0]["full"]["spoken"] = "ఇది క్ల\u0430స్ వాక్యం."
if mode == "dupe_first":
    lines.append(lines[0])
if mode == "extra":
    lines.append({**lines[0], "id": 999})
if mode == "full_only":
    lines = [{k: v for k, v in x.items() if k not in ("fuller", "concise", "very_concise")} for x in lines]
if mode == "reword":  # a fit that rewrites full too, with its own delivery
    for x in lines:
        x["full"] = wording(["వేరే"] + x["full"]["spoken"].split(), [(e["i"] + 1, e["en"]) for e in x["full"]["english"]])
        x["delivery"] = {"emotion": "sad", "energy": "low", "emphasis": [0]}
data = {"lines": lines}
if mode == "glossary":
    data["glossary_additions"] = [{"term": "Kite", "spoken": "కైట్", "keep_english": True},
                                  {"term": "string", "spoken": "string"}]
finish(data)
"""

META = VideoMeta("Kites over the hill", "Open Sky", talk_shares=(("S1", 0.6), ("S2", 0.4)))
ENGLISH = ["The kite climbs over the hill.", "Wind pushes it higher.", "Everyone watches from the field.",
           "Then the string snaps.", "It drifts toward the trees."]


def specs(ids=(1, 2), want=("full",), speaker="S1", **kw) -> tuple[LineSpec, ...]:
    return tuple(LineSpec(i, speaker, ENGLISH[(i - 1) % len(ENGLISH)], 2.0 * i, 2.0 * i + 1.8, 1.6, 9.0, want, **kw)
                 for i in ids)


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.replace("@PYTHON@", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    record, ends, signals = tmp_path / "record.jsonl", tmp_path / "ends.jsonl", tmp_path / "signals"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("FAKE_ENDS", str(ends))
    monkeypatch.setenv("FAKE_SIGNALS", str(signals))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_REVIEW", raising=False)
    events: list[dict] = []

    def make(brief: Brief | None = None, video: str = "vid1", **kw) -> ClaudeTranslator:
        tr = ClaudeTranslator(tmp_path / "cache", video, brief or brief_v0(META), binary=str(exe), trace=events.append,
                              **kw)
        tr.cli.timeout, tr.cli.startup_timeout, tr.cli.grace, tr.cli.backoff = 40.0, 20.0, 1.0, 0.01
        return tr

    def calls() -> list[dict]:
        if not record.exists():
            return []
        out = [json.loads(x) for x in record.read_text().splitlines()]
        for c in out:
            c["msg"] = json.loads(c["stdin"])
            c["arg"] = lambda flag, argv=c["argv"]: argv[argv.index(flag) + 1] if flag in argv else None
        return out

    async def started(n: int = 1) -> None:
        for _ in range(400):
            if len(calls()) >= n:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("the fake CLI never started")

    return SimpleNamespace(make=make, calls=calls, events=events, env=monkeypatch, tmp=tmp_path, started=started,
                           ends=ends, signals=signals)


def cli_events(events, **match):
    return [e for e in events if e["event"] == "claude" and all(e.get(k) == v for k, v in match.items())]


# ---- the call --------------------------------------------------------------------------------------------------------
async def test_scene_call_defaults_and_message(fake):
    tr = fake.make()
    res = await tr.translate(SceneRequest(5, specs(), context_before=(("Hello there.", "నమస్కారం."),),
                                          context_after_en=("Next line.",)))
    (call,) = fake.calls()
    assert call["arg"]("--model") == "claude-opus-5-5" and call["arg"]("--effort") == "medium"
    assert call["arg"]("--fallback-model") == "claude-sonnet-5"
    assert call["arg"]("--system-prompt") == sp.system_prompt(brief_v0(META))
    assert json.loads(call["arg"]("--json-schema")) == sp.SCENE_SCHEMA
    assert call["stdin"] == sp.user_message(SceneRequest(5, specs(), "scene", (("Hello there.", "నమస్కారం."),),
                                                         ("Next line.",)))
    assert set(res.lines) == {1, 2} and res.skipped == {} and res.calls == 1
    one = res.lines[1]
    assert one.full == Wording("కైట్ వాక్యం కము ఇది.", ((0, "kite"),))
    assert one.model == "claude-opus-5-5" and one.brief_version == 0 and not one.cached and one.flags == ()


async def test_the_fallback_is_used_only_when_the_cli_can_run_it(fake):
    fake.env.setenv("FAKE_VERSION", "2.1.250 (Claude Code)")
    await fake.make().translate(SceneRequest(1, specs((1,))))
    assert "--fallback-model" not in fake.calls()[-1]["argv"]


async def test_usage_is_recorded_per_cli_call_and_per_request(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(5, specs()))
    (usage,) = cli_events(fake.events)
    assert usage["call"] == "scene" and usage["scene"] == 5 and usage["lines"] == 2 and usage["output_tokens"] == 20
    assert usage["model"] == "claude-opus-5-5" and usage["effort"] == "medium"
    (ev,) = [e for e in fake.events if e["event"] == "translate"]
    assert {k: ev[k] for k in ("call", "scene", "lines", "cached", "answered", "skipped", "calls", "model", "prompt_hash",
                               "brief", "error", "dropped")} == \
        {"call": "scene", "scene": 5, "lines": 2, "cached": 0, "answered": 2, "skipped": [], "calls": 1,
         "model": "claude-opus-5-5", "prompt_hash": sp.PROMPT_HASH, "brief": 0, "error": None, "dropped": False}


# ---- the line cache --------------------------------------------------------------------------------------------------
async def test_a_rewatch_makes_no_call(fake):
    await fake.make().translate(SceneRequest(1, specs((1, 2, 3))))
    rows = (fake.tmp / "cache" / "vid1" / "lines.jsonl").read_text().splitlines()
    assert len(rows) == 3 and all(json.loads(r)["prompt_hash"] == sp.PROMPT_HASH for r in rows)
    again = await fake.make().translate(SceneRequest(9, specs((3, 1, 2))))  # another run, another scene cut
    assert len(fake.calls()) == 1 and again.calls == 0
    assert all(x.cached for x in again.lines.values()) and again.lines[1].full.spoken == "కైట్ వాక్యం కము ఇది."
    assert again.lines[1].model == "claude-opus-5-5" and again.lines[1].brief_version == 0


@pytest.mark.parametrize("change", ["speaker", "style", "model", "prompt", "video"])
async def test_the_cache_misses_when_its_key_changes(fake, monkeypatch, change):
    await fake.make().translate(SceneRequest(1, specs((1,))))
    kw, lines = {}, specs((1,))
    if change == "speaker":
        lines = specs((1,), speaker="S2")
    elif change == "style":
        kw["style"] = "formal"
    elif change == "model":
        kw.update(model="claude-sonnet-5", fallback_model="claude-opus-5-5")
    elif change == "prompt":
        monkeypatch.setattr(ct, "PROMPT_HASH", "0123456789ab")
    else:
        kw["video"] = "vid2"
    res = await fake.make(**kw).translate(SceneRequest(1, lines))
    assert len(fake.calls()) == 2 and not res.lines[1].cached


async def test_a_line_made_from_metadata_alone_is_made_again_once_a_brief_exists_but_v2_reuses_v1s(fake):
    await fake.make().translate(SceneRequest(1, specs((1,))))
    assert (await fake.make().translate(SceneRequest(1, specs((1,))))).lines[1].cached  # still v0: served
    tr = fake.make()
    tr.use_brief(Brief(1, META, topic="kites"))
    res = await tr.translate(SceneRequest(1, specs((1,))))
    assert len(fake.calls()) == 2 and not res.lines[1].cached and res.lines[1].brief_version == 1
    tr = fake.make()
    tr.use_brief(Brief(2, META, topic="kites"))
    res = await tr.translate(SceneRequest(1, specs((1,))))
    assert len(fake.calls()) == 2 and res.lines[1].cached and res.lines[1].brief_version == 1


async def test_a_call_records_the_brief_it_was_sent_with_even_if_a_swap_lands_meanwhile(fake):
    fake.env.setenv("FAKE_MODE", "pause")
    tr = fake.make()
    task = tr.submit(SceneRequest(1, specs((1,))))
    await fake.started()
    tr.use_brief(Brief(1, META, topic="kites"))
    res = await task
    assert fake.calls()[0]["arg"]("--system-prompt") == sp.system_prompt(brief_v0(META))
    assert res.lines[1].brief_version == 0
    (row,) = [json.loads(r) for r in (fake.tmp / "cache" / "vid1" / "lines.jsonl").read_text().splitlines()]
    assert row["brief"] == 0


def test_a_row_written_after_a_torn_line_survives_the_next_load(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b", "li', encoding="utf-8")  # a crash mid-write
    rows = ct._Jsonl(path)
    assert set(rows.rows) == {"a"}
    rows.put({"key": "c"})
    rows.put({"key": "d"})
    assert set(ct._Jsonl(path).rows) == {"a", "c", "d"}


async def test_cached_lines_are_context_only_for_the_rest_of_the_scene(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1, 3))))
    await tr.translate(SceneRequest(1, specs((1, 2, 3))))
    second = fake.calls()[-1]["msg"]
    assert [x["id"] for x in second["lines"]] == [2]
    assert [(x["id"], x["te"]) for x in second["context_done"]] == [(1, "కైట్ వాక్యం కము ఇది."), (3, "వాక్యం చము ఇది.")]


async def test_a_tier_the_cache_lacks_goes_to_one_fit_call(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1, 2))))
    res = await tr.translate(SceneRequest(1, specs((1, 2), want=("full", "concise"))))
    fit = fake.calls()[-1]
    assert len(fake.calls()) == 2 and fit["msg"]["call"] == "fit" and fit["arg"]("--effort") == "low"
    assert [(x["id"], x["want"], x["current"]) for x in fit["msg"]["lines"]] == \
        [(1, ["concise"], "కైట్ వాక్యం కము ఇది."), (2, ["concise"], "వాక్యం గము ఇది.")]
    assert fit["msg"]["lines"][0]["overflow_aksharas"] == round(count_telugu("కైట్ వాక్యం కము ఇది.") - 9.0) == -2
    assert res.lines[1].tiers["concise"].spoken == "కైట్ వాక్యం కము" and "full" in res.lines[1].tiers
    third = await tr.translate(SceneRequest(1, specs((1, 2), want=("full", "concise"))))
    assert len(fake.calls()) == 2 and all(x.cached for x in third.lines.values())


async def test_a_fit_takes_only_the_tiers_it_was_asked_for_and_never_rewrites_full(fake):
    tr = fake.make()
    first = (await tr.translate(SceneRequest(1, specs((1,))))).lines[1]
    fake.env.setenv("FAKE_MODE", "ok,reword")
    res = await tr.translate(SceneRequest(1, specs((1,), want=("full", "concise"))))
    assert fake.calls()[-1]["msg"]["call"] == "fit"
    line = res.lines[1]
    assert line.full == first.full == Wording("కైట్ వాక్యం కము ఇది.", ((0, "kite"),))
    assert line.delivery == first.delivery and line.tiers["concise"].spoken == "కైట్ వాక్యం కము"
    again = (await fake.make().translate(SceneRequest(1, specs((1,), want=("full", "concise"))))).lines[1]
    assert again.cached and again.full == first.full and again.delivery == first.delivery and len(fake.calls()) == 2


async def test_the_sessions_fit_of_a_chosen_shorter_tier_adds_that_tier_only(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1,), want=("full", "concise"))))
    fake.env.setenv("FAKE_MODE", "ok,reword")
    fit = replace(specs((1,), want=("very_concise",))[0], current="కైట్ వాక్యం కము", overflow=3.0)
    res = await tr.translate(SceneRequest(1, (fit,), "fit"))
    msg = fake.calls()[-1]["msg"]["lines"][0]
    assert (msg["current"], msg["overflow_aksharas"], msg["want"]) == ("కైట్ వాక్యం కము", 3, ["very_concise"])
    tiers = {k: w.spoken for k, w in res.lines[1].tiers.items()}
    assert tiers == {"full": "కైట్ వాక్యం కము ఇది.", "concise": "కైట్ వాక్యం కము", "very_concise": "కైట్ వాక్యం"}
    assert res.skipped == {} and res.lines[1].brief_version == 0


async def test_a_sessions_fit_that_fails_keeps_the_cached_line_and_never_skips_it(fake, caplog):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1,))))
    fake.env.setenv("FAKE_MODE", "ok,latin_first")
    fit = replace(specs((1,), want=("concise",))[0], current="కైట్ వాక్యం కము ఇది.", overflow=2.0)
    with caplog.at_level("INFO", logger="maata.translate"):
        res = await tr.translate(SceneRequest(1, (fit,), "fit"))
    assert len(fake.calls()) == 4 and res.skipped == {}  # the fit asked all, again, then per line
    assert res.lines[1].full.spoken == "కైట్ వాక్యం కము ఇది." and "fit failed" in res.lines[1].flags
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


async def test_a_fit_with_no_line_to_fold_into_is_answered_but_not_cached(fake):
    tr = fake.make()
    fit = replace(specs((1,), want=("concise",))[0], current="కైట్ వాక్యం కము ఇది.", overflow=2.0)
    res = await tr.translate(SceneRequest(1, (fit,), "fit"))
    assert res.lines[1].tiers["concise"].spoken == "కైట్ వాక్యం కము"
    assert not (fake.tmp / "cache" / "vid1" / "lines.jsonl").exists()


async def test_a_cached_line_whose_fit_fails_is_kept(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1,))))
    fake.env.setenv("FAKE_MODE", "ok,latin_first")
    res = await tr.translate(SceneRequest(1, specs((1,), want=("full", "concise"))))
    assert res.skipped == {} and res.lines[1].full.spoken == "కైట్ వాక్యం కము ఇది."
    assert "fit failed" in res.lines[1].flags and "concise" not in res.lines[1].tiers


# ---- recovery ----------------------------------------------------------------------------------------------------------
async def test_missing_ids_are_asked_for_once_more_with_the_answered_ones_as_context(fake):
    fake.env.setenv("FAKE_MODE", "drop_last,ok")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2, 3))))
    first, second = fake.calls()
    assert [x["id"] for x in second["msg"]["lines"]] == [3]
    assert second["msg"]["lines"][0]["problems"] == ["missing from the reply"]
    assert [x["id"] for x in second["msg"]["context_done"]] == [1, 2]
    assert set(res.lines) == {1, 2, 3} and res.calls == 2 and res.skipped == {}


async def test_then_one_call_per_line(fake):
    fake.env.setenv("FAKE_MODE", "latin_first,latin_first,ok")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2))))
    msgs = [c["msg"] for c in fake.calls()]
    assert [[x["id"] for x in m["lines"]] for m in msgs] == [[1, 2], [1], [1]]
    assert msgs[1]["lines"][0]["problems"] == ["full: Latin letters in test"]
    assert set(res.lines) == {1, 2} and res.calls == 3


async def test_per_line_calls_run_one_per_pending_line(fake):
    fake.env.setenv("FAKE_MODE", "drop_last,drop_last,ok")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2, 3))))
    assert [[x["id"] for x in c["msg"]["lines"]] for c in fake.calls()] == [[1, 2, 3], [3], [3]]
    assert set(res.lines) == {1, 2, 3}


async def test_a_line_that_never_comes_back_usable_is_skipped_never_english(fake):
    fake.env.setenv("FAKE_MODE", "cyrillic_first")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2))))
    assert len(fake.calls()) == 3 and set(res.lines) == {2}
    assert res.skipped == {1: "full: Cyrillic letters in క్ల\u0430స్"}
    rows = [json.loads(r) for r in (fake.tmp / "cache" / "vid1" / "lines.jsonl").read_text().splitlines()]
    assert len(rows) == 1  # only the good line is cached
    ev = [e for e in fake.events if e["event"] == "translate"][-1]
    assert ev["skipped"] == [1] and ev["calls"] == 3


async def test_a_reply_that_fails_the_schema_counts_as_every_id_missing(fake):
    fake.env.setenv("FAKE_MODE", "garbage_json,ok")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2))))
    assert res.calls == 2 and set(res.lines) == {1, 2}
    assert fake.calls()[1]["msg"]["lines"][0]["problems"][0].startswith("bad_output")


async def test_duplicated_and_unasked_ids(fake):
    fake.env.setenv("FAKE_MODE", "dupe_first,extra")
    res = await fake.make().translate(SceneRequest(1, specs((1, 2))))
    msgs = [c["msg"] for c in fake.calls()]
    assert msgs[1]["lines"][0]["problems"] == ["returned 2 times"]
    assert set(res.lines) == {1, 2} and 999 not in res.lines


@pytest.mark.parametrize("mode,kind", [("not_signed_in", "not_signed_in"), ("session_limit", "usage_limit")])
async def test_failures_the_user_must_fix_are_raised(fake, mode, kind):
    fake.env.setenv("FAKE_MODE", f"drop_last,{mode}")
    tr = fake.make()
    with pytest.raises(ClaudeCLIError) as err:
        await tr.translate(SceneRequest(1, specs((1, 2))))
    assert err.value.kind == kind
    assert [e for e in fake.events if e["event"] == "translate"][-1]["error"] == kind
    fake.env.setenv("FAKE_MODE", "ok")  # the answered line was cached before the failure
    res = await tr.translate(SceneRequest(1, specs((1, 2))))
    assert [x["id"] for x in fake.calls()[-1]["msg"]["lines"]] == [2] and res.lines[1].cached


# ---- other call types ------------------------------------------------------------------------------------------------
async def test_retranslate_names_the_missing_words_and_leaves_the_line_cache_to_the_review(fake):
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1,))))
    res = await tr.translate(SceneRequest(1, specs((1,), missing=("over the hill",)), "retranslate"))
    call = fake.calls()[-1]
    assert call["msg"]["lines"][0]["missing"] == ["over the hill"] and call["arg"]("--effort") == "medium"
    assert res.lines[1].full.spoken == "మళ్ళీ కైట్ వాక్యం కము ఇది." and res.lines[1].full.english == ((1, "kite"),)
    again = await tr.translate(SceneRequest(1, specs((1,))))  # not judged yet: the cache keeps the line it had
    assert again.lines[1].cached and again.lines[1].full.spoken == "కైట్ వాక్యం కము ఇది."


async def test_a_rephrase_is_queued_not_awaited_and_carries_the_qa_finding(fake):
    tr = fake.make()
    line = replace(specs((2,))[0], current="వాక్యం గము ఇది.", finding={"cer": 0.41, "skipped": ["గము"]})
    task = tr.submit(SceneRequest(1, (line,), "rephrase", deadline=time.time() + 60))
    assert isinstance(task, asyncio.Task)
    res = await task
    call = fake.calls()[-1]
    assert call["msg"]["call"] == "rephrase" and call["arg"]("--effort") == "low"
    assert call["msg"]["lines"][0]["failing"] == "వాక్యం గము ఇది." and call["msg"]["lines"][0]["finding"]["cer"] == 0.41
    assert not res.dropped and res.lines[2].full.spoken == "మళ్ళీ వాక్యం గము ఇది."


async def test_a_rephrase_that_misses_its_deadline_is_dropped_and_its_usage_logged(fake):
    fake.env.setenv("FAKE_MODE", "pause")
    fake.env.setenv("FAKE_PAUSE", "1.0")
    tr = fake.make()
    res = await tr.submit(SceneRequest(1, specs((2,)), "rephrase", deadline=time.time() + 0.3))
    assert res.dropped and res.lines == {} and res.skipped == {}
    assert len(cli_events(fake.events, call="rephrase", error=None)) == 1  # the call ran and its usage was recorded
    assert not (fake.tmp / "cache" / "vid1" / "lines.jsonl").exists()   # nothing cached from a late answer
    late = await tr.submit(SceneRequest(1, specs((3,)), "rephrase", deadline=time.time() - 1))
    assert late.dropped and len(fake.calls()) == 1  # already late when it got a slot: never started


# ---- concurrency and cancellation ------------------------------------------------------------------------------------
async def test_at_most_n_calls_run_at_once(fake):
    fake.env.setenv("FAKE_MODE", "pause")
    fake.env.setenv("FAKE_PAUSE", "0.6")
    tr = fake.make(concurrency=2)
    results = await asyncio.gather(*(tr.submit(SceneRequest(k, specs((k,)))) for k in range(1, 5)))
    assert all(len(r.lines) == 1 for r in results)
    starts = {c["pid"]: c["t0"] for c in fake.calls()}
    spans = [(starts[e["pid"]], e["t1"]) for e in map(json.loads, fake.ends.read_text().splitlines())]
    overlap = max(sum(1 for a, b in spans if a <= t < b) for t, _ in spans)
    assert overlap == 2


async def test_cancelling_an_in_flight_call_interrupts_the_cli_and_frees_its_slot(fake):
    fake.env.setenv("FAKE_MODE", "slow,ok")
    tr = fake.make(concurrency=1)
    task = tr.submit(SceneRequest(1, specs((1,))))
    await fake.started()
    t0 = time.monotonic()
    assert tr.cancel() == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - t0 < 10
    assert fake.signals.read_text().split() == ["SIGINT"]
    with pytest.raises(ProcessLookupError):
        os.kill(fake.calls()[0]["pid"], 0)
    assert cli_events(fake.events, error="cancelled")
    res = await tr.translate(SceneRequest(2, specs((2,))))  # the slot is free again
    assert set(res.lines) == {2}


async def test_cancelling_queued_requests_drops_them_before_they_start(fake):
    fake.env.setenv("FAKE_MODE", "slow")
    tr = fake.make(concurrency=1)
    running = tr.submit(SceneRequest(1, specs((1,))))
    queued = tr.submit(SceneRequest(2, specs((2,))))
    await fake.started()
    assert tr.cancel(lambda r: r.scene == 2) == 1
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert not running.done()
    await tr.aclose()
    assert running.cancelled() and len(fake.calls()) == 1


async def test_a_second_cancel_still_waits_for_the_cli_to_stop_before_freeing_the_slot(fake, caplog):
    fake.env.setenv("FAKE_MODE", "slow")
    fake.env.setenv("FAKE_STOP_DELAY", "0.6")
    tr = fake.make(concurrency=1)
    task = tr.submit(SceneRequest(1, specs((1,))))
    await fake.started()
    t0 = time.monotonic()
    assert tr.cancel(lambda r: r.scene == 1) == 1  # a seek
    await asyncio.sleep(0.1)
    await tr.aclose()                               # then the session closes
    assert time.monotonic() - t0 >= 0.5 and task.cancelled()
    with pytest.raises(ProcessLookupError):
        os.kill(fake.calls()[0]["pid"], 0)
    gc.collect()
    assert not [r for r in caplog.records if "never retrieved" in r.getMessage()]


# ---- glossary and brief ----------------------------------------------------------------------------------------------
async def test_glossary_additions_travel_in_later_messages_until_a_brief_swap(fake):
    fake.env.setenv("FAKE_MODE", "glossary,ok")
    tr = fake.make()
    res = await tr.translate(SceneRequest(1, specs((1,))))
    assert [g.term for g in res.glossary_additions] == ["Kite"]  # the Latin "string" spelling was dropped
    await tr.translate(SceneRequest(2, specs((2,))))
    assert fake.calls()[-1]["msg"]["glossary_recent"] == [{"term": "Kite", "spoken": "కైట్"}]
    assert '"term":"Kite"' not in fake.calls()[-1]["arg"]("--system-prompt")  # not in the cached prompt yet
    tr.use_brief(Brief(1, META, topic="kites", glossary=(GlossaryEntry("wind", "గాలి", False),)))
    await tr.translate(SceneRequest(3, specs((3,))))
    last = fake.calls()[-1]
    assert last["msg"]["glossary_recent"] == []
    assert '{"term":"Kite","spoken":"కైట్","keep_english":true}' in last["arg"]("--system-prompt")
    assert [g.term for g in tr.brief.glossary] == ["wind", "Kite"]


async def test_glossary_additions_survive_every_brief_swap(fake):
    fake.env.setenv("FAKE_MODE", "glossary,ok")
    tr = fake.make()
    await tr.translate(SceneRequest(1, specs((1,))))
    v1 = Brief(1, META, topic="kites", glossary=(GlossaryEntry("wind", "గాలి", False),))
    tr.use_brief(v1)
    tr.use_brief(parse_brief(META, {"topic": "", "register": "", "speakers": [], "glossary": []}, v1))  # v2 from v1
    await tr.translate(SceneRequest(2, specs((2,))))
    last = fake.calls()[-1]
    assert tr.brief.version == 2 and [g.term for g in tr.brief.glossary] == ["wind", "Kite"]
    assert '{"term":"Kite","spoken":"కైట్","keep_english":true}' in last["arg"]("--system-prompt")
    assert last["msg"]["glossary_recent"] == []


async def test_the_brief_wins_a_glossary_conflict(fake):
    fake.env.setenv("FAKE_MODE", "glossary")
    tr = fake.make(Brief(1, META, glossary=(GlossaryEntry("kite", "గాలిపటం", False),)))
    res = await tr.translate(SceneRequest(1, specs((1,))))
    assert res.glossary_additions == [] and tr.glossary_recent == {}
    assert "glossary: kite not written గాలిపటం" in res.lines[1].flags  # the fake says కైట్


async def test_brief_v1_is_one_call_cached_by_its_transcript(fake):
    tr = fake.make()
    transcript = [("S1", ENGLISH[0]), ("S2", ENGLISH[1])]
    v1 = await tr.make_brief(META, transcript)
    call = fake.calls()[-1]
    assert call["arg"]("--system-prompt") == sp.BRIEF_SYSTEM and json.loads(call["arg"]("--json-schema")) == sp.BRIEF_SCHEMA
    assert call["msg"]["transcript"][0] == {"speaker": "S1", "en": ENGLISH[0]}
    assert v1.version == 1 and [g.term for g in v1.glossary] == ["kite"]  # the Latin spelling was dropped
    assert [(s.id, s.gender) for s in v1.speakers] == [("S1", "female"), ("S2", "female")]
    assert await tr.make_brief(META, transcript) == v1 and len(fake.calls()) == 1
    assert [e["cache_hit"] for e in fake.events if e["event"] == "brief"] == [False, True]
    v2 = await tr.make_brief(META, transcript + [("S1", ENGLISH[2])], v1)
    assert "previous" in fake.calls()[-1]["msg"] and v2.version == 2 and v2.idioms == ("high as a kite",)
    assert v2.topic == "kites" and [g.term for g in v2.glossary] == ["kite"]


def test_a_brief_diff_keeps_what_it_leaves_out():
    v1 = parse_brief(META, {"topic": "kites", "register": "casual", "numbers": "lakhs",
                            "speakers": [{"id": "S1", "gender": "male"}],
                            "glossary": [{"term": "kite", "spoken": "కైట్", "keep_english": True}],
                            "asr_fixes": [{"heard": "kyte", "meant": "kite"}]})
    v2 = parse_brief(META, {"topic": "", "register": "", "speakers": [{"id": "S2", "gender": "female"}],
                            "glossary": [{"term": "Kite", "spoken": "గాలిపటం", "keep_english": False}],
                            "entities": ["Open Sky"]}, v1)
    assert (v2.version, v2.topic, v2.register, v2.numbers) == (2, "kites", "casual", "lakhs")
    assert [s.id for s in v2.speakers] == ["S1", "S2"]
    assert [(g.term, g.spoken) for g in v2.glossary] == [("Kite", "గాలిపటం")]
    assert v2.entities == ("Open Sky",) and v2.asr_fixes == (("kyte", "kite"),)
    assert parse_brief(META, "junk") == Brief(1, META)


def test_a_brief_diff_changes_only_the_speaker_fields_it_names():
    v1 = parse_brief(META, {"topic": "kites", "register": "casual", "glossary": [], "speakers": [
        {"id": "S1", "name": "Meena", "gender": "female", "role": "host", "audience": "familiar",
         "address": [{"to": "S2", "form": "familiar"}]}]})
    v2 = parse_brief(META, {"topic": "", "register": "", "glossary": [], "speakers": [{"id": "S1", "role": "teacher"}]},
                     v1)
    assert v2.speakers == (SpeakerNote("S1", "Meena", "female", "teacher", "familiar", (("S2", "familiar"),)),)
    v3 = parse_brief(META, {"topic": "", "register": "", "glossary": [], "speakers": [
        {"id": "S1", "name": "", "gender": "unknown", "address": [{"to": "S2", "form": "polite"},
                                                                  {"to": "S3", "form": "polite"}]}]}, v2)
    assert v3.speakers == (SpeakerNote("S1", "Meena", "female", "teacher", "familiar",
                                       (("S2", "polite"), ("S3", "polite"))),)
    assert parse_brief(META, {"speakers": [{"id": "S2"}]}).speakers == (SpeakerNote("S2"),)  # v1: the defaults


# ---- the coverage review (§4.6) ---------------------------------------------------------------------------------------
def rows(fake, video="vid1") -> dict[str, dict]:
    """The line cache's rows as a later load sees them: the last one for each key."""
    out = {}
    for raw in (fake.tmp / "cache" / video / "lines.jsonl").read_text().splitlines():
        row = json.loads(raw)
        out[row["key"]] = row
    return out


async def reviewed(fake, ids=(1, 2), chosen="full", tr=None, **kw):
    tr = tr or fake.make(**kw)
    req = SceneRequest(3, specs(ids), context_before=(("Look up there.", "అటు చూడండి."),), context_after_en=("Wow.",))
    res = await tr.translate(req)
    return tr, req, res, await tr.review(req, res.lines, {i: chosen for i in res.lines})


async def test_the_review_is_one_call_on_its_own_model_prompt_and_schema(fake):
    tr, req, res, out = await reviewed(fake)
    scene, review = fake.calls()
    assert review["arg"]("--model") == "claude-sonnet-5" and review["arg"]("--effort") == "high"
    assert review["arg"]("--fallback-model") == "claude-opus-5-5"
    assert review["arg"]("--system-prompt") == sp.REVIEW_SYSTEM
    assert json.loads(review["arg"]("--json-schema")) == sp.REVIEW_SCHEMA
    assert review["stdin"] == sp.review_message(req, [(s, res.lines[s.id].full) for s in req.lines])
    assert review["msg"]["context_before"] == [{"en": "Look up there.", "te": "అటు చూడండి."}]
    assert out.error is None and out.calls == 1
    assert {i: x.coverage for i, x in out.lines.items()} == {i: Coverage("C", tier="full") for i in (1, 2)}
    assert out.lines[1].full == res.lines[1].full
    (usage,) = cli_events(fake.events, call="review")
    assert usage["model"] == "claude-sonnet-5" and usage["effort"] == "high" and usage["scene"] == 3
    (ev,) = [e for e in fake.events if e["event"] == "review"]
    assert ev["classes"] == {"C": 2, "m": 0, "P": 0, "E": 0} and ev["unreviewed"] == [] and ev["error"] is None
    assert ev["review_hash"] == sp.REVIEW_HASH


async def test_the_review_model_and_effort_are_configurable(fake):
    await reviewed(fake, review_model="claude-opus-5-5", review_fallback=None, review_effort="low")
    review = fake.calls()[-1]
    assert review["arg"]("--model") == "claude-opus-5-5" and review["arg"]("--effort") == "low"
    assert "--fallback-model" not in review["argv"]


async def test_classes_are_stored_with_the_line_and_a_rewatch_reviews_nothing(fake):
    await reviewed(fake, chosen="full")
    assert all(r["coverage"]["class"] == "C" and r["coverage"]["tier"] == "full" for r in rows(fake).values())
    tr, req, res, out = await reviewed(fake)  # another run over the same cache
    assert len(fake.calls()) == 2 and res.calls == 0 and out.calls == 0
    assert all(x.cached and x.coverage == Coverage("C", tier="full") for x in out.lines.values())


async def test_p_and_e_lines_get_one_retranslation_from_the_english_with_the_missing_words_named(fake):
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"], "2": ["E", [], "negation"]}))
    tr, req, res, out = await reviewed(fake, ids=(1, 2, 3))
    redo = fake.calls()[-1]
    assert redo["msg"]["call"] == "retranslate" and len(fake.calls()) == 3
    assert redo["arg"]("--model") == "claude-opus-5-5" and redo["arg"]("--effort") == "medium"
    assert redo["arg"]("--system-prompt") == sp.system_prompt(brief_v0(META))  # the scene prompt: never the Telugu
    by_id = {x["id"]: x for x in redo["msg"]["lines"]}
    assert set(by_id) == {1, 2} and by_id[1]["missing"] == ["over the hill"] and "problems" not in by_id[1]
    assert by_id[2]["missing"] == [] and "meaning error (negation)" in by_id[2]["problems"][0]
    assert [x["id"] for x in redo["msg"]["context_done"]] == [3]
    # the re-translations say more and flag nothing: classed C by the checks, and kept
    for i, first in ((1, "P"), (2, "E")):
        assert out.lines[i].full.spoken.startswith("మళ్ళీ")
        assert out.lines[i].coverage == Coverage("C", tier="full", by="validators", first=first)
    assert out.lines[3].coverage.cls == "C" and out.lines[3].full == res.lines[3].full
    stored = {r["line"]["id"]: r for r in rows(fake).values()}
    assert stored[1]["line"]["full"]["spoken"].startswith("మళ్ళీ") and stored[1]["coverage"]["first"] == "P"
    (ev,) = [e for e in fake.events if e["event"] == "review"]
    assert ev["retranslated"] == [1, 2] and ev["replaced"] == [1, 2] and ev["calls"] == 2
    again = await fake.make().translate(req)  # a re-watch serves the kept wordings with their classes
    assert again.calls == 0 and again.lines[2].coverage.by == "validators"


async def test_the_reviewed_wording_stays_unless_the_retranslation_classes_better(fake):
    lines = (LineSpec(1, "S1", "The kite climbs over the hill.", 2.0, 3.8, 1.6, 9.0),
             LineSpec(2, "S1", "It never comes down again.", 4.0, 5.8, 1.6, 9.0),
             LineSpec(3, "S1", "Two more kites follow it.", 6.0, 7.8, 1.6, 9.0))
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over"], "none"], "2": ["E", [], "negation"],
                                               "3": ["P", ["two"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,no_longer")
    tr = fake.make()
    req = SceneRequest(1, lines)
    res = await tr.translate(req)
    out = await tr.review(req, res.lines, {1: "full", 2: "full", 3: "full"})
    # 1: no longer than before, so still P (a tie keeps the reviewed one); 2: the negation still isn't said, E again;
    # 3: the number still isn't said, E, worse than P
    assert [out.lines[i].coverage.cls for i in (1, 2, 3)] == ["P", "E", "P"]
    assert all(out.lines[i].coverage.by == "review" and out.lines[i].full == res.lines[i].full for i in (1, 2, 3))
    assert all(r["coverage"]["by"] == "review" for r in rows(fake).values())  # the cache holds the reviewed ones again
    assert (await fake.make().translate(req)).lines[1].full == res.lines[1].full


async def test_a_line_reviewed_on_a_shorter_tier_keeps_that_tier_in_its_class(fake):
    tr = fake.make()
    req = SceneRequest(1, specs((1,), want=("full", "concise")))
    res = await tr.translate(req)
    out = await tr.review(req, res.lines, {1: "concise"})
    assert fake.calls()[-1]["msg"]["lines"][0]["te"] == res.lines[1].tiers["concise"].spoken
    assert out.lines[1].coverage.tier == "concise"
    req2 = SceneRequest(2, specs((2,)))
    res2 = await tr.translate(req2)
    out2 = await tr.review(req2, res2.lines, {2: "very_concise"})  # a tier the line hasn't: its `full` is reviewed
    assert fake.calls()[-1]["msg"]["lines"][0]["te"] == res2.lines[2].full.spoken
    assert out2.lines[2].coverage.tier == "full"


async def test_ids_the_review_lacks_are_asked_once_more_then_left_unreviewed_and_unstored(fake):
    fake.env.setenv("FAKE_MODE", "ok,drop_last,drop_last,ok")
    tr, req, res, out = await reviewed(fake, ids=(1, 2, 3))
    first, second = fake.calls()[1:3]
    assert [x["id"] for x in second["msg"]["lines"]] == [3]
    assert out.lines[3].coverage is None and out.lines[1].coverage.cls == "C" and out.error is None
    (ev,) = [e for e in fake.events if e["event"] == "review"]
    assert ev["unreviewed"] == [3] and ev["calls"] == 2
    assert [r["coverage"] is None for r in sorted(rows(fake).values(), key=lambda r: r["line"]["id"])] == \
        [False, False, True]
    out2 = await tr.review(req, (await tr.translate(req)).lines, {i: "full" for i in (1, 2, 3)})  # asked about next time
    assert [x["id"] for x in fake.calls()[-1]["msg"]["lines"]] == [3] and out2.lines[3].coverage.cls == "C"


async def test_a_review_reply_that_fails_its_schema_is_asked_again_once(fake):
    fake.env.setenv("FAKE_MODE", "ok,garbage_json,garbage_json")
    tr, req, res, out = await reviewed(fake)
    assert len(fake.calls()) == 3 and out.error is None and out.calls == 2
    assert all(x.coverage is None for x in out.lines.values()) and set(out.lines) == {1, 2}


async def test_a_review_failure_the_user_must_fix_leaves_the_lines_unreviewed_and_comes_back_in_error(fake):
    fake.env.setenv("FAKE_MODE", "ok,not_signed_in")
    tr, req, res, out = await reviewed(fake)
    assert out.error is not None and out.error.kind == "not_signed_in"
    assert {i: x.full for i, x in out.lines.items()} == {i: x.full for i, x in res.lines.items()}
    assert all(x.coverage is None for x in out.lines.values()) and all(r["coverage"] is None for r in rows(fake).values())
    assert [e["error"] for e in fake.events if e["event"] == "review"] == ["not_signed_in"]


async def test_a_retranslation_that_fails_the_user_leaves_its_line_to_be_reviewed_again(fake):
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,session_limit")
    tr, req, res, out = await reviewed(fake)
    assert out.error.kind == "usage_limit" and out.lines[1].coverage is None and out.lines[2].coverage.cls == "C"
    assert {r["line"]["id"]: r["coverage"] for r in rows(fake).values()}[1] is None


async def test_a_review_is_cancelled_like_a_request(fake):
    fake.env.setenv("FAKE_MODE", "ok,slow")
    tr = fake.make()
    req = SceneRequest(1, specs((1,)))
    res = await tr.translate(req)
    task = asyncio.ensure_future(tr.review(req, res.lines, {1: "full"}))
    await fake.started(2)
    assert tr.cancel(lambda r: r.scene == 1) == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.signals.read_text().split() == ["SIGINT"]


async def test_a_fit_keeps_the_lines_class(fake):
    tr, req, res, out = await reviewed(fake, ids=(1,))
    fake.env.setenv("FAKE_MODE", "reword")  # a fit that rewrites full too: never taken
    fit = replace(specs((1,), want=("concise",))[0], current=out.lines[1].full.spoken, overflow=2.0)
    got = (await tr.translate(SceneRequest(1, (fit,), "fit"))).lines[1]
    assert got.coverage == Coverage("C", tier="full") and got.tiers["concise"].spoken == "కైట్ వాక్యం కము"
    assert rows(fake)[ct.line_key(fit, "colloquial")]["coverage"]["class"] == "C"


def test_a_fit_never_loses_a_tier_the_line_had(fake):
    tr = fake.make()
    spec = replace(specs((1,))[0], want=("concise", "very_concise"))
    old = LineResult(1, {"full": Wording("ఒకటి రెండు మూడు నాలుగు."), "concise": Wording("ఒకటి రెండు మూడు."),
                         "very_concise": Wording("ఒకటి రెండు.")}, coverage=Coverage("m", tier="concise"))
    # a new concise no shorter than full is left out, and the line's own stays; the new very_concise fits under it
    new = LineResult(1, {"full": Wording("ఒకటి రెండు మూడు నాలుగు."), "concise": Wording("ఒకటి రెండు మూడు నాలుగు ఐదు."),
                         "very_concise": Wording("ఒకటి.")})
    got = tr._merged(old, new, spec, {})
    assert {k: w.spoken for k, w in got.tiers.items()} == \
        {"very_concise": "ఒకటి.", "concise": "ఒకటి రెండు మూడు.", "full": "ఒకటి రెండు మూడు నాలుగు."}
    assert "fit: concise left out" in got.flags and got.coverage == old.coverage
    # a new very_concise longer than the line's concise would push that out: left out too
    worse = LineResult(1, {"full": new.full, "very_concise": Wording("ఒకటి రెండు మూడు నాలుగు.")})
    got = tr._merged(old, worse, replace(spec, want=("very_concise",)), {})
    assert got.tiers == old.tiers and "fit: very_concise left out" in got.flags


async def test_a_retranslation_is_classed_on_the_tier_the_review_classed(fake):
    """The review found a phrase missing from `concise`: the re-translation's `concise` is set against it, never its
    `full` (longer by construction). The same wording back is still P, and the reviewed line stays."""
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,no_longer")
    tr = fake.make()
    req = SceneRequest(1, specs((1,), want=("full", "concise")))
    res = await tr.translate(req)
    out = await tr.review(req, res.lines, {1: "concise"})
    assert fake.calls()[-1]["msg"]["call"] == "retranslate"
    assert out.lines[1].coverage == Coverage("P", ("over the hill",), tier="concise")
    assert out.lines[1].tiers == res.lines[1].tiers
    assert [e["replaced"] for e in fake.events if e["event"] == "review"] == [[]]
    # a longer `concise` back: C on `concise`, and kept
    fake.env.setenv("FAKE_MODE", "ok")
    tr = fake.make(video="vid2")
    res = await tr.translate(req)
    out = await tr.review(req, res.lines, {1: "concise"})
    assert out.lines[1].coverage == Coverage("C", tier="concise", by="validators", first="P")
    assert out.lines[1].tiers["concise"].spoken == "మళ్ళీ కైట్ వాక్యం కము"


async def test_a_retranslation_without_the_reviewed_tier_is_classed_on_full_against_full(fake):
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,full_only")
    tr = fake.make()
    req = SceneRequest(1, specs((1,), want=("full", "concise")))
    res = await tr.translate(req)
    out = await tr.review(req, res.lines, {1: "concise"})
    assert set(out.lines[1].tiers) == {"full"} and out.lines[1].full.spoken == "మళ్ళీ కైట్ వాక్యం కము ఇది."
    assert out.lines[1].coverage == Coverage("C", tier="full", by="validators", first="P")


async def test_a_retranslation_is_cached_only_once_the_review_has_judged_it(fake):
    """Line 2's re-translation is left out of the first reply and asked for again, which hits a usage limit: line 1's,
    back already, is judged, kept and stored; line 2 keeps its first wording, unclassed, and so does its cache row."""
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"], "2": ["P", ["higher"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,drop_last,session_limit")
    tr, req, res, out = await reviewed(fake)
    assert out.error.kind == "usage_limit" and len(fake.calls()) == 4
    assert out.lines[1].coverage == Coverage("C", tier="full", by="validators", first="P")
    assert out.lines[2].coverage is None and out.lines[2].full == res.lines[2].full
    stored = {r["line"]["id"]: r for r in rows(fake).values()}
    assert stored[1]["line"]["full"]["spoken"].startswith("మళ్ళీ") and stored[1]["coverage"]["class"] == "C"
    assert stored[2]["line"]["full"] == ct.line_json(res.lines[2])["full"] and stored[2]["coverage"] is None


async def test_a_review_cancelled_during_its_retranslation_leaves_the_cache_as_it_was(fake):
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["P", ["over the hill"], "none"], "2": ["P", ["higher"], "none"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,drop_last,slow")
    tr = fake.make()
    req = SceneRequest(1, specs((1, 2)))
    res = await tr.translate(req)
    task = asyncio.ensure_future(tr.review(req, res.lines, {1: "full", 2: "full"}))
    await fake.started(4)  # line 1's re-translation is back; line 2's is asked for again
    assert tr.cancel(lambda r: r.scene == 1) == 1
    with pytest.raises(asyncio.CancelledError):
        await task
    stored = {r["line"]["id"]: r for r in rows(fake).values()}
    assert all(stored[i]["line"]["full"] == ct.line_json(res.lines[i])["full"] and stored[i]["coverage"] is None
               for i in (1, 2))


async def test_a_retranslation_asked_again_still_carries_the_review_finding(fake):
    fake.env.setenv("FAKE_REVIEW", json.dumps({"1": ["E", [], "negation"]}))
    fake.env.setenv("FAKE_MODE", "ok,ok,drop_last,ok")
    tr, req, res, out = await reviewed(fake, ids=(1,))
    first, again = (c["msg"]["lines"][0] for c in fake.calls()[2:])
    assert "meaning error (negation)" in first["problems"][0]
    assert again["problems"] == first["problems"] + ["missing from the reply"]
    assert out.lines[1].coverage.by == "validators"


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


LONG = "గాలిపటం పచ్చని కొండ మీదుగా నెమ్మదిగా పైకి పైకి వెళ్తుంది."
MID = "గాలిపటం కొండ మీదుగా పైకి వెళ్తుంది."
SHORT = "గాలిపటం పైకి."
KITE = LineSpec(1, "S1", "The kite climbs slowly over the green hill.", 0.0, 4.0, 4.0, 14.0,
                ("full", "concise", "very_concise"))


def refit() -> Refit:
    """`full` far over the slot and `very_concise` far under it; a fit gives a `concise` and a `very_concise` between."""
    return Refit({"full": LONG, "very_concise": SHORT}, {"concise": MID, "very_concise": "గాలిపటం కొండ పైకి."})


async def test_a_fit_under_the_band_is_cut_from_the_longer_wording_so_the_tier_it_asks_for_is_kept(tmp_path):
    """The band lies between `full` and `very_concise`: the session asks for `concise`, cut from `full` (the fit's
    "current", which it copies into its `full`), so the new tier is checked against the line's own `full`. Given the
    short pick as "current" instead, the same answer would be longer than the `full` the fit copied, and dropped."""
    tr = ClaudeTranslator(tmp_path, "vid", brief_v0(META), cli=refit())
    line = (await tr.translate(SceneRequest(1, (KITE,)))).lines[1]
    assert set(line.tiers) == {"full", "very_concise"}
    short = replace(KITE, want=("concise",), current=SHORT, overflow=-8.0)
    assert (await tr.translate(SceneRequest(1, (short,), "fit"))).lines[1].tiers == line.tiers
    fit = replace(KITE, want=("concise",), current=LONG, overflow=count_telugu(LONG) - KITE.target_aksharas)
    got = (await tr.translate(SceneRequest(1, (fit,), "fit"))).lines[1]
    assert {k: w.spoken for k, w in got.tiers.items()} == {"full": LONG, "concise": MID, "very_concise": SHORT}
    # the pick again (`very_concise`, under the band, `concise` over it): cut from `concise`
    again = replace(KITE, want=("very_concise",), current=MID, overflow=4.0)
    got = (await tr.translate(SceneRequest(1, (again,), "fit"))).lines[1]
    assert got.tiers["very_concise"].spoken == "గాలిపటం కొండ పైకి." and got.full.spoken == LONG


async def test_a_fit_that_replaces_the_reviewed_wording_leaves_the_line_unreviewed(tmp_path):
    tr = ClaudeTranslator(tmp_path, "vid", brief_v0(META), cli=refit())
    req = SceneRequest(1, (KITE,))
    res = await tr.translate(req)
    line = (await tr.review(req, res.lines, {1: "very_concise"})).lines[1]
    assert line.coverage == Coverage("C", tier="very_concise")
    fit = replace(KITE, want=("concise",), current=LONG, overflow=6.0)  # another tier added: the class stays
    line = (await tr.translate(SceneRequest(1, (fit,), "fit"))).lines[1]
    assert "concise" in line.tiers and line.coverage == Coverage("C", tier="very_concise")
    fit = replace(KITE, want=("very_concise",), current=MID, overflow=4.0)  # the reviewed wording itself replaced
    line = (await tr.translate(SceneRequest(1, (fit,), "fit"))).lines[1]
    assert line.tiers["very_concise"].spoken == "గాలిపటం కొండ పైకి." and line.coverage is None
    assert tr._lines.rows[ct.line_key(KITE, "colloquial")]["coverage"] is None
    n = len(tr.review_cli.calls)
    again = await tr.review(req, (await tr.translate(req)).lines, {1: "very_concise"})  # reviewed the next time
    assert len(tr.review_cli.calls) == n + 1 and again.lines[1].coverage == Coverage("C", tier="very_concise")


# ---- the mock ----------------------------------------------------------------------------------------------------------
async def test_the_mock_translator_is_deterministic_telugu_and_never_calls_claude(tmp_path):
    from maata_engine.backends.mock import _LINES

    lines = tuple(LineSpec(i, "S1", en, 2.0 * i, 2.0 * i + 1.5, 1.4, 8.0, ("full", "concise", "fuller"))
                  for i, en in enumerate([x[0] for x in _LINES] + ["Something no demo line says, with a pause."], 1))
    tr = MockSceneTranslator(tmp_path, "demo", brief_v0(META))
    res = await tr.translate(SceneRequest(1, lines))
    again = await MockSceneTranslator(tmp_path / "other", "demo", brief_v0(META)).translate(SceneRequest(1, lines))
    assert {i: x.tiers for i, x in res.lines.items()} == {i: x.tiers for i, x in again.lines.items()}
    assert res.skipped == {} and len(res.lines) == len(lines)
    for x in res.lines.values():
        assert all(script_problems(t.spoken) == [] for t in x.tiers.values())
        assert list(x.tiers) == ["concise", "full", "fuller"]
    assert res.lines[2].full.english == ((5, "project"), (6, "build"))
    assert isinstance(tr.cli, MockClaude) and tr.model == "mock"


async def test_the_mock_can_be_slow_and_cancelled(tmp_path):
    tr = MockSceneTranslator(tmp_path, "demo", brief_v0(META), delay=30.0)
    task = tr.submit(SceneRequest(1, specs((1,))))
    await asyncio.sleep(0.1)
    t0 = time.monotonic()
    tr.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - t0 < 2
