"""The sealed `claude -p` client, against a fake CLI that records what it was given and replays canned replies.

FAKE_MODE is a comma-separated script: the n-th `-p` call plays the n-th mode (the last one repeats).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from maata_engine.claude_cli import (QUIET_ENV, SETTINGS, WORKDIR, ClaudeCLI, ClaudeCLIError, check_schema, classify_failure,
                                     failure, parse_output, parse_reset, validate)

FAKE = r"""#!@PYTHON@
import json, os, signal, sys, time
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "2.1.281 (Claude Code)"))
    sys.exit(0)
path = os.environ["FAKE_RECORD"]
n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
modes = os.environ.get("FAKE_MODE", "ok").split(",")
mode = modes[min(n, len(modes) - 1)]
if mode == "stubborn":  # ignores SIGINT and SIGTERM; only SIGKILL stops it
    def note(sig, _):
        open(os.environ["FAKE_SIGNALS"], "a").write(signal.Signals(sig).name + "\n")
    signal.signal(signal.SIGINT, note)
    signal.signal(signal.SIGTERM, note)
prompt = sys.stdin.read()
open(path, "a").write(json.dumps({"argv": argv, "stdin": prompt, "env": dict(os.environ), "cwd": os.getcwd(),
                                  "pid": os.getpid(), "files": sorted(os.listdir("."))}) + "\n")

def emit(ev):
    print(json.dumps(ev), flush=True)

def result(**kw):
    emit({"type": "result", "subtype": "success", "is_error": False, **kw})

def error(text, code=1):
    result(is_error=True, result=text)
    sys.exit(code)

if mode in ("hang", "stubborn"):
    for _ in range(300):
        time.sleep(0.1)
    sys.exit(0)
if mode == "garbage":
    print("segfault-ish noise")
    sys.exit(3)
emit({"type": "system", "subtype": "init", "model": argv[argv.index("--model") + 1]})
if mode == "slow":
    time.sleep(30)
if mode == "not_signed_in":
    error("Not logged in · Please run /login")
if mode == "oauth_expired":
    error("Failed to authenticate: OAuth session expired and could not be refreshed")
if mode == "outdated":
    error("API Error: 400 Claude Code 2.1.281 does not support this model; version 2.1.300 or newer is required. "
          "Run 'claude update', then try again.")
if mode == "throttle":
    error("API Error: Server is temporarily limiting requests (not your usage limit)")
if mode == "session_limit":
    error("You've hit your session limit · resets 3pm (Europe/London)")
if mode == "opus_limit":
    error("You've hit your Opus limit · resets Oct 7, 2am (America/Los_Angeles)")
if mode == "old_limit":
    error("Claude AI usage limit reached|1790300000")
if mode == "rejected":
    emit({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "seven_day_sonnet",
                                                          "resetsAt": 1790300000}})
    error("Request failed")
if mode == "max_retries":
    emit({"type": "result", "subtype": "error_max_structured_output_retries", "is_error": True})
    sys.exit(1)
if mode == "fenced":
    result(result='```json\n{"lines": ["a"]}\n```')
    sys.exit(0)
if mode == "fenced_bad":
    result(result='```json\n{"lines": [""], "extra": 1}\n```')
    sys.exit(0)
if mode == "prose":
    result(result="Sorry, I can't.")
    sys.exit(0)
emit({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "five_hour",
                                                      "utilization": 0.86, "resetsAt": 1790300000000}})
model = argv[argv.index("--model") + 1]
usage = {"input_tokens": 12, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 300,
         "cache_creation": {"ephemeral_1h_input_tokens": 300}, "output_tokens": 5}
models = {model: {"outputTokens": 5}}
if mode == "fallback":
    models = {model: {"outputTokens": 0}, "claude-opus-5-5": {"outputTokens": 5}}
result(result="నమస్కారం", structured_output={"lines": ["నమస్కారం"]}, usage=usage, modelUsage=models)
"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["lines"],
          "properties": {"lines": {"type": "array", "items": {"type": "string", "minLength": 1}}}}


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.replace("@PYTHON@", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("FAKE_SIGNALS", str(tmp_path / "signals"))
    monkeypatch.setenv("CLAUDECODE", "1")  # as when the engine runs inside a Claude Code session
    events: list[dict] = []
    delays: list[float] = []

    def make(**kw) -> ClaudeCLI:
        opts = dict(effort="high", binary=str(exe), timeout=8, startup_timeout=5, grace=0.5, retries=2, backoff=0.01,
                    trace=events.append)
        cli = ClaudeCLI(tmp_path / "cache", **{**opts, **kw})
        cli._sleep = delays.append  # record back-offs instead of sleeping
        return cli

    def calls() -> list[dict]:
        return [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []

    return SimpleNamespace(cli=make(), make=make, calls=calls, events=events, delays=delays, env=monkeypatch, tmp=tmp_path)


def test_call_is_sealed_and_prompt_goes_on_stdin(fake):
    reply = fake.cli.ask("SYSTEM PROMPT", "a transcript line that must not appear in argv", schema=SCHEMA, call="scene")
    rec = fake.calls()[-1]
    argv = rec["argv"]
    assert argv[:1] == ["-p"]
    for flag in ("--safe-mode", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands", "--verbose"):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "SYSTEM PROMPT"
    assert json.loads(argv[argv.index("--settings") + 1]) == {"fastMode": False} == json.loads(SETTINGS)
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--fallback-model") + 1] == "claude-sonnet-5"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert "--bare" not in argv
    assert rec["stdin"] == "a transcript line that must not appear in argv"
    assert not any("transcript line" in a for a in argv)
    assert "CLAUDECODE" not in rec["env"]
    assert all(rec["env"].get(k) == v for k, v in QUIET_ENV.items()) and QUIET_ENV["DISABLE_FEEDBACK_COMMAND"] == "1"
    assert reply.data == {"lines": ["నమస్కారం"]} and reply.text == "నమస్కారం"
    assert reply.usage["output_tokens"] == 5 and reply.model == "claude-opus-5-5"
    assert reply.rate_limit == {"status": "allowed_warning", "type": "five_hour", "utilization": 0.86,
                                "resets_at": 1790300000.0}
    assert reply.startup_s is not None and reply.seconds >= reply.startup_s


def test_every_call_runs_in_one_fixed_empty_directory(fake):
    fake.cli.ask("s", "p", schema=SCHEMA)
    fake.make(model="claude-opus-5-5", fallback_model=None).ask("s", "p", schema=SCHEMA)
    cwds = {os.path.realpath(c["cwd"]) for c in fake.calls()}
    assert cwds == {os.path.realpath(fake.tmp / "cache" / WORKDIR)}
    assert all(c["files"] == [] for c in fake.calls())


def test_each_call_leaves_a_usage_record(fake):
    cli = fake.make(model="claude-sonnet-5", fallback_model="claude-opus-5-5")  # so the fallback answer is visible
    cli.ask("s", "p", schema=SCHEMA, call="scene")
    fake.env.setenv("FAKE_MODE", "fallback")
    reply = cli.ask("s", "p", schema=SCHEMA, call="review")
    assert reply.model == "claude-opus-5-5"  # the model that answered, not the one asked for
    first, second = fake.events
    assert first == {"event": "claude", "call": "scene", "requested": "claude-sonnet-5", "model": "claude-sonnet-5",
                     "effort": "high", "attempt": 1, "wall_s": first["wall_s"], "startup_s": first["startup_s"],
                     "input_tokens": 12, "cache_read_tokens": 9000, "cache_creation_tokens": 300, "cache_1h_tokens": 300,
                     "output_tokens": 5, "rate_limit": first["rate_limit"], "error": None}
    assert first["wall_s"] >= first["startup_s"] >= 0
    assert second["call"] == "review" and second["requested"] == "claude-sonnet-5" and second["model"] == "claude-opus-5-5"


@pytest.mark.parametrize("version,models", [("2.1.250 (Claude Code)", ("claude-sonnet-5",)),
                                            ("2.1.281 (Claude Code)", ("claude-sonnet-5", "claude-opus-5-5"))])
def test_version_gate_reports_which_models_the_cli_can_run(fake, version, models):
    fake.env.setenv("FAKE_VERSION", version)
    assert fake.cli.probe().models == models
    fake.cli.ask("s", "p", schema=SCHEMA)
    argv = fake.calls()[-1]["argv"]
    # the fallback only when this CLI can run it
    assert ("--fallback-model" in argv) == ("claude-opus-5-5" in models)


def test_too_old_cli_is_refused_before_any_call(fake):
    fake.env.setenv("FAKE_VERSION", "2.1.201 (Claude Code)")
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "outdated" and "claude update" in str(err.value)
    fake.env.setenv("FAKE_VERSION", "2.1.250 (Claude Code)")
    with pytest.raises(ClaudeCLIError) as err:  # Opus 5.5 needs 2.1.280, and there is no fallback to run instead
        fake.make(model="claude-opus-5-5", fallback_model=None).ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "outdated" and "2.1.280" in str(err.value)
    assert fake.calls() == [] and fake.events == []


def test_a_cli_too_old_for_the_model_runs_the_fallback_and_says_so(fake):
    fake.env.setenv("FAKE_VERSION", "2.1.250 (Claude Code)")
    cli = fake.make(model="claude-opus-5-5", fallback_model="claude-sonnet-5")
    assert cli.ask("s", "p", schema=SCHEMA).model == "claude-sonnet-5"
    argv = fake.calls()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5" and "--fallback-model" not in argv
    health = cli.health()
    assert health["problem"] is None and health["model"] == "claude-sonnet-5"
    assert "2.1.280" in health["message"] and "claude update" in health["message"]


@pytest.mark.parametrize("model,fallback", [("opus", None), ("claude-fable-5-1", None), ("claude-sonnet-5", "best"),
                                            ("claude-sonnet-5", "claude-fable-5-1")])
def test_only_pinned_model_ids_and_never_fable(tmp_path, model, fallback):
    with pytest.raises(ValueError):
        ClaudeCLI(tmp_path, model=model, fallback_model=fallback)


def test_json_recovered_from_text_counts_only_if_it_validates(fake):
    fake.env.setenv("FAKE_MODE", "fenced")
    assert fake.cli.ask("s", "p", schema=SCHEMA).data == {"lines": ["a"]}
    fake.env.setenv("FAKE_MODE", "fenced_bad")
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "bad_output"


@pytest.mark.parametrize("mode,kind", [("not_signed_in", "not_signed_in"), ("oauth_expired", "not_signed_in"),
                                       ("outdated", "outdated"), ("garbage", "failed"), ("prose", "bad_output"),
                                       ("max_retries", "bad_output"), ("session_limit", "usage_limit")])
def test_failures_map_to_kinds_the_ui_can_explain_and_are_not_retried(fake, mode, kind):
    fake.env.setenv("FAKE_MODE", mode)
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == kind
    assert len(fake.calls()) == 1 and fake.delays == []
    assert [e["error"] for e in fake.events] == [kind]


def test_a_session_limit_names_itself_and_its_reset(fake):
    fake.env.setenv("FAKE_MODE", "session_limit")
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.limit == "session" and err.value.resets == "3pm (Europe/London)"
    assert err.value.resets_at is not None and err.value.resets_at > time.time()


def test_the_transient_throttle_is_retried_with_exponential_backoff(fake):
    fake.env.setenv("FAKE_MODE", "throttle,throttle,ok")
    assert fake.cli.ask("s", "p", schema=SCHEMA).data == {"lines": ["నమస్కారం"]}
    assert fake.delays == [0.01, 0.02]
    assert [(e["attempt"], e["error"]) for e in fake.events] == [(1, "transient"), (2, "transient"), (3, None)]


def test_retries_give_up_after_the_limit(fake):
    fake.env.setenv("FAKE_MODE", "throttle")
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "transient"
    assert len(fake.calls()) == 3 and fake.delays == [0.01, 0.02]


def test_a_cli_that_never_starts_is_stopped_then_retried_after_a_backoff(fake):
    fake.env.setenv("FAKE_MODE", "hang,ok")
    cli = fake.make(startup_timeout=0.5)
    assert cli.ask("s", "p", schema=SCHEMA).data == {"lines": ["నమస్కారం"]}
    assert fake.delays == [0.01]  # never an immediate retry (#91987)
    assert [e["error"] for e in fake.events] == ["stalled", None]


def test_watchdog_escalates_from_sigint_to_sigterm_to_sigkill(fake):
    fake.env.setenv("FAKE_MODE", "stubborn")
    cli = fake.make(startup_timeout=1.0, grace=0.3, retries=0)
    with pytest.raises(ClaudeCLIError) as err:
        cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "stalled" and "#91987" in str(err.value)
    assert (fake.tmp / "signals").read_text().split() == ["SIGINT", "SIGTERM"]
    with pytest.raises(ProcessLookupError):  # killed and reaped
        os.kill(fake.calls()[-1]["pid"], 0)


@pytest.mark.parametrize("mode,startup", [("slow", 5.0), ("hang", 90.0)], ids=["started", "never-started"])
def test_an_answer_that_takes_too_long_times_out(fake, mode, startup):
    # With the overall timeout shorter than the startup one, it is a timeout (not retried) whether or not the CLI started.
    fake.env.setenv("FAKE_MODE", mode)
    with pytest.raises(ClaudeCLIError) as err:
        fake.make(timeout=1.0, startup_timeout=startup).ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "timeout" and "within 1 s" in str(err.value)
    assert fake.delays == [] and len(fake.calls()) == 1


def test_a_family_limit_moves_calls_to_the_other_family(fake):
    fake.env.setenv("FAKE_MODE", "opus_limit,ok")
    cli = fake.make(model="claude-opus-5-5", fallback_model="claude-sonnet-5")
    assert cli.ask("s", "p", schema=SCHEMA).model == "claude-sonnet-5"
    cli.ask("s", "p", schema=SCHEMA)  # the limit is remembered until it resets
    argvs = [c["argv"] for c in fake.calls()]
    assert [a[a.index("--model") + 1] for a in argvs] == ["claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5"]
    assert "--fallback-model" in argvs[0] and all("--fallback-model" not in a for a in argvs[1:])
    assert fake.delays == [0.01]


@pytest.mark.parametrize("modes,retries", [("opus_limit,ok", 0), ("throttle,throttle,opus_limit,ok", 2)],
                         ids=["no-retries", "last-attempt"])
def test_a_family_limit_switches_even_with_the_retries_used_up(fake, modes, retries):
    fake.env.setenv("FAKE_MODE", modes)
    cli = fake.make(model="claude-opus-5-5", fallback_model="claude-sonnet-5", retries=retries)
    assert cli.ask("s", "p", schema=SCHEMA).model == "claude-sonnet-5"
    assert "opus" in cli._limited
    cli.ask("s", "p", schema=SCHEMA)  # straight to Sonnet: the limit was remembered
    argvs = [c["argv"] for c in fake.calls()]
    assert [a[a.index("--model") + 1] for a in argvs[-3:]] == ["claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5"]
    assert fake.delays[-1] == 0.01  # the switch waits one plain back-off


def test_a_rejected_rate_limit_event_names_the_limit(fake):
    fake.env.setenv("FAKE_MODE", "rejected,ok")
    reply = fake.make(model="claude-sonnet-5", fallback_model="claude-opus-5-5").ask("s", "p", schema=SCHEMA)
    # Sonnet's weekly limit: the call moves to Opus
    assert reply.model == "claude-opus-5-5"
    # the failed attempt's usage record keeps the rejected event (§4.8's usage run reads it)
    assert fake.events[0]["error"] == "usage_limit"
    assert fake.events[0]["rate_limit"] == {"status": "rejected", "type": "seven_day_sonnet", "resets_at": 1790300000}
    fake.env.setenv("FAKE_MODE", "rejected")
    with pytest.raises(ClaudeCLIError) as err:
        fake.make(model="claude-sonnet-5", fallback_model=None).ask("s", "p", schema=SCHEMA)
    assert (err.value.kind, err.value.limit, err.value.resets_at) == ("usage_limit", "sonnet", 1790300000)
    assert err.value.rate_limit["status"] == "rejected"


def test_old_limit_message_carries_its_reset_epoch(fake):
    fake.env.setenv("FAKE_MODE", "old_limit")
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA)
    assert err.value.kind == "usage_limit" and err.value.resets_at == 1790300000.0


def test_missing_binary_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("MAATA_CLAUDE_BIN", raising=False)
    monkeypatch.setattr("maata_engine.claude_cli.Path.home", lambda: tmp_path)
    cli = ClaudeCLI(tmp_path / "c", binary=str(tmp_path / "nope"))
    assert cli.status() == {"loggedIn": False, "error": "missing"}
    with pytest.raises(ClaudeCLIError) as err:
        cli.ask("s", "p")
    assert err.value.kind == "missing"


@pytest.mark.parametrize("text,kind", [
    ("Please run /login", "not_signed_in"),
    ("Not logged in · Please run /login", "not_signed_in"),
    ("OAuth token revoked", "not_signed_in"),
    ("OAuth token has expired", "not_signed_in"),
    ("Failed to authenticate: OAuth session expired and could not be refreshed", "not_signed_in"),
    ("You've hit your session limit · resets 3pm", "usage_limit"),
    ("You've hit your weekly limit · resets Oct 7, 2am", "usage_limit"),
    ("You've hit your Sonnet limit", "usage_limit"),
    ("Claude AI usage limit reached|123", "usage_limit"),
    ("API Error: Server is temporarily limiting requests (not your usage limit)", "transient"),
    ("API Error: 529 Overloaded", "transient"),
    ("API Error: 400 Claude Code 2.1.201 does not support this model; version 2.1.280 or newer is required. "
     "Run 'claude update', or update the Claude desktop app, then try again.", "outdated"),
    ("boom", "failed"),
])
def test_documented_messages_are_classified(text, kind):
    assert classify_failure(text) == kind


def test_limit_errors_name_the_limit():
    assert failure("You've hit your Opus limit · resets 3pm", "m").limit == "opus"
    assert failure("You've hit your weekly limit", "m").limit == "weekly"
    assert failure("Claude AI usage limit reached|1790300000", "m").limit is None


def test_parse_edge_cases():
    with pytest.raises(ClaudeCLIError):
        parse_output(json.dumps({"subtype": "error_during_execution", "result": ""}), "", 1)
    ok = parse_output(json.dumps({"subtype": "success", "result": "plain"}), "", 0)
    assert ok.text == "plain" and ok.data is None  # no schema: no unvalidated JSON either
    with pytest.raises(ClaudeCLIError) as err:  # success without structured output, and nothing valid in the text
        parse_output(json.dumps({"type": "result", "subtype": "success", "result": "{}"}), "", 0, SCHEMA)
    assert err.value.kind == "bad_output"
    with pytest.raises(ClaudeCLIError) as err:
        parse_output("", "", 1)
    assert err.value.kind == "failed"


@pytest.mark.parametrize("sep", [" ", " ", "\u0085"])
def test_stream_lines_split_only_at_newlines(sep):
    # Node's JSON.stringify leaves these raw inside strings; they must not cut an event in two.
    init = json.dumps({"type": "system", "subtype": "init"})
    text = f"ఒకటి{sep}రెండు"
    res = json.dumps({"type": "result", "subtype": "success", "result": text, "structured_output": {"lines": [text]}},
                     ensure_ascii=False)
    assert parse_output(f"{init}\n{res}\n", "", 0, SCHEMA).data == {"lines": [text]}


def test_parse_reset():
    london, la = ZoneInfo("Europe/London"), ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 9, 24, 10, 0, tzinfo=london).timestamp()
    assert parse_reset("You've hit your session limit · resets 3pm (Europe/London)", now) == \
        ("3pm (Europe/London)", datetime(2026, 9, 24, 15, 0, tzinfo=london).timestamp())
    assert parse_reset("resets 9:30am (Europe/London)", now)[1] == datetime(2026, 9, 25, 9, 30, tzinfo=london).timestamp()
    assert parse_reset("resets Oct 7, 2am (America/Los_Angeles)", now)[1] == datetime(2026, 10, 7, 2, 0, tzinfo=la).timestamp()
    assert parse_reset("resets Jan 2, 1pm (America/Los_Angeles)", now)[1] == datetime(2027, 1, 2, 13, 0, tzinfo=la).timestamp()
    assert parse_reset("usage limit reached|1790300000", now) == (None, 1790300000.0)
    assert parse_reset("resets soon", now) == ("soon", None)
    assert parse_reset("no reset here", now) == (None, None)


# The scene schema of ARCHITECTURE §4.4: the local validator must cover everything it uses.
SCENE = {"type": "object", "additionalProperties": False, "required": ["lines"],
         "definitions": {"wording": {"type": "object", "additionalProperties": False, "required": ["spoken", "english"],
                                     "properties": {"spoken": {"type": "string", "minLength": 1},
                                                    "english": {"type": "array", "items": {
                                                        "type": "object", "additionalProperties": False,
                                                        "required": ["i", "en"],
                                                        "properties": {"i": {"type": "integer", "minimum": 0},
                                                                       "en": {"type": "string"}}}}}}},
         "properties": {
             "lines": {"type": "array", "items": {
                 "type": "object", "additionalProperties": False, "required": ["id", "full", "delivery"],
                 "properties": {
                     "id": {"type": "integer"},
                     "full": {"$ref": "#/definitions/wording"},
                     "concise": {"$ref": "#/definitions/wording"},
                     "pieces": {"type": "array", "items": {"type": "string"}},
                     "unfinished": {"type": "boolean"},
                     "delivery": {"type": "object", "additionalProperties": False, "required": ["emotion", "energy"],
                                  "properties": {"emotion": {"enum": ["neutral", "happy", "sad"]},
                                                 "energy": {"enum": ["low", "mid", "high"]},
                                                 "question": {"type": "boolean"},
                                                 "emphasis": {"type": "array", "items": {"type": "integer"}}}}}}},
             "glossary_additions": {"type": "array", "items": {
                 "type": "object", "required": ["term", "spoken"],
                 "properties": {"term": {"type": "string"}, "spoken": {"type": "string"},
                                "keep_english": {"type": "boolean"}}}}}}


def _line(**kw) -> dict:
    line = {"id": 3, "full": {"spoken": "ఈ పుస్తకం చాలా బాగుంది", "english": []},
            "delivery": {"emotion": "happy", "energy": "mid", "emphasis": [1]}}
    return {**line, **kw}


def test_the_scene_schema_is_fully_checked():
    check_schema(SCENE)
    assert validate({"lines": [_line()], "glossary_additions": [{"term": "cloud", "spoken": "క్లౌడ్", "note": "x"}]},
                    SCENE) == []
    assert validate({"lines": [_line(id=3.0)]}, SCENE) == []  # 3.0 is an integer in JSON Schema
    bad = {
        "missing id": {"lines": [{k: v for k, v in _line().items() if k != "id"}]},
        "bool id": {"lines": [_line(id=True)]},
        "empty spoken": {"lines": [_line(full={"spoken": "", "english": []})]},
        "negative index": {"lines": [_line(full={"spoken": "ఇది", "english": [{"i": -1, "en": "it"}]})]},
        "unknown emotion": {"lines": [_line(delivery={"emotion": "bored", "energy": "mid"})]},
        "extra field": {"lines": [_line(note="x")]},
        "wrong tier shape": {"lines": [_line(concise="short")]},
        "not an object": ["lines"],
        "no lines": {},
    }
    for why, doc in bad.items():
        assert validate(doc, SCENE), why


@pytest.mark.parametrize("schema", [{"type": "string", "format": "date"}, {"type": "string", "pattern": "x"},
                                    {"type": "decimal"}, {"$ref": "#/definitions/nope"},
                                    {"type": "array", "items": [{"type": "string"}]}])
def test_schemas_the_validator_cannot_fully_check_are_refused(schema):
    with pytest.raises(ValueError):
        check_schema(schema)


def test_effort_and_tags_can_be_set_per_call(fake):
    fake.cli.ask("s", "p", schema=SCHEMA, call="fit", effort="low", tags={"scene": 4, "lines": 2})
    argv = fake.calls()[-1]["argv"]
    assert argv[argv.index("--effort") + 1] == "low"
    ev = fake.events[-1]
    assert ev["effort"] == "low" and ev["scene"] == 4 and ev["lines"] == 2 and ev["call"] == "fit"
    fake.cli.ask("s", "p", schema=SCHEMA)
    argv = fake.calls()[-1]["argv"]
    assert argv[argv.index("--effort") + 1] == "high" and fake.events[-1]["effort"] == "high"


def test_cancel_stops_a_call_in_flight(fake):
    fake.env.setenv("FAKE_MODE", "slow")
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(ClaudeCLIError) as err:
        fake.make(timeout=30, startup_timeout=20).ask("s", "p", schema=SCHEMA, cancel=cancel)
    assert err.value.kind == "cancelled" and time.monotonic() - t0 < 5
    with pytest.raises(ProcessLookupError):  # stopped and reaped
        os.kill(fake.calls()[-1]["pid"], 0)
    assert [e["error"] for e in fake.events] == ["cancelled"] and fake.delays == []


def test_cancel_ends_a_backoff_and_a_cancelled_call_never_starts(fake):
    fake.env.setenv("FAKE_MODE", "throttle")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(ClaudeCLIError) as err:
        fake.make(backoff=30).ask("s", "p", schema=SCHEMA, cancel=cancel)
    assert err.value.kind == "cancelled" and time.monotonic() - t0 < 5 and len(fake.calls()) == 1
    with pytest.raises(ClaudeCLIError) as err:
        fake.cli.ask("s", "p", schema=SCHEMA, cancel=cancel)
    assert err.value.kind == "cancelled" and len(fake.calls()) == 1
