"""Codex transport against an isolated fake executable. No model, network, or real account is used."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from maata_engine import codex_cli as cc
from maata_engine.claude_cli import ClaudeCLIError, validate
from maata_engine.text.scene_prompt import BRIEF_SCHEMA, REVIEW_SCHEMA, SCENE_SCHEMA

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"],
          "properties": {"text": {"type": "string"}, "optional": {"type": "integer"},
                         "nullable": {"type": ["string", "null"]}}}
NOTICE = "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed."

FAKE = r'''#!@PYTHON@
import json, os, signal, sys, time
from pathlib import Path
argv = sys.argv[1:]
if argv == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "codex-cli 0.160.0")); sys.exit(0)
if argv == ["login", "status"]:
    print(os.environ.get("FAKE_AUTH", "Logged in using ChatGPT"), file=sys.stderr)
    sys.exit(1 if "Not logged" in os.environ.get("FAKE_AUTH", "") else 0)
assert argv[0] == "exec"
path = Path(os.environ["FAKE_RECORD"])
n = len(path.read_text().splitlines()) if path.exists() else 0
modes = os.environ.get("FAKE_MODE", "ok").split(",")
mode = modes[min(n, len(modes)-1)]
configs = {argv[i+1].split("=",1)[0]:argv[i+1].split("=",1)[1] for i,x in enumerate(argv) if x == "-c"}
schema_path = argv[argv.index("--output-schema")+1] if "--output-schema" in argv else None
record = {"argv":argv,"cwd":os.getcwd(),"stdin":sys.stdin.read(),"pid":os.getpid(),
          "files":os.listdir("."),"env_names":list(os.environ), "schema_path":schema_path,
          "schema":json.loads(Path(schema_path).read_text()) if schema_path else None,
          "instructions":Path(json.loads(configs["model_instructions_file"])).read_text()}
with path.open("a") as f: f.write(json.dumps(record)+"\n")
def emit(row): print(json.dumps(row,ensure_ascii=False),flush=True)
if mode == "stubborn":
    def note(sig, frame):
        with open(os.environ["FAKE_SIGNALS"],"a") as f: f.write(signal.Signals(sig).name+"\n")
    signal.signal(signal.SIGINT,note);signal.signal(signal.SIGTERM,note)
if mode in ("hang","stubborn"): time.sleep(60)
emit({"type":"thread.started","thread_id":"fake-thread"})
if mode == "notice": emit({"type":"item.completed","item":{"type":"error","message":@NOTICE@}})
emit({"type":"turn.started"})
if mode == "slow": time.sleep(60)
errors = {"auth":"Not logged in", "transient":"HTTP 503 temporarily unavailable",
          "limit":"You've hit your session limit · resets 3pm (Europe/London)",
          "outdated":"unexpected argument '--ignore-user-config'", "failed":"Model not available"}
if mode in errors:
    emit({"type":"turn.failed","error":{"message":errors[mode]}});sys.exit(1)
if mode == "tool":
    emit({"type":"item.started","item":{"type":"command_execution","command":"touch unsafe"}});time.sleep(60)
data = {"text":"నమస్కారం","optional":None,"nullable":None}
if mode == "invalid": data = {"text":99}
if mode == "required_null": data = {"text":None}
if mode == "top_error": emit({"type":"error","message":"authentication failed"})
emit({"type":"item.completed","item":{"type":"agent_message","text":"not JSON" if mode == "malformed" else json.dumps(data,ensure_ascii=False)}})
emit({"type":"turn.completed","usage":{"input_tokens":101,"cached_input_tokens":70,"cache_write_input_tokens":3,"output_tokens":12,"reasoning_output_tokens":4}})
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(FAKE.replace("@PYTHON@", sys.executable).replace("@NOTICE@", repr(NOTICE)))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    record, signals = tmp_path / "calls.jsonl", tmp_path / "signals"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("FAKE_SIGNALS", str(signals))
    events, delays = [], []

    def make(**kw):
        opts = {"binary": str(exe), "timeout": 4, "startup_timeout": 2, "grace": .15,
                "backoff": .01, "retries": 2, "trace": events.append}
        cli = cc.CodexCLI(tmp_path / "cache", **(opts | kw))
        cli._sleep = delays.append
        return cli

    def calls():
        return [json.loads(row) for row in record.read_text().splitlines()] if record.exists() else []

    return SimpleNamespace(make=make, cli=make(), calls=calls, events=events, delays=delays, env=monkeypatch,
                           root=tmp_path, exe=exe, signals=signals)


def test_default_request_is_sealed_and_all_task_text_stays_on_stdin(fake):
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID", "CLAUDECODE"):
        fake.env.setenv(name, "not-for-the-child")
    reply = fake.cli.ask("private video brief", "private transcript", schema=SCHEMA, call="scene", tags={"scene": 2})
    rec = fake.calls()[0]
    assert json.loads(rec["stdin"]) == {"instructions": "private video brief", "input": "private transcript"}
    assert not any(text in arg for arg in rec["argv"] for text in ("private video brief", "private transcript"))
    assert rec["instructions"] == cc.SERVICE_INSTRUCTIONS and "private" not in rec["instructions"]
    assert rec["files"] == [] and Path(rec["cwd"]).name == cc.WORKDIR
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json", "--skip-git-repo-check"):
        assert flag in rec["argv"]
    assert rec["argv"][rec["argv"].index("--sandbox") + 1] == "read-only"
    assert rec["argv"][rec["argv"].index("--model") + 1] == "gpt-6-luna"
    config = {rec["argv"][i + 1].split("=", 1)[0]: rec["argv"][i + 1].split("=", 1)[1]
              for i, arg in enumerate(rec["argv"]) if arg == "-c"}
    assert json.loads(config["model_reasoning_effort"]) == "low"
    for key, expected in cc.CONFIG.items():
        assert json.loads(config[key]) == expected
    assert config["mcp_servers"] == config["plugins"] == config["hooks"] == "{}"
    disabled = {rec["argv"][i + 1] for i, arg in enumerate(rec["argv"]) if arg == "--disable"}
    assert disabled == set(cc.DISABLED_FEATURES)
    assert not (set(rec["env_names"]) & cc._DROP_ENV)
    assert "--output-last-message" not in rec["argv"]
    assert reply.data == {"text": "నమస్కారం", "nullable": None}
    assert reply.model == cc.DEFAULT_MODEL and reply.seconds >= reply.startup_s >= 0
    assert not Path(rec["schema_path"]).exists()
    assert not list((fake.root / "cache").glob("codex-call-*"))
    for path in (fake.root / "cache").rglob("*"):
        if path.is_file():
            assert "private" not in path.read_text()


def test_usage_and_per_call_effort_are_recorded_without_prompt(fake):
    fake.cli.ask("system", "transcript", SCHEMA, "review", effort="medium", tags={"scene": 4, "lines": 2})
    event, = fake.events
    assert event | {"wall_s": 0, "startup_s": 0} == {
        "event": "claude", "provider": "codex", "call": "review", "scene": 4, "lines": 2, "attempt": 1,
        "requested": "gpt-6-luna", "model": "gpt-6-luna", "effort": "medium", "wall_s": 0, "startup_s": 0,
        "input_tokens": 101, "cache_read_tokens": 70, "cache_creation_tokens": 3, "cache_1h_tokens": 0,
        "output_tokens": 12, "reasoning_output_tokens": 4, "rate_limit": None, "error": None}
    assert "model_reasoning_effort=\"medium\"" in fake.calls()[0]["argv"]


def test_disabled_code_mode_startup_notice_does_not_discard_valid_answer(fake):
    fake.env.setenv("FAKE_MODE", "notice")
    assert fake.cli.ask("s", "p", SCHEMA).data["text"] == "నమస్కారం"


@pytest.mark.parametrize("mode,kind", [("invalid", "bad_output"), ("required_null", "bad_output"),
                                      ("malformed", "bad_output"), ("auth", "not_signed_in"),
                                      ("limit", "usage_limit"), ("outdated", "outdated"), ("failed", "failed"),
                                      ("top_error", "not_signed_in")])
def test_failures_do_not_become_success_or_silently_switch_models(fake, mode, kind):
    fake.env.setenv("FAKE_MODE", mode)
    with pytest.raises(ClaudeCLIError) as error:
        fake.cli.ask("s", "p", SCHEMA)
    assert error.value.kind == kind and "Claude" not in str(error.value)
    assert len(fake.calls()) == 1 and fake.events[0]["error"] == kind
    assert not list((fake.root / "cache").glob("codex-call-*"))
    if mode == "limit":
        assert error.value.limit == "session" and error.value.resets_at is not None


def test_transient_failures_retry_with_backoff_and_keep_model(fake):
    fake.env.setenv("FAKE_MODE", "transient,transient,ok")
    assert fake.cli.ask("s", "p", SCHEMA).data["text"] == "నమస్కారం"
    assert fake.delays == [.01, .02]
    assert [e["error"] for e in fake.events] == ["transient", "transient", None]
    assert {e["model"] for e in fake.events} == {cc.DEFAULT_MODEL}


def test_retry_budget_is_bounded(fake):
    fake.env.setenv("FAKE_MODE", "transient")
    with pytest.raises(ClaudeCLIError, match="Codex"):
        fake.cli.ask("s", "p", SCHEMA)
    assert len(fake.calls()) == 3 and fake.delays == [.01, .02]


@pytest.mark.parametrize("mode,kind", [("hang", "stalled"), ("slow", "timeout"), ("tool", "failed")])
def test_watchdog_stops_hangs_and_unexpected_tools(fake, mode, kind):
    fake.env.setenv("FAKE_MODE", mode)
    with pytest.raises(ClaudeCLIError) as error:
        fake.make(timeout=.7, startup_timeout=.35, retries=0).ask("s", "p", SCHEMA)
    assert error.value.kind == kind
    with pytest.raises(ProcessLookupError):
        os.kill(fake.calls()[0]["pid"], 0)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process-group signals")
def test_watchdog_escalates_to_kill_for_stubborn_process(fake):
    fake.env.setenv("FAKE_MODE", "stubborn")
    with pytest.raises(ClaudeCLIError) as error:
        fake.make(timeout=1, startup_timeout=.35, retries=0).ask("s", "p", SCHEMA)
    assert error.value.kind == "stalled"
    assert fake.signals.read_text().splitlines() == ["SIGINT", "SIGTERM"]
    with pytest.raises(ProcessLookupError):
        os.kill(fake.calls()[0]["pid"], 0)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process-group signals")
def test_sigkill_is_reaped_even_when_the_last_grace_wait_expires(fake, monkeypatch):
    class SlowReap:
        pid, reaped = 12345, False

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                raise cc.subprocess.TimeoutExpired("fake", timeout)
            self.reaped = True
            return -9

    sent = []
    monkeypatch.setattr(cc.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    proc = SlowReap()
    fake.cli._stop(proc)
    assert proc.reaped and sent == [(proc.pid, sig) for sig in (cc.signal.SIGINT, cc.signal.SIGTERM, cc.signal.SIGKILL)]


def test_cancel_stops_an_inflight_request_and_does_not_start_an_already_cancelled_one(fake):
    fake.env.setenv("FAKE_MODE", "slow")
    cancel = threading.Event()
    timer = threading.Timer(.4, cancel.set)
    timer.start()
    with pytest.raises(ClaudeCLIError) as error:
        fake.cli.ask("s", "p", SCHEMA, cancel=cancel)
    timer.join()
    assert error.value.kind == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(fake.calls()[0]["pid"], 0)
    with pytest.raises(ClaudeCLIError) as error:
        fake.cli.ask("s", "p", SCHEMA, cancel=cancel)
    assert error.value.kind == "cancelled" and len(fake.calls()) == 1


def test_cancel_interrupts_retry_backoff(fake):
    fake.env.setenv("FAKE_MODE", "transient")
    cancel = threading.Event()
    timer = threading.Timer(.4, cancel.set)
    timer.start()
    with pytest.raises(ClaudeCLIError) as error:
        fake.make(backoff=20).ask("s", "p", SCHEMA, cancel=cancel)
    timer.join()
    assert error.value.kind == "cancelled" and len(fake.calls()) == 1


def test_cancel_during_version_probe_never_starts_a_text_request(fake, monkeypatch):
    cancel = threading.Event()
    monkeypatch.setattr(fake.cli, "probe", lambda: cancel.set())
    monkeypatch.setattr(fake.cli, "_run", lambda *args: pytest.fail("cancelled probe started a text request"))
    with pytest.raises(ClaudeCLIError) as error:
        fake.cli.ask("private brief", "private transcript", SCHEMA, cancel=cancel)
    assert error.value.kind == "cancelled" and not fake.calls()
    assert not list((fake.root / "cache").glob("codex-call-*"))


def test_cancel_before_process_spawn_does_not_open_a_subprocess(fake, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(cc.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("cancelled call spawned a process"))
    with pytest.raises(ClaudeCLIError) as error:
        fake.cli._run([str(fake.exe), "exec"], "private transcript", cancel)
    assert error.value.kind == "cancelled" and not fake.calls()


@pytest.mark.parametrize("schema", [SCENE_SCHEMA, BRIEF_SCHEMA, REVIEW_SCHEMA])
def test_production_schemas_become_strict_without_mutating_the_original(schema):
    original = json.dumps(schema, ensure_ascii=False)
    wire = cc.strict_schema(schema)
    assert json.dumps(schema, ensure_ascii=False) == original
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node.get("properties", {}))
        for value in node.values():
            if isinstance(value, dict):
                visit(value)
                for sub in value.values():
                    visit(sub)
            elif isinstance(value, list):
                for sub in value:
                    visit(sub)
    visit(wire)


def test_nullable_required_and_refs_survive_adapter_but_optional_absent_fields_are_removed():
    schema = {"type": "object", "required": ["nested"], "properties": {"nested": {"$ref": "#/definitions/entry"}},
              "definitions": {"entry": {"type": "object", "required": ["nullable", "text"], "properties": {
                  "nullable": {"type": ["string", "null"]}, "text": {"type": "string"},
                  "optional": {"$ref": "#/definitions/wording"}}},
                  "wording": {"type": "object", "required": ["spoken"], "properties": {"spoken": {"type": "string"}}}}}
    value = {"nested": {"nullable": None, "text": "value", "optional": None}}
    got = cc.prune_optional_nulls(value, schema)
    assert got == {"nested": {"nullable": None, "text": "value"}} and validate(got, schema) == []
    value["nested"]["text"] = None
    assert validate(cc.prune_optional_nulls(value, schema), schema)
    assert cc.strict_schema(schema)["properties"]["nested"] == {"$ref": "#/definitions/entry"}


def test_concurrent_calls_keep_schema_files_independent_and_workdir_empty(fake):
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda text: fake.cli.ask("s", text, SCHEMA), ("one", "two")))
    assert len(replies) == 2 and len({r["schema_path"] for r in fake.calls()}) == 2
    assert all(r["files"] == [] and not Path(r["schema_path"]).exists() for r in fake.calls())


def test_health_checks_chatgpt_signin_without_exposing_credentials(fake):
    assert fake.cli.health() == {"provider": "codex", "installed": True, "version": "0.160.0", "signedIn": True,
                                 "models": [cc.DEFAULT_MODEL], "model": cc.DEFAULT_MODEL, "problem": None, "message": ""}
    for auth in ("Not logged in", "Logged in using an API key"):
        fake.env.setenv("FAKE_AUTH", auth)
        health = fake.cli.health()
        assert health["signedIn"] is False and health["problem"] == "not_signed_in"
        assert "codex login" in health["message"]
    fake.env.setenv("FAKE_AUTH", "unknown response")
    assert fake.cli.health()["signedIn"] is None


def test_old_or_missing_cli_fails_before_model_call(fake, monkeypatch):
    fake.env.setenv("FAKE_VERSION", "codex-cli 0.159.0")
    assert fake.cli.health()["problem"] == "outdated"
    with pytest.raises(ClaudeCLIError):
        fake.cli.ask("s", "p", SCHEMA)
    assert not fake.calls()
    monkeypatch.setattr(cc, "find_binary", lambda explicit=None: None)
    assert fake.make().health()["problem"] == "missing"


def test_finder_path_falls_back_to_the_bundled_cli(monkeypatch):
    bundled = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"
    monkeypatch.delenv("MAATA_CODEX_BIN", raising=False)
    monkeypatch.setattr(cc.shutil, "which", lambda _: None)
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == bundled)
    monkeypatch.setattr(cc.os, "access", lambda path, _: path == bundled)
    assert cc.find_binary() == bundled


@pytest.mark.parametrize("kwargs", [{"model": "claude-haiku-4-5-20251001"}, {"model": "fast"},
                                   {"fallback_model": "gpt-6-sol"}, {"effort": "made-up"}])
def test_unsupported_model_fallback_or_effort_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        cc.CodexCLI(tmp_path, **kwargs)
