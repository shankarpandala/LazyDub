"""The `claude` CLI (Claude Code, headless) as the engine's only text model (ADR-019).

Translation text goes out through the maintainer's own signed-in CLI; audio never does. Every call is sealed off from the
user's Claude Code setup: `--safe-mode` (no hooks, plugins, MCP servers, CLAUDE.md or skills, but subscription sign-in
still works), no tools, our own system prompt, nothing persisted, fast mode off, non-essential traffic and telemetry off.
The prompt goes in on stdin, so long scenes never hit argv limits and no transcript line shows up in `ps`. The system
prompt does go in argv, and with it the scene translator's video brief: the metadata and, from brief v1 on, the names,
terms, idioms and ASR fixes it quotes from the transcript.

Hardening (docs/research/dubbing-2026-09/ARCHITECTURE.md §4.4, §4.9):
- a version gate: schema validation needs 2.1.205, and each model has its own minimum (the fallback is dropped when the
  installed CLI can't run it);
- failures classed from the CLI's documented messages, so the UI can say what to do;
- a watchdog: no stream event at start (#91987: an interactive session on the same version holds a lock) or no answer in
  time -> SIGINT, then SIGTERM, then SIGKILL; transient failures and stalls are retried after an exponential back-off,
  never at once;
- a cancel event stops a call (SIGINT first), also while it waits to retry, so a seek can drop calls it no longer needs;
- a schema-bound reply counts only if it validates, whether it came as `structured_output` or was recovered from text;
- one fixed working directory under the engine's cache dir: the CLI puts the working directory in the prompt prefix, so
  separate calls share the prompt cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("maata.claude")

# Env the CLI inherits: telemetry, error reports, /feedback and every non-essential request off; no auto-update mid-session.
QUIET_ENV = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
# Set when the engine itself runs inside a Claude Code session (development); the CLI refuses to nest under it.
NESTING_ENV = ("CLAUDECODE",)
# Fast mode bills usage credits only on Pro/Max, so it is never on, whatever the user's own settings say.
SETTINGS = '{"fastMode":false}'
WORKDIR = "claude-cwd"  # under the engine's cache dir; always empty

MIN_VERSION = (2, 1, 205)  # an invalid --json-schema fails at start from here on; before, it was silently ignored
# The server enforces a minimum CLI version per model; a model not listed here is left to the server to refuse.
MODEL_MIN_VERSION = {"claude-sonnet-5": MIN_VERSION, "claude-opus-5-5": (2, 1, 280)}
# The §7.1 bake-off (ADR-019, 2026-09-25): Opus 5.5 at medium effort had no major meaning errors on 75 lines under both
# an Opus and a Sonnet judge panel, against 3 lines for Sonnet 5, and ran 2.4x faster. Sonnet 5 covers an Opus weekly
# limit and CLIs older than 2.1.280.
DEFAULT_MODEL, DEFAULT_FALLBACK = "claude-opus-5-5", "claude-sonnet-5"
LIMIT_HOLD = 3600.0  # s a family stays avoided after its weekly limit is hit, when the CLI didn't say when it resets

RETRYABLE = frozenset({"transient", "stalled"})
_STALL = ("Claude Code didn't start within {:.0f} s. If Claude Code is open in another window on the same version, it can "
          "block Maata (known issue #91987).")


class ClaudeCLIError(RuntimeError):
    """`kind` tells the UI what to show: missing | not_signed_in | outdated | usage_limit | transient | stalled | timeout |
    bad_output | failed | cancelled. `ask` retries `transient` and `stalled` itself. A usage limit names which one
    (`limit`: session | weekly | opus | sonnet | overage, when known) and when it resets (`resets_at`, epoch s; `resets`,
    the CLI's words)."""

    def __init__(self, kind: str, message: str, limit: str | None = None, resets: str | None = None,
                 resets_at: float | None = None) -> None:
        super().__init__(message)
        self.kind, self.limit, self.resets, self.resets_at = kind, limit, resets, resets_at
        self.usage: dict = {}          # what the failed call used, when the CLI reported it
        self.model: str | None = None
        self.rate_limit: dict | None = None  # the last `rate_limit_event`, e.g. the rejected one behind a usage limit


@dataclass(slots=True)
class ClaudeReply:
    text: str
    data: dict | list | None  # the validated object when a JSON schema was given
    seconds: float
    usage: dict = field(default_factory=dict)
    model: str | None = None  # the model that answered (from `modelUsage`; differs from the one asked for after a fallback)
    rate_limit: dict | None = None  # the last `rate_limit_event`, normalised (see `_rate_limit`)
    startup_s: float | None = None  # seconds to the CLI's first stream event


@dataclass(frozen=True, slots=True)
class CLIInfo:
    version: tuple[int, int, int]

    @property
    def text(self) -> str:
        return ".".join(map(str, self.version))

    @property
    def models(self) -> tuple[str, ...]:
        """The known models this version can run."""
        return tuple(m for m in MODEL_MIN_VERSION if self.supports(m))

    def supports(self, model: str) -> bool:
        return self.version >= MODEL_MIN_VERSION.get(model, MIN_VERSION)


def find_binary(explicit: str | None = None) -> str | None:
    """`MAATA_CLAUDE_BIN`, then PATH, then the installer's default location."""
    for cand in (explicit, os.environ.get("MAATA_CLAUDE_BIN"), shutil.which("claude"), str(Path.home() / ".local/bin/claude")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def parse_version(text: str) -> tuple[int, int, int] | None:
    """`claude --version` prints e.g. "2.1.281 (Claude Code)"."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def _check_model(model: str) -> None:
    # Full ids only: aliases resolve differently per CLI version (`opus` was Opus 4.8 on 2.1.201). Never Fable: in -p it
    # can bill usage credits without asking (research 05 A7).
    if not model.startswith("claude-") or "fable" in model.lower():
        raise ValueError(f"not an allowed model id: {model!r}")


def _family(model: str) -> str | None:
    return next((f for f in ("opus", "sonnet") if f in model), None)


# ---- failure classes ---------------------------------------------------------------------------------------------------
# The CLI's documented messages (code.claude.com/docs/en/errors, read 2026-09-24; research 05 A1, A10), e.g.
#   "You've hit your weekly limit · resets …" (and session, Opus, Sonnet)       -> usage_limit
#   "API Error: Server is temporarily limiting requests (not your usage limit)"  -> transient
#   "Not logged in · Please run /login", "OAuth token has expired",
#   "Failed to authenticate: OAuth session expired and could not be refreshed"    -> not_signed_in
#   "API Error: 400 Claude Code 2.1.201 does not support this model; version 2.1.280 or newer is required"  -> outdated
_OUTDATED = re.compile(r"does not support this model|or newer is required|claude_code_version_too_old|run .claude update", re.I)
_AUTH = re.compile(r"not (?:logged|signed) in|please run /login|invalid api key|failed to authenticate|authentication_failed|"
                   r"oauth (?:token|session)\b[^\n]*?(?:expired|revoked)|unauthori[sz]ed", re.I)
_LIMIT = re.compile(r"hit your (?:(session|weekly|opus|sonnet) )?limit|usage limit reached", re.I)
_TRANSIENT = re.compile(r"temporarily limiting requests|not your usage limit|overloaded|too many requests|rate.?limited|"
                        r"api error: (?:429|5\d\d)\b", re.I)
_LIMIT_TYPES = {"five_hour": "session", "seven_day": "weekly", "seven_day_opus": "opus", "seven_day_sonnet": "sonnet",
                "overage": "overage"}


def classify_failure(text: str) -> str:
    for kind, rx in (("outdated", _OUTDATED), ("not_signed_in", _AUTH), ("usage_limit", _LIMIT), ("transient", _TRANSIENT)):
        if rx.search(text):
            return kind
    return "failed"


def failure(text: str, message: str, now: float | None = None) -> ClaudeCLIError:
    """The error `text` describes; a usage limit also gets its name and reset time when the text has them."""
    kind = classify_failure(text)
    if kind != "usage_limit":
        return ClaudeCLIError(kind, message)
    m = _LIMIT.search(text)
    resets, at = parse_reset(text, now)
    return ClaudeCLIError(kind, message, limit=m[1].lower() if m and m[1] else None, resets=resets, resets_at=at)


_EPOCH = re.compile(r"\|(\d{10})\b")  # "Claude AI usage limit reached|1790300000" (older CLIs)
_RESETS = re.compile(r"\bresets\s+(?:at\s+)?([^·\n]+)", re.I)
_CLOCK = re.compile(r"(?:\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(?:at\s+)?)?"
                    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b(?:\s*\(([^)]+)\))?", re.I)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def parse_reset(text: str, now: float | None = None) -> tuple[str | None, float | None]:
    """When a usage limit resets: (the CLI's words, epoch seconds), each None when the text doesn't say. The words look
    like "3pm (Europe/London)" or "Oct 7, 2am (America/Los_Angeles)"; a time without a date is its next occurrence, in
    the named zone or else local time."""
    if m := _EPOCH.search(text):
        return None, float(m[1])
    m = _RESETS.search(text)
    if not m:
        return None, None
    words = m[1].strip().rstrip(".")
    c = _CLOCK.search(words)
    if not c:
        return words, None
    mon, day, hour, minute, ampm, zone = c.groups()
    try:
        tz = ZoneInfo(zone.strip()) if zone else None
    except (ZoneInfoNotFoundError, ValueError):
        tz = None
    ref = datetime.fromtimestamp(time.time() if now is None else now, tz)  # tz None: local time
    try:
        at = ref.replace(hour=int(hour) % 12 + (12 if ampm.lower() == "pm" else 0), minute=int(minute or 0), second=0,
                         microsecond=0)
        if mon:
            at = at.replace(month=_MONTHS.index(mon.lower()) + 1, day=int(day))
            if at < ref - timedelta(days=1):  # a date that has passed is next year's (read in December for January)
                at = at.replace(year=at.year + 1)
        elif at <= ref:
            at += timedelta(days=1)
    except ValueError:
        return words, None
    return words, at.timestamp()


def _rate_limit(ev: dict) -> dict:
    """A `rate_limit_event` (stream-json), trimmed to what the engine uses. `utilization` is optional in the CLI's own
    types and the overage fields appear only in raw events, so each is kept only when present (research 05 A5)."""
    info = ev.get("rate_limit_info") or ev
    at = info.get("resetsAt")
    if isinstance(at, (int, float)) and at > 1e12:  # milliseconds
        at = at / 1000
    out = {"status": info.get("status"), "type": info.get("rateLimitType"), "utilization": info.get("utilization"),
           "resets_at": at, "overage_status": info.get("overageStatus"), "using_overage": info.get("isUsingOverage")}
    return {k: v for k, v in out.items() if v is not None}


def _limit_error(rate: dict) -> ClaudeCLIError:
    limit = _LIMIT_TYPES.get(rate.get("type") or "")
    return ClaudeCLIError("usage_limit", f"Claude usage limit reached ({limit or 'plan'})", limit=limit,
                          resets_at=rate.get("resets_at"))


# ---- structured output -------------------------------------------------------------------------------------------------
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "minLength", "minimum", "$ref",
             "definitions"}
_ANNOTATIONS = {"$schema", "$comment", "title", "description", "default", "examples"}


def check_schema(schema: dict) -> None:
    """Raise ValueError when `schema` uses anything `validate` doesn't check, so a new schema can't be half-validated.
    `format` is left out on purpose: the CLI treats it as an annotation only (research 05 A12)."""

    def walk(node: object, where: str) -> None:
        if isinstance(node, bool):
            return
        if not isinstance(node, dict):
            raise ValueError(f"{where}: not a schema")
        unknown = set(node) - _KEYWORDS - _ANNOTATIONS
        if unknown:
            raise ValueError(f"{where}: unsupported schema keyword(s) {sorted(unknown)}")
        types = node.get("type", [])
        for t in types if isinstance(types, list) else [types]:
            if t not in _TYPES and t not in ("integer", "number"):
                raise ValueError(f"{where}: unknown type {t!r}")
        if "$ref" in node:
            _resolve(schema, node["$ref"])
        for key in ("properties", "definitions"):
            for name, sub in (node.get(key) or {}).items():
                walk(sub, f"{where}.{key}.{name}")
        for key in ("items", "additionalProperties"):
            if key in node:
                walk(node[key], f"{where}.{key}")

    walk(schema, "$")


def _resolve(root: dict, ref: str) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $refs are supported: {ref!r}")
    node: object = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"unresolvable $ref {ref!r}")
        node = node[part]
    return node  # type: ignore[return-value]


def _is_type(value: object, t: str | list) -> bool:
    if isinstance(t, list):
        return any(_is_type(value, x) for x in t)
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t])


def validate(value: object, schema: dict | bool, root: dict | None = None, path: str = "$") -> list[str]:
    """Where `value` breaks `schema`, empty when it conforms. Covers the draft-07 subset the engine's schemas use (see
    `check_schema`); a `$ref` replaces its siblings, as in draft-07."""
    if isinstance(schema, bool):
        return [] if schema else [f"{path}: not allowed"]
    root = schema if root is None else root
    if "$ref" in schema:
        return validate(value, _resolve(root, schema["$ref"]), root, path)
    if "type" in schema and not _is_type(value, schema["type"]):
        return [f"{path}: expected {schema['type']}"]
    errs: list[str] = []
    if "enum" in schema and not any(e == value and isinstance(e, bool) == isinstance(value, bool) for e in schema["enum"]):
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errs.append(f"{path}: shorter than {schema['minLength']}")
    if _is_type(value, "number") and "minimum" in schema and value < schema["minimum"]:  # type: ignore[operator]
        errs.append(f"{path}: below {schema['minimum']}")
    if isinstance(value, dict):
        props, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        errs += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], root, f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: unexpected {k!r}")
            elif isinstance(extra, dict):
                errs += validate(v, extra, root, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], root, f"{path}[{i}]")
    return errs


def _json_in(text: str) -> dict | list | None:
    """The first JSON object or array in a reply, tolerating a ```json fence around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---- output ------------------------------------------------------------------------------------------------------------
def _events(stdout: str) -> tuple[list[dict], str]:
    """The JSON objects the CLI printed (one document with `json`, one per line with `stream-json`) and the rest of the
    text, which is where a CLI that fails before it starts streaming says why."""
    try:
        doc = json.loads(stdout)
        return ([doc], "") if isinstance(doc, dict) else ([], stdout)
    except json.JSONDecodeError:
        pass
    events, other = [], []
    # "\n" only: `splitlines` also breaks at U+2028, U+2029 and U+0085, which JSON.stringify leaves raw inside strings.
    for line in stdout.split("\n"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            ev = None
        if isinstance(ev, dict):
            events.append(ev)
        elif line.strip():
            other.append(line)
    return events, "\n".join(other)


def _resolved_model(doc: dict) -> str | None:
    """The model that did the work: `modelUsage` may list several after a fallback; the one with the most output wins."""
    mu = doc.get("modelUsage") or {}
    return max(mu, key=lambda k: (mu[k] or {}).get("outputTokens", 0) if isinstance(mu[k], dict) else 0) if mu else None


def parse_output(stdout: str, stderr: str, returncode: int, schema: dict | None = None) -> ClaudeReply:
    """Turn the CLI's output into a reply, or raise with a `kind` the UI can act on. With a schema, the reply's `data` is
    `structured_output`, or JSON recovered from the text, and either counts only if it validates against the schema."""
    events, other = _events(stdout)
    results = [e for e in events if e.get("type", "result") == "result"]
    limits = [_rate_limit(e) for e in events if e.get("type") == "rate_limit_event"]
    rate = limits[-1] if limits else None
    rejected = rate is not None and rate.get("status") == "rejected"
    if not results:
        blob = f"{other}\n{stderr}".strip()
        if rejected:
            err = _limit_error(rate)
        elif blob:
            err = failure(blob, f"The Claude CLI failed (exit {returncode}): {blob[:300]}")
        else:
            err = ClaudeCLIError("failed", f"The Claude CLI failed (exit {returncode}) without saying why")
        err.rate_limit = rate
        raise err
    doc = results[-1]
    text, usage, model = doc.get("result") or "", doc.get("usage") or {}, _resolved_model(doc)
    subtype = doc.get("subtype", "success")
    err: ClaudeCLIError | None = None
    if doc.get("is_error") or subtype != "success":
        if subtype == "error_max_structured_output_retries":
            err = ClaudeCLIError("bad_output", "Claude couldn't produce output matching the schema")
        else:
            err = failure(f"{text}\n{other}\n{stderr}", f"Claude reported an error: {text[:300] or subtype}")
            if rejected and err.kind in ("failed", "usage_limit"):  # the event names the limit and its reset exactly
                ev = _limit_error(rate)
                err = ClaudeCLIError("usage_limit", str(err), limit=ev.limit or err.limit, resets=err.resets,
                                     resets_at=ev.resets_at or err.resets_at)
    data = None
    if err is None and schema is not None:
        data = doc.get("structured_output")
        if data is None:  # an answer in the text instead: accepted only if it validates (research 05 A12)
            data = _json_in(text)
        problems = validate(data, schema) if data is not None else ["no structured output"]
        if problems:
            err = ClaudeCLIError("bad_output", f"Claude's reply didn't match the schema ({problems[0]}): {text[:160]!r}")
    if err is not None:
        err.usage, err.model, err.rate_limit = usage, model, rate
        raise err
    return ClaudeReply(text=text, data=data, seconds=0.0, usage=usage, model=model, rate_limit=rate)


@dataclass(slots=True)
class _Run:
    stdout: str
    stderr: str
    returncode: int
    startup_s: float | None


class ClaudeCLI:
    """One sealed `claude -p` call per request. Thread-safe: every call is its own process.

    `trace`, when given, receives one usage record per CLI process (units.jsonl's "claude" event): the call type, the
    model asked for and the one that answered, seconds, and input, cache-read, cache-creation and output tokens.
    """

    def __init__(self, cache_dir: Path, model: str = DEFAULT_MODEL, fallback_model: str | None = DEFAULT_FALLBACK,
                 effort: str | None = None, binary: str | None = None, timeout: float = 240.0,
                 startup_timeout: float = 90.0, grace: float = 5.0, retries: int = 3, backoff: float = 10.0,
                 trace: Callable[[dict], None] | None = None) -> None:
        for m in (model, fallback_model):
            if m is not None:
                _check_model(m)
        self.model, self.fallback_model, self.effort = model, fallback_model, effort
        self.timeout, self.startup_timeout, self.grace = timeout, startup_timeout, grace
        self.retries, self.backoff, self.trace = retries, backoff, trace
        self.binary = find_binary(binary)
        # One fixed, empty directory for every call of the engine: nothing to discover there even if safe mode ever
        # changes, and the same prompt prefix for all calls (research 05 A8).
        self.workdir = cache_dir / WORKDIR
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._info: CLIInfo | None = None
        self._limited: dict[str, float] = {}  # model family -> epoch until which its weekly limit holds
        self._sleep = time.sleep

    # -- plumbing ---------------------------------------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in NESTING_ENV}
        env.update(QUIET_ENV)
        return env

    def _missing(self) -> ClaudeCLIError:
        return ClaudeCLIError("missing", "The Claude CLI isn't installed. Install Claude Code, then run `claude auth login`.")

    def _command(self, system: str, schema: dict | None, model: str, fallback: str | None, effort: str | None) -> list[str]:
        if not self.binary:
            raise self._missing()
        cmd = [self.binary, "-p", "--output-format", "stream-json", "--verbose", "--model", model, "--system-prompt", system,
               "--tools", "", "--safe-mode", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands",
               "--settings", SETTINGS]
        if effort:
            cmd += ["--effort", effort]
        if fallback:
            cmd += ["--fallback-model", fallback]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False, separators=(",", ":"))]
        return cmd

    def probe(self) -> CLIInfo:
        """The installed CLI's version, checked once: raises `missing`, or `outdated` below MIN_VERSION."""
        if self._info is not None:
            return self._info
        if not self.binary:
            raise self._missing()
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=30,
                                 env=self._env(), cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError("failed", "`claude --version` didn't answer") from exc
        except OSError as exc:
            raise ClaudeCLIError("missing", f"Couldn't start the Claude CLI: {exc}") from exc
        version = parse_version(out.stdout)
        if version is None:
            raise ClaudeCLIError("failed", f"Couldn't read the Claude CLI's version: {(out.stdout + out.stderr)[:120]!r}")
        info = CLIInfo(version)
        if version < MIN_VERSION:
            raise ClaudeCLIError("outdated", f"Claude Code {info.text} is too old; Maata needs "
                                             f"{'.'.join(map(str, MIN_VERSION))} or newer. Run `claude update`.")
        log.info("Claude Code %s; can run %s", info.text, ", ".join(info.models))
        self._info = info
        return info

    def status(self) -> dict:
        """`claude auth status` as a dict ({"loggedIn": bool, ...}); {"loggedIn": False, "error": ...} when it can't run."""
        if not self.binary:
            return {"loggedIn": False, "error": "missing"}
        try:
            out = subprocess.run([self.binary, "auth", "status"], capture_output=True, text=True, timeout=30,
                                 env=self._env(), cwd=self.workdir)
            return json.loads(out.stdout)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
            return {"loggedIn": False, "error": str(exc)[:200]}

    def health(self) -> dict:
        """What the UI shows about the CLI when it connects: installed, version, signed in (None when `claude auth
        status` can't tell), the models this version can run, and the problem (a ClaudeCLIError kind, with a message)
        when there is one."""
        out: dict = {"installed": self.binary is not None, "version": None, "signedIn": None, "models": [],
                     "model": self.model, "problem": None, "message": ""}
        try:
            info = self.probe()
        except ClaudeCLIError as err:
            return {**out, "problem": err.kind, "message": str(err)}
        out.update(version=info.text, models=list(info.models))
        auth = self.status()
        if isinstance(auth.get("loggedIn"), bool) and "error" not in auth:
            out["signedIn"] = auth["loggedIn"]
        if out["signedIn"] is False:
            out.update(problem="not_signed_in", message="Claude Code isn't signed in. Run `claude auth login`.")
        elif not info.supports(self.model):
            need = ".".join(map(str, MODEL_MIN_VERSION.get(self.model, MIN_VERSION)))
            if self.fallback_model and info.supports(self.fallback_model):  # translation still runs, on the fallback
                out.update(model=self.fallback_model,
                           message=f"Claude Code {info.text} can't run {self.model} (it needs {need} or newer), so Maata "
                                   f"translates with {self.fallback_model}. Run `claude update` for the better model.")
            else:
                out.update(problem="outdated", message=f"Claude Code {info.text} can't run {self.model}; it needs {need} "
                                                       "or newer. Run `claude update`.")
        return out

    def _models(self, info: CLIInfo) -> tuple[str, str | None]:
        """This call's model and fallback: the configured pair, swapped while the model's family is at its weekly limit
        (the fallback covers overload, not limits); the fallback only when this CLI can run it."""
        model, fallback = self.model, self.fallback_model
        if fallback and self._is_limited(_family(model)) and not self._is_limited(_family(fallback)):
            model, fallback = fallback, None
        if not info.supports(model) and fallback and info.supports(fallback):
            model, fallback = fallback, None  # an older CLI runs the fallback until `claude update` (health() says so)
        if not info.supports(model):
            need = ".".join(map(str, MODEL_MIN_VERSION.get(model, MIN_VERSION)))
            raise ClaudeCLIError("outdated", f"Claude Code {info.text} can't run {model}; it needs {need} or newer. "
                                             "Run `claude update`.")
        return model, fallback if fallback and info.supports(fallback) else None

    def _is_limited(self, family: str | None) -> bool:
        return family is not None and time.time() < self._limited.get(family, 0.0)

    # -- one process, under the watchdog ----------------------------------------------------------------------------
    def _wait(self, proc: subprocess.Popen, timeout: float, cancel: threading.Event | None) -> None:
        """`proc.wait(timeout)`, except that setting `cancel` stops the process and raises `cancelled` within 0.1 s."""
        if cancel is None:
            proc.wait(timeout=timeout)
            return
        end = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                self._stop(proc)
                raise ClaudeCLIError("cancelled", "The Claude call was cancelled")
            left = end - time.monotonic()
            try:
                proc.wait(timeout=max(min(left, 0.1), 0.0))
                return
            except subprocess.TimeoutExpired:
                if left <= 0.1:
                    raise

    def _run(self, cmd: list[str], prompt: str, cancel: threading.Event | None = None) -> _Run:
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", errors="replace", env=self._env(), cwd=self.workdir,
                                    start_new_session=True)  # its own process group, so the watchdog reaches any helper
        except OSError as exc:
            raise ClaudeCLIError("missing", f"Couldn't start the Claude CLI: {exc}") from exc
        out: list[str] = []
        err: list[str] = []
        first: list[float] = []

        def feed() -> None:
            try:
                with proc.stdin:
                    proc.stdin.write(prompt)
            except OSError:
                pass  # it exited or stopped reading; its output says why

        def read_out() -> None:
            for line in proc.stdout:
                if not first:
                    first.append(time.monotonic() - t0)
                out.append(line)

        readers = [threading.Thread(target=read_out, daemon=True),
                   threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)]
        for th in (threading.Thread(target=feed, daemon=True), *readers):
            th.start()
        try:
            self._wait(proc, min(self.startup_timeout, self.timeout), cancel)
        except subprocess.TimeoutExpired:
            if not first and self.startup_timeout <= self.timeout:  # else the overall timeout ran out first
                self._stop(proc)
                raise ClaudeCLIError("stalled", _STALL.format(self.startup_timeout)) from None
            try:
                self._wait(proc, max(self.timeout - (time.monotonic() - t0), 0.0), cancel)
            except subprocess.TimeoutExpired:
                self._stop(proc)
                raise ClaudeCLIError("timeout", f"Claude didn't answer within {self.timeout:.0f} s") from None
        finally:
            for th, pipe in zip(readers, (proc.stdout, proc.stderr)):
                th.join(timeout=self.grace)
                if not th.is_alive():
                    pipe.close()
        return _Run("".join(out), "".join(err), proc.returncode, first[0] if first else None)

    def _stop(self, proc: subprocess.Popen) -> None:
        """SIGINT (the CLI ends its turn), then SIGTERM, then SIGKILL, each after `grace` s (research 05 A5, A9)."""
        if not hasattr(os, "killpg"):  # Windows: no signals to a child's group, only TerminateProcess
            proc.kill()
            proc.wait()
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                return
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=self.grace)
                return
            except subprocess.TimeoutExpired:
                continue

    # -- the call ---------------------------------------------------------------------------------------------------
    def ask(self, system: str, prompt: str, schema: dict | None = None, call: str = "text", *, effort: str | None = None,
            cancel: threading.Event | None = None, tags: dict | None = None) -> ClaudeReply:
        """One answer. `call` names the call type in the usage trace (brief, scene, fit, review, ...), and `tags` adds
        fields to its records (the scene number). `effort` overrides the client's for this call. Setting `cancel` stops
        the process (SIGINT first) and raises `cancelled`, also during a back-off. Transient failures and stalls at
        start are retried up to `retries` times after `backoff` s, doubling each time; a model family's weekly limit
        moves this call and later ones to the fallback model, even with no retries left (§4.8 re-runs the scene on the
        other model); anything else raises."""
        if schema is not None:
            check_schema(schema)
        effort = effort or self.effort
        info = self.probe()
        attempt = 0
        while True:
            attempt += 1
            if cancel is not None and cancel.is_set():
                raise ClaudeCLIError("cancelled", "The Claude call was cancelled")
            model, fallback = self._models(info)
            t0 = time.monotonic()
            run: _Run | None = None
            try:
                run = self._run(self._command(system, schema, model, fallback, effort), prompt, cancel)
                reply = parse_output(run.stdout, run.stderr, run.returncode, schema)
            except ClaudeCLIError as err:
                self._record(call, attempt, model, effort, time.monotonic() - t0, run, err.usage, err.model, err.rate_limit,
                             err, tags)
                delay = self._retry_after(err, model, fallback, attempt)
                if delay is None:
                    raise
                log.warning("claude %s call failed (%s: %s); retrying in %.0f s", call, err.kind, err, delay)
                if cancel is None:
                    self._sleep(delay)
                elif cancel.wait(delay):
                    raise ClaudeCLIError("cancelled", "The Claude call was cancelled") from err
                continue
            reply.seconds, reply.startup_s = time.monotonic() - t0, run.startup_s
            self._record(call, attempt, model, effort, reply.seconds, run, reply.usage, reply.model, reply.rate_limit, None,
                         tags)
            return reply

    def _retry_after(self, err: ClaudeCLIError, model: str, fallback: str | None, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None to give up. Never 0 (#91987: an immediate retry hangs too).
        A family limit is remembered whatever the attempt; moving to the other family isn't a retry of the same call, so
        it may take one attempt past the retry budget."""
        if err.kind == "usage_limit" and err.limit and err.limit == _family(model) and fallback \
                and _family(fallback) != err.limit:
            now = time.time()
            self._limited[err.limit] = err.resets_at if err.resets_at and err.resets_at > now else now + LIMIT_HOLD
            return self.backoff if attempt <= self.retries + 1 else None
        if err.kind not in RETRYABLE or attempt > self.retries:
            return None
        return self.backoff * 2 ** (attempt - 1)

    def _record(self, call: str, attempt: int, model: str, effort: str | None, seconds: float, run: _Run | None,
                usage: dict, resolved: str | None, rate: dict | None, err: ClaudeCLIError | None,
                tags: dict | None = None) -> None:
        startup = run.startup_s if run else None
        ev = {"event": "claude", "call": call, "requested": model, "model": resolved or model, "effort": effort,
              "attempt": attempt, "wall_s": round(seconds, 2), "startup_s": None if startup is None else round(startup, 2),
              "input_tokens": usage.get("input_tokens"), "cache_read_tokens": usage.get("cache_read_input_tokens"),
              "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
              "cache_1h_tokens": (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens"),
              "output_tokens": usage.get("output_tokens"), "rate_limit": rate, "error": err.kind if err else None,
              **(tags or {})}
        log.info("claude %s on %s: %.1f s (first event %s s), in %s + %s cached + %s written / out %s tokens%s", call,
                 ev["model"], seconds, ev["startup_s"], ev["input_tokens"], ev["cache_read_tokens"],
                 ev["cache_creation_tokens"], ev["output_tokens"], f", {err.kind}" if err else "")
        if self.trace is not None:
            try:
                self.trace(ev)
            except Exception:
                log.warning("could not record a claude call", exc_info=True)
