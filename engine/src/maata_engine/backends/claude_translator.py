"""Scene translation through the Claude CLI (docs/research/dubbing-2026-09/ARCHITECTURE.md §4; ADR-019).

One `ClaudeTranslator` per video. Its calls:
- brief v0: the video's metadata, placed as-is in the system prompt; no call (`brief_v0`);
- brief v1 (and v2, a diff of v1): one call on the transcript so far, cached by transcript hash; swapped in by
  `use_brief` at a scene boundary, when the scenes' glossary additions are folded in;
- scene: one call per scene for the lines the per-line cache lacks; ids that come back missing or unusable are asked
  for once more (with what was wrong), then one call per id, then marked skipped: never padded, never English (§4.9);
- fit, re-translate and rephrase: the same system prompt and schema, the call type in the message; a fit only adds
  tiers to the line it fits (or a wording of one closer to the slot), never a new `full`; a rephrase comes from the
  voicer through `submit`, is never awaited there, and is dropped when its deadline passes;
- review: the coverage review of a scene's chosen wordings (§4.6), on its own model (Sonnet 5 at high effort by
  default), its own system prompt and schema; each line it classes P or E gets one re-translation from the English with
  the missing words named, which the deterministic checks class (qa/coverage.py), and the better of the two is kept.

At most `concurrency` CLI processes run at once, the review's included, without any GPU lock. Cancelling a request (or
a review) drops it if it is still queued and stops its process (SIGINT first) if it is in flight. Every validated line
(a re-translation only once the review has judged it) goes into `<cache>/<video_id>/lines.jsonl` with its coverage class
once it has one, so a re-watch makes no call. Each CLI process leaves a usage record (tagged with its scene), each
request a "translate" record and each review a "review" record, through `trace`.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Coroutine

from ..claude_cli import DEFAULT_FALLBACK, DEFAULT_MODEL, ClaudeCLI, ClaudeCLIError, ClaudeReply
from ..qa.coverage import REDO, check_review, coverage_from, coverage_json, finding, rank, redo_class
from ..qa.validators import check_line, check_reply, glossary_entries
from ..text.akshara import count_telugu
from ..text.scene_prompt import (BRIEF_HASH, BRIEF_SCHEMA, BRIEF_SYSTEM, PROMPT_HASH, REVIEW_HASH, REVIEW_SCHEMA,
                                 REVIEW_SYSTEM, SCENE_SCHEMA, brief_message, review_message, system_prompt, user_message)
from .base import (CALLS, Brief, Coverage, GlossaryEntry, LineResult, LineSpec, SceneRequest, SceneResult, SpeakerNote,
                   VideoMeta)

log = logging.getLogger("maata.translate")

EFFORT, LIGHT_EFFORT = "medium", "low"  # brief, scene and re-translate calls; fit and rephrase calls (§4.2)
# The coverage review (§4.2, §4.6): Sonnet 5 at high effort. With Opus 5.5 translating, a Sonnet reviewer draws on the
# other family's weekly limit, and its misses are less likely to be the scene model's own.
REVIEW_MODEL, REVIEW_FALLBACK, REVIEW_EFFORT = "claude-sonnet-5", "claude-opus-5-5", "high"
CONCURRENCY = 3
# Failures of one reply, answered by asking again for fewer lines. Everything else (not signed in, a usage limit, the
# CLI missing or too old, a stall or throttle `ask` already retried) is the user's to fix, so it raises.
LADDER = frozenset({"bad_output", "timeout"})
LINES_FILE, BRIEFS_FILE = "lines.jsonl", "briefs.jsonl"


class _Late(Exception):
    """A rephrase whose deadline passed before it could start or before its answer came back."""


def brief_v0(meta: VideoMeta) -> Brief:
    """What scene 1 knows before brief v1 exists: the metadata alone."""
    return Brief(0, meta)


def line_key(spec: LineSpec, style: str) -> str:
    """A unit's key in the line cache: its English and speaker (§4.7), and the style, which one prompt serves both of."""
    return hashlib.sha256(json.dumps([spec.en, spec.speaker, style], ensure_ascii=False).encode()).hexdigest()[:20]


def line_json(line: LineResult) -> dict:
    """A validated line in the reply's own shape, as the cache stores it."""
    out: dict = {"id": line.id}
    for name, w in line.tiers.items():
        out[name] = {"spoken": w.spoken, "english": [{"i": i, "en": en} for i, en in w.english]}
    d = line.delivery
    out["delivery"] = {"emotion": d.emotion, "energy": d.energy, "question": d.question, "emphasis": list(d.emphasis)}
    if line.pieces:
        out["pieces"], out["moved"] = list(line.pieces), list(line.moved)
    if line.unfinished:
        out["unfinished"] = True
    return out


class _Jsonl:
    """An append-only JSONL file of rows keyed by "key" and filtered on load; the last row for a key wins."""

    def __init__(self, path: Path, **match: object) -> None:
        self.path, self.rows, self._torn = path, {}, False
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        self._torn = bool(text) and not text.endswith("\n")  # the next row must not land on the torn line
        for raw in text.split("\n"):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a torn last line after a crash
            if isinstance(row, dict) and isinstance(row.get("key"), str) and all(row.get(k) == v for k, v in match.items()):
                self.rows[row["key"]] = row

    def put(self, row: dict) -> None:
        self.rows[row["key"]] = row
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write("\n" * self._torn + json.dumps(row, ensure_ascii=False) + "\n")
            self._torn = False
        except OSError:
            log.warning("could not write %s", self.path.name, exc_info=True)


class ClaudeTranslator:
    def __init__(self, cache_dir: Path, video_id: str, brief: Brief, *, style: str = "colloquial",
                 cli: ClaudeCLI | None = None, model: str = DEFAULT_MODEL, fallback_model: str | None = DEFAULT_FALLBACK,
                 effort: str = EFFORT, light_effort: str = LIGHT_EFFORT, concurrency: int = CONCURRENCY,
                 review_cli: ClaudeCLI | None = None, review_model: str = REVIEW_MODEL,
                 review_fallback: str | None = REVIEW_FALLBACK, review_effort: str = REVIEW_EFFORT,
                 trace: Callable[[dict], None] | None = None, binary: str | None = None) -> None:
        self.cli = cli or ClaudeCLI(cache_dir, model=model, fallback_model=fallback_model, effort=effort, binary=binary,
                                    trace=trace)
        # A client given for the scene calls (the mock, a test's) answers the reviews too unless one is given for them.
        self.review_cli = review_cli or cli or ClaudeCLI(cache_dir, model=review_model, fallback_model=review_fallback,
                                                         effort=review_effort, binary=binary, trace=trace)
        self.model: str = self.cli.model
        self.prompt_hash = PROMPT_HASH
        self.style = "formal" if style == "formal" else "colloquial"
        self.effort, self.light_effort, self.review_effort = effort, light_effort, review_effort
        self.trace = trace
        self.brief = brief
        self._system = system_prompt(brief)
        self.glossary_recent: dict[str, GlossaryEntry] = {}  # scene additions since the last brief swap, by term
        self._additions: dict[str, GlossaryEntry] = {}       # every scene addition, folded into each brief swapped in
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[asyncio.Task, SceneRequest] = {}
        video_dir = cache_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        self._lines = _Jsonl(video_dir / LINES_FILE, prompt_hash=PROMPT_HASH, model=self.model)
        self._briefs = _Jsonl(video_dir / BRIEFS_FILE, prompt_hash=BRIEF_HASH, model=self.model)

    # ---- the brief -----------------------------------------------------------------------------------------------
    def use_brief(self, brief: Brief) -> None:
        """Swap `brief` in for the requests that start from now on (requests under way keep theirs, retries included,
        so a scene never mixes briefs). Every glossary addition
        the scenes have made joins it, except terms it already has: conflicts go to the brief (§4.6). All of them, not
        only those since the last swap, since a v2 made from v1 as `make_brief` returned it lacks the ones folded in."""
        have = {g.term.lower() for g in brief.glossary}
        extra = tuple(g for k, g in self._additions.items() if k not in have)
        self.brief = replace(brief, glossary=brief.glossary + extra)
        self.glossary_recent.clear()
        self._system = system_prompt(self.brief)

    async def make_brief(self, meta: VideoMeta, transcript: Sequence[tuple[str, str]],
                         previous: Brief | None = None) -> Brief:
        """Brief v1 from the (speaker, English) transcript so far, or with `previous` (v1) the next version, from its
        diff. The reply is cached by the call's whole message, so an unchanged transcript costs nothing."""
        t0 = time.monotonic()
        msg = brief_message(meta, transcript, previous)
        key = hashlib.sha256(msg.encode()).hexdigest()[:20]
        row = self._briefs.rows.get(key)
        hit = row is not None
        if row is None:
            reply = await self._ask("brief", BRIEF_SYSTEM, msg, BRIEF_SCHEMA, self.effort)
            row = {"key": key, "prompt_hash": BRIEF_HASH, "model": self.model, "answered_by": reply.model,
                   "data": reply.data, "at": round(time.time(), 3)}
            self._briefs.put(row)
        brief = parse_brief(meta, row["data"], previous if previous is not None and previous.version >= 1 else None)
        self._emit({"event": "brief", "version": brief.version, "cache_hit": hit, "glossary": len(brief.glossary),
                    "wall_s": round(time.monotonic() - t0, 2)})
        return brief

    # ---- requests ------------------------------------------------------------------------------------------------
    async def translate(self, req: SceneRequest) -> SceneResult:
        if req.call not in CALLS:
            raise ValueError(f"unknown call {req.call!r}")
        t0 = time.monotonic()
        result = SceneResult(req.scene, req.call)
        pin = (self._system, self.brief.version)  # every call of this request uses the brief it started under
        try:
            if req.call == "scene":
                await self._scene(req, result, pin)
            else:  # a fit adds tiers to the line the session holds, even one made under brief v0; the others replace it
                base = ({s.id: line for s in req.lines if (line := self._cached(s, any_brief=True))}
                        if req.call == "fit" else None)
                await self._ladder(req, list(req.lines), [], result, pin, base)
        except _Late:
            result.dropped, result.lines, result.skipped = True, {}, {}
        except ClaudeCLIError as err:
            result.seconds = time.monotonic() - t0
            self._emit(self._event(req, result, err.kind))
            raise
        result.seconds = time.monotonic() - t0
        self._emit(self._event(req, result))
        return result

    def submit(self, req: SceneRequest) -> asyncio.Task[SceneResult]:
        """`translate` as a task the caller need not await; `cancel` reaches it."""
        return self._track(self.translate(req), req)

    def _track(self, coro: Coroutine[Any, Any, SceneResult], req: SceneRequest) -> asyncio.Task[SceneResult]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks[task] = req
        task.add_done_callback(lambda t: self._tasks.pop(t, None))
        return task

    def cancel(self, match: Callable[[SceneRequest], bool] | None = None) -> int:
        """Cancel the submitted requests `match` picks (all by default): queued ones never start, and in-flight ones
        get SIGINT (§3.12). Lines already validated stay in the line cache. Returns how many were cancelled."""
        hit = [t for t, r in self._tasks.items() if not t.done() and (match is None or match(r))]
        for t in hit:
            t.cancel()
        return len(hit)

    async def aclose(self) -> None:
        """Cancel every submitted request and return once each has ended, its CLI process included."""
        tasks = list(self._tasks)
        self.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- the coverage review (§4.6) ------------------------------------------------------------------------------
    async def review(self, req: SceneRequest, lines: Mapping[int, LineResult], chosen: Mapping[int, str]) -> SceneResult:
        """The coverage review of scene `req`'s `lines`, each on the tier `chosen` for it, as a task `cancel` reaches as
        it reaches `req`. A line that has a class already (from the line cache) keeps it; the others go to one review
        call, and the ids its reply lacks to one more. A line classed P or E gets one re-translation from its English
        with the missing words named (an E line with what was wrong); the deterministic checks class that one, and it
        replaces the line only with a better class. Every class is stored with its line in the line cache (a
        re-translation only here, once judged).
        The result holds the lines to use, each with its class (None where the review had no usable answer). A failure
        the user must fix leaves the lines it reached unreviewed and comes back in `error`: a P or E line whose
        re-translation never came back too. Nothing is stored for them, so a later showing reviews them."""
        return await self._track(self._review(req, lines, chosen), req)

    async def _review(self, req: SceneRequest, lines: Mapping[int, LineResult],
                      chosen: Mapping[int, str]) -> SceneResult:
        t0 = time.monotonic()
        result = SceneResult(req.scene, "review", lines=dict(lines))
        specs = {s.id: s for s in req.lines if s.id in lines}
        tiers = {i: t if (t := chosen.get(i)) in lines[i].tiers else "full" for i in specs}
        said = {i: lines[i].tiers[t] for i, t in tiers.items()}  # the wording each line would be voiced with
        todo = [s for s in specs.values() if lines[s.id].coverage is None]
        verdicts: dict[int, Coverage] = {}
        why: dict[int, str] = {}
        again: list[LineSpec] = []
        redo = SceneResult(req.scene, "retranslate")
        try:
            for _ in range(2):
                pending = [s for s in todo if s.id not in verdicts]
                if not pending:
                    break
                result.calls += 1
                msg = review_message(req, [(s, said[s.id]) for s in pending])
                try:
                    reply = await self._ask("review", REVIEW_SYSTEM, msg, REVIEW_SCHEMA, self.review_effort,
                                            tags={"scene": req.scene, "lines": len(pending)}, cli=self.review_cli)
                except ClaudeCLIError as err:
                    if err.kind not in LADDER:
                        raise
                    why = {s.id: f"{err.kind}: {err}" for s in pending}
                    continue
                got, why = check_review(reply.data, [s.id for s in pending])
                verdicts.update({i: replace(c, tier=tiers[i]) for i, c in got.items()})
            again = [replace(specs[i], missing=c.missing, problems=finding(c)) for i, c in verdicts.items()
                     if c.cls in REDO]
            if again:  # the rest of the scene, as said, is their context
                done = [(s, said[s.id].spoken) for s in specs.values() if s.id not in {x.id for x in again}]
                await self._ladder(replace(req, call="retranslate", lines=tuple(again)), again, done, redo,
                                   (self._system, self.brief.version))
        except ClaudeCLIError as err:
            result.error = err
        replaced = []
        for i, c in verdicts.items():
            new = redo.lines.get(i)
            if c.cls in REDO and new is None and result.error is not None:
                continue  # its re-translation never came back: it is reviewed again next time
            line = replace(lines[i], coverage=c)
            if new is not None:  # like with like: the new wording on the tier the review classed, else both `full`s
                t = c.tier if c.tier in new.tiers else "full"
                ours = redo_class(specs[i].en, c, lines[i].tiers[t], new.tiers[t], t)
                if rank(ours) < rank(c):
                    line = replace(new, coverage=ours)
                    replaced.append(i)
            result.lines[i] = line
            self._store(specs[i], line)
        unreviewed = sorted(s.id for s in todo if result.lines[s.id].coverage is None)
        for i in unreviewed:
            log.info("scene %s: line %s unreviewed: %s", req.scene, i, why.get(i, getattr(result.error, "kind", "")))
        result.calls += redo.calls
        result.seconds = time.monotonic() - t0
        self._emit({"event": "review", "scene": req.scene, "lines": len(specs), "stored": len(specs) - len(todo),
                    "classes": {k: sum(1 for c in verdicts.values() if c.cls == k) for k in ("C", "m", "P", "E")},
                    "retranslated": sorted(x.id for x in again), "replaced": sorted(replaced),
                    "unreviewed": unreviewed, "calls": result.calls, "wall_s": round(result.seconds, 2),
                    "model": self.review_cli.model, "review_hash": REVIEW_HASH,
                    "error": getattr(result.error, "kind", None)})
        return result

    # ---- scene calls ---------------------------------------------------------------------------------------------
    async def _scene(self, req: SceneRequest, result: SceneResult, pin: tuple[str, int]) -> None:
        """Cached lines are served and passed as context only; lines the cache lacks go to one scene call; cached lines
        without a tier asked for now go to one fit call (§4.7). A scene with every line cached makes no call."""
        todo: list[LineSpec] = []
        partial: dict[int, LineResult] = {}
        for s in req.lines:
            line = self._cached(s)
            if line is None:
                todo.append(s)
            elif all(t in line.tiers for t in s.want):
                result.lines[s.id] = line
            else:
                partial[s.id] = line
        done = [(s, (result.lines.get(s.id) or partial[s.id]).full.spoken) for s in req.lines
                if s.id in result.lines or s.id in partial]
        fits = [replace(s, want=tuple(t for t in s.want if t not in partial[s.id].tiers),
                        current=partial[s.id].full.spoken,
                        overflow=count_telugu(partial[s.id].full.spoken) - s.target_aksharas)
                for s in req.lines if s.id in partial]
        jobs = []
        if todo:
            jobs.append(self._ladder(req, todo, done, result, pin))
        if fits:
            fit = replace(req, call="fit", lines=tuple(fits), context_before=(), context_after_en=())
            jobs.append(self._ladder(fit, fits, [], result, pin, base=partial))
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)
        for o in outcomes:
            if isinstance(o, BaseException):
                raise o

    async def _ladder(self, req: SceneRequest, specs: list[LineSpec], done: list[tuple[LineSpec, str]],
                      result: SceneResult, pin: tuple[str, int], base: Mapping[int, LineResult] | None = None) -> None:
        """One call for `specs`; the ids it didn't answer usably once more, each with what was wrong (after the
        problems it came with: a re-translation's review finding); then one call per id; then skipped (§4.9). Lines
        answered on the way become context for the rest. In a fit, a line with a `base` is never skipped: it keeps the
        tiers it had, flagged."""
        pending, own = specs, {s.id: s.problems for s in specs}
        for rung in ("all", "again", "each"):
            if not pending:
                return
            groups = [[s] for s in pending] if rung == "each" else [pending]
            outcomes = await asyncio.gather(*(self._once(req, g, done, result, pin, base) for g in groups),
                                            return_exceptions=True)
            rejected: dict[int, list[str]] = {}
            for o in outcomes:
                if isinstance(o, BaseException):
                    raise o
                rejected.update(o)
            done = done + [(s, result.lines[s.id].full.spoken) for s in pending if s.id in result.lines]
            pending = [replace(s, problems=tuple(dict.fromkeys(own[s.id] + tuple(rejected[s.id]))))
                       for s in pending if s.id in rejected]
        for s in pending:
            why = "; ".join(s.problems) or "no usable answer"
            if base is not None and s.id in base:  # a cached line whose fit failed is still a good line
                result.lines[s.id] = replace(base[s.id], flags=base[s.id].flags + ("fit failed",))
                log.info("scene %s: line %s keeps its cached tiers; the fit failed: %s", req.scene, s.id, why)
            else:
                result.skipped[s.id] = why
                log.warning("scene %s: line %s got no usable %s answer: %s", req.scene, s.id, req.call, why)

    async def _once(self, req: SceneRequest, specs: list[LineSpec], done: list[tuple[LineSpec, str]],
                    result: SceneResult, pin: tuple[str, int],
                    base: Mapping[int, LineResult] | None) -> dict[int, list[str]]:
        """One call for `specs`: validated lines go into `result` and the line cache; returns the rejected ids. In a fit
        (`base` given) an answer is folded into the line it fits; with no line to fold into, it is returned uncached.
        `pin` is the system prompt and brief version of the request, so a swap never lands inside one."""
        sub = replace(req, lines=tuple(specs))
        msg = user_message(sub, self.style, list(self.glossary_recent.values()), done)
        effort = self.light_effort if req.call in ("fit", "rephrase") else self.effort
        glossary = self._glossary()
        system, version = pin
        result.calls += 1
        try:
            reply = await self._ask(req.call, system, msg, SCENE_SCHEMA, effort, req.deadline,
                                    {"scene": req.scene, "lines": len(specs)})
        except ClaudeCLIError as err:
            if err.kind not in LADDER:
                raise
            return {s.id: [f"{err.kind}: {err}"] for s in specs}
        if req.deadline is not None and time.time() > req.deadline:
            raise _Late
        checked = check_reply(reply.data, specs, glossary)
        if checked.unexpected:
            log.warning("scene %s: the reply also had ids nobody asked for: %s", req.scene, checked.unexpected)
        by_id = {s.id: s for s in specs}
        for lid, line in checked.lines.items():
            old = base.get(lid) if base is not None else None
            if old is not None:
                line = self._merged(old, line, by_id[lid], glossary)
            line.model = reply.model or self.model
            line.brief_version = version if old is None else old.brief_version  # the brief its `full` was made under
            result.lines[lid] = line
            if base is not None and old is None:
                continue  # a fit's `full` is only the wording it was given (§4.3), never a line to cache
            if req.call == "retranslate":
                continue  # the review keeps this or the reviewed wording, and stores the one it keeps with its class
            self._store(by_id[lid], line)
        for g in checked.glossary_additions:
            if g.term.lower() not in {b.term.lower() for b in self.brief.glossary}:  # conflicts go to the brief
                self.glossary_recent.setdefault(g.term.lower(), g)
                self._additions.setdefault(g.term.lower(), g)
                result.glossary_additions.append(g)
        return checked.rejected

    def _merged(self, old: LineResult, new: LineResult, spec: LineSpec, glossary: Mapping[str, str]) -> LineResult:
        """A fit's answer folded into the line it fits: only the tiers `want` names are taken from it, each only where
        the akshara order still holds with every tier the line had (a wording asked for again replaces its own only
        then). The line's `full`, english map, pieces and delivery are kept: a fit never rewrites the natural line
        (§4.2), and the emphasis and pieces belong to that `full`. So is its coverage class, unless the fit replaced the
        very wording the review classed: the line is then unreviewed, and reviewed the next time it is shown."""
        raw, fitted = line_json(old), line_json(new)
        left_out, taken = [], set()
        for k in (k for k in spec.want if k != "full" and k in fitted):
            got, _ = check_line({**raw, k: fitted[k]}, spec, glossary)
            if got is not None and k in got.tiers and set(old.tiers) <= set(got.tiers):
                raw[k] = fitted[k]
                taken.add(k)
            else:
                left_out.append(f"fit: {k} left out")
        line, _ = check_line(raw, spec, glossary)
        if line is None:
            return replace(old)
        line.flags += tuple(left_out)
        c = old.coverage
        kept = c is None or c.tier not in taken or old.tiers.get(c.tier) == line.tiers.get(c.tier)
        line.coverage = c if kept else None
        return line

    def _store(self, spec: LineSpec, line: LineResult) -> None:
        """A validated line into the line cache, with its coverage class when it has one."""
        self._lines.put({"key": line_key(spec, self.style), "prompt_hash": PROMPT_HASH, "model": self.model,
                         "answered_by": line.model, "brief": line.brief_version,
                         "coverage": coverage_json(line.coverage), "line": line_json(line), "at": round(time.time(), 3)})

    def _cached(self, spec: LineSpec, any_brief: bool = False) -> LineResult | None:
        """The cached line for `spec`, if it still validates. A line made under brief v0 (the metadata alone: no
        speaker genders, address forms or glossary) is made again once a brief is in use; v1's serve v2 (§4.7)."""
        row = self._lines.rows.get(line_key(spec, self.style))
        if row is None or (not any_brief and row.get("brief") == 0 and self.brief.version >= 1):
            return None
        line, why = check_line(row["line"], replace(spec, want=()), self._glossary())
        if line is None:  # stored under looser checks than today's
            log.info("line cache entry for %s no longer validates: %s", spec.id, why)
            return None
        line.cached, line.model, line.brief_version, line.coverage = (True, row.get("answered_by"), row.get("brief"),
                                                                     coverage_from(row.get("coverage")))
        return line

    def _glossary(self) -> dict[str, str]:
        out = {g.term: g.spoken for g in self.glossary_recent.values()}
        out.update({g.term: g.spoken for g in self.brief.glossary})
        return out

    # ---- plumbing ------------------------------------------------------------------------------------------------
    async def _ask(self, call: str, system: str, prompt: str, schema: dict, effort: str, deadline: float | None = None,
                   tags: dict | None = None, cli: ClaudeCLI | None = None) -> ClaudeReply:
        """One CLI call (on `cli`, by default the scene model's) in a worker thread, under the concurrency semaphore. If
        the awaiting task is cancelled, the process gets SIGINT and the slot frees once it has exited, however many
        more cancels arrive meanwhile (a seek's cancel, then the session's close)."""
        cancel = threading.Event()
        async with self._sem:
            if deadline is not None and time.time() > deadline:
                raise _Late
            fut = asyncio.get_running_loop().run_in_executor(None, functools.partial(
                (cli or self.cli).ask, system, prompt, schema, call, effort=effort, cancel=cancel, tags=tags))
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError:
                cancel.set()
                while not fut.done():
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(fut)
                if not fut.cancelled():
                    fut.exception()  # retrieved: the call's own "cancelled" error isn't news
                raise

    def _event(self, req: SceneRequest, result: SceneResult, error: str | None = None) -> dict:
        return {"event": "translate", "call": req.call, "scene": req.scene, "lines": len(req.lines),
                "cached": sum(1 for x in result.lines.values() if x.cached),
                "answered": sum(1 for x in result.lines.values() if not x.cached), "skipped": sorted(result.skipped),
                "calls": result.calls, "wall_s": round(result.seconds, 2), "dropped": result.dropped,
                "model": self.model, "prompt_hash": self.prompt_hash, "brief": self.brief.version, "error": error}

    def _emit(self, ev: dict) -> None:
        if self.trace is not None:
            try:
                self.trace(ev)
            except Exception:
                log.warning("could not record a translate event", exc_info=True)


# ---- briefs ----------------------------------------------------------------------------------------------------------
_GENDERS, _FORMS = ("male", "female"), ("polite", "familiar")  # "unknown" is the default, never news


def _strings(raw: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(s.strip() for s in raw if isinstance(s, str) and s.strip())) if isinstance(raw, list) else ()


def _speaker(raw: dict, known: SpeakerNote) -> SpeakerNote:
    """`known` updated by what `raw` gives: only fields present, non-empty and valid, and `address` per speaker
    addressed. An "unknown" gender never erases a known one."""
    address = dict(known.address) | {a["to"]: a["form"] for a in raw.get("address") or []
                                     if isinstance(a, dict) and isinstance(a.get("to"), str) and a.get("form") in _FORMS}
    return SpeakerNote(known.id, str(raw.get("name") or "") or known.name,
                       raw["gender"] if raw.get("gender") in _GENDERS else known.gender,
                       str(raw.get("role") or "") or known.role,
                       raw["audience"] if raw.get("audience") in _FORMS else known.audience, tuple(address.items()))


def parse_brief(meta: VideoMeta, data: object, previous: Brief | None = None) -> Brief:
    """A brief from the brief call's reply. With `previous`, the reply is a diff: what it names replaces or joins the
    previous brief's, and what it leaves out (or empty) stays, down to each field of a speaker it names."""
    d = data if isinstance(data, dict) else {}
    before = {s.id: s for s in previous.speakers} if previous is not None else {}
    speakers: dict[str, SpeakerNote] = {}
    for s in d.get("speakers") or []:
        if isinstance(s, dict) and isinstance(s.get("id"), str):
            speakers[s["id"]] = _speaker(s, speakers.get(s["id"]) or before.get(s["id"]) or SpeakerNote(s["id"]))
    fixes = tuple((f["heard"], f["meant"]) for f in d.get("asr_fixes") or []
                  if isinstance(f, dict) and isinstance(f.get("heard"), str) and isinstance(f.get("meant"), str))
    brief = Brief(1, meta, str(d.get("topic") or ""), str(d.get("register") or ""), tuple(speakers.values()),
                  tuple(glossary_entries(d.get("glossary"))), _strings(d.get("entities")), _strings(d.get("idioms")),
                  str(d.get("numbers") or ""), fixes)
    if previous is None:
        return brief
    by_id = before | speakers
    terms = {g.term.lower(): g for g in previous.glossary} | {g.term.lower(): g for g in brief.glossary}
    heard = dict(previous.asr_fixes) | dict(brief.asr_fixes)
    return Brief(previous.version + 1, meta, brief.topic or previous.topic, brief.register or previous.register,
                 tuple(by_id.values()), tuple(terms.values()), tuple(dict.fromkeys(previous.entities + brief.entities)),
                 tuple(dict.fromkeys(previous.idioms + brief.idioms)), brief.numbers or previous.numbers,
                 tuple(heard.items()))
