#!/usr/bin/env python3
"""One bounded production correction smoke on authored text; no audio or user jobs.

From the repository root:
    engine/.venv/bin/python scripts/experiments/check_semantic_correction.py --out-dir PATH

The production translator/reviewer and sealed signed-in Codex transport are unchanged.
Transport retries are disabled solely to make the six-call experiment budget exact;
the translator's own bounded malformed-output ladder is retained. There is one run,
with a 180-second operation deadline plus subprocess cancellation cleanup.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from maata_engine.backends.base import Brief, LineResult, LineSpec, SceneRequest, SpeakerNote, VideoMeta, Wording
from maata_engine.backends.claude_translator import ClaudeTranslator, line_json
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.codex_cli import CodexCLI, DEFAULT_EFFORT, DEFAULT_MODEL
from maata_engine.qa.coverage import APPROVAL_POLICY, coverage_json
from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH, SHOTS_VERSION

MAX_CALLS = 6
TIMEOUT_SECONDS = 180


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixture() -> tuple[SceneRequest, Brief, dict[int, LineResult]]:
    # Authored specifically for this smoke; never copied from a user's video.
    rows = (
        ("Ravi will return the red book tomorrow.", "రవి ఎర్ర పుస్తకాన్ని తిరిగి ఇస్తాడు.", 7.0, 40.0),
        ("The battery may last six hours, but that is not guaranteed.", "ఈ బ్యాటరీ ఆరు గంటలు పనిచేస్తుంది.", 9.0, 50.0),
    )
    specs, originals, start = [], {}, 0.0
    for i, (en, te, speech, target) in enumerate(rows, 1):
        specs.append(LineSpec(i, "S1", en, start, start + speech, speech, target, want=("full",)))
        originals[i] = LineResult(i, {"full": Wording(te)})
        start += speech + 0.5
    req = SceneRequest(1, tuple(specs))
    brief = Brief(1, VideoMeta("Two original semantic correction controls", "Maata experiments"),
                  topic="Two independent everyday examples",
                  register="Natural everyday Telugu, neutral and polite",
                  speakers=(SpeakerNote("S1", role="narrator", audience="polite"),), entities=("Ravi",))
    return req, brief, originals


def outputs(lines: dict[int, LineResult]) -> list[dict]:
    return [{**line_json(line), "coverage": coverage_json(line.coverage), "model": line.model,
             "flags": list(line.flags)} for _, line in sorted(lines.items())]


class RecordedCLI:
    """Observe the normal transport without replacing its prompt, parser or semantics."""

    def __init__(self, cli: CodexCLI, dest: Path):
        self.cli, self.dest = cli, dest
        self.model, self.effort, self.fallback_model = cli.model, cli.effort, cli.fallback_model
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def ask(self, system, prompt, schema=None, call="text", *, effort=None, cancel=None, tags=None):
        with self.lock:
            if len(self.calls) >= MAX_CALLS:
                raise ClaudeCLIError("failed", f"Original-text experiment reached its {MAX_CALLS}-call limit.")
            record = {"number": len(self.calls) + 1, "call": call, "tags": tags,
                      "model": self.model, "effort": effort if effort is not None else self.effort,
                      "system": system, "prompt": prompt, "schema": schema, "status": "started"}
            self.calls.append(record)
            path = self.dest / f"call-{record['number']:02d}.json"
            write_json(path, record)
        started = time.monotonic()
        try:
            reply = self.cli.ask(system, prompt, schema, call, effort=effort, cancel=cancel, tags=tags)
            record.update(status="completed", reply=asdict(reply))
            return reply
        except Exception as exc:
            record.update(status="failed", error={"kind": getattr(exc, "kind", type(exc).__name__),
                                                  "message": str(exc), "usage": getattr(exc, "usage", {})})
            raise
        finally:
            record["wall_seconds"] = round(time.monotonic() - started, 3)
            write_json(path, record)


def provenance() -> dict:
    repo = Path(__file__).resolve().parents[2]
    paths = ["engine/src/maata_engine/backends/claude_translator.py",
             "engine/src/maata_engine/backends/base.py", "engine/src/maata_engine/qa/coverage.py",
             "engine/src/maata_engine/text/scene_prompt.py", "engine/src/maata_engine/codex_cli.py",
             "scripts/experiments/check_semantic_correction.py"]
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "source_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths},
            "platform": platform.platform(), "architecture": platform.machine(),
            "python": platform.python_version(), "prompt_version": SHOTS_VERSION,
            "prompt_hash": PROMPT_HASH, "review_hash": REVIEW_HASH, "approval_policy_base": APPROVAL_POLICY}


async def check(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=False)
    req, brief, originals = fixture()
    write_json(dest / "inputs.json", {"authored_originals": True, "request": asdict(req), "brief": asdict(brief),
                                     "initial_wordings": outputs(originals),
                                     "expected_defects": {"1": "Missing tomorrow",
                                                          "2": "Possibility and lack of guarantee become certainty"}})
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Single original-text production correction-path smoke; no audio or user jobs",
              "limits": {"model_calls": MAX_CALLS, "operation_timeout_seconds": TIMEOUT_SECONDS,
                         "transport_retries": 0, "outer_repeats": 0},
              "provenance": provenance(), "human_native_speaker_review": "not performed",
              "limitations": ["Automated semantic review is not ground truth.",
                              "Two controls do not establish general Telugu translation quality.",
                              "No speech synthesis, pronunciation, voice or audio quality was tested."]}
    write_json(dest / "report.json", report)
    events: list[dict] = []
    event_lock = threading.Lock()
    started = time.monotonic()

    def trace(event):
        with event_lock:
            ev = {"observed_seconds": round(time.monotonic() - started, 3), **event}
            events.append(ev)
            with (dest / "trace.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")

    with tempfile.TemporaryDirectory(prefix="maata-original-correction-") as temp:
        cache = Path(temp)
        cli = CodexCLI(cache, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, retries=0,
                       timeout=TIMEOUT_SECONDS, trace=trace)
        recorded = RecordedCLI(cli, dest)
        tr = ClaudeTranslator(cache, "original-correction-smoke", brief, cli=recorded, trace=trace)
        report.update(model=tr.model, effort=tr.effort, review_model=tr.review_cli.model,
                      review_effort=tr.review_effort, approval_policy=tr.approval_policy,
                      isolated_cache=True, production_cache_touched=False)
        try:
            result = await asyncio.wait_for(tr.review(req, originals, {1: "full", 2: "full"}), TIMEOUT_SECONDS)
            report.update(result={"calls": result.calls, "seconds": result.seconds,
                                  "review_pending": result.review_pending, "output": outputs(result.lines),
                                  "error": ({"kind": getattr(result.error, "kind", type(result.error).__name__),
                                             "message": str(result.error)} if result.error else None)})
            primary = [c for c in recorded.calls if c["call"] == "review" and not (c["tags"] or {}).get("correction")]
            verdicts = {row["id"]: row for c in primary if c["status"] == "completed"
                        for row in (c["reply"]["data"] or {}).get("lines", [])}
            correction = [c for c in recorded.calls if (c["tags"] or {}).get("correction")]
            detected = all(verdicts.get(i, {}).get("class") in ("P", "E") for i in originals)
            approved = all(i in result.lines and result.lines[i].coverage is not None
                           and result.lines[i].coverage.cls == "C" and result.lines[i].coverage.by == "review"
                           and result.lines[i].coverage.first in ("P", "E")
                           and result.lines[i].full.spoken != originals[i].full.spoken for i in originals)
            report["checks"] = {"both_primary_defects_detected": detected,
                                "generated_correction_call": any(c["call"] == "retranslate" for c in recorded.calls),
                                "one_fresh_batched_correction_review": len(correction) == 1,
                                "both_changed_candidates_explicitly_C": approved,
                                "no_pending_review": not result.review_pending, "no_error": result.error is None}
            report["status"] = "passed" if all(report["checks"].values()) else "needs_review"
        except Exception as exc:
            report.update(status="timeout" if isinstance(exc, TimeoutError) else "failed",
                          error={"kind": getattr(exc, "kind", type(exc).__name__), "message": str(exc)})
        finally:
            await tr.aclose()
            rows = {}
            for path in sorted(cache.rglob("*.jsonl")):
                # All cache content belongs to the two authored lines; never copy CLI state or credentials.
                rows[str(path.relative_to(cache))] = [json.loads(row) for row in path.read_text().splitlines() if row]
            write_json(dest / "cache-rows.json", rows)
            report.update(model_calls=len(recorded.calls), wall_seconds=round(time.monotonic() - started, 3),
                          cli_version=cli._info.text if cli._info is not None else None,
                          cache_files=list(rows), trace_events=len(events))
            write_json(dest / "report.json", report)
    print(json.dumps({"status": report["status"], "calls": report["model_calls"],
                      "wall_seconds": report["wall_seconds"], "output": str(dest.resolve())}))
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(check(args.out_dir)) else 1)
