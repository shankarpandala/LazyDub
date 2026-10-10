#!/usr/bin/env python3
"""Smoke-check the production translation prompt on five original English lines, with no audio.

From the repository root:
    engine/.venv/bin/python scripts/check-translation.py --out PATH.json

Uses the normal sealed Codex CLI and production local validators/reviewer. This is a schema and automated-meaning
smoke check, not a native-speaker quality evaluation or an end-to-end speed benchmark. The isolated temporary cache
ensures every rerun makes fresh calls without changing the user's jobs or cached translations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from maata_engine.backends.base import Brief, GlossaryEntry, LineSpec, SceneRequest, SpeakerNote, VideoMeta
from maata_engine.backends.claude_translator import ClaudeTranslator, line_json
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.qa.coverage import coverage_json
from maata_engine.text.akshara import count_telugu
from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH, SHOTS_VERSION


def request() -> SceneRequest:
    # Original examples only; no downloaded transcript or personal data.
    examples = [
        ("Ravi paid 250 rupees for two tickets, but Meena did not buy one.", 8.0, 40.0),
        ("Open WhatsApp, turn off notifications, and restart the phone only if it still freezes.", 10.0, 50.0),
        ("The battery may last about six hours; it is not guaranteed to last all day.", 8.0, 40.0),
        ("Did you send the final PDF to Dr. Anjali before 9 AM, or only save it on your laptop?", 10.0, 50.0),
        ("I mean, the update is faster, but it does not fix the camera problem.", 7.5, 38.0),
    ]
    lines, start = [], 0.0
    for i, (english, speech, target) in enumerate(examples, 1):
        lines.append(LineSpec(i, "S1", english, start, start + speech, speech, target,
                              want=("full", "concise") if i == 5 else ("full",)))
        start += speech + 0.5
    return SceneRequest(1, tuple(lines), context_before=(("Here are five everyday examples.", None),),
                        context_after_en=("Those are all the examples.",))


def machine() -> dict:
    out = {"os": platform.platform(), "python": platform.python_version(), "architecture": platform.machine()}
    if platform.system() == "Darwin":
        out["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        out["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return out


def outputs(lines: dict) -> list[dict]:
    return [{**line_json(line), "model": line.model, "flags": list(line.flags),
             "aksharas": {tier: count_telugu(wording.spoken) for tier, wording in line.tiers.items()},
             "coverage": coverage_json(line.coverage)} for _, line in sorted(lines.items())]


async def check(dest: Path) -> bool:
    req = request()
    brief = Brief(1, VideoMeta("Original translation smoke examples", "Maata"), topic="Everyday examples",
                  register="Neutral and polite, addressing the audience with మీరు",
                  speakers=(SpeakerNote("S1", role="narrator", audience="polite"),),
                  glossary=(GlossaryEntry("WhatsApp", "వాట్సాప్"), GlossaryEntry("PDF", "పీడీఎఫ్")),
                  entities=("Ravi", "Meena", "Dr. Anjali"))
    events: list[dict] = []
    report = {"scope": "Original-text translation/schema/automated-meaning smoke; no audio or render-speed claim",
              "created_utc": datetime.now(timezone.utc).isoformat(), "machine": machine(),
              "prompt_version": SHOTS_VERSION, "prompt_hash": PROMPT_HASH, "review_hash": REVIEW_HASH,
              "request": asdict(req), "brief": asdict(brief), "human_review": "not performed",
              "events": events, "status": "running"}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="maata-translation-smoke-") as cache:
        tr = ClaudeTranslator(Path(cache), "original-smoke", brief, trace=events.append)
        report.update(requested_model=tr.model, effort=tr.effort, review_model=tr.review_cli.model,
                      review_effort=tr.review_effort)
        try:
            report["cli_health"] = await asyncio.to_thread(tr.cli.health)
            if report["cli_health"]["problem"]:
                raise ClaudeCLIError(report["cli_health"]["problem"], report["cli_health"]["message"])
            translated = await tr.translate(req)
            report.update(translation_seconds=translated.seconds, translation_calls=translated.calls,
                          skipped=translated.skipped, initial_output=outputs(translated.lines))
            reviewed = await tr.review(req, translated.lines, {i: "full" for i in translated.lines})
            report.update(review_seconds=reviewed.seconds, review_calls=reviewed.calls,
                          reviewed_output=outputs(reviewed.lines))
            if reviewed.error:
                raise reviewed.error
            report["status"] = "passed" if (len(reviewed.lines) == len(req.lines) and not translated.skipped
                                            and all(line.coverage is not None and line.coverage.cls == "C"
                                                    for line in reviewed.lines.values())) else "needs_review"
        except ClaudeCLIError as error:
            report.update(status="blocked", error={"kind": error.kind, "message": str(error)})
        finally:
            await tr.aclose()
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(dest.resolve()),
                      "wall_seconds": report["wall_seconds"]}))
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(check(args.out)) else 1)
