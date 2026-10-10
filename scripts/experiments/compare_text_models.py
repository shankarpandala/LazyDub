#!/usr/bin/env python3
"""Bounded original-text screen: three generation calls and one blind semantic review.

Research only. No production imports this script; no audio, translation caches, repairs,
fallbacks or automatic adoption. Run from root with engine/.venv/bin/python and --run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from maata_engine.backends.base import Brief, GlossaryEntry, LineSpec, SceneRequest, SpeakerNote, VideoMeta
from maata_engine.backends.claude_translator import line_json
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.codex_cli import CodexCLI
from maata_engine.qa.coverage import check_review, coverage_json
from maata_engine.qa.validators import check_reply
from maata_engine.text.akshara import count_units
from maata_engine.text.scene_prompt import (PROMPT_HASH, REVIEW_HASH, REVIEW_SCHEMA, REVIEW_SYSTEM, SCENE_SCHEMA,
                                          SHOTS_VERSION, review_message, system_prompt, user_message)
from maata_engine.text.tenglish import latin_spoken

ARMS = (("luna", "gpt-6-luna", "low"), ("sol", "gpt-6.1-sol", "medium"),
        ("astra", "gpt-6-astra", "medium"))
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs/spikes/results/model-evaluation-2026-10-10/text"


def fixture():
    examples = [
        ("S1", "Meena, the bus tickets to Vijayawada cost two hundred and fifty rupees for both of us.", 8.0),
        ("S2", "You haven't paid yet, have you? Seriously?", 4.0),
        ("S1", "No, I only saved the booking page on my phone.", 5.0),
        ("S2", "Good, it may arrive by six, but the driver hasn't confirmed that.", 6.0),
        ("S1", "Let's not count our chickens before they hatch; we still need to check the date.", 7.5),
        ("S2", "Then send me the link on WhatsApp, and I'll check it after the meeting.", 7.0),
    ]
    lines, start = [], 0.0
    for i, (speaker, en, speech) in enumerate(examples, 1):
        lines.append(LineSpec(i, speaker, en, start, start + speech, speech, speech * 5,
                              want=("full",), to="S2" if speaker == "S1" else "S1"))
        start += speech + .3
    req = SceneRequest(1, tuple(lines), context_before=((
        "Ravi and Meena are close friends planning a bus trip to Vijayawada. They are comparing the fare before paying.",
        None),), context_after_en=("They agree to check the travel date and arrival time before buying the tickets.",))
    brief = Brief(1, VideoMeta("Original dialogue: checking a bus trip", "Maata original fixture"),
                  topic="Two close friends check the price, travel date and uncertain bus arrival before paying.",
                  register="Friendly everyday dialogue; Ravi and Meena address each other familiarly with నువ్వు.",
                  speakers=(SpeakerNote("S1", "Ravi", "male", "friend", "familiar", (("S2", "familiar"),)),
                            SpeakerNote("S2", "Meena", "female", "friend", "familiar", (("S1", "familiar"),))),
                  glossary=(GlossaryEntry("WhatsApp", "వాట్సాప్"),), entities=("Ravi", "Meena", "Vijayawada"),
                  idioms=("count our chickens before they hatch",), numbers="Use spoken Telugu numbers and rupees.")
    expectations = {
        1: "Meena is addressed; Vijayawada destination; 250 rupees total for both, not per person.",
        2: "Expressive question seeking confirmation that payment has not happened yet; preserve surprise.",
        3: "No payment; only saving the booking page on the speaker's own phone.",
        4: "Possible arrival by six, not guaranteed; driver has not confirmed it.",
        5: "Do not assume a good outcome prematurely; checking the date is still necessary; no literal chickens.",
        6: "Send the link to the speaker on WhatsApp; the speaker will check after the meeting, not before.",
    }
    return req, brief, expectations


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class EvidenceCLI(CodexCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = []

    def _run(self, *args, **kwargs):
        run = super()._run(*args, **kwargs)
        # Preserve returned model text and all errors, but omit ephemeral account/thread
        # identifiers and internal reasoning. Inputs and final outputs are original text.
        events = []
        for row in run.stdout.splitlines():
            try:
                event = json.loads(row)
            except json.JSONDecodeError:
                events.append({"unparsed": row})
                continue
            if event.get("type") == "thread.started":
                events.append({"type": "thread.started"})
            elif isinstance(event.get("item"), dict) and event["item"].get("type") == "reasoning":
                events.append({"type": event.get("type"), "item": {"type": "reasoning", "omitted": True}})
            else:
                events.append(event)
        self.raw.append({"returncode": run.returncode, "stdout_events": events, "stderr": run.stderr})
        return run


def ask(cache, model, effort, system, prompt, schema, call):
    events = []
    cli = EvidenceCLI(cache, model=model, effort=effort, retries=0, timeout=180, trace=events.append)
    started = time.monotonic()
    out = {"model_requested": model, "effort": effort, "calls_permitted": 1, "retries": 0}
    try:
        reply = cli.ask(system, prompt, schema, call)
        out.update(status="ok", model_returned=reply.model, data=reply.data, usage=reply.usage,
                   cli_seconds=reply.seconds, startup_s=reply.startup_s)
    except Exception as exc:
        out.update(status="error", error={"kind": getattr(exc, "kind", type(exc).__name__), "message": str(exc)})
    out.update(wall_s=round(time.monotonic() - started, 3), trace=events, raw=cli.raw)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Make at most four real signed-in Codex calls")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    dest = args.out.resolve()
    if (dest / "results.json").exists():
        p.error("Results already exist; choose another output directory to preserve evidence")
    req, brief, expectations = fixture()
    system, prompt = system_prompt(brief), user_message(req)
    inputs = {"authorship": "Original examples authored for this research screen; no downloaded transcript",
              "request": asdict(req), "brief": asdict(brief), "expectations_not_sent_to_models": expectations,
              "system": system, "prompt": prompt, "schema": SCENE_SCHEMA,
              "prompt_hash": PROMPT_HASH, "review_hash": REVIEW_HASH, "shots_version": SHOTS_VERSION,
              "arms": ARMS, "review_model": "gpt-6-astra", "review_effort": "high"}
    write(dest / "inputs.json", inputs)
    if not args.run:
        print(json.dumps({"status": "prepared", "fixture": str(dest / "inputs.json"), "calls": 0}))
        return
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "original text screen only",
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "platform": platform.platform(), "arms": [], "status": "running",
              "limitations": ["One generation per configuration; no throughput or significance claim",
                              "Only full wording requested/reviewed; no fit, planner or corrective ladder",
                              "Meaning reviewer does not judge style or pronunciation; no native-human or audio QA",
                              "One Astra reviewer can have correlated errors, including self-preference",
                              "Fixed arm order; provider cache/server load uncontrolled; other local research may run"]}
    reviewed = []
    with tempfile.TemporaryDirectory(prefix="maata-original-model-comparison-") as tmp:
        for label, model, effort in ARMS:
            print(json.dumps({"event": "start", "arm": label, "model": model, "effort": effort}), flush=True)
            result = ask(Path(tmp) / label, model, effort, system, prompt, SCENE_SCHEMA, "scene")
            result["arm"] = label
            if result["status"] == "ok":
                checked = check_reply(result["data"], req.lines, {g.term: g.spoken for g in brief.glossary})
                result.update(rejected=checked.rejected, unexpected=checked.unexpected,
                              validated=[{**line_json(line), "flags": list(line.flags),
                                          "tts_telugu": line.full.spoken,
                                          "tts_latin_proposal": latin_spoken(line.full.spoken, line.full.english),
                                          "units": count_units(line.full.spoken)}
                                         for _, line in sorted(checked.lines.items())])
                reviewed.extend((label, s, checked.lines[s.id].full) for s in req.lines if s.id in checked.lines)
            report["arms"].append(result)
            write(dest / "results.json", report)
            print(json.dumps({"event": "complete", "arm": label, "status": result["status"],
                              "wall_s": result["wall_s"], "validated": len(result.get("validated", []))}), flush=True)
        # Random opaque IDs and shuffled rows remove model IDs and arm order from the
        # one independent review. English scene context is identical for every candidate.
        rng = random.Random(20261010)
        rng.shuffle(reviewed)
        opaque_ids = rng.sample(range(1000, 10000), len(reviewed))
        rows, keys = [], {}
        for opaque, (label, spec, wording) in zip(opaque_ids, reviewed):
            rows.append((replace(spec, id=opaque), wording))
            keys[opaque] = {"arm": label, "line": spec.id, "tier": "full"}
        if rows:
            review_prompt = review_message(req, rows)
            write(dest / "blind-review-input.json", {"system": REVIEW_SYSTEM, "prompt": review_prompt,
                                                      "schema": REVIEW_SCHEMA})
            print(json.dumps({"event": "start", "call": "blind_review", "lines": len(rows)}), flush=True)
            review = ask(Path(tmp) / "review", "gpt-6-astra", "high", REVIEW_SYSTEM, review_prompt,
                         REVIEW_SCHEMA, "review")
            review["key_not_sent_to_model"] = keys
            if review["status"] == "ok":
                coverage, missing = check_review(review["data"], list(keys))
                review["missing_or_invalid"] = missing
                review["verdicts"] = [{**keys[i], "coverage": coverage_json(c)} for i, c in coverage.items()]
            report["review"] = review
        report["status"] = "complete"
        write(dest / "results.json", report)
    summary = {"scope": report["scope"], "limitations": report["limitations"],
               "generation": [{"arm": arm["arm"], "model": arm["model_requested"], "effort": arm["effort"],
                               "status": arm["status"], "wall_s": arm["wall_s"], "usage": arm.get("usage"),
                               "validated": len(arm.get("validated", [])), "rejected": arm.get("rejected"),
                               "meaning": dict(Counter(v["coverage"]["class"] for v in
                                   report.get("review", {}).get("verdicts", []) if v["arm"] == arm["arm"]))}
                              for arm in report["arms"]],
               "review_status": report.get("review", {}).get("status", "not_run"),
               "review_wall_s": report.get("review", {}).get("wall_s"),
               "actual_calls": sum(len(arm["trace"]) for arm in report["arms"]) +
                   len(report.get("review", {}).get("trace", [])), "production_changed": False}
    write(dest / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
