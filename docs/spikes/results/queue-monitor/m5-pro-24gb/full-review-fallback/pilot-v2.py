"""Original text only: selected-only vs same-call approved-full review, no audio."""
import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from maata_engine.backends.base import Brief, LineSpec, LineResult, SceneRequest, VideoMeta, Wording
from maata_engine.backends.claude_translator import ClaudeTranslator, line_json
from maata_engine.codex_cli import CodexCLI
from maata_engine.qa.coverage import approved_full, coverage_json
from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH, SHOTS_VERSION
from maata_engine.text.tenglish import latin_spoken


class RecordingCLI(CodexCLI):
    def __init__(self, *args, records, **kwargs):
        self.records = records
        super().__init__(*args, **kwargs)

    def ask(self, system, prompt, schema, call, **kwargs):
        record = {"call": call, "prompt": prompt, "started_utc": datetime.now(timezone.utc).isoformat()}
        self.records.append(record)
        try:
            reply = super().ask(system, prompt, schema, call, **kwargs)
            record["reply"] = asdict(reply)
            return reply
        except Exception as exc:
            record["error"] = {"kind": getattr(exc, "kind", None), "message": str(exc)}
            raise


def make_request(cases):
    specs, lines, chosen = [], {}, {}
    for i, c in enumerate(cases, 1):
        specs.append(LineSpec(i, "S1", c["source"], (i - 1) * 9., (i - 1) * 9. + 8., 8., 40.))
        tiers = {"full": Wording(c["full"], tuple(map(tuple, c.get("full_english", []))))}
        tiers[c["selected_tier"]] = Wording(c["selected"], tuple(map(tuple, c.get("selected_english", []))))
        lines[i] = LineResult(i, tiers)
        chosen[i] = c["selected_tier"]
    return SceneRequest(1, tuple(specs)), lines, chosen


async def run(dest, fixture):
    cases = json.loads(fixture.read_text())["cases"]
    positives = [c for c in cases if c["id"] in {"omitted_object_and_time",
                  "cache_and_duration_do_not_resurrect_omission", "mapped_conditional", "tight_mapped_both_condition"}]
    controls = [c for c in cases if c not in positives]
    order = list("ABBAABBAAB")  # five per policy; record every call/error, never discard a trial
    report = {"scope": "Original fixed-wording review component comparison, not whole-video throughput",
              "created_utc": datetime.now(timezone.utc).isoformat(), "order": order,
              "machine": {"os": platform.platform(), "python": platform.python_version(),
                          "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()},
              "prompt_version": SHOTS_VERSION, "prompt_hash": PROMPT_HASH, "review_hash": REVIEW_HASH,
              "limits": ["Concurrent production queue and uncontrolled provider load",
                         "Fixed existing wordings, same current prompt in both arms, no translation generation",
                         "Positive timing cohort excludes uncertainty sources; those move to conservative controls",
                         "Eligibility fixed for this text component; real planner gate is tested separately",
                         "Automated meaning review; no native human language or audio listening assessment",
                         "Baseline corrective outputs classified by validators, not a fresh semantic review"],
              "runs": []}
    dest.mkdir(parents=True, exist_ok=True)
    def save():
        (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for n, arm in enumerate(order + ["controls"], 1):
        selected = controls if arm == "controls" else positives
        req, lines, chosen = make_request(selected)
        offered = {i for i, c in enumerate(selected, 1) if arm != "A" and c["selected_tier"] != "full"
                   and c["id"] != "missing_fallback_verdict"}
        entry = {"trial": n, "arm": arm, "case_ids": [c["id"] for c in selected], "request": asdict(req),
                 "initial": [line_json(v) for v in lines.values()], "chosen": chosen,
                 "offered": sorted(offered), "records": [], "events": []}
        report["runs"].append(entry)
        with tempfile.TemporaryDirectory(prefix="maata-full-review-") as cache:
            brief = Brief(1, VideoMeta("Original meaning-review comparison", "Maata"), topic="Independent everyday examples")
            cli = RecordingCLI(Path(cache), records=entry["records"], trace=entry["events"].append)
            tr = ClaudeTranslator(Path(cache), "original", brief, cli=cli, trace=entry["events"].append)
            started = time.monotonic()
            try:
                result = await tr.review(req, lines, chosen, fallbacks=offered, fallback_check=lambda i: i in offered)
                entry.update(wall_s=time.monotonic() - started, calls=result.calls,
                             error=str(result.error) if result.error else None,
                             output=[{**line_json(v), "coverage": coverage_json(v.coverage),
                                      "approved_full": approved_full(v),
                                      "full_tts": latin_spoken(v.full.spoken, v.full.english)} for v in result.lines.values()])
            except Exception as exc:
                entry.update(wall_s=time.monotonic() - started, error=str(exc))
            finally:
                await tr.aclose()
        save()
        print(json.dumps({"trial": n, "arm": arm, "wall_s": round(entry["wall_s"], 3),
                          "calls": len(entry["records"]), "approved": sum(x["approved_full"] for x in entry.get("output", [])),
                          "error": entry.get("error")}), flush=True)
        if entry.get("error"):
            break
    report["summary"] = {arm: {"n": len(rows := [r for r in report["runs"] if r["arm"] == arm]),
        "median_wall_s": statistics.median(r["wall_s"] for r in rows),
        "call_counts": [len(r["records"]) for r in rows],
        "approved_counts": [sum(x["approved_full"] for x in r.get("output", [])) for r in rows]}
        for arm in ("A", "B") if any(r["arm"] == arm for r in report["runs"])}
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.out, args.fixture))
