#!/usr/bin/env python3
"""Read live Maata timing/quality aggregates without modifying its queue or calling a model."""

import argparse
import json
from pathlib import Path

from maata_engine.diagnostics import snapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, default=Path.home() / "Library/Caches/Maata")
parser.add_argument("--window", type=float, default=900, help="Seconds of legacy traces without run boundaries")
parser.add_argument("--previous", type=Path, help="An earlier snapshot for progress rates")
parser.add_argument("--out", type=Path, help="Save the complete transcript-free JSON report")
args = parser.parse_args()
if args.window <= 0:
    parser.error("--window must be positive")
previous = json.loads(args.previous.read_text()) if args.previous and args.previous.exists() else None
report = snapshot(args.cache, window_s=args.window, previous=previous)
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = args.out.with_suffix(args.out.suffix + ".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(args.out)
for job in report["jobs"]:
    if job["status"] == "done":
        continue
    print(f"{job['video_id']} {job['status']} · {job['duration_s'] or 0:.0f}s media · {job['stage'] or 'not started'}")
    print(f"  Scope: {job['scope']}; {job['observed_span_s']:.0f}s observed")
    for stage, p in job["progress"].items():
        if p["state"] == "running":
            print(f"  {stage}: {p['done']}/{p['total']} {p['unit']}")
    for call in job["timings"]["text_calls"]:
        latency = (f"successful median {call['successful_p50_s']:.1f}s; max {call['successful_max_s']:.1f}s"
                   if call['successful_calls'] else "no successful calls")
        print(f"  {call['model']} {call['call']}: {call['successful_calls']}/{call['calls']} successful; "
              f"{latency}; errors {call['errors']}")
print(f"Queued media: {report['queued_media_s'] / 3600:.2f} hours")
if args.out:
    print(f"Report: {args.out.resolve()}")
