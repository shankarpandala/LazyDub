#!/usr/bin/env python3
"""Measure a fresh whole-video render with an experimental translation scene size.

Use the engine Python, under verify_mac.py's text-only proxy/network sandbox on the Mac.
This changes only this process's scene cutter, never the user's settings or production defaults.
Run each arm in a separate process/cache; alternate their order when comparing them.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_inference_components import thermal_state
from maata_engine import bench, render
from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH


async def measure(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mx.random.seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    provenance = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ("engine/uv.lock", "engine/models.lock.json")},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "mlx", "transformers", "chatterbox-tts")},
        "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip(),
        "thermal_start": thermal_state(),
        "seed": args.seed,
    }
    original_cut = render.RenderJob._cut_scenes
    original_keep = render.RenderJob._keep
    original_run = render.RenderJob.run
    started = 0.0
    takes: list[dict] = []

    async def run(job):
        nonlocal started
        started = time.perf_counter()
        return await original_run(job)

    def keep(job, st, *pos, **kwargs):
        original_keep(job, st, *pos, **kwargs)
        takes.append({"unit": st.unit.id, "seconds": round(time.perf_counter() - started, 4)})

    def cut(job):
        return original_cut(job, max_seconds=args.scene_seconds, max_units=args.scene_lines)

    render.RenderJob._cut_scenes = cut
    render.RenderJob._keep = keep
    render.RenderJob.run = run
    try:
        result = await bench._pipeline(argparse.Namespace(
            file=args.video, models=args.models, cache=args.cache, out=args.mp4_dir,
            backend="apple", translator=None, tts_script="telugu", baseline=None))
    finally:
        render.RenderJob._cut_scenes = original_cut
        render.RenderJob._keep = original_keep
        render.RenderJob.run = original_run
    trace_path = Path(args.cache) / bench.BENCH_ID / "units.jsonl"
    events = [json.loads(row) for row in trace_path.read_text().splitlines()]
    scene_events = [e for e in events if e.get("event") == "scene" and not e.get("dropped")]
    video = Path(args.video)
    return {
        "scope": "Fresh whole-video render; original local fixture; automated quality checks, no human listening",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **provenance, "thermal_end": thermal_state(),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "scene_seconds": args.scene_seconds, "scene_lines": args.scene_lines,
        "prompt_hash": PROMPT_HASH, "review_hash": REVIEW_HASH,
        "first_take_seconds": takes[0]["seconds"] if takes else None,
        "take_ready": takes, "scenes": scene_events, "pipeline": result,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--scene-seconds", type=float, required=True)
    parser.add_argument("--scene-lines", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Hold local calibration randomness fixed across arms")
    parser.add_argument("--cache", required=True, help="Dedicated benchmark cache; localbench0 is cleared")
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--models", default=str(Path.home() / "Library/Application Support/Maata/Models"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not (0 < args.scene_seconds < float("inf") and 0 < args.scene_lines <= 30):
        parser.error("positive finite seconds and 1..30 lines required")
    report = asyncio.run(measure(args))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(dest.resolve()), "pipeline_seconds": report["pipeline"]["pipeline_seconds"],
                      "first_take_seconds": report["first_take_seconds"]}))


if __name__ == "__main__":
    main()
