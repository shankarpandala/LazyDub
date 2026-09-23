"""maata-bench (spec §11): fetch pinned models, and measure the pipeline on a local audio file.

    maata-bench fetch [--backend apple]           download + verify the models for a backend
    maata-bench pipeline FILE [--backend apple]   run the full dub pipeline, print JSON metrics

Every number printed comes from this machine; nothing is estimated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from .backends import detect, load_backend
from .models import FetchError, fetch_model, models_for


def _machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "darwin":
        import subprocess

        for key, cmd in {"chip": ["sysctl", "-n", "machdep.cpu.brand_string"], "ram_bytes": ["sysctl", "-n", "hw.memsize"]}.items():
            try:
                info[key] = subprocess.check_output(cmd, text=True).strip()
            except Exception:
                pass
    return info


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.models).expanduser()
    token = os.environ.get("HF_TOKEN")
    for m in models_for(args.backend or detect()):
        print(f"→ {m['id']}  ({m['total_bytes'] / 1e9:.2f} GB, {m['license']})", flush=True)
        last = [0.0]

        def progress(path: str, got: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] > 0.5 or got == total:
                print(f"\r   {path}: {got / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                last[0] = now

        try:
            fetch_model(m, root, token, progress)
            print("\r   verified ✓" + " " * 60)
        except FetchError as e:
            print(f"\n   ✗ {e}")
            return 1
    return 0


async def _pipeline(args: argparse.Namespace) -> dict:
    from .resolve import ResolvedVideo, decode_audio
    from .session import Session
    from .urls import VideoRef

    t0 = time.perf_counter()
    backend = load_backend(args.backend, Path(args.models).expanduser())
    load_s = time.perf_counter() - t0
    audio = decode_audio(Path(args.file))
    seconds = len(audio) / 16_000

    class LocalResolver:
        def resolve(self, ref: VideoRef, cache_dir: Path) -> ResolvedVideo:
            return ResolvedVideo(ref.video_id, Path(args.file).name, seconds, "local", Path(args.file))

        def load_audio(self, video: ResolvedVideo):
            return audio

    events: list[dict] = []
    first_audio: list[float] = []
    started = [0.0]

    async def sj(m: dict) -> None:
        events.append(m)

    async def sb(_: bytes) -> None:
        if not first_audio:
            first_audio.append(time.perf_counter() - started[0])

    s = Session(backend, LocalResolver(), Path(args.cache).expanduser(), sj, sb)
    started[0] = time.perf_counter()
    await s.open("https://youtu.be/localbench0")
    errors = [e for e in events if e["type"] == "error"]
    if errors:
        raise SystemExit(f"pipeline failed: {errors[0]['message']}")
    while s._task and not s._task.done():
        await asyncio.sleep(0.2)
        if s._next_window() is None:
            break
    wall = time.perf_counter() - started[0]
    units = [e for e in events if e["type"] == "unit"]
    rates = [u["audioRate"] for u in units]
    added = sum(ed["added"] for u in units for ed in u["edits"])
    return {
        "machine": _machine(), "backend": backend.name, "device": backend.device, "file": Path(args.file).name,
        "audio_seconds": round(seconds, 2), "model_load_seconds": round(load_s, 2), "pipeline_seconds": round(wall, 2),
        "throughput_x_realtime": round(seconds / wall, 2) if wall else None,
        "time_to_first_audio_seconds": round(first_audio[0], 2) if first_audio else None,
        "units": len(units), "skipped": sum(1 for e in events if e["type"] == "unit_skipped"),
        "speedup_max": max(rates, default=None), "added_video_seconds_per_minute": round(added / (seconds / 60), 2) if seconds else None,
        "window_seconds": [round(e["seconds"], 2) for e in events if e["type"] == "stage_time"],
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maata-bench")
    p.add_argument("--models", default="~/Library/Application Support/Maata/Models" if sys.platform == "darwin" else "~/.local/share/maata/models")
    p.add_argument("--cache", default="~/Library/Caches/Maata" if sys.platform == "darwin" else "~/.cache/maata")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--backend", choices=["apple", "cuda"])
    pl = sub.add_parser("pipeline")
    pl.add_argument("file")
    pl.add_argument("--backend", choices=["apple", "cuda", "mock"])
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        raise SystemExit(cmd_fetch(args))
    print(json.dumps(asyncio.run(_pipeline(args)), indent=2))


if __name__ == "__main__":
    main()
