"""maata-bench (spec §11): fetch pinned models, and measure the pipeline on a local video file.

    maata-bench fetch [--backend apple]           download + verify the models for a backend
    maata-bench pipeline FILE [--backend apple]   dub the whole video to an MP4 (a render job), print JSON metrics
        [--out DIR]                               ...saving the MP4 in DIR (default: <cache>/out)
        [--translator mock]                       ...with the mock translator instead of the Codex CLI (offline)
        [--tts-script latin]                      ...with English words given to the TTS in Latin script (decision D6)
        [--baseline JSON]                         ...with the timing targets measured against a committed run's JSON

Every number printed comes from this machine; nothing is estimated. The pipeline translates through the backend's
translator: the user's signed-in Codex CLI on apple and cuda, the mock one on the mock backend. The input file
is read in place and never changed (OFFLINE-RENDER §2.2: `LocalResolver`).

The bench's job lives in its own cache (`--cache`, default ~/Library/Caches/Maata-bench), which Maata never scans: a bench
job never shows in the Library, counts toward its retention, or is continued by the app after an interrupted run. Each run
starts from a clean job, so every stage (translation included) is measured on this run's file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from .backends import detect, load_backend
from .backends.claude_translator import ClaudeTranslator
from .backends.mock import MockSceneTranslator
from .models import FetchError, fetch_model, models_for
from .qa.coverage import CLASSES, voiced_class
from .timing.metrics import guards, pct, speech_metrics


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
    import resource  # (not on Windows: the engine imports this module for its stats)

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _mlx_peak_bytes() -> int | None:
    """MLX's peak memory in this process (Whisper and the separator run on it), or None where no model ran on MLX."""
    mx = sys.modules.get("mlx.core")
    get = getattr(mx, "get_peak_memory", None) if mx is not None else None
    return int(get()) if get is not None else None


REQUIRED_RATE_BAND = (0.9, 1.2)  # Descript's natural window: 73-83 % of its lines need a rate inside it (§7 step 4)


def _shares(labels: list[str]) -> dict:
    n = len(labels)
    return {k: round(labels.count(k) / n, 3) if n else None for k in CLASSES + ("other_tier", "unreviewed")}


def dub_metrics(units: Iterable) -> dict:
    """The §7 step 4 yardsticks over the voiced lines (`RenderJob.final_voiced`), as they were finally voiced:
    - `coverage`: the share of lines in each coverage class, counted only where the class is of the wording voiced
      (`qa.coverage.voiced_class`), with the lines whose class is of another of their wordings (`other_tier`) and those
      with none (`unreviewed`); `coverage_reviewed`: the same for the review's own classes, before any re-translation
      replaced a wording (§4.6); `retranslations_kept`: how many did;
    - `speech_fill`: seconds of dub played over seconds of source speech, summed over lines (and its per-line median);
    - `required_rate_in_band_share`: the share of lines whose take needs an audio rate within [0.9, 1.2] to fill exactly
      their speech time."""
    lines = [st for st in units if st.voiced and st.plan is not None and st.take_s and st.speech_s > 0]
    cov = [st.line.coverage for st in lines]
    rates = [st.take_s / st.speech_s for st in lines]
    lo, hi = REQUIRED_RATE_BAND
    return {
        "lines": len(lines),
        "coverage": _shares([voiced_class(st.line.coverage, st.tier) for st in lines]),
        "coverage_reviewed": _shares([voiced_class(st.line.coverage, st.tier, first=True) for st in lines]),
        "retranslations_kept": sum(1 for c in cov if c is not None and c.by == "validators"),
        "speech_fill": round(sum(st.plan.played for st in lines) / sum(st.speech_s for st in lines), 3) if lines else None,
        "speech_fill_median": round(statistics.median(st.plan.played / st.speech_s for st in lines), 3) if lines else None,
        "required_rate_in_band_share": round(sum(lo <= r <= hi for r in rates) / len(rates), 3) if rates else None,
    }


def render_metrics(events: Iterable[dict], video_seconds: float) -> dict:
    """The offline render's yardsticks (OFFLINE-RENDER §7) from units.jsonl's events of one run: each stage's seconds
    and GPU seconds (its `stage` event; `cached` when it was served from disk; the separate stage's `blocks` made), the
    GPU seconds per second of video, the engine's peak resident memory and MLX's peak memory, and the Claude calls and
    tokens by call type (the CLI's `claude` events)."""
    events = list(events)
    stages = {e["key"]: {"seconds": e["seconds"], "gpu_s": e["gpu_s"], "cached": e["cached"],
                         **({"blocks": e["blocks"]} if "blocks" in e else {})}
              for e in events if e.get("event") == "stage"}
    gpu = sum(x["gpu_s"] for x in stages.values())
    claude: dict[str, dict] = {}
    for e in events:
        if e.get("event") == "claude":
            c = claude.setdefault(e["call"], {"calls": 0, "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0})
            c["calls"] += 1
            for k in ("input_tokens", "cache_read_tokens", "output_tokens"):
                c[k] += e.get(k) or 0
    return {"stages": stages, "gpu_s": round(gpu, 2),
            "gpu_s_per_video_s": round(gpu / video_seconds, 4) if video_seconds else None,
            "peak_rss_bytes": _peak_rss_bytes(), "mlx_peak_bytes": _mlx_peak_bytes(), "claude": claude}


def timing_metrics(events: Iterable[dict], baseline: Mapping | None = None) -> dict:
    """The §3.10 yardsticks from units.jsonl's unit and skipped events, as the lines were finally voiced (a rephrase
    that replaced a take replaces its voiced time, how it was said and its freeze):
    - onset lag p95, rate p90, `rate_step_p90` (a speaker's rate change from one line to the next), freezes per 10 min;
    - the speech-level metrics (`timing.metrics.speech_metrics`): end error, overlap, seconds of speech per minute with
      no dub (skipped lines included), anchor onset error;
    - how lines were said (whole, pieces at hard breaks, joined, anchored);
    - `guards`: the acceptance guards and targets, against `baseline` (a committed run's `timing` block) where they
      are measured against one."""
    events = list(events)
    units = {e["id"]: dict(e) for e in events if e.get("event") == "unit"}
    for e in events:
        if e.get("event") == "rephrase" and e.get("outcome") == "replaced" and e["id"] in units:
            units[e["id"]].update({k: e[k] for k in ("voiced", "anchor_errors", "said", "freeze") if k in e})
    rows = sorted(units.values(), key=lambda u: (u["start"], u["id"]))
    skipped = [e for e in events if e.get("event") == "skipped"]
    steps, last = [], {}
    for u in rows:
        if u["speaker"] in last:
            steps.append(abs(u["audio_rate"] - last[u["speaker"]]))
        last[u["speaker"]] = u["audio_rate"]
    spans = rows + skipped
    minutes = (max(u["end"] for u in spans) - min(u["start"] for u in spans)) / 60.0 if spans else 0.0
    said = [u.get("said", "whole") for u in rows]
    out = {
        "lines": len(rows), "skipped": len(skipped),
        "lag_p95": round(pct([u["lag"] for u in rows], 0.95), 4) if rows else None,
        "rate_p90": round(pct([u["audio_rate"] for u in rows], 0.90), 4) if rows else None,
        "rate_step_p90": round(pct(steps, 0.90), 4) if steps else None,
        "freeze_s_per_10min": round(10.0 * sum(u["freeze"] for u in rows) / minutes, 4) if minutes > 0 else None,
        "said": {k: said.count(k) for k in ("whole", "pieces", "joined", "anchored")},
        **speech_metrics([*rows, *({**e, "voiced": []} for e in skipped)]),
    }
    out["guards"] = guards(out, baseline)
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.models).expanduser()
    token = os.environ.get("HF_TOKEN")
    missing_gated: list[str] = []
    for m in models_for(args.backend or detect()):
        total = m.get("total_bytes")
        if not isinstance(total, (int, float)) or total < 0:
            total = sum(f["size"] for f in m["files"])
        print(f"→ {m['id']}  ({total / 1e9:.2f} GB, {m['license']})", flush=True)
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
            if not m.get("gated"):
                return 1
            # Optional for dubbing: the engine falls back to a single speaker without it.
            missing_gated.append(f"https://huggingface.co/{m['repo']}")
    for url in missing_gated:
        print(f"! Skipped a gated model. Accept its terms at {url}, set HF_TOKEN, then re-run to add it.")
    return 0


BENCH_ID = "localbench0"  # the bench's job folder in its cache (a video id's shape)


def _fresh(job_dir: Path, mp4: Path) -> None:
    """A clean job for each run: the bench's job folder goes (a job on another file, or one whose stages would be
    served from disk), and the MP4 its last run saved at `mp4`, where this run saves (replaced, as a job replaces its
    own earlier file, not saved beside it as " (2)")."""
    from .render import _read_json

    out = (_read_json(job_dir / "render" / "job.json") or {}).get("output") or {}
    if out.get("path") == str(mp4) and mp4.is_file() and mp4.stat().st_size == out.get("bytes"):
        mp4.unlink()
    shutil.rmtree(job_dir, ignore_errors=True)


async def _pipeline(args: argparse.Namespace) -> dict:
    import av

    from . import export
    from .render import RenderJob, RenderSettings
    from .resolve import LocalResolver
    from .text.tenglish import latin_ratio

    path = Path(args.file).expanduser().resolve()
    try:  # before any model loads or Claude call: the MP4 copies the file's video stream
        export.probe(path)
    except (ValueError, OSError, av.FFmpegError) as e:
        raise SystemExit(f"pipeline: {path.name} can't be dubbed to a video: {e}") from None
    t0 = time.perf_counter()
    backend = load_backend(args.backend, Path(args.models).expanduser())
    load_s = time.perf_counter() - t0
    if args.translator:
        backend.translator = MockSceneTranslator if args.translator == "mock" else ClaudeTranslator
    cache = Path(args.cache).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else cache / "out"
    _fresh(cache / BENCH_ID, out_dir / export.output_name(path.stem, BENCH_ID, None))
    errors: list[dict] = []

    async def on_event(m: dict) -> None:
        if m.get("type") == "claude_error":
            errors.append(m)

    job = RenderJob(backend, LocalResolver(path), cache, BENCH_ID, RenderSettings(tts_script=args.tts_script),
                    on_event=on_event, output_dir=out_dir)
    since = time.time()
    started = time.perf_counter()
    status = await job.run()
    wall = time.perf_counter() - started
    if status != "done":
        raise SystemExit(f"pipeline {status}: {job.doc['error'] or (errors[0]['message'] if errors else '')}")
    trace = [e for line in (job.cache_dir / BENCH_ID / "units.jsonl").read_text(encoding="utf-8").splitlines()
             if (e := json.loads(line)).get("t", 0) >= since]  # (this run's: the job is a clean one)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("timing") if args.baseline else None
    seconds = float(job.doc["duration"])
    lines = [e for e in trace if e.get("event") == "unit"]
    out = job.doc["output"]
    return {
        "machine": _machine(), "backend": backend.name, "device": backend.device, "file": path.name,
        "translator": backend.translator.__name__, "translation_model": job.tr.model,
        "translation_provider": "mock" if args.translator == "mock" or backend.name == "mock" and not args.translator
                                else "codex", "tts_script": args.tts_script,
        "audio_seconds": round(seconds, 2), "model_load_seconds": round(load_s, 2), "pipeline_seconds": round(wall, 2),
        "throughput_x_realtime": round(seconds / wall, 2) if wall else None,
        "calibration_seconds": round(job.calibrate_s, 2), "speakers": len(job.registry.speakers),
        "units": len(lines), "skipped": sum(1 for e in trace if e.get("event") == "skipped"),
        "speedup_max": max((u["audio_rate"] for u in lines), default=None),
        "latin_ratio": round(sum(latin_ratio(u["telugu"]) for u in lines) / len(lines), 3) if lines else None,
        "render": render_metrics(trace, seconds),
        "dub": dub_metrics(job.final_voiced()),
        "timing": timing_metrics(trace, baseline),
        "planner": job._final.stats() if job._final is not None else None,
        "export": {"path": out["path"], "bytes": out["bytes"], "seconds": job.doc["stages"]["export"]["seconds"],
                   "video_seconds": job.doc["stages"]["export"]["total"], "loudness": out["loudness"],
                   "warning": out["warning"], "video": {k: v for k, v in job.doc["source"]["video"].items()
                                                         if k != "fingerprint"}},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maata-bench")
    p.add_argument("--models", default="~/Library/Application Support/Maata/Models" if sys.platform == "darwin" else "~/.local/share/maata/models")
    p.add_argument("--cache", help="the bench's own cache, which Maata never scans",
                   default="~/Library/Caches/Maata-bench" if sys.platform == "darwin" else "~/.cache/maata-bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--backend", choices=["apple", "cuda"])
    pl = sub.add_parser("pipeline")
    pl.add_argument("file", help="a local video file (it is read in place, never changed)")
    pl.add_argument("--backend", choices=["apple", "cuda", "mock"])
    pl.add_argument("--out", help="the folder the MP4 is saved in (default: <cache>/out)")
    pl.add_argument("--translator", choices=["codex", "mock", "claude"],
                    help="default: the backend's own; claude is a legacy alias for codex")
    pl.add_argument("--tts-script", choices=["telugu", "latin"], default="telugu")
    pl.add_argument("--baseline", help="a committed pipeline JSON: the timing targets are measured against its own")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        raise SystemExit(cmd_fetch(args))
    print(json.dumps(asyncio.run(_pipeline(args)), indent=2))


if __name__ == "__main__":
    main()
