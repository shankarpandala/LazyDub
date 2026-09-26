"""maata-bench (spec §11): fetch pinned models, and measure the pipeline on a local audio file.

    maata-bench fetch [--backend apple]           download + verify the models for a backend
    maata-bench pipeline FILE [--backend apple]   run the full dub pipeline, print JSON metrics
        [--translator mock]                       ...with the mock translator instead of the Claude CLI (offline)
        [--tts-script latin]                      ...with English words given to the TTS in Latin script (decision D6)
        [--timing v1]                             ...timed as before timing v2: the baseline run (v2-whole: v2, no pieces)
        [--baseline JSON]                         ...with the timing targets measured against a committed run's JSON

Every number printed comes from this machine; nothing is estimated. The pipeline translates through the backend's
translator: the user's signed-in Claude CLI on apple and cuda (ADR-019), the mock one on the mock backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
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
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


REQUIRED_RATE_BAND = (0.9, 1.2)  # Descript's natural window: 73-83 % of its lines need a rate inside it (§7 step 4)


def _shares(labels: list[str]) -> dict:
    n = len(labels)
    return {k: round(labels.count(k) / n, 3) if n else None for k in CLASSES + ("other_tier", "unreviewed")}


def dub_metrics(units: Iterable) -> dict:
    """The §7 step 4 yardsticks over the voiced lines (session `UnitState`s), as they were finally voiced:
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


def realtime_metrics(events: Iterable[dict]) -> dict:
    """The §7 step 5 yardsticks from units.jsonl's unit events: the voicer's waits for the GPU (`lock_wait_s`, p95 by
    nearest rank; the target is 2 s or less), how many lines got each number of takes from the throughput governor
    (`takes_n`), the retakes of batches that all failed, and why takes failed."""
    units = [e for e in events if e.get("event") == "unit"]
    waits = sorted(u["lock_wait_s"] for u in units if u.get("lock_wait_s") is not None)
    failures = [f for u in units for f in u.get("take_failures") or ()]
    return {
        "voicer_lock_wait_p95_s": waits[max(0, -(-95 * len(waits) // 100) - 1)] if waits else None,
        "takes_n": {str(n): sum(1 for u in units if u.get("takes_n") == n)
                    for n in sorted({u["takes_n"] for u in units if u.get("takes_n") is not None})},
        "retakes": sum(u.get("retakes") or 0 for u in units),
        "take_failures": {f: failures.count(f) for f in sorted(set(failures))},
    }


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
            if not m.get("gated"):
                return 1
            # Optional for dubbing: the engine falls back to a single speaker without it.
            missing_gated.append(f"https://huggingface.co/{m['repo']}")
    for url in missing_gated:
        print(f"! Skipped a gated model. Accept its terms at {url}, set HF_TOKEN, then re-run to add it.")
    return 0


async def _pipeline(args: argparse.Namespace) -> dict:
    from .resolve import ResolvedVideo, decode_audio
    from .session import Session
    from .urls import VideoRef

    t0 = time.perf_counter()
    backend = load_backend(args.backend, Path(args.models).expanduser())
    load_s = time.perf_counter() - t0
    if args.translator:
        backend.translator = MockSceneTranslator if args.translator == "mock" else ClaudeTranslator
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

    s = Session(backend, LocalResolver(), Path(args.cache).expanduser(), sj, sb, lookahead=seconds + 60, prepass=min(600.0, seconds),
                tts_script=args.tts_script, timing=args.timing)
    started[0], since = time.perf_counter(), time.time()
    await s.open("https://youtu.be/localbench0")
    prepass_s = None
    while True:
        await asyncio.sleep(0.2)
        errors = [e for e in events if e["type"] in ("error", "claude_error")]
        if errors:
            await s.close()
            raise SystemExit(f"pipeline failed: {errors[0]['message']}")
        if prepass_s is None and any(e["type"] == "speaker_scan" and e["phase"] == "ready" for e in events):
            prepass_s = time.perf_counter() - started[0]
        if s.ready.covered(0.0, seconds - 0.5) or all(t.done() for t in s._tasks):
            break
    wall = time.perf_counter() - started[0]
    await s.close()
    # from this run's events only: units.jsonl keeps those of earlier runs on the same file
    trace = [e for line in (s._dir / "units.jsonl").read_text(encoding="utf-8").splitlines()
             if (e := json.loads(line)).get("t", 0) >= since]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("timing") if args.baseline else None
    units = [e for e in events if e["type"] == "unit"]
    rates = [u["audioRate"] for u in units]
    added = sum(ed["added"] for u in units for ed in u["edits"])
    from .text.tenglish import latin_ratio

    return {
        "machine": _machine(), "backend": backend.name, "device": backend.device, "file": Path(args.file).name,
        "translator": backend.translator.__name__, "tts_script": args.tts_script, "timing_mode": s.timing,
        "audio_seconds": round(seconds, 2), "model_load_seconds": round(load_s, 2), "pipeline_seconds": round(wall, 2),
        "throughput_x_realtime": round(seconds / wall, 2) if wall else None,
        "speaker_prepass_seconds": round(prepass_s, 2) if prepass_s is not None else None,
        "calibration_seconds": round(s.calibrate_s, 2),  # of the voices' clones, most inside the pre-pass
        "time_to_first_audio_seconds": round(first_audio[0], 2) if first_audio else None,
        "speakers": len(s.registry.speakers),
        "units": len(units), "skipped": sum(1 for e in events if e["type"] == "unit_skipped"),
        "speedup_max": max(rates, default=None), "added_video_seconds_per_minute": round(added / (seconds / 60), 2) if seconds else None,
        "latin_ratio": round(sum(latin_ratio(u["telugu"]) for u in units) / len(units), 3) if units else None,
        "dub": dub_metrics(s.units.values()),
        "realtime": realtime_metrics(trace),
        "timing": timing_metrics(trace, baseline),
        "planner": s.planner.stats(),
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
    pl.add_argument("--translator", choices=["claude", "mock"], help="default: the backend's own")
    pl.add_argument("--tts-script", choices=["telugu", "latin"], default="telugu")
    pl.add_argument("--timing", choices=["v2", "v2-whole", "v1"], default="v2",
                    help="v1: timed as before timing v2, for the baseline run (ADR-017); v2-whole: v2 with every "
                         "sentence said whole")
    pl.add_argument("--baseline", help="a committed pipeline JSON (a --timing v1 run): the timing targets are measured "
                                       "against its own")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        raise SystemExit(cmd_fetch(args))
    print(json.dumps(asyncio.run(_pipeline(args)), indent=2))


if __name__ == "__main__":
    main()
