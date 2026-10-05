"""Read-only, transcript-free queue diagnostics. Concurrent stage times must never be added as wall time."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value, default=0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _events(path: Path) -> list[dict]:
    events = []
    try:
        with path.open() as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:  # an in-flight append may have an incomplete final row
                    continue
                if isinstance(event, dict) and isinstance(event.get("t"), (int, float)):
                    events.append(event)
    except OSError:
        pass
    return events


def _union(intervals: list[tuple[float, float]]) -> float:
    end, total = float("-inf"), 0.0
    for a, b in sorted(intervals):
        if b > a:
            total += max(0.0, b - max(a, end))
            end = max(end, b)
    return round(total, 3)


def _timings(events: list[dict], since: float, until: float) -> dict:
    calls = defaultdict(list)
    for e in events:
        if e.get("event") == "claude":  # legacy event name; provider is explicit in current engines
            calls[(str(e.get("provider") or "legacy"), str(e.get("model") or "unknown"),
                   str(e.get("call") or "unknown"))].append(e)
    text = []
    for (provider, model, kind), rows in sorted(calls.items()):
        durations = [_number(e.get("wall_s")) for e in rows]
        input_tokens = sum(_number(e.get("input_tokens")) for e in rows)
        cached = sum(_number(e.get("cache_read_tokens")) for e in rows)
        # Codex input usage includes cached tokens; Claude usage reports uncached input separately.
        total_input = (input_tokens if provider == "codex" else
                       input_tokens + cached + sum(_number(e.get("cache_creation_tokens")) for e in rows)
                       if provider in ("claude", "anthropic") or model.startswith("claude-") else None)
        text.append({"provider": provider, "model": model, "call": kind, "calls": len(rows),
                     "p50_s": round(statistics.median(durations), 3), "max_s": round(max(durations), 3),
                     "summed_call_s": round(sum(durations), 3),
                     "active_union_s": _union([(max(since, e["t"] - d), min(until, e["t"]))
                                               for e, d in zip(rows, durations)]),
                     "errors": dict(Counter(e["error"] for e in rows if e.get("error"))),
                     "input_tokens": int(input_tokens), "cached_input_tokens": int(cached),
                     "total_input_tokens": int(total_input) if total_input is not None else None,
                     "cached_input_fraction": round(cached / total_input, 3) if total_input else None})
    starts, finished = {}, {}
    for e in events:
        if e.get("event") == "stage_start":
            starts[e["key"]] = e["t"]
        elif e.get("event") == "stage_timing":
            finished[e["key"]] = {k: e.get(k) for k in ("wall_s", "gpu_s", "outcome", "cached")}
    stages = [{"stage": key, **finished.get(key, {"wall_s": round(until - t, 3), "gpu_s": None,
                                                 "outcome": "in_progress", "cached": None})}
              for key, t in starts.items()]
    latest = {e["id"]: e for e in events if e.get("event") == "unit" and "id" in e}
    gpu_work = defaultdict(list)
    for e in events:
        if e.get("event") == "gpu_work":
            gpu_work[(str(e.get("stage") or "unknown"), str(e.get("operation") or "unknown"))].append(e)
    return {"text_calls": text, "stages": stages,
            "gpu_work": [{"stage": stage, "operation": operation, "calls": len(rows),
                          "run_s": round(sum(_number(e.get("run_s")) for e in rows), 3),
                          "queue_wait_s": round(sum(_number(e.get("queue_wait_s")) for e in rows), 3)}
                         for (stage, operation), rows in sorted(gpu_work.items())],
            "finalized_units_in_scope": len(latest),
            "finalized_coverage": dict(Counter(str(e.get("coverage_voiced", "unknown")) for e in latest.values())),
            "finalized_coverage_provenance": dict(Counter(str((e.get("coverage") or {}).get("by", "unknown"))
                                                          for e in latest.values())),
            "skipped_events": sum(e.get("event") == "skipped" for e in events),
            "note": "Stage and model-call times overlap. Finalized units can include restored takes; their stored "
                    "synthesis costs are deliberately not summed as work performed in this run. GPU work records "
                    "completed scheduled operations only, not in-flight work or hardware utilization."}


def snapshot(cache: Path, *, now: float | None = None, window_s: float = 900,
             previous: dict | None = None) -> dict:
    now = time.time() if now is None else now
    old = {j["video_id"]: j for j in (previous or {}).get("jobs", [])}
    jobs = []
    for path in sorted(cache.glob("*/render/job.json")):
        doc = _read(path)
        if not doc:
            continue
        vid = path.parent.parent.name
        events = _events(path.parent.parent / "units.jsonl")
        starts = [e for e in events if e.get("event") == "run_start" and e.get("run_id")]
        run = starts[-1] if starts else None
        since = run["t"] if run else now - window_s
        run_id = run["run_id"] if run else None
        selected = [e for e in events if since <= e["t"] <= now and
                    (run_id is None or e.get("run_id") == run_id)]
        ended = next((e for e in reversed(selected) if e.get("event") == "run_end"), None)
        stages = {k: {field: s.get(field) for field in ("state", "done", "total", "unit")}
                  for k, s in doc.get("stages", {}).items()}
        delta = None
        prior = old.get(vid)
        interval = now - _number((previous or {}).get("observed_at"), now)
        if prior and interval > 0 and run_id == prior.get("run_id"):
            delta = {}
            for k, stage in stages.items():
                a = (prior.get("progress", {}).get(k) or {})
                change = _number(stage.get("done")) - _number(a.get("done"))
                if stage.get("total") == a.get("total") and change >= 0:
                    delta[k] = {"advanced": round(change, 3), "interval_s": round(interval, 3),
                                "units_per_second": round(change / interval, 4), "unit": stage.get("unit")}
        jobs.append({"video_id": vid, "status": doc.get("status"), "stage": doc.get("stage"),
                     "duration_s": doc.get("duration"), "queue_at": doc.get("queuedAt"),
                     "state_age_s": round(max(0, now - _number(doc.get("updatedAt"), now)), 1),
                     "run_id": run_id, "scope": "latest_run" if run else "recent_window_without_run_boundary",
                     "scope_started_at": since,
                     "run_wall_s": ended.get("wall_s") if ended else None,
                     "observed_span_s": round((ended["t"] if ended else now) - since, 3),
                     "progress": stages, "progress_delta": delta, "timings": _timings(selected, since, ended["t"] if ended else now),
                     "output_ready": bool(doc.get("output")), "quality_report": doc.get("report"),
                     "coverage": doc.get("coverage"), "has_error": bool(doc.get("error"))})
    queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["queue_at"] or 0)
    return {"observed_at": now, "window_s": window_s, "jobs": jobs,
            "queue": [j["video_id"] for j in queued],
            "queued_media_s": sum(_number(j["duration_s"]) for j in queued),
            "note": "Read-only local aggregate; no titles, source text, translations, credentials or media included. "
                    "Old traces without run boundaries use a recent window, not a claimed fresh-run duration."}
