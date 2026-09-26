"""Timing yardsticks at speech level (ARCHITECTURE §3.10 metrics and acceptance). Pure.

A line is measured against its source speaker's diarized speech inside the line, not its span: §4.3/§4.5 size lines
to speech time, so a dub that ends early only because the source paused at its end is no error. The rows are the
planner's placed lines (`TimelinePlanner.stats`) or units.jsonl's `unit` and `skipped` events (`maata-bench`): dicts
with `start`, `end`, `speech` (the source speech inside the line, [[a, b], ...]), `voiced` (where the dub is voiced, in
video time; empty for a line with no dub) and `anchor_errors` (the voiced onset of each part timed against an English
onset inside the line, minus that onset).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

Span = tuple[float, float]
LONG_LINE = 8.0   # s of source span: a long line (§3.10's long-line end error)
# Regression guards (§3.10 acceptance): today's build already meets them, and they must stay met.
GUARDS = {"lag_p95": 0.3, "rate_p90": 1.1, "freeze_s_per_10min": 1.0}
SILENT_TARGET = 3.0      # s of source speech with no dub per minute of it (baseline 8.2)
END_ERROR_TARGET = 0.3   # s: the speech-level end error's median within +-this


def pct(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default)."""
    v = sorted(xs)
    k = (len(v) - 1) * q
    i = math.floor(k)
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (k - i)


def union(spans: Iterable) -> list[Span]:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(spans: list[Span]) -> float:
    return sum(b - a for a, b in spans)


def overlap(x: list[Span], y: list[Span]) -> float:
    """Seconds two unions (`union`) share."""
    i = j = 0
    got = 0.0
    while i < len(x) and j < len(y):
        got += max(min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]), 0.0)
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return got


def _r(v: float) -> float:
    return round(v, 4)


def line_metrics(u: Mapping) -> dict:
    """One line's speech-level end error (its last voiced moment minus the end of its source's last speech) and overlap
    (voiced time against speech time, intersection over union); None for a line with no dub or no speech."""
    speech, voiced = union(u.get("speech") or ()), union(u.get("voiced") or ())
    if not voiced or not speech:
        return {"end_error_s": None, "overlap_speech": None}
    both = overlap(speech, voiced)
    return {"end_error_s": _r(voiced[-1][1] - speech[-1][1]),
            "overlap_speech": _r(both / (length(speech) + length(voiced) - both))}


def speech_metrics(rows: Iterable[Mapping]) -> dict:
    """- `end_error_p50` (p10, p90): the dub's last voiced moment minus the end of the source's last speech inside the
      line; `end_early_1s_share`: the share of lines ending more than 1 s early by it; `long_end_error_p50`: its median
      over lines longer than LONG_LINE;
    - `overlap_speech_mean`: per line, the dub's voiced time against the source speech as intersection over union;
    - `silent_s_per_min`: seconds of source speech with no dub over it, per minute of source speech (skipped lines count);
    - `anchor_err_p50` / `anchor_err_p95`: the absolute onset error of the parts timed against an English onset."""
    rows = list(rows)
    ends, long_ends, ious, errs = [], [], [], []
    speech_all, voiced_all = [], []
    for u in rows:
        speech_all += union(u.get("speech") or ())
        voiced_all += union(u.get("voiced") or ())
        errs += [abs(float(e)) for e in u.get("anchor_errors") or ()]
        m = line_metrics(u)
        if m["end_error_s"] is None:
            continue
        ends.append(m["end_error_s"])
        if float(u["end"]) - float(u["start"]) > LONG_LINE:
            long_ends.append(m["end_error_s"])
        ious.append(m["overlap_speech"])
    speech_u, voiced_u = union(speech_all), union(voiced_all)
    heard = length(speech_u)
    out = {
        "speech_lines": len(ends),
        "end_error_p10": _r(pct(ends, 0.10)) if ends else None,
        "end_error_p50": _r(pct(ends, 0.50)) if ends else None,
        "end_error_p90": _r(pct(ends, 0.90)) if ends else None,
        "end_early_1s_share": _r(sum(e < -1.0 for e in ends) / len(ends)) if ends else None,
        "long_lines": len(long_ends),
        "long_end_error_p50": _r(pct(long_ends, 0.50)) if long_ends else None,
        "overlap_speech_mean": _r(sum(ious) / len(ious)) if ious else None,
        "silent_s_per_min": _r((heard - overlap(speech_u, voiced_u)) / (heard / 60.0)) if heard > 0 else None,
        "anchors": len(errs),
        "anchor_err_p50": _r(pct(errs, 0.50)) if errs else None,
        "anchor_err_p95": _r(pct(errs, 0.95)) if errs else None,
    }
    return out


def guards(stats: Mapping, baseline: Mapping | None = None) -> dict:
    """§3.10's acceptance, from `stats` (the planner's, with its speech-level metrics): the regression guards, each with
    its value, limit and whether it holds; `rate_step_p90` is held against `baseline`'s (the committed step-0 run), and
    the improvement targets that are measured against it too. Without a baseline those say None."""
    def held(value, limit, ok) -> dict:
        return {"value": value, "limit": limit, "ok": None if value is None or limit is None else ok}

    base = baseline or {}
    out = {k: held(stats.get(k), lim, stats.get(k) is not None and stats[k] <= lim + 1e-9) for k, lim in GUARDS.items()}
    step, step0 = stats.get("rate_step_p90"), base.get("rate_step_p90")
    out["rate_step_p90"] = held(step, step0, step is not None and step0 is not None and step <= step0 + 1e-9)
    silent, end = stats.get("silent_s_per_min"), stats.get("end_error_p50")
    out["silent_s_per_min"] = held(silent, SILENT_TARGET, silent is not None and silent <= SILENT_TARGET)
    out["end_error_p50"] = held(end, END_ERROR_TARGET, end is not None and abs(end) <= END_ERROR_TARGET)
    early, early0 = stats.get("end_early_1s_share"), base.get("end_early_1s_share")
    out["end_early_1s_share"] = held(early, None if early0 is None else early0 / 2,
                                     early is not None and early0 is not None and early <= early0 / 2 + 1e-9)
    long_, long0 = stats.get("long_end_error_p50"), base.get("long_end_error_p50")
    out["long_end_error_p50"] = held(long_, None if long0 is None else abs(long0) / 2,
                                     long_ is not None and long0 is not None and abs(long_) <= abs(long0) / 2 + 1e-9)
    return out
