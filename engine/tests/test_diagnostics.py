"""Diagnostics must separate runs, tolerate appends, and never disclose transcript content."""

import json

from maata_engine.diagnostics import snapshot


def write_job(root, events, done=20):
    folder = root / "example0001"
    (folder / "render").mkdir(parents=True, exist_ok=True)
    (folder / "render/job.json").write_text(json.dumps({
        "title": "private title", "status": "running", "duration": 120, "updatedAt": 120,
        "stage": "voice_lines", "stages": {"voice_lines": {"state": "running", "done": done,
            "total": 100, "unit": "speech s", "seconds": 10000, "gpu_s": 5000}}}))
    (folder / "units.jsonl").write_text("\n".join(json.dumps(e) for e in events) + '\n{"t":')


def test_latest_run_excludes_history_and_sums_only_current_intervals(tmp_path):
    events = [
        {"t": 5, "event": "claude", "run_id": "old", "wall_s": 999, "call": "scene"},
        {"t": 100, "event": "run_start", "run_id": "new"},
        {"t": 101, "event": "stage_start", "run_id": "new", "key": "translate"},
        {"t": 120, "event": "claude", "run_id": "new", "wall_s": 10, "call": "scene",
         "provider": "codex", "model": "gpt-6-luna", "input_tokens": 100, "cache_read_tokens": 50},
        {"t": 125, "event": "claude", "run_id": "new", "wall_s": 10, "call": "scene",
         "provider": "codex", "model": "gpt-6-luna", "source": "private transcript"},
        {"t": 130, "event": "stage_timing", "run_id": "new", "key": "translate",
         "wall_s": 29, "gpu_s": 0, "outcome": "done", "cached": False},
        {"t": 131, "event": "unit", "run_id": "new", "id": 1, "synth_s": 999,
         "source": "private transcript", "wording": "private translation", "coverage_voiced": "C"},
        {"t": 132, "event": "unit", "run_id": "new", "id": 1, "coverage_voiced": "m",
         "coverage": {"by": "validators"}},
        {"t": 132, "event": "gpu_work", "run_id": "new", "stage": "voice_lines",
         "operation": "vocode", "run_s": 2, "queue_wait_s": 1},
        {"t": 132, "event": "gpu_work", "run_id": "old", "run_s": 999},
        {"t": 133, "event": "run_end", "run_id": "new", "wall_s": 33, "status": "done"},
    ]
    write_job(tmp_path, events)
    report = snapshot(tmp_path, now=150)
    job = report["jobs"][0]
    assert job["run_id"] == "new" and job["run_wall_s"] == 33
    assert job["timings"]["stages"] == [{"stage": "translate", "wall_s": 29, "gpu_s": 0,
                                           "outcome": "done", "cached": False}]
    call, = job["timings"]["text_calls"]
    assert call["summed_call_s"] == 20 and call["active_union_s"] == 15
    assert job["timings"]["finalized_units_in_scope"] == 1
    assert job["timings"]["finalized_coverage"] == {"m": 1}
    assert job["timings"]["finalized_coverage_provenance"] == {"validators": 1}
    assert job["timings"]["gpu_work"] == [{"stage": "voice_lines", "operation": "vocode", "calls": 1,
                                           "run_s": 2, "queue_wait_s": 1}]
    for forbidden in ("private", "10000", "5000", "999"):
        assert forbidden not in json.dumps(report)


def test_legacy_window_is_explicit_and_does_not_claim_a_fresh_run(tmp_path):
    write_job(tmp_path, [{"t": 50, "event": "claude", "wall_s": 30},
                         {"t": 140, "event": "claude", "wall_s": 20}])
    job = snapshot(tmp_path, now=150, window_s=30)["jobs"][0]
    assert job["scope"] == "recent_window_without_run_boundary"
    assert job["run_wall_s"] is None and job["run_id"] is None
    assert job["timings"]["text_calls"][0]["calls"] == 1


def test_progress_delta_never_crosses_run_reset_or_counts_regression(tmp_path):
    events = [{"t": 100, "event": "run_start", "run_id": "a"}]
    write_job(tmp_path, events)
    before = snapshot(tmp_path, now=130)
    write_job(tmp_path, events, done=40)
    job = snapshot(tmp_path, now=140, previous=before)["jobs"][0]
    assert job["progress_delta"]["voice_lines"]["units_per_second"] == 2
    write_job(tmp_path, events, done=5)
    assert snapshot(tmp_path, now=140, previous=before)["jobs"][0]["progress_delta"] == {}
    write_job(tmp_path, events + [{"t": 135, "event": "run_start", "run_id": "b"}], done=40)
    assert snapshot(tmp_path, now=140, previous=before)["jobs"][0]["progress_delta"] is None


def test_token_cache_fraction_uses_provider_specific_accounting(tmp_path):
    write_job(tmp_path, [
        {"t": 140, "event": "claude", "provider": "codex", "model": "gpt-6-luna",
         "input_tokens": 100, "cache_read_tokens": 80},
        {"t": 141, "event": "claude", "model": "claude-haiku-4-5-20251001",
         "input_tokens": 2, "cache_read_tokens": 80, "cache_creation_tokens": 18},
        {"t": 142, "event": "claude", "input_tokens": 2, "cache_read_tokens": 80},
    ])
    calls = {r["model"]: r for r in snapshot(tmp_path, now=150)["jobs"][0]["timings"]["text_calls"]}
    assert calls["gpt-6-luna"]["cached_input_fraction"] == .8
    assert calls["claude-haiku-4-5-20251001"]["cached_input_fraction"] == .8
    assert calls["unknown"]["cached_input_fraction"] is None


def test_usage_limit_rejections_do_not_masquerade_as_fast_successes(tmp_path):
    write_job(tmp_path, [
        {"t": 110, "event": "claude", "call": "scene", "wall_s": 20},
        {"t": 120, "event": "claude", "call": "scene", "wall_s": 30},
        *[{"t": 130 + i, "event": "claude", "call": "scene", "wall_s": 2, "error": "usage_limit"}
          for i in range(5)],
        {"t": 140, "event": "claude", "call": "review", "wall_s": 2, "error": "usage_limit"},
    ])
    calls = {r["call"]: r for r in snapshot(tmp_path, now=150)["jobs"][0]["timings"]["text_calls"]}
    assert calls["scene"]["p50_s"] == 2  # historical all-attempt metric remains explicitly available
    assert calls["scene"]["successful_calls"] == 2
    assert calls["scene"]["successful_p50_s"] == 25
    assert calls["scene"]["successful_max_s"] == 30
    assert calls["review"]["successful_calls"] == 0
    assert calls["review"]["successful_p50_s"] is None
    assert calls["review"]["successful_max_s"] is None
