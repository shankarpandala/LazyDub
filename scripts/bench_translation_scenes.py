#!/usr/bin/env python3
"""Compare translation scene caps without loading or synthesizing audio.

Default: inspect the requests, with no Claude calls. After reviewing the harness:
  engine/.venv/bin/python scripts/bench_translation_scenes.py --run --order AB --out /tmp/text-pilot.json
  engine/.venv/bin/python scripts/bench_translation_scenes.py --run --order ABBAABBAAB --out /tmp/text-paired.json

A uses 150 seconds / 30 lines; B uses 30 seconds / 6 lines. Every arm has a fresh local line cache. Provider prompt
caches cannot be cleared: their usage is retained, and ABBA balances order only approximately. The production render
translation, lane, context, duration-pick, review and fit methods run unchanged. Calibration and the whole-video brief
are held fixed; audio feedback, voiced-wording reviews, export and final video latency are outside this experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from maata_engine.backends.base import Backend, Brief, GlossaryEntry, SpeakerNote, VideoMeta
from maata_engine.backends.claude_translator import ClaudeTranslator, LIGHT_EFFORT, line_json, parse_brief
from maata_engine.dubber import UnitState, VoiceState
from maata_engine.qa.coverage import coverage_json, voiced_class
from maata_engine.render import LANES, RenderJob, RenderSettings
from maata_engine.speakers import SpeakerRegistry
from maata_engine.text.akshara import count_units
from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH, SHOTS_VERSION
from maata_engine.timing.duration import VoiceKey
from maata_engine.types import SourceUnit, TimedWord, VoiceKind

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "scripts/fixtures/translation-scenes.json"
CAPS = {"A": (150.0, 30), "B": (30.0, 6)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def hygiene() -> dict:
    if platform.system() != "Darwin":
        return {"thermal_state": None, "power": "unavailable"}
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes, objc.objc_getClass.restype = [ctypes.c_char_p], ctypes.c_void_p
    objc.sel_registerName.argtypes, objc.sel_registerName.restype = [ctypes.c_char_p], ctypes.c_void_p
    objc.objc_msgSend.argtypes, objc.objc_msgSend.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p
    instance = objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    objc.objc_msgSend.restype = ctypes.c_long
    return {"thermal_state": int(objc.objc_msgSend(instance, objc.sel_registerName(b"thermalState"))),
            "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0]}


def brief_from(data: dict) -> Brief:
    meta = dict(data["meta"])
    for key in ("chapters", "talk_shares"):
        meta[key] = tuple(tuple(x) for x in meta.get(key, ()))
    meta["tags"] = tuple(meta.get("tags", ()))
    speakers = tuple(SpeakerNote(**{**s, "address": tuple(tuple(x) for x in s.get("address", ()))})
                     for s in data.get("speakers", ()))
    return Brief(data["version"], VideoMeta(**meta), data.get("topic", ""), data.get("register", ""), speakers,
                 tuple(GlossaryEntry(**g) for g in data.get("glossary", ())), tuple(data.get("entities", ())),
                 tuple(data.get("idioms", ())), data.get("numbers", ""),
                 tuple(tuple(x) for x in data.get("asr_fixes", ())))


class NoAudio:
    """Fail visibly if the text benchmark ever attempts an audio backend call."""

    def __getattr__(self, name):
        raise RuntimeError(f"Audio is forbidden in this benchmark: {name}")


def backend() -> Backend:
    forbidden = NoAudio()
    return Backend("text-benchmark", "none", forbidden, forbidden, ClaudeTranslator, forbidden)


class TextJob(RenderJob):
    def __init__(self, cache: Path, fixture: dict, cap: tuple[float, int]):
        super().__init__(backend(), None, cache, "textbench00", RenderSettings.from_json(fixture["settings"]))
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        self.duration = fixture["duration"]
        self.events: list[dict] = []
        self.requests: list[dict] = []
        self.ready: dict[int, float] = {}
        self.contiguous: list[dict] = []
        self.started = time.monotonic()
        self.registry = SpeakerRegistry.restore(fixture["registry"], 0.0, self.duration)
        states = []
        for row in fixture["units"]:
            unit = dict(row["unit"])
            unit["words"] = [TimedWord(**w) for w in unit.get("words", ())]
            states.append(UnitState(SourceUnit(**unit), row["next_start"], row["speech_s"], row.get("music", False)))
        self._index(states)
        for speaker, row in fixture["voices"].items():
            key = VoiceKey(**row["key"])
            self.voices[speaker] = VoiceState(voice=object(), kind=VoiceKind.CLONED, status="cloned", key=key)
            self.estimator.calibrate(key, row["calibration"])

    def _duration(self):
        return self.duration

    def _cut_scenes(self):
        return super()._cut_scenes(max_seconds=self.cap[0], max_units=self.cap[1])

    def _trace(self, rec):
        self.events.append({**rec, "elapsed_s": round(time.monotonic() - self.started, 6)})

    def _scene_request(self, sc, lines, whole=False):
        req = super()._scene_request(sc, lines, whole)
        self.requests.append({"elapsed_s": time.monotonic() - self.started, "request": asdict(req)})
        return req

    async def _review(self, req, lines):
        self.events.append({"event": "review_picks", "scene": req.scene,
                            "chosen": {str(i): self._pick(self.units[i], line)[0] for i, line in lines.items()},
                            "elapsed_s": time.monotonic() - self.started})
        return await super()._review(req, lines)

    async def _fit(self, req):
        self.requests.append({"elapsed_s": time.monotonic() - self.started, "request": asdict(req)})
        await super()._fit(req)

    async def _claude_failed(self, err):
        # A controlled comparison aborts on provider failure instead of retrying the job after a long hold.
        raise err

    def capture_ready(self):
        now = time.monotonic() - self.started
        for sc in self.scenes:
            if sc.ready and sc.no not in self.ready:
                self.ready[sc.no] = now
        prefix = []
        for sc in self.scenes:
            if sc.no not in self.ready:
                break
            prefix.append(sc)
        count = len(prefix)
        if count and (not self.contiguous or count != self.contiguous[-1]["scenes"]):
            self.contiguous.append({"elapsed_s": now, "scenes": count,
                                    "lines": sum(len(sc.lines) for sc in prefix),
                                    "source_until_s": prefix[-1].lines[-1].unit.end})

    def _progress(self, key, done, total):
        super()._progress(key, done, total)
        self.capture_ready()

    def _side(self, coro):
        task = super()._side(coro)
        task.add_done_callback(lambda _: self.capture_ready())
        return task


def extract(cache: Path) -> dict:
    """Freeze the source units and initial calibrated rates, never the prior translated text or learned TTS rates."""
    render = cache / "render"
    old = json.loads((render / "job.json").read_text())
    registry = SpeakerRegistry.restore(json.loads((render / "diarization.json").read_text()), 0, old["duration"])
    with tempfile.TemporaryDirectory(prefix="maata-text-extract-") as tmp:
        job = RenderJob(backend(), None, Path(tmp), "textbench00", RenderSettings.from_json(old["settings"]))
        job.registry = registry
        states = job._segment([r for r in read_rows(render / "transcript.jsonl") if "a" in r])
    trace = read_rows(cache / "units.jsonl")
    expected = {r["id"]: r for r in trace if r.get("event") == "unit"}
    assert len(states) == len(expected), "The frozen source no longer produces the same units"
    for st in states:
        got = expected[st.unit.id]
        assert (st.unit.text, st.unit.speaker) == (got["source"], got["speaker"])
        assert abs(st.unit.start - got["start"]) < 1e-6 and abs(st.unit.end - got["end"]) < 1e-6
        assert abs(st.speech_s - got["speech_s"]) <= 0.0051
        assert len(st.unit.breaks) == got["breaks"] and len(st.unit.anchors) == got["anchors"]
    source = old["source"]
    meta = VideoMeta(old["title"], old["channel"], source.get("description", ""),
                     tuple(tuple(x) for x in source.get("chapters", ())), tuple(source.get("tags", ())),
                     tuple(sorted(registry.talk_share().items())))
    brief = parse_brief(meta, read_rows(cache / "briefs.jsonl")[-1]["data"])
    voices = json.loads((render / "voices.json").read_text())
    scene_call = next(r for r in trace if r.get("event") == "claude" and r.get("call") == "scene")
    review_call = next(r for r in trace if r.get("event") == "claude" and r.get("call") == "review")
    paths = [cache / "units.jsonl", cache / "briefs.jsonl", render / "voices.json",
             render / "transcript.jsonl", render / "diarization.json"]
    return {"version": 1, "scope": "Original generated-video transcript, exact segmentation and initial calibration",
            "provenance": {"source_cache": str(cache), "files_sha256": {str(p.relative_to(cache)): digest(p) for p in paths},
                           "audio_models": voices["inputs"], "unit_equivalence_checked": True},
            "duration": old["duration"], "settings": old["settings"],
            "registry": {**registry.dump(), "centroids": {}},
            "brief": asdict(brief), "models": {"scene": scene_call["model"], "review": review_call["model"],
                                               "effort": scene_call["effort"], "review_effort": review_call["effort"],
                                               "light_effort": LIGHT_EFFORT},
            "voices": {sid: {"key": v["key"], "calibration": v["calibration"]} for sid, v in voices["speakers"].items()},
            "units": [{"unit": asdict(st.unit), "next_start": st.next_start,
                       "speech_s": st.speech_s, "music": st.music} for st in states]}


def normalize_fixture(data: dict, calibration_path: Path) -> dict:
    """The optional original boundary fixture has text/times only; use the same recorded pace for both arms."""
    if "units" in data:
        return data
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    lines = data["lines"]
    speakers = {line["speaker"] for line in lines}
    turns = [[line["speaker"], line["start"], line["end"]] for line in lines]
    units = [{"unit": asdict(SourceUnit(line["id"], line["speaker"], line["start"], line["end"], line["text"])),
              "next_start": lines[i + 1]["start"] if i + 1 < len(lines) else None,
              "speech_s": line.get("speech_s", line["end"] - line["start"]), "music": False}
             for i, line in enumerate(lines)]
    return {**data, "units": units, "models": calibration["models"],
            "voices": {sid: calibration["voices"][sid] for sid in speakers},
            "registry": {"turns": turns, "exclusive": turns, "stray": [], "centroids": {}},
            "calibration_provenance": {"fixture": str(calibration_path), "sha256": digest(calibration_path),
                                       "note": "Fixed initial calibrated pace; original text has no corresponding audio"}}


def plan(fixture: dict) -> dict:
    out = {}
    with tempfile.TemporaryDirectory(prefix="maata-text-plan-") as tmp:
        for arm, cap in CAPS.items():
            job = TextJob(Path(tmp) / arm, fixture, cap)
            scenes = job._cut_scenes()
            n = len(scenes)
            lanes = [[s.no for s in scenes[i * n // LANES:(i + 1) * n // LANES]] for i in range(LANES)]
            out[arm] = {"cap_seconds": cap[0], "cap_lines": cap[1], "lanes": lanes,
                        "scenes": [{"scene": sc.no, "request": asdict(job._scene_request(sc, sc.lines))} for sc in scenes]}
    return out


async def run_arm(fixture: dict, arm: str, mock: bool, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"maata-text-{arm}-") as tmp:
        job = TextJob(Path(tmp), fixture, CAPS[arm])
        models = fixture["models"]
        factory = ClaudeTranslator
        if mock:
            from maata_engine.backends.mock import MockSceneTranslator
            factory = MockSceneTranslator
        tr = factory(Path(tmp), job.video_id, brief_from(fixture["brief"]), style=job.style, trace=job._trace,
                     model=models["scene"], review_model=models["review"], effort=models["effort"],
                     light_effort=models["light_effort"], review_effort=models["review_effort"],
                     fallback_model=None, review_fallback=None)
        job.tr = tr
        tr.call_priority = job._translation_priority
        record = {"arm": arm, "cap_seconds": CAPS[arm][0], "cap_lines": CAPS[arm][1], "mock": mock,
                  "local_cache": "fresh empty temporary directory", "status": "running", "hygiene_start": hygiene()}
        try:
            if not mock:
                record["cli_health"] = await asyncio.to_thread(tr.cli.health)
                if record["cli_health"].get("problem"):
                    raise RuntimeError(record["cli_health"])
            job.started = time.monotonic()
            await asyncio.wait_for(job._translate(), timeout=timeout)
            job.capture_ready()
            record["translation_stage_seconds"] = time.monotonic() - job.started
            record["status"] = "completed"
        except Exception as exc:
            record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        finally:
            await tr.aclose()
            if job._side_tasks:
                await asyncio.gather(*list(job._side_tasks), return_exceptions=True)
        calls = [r for r in job.events if r.get("event") == "claude"]
        record["hygiene_end"] = hygiene()
        checks = [record["hygiene_start"], record["hygiene_end"]]
        record["valid_hygiene"] = all(r["thermal_state"] in (0, 1) and "AC Power" in r["power"] for r in checks)
        mismatches = [r for r in calls if r.get("model") != models["review" if r["call"] == "review" else "scene"]]
        record.update(first_scene_ready_seconds=min(job.ready.values(), default=None),
                      first_contiguous_ready_seconds=job.ready.get(1),
                      all_text_ready_seconds=max(job.ready.values()) if len(job.ready) == len(job.scenes) and job.ready else None,
                      first_audio_seconds=None, whole_video_finish_seconds=None,
                      ready_by_scene={str(k): v for k, v in job.ready.items()}, contiguous_ready=job.contiguous,
                      scenes=[{"scene": sc.no, "ids": [st.unit.id for st in sc.lines]} for sc in job.scenes],
                      coverage=job._coverage(job._order), skipped=job.skipped,
                      model_mismatch=bool(mismatches) and not mock, calls=len(calls),
                      usage={k: sum(r.get(k, 0) or 0 for r in calls) for k in
                             ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "cache_1h_tokens")},
                      events=job.events, requests=job.requests,
                      outputs=[{**line_json(st.line), "model": st.line.model, "cached": st.line.cached,
                                "flags": list(st.line.flags), "chosen_tier": st.tier,
                                "coverage": coverage_json(st.line.coverage),
                                "selected_coverage": voiced_class(st.line.coverage, st.tier),
                                "aksharas": {t: count_units(w.spoken) for t, w in st.line.tiers.items()},
                                "predicted_seconds": {t: job.estimator.estimate(w.spoken, job._key(st.unit.speaker))
                                                      for t, w in st.line.tiers.items()}}
                               for st in job._order if st.line])
        record["complete_output"] = len(record["outputs"]) == len(fixture["units"]) and not job.skipped
        record["quality_eligible"] = record["complete_output"] and all(
            r["selected_coverage"] in ("C", "m") for r in record["outputs"])
        record["valid_timing"] = (record["status"] == "completed" and not record["model_mismatch"]
                                  and record["quality_eligible"] and record["valid_hygiene"] and not mock)
        record["automated_quality_complete"] = record["complete_output"] and all(
            r["selected_coverage"] == "C" for r in record["outputs"])
        return record


def summary(runs: list[dict]) -> dict:
    result = {}
    for metric in ("first_contiguous_ready_seconds", "all_text_ready_seconds", "translation_stage_seconds"):
        values = {arm: [r[metric] for r in runs if r["arm"] == arm and r["valid_timing"] and r.get(metric) is not None]
                  for arm in CAPS}
        medians = {arm: statistics.median(v) if v else None for arm, v in values.items()}
        def stats(v):
            if not v:
                return {"n": 0, "p50_s": None, "p95_s": None, "max_s": None}
            ordered = sorted(v)
            rank = (len(v) - 1) * .95
            lo = int(rank)
            p95 = ordered[lo] + (ordered[min(lo + 1, len(v) - 1)] - ordered[lo]) * (rank - lo)
            return {"n": len(v), "p50_s": statistics.median(v), "p95_s": p95, "max_s": max(v)}
        enough = all(len(v) >= 5 for v in values.values())
        result[metric] = {"samples": values, "statistics": {arm: stats(v) for arm, v in values.items()},
                          "minimum_five_per_configuration": enough,
                          "candidate_reduction_percent": 100 * (1 - medians["B"] / medians["A"])
                          if enough and medians["A"] and medians["B"] is not None else None}
    return result


async def main(args) -> bool:
    if args.extract_cache:
        save(args.fixture, extract(args.extract_cache))
    fixture = normalize_fixture(json.loads(args.fixture.read_text(encoding="utf-8")), args.calibration_fixture)
    report = {"scope": "Isolated text scene-shape comparison; no first-audio or whole-video speed claim",
              "created_utc": datetime.now(timezone.utc).isoformat(), "fixture": fixture,
              "fixture_sha256": digest(args.fixture), "script_sha256": digest(Path(__file__)),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {p: digest(ROOT / p) for p in ("engine/src/maata_engine/render.py",
                                "engine/src/maata_engine/dubber.py", "engine/src/maata_engine/backends/claude_translator.py",
                                "engine/src/maata_engine/text/scene_prompt.py", "engine/models.lock.json", "engine/uv.lock")},
              "machine": {"platform": platform.platform(), "python": platform.python_version()},
              "prompt_hash": PROMPT_HASH, "prompt_version": SHOTS_VERSION, "review_hash": REVIEW_HASH,
              "lanes": LANES, "plan": plan(fixture), "order": args.order, "runs": [],
              "method": {"brief": "frozen before timing", "duration_calibration": "frozen original three takes per voice",
                         "local_cache": "fresh for every arm", "provider_prompt_cache": "uncontrolled; token usage retained",
                         "audio": "no models loaded; no duration feedback, voiced-wording review or export",
                         "priority": "production callback; cursor stays at first line because there is no voicer",
                         "models": "fixed requested model IDs; fallback disabled; mismatches invalidate timing",
                         "quality": "all requests, selected tiers and automated coverage retained; human review not performed"},
              "status": "dry_run"}
    if platform.system() == "Darwin":
        report["machine"].update(chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                                 memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)))
    save(args.out, report)
    if args.run or args.mock:
        report["status"] = "running"
        for arm in args.order:
            result = await run_arm(fixture, arm, args.mock, args.timeout)
            report["runs"].append(result)
            report["summary"] = summary(report["runs"])
            save(args.out, report)
            print(json.dumps({k: result.get(k) for k in ("arm", "status", "translation_stage_seconds",
                                                       "first_contiguous_ready_seconds", "all_text_ready_seconds", "calls")}))
            if result["status"] != "completed" or result["model_mismatch"]:
                report["status"] = "failed"
                break
        else:
            report["status"] = "mock_verified" if args.mock else "completed"
        save(args.out, report)
    print(json.dumps({"status": report["status"], "output": str(args.out.resolve()),
                      "scenes": {arm: len(p["scenes"]) for arm, p in report["plan"].items()}}))
    return report["status"] != "failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--calibration-fixture", type=Path, default=DEFAULT_FIXTURE,
                        help="Initial voice calibration for a text-only fixture containing lines instead of units")
    parser.add_argument("--extract-cache", type=Path, help="Freeze a completed local render into --fixture; no CLI calls")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Make real sealed Claude CLI calls (default is dry-run)")
    mode.add_argument("--mock", action="store_true", help="Verify orchestration with the production mock, no CLI")
    parser.add_argument("--order", default="AB", help="AB quality/pilot; ABBAABBAAB has five runs per configuration")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-arm timeout, seconds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.order or set(args.order) - set(CAPS):
        parser.error("--order must contain only A and B")
    raise SystemExit(0 if asyncio.run(main(args)) else 1)
