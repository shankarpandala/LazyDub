#!/usr/bin/env python3
"""Three original timed lines through the production adapter, never a real job.

Run synthesize under network denial, then export/roundtrip in fresh engine processes.
Only synthetic video is exported. No translation, source-media ASR, queue operation or app launch occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import omnivoice_smoke as common


def synthesize(out: Path) -> None:
    common.offline()
    sys.path.insert(0, str(common.REPO / "engine/src"))
    import numpy as np
    import soundfile as sf
    import perth
    from maata_engine.backends.omnivoice import OmniVoiceTTS
    from maata_engine.qa.validators import wording
    from maata_engine.timing.planner import LineSlot, PlannerSettings, TimelinePlanner

    if (out / "results.json").exists():
        raise RuntimeError("Refusing an extra synthesis trial in the same directory")
    model = common.PRODUCTION_MODELS / "omnivoice"
    # Match Apple backend wiring, using the production automatic-mode default.
    # No legacy reference metadata is read or explicitly supplied by this runner.
    kwargs = {"model_dir": model, "voice_dir": model / "voices"}
    tts = OmniVoiceTTS(**kwargs)
    identity = tts.cache_identity
    assert identity["mode"] == "automatic" and identity["seed"] == 20261010
    assert identity["seed_policy"] == "fixed-before-every-generation"
    assert identity["voice_policy"] == "automatic-telugu-seed-v1"
    assert identity["reference_audio_sha256"] is None
    if tts.missing():
        raise RuntimeError(f"Adapter assets missing: {tts.missing()}")
    phrase_list = []
    for item in common.PHRASES:
        checked, problems, flags = wording({"spoken": item["text"], "english": []})
        assert checked is not None and not problems
        phrase_list.append({**item, "text": checked.spoken, "seed": 20261010})
    report = {"version": 1, "scope": "Three original Telugu lines through production adapter and timeline planner; component validation only.",
        "phrases": phrase_list, "samples": [], "runs": {"omnivoice-adapter": {"cache_identity": identity}},
        "implementation_sha256": {str(path.relative_to(common.REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (common.REPO / "engine/src/maata_engine/backends/omnivoice.py",
                         common.REPO / "engine/src/maata_engine/backends/_omnivoice_worker.py", Path(__file__).resolve())},
        "limitations": ["No full-video or native-listening acceptance claim.",
            "Parent CPU tests may be concurrent; first/warm times are different sentences and not repeated speed trials.",
            "Automatic mode, no reference/instruction/duration/speed arguments; seed 20261010 reset on every call, 32 steps.",
            "The fixed seed is predeclared, not selected by retries; automatic mode does not guarantee consistent speaker identity.",
            "Original auto B used seeds 20261010/11/12 per sentence; this trial deliberately uses 20261010 for all three.",
            "Planner coverage uses authored source windows and whole waveform spans, not measured source/dub VAD or native listening.",
            "ASR CER is only a diagnostic; it does not prove natural pronunciation or prosody."]}
    settings = PlannerSettings(max_freeze=0.0, freeze_budget=0.0)
    planner = TimelinePlanner(settings)
    slots = [LineSlot(1, "S1", 0.0, 3.5, 4.0, None),
             LineSlot(2, "S2", 4.0, 8.75, 9.5, 3.5),
             LineSlot(3, "S1", 9.5, 13.5, None, 8.75)]
    takes, rendered, pids = [], [], []
    try:
        for index, (phrase, slot) in enumerate(zip(phrase_list, slots)):
            prepared = time.perf_counter()
            voice = tts.preset_voice(slot.speaker)
            prepare_s = time.perf_counter() - prepared
            started = time.perf_counter()
            # A deliberately tight advisory cap on line2 verifies that completed
            # natural speech is not cropped or mislabeled a token-cap failure.
            cap = 2.0 if index == 1 else 15.0
            take = tts.synthesize_mel(phrase["text"], voice, "te", cap)
            generation_s = time.perf_counter() - started
            pid = tts._process.pid
            pids.append(pid)
            plan = planner.place(slot, take.seconds)
            started = time.perf_counter()
            audio = tts.vocode(take, plan.rate)
            render_s = time.perf_counter() - started
            raw_path, marked_path = out / f"adapter-{phrase['id']}-raw.wav", out / f"adapter-{phrase['id']}-marked.wav"
            sf.write(raw_path, take.samples, tts.sample_rate, subtype="FLOAT")
            sf.write(marked_path, audio, tts.sample_rate, subtype="FLOAT")
            packed = out / f"adapter-{phrase['id']}-take.npz"
            np.savez_compressed(packed, **tts.pack_take(take))
            sample = {**phrase, "model": "omnivoice-adapter", "sample_rate": tts.sample_rate,
                "raw_path": str(raw_path), "marked_path": str(marked_path), "path": str(marked_path),
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "packed_take": str(packed),
                "prepare_seconds": prepare_s, "generation_seconds": generation_s, "render_and_watermark_seconds": render_s,
                "duration_seconds": len(audio) / tts.sample_rate, "natural_duration_seconds": take.seconds,
                "advisory_max_seconds": cap, "capped": take.capped, "over_budget": take.over_budget,
                "rtf": generation_s / take.seconds, "call": "first" if index == 0 else "warm (different sentence)",
                "worker_pid": pid, "plan": asdict(plan), "slot": asdict(slot),
                "end_error_seconds": plan.end - slot.end, "early_end_seconds": max(0.0, slot.end - plan.end),
                "waveform": common.stats(audio, tts.sample_rate)}
            assert not take.capped, "Completed natural speech must not be falsely classified as a cutoff"
            assert take.over_budget == (take.seconds >= cap)
            report["samples"].append(sample)
            takes.append(take)
            rendered.append(audio)
            common.save(out, report)
            print(json.dumps({k: sample[k] for k in ("id", "generation_seconds", "natural_duration_seconds", "capped")}), flush=True)
    finally:
        tts.release()
    report["persistent_worker_reused"] = len(set(pids)) == 1
    report["planner_stats_whole_waveform_proxy"] = planner.stats()
    assert report["persistent_worker_reused"]

    # Resume the saved natural takes without spawning OmniVoice or synthesizing
    # again; rate1 and planned-rate output must match exactly after re-watermark.
    restored = OmniVoiceTTS(**kwargs)
    resume = []
    for sample, previous in zip(report["samples"], rendered):
        with np.load(sample["packed_take"], allow_pickle=False) as arrays:
            take = restored.unpack_take(dict(arrays))
        audio = restored.vocode(take, sample["plan"]["rate"])
        resume.append(bool(np.array_equal(audio, previous)))
    report["saved_take_resume_pcm_equal"] = resume
    report["resume_spawned_worker"] = restored._process is not None
    restored.release()
    assert all(resume) and not report["resume_spawned_worker"]

    # Price a tighter source window using an existing take only. No new text or
    # synthesis; report the resulting timing pressure rather than cut speech.
    control = TimelinePlanner(settings)
    tight = LineSlot(99, "S1", 0.0, 0.8, 0.9, None)
    report["tight_window_control"] = asdict(control.place(tight, takes[1].seconds))
    assert report["tight_window_control"]["overdraft"] > 0
    sr = tts.sample_rate
    final_length = max(round(s["plan"]["start"] * sr) + len(a) for s, a in zip(report["samples"], rendered)) + sr // 4
    scene = np.zeros(final_length, np.float32)
    for sample, audio in zip(report["samples"], rendered):
        start = round(sample["plan"]["start"] * sr)
        scene[start:start + len(audio)] += audio
    path = out / "adapter-timed-scene.wav"
    sf.write(path, scene, sr, subtype="FLOAT")
    report["timed_scene_path"] = str(path)
    detector = perth.PerthImplicitWatermarker()
    for sample, audio in zip(report["samples"], rendered):
        probability = float(detector.get_watermark(audio, sample_rate=sr, round=False))
        sample["watermark"] = {"marked_probability": probability, "threshold": 0.5, "detected": probability >= 0.5}
        assert probability >= 0.5
    report["runs"]["omnivoice-adapter"].update(status="complete", memory=common.memory())
    common.save(out, report)


def export_check(out: Path) -> None:
    """Only generated PCM and synthetic video, through the production PyAV/FFmpeg AAC export."""
    common.offline()
    sys.path.insert(0, str(common.REPO / "engine/src"))
    import av
    import numpy as np
    import perth
    import soundfile as sf
    from maata_engine import export as mp4, mix
    from maata_engine.resolve import synth_video
    from maata_engine.subtitles import Cue
    from maata_engine.timing import pauses
    from maata_engine.timing.metrics import line_metrics, speech_metrics

    report = common.load_report(out)
    video, dest = out / "original-synthetic-video.mp4", out / "adapter-timed-export.mp4"
    duration = max(s["plan"]["start"] + s["duration_seconds"] for s in report["samples"]) + 0.5
    synth_video(video, duration, audio=False)
    lines, cues, metric_rows = [], [], []
    for sample in report["samples"]:
        audio, sr = sf.read(sample["marked_path"], dtype="float32")
        assert sr == 24000
        pcm = out / f"adapter-{sample['id']}-export-pcm.npy"
        np.save(pcm, audio.astype(np.float16))
        start = sample["plan"]["start"]
        spoken = pauses.voiced(pauses.pauses(audio, sr), len(audio) / sr)
        metric = {"start": sample["slot"]["start"], "end": sample["slot"]["end"],
            "speech": [[sample["slot"]["start"], sample["slot"]["end"]]],
            "voiced": [[start + a, start + b] for a, b in spoken]}
        metric_rows.append(metric)
        sample["energy_voiced_vs_authored_window"] = {**line_metrics(metric), "voiced": metric["voiced"]}
        lines.append(mix.Line(start, sample["slot"]["speaker"], pcm, len(audio)))
        cues.append(Cue(start, start + len(audio) / sr, sample["slot"]["speaker"], sample["text"]))
    started = time.perf_counter()
    result = mp4.export(dest, video=video, lines=lines, voice_sr=24000, telugu=cues, english=[],
        audio_start=0.0, stop=None, metadata={"title": "Original OmniVoice adapter validation",
            "comment": "Three original synthetic Telugu utterances; AI voices. No source media."})
    elapsed = time.perf_counter() - started
    parts = []
    with av.open(str(dest)) as container:
        stream = container.streams.audio[0]
        codec, encoded_sr = stream.codec_context.name, stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        for frame in container.decode(stream):
            parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    decoded = np.concatenate(parts).astype(np.float32)
    decoded_path = out / "adapter-timed-export-decoded.wav"
    sf.write(decoded_path, decoded, 24000, subtype="FLOAT")
    detector = perth.PerthImplicitWatermarker()
    probabilities = []
    for sample in report["samples"]:
        a = round(sample["plan"]["start"] * 24000)
        b = a + round(sample["duration_seconds"] * 24000)
        probability = float(detector.get_watermark(decoded[a:b], sample_rate=24000, round=False))
        probabilities.append({"id": sample["id"], "probability": probability, "detected": probability >= 0.5})
    report["export"] = {"path": str(dest), "decoded_path": str(decoded_path), "seconds": elapsed,
        "path_under_test": "production export.export: float16 PCM, mastering, PyAV/FFmpeg AAC, final decoded loudness",
        "encoder": mp4.aac(), "codec": codec, "encoded_sample_rate": encoded_sr,
        "result": result, "watermarks_by_line": probabilities,
        "all_watermarks_detected": all(x["detected"] for x in probabilities),
        "true_peak_passed": result["loudness"]["TP"] is not None and result["loudness"]["TP"] <= mix.TP_CEIL}
    report["energy_voiced_vs_authored_window"] = {"scope": "Production energy-pause detector versus authored fixed windows; no recorded source speech exists, so not full dubbing acceptance.",
        "metrics": speech_metrics(metric_rows)}
    common.save(out, report)
    print(json.dumps(report["export"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["synthesize", "roundtrip", "export"])
    parser.add_argument("--out", type=Path, default=common.ROOT / "audio/adapter-auto")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "synthesize":
        synthesize(args.out)
    elif args.stage == "roundtrip":
        common.roundtrip(args.out)
    else:
        export_check(args.out)


if __name__ == "__main__":
    main()
