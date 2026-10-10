#!/usr/bin/env python3
"""Isolated, original-text OmniVoice audition; never used by the production engine.

Run each synthesis/QA stage in a fresh process under sandbox-exec network deny.
The fetch stage alone uses Maata's pinned, hash-checking model downloader.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import unicodedata

REPO = Path(__file__).resolve().parents[2]
ROOT = Path.home() / "Library/Caches/Maata/model-evaluation-2026-10-10"
LOCK = Path(__file__).with_name("omnivoice") / "models.lock.json"
PRODUCTION_MODELS = Path.home() / "Library/Application Support/Maata/Models"
PHRASES = [
    {"id": "conversation", "text": "నమస్కారం! మీరు ఎలా ఉన్నారు? నేను బాగున్నాను."},
    {"id": "names_numbers", "text": "రవి రేపు ఉదయం తొమ్మిది గంటలకు వాట్సాప్‌లో రెండు ఫైళ్లు పంపుతాడు."},
    {"id": "expressive", "text": "నిజంగానా? ఇంత మంచి వార్త నాకు ఇప్పుడే చెబుతున్నారా!"},
]
TRIAL_PROVENANCE = None


def offline() -> None:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(ROOT / "hf-offline"))
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


def load_report(out: Path) -> dict:
    path = out / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "scope": f"{len(PHRASES)} original Telugu audition phrase(s); no downloaded media.",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(),
            "phrases": PHRASES, "trial_provenance": TRIAL_PROVENANCE, "runs": {}, "samples": [],
            "limitations": ["No native listening judgment has been made.",
                *( ["Pinned FLEURS Telugu reference establishes language/text provenance, not native-listener certification.",
                    "Dataset overlap with training is unknown; this is an audition, not an accuracy benchmark."]
                   if TRIAL_PROVENANCE and TRIAL_PROVENANCE.get("reference_manifest")
                   else ["OmniVoice auto voice has no reference; native accent is unverified.",
                         "Chatterbox uses builtin conds.pt; native reference provenance is unknown."]),
                "Each phrase has one seed/take per model; these are not repeated speed trials.",
                *(["The original names_numbers fixture contains U+200C in వాట్సాప్‌లో; both models receive it unchanged. Production normalizes/rejects joiners, so that phrase is not an exact production-text comparison."]
                  if any("\u200c" in p["text"] for p in PHRASES) else []),
                "Whisper roundtrip errors are diagnostic, not proof of pronunciation or naturalness."]}


def save(out: Path, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(out / "results.json")


def memory(torch=None) -> dict:
    result = {"process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if torch is not None and torch.backends.mps.is_available():
        result.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


def stats(audio, sr: int) -> dict:
    import numpy as np
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(x) or not np.isfinite(x).all() or float(np.max(np.abs(x))) < 1e-6:
        raise RuntimeError("Empty, nonfinite or numerically silent generated waveform")
    active = np.flatnonzero(np.abs(x) > max(1e-4, float(np.max(np.abs(x))) * 0.01))
    tail = x[-round(sr * 0.1):]
    return {"frames": len(x), "peak": float(np.max(np.abs(x))),
            "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
            "nonfinite": int((~np.isfinite(x)).sum()),
            "last_100ms_rms": float(np.sqrt(np.mean(tail.astype(np.float64) ** 2))),
            "trailing_below_1pct_peak_s": float((len(x) - 1 - active[-1]) / sr) if len(active) else len(x) / sr,
            "tail_check_limit": "Waveform energy cannot establish word-ending completeness."}


def verify_model(model: dict, root: Path) -> None:
    sys.path.insert(0, str(REPO / "engine/src"))
    from maata_engine.models import _spec, verify
    bad = [f["path"] for f in model["files"] if not verify(root / model["id"] / f["path"], _spec(f))]
    if bad:
        raise RuntimeError(f"Pinned model files missing or invalid: {bad}")


def versions(names: list[str]) -> dict:
    return {name: importlib.metadata.version(name) for name in names}


def fetch() -> None:
    sys.path.insert(0, str(REPO / "engine/src"))
    from maata_engine.models import fetch_model
    for model in json.loads(LOCK.read_text())["models"]:
        fetch_model(model, ROOT / "models")
        verify_model(model, ROOT / "models")


def synthesize(out: Path, kind: str) -> None:
    offline()
    import numpy as np
    import soundfile as sf
    import torch
    report = load_report(out)
    if any(s["model"] == kind for s in report["samples"]):
        raise RuntimeError("Refusing accidental extra generation; use a new output directory for a new trial.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This evaluation requires Apple MPS")
    identity = {"device": "mps", "python": sys.version,
                "versions": versions(["torch", "torchaudio", "transformers"]),
                "load_timing_scope": "model construction plus MPS synchronization; excludes imports, process startup and hash verification; local file cache warmed by verification",
                "generation_timing_scope": ("synthesis plus existing PerTh" if kind == "chatterbox"
                                             else "synthesis only; PerTh applied in separate QA stage")}
    decoder_times = []
    before_watermark = []
    if kind == "omnivoice":
        model_info = json.loads(LOCK.read_text())["models"][0]
        verify_model(model_info, ROOT / "models")
        import omnivoice.models.omnivoice as impl
        def only_local(value):
            if not Path(value).is_dir():
                raise RuntimeError("Automatic remote model resolution is disabled")
            return str(Path(value).resolve())
        def no_asr(*args, **kwargs):
            raise RuntimeError("OmniVoice automatic ASR is disabled")
        impl._resolve_model_path = only_local
        impl.OmniVoice.load_asr_model = no_asr
        started = time.perf_counter()
        model = impl.OmniVoice.from_pretrained(
            ROOT / "models/omnivoice", device_map="mps", dtype=torch.float16,
            local_files_only=True, trust_remote_code=False, load_asr=False)
        torch.mps.synchronize()
        load_s = time.perf_counter() - started
        sr = int(model.sampling_rate)
        decode = model.audio_tokenizer.decode
        def timed_decode(*args, **kwargs):
            start = time.perf_counter()
            result = decode(*args, **kwargs)
            decoder_times.append(time.perf_counter() - start)
            return result
        model.audio_tokenizer.decode = timed_decode
        reference_kwargs = {}
        reference_identity = {}
        if TRIAL_PROVENANCE and TRIAL_PROVENANCE.get("reference_manifest"):
            ref_manifest = Path(TRIAL_PROVENANCE["reference_manifest"])
            reference = json.loads(ref_manifest.read_text())
            if hashlib.sha256(Path(reference["wav_path"]).read_bytes()).hexdigest() != reference["wav_sha256"]:
                raise RuntimeError("Reference checksum mismatch")
            if not reference.get("reference_transcript"):
                raise RuntimeError("Exact supplied reference transcript is required; no automatic ASR")
            started = time.perf_counter()
            prompt = model.create_voice_clone_prompt(ref_audio=reference["wav_path"],
                ref_text=reference["reference_transcript"], preprocess_prompt=False)
            reference_kwargs["voice_clone_prompt"] = prompt
            reference_identity = {"mode": "Pinned human FLEURS Telugu reference and exact dataset transcript",
                "reference_encode_seconds": time.perf_counter() - started,
                "reference_manifest": str(ref_manifest), "reference_wav_sha256": reference["wav_sha256"],
                "reference_preprocess_prompt": False,
                "reference_note": "Full 10.98s reference; upstream codec rounds to its frame boundary. No arbitrary transcript/audio trimming or ASR."}
        generate = lambda text: model.generate(text=text, language="te", num_step=32,
                                               normalize_text=False, **reference_kwargs)[0]
        identity.update(repo=model_info["repo"], revision=model_info["revision"],
                        source_commit="08be0b4ccbac3e13e374e86fbfead4b4cac343e2",
                        mode="auto (no reference or voice-design instruction)", steps=32,
                        codec_device=str(model.audio_tokenizer.device), dtype="float16",
                        duration="estimated by upstream", postprocess="upstream defaults")
        identity.update(reference_identity)
    else:
        sys.path.insert(0, str(REPO / "engine/src"))
        from maata_engine.backends.torch_common import ChatterboxTeluguTTS
        prod_lock = json.loads((REPO / "engine/models.lock.json").read_text())
        model_info = next(m for m in prod_lock["models"] if m["id"] == "chatterbox-telugu")
        verify_model(model_info, PRODUCTION_MODELS)
        started = time.perf_counter()
        model = ChatterboxTeluguTTS(PRODUCTION_MODELS / "chatterbox-telugu", "mps", presets_dir=None)
        voice = model.preset_voice("builtin")
        torch.mps.synchronize()
        load_s = time.perf_counter() - started
        sr = int(model.sample_rate)
        apply = model.model.watermarker.apply_watermark
        def capture_raw(audio, *args, **kwargs):
            before_watermark.append(np.asarray(audio, np.float32).copy())
            return apply(audio, *args, **kwargs)
        model.model.watermarker.apply_watermark = capture_raw
        generate = lambda text: model.synthesize(text, voice, language="te")
        identity.update(repo=model_info["repo"], revision=model_info["revision"],
                        mode="builtin conds.pt; native reference provenance unknown",
                        cfm_steps=10, t3_dtype="bfloat16", cfg_weight=0.5, exaggeration=0.5)
    report["runs"][kind] = {**identity, "load_seconds": load_s, "after_load_memory": memory(torch),
                             "sample_rate": sr, "status": "generating"}
    save(out, report)
    for index, phrase in enumerate(PHRASES):
        seed = phrase.get("seed", 20261010 + index)
        torch.manual_seed(seed)
        np.random.seed(seed)
        decoder_times.clear()
        before_watermark.clear()
        torch.mps.synchronize()
        started = time.perf_counter()
        audio = np.asarray(generate(phrase["text"]), np.float32).reshape(-1)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        raw = before_watermark[-1] if before_watermark else audio
        raw_path = out / f"{kind}-{phrase['id']}-raw.wav"
        sf.write(raw_path, raw, sr, subtype="FLOAT")
        marked_path = None
        if kind == "chatterbox":
            marked_path = out / f"{kind}-{phrase['id']}-marked.wav"
            sf.write(marked_path, audio, sr, subtype="FLOAT")
        sample = {**phrase, "model": kind, "raw_path": str(raw_path),
                  "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  "marked_path": str(marked_path) if marked_path else None,
                  "path": str(marked_path) if marked_path else None,
                  "sample_rate": sr, "generation_seconds": elapsed,
                  "duration_seconds": len(audio) / sr, "rtf": elapsed / (len(audio) / sr),
                  "call": "first" if index == 0 else "warm (different sentence)",
                  "seed": seed, "codec_decode_seconds": sum(decoder_times) if decoder_times else None,
                  "waveform": stats(audio, sr), "memory": memory(torch)}
        report["samples"].append(sample)
        save(out, report)
        print(json.dumps({k: sample[k] for k in ("model", "id", "generation_seconds", "duration_seconds", "rtf")}), flush=True)
    report["runs"][kind].update(status="complete", final_memory=memory(torch))
    save(out, report)


def watermark(out: Path) -> None:
    offline()
    import numpy as np
    import soundfile as sf
    import perth
    det = perth.PerthImplicitWatermarker()
    report = load_report(out)
    for sample in report["samples"]:
        if "watermark" in sample:
            continue
        raw, sr = sf.read(sample["raw_path"], dtype="float32")
        if sample["marked_path"]:
            marked, check_sr = sf.read(sample["marked_path"], dtype="float32")
            assert check_sr == sr
            action = "preserved production Chatterbox PerTh"
        else:
            started = time.perf_counter()
            marked = np.asarray(det.apply_watermark(raw, sample_rate=sr), np.float32)
            sample["watermark_apply_seconds"] = time.perf_counter() - started
            sample["marked_path"] = str(out / f"{sample['model']}-{sample['id']}-marked.wav")
            sf.write(sample["marked_path"], marked, sr, subtype="FLOAT")
            action = f"applied existing local PerTh to {sample['model']} output"
        sample["path"] = sample["marked_path"]
        sample["watermark"] = {"action": action, "package": importlib.metadata.version("resemble-perth"),
            "raw_probability": float(det.get_watermark(raw, sample_rate=sr, round=False)),
            "marked_probability": float(det.get_watermark(marked, sample_rate=sr, round=False)),
            "threshold": 0.5, "marked_sha256": hashlib.sha256(Path(sample["marked_path"]).read_bytes()).hexdigest()}
        sample["watermark"]["detected"] = sample["watermark"]["marked_probability"] >= 0.5
        sample["marked_waveform"] = stats(marked, sr)
        save(out, report)


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text) if c.isalnum() or unicodedata.category(c).startswith("M"))


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def roundtrip(out: Path) -> None:
    offline()
    import mlx_whisper
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    model_info = next(m for m in json.loads((REPO / "engine/models.lock.json").read_text())["models"]
                      if m["id"] == "whisper-large-v3-turbo-mlx")
    verify_model(model_info, PRODUCTION_MODELS)
    report = load_report(out)
    for sample in report["samples"]:
        if "roundtrip" in sample:
            continue
        audio, sr = sf.read(sample["path"], dtype="float32")
        divisor = gcd(sr, 16000)
        audio = np.asarray(resample_poly(audio, 16000 // divisor, sr // divisor), np.float32)
        started = time.perf_counter()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=str(PRODUCTION_MODELS / model_info["id"]),
            language="te", word_timestamps=True, condition_on_previous_text=False, verbose=None,
            temperature=0.0)
        result_path = out / f"{sample['model']}-{sample['id']}-asr.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        expected, actual = normalized(sample["text"]), normalized(result["text"])
        sample["roundtrip"] = {"model": model_info["repo"], "revision": model_info["revision"],
            "forced_language": "te", "text": result["text"], "raw_result_path": str(result_path),
            "seconds": time.perf_counter() - started, "normalized_character_errors": edit_distance(expected, actual),
            "expected_characters": len(expected), "cer": edit_distance(expected, actual) / max(1, len(expected)),
            "normalization": "NFC, remove whitespace and punctuation; retain letters/numbers/combining marks",
            "memory": memory()}
        save(out, report)
        print(json.dumps({"stage": "roundtrip", "model": sample["model"], "id": sample["id"],
                          "cer": sample["roundtrip"]["cer"]}), flush=True)


def main() -> None:
    global TRIAL_PROVENANCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "omnivoice", "chatterbox", "watermark", "roundtrip"])
    parser.add_argument("--out", type=Path, default=ROOT / "audio")
    parser.add_argument("--fixture", type=Path, help="Explicitly authorized original-only bounded fixture JSON")
    args = parser.parse_args()
    if args.fixture:
        fixture = json.loads(args.fixture.read_text())
        phrases = fixture["phrases"]
        if not 1 <= len(phrases) <= 3 or any(not p.get("id") or not p.get("text") for p in phrases):
            raise ValueError("Fixture must contain one to three named original phrases")
        PHRASES[:] = phrases
        TRIAL_PROVENANCE = fixture.get("provenance")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "fetch":
        fetch()
    elif args.stage in ("omnivoice", "chatterbox"):
        synthesize(args.out, args.stage)
    elif args.stage == "watermark":
        watermark(args.out)
    else:
        roundtrip(args.out)


if __name__ == "__main__":
    main()
