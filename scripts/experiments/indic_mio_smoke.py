#!/usr/bin/env python3
"""Bounded original Telugu Indic-Mio audition. Run under sandbox-exec network deny."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import omnivoice_smoke as common

LOCK = Path(__file__).with_name("indic_mio") / "models.lock.json"


def synthesize(out: Path, reference: dict) -> None:
    common.offline()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from miocodec import MioCodecModel, load_audio

    lock = json.loads(LOCK.read_text())["models"]
    for model in lock:
        common.verify_model(model, common.ROOT / "models")
    if hashlib.sha256(Path(reference["wav_path"]).read_bytes()).hexdigest() != reference["wav_sha256"]:
        raise RuntimeError("Reference checksum mismatch")
    report = common.load_report(out)
    if report["samples"]:
        raise RuntimeError("Refusing extra synthesis into an existing trial")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    # The upstream constructor downloads WavLM redundantly before replacing it
    # with the codec checkpoint. Build exactly its architecture instead, then
    # require every codec/SSL weight to load strictly. No random SSL weights may
    # survive and there is no automatic download path.
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    assert bundle._model_type == "WavLM" and bundle._normalize_waveform is False
    original_get_model = bundle.get_model
    bundle.get_model = lambda **kwargs: torchaudio.models.wavlm_model(**bundle._params).eval()
    started = time.perf_counter()
    codec_dir = common.ROOT / "models/miocodec-24k"
    try:
        codec = MioCodecModel.from_hparams(str(codec_dir / "config.yaml"))
    finally:
        bundle.get_model = original_get_model
    state = load_file(str(codec_dir / "model.safetensors"), device="cpu")
    loaded = codec.load_state_dict(state, strict=True)
    del state
    assert not loaded.missing_keys and not loaded.unexpected_keys
    codec.eval().to("cpu")
    codec_load_s = time.perf_counter() - started
    sr = int(codec.config.sample_rate)
    assert sr == 24000 and codec.config.use_wave_decoder
    started = time.perf_counter()
    reference_audio = load_audio(reference["wav_path"], sample_rate=sr)
    with torch.inference_mode():
        embedding = codec.encode(reference_audio, return_content=False, return_global=True).global_embedding
    if embedding is None or not torch.isfinite(embedding).all():
        raise RuntimeError("Invalid actual reference speaker embedding")
    reference_encode_s = time.perf_counter() - started

    model_dir = common.ROOT / "models/indic-mio"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("mps")
    torch.mps.synchronize()
    text_load_s = time.perf_counter() - started
    offset = tokenizer.convert_tokens_to_ids("<|s_0|>")
    assert offset == 151669 and tokenizer.convert_tokens_to_ids("<|s_12799|>") == offset + 12799
    report["runs"]["indic-mio"] = {"device": "mps", "codec_device": "cpu", "dtype": "bfloat16",
        "codec_dtype": "float32", "sample_rate": sr, "python": sys.version,
        "versions": common.versions(["torch", "torchaudio", "transformers", "miocodec"]),
        "models": [{k: m[k] for k in ("id", "repo", "revision")} for m in lock],
        "codec_source_commit": "77473544375d57e96cbdfd5d7d257e8f280fa8e3",
        "mode": "Actual FLEURS Telugu reference global embedding, not a stock or invented embedding",
        "codec_strict_state_dict": True, "codec_load_seconds": codec_load_s,
        "reference_encode_seconds": reference_encode_s, "text_model_load_seconds": text_load_s,
        "load_seconds": codec_load_s + text_load_s, "reference_duration_seconds": len(reference_audio) / sr,
        "reference_manifest": str(common.ROOT / "references/reference.json"),
        "reference_wav_sha256": reference["wav_sha256"],
        "generation": {"do_sample": True, "temperature": 0.9, "top_p": 0.9,
                       "max_new_tokens": 1024, "use_cache": True},
        "generation_timing_scope": "MPS text token generation plus CPU codec decode; excludes PerTh",
        "concurrency": "Parent CPU test suite may be active; no paired speed claim", "status": "generating",
        "after_load_memory": common.memory(torch)}
    report["limitations"] = ["No native listening judgment has been made.",
        "One take per original phrase. Parent CPU tests may affect timing.",
        "FLEURS establishes Telugu language/text provenance, not native-listener certification.",
        "Dataset overlap with training is unknown; this is an audition, not an accuracy benchmark.",
        "Codec uses upstream CPU SDPA, not the recommended CUDA FlashAttention path.",
        "Whisper CER and waveform tails cannot establish pronunciation/naturalness or exact word-ending completeness."]
    common.save(out, report)
    for index, phrase in enumerate(common.PHRASES):
        seed = 20261010 + index
        torch.manual_seed(seed)
        np.random.seed(seed)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": phrase["text"]}],
                                               tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                                    temperature=0.9, top_p=0.9, use_cache=True)
        torch.mps.synchronize()
        token_s = time.perf_counter() - started
        generated = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
        codes = [token - offset for token in generated if offset <= token < offset + 12800]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if not codes or len(generated) >= 1024 or not any(token in eos for token in generated):
            raise RuntimeError("Empty or capped/non-terminated speech token generation")
        started_decode = time.perf_counter()
        with torch.inference_mode():
            audio = codec.decode(global_embedding=embedding,
                                 content_token_indices=torch.tensor(codes, dtype=torch.long)).cpu().numpy()
        decode_s = time.perf_counter() - started_decode
        audio = np.asarray(audio, np.float32).reshape(-1)
        elapsed = time.perf_counter() - started
        path = out / f"indic-mio-{phrase['id']}-raw.wav"
        sf.write(path, audio, sr, subtype="FLOAT")
        sample = {**phrase, "model": "indic-mio", "raw_path": str(path), "marked_path": None,
            "path": None, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed, "sample_rate": sr, "generation_seconds": elapsed, "text_generation_seconds": token_s,
            "codec_decode_seconds": decode_s, "duration_seconds": len(audio) / sr,
            "rtf": elapsed / (len(audio) / sr), "generated_tokens": len(generated), "speech_tokens": len(codes),
            "terminated": True, "call": "first" if index == 0 else "warm (different sentence)",
            "waveform": common.stats(audio, sr), "memory": common.memory(torch)}
        report["samples"].append(sample)
        common.save(out, report)
        print(json.dumps({k: sample[k] for k in ("id", "generation_seconds", "duration_seconds", "rtf")}), flush=True)
    report["runs"]["indic-mio"].update(status="complete", final_memory=common.memory(torch))
    common.save(out, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "synthesize", "watermark", "roundtrip"])
    parser.add_argument("--out", type=Path, default=common.ROOT / "audio/indic-mio-native")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(common.REPO / "engine/src"))
    from maata_engine.qa.validators import wording
    repaired = []
    for phrase in common.PHRASES:
        word, problems, flags = wording({"spoken": phrase["text"], "english": []})
        if word is None or problems:
            raise RuntimeError("Invalid original normalized fixture")
        repaired.append({**phrase, "text": word.spoken})
    common.PHRASES[:] = repaired
    common.TRIAL_PROVENANCE = {"normalizer": "maata_engine.qa.validators.wording", "label": "C"}
    reference = json.loads((common.ROOT / "references/reference.json").read_text())
    if args.stage == "fetch":
        from maata_engine.models import fetch_model
        for model in json.loads(LOCK.read_text())["models"]:
            fetch_model(model, common.ROOT / "models")
    elif args.stage == "synthesize":
        synthesize(args.out, reference)
    elif args.stage == "watermark":
        common.watermark(args.out)
    else:
        common.roundtrip(args.out)


if __name__ == "__main__":
    main()
