"""Paired, offline M5 Pro component benchmarks; these are not whole-render speed or quality measurements.

From the repository root (the script re-executes itself under sandbox-exec, denying outbound network):

    engine/.venv/bin/python scripts/bench_inference_components.py --component separator
    engine/.venv/bin/python scripts/bench_inference_components.py --component t3-buffer

The separator compares the current implementation with the pre-optimization Git revision below, on the same pinned
local weights and deterministic synthetic input. The optional T3 buffer is an experiment, NOT production code: it
passed token equivalence but did not improve the first M5 Pro timings, so it was not retained. No audio is saved and
no downloads are attempted. Two warmup pairs precede five measured pairs by default; the order alternates per pair.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "23503cbece824be13383af67297f75028daec0e4"
SOURCES = ("engine/src/maata_engine/separation/mel_roformer.py", "engine/src/maata_engine/backends/torch_common.py")
NETWORK_POLICY = "(version 1)(allow default)(deny network-outbound)"


def thermal_state() -> int:
    """NSProcessInfo: 0 nominal, 1 fair, 2 serious, 3 critical; no additional dependency."""
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes, objc.objc_getClass.restype = [ctypes.c_char_p], ctypes.c_void_p
    objc.sel_registerName.argtypes, objc.sel_registerName.restype = [ctypes.c_char_p], ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.objc_msgSend.restype = ctypes.c_void_p
    instance = objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    objc.objc_msgSend.restype = ctypes.c_long
    return int(objc.objc_msgSend(instance, objc.sel_registerName(b"thermalState")))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def method_from_revision(revision: str, path: str, cls: str, name: str, namespace: dict):
    source = git("show", f"{revision}:{path}")
    tree = ast.parse(source)
    typ = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fun = next(n for n in typ.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return function_from_ast(fun, namespace), ast.unparse(fun)


def function_from_ast(fun: ast.FunctionDef, namespace: dict):
    scope = dict(namespace)
    exec(compile(ast.Module(body=[fun], type_ignores=[]), "<benchmark-reference>", "exec"), scope)
    return scope[fun.name]


def summarize(values: list[float]) -> dict:
    import numpy as np

    return {"n": len(values), "seconds": values, "p50_s": statistics.median(values),
            "p95_s": float(np.percentile(values, 95)), "max_s": max(values)}


def separator(args, out: dict) -> None:
    import mlx.core as mx
    import numpy as np
    from maata_engine.separation import mel_roformer as mr

    old, source = method_from_revision(args.baseline, SOURCES[0], "BandSplit", "merge", vars(mr))
    new = mr.BandSplit.merge
    folder = args.models / "mel-roformer-kim-vocal-2-mlx"
    model_info = {"config": json.loads((folder / "config.json").read_text()),
                  "weights_sha256": digest(folder / "model.safetensors")}
    t0 = time.perf_counter()
    model = mr.MelRoFormer.from_pretrained(folder, mx.bfloat16)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    rng = np.random.default_rng(47)
    t = np.arange(model.config.chunk_size, dtype=np.float32) / model.config.sample_rate
    base = (0.07 * np.sin(2 * np.pi * 177 * t) + 0.03 * np.sin(2 * np.pi * 440 * t)
            + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    mixture = np.stack([np.stack([base, base * .83]), np.stack([base * .67, base[::-1]])])
    times, peaks, errors, checks, thermals = {"old": [], "new": []}, {"old": [], "new": []}, [], [], []
    limit = mx.set_cache_limit(1 << 30)
    try:
        for run in range(args.warmup + args.runs):
            outputs = {}
            order = (("old", old), ("new", new)) if run % 2 == 0 else (("new", new), ("old", old))
            for name, fn in order:
                mr.BandSplit.merge = fn
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                y = model(mx.array(mixture))
                mx.eval(y)
                outputs[name] = np.array(y, np.float32)
                peak = mx.get_peak_memory()
                del y
                mx.clear_cache()
                elapsed = time.perf_counter() - t0
                if run >= args.warmup:
                    times[name].append(elapsed)
                    peaks[name].append(peak)
                print(f"separator pair {run}: {name} {elapsed:.4f}s", flush=True)
            errors.append(float(np.max(np.abs(outputs["old"] - outputs["new"]))))
            checks.append(bool(np.allclose(outputs["old"], outputs["new"], rtol=1e-5, atol=1e-7)))
            thermals.append(thermal_state())
        out["separator"] = {"timing": {k: summarize(v) for k, v in times.items()}, "mlx_peak_bytes": peaks,
                            "max_abs_errors": errors, "numerical_checks": checks,
                            "tolerance": {"rtol": 1e-5, "atol": 1e-7}, "thermal_states": thermals, "model": model_info,
                            "load_seconds": load_s, "baseline_method": source,
                            "input": {"shape": list(mixture.shape), "seed": 47,
                                      "sha256": hashlib.sha256(mixture.tobytes()).hexdigest()},
                            "note": "Real-weight forward, batch=2, bf16, 8s chunks; no block overlap-add/media I/O. "
                                    "Tiny full-wave differences may arise from iSTFT scatter reduction order; "
                                    "the band-mask merge has separate bitwise equivalence tests."}
    finally:
        mr.BandSplit.merge = new
        del model
        mx.set_cache_limit(limit)
        gc.collect()
        mx.clear_cache()


def t3_buffer(args, out: dict) -> None:
    import numpy as np
    import torch
    from maata_engine.backends import torch_common as tc

    old, source = method_from_revision(args.baseline, SOURCES[1], "ChatterboxTeluguTTS", "_t3_tokens", vars(tc))
    edits = {
        "generated = bos.expand(n, 1).clone()": "history = torch.empty((n, max_new_tokens + 1), dtype=bos.dtype, "
        "device=bos.device)\n    history[:, :1] = bos\n    generated = history[:, :1]",
        "generated = torch.cat([generated, nxt], dim=1)": "history[:, i + 1:i + 2] = nxt\n        "
        "generated = history[:, :i + 2]",
    }
    candidate = source
    for before, after in edits.items():
        if candidate.count(before) != 1:
            raise ValueError("The baseline decoder changed; review the token-buffer experiment before rerunning it")
        candidate = candidate.replace(before, after)
    new = function_from_ast(ast.parse(candidate).body[0], vars(tc))
    folder = args.models / "chatterbox-telugu"
    model_info = {"config": json.loads((folder / "config.json").read_text()),
                  "t3_weights_sha256": digest(folder / "t3_mtl_te.safetensors")}
    tts = tc.ChatterboxTeluguTTS(folder, "mps")
    t0 = time.perf_counter()
    voice = tc.ChatterboxVoice(tts.builtin)
    torch.mps.synchronize()
    load_s = time.perf_counter() - t0
    line = "ఇది ఒక చిన్న పరీక్ష. మనం కలిసి మంచి పని చేద్దాం."
    times, checks, thermals = {"old": [], "candidate": []}, [], []
    try:
        for run in range(args.warmup + args.runs):
            outputs = {}
            order = (("old", old), ("candidate", new)) if run % 2 == 0 else (("candidate", new), ("old", old))
            for name, fn in order:
                tts._t3_tokens = types.MethodType(fn, tts)
                torch.mps.synchronize()
                t0 = time.perf_counter()
                takes = tts.synthesize_takes(line, voice, max_seconds=4.0, n=2, seeds=[11, 22])
                torch.mps.synchronize()
                elapsed = time.perf_counter() - t0
                outputs[name] = [(q.speech.cpu().numpy().copy(), q.capped, q.t3_steps) for q in takes]
                if run >= args.warmup:
                    times[name].append(elapsed)
                print(f"T3 pair {run}: {name} {elapsed:.4f}s", flush=True)
            checks.append(all(np.array_equal(a[0], b[0]) and a[1:] == b[1:]
                              for a, b in zip(outputs["old"], outputs["candidate"])))
            thermals.append(thermal_state())
        out["t3_buffer_experiment"] = {"timing": {k: summarize(v) for k, v in times.items()},
                                       "same_tokens_caps_steps": checks, "thermal_states": thermals,
                                       "model": model_info, "load_seconds": load_s,
                                       "baseline_method": source, "candidate_method": candidate,
                                       "input": {"text": line, "n": 2, "seeds": [11, 22], "max_seconds": 4.0},
                                       "note": "Not shipped: initial timings showed no gain. T3 only; no flow/vocoder."}
    finally:
        tts.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path.home() / "Library/Application Support/Maata/models")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--component", choices=("separator", "t3-buffer", "both"), default="separator")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_sandboxed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This benchmark targets Apple Silicon macOS")
    if args.runs < 5 or args.warmup < 2:
        parser.error("Use at least five measured pairs and two warmup pairs")
    if not args._sandboxed:
        subprocess.run(["sandbox-exec", "-p", NETWORK_POLICY, sys.executable, str(Path(__file__).resolve()),
                        *sys.argv[1:], "--_sandboxed"], check=True)
        return
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "engine/src"))
    hardware = json.loads(subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"], text=True))
    machine = hardware["SPHardwareDataType"][0]
    out = {"scope": "Exploratory paired component measurements, not whole-render speed or listening quality.",
           "offline": True, "network_policy": NETWORK_POLICY, "baseline_git_sha": git("rev-parse", args.baseline),
           "current_git_sha": git("rev-parse", "HEAD"), "python": platform.python_version(),
           "macos": platform.mac_ver()[0], "machine": {k: machine.get(k) for k in ("chip_type", "physical_memory")},
           "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).splitlines()[0],
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "mlx", "transformers", "chatterbox-tts")},
           "thermal_start": thermal_state(), "runs": args.runs, "warmup_pairs": args.warmup,
           "source_sha256": {p: digest(ROOT / p) for p in SOURCES}, "git_diff": git("diff", "--", *SOURCES),
           "benchmark_script": "scripts/bench_inference_components.py", "script_sha256": digest(Path(__file__)),
           "arguments": [x for x in sys.argv[1:] if x != "--_sandboxed"]}
    if args.component in ("separator", "both"):
        separator(args, out)
    if args.component in ("t3-buffer", "both"):
        t3_buffer(args, out)
    out["thermal_end"] = thermal_state()
    states = [out["thermal_start"], out["thermal_end"]]
    states += [s for k in ("separator", "t3_buffer_experiment") for s in out.get(k, {}).get("thermal_states", [])]
    out["timings_valid_thermal"] = max(states) < 2
    out["numerical_checks_passed"] = all(out.get("separator", {}).get("numerical_checks", [True])) and \
        all(out.get("t3_buffer_experiment", {}).get("same_tokens_caps_steps", [True]))
    path = args.output or ROOT / "docs/spikes/results/inference-components/m5-pro-24gb" / f"{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(path)
    if not out["numerical_checks_passed"]:
        raise SystemExit("Component equivalence failed; inspect the recorded result")


if __name__ == "__main__":
    main()
