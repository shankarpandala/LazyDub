"""Standalone worker logic with a tiny injected model; no Torch/model load or audio generation."""

import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import contextlib
import subprocess
import time

import numpy as np
import pytest

from maata_engine.backends import _omnivoice_worker as worker


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_runtime", lambda: None)
    root = tmp_path / "model"
    (root / "audio_tokenizer").mkdir(parents=True)
    ref = tmp_path / "native.wav"
    ref.write_bytes(b"voice fixture")
    config = {"model_dir": str(root), "voice_dir": str(tmp_path / "voices"), "device": "mps",
              "mode": "reference-experiment", "seed": 20261010,
              "steps": 32, "dtype": "float16", "reference_path": str(ref), "reference_text": "నమస్కారం!",
              "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(), "voice_identity": "a" * 64}
    calls = []

    class Prompt:
        ref_text = config["reference_text"]
        ref_audio_tokens = SimpleNamespace(size=lambda axis: 50)

        def save(self, path):
            Path(path).write_text(self.ref_text)

        @classmethod
        def load(cls, path, map_location):
            calls.append(("load_prompt", map_location))
            prompt = cls()
            prompt.ref_text = Path(path).read_text()
            return prompt

    class Model:
        sampling_rate = 24000
        audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=25))
        tokens = 100

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("load_model", path, kwargs))
            return cls()

        def create_voice_clone_prompt(self, **kwargs):
            calls.append(("create_prompt", kwargs))
            return Prompt()

        def _estimate_target_tokens(self, *args):
            return self.tokens

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return [np.full(24000, .1, np.float32)]

    torch = ModuleType("torch")
    torch.float16, torch.float32, torch.inference_mode = "fp16", "fp32", contextlib.nullcontext
    torch.manual_seed = lambda seed: calls.append(("seed", seed))
    impl = ModuleType("omnivoice.models.omnivoice")
    impl.OmniVoice, impl.VoiceClonePrompt = Model, Prompt
    top = ModuleType("omnivoice")
    middle = ModuleType("omnivoice.models")
    middle.omnivoice, top.models = impl, middle
    for key, value in {"torch": torch, "omnivoice": top, "omnivoice.models": middle,
                       "omnivoice.models.omnivoice": impl}.items():
        monkeypatch.setitem(sys.modules, key, value)
    return config, calls, impl


def test_local_load_and_exact_native_prompt_reuse(model):
    config, calls, impl = model
    first = worker.SpeechWorker(config)
    first.generate("బాగున్నారా?")
    worker.SpeechWorker(config)
    loading = next(x for x in calls if x[0] == "load_model")
    assert loading[2] == {"device_map": "mps", "dtype": "fp16", "local_files_only": True,
                          "trust_remote_code": False, "load_asr": False}
    create = [x for x in calls if x[0] == "create_prompt"]
    assert len(create) == 1 and create[0][1] == {"ref_audio": config["reference_path"],
                     "ref_text": config["reference_text"], "preprocess_prompt": False}
    generate = next(x for x in calls if x[0] == "generate")[1]
    assert generate["num_step"] == 32 and generate["language"] == "te"
    assert generate["normalize_text"] is False
    assert "duration" not in generate and "speed" not in generate
    with pytest.raises(RuntimeError, match="remote"):
        impl._resolve_model_path("k2-fsa/OmniVoice")
    with pytest.raises(RuntimeError, match="ASR"):
        impl.OmniVoice.load_asr_model(None)


def test_pathological_duration_is_rejected_before_any_generation(model):
    config, calls, _ = model
    running = worker.SpeechWorker(config)
    running.model.tokens = 1501
    with pytest.raises(RuntimeError, match="60-second"):
        running.generate("నమస్కారం!")
    assert not any(x[0] == "generate" for x in calls)


def test_worker_reference_checksum_cannot_be_bypassed_by_existing_prompt(model):
    config, _, _ = model
    worker.SpeechWorker(config)
    Path(config["reference_path"]).write_bytes(b"replaced reference")
    with pytest.raises(RuntimeError, match="checksum"):
        worker.SpeechWorker(config)


def test_persisted_prompt_must_have_exact_transcript(model):
    config, _, _ = model
    worker.SpeechWorker(config)
    (Path(config["voice_dir"]) / (config["voice_identity"] + ".pt")).write_text("wrong reference text")
    with pytest.raises(RuntimeError, match="transcript"):
        worker.SpeechWorker(config)


def test_automatic_mode_exact_audition_parameters_and_seed_every_call(model, monkeypatch):
    config, calls, _ = model
    config.update(mode="automatic", reference_path=None, reference_text="", reference_sha256=None)
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy_seed", seed)))
    running = worker.SpeechWorker(config)
    assert running.prompt is None
    running.generate("నమస్కారం!")
    running.generate("బాగున్నారా?")
    assert not any(c[0] in ("create_prompt", "load_prompt") for c in calls)
    assert [c[1] for c in calls if c[0] == "seed"] == [20261010, 20261010]
    assert [c[1] for c in calls if c[0] == "numpy_seed"] == [20261010, 20261010]
    generated = [c[1] for c in calls if c[0] == "generate"]
    for params, text in zip(generated, ["నమస్కారం!", "బాగున్నారా?"]):
        assert params == {"text": text, "language": "te", "num_step": 32, "normalize_text": False}


def test_automatic_mode_refuses_hidden_reference_or_changed_seed(model):
    config, _, _ = model
    config["mode"] = "automatic"
    with pytest.raises(RuntimeError, match="must not receive reference"):
        worker.SpeechWorker(config)
    config.update(reference_path=None, reference_text="", reference_sha256=None, seed=99)
    with pytest.raises(RuntimeError, match="fixed seed"):
        worker.SpeechWorker(config)


def test_profile_generation_uses_exact_instruction_without_reference(model):
    config, calls, _ = model
    config.update(mode="automatic", reference_path=None, reference_text="", reference_sha256=None)
    running = worker.SpeechWorker(config)
    for profile in ("male", "female", "male"):
        running.generate("రవి రేపు వస్తాడు.", profile)
    generated = [c[1] for c in calls if c[0] == "generate"]
    assert [g["instruct"] for g in generated] == ["male", "female", "male"]
    assert all(g["language"] == "te" and g["normalize_text"] is False for g in generated)
    assert all("voice_clone_prompt" not in g and "duration" not in g and "speed" not in g for g in generated)
    assert [c[1] for c in calls if c[0] == "seed"] == [20261010] * 3
    assert not any(c[0] in ("create_prompt", "load_prompt") for c in calls)


@pytest.mark.parametrize("profile,instruct,identity,mode", [
    (None, None, "a" * 64, "automatic"),
    ("male", "female", "a" * 64, "automatic"),
    ("male", "male, low pitch", "a" * 64, "automatic"),
    ("female", "female", "bad", "automatic"),
    ("female", "female", "a" * 64, "reference-experiment"),
])
def test_profile_protocol_rejects_untrusted_or_mismatched_values(profile, instruct, identity, mode):
    with pytest.raises(ValueError):
        worker.checked_profile(profile, instruct, identity, mode)


def test_invalid_profile_request_is_rejected_before_loading_model(tmp_path, monkeypatch):
    message = {"version": 2, "id": 1, "op": "generate", "config": {"mode": "automatic"},
               "text": "నమస్కారం!", "profile": "male", "instruct": "female", "voice_identity": "a" * 64}
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO((json.dumps(message) + "\n").encode())))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    def forbidden(config):
        pytest.fail("Invalid profile must fail before model loading")
    worker.serve(tmp_path, forbidden, stop_on_eof=False)
    result = json.loads(stdout.getvalue())
    assert result["ok"] is False and "profile" in result["error"]
    assert not (tmp_path / "audio-1.npy").exists()


def test_protocol_reuses_worker_and_refuses_config_change(tmp_path, monkeypatch):
    loaded = []

    class Tiny:
        np = np

        def __init__(self, config):
            loaded.append(config)

        def generate(self, text, profile):
            print("library noise")
            return np.full(1000, .1, np.float32), .04

    messages = [{"version": 2, "id": i, "op": "generate", "config": {"reference": "B"}, "text": "మాట",
                 "profile": "male", "instruct": "male", "voice_identity": "b" * 64}
                for i in (1, 2, 3)]
    messages[-1]["config"] = {"reference": "other"}
    stdin = SimpleNamespace(buffer=io.BytesIO(("\n".join(json.dumps(m) for m in messages) + "\n").encode()))
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    worker.serve(tmp_path, Tiny, stop_on_eof=False)
    replies = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert len(loaded) == 1 and [x["ok"] for x in replies] == [True, True, False]
    assert "cannot change" in replies[-1]["error"]
    assert "library noise" in stderr.getvalue() and "library noise" not in stdout.getvalue()
    assert (tmp_path / "audio-1.npy").is_file() and (tmp_path / "audio-2.npy").is_file()
    assert not (tmp_path / "audio-3.npy").exists()


def test_runtime_rejects_dependency_or_source_pin_mismatch(monkeypatch):
    metadata = worker.importlib.metadata
    monkeypatch.setattr(metadata, "version", lambda name: {"torch":"2.8.0", "torchaudio":"2.8.0",
                                                          "transformers":"5.3.0"}[name])
    source = {"vcs_info": {"commit_id": worker.SOURCE_REVISION}}
    monkeypatch.setattr(metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda filename: json.dumps(source)))
    worker.verify_runtime()
    source["vcs_info"]["commit_id"] = "unapproved"
    with pytest.raises(RuntimeError, match="source revision"):
        worker.verify_runtime()
    monkeypatch.setattr(metadata, "version", lambda name: "0.0")
    with pytest.raises(RuntimeError, match="pinned torch"):
        worker.verify_runtime()


def test_parent_eof_stops_worker_even_while_model_is_loading(tmp_path):
    script = tmp_path / "lifeline.py"
    script.write_text('''import importlib.util, pathlib, sys, time
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
class SleepingModel:
    def __init__(self, config):
        pathlib.Path(sys.argv[2], 'loading').touch()
        time.sleep(30)
w.serve(pathlib.Path(sys.argv[2]), SleepingModel)
''')
    proc = subprocess.Popen([sys.executable, str(script), worker.__file__, str(tmp_path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(json.dumps({"version":2,"id":1,"op":"prepare","config":{}}).encode() + b"\n")
        proc.stdin.flush()
        until = time.monotonic() + 3
        while not (tmp_path / "loading").exists() and time.monotonic() < until:
            time.sleep(.01)
        assert (tmp_path / "loading").exists()
        proc.stdin.close()  # exactly the lifeline loss caused by hard parent termination
        assert proc.wait(timeout=2) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()
