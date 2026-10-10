"""No model imports/inference: protocol processes, cancellation, PCM provenance and watermark order."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from maata_engine.backends.base import Cancelled
from maata_engine.backends.omnivoice import OmniVoiceError, OmniVoiceTTS, WaveformTake, SAMPLE_RATE


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    model = tmp_path / "Models/omnivoice"
    for name in ("model.safetensors", "config.json", "tokenizer.json", "audio_tokenizer/model.safetensors",
                 "audio_tokenizer/config.json"):
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-only")
    reference = tmp_path / "native.wav"
    reference.write_bytes(b"original Voice B fixture (not played by fake worker)")
    tts = OmniVoiceTTS(model, Path(sys.executable), seed_reference=(reference, "నమస్కారం!"),
                      reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                      request_timeout=0.6, load_timeout=0.6, stop_grace=0.1)
    script = tmp_path / "fake_worker.py"
    script.write_text('''import json, os, signal, sys, time
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
mode = sys.argv[2]
for raw in sys.stdin:
    msg = json.loads(raw)
    rid = msg['id']
    if mode == 'garbage':
        print('not JSON', flush=True)
        continue
    if mode == 'wrong-id':
        rid += 1
    if mode == 'long-line':
        print('x' * 70000, flush=True)
        continue
    if msg['op'] == 'generate':
        if mode == 'hang':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            (root/'started').touch()
            time.sleep(30)
        x = np.full(24000, .1, dtype=np.float32)
        if mode == 'nan':
            x[0] = float('nan')
        if mode == 'missing':
            pass
        else:
            np.save(root / f'audio-{msg["id"]}.npy', x, allow_pickle=False)
    print(json.dumps({'version':2, 'id':rid, 'ok':True, 'sample_rate':24000,
                      'profile':'female' if mode == 'wrong-profile' else msg.get('profile'), 'instruct':msg.get('instruct'),
                      'voice_identity':msg.get('voice_identity'),
                      'pid':os.getpid(), 'offline':os.environ.get('HF_HUB_OFFLINE'),
                      'pythonpath':os.environ.get('PYTHONPATH')}), flush=True)
''')
    tts.test_mode = "normal"
    monkeypatch.setattr(tts, "_command", lambda work: [sys.executable, "-I", "-u", str(script), str(work), tts.test_mode])
    monkeypatch.setattr(tts, "_verify_models", lambda: None)
    try:
        yield tts
    finally:
        tts.release()


def wait_for(path, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path():
            return
        time.sleep(0.01)
    raise AssertionError("Worker did not reach expected state")


def test_persistent_process_keeps_native_voice_and_offline_environment(adapter, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/untrusted/imports")
    first = adapter._request("prepare")
    second = adapter._request("prepare")
    assert first["pid"] == second["pid"]
    assert first["offline"] == "1" and first["pythonpath"] is None
    a = adapter.prepare_voice(np.ones(16000), 16000)
    b = adapter.prepare_voice(np.zeros(32000), 16000)
    assert a == b == adapter.preset_voice("native")
    assert adapter.native_voice


def test_natural_duration_flags_cap_without_cutting_audio(adapter):
    voice = adapter.preset_voice("native")
    take = adapter.synthesize_mel("నమస్కారం!", voice, max_seconds=0.2)
    assert take.seconds == 1.0 and len(take.samples) == SAMPLE_RATE and take.over_budget and not take.capped
    from maata_engine.qa.take import failure
    assert failure(take.seconds, take.capped, 1.0) is None  # complete slow audio must teach the estimator its pace
    assert not list(Path(adapter._temp.name).glob("audio-*.npy"))


def test_release_reaps_persistent_worker_and_cleans_private_outputs(adapter):
    adapter.preset_voice("native")
    proc, work = adapter._process, Path(adapter._temp.name)
    adapter.release()
    assert proc.poll() is not None and not work.exists()
    assert adapter._process is None
    adapter.release()  # idempotent


def test_cancel_running_call_kills_uncooperative_worker_and_allows_fresh_restart(adapter):
    adapter.test_mode = "hang"
    adapter.request_timeout = 10
    voice = adapter.preset_voice("native")
    proc, work = adapter._process, Path(adapter._temp.name)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(adapter.synthesize_mel, "నమస్కారం!", voice)
        wait_for(lambda: (work / "started").exists())
        start = time.monotonic()
        adapter.cancel()
        assert time.monotonic() - start < .2
        with pytest.raises(Cancelled):
            result.result(timeout=3)
    assert proc.poll() is not None and not work.exists()
    adapter.test_mode = "normal"
    assert adapter.synthesize_mel("నమస్కారం!", voice).seconds == 1
    assert adapter._process.pid != proc.pid


def test_cancel_rejects_queued_call_instead_of_starting_another_model(adapter):
    adapter.test_mode = "hang"
    adapter.request_timeout = 10
    voice = adapter.preset_voice("native")
    work = Path(adapter._temp.name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(adapter.synthesize_mel, "నమస్కారం!", voice)
        wait_for(lambda: (work / "started").exists())
        second = pool.submit(adapter.synthesize_mel, "బాగున్నారా?", voice)
        time.sleep(.04)  # second request takes its cancellation epoch before waiting on the first's lock
        adapter.cancel()
        for result in (first, second):
            with pytest.raises(Cancelled):
                result.result(timeout=3)
    assert adapter._process is None


def test_request_timeout_reaps_worker(adapter):
    adapter.test_mode = "hang"
    voice = adapter.preset_voice("native")
    proc, work = adapter._process, Path(adapter._temp.name)
    with pytest.raises(OmniVoiceError, match="timeout"):
        adapter.synthesize_mel("నమస్కారం!", voice)
    assert proc.poll() is not None and not work.exists()


@pytest.mark.parametrize("mode,pattern", [("garbage", "JSON"), ("wrong-id", "identity"),
    ("long-line", "protocol"), ("nan", "nonfinite"), ("missing", "missing")])
def test_bad_worker_response_fails_closed_and_reaps(adapter, mode, pattern):
    adapter.test_mode = mode
    with pytest.raises(OmniVoiceError, match=pattern):
        adapter.synthesize_mel("నమస్కారం!", adapter._voice)
    assert adapter._process is None and adapter._temp is None


def test_reference_changed_after_identity_creation_is_rejected(adapter):
    adapter.reference_path.write_bytes(b"different voice")
    with pytest.raises(OmniVoiceError, match="reference changed"):
        adapter.preset_voice("native")
    assert adapter._process is None


def test_wrong_reference_checksum_never_starts(adapter):
    adapter._expected_reference = "f" * 64
    with pytest.raises(OmniVoiceError, match="checksum"):
        adapter.preset_voice("native")
    assert adapter._process is None


def test_cache_identity_includes_actual_reference_and_generation_policy(adapter):
    identity = adapter.cache_identity
    assert identity["reference_audio_sha256"] == hashlib.sha256(adapter.reference_path.read_bytes()).hexdigest()
    assert identity["reference_text_sha256"] == hashlib.sha256("నమస్కారం!".encode()).hexdigest()
    assert identity["model"] == adapter.model_revision and identity["steps"] == 32
    assert identity["voice_policy"] and identity["watermark_version"] and identity["precision"] == "float16"
    identity["steps"] = 1
    assert adapter.cache_identity["steps"] == 32


def test_pack_roundtrip_retains_unmarked_pcm_and_cap(adapter):
    take = WaveformTake(np.full(SAMPLE_RATE, .1, np.float32), over_budget=True)
    packed = adapter.pack_take(take)
    assert "samples" not in packed  # restoration must not bypass final-rate watermarking via plain PCM
    restored = adapter.unpack_take(packed)
    np.testing.assert_array_equal(restored.samples, take.samples)
    assert restored.seconds == take.seconds and restored.over_budget and not restored.capped


def test_vocode_applies_watermark_after_stretch_and_keeps_raw_take(adapter, monkeypatch):
    raw = np.full(SAMPLE_RATE, .1, np.float32)
    take = WaveformTake(raw.copy())
    marked_inputs = []

    def stretch(x, rate, sr):
        assert sr == SAMPLE_RATE and rate == 1.25
        return x[:19000].copy() * 2

    def watermark(x, sample_rate):
        marked_inputs.append(x.copy())
        return x + .01

    monkeypatch.setattr("maata_engine.backends.omnivoice.wsola", stretch)
    adapter._watermarker = SimpleNamespace(apply_watermark=watermark)
    first = adapter.vocode(take, 1.25)
    restored = adapter.unpack_take(adapter.pack_take(take))
    second = adapter.vocode(restored, 1.0)
    assert len(first) == 19000 and len(second) == SAMPLE_RATE
    np.testing.assert_allclose(marked_inputs[0], .2)
    np.testing.assert_allclose(marked_inputs[1], .1)
    np.testing.assert_array_equal(take.samples, raw)


@pytest.mark.parametrize("rate", [0, .9, float("nan"), 2])
def test_vocode_refuses_invalid_rate(adapter, rate):
    with pytest.raises(ValueError):
        adapter.vocode(WaveformTake(np.ones(1000, np.float32)), rate)


def test_wrong_sample_rate_cache_is_rejected(adapter):
    with pytest.raises(OmniVoiceError, match="sample rate"):
        adapter.unpack_take({"omnivoice_sample_rate": np.int64(16000)})


def test_idle_cancel_also_reaps_worker(adapter):
    adapter.preset_voice("native")
    proc = adapter._process
    adapter.cancel()
    wait_for(lambda: proc.poll() is not None)


def test_bound_paused_job_cannot_start_worker_but_next_admitted_job_can(adapter):
    stop = threading.Event()
    adapter.bind_cancel_event(stop)
    stop.set()
    with pytest.raises(Cancelled):
        adapter.preset_voice("native")
    assert adapter._process is None
    adapter.bind_cancel_event(threading.Event())
    assert adapter.synthesize_mel("నమస్కారం!", adapter._voice).seconds == 1


def test_bound_cancellation_during_start_never_submits_model_request(adapter, monkeypatch):
    stop = threading.Event()
    adapter.bind_cancel_event(stop)
    start = adapter._start
    def stop_after_start():
        start()
        stop.set()
    monkeypatch.setattr(adapter, "_start", stop_after_start)
    with pytest.raises(Cancelled):
        adapter.synthesize_mel("నమస్కారం!", adapter._voice)
    assert adapter._counter == 0 and adapter._process is None


def test_model_pins_reverified_after_file_change(adapter, monkeypatch):
    from maata_engine import models
    content = b"pinned tiny model"
    path = adapter.model_dir / "model.safetensors"
    path.write_bytes(content)
    manifest = {"models": [{"id": "omnivoice", "revision": adapter.model_revision,
                 "files": [{"path": "model.safetensors", "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()}]}]}
    monkeypatch.setattr(models, "load_lock", lambda: manifest)
    verify = OmniVoiceTTS._verify_models.__get__(adapter)
    verify()
    signature = adapter._verified_signature
    verify()
    assert adapter._verified_signature == signature
    path.write_bytes(b"changed tiny file")
    with pytest.raises(OmniVoiceError, match="checksum"):
        verify()


def test_automatic_default_needs_no_reference_and_ignores_old_conditioned_metadata(adapter, tmp_path):
    voices = tmp_path / "auto-voices"
    voices.mkdir()
    (voices / "native.json").write_text("old or invalid experimental conditioning metadata")
    auto = OmniVoiceTTS(adapter.model_dir, Path(sys.executable), voice_dir=voices)
    try:
        assert auto.missing() == [] and auto._process is None
        assert auto.reference_path is None and auto.reference_text == ""
        identity = auto.cache_identity
        assert identity["mode"] == "automatic" and identity["seed"] == 20261010
        assert identity["seed_policy"] == "fixed-before-every-generation"
        assert identity["reference_audio_sha256"] is None
        assert identity != adapter.cache_identity  # explicit reference experiment cannot serve automatic takes
    finally:
        auto.release()


def test_stderr_error_tail_does_not_include_arbitrary_upstream_text(adapter, monkeypatch):
    script = adapter.model_dir.parent / "failing.py"
    script.write_text("import sys\nprint('transcript: do not surface me',file=sys.stderr)\n"
                      "print(\"ModuleNotFoundError: No module named 'omnivoice'\",file=sys.stderr)\n")
    monkeypatch.setattr(adapter, "_command", lambda work: [sys.executable, str(script)])
    with pytest.raises(OmniVoiceError, match="ModuleNotFoundError") as exc:
        adapter.preset_voice("native")
    assert "transcript" not in str(exc.value) and adapter._process is None


def test_production_worker_launch_isolates_sibling_adapter_from_upstream_import(adapter, tmp_path):
    import subprocess
    command = OmniVoiceTTS._command(adapter, tmp_path)
    runtime = command.index(str(adapter.runtime_python))
    assert command[runtime + 1:runtime + 3] == ["-I", "-u"]
    # Mirror the worker's sibling collision with a stdlib package, without loading OmniVoice or a model.
    (tmp_path / "fractions.py").write_text("raise RuntimeError('sibling shadow imported')\n")
    probe = tmp_path / "worker.py"
    probe.write_text("import fractions; print(fractions.Fraction(1, 2))\n")
    result = subprocess.run([sys.executable, "-I", "-u", str(probe)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and result.stdout.strip() == "1/2"


def test_profiles_have_distinct_cache_identity_and_one_worker(adapter, monkeypatch):
    auto = OmniVoiceTTS(adapter.model_dir, Path(sys.executable))
    monkeypatch.setattr(auto, "_command", adapter._command)
    monkeypatch.setattr(auto, "_verify_models", lambda: None)
    try:
        voices = [auto.preset_voice(p) for p in ("male", "female", "native")]
        assert len({v.identity for v in voices}) == 3
        pid = auto._process.pid
        for voice in voices:
            take = auto.synthesize_mel("నమస్కారం!", voice)
            assert take.seconds == 1 and auto._process.pid == pid
        male, female = (auto.profile_cache_identity(p) for p in ("male", "female"))
        assert male["instruct"] == "male" and female["instruct"] == "female" and male != female
        assert male["seed"] == female["seed"] == 20261010
        returned = auto.cache_identity
        returned["profiles"]["male"] = "female"
        assert auto.profile_cache_identity("male")["instruct"] == "male"
        with pytest.raises(OmniVoiceError, match="Unknown"):
            auto.preset_voice("male, whisper")
        from maata_engine.backends.omnivoice import OmniVoiceVoice
        forged = OmniVoiceVoice(voices[0].identity, "female")
        with pytest.raises(OmniVoiceError, match="configured"):
            auto.synthesize_mel("నమస్కారం!", forged)
    finally:
        auto.release()


def test_mismatched_worker_profile_is_rejected(adapter):
    adapter.test_mode = "wrong-profile"
    voice = adapter.preset_voice("native")
    with pytest.raises(OmniVoiceError, match="profile did not match"):
        adapter.synthesize_mel("నమస్కారం!", voice)
    assert adapter._process is None


def test_reference_mode_cannot_accept_gender_profiles(adapter):
    with pytest.raises(OmniVoiceError, match="reference conditioning"):
        adapter.preset_voice("male")
    assert adapter._process is None
