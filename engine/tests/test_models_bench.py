import hashlib
import json
import wave

import numpy as np

from maata_engine.bench import main as bench_main
from maata_engine.models import FileSpec, load_lock, verify


def test_lock_is_pinned_and_complete():
    lock = load_lock()
    ids = {m["id"] for m in lock["models"]}
    assert {"whisper-large-v3-turbo-mlx", "translategemma-4b-it-4bit-mlx", "chatterbox-telugu"} <= ids
    for m in lock["models"]:
        assert len(m["revision"]) == 40  # a commit, never a branch
        assert m["files"] and all(f["size"] > 0 and ("sha256" in f or "git_oid" in f) for f in m["files"])
        assert m["license"]


def test_verify_sha256_and_git_oid(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"maata")
    sha = hashlib.sha256(b"maata").hexdigest()
    git = hashlib.sha1(b"blob 5\0maata").hexdigest()
    assert verify(p, FileSpec("f.bin", 5, sha, None))
    assert verify(p, FileSpec("f.bin", 5, None, git))
    assert not verify(p, FileSpec("f.bin", 5, "0" * 64, None))
    assert not verify(p, FileSpec("f.bin", 6, sha, None))


def test_bench_pipeline_on_local_file_with_mock_backend(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    sr = 16000
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.random.default_rng(0).standard_normal(sr * 70) * 300).astype("<i2").tobytes())
    bench_main(["--cache", str(tmp_path), "--models", str(tmp_path), "pipeline", str(wav), "--backend", "mock"])
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "mock" and out["units"] > 0 and out["skipped"] == 0
    assert out["speedup_max"] <= 1.2 + 1e-9
    assert out["throughput_x_realtime"] > 1
