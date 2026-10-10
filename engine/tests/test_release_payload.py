"""Release payload allowlist, reproducibility and source integrity."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_runtime", ROOT / "scripts/prepare-runtime.py")
prepare_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_runtime)


def fixture_source(tmp_path):
    root = tmp_path / "source"
    for name in prepare_runtime.FIXED_FILES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    p = root / "engine/src/maata_engine/server.py"
    p.parent.mkdir(parents=True)
    p.write_text("# engine\n")
    config = root / "app/src-tauri/tauri.conf.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"version":"0.1.0"}')
    return root


def test_payload_excludes_local_data_and_rebuild_removes_stale_files(tmp_path):
    root = fixture_source(tmp_path)
    for name in ("index.html", "engine/.venv/token", "engine/models/audio.wav",
                 "engine/src/maata_engine/__pycache__/server.pyc"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("private")
    target = tmp_path / "payload"
    first = prepare_runtime.prepare(root, target)
    (target / "stale.txt").write_text("stale")
    second = prepare_runtime.prepare(root, target)
    assert first == second
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == set(first["files"]) | {"release.json"}
    assert json.loads((target / "release.json").read_text()) == first
    (root / "engine/src/maata_engine/server.py").write_text("# changed\n")
    assert prepare_runtime.prepare(root, target)["runtime_id"] != first["runtime_id"]


def test_symlink_source_rejected_without_replacing_existing_payload(tmp_path):
    root = fixture_source(tmp_path)
    target = tmp_path / "payload"
    first = prepare_runtime.prepare(root, target)
    script = root / "scripts/install-runtime.sh"
    script.unlink()
    script.symlink_to(root / "scripts/setup-omnivoice.sh")
    with pytest.raises(ValueError, match="symlinks"):
        prepare_runtime.prepare(root, target)
    assert json.loads((target / "release.json").read_text()) == first
