#!/usr/bin/env python3
"""Stage only release engine sources and locks, with a reproducible file manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXED_FILES = (
    "LICENSE",
    "engine/pyproject.toml",
    "engine/uv.lock",
    "engine/models.lock.json",
    "engine/runtimes/omnivoice/pyproject.toml",
    "engine/runtimes/omnivoice/uv.lock",
    "engine/runtimes/omnivoice/models.lock.json",
    "scripts/setup-omnivoice.sh",
    "scripts/install-runtime.sh",
)


def prepare(root: Path, destination: Path) -> dict:
    sources = [root / name for name in FIXED_FILES]
    sources += sorted((root / "engine/src/maata_engine").rglob("*.py"))
    if not any(p.name == "server.py" for p in sources):
        raise ValueError("Engine source tree is missing")
    files = {}
    for source in sources:
        relative = source.relative_to(root)
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p != root.parent):
            raise ValueError(f"Release sources must not be symlinks: {relative}")
        files[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    version = json.loads((root / "app/src-tauri/tauri.conf.json").read_text())["version"]
    release = {"schema": 1, "runtime_id": digest, "payload_sha256": digest,
               "files": files, "app_version": version, "platform": "macos-arm64"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".runtime-", dir=destination.parent))
    backup = staging.with_name(staging.name + "-previous")
    try:
        for source in sources:
            target = staging / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (staging / "release.json").write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
        if destination.is_symlink():
            raise ValueError("Release destination must not be a symlink")
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "app/src-tauri/resources/runtime")
    args = parser.parse_args()
    result = prepare(ROOT, args.out.absolute())
    print(f"Staged {len(result['files'])} runtime files; ID {result['runtime_id']}")
