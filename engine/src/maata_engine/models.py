"""Pinned model fetching (spec §3.3, §7): every file by commit, size and hash; resumable; atomic.

The app's Model Manager uses the same lock format; this is the engine-side fetcher used by
`maata-bench fetch` and first-run setup.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LOCK = Path(__file__).resolve().parents[2] / "models.lock.json"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str | None
    git_oid: str | None


def load_lock(path: Path = LOCK) -> dict:
    return json.loads(path.read_text())


def _git_blob_sha1(data_path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {data_path.stat().st_size}\0".encode())
    with data_path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def verify(p: Path, spec: FileSpec) -> bool:
    if not p.is_file() or p.stat().st_size != spec.size:
        return False
    if spec.sha256:
        return _sha256(p) == spec.sha256
    return spec.git_oid is None or _git_blob_sha1(p) == spec.git_oid


def fetch_model(model: dict, dest_root: Path, token: str | None = None,
                progress: Callable[[str, int, int], None] | None = None) -> Path:
    dest = dest_root / model["id"]
    dest.mkdir(parents=True, exist_ok=True)
    for f in model["files"]:
        spec = FileSpec(f["path"], f["size"], f.get("sha256"), f.get("git_oid"))
        final = dest / spec.path
        if verify(final, spec):
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have > spec.size:
            part.unlink()
            have = 0
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{spec.path}"
        req = urllib.request.Request(url, headers={"User-Agent": "maata/0.1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r, part.open("ab" if have and r.status == 206 else "wb") as out:
                got = have if r.status == 206 else 0
                for b in iter(lambda: r.read(CHUNK), b""):
                    out.write(b)
                    got += len(b)
                    if progress:
                        progress(spec.path, got, spec.size)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise FetchError(f"{model['repo']} needs a Hugging Face token (accept its terms first).") from e
            raise FetchError(f"Download failed for {spec.path}: HTTP {e.code}") from e
        if not verify(part, spec):
            part.unlink(missing_ok=True)
            raise FetchError(f"Checksum mismatch for {model['id']}/{spec.path}; the partial file was removed.")
        os.replace(part, final)  # atomic move into place
    return dest


def models_for(backend: str, lock: dict | None = None) -> list[dict]:
    lock = lock or load_lock()
    return [m for m in lock["models"] if backend in m["backends"]]
