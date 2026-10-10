#!/usr/bin/env python3
"""Fetch one attributed Telugu reference using a bounded archive prefix, never the dataset.

This explicit network stage is separate from every offline synthesis process.
Audio and dataset transcript remain in the local evaluation cache, not the repo.
"""
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/te_in/"
MEMBER = "dev/1011590415238839102.wav"
AUDIO_SHA256 = "31c67dc8ac73afc639cc92b807341b95127f46f89c12b3bdd27a97c6f5d3a1e1"
TSV_SHA256 = "fbedc2a5b7e394ff5a6e8c269f616017d2ac9a247ffa707c10efaa08132a0630"
ROOT = Path.home() / "Library/Caches/Maata/model-evaluation-2026-10-10/references"


class BoundedReader:
    def __init__(self, raw):
        self.raw, self.read_bytes = raw, 0

    def read(self, n=-1):
        n = 65536 if n < 0 else n
        if self.read_bytes + n > 4_000_000:
            raise RuntimeError("Reference is outside the authorized 4 MB archive prefix")
        data = self.raw.read(n)
        self.read_bytes += len(data)
        return data


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE + "dev.tsv", timeout=25) as response:
        rows = response.read(350270)
    assert len(rows) == 350269 and hashlib.sha256(rows).hexdigest() == TSV_SHA256
    with urlopen(BASE + "audio/dev.tar.gz", timeout=25) as response:
        source = BoundedReader(response)
        with tarfile.open(fileobj=source, mode="r|gz") as archive:
            for member in archive:
                if member.name != MEMBER:
                    continue
                assert member.isfile() and member.size == 702778
                audio = archive.extractfile(member).read()
                assert hashlib.sha256(audio).hexdigest() == AUDIO_SHA256
                break
            else:
                raise RuntimeError("Pinned reference member missing")
    matching = [line.split("\t") for line in rows.decode().splitlines()
                if Path(MEMBER).name in line.split("\t")]
    assert len(matching) == 1
    path = ROOT / "fleurs-te-dev-first.wav"
    path.write_bytes(audio)
    (ROOT / "fleurs-te-dev.tsv").write_bytes(rows)
    manifest = {"dataset": "google/fleurs", "revision": REVISION, "config": "te_in",
        "split": "validation", "license": "CC-BY-4.0",
        "attribution": "Google FLEURS dataset, Conneau et al., 2022, https://huggingface.co/datasets/google/fleurs",
        "archive_url": BASE + "audio/dev.tar.gz", "archive_member": MEMBER,
        "downloaded_archive_prefix_bytes": source.read_bytes,
        "wav_path": str(path), "wav_bytes": len(audio), "wav_sha256": AUDIO_SHA256,
        "tsv_url": BASE + "dev.tsv", "tsv_sha256": TSV_SHA256,
        "dataset_row_fields": matching[0], "reference_transcript": matching[0][2],
        "transcript_source": "FLEURS dev.tsv raw_transcription column2 (zero-based), full matching filename row",
        "native_provenance_limit": "Dataset-labeled Telugu reference with supplied text; no native-listener certification."}
    (ROOT / "reference.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
