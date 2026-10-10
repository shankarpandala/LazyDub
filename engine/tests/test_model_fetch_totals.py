"""Fetching manifests with optional display totals still verifies every pinned file."""
from argparse import Namespace
from copy import deepcopy

import pytest

from maata_engine import bench
from maata_engine.models import load_lock


@pytest.mark.parametrize("total", [None, -1, "unknown"])
def test_fetch_derives_missing_or_invalid_total_from_pinned_files(monkeypatch, tmp_path, capsys, total):
    model = next(deepcopy(m) for m in load_lock()["models"] if m["id"] == "omnivoice")
    if total is None:
        model.pop("total_bytes", None)
    else:
        model["total_bytes"] = total
    calls = []
    monkeypatch.setattr(bench, "models_for", lambda _: [model])
    monkeypatch.setattr(bench, "fetch_model", lambda *args: calls.append(args))
    assert bench.cmd_fetch(Namespace(models=str(tmp_path), backend="apple")) == 0
    assert len(calls) == 1 and calls[0][0] is model and calls[0][1] == tmp_path
    assert f'{sum(f["size"] for f in model["files"]) / 1e9:.2f} GB' in capsys.readouterr().out


def test_fetch_missing_display_total_does_not_hide_verification_failure(monkeypatch, tmp_path):
    model = {"id": "test", "license": "test", "files": [{"path": "one", "size": 12}]}
    monkeypatch.setattr(bench, "models_for", lambda _: [model])
    def bad_fetch(*_):
        raise bench.FetchError("checksum mismatch")
    monkeypatch.setattr(bench, "fetch_model", bad_fetch)
    assert bench.cmd_fetch(Namespace(models=str(tmp_path), backend="apple")) == 1
