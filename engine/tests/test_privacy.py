"""No telemetry (CLAUDE.md): libraries that phone home by default must be switched off by the engine."""

import importlib
import os

import pytest

import maata_engine


def test_engine_forces_telemetry_off_even_if_inherited_env_enables_it(monkeypatch):
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    importlib.reload(maata_engine)
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_pyannote_sees_metrics_disabled():
    metrics = pytest.importorskip("pyannote.audio.telemetry.metrics")  # only with the apple/cuda extras
    importlib.reload(maata_engine)
    assert metrics.is_metrics_enabled() is False
