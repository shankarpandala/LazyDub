"""Every test's default output and config folders are its own temporary ones: no test writes into ~/Movies/Maata or
~/Library/Application Support/Maata, whatever it forgets to pass.

Every render job a test runs to `done` saves an MP4 of the demo's 3 minutes; AudioToolbox's AAC at its best quality is
most of that, so tests encode at its fastest (`aac_at_quality` 2) unless marked `product_aac`: the tests of the audio
itself (loudness and true peak after AAC, A/V sync, byte-identical exports) run with the product's encoder."""

from __future__ import annotations

import pytest

from maata_engine import export, settings


def pytest_configure(config):
    config.addinivalue_line("markers", "product_aac: encode with the product's AAC options")


@pytest.fixture(autouse=True)
def _private_folders(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(settings, "DEFAULT_OUTPUT", home / "Movies" / "Maata")
    monkeypatch.setattr(settings, "default_config_dir", lambda: home / "config")


@pytest.fixture(autouse=True)
def _quick_aac(request, monkeypatch):
    if export.aac() == "aac_at" and request.node.get_closest_marker("product_aac") is None:
        monkeypatch.setattr(export, "AAC_OPTIONS", {"aac_at_quality": "2"})
