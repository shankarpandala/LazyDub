"""The engine's settings (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §4): settings.json in the config folder (on
macOS ~/Library/Application Support/Maata, next to Models; ~/.config/maata elsewhere), not in the cache, which users and
cleaners treat as disposable, nor in the UI's storage, whose origin changes with the port at every launch. The only
setting is `outputDir`, where the dubbed MP4s go."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_OUTPUT = Path("~/Movies/Maata")  # created on first use; ~/Movies is not a folder macOS asks permission for


class SettingsError(ValueError):
    """A setting that can't be used; the message says why."""


def default_config_dir() -> Path:
    return Path("~/Library/Application Support/Maata" if sys.platform == "darwin" else "~/.config/maata").expanduser()


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT.expanduser()


def load(config_dir: Path) -> dict:
    """The settings saved in `config_dir`, over the defaults; a missing or unreadable file gives the defaults."""
    try:
        saved = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    out = {"outputDir": str(default_output_dir())}
    if isinstance(saved, dict) and isinstance(saved.get("outputDir"), str) and saved["outputDir"].strip():
        out["outputDir"] = saved["outputDir"]
    return out


def save(config_dir: Path, settings: dict) -> None:
    """Write settings.json whole, then put it in place."""
    config_dir.mkdir(parents=True, exist_ok=True)
    tmp = config_dir / "settings.json.tmp"
    tmp.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, config_dir / "settings.json")


def output_dir(path: str | Path) -> Path:
    """`path` as the output folder: absolute (a leading ~ is the home folder), created if missing, and writable;
    `SettingsError` saying why not."""
    folder = Path(str(path).strip()).expanduser()
    if not str(path).strip() or not folder.is_absolute():
        raise SettingsError("Choose a folder by its full path, such as ~/Movies/Maata.")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as e:
        raise SettingsError(f"Maata can't save videos in {folder} ({e.strerror or e}). Choose another folder.") from e
    return folder
