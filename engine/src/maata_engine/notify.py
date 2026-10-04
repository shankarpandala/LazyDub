"""Notifications (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §4) on macOS, through /usr/bin/osascript's `display
notification`, so no dependency is added: they show under Script Editor's name and icon, once Script Editor may notify
(System Settings ▸ Notifications). The texts go to the script as its arguments (`argv`), never spliced into it. Nothing
depends on them: the Library and the job view show every event they report."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Callable

log = logging.getLogger("maata.notify")

OSASCRIPT = "/usr/bin/osascript"
SCRIPT = ("-e", "on run argv", "-e", "display notification (item 2 of argv) with title (item 1 of argv)", "-e", "end run")


def spawn(cmd: list[str]) -> None:
    """Start `cmd` detached, its output discarded (osascript here; the engine's `open` too)."""
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


RUNNER: Callable[[list[str]], None] = spawn  # how the command runs (tests stub it)


def command(title: str, body: str) -> list[str]:
    """osascript's arguments: the fixed script, then the title and the body as the script's argv."""
    return [OSASCRIPT, *SCRIPT, title, body]


def post(title: str, body: str) -> None:
    """Post a notification on macOS; elsewhere (or without osascript) nothing. Never raises."""
    if sys.platform != "darwin" or not os.path.exists(OSASCRIPT):
        return
    try:
        RUNNER(command(title, body))
    except OSError:
        log.warning("couldn't post a notification", exc_info=True)
