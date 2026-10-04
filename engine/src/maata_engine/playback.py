"""The binary frame of ADR-007 (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §4, ADR-021): a render's Hear voice
sample, read from render/voices/ alone. Nothing of a render is played in the app (§6): its dub is a file. The original
audio (audio.<ext>, render/audio16k.f32) is never opened here."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

SAMPLE_ID = 0xFFFFFF00  # frame id of a speaker's Hear voice sample: this plus the speaker's index (S1: 0)

SendBytes = Callable[[bytes], Awaitable[None]]


def frame(uid: int, samples: np.ndarray, sr: int) -> bytes:
    """One binary frame (ADR-007): a header of its id, the sample rate, the sample count and 0, then float32 samples."""
    out = np.asarray(samples, np.float32)
    return struct.pack("<IIII", uid, sr, len(out), 0) + out.tobytes()


async def voice_sample(render_dir: Path, sid: str, sr: int, send_bytes: SendBytes) -> bool:
    """Send a speaker's Hear voice sample (render/voices/<id>.npy: synthetic Telugu said by their voice, never the
    source) as one frame, id SAMPLE_ID + their index. False when they have none (a stock voice, or not built yet)."""
    digits = sid.lstrip("S")
    if not digits.isdigit() or not 1 <= int(digits) <= 255:
        return False
    pcm = await asyncio.to_thread(_load, render_dir / "voices" / f"{sid}.npy")
    if pcm is None:
        return False
    await send_bytes(frame(SAMPLE_ID + int(digits) - 1, pcm, sr))
    return True


def _load(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=False).astype(np.float32)
    except (OSError, ValueError):
        return None
