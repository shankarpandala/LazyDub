"""Backend selection: Apple Silicon first (the reference machine), then CUDA, else the demo engine."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from .base import Backend


def detect() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "mock"


def load_backend(name: str | None, models_dir: Path) -> Backend:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # spec §3.3: never download on our behalf
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(models_dir / ".hf"))
    choice = name or detect()
    if choice == "apple":
        from .apple import make_apple_backend

        return make_apple_backend(models_dir)
    if choice == "cuda":
        from .cuda import make_cuda_backend

        return make_cuda_backend(models_dir)
    from .mock import make_mock_backend

    return make_mock_backend()
