"""Stream resolution and audio ingest (spec §6.1, ADR-010).

yt-dlp runs in-process with remote components and self-update disabled; it only talks to YouTube.
The UI plays video through YouTube's embedded player, so only the audio stream is fetched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .backends.base import SR_ANALYSIS
from .urls import VideoRef


class ResolveError(RuntimeError):
    """User-facing resolution failure."""


@dataclass(slots=True)
class ResolvedVideo:
    video_id: str
    title: str
    duration: float
    channel: str
    audio_path: Path | None


class Resolver(Protocol):
    def resolve(self, ref: VideoRef, cache_dir: Path) -> ResolvedVideo: ...
    def load_audio(self, video: ResolvedVideo) -> np.ndarray: ...


class YtDlpResolver:
    def resolve(self, ref: VideoRef, cache_dir: Path) -> ResolvedVideo:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError

        out_dir = cache_dir / ref.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": str(out_dir / "audio.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "ignoreconfig": True,
            "remote_components": [],  # never fetch solver scripts from the network (confirm at pin)
            "cachedir": str(cache_dir / ".yt-dlp"),
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={ref.video_id}", download=False)
                if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                    raise ResolveError("Live streams and premieres aren't supported yet.")
                if (info.get("age_limit") or 0) >= 18:
                    raise ResolveError("Age-restricted videos aren't supported.")
                if info.get("availability") in ("private", "premium_only", "subscriber_only", "needs_auth"):
                    raise ResolveError("This video isn't public, so it can't be dubbed.")
                existing = sorted(out_dir.glob("audio.*"))
                if not existing:
                    ydl.process_info(info)
                    existing = sorted(out_dir.glob("audio.*"))
        except DownloadError as e:
            msg = str(e)
            if "Private video" in msg:
                raise ResolveError("This video is private.") from e
            if "Sign in to confirm" in msg:
                raise ResolveError("YouTube is asking for a sign-in check. Try again in a few minutes.") from e
            raise ResolveError("Couldn't load this video from YouTube.") from e
        return ResolvedVideo(ref.video_id, info.get("title") or "", float(info.get("duration") or 0), info.get("channel") or "", existing[0] if existing else None)

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        return decode_audio(video.audio_path)  # type: ignore[arg-type]


def decode_audio(path: Path, sr: int = SR_ANALYSIS) -> np.ndarray:
    """Decode any container to mono float32 at `sr` (PyAV)."""
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class DemoResolver:
    """No network: a synthetic 3-minute 'video' for the demo engine and CI."""

    def resolve(self, ref: VideoRef, cache_dir: Path) -> ResolvedVideo:
        return ResolvedVideo(ref.video_id, "Demo video", 180.0, "Maata demo", None)

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (0.01 * rng.standard_normal(int(video.duration * SR_ANALYSIS))).astype(np.float32)
