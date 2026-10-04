"""Synthetic media and a YouTube-shaped resolver for the tests (OFFLINE-RENDER §2.2, §9): no downloads, PyAV and numpy
only. The two-file fake has what YouTube's DASH files have and the demo's single file hasn't: a video-only video.mp4
(libx264 with B-frames) and an audio-only audio.m4a whose first sample comes after its stream's 0 (an edit list, with
the encoder's priming). And a render job with no stage run, for the tests of its lines one at a time (`bare_job`)."""

from __future__ import annotations

import functools
import json
import tempfile
from pathlib import Path

import numpy as np

from maata_engine.backends.base import SR_ANALYSIS, Backend
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.dubber import UnitState
from maata_engine.render import RenderJob, RenderSettings, _said_of
from maata_engine.resolve import DEMO_SECONDS, DemoResolver, ResolvedVideo, VideoFile, synth_video
from maata_engine.timing.planner import Plan, TimelinePlanner

AUDIO_START = 0.25  # s: where the fake's audio stream's first sample is meant to be in its stream


def onset(x: np.ndarray, sr: int, hz: float = 1000.0, window: float = 0.001) -> float:
    """When a `hz` tone starts in `x`: a narrow quadrature detector (a centred `window` s average of x·e^(-iωt)) first
    over half its peak, in seconds from x's first sample."""
    t = np.arange(len(x)) / sr
    z = x * np.exp(-2j * np.pi * hz * t)
    n = max(1, int(round(window * sr)))
    mag = np.abs(np.convolve(z, np.ones(n) / n, mode="same"))
    return float(np.argmax(mag > mag.max() / 2)) / sr


def white_frames(path: Path) -> list[float]:
    """The times of the video's white frames (luma over 220)."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        h = s.codec_context.height
        return [float(f.time) for f in c.decode(s) if f.to_ndarray(format="yuv420p")[:h].mean() > 220]


def decode(path: Path, sr: int = 44_100) -> np.ndarray:
    """The first audio stream, mono float32 at `sr`."""
    import av

    out = []
    with av.open(str(path)) as c:
        r = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in c.decode(c.streams.audio[0]):
            out += [f.to_ndarray().reshape(-1) for f in r.resample(frame)]
        out += [f.to_ndarray().reshape(-1) for f in r.resample(None)]
    return np.concatenate(out)


@functools.lru_cache(maxsize=None)
def _two_files() -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        audio, video = Path(tmp) / "audio.m4a", Path(tmp) / "video.mp4"
        synth_video(audio, video_codec=None, audio_start=AUDIO_START)
        synth_video(video, audio=False)
        return audio.read_bytes(), video.read_bytes()


class YouTubeShaped:
    """A resolver whose 'downloads' are two files in the cache, as yt-dlp leaves them: audio.m4a at fetch (with a
    thumbnail), video.mp4 at the video stage (owned by the job). It counts its video stage's calls (`videos`) and the
    video files it wrote (`downloads`)."""

    def __init__(self) -> None:
        self.videos = 0
        self.downloads = 0
        self.resolves = 0

    def resolve(self, ref, cache_dir: Path, download: bool = True, progress=None) -> ResolvedVideo:
        self.resolves += 1
        d = cache_dir / ref.video_id
        audio = d / "audio.m4a"
        if download and not audio.exists():
            d.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(_two_files()[0])
            (d / "thumb.jpg").write_bytes(b"\xff\xd8\xff thumbnail")
        return ResolvedVideo(ref.video_id, "A two-file video", DEMO_SECONDS, "Maata tests", audio if audio.exists() else None,
                             "Two files, as YouTube serves them.", ((0.0, "Start"),), ("test",),
                             thumb_path=d / "thumb.jpg" if (d / "thumb.jpg").exists() else None)

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        raise AssertionError("a render decodes the audio file")

    def fetch_video(self, video: ResolvedVideo, cache_dir: Path, progress=None) -> VideoFile:
        self.videos += 1
        path = cache_dir / video.video_id / "video.mp4"
        data = _two_files()[1]
        if not path.exists():
            if progress is not None:
                for k in range(1, 5):
                    progress(len(data) * k // 4, len(data))
            path.write_bytes(data)
            self.downloads += 1
        return VideoFile(path, True)


VIDEO_URL = "https://youtu.be/dQw4w9WgXcQ"


def bare_job(tmp_path, tts=None, transcriber=None, translator=MockSceneTranslator, events: list | None = None,
             seconds: float = 600.0, **settings) -> RenderJob:
    """A render job with no stage run, for its lines one at a time: the mock backend (with `tts`, `transcriber` and
    `translator` in place of the mock's), RenderSettings(**settings), its render folder made, `seconds` of silent audio
    and its events in `events`. Its lines go in with `put`; `job.tr` is set by the test that needs one."""
    async def on_event(msg: dict) -> None:
        if events is not None:
            events.append(msg)

    b = Backend("mock", "cpu", transcriber or MockTranscriber(), MockDiarizer(), translator, tts or MockTTS())
    job = RenderJob(b, DemoResolver(), tmp_path, VIDEO_URL, RenderSettings(**settings), on_event=on_event)
    job.render_dir.mkdir(parents=True, exist_ok=True)
    job.audio = np.zeros(int(seconds * SR_ANALYSIS), np.float32)
    return job


def put(job: RenderJob, *states: UnitState) -> None:
    """Add lines to the job's units, indexed in onset order as the units stage indexes them."""
    job._index(sorted([*job.units.values(), *states], key=lambda st: (st.unit.start, st.unit.id)))


async def voice(job: RenderJob, st: UnitState) -> Plan:
    """A line voiced as the dub loop voices it (`_voice_line`: its take placed on W and kept), then made final as the
    final plan places it on its own (`_final_pcm`: its PCM and its `unit` trace event). Returns that final plan."""
    await job._voice_line(st)
    plan = job._plan(st, _said_of(st.take), TimelinePlanner(job.planner.s))
    await job._final_pcm(st, plan, job._pcm_key(st, plan))
    return plan


def pcm(job: RenderJob, st: UnitState) -> np.ndarray:
    """A final line's PCM, as `voice` saved it."""
    return np.load(job.render_dir / "pcm" / f"{job.pcm[st.unit.id][0]}.npy").astype(np.float32)


def trace(job: RenderJob, event: str | None = None) -> list[dict]:
    """The job's units.jsonl, or its events of one kind."""
    path = job._dir / "units.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    return [r for r in rows if event is None or r["event"] == event]
