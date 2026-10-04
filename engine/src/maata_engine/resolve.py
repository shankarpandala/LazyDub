"""Stream resolution and media ingest (spec §6.1, ADR-010; OFFLINE-RENDER §2.2).

yt-dlp runs in-process with remote components and self-update disabled; it only talks to YouTube. A render fetches the
audio first (with the thumbnail), and the video-only stream later, beside the dub (`fetch_video`): the dubbed MP4 copies
that stream. Every YoutubeDL starts from `ytdlp_options` (scripts/check-youtube-video.sh imports them too).
"""

from __future__ import annotations

import copy
import functools
import itertools
import logging
import math
import os
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from .backends.base import SR_ANALYSIS
from .urls import VideoRef

log = logging.getLogger("maata.resolve")


class ResolveError(RuntimeError):
    """User-facing resolution failure."""


@dataclass(slots=True)
class ResolvedVideo:
    video_id: str
    title: str
    duration: float
    channel: str
    audio_path: Path | None
    # What the translation brief starts from (ARCHITECTURE §3.1, §4.2 brief v0): yt-dlp's metadata, as the uploader wrote it.
    description: str = ""
    chapters: tuple[tuple[float, str], ...] = ()  # (start s, title)
    tags: tuple[str, ...] = ()
    thumb_path: Path | None = None  # the thumbnail as YouTube serves it (the Library shows it from the cache)


@dataclass(frozen=True, slots=True)
class VideoFile:
    """The video stream the dubbed MP4 copies (§2.2). `owned`: the job's own download (`<cache>/<id>/video.<ext>`),
    deleted once the whole video's MP4 is written; never a file a resolver found elsewhere (the demo's, a bench's)."""

    path: Path
    owned: bool


Progress = Callable[[int, int], None]  # (bytes downloaded, bytes in all; 0 when unknown), from the downloading thread

# The audio: m4a first (AAC; YouTube's DASH audio). The video (§2.2): H.264 up to 1080p first (QuickTime plays it
# everywhere), then AV1 in MP4 (copied as is), then any video-only stream up to 1080p (VP9: copied with a warning), last
# a muxed H.264 format (its audio is ignored). Never `+` (a merge needs ffmpeg), never an unbounded `bv`.
AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio"
VIDEO_FORMAT = ("bv[vcodec^=avc1][height<=1080][ext=mp4]/bv[vcodec^=av01][height<=1080][ext=mp4]/bv[height<=1080]/"
                "b[vcodec^=avc1][ext=mp4][height<=1080]")
_IMAGES = (".jpg", ".jpeg", ".png", ".webp", ".gif")  # a thumbnail yt-dlp writes under the media's name, then moves
# s for which fetch's extract_info result serves the video download: YouTube's signed format URLs in it last about 6 h.
# Wall-clock time, which goes on while the Mac sleeps (a paused or slept job asks YouTube again).
INFO_FRESH = 3600.0
# YouTube's media servers now and then refuse a freshly signed URL (HTTP 403); new URLs a moment later download fine
# (2026-10-04, video 0eFln_mdXGY: the audio failed once and passed on the next try). So a 403 asks for the info again
# and retries, after these waits, before the user sees "Couldn't load this video".
FORBIDDEN_WAITS = (2.0, 5.0)
_sleep = time.sleep  # tests replace it


def ytdlp_options(cache_dir: Path, progress_hook: Callable[[dict], None] | None = None) -> dict:
    """What every YoutubeDL of the engine is made with: quiet, no config files, no remote components, and `fixup`
    "never", so the files are what YouTube sent whether or not an ffmpeg is on the PATH (a Finder launch has none)."""
    return {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreconfig": True,
        "remote_components": [],  # never fetch solver scripts from the network (confirm at pin)
        "fixup": "never",
        "cachedir": str(cache_dir / ".yt-dlp"),
        "progress_hooks": [progress_hook] if progress_hook is not None else [],
    }


def audio_options(out_dir: Path, cache_dir: Path, progress_hook: Callable[[dict], None] | None = None) -> dict:
    """fetch's YoutubeDL (§2.2): the audio as audio.<ext>, and the thumbnail as YouTube serves it (no conversion, so no
    ffmpeg) as thumb.<ext>."""
    return {**ytdlp_options(cache_dir, progress_hook), "format": AUDIO_FORMAT, "writethumbnail": True,
            "outtmpl": {"default": str(out_dir / "audio.%(ext)s"), "thumbnail": str(out_dir / "thumb.%(ext)s")}}


def video_options(out_dir: Path, cache_dir: Path, progress_hook: Callable[[dict], None] | None = None) -> dict:
    """The video stage's YoutubeDL (§2.2): the audio's options, with VIDEO_FORMAT, as video.<ext>, no thumbnail."""
    return {**audio_options(out_dir, cache_dir, progress_hook), "format": VIDEO_FORMAT, "writethumbnail": False,
            "outtmpl": str(out_dir / "video.%(ext)s")}


class Resolver(Protocol):
    def resolve(self, ref: VideoRef, cache_dir: Path, download: bool = True,
                progress: Progress | None = None) -> ResolvedVideo:
        """The video's metadata, and its audio in `cache_dir`/<id>/ (`audio_path`). With `download=False`, metadata only:
        `audio_path` is the audio already there, if any."""
        ...

    def load_audio(self, video: ResolvedVideo) -> np.ndarray: ...

    def fetch_video(self, video: ResolvedVideo, cache_dir: Path, progress: Progress | None = None) -> VideoFile:
        """The video stream the MP4 copies (§2.2): downloaded into `cache_dir`/<id>/ unless it is there already."""
        ...


class YtDlpResolver:
    def __init__(self) -> None:
        # video id -> (time.time() it came, fetch's extract_info result), for the video download (§2.2), until it is
        # used or INFO_FRESH has passed
        self._info: dict[str, tuple[float, dict]] = {}

    def resolve(self, ref: VideoRef, cache_dir: Path, download: bool = True,
                progress: Progress | None = None) -> ResolvedVideo:
        from yt_dlp import YoutubeDL

        out_dir = cache_dir / ref.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        opts = audio_options(out_dir, cache_dir, _hook(progress))
        with _errors(), YoutubeDL(opts) as ydl:
            info = ydl.extract_info(watch_url(ref.video_id), download=False)
            if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                raise ResolveError("Live streams and premieres aren't supported yet.")
            if (info.get("age_limit") or 0) >= 18:
                raise ResolveError("Age-restricted videos aren't supported.")
            if info.get("availability") in ("private", "premium_only", "subscriber_only", "needs_auth"):
                raise ResolveError("This video isn't public, so it can't be dubbed.")
            now = time.time()
            self._info = {k: x for k, x in self._info.items() if now - x[0] < INFO_FRESH}
            self._info[ref.video_id] = (now, info)
            existing = _audio_files(out_dir)
            if not existing and download:  # the audio, and the thumbnail
                info = _download(ydl.process_info, info, lambda: ydl.extract_info(watch_url(ref.video_id), download=False),
                                 "audio")
                self._info[ref.video_id] = (time.time(), info)  # new URLs after a retry: the video stage uses them
                existing = _audio_files(out_dir)
            elif download and not _media(out_dir, "thumb"):  # an audio fetched before thumbnails were: the thumbnail
                with YoutubeDL({**opts, "skip_download": True}) as thumbs:
                    thumbs.process_info(copy.deepcopy(info))
        thumbs = _media(out_dir, "thumb")
        return ResolvedVideo(ref.video_id, info.get("title") or "", float(info.get("duration") or 0), info.get("channel") or "",
                             existing[0] if existing else None, *metadata(info), thumb_path=thumbs[0] if thumbs else None)

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        return decode_audio(video.audio_path)  # type: ignore[arg-type]

    def fetch_video(self, video: ResolvedVideo, cache_dir: Path, progress: Progress | None = None) -> VideoFile:
        """A second YoutubeDL with VIDEO_FORMAT downloads video.<ext> from the info fetch's `extract_info` returned, kept
        in memory for INFO_FRESH s (its format URLs expire); a resume without it, or after that, asks once more. The
        total its progress reports is the chosen format's size (`filesize`, else `filesize_approx`). The kept info is
        let go of once the video is there."""
        from yt_dlp import YoutubeDL

        out_dir = cache_dir / video.video_id
        kept = self._info.get(video.video_id)
        if got := _media(out_dir, "video"):
            self._info.pop(video.video_id, None)
            return VideoFile(got[0], True)
        out_dir.mkdir(parents=True, exist_ok=True)
        with _errors(), YoutubeDL(video_options(out_dir, cache_dir, _hook(progress, sized=True))) as ydl:
            fresh = kept is not None and time.time() - kept[0] < INFO_FRESH
            info = kept[1] if fresh else ydl.extract_info(watch_url(video.video_id), download=False)
            _download(functools.partial(ydl.process_ie_result, download=True), info,
                      lambda: ydl.extract_info(watch_url(video.video_id), download=False), "video")
        self._info.pop(video.video_id, None)
        got = _media(out_dir, "video")
        if not got:
            raise ResolveError("YouTube didn't send this video's picture.")
        return VideoFile(got[0], True)


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _hook(progress: Progress | None, sized: bool = False) -> Callable[[dict], None]:
    """yt-dlp's progress hook as `progress(done, total)`. `sized`: the total is the chosen format's size (`filesize`,
    else `filesize_approx`) where the info has one. An exception `progress` raises (a pause) stops the download."""

    def hook(d: dict) -> None:
        if progress is None or d.get("status") not in ("downloading", "finished"):
            return
        info = d.get("info_dict") or {}
        total = int((sized and (info.get("filesize") or info.get("filesize_approx")))
                    or d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
        progress(int(d.get("downloaded_bytes") or total), total)

    return hook


def _download(fetch: Callable[[dict], object], info: dict, refresh: Callable[[], dict], what: str) -> dict:
    """`fetch(info)` (a copy each time), asking for new signed URLs (`refresh`) and trying again after each of
    FORBIDDEN_WAITS when YouTube answers HTTP 403. Returns the info the download used; any other error, or a 403 on the
    last try, goes on to `_errors`."""
    from yt_dlp.utils import DownloadError

    for attempt in range(len(FORBIDDEN_WAITS) + 1):
        try:
            fetch(copy.deepcopy(info))
            return info
        except DownloadError as err:
            if attempt == len(FORBIDDEN_WAITS) or "HTTP Error 403" not in str(err):
                raise
            log.warning("YouTube refused the %s download (HTTP 403); asking for new URLs (retry %d of %d)",
                        what, attempt + 1, len(FORBIDDEN_WAITS))
            _sleep(FORBIDDEN_WAITS[attempt])
            info = refresh()
    return info


class _errors:
    """yt-dlp's download errors as the messages the user sees."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind, err, tb) -> bool:
        from yt_dlp.utils import DownloadError

        if not isinstance(err, DownloadError):
            return False
        msg = str(err)
        if "Private video" in msg:
            raise ResolveError("This video is private.") from err
        if "Sign in to confirm" in msg:
            raise ResolveError("YouTube is asking for a sign-in check. Try again in a few minutes.") from err
        raise ResolveError("Couldn't load this video from YouTube.") from err


def _media(out_dir: Path, stem: str) -> list[Path]:
    """The finished files `stem`.* in `out_dir`: never yt-dlp's partial files (<stem>.<ext>.part, .ytdl) of a download
    that was stopped, and (but for the thumbnail) never an image: yt-dlp writes the thumbnail under the media's name
    first and moves it to thumb.<ext> once the download is done."""
    return sorted(p for p in out_dir.glob(f"{stem}.*") if ".part" not in p.name and not p.name.endswith(".ytdl")
                  and (stem == "thumb") == (p.suffix.lower() in _IMAGES))


def _audio_files(out_dir: Path) -> list[Path]:
    """The downloaded audio in `out_dir`, never yt-dlp's partial files (audio.m4a.part, audio.m4a.ytdl) of a download
    that was stopped, nor a thumbnail on its way to thumb.<ext>."""
    return _media(out_dir, "audio")


def metadata(info: dict) -> tuple[str, tuple[tuple[float, str], ...], tuple[str, ...]]:
    """The description, chapters and tags of a yt-dlp info dict; anything missing or malformed is left out."""
    chapters = tuple((float(c["start_time"]), str(c["title"]).strip()) for c in info.get("chapters") or []
                     if isinstance(c, dict) and isinstance(c.get("start_time"), (int, float))
                     and str(c.get("title") or "").strip())
    tags = tuple(dict.fromkeys(t.strip() for t in info.get("tags") or [] if isinstance(t, str) and t.strip()))
    description = info.get("description")
    return (description.strip() if isinstance(description, str) else ""), chapters, tags


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


def decode_audio_to(src: Path, dest: Path, sr: int = SR_ANALYSIS) -> int:
    """Decode any container to mono float32 at `sr` straight into the raw file `dest` (PyAV), frame by frame: no more
    than a frame is held in memory, where `decode_audio` holds the whole video about three times at its peak
    (OFFLINE-RENDER §2.2). It goes to `dest`.part, renamed once complete, so `dest` is never half written. Sample 0 is
    the first decoded sample (`audio_start` says where that is in the stream), and the decode stops at the stream's
    declared duration where it has one: the encoder's padding after the last frame (AAC's, up to a frame) isn't the
    video's. Returns the number of samples written."""
    import av

    part = dest.with_name(dest.name + ".part")
    n = 0
    with av.open(str(src)) as container, part.open("wb") as f:
        stream = container.streams.audio[0]
        limit = math.ceil(float(stream.duration * stream.time_base) * sr) if stream.duration else None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in itertools.chain(container.decode(stream), [None]):  # None flushes the resampler
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if limit is not None:
                    samples = samples[:max(limit - n, 0)]
                f.write(samples.tobytes())
                n += len(samples)
    os.replace(part, dest)
    return n


def decode_stereo(src: Path, sr: int = 44_100) -> Iterator[np.ndarray]:
    """`src`'s audio decoded to stereo float32 at `sr`, frame by frame ([2, m] each): the separator's input (§2.14).
    Samples are counted as `decode_audio_to` counts them: sample 0 is the first decoded sample, and the decode stops at
    the stream's declared duration where it has one, so sample i is the same instant at either rate. A mono stream is
    duplicated to both channels; more channels are mixed down to two by the resampler. Close the generator to stop
    early."""
    import av

    with av.open(str(src)) as container:
        stream = container.streams.audio[0]
        limit = math.ceil(float(stream.duration * stream.time_base) * sr) if stream.duration else None
        mono = stream.codec_context.layout.nb_channels == 1
        resampler = av.AudioResampler(format="fltp", layout="mono" if mono else "stereo", rate=sr)
        n = 0
        for frame in itertools.chain(container.decode(stream), [None]):  # None flushes the resampler
            for out in resampler.resample(frame):
                x = out.to_ndarray().astype(np.float32, copy=False)
                if mono:
                    x = np.repeat(x, 2, axis=0)
                if limit is not None:
                    x = x[:, :max(limit - n, 0)]
                if x.shape[1]:
                    n += x.shape[1]
                    yield x


def audio_start(src: Path) -> float:
    """The time in its stream of the first sample a decode of `src`'s audio gives (after libavformat applies the edit
    list and the AAC priming): the one origin of every decode (§2.2). Dub time t is that much after the stream's 0."""
    import av

    with av.open(str(src)) as container:
        for frame in container.decode(container.streams.audio[0]):
            return float(frame.time or 0.0)
    return 0.0


# ---- the demo, a local file ------------------------------------------------------------------------------------------
DEMO_SECONDS = 180.0
DEMO_FPS = 10
DEMO_SIZE = (320, 180)
DEMO_MARK = 10.0  # s: the white frame and the start of the 1 kHz beep
# Colour bars in BT.601 limited-range Y, U, V (grey, yellow, cyan, green, magenta, red, blue).
_BARS = ((180, 128, 128), (168, 44, 136), (145, 147, 44), (133, 63, 52), (63, 193, 204), (51, 109, 212), (28, 212, 120))


def synth_video(dest: Path, seconds: float = DEMO_SECONDS, *, video_codec: str | None = "libx264",
                audio: bool = True, audio_start: float = 0.0, video_start: float = 0.0, container: str = "mp4",
                codec_options: dict | None = None) -> None:
    """A synthetic test card, written with PyAV only: colour bars rolling 2 px a frame at DEMO_FPS with one white frame
    at exactly DEMO_MARK s (`video_codec`, with its defaults: libx264's B-frames, so packets have dts != pts; None: no
    video), and AAC 44.1 kHz stereo (`audio`): a soft chord with a 1 kHz beep from exactly DEMO_MARK s to half a second
    after it. `audio_start`: the audio stream's first sample is at that time in its stream (YouTube's DASH audio may
    start late); `video_start`: the video's first frame is (its frames before it are left out). Written to `dest`.part,
    then put in place."""
    import av

    from .export import aac

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = DEMO_SIZE
    frames = int(round(seconds * DEMO_FPS))
    with av.open(str(part), "w", format=container) as out:
        v = a = None
        if video_codec is not None:
            v = out.add_stream(video_codec, rate=DEMO_FPS, options=codec_options or {})
            v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        if audio:
            a = out.add_stream(aac(), rate=44_100)
            a.layout = "stereo"
        cols = np.arange(w) * len(_BARS) // w
        planes = [np.asarray([c[i] for c in _BARS], np.uint8)[cols] for i in range(3)]
        sr, block = 44_100, 4_410  # one frame's audio
        first = int(round(audio_start * sr))
        late = int(round(video_start * DEMO_FPS))  # the first frame
        for k in range(frames):
            if v is not None and k >= late:
                if k == int(round(DEMO_MARK * DEMO_FPS)):
                    y, u, vv = np.full((h, w), 235, np.uint8), np.full((h // 2, w // 2), 128, np.uint8), \
                        np.full((h // 2, w // 2), 128, np.uint8)
                else:
                    y = np.broadcast_to(np.roll(planes[0], 2 * k), (h, w))
                    u, vv = (np.broadcast_to(np.roll(p[::2], k), (h // 2, w // 2)) for p in planes[1:])
                frame = av.VideoFrame.from_ndarray(np.concatenate([y.ravel(), u.ravel(), vv.ravel()]).reshape(-1, w),
                                                   format="yuv420p")
                frame.pts = k
                for p in v.encode(frame):
                    out.mux(p)
            if a is not None:
                n = np.arange(first + k * block, first + (k + 1) * block)
                t = n / sr
                s = 0.02 * (np.sin(2 * np.pi * 220.0 * t) + np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t))
                s = s + np.where((n >= int(DEMO_MARK * sr)) & (t < DEMO_MARK + 0.5),
                                 0.3 * np.sin(2 * np.pi * 1000.0 * (n - int(DEMO_MARK * sr)) / sr), 0.0)
                af = av.AudioFrame.from_ndarray(np.stack([s, s]).astype(np.float32), format="fltp", layout="stereo")
                af.sample_rate, af.pts, af.time_base = sr, int(n[0]), Fraction(1, sr)
                for p in a.encode(af):
                    out.mux(p)
        for s in (v, a):
            if s is not None:
                for p in s.encode(None):
                    out.mux(p)
    os.replace(part, dest)


@functools.lru_cache(maxsize=1)
def _demo_bytes() -> bytes:
    """The demo video, made once a process (a few seconds of encoding): every demo job folder gets a copy."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.mp4"
        synth_video(path)
        return path.read_bytes()


def write_demo(dest: Path) -> Path:
    """`dest` (the demo's demo.mp4), written once."""
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(_demo_bytes())
        os.replace(part, dest)
    return dest


class DemoResolver:
    """No network: a synthetic 3-minute video for the demo engine and CI, <cache>/<id>/demo.mp4 (`synth_video`), written
    once. It is both the render's audio and its video, never owned by the job (never deleted). `load_audio` (noise of
    the video's length) stands in only when there is no file to decode."""

    def resolve(self, ref: VideoRef, cache_dir: Path, download: bool = True,
                progress: Progress | None = None) -> ResolvedVideo:
        path = cache_dir / ref.video_id / "demo.mp4"
        if download:
            write_demo(path)
        return ResolvedVideo(ref.video_id, "Demo video", DEMO_SECONDS, "Maata demo", path if path.is_file() else None,
                             "A synthetic video for the demo engine.", ((0.0, "Start"),), ("demo",))

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (0.01 * rng.standard_normal(int(video.duration * SR_ANALYSIS))).astype(np.float32)

    def fetch_video(self, video: ResolvedVideo, cache_dir: Path, progress: Progress | None = None) -> VideoFile:
        return VideoFile(write_demo(cache_dir / video.video_id / "demo.mp4"), False)


class LocalResolver:
    """A local video file outside the cache (maata-bench): its audio and its video, never owned by the job. The job
    records its absolute path, so a resume finds it, and nothing deletes it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def resolve(self, ref: VideoRef, cache_dir: Path, download: bool = True,
                progress: Progress | None = None) -> ResolvedVideo:
        import av

        if not self.path.is_file():
            raise ResolveError(f"There is no file at {self.path}.")
        with av.open(str(self.path)) as c:
            duration = float(c.duration or 0) / 1e6
        return ResolvedVideo(ref.video_id, self.path.stem, duration, "local", self.path)

    def load_audio(self, video: ResolvedVideo) -> np.ndarray:
        return decode_audio(self.path)

    def fetch_video(self, video: ResolvedVideo, cache_dir: Path, progress: Progress | None = None) -> VideoFile:
        return VideoFile(self.path, False)
