"""Fetching the media (OFFLINE-RENDER §2.2): yt-dlp's two instances (the audio with its thumbnail, then the video beside
the dub) with the engine's options, checked on a monkeypatched YoutubeDL (no network); partial files and a thumbnail on
its way ignored; the demo's test card (a beep and a white frame at exactly 10.0 s, B-frames); the local file; and the
one origin of every decode."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from fakes import AUDIO_START, YouTubeShaped, onset, white_frames  # noqa: E402

from maata_engine import resolve  # noqa: E402
from maata_engine.resolve import (VIDEO_FORMAT, DemoResolver, LocalResolver, YtDlpResolver, audio_start,  # noqa: E402
                                  decode_audio_to, synth_video)
from maata_engine.urls import VideoRef  # noqa: E402

VID = "abcdefghijk"
INFO = {"id": VID, "title": "A talk", "duration": 600, "channel": "A channel", "description": " About it. ",
        "tags": ["talk"], "formats": [{"format_id": "137"}, {"format_id": "140"}]}


class FakeYDL:
    """yt-dlp's YoutubeDL as the resolver uses it: every instance's options, the calls, and 'downloads' that write what
    yt-dlp writes (its progress hook called on the way)."""

    made: list["FakeYDL"] = []
    refuse: list[str] = []  # errors the next downloads raise, in order (a 403 YouTube's media servers answer now and then)

    def __init__(self, opts: dict) -> None:
        self.opts, self.calls = opts, []
        FakeYDL.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def _out(self, name: str) -> Path:
        tmpl = self.opts["outtmpl"]
        return Path(tmpl["default"] if isinstance(tmpl, dict) else tmpl).with_name(name)

    def _hook(self, done: int, total: int, info: dict) -> None:
        for h in self.opts["progress_hooks"]:
            h({"status": "downloading", "downloaded_bytes": done, "total_bytes": total, "info_dict": info})

    def extract_info(self, url: str, download: bool = True) -> dict:
        self.calls.append(("extract_info", url, download))
        return copy.deepcopy(INFO)

    def _refused(self) -> None:
        if FakeYDL.refuse:
            from yt_dlp.utils import DownloadError

            raise DownloadError(FakeYDL.refuse.pop(0))

    def process_info(self, info: dict) -> None:
        self.calls.append(("process_info", self.opts.get("skip_download", False)))
        if not self.opts.get("skip_download"):
            self._refused()
        out = self._out("x")
        if self.opts.get("writethumbnail"):  # yt-dlp writes it under the media's name, and moves it once done
            out.with_name("thumb.webp").write_bytes(b"RIFF thumb")
        if not self.opts.get("skip_download"):
            self._hook(1000, 4000, {"filesize": 5000})
            out.with_name("audio.m4a").write_bytes(b"audio")

    def process_ie_result(self, info: dict, download: bool = True) -> dict:
        self.calls.append(("process_ie_result", info.get("id"), download))
        self._refused()
        self._hook(2_000_000, 3_000_000, {"filesize": None, "filesize_approx": 9_000_000})
        self._out("x").with_name("video.mp4").write_bytes(b"video")
        return info


@pytest.fixture
def ydl(monkeypatch):
    yt_dlp = pytest.importorskip("yt_dlp")  # the `resolve` extra; CI installs it, a bare `uv sync --group dev` skips these

    FakeYDL.made, FakeYDL.refuse = [], []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


FORBIDDEN = "ERROR: unable to download video data: HTTP Error 403: Forbidden"


@pytest.fixture
def waits(monkeypatch):
    got: list[float] = []
    monkeypatch.setattr(resolve, "_sleep", got.append)
    return got


def test_a_403_on_the_audio_asks_for_new_urls_and_tries_again(tmp_path, ydl, waits):
    ydl.refuse = [FORBIDDEN]
    v = YtDlpResolver().resolve(VideoRef(VID), tmp_path)
    assert v.audio_path == tmp_path / VID / "audio.m4a"
    assert [c[0] for c in ydl.made[0].calls] == ["extract_info", "process_info", "extract_info", "process_info"]
    assert waits == [resolve.FORBIDDEN_WAITS[0]]


def test_a_403_on_the_video_asks_for_new_urls_and_tries_again(tmp_path, ydl, waits):
    r = YtDlpResolver()
    v = r.resolve(VideoRef(VID), tmp_path)
    ydl.refuse = [FORBIDDEN, FORBIDDEN]
    assert r.fetch_video(v, tmp_path).path == tmp_path / VID / "video.mp4"
    assert [c[0] for c in ydl.made[-1].calls] == ["process_ie_result", "extract_info", "process_ie_result",
                                                  "extract_info", "process_ie_result"]
    assert waits == list(resolve.FORBIDDEN_WAITS)


def test_a_403_every_time_still_fails_with_the_message_the_user_sees(tmp_path, ydl, waits):
    ydl.refuse = [FORBIDDEN] * (len(resolve.FORBIDDEN_WAITS) + 1)
    with pytest.raises(resolve.ResolveError, match="Couldn't load this video from YouTube"):
        YtDlpResolver().resolve(VideoRef(VID), tmp_path)
    assert waits == list(resolve.FORBIDDEN_WAITS) and not (tmp_path / VID / "audio.m4a").exists()


def test_other_download_errors_are_not_retried(tmp_path, ydl, waits):
    ydl.refuse = ["ERROR: unable to download video data: HTTP Error 404: Not Found"]
    with pytest.raises(resolve.ResolveError, match="Couldn't load this video from YouTube"):
        YtDlpResolver().resolve(VideoRef(VID), tmp_path)
    assert waits == [] and [c[0] for c in ydl.made[0].calls] == ["extract_info", "process_info"]


def test_the_audio_and_the_video_are_two_youtubedls_with_the_engines_options(tmp_path, ydl):
    r = YtDlpResolver()
    seen: list[tuple[int, int]] = []
    v = r.resolve(VideoRef(VID), tmp_path, progress=lambda d, t: seen.append((d, t)))
    d = tmp_path / VID
    assert v.audio_path == d / "audio.m4a" and v.thumb_path == d / "thumb.webp" and v.description == "About it."
    audio = ydl.made[0]
    assert [c[0] for c in audio.calls] == ["extract_info", "process_info"]
    assert audio.opts["outtmpl"] == {"default": str(d / "audio.%(ext)s"), "thumbnail": str(d / "thumb.%(ext)s")}
    assert audio.opts["writethumbnail"] is True and audio.opts["format"] == "bestaudio[ext=m4a]/bestaudio"
    assert seen == [(1000, 4000)]  # the audio's own total
    seen.clear()
    kept = r._info[VID][1]
    got = r.fetch_video(v, tmp_path, progress=lambda d, t: seen.append((d, t)))
    video = ydl.made[-1]
    assert got == resolve.VideoFile(d / "video.mp4", True)
    # the info fetch's extract_info returned, kept: no second extract_info; a copy, so it stays as it was; then let go
    assert video.calls == [("process_ie_result", VID, True)] and kept == INFO and VID not in r._info
    assert video.opts["format"] == VIDEO_FORMAT and video.opts["outtmpl"] == str(d / "video.%(ext)s")
    assert video.opts["writethumbnail"] is False
    assert seen == [(2_000_000, 9_000_000)]  # the chosen format's size (filesize, else filesize_approx)
    for x in ydl.made:  # both: never a fixup that depends on an ffmpeg being on the PATH, no remote components
        assert x.opts["fixup"] == "never" and x.opts["remote_components"] == [] and "+" not in x.opts["format"]
        assert x.opts["ignoreconfig"] is True and x.opts["noplaylist"] is True
    assert VIDEO_FORMAT == ("bv[vcodec^=avc1][height<=1080][ext=mp4]/bv[vcodec^=av01][height<=1080][ext=mp4]/"
                            "bv[height<=1080]/b[vcodec^=avc1][ext=mp4][height<=1080]")
    # a resume (a new resolver, no info in memory) asks once more; a file already there is not downloaded again
    again = YtDlpResolver()
    assert again.fetch_video(v, tmp_path) == got and len(ydl.made) == 2  # there: no YoutubeDL at all
    (d / "video.mp4").unlink()
    again.fetch_video(v, tmp_path)
    assert [c[0] for c in ydl.made[-1].calls] == ["extract_info", "process_ie_result"]


def test_the_kept_info_serves_the_video_only_while_youtubes_urls_last(tmp_path, ydl):
    r = YtDlpResolver()
    v = r.resolve(VideoRef(VID), tmp_path)
    at, info = r._info[VID]
    r._info[VID] = (at - resolve.INFO_FRESH - 1, info)  # paused, or the Mac asleep, past the window: its URLs expired
    r.fetch_video(v, tmp_path)
    assert [c[0] for c in ydl.made[-1].calls] == ["extract_info", "process_ie_result"] and VID not in r._info
    # a video already there lets go of it too; and a resolve drops what is past the window, whatever video it was for
    r.resolve(VideoRef(VID), tmp_path)
    assert r.fetch_video(v, tmp_path).path.name == "video.mp4" and VID not in r._info
    r._info["zyxwvutsrqp"] = (at - resolve.INFO_FRESH - 1, info)
    r.resolve(VideoRef(VID), tmp_path)
    assert list(r._info) == [VID]


def test_an_audio_fetched_before_thumbnails_gets_one_without_downloading_again(tmp_path, ydl):
    d = tmp_path / VID
    d.mkdir()
    (d / "audio.m4a").write_bytes(b"audio")
    v = YtDlpResolver().resolve(VideoRef(VID), tmp_path)
    assert v.thumb_path == d / "thumb.webp" and [x.opts.get("skip_download") for x in ydl.made] == [None, True]
    assert ydl.made[1].calls == [("process_info", True)]
    assert YtDlpResolver().resolve(VideoRef(VID), tmp_path, download=False).thumb_path == d / "thumb.webp"


def test_partial_files_and_a_thumbnail_on_its_way_are_never_the_media(tmp_path):
    for name in ("audio.m4a.part", "audio.m4a.ytdl", "audio.webp", "video.mp4.part", "video.mp4.part-Frag3",
                 "video.webp", "thumb.webp.part"):
        (tmp_path / name).write_bytes(b"x")
    assert resolve._media(tmp_path, "audio") == [] and resolve._media(tmp_path, "video") == []
    assert resolve._media(tmp_path, "thumb") == []
    (tmp_path / "audio.webm").write_bytes(b"x")
    (tmp_path / "video.mp4").write_bytes(b"x")
    (tmp_path / "thumb.jpg").write_bytes(b"x")
    assert resolve._audio_files(tmp_path) == [tmp_path / "audio.webm"]
    assert resolve._media(tmp_path, "video") == [tmp_path / "video.mp4"]
    assert resolve._media(tmp_path, "thumb") == [tmp_path / "thumb.jpg"]


def test_the_demo_is_a_test_card_with_its_beep_and_white_frame_at_exactly_ten_seconds(tmp_path):
    r = DemoResolver()
    v = r.resolve(VideoRef(VID), tmp_path, download=False)
    assert v.audio_path is None  # metadata only: nothing written
    v = r.resolve(VideoRef(VID), tmp_path)
    path = tmp_path / VID / "demo.mp4"
    assert v.audio_path == path and r.fetch_video(v, tmp_path) == resolve.VideoFile(path, False)  # never owned
    mtime = path.stat().st_mtime_ns
    r.resolve(VideoRef(VID), tmp_path)
    assert path.stat().st_mtime_ns == mtime  # written once
    with av.open(str(path)) as c:
        vs, a = c.streams.video[0], c.streams.audio[0]
        assert (vs.codec_context.width, vs.codec_context.height, float(vs.average_rate)) == (320, 180, 10.0)
        assert float(vs.duration * vs.time_base) == 180.0
        assert (a.codec_context.sample_rate, a.codec_context.channels) == (44_100, 2)
        packets = [p for p in c.demux(vs) if p.size]
    assert any(p.dts != p.pts for p in packets) and sum(p.is_keyframe for p in packets) > 1  # B-frames, GOPs
    assert white_frames(path) == [10.0]
    assert audio_start(path) == 0.0
    with av.open(str(path)) as c:
        x = np.concatenate([f.to_ndarray()[0] for f in c.decode(c.streams.audio[0])])
    assert abs(onset(x, 44_100) - 10.0) <= 0.001


def test_the_two_file_fake_starts_its_audio_late_and_every_decode_counts_from_its_first_sample(tmp_path):
    r = YouTubeShaped()
    v = r.resolve(VideoRef(VID), tmp_path)
    video = r.fetch_video(v, tmp_path)
    assert video.owned and white_frames(video.path) == [10.0]
    with av.open(str(video.path)) as c:
        assert not c.streams.audio and any(p.dts != p.pts for p in c.demux(c.streams.video[0]) if p.size)
    start = audio_start(v.audio_path)
    assert 0.0 < start <= AUDIO_START  # (the AAC encoder's priming is decoded too, before the stream's first sample)
    dest = tmp_path / "a16.f32"
    decode_audio_to(v.audio_path, dest, 16_000)
    x = np.fromfile(dest, np.float32)
    # the beep is at 10.0 s in the stream: start + its place in the decode
    assert abs(start + onset(x, 16_000) - 10.0) <= 0.001


def test_a_decode_stops_at_the_streams_declared_duration(tmp_path):
    path = tmp_path / "card.mp4"
    synth_video(path, 12.0)
    n = decode_audio_to(path, tmp_path / "a.f32", 16_000)
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        declared = float(s.duration * s.time_base)
        decoded = sum(f.samples for f in c.decode(s)) / 44_100
    assert decoded > declared  # the encoder's padding after the last frame
    assert n == pytest.approx(declared * 16_000, abs=2)


def test_a_local_file_is_read_in_place_and_never_owned(tmp_path):
    path = tmp_path / "talk.mp4"
    synth_video(path, 12.0)
    r = LocalResolver(path)
    v = r.resolve(VideoRef(VID), tmp_path / "cache")
    assert v.audio_path == path.resolve() and v.title == "talk" and v.duration == pytest.approx(12.0, abs=0.05)
    assert r.fetch_video(v, tmp_path / "cache") == resolve.VideoFile(path.resolve(), False)
    assert not (tmp_path / "cache").exists()  # nothing copied into the cache
    with pytest.raises(resolve.ResolveError):
        LocalResolver(tmp_path / "missing.mp4").resolve(VideoRef(VID), tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="AudioToolbox")
def test_aac_at_is_the_encoder_on_macos():
    from maata_engine import export

    assert export.aac() == "aac_at"


def test_resolve_keeps_the_metadata_the_brief_starts_from():
    info = {"description": "  How to fold a paper boat.\n", "tags": ["origami", " origami", "", 5, "boats"],
            "chapters": [{"start_time": 0, "title": "Intro"}, {"start_time": 42.5, "title": " Folding "},
                         {"start_time": None, "title": "Broken"}, {"title": "No start"}]}
    assert resolve.metadata(info) == ("How to fold a paper boat.", ((0.0, "Intro"), (42.5, "Folding")),
                                      ("origami", "boats"))
    assert resolve.metadata({}) == ("", (), ())
