"""The dubbed MP4 (OFFLINE-RENDER §2.17, §9), on the demo's single file and on the YouTube-shaped two files: the video
copied packet for packet, the Telugu AAC, the two mov_text tracks, A/V sync within 2 ms, faststart, the metadata, the
preview's cut at a keyframe, AV1 and VP9 copied, names, a pause that leaves no file, and, through a render job, the
sources a whole export lets go of, a re-run served without downloading, a bench's input left as it was, and the
background sound (MockSeparator: half the source) under the voices, in sync with the picture, never separated again for
an unchanged re-run. Synthetic media only (PyAV and numpy); the mock backend, no models, no Claude."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from fakes import YouTubeShaped, decode, onset, white_frames  # noqa: E402

from maata_engine import export as mp4  # noqa: E402
from maata_engine import mix, render, subtitles  # noqa: E402
from maata_engine.backends.base import Backend, Cancelled  # noqa: E402
from maata_engine.backends.mock import (MockDiarizer, MockSceneTranslator, MockSeparator, MockTranscriber,  # noqa: E402
                                        MockTTS)
from maata_engine.render import RenderJob, RenderSettings  # noqa: E402
from maata_engine.resolve import DemoResolver, LocalResolver, audio_start, synth_video  # noqa: E402
from maata_engine.subtitles import Cue  # noqa: E402
from maata_engine.urls import VideoRef  # noqa: E402

pytestmark = pytest.mark.product_aac  # the export's own tests encode as the product does (conftest)

SR = 24_000
VID = "demo0000001"
META = {"title": "A talk (Telugu)", "artist": "A channel", "comment": "Telugu dub by Maata (AI voices) of "
        "https://youtu.be/demo0000001", "date": "2026-10-04"}


def sources(tmp_path, kind: str) -> tuple[Path, float]:
    """The video to copy and the audio's start: the demo's one file, or the two-file fake's."""
    if kind == "demo":
        r = DemoResolver()
        v = r.resolve(VideoRef(VID), tmp_path)
        return r.fetch_video(v, tmp_path).path, audio_start(v.audio_path)
    r = YouTubeShaped()
    v = r.resolve(VideoRef(VID), tmp_path)
    return r.fetch_video(v, tmp_path).path, audio_start(v.audio_path)


def click_line(tmp_path, name: str, start: float, speaker: str = "S1", seconds: float = 1.0) -> mix.Line:
    """A line whose PCM is a click at its first sample, then a quiet tone."""
    t = np.arange(int(seconds * SR)) / SR
    x = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x[0] = 0.9
    path = tmp_path / f"{name}.npy"
    np.save(path, x.astype(np.float16))
    return mix.Line(start, speaker, path, len(x))


def packets(path: Path) -> list[bytes]:
    with av.open(str(path)) as c:
        return [bytes(p) for p in c.demux(c.streams.video[0]) if p.size]


def cues_of(path: Path, index: int) -> list[tuple[int, int, str]]:
    """A subtitle track's non-empty samples: (pts ms, duration ms, text)."""
    with av.open(str(path)) as c:
        s = c.streams.subtitles[index]
        out = []
        for p in c.demux(s):
            data = bytes(p)
            if p.size and len(data) > 2:
                out.append((int(p.pts * s.time_base * 1000), int(p.duration * s.time_base * 1000),
                            data[2:2 + int.from_bytes(data[:2], "big")].decode("utf-8")))
        return out


def do_export(dest: Path, video: Path, start: float, lines: list[mix.Line], stop: float | None = None, **kw) -> dict:
    telugu = [{"start": x.start, "samples": x.samples, "speaker": x.speaker, "telugu": "మళ్ళీ ఛానల్ కి స్వాగతం"}
              for x in lines]
    english = [Cue(x.start, x.start + 0.9, x.speaker, "Welcome back to the channel.") for x in lines]
    return mp4.export(dest, video=video, lines=lines, voice_sr=SR, telugu=subtitles.telugu_cues(telugu, SR),
                      english=english, audio_start=start, stop=stop, metadata=META, **kw)


@pytest.mark.parametrize("kind", ["demo", "two files"])
def test_the_mp4_copies_the_video_and_carries_the_telugu_audio_and_two_subtitle_tracks(tmp_path, kind):
    video, start = sources(tmp_path, kind)
    off = start - 0.0  # the source video starts at 0: dub time t plays at t + audioStart
    flash = white_frames(video)[0]
    lines = [click_line(tmp_path, "a", flash - off), click_line(tmp_path, "b", 40.0, "S2", 3.0),
             click_line(tmp_path, "c", 179.5, "S1", 2.0)]  # the last runs 1.5 s past the video's end
    dest = tmp_path / "out" / "A talk (Telugu).mp4"
    dest.parent.mkdir()
    seen: list[tuple] = []
    got = do_export(dest, video, start, lines, progress=lambda d, t, x: seen.append((d, t, x, dest.exists(),
                                                                                     mp4.part_path(dest).exists())))
    # the file appears under its name only once done; the .part.mp4 is gone
    assert not any(s[3] for s in seen) and any(s[4] for s in seen) and dest.is_file()
    assert not mp4.part_path(dest).exists() and got["bytes"] == dest.stat().st_size
    assert any(s[2] == mp4.FINISHING and s[0] < s[1] for s in seen)
    assert seen[-1][2] == mp4.CHECKING and seen[-1][0] == seen[-1][1]
    assert all(s[0] < s[1] for s in seen[:-1])  # 100% waits for the encoded-audio quality check
    assert [s[0] for s in seen] == sorted(s[0] for s in seen)
    assert max(s[0] for s in seen if s[0] <= seen[-1][1] / 3 + 1e-6) > 0  # pass 1
    with av.open(str(dest)) as c:
        assert [s.type for s in c.streams] == ["video", "audio", "subtitle", "subtitle"]
        v, a = c.streams.video[0], c.streams.audio[0]
        assert a.codec_context.name == "aac" and a.codec_context.sample_rate == 44_100 and a.codec_context.channels == 2
        assert a.metadata["language"] == "tel" and a.disposition & av.stream.Disposition.default
        s_te, s_en = c.streams.subtitles
        assert [s.codec_context.name for s in (s_te, s_en)] == ["mov_text", "mov_text"]
        assert [(s.metadata["language"], s.metadata.get("name")) for s in (s_te, s_en)] == [("tel", "Telugu"),
                                                                                            ("eng", "English")]
        # neither subtitle track is enabled (the muxer enables the first of a type; the export clears it): QuickTime
        # shows them only once picked (M12); the video and the audio stay enabled
        assert not s_te.disposition & av.stream.Disposition.default and not s_en.disposition & av.stream.Disposition.default
        assert v.disposition & av.stream.Disposition.default
        assert {k: c.metadata.get(k) for k in META} == META
        video_s = float(v.duration * v.time_base)
        audio_s = float(a.duration * a.time_base)
    assert packets(dest) == packets(video)  # stream copy: every packet, byte for byte
    with av.open(str(video)) as c:
        assert video_s == pytest.approx(float(c.streams.video[0].duration * c.streams.video[0].time_base))
    # the audio spans the video, within a frame, plus the last line's overrun
    overrun = lines[-1].start + off + lines[-1].samples / SR - video_s
    assert overrun > 1.0 and abs(audio_s - (video_s + overrun)) <= 0.1 and got["seconds"] == pytest.approx(audio_s, abs=0.03)
    # A/V sync: the click planned at the white frame's dub time is heard at that frame, within 2 ms
    x = decode(dest)
    heard = int(np.argmax(np.abs(x[:int(30 * 44_100)]))) / 44_100
    assert abs(heard - white_frames(dest)[0]) <= 0.002 and white_frames(dest)[0] == flash
    # the cues: Telugu at the dub's times, English at the source's, both on the file's clock
    te = cues_of(dest, 0)
    assert te[0][0] == round((flash) * 1000) and te[0][2] == "మళ్ళీ ఛానల్ కి స్వాగతం"
    assert [c[0] for c in te] == [round((x.start + off) * 1000) for x in lines]
    en = cues_of(dest, 1)
    assert [(c[0], c[2]) for c in en] == [(round((x.start + off) * 1000), "Welcome back to the channel.") for x in lines]
    data = dest.read_bytes()
    assert 0 <= data.find(b"moov") < data.find(b"mdat")  # faststart: QuickTime opens it at once
    assert got["loudness"]["TP"] <= mix.TP_CEIL and got["warning"] == mp4.NO_BED


def test_a_preview_ends_at_the_first_keyframe_after_its_last_lines_end(tmp_path):
    video, start = sources(tmp_path, "demo")
    lines = [click_line(tmp_path, "a", 10.0), click_line(tmp_path, "b", 18.0, "S2", 9.0)]  # straddles 20 s, to 27 s
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        src = [(float(p.pts * s.time_base), p.is_keyframe) for p in c.demux(s) if p.size]
    after_stop = min(t for t, k in src if k and t >= 20.0)
    key = min(t for t, k in src if k and t >= 27.0)  # not the stop point's: the straddling line's end
    assert after_stop < 27.0 < key
    dest = tmp_path / "Preview (Telugu, first 1 min).mp4"
    got = do_export(dest, video, start, lines, stop=20.0)
    n = next(i for i, (t, k) in enumerate(src) if k and t == key)
    assert packets(dest) == packets(video)[:n]  # everything decoded before that keyframe, nothing after
    with av.open(str(dest)) as c:
        a = c.streams.audio[0]
        assert float(a.duration * a.time_base) == pytest.approx(key, abs=0.03) and got["seconds"] == pytest.approx(key)
    assert all(c[0] + c[1] <= round(key * 1000) for i in (0, 1) for c in cues_of(dest, i))
    x = decode(dest)
    assert np.abs(x[int(26.0 * 44_100):int(27.0 * 44_100)]).max() > 0.01  # the line is heard to its end


def test_a_video_that_starts_after_the_audio_has_no_cue_before_the_files_start(tmp_path):
    video = tmp_path / "late.mp4"
    synth_video(video, 12.0, audio=False, video_start=0.5)  # its first frame half a second after the audio's first sample
    assert mp4.probe(video)["start"] == 0.5
    lines = [click_line(tmp_path, "a", 0.2), click_line(tmp_path, "b", 3.0, "S2")]  # dub time t plays at t - 0.5
    telugu = subtitles.telugu_cues([{"start": x.start, "samples": x.samples, "speaker": x.speaker,
                                     "telugu": "మళ్ళీ స్వాగతం"} for x in lines], SR)
    english = [Cue(0.1, 0.4, "S1", "Hi."), Cue(0.2, 1.4, "S1", "Welcome back."), Cue(3.0, 3.9, "S2", "Thanks.")]
    dest = tmp_path / "Late (Telugu).mp4"
    mp4.export(dest, video=video, lines=lines, voice_sr=SR, telugu=telugu, english=english, audio_start=0.0, stop=None,
               metadata=META)
    # what began before the file's 0 shows from it, what ended before it is left out, and the rest is exactly in time
    assert cues_of(dest, 0) == [(0, 700, "మళ్ళీ స్వాగతం"), (2500, 1000, "మళ్ళీ స్వాగతం")]
    assert cues_of(dest, 1) == [(0, 900, "Welcome back."), (2500, 900, "Thanks.")]
    x = decode(dest)
    assert abs(int(np.argmax(np.abs(x[44_100:]))) / 44_100 + 1.0 - 2.5) <= 0.002  # the second line's click, at 2.5 s


@pytest.mark.parametrize("codec", ["av1", "vp9"])
def test_av1_and_vp9_are_copied_as_they_are(tmp_path, codec):
    if codec == "av1":
        video = tmp_path / "av1.mp4"
        synth_video(video, 12.0, video_codec="libsvtav1", audio=False, codec_options={"preset": "12"})
    else:
        video = tmp_path / "vp9.webm"
        synth_video(video, 12.0, video_codec="libvpx-vp9", audio=False, container="webm",
                    codec_options={"deadline": "realtime", "cpu-used": "8"})
    assert mp4.probe(video)["codec"] == codec
    dest = tmp_path / "out.mp4"
    got = do_export(dest, video, 0.0, [click_line(tmp_path, "a", 10.0)])
    assert mp4.probe(dest)["codec"] == codec and packets(dest) == packets(video)
    assert abs(int(np.argmax(np.abs(decode(dest)))) / 44_100 - white_frames(dest)[0]) <= 0.002
    assert (mp4.VP9 in (got["warning"] or "")) == (codec == "vp9")


def test_names_are_sanitised_and_never_take_someone_elses_file(tmp_path):
    assert mp4.sanitize('  ..A/B\\\\C: "quoted" <x>|y*?\x07 \t z  ', "abcdefghijk") == 'A B C quoted x y z'
    assert mp4.sanitize("...", "abcdefghijk") == "abcdefghijk"
    long = "తెలుగు" * 20  # 18 bytes each: cut at 150 bytes, on a character
    cut = mp4.sanitize(long, "x")
    assert len(cut.encode()) <= mp4.NAME_BYTES and long.startswith(cut) and cut
    assert mp4.output_name("A talk", "x", None) == "A talk (Telugu).mp4"
    assert mp4.output_name("A talk", "x", 900.0) == "A talk (Telugu, first 15 min).mp4"
    name = "A talk (Telugu).mp4"
    assert mp4.output_path(tmp_path, name, None) == tmp_path / name
    (tmp_path / name).write_bytes(b"someone else's")
    assert mp4.output_path(tmp_path, name, None) == tmp_path / "A talk (Telugu) (2).mp4"
    assert mp4.output_path(tmp_path, name, tmp_path / name) == tmp_path / name  # the job's own: replaced
    (tmp_path / "A talk (Telugu) (2).mp4").write_bytes(b"x")
    assert mp4.output_path(tmp_path, name, None) == tmp_path / "A talk (Telugu) (3).mp4"
    numbered = tmp_path / "A talk (Telugu) (2).mp4"
    assert mp4.output_path(tmp_path, name, numbered) == numbered  # the job's own numbered file: replaced
    # not a file of another name (a preview's stays beside the whole video's), nor one in another folder
    preview = tmp_path / "A talk (Telugu, first 15 min).mp4"
    assert mp4.output_path(tmp_path, name, preview) == tmp_path / "A talk (Telugu) (3).mp4"
    assert mp4.output_path(tmp_path / "elsewhere", name, tmp_path / name) == tmp_path / "elsewhere" / name
    assert mp4.part_path(tmp_path / name) == tmp_path / ".A talk (Telugu).part.mp4"


def test_a_pause_mid_export_leaves_no_file_and_the_export_again_is_the_same_bytes(tmp_path):
    video, start = sources(tmp_path, "demo")
    lines = [click_line(tmp_path, f"l{k}", 5.0 + 15.0 * k, "S1" if k % 2 else "S2", 2.0) for k in range(10)]
    dest = tmp_path / "out" / "A talk (Telugu).mp4"
    dest.parent.mkdir()
    cancel = threading.Event()

    def progress(done: float, total: float, extra: str | None) -> None:
        if done > total / 2:  # in the second pass, with the .part.mp4 half written
            assert mp4.part_path(dest).exists()
            cancel.set()

    with pytest.raises(Cancelled):
        do_export(dest, video, start, lines, cancel=cancel, progress=progress)
    assert not dest.exists() and not mp4.part_path(dest).exists() and not list(dest.parent.iterdir())
    do_export(dest, video, start, lines)
    other = tmp_path / "again.mp4"
    do_export(other, video, start, lines)
    assert dest.read_bytes() == other.read_bytes()  # an export made again from the start is the same file


def test_a_pause_during_encoded_audio_quality_check_stops_at_a_block_and_preserves_a_prior_output(tmp_path):
    video = tmp_path / "source.mp4"
    synth_video(video, seconds=24.0)
    lines = [click_line(tmp_path, "a", 1.0, seconds=2.0)]
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"previous finished output")
    cancel = threading.Event()
    checks: list[tuple[float, float]] = []

    def progress(done: float, total: float, extra: str | None) -> None:
        if extra == mp4.CHECKING:
            checks.append((done, total))
            assert dest.read_bytes() == b"previous finished output" and mp4.part_path(dest).exists()
            if len(checks) == 2:  # phase start, then one 10-second block of decoded AAC
                cancel.set()

    with pytest.raises(Cancelled):
        do_export(dest, video, audio_start(video), lines, cancel=cancel, progress=progress)
    assert len(checks) == 2 and checks[0][0] < checks[1][0] < checks[1][1]
    assert dest.read_bytes() == b"previous finished output" and not mp4.part_path(dest).exists()


def test_quality_check_progress_keeps_loudness_results_and_export_bytes(tmp_path):
    video = tmp_path / "source.mp4"
    synth_video(video, seconds=12.0)
    lines = [click_line(tmp_path, "a", 1.0, seconds=2.0)]
    plain, tracked = tmp_path / "plain.mp4", tmp_path / "tracked.mp4"
    baseline = do_export(plain, video, audio_start(video), lines)
    seen = []
    actual = do_export(tracked, video, audio_start(video), lines,
                       progress=lambda done, total, extra: seen.append((done, total, extra)))
    assert plain.read_bytes() == tracked.read_bytes()
    assert actual["loudness"] == baseline["loudness"] == mp4._measure(tracked)
    checking = [(done, total) for done, total, extra in seen if extra == mp4.CHECKING]
    assert len(checking) >= 4  # phase start, full block, final partial block, meter flushed
    assert checking[-1][0] == checking[-1][1]
    assert all(done < total for done, total in checking[:-1])


# ---- through a render job ------------------------------------------------------------------------------------------
def backend(tts=None) -> Backend:
    return Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts or MockTTS())


def job_for(tmp_path, resolver, settings=None, b=None, out="out") -> RenderJob:
    return RenderJob(b or backend(), resolver, tmp_path / "cache", f"https://youtu.be/{VID}", settings,
                     output_dir=tmp_path / out)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_a_whole_export_lets_go_of_the_downloaded_video_and_a_preview_keeps_its_sources(tmp_path):
    fake = YouTubeShaped()
    preview = job_for(tmp_path, fake, RenderSettings(stop_at=60.0))
    assert await preview.run() == "done"
    d = tmp_path / "cache" / VID
    out = preview.doc["output"]
    assert out["kind"] == "preview" and Path(out["path"]).name == "A two-file video (Telugu, first 1 min).mp4"
    assert (d / "video.mp4").exists() and (d / "render" / "audio16k.f32").exists()  # continuing needs them
    src = preview.doc["source"]
    assert src["video"]["owned"] and src["video"]["file"] == "video.mp4" and src["video"]["codec"] == "h264"
    assert src["audioStart"] == pytest.approx(audio_start(d / "audio.m4a"), abs=1e-6) and src["thumb"] == "thumb.jpg"
    whole = job_for(tmp_path, fake, RenderSettings())  # "Continue to the whole video"
    assert await whole.run() == "done" and fake.videos == 1  # the video was there: no download
    assert not (d / "video.mp4").exists() and not (d / "render" / "audio16k.f32").exists()
    assert (d / "audio.m4a").exists() and Path(out["path"]).exists()  # the preview's file stays
    doc = json.loads((d / "render" / "job.json").read_text())
    assert doc["output"]["kind"] == "whole" and doc["source"]["video"]["fingerprint"]["size"] > 0  # (kept)
    with av.open(doc["output"]["path"]) as c:
        assert [s.type for s in c.streams] == ["video", "audio", "subtitle", "subtitle"]
    # the demo's file isn't the job's: it stays
    demo = job_for(tmp_path / "demo", DemoResolver())
    assert await demo.run() == "done" and (tmp_path / "demo" / "cache" / VID / "demo.mp4").exists()
    assert not demo.doc["source"]["video"]["owned"]


async def test_a_done_job_run_again_unchanged_serves_its_mp4_without_downloading(tmp_path):
    fake = YouTubeShaped()
    job = job_for(tmp_path, fake)
    assert await job.run() == "done" and fake.videos == 1
    path = Path(job.doc["output"]["path"])
    mtime = path.stat().st_mtime_ns
    again = job_for(tmp_path, fake)
    assert await again.run() == "done"
    assert fake.videos == 1 and path.stat().st_mtime_ns == mtime  # served: the video stage and the export did nothing
    trace = [json.loads(x) for x in (tmp_path / "cache" / VID / "units.jsonl").read_text().splitlines()]
    assert [(e["key"], e["cached"]) for e in trace if e["event"] == "stage" and e["key"] in ("video", "export")][-2:] \
        == [("video", True), ("export", True)]
    # lines that change: the export must mux again, so the video is downloaded again (by the video stage's code)
    changed = job_for(tmp_path, fake, RenderSettings(tts_script="latin"))
    assert await changed.run() == "done" and fake.videos == 2 and path.stat().st_mtime_ns != mtime
    assert changed.doc["output"]["path"] == str(path)  # its own file, replaced
    # the TTS read the English words in Latin script; the Telugu track shows the Telugu-script wording (§2.16)
    said = [json.loads(x) for x in (tmp_path / "cache" / VID / "units.jsonl").read_text().splitlines()]
    assert any(re.search("[A-Za-z]", e["telugu"]) for e in said if e["event"] == "unit")
    te = cues_of(path, 0)
    assert te and not any(re.search("[A-Za-z]", c[2]) for c in te)
    # a file the user moved: the snapshot says so
    path.unlink()
    assert changed.snapshot()["output"]["note"] == "File moved or deleted" and changed.snapshot()["output"]["missing"]
    # another file under its name now (another job's of the same title): never replaced; the job saves beside it
    path.write_bytes(b"another job's video")
    beside = job_for(tmp_path, fake, RenderSettings(tts_script="latin"))
    assert await beside.run() == "done" and path.read_bytes() == b"another job's video"
    second = path.with_name("A two-file video (Telugu) (2).mp4")
    assert beside.doc["output"]["path"] == str(second) and second.is_file()
    # with that name free again, the job's next file still replaces its own " (2)": no orphaned earlier output
    path.unlink()
    last = job_for(tmp_path, fake)  # Telugu-script TTS again: new lines, so a new file
    assert await last.run() == "done" and last.doc["output"]["path"] == str(second) and not path.exists()


async def test_another_video_format_or_a_changed_video_file_downloads_the_video_again(tmp_path, monkeypatch):
    fake = YouTubeShaped()
    first = job_for(tmp_path, fake, RenderSettings(stop_at=60.0))  # a preview: its video.mp4 stays
    assert await first.run() == "done" and fake.downloads == 1
    # the selector changed (an engine update): the file there was chosen by the old one, so it is fetched again
    monkeypatch.setattr(render, "VIDEO_FORMAT", "bv[vcodec^=avc1][height<=720][ext=mp4]")
    again = job_for(tmp_path, fake, RenderSettings(stop_at=60.0))
    assert await again.run() == "done" and fake.downloads == 2
    assert again.doc["stages"]["video"]["inputs"]["format"] == "bv[vcodec^=avc1][height<=720][ext=mp4]"
    # a file that isn't the one recorded any more is fetched again too, not recorded as it is
    (tmp_path / "cache" / VID / "video.mp4").write_bytes(b"not the video")
    third = job_for(tmp_path, fake, RenderSettings(stop_at=60.0))
    assert await third.run() == "done" and fake.downloads == 3
    assert third.doc["source"]["video"]["fingerprint"] == first.doc["source"]["video"]["fingerprint"]


async def test_a_pause_mid_export_then_a_resume_gives_the_same_file(tmp_path, monkeypatch):
    paused = job_for(tmp_path / "a", DemoResolver())
    real = mp4.export

    def export(dest, **kw):
        progress = kw["progress"]

        def pausing(done, total, extra):
            if done > total / 2:
                paused.pause()
            progress(done, total, extra)

        return real(dest, **{**kw, "progress": pausing})

    monkeypatch.setattr(render.mp4, "export", export)
    assert await paused.run() == "paused"
    st = paused.doc["stages"]["export"]
    assert paused.doc["stage"] == "export" and st["state"] == "todo" and st["attempts"] == 0
    out = tmp_path / "a" / "out"
    assert list(out.iterdir()) == [] and paused.doc["output"] is None  # no part, no file
    monkeypatch.setattr(render.mp4, "export", real)
    assert await paused.run() == "done"
    straight = job_for(tmp_path / "b", DemoResolver())
    assert await straight.run() == "done"
    assert Path(paused.doc["output"]["path"]).read_bytes() == Path(straight.doc["output"]["path"]).read_bytes()


async def test_a_bench_input_is_never_changed_or_deleted(tmp_path):
    video = tmp_path / "talk.mp4"
    synth_video(video, 40.0)
    before = digest(video)
    job = job_for(tmp_path, LocalResolver(video))
    assert await job.run() == "done" and job.doc["output"]["kind"] == "whole"
    assert video.exists() and digest(video) == before
    assert job.doc["source"]["file"] == str(video.resolve()) and job.doc["source"]["video"]["file"] == str(video.resolve())
    assert not job.doc["source"]["video"]["owned"]
    again = job_for(tmp_path, LocalResolver(video))  # a resume finds it by its absolute path
    assert await again.run() == "done" and digest(video) == before


async def test_the_video_stage_checks_the_disk_before_downloading(tmp_path, monkeypatch):
    fake = YouTubeShaped()
    job = job_for(tmp_path, fake)
    monkeypatch.setattr(render, "VIDEO_DISK_PER_HOUR", 1e18)  # what the bed and the audio would need: more than any disk
    assert await job.run() == "failed" and job.doc["stage"] == "video"
    assert "GB of free disk space" in job.doc["error"] and not (tmp_path / "cache" / VID / "video.mp4").exists()
    assert fake.videos == 1  # stopped at the download's first progress, before writing it


# ---- the background sound under the voices (§2.14, §2.15) ---------------------------------------------------------
class Gap(MockTranscriber):
    """The mock's speech with nothing said from 7 s to 13 s, so no Telugu line plays near the beep (10 s)."""

    def transcribe(self, audio, language=None, speech=()):
        got = super().transcribe(audio, language, speech)
        got.words = [w for w in got.words if w.end <= 7.0 or w.start >= 13.0]
        return got


def separating() -> Backend:
    return Backend("mock", "cpu", Gap(), MockDiarizer(), MockSceneTranslator, MockTTS(), separator=MockSeparator())


@pytest.mark.parametrize("kind", ["demo", "two files"])
async def test_the_background_sound_plays_under_the_voices_in_sync_with_the_picture(tmp_path, kind):
    b = separating()
    job = job_for(tmp_path, DemoResolver() if kind == "demo" else YouTubeShaped(), b=b)
    assert await job.run() == "done"
    out = Path(job.doc["output"]["path"])
    manifest = json.loads((tmp_path / "cache" / VID / "render" / "manifest.json").read_text())
    spans = [(x["start"], x["start"] + x["samples"] / SR) for x in manifest["lines"]]
    assert spans and not any(a < 11.5 and b > 8.5 for a, b in spans)  # no line near the beep
    assert job.doc["output"]["warning"] is None and job.doc["output"]["inputs"]["bed"]["separator"] == "MockSeparator"
    assert b.separator.calls > 0 and job.doc["bed"]["until"] == pytest.approx(job.doc["duration"], abs=0.05)
    # the beep, now only in the bed, is heard at the white frame, within 2 ms (the A/V origin of §2.2, both files)
    x = decode(out)
    flash = white_frames(out)[0]
    heard = 9.0 + onset(x[int(9.0 * 44_100):int(11.0 * 44_100)], 44_100)
    assert abs(heard - flash) <= 0.002
    # the bed is under the voices: between lines the original's chord is there, ducked under the English speech
    level = lambda a, b: float(np.sqrt(np.mean(x[int(a * 44_100):int(b * 44_100)] ** 2)))  # noqa: E731
    assert level(9.2, 9.8) > 0.003
    # a whole export lets go of the bed with the other sources
    d = tmp_path / "cache" / VID / "render"
    assert not (d / "bed").exists() and not (d / "bed.json").exists() and not (d / "bed_vocals.npy").exists()


async def test_a_done_job_run_again_unchanged_calls_neither_the_separator_nor_the_video_download(tmp_path):
    fake, b = YouTubeShaped(), separating()
    job = job_for(tmp_path, fake, b=b)
    assert await job.run() == "done"
    calls, chunks, path = b.separator.calls, b.separator.chunks, Path(job.doc["output"]["path"])
    mtime = path.stat().st_mtime_ns
    again = job_for(tmp_path, fake, b=b)
    assert await again.run() == "done"
    assert b.separator.calls == calls and fake.videos == 1 and path.stat().st_mtime_ns == mtime
    trace = [json.loads(x) for x in (tmp_path / "cache" / VID / "units.jsonl").read_text().splitlines()]
    assert [e["cached"] for e in trace if e["event"] == "stage" and e["key"] in ("separate", "video", "export")] == \
        [False, False, False, True, True, True]  # served from the output, all three
    assert again.doc["stages"]["separate"]["attempts"] == 1
    # lines that change: the export mixes again, so it makes the bed's blocks again (and downloads the video again)
    changed = job_for(tmp_path, fake, RenderSettings(tts_script="latin"), b=b)
    assert await changed.run() == "done"
    assert b.separator.chunks == 2 * chunks and b.separator.calls == 2 * calls  # every block again
    assert fake.videos == 2 and path.stat().st_mtime_ns != mtime and changed.doc["output"]["warning"] is None
    st = changed.doc["stages"]["separate"]
    assert st["state"] == "done" and st["attempts"] == 1  # the export's re-make: its start is the export's
    assert not (tmp_path / "cache" / VID / "render" / "bed").exists()  # and let go of again


async def test_a_done_mp4_doesnt_need_the_separator_model_until_its_lines_change(tmp_path):
    """The run's own check (render) skips a whole video's MP4 that still matches: it needs the separator only to mix
    again. Then the export says how to fetch the model before making any block (§2.14)."""
    fake, b = YouTubeShaped(), separating()
    job = job_for(tmp_path, fake, b=b)
    assert await job.run() == "done"
    calls = b.separator.calls
    b.separator.missing = lambda: ["config.json", "model.safetensors"]  # the model deleted since
    again = job_for(tmp_path, fake, b=b)
    assert await again.run() == "done" and b.separator.calls == calls
    changed = job_for(tmp_path, fake, RenderSettings(tts_script="latin"), b=b)
    assert await changed.run() == "failed" and changed.doc["stage"] == "export"
    assert "maata-bench fetch --backend mock" in changed.doc["error"] and b.separator.calls == calls
    assert changed.doc["stages"]["separate"]["state"] == "done"  # (the bed's row: nothing started)


@pytest.mark.parametrize("why", ["read-only", "full"])
async def test_the_export_checks_the_output_folder_before_downloading_or_separating_again(tmp_path, monkeypatch, why):
    """A whole video's export deleted the video and the bed: an export again (lines changed) checks the output folder
    first, so a folder it can't use fails it before minutes of download and separation."""
    fake, b = YouTubeShaped(), separating()
    job = job_for(tmp_path, fake, b=b)
    assert await job.run() == "done"
    calls, out = b.separator.calls, tmp_path / "out"
    if why == "full":
        monkeypatch.setattr(render.mp4, "OUTPUT_PER_HOUR", 1e18)
    else:
        out.chmod(0o500)
    try:
        changed = job_for(tmp_path, fake, RenderSettings(tts_script="latin"), b=b)
        assert await changed.run() == "failed" and changed.doc["stage"] == "export"
    finally:
        out.chmod(0o700)
    assert ("can't save videos in" if why == "read-only" else "GB of free disk space") in changed.doc["error"]
    assert fake.videos == 1 and b.separator.calls == calls  # neither downloaded nor separated again
