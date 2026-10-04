"""The background sound's chunking and stage (OFFLINE-RENDER §2.14, §9): one global chunk grid, so blocks made one at a
time, in a preview and its continuation, or across a pause, equal a whole-file run's; torn blocks and changed inputs;
the vocals envelope; mono, a 3 s file and 48 kHz Opus; one sample origin for the 16 kHz and the 44.1 kHz decodes. The
mock backend's MockSeparator (vocals = half of each chunk) and stand-ins; synthetic media only (PyAV and numpy)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from fakes import onset  # noqa: E402

from maata_engine import render, separate  # noqa: E402
from maata_engine.backends.base import Backend  # noqa: E402
from maata_engine.backends.mock import (MockDiarizer, MockSceneTranslator, MockSeparator, MockTranscriber,  # noqa: E402
                                        MockTTS)
from maata_engine.render import RenderJob, RenderSettings  # noqa: E402
from maata_engine.resolve import (DEMO_MARK, LocalResolver, audio_start, decode_audio_to,  # noqa: E402
                                  decode_stereo, synth_video)

SR = separate.SR
VID = "sepa0000001"


def tones(path: Path, seconds: float, *, rate: int = 44_100, layout: str = "stereo", codec: str = "aac",
          container: str = "mp4", beep: float | None = None) -> Path:
    """An audio-only file: a soft chord, a different one on the right channel, and a 1 kHz beep from `beep` s."""
    out = av.open(str(path), "w", format=container)
    s = out.add_stream(codec, rate=rate)
    s.layout = layout
    channels = 1 if layout == "mono" else 2
    n, block = 0, 960
    total = int(seconds * rate)
    while n < total:
        t = (n + np.arange(min(block, total - n))) / rate
        left = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 330 * t)
        right = 0.1 * np.sin(2 * np.pi * 277 * t) + 0.05 * np.sin(2 * np.pi * 415 * t)
        if beep is not None:
            left = left + np.where(t >= beep, 0.3 * np.sin(2 * np.pi * 1000 * (t - beep)), 0.0)
            right = right + np.where(t >= beep, 0.3 * np.sin(2 * np.pi * 1000 * (t - beep)), 0.0)
        x = np.stack([left, right][:channels]).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(x, format="fltp", layout=layout)
        frame.sample_rate, frame.pts, frame.time_base = rate, n, Fraction(1, rate)
        for p in s.encode(frame):
            out.mux(p)
        n += x.shape[1]
    for p in s.encode(None):
        out.mux(p)
    out.close()
    return path


def job_on(tmp_path, src: Path, sep: MockSeparator | None = None, stop_at: float | None = None,
           cache: str = "cache") -> RenderJob:
    b = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, MockTTS(),
                separator=sep if sep is not None else MockSeparator())
    return RenderJob(b, LocalResolver(src), tmp_path / cache, f"https://youtu.be/{VID}", RenderSettings(stop_at=stop_at),
                     output_dir=tmp_path / "out")


async def separate_only(job: RenderJob) -> None:
    """The fetch and separate stages alone, as `run()` starts them."""
    job._stop.clear()
    job.render_dir.mkdir(parents=True, exist_ok=True)
    await job._stage("fetch")
    await job._stage("separate")


def blocks(job: RenderJob) -> dict[int, bytes]:
    return {int(p.stem): p.read_bytes() for p in sorted((job.render_dir / "bed").glob("*.npy"))}


def bed_of(job: RenderJob) -> np.ndarray:
    folder = job.render_dir / "bed"
    return np.concatenate([np.load(p).astype(np.float32) for p in sorted(folder.glob("*.npy"))], axis=1)


def decoded(src: Path) -> np.ndarray:
    return np.concatenate(list(decode_stereo(src)), axis=1)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "demo.mp4"
    synth_video(path, 150.0)  # two and a half blocks, the beep at DEMO_MARK
    return path


# ---- the grid (pure) -----------------------------------------------------------------------------------------------
def whole_file(x: np.ndarray, vocals, chunk: int = separate.CHUNK) -> np.ndarray:
    """A whole-file run, written out plainly: the padded file, every chunk on the grid, overlap-add, the window sum."""
    n, pad = x.shape[1], chunk // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect" if n > chunk else "constant")
    count = separate.chunk_count(n, chunk)
    xp = np.pad(xp, ((0, 0), (0, max(0, (count - 1) * pad + chunk - xp.shape[1]))))
    out, wsum, w = np.zeros(xp.shape, np.float32), np.zeros(xp.shape[1], np.float32), separate.window(chunk)
    for j in range(count):
        out[:, j * pad:j * pad + chunk] += vocals(xp[None, :, j * pad:j * pad + chunk])[0] * w
        wsum[j * pad:j * pad + chunk] += w
    return (out / np.where(wsum > 0, wsum, 1))[:, pad:pad + n]


def context(batch: np.ndarray) -> np.ndarray:
    """A stand-in separator whose output depends on the whole chunk (its mean) and the position in it, so a chunk cut
    anywhere but on the grid gives other samples."""
    ramp = np.linspace(0.2, 0.8, batch.shape[-1], dtype=np.float32)
    return np.tanh(batch) * ramp + batch.mean(axis=-1, keepdims=True)


@pytest.mark.parametrize("seconds", [3.0, 60.0, 61.3, 150.3])
def test_blocks_one_at_a_time_equal_a_whole_file_run(seconds):
    n = int(seconds * SR)
    x = (0.3 * np.random.default_rng(int(seconds)).standard_normal((2, n))).astype(np.float32)
    ref = whole_file(x, context)
    mock = MockSeparator()
    for k in range(separate.block_count(n)):
        lo, hi = separate.block_span(k, n)
        a, b = separate.block_range(k, n)
        bed, ms = separate.block_bed(k, n, x[:, lo:hi], lo, context)  # only the samples the block's chunks read
        assert np.abs((x[:, a:b] - bed) - ref[:, a:b]).max() < 1e-5  # across chunk joins and block edges
        # MockSeparator: the bed is exactly half the mixture, the vocals' mean square a quarter of the mixture's
        bed, ms = separate.block_bed(k, n, x[:, lo:hi], lo, mock.vocals)
        assert np.abs(bed - x[:, a:b] / 2).max() < 1e-6
        f = separate.frame_samples()
        sq = (x[:, a:b].astype(np.float64) / 2) ** 2
        want = [sq[:, i:i + f].mean() for i in range(0, b - a, f)]
        assert np.allclose(ms, want, rtol=1e-5) and len(ms) == separate.frames_in(b - a)
    # a block's chunks: those that touch it, one more than its share (16 for a whole 60 s block)
    if seconds == 150.3:
        assert [len(separate.block_chunks(k, n)) for k in range(3)] == [16, 16, 9]
        assert mock.chunks == sum(len(separate.block_chunks(k, n)) for k in range(3))


def test_a_caller_that_doesnt_know_the_length_yet_gets_the_same_block():
    n = int(150.3 * SR)
    x = (0.3 * np.random.default_rng(5).standard_normal((2, n))).astype(np.float32)
    lo, hi = separate.block_span(0, 1 << 62)
    assert hi < n
    early, _ = separate.block_bed(0, hi, x[:, lo:hi], lo, context)  # the decode has reached `hi`, no further
    known, _ = separate.block_bed(0, n, x[:, lo:hi], lo, context)
    assert np.array_equal(early, known)


def test_the_buffer_keeps_only_its_window():
    buf = separate.Buffer()
    buf.drop(10)
    buf.put(np.arange(16, dtype=np.float32).reshape(2, 8))  # samples 0-7: all before the window
    assert buf.at == buf.end == 8 and buf.get().shape == (2, 0)
    buf.put(np.arange(16, 32, dtype=np.float32).reshape(2, 8))  # 8-15: 10-15 kept
    assert (buf.at, buf.end) == (10, 16) and buf.get()[0].tolist() == [18, 19, 20, 21, 22, 23]
    buf.drop(14)
    assert buf.at == 14 and buf.get()[1].tolist() == [30, 31]


# ---- the stage -----------------------------------------------------------------------------------------------------
async def test_the_stage_makes_the_bed_and_its_vocals_envelope(tmp_path, demo):
    sep = MockSeparator()
    job = job_on(tmp_path, demo, sep)
    await separate_only(job)
    d = job.render_dir
    doc = json.loads((d / "bed.json").read_text())
    assert doc["blocks"] == [0, 1, 2] and doc["inputs"]["separator"] == "MockSeparator"
    assert doc["inputs"]["dtype"] == separate.SEP_DTYPE and doc["inputs"]["version"] == separate.SEP_VERSION
    assert doc["inputs"]["audio"] == render._fingerprint(demo) and doc["inputs"]["block"] == 60.0
    x = decoded(demo)
    n = x.shape[1]
    got = {k: np.load(d / "bed" / f"{k:05d}.npy") for k in range(3)}
    assert [g.shape for g in got.values()] == [(2, 2_646_000), (2, 2_646_000), (2, n - 2 * 2_646_000)]
    assert all(g.dtype == np.float16 for g in got.values())
    bed = np.concatenate([g.astype(np.float32) for g in got.values()], axis=1)
    assert np.abs(bed - x / 2).max() < 1e-3  # half the mixture (float16)
    vocals = np.load(d / "bed_vocals.npy")
    f = separate.frame_samples()
    want = [((x[:, i:i + f].astype(np.float64) / 2) ** 2).mean() for i in range(0, n, f)]
    assert vocals.dtype == np.float32 and len(vocals) == len(want) and np.allclose(vocals, want, rtol=1e-4)
    st = job.doc["stages"]["separate"]
    assert st["state"] == "done" and st["unit"] == "video s" and st["done"] == st["total"] == 150.0
    assert job.doc["bed"] == {"separator": "MockSeparator", "until": 150.0, "note": None}
    assert sep.released == 1 and job._separated.is_set()
    # run again unchanged: served from disk, no separator call
    calls = sep.calls
    await separate_only(job)
    assert sep.calls == calls and st["attempts"] == 1


async def test_a_preview_then_continuing_adds_only_the_missing_blocks(tmp_path, demo):
    sep = MockSeparator()
    preview = job_on(tmp_path, demo, sep, stop_at=20.0)  # its range: 20 + 60 s, so blocks 0 and 1
    await separate_only(preview)
    first = blocks(preview)
    assert sorted(first) == [0, 1] and preview.doc["stages"]["separate"]["total"] == 80.0
    assert sep.chunks == len(separate.block_chunks(0, 1 << 62)) * 2
    whole = job_on(tmp_path, demo, sep)  # "Continue to the whole video"
    before = sep.chunks
    await separate_only(whole)
    n = decoded(demo).shape[1]
    assert sep.chunks - before == len(separate.block_chunks(2, n))  # only block 2's chunks
    got = blocks(whole)
    assert sorted(got) == [0, 1, 2] and all(got[k] == first[k] for k in (0, 1))
    straight = job_on(tmp_path, demo, MockSeparator(), cache="straight")
    await separate_only(straight)
    assert blocks(straight) == got  # the same bytes as one run over the whole file
    assert np.array_equal(np.load(whole.render_dir / "bed_vocals.npy"), np.load(straight.render_dir / "bed_vocals.npy"))


class PausingSeparator(MockSeparator):
    """Pauses its job during call number `on` (from the worker thread, as the UI's pause arrives mid-batch)."""

    def __init__(self, on: int) -> None:
        super().__init__()
        self.on = on
        self.job: RenderJob | None = None

    def vocals(self, batch):
        out = super().vocals(batch)
        if self.calls == self.on:
            self.job.pause()
        return out


async def test_a_pause_stops_between_batches_and_a_resume_skips_the_done_blocks(tmp_path, demo):
    sep = PausingSeparator(on=10)  # inside block 1 (block 0 is 8 calls of 2 chunks)
    job = job_on(tmp_path, demo, sep)
    sep.job = job
    with pytest.raises(render._Paused):
        await separate_only(job)
    st = job.doc["stages"]["separate"]
    assert sep.calls == 10 and st["state"] == "todo" and st["attempts"] == 0 and sep.released == 1
    assert sorted(blocks(job)) == [0] and json.loads((job.render_dir / "bed.json").read_text())["blocks"] == [0]
    assert not job._separated.is_set()
    await separate_only(job)  # decoded from the start again, block 0 skipped
    n = decoded(demo).shape[1]
    assert sep.chunks == 20 + len(separate.block_chunks(1, n)) + len(separate.block_chunks(2, n))
    straight = job_on(tmp_path, demo, MockSeparator(), cache="straight")
    await separate_only(straight)
    assert blocks(job) == blocks(straight)  # identical samples


async def test_a_torn_block_is_made_again_and_changed_inputs_make_every_block_again(tmp_path, demo, monkeypatch):
    sep = MockSeparator()
    src = tmp_path / "talk.mp4"
    src.write_bytes(demo.read_bytes())
    job = job_on(tmp_path, src, sep)
    await separate_only(job)
    good = blocks(job)
    path = job.render_dir / "bed" / "00001.npy"
    path.write_bytes(good[1][:len(good[1]) // 2])  # torn
    before = sep.chunks
    await separate_only(job)
    assert sep.chunks - before == len(separate.block_chunks(1, 1 << 62)) and blocks(job) == good
    # a vocals envelope that lost a block's frames: that block again
    vocals = np.load(job.render_dir / "bed_vocals.npy")
    vocals[700] = np.nan
    np.save(job.render_dir / "bed_vocals.npy", vocals)
    before = sep.chunks
    await separate_only(job)
    assert sep.chunks - before == len(separate.block_chunks(1, 1 << 62)) and blocks(job) == good
    # the inputs: the compute dtype, the version, another source: every block again
    for name, value in (("SEP_DTYPE", "float32"), ("SEP_VERSION", separate.SEP_VERSION + 1)):
        monkeypatch.setattr(separate, name, value)
        before = sep.chunks
        await separate_only(job)
        assert sep.chunks - before == 41 and sorted(blocks(job)) == [0, 1, 2]
        assert json.loads((job.render_dir / "bed.json").read_text())["inputs"][name.split("_")[1].lower()] == value
    tones(src, 30.0)  # the source's bytes changed (its fingerprint): fetched again, and every block of the new audio
    again = job_on(tmp_path, src, sep)
    await separate_only(again)
    assert sorted(blocks(again)) == [0] and len(np.load(again.render_dir / "bed_vocals.npy")) == 300


async def test_mono_is_duplicated_and_a_3_s_file_and_48_khz_opus_work(tmp_path):
    mono = tones(tmp_path / "mono.m4a", 70.0, layout="mono")
    job = job_on(tmp_path, mono, cache="mono")
    await separate_only(job)
    bed = bed_of(job)
    assert bed.shape[0] == 2 and np.array_equal(bed[0], bed[1]) and np.abs(bed[0]).max() > 0.05
    x = np.concatenate(list(decode_stereo(mono)), axis=1)
    assert np.array_equal(x[0], x[1]) and np.abs(bed - x / 2).max() < 1e-3
    short = tones(tmp_path / "short.m4a", 3.0)
    job = job_on(tmp_path, short, cache="short")
    await separate_only(job)
    x = decoded(short)
    assert x.shape[1] < separate.CHUNK  # shorter than a chunk: zero padding, two chunks
    assert bed_of(job).shape == x.shape and np.abs(bed_of(job) - x / 2).max() < 1e-3
    assert job.doc["stages"]["separate"]["done"] == pytest.approx(3.0, abs=0.05)
    opus = tones(tmp_path / "opus.webm", 65.0, rate=48_000, codec="libopus", container="webm")
    job = job_on(tmp_path, opus, cache="opus")
    await separate_only(job)
    x = decoded(opus)
    assert abs(x.shape[1] / SR - 65.0) < 0.05  # resampled to 44.1 kHz
    assert bed_of(job).shape == x.shape and np.abs(bed_of(job) - x / 2).max() < 1e-3
    assert sorted(blocks(job)) == [0, 1]


async def test_no_separator_means_no_bed_and_job_json_says_so(tmp_path, demo):
    job = job_on(tmp_path, demo)
    job.b.separator = None
    await separate_only(job)
    assert job.doc["bed"]["separator"] is None and "Telugu voices only" in job.doc["bed"]["note"]
    assert not (job.render_dir / "bed").exists() and job._separated.is_set()
    assert job.doc["stages"]["separate"]["state"] == "done"


def test_the_16_khz_and_the_44_1_khz_decodes_put_the_beep_at_the_same_time(tmp_path, demo):
    decode_audio_to(demo, tmp_path / "a.f32")
    low = np.fromfile(tmp_path / "a.f32", np.float32)
    high = decoded(demo)
    a = int((DEMO_MARK - 2) * 16_000)
    b = int((DEMO_MARK - 2) * SR)
    at16 = DEMO_MARK - 2 + onset(low[a:a + 4 * 16_000], 16_000)
    at44 = DEMO_MARK - 2 + onset(high[0, b:b + 4 * SR], SR)
    assert abs(at16 - DEMO_MARK) <= 0.001 and abs(at44 - at16) <= 0.001
    # and on a file whose audio starts after its stream's 0 (as the two-file fake's): both count from its first sample
    late = tmp_path / "late.m4a"
    synth_video(late, 20.0, video_codec=None, audio_start=0.25)
    first = audio_start(late)
    assert first > 0.1
    decode_audio_to(late, tmp_path / "b.f32")
    low = np.fromfile(tmp_path / "b.f32", np.float32)
    high = decoded(late)
    at16 = onset(low[int(8 * 16_000):int(12 * 16_000)], 16_000) + 8
    at44 = onset(high[0, int(8 * SR):int(12 * SR)], SR) + 8
    assert abs(at16 - (DEMO_MARK - first)) <= 0.001 and abs(at44 - at16) <= 0.001
