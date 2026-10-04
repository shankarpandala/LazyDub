"""The offline render's stages so far (OFFLINE-RENDER §2.1-§2.8) on the mock backend and the demo video: fetch, speakers,
transcript and units, job.json and the stage files, pause and resume, a speaker-count re-run, torn rows, the units pass
off the event loop; voices from the whole video with their stored calibration and Hear voice sample, the brief in parts,
and the three translation lanes with their retries, preview range and coverage report. No models, no Claude (MockClaude
and stand-ins built on it), no network. All test text is original."""

from __future__ import annotations

import asyncio
import errno
import gc
import json
import logging
import math
import shutil
import sys
import threading
import time
import wave
from dataclasses import dataclass, replace

import numpy as np
import pytest

from maata_engine import dubber, render
from maata_engine.backends import mock
from maata_engine.backends.base import COARSE_STEP, Backend, Cancelled, Coverage, LineResult, Transcript, Wording
from maata_engine.backends.claude_translator import ClaudeTranslator
from maata_engine.backends.mock import MockClaude, MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.claude_cli import ClaudeCLIError
from maata_engine.dubber import CALIBRATION_TE, Dubber
from maata_engine.render import STAGES, RenderJob, RenderSettings, _read_rows
from maata_engine.resolve import DemoResolver, ResolvedVideo, ResolveError, decode_audio, decode_audio_to
from maata_engine.speakers import DiarBlock
from maata_engine.text.akshara import mixed_units
from maata_engine.types import SpeakerTurn, TimedWord, VoiceKind

URL = "https://youtu.be/demo0000001"
VID = "demo0000001"


class Transcriber(MockTranscriber):
    """The mock transcriber, counting its calls; it pauses `job` during call number `pause_on`, and takes `delay` s."""

    def __init__(self, pause_on: int | None = None, delay: float = 0.0) -> None:
        self.calls = 0
        self.pause_on = pause_on
        self.delay = delay
        self.job: RenderJob | None = None

    def transcribe(self, audio, language=None, speech=()):
        self.calls += 1
        if self.calls == self.pause_on:
            self.job.pause()  # from the worker thread, as the UI's pause arrives mid-chunk
        time.sleep(self.delay)
        return super().transcribe(audio, language, speech)


class UniqueTranscriber(Transcriber):
    """The mock's speech, but every sentence different (original text), so the line cache serves no line of a run from
    another line of it."""

    ADJ = ("quiet", "bright", "early", "narrow", "gentle", "heavy", "hidden", "simple")
    NOUN = ("river", "garden", "window", "letter", "market", "bridge", "kitchen", "lantern")
    VERB = ("painted", "cleaned", "opened", "found", "moved", "watched", "fixed", "closed")

    def transcribe(self, audio, language=None, speech=()):
        self.calls += 1
        words: list[TimedWord] = []
        t, i = 0.6, 0
        while t < len(audio) / 16_000 - 2:
            n = 100 * self.calls + i
            src = f"The {self.ADJ[n % 8]} {self.NOUN[n // 8 % 8]} was {self.VERB[n // 64 % 8]} today.".split()
            for wd in src:
                d = 0.18 + 0.04 * len(wd)
                words.append(TimedWord(wd, round(t, 3), round(t + d, 3)))
                t += d + 0.06
            t += 0.9
            i += 1
        return Transcript(words, language or "en")


def backend(transcriber: Transcriber | None = None, tts: MockTTS | None = None) -> Backend:
    return Backend("mock", "cpu", transcriber or Transcriber(), MockDiarizer(), MockSceneTranslator, tts or MockTTS())


def job_for(tmp_path, b: Backend, settings: RenderSettings | None = None, events: list | None = None) -> RenderJob:
    async def on_event(msg: dict) -> None:
        if events is not None:
            events.append(msg)

    job = RenderJob(b, DemoResolver(), tmp_path, URL, settings, on_event=on_event)
    if isinstance(b.transcriber, Transcriber):
        b.transcriber.job = job
    return job


def rows(tmp_path) -> list[dict]:
    return _read_rows(tmp_path / VID / "render" / "transcript.jsonl")


def in_order(keys: list[str]) -> list[str]:
    """Stage keys in the order of STAGES: the stages of a group (voices and the brief) finish in either order."""
    order = [k for k, _, _ in STAGES]
    return sorted(keys, key=order.index)


def contiguous(data: list[dict]) -> bool:
    return data[0]["a"] == 0.0 and all(r["a"] == p["next"] for p, r in zip(data, data[1:]))


async def test_fetch_through_units_complete(tmp_path):
    events: list[dict] = []
    b = backend()
    job = job_for(tmp_path, b, events=events)
    assert await job.run() == "done"
    d = tmp_path / VID / "render"
    doc = json.loads((d / "job.json").read_text())
    assert doc["status"] == "done" and doc["videoId"] == VID and doc["title"] == "Demo video"
    assert doc["duration"] == 180.0 and doc["settings"]["speakers"] is None
    assert [doc["stages"][k]["state"] for k, _, _ in STAGES] == ["done"] * len(STAGES)
    assert all(doc["stages"][k]["attempts"] == 1 for k, _, _ in STAGES if k != "separate")
    # this backend has no separator: no background sound, nothing for that stage to do, and job.json says so
    assert doc["stages"]["separate"]["attempts"] == 0 and doc["bed"]["separator"] is None and doc["bed"]["note"]
    # fetch: the analysis audio, decoded once from the demo's demo.mp4 (its AAC, to the stream's duration), and let go
    # of once the whole video's MP4 is written (§2.17); the demo's own file stays
    assert not (d / "audio16k.f32").exists() and len(job.audio) == 0 and (tmp_path / VID / "demo.mp4").is_file()
    assert doc["source"]["file"] == "demo.mp4" and doc["source"]["audioStart"] == 0.0
    assert doc["output"]["kind"] == "whole" and doc["output"]["path"].endswith("Demo video (Telugu).mp4")
    # speakers: the whole file at once; the 4 s blip near A's voice is merged into A, so two speakers
    diar = json.loads((d / "diarization.json").read_text())
    assert b.diarizer.calls == 1 and diar["step"] == 0.1
    assert diar["inputs"]["speakers"] == "auto" and diar["inputs"]["bounds"] == [1, 6]
    assert diar["labels"] == {"A": "S1", "B": "S2", "C": "S1"} and diar["merged"] == [["S3", "S1", "talk", 4.0]]
    assert [(s["id"], s["talkSeconds"], s["firstAt"]) for s in diar["speakers"]] == [("S1", 100.0, 0.0), ("S2", 80.0, 14.0)]
    assert list(job.registry.speakers) == ["S1", "S2"] and set(job.voices) == {"S1", "S2"}
    # transcript: one row per chunk after the inputs, contiguous from 0 to the end
    got = rows(tmp_path)
    assert got[0]["inputs"]["transcriber"] == "Transcriber" and "diarizer" not in got[0]["inputs"]
    assert contiguous(got[1:]) and got[-1]["next"] >= 179.8 and len(got) - 1 == b.transcriber.calls
    assert all(r["language"] == "en" for r in got[1:])
    # units: the whole transcript, ids in onset order, speakers from the settled registry
    units = [job.units[k] for k in sorted(job.units)]
    assert len(units) == doc["stages"]["units"]["total"] > 30
    assert [st.unit.start for st in units] == sorted(st.unit.start for st in units)
    assert {st.unit.speaker for st in units} == {"S1", "S2"}
    assert all(st.unit.speaker == "S1" for st in units if 100.0 <= st.unit.start < 104.0)
    assert all(st.next_start == nxt.unit.start for st, nxt in zip(units, units[1:])) and units[-1].next_start is None
    assert all(st.speech_s > 0 for st in units)
    # every stage traced when it finished; the UI heard the job and the speakers found
    trace = [json.loads(x) for x in (tmp_path / VID / "units.jsonl").read_text().splitlines()]
    assert in_order([e["key"] for e in trace if e["event"] == "stage"]) == [k for k, _, _ in STAGES]
    assert sum(e["event"] == "asr" for e in trace) == b.transcriber.calls
    found = [e for e in events if e["type"] == "speakers_found"]
    assert len(found) == 1 and [s["id"] for s in found[0]["speakers"]] == ["S1", "S2"]
    assert len(found[0]["speakers"][0]["activity"]) == 120
    assert found[0]["merged"] == [{"from": "S3", "into": "S1", "why": "talk", "talkSeconds": 4.0}]
    assert events[-1]["type"] == "render" and events[-1]["status"] == "done"
    assert [s["key"] for s in events[-1]["stages"]] == [k for k, _, _ in STAGES]


async def test_pause_mid_transcript_then_resume(tmp_path):
    t1 = Transcriber(pause_on=2)
    b = backend(t1)
    assert await job_for(tmp_path, b).run() == "paused"
    doc = json.loads((tmp_path / VID / "render" / "job.json").read_text())
    assert doc["status"] == "paused" and doc["stage"] == "transcript"
    assert doc["stages"]["transcript"]["state"] == "todo" and doc["stages"]["transcript"]["attempts"] == 0
    assert len(rows(tmp_path)) - 1 == 2 == t1.calls  # the chunk heard when the pause came is kept
    # Resumed by a new job (as after a restart), with the same diarizer: nothing heard again.
    t2 = Transcriber()
    b.transcriber = t2
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    assert b.diarizer.calls == 1  # served from diarization.json
    data = rows(tmp_path)[1:]
    assert contiguous(data) and data[-1]["next"] >= 179.8 and t1.calls + t2.calls == len(data)
    assert job.doc["stages"]["transcript"]["attempts"] == 1
    trace = [json.loads(x) for x in (tmp_path / VID / "units.jsonl").read_text().splitlines()]
    asr = [e["a"] for e in trace if e["event"] == "asr"]
    assert asr[2] == round(max(0.0, data[1]["next"] - 0.3), 2)  # the first chunk after the pause starts at the last row's next
    stages = [(e["key"], e["cached"]) for e in trace if e["event"] == "stage" and e["key"] in ("speakers", "transcript",
                                                                                                  "units")]
    assert stages[-3:] == [("speakers", True), ("transcript", False), ("units", False)]


async def test_a_pause_then_an_immediate_run_never_runs_twice(tmp_path):
    """Resume straight after a pause, while the paused run is still inside an ASR chunk: the new run waits for that one
    to stop, so no chunk is heard twice. A pause that comes after the resume was asked for wins over it."""
    t = Transcriber(delay=0.05)
    job = job_for(tmp_path, backend(t))
    first = asyncio.create_task(job.run())
    while t.calls < 2:
        await asyncio.sleep(0.005)
    job.pause()
    second = asyncio.create_task(job.run())
    assert await first == "paused" and await second == "done"
    data = rows(tmp_path)[1:]
    assert contiguous(data) and len(data) == t.calls and len({r["a"] for r in data}) == len(data)
    # Pause, resume, pause again, all during one chunk: the resume never starts.
    t = Transcriber(delay=0.05)
    job = job_for(tmp_path / "again", backend(t))
    first = asyncio.create_task(job.run())
    while t.calls < 2:
        await asyncio.sleep(0.005)
    job.pause()
    second = asyncio.create_task(job.run())
    await asyncio.sleep(0)  # the resume is asked for (it waits for the first run)
    job.pause()
    assert await first == "paused" and await second == "paused"
    assert len(_read_rows(tmp_path / "again" / VID / "render" / "transcript.jsonl")) - 1 == t.calls == 2


async def test_a_job_resumed_by_its_video_id_keeps_its_settings(tmp_path):
    b = backend()
    settings = RenderSettings(speakers=1, style="formal", speed_cap=1.1)
    assert await job_for(tmp_path, b, settings).run() == "done"
    job = RenderJob(b, DemoResolver(), tmp_path, VID)  # as the engine resumes a job from its cache
    assert job.settings == settings and (job.style, job.speed_cap, job.allow_freeze) == ("formal", 1.1, False)
    assert await job.run() == "done" and b.diarizer.calls == 1  # the count of 1 is kept: diarization from disk
    assert json.loads((tmp_path / VID / "render" / "job.json").read_text())["settings"] == settings.to_json()
    assert RenderJob(b, DemoResolver(), tmp_path / "new", VID).settings == RenderSettings()  # a new job: the defaults
    # A job.json from before freezes went (§2.11): its allowFreeze is ignored, and never written again.
    old = {**settings.to_json(), "allowFreeze": True}
    assert "allowFreeze" not in settings.to_json() and RenderSettings.from_json(old) == settings


class WaitingDiarizer(MockDiarizer):
    """Takes up to 5 s, stopping as soon as its cancel event is set, as pyannote's hook stops it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def diarize(self, audio, *, cancel=None, **kwargs):
        self.started.set()
        if cancel is not None and cancel.wait(5):
            self.calls += 1
            raise Cancelled("diarization cancelled")
        return super().diarize(audio, cancel=cancel, **kwargs)


async def test_cancelling_the_run_stops_diarization_at_once(tmp_path):
    """The engine stopping cancels the run's task: the diarization it waits out is told to stop, not run to its end,
    and the job stays `running` (the engine finds it interrupted), but not as a killed start: no coarse step next."""
    b = backend()
    b.diarizer = WaitingDiarizer()
    task = asyncio.create_task(job_for(tmp_path, b).run())
    while not b.diarizer.started.is_set():
        await asyncio.sleep(0.005)
    t0 = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - t0 < 2.0 and b.diarizer.calls == 1
    doc = json.loads((tmp_path / VID / "render" / "job.json").read_text())
    assert doc["status"] == "running" and doc["stage"] == "speakers"
    assert doc["stages"]["speakers"]["state"] == "todo" and doc["stages"]["speakers"]["attempts"] == 0
    b.diarizer = MockDiarizer()
    job = job_for(tmp_path, b)
    assert await job.run() == "done" and b.diarizer.step is None and job.doc["stages"]["speakers"]["attempts"] == 1


class Offline(DemoResolver):
    """No network, or YouTube asking for a sign-in check."""

    def resolve(self, ref, cache_dir, download=True, progress=None):
        raise ResolveError("YouTube is asking for a sign-in check. Try again in a few minutes.")


class FileResolver(DemoResolver):
    """A 'download' that copies a wav into the cache, where yt-dlp leaves audio.<ext>."""

    def __init__(self, wav) -> None:
        self.wav = wav
        self.calls = 0

    def resolve(self, ref, cache_dir, download=True, progress=None):
        self.calls += 1
        path = cache_dir / ref.video_id / "audio.wav"
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.wav, path)
        return ResolvedVideo(ref.video_id, "A tone", 30.0, "Maata tests", path if path.exists() else None,
                             "Thirty seconds of a tone.", ((0.0, "Tone"),), ("test",))


def write_tone(path, seconds: float, hz: float) -> None:
    sr = 16_000
    t = np.arange(int(seconds * sr)) / sr
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((0.3 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16).tobytes())


async def test_a_fetched_job_resumes_offline(tmp_path):
    """Everything fetch made is on disk: a resume needs no resolver (no network), with the metadata job.json kept."""
    b = backend()
    assert await job_for(tmp_path, b).run() == "done"
    job = RenderJob(b, Offline(), tmp_path, VID)
    assert await job.run() == "done"
    v = job.video
    assert (v.title, v.channel, v.duration, v.audio_path) == ("Demo video", "Maata demo", 180.0, tmp_path / VID / "demo.mp4")
    assert (v.description, v.chapters, v.tags) == ("A synthetic video for the demo engine.", ((0.0, "Start"),), ("demo",))
    # (the whole video's export let go of the analysis audio: decoded again from demo.mp4, with no resolver call)
    trace = [json.loads(x) for x in (tmp_path / VID / "units.jsonl").read_text().splitlines()]
    assert [e["cached"] for e in trace if e["event"] == "stage" and e["key"] == "fetch"] == [False, False]
    assert job.doc["stages"]["fetch"]["attempts"] == 1


async def test_a_downloaded_job_resumes_offline_until_its_audio_changes(tmp_path):
    pytest.importorskip("av")
    wav = tmp_path / "tone.wav"
    write_tone(wav, 30.0, 220.0)
    cache, b, online = tmp_path / "cache", backend(), FileResolver(wav)
    assert await RenderJob(b, online, cache, VID).run() == "done" and online.calls == 2  # metadata, then the download
    assert json.loads((cache / VID / "render" / "job.json").read_text())["source"]["file"] == "audio.wav"
    analysis = cache / VID / "render" / "audio16k.f32"
    assert not analysis.exists()  # a finished render lets go of its analysis audio (§2.1)...
    job = RenderJob(b, Offline(), cache, VID)
    assert await job.run() == "done"
    assert job.video.audio_path == cache / VID / "audio.wav" and job.video.title == "A tone"
    # ...and a re-run decodes it again from the source, offline
    assert job.doc["stages"]["fetch"]["attempts"] == 1 and not analysis.exists()
    trace = [json.loads(x) for x in (cache / VID / "units.jsonl").read_text().splitlines()]
    assert [e["cached"] for e in trace if e["event"] == "stage" and e["key"] == "fetch"] == [False, False]
    # Other bytes in the cache (a re-download that differs): fetch must run again, which needs the resolver.
    write_tone(cache / VID / "audio.wav", 30.0, 330.0)
    job = RenderJob(b, Offline(), cache, VID)
    assert await job.run() == "failed" and job.doc["stage"] == "fetch" and "sign-in" in job.doc["error"]


class LateDiarizer(MockDiarizer):
    """A from the start, a 4 s blip near A's voice at 100 s, and B only from 104 s: the blip is registered before B."""

    def diarize(self, audio, **kwargs):
        self.calls += 1
        end = len(audio) / 16_000
        turns = [SpeakerTurn("A", 0.0, 100.0), SpeakerTurn("C", 100.0, 104.0), SpeakerTurn("B", 104.0, end)]
        e = np.eye(8, dtype=np.float32)
        return DiarBlock(0.0, end, turns, list(turns), {"A": e[0], "B": e[1], "C": (0.95 * e[0] + 0.31 * e[7])}, step=0.1)


async def test_a_merged_speaker_is_never_named_after_a_live_one(tmp_path):
    b = backend()
    b.diarizer = LateDiarizer()
    events: list[dict] = []
    assert await job_for(tmp_path, b, events=events).run() == "done"
    diar = json.loads((tmp_path / VID / "render" / "diarization.json").read_text())
    assert [s["id"] for s in diar["speakers"]] == ["S1", "S2"] and diar["labels"] == {"A": "S1", "B": "S2", "C": "S1"}
    # The blip was first registered as S2, which B is now: it is named above every live id.
    assert diar["merged"] == [["S3", "S1", "talk", 4.0]]
    found = next(e for e in events if e["type"] == "speakers_found")
    assert found["merged"] == [{"from": "S3", "into": "S1", "why": "talk", "talkSeconds": 4.0}]


async def test_a_speakers_stage_that_failed_keeps_the_pipelines_step(tmp_path):
    """An ordinary error is not a kill: the next start diarizes at the pipeline's own step. After a killed start, the
    coarse step holds until the stage finishes, even when an ordinary error came in between."""
    for name, stage, step in (("failed", {"state": "failed", "attempts": 1}, None),
                              ("killed", {"state": "failed", "attempts": 2, "interrupted": 1}, COARSE_STEP)):
        b = backend()
        d = tmp_path / name / VID / "render"
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"stages": {"speakers": stage}}))
        job = job_for(tmp_path / name, b)
        assert await job.run() == "done"
        assert b.diarizer.step == step and job.doc["stages"]["speakers"]["attempts"] == stage["attempts"] + 1


async def test_speaker_count_reruns_diarization_but_not_asr_and_ids_stay(tmp_path):
    b = backend()
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    heard = b.transcriber.calls
    transcript = (tmp_path / VID / "render" / "transcript.jsonl").read_bytes()
    one = job_for(tmp_path, b, RenderSettings(speakers=1))
    assert await one.run() == "done"
    assert b.diarizer.calls == 2 and b.transcriber.calls == heard
    assert (tmp_path / VID / "render" / "transcript.jsonl").read_bytes() == transcript
    assert list(one.registry.speakers) == ["S1"] and {st.unit.speaker for st in one.units.values()} == {"S1"}
    assert one.doc["settings"]["speakers"] == 1 and one.doc["stages"]["speakers"]["attempts"] == 1
    diar = json.loads((tmp_path / VID / "render" / "diarization.json").read_text())
    assert diar["inputs"]["speakers"] == 1 and diar["merged"] == []  # a count is never second-guessed
    # Back to two: the same people get the same ids, so their line keys and voices carry over.
    two = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await two.run() == "done"
    assert b.diarizer.calls == 3 and b.transcriber.calls == heard
    times = (5.0, 20.0, 50.0, 120.0, 160.0)  # the 14 s grid's turns 0, 1, 3, 8, 11; away from the auto run's blip
    assert [two.registry.speaker_at(t) for t in times] == [first.registry.speaker_at(t) for t in times] \
        == ["S1", "S2", "S2", "S1", "S2"]
    again = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await again.run() == "done" and b.diarizer.calls == 3  # unchanged inputs: served from disk


async def test_an_interrupted_speakers_stage_runs_again_at_the_coarse_step(tmp_path):
    b = backend()
    d = tmp_path / VID / "render"
    d.mkdir(parents=True)
    # The engine died during diarization (an out-of-memory kill leaves no exception): the stage is still `running`.
    (d / "job.json").write_text(json.dumps({"stages": {"speakers": {"state": "running", "attempts": 1}}}))
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    assert b.diarizer.step == COARSE_STEP and json.loads((d / "diarization.json").read_text())["step"] == COARSE_STEP
    assert job.doc["stages"]["speakers"]["attempts"] == 2


# The engine died writing the third row: cut in ASCII, or inside a UTF-8 character (rows are written unescaped).
TORN = {"ascii": b'{"a": 118.2, "b": 177.0, "next": 17',
        "inside-a-character": '{"a": 118.2, "words": [["so\u2014'.encode()[:-2]}


@pytest.mark.parametrize("tail", TORN.values(), ids=TORN.keys())
async def test_a_torn_transcript_row_is_ignored(tmp_path, tail):
    clean = tmp_path / "clean"
    ref = job_for(clean, backend())
    assert await ref.run() == "done"
    b = backend(Transcriber(pause_on=2))
    assert await job_for(tmp_path, b).run() == "paused"
    path = tmp_path / VID / "render" / "transcript.jsonl"
    with path.open("ab") as f:
        f.write(tail)
    b.transcriber = Transcriber()
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    lines = path.read_bytes().splitlines()
    assert sum(1 for line in lines if not _parses(line)) == 1 and _parses(lines[-1])  # the next row has its own line
    assert contiguous(rows(tmp_path)[1:])
    assert [(u.unit.speaker, u.unit.start, u.unit.text) for u in job.units.values()] == \
           [(u.unit.speaker, u.unit.start, u.unit.text) for u in ref.units.values()]


def _parses(line: bytes) -> bool:
    try:
        json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def test_a_row_keeps_a_line_separator_inside_its_text(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"a": 0.0, "words": [["one\u2028two\x85", 0.1, 0.4, 0.9]]}
    render._append_row(path, row)
    render._append_row(path, {"a": 1.0})
    assert _read_rows(path) == [row, {"a": 1.0}]


def test_peak_rss_is_none_without_the_resource_module(monkeypatch):
    assert render._peak_rss_bytes() > 0
    monkeypatch.setitem(sys.modules, "resource", None)  # as on Windows
    assert render._peak_rss_bytes() is None


async def test_heavy_passes_do_not_block_the_loop(tmp_path, monkeypatch, caplog):
    """2,000 synthetic sentences are segmented in a worker thread: no callback of the event loop runs for 0.1 s or more
    (asyncio's debug mode reports each one that does), and a coroutine ticking every 5 ms keeps ticking. Each chunk's
    segmentation is slowed a little, so the pass lasts long enough that a pass on the loop would show.
    What is measured is the loop's own stalls, not the largest gap between ticks: inside the full suite a generation-2
    collection of the cyclic GC over the heap the earlier tests left took 0.21 s once, and it stops every thread,
    whichever thread set it off (measured with gc.callbacks; asyncio reported no slow callback then). The GC is off for
    the pass for the same reason: one set off inside a loop callback would count against the loop."""
    real = render.segment

    def slow(*args, **kwargs):
        time.sleep(0.003)
        return real(*args, **kwargs)

    monkeypatch.setattr(render, "segment", slow)
    job = job_for(tmp_path, backend())
    job.render_dir.mkdir(parents=True)
    n, per = 2000, 3.0  # one sentence of 2.3 s every 3 s: each is a unit of its own
    end = n * per
    turns = [SpeakerTurn("AB"[k % 2], k * 15.0, min((k + 1) * 15.0, end)) for k in range(math.ceil(end / 15.0))]
    job.registry.add_block(DiarBlock(0.0, end, turns, list(turns), {"A": np.eye(4)[0], "B": np.eye(4)[1]}))
    words = []
    for k in range(n):
        t = k * per
        words += [[w, t + 0.4 * i, t + 0.4 * i + 0.3, 0.9]
                  for i, w in enumerate(("We", "walked", "along", "the", "river", "today."))]
    job._rows = [{"a": a, "b": min(a + 60.0, end), "next": min(a + 60.0, end),
                  "words": [w for w in words if a <= w[1] < a + 60.0]} for a in np.arange(0.0, end, 60.0).tolist()]
    gaps, stop = [], False

    async def tick() -> None:
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    loop = asyncio.get_running_loop()
    caplog.set_level(logging.WARNING, logger="asyncio")
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1
    gc.disable()
    try:
        ticker = asyncio.create_task(tick())
        t0 = time.perf_counter()
        await asyncio.create_task(job._units())  # a task's steps, as a stage's: each one begun in debug mode is timed
        took = time.perf_counter() - t0
        stop = True
        await ticker
    finally:
        gc.enable()
        loop.set_debug(False)
    assert len(job.units) == n and [job.units[k].unit.start for k in range(n)] == [k * per for k in range(n)]
    assert took > 0.25
    held = [r.getMessage() for r in caplog.records if r.name == "asyncio" and r.getMessage().startswith("Executing")]
    assert held == []  # the loop was never held for the pass
    assert len(gaps) >= 10


class Clocks:
    """The wall clock and the monotonic one, set by hand: `time.monotonic()` stops while a Mac sleeps (§4)."""

    def __init__(self) -> None:
        self.wall, self.mono = 1_700_000_000.0, 500.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def perf_counter(self) -> float:
        return self.mono

    def run(self, awake: float, asleep: float = 0.0) -> None:
        self.wall += awake + asleep
        self.mono += awake


async def test_a_sleep_is_noticed_at_the_next_progress_tick(tmp_path, monkeypatch):
    """§4: at each progress tick the wall clock is compared with the monotonic one; a wall-clock gap more than
    SLEEP_GAP s past the monotonic gap is a sleep, kept as job.json's `slept` (the last tick before, the first after,
    wall clock) and sent in the render snapshot. Time awake, however long, is not; a new run starts with none."""
    clocks = Clocks()
    monkeypatch.setattr(render, "time", clocks)
    events: list[dict] = []
    job = job_for(tmp_path, backend(), events=events)
    job.render_dir.mkdir(parents=True)
    job.doc["status"] = "running"
    job._progress("transcript", 10.0, 180.0)
    clocks.run(awake=3600.0)  # a long stage, awake
    job._progress("transcript", 20.0, 180.0)
    clocks.run(awake=5.0, asleep=render.SLEEP_GAP - 1)  # under SLEEP_GAP: a clock adjustment, not a sleep
    job._progress("transcript", 30.0, 180.0)
    assert job.doc["slept"] is None
    before = clocks.wall
    clocks.run(awake=2.0, asleep=7 * 3600.0)  # the lid closed overnight
    job._progress("transcript", 40.0, 180.0)
    slept = {"at": round(before, 1), "resumed": round(clocks.wall, 1)}
    assert job.doc["slept"] == slept
    await asyncio.sleep(0.01)  # its save: at once, not after PROGRESS_EVERY
    assert json.loads((job.render_dir / "job.json").read_text())["slept"] == slept
    assert events[-1]["type"] == "render" and events[-1]["slept"] == slept and job.snapshot()["slept"] == slept
    job._ticked = (clocks.wall - 10_000.0, clocks.mono)  # a run starts afresh: no note, and no gap from before
    job.doc.update(status="paused")
    task = asyncio.create_task(job.run())
    await asyncio.sleep(0.01)
    job.pause()
    await task
    assert job.doc["slept"] is None


async def test_indexed_queries_match_the_scans(tmp_path):
    job = job_for(tmp_path, backend())
    assert await job.run() == "done"
    units = [job.units[k] for k in sorted(job.units)]
    for st in units[::3]:
        st.line, st.tier = LineResult(st.unit.id, {"full": Wording(f"పదం {st.unit.id}")}), "full"
    for st in units:
        u = st.unit
        assert job._context_for(st) == Dubber._context_for(job, st)
        assert job._slot(st) == Dubber._slot(job, st)
        assert job._window(st) == Dubber._window(job, st)
        breaks = ((u.start + 0.5, u.end + 3.0), (u.start - 6.0, u.start - 0.2))
        assert job._heard_in(st, breaks) == Dubber._heard_in(job, st, breaks)


def test_decode_audio_to_writes_float32_frame_by_frame(tmp_path):
    pytest.importorskip("av")
    src, dest = tmp_path / "tone.wav", tmp_path / "audio16k.f32"
    sr = 8_000
    t = np.arange(int(1.5 * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.stack([tone, tone], axis=1).tobytes())
    n = decode_audio_to(src, dest, 16_000)
    got = np.fromfile(dest, np.float32)
    assert n == len(got) and abs(n - 24_000) <= 64
    assert not dest.with_name(dest.name + ".part").exists()
    np.testing.assert_allclose(got, decode_audio(src, 16_000), atol=1e-6)
    assert 0.2 < float(np.abs(got[1000:-1000]).max()) < 0.5  # the tone, not silence (a -3 dB stereo downmix: 0.42)


# ---- voices (§2.6) --------------------------------------------------------------------------------------------------
@dataclass
class MelTake:
    """Chatterbox's MelTake as the render sees it: what was said, its natural length, a mel (80 bins, 50 frames a second,
    made from the text), whether it ran to its cap, and the timings VoiceCost reads."""

    text: str
    seconds: float
    mel: np.ndarray
    capped: bool = False
    t3_s: float = 0.01
    t3_tokens: int = 10
    t3_steps: int = 10
    flow_s: float = 0.0
    cfm_steps: int = 6


class MelTTS(MockTTS):
    """A CPU stand-in for Chatterbox's voice and take API, in the style of test_batched_takes.py: voices from parts; mel
    takes of `overhead` + aksharas / `rate` s, one at a time (`synthesize_mel`, calibration) or `n` in a batch
    (`synthesize_takes`, each `spread` longer than the one before; takes of `fail` run to their cap); `pack_take` and
    `unpack_take` as Chatterbox keeps a take; and a vocoder that records each (take, rate) it is asked for and gives a
    tone of the take's length. It counts the voices built, the calibration takes and the batches."""

    cfg_weight, exaggeration, cfm_steps = 0.5, 0.7, 6

    def __init__(self, rate: float = 3.0, overhead: float = 0.3, spread: float = 0.0, fail: str | None = None) -> None:
        self.rate, self.overhead, self.spread, self.fail = rate, overhead, spread, fail
        self.built = self.takes = 0
        self.batches: list[tuple[str, int]] = []
        self.vocoded: list[tuple[MelTake, float]] = []

    def prepare_voice_parts(self, timbre, pool, prompt, sample_rate, weight):
        self.built += 1
        return {"f0": 140.0}

    def _take(self, text: str, max_seconds: float | None, k: int = 0) -> MelTake:
        seconds = (self.overhead + mixed_units(text) / self.rate) * (1.0 + self.spread * k)
        capped = self.fail is not None and self.fail in text
        if max_seconds:
            seconds = max_seconds if capped else min(seconds, max_seconds)
        frames = max(1, round(50 * seconds))
        mel = np.sin(0.01 * np.arange(80 * frames, dtype=np.float32).reshape(80, frames) + len(text) + k)
        return MelTake(text, seconds, mel.astype(np.float32), capped)

    def synthesize_mel(self, text, voice, language="te", max_seconds=None) -> MelTake:
        self.takes += 1
        return self._take(text, max_seconds)

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1) -> list[MelTake]:
        self.batches.append((text, n))
        return [self._take(text, max_seconds, k) for k in range(n)]

    def vocode(self, take: MelTake, rate: float = 1.0) -> np.ndarray:
        self.vocoded.append((take, rate))
        n = int(take.seconds / rate * self.sample_rate)
        return (0.1 * np.sin(2 * np.pi * 140.0 * np.arange(n) / self.sample_rate)).astype(np.float32)

    def pack_take(self, take: MelTake) -> dict[str, np.ndarray]:
        return {"mel": take.mel.astype(np.float16), "text": np.str_(take.text)}

    def unpack_take(self, d: dict[str, np.ndarray]) -> MelTake:
        return MelTake(str(d["text"]), float(d["seconds"]), np.asarray(d["mel"], np.float32))

    def calibration_vocodes(self) -> int:
        return sum(1 for take, _ in self.vocoded if take.text in {w.spoken for w in CALIBRATION_TE})


def trace_of(tmp_path) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / VID / "units.jsonl").read_text().splitlines()]


async def test_voices_are_stored_and_a_resume_rebuilds_them_with_no_synthesis(tmp_path):
    tts = MelTTS()
    b = backend(tts=tts)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    d = tmp_path / VID / "render"
    doc = json.loads((d / "voices.json").read_text())
    assert list(doc["speakers"]) == ["S1", "S2"]  # most talk first
    assert (tts.built, tts.takes, tts.calibration_vocodes()) == (2, 2 * len(CALIBRATION_TE), 2)
    for sid, e in doc["speakers"].items():
        v = job.voices[sid]
        assert v.status == "cloned" and v.kind is VoiceKind.CLONED
        assert v.key.reference == e["key"]["reference"] == f"v1:{e['hash']}"
        assert (e["how"], e["key"]["voice"], e["key"]["cfg"], e["key"]["exaggeration"]) == ("single-clip", sid, 0.5, 0.7)
        assert e["timbre"][1] - e["timbre"][0] >= 8.0 and e["clips"] and e["speech"] > 0
        assert all(a >= 60.0 for a, _ in [e["timbre"]] + e["clips"][:3])  # the best speech, after the first minute
        assert [spoken for spoken, _ in e["calibration"]] == [w.spoken for w in CALIBRATION_TE]
        assert e["pace"] == pytest.approx(3.0) and e["overhead"] == pytest.approx(0.3)
        assert job.estimator.rate(v.key) == pytest.approx(3.0)
        # The Hear voice sample: the first calibration take, vocoded (synthetic Telugu, never the source audio).
        sample = np.load(d / "voices" / f"{sid}.npy")
        assert e["sample"] and sample.dtype == np.float16 and len(sample) == int(e["calibration"][0][1] * 24_000)
    stages = json.loads((d / "job.json").read_text())["stages"]
    assert [stages["voices"][k] for k in ("state", "done", "total", "unit")] == ["done", 2, 2, "speakers"]
    assert [stages["brief"][k] for k in ("state", "done", "total", "unit")] == ["done", 1, 1, "parts"]
    assert [stages["translate"][k] for k in ("state", "done", "total", "unit")] == ["done", len(job.units),
                                                                                   len(job.units), "lines"]
    # Resumed by a new job (as after a restart): each voice is rebuilt from its stored spans and its pace re-fitted from
    # the stored calibration, with no synthesis and no vocoding (every line comes back from its take row).
    b.tts = tts2 = MelTTS()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert (tts2.built, tts2.takes, tts2.batches, tts2.vocoded) == (2, 0, [], [])
    assert json.loads((d / "voices.json").read_text()) == doc
    for sid in doc["speakers"]:
        k1, k2 = job.voices[sid].key, again.voices[sid].key
        assert k2 == k1 and again.voices[sid].status == "cloned"
        assert (again.estimator.rate(k2), again.estimator.overhead(k2)) == (job.estimator.rate(k1), job.estimator.overhead(k1))
    trace = trace_of(tmp_path)
    assert [e["speaker"] for e in trace if e["event"] == "clone"] == ["S1", "S2"]  # the first run's only
    assert [e["cached"] for e in trace if e["event"] == "stage" and e["key"] == "voices"] == [False, True]


async def test_a_voice_of_the_mock_has_a_hear_voice_sample_too(tmp_path):
    """MockTTS has no mel takes, so it isn't calibrated: its sample is the first calibration sentence said by the
    speaker's voice (synthetic, never the source audio), as the demo's Hear voice needs."""
    job = job_for(tmp_path, backend())
    assert await job.run() == "done"
    d = tmp_path / VID / "render"
    doc = json.loads((d / "voices.json").read_text())["speakers"]
    for sid, e in doc.items():
        assert e["sample"] and e["calibration"] == []
        want = MockTTS().synthesize(CALIBRATION_TE[0].spoken, job.voices[sid].voice)
        np.testing.assert_allclose(np.load(d / "voices" / f"{sid}.npy").astype(np.float32), want, atol=1e-3)
    again = job_for(tmp_path, backend())
    assert await again.run() == "done"
    assert json.loads((d / "voices.json").read_text())["speakers"] == doc
    assert all((d / "voices" / f"{sid}.npy").is_file() for sid in doc)


class ShortSpeakerDiarizer(MockDiarizer):
    """A throughout but for 3 s of B at 170 s: too little clean speech for B's clone."""

    def diarize(self, audio, **kwargs):
        self.calls += 1
        end = len(audio) / 16_000
        turns = [SpeakerTurn("A", 0.0, 170.0), SpeakerTurn("B", 170.0, 173.0), SpeakerTurn("A", 173.5, end)]
        e = np.eye(8, dtype=np.float32)
        return DiarBlock(0.0, end, turns, list(turns), {"A": e[0], "B": e[1]}, step=0.1)


async def test_a_speaker_with_too_little_clean_speech_gets_a_preset(tmp_path):
    tts = MelTTS()
    b = backend(tts=tts)
    b.diarizer = ShortSpeakerDiarizer()
    job = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await job.run() == "done"
    d = tmp_path / VID / "render"
    doc = json.loads((d / "voices.json").read_text())
    assert doc["speakers"]["S2"] == {"how": "preset", "preset": "preset_f", "refSeconds": pytest.approx(2.4),
                                     "sample": False}
    assert (job.voices["S2"].status, job.voices["S2"].kind, job.voices["S2"].key) == ("preset", VoiceKind.PRESET, None)
    assert job.voices["S1"].status == "cloned" and tts.built == 1 and tts.takes == len(CALIBRATION_TE)
    assert not (d / "voices" / "S2.npy").exists() and (d / "voices" / "S1.npy").exists()
    b.tts = tts2 = MelTTS()
    again = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await again.run() == "done"
    assert (tts2.built, tts2.takes) == (1, 0) and again.voices["S2"].status == "preset"


class Stealing(MockDiarizer):
    """The mock's two speakers on their 14 s grid, with B speaking over [a, b) of A's time as well."""

    SPAN = (0.0, 0.0)

    def diarize(self, audio, **kwargs):
        block = super().diarize(audio, **kwargs)
        a, b = self.SPAN
        turns = []
        for t in block.turns:
            if t.speaker == "A" and t.start < b and a < t.end:
                turns += [SpeakerTurn("A", t.start, a), SpeakerTurn("B", a, b), SpeakerTurn("A", b, t.end)]
            else:
                turns.append(t)
        turns = [t for t in turns if t.end > t.start]
        return DiarBlock(block.start, block.end, turns, list(turns), block.centroids, step=block.step)


class StealsLittle(Stealing):
    SPAN = (90.0, 93.0)   # 3 s of the speech S1's voice was built from


class StealsMuch(Stealing):
    SPAN = (112.0, 126.0)  # a whole turn S1's voice has two clips in


async def test_a_speaker_keeps_their_voice_while_its_spans_are_still_their_speech(tmp_path):
    """After a speakers re-run, S1 keeps a voice while at least 90 % of the speech in its spans is still theirs; below
    that it is made again, and S2, whose speech didn't move, keeps theirs."""
    b = backend(tts=MelTTS())
    assert await job_for(tmp_path, b).run() == "done"
    path = tmp_path / VID / "render" / "voices.json"
    first = json.loads(path.read_text())["speakers"]
    spans = [tuple(first["S1"]["timbre"])] + [tuple(c) for c in first["S1"]["clips"]]
    b.diarizer, b.tts = StealsLittle(), MelTTS()
    little = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await little.run() == "done"
    own = little._own_speech("S1", spans)
    assert 0.9 * first["S1"]["speech"] <= own < first["S1"]["speech"]  # some of it is S2's now...
    assert (b.tts.built, b.tts.takes) == (2, 0) and json.loads(path.read_text())["speakers"] == first  # ...it is kept
    b.diarizer, b.tts = StealsMuch(), MelTTS()
    much = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await much.run() == "done"
    assert much._own_speech("S1", spans) < 0.9 * first["S1"]["speech"]
    now = json.loads(path.read_text())["speakers"]
    assert (b.tts.built, b.tts.takes) == (2, len(CALIBRATION_TE))  # S1's voice made again, S2's rebuilt from disk
    assert now["S1"]["hash"] != first["S1"]["hash"] and now["S2"] == first["S2"]
    assert much.voices["S1"].key.reference == f"v1:{now['S1']['hash']}"


# ---- brief (§2.7) ---------------------------------------------------------------------------------------------------
async def test_the_brief_is_made_before_the_first_scene_call(tmp_path):
    job = job_for(tmp_path, backend())
    assert await job.run() == "done"
    calls = job.tr.cli.calls
    kinds = [c["call"] for c in calls]
    assert kinds[0] == "brief" and kinds.count("brief") == 1 and "scene" in kinds
    assert job.tr.brief.version == 1 and job.doc["brief"] == {"version": 1, "parts": 1, "made": 1, "gaveUp": None}
    brief = calls[0]["message"]
    assert [s["id"] for s in brief["video"]["speakers"]] == ["S1", "S2"]  # the whole file's talk shares
    assert [(x["speaker"], x["en"]) for x in brief["transcript"]] == [(st.unit.speaker, st.unit.text) for st in job._order]
    assert all(BRIEF_V1 in c["system"] for c in calls if c["call"] == "scene")  # every scene under the brief


BRIEF_V1 = "VIDEO BRIEF (version 1)"


async def test_a_long_transcript_makes_the_brief_in_parts_and_a_resume_finds_them(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "BRIEF_PART_WORDS", 200)
    b = backend()
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    kinds = [c["call"] for c in job.tr.cli.calls]
    briefs = [c["message"] for c in job.tr.cli.calls if c["call"] == "brief"]
    n = len(briefs)
    assert n >= 3 and max(i for i, k in enumerate(kinds) if k == "brief") < kinds.index("scene")
    assert "previous" not in briefs[0] and all(m["previous"]["topic"] == "Demo video" for m in briefs[1:])
    # The parts are the whole transcript in order, each of whole transcript rows and at most 200 words.
    assert [(x["speaker"], x["en"]) for m in briefs for x in m["transcript"]] == \
           [(st.unit.speaker, st.unit.text) for st in job._order]
    assert all(sum(len(x["en"].split()) for x in m["transcript"]) <= 200 for m in briefs)
    starts = [r["a"] for r in job._rows]
    at = np.cumsum([0] + [len(m["transcript"]) for m in briefs[:-1]])
    assert [max(a for a in starts if a <= job._order[k].unit.start) for k in at] == starts[:n]  # parts start at rows
    assert job.tr.brief.version == n and job.doc["brief"] == {"version": n, "parts": n, "made": n, "gaveUp": None}
    stage = job.doc["stages"]["brief"]
    assert (stage["done"], stage["total"]) == (n, n)
    # A resumed render finds every part in briefs.jsonl, and every line in lines.jsonl.
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert again.tr.cli.calls == [] and again.tr.brief.version == n
    hits = [e["cache_hit"] for e in trace_of(tmp_path) if e["event"] == "brief"]
    assert hits == [False] * n + [True] * n


class UnusableBriefClaude(MockClaude):
    """No usable reply to any brief part after the first."""

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "brief" and "previous" in json.loads(prompt):
            self.calls.append({"call": call, "message": json.loads(prompt), "failed": True})
            raise ClaudeCLIError("bad_output", "the reply didn't match the schema")
        return super().ask(system, prompt, schema, call, **kw)


def translator_with(cli: MockClaude):
    def make(cache_dir, video_id, brief, **kw) -> ClaudeTranslator:
        return ClaudeTranslator(cache_dir, video_id, brief, cli=cli, **kw)

    return make


async def test_after_three_unusable_replies_the_brief_goes_on_with_what_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "BRIEF_PART_WORDS", 200)
    cli = UnusableBriefClaude()
    b = backend()
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    briefs = [c for c in cli.calls if c["call"] == "brief"]
    assert [c.get("failed", False) for c in briefs] == [False, True, True, True]  # part 2 three times, part 3 never
    assert job.tr.brief.version == 1
    doc = json.loads((tmp_path / VID / "render" / "job.json").read_text())
    assert doc["brief"]["version"] == 1 and doc["brief"]["made"] == 1 and doc["brief"]["parts"] >= 3
    assert doc["brief"]["gaveUp"]["part"] == 2 and doc["brief"]["gaveUp"]["why"].startswith("bad_output")
    assert doc["stages"]["brief"]["state"] == "done" and doc["stages"]["translate"]["state"] == "done"


class HeldBriefClaude(MockClaude):
    """Claude held (not signed in) for the first brief call; records the job's count of Claude failures at each call."""

    job: RenderJob | None = None

    def __init__(self) -> None:
        super().__init__()
        self.fails: list[tuple[str, int]] = []

    def ask(self, system, prompt, schema=None, call="text", **kw):
        self.fails.append((call, self.job._claude_fails))
        if call == "brief" and len(self.fails) == 1:
            self.calls.append({"call": call, "message": json.loads(prompt), "failed": True})
            raise ClaudeCLIError("not_signed_in", "Sign in to Claude Code.")
        return super().ask(system, prompt, schema, call, **kw)


async def test_a_brief_part_that_goes_through_after_a_hold_clears_it(tmp_path, monkeypatch):
    """The part asked again after the hold went through Claude: the UI's banner and the back-off clear there, not at the
    first scene call after the whole brief."""
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.05)
    monkeypatch.setattr(render, "BRIEF_PART_WORDS", 200)
    cli = HeldBriefClaude()
    b = backend()
    b.translator = translator_with(cli)
    events: list[dict] = []
    job = cli.job = job_for(tmp_path, b, events=events)
    assert await job.run() == "done"
    briefs = [n for call, n in cli.fails if call == "brief"]
    assert len(briefs) >= 4 and briefs[:2] == [0, 1]  # part 1 held, then asked again
    assert set(briefs[2:]) == {0} and all(n == 0 for call, n in cli.fails if call != "brief")
    kinds = [e["type"] for e in events if e["type"] in ("claude_error", "claude_ok")]
    assert kinds == ["claude_error", "claude_ok"]


# ---- translate (§2.8) -----------------------------------------------------------------------------------------------
async def test_translation_waits_for_voices(tmp_path):
    """Every line is sized at its voice's calibrated pace (3 aksharas/s after 0.3 s), not at the prior's."""
    job = job_for(tmp_path, backend(tts=MelTTS(rate=3.0, overhead=0.3)))
    assert await job.run() == "done"
    scenes = [c["message"] for c in job.tr.cli.calls if c["call"] == "scene"]
    assert scenes
    for msg in scenes:
        for x in msg["lines"]:
            st = job.units[x["id"]]
            assert x["target_aksharas"] == pytest.approx(max(st.speech_s - 0.3, 0.0) * 3.0, abs=1.0)
            assert abs(x["target_aksharas"] - max(st.speech_s - 0.15, 0.0) * 6.1) > 1.0 or st.speech_s < 0.5


async def test_each_scene_but_a_lanes_first_has_the_telugu_before_it(tmp_path, monkeypatch):
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 20.0)
    job = job_for(tmp_path, backend())
    assert await job.run() == "done"
    n = len(job.scenes)
    firsts = {job.scenes[i * n // render.LANES].no for i in range(render.LANES)}
    # (The demo repeats its sentences, so some scenes are served whole from the line cache and make no call.)
    asked = {c["message"]["scene"]: c["message"] for c in job.tr.cli.calls if c["call"] == "scene"}
    assert n >= 6 and len(set(asked) - firsts) >= 3 and asked[1]["context_before"] == []
    for no, msg in asked.items():
        if no not in firsts:
            assert msg["context_before"] and all(x.get("te") for x in msg["context_before"])
    assert all(sc.reviewed and sc.fitted and not sc.held for sc in job.scenes)
    assert [st.unit.id for sc in job.scenes for st in sc.lines] == sorted(job.units)  # every line in one scene


async def test_a_second_run_makes_no_scene_calls(tmp_path):
    b = backend()
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    assert [c["call"] for c in first.tr.cli.calls].count("scene") >= 2
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert again.tr.cli.calls == []
    assert {k: (st.tier, st.telugu) for k, st in again.units.items()} == \
           {k: (st.tier, st.telugu) for k, st in first.units.items()}
    assert again.doc["coverage"] == first.doc["coverage"]


class HeldClaude(MockClaude):
    """Claude held (not signed in) once for each (call, scene) in `fail`; scene None for the brief."""

    def __init__(self, fail: set) -> None:
        super().__init__()
        self.fail = set(fail)

    def ask(self, system, prompt, schema=None, call="text", **kw):
        msg = json.loads(prompt)
        if (call, msg.get("scene")) in self.fail:
            self.fail.discard((call, msg.get("scene")))
            self.calls.append({"call": call, "message": msg, "failed": True})
            raise ClaudeCLIError("not_signed_in", "Sign in to Claude Code.")
        return super().ask(system, prompt, schema, call, **kw)


async def test_a_released_scene_and_a_failed_review_are_asked_again_after_the_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.05)
    cli = HeldClaude({("brief", None), ("scene", 1), ("review", 2)})
    b = backend()
    b.translator = translator_with(cli)
    events: list[dict] = []
    job = job_for(tmp_path, b, events=events)
    seen: list[tuple[int, bool, bool]] = []  # (scene, held, released) whenever a lane is about to ask
    before_call = job._before_call

    async def watched() -> None:
        seen.extend((sc.no, sc.held, sc.released) for sc in job.scenes)
        await before_call()

    job._before_call = watched
    reviewing: list[tuple[int, bool]] = []  # (scene, held) at each review
    review = job._review

    async def watched_review(req, lines):
        reviewing.append((req.scene, job.scenes[req.scene - 1].held))
        return await review(req, lines)

    job._review = watched_review
    assert await job.run() == "done"
    assert (1, True, True) not in seen and (1, False, True) in seen  # released: untranslated, never `held`
    assert (2, True, False) in seen  # its review failed: translated, unreviewed, voiceable during the hold
    # Asked again, its lines lost their wording: no longer `held` (voiceable) while the retry is under way.
    assert [r for r in reviewing if r[0] == 2] == [(2, False), (2, False)]
    # Each line's k is learned once, though scene 2's lines went through two scene requests.
    assert sum(len(v) for v in job._ratios.values()) == \
           sum(1 for st in job.units.values() if mixed_units(st.unit.text) >= dubber.K_MIN_SYLLABLES)
    asked = [(c["call"], c["message"].get("scene"), c.get("failed", False)) for c in cli.calls]
    assert asked.count(("brief", None, True)) == asked.count(("brief", None, False)) == 1
    assert asked.count(("scene", 1, True)) == asked.count(("scene", 1, False)) == 1   # released, then asked again
    assert asked.count(("review", 2, True)) == asked.count(("review", 2, False)) == 1  # failed, then reviewed
    assert asked.count(("scene", 2, False)) == 1  # the second time its lines came from the line cache
    assert len(job.scenes) == 2 and all(sc.reviewed and not sc.held for sc in job.scenes)
    assert all(st.line.coverage.cls == "C" for st in job.units.values())
    assert any(e["type"] == "claude_error" and e["kind"] == "not_signed_in" for e in events)
    statuses = [e["status"] for e in events if e["type"] == "render"]
    assert "waiting" in statuses and statuses[-1] == "done"
    assert job.doc["coverage"]["C"] == len(job.units) and job.doc["coverage"]["unreviewed"] == 0


class SkippingClaude(MockClaude):
    """Never answers for the line with id `skip`: the translator's ladder gives up on it."""

    def __init__(self, skip: int) -> None:
        super().__init__()
        self.skip = skip

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "scene":
            reply.data["lines"] = [x for x in reply.data["lines"] if x["id"] != self.skip]
        return reply


async def test_the_coverage_counts_land_in_job_json_and_a_skipped_line_stays_skipped(tmp_path):
    cli = SkippingClaude(skip=5)
    b = backend()
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    n = len(job.units)
    want = {"C": n - 1, "m": 0, "P": 0, "E": 0, "otherTier": 0, "unreviewed": 0, "skipped": 1, "lines": n}
    assert job.doc["coverage"] == want
    assert json.loads((tmp_path / VID / "render" / "job.json").read_text())["coverage"] == want
    assert list(job.skipped) == [5] and job.units[5].line is None and job.units[5].voiced
    asked = [c for c in cli.calls if c["call"] == "scene" and any(x["id"] == 5 for x in c["message"]["lines"])]
    assert len(asked) == 3  # the ladder (all, again, each), and never again
    assert all(sc.reviewed for sc in job.scenes)


async def test_a_line_skipped_once_is_not_asked_again_by_a_resume(tmp_path):
    cli = SkippingClaude(skip=5)
    b = backend(UniqueTranscriber())
    b.translator = translator_with(cli)
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    assert list(first.skipped) == [5]
    rows = _read_rows(tmp_path / VID / "render" / "skipped.jsonl")
    assert [(r["unit"], r["why"]) for r in rows] == [(5, first.skipped[5])]
    cli.calls.clear()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert cli.calls == []  # not the ladder's three calls again
    assert again.skipped == first.skipped and again.units[5].voiced and again.units[5].line is None
    assert again.doc["coverage"] == first.doc["coverage"] and again.doc["coverage"]["skipped"] == 1


class SkipThenPause(SkippingClaude):
    """Skips line `skip` while `skip` is set, and pauses `job` from inside the first review call."""

    job: RenderJob | None = None

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "review" and not any(c["call"] == "review" for c in self.calls):
            self.job.pause()
        return super().ask(system, prompt, schema, call, **kw)


async def test_a_run_after_a_pause_on_the_same_job_counts_each_line_once(tmp_path):
    """The run after a pause starts again from the transcript's units: what the paused run learned and skipped is not
    counted again. Line 5 is skipped in the first run while a line said the same way (the demo repeats its sentences)
    is answered, so the second run serves line 5 from the line cache."""
    cli = SkipThenPause(skip=5)
    b = backend()
    b.translator = translator_with(cli)
    job = cli.job = job_for(tmp_path, b)
    assert await job.run() == "paused"
    assert list(job.skipped) == [5] and job._ratios  # the premise
    cli.skip = None
    assert await job.run() == "done"
    cov = job.doc["coverage"]
    assert sum(cov[k] for k in ("C", "m", "P", "E", "otherTier", "unreviewed", "skipped")) == cov["lines"] == len(job.units)
    assert job.skipped == {} and job.units[5].line is not None and cov["skipped"] == 0
    assert sum(len(v) for v in job._ratios.values()) == \
           sum(1 for st in job.units.values() if mixed_units(st.unit.text) >= dubber.K_MIN_SYLLABLES)


async def test_a_preview_translates_only_the_scenes_before_its_stop_point(tmp_path, monkeypatch):
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 20.0)
    b = backend()
    job = job_for(tmp_path, b, RenderSettings(stop_at=60.0))
    assert await job.run() == "done"
    inside = [sc for sc in job.scenes if sc.lines[0].unit.start < 60.0 + render.PAST_STOP]
    rest = job.scenes[len(inside):]
    assert inside and rest and all(sc.reviewed for sc in inside) and not any(sc.reviewed for sc in rest)
    ids = {st.unit.id for sc in inside for st in sc.lines}
    asked = {x["id"] for c in job.tr.cli.calls if c["call"] == "scene" for x in c["message"]["lines"]}
    assert asked <= ids and all(st.line is not None for st in job.units.values() if st.unit.id in ids)
    assert all(st.line is None and st.scene is None for sc in rest for st in sc.lines)
    stage = job.doc["stages"]["translate"]
    assert stage["done"] == stage["total"] == len(ids)
    # The lines voiced (the coverage report's, of the wordings voiced) start before the stop point plus PAST_STOP.
    voiced = {st.unit.id for st in job.units.values() if st.unit.start < 60.0 + render.PAST_STOP}
    assert voiced < ids and job.doc["coverage"]["lines"] == len(voiced)
    assert {st.unit.id for st in job.units.values() if st.take is not None} == voiced
    # Everything before translation covers the whole video: the brief read the whole transcript.
    assert sum(len(c["message"]["transcript"]) for c in job.tr.cli.calls if c["call"] == "brief") == len(job.units)
    # Continuing to the whole video asks only for the scenes the preview left.
    whole = job_for(tmp_path, b, RenderSettings())
    assert await whole.run() == "done"
    asked = {x["id"] for c in whole.tr.cli.calls if c["call"] == "scene" for x in c["message"]["lines"]}
    assert asked and not asked & ids and all(sc.reviewed for sc in whole.scenes)
    assert whole.doc["coverage"]["lines"] == len(whole.units)


async def test_continuing_a_preview_serves_its_scenes_first_and_the_rest_in_three_lanes(tmp_path, monkeypatch):
    """§3: the scenes the preview translated come from the line cache first, in order and with no call; the scenes left
    are split into three lanes, and the first of them has the preview's Telugu before it."""
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 20.0)
    b = backend(UniqueTranscriber())
    assert await job_for(tmp_path, b, RenderSettings(stop_at=60.0)).run() == "done"
    job = job_for(tmp_path, b, RenderSettings())
    lanes: list[tuple[list[int], bool]] = []
    lane = job._lane

    async def watched(scenes, total, whole=False):
        lanes.append(([sc.no for sc in scenes], whole))
        await lane(scenes, total, whole)

    job._lane = watched
    assert await job.run() == "done"
    served = [no for nos, w in lanes if w for no in nos]
    split = [nos for nos, w in lanes if not w]
    preview = [sc.no for sc in job.scenes if sc.lines[0].unit.start < 60.0 + render.PAST_STOP]
    assert served == preview and all(len(nos) == 1 for nos, w in lanes if w)
    assert len(split) == 3 and [no for nos in split for no in nos] == list(range(len(preview) + 1, len(job.scenes) + 1))
    asked = {c["message"]["scene"]: c["message"] for c in job.tr.cli.calls if c["call"] == "scene"}
    assert set(asked) == {no for nos in split for no in nos}  # the preview's scenes made no call
    first = asked[split[0][0]]
    assert first["context_before"] and all(x.get("te") for x in first["context_before"])
    assert all(sc.reviewed for sc in job.scenes)


class PausingClaude(MockClaude):
    """Pauses `job` from inside the first scene call, as the UI's pause arrives while Claude answers."""

    job: RenderJob | None = None

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "scene" and not any(c["call"] == "scene" for c in self.calls):
            self.job.pause()
        return super().ask(system, prompt, schema, call, **kw)


async def test_a_pause_lets_the_scene_calls_in_flight_finish_and_starts_no_new_call(tmp_path):
    cli = PausingClaude()
    b = backend()
    b.translator = translator_with(cli)
    job = cli.job = job_for(tmp_path, b)
    assert await job.run() == "paused"
    assert job.doc["stage"] == "translate" and job.doc["stages"]["translate"]["state"] == "todo"
    assert job.scenes and not any(sc.reviewed or sc.held for sc in job.scenes)
    kinds = [c["call"] for c in cli.calls]
    assert "scene" in kinds and "review" not in kinds and "fit" not in kinds
    first = {x["id"] for c in cli.calls if c["call"] == "scene" for x in c["message"]["lines"]}
    b.translator = MockSceneTranslator
    again = job_for(tmp_path, b)
    whole: list[int] = []  # the scenes served whole from the line cache, ahead of the lanes
    lane = again._lane

    async def watched(scenes, total, whole_served=False):
        whole.extend(sc.no for sc in scenes if whole_served)
        await lane(scenes, total, whole_served)

    again._lane = watched
    assert await again.run() == "done"
    asked = {x["id"] for c in again.tr.cli.calls if c["call"] == "scene" for x in c["message"]["lines"]}
    assert not asked & first  # what the paused run's calls answered is in the line cache
    reviewed = {c["message"]["scene"] for c in again.tr.cli.calls if c["call"] == "review"}
    assert reviewed and all(sc.reviewed for sc in again.scenes)
    assert not reviewed & set(whole)  # cached but unreviewed lines are no scene served whole: they go to the lanes


class BrokenTTS(MockTTS):
    def prepare_voice(self, reference, sample_rate):
        raise RuntimeError("the voice model is broken")


async def test_an_error_in_the_voices_fails_the_job_and_stops_the_brief(tmp_path):
    b = backend(tts=BrokenTTS())
    b.translator = translator_with(MockClaude(delay=5.0))
    job = job_for(tmp_path, b)
    t0 = time.perf_counter()
    assert await job.run() == "failed"
    assert time.perf_counter() - t0 < 3.0  # the brief's call was stopped, not waited out
    doc = json.loads((tmp_path / VID / "render" / "job.json").read_text())
    assert doc["stage"] == "voices" and doc["stages"]["voices"]["state"] == "failed"
    assert doc["stages"]["brief"]["state"] == "todo" and doc["stages"]["brief"]["attempts"] == 0


class Pausing(MockDiarizer):
    """The mock's 14 s turns, each said in 2 s stretches with 0.5 s pauses between: a voice's clean spans, which run
    across pauses that short, hold 20 % silence."""

    def diarize(self, audio, **kwargs):
        block = super().diarize(audio, **kwargs)
        turns = [SpeakerTurn(t.speaker, a, min(a + 2.0, t.end)) for t in block.turns
                 for a in np.arange(t.start, t.end, 2.5).tolist()]
        return DiarBlock(block.start, block.end, turns, list(turns), block.centroids, step=block.step)


async def test_a_voice_whose_spans_hold_pauses_is_kept_on_resume(tmp_path):
    b = backend(tts=MelTTS())
    b.diarizer = Pausing()
    job = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await job.run() == "done"
    e = json.loads((tmp_path / VID / "render" / "voices.json").read_text())["speakers"]["S1"]
    spans = render._union([tuple(e["timbre"])] + [tuple(c) for c in e["clips"]])
    assert e["speech"] < 0.9 * sum(b - a for a, b in spans)  # the spans are over 10 % pauses
    b.tts = MelTTS()
    assert await job_for(tmp_path, b, RenderSettings(speakers=2)).run() == "done"
    assert b.tts.takes == 0


class TerseClaude(MockClaude):
    """Answers a scene call with `full` alone, whatever tiers it asked for, so lines that run long get a fit; each fit
    takes a moment."""

    FIT_DELAY = 0.2

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "fit":
            time.sleep(self.FIT_DELAY)
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "scene":
            for x in reply.data["lines"]:
                for tier in ("fuller", "concise", "very_concise"):
                    x.pop(tier, None)
        return reply


async def test_each_scene_keeps_its_fit_and_the_stage_waits_for_it(tmp_path):
    cli = TerseClaude()
    b = backend(tts=MelTTS(rate=2.0))  # a slow voice: most lines run long in `full`
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    fitted = {c["message"]["scene"] for c in cli.calls if c["call"] == "fit"}
    assert fitted and all(sc.fit is not None for sc in job.scenes if sc.no in fitted)
    assert all(sc.fitted for sc in job.scenes)  # translate ends once every fit is back...
    assert any("concise" in st.line.tiers for st in job.units.values())  # ...and folded into its lines


class LateFitClaude(TerseClaude):
    """Terse; Claude is held for scene 2's first review, and the fit scene 2 then asks for is slow to come back."""

    FIT_DELAY = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.late: float | None = None  # when the slow fit came back (monotonic s)

    def ask(self, system, prompt, schema=None, call="text", **kw):
        msg = json.loads(prompt)
        if call == "review" and msg["scene"] == 2 and not any(c["call"] == "review" and c["message"]["scene"] == 2
                                                               for c in self.calls):
            self.calls.append({"call": call, "message": msg, "failed": True})
            raise ClaudeCLIError("not_signed_in", "Sign in to Claude Code.")
        first_fit = call == "fit" and msg["scene"] == 2 and not any(c["call"] == "fit" and c["message"]["scene"] == 2
                                                                    for c in self.calls)
        reply = super().ask(system, prompt, schema, call, **kw)
        if first_fit:
            time.sleep(0.6)
            self.late = time.monotonic()
        return reply


async def test_a_late_fit_never_undoes_the_review_of_a_scene_asked_again(tmp_path, monkeypatch):
    """Scene 2's review fails while Claude is held, after the scene asked for a fit; the lane asks for the scene again
    only once that fit is back, so the fit can't fold its unreviewed line into the lines the second review classed."""
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.05)
    cli = LateFitClaude()
    b = backend(UniqueTranscriber(), MelTTS(rate=2.0))
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    ended = time.monotonic()
    for _ in range(100):  # a fit nobody waited for would come back after the run
        if cli.late is not None:
            break
        await asyncio.sleep(0.02)
    reviews = [c for c in cli.calls if c["call"] == "review" and c["message"]["scene"] == 2]
    # (Its voiced wordings are reviewed after those two: a shorter tier or a rephrase voiced has no class.)
    assert [c.get("failed", False) for c in reviews][:2] == [True, False] and cli.late is not None  # the premise
    assert all(st.line.coverage is not None and st.line.coverage.cls == "C" for st in job.units.values())
    assert cli.late < ended


class PauseInFitClaude(TerseClaude):
    """Terse; Claude is held for scene 2's first review, and the fit scene 2 then asks for comes back once the hold is
    over, with the job paused while it was out."""

    FIT_DELAY = 0.0
    job: RenderJob | None = None

    def ask(self, system, prompt, schema=None, call="text", **kw):
        msg = json.loads(prompt)
        if call == "review" and msg["scene"] == 2 and not any(c["call"] == "review" and c["message"]["scene"] == 2
                                                               for c in self.calls):
            self.calls.append({"call": call, "message": msg, "failed": True})
            raise ClaudeCLIError("not_signed_in", "Sign in to Claude Code.")
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "fit" and msg["scene"] == 2:
            time.sleep(0.3)
            self.job.pause()
        return reply


async def test_a_pause_while_a_lane_waits_for_a_fit_starts_no_new_request(tmp_path, monkeypatch):
    """A lane asks for a scene again only once its earlier fit is back; a pause that came while it waited stops the lane
    there, before the request (whose cached lines could need a fit call of their own)."""
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 0.05)
    cli = PauseInFitClaude()
    b = backend(UniqueTranscriber(), MelTTS(rate=2.0))
    b.translator = translator_with(cli)
    job = cli.job = job_for(tmp_path, b)
    entered: list[tuple[int, bool]] = []  # (scene, paused) as each scene request starts
    scene = job._scene

    async def watched(req):
        entered.append((req.scene, job._stop.is_set()))
        await scene(req)

    job._scene = watched
    assert await job.run() == "paused"
    assert any(c["call"] == "fit" and c["message"]["scene"] == 2 for c in cli.calls)  # the premise
    assert (2, False) in entered and not any(paused for _, paused in entered)
    assert job.scenes[1].held and not job.scenes[1].reviewed  # translated, unreviewed: asked again on resume


# ---- the dub loop (§2.9, §2.10) ---------------------------------------------------------------------------------------
_TE_SYLLABLES = [c + v for c in "కగచజటడతదనపబమ" for v in ("ా", "ి", "ు", "ే")]


def unique_telugu(monkeypatch) -> None:
    """Each word of UniqueTranscriber's English its own Telugu-script stand-in (the mock has a dozen, shared by many
    words), with the words that tell its sentences apart first, so no two of its lines are said the same way in any
    tier (the mock's shorter tiers drop trailing words; "the gentle bridge" would end two sentences' shortest): no take
    file and no PCM file is shared between two lines. Every stand-in is two aksharas, so the order changes no length."""
    common = ("the", "was", "today")
    vocab = [*common, *UniqueTranscriber.ADJ, *UniqueTranscriber.NOUN, *UniqueTranscriber.VERB]
    te = {w: _TE_SYLLABLES[i] + "ల" for i, w in enumerate(vocab)}
    fake = mock._fake_spoken

    def spoken(en: str) -> tuple[list[str], list[tuple[int, str]]]:
        words = sorted((w.lower().strip(".,?!") for w in en.split()), key=lambda w: w in common)
        return ([te[w] for w in words], []) if words and all(w in te for w in words) else fake(en)

    monkeypatch.setattr(mock, "_fake_spoken", spoken)


def take_rows(tmp_path) -> list[dict]:
    return _read_rows(tmp_path / VID / "render" / "takes.jsonl")


def texts_of(tts: MelTTS) -> set[str]:
    return {text for text, _ in tts.batches}


async def test_the_dub_loop_voices_every_line_on_the_working_plan_and_keeps_its_take(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    tts = MelTTS()
    b = backend(UniqueTranscriber(), tts)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    lines = [st for sc in job.scenes for st in sc.lines]
    assert lines and all(st.voiced and st.take is not None and st.plan is not None for st in lines)
    assert set(job.planner._placed) == {st.unit.id for st in lines}  # every line voiced is placed on W
    rows = take_rows(tmp_path)
    first = list(dict.fromkeys(r["unit"] for r in rows))
    assert first == sorted(first) == [st.unit.id for st in lines]  # voiced in onset order
    for st in lines:
        row = st.take
        key = job._key(st.unit.speaker)
        assert row["line"] == job._line_key(st) and row["voice"] == [key.voice, 0.5, 0.7, key.reference]
        assert key.reference.startswith("v1:") and row["wording"] in (st.tier, f"piece:{len(row['takes'])}")
        words = job._piece_words(st.line) if len(row["takes"]) > 1 else [st.line.tiers[st.tier]]
        assert [t["spoken"] for t in row["takes"]] == [w.spoken for w in words]
        n = 3 if st.speech_s < 3.0 else 2
        for w, t in zip(words, row["takes"]):
            assert t["key"] == job._take_key(w.spoken, row["voice"], n)  # the TTS reads Telugu script
            assert (w.spoken, n) in tts.batches  # TAKES_N in one batch, 3 under 3 s of speech
            assert (tmp_path / VID / "render" / "takes" / f"{t['key']}.npz").is_file()
        assert sum(t["seconds"] for t in row["takes"]) == pytest.approx(st.take_s)
    assert {n for _, n in tts.batches} == {2, 3}
    assert all(rate == 1.0 for _, rate in tts.vocoded)  # only natural audio so far: PCM at the planned rate is step 5
    stage = job.doc["stages"]["voice_lines"]
    total = round(sum(st.speech_s for st in lines), 1)
    assert (stage["state"], stage["done"], stage["total"], stage["unit"]) == ("done", total, total, "speech s")
    cov = job.doc["coverage"]
    assert cov["C"] == cov["lines"] == len(lines) and job.flags == {}


class CancelOnBatch(MelTTS):
    """Counts its batches; `started` is set once batch number `at` starts."""

    def __init__(self, at: int, **kw) -> None:
        super().__init__(**kw)
        self.at, self.started = at, threading.Event()

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1):
        if len(self.batches) + 1 == self.at:
            self.started.set()
            time.sleep(0.05)
        return super().synthesize_takes(text, voice, language, max_seconds, n)


async def test_cancel_during_the_dub_then_resume_voices_only_the_lines_without_rows(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    tts = CancelOnBatch(at=12)
    b = backend(UniqueTranscriber(), tts)
    job = job_for(tmp_path, b)
    task = asyncio.create_task(job.run())
    while not tts.started.is_set():
        await asyncio.sleep(0.002)
    task.cancel()  # the engine stopping: job.json stays `running`, an interrupted render
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads((tmp_path / VID / "render" / "job.json").read_text())["status"] == "running"
    had = {r["line"] for r in take_rows(tmp_path)}
    kept = {st.unit.id: st.tier for st in job.units.values() if st.take is not None}
    assert 0 < len(kept) < len(job.units)
    heard, diarized = b.transcriber.calls, b.diarizer.calls
    b.tts = tts2 = MelTTS()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert (b.transcriber.calls, b.diarizer.calls) == (heard, diarized)  # no stage before the dub ran again
    left = {st.unit.id for st in again.units.values() if again._line_key(st) not in had}
    assert again._voiced_now == left and len(left) == len(job.units) - len(kept)  # only these were voiced...
    assert texts_of(tts2) <= {w.spoken for k in left for w in again.units[k].line.tiers.values()}  # ...and synthesized
    assert {k: again.units[k].tier for k in kept} == kept  # the others restored as voiced, with no `_choose`
    assert all(st.take is not None for st in again.units.values())


class PausingMockTTS(MockTTS):
    """MockTTS (no mel takes: its voices aren't calibrated, their pace is learned from their lines), pausing `job` from
    inside synthesis number `at`."""

    job: RenderJob | None = None

    def __init__(self, at: int | None) -> None:
        self.at, self.calls = at, 0

    def synthesize(self, text, voice, language="te", max_seconds=None):
        self.calls += 1
        if self.calls == self.at:
            self.job.pause()
        return super().synthesize(text, voice, language, max_seconds)


async def test_a_run_after_a_pause_counts_each_take_once_in_the_estimator(tmp_path, monkeypatch):
    """A voice's pace is its calibration (none, for the mock) and then its takes, each counted once: a run after a
    pause on the same job replays the takes voiced before it, and a new job that finds every take in its row ends where
    it did."""
    unique_telugu(monkeypatch)
    tts = PausingMockTTS(at=14)
    b = backend(UniqueTranscriber(), tts)
    job = tts.job = job_for(tmp_path, b)
    assert await job.run() == "paused"
    assert 0 < len(take_rows(tmp_path)) < len(job.units)
    tts.at = None
    assert await job.run() == "done" and job._resynths == 0  # (every take made was kept: each has a row)
    b.tts = PausingMockTTS(at=None)
    fresh = job_for(tmp_path, b)
    assert await fresh.run() == "done" and b.tts.calls == 0
    for sid in ("S1", "S2"):
        a, z = job.estimator.voices[job._key(sid)], fresh.estimator.voices[fresh._key(sid)]
        assert a.n == z.n == sum(1 for r in take_rows(tmp_path) if r["voice"][0] == sid for _ in r["takes"])
        assert (a.inv_rate, a.overhead) == pytest.approx((z.inv_rate, z.overhead))


class PauseOnBatch(MelTTS):
    """Pauses `job` from inside batch number `at`."""

    job: RenderJob | None = None

    def __init__(self, at: int, **kw) -> None:
        super().__init__(**kw)
        self.at = at

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1):
        if len(self.batches) + 1 == self.at:
            self.job.pause()
        return super().synthesize_takes(text, voice, language, max_seconds, n)


async def test_a_pause_in_the_dub_loop_stops_it_and_a_run_voices_what_is_left(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    tts = PauseOnBatch(at=10)
    b = backend(UniqueTranscriber(), tts)
    job = tts.job = job_for(tmp_path, b)
    assert await job.run() == "paused"
    stage = job.doc["stages"]["voice_lines"]
    assert stage["state"] == "todo" and stage["attempts"] == 0 and 0 < stage["done"] < stage["total"]
    first = {r["line"] for r in take_rows(tmp_path)}
    assert 0 < len(first) < len(job.units) and len(tts.batches) == 10
    calls = len(job.tr.cli.calls)
    tts.at = None
    assert await job.run() == "done"  # the same job, as a resume after a pause
    synthesized = {text for text, _ in tts.batches[10:]}
    before = {w.spoken for st in job.units.values() if job._line_key(st) in first for w in st.line.tiers.values()}
    assert synthesized and not synthesized & before  # no line with a row was voiced again
    assert job._voiced_now == {st.unit.id for st in job.units.values() if job._line_key(st) not in first}
    assert all(st.take is not None for st in job.units.values())
    assert not any(c["call"] == "scene" for c in job.tr.cli.calls[calls:])  # its scenes came from the line cache
    assert len({r["line"] for r in take_rows(tmp_path)}) == len(job.units)
    # The estimator counts each take once: the calibration, then the takes' paces replayed from their rows or learned
    # as they were voiced, the same as a new job's that finds them all in their rows.
    assert job._resynths == 0  # (the premise: every take made was kept, so every take has a row)
    b.tts = MelTTS()
    fresh = job_for(tmp_path, b)
    assert await fresh.run() == "done" and b.tts.batches == []
    for sid in ("S1", "S2"):
        a, z = job.estimator.voices[job._key(sid)], fresh.estimator.voices[fresh._key(sid)]
        assert (a.n, a.inv_rate, a.overhead) == pytest.approx((z.n, z.inv_rate, z.overhead))


async def test_take_files_load_back_into_an_equal_said_and_the_mel_round_trips(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    for tts in (MelTTS(spread=0.05), MockTTS()):
        d = tmp_path / type(tts).__name__
        job = job_for(d, backend(UniqueTranscriber(), tts))
        saids, keep = {}, job._keep

        def kept(st, said, *args, **kw):  # (the loop lets go of a line's takes once its PCM is written)
            saids[st.unit.id] = said
            return keep(st, said, *args, **kw)

        job._keep = kept
        assert await job.run() == "done"
        assert saids and job._said == {}
        for uid, said in saids.items():
            got = await job._load_said(job.units[uid].take)
            assert got.seconds == said.seconds and got.pauses == said.pauses and got.paces == said.paces
            assert [w.spoken for w in got.wordings] == [w.spoken for w in said.wordings]
            for a, x in zip(got.takes, said.takes):
                if isinstance(tts, MelTTS):  # the mel, kept as float16
                    assert a.seconds == x.seconds and a.mel.dtype == np.float32
                    np.testing.assert_allclose(a.mel, x.mel, atol=2e-3)
                else:  # MockTTS: its samples, exactly
                    np.testing.assert_array_equal(a, x)
            assert all(audio is None for audio in got.audio) if isinstance(tts, MelTTS) else \
                all(np.array_equal(a, x) for a, x in zip(got.audio, said.audio))


class RephraseClaude(MockClaude):
    """The mock, whose rephrase of a line is the first two words of the wording that ran long, with no shorter tier: a
    wording that fits where the others ran long."""

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "rephrase":
            for x, asked in zip(reply.data["lines"], json.loads(prompt)["lines"]):
                x["full"] = {"spoken": " ".join(asked["failing"].split()[:2]), "english": []}
                for tier in ("concise", "very_concise", "pieces"):
                    x.pop(tier, None)
        return reply


class TerseRephraseClaude(TerseClaude, RephraseClaude):
    """Terse scene answers (a line's shorter tiers come from its fit), rephrases as RephraseClaude's."""

    FIT_DELAY = 0.0


class ShortenableClaude(RephraseClaude):
    """RephraseClaude, whose scene answers carry `concise` and `very_concise` whether asked for or not."""

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "scene":
            for x in reply.data["lines"]:
                words = x["full"]["spoken"].split()
                for tier, k in (("concise", math.ceil(0.75 * len(words))), ("very_concise", math.ceil(0.5 * len(words)))):
                    if tier not in x and 0 < k < len(words):
                        x[tier] = {"spoken": " ".join(words[:k]), "english": []}
        return reply


def rephrase_job(tmp_path, cli: MockClaude, monkeypatch):
    """A voice far slower than its estimate (1 akshara/s: its calibration takes all run to their cap), so lines run
    long: each is voiced in the `full` its review classed, then in its shortest tier, then (still long) in its scene's
    rephrase. No fix-up budget, so every line can have them."""
    monkeypatch.setattr(render, "FIXUP_SHARE", 10.0)
    unique_telugu(monkeypatch)
    tts = MelTTS(rate=1.0)
    b = backend(UniqueTranscriber(), tts)
    b.translator = translator_with(cli)
    return job_for(tmp_path, b), b, tts


async def test_fixups_shorter_tier_then_one_rephrase_per_scene(tmp_path, monkeypatch):
    cli = ShortenableClaude()
    job, b, tts = rephrase_job(tmp_path, cli, monkeypatch)
    voiced: dict[int, list[str]] = {}  # unit id -> each wording `_say` was asked for, in order
    say = job._say

    async def watched(st, words, *args, **kw):
        voiced.setdefault(st.unit.id, []).append(" ".join(w.spoken for w in words))
        return await say(st, words, *args, **kw)

    job._say = watched
    assert await job.run() == "done"
    rows = take_rows(tmp_path)
    rephrased = [st for st in job.units.values() if st.take["wording"] == "rephrase"]
    assert len(rephrased) >= 5
    three = 0
    for st in rephrased:
        cached = job.tr.cached(job._line_spec(st, ("full",)))
        size = {w.spoken: len(w.spoken.split()) for w in cached.tiers.values()}
        *tiers, rephrase = voiced[st.unit.id]  # a tier of the line, a shorter one if it has one, then the rephrase
        assert rephrase == st.line.full.spoken and rephrase not in size
        if any(n < size[tiers[0]] for n in size.values()):
            assert len(tiers) == 2 and size[tiers[1]] < size[tiers[0]]
            three += 1
        else:
            assert len(tiers) == 1
        mine = [r for r in rows if r["line"] == job._line_key(st)]
        assert [r["wording"] for r in mine][1:] == ["rephrase"]  # first: the shorter tier, or the take it didn't beat
        assert st.tier == "full" and len(st.line.full.spoken.split()) == 2 and st.take_s < mine[0]["takes"][0]["seconds"]
        assert ("long" in job.flags.get(st.unit.id, [])) == st.plan.needs_shorter  # flagged while it still runs long
        # lines.jsonl still serves the reviewed line: the rephrase is in fixups.jsonl only
        assert cached.coverage is not None and cached.full.spoken != st.line.full.spoken
    assert three >= 3
    assert any(st.plan.needs_shorter for st in rephrased)  # kept (shorter), still long: flagged
    asked = [c["message"]["scene"] for c in cli.calls if c["call"] == "rephrase"]
    assert sorted(asked) == sorted(set(asked)) == sorted({job.units[st.unit.id].scene for st in rephrased})
    fixups = _read_rows(tmp_path / VID / "render" / "fixups.jsonl")
    assert {f["line"] for f in fixups if f["kind"] == "rephrase"} >= {job._line_key(st) for st in rephrased}
    lines_cache = (tmp_path / VID / "lines.jsonl").read_text(encoding="utf-8")
    assert not any(json.dumps(st.line.full.spoken, ensure_ascii=False) in lines_cache for st in rephrased)
    # A second run asks Claude for nothing and synthesizes nothing: every line comes back as it was voiced.
    b.tts = tts2 = MelTTS(rate=1.0)
    cli.calls.clear()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert cli.calls == [] and tts2.batches == [] and len(take_rows(tmp_path)) == len(rows)
    assert {k: (st.tier, st.telugu, st.take["tts"]) for k, st in again.units.items()} == \
           {k: (st.tier, st.telugu, st.take["tts"]) for k, st in job.units.items()}


async def test_fixups_stay_within_their_share_of_the_lines(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS(rate=1.5))
    b.translator = translator_with(TerseRephraseClaude())
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    voiced = sum(1 for st in job.units.values() if st.take is not None)
    assert 3 <= job._resynths <= max(3, render.FIXUP_SHARE * voiced) + 1
    long = {k for k, why in job.flags.items() if "long" in why}  # the lines left long are flagged...
    assert long and all(job.units[k].plan.needs_shorter and job.units[k].take["wording"] != "rephrase" for k in long)
    b.tts = MelTTS(rate=1.5)
    again = job_for(tmp_path, b)
    assert await again.run() == "done" and b.tts.batches == []
    assert again.flags == job.flags  # ...and flagged again when they come back from their rows


class SlowPauseOnBatch(PauseOnBatch):
    def __init__(self, at: int) -> None:
        super().__init__(at, rate=1.0)


async def test_a_run_with_fix_ups_after_a_pause_learns_what_a_new_job_finds_in_its_rows(tmp_path, monkeypatch):
    """Each take made teaches the estimator once, and its row keeps what it taught: the first take a shorter tier
    replaced, a fix-up's take kept or not, and a take made just before a pause whose line's row was never written (it
    teaches when its line is voiced again, from its file). A job paused in the dub loop and run again ends where a new
    job that finds every row does."""
    monkeypatch.setattr(render, "FIXUP_SHARE", 10.0)
    unique_telugu(monkeypatch)
    tts = SlowPauseOnBatch(at=6)
    b = backend(UniqueTranscriber(), tts)
    b.translator = translator_with(ShortenableClaude())
    job = tts.job = job_for(tmp_path, b)
    assert await job.run() == "paused"
    takes = tmp_path / VID / "render" / "takes"
    named = {m[0] for r in take_rows(tmp_path) for m in r["made"]}
    window = {p.stem: job._read_take(p.stem)["pace"] for p in takes.glob("*.npz") if p.stem not in named}
    assert any(np.isfinite(p) for p in window.values())  # the premise: a take made and learned from, with no row yet
    tts.at = None
    assert await job.run() == "done"
    rows = take_rows(tmp_path)
    files = {p.stem for p in takes.glob("*.npz")}
    taught = {m[0]: m[2] for r in rows for m in r["made"]}
    assert all(taught[k] == (float(p) if np.isfinite(p) else None) for k, p in window.items())  # taught from its file
    assert any(len(r["made"]) > len(r["takes"]) for r in rows)  # a first take its shorter tier replaced
    assert any(r["wording"] == "rephrase" for r in rows)  # a rephrase's take
    made = [m[0] for r in rows for m in r["made"]]
    assert len(made) == len(set(made)) and set(made) >= files  # each take taught once, and every take is in a row...
    # ...and the take files left are those of the rows the lines use: the others were collected (§2.13)
    assert files == {t["key"] for st in job.units.values() for t in st.take["takes"]}
    b.tts = MelTTS(rate=1.0)
    fresh = job_for(tmp_path, b)
    assert await fresh.run() == "done" and b.tts.batches == []
    assert fresh._resynths == job._resynths > 0
    for sid in ("S1", "S2"):
        a, z = job.estimator.voices[job._key(sid)], fresh.estimator.voices[fresh._key(sid)]
        assert a.n == z.n == sum(1 for r in rows if r["voice"][0] == sid for m in r["made"] if m[2] is not None)
        assert (a.inv_rate, a.overhead) == pytest.approx((z.inv_rate, z.overhead))


async def test_a_take_the_disk_cant_keep_fails_the_job_and_a_resume_goes_on(tmp_path, monkeypatch):
    """A full disk while the dub loop keeps a take stops the render saying so (never every line after it skipped as a
    failed synthesis); once there is room, a resume goes on from the rows on disk."""
    unique_telugu(monkeypatch)
    save, saved = render._save_npz, []

    def full(path, arrays):
        if len(saved) == 6:
            raise OSError(errno.ENOSPC, "No space left on device")
        save(path, arrays)
        saved.append(path)

    monkeypatch.setattr(render, "_save_npz", full)
    b = backend(UniqueTranscriber(), MelTTS())
    job = job_for(tmp_path, b)
    assert await job.run() == "failed"
    assert "disk space" in job.doc["error"] and "No space left on device" in job.doc["error"]
    assert job.doc["stages"]["voice_lines"]["state"] == "failed" and job.skipped == {}
    kept = {r["line"] for r in take_rows(tmp_path)}
    assert 0 < len(kept) < len(job.units)
    monkeypatch.setattr(render, "_save_npz", save)
    b.tts = tts = MelTTS()
    assert await job.run() == "done"
    assert job.skipped == {} and all(st.take is not None for st in job.units.values())
    assert job._voiced_now == {k for k, st in job.units.items() if job._line_key(st) not in kept}
    assert texts_of(tts) <= {w.spoken for k in job._voiced_now for w in job.units[k].line.tiers.values()}


async def test_a_fix_up_take_is_kept_only_when_it_passes_and_a_rephrase_only_when_shorter(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    tts = MelTTS(fail="ఫెయిల్")
    job = job_for(tmp_path, backend(UniqueTranscriber(), tts))
    assert await job.run() == "done"
    st = max(job.units.values(), key=lambda x: x.take_s)
    rows = len(take_rows(tmp_path))
    words = st.line.tiers[st.tier].spoken.split()

    async def fix(kind: str, spoken: str) -> bool:
        line = LineResult(st.unit.id, {"full": Wording(spoken)})
        f = render._Fix(st, line, "full", kind, job._fixup_key(kind, job._line_key(st), spoken),
                        asyncio.get_running_loop().create_future())
        await job._voice_fix(f)
        return f.done.result()

    take, spent = st.take, job._resynths
    assert not await fix("rephrase", " ".join(words + ["కల"]))  # longer: not kept
    assert not await fix("rephrase", "ఫెయిల్ కల")  # shorter, but its takes ran to their cap: not kept
    # The line keeps its take, and its row is restated with what each made (a take new to the render) cost and taught.
    restated = take_rows(tmp_path)[rows:]
    assert len(restated) == 2 and st.take == restated[-1] and job._resynths == spent + 2
    assert all({k: v for k, v in r.items() if k not in ("fixups", "made", "cost")} ==
               {k: v for k, v in take.items() if k not in ("fixups", "made", "cost")} for r in restated)
    assert [r["fixups"] for r in restated] == [take["fixups"] + 1, take["fixups"] + 2]
    assert [[m[1] for m in r["made"]] for r in restated] == [[" ".join(words + ["కల"])], ["ఫెయిల్ కల"]]
    assert await fix("rephrase", " ".join(words[:2]))  # shorter, and it passes: kept, in place of the take
    assert st.take["wording"] == "rephrase" and st.take["tier"] == "full" and st.tier == "full"
    assert st.take["fixup"] == job._fixup_key("rephrase", job._line_key(st), " ".join(words[:2])) != \
        job._fixup_key("rephrase", job._line_key(st), " ".join(words))  # keyed by the wording shortened
    assert st.plan.start == take_plan_start(job, st) and st.take_s < take["takes"][0]["seconds"] * 2
    assert st.take["fixups"] == take["fixups"] + 3
    assert await fix("retranslate", " ".join(words * 2))  # the review's re-translation: kept, whatever its length
    assert st.take["wording"] == "retranslate" and len(take_rows(tmp_path)) == rows + 4
    assert st.take["fixups"] == take["fixups"] + 3  # a re-translation isn't a fix-up the budget counts
    assert not await fix("rephrase", "ఫెయిల్ కల")  # its take from its file, made already: free, and no row
    assert len(take_rows(tmp_path)) == rows + 4 and job._resynths == spent + 3


def take_plan_start(job: RenderJob, st) -> float:
    return job.planner._placed[st.unit.id][1].start  # where W placed the line's first take: a fix-up keeps its start


class ClassesRephrasesP(ShortenableClaude):
    """Reviews every rephrase (a two-word wording) as P: its re-translation (the line with a word more) replaces it."""

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "retranslate":
            for x in reply.data["lines"]:
                x["full"]["spoken"] += " మళ్ళీ"
        if call == "review":
            said = {x["id"]: x["te"] for x in json.loads(prompt)["lines"]}
            for x in reply.data["lines"]:
                if len(said[x["id"]].split()) == 2:
                    x.update({"class": "P", "missing": ["today"]})
        return reply


async def test_every_voiced_wording_is_reviewed(tmp_path, monkeypatch):
    cli = ClassesRephrasesP()
    job, b, tts = rephrase_job(tmp_path, cli, monkeypatch)
    assert await job.run() == "done"
    voiced = [st for st in job.units.values() if st.take is not None]
    assert all(dubber.voiced_class(st.line.coverage, st.tier) in dubber.CLASSES for st in voiced)
    assert job.doc["coverage"]["otherTier"] == job.doc["coverage"]["unreviewed"] == 0 and job.flags.keys() <= \
        {st.unit.id for st in voiced if "long" in job.flags.get(st.unit.id, [])}
    reviewed = [(x["id"], x["te"]) for c in cli.calls if c["call"] == "review" for x in c["message"]["lines"]]
    rows = take_rows(tmp_path)
    fixed = [r for r in rows if r["wording"] in render.FIXUP_WORDINGS]
    for r in fixed:  # each fix-up wording was reviewed on the wording voiced, but a re-translation never again
        te = r["takes"][0]["spoken"]
        assert ((r["unit"], te) in reviewed) == (r["wording"] == "rephrase")
    retranslated = [st for st in voiced if st.take["wording"] == "retranslate"]
    assert retranslated and all(st.line.coverage.by == "validators" for st in retranslated)
    assert all(st.line.coverage.cls == "C" and st.line.coverage.first == "P" for st in retranslated)
    shorter = [st.take for st in voiced if st.take["wording"] == "very_concise"]  # a tier the scene review didn't class
    assert shorter and all((r["unit"], r["takes"][0]["spoken"]) in reviewed for r in shorter)
    fixups = _read_rows(tmp_path / VID / "render" / "fixups.jsonl")
    assert {f["kind"] for f in fixups} == {"rephrase", "review", "settled"}
    assert all(f["coverage"] is not None for f in fixups if f["kind"] == "review")
    assert sum(f["kind"] == "settled" for f in fixups) == len(job.scenes)


class VoicedReviewP(ShortenableClaude):
    """Rephrases longer than the wording they shorten (none is voiced); a voiced-wording review (its scene reviewed
    already) that classes every line P; re-translations that end in a word the TTS runs to its cap on."""

    job: RenderJob | None = None

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        msg = json.loads(prompt)
        if call == "rephrase":
            for x, asked in zip(reply.data["lines"], msg["lines"]):
                x["full"]["spoken"] = asked["failing"] + " కల కల"
        if call == "review" and self.job.scenes[msg["scene"] - 1].reviewed:
            for x in reply.data["lines"]:
                x.update({"class": "P", "missing": ["today"]})
        if call == "retranslate":
            for x in reply.data["lines"]:
                for tier in ("full", "concise", "very_concise"):
                    if tier in x:
                        x[tier]["spoken"] += " ఫెయిల్"
        return reply


async def test_a_re_translation_whose_take_fails_leaves_the_wording_voiced_in_the_line_cache(tmp_path, monkeypatch):
    """The voiced-wording review classes a line's own wording P and its re-translation replaces it, but the
    re-translation's take runs to its cap: the line stays voiced as it was, classed P. The line cache holds that wording
    with that class, never the re-translation no take kept, so a resume restores the line as voiced: no call and no
    synthesis."""
    monkeypatch.setattr(render, "FIXUP_SHARE", 10.0)
    unique_telugu(monkeypatch)
    cli = VoicedReviewP()
    b = backend(UniqueTranscriber(), MelTTS(rate=1.0, fail="ఫెయిల్"))
    b.translator = translator_with(cli)
    job = cli.job = job_for(tmp_path, b)
    assert await job.run() == "done"
    assert not any(st.take["wording"] in render.FIXUP_WORDINGS for st in job.units.values())
    failed = [st for st in job.units.values() if (st.line.coverage.cls, st.line.coverage.tier) == ("P", st.tier)]
    assert failed and any(c["call"] == "retranslate" for c in cli.calls)  # the premise
    for st in failed:
        cached = job.tr.cached(job._line_spec(st, ("full",)))
        assert cached.tiers[st.tier].spoken == st.line.tiers[st.tier].spoken
        assert not any("ఫెయిల్" in w.spoken for w in cached.tiers.values())
        assert (cached.coverage.cls, cached.coverage.tier, cached.coverage.by) == ("P", st.tier, "review")
    b.tts = MelTTS(rate=1.0, fail="ఫెయిల్")
    cli.calls.clear()
    again = cli.job = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert b.tts.batches == [] and cli.calls == [] and again._voiced_now == set()
    assert {k: (st.tier, st.take["tts"], st.line.coverage.cls) for k, st in again.units.items()} == \
           {k: (st.tier, st.take["tts"], st.line.coverage.cls) for k, st in job.units.items()}


class PauseOnVoicedReview(ShortenableClaude):
    """ShortenableClaude, pausing `job` from inside the first voiced-wording review (its scene reviewed already)."""

    job: RenderJob | None = None
    paused = False

    def ask(self, system, prompt, schema=None, call="text", **kw):
        if call == "review" and not self.paused and self.job.scenes[json.loads(prompt)["scene"] - 1].reviewed:
            self.paused = True
            self.job.pause()
        return super().ask(system, prompt, schema, call, **kw)


async def test_a_scene_whose_fix_ups_a_pause_cut_short_is_settled_again_but_a_rephrase_never_rephrased(tmp_path,
                                                                                                      monkeypatch):
    """A pause during a scene's voiced-wording review leaves the scene unsettled: a resume decides its fix-ups again,
    from fixups.jsonl, but a line restored in a rephrase (still long) isn't rephrased again, which the run before never
    asked for."""
    cli = PauseOnVoicedReview()
    job, b, tts = rephrase_job(tmp_path, cli, monkeypatch)
    cli.job = job
    assert await job.run() == "paused"
    rows = take_rows(tmp_path)
    rephrases = {r["takes"][0]["spoken"] for r in rows if r["wording"] == "rephrase"}
    settled = [f for f in _read_rows(tmp_path / VID / "render" / "fixups.jsonl") if f["kind"] == "settled"]
    assert rephrases and len(settled) < len(job.scenes)  # the premise
    cli.calls.clear()
    assert await job.run() == "done"
    asked = [x["failing"] for c in cli.calls if c["call"] == "rephrase" for x in c["message"]["lines"]]
    assert not set(asked) & rephrases
    restored = [st for st in job.units.values() if st.unit.id in job._restored and st.take["wording"] == "rephrase"]
    assert restored and any(st.plan.needs_shorter for st in restored)  # still long: flagged, not rephrased again
    assert all("long" in job.flags[st.unit.id] for st in restored if st.plan.needs_shorter)


class SlowFitClaude(TerseClaude):
    FIT_DELAY = 0.3


async def test_the_dub_loop_waits_for_a_scenes_fit(tmp_path):
    cli = SlowFitClaude()
    b = backend(UniqueTranscriber(), MelTTS(rate=2.0, spread=0.3))  # takes slower than calibrated: the pace moves
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    seen: list[tuple[int, bool, bool]] = []  # (scene, fit back, reviewed) as each line starts
    voice_line = job._voice_line

    async def watched(st):
        sc = job.scenes[st.scene - 1]
        seen.append((sc.no, sc.fit is not None and sc.fit.done(), sc.reviewed))
        return await voice_line(st)

    job._voice_line = watched
    assert await job.run() == "done"
    fitted = {c["message"]["scene"] for c in cli.calls if c["call"] == "fit"}
    assert fitted and seen  # the premise: scenes with a fit
    assert all(done and reviewed for no, done, reviewed in seen if no in fitted)
    assert all(reviewed for _, _, reviewed in seen)
    # A resume restores every line in the wording its take row says: it asks for no fit, nor anything else.
    cli.calls.clear()
    b.tts = MelTTS(rate=2.0, spread=0.3)
    assert await job_for(tmp_path, b).run() == "done"
    assert cli.calls == [] and b.tts.batches == []


async def test_a_scene_reached_while_claude_is_held_is_voiced_unreviewed_and_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(dubber, "CLAUDE_BACKOFF", 1.0)
    unique_telugu(monkeypatch)
    cli = HeldClaude({("review", 2)})
    b = backend(UniqueTranscriber())
    b.translator = translator_with(cli)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    held = job.scenes[1]
    assert held.settled and held.reviewed  # asked again after the hold: nothing left to ask for
    assert all(st.take is not None and st.line.coverage is None for st in held.lines)  # voiced unreviewed
    assert all("unreviewed" in job.flags[st.unit.id] for st in held.lines)
    reviews = [c.get("failed", False) for c in cli.calls if c["call"] == "review" and c["message"]["scene"] == 2]
    assert reviews == [True]  # no voiced-wording review while held, and nothing to review after it
    fixups = {e["scene"]: e for e in trace_of(tmp_path) if e["event"] == "fixups"}
    assert fixups[2]["held"] and fixups[2]["flagged"] == len(held.lines)
    unreviewed = [st for st in job.units.values() if dubber.voiced_class(st.line.coverage, st.tier) == "unreviewed"]
    assert job.doc["coverage"]["unreviewed"] == len(unreviewed) >= len(held.lines)
    assert all(job.flags.get(st.unit.id) for st in unreviewed)


class Rediarized(MockDiarizer):
    """The mock's diarization under another name: its inputs change, so the speakers stage runs again."""


async def test_a_speaker_rerun_with_the_same_count_keeps_every_take_key(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await first.run() == "done"
    keys = {first._line_key(st): [t["key"] for t in st.take["takes"]] for st in first.units.values()}
    b.diarizer, b.tts = Rediarized(), MelTTS()
    again = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await again.run() == "done"
    assert b.diarizer.calls == 1  # diarized again, with the ids kept
    assert {again._line_key(st): [t["key"] for t in st.take["takes"]] for st in again.units.values()} == keys
    assert b.tts.batches == [] and again.tr.cli.calls == []


async def test_after_one_speaker_no_line_gets_a_take_of_another_line_key(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    before = {first._line_key(st): st.take for st in first.units.values()}
    s1 = {first._line_key(st) for st in first.units.values() if st.unit.speaker == "S1"}
    b.tts = MelTTS()
    one = job_for(tmp_path, b, RenderSettings(speakers=1))
    assert await one.run() == "done"
    assert {st.unit.speaker for st in one.units.values()} == {"S1"}
    for st in one.units.values():
        key = one._line_key(st)
        assert st.take["line"] == key  # its own line key's take, never another line's
        if key in s1:
            assert st.take == before[key]  # kept its speaker: restored as it was
        else:
            assert key not in before and st.take["voice"] == one._voice_row(one._key("S1"))
    moved = [st for st in one.units.values() if one._line_key(st) not in s1]
    assert moved and one._voiced_now == {st.unit.id for st in moved}  # voiced again: in S1's voice now
    assert texts_of(b.tts) <= {w.spoken for st in moved for w in st.line.tiers.values()}


async def test_a_torn_row_and_a_torn_take_file_are_ignored(tmp_path, monkeypatch):
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    d = tmp_path / VID / "render"
    rows = take_rows(tmp_path)
    last, early = rows[-1], rows[3]
    assert sum(r["line"] == last["line"] for r in rows) == sum(r["line"] == early["line"] for r in rows) == 1
    data = (d / "takes.jsonl").read_bytes()
    (d / "takes.jsonl").write_bytes(data[:-len(json.dumps(last).encode()) // 2 - 1])  # the engine died writing the row
    npz = d / "takes" / f"{early['takes'][0]['key']}.npz"
    npz.write_bytes(npz.read_bytes()[:200])  # a take file cut short
    b.tts = tts = MelTTS()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert texts_of(tts) == {t["spoken"] for t in early["takes"]}  # only the torn file's take is made again
    assert again._read_take(early["takes"][0]["key"]) is not None  # whole again
    got = {st.take["line"]: st.take for st in again.units.values()}
    def said(row: dict) -> tuple:
        return row["wording"], [{k: v for k, v in t.items() if k != "pace"} for t in row["takes"]]

    for old in (last, early):  # each voiced again as it was: the torn row's line from its take file, with no synthesis
        assert said(got[old["line"]]) == said(old)
    assert got[last["line"]]["cost"]["takes"] == 0 and got[early["line"]]["cost"]["takes"] > 0
    new = take_rows(tmp_path)[len(rows) - 1:]
    assert sorted(r["line"] for r in new) == sorted([last["line"], early["line"]])


async def test_a_line_said_again_by_its_speaker_is_voiced_for_its_own_slot_from_the_take_files(tmp_path):
    """A line whose English and speaker were voiced before (the demo repeats its sentences) is another occurrence, with
    a slot of its own: it is voiced for it, with a row of its own, and a wording the same voice said before comes from
    its take file (§2.1), with no synthesis. A resume restores each from its own row."""
    tts = MelTTS()
    b = backend(tts=tts)
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    keys = {job._line_key(st) for st in job.units.values()}
    assert len(keys) < len(job.units)  # the premise
    assert job._voiced_now == set(job.units)  # each voiced for its own slot...
    made = [m[0] for r in take_rows(tmp_path) for m in r["made"]]
    assert len(tts.batches) == len(made) == len(set(made))  # ...no take made twice...
    assert any(st.take["cost"]["takes"] == 0 for st in job.units.values())  # ...a wording said before from its file
    for st in job.units.values():
        assert (st.take["line"], st.take["at"]) == (job._line_key(st), round(st.unit.start, 3)) and st.plan is not None
    b.tts = MelTTS()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert b.tts.batches == [] and again._voiced_now == set()
    assert {k: st.take for k, st in again.units.items()} == {k: st.take for k, st in job.units.items()}


class LongDemoClaude(ShortenableClaude):
    """ShortenableClaude, whose rephrase of a line is the first word of the wording that ran long."""

    def ask(self, system, prompt, schema=None, call="text", **kw):
        reply = super().ask(system, prompt, schema, call, **kw)
        if call == "rephrase":
            for x in reply.data["lines"]:
                x["full"]["spoken"] = x["full"]["spoken"].split()[0]
        return reply


async def test_the_fix_up_budget_counts_the_takes_fix_ups_make_and_a_rephrase_stays_with_its_line(tmp_path,
                                                                                                    monkeypatch):
    """Repeated lines (the demo's) are long alike, and their shorter tiers and rephrases are said alike: only a fix-up
    that makes a take counts in the budget, one from a take file is free. A rephrase is voiced only for a line of its own
    that ran long, never for another occurrence of it."""
    monkeypatch.setattr(render, "FIXUP_SHARE", 0.3)
    tts = MelTTS(rate=1.0)  # far slower than its estimate: lines run long
    b = backend(tts=tts)
    b.translator = translator_with(LongDemoClaude())
    job = job_for(tmp_path, b)
    fix_ups = []  # (unit, it made a take): every `_say` of a line after its first is a fix-up (a shorter tier, a rephrase)
    said, say = set(), job._say

    async def watched(st, words, *args, **kw):
        before = len(tts.batches)
        got = await say(st, words, *args, **kw)
        if st.unit.id in said:
            fix_ups.append((st.unit.id, len(tts.batches) > before))
        said.add(st.unit.id)
        return got

    long_after_voicing = set()
    voice_line = job._voice_line

    async def voiced(st):
        got = await voice_line(st)
        if st.unit.id in job._long:
            long_after_voicing.add(st.unit.id)
        return got

    job._say, job._voice_line = watched, voiced
    assert await job.run() == "done"
    assert any(made for _, made in fix_ups) and any(not made for _, made in fix_ups)  # the premise
    assert job._resynths == sum(made for _, made in fix_ups)  # only the fix-ups that made a take
    budget = max(3, 0.3 * len(job.units))
    assert job._resynths <= budget + 1 < len(fix_ups)  # the free ones would have run past it
    rephrased = {k for k, st in job.units.items() if st.take["wording"] == "rephrase"}
    assert rephrased and rephrased <= long_after_voicing  # never for a line that fit
    assert all(st.take["at"] == round(st.unit.start, 3) for st in job.units.values())
    # A resume spends the budget as this run did, and asks for nothing.
    b.tts = MelTTS(rate=1.0)
    job.tr.cli.calls.clear()
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert b.tts.batches == [] and again.tr.cli.calls == [] and again._resynths == job._resynths
    assert {k: st.take for k, st in again.units.items()} == {k: st.take for k, st in job.units.items()}


async def test_a_line_whose_wording_or_voice_changed_is_voiced_again(tmp_path, monkeypatch):
    """A take row matches its line only while the line still says the same (the TTS text's hash) with the same voice:
    a line translated again since, or whose speaker's voice was built again, is voiced again."""
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await first.run() == "done"
    st = first.units[4]
    cache = tmp_path / VID / "lines.jsonl"
    rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
    for row in rows:  # the line cache now says line 4 another way (as after a translation made again)
        if row["key"] == first._line_key(st):
            for tier in row["line"].values():
                if isinstance(tier, dict) and "spoken" in tier:
                    tier["spoken"] = tier["spoken"].replace(tier["spoken"].split()[0], "మరో", 1)
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    b.tts = MelTTS()
    again = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await again.run() == "done"
    assert again._voiced_now == {4} and again.units[4].take["tts"] != st.take["tts"]
    # S1's voice built again (a re-run gave S2 a whole turn of the speech it was built from): S1's lines are voiced
    # again in it, S2's come back from their rows.
    b.diarizer, b.tts = StealsMuch(), MelTTS()
    rebuilt = job_for(tmp_path, b, RenderSettings(speakers=2))
    assert await rebuilt.run() == "done"
    assert rebuilt.voices["S1"].key != again.voices["S1"].key and rebuilt.voices["S2"].key == again.voices["S2"].key
    same = {k for k, st in rebuilt.units.items() if again._line_key(again.units[k]) == rebuilt._line_key(st)
            and st.unit.speaker == "S2"}
    assert same and not same & rebuilt._voiced_now
    assert {k for k, st in rebuilt.units.items() if st.unit.speaker == "S1"} <= rebuilt._voiced_now


async def test_the_wording_voiced_is_the_one_reviewed_unless_it_left_the_band_and_another_fits(tmp_path):
    job = job_for(tmp_path, backend(UniqueTranscriber(), MelTTS()))
    assert await job.run() == "done"
    st = next(s for s in job.units.values() if s.speech_s >= 2.0)
    key = job._key(st.unit.speaker)

    def say(seconds: float) -> Wording:  # a wording predicted to last `seconds` at the voice's pace
        n = max(1, round((seconds - job.estimator.overhead(key)) * job.estimator.rate(key)))
        return Wording(" ".join(["కల"] * (n // 2) + ["క"] * (n % 2)))

    def line(**tiers: float) -> LineResult:
        return LineResult(st.unit.id, {t: say(secs * st.speech_s) for t, secs in tiers.items()},
                          coverage=Coverage("C", tier="full"))

    st.line = line(fuller=1.05, full=0.95, concise=0.6)  # the reviewed `full` fits: kept, though `fuller` fits too
    job._wording(st, key)
    assert st.tier == "full" and job._pick(st, st.line, key)[0] == "fuller"
    st.line = line(full=1.6, concise=1.0)  # `full` has left the band and `concise` fits: `concise`
    job._wording(st, key)
    assert st.tier == "concise" and st.telugu == st.line.tiers["concise"].spoken
    st.line = line(full=1.6, concise=1.4)  # nothing fits (the line is placed on W: nothing absorbed): `full` stays
    job._wording(st, key)
    assert st.tier == "full"


async def test_a_line_restored_from_its_take_row_asks_for_no_fit(tmp_path, monkeypatch):
    """On a resume the line cache serves every scene, and a pick made at a pace the takes have since moved could want a
    tier the line lacks: a line with a take row that still matches is voiced as that row says, so it asks for none."""
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    wanted = []

    def fit_spec(self, st):
        wanted.append(st.unit.id)
        return self._line_spec(st, ("concise",))

    monkeypatch.setattr(Dubber, "_fit_spec", fit_spec)  # every pick wants a tier the line lacks
    again = job_for(tmp_path, b)
    assert await again.run() == "done"
    assert wanted == [] and not any(c["call"] == "fit" for c in again.tr.cli.calls)
    st = next(iter(again.units.values()))
    assert again._fit_spec(st) is None  # restorable: no fit
    again._take_rows.clear()
    assert again._fit_spec(st) is not None and wanted == [st.unit.id]  # no row: the fit it would ask for


# ---- the final plan, line PCM and the manifest (§2.11-§2.13, §3) -------------------------------------------------------
MANIFEST_LINE = {"id": int, "speaker": str, "start": float, "srcStart": float, "srcEnd": float, "audioRate": float,
                 "audioWall": float, "lag": float, "said": str, "tier": str, "voice": str, "source": str, "telugu": str,
                 "coverage": str, "flags": list, "pcm": str, "samples": int}


def manifest_of(tmp_path) -> dict:
    return json.loads((tmp_path / VID / "render" / "manifest.json").read_text())


def check_manifest(m: dict, render_dir) -> None:
    """The manifest as §2.13 has it (version 2, the export's input): its fields and nothing the player needed, every
    line with its fields and a PCM file of `samples` frames (float16, 24 kHz), in onset order."""
    assert m.keys() == {"version", "videoId", "title", "channel", "duration", "stopAt", "complete", "sampleRate",
                        "settings", "speakers", "lines", "skipped", "stats"}
    assert m["version"] == 2 and m["sampleRate"] == 24000 and m["complete"] is True
    assert m["settings"].keys() == {"style", "speedCap", "ttsScript", "speakers", "presets"}
    for x in m["lines"]:
        assert x.keys() == MANIFEST_LINE.keys()  # no end, budget, units, freeze or edits
        for f, t in MANIFEST_LINE.items():
            assert isinstance(x[f], (int, float) if t is float else t), (f, x[f])
        assert x["voice"] in ("cloned", "preset") and x["telugu"]
        pcm = np.load(render_dir / "pcm" / f"{x['pcm']}.npy")
        assert pcm.dtype == np.float16 and pcm.shape == (x["samples"],)
        assert x["samples"] == pytest.approx(x["audioWall"] * 24000, abs=0.02 * 24000)
    starts = [x["start"] for x in m["lines"]]
    assert starts == sorted(starts)
    assert m["stats"].keys() == {"lines", "coverage", "lag_p95", "rate_p90", "overdraft_s", "overdraft_carried_s",
                                 "fixups"}
    assert m["stats"]["lines"] == len(m["lines"])


async def test_demo_renders_end_to_end(tmp_path, monkeypatch):
    """§9: the mock backend renders the demo video end to end, offline: every stage done; the manifest valid, each
    line's PCM file of `samples` frames; the lines in order over 0-180 s; the auto speaker blip merged; MockClaude
    asked; no process spawned."""
    import subprocess

    def no_process(*args, **kw):
        raise AssertionError(f"a process was spawned: {args}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_process)
    events: list[dict] = []
    tts = MelTTS()
    b = backend(tts=tts)
    job = job_for(tmp_path, b, events=events)
    assert await job.run() == "done"
    d = tmp_path / VID / "render"
    doc = json.loads((d / "job.json").read_text())
    assert [doc["stages"][k]["state"] for k, _, _ in STAGES] == ["done"] * len(STAGES) and doc["finalUntil"] == 180.0
    m = manifest_of(tmp_path)
    check_manifest(m, d)
    assert m["stopAt"] is None and m["skipped"] == [] and m["stats"]["overdraft_s"] >= 0.0
    assert [x["id"] for x in m["lines"]] == sorted(job.units) and m["lines"][0]["srcStart"] < 2.0
    assert m["lines"][-1]["srcEnd"] > 170.0 and all(0.0 <= x["start"] and x["srcEnd"] <= 180.0 for x in m["lines"])
    assert [s["id"] for s in m["speakers"]] == ["S1", "S2"] and all(s["sample"] for s in m["speakers"])  # blip merged
    assert m["stats"]["coverage"]["C"] == 1.0 and len(job.tr.cli.calls) > 0
    assert {(x["pcm"], x["samples"]) for x in m["lines"]} == {job.pcm[x["id"]] for x in m["lines"]}
    # A line played faster than 1x had its take's mel vocoded again at its rate (the watermarked path, ADR-014).
    fast = {round(x["audioRate"], 3) for x in m["lines"] if x["audioRate"] > 1.001}
    assert fast and fast <= {round(r, 3) for _, r in tts.vocoded}
    # The `unit` trace event with each line's PCM.
    units = [e for e in trace_of(tmp_path) if e["event"] == "unit"]
    assert sorted(e["id"] for e in units) == sorted(job.units) and all(e["pcm"] and e["samples"] for e in units)
    # (the demo repeats its sentences: a line said as one before, at the same rate, found that one's PCM)
    assert 0 < sum(e["pcm_made"] for e in units) == len({x["pcm"] for x in m["lines"]}) < len(units)
    assert job.doc["stages"]["finish"]["total"] == len(job.units)
    assert not any(e["type"] == "manifest" for e in events)  # nothing is watched in the app any more (§4)


async def test_the_manifest_is_written_once_when_every_line_is_final(tmp_path, monkeypatch):
    """§2.13: no manifest.json exists before the finish stage completes; it is written once, atomically, when every line
    of the range is final, while job.json's `finalUntil` follows the finished prefix at each progress tick."""
    d = tmp_path / VID / "render"
    writes: list[tuple[bool, int, int]] = []
    real = render._write_json

    def write(path, doc):
        if path.name == "manifest.json":
            writes.append((path.exists(), job._finished(job._final_lines), len(job._final_lines)))
        real(path, doc)

    monkeypatch.setattr(render, "_write_json", write)
    seen: list[tuple[str, float, float, bool]] = []

    async def on_event(msg: dict) -> None:
        if msg["type"] == "render":
            fin = next(s for s in msg["stages"] if s["key"] == "finish")
            seen.append((fin["state"], fin["done"], fin["total"], (d / "manifest.json").exists()))

    job = job_for(tmp_path, backend(tts=MelTTS()))
    job.on_event = on_event
    ticks: list[float | None] = []
    progress = job._progress

    def tick(key, done, total):
        if key == "finish":
            ticks.append(job.doc["finalUntil"])
        progress(key, done, total)

    job._progress = tick
    assert await job.run() == "done"
    n = len(job._final_lines)
    assert writes == [(False, n, n)]  # once, with every line final, and none there before
    assert seen and all(not there for state, done, total, there in seen if state != "done" and (done < total or not total))
    assert seen[-1][0] == "done" and seen[-1][3]
    assert ticks[0] < 180.0 and ticks == sorted(ticks) and ticks[-1] == 180.0 == job.doc["finalUntil"]
    # A pause (or a failure) writes no manifest either: a run that stops before the end leaves none.
    (d / "manifest.json").unlink()
    again = job_for(tmp_path / "again", backend(tts=MelTTS()))
    task = asyncio.create_task(again.run())
    while again.doc["stages"]["finish"]["state"] != "running":
        await asyncio.sleep(0.001)
    again.pause()
    assert await task == "paused" and not (tmp_path / "again" / VID / "render" / "manifest.json").exists()


class SlowLines(MelTTS):
    """MelTTS whose lines run `slow` times longer than its calibration says: every take outlasts its estimate."""

    def __init__(self, slow: float = 1.6, **kw) -> None:
        super().__init__(**kw)
        self.slow = slow

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1):
        takes = super().synthesize_takes(text, voice, language, None, n)
        for t in takes:
            t.seconds = min(t.seconds * self.slow, max_seconds or math.inf)
        return takes


class Recording(render.TimelinePlanner):
    """A planner that keeps what each `place` was told of the lines ahead."""

    def __init__(self, s) -> None:
        super().__init__(s)
        self.asked: dict[int, list[float | None]] = {}

    def place(self, slot, duration, *args, ahead=None, **kw):
        self.asked[slot.id] = [d for _, d in ahead or ()]
        return super().place(slot, duration, *args, ahead=ahead, **kw)


async def test_final_plan_is_the_same_as_planning_at_the_end(tmp_path, monkeypatch):
    """F, run progressively while the dub loop voices, shorter tiers and rephrases replacing takes, gives each line the
    plan a planner placing every line after the last take gives it (§2.11): the plan never looks past its window, and F
    places a line only once its window's takes are final."""
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 20.0)
    job, b, tts = rephrase_job(tmp_path, ShortenableClaude(), monkeypatch)
    batches: list[int] = []
    place = job._place_final

    def placed(planner, lines):
        batches.append(len(lines))
        return place(planner, lines)

    job._place_final = placed
    assert await job.run() == "done"
    assert len(batches) > 3 and any(r["wording"] == "rephrase" for r in take_rows(tmp_path))  # progressive; fix-ups
    lines = job._final_lines
    at_end = dict((st.unit.id, plan) for st, plan in job._place_final(render.TimelinePlanner(job.planner.s), lines))
    assert at_end == job.final and len(at_end) == len(lines)
    assert [x["pcm"] for x in manifest_of(tmp_path)["lines"]] == [job._pcm_key(st, at_end[st.unit.id]) for st in lines]


async def test_both_planners_run_with_no_freezes(tmp_path):
    """§2.11: a file keeps the video's timing: the working plan W and the final plan F both have max_freeze ==
    freeze_budget == 0, also for a job whose job.json still says allowFreeze (from before freezes went)."""
    job = job_for(tmp_path, backend(tts=MelTTS()))
    assert await job.run() == "done"
    w, f = job.planner, job._final
    assert f is not None and f is not w
    assert (w.s.max_freeze, w.s.freeze_budget, f.s.max_freeze, f.s.freeze_budget) == (0.0, 0.0, 0.0, 0.0)
    assert all(p.freeze == 0.0 for p in job.final.values())
    path = tmp_path / VID / "render" / "job.json"
    doc = json.loads(path.read_text())
    doc["settings"]["allowFreeze"] = True
    path.write_text(json.dumps(doc))
    old = RenderJob(job.b, DemoResolver(), tmp_path, VID)
    assert (old.planner.s.max_freeze, old.planner.s.freeze_budget) == (0.0, 0.0) and not old.allow_freeze


class LongLine(MelTTS):
    """MelTTS whose takes of the first line it voices, in every wording (they start with the same two words: see
    unique_telugu), last `slow` times as long."""

    def __init__(self, slow: float = 3.0, **kw) -> None:
        super().__init__(**kw)
        self.slow, self.lead = slow, None

    def synthesize_takes(self, text, voice, language="te", max_seconds=None, n=1):
        if self.lead is None:
            self.lead = text.split()[:2]
        takes = super().synthesize_takes(text, voice, language, None, n)
        if text.split()[:2] == self.lead:
            for t in takes:
                t.seconds *= self.slow
        return takes


async def test_an_overrun_drifts_with_no_freeze_and_the_speakers_next_line_waits_for_it(tmp_path, monkeypatch):
    """§2.11: a line no wording, speed-up, early start or lag can fit plays whole as overdraft, never a freeze; the
    next line of the same speaker starts after it (`min_gap` later, no overlap) and late, the lateness carried down
    the chain until a pause absorbs it; the line is flagged `long`, and the stats count the overdraft: each line its own
    (`overdraft_s`), the lateness carried apart (`overdraft_carried_s`), as the planner's stats do."""
    unique_telugu(monkeypatch)
    job = job_for(tmp_path, backend(UniqueTranscriber(), LongLine(slow=5.0)))  # (long enough to carry overdraft)
    assert await job.run() == "done"
    s = job._final.s
    first, nxt, *rest = [st for st in job._final_lines if st.unit.speaker == job._final_lines[0].unit.speaker]
    p, q = job.final[first.unit.id], job.final[nxt.unit.id]
    assert p.freeze == 0.0 and p.overdraft > 0.5 and p.carried == 0.0  # its own overrun, as overdraft
    assert p.wall == pytest.approx(first.take_s / p.rate) and p.rate == pytest.approx(s.speed_cap)  # whole, flat out
    assert q.start >= p.start + p.wall + s.min_gap - 1e-6 and q.lag > s.max_lag  # after it, late
    assert any(job.final[st.unit.id].lag <= s.max_lag for st in rest)  # absorbed further on
    assert all(plan.freeze == 0.0 for plan in job.final.values())
    assert "long" in job.flags[first.unit.id]
    m = manifest_of(tmp_path)
    assert "long" in next(x for x in m["lines"] if x["id"] == first.unit.id)["flags"]
    # Each line counts the overdraft its own audio caused; the lateness carried down the chain is counted apart, once.
    plans = [job.final[x["id"]] for x in m["lines"]]
    assert m["stats"]["overdraft_s"] == round(sum(x.own_overdraft for x in plans), 2) >= round(p.overdraft, 2) > 0.0
    assert m["stats"]["overdraft_carried_s"] == round(sum(x.carried for x in plans), 2) > 0.0
    assert m["stats"]["overdraft_s"] < round(sum(x.overdraft for x in plans), 2)


async def test_final_plan_uses_actual_durations(tmp_path, monkeypatch):
    """F prices the lines of each window at their takes' actual seconds, never an estimate (§2.11): with every take
    outlasting its estimate, W planned lines against estimates that F corrects, and F's plans are the ones played."""
    monkeypatch.setattr(render, "TimelinePlanner", Recording)
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), SlowLines())
    job = job_for(tmp_path, b)
    assert await job.run() == "done"
    w, f = job.planner, job._final
    assert isinstance(f, Recording) and f is not w
    for st in job._final_lines:
        window = job._window(st)
        assert f.asked[st.unit.id] == [x.take_s for x in window]  # actual seconds, for every line of its window
        est = w.asked[st.unit.id]
        assert all(e is None or e < x.take_s for e, x in zip(est, window))  # W's: estimates, short of the takes
    changed = [st for st in job._final_lines if job.final[st.unit.id] != st.plan]
    assert changed  # F's plan corrects W's where the lines after it turned out longer
    lines = {x["id"]: x for x in manifest_of(tmp_path)["lines"]}
    for st in job._final_lines:  # what plays is F's plan
        plan, x = job.final[st.unit.id], lines[st.unit.id]
        assert (x["start"], x["audioRate"], x["audioWall"]) == (round(plan.start, 3), round(plan.rate, 4),
                                                                round(plan.wall, 4))


def test_quit_puts_back_only_the_stages_this_run_has_running(tmp_path):
    """`quit()` (§4) writes the graceful state at once: the stages running now go back to `todo` with their start not
    counted; a stage a crash left `running` that this run hasn't reached yet keeps its count (the crash-loop guard)."""
    job = job_for(tmp_path, backend())
    st = job.doc["stages"]
    st["speakers"].update(state="running", attempts=1)
    st["voice_lines"].update(state="running", attempts=2)  # a crash's, not started again yet
    job._active.append("speakers")
    job.quit()
    doc = json.loads((tmp_path / VID / "render" / "job.json").read_text())
    assert doc["status"] == "interrupted"
    assert [(doc["stages"][k]["state"], doc["stages"][k]["attempts"]) for k in ("speakers", "voice_lines")] == \
           [("todo", 0), ("running", 2)]


async def test_a_preview_continued_has_the_span_it_adds_left(tmp_path):
    """§4: a job's time left by the priors (`render.left`: the jobs-ahead estimate; and the ETA) counts a stage done for
    a preview's range by the video seconds the whole video adds, since what the preview did is served from its caches;
    a stage done for the range asked has nothing left."""
    b = backend()
    assert await job_for(tmp_path, b, RenderSettings(stop_at=60.0)).run() == "done"
    path = tmp_path / VID / "render" / "job.json"
    doc = json.loads(path.read_text())
    assert doc["stages"]["voice_lines"]["span"] == 60.0 + render.PAST_STOP and doc["stages"]["speakers"]["span"] == 180.0
    assert render.left(doc) == 0.0
    whole = job_for(tmp_path, b, RenderSettings())  # "Continue to the whole video", queued behind another job
    await whole.hold("queued", time.time())
    # the background sound's added span then the dub loop's, in series (the longest of their group), and the export's
    # (a group of its own)
    added = (sum(render.PRIORS["voice_lines"]) + sum(render.PRIORS["export"])) / 2 * (180.0 - 60.0 - render.PAST_STOP) \
        + sum(render.PRIORS["separate"]) / 2 * (180.0 - 60.0 - 2 * render.PAST_STOP)
    assert render.left(json.loads(path.read_text())) == pytest.approx(added) and added > 60
    assert whole._eta("voice_lines", time.monotonic()) == pytest.approx(
        sum(render.PRIORS["voice_lines"]) / 2 * (180.0 - 60.0 - render.PAST_STOP))
    # the job view shows the stages the whole video adds to as still to do (job.json keeps them done, with their span):
    # it never ticks "Saving the video" for the preview's file while the whole video is dubbed
    shown = {s["key"]: s["state"] for s in whole.snapshot()["stages"]}
    assert {k for k, v in shown.items() if v == "todo"} == {"separate", "translate", "voice_lines", "finish", "export"}
    assert {json.loads(path.read_text())["stages"][k]["state"] for k in shown} == {"done"}
    assert await whole.run() == "done"
    assert render.left(json.loads(path.read_text())) == 0.0
    assert {s["state"] for s in whole.snapshot()["stages"]} == {"done"}


async def test_preview_then_continue_reuses_caches(tmp_path, monkeypatch):
    """§3: a preview (stopAt 60) voices the lines that start before 90 s and makes final those before 60 s: done, with
    finalUntil 60. Continuing to the whole video diarizes and transcribes nothing again, and each line the preview made
    final keeps its takes and its PCM key, whose file isn't written again."""
    monkeypatch.setattr(dubber, "SCENE_MAX_S", 20.0)
    unique_telugu(monkeypatch)
    tts = MelTTS(spread=0.05)
    b = backend(UniqueTranscriber(), tts)
    preview = job_for(tmp_path, b, RenderSettings(stop_at=60.0))
    assert await preview.run() == "done"
    m = manifest_of(tmp_path)
    check_manifest(m, tmp_path / VID / "render")
    assert m["stopAt"] == 60.0 and preview.doc["finalUntil"] == 60.0
    assert {x["id"] for x in m["lines"]} == {k for k, st in preview.units.items() if st.unit.start < 60.0}
    assert {k for k, st in preview.units.items() if st.take is not None} == \
           {k for k, st in preview.units.items() if st.unit.start < 60.0 + render.PAST_STOP}
    before = {x["id"]: (preview.units[x["id"]].take["takes"], x["pcm"]) for x in m["lines"]}
    files = {x["pcm"]: (tmp_path / VID / "render" / "pcm" / f"{x['pcm']}.npy").stat().st_mtime_ns for x in m["lines"]}
    heard, diarized, made = b.transcriber.calls, b.diarizer.calls, len(tts.batches)
    whole = job_for(tmp_path, b, RenderSettings())
    assert await whole.run() == "done"
    assert (b.transcriber.calls, b.diarizer.calls) == (heard, diarized)
    m2 = manifest_of(tmp_path)
    check_manifest(m2, tmp_path / VID / "render")
    assert m2["stopAt"] is None and whole.doc["finalUntil"] == 180.0
    after = {x["id"]: (whole.units[x["id"]].take["takes"], x["pcm"]) for x in m2["lines"]}
    assert {k: after[k] for k in before} == before  # the same takes and PCM keys...
    assert all((tmp_path / VID / "render" / "pcm" / f"{k}.npy").stat().st_mtime_ns == t for k, t in files.items())
    assert whole._voiced_now and not whole._voiced_now & set(before)  # restored from their rows, not voiced again
    assert len(tts.batches) > made
    made = [e["id"] for e in trace_of(tmp_path) if e["event"] == "unit" and e["pcm_made"]]
    assert sorted(made) == sorted(whole.units)  # ...each line's PCM made once in all


async def test_orphan_takes_and_pcm_are_collected(tmp_path, monkeypatch):
    """§2.13: after a manifest rewrite, the take files no take row a line uses names and the PCM files the manifest
    doesn't name are deleted: here a stray pair, and what a speaker's switch to a stock voice (`set_voice`) left."""
    unique_telugu(monkeypatch)
    b = backend(UniqueTranscriber(), MelTTS())
    first = job_for(tmp_path, b)
    assert await first.run() == "done"
    d = tmp_path / VID / "render"
    shutil.copy(next((d / "takes").glob("*.npz")), d / "takes" / "0123456789abcdef0123.npz")
    shutil.copy(next((d / "pcm").glob("*.npy")), d / "pcm" / "0123456789abcdef0123.npy")
    s2 = {k for k, st in first.units.items() if st.unit.speaker == "S2"}
    old = {k: (first.units[k].take["takes"][0]["key"], first.pcm[k][0]) for k in s2}
    again = job_for(tmp_path, b, replace(first.settings, presets=("S2",)))
    assert await again.run() == "done"
    assert again._voiced_now == s2 and all(again.units[k].take["voice"][3] == "v1:preset" for k in s2)
    takes = {p.stem for p in (d / "takes").glob("*.npz")}
    pcm = {p.stem for p in (d / "pcm").glob("*.npy")}
    assert takes == {t["key"] for st in again.units.values() for t in st.take["takes"]}
    assert pcm == {x["pcm"] for x in manifest_of(tmp_path)["lines"]}
    assert "0123456789abcdef0123" not in takes | pcm
    assert not {t for t, _ in old.values()} & takes and not {p for _, p in old.values()} & pcm
    # S1 was untouched: its lines kept their takes and PCM.
    assert all(again.pcm[k] == first.pcm[k] for k in again.units if k not in s2)


# ---- the video, the TTS let go of, the export (§2.2, §2.13, §2.17) ----------------------------------------------------
class ReleasingTTS(MockTTS):
    """The mock's voice, recording its syntheses and its releases in order."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def synthesize(self, text, voice, language="te", max_seconds=None):
        self.log.append("synthesize")
        return super().synthesize(text, voice, language, max_seconds)

    def release(self) -> None:
        self.log.append("release")


@pytest.mark.parametrize("stop_at", [None, 60.0])
async def test_the_tts_is_let_go_of_once_the_dub_is_final_and_the_dub_loop_has_ended(tmp_path, stop_at):
    """§2.13: the export needs no GPU. In a preview the dub loop outlives the final plan (it voices the lines past the
    stop point): the model goes only once both have ended, so it is never loaded twice."""
    tts = ReleasingTTS()
    job = job_for(tmp_path, backend(tts=tts), RenderSettings(stop_at=stop_at))
    assert await job.run() == "done"
    assert tts.log.count("release") == 1 and tts.log[-1] == "release" and "synthesize" in tts.log
    # (the demo's preview settles its one scene before its final plan ends: the rule itself, as a preview of a longer
    # video meets it, with the final plan done while the dub loop still voices the lines past the stop point)
    tts.log.clear()
    job._final_done, job._dubbing = True, True
    await job._let_go_of_tts()
    assert tts.log == []
    job._dubbing = False
    await job._let_go_of_tts()
    assert tts.log == ["release"]


async def test_the_export_shows_finishing_the_file_while_it_is_written(tmp_path):
    events: list[dict] = []
    job = job_for(tmp_path, backend(), events=events)
    assert await job.run() == "done"
    rows = [s for e in events if e["type"] == "render" for s in e["stages"] if s["key"] == "export"]
    assert any(s.get("extra") == "Finishing the file…" and s["state"] == "running" for s in rows)
    assert "extra" not in rows[-1] and rows[-1]["done"] == rows[-1]["total"] == 180.0
    assert events[-1]["output"]["kind"] == "whole" and not events[-1]["output"]["missing"]


def test_the_estimate_counts_the_video_download_beside_the_dub_and_the_export_after_it():
    whole, preview = render.estimate(10_506.0), render.estimate(10_506.0, 900.0)
    for i in (0, 1):
        span = {k: render._span(k, 10_506.0, None) for k, _, _ in STAGES}
        sec = {k: render.PRIORS[k][i] * span[k] for k in span}
        side = max(sec["translate"], sec["video"], sec["separate"] + max(sec["voice_lines"], sec["finish"]))
        want = sum(side if "separate" in g else max(sec[k] for k in g) for g in render.ORDER)
        assert whole["seconds"][i] == round(want)
        # the GPU separates, then dubs: in series, not beside each other
        assert side > max(sec[k] for k in render.ORDER[5]) and side == pytest.approx(sec["separate"] + sec["voice_lines"])
    # a preview separates to its stop point plus twice PAST_STOP (its cut), and dubs to its stop point plus PAST_STOP
    assert render._span("separate", 10_506.0, 900.0) == 900.0 + 2 * render.PAST_STOP
    lo, hi = render.PRIORS["separate"]
    assert render.estimate(600.0)["seconds"][1] - render.estimate(600.0)["seconds"][0] >= round((hi - lo) * 600.0) - 1
    lo, hi = render.PRIORS["export"]
    # the export of a preview covers its range; the video download is the whole stream either way
    assert render._span("export", 10_506.0, 900.0) == 900.0 + render.PAST_STOP
    assert render._span("video", 10_506.0, 900.0) == 10_506.0
    assert whole["seconds"][1] - preview["seconds"][1] > hi * (10_506.0 - 930.0)


# ---- the background sound (OFFLINE-RENDER §2.14) --------------------------------------------------------------------
class SlowSeparator(mock.MockSeparator):
    """The mock separator, `delay` s a call, logging each call into `log`."""

    def __init__(self, log: list, delay: float = 0.05) -> None:
        super().__init__()
        self.log, self.delay = log, delay

    def vocals(self, batch):
        time.sleep(self.delay)
        self.log.append("separate")
        return super().vocals(batch)


class LoggingTTS(MockTTS):
    def __init__(self, log: list) -> None:
        self.log = log

    def synthesize(self, text, voice, language="te", max_seconds=None):
        self.log.append("tts")
        return super().synthesize(text, voice, language, max_seconds)


async def test_the_dub_loop_waits_for_the_separator(tmp_path):
    """§2.14: in the group where the GPU runs `separate`, then the dub loop, no line is voiced until the background
    sound is done (the translation goes on meanwhile), and the dub loop's row says why it waits."""
    log: list[str] = []
    b = backend(tts=LoggingTTS(log))
    b.separator = SlowSeparator(log)
    events: list[dict] = []
    job = job_for(tmp_path, b, RenderSettings(stop_at=60.0), events=events)
    assert await job.run() == "done"
    first, last = log.index("separate"), len(log) - 1 - log[::-1].index("separate")
    assert "tts" not in log[first:last] and "tts" in log[last:]  # (the voices stage calibrated before it)
    rows = [r for e in events if e["type"] == "render" for r in e["stages"] if r["key"] == "voice_lines"]
    assert any(r.get("extra") == render.WAITING_BED and r["state"] == "running" for r in rows)
    assert "extra" not in rows[-1]
    assert job.doc["bed"]["separator"] == "SlowSeparator" and job.doc["stages"]["separate"]["state"] == "done"
    assert b.separator.released >= 1


async def test_a_pause_while_the_dub_loop_waits_for_the_separator_stops_both(tmp_path):
    b = backend()
    sep = b.separator = SlowSeparator([], delay=0.1)
    job = job_for(tmp_path, b, RenderSettings(stop_at=60.0))
    real = sep.vocals

    def pausing(batch):
        if sep.calls == 3:
            job.pause()
        return real(batch)

    sep.vocals = pausing
    assert await job.run() == "paused"
    st = job.doc["stages"]
    assert st["separate"]["state"] == "todo" and st["voice_lines"]["state"] == "todo"
    assert not (tmp_path / VID / "render" / "takes.jsonl").exists() or not _read_rows(
        tmp_path / VID / "render" / "takes.jsonl")  # no line voiced
    assert await job.run() == "done"


async def test_a_speakers_rerun_doesnt_separate_again(tmp_path):
    """§2.3, §2.14: the bed doesn't depend on the speakers. A speaker-count change re-runs diarization, and the export
    mixes again over the same bed, its balance from the new turns, with no separator call."""
    b = backend()
    b.separator = mock.MockSeparator()
    first = job_for(tmp_path, b, RenderSettings(stop_at=60.0))  # a preview: its bed stays
    assert await first.run() == "done"
    calls, bed = b.separator.calls, (tmp_path / VID / "render" / "bed.json").read_bytes()
    assert calls > 0 and first.doc["output"]["inputs"]["bed"]["separator"] == "MockSeparator"
    one = job_for(tmp_path, b, RenderSettings(stop_at=60.0, speakers=1))
    assert await one.run() == "done"
    assert b.diarizer.calls == 2 and b.separator.calls == calls
    assert (tmp_path / VID / "render" / "bed.json").read_bytes() == bed
    a, c = first.doc["output"]["inputs"], one.doc["output"]["inputs"]
    assert a["bed"] == c["bed"] and a["turns"] != c["turns"]  # (the speech, and so the duck spans, cover the same time)
    trace = [json.loads(x) for x in (tmp_path / VID / "units.jsonl").read_text().splitlines()]
    assert [e["cached"] for e in trace if e["event"] == "stage" and e["key"] == "separate"] == [False, True]


class Unfetched(mock.MockSeparator):
    """A separator whose models folder lacks its files (one filled before the separator was pinned), until fetched."""

    def __init__(self) -> None:
        super().__init__()
        self.lacking = ["config.json", "model.safetensors"]

    def missing(self) -> list[str]:
        return list(self.lacking)


async def test_a_job_whose_separator_model_isnt_downloaded_fails_at_once_saying_how_to_fetch_it(tmp_path):
    """The separator is the one model a models folder filled before it was pinned lacks: a job that will separate fails
    before its first stage (not after the GPU stages, with "Rendering stopped unexpectedly."), and resumes once it is
    fetched."""
    b = backend()
    sep = b.separator = Unfetched()
    job = job_for(tmp_path, b, RenderSettings(stop_at=60.0))
    assert await job.run() == "failed"
    assert job.doc["error"] == ("The background-sound model isn't downloaded. Run `uv run maata-bench fetch --backend "
                                "mock` in Maata's engine folder, then resume.")
    assert job.doc["stage"] == "separate" and not any(st["attempts"] for st in job.doc["stages"].values())
    assert b.diarizer.calls == 0 and sep.calls == 0
    sep.lacking = []  # `maata-bench fetch`
    assert await job.run() == "done" and sep.calls > 0 and job.doc["error"] is None


async def test_with_no_separator_the_estimate_the_time_left_and_the_eta_count_no_background_sound(tmp_path):
    """With no separator the separate stage is served at once: the prepare view's estimate, a queued job's time left
    and a running job's ETA count nothing for it, which is the estimate from before the background sound."""
    for duration, stop in ((10_506.0, None), (10_506.0, 900.0), (600.0, None)):
        got = render.estimate(duration, stop, separates=False)
        for i in (0, 1):
            sec = {k: render.PRIORS[k][i] * render._span(k, duration, stop) for k, _, _ in STAGES}
            want = sum(max(sec[k] for k in g if k != "separate") for g in render.ORDER)  # every group side by side
            assert got["seconds"][i] == round(want)
            assert render.estimate(duration, stop)["seconds"][i] > got["seconds"][i]  # (with one: in series)
        assert got["claudeCalls"] == render.estimate(duration, stop)["claudeCalls"]
    queued = {"duration": 600.0, "settings": {"stopAt": None}, "stages": {}}
    mid = {k: sum(render.PRIORS[k]) / 2 * 600.0 for k, _, _ in STAGES}
    assert render.left(queued, separates=False) == pytest.approx(
        sum(max(mid[k] for k in g if k != "separate") for g in render.ORDER))
    side = max(mid["translate"], mid["video"], mid["voice_lines"])
    assert render.left(queued) - render.left(queued, separates=False) == pytest.approx(
        max(mid["translate"], mid["video"], mid["separate"] + mid["voice_lines"]) - side)
    none, one = backend(), backend()
    one.separator = mock.MockSeparator()
    for name, b, want in (("none", none, 0.0), ("one", one, mid["separate"])):
        job = job_for(tmp_path / name, b)
        job.doc["duration"] = 600.0
        assert job._eta("separate", time.monotonic()) == pytest.approx(want)
