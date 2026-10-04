"""End-to-end over the real WebSocket with the demo engine: no network, no models."""

import asyncio
import json
import struct
import sys

import pytest
from websockets.asyncio.client import connect

from maata_engine.server import Engine
from websockets.asyncio.server import serve


@pytest.fixture
async def engine_url(tmp_path):
    eng = Engine("mock", tmp_path / "models", tmp_path / "cache", None, "tok", demo=True)
    async with serve(eng.handler, "127.0.0.1", 0, process_request=eng.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws"


async def test_rejects_bad_token(engine_url):
    with pytest.raises(Exception):
        async with connect(engine_url + "?token=nope") as ws:
            await ws.recv()


def test_stdin_lifeline_exits_when_shell_goes_away(tmp_path):
    """The shell holds the engine's stdin; if it dies (even SIGKILL), the engine must not linger."""
    import subprocess
    import sys

    p = subprocess.Popen(
        [sys.executable, "-m", "maata_engine.server", "--backend", "mock", "--demo", "--port", "0",
         "--models", str(tmp_path / "m"), "--cache", str(tmp_path / "c"), "--config-dir", str(tmp_path / "cfg"),
         "--stdin-lifeline"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        assert p.stdout.readline().startswith("MAATA_ENGINE_READY ")
        p.stdin.close()  # what the kernel does to the pipe when the shell process dies
        assert p.wait(timeout=10) == 0
    finally:
        p.kill()


# ---- the offline render over the WebSocket (OFFLINE-RENDER §4, §9) ---------------------------------------------------
import contextlib
import functools
import threading
import time

import numpy as np

from maata_engine import render as render_mod
from maata_engine.backends.base import COARSE_STEP
from maata_engine.backends.mock import MockTTS
from maata_engine.playback import SAMPLE_ID
from maata_engine.render import STAGES, RenderJob, RenderSettings, _read_rows
from maata_engine.resolve import DemoResolver, decode_audio
from maata_engine.server import CRASH_STARTS, retain

URL_A, VID_A = "https://youtu.be/demo0000001", "demo0000001"
URL_B, VID_B = "https://youtu.be/demo0000002", "demo0000002"
URL_C, VID_C = "https://youtu.be/demo0000003", "demo0000003"
URL_D, VID_D = "https://youtu.be/demo0000004", "demo0000004"


class SlowTTS(MockTTS):
    """The mock's voice, taking `delay` s a synthesis, so a render lasts long enough to pause and ask about."""

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay

    def synthesize(self, text, voice, language="te", max_seconds=None):
        time.sleep(self.delay)
        return super().synthesize(text, voice, language, max_seconds)


class GateTTS(SlowTTS):
    """SlowTTS whose syntheses wait at a gate while it is shut (`waiting` says one is): a model call in its worker thread
    that outlasts the engine's QUIT_WAIT."""

    def __init__(self) -> None:
        super().__init__()
        self.open, self.waiting = threading.Event(), threading.Event()
        self.open.set()

    def synthesize(self, text, voice, language="te", max_seconds=None):
        if not self.open.is_set():
            self.waiting.set()
            self.open.wait(30)
        return super().synthesize(text, voice, language, max_seconds)


async def stop(e) -> None:
    """Leave the engine idle: nothing queued, the job running paused and its run returned."""
    e.queue.clear()
    if e.job is not None:
        e.job.pause()
    if e._job_task is not None:
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(e._job_task, 20)


@pytest.fixture
async def eng(tmp_path):
    """A demo engine (mock backend, DemoResolver) serving on a free port; its jobs are stopped at the end."""
    e = Engine("mock", tmp_path / "models", tmp_path / "cache", None, "tok", demo=True)
    async with serve(e.handler, "127.0.0.1", 0, process_request=e.process_request, max_size=2**24) as server:
        e.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws?token=tok"
        try:
            yield e
        finally:
            await stop(e)


class Window:
    """A connected UI window: every message and frame it receives, in order."""

    def __init__(self, ws) -> None:
        self.ws, self.msgs, self.frames = ws, [], []
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        with contextlib.suppress(Exception):
            async for m in self.ws:
                if isinstance(m, bytes):
                    uid, sr, n, _ = struct.unpack("<IIII", m[:16])
                    assert len(m) == 16 + 4 * n
                    self.frames.append((uid, sr, np.frombuffer(m[16:], np.float32)))
                else:
                    self.msgs.append(json.loads(m))

    async def send(self, **msg) -> None:
        await self.ws.send(json.dumps(msg))

    async def until(self, pred, timeout: float = 30.0, after: int = 0):
        """The first message from number `after` on (received already, or waited for) for which `pred` holds."""
        t0 = time.monotonic()
        while True:
            for m in self.msgs[after:]:
                if pred(m):
                    return m
            assert time.monotonic() - t0 < timeout, f"timed out; last messages: {self.msgs[-3:]}"
            await asyncio.sleep(0.01)

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == kind]


@contextlib.asynccontextmanager
async def window(e):
    async with connect(e.url, max_size=2**24) as ws:
        w = Window(ws)
        await w.until(lambda m: m["type"] == "hello")
        yield w


def done(vid: str):
    return lambda m: m["type"] == "render" and m["videoId"] == vid and m["status"] == "done"


def status(vid: str, want: str):
    return lambda m: m["type"] == "render" and m["videoId"] == vid and m["status"] == want


def job_json(e, vid: str) -> dict:
    return json.loads((e.cache_dir / vid / "render" / "job.json").read_text())


def waiting(e) -> list[str]:
    return [vid for vid, _ in e.queue]


async def until(pred, timeout: float = 20.0) -> None:
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "timed out"
        await asyncio.sleep(0.005)


async def stage_running(e, vid: str, stage: str = "voice_lines", attempts: int | None = None) -> None:
    """Wait until `vid`'s job runs on `e` and is in `stage` (its start number `attempts`: a job.json a kill left says
    `running` before the stage has started again)."""
    await until(lambda: e.job is not None and e.job.video_id == vid
                and e.job.doc["stages"][stage]["state"] == "running"
                and (attempts is None or e.job.doc["stages"][stage]["attempts"] == attempts))


async def test_prepare_reports_progress_to_done(eng):
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A, speakers="auto", style="colloquial", stopAt=None, speedCap=1.2,
                     allowFreeze=True, ttsScript="telugu")  # (an old UI's allowFreeze is ignored)
        await w.until(done(VID_A))
    renders = w.of("render")
    # The stages walk to done, in order, each reported running before done; ETAs while it runs; not queued.
    order = [k for k, _, _ in STAGES]
    started = [next(i for i, m in enumerate(renders) if m["stage"] == k) for k in order if any(m["stage"] == k for m in renders)]
    assert started == sorted(started)
    assert [s["state"] for s in renders[-1]["stages"]] == ["done"] * len(STAGES) and renders[-1]["finalUntil"] == 180.0
    assert all(isinstance(m["eta"], (int, float)) for m in renders if m["status"] == "running")
    assert all(m["position"] is None and m["slept"] is None for m in renders)
    # Nothing is watched or played in the app (§4, §6): no manifest is pushed; it is the export's input, on disk.
    assert not w.of("manifest") and not w.frames
    m = json.loads((eng.cache_dir / VID_A / "render" / "manifest.json").read_text())
    assert m["version"] == 2 and m["complete"] and [s["id"] for s in m["speakers"]] == ["S1", "S2"]
    assert "allowFreeze" not in job_json(eng, VID_A)["settings"]


async def test_hello_carries_a_snapshot(eng):
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.until(lambda m: m["type"] == "render" and m["stage"] == "voice_lines")
        async with connect(eng.url) as ws:  # a new or reconnected window is up to date at once
            hello = json.loads(await ws.recv())
        assert hello["type"] == "hello" and hello["render"]["videoId"] == VID_A
        assert hello["render"]["status"] == "running" and len(hello["render"]["stages"]) == len(STAGES)
        assert hello["render"]["position"] is None
        assert [(r["videoId"], r["status"], r["position"]) for r in hello["renders"]] == [(VID_A, "running", None)]
        assert hello["renders"][0]["title"] == "Demo video" and hello["renders"][0]["bytes"] > 0


async def test_pause_and_resume(eng):
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.until(lambda m: m["type"] == "render" and m["stage"] == "voice_lines")
        await w.send(type="pause", videoId=VID_A)
        paused = await w.until(status(VID_A, "paused"))
        assert eng.job is None or eng.job.stopping
        await asyncio.wait_for(eng._job_task, 10)
        doc = job_json(eng, VID_A)
        assert doc["status"] == "paused" and paused["eta"] is None
        n = len(_read_rows(eng.cache_dir / VID_A / "render" / "takes.jsonl"))
        await w.send(type="resume", videoId=VID_A)
        await w.until(done(VID_A))
    assert len(_read_rows(eng.cache_dir / VID_A / "render" / "takes.jsonl")) > n  # it went on from where it stopped
    assert eng.backend.diarizer.calls == 1  # from disk


async def test_a_second_prepare_is_queued_and_runs_after_the_first(eng):
    """§4: one job runs at a time. A prepare while another runs is `queued` (job.json, `queuedAt`), its render snapshot
    sent to every window with its place in line, also in `renders`; it runs once the first is done. There is no
    `busy` any more."""
    eng.backend.tts = SlowTTS()
    async with window(eng) as w, window(eng) as other:
        await w.send(type="prepare", url=URL_A)
        await w.until(status(VID_A, "running"))
        await w.send(type="prepare", url=URL_B)
        for x in (w, other):  # to every window
            queued = await x.until(status(VID_B, "queued"))
            assert queued["position"] == 1 and queued["eta"] is None
        doc = job_json(eng, VID_B)
        assert doc["status"] == "queued" and doc["queuedAt"] > 0 and eng.job.video_id == VID_A
        await w.send(type="renders")
        items = {x["videoId"]: x for x in (await w.until(lambda m: m["type"] == "renders"))["items"]}
        assert (items[VID_A]["status"], items[VID_A]["position"]) == ("running", None)
        assert (items[VID_B]["status"], items[VID_B]["position"]) == ("queued", 1)
        await w.until(done(VID_A))
        await w.until(done(VID_B))
    a_done = next(i for i, m in enumerate(w.msgs) if done(VID_A)(m))
    b_runs = next(i for i, m in enumerate(w.msgs) if status(VID_B, "running")(m))
    assert a_done < b_runs  # one at a time, in order
    assert not w.of("busy")


async def test_inspect_says_how_much_work_is_ahead(eng):
    """The `video` estimate's `ahead` (§4): None while nothing runs; with a job running, its time left and the queued
    jobs' (by the priors), never counting the video asked about."""
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="inspect", url=URL_C)
        video = await w.until(lambda m: m["type"] == "video")
        assert video["estimate"]["ahead"] is None and len(video["estimate"]["seconds"]) == 2
        await w.send(type="prepare", url=URL_A)
        await w.until(status(VID_A, "running"))
        n = len(w.msgs)
        await w.send(type="inspect", url=URL_C)
        alone = (await w.until(lambda m: m["type"] == "video", after=n))["estimate"]["ahead"]
        await w.send(type="inspect", url=URL_B)  # its metadata is kept: a queued job that hasn't fetched shows it
        await w.until(lambda m: m["type"] == "video" and m["videoId"] == VID_B, after=n)
        await w.send(type="prepare", url=URL_B)
        await w.until(status(VID_B, "queued"))
        assert job_json(eng, VID_B)["duration"] == 180.0 and job_json(eng, VID_B)["title"] == "Demo video"
        n = len(w.msgs)
        await w.send(type="inspect", url=URL_C)
        both = (await w.until(lambda m: m["type"] == "video", after=n))["estimate"]["ahead"]
    assert isinstance(alone, int) and alone > 0 and render_mod.left(job_json(eng, VID_B)) > 0
    assert both > alone  # B's whole render, by the priors, comes on top of what A has left


async def test_pausing_the_running_job_starts_the_next_and_pausing_a_queued_one_takes_it_out(eng):
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.until(status(VID_A, "running"))
        await w.send(type="prepare", url=URL_B)
        await w.send(type="prepare", url=URL_C)
        await w.until(status(VID_C, "queued"))
        assert waiting(eng) == [VID_B, VID_C]
        await w.send(type="pause", videoId=VID_C)  # queued: out of the queue, paused
        paused = await w.until(status(VID_C, "paused"))
        assert paused["position"] is None and waiting(eng) == [VID_B] and job_json(eng, VID_C)["status"] == "paused"
        await w.send(type="pause", videoId=VID_A)  # running: it stops, and the next in line starts
        await w.until(status(VID_A, "paused"))
        await w.until(status(VID_B, "running"))
        assert eng.job.video_id == VID_B and waiting(eng) == []
        await w.until(done(VID_B))
    await until(lambda: eng.job is None)
    assert job_json(eng, VID_A)["status"] == "paused" and job_json(eng, VID_C)["status"] == "paused"


async def test_resume_queues_at_the_back_and_a_rerun_of_the_running_job_goes_first(eng):
    await run_job(eng, URL_C)  # a job done before: a resume (or a re-run) of it waits its turn at the back
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.until(status(VID_A, "running"))
        await w.send(type="prepare", url=URL_B)
        await w.until(status(VID_B, "queued"))
        await w.send(type="resume", videoId=VID_C)
        await w.until(status(VID_C, "queued"))
        assert waiting(eng) == [VID_B, VID_C] and job_json(eng, VID_C)["queuedAt"] > job_json(eng, VID_B)["queuedAt"]
        n = len(w.msgs)
        await w.send(type="resume", videoId=VID_B)  # queued already: keeps its place
        assert (await w.until(status(VID_B, "queued"), after=n))["position"] == 1
        assert waiting(eng) == [VID_B, VID_C]
        # A re-run of the job running pauses it and is first in line, with its new settings, from when it has stopped.
        n = len(w.msgs)
        await w.send(type="set_speakers", videoId=VID_A, speakers=1)
        first = await w.until(lambda m: status(VID_A, "running")(m) and m["position"] == 1, after=n)
        k = w.msgs.index(first)
        await w.until(status(VID_A, "paused"), after=k)
        again = await w.until(lambda m: status(VID_A, "running")(m) and m["position"] is None, after=k)
        assert w.msgs.index(again) > w.msgs.index(next(m for m in w.msgs[k:] if status(VID_A, "paused")(m)))
        assert eng.job.video_id == VID_A and eng.job.settings.speakers == 1 and waiting(eng) == [VID_B, VID_C]
        await stop(eng)
    assert job_json(eng, VID_A)["settings"]["speakers"] == 1


async def test_renders_lists_jobs(eng):
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A, stopAt=60)
        await w.until(done(VID_A))
        await w.send(type="renders")
        items = (await w.until(lambda m: m["type"] == "renders"))["items"]
    assert [(x["videoId"], x["status"], x["stopAt"], x["position"]) for x in items] == [(VID_A, "done", 60.0, None)]
    x = items[0]
    assert x["finalUntil"] == 60.0 and x["bytes"] > 0 and not x["expired"] and x["eta"] is None
    assert "watchedAt" not in x


async def test_set_speakers_reruns(eng):
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.until(done(VID_A))
        n = len(w.msgs)
        await w.send(type="set_speakers", videoId=VID_A, speakers=1)
        await w.until(done(VID_A), after=n)
        found = [m for m in w.msgs[n:] if m["type"] == "speakers_found"]
    assert eng.backend.diarizer.calls == 2 and [s["id"] for s in found[-1]["speakers"]] == ["S1"]
    doc = job_json(eng, VID_A)
    assert doc["settings"]["speakers"] == 1
    manifest = json.loads((eng.cache_dir / VID_A / "render" / "manifest.json").read_text())
    assert {x["speaker"] for x in manifest["lines"]} == {"S1"}


def crashed(e, vid: str, stage: str, attempts: int, updated: float) -> None:
    """Leave `vid`'s job.json as an engine killed during `stage` leaves it."""
    path = e.cache_dir / vid / "render" / "job.json"
    doc = json.loads(path.read_text())
    doc.update(status="running", stage=stage, updatedAt=updated)
    doc["stages"][stage].update(state="running", attempts=attempts)
    path.write_text(json.dumps(doc))


def set_status(e, vid: str, **fields) -> None:
    path = e.cache_dir / vid / "render" / "job.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


async def run_job(e, url: str, **kw) -> None:
    assert await RenderJob(e.backend, DemoResolver(), e.cache_dir, url, RenderSettings(**kw)).run() == "done"


async def test_stale_running_jobs_are_interrupted_and_continue_at_startup(eng):
    await run_job(eng, URL_A)
    await run_job(eng, URL_B)
    crashed(eng, VID_A, "voice_lines", 1, time.time() - 100)
    crashed(eng, VID_B, "voice_lines", 1, time.time())  # the latest: first
    (eng.cache_dir / VID_B / "render" / "diarization.json").unlink()  # its speakers too, killed there first
    doc = job_json(eng, VID_B)
    doc["stages"]["speakers"].update(state="running", attempts=1)
    (eng.cache_dir / VID_B / "render" / "job.json").write_text(json.dumps(doc))
    at_launch, launch = [], eng._launch

    def launched(job):
        at_launch.append({vid: job_json(eng, vid)["status"] for vid in (VID_A, VID_B)})
        launch(job)

    eng._launch = launched
    await eng.start()
    assert eng.job is not None and eng.job.video_id == VID_B and waiting(eng) == [VID_A]
    await until(lambda: len(at_launch) == 2)  # both found interrupted: the latest continues first, then the other
    assert at_launch == [{VID_A: "interrupted", VID_B: "interrupted"}, {VID_A: "interrupted", VID_B: "done"}]
    doc = job_json(eng, VID_B)
    assert doc["stages"]["speakers"]["interrupted"] == 1 and eng.backend.diarizer.step == COARSE_STEP
    await until(lambda: eng.job is None)
    assert job_json(eng, VID_A)["status"] == "done"


async def test_startup_queues_interrupted_jobs_first_then_the_queued_in_their_order(eng):
    for url in (URL_A, URL_B, URL_C, URL_D):
        await run_job(eng, url)
    now = time.time()
    set_status(eng, VID_A, status="interrupted", updatedAt=now - 100)  # a quit
    crashed(eng, VID_B, "voice_lines", 1, now)                          # a crash, the latest
    set_status(eng, VID_C, status="queued", queuedAt=now - 10)
    set_status(eng, VID_D, status="queued", queuedAt=now - 50)          # queued before C
    eng._launch = lambda job: eng.queue.insert(0, (job.video_id, None))  # (look, don't run)
    await eng.start()
    assert waiting(eng) == [VID_B, VID_A, VID_D, VID_C]
    assert job_json(eng, VID_B)["status"] == "interrupted"


async def test_retention_at_startup_never_expires_a_job_in_the_queue(eng, monkeypatch):
    """The jobs startup queues are about to run, the interrupted ones too, whose job.json still says `interrupted`: the
    retention that runs once the first has started expires none of them, even over the cache cap; another job, yes."""
    from maata_engine import server

    for url in (URL_A, URL_B, URL_C):
        await run_job(eng, url)
    now = time.time()
    set_status(eng, VID_A, status="interrupted", updatedAt=now - 10)   # first in line
    set_status(eng, VID_B, status="interrupted", updatedAt=now - 100)  # second
    set_status(eng, VID_C, updatedAt=now - 200)                         # done
    monkeypatch.setattr(server, "retain", functools.partial(server.retain, cap=0))
    eng._launch = lambda job: setattr(eng, "job", job)  # (look, don't run)
    await eng.start()
    assert eng.job.video_id == VID_A and waiting(eng) == [VID_B]
    assert {vid: job_json(eng, vid)["expired"] for vid in (VID_A, VID_B, VID_C)} == \
           {VID_A: False, VID_B: False, VID_C: True}
    assert (eng.cache_dir / VID_B / "render" / "pcm").is_dir() and not (eng.cache_dir / VID_C / "render" / "pcm").exists()


async def test_the_crash_loop_guard_fails_a_job_whose_stage_never_finishes(eng):
    await run_job(eng, URL_A)
    crashed(eng, VID_A, "voice_lines", 3, time.time())
    await eng.start()
    doc = job_json(eng, VID_A)
    assert eng.job is None and doc["status"] == "failed" and "Telugu speech" in doc["error"]
    async with window(eng) as w:  # the user can still resume it
        await w.send(type="resume", videoId=VID_A)
        await w.until(done(VID_A))


def engine_on(cache) -> Engine:
    """Maata launched again on the same cache."""
    e = Engine("mock", cache.parent / "models", cache, None, "tok", demo=True)
    e.backend.tts = SlowTTS()
    return e


async def test_a_graceful_quit_interrupts_the_running_stage_without_counting_its_start(eng):
    """§4: on the lifeline's EOF the engine stops gracefully: the running stage goes back to `todo` with its start not
    counted (`attempts` as before the run), the job is written `interrupted`, and the engine stops."""
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        await w.send(type="prepare", url=URL_B)
        await w.until(status(VID_B, "queued"))
        await stage_running(eng, VID_A)
        assert eng.job.doc["stages"]["voice_lines"]["attempts"] == 1
        await asyncio.wait_for(eng.shutdown(), 10)  # what the lifeline thread's call_soon_threadsafe(engine.quit) runs
    assert eng.stopped.is_set() and eng.job is None and eng._job_task.done()
    doc = job_json(eng, VID_A)
    assert doc["status"] == "interrupted" and doc["elapsed"] > 0
    st = doc["stages"]["voice_lines"]
    assert (st["state"], st["attempts"], st["interrupted"]) == ("todo", 0, 0)
    assert doc["stages"]["speakers"]["state"] == "done" and doc["stages"]["speakers"]["interrupted"] == 0
    assert job_json(eng, VID_B)["status"] == "queued" and waiting(eng) == [VID_B]  # nothing started on the way out
    await eng.start()  # (a quitting engine starts nothing)
    assert eng.job is None


async def gate_shut_in_dub_loop(e, vid: str) -> GateTTS:
    """Run `vid` on `e` until the dub loop's synthesis is held at a shut gate."""
    tts = e.backend.tts = GateTTS()
    await e._submit(vid, RenderSettings())
    await stage_running(e, vid)
    tts.open.clear()
    await until(tts.waiting.is_set)
    return tts


async def test_a_quit_is_on_disk_before_a_long_model_call_returns(eng):
    """A model call can't be stopped, and may outlast QUIT_WAIT, after which the lifeline's fallback exits at once: the
    graceful state (the stage `todo` with its start not counted, the job `interrupted`) is on disk as soon as the quit
    begins, not once the call has returned, so the fallback never leaves a job that looks crashed (§4)."""
    tts = await gate_shut_in_dub_loop(eng, VID_A)
    quitting = asyncio.create_task(eng.shutdown())
    await until(lambda: job_json(eng, VID_A)["status"] != "running")
    doc = job_json(eng, VID_A)  # what the fallback's os._exit would leave
    st = doc["stages"]["voice_lines"]
    assert doc["status"] == "interrupted" and (st["state"], st["attempts"], st["interrupted"]) == ("todo", 0, 0)
    await asyncio.sleep(0.2)
    assert not quitting.done() and eng.job is not None  # the call is still being waited out
    tts.open.set()
    await asyncio.wait_for(quitting, 10)
    after = job_json(eng, VID_A)
    st = after["stages"]["voice_lines"]
    assert after["status"] == "interrupted" and (st["state"], st["attempts"]) == ("todo", 0)  # not taken off twice
    assert after["elapsed"] >= doc["elapsed"] > 0


async def test_a_quit_while_a_pause_finishes_leaves_the_job_paused(eng):
    """§2.1, §4: a job the user paused stays paused. Quitting before its pause has finished (a model call returning, or
    Claude calls finishing into the cache) writes it `paused`, and the next launch doesn't run it; but with a re-run
    of it waiting for that pause (set_speakers here), it is `interrupted` with the re-run's settings, and continues."""
    tts = await gate_shut_in_dub_loop(eng, VID_A)
    await eng._pause(VID_A)
    assert eng.job is not None and eng.job.stopping  # its run hasn't returned
    quitting = asyncio.create_task(eng.shutdown())
    await until(lambda: job_json(eng, VID_A)["status"] != "running")
    doc = job_json(eng, VID_A)
    assert doc["status"] == "paused" and doc["stages"]["voice_lines"]["state"] == "todo"
    tts.open.set()
    await asyncio.wait_for(quitting, 10)
    doc = job_json(eng, VID_A)
    assert doc["status"] == "paused" and doc["stages"]["voice_lines"]["attempts"] == 0
    e = engine_on(eng.cache_dir)
    await e.start()
    assert e.job is None and waiting(e) == []
    tts = await gate_shut_in_dub_loop(e, VID_A)  # (resumed)
    await e._submit(VID_A, RenderSettings(speakers=1))  # a re-run of it: it pauses, and the re-run waits for it
    assert e.job.stopping and waiting(e) == [VID_A]
    quitting = asyncio.create_task(e.shutdown())
    await until(lambda: job_json(e, VID_A)["status"] != "running")
    assert job_json(e, VID_A)["status"] == "interrupted" and job_json(e, VID_A)["settings"]["speakers"] == 1
    tts.open.set()
    await asyncio.wait_for(quitting, 10)
    e = engine_on(eng.cache_dir)
    e._launch = lambda job: setattr(e, "job", job)  # (look, don't run)
    await e.start()
    assert e.job is not None and e.job.video_id == VID_A and e.job.settings.speakers == 1


async def test_three_graceful_quits_never_trip_the_crash_loop_guard_but_three_crashes_do(tmp_path):
    cache = tmp_path / "quits"
    e = engine_on(cache)
    await e._submit(URL_A, RenderSettings())
    for n in range(CRASH_STARTS):
        await stage_running(e, VID_A, attempts=1)  # a quit's start never counts: the first, every time
        await asyncio.wait_for(e.shutdown(), 10)
        assert job_json(e, VID_A)["status"] == "interrupted"
        e = engine_on(cache)
        await e.start()
    assert e.job is not None and e.job.video_id == VID_A  # still continued
    await stage_running(e, VID_A)
    await stop(e)
    # The same, killed instead: job.json as the kill left it, the stage still `running`.
    cache = tmp_path / "crashes"
    e = engine_on(cache)
    await e._submit(URL_A, RenderSettings())
    for n in range(CRASH_STARTS):
        await stage_running(e, VID_A, attempts=n + 1)
        left_by_kill = (cache / VID_A / "render" / "job.json").read_bytes()
        e._job_task.cancel()  # (the test's own clean-up; a kill runs no code)
        await asyncio.wait([e._job_task])
        (cache / VID_A / "render" / "job.json").write_bytes(left_by_kill)
        e = engine_on(cache)
        await e.start()
    doc = job_json(e, VID_A)
    assert e.job is None and doc["status"] == "failed" and f"stopped {CRASH_STARTS} times" in doc["error"]


async def test_the_lifeline_thread_asks_the_loop_to_stop(eng, monkeypatch):
    """The stdin lifeline (main's thread) hands the stop to the loop with call_soon_threadsafe, and arms the exit
    fallback; a closed loop exits at once."""
    import io
    import threading

    from maata_engine import server

    timers, exits = [], []

    class Timer:
        def __init__(self, interval, fn, args):
            timers.append((interval, fn, args))

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server.os, "_exit", exits.append)
    monkeypatch.setattr(server.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"")))
    t = threading.Thread(target=server._stop_when_stdin_closes, args=(asyncio.get_running_loop(), eng))
    t.start()
    t.join(5)
    await until(lambda: eng.stopped.is_set())
    assert timers == [(server.QUIT_WAIT, exits.append, (0,))] and exits == []
    loop = asyncio.new_event_loop()
    loop.close()
    server._stop_when_stdin_closes(loop, eng)
    assert exits == [0]


async def test_remove_refuses_what_isnt_a_video_of_the_cache_and_a_running_job(eng, tmp_path):
    await run_job(eng, URL_A)
    eng.backend.tts = SlowTTS()
    a = eng.cache_dir / VID_A
    for name in ("audio.m4a", "audio.m4a.part", "video.mp4", "thumb.webp", "lines.jsonl", "briefs.jsonl"):
        (a / name).write_bytes(b"x")
    (a / "pcm").mkdir()  # a streaming session's line PCM (no job.json covers it once the render goes)
    (a / "pcm" / "3.npy").write_bytes(b"x")
    outside = tmp_path / "outside" / VID_C
    (outside / "render").mkdir(parents=True)
    (outside / "render" / "job.json").write_text("{}")
    (eng.cache_dir / "demo0000009").symlink_to(outside, target_is_directory=True)  # a link out of the cache
    async with window(eng) as w:
        for bad in ("", "..", ".", "../" + VID_A, f"{VID_A}/render", "/tmp", "demo0000009", VID_A + "x"):
            n = len(w.msgs)
            await w.send(type="remove", videoId=bad, forget=True)
            err = await w.until(lambda m: m["type"] == "error", after=n)
            assert "no dub" in err["message"], bad
        assert (outside / "render" / "job.json").exists() and (a / "render" / "job.json").exists()
        await w.send(type="prepare", url=URL_B)
        await w.until(status(VID_B, "running"))
        n = len(w.msgs)
        await w.send(type="remove", videoId=VID_B)
        err = await w.until(lambda m: m["type"] == "error", after=n)
        assert err["message"] == "Pause it first." and (eng.cache_dir / VID_B / "render" / "job.json").exists()
        await w.send(type="prepare", url=URL_C)
        await w.until(status(VID_C, "queued"))
        n = len(w.msgs)
        await w.send(type="remove", videoId=VID_C)  # queued: it leaves the queue first, then goes
        await w.until(lambda m: m["type"] == "renders", after=n)
        assert waiting(eng) == [] and not (eng.cache_dir / VID_C / "render").exists()
        n = len(w.msgs)
        await w.send(type="remove", videoId=VID_A)
        items = (await w.until(lambda m: m["type"] == "renders", after=n))["items"]
        assert VID_A not in {x["videoId"] for x in items}
        # Its render, media and streaming PCM went; its translations and trace stayed, for a dub of it again.
        assert sorted(p.name for p in a.iterdir()) == ["briefs.jsonl", "lines.jsonl", "units.jsonl"]
        n = len(w.msgs)
        await w.send(type="remove", videoId=VID_A, forget=True)
        await w.until(lambda m: m["type"] == "renders", after=n)
        assert not a.exists()
    assert (outside / "render" / "job.json").exists()


async def test_voice_sample_is_dub_audio(eng):
    await run_job(eng, URL_A)
    async with window(eng) as w:
        await w.send(type="voice_sample", videoId="../" + VID_A, speaker="S2")  # not a video of the cache: nothing
        await w.send(type="voice_sample", videoId=VID_A, speaker="S2")
        t0 = time.monotonic()
        while not w.frames:
            assert time.monotonic() - t0 < 10
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
    assert len(w.frames) == 1
    uid, sr, audio = w.frames[0]
    sample = np.load(eng.cache_dir / VID_A / "render" / "voices" / "S2.npy").astype(np.float32)
    assert uid == SAMPLE_ID + 1 and sr == 24000 and np.array_equal(audio, sample)  # S2's voice saying a fixed sentence
    src = decode_audio(eng.cache_dir / VID_A / "demo.mp4", 24_000)  # (the analysis audio went with the MP4 written)
    assert len(audio) and not np.allclose(audio[:1000], src[:1000])  # dub audio, not the source


async def test_retention_expires_voice_data_and_keeps_translations(eng):
    await run_job(eng, URL_A)
    await run_job(eng, URL_B)
    await run_job(eng, URL_C)
    cache = eng.cache_dir
    for vid, age in ((VID_A, 20), (VID_B, 1), (VID_C, 30)):
        set_status(eng, vid, updatedAt=time.time() - age * 86400)
        (cache / vid / "audio.m4a").write_bytes(b"x" * 1000)
    set_status(eng, VID_C, status="queued")  # about to run: never expired, however old
    assert retain(cache, (), time.time()) == [VID_A]  # not updated for 20 days
    a, b = cache / VID_A, cache / VID_B
    for name in ("voices", "voices.json", "takes", "takes.jsonl", "pcm", "manifest.json"):
        assert not (a / "render" / name).exists() and (b / "render" / name).exists()
    assert not (a / "render" / "audio16k.f32").exists()  # (B's went when its MP4 was written)
    assert not (a / "audio.m4a").exists() and (b / "audio.m4a").exists()
    for name in ("lines.jsonl", "briefs.jsonl", "render/transcript.jsonl", "render/diarization.json"):
        assert (a / name).exists()  # translations and the transcript are kept
    doc = job_json(eng, VID_A)
    assert doc["expired"] and doc["finalUntil"] is None and doc["stages"]["voice_lines"]["state"] == "todo"
    # Over the cap: the render updated least recently goes first; the running one and a queued one never.
    assert retain(cache, {VID_B}, time.time(), cap=0) == []
    assert retain(cache, (), time.time(), cap=0) == [VID_B]
    assert not job_json(eng, VID_C)["expired"] and (cache / VID_C / "render" / "pcm").exists()
    set_status(eng, VID_C, status="done")
    async with window(eng) as w:
        await w.send(type="renders")
        items = (await w.until(lambda m: m["type"] == "renders"))["items"]
        assert {x["videoId"]: x["expired"] for x in items} == {VID_A: True, VID_B: True, VID_C: False}
        assert [x["videoId"] for x in items] == [VID_B, VID_A, VID_C]  # by updatedAt, newest first
        await w.send(type="prepare", url=URL_A)  # prepared again: voiced again, no longer expired
        await w.until(done(VID_A))
    doc = job_json(eng, VID_A)
    assert not doc["expired"] and (a / "render" / "manifest.json").exists()


# ---- the MP4, settings, open_output, notifications, retention (OFFLINE-RENDER §2.17, §4) ------------------------------
from pathlib import Path  # noqa: E402

from maata_engine import export as mp4  # noqa: E402
from maata_engine import notify  # noqa: E402
from maata_engine import settings as config  # noqa: E402
from maata_engine.server import Engine as _Engine  # noqa: E402


async def test_prepare_to_done_saves_the_mp4_and_says_where(eng):
    async with window(eng) as w:
        hello = w.of("hello")[0]
        assert hello["settings"] == {"outputDir": str(config.default_output_dir())}
        await w.send(type="prepare", url=URL_A)
        last = await w.until(done(VID_A))
        out = last["output"]
        assert out["kind"] == "whole" and Path(out["path"]) == config.default_output_dir() / "Demo video (Telugu).mp4"
        assert out["bytes"] == Path(out["path"]).stat().st_size and not out["missing"] and "inputs" not in out
        assert [s["key"] for s in last["stages"]][-2:] == ["video", "export"]
        await w.send(type="renders")
        items = (await w.until(lambda m: m["type"] == "renders"))["items"]
        assert items[0]["output"] == out


async def test_open_output_opens_only_the_jobs_own_file(eng, tmp_path):
    ran: list[list[str]] = []
    eng.opener = ran.append
    async with window(eng) as w:
        n = len(w.msgs)
        await w.send(type="open_output", videoId=VID_A, reveal=False)  # no job: nothing to open
        assert "no saved video" in (await w.until(lambda m: m["type"] == "error", after=n))["message"]
        await w.send(type="prepare", url=URL_A)
        path = Path((await w.until(done(VID_A)))["output"]["path"])
        for bad in ("../" + VID_A, "/tmp", ""):
            n = len(w.msgs)
            await w.send(type="open_output", videoId=bad, reveal=True, path="/etc/passwd")
            await w.until(lambda m: m["type"] == "error", after=n)
        await w.send(type="open_output", videoId=VID_A, reveal=False, path="/etc/passwd")  # a path sent is ignored
        await until(lambda: len(ran) == 1)  # (each request is a task of its own: one at a time, in order)
        await w.send(type="open_output", videoId=VID_A, reveal=True)
        await until(lambda: len(ran) == 2)
        path.unlink()  # moved away by the user: its folder
        await w.send(type="open_output", videoId=VID_A, reveal=True)
        await until(lambda: len(ran) == 3)
    tail = [str(path), str(path), str(path.parent)]
    assert [r[-1] for r in ran] == tail and all("/etc/passwd" not in r for r in ran)
    if sys.platform == "darwin":
        assert ran == [["/usr/bin/open", str(path)], ["/usr/bin/open", "-R", str(path)], ["/usr/bin/open", str(path.parent)]]


async def test_settings_are_checked_saved_broadcast_and_kept_across_restarts(eng, tmp_path):
    folder = tmp_path / "Videos" / "Dubs"
    async with window(eng) as w, window(eng) as other:
        n = len(w.msgs)
        await w.send(type="settings", outputDir="relative/folder")
        assert "full path" in (await w.until(lambda m: m["type"] == "error", after=n))["message"]
        blocked = tmp_path / "file"
        blocked.write_text("x")
        n = len(w.msgs)
        await w.send(type="settings", outputDir=str(blocked / "sub"))  # can't be made: a file is in the way
        assert "can't save videos" in (await w.until(lambda m: m["type"] == "error", after=n))["message"]
        await w.send(type="settings", outputDir=str(folder))
        for x in (w, other):  # to every window
            assert (await x.until(lambda m: m["type"] == "settings"))["outputDir"] == str(folder)
        assert folder.is_dir()
        await w.send(type="prepare", url=URL_A)
        path = Path((await w.until(done(VID_A)))["output"]["path"])
    assert path.parent == folder
    again = _Engine("mock", tmp_path / "models", eng.cache_dir, None, "tok", demo=True, config_dir=eng.config_dir)
    assert again.settings == {"outputDir": str(folder)}  # in the config folder, not the cache
    assert json.loads((eng.config_dir / "settings.json").read_text()) == {"outputDir": str(folder)}


async def test_notifications_fire_for_speakers_done_failed_holds_and_a_crash(eng, monkeypatch):
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(notify, "post", lambda title, body: posted.append((title, body)))
    async with window(eng) as w:  # the demo engine never notifies
        await w.send(type="prepare", url=URL_A)
        await w.until(done(VID_A))
    assert posted == []
    eng.notifications = True  # as a real backend's engine
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_B)
        await w.until(done(VID_B))
    found = [b for _, b in posted if "speakers" in b]
    assert len(found) == 1 and found[0].startswith("Maata found 2 speakers in Demo video. Check them before the Telugu "
                                                    "speech starts, in about ")
    assert any(b == "Demo video is dubbed: Demo video (Telugu) (2).mp4 is saved." for _, b in posted)
    posted.clear()
    async with window(eng) as w:  # served from disk: the speakers aren't news
        await w.send(type="resume", videoId=VID_B)
        await w.until(lambda m: done(VID_B)(m), after=len(w.msgs))
    assert not [b for _, b in posted if "speakers" in b]
    # the first Claude hold of a run, once
    posted.clear()
    for _ in range(2):
        await eng._job_event({"type": "claude_error", "videoId": VID_C, "kind": "usage_limit", "message": "Limit reached.",
                              "resetsAt": None})
    assert posted == [("Maata", f"Claude is holding back {VID_C}: Limit reached.")]
    # a failure
    posted.clear()
    eng.backend.tts = BrokenSynth()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_D, speakers=1)
        await w.until(status(VID_D, "failed"))
    assert [b for _, b in posted if b.startswith("Dubbing Demo video stopped: ")]
    # a job a crash left running continues at the next start, and says so
    posted.clear()
    crashed(eng, VID_A, "voice_lines", 1, time.time())
    eng.backend.tts = MockTTS()
    await eng.start()
    assert posted[0] == ("Maata", "Maata restarted after a problem and is continuing Demo video.")
    await stop(eng)


class BrokenSynth(MockTTS):
    def prepare_voice(self, reference, sample_rate):
        raise OSError(28, "No space left on device")


def test_a_notification_passes_its_texts_as_arguments_never_inside_the_script(monkeypatch):
    ran: list[list[str]] = []
    monkeypatch.setattr(notify, "RUNNER", ran.append)
    monkeypatch.setattr(notify.sys, "platform", "darwin")
    monkeypatch.setattr(notify.os.path, "exists", lambda p: p == notify.OSASCRIPT)
    notify.post('Maata', 'A "quoted" title" & do shell script "rm -rf ~"')
    assert ran == [["/usr/bin/osascript", "-e", "on run argv", "-e",
                    "display notification (item 2 of argv) with title (item 1 of argv)", "-e", "end run", "Maata",
                    'A "quoted" title" & do shell script "rm -rf ~"']]
    monkeypatch.setattr(notify.sys, "platform", "linux")
    notify.post("Maata", "x")
    assert len(ran) == 1  # osascript only on macOS


async def test_a_paused_job_holding_a_big_video_survives_another_job_ending(eng, monkeypatch):
    """§4: the 5 GB cap trims only done whole-video jobs (least recently updated first) and expired leftovers: a paused
    job waits to be continued, and keeps its takes, its PCM and its 6 GB video, whatever the cache holds."""
    await run_job(eng, URL_A)
    set_status(eng, VID_A, status="paused", updatedAt=time.time() - 3600)
    big = eng.cache_dir / VID_A / "video.mp4"
    with big.open("wb") as f:
        f.truncate(6_000_000_000)  # sparse: 6 GB to the cap's count, nothing on the disk
    eng.backend.tts = SlowTTS(0.0)
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_B)
        await w.until(done(VID_B))
        await until(lambda: eng.job is None)
        await asyncio.sleep(0.2)  # retention runs once the job has ended
    a = job_json(eng, VID_A)
    assert not a["expired"] and big.stat().st_size == 6_000_000_000
    assert (eng.cache_dir / VID_A / "render" / "pcm").is_dir() and (eng.cache_dir / VID_A / "render" / "takes").is_dir()
    assert job_json(eng, VID_B)["expired"]  # the done whole job went instead (still over the cap)
    assert Path(job_json(eng, VID_B)["output"]["path"]).is_file()  # never the output folder


async def test_remove_deletes_the_jobs_unfinished_mp4_too(eng, tmp_path):
    await run_job(eng, URL_A)
    part = config.default_output_dir() / ".Demo video (Telugu).part.mp4"
    part.write_bytes(b"half a file")
    done_mp4 = Path(job_json(eng, VID_A)["output"]["path"])
    set_status(eng, VID_A, part=str(part))
    async with window(eng) as w:
        n = len(w.msgs)
        await w.send(type="remove", videoId=VID_A)
        await w.until(lambda m: m["type"] == "renders", after=n)
    assert not part.exists() and done_mp4.is_file()  # its finished MP4 always stays


def test_remove_never_deletes_the_running_jobs_part_of_the_same_name(tmp_path):
    from maata_engine import server

    # a job paused mid-export recorded .X (Telugu).part.mp4; another of the same title is writing that name now
    video_dir = tmp_path / "cache" / VID_A
    (video_dir / "render").mkdir(parents=True)
    part = tmp_path / "out" / ".Demo video (Telugu).part.mp4"
    part.parent.mkdir()
    part.write_bytes(b"the running export")
    (video_dir / "render" / "job.json").write_text(json.dumps({"part": str(part)}))
    server._delete(video_dir, False, running=tmp_path / "out" / "." / part.name)
    assert part.read_bytes() == b"the running export" and not (video_dir / "render").exists()
    (video_dir / "render").mkdir()
    (video_dir / "render" / "job.json").write_text(json.dumps({"part": str(part)}))
    server._delete(video_dir, False)
    assert not part.exists()


# ---- what the job UI reads (OFFLINE-RENDER §5) -----------------------------------------------------------------------
def test_the_page_may_load_no_youtube_frame_or_script():
    """Nothing plays in the app (§6): the CSP keeps 'self' and i.ytimg.com images (New dub's thumbnail), and allows no
    YouTube frame or script."""
    from maata_engine.server import _CSP

    rules = dict(r.strip().split(" ", 1) for r in _CSP.split(";"))
    assert rules["script-src"] == "'self'" and rules["frame-src"] == "'none'"
    assert "https://i.ytimg.com" in rules["img-src"] and "'self'" in rules["img-src"]
    assert "youtube.com" not in _CSP and "s.ytimg.com" not in _CSP and "youtube-nocookie" not in _CSP


def _get(e, path: str) -> tuple[int, bytes, str]:
    import urllib.error
    import urllib.request

    url = e.url.replace("ws://", "http://").split("/ws", 1)[0] + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as err:
        return err.code, b"", ""


async def test_renders_items_carry_the_cached_thumbnail_behind_the_token(eng):
    """The Library shows each job's cached thumbnail from the loopback (§5): the item's `thumb` is a token-guarded URL of
    this engine, only for a thumbnail job.json records in the job's own folder; none for the demo (it has none)."""
    await run_job(eng, URL_A)
    await run_job(eng, URL_B)
    jpeg = b"\xff\xd8\xff\xe0fake jpeg"
    (eng.cache_dir / VID_A / "thumb.jpg").write_bytes(jpeg)
    set_status(eng, VID_A, source={**job_json(eng, VID_A)["source"], "thumb": "thumb.jpg"})
    (eng.cache_dir / "secret.jpg").write_bytes(b"not a job's")
    set_status(eng, VID_B, source={**job_json(eng, VID_B)["source"], "thumb": "../secret.jpg"})
    async with window(eng) as w:
        await w.send(type="renders")
        items = {x["videoId"]: x for x in (await w.until(lambda m: m["type"] == "renders"))["items"]}
    assert items[VID_A]["thumb"] == f"/thumb/{VID_A}?token=tok" and items[VID_B]["thumb"] is None
    assert await asyncio.to_thread(_get, eng, items[VID_A]["thumb"]) == (200, jpeg, "image/jpeg")
    assert (await asyncio.to_thread(_get, eng, f"/thumb/{VID_A}"))[0] == 403  # no token
    assert (await asyncio.to_thread(_get, eng, f"/thumb/{VID_A}?token=nope"))[0] == 403
    for bad in (VID_B, "..", "../" + VID_A, VID_C):  # a name out of its folder, not an id, no job
        assert (await asyncio.to_thread(_get, eng, f"/thumb/{bad}?token=tok"))[0] == 404


async def test_renders_items_carry_their_options_and_the_video_both_estimates(eng):
    """New dub starts from the last job's options and Continue keeps a job's (§5): the items carry `settings` and
    `createdAt`. The `video` estimate has the first 15 minutes' too, so switching the length asks YouTube nothing."""
    await run_job(eng, URL_A, style="formal", speed_cap=1.1, stop_at=60.0)
    async with window(eng) as w:
        await w.send(type="renders")
        item = (await w.until(lambda m: m["type"] == "renders"))["items"][0]
        await w.send(type="inspect", url="https://youtu.be/" + "x" * 11)
        video = await w.until(lambda m: m["type"] == "video")
    assert item["settings"] == {"speakers": None, "style": "formal", "stopAt": 60.0, "speedCap": 1.1,
                                "ttsScript": "telugu", "presets": []}
    assert item["createdAt"] == job_json(eng, VID_A)["createdAt"] and item["createdAt"] > 0
    assert item["error"] is None and item["slept"] is None
    # what the job view shows of a job that isn't running (it sends no `render`): its stages, elapsed time and report
    doc = job_json(eng, VID_A)
    assert [(s["key"], s["label"], s["state"], s["done"], s["eta"]) for s in item["stages"]] == \
        [(k, label, doc["stages"][k]["state"], doc["stages"][k]["done"], None) for k, label, _ in STAGES]
    assert item["elapsed"] == doc["elapsed"] > 0 and item["report"] == doc["report"] and item["coverage"] == doc["coverage"]
    # continued to the whole video but not run that far (queued, or paused before): the stages the whole video adds to
    # show as still to do, while job.json keeps them done for the preview's range (their `span`, for the ETA)
    doc["settings"]["stopAt"] = None
    (eng.cache_dir / VID_A / "render" / "job.json").write_text(json.dumps(doc))
    async with window(eng) as w:
        await w.send(type="renders")
        item = (await w.until(lambda m: m["type"] == "renders"))["items"][0]
    assert {s["key"] for s in item["stages"] if s["state"] == "todo"} == {"separate", "translate", "voice_lines",
                                                                         "finish", "export"}
    assert {s["state"] for s in item["stages"] if s["state"] != "todo"} == {"done"}
    est = video["estimate"]
    assert est["preview"] == render_mod.estimate(180.0, 900.0, separates=eng.backend.separator is not None)
    assert {k: est[k] for k in ("seconds", "claudeCalls")} == render_mod.estimate(180.0)
    assert est["ahead"] is None


async def test_a_done_job_reports_its_flagged_lines_and_overdraft(eng):
    """The Done card's figures (§5) that only the manifest had: the final lines, those flagged long or unreviewed, and
    the overdraft seconds, in job.json and the render snapshot."""
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        last = await w.until(done(VID_A))
    manifest = json.loads((eng.cache_dir / VID_A / "render" / "manifest.json").read_text())
    report = last["report"]
    assert report == job_json(eng, VID_A)["report"]
    assert report["lines"] == len(manifest["lines"]) > 0
    assert report["long"] == sum("long" in x["flags"] for x in manifest["lines"])
    assert report["unreviewed"] == sum("unreviewed" in x["flags"] for x in manifest["lines"])
    assert report["overdraft"] == manifest["stats"]["overdraft_s"]
    assert report["carried"] == manifest["stats"]["overdraft_carried_s"]
    finish = [k for k, _, _ in STAGES].index("finish")
    assert all(m["report"] is None for m in w.of("render") if m["stages"][finish]["state"] != "done")


def _same_check(a: dict, b: dict) -> None:
    """Two `speakers_found` of one diarization: the same speakers (activity to rounding), merges, mode and bounds."""
    keys = ("videoId", "mode", "bounds", "merged")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    strip = lambda f: [{k: v for k, v in s.items() if k != "activity"} for s in f["speakers"]]  # noqa: E731
    assert strip(a) == strip(b)
    for x, y in zip(a["speakers"], b["speakers"]):
        assert len(x["activity"]) == len(y["activity"]) == 120 and np.allclose(x["activity"], y["activity"], atol=0.01)


async def test_a_window_connecting_after_the_speakers_stage_still_gets_the_speaker_check(eng):
    """A job broadcasts `speakers_found` only while its speakers stage runs; a window that connects later (a reload, an
    engine restarted, a relaunch) gets it right after `hello`, for the running job and for one paused or done (§5)."""
    eng.backend.tts = SlowTTS()
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A)
        live = await w.until(lambda m: m["type"] == "speakers_found")
        await stage_running(eng, VID_A, "voice_lines")
        async with window(eng) as late:
            found = await late.until(lambda m: m["type"] == "speakers_found" and m["videoId"] == VID_A)
            assert late.msgs.index(found) > late.msgs.index(late.of("hello")[0])
        assert live["fresh"] and not found["fresh"] and found["freeFor"] == 0  # (the stages before the speech are done)
        _same_check(live, found)
        await w.send(type="pause", videoId=VID_A)
        await w.until(status(VID_A, "paused"))
        await until(lambda: eng.job is None)
    async with window(eng) as later:
        _same_check(live, await later.until(lambda m: m["type"] == "speakers_found" and m["videoId"] == VID_A))


async def test_no_speaker_check_from_disk_until_the_speakers_stage_is_done_for_the_jobs_count(eng):
    await run_job(eng, URL_A, stop_at=60.0)
    video_dir = eng.cache_dir / VID_A
    assert render_mod.found_on_disk(video_dir) is not None
    doc = job_json(eng, VID_A)
    hint = {**doc, "settings": {**doc["settings"], "speakers": 1}}  # a set_speakers waiting to run
    assert render_mod.found_on_disk(video_dir, hint) is None
    rerun = {**doc, "stages": {**doc["stages"], "speakers": {**doc["stages"]["speakers"], "state": "running"}}}
    assert render_mod.found_on_disk(video_dir, rerun) is None  # diarizing again
    (video_dir / "render" / "diarization.json").write_text("{")
    assert render_mod.found_on_disk(video_dir) is None
    async with window(eng) as w:
        await w.send(type="renders")
        await w.until(lambda m: m["type"] == "renders")
    assert not w.of("speakers_found")


async def test_inspect_answers_name_the_link_they_answer(eng):
    """The UI shows only the answer to the link pasted last (§5): `video` and inspect's `error` carry the `url` asked."""
    async with window(eng) as w:
        await w.send(type="inspect", url=URL_C)
        assert (await w.until(lambda m: m["type"] == "video"))["url"] == URL_C
        await w.send(type="inspect", url="not a link")
        err = await w.until(lambda m: m["type"] == "error")
    assert err["url"] == "not a link" and not err["retryable"]


async def test_the_output_says_whether_it_has_the_background_sound(eng):
    """The Done card says "over the original music and sounds" only when the MP4 has the bed (§5)."""
    async with window(eng) as w:
        await w.send(type="prepare", url=URL_A, stopAt=60)
        out = (await w.until(done(VID_A)))["output"]
        assert out["bed"] is True and out["warning"] is None
        eng.backend.separator = None
        await w.send(type="prepare", url=URL_B, stopAt=60)
        out = (await w.until(done(VID_B)))["output"]
    assert out["bed"] is False and out["warning"] == mp4.NO_BED
