"""The GPU scheduler (ARCHITECTURE §5.3): one owner at a time, the most urgent waiter next, first come first served within
a priority, no starvation, and cancellation that never loses or leaks the GPU, nor lets it go while a cancelled model
call is still running (`Dubber._on_gpu`)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from maata_engine.backends.mock import make_mock_backend
from maata_engine.dubber import Dubber, VoiceCost, VoiceState
from maata_engine.gpu import BACKGROUND, FRONTIER, URGENT, VOICE, GpuScheduler
from maata_engine.render import RenderJob
from maata_engine.resolve import DemoResolver
from maata_engine.types import VoiceKind


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def _queue(gpu: GpuScheduler, asks: list[tuple[str, int]], order: list[str]) -> list[asyncio.Task]:
    """Hold the GPU and queue `asks` behind the holder in that order; once it lets go, the order they get it in lands
    in `order`."""
    await gpu.acquire(VOICE)
    tasks = []
    for name, priority in asks:
        tasks.append(asyncio.create_task(_job(gpu, name, priority, order)))
        await asyncio.sleep(0)  # queued in this order
    return tasks


async def test_the_most_urgent_waiter_goes_next_and_equal_ones_in_turn():
    gpu, order = GpuScheduler(), []
    tasks = await _queue(gpu, [("bg", BACKGROUND), ("voice-a", VOICE), ("frontier", FRONTIER), ("voice-b", VOICE),
                               ("urgent", URGENT)], order)
    assert gpu.waiting == 5
    gpu.release()
    await asyncio.gather(*tasks)
    assert order == ["urgent", "frontier", "voice-a", "voice-b", "bg"]
    assert not gpu.owned and gpu.waiting == 0


async def test_one_owner_at_a_time():
    gpu, inside, most = GpuScheduler(), [0], [0]

    async def job(priority: int) -> None:
        async with gpu.hold(priority):
            inside[0] += 1
            most[0] = max(most[0], inside[0])
            await asyncio.sleep(0.002)
            inside[0] -= 1

    await asyncio.gather(*(job(p) for p in [URGENT, FRONTIER, VOICE, BACKGROUND] * 5))
    assert most[0] == 1 and not gpu.owned


async def test_a_long_waiter_moves_up_a_priority_per_age_waited():
    clock = Clock()
    gpu, order = GpuScheduler(age=20.0, clock=clock), []
    tasks = await _queue(gpu, [("bg", BACKGROUND)], order)
    clock.t = 41.0  # two ages waited: background ranks as frontier now, ahead of a fresh voicer hold
    tasks.append(asyncio.create_task(_job(gpu, "voice", VOICE, order)))
    await asyncio.sleep(0)
    gpu.release()
    await asyncio.gather(*tasks)
    assert order == ["bg", "voice"]


async def test_background_work_never_starves_under_a_stream_of_urgent_holds():
    """Urgent holds of 10 s each, with another urgent one always waiting: the background waiter still gets its turn
    once it has waited three ages (it then ties with a fresh urgent waiter and was there first)."""
    clock = Clock()
    gpu, got = GpuScheduler(age=20.0, clock=clock), []
    await gpu.acquire(URGENT)
    bg = asyncio.create_task(_job(gpu, "bg", BACKGROUND, got))
    await asyncio.sleep(0)
    urgent_holds = 1
    while not got:
        clock.t += 10.0
        nxt = asyncio.create_task(gpu.acquire(URGENT))
        await asyncio.sleep(0)
        gpu.release()  # the running urgent hold ends
        for _ in range(3):
            await asyncio.sleep(0)
        await nxt      # the next urgent hold runs (straight away, or once the background one is done)
        urgent_holds += 1
        assert urgent_holds < 20
    gpu.release()
    await bg
    assert clock.t == 60.0 and not gpu.owned


async def _job(gpu: GpuScheduler, name: str, priority: int, order: list[str]) -> None:
    async with gpu.hold(priority):
        order.append(name)


async def test_a_cancelled_waiter_leaves_the_queue_and_the_rest_go_on():
    gpu, order = GpuScheduler(), []
    tasks = await _queue(gpu, [("a", VOICE), ("b", VOICE), ("c", VOICE)], order)
    tasks[1].cancel()
    await asyncio.sleep(0)
    assert gpu.waiting == 2
    gpu.release()
    await asyncio.gather(tasks[0], tasks[2])
    with pytest.raises(asyncio.CancelledError):
        await tasks[1]
    assert order == ["a", "c"] and not gpu.owned


async def test_a_waiter_cancelled_as_it_is_handed_the_gpu_hands_it_on():
    gpu, order = GpuScheduler(), []
    tasks = await _queue(gpu, [("a", URGENT), ("b", VOICE)], order)
    gpu.release()        # hands it to "a"...
    tasks[0].cancel()    # ...which is cancelled before it runs
    await asyncio.gather(*tasks, return_exceptions=True)
    assert order == ["b"] and not gpu.owned


async def test_a_holder_cancelled_mid_hold_lets_go():
    gpu, order = GpuScheduler(), []

    async def long_hold() -> None:
        async with gpu.hold(VOICE):
            await asyncio.sleep(10)

    holder = asyncio.create_task(long_hold())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(_job(gpu, "next", BACKGROUND, order))
    await asyncio.sleep(0)
    holder.cancel()
    await asyncio.gather(holder, return_exceptions=True)
    await waiter
    assert order == ["next"] and not gpu.owned


async def test_a_free_gpu_is_taken_at_once_whatever_the_priority():
    gpu = GpuScheduler()
    async with gpu.hold(BACKGROUND):
        assert gpu.owned and gpu.waiting == 0
    assert not gpu.owned


async def test_cancel_keeps_the_hold_until_the_call_returns():
    """OFFLINE-RENDER §2.1: a model call runs in a worker thread, which a cancel can't stop, so the task awaiting it
    keeps the GPU until the call has returned, through a second cancel too. A call asked for straight after the cancel,
    however urgent, starts only then: two calls never run on the GPU side by side."""
    gpu = GpuScheduler()
    dubber = Dubber(make_mock_backend(), gpu)
    lock, go = threading.Lock(), threading.Event()
    inside, most, order = [0], [0], []

    def call(name: str) -> str:
        with lock:
            inside[0] += 1
            most[0] = max(most[0], inside[0])
            order.append(f"{name} in")
        go.wait(5.0)  # the first blocks until the test lets it go; the second finds it set
        with lock:
            inside[0] -= 1
            order.append(f"{name} out")
        return name

    first = asyncio.create_task(dubber._on_gpu(call, "first", priority=BACKGROUND))
    for _ in range(500):
        if order:
            break
        await asyncio.sleep(0.01)
    assert order == ["first in"]
    first.cancel()
    second = asyncio.create_task(dubber._on_gpu(call, "second", priority=URGENT))
    await asyncio.sleep(0.2)
    first.cancel()
    await asyncio.sleep(0.05)
    assert order == ["first in"] and not first.done() and gpu.owned and gpu.waiting == 1
    go.set()
    got, seconds = await second
    with pytest.raises(asyncio.CancelledError):
        await first
    assert got == "second" and seconds >= 0.0
    assert most[0] == 1 and order == ["first in", "first out", "second in", "second out"]
    assert not gpu.owned and gpu.waiting == 0


async def test_a_preset_voice_is_made_holding_the_gpu(tmp_path):
    """A preset may be the first thing to touch the TTS, which then loads its weights (Chatterbox loads on first use), and
    a preset from a wav is conditioned on the GPU: made, it is a model call like any other, never beside another owner.
    Asked for again, it is the TTS's cached copy: the dub loop doesn't queue for the GPU for it."""
    gpu = GpuScheduler()
    b = make_mock_backend()
    held: list[bool] = []
    preset = b.tts.preset_voice

    def watched(name: str) -> dict:
        held.append(gpu.owned)
        return preset(name)

    b.tts.preset_voice = watched
    job = RenderJob(b, DemoResolver(), tmp_path, "https://youtu.be/dQw4w9WgXcQ", gpu=gpu)
    job.voices["S1"] = VoiceState(use_preset=True)
    _, kind, _ = await job._voice_for("S1", VoiceCost())
    assert kind is VoiceKind.PRESET and held == [True] and not gpu.owned
    async with gpu.hold(BACKGROUND):  # another owner has the GPU: the preset, made already, doesn't wait for it
        _, kind, _ = await asyncio.wait_for(job._voice_for("S1", VoiceCost()), 1.0)
    assert kind is VoiceKind.PRESET and held == [True, True]
