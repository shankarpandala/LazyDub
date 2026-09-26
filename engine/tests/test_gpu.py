"""The GPU scheduler (ARCHITECTURE §5.3): one owner at a time, the most urgent waiter next, first come first served within
a priority, no starvation, and cancellation that never loses or leaks the GPU."""

from __future__ import annotations

import asyncio

import pytest

from maata_engine.gpu import BACKGROUND, FRONTIER, URGENT, VOICE, GpuScheduler


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
