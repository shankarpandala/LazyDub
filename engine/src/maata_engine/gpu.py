"""One GPU owner at a time, by priority (ARCHITECTURE §5.3; ADR-004, ADR-016).

MLX work (ASR) and MPS/CUDA work (diarization, TTS) of every session take turns on the one GPU: running them side by
side measured slower than taking turns on the M5 Pro (ADR-016). Translation goes through the Claude CLI (ADR-019) and
never asks for it. The priorities, most urgent first:
1. URGENT: the voicer while the dub reaches less than a minute past the playhead;
2. FRONTIER: ASR and diarization the translator is waiting on (the scene frontier);
3. VOICE: the voicer in normal operation;
4. BACKGROUND: listening and diarizing beyond the frontier, and building the voices of speakers found later.
Within a priority, first come first served. A waiter moves up a priority for every `age` seconds it has waited, so
nothing starves: a background hold waits at most about three times `age` behind a stream of urgent ones.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

URGENT, FRONTIER, VOICE, BACKGROUND = 1, 2, 3, 4
AGE_S = 20.0  # s of waiting that count as one priority more


@dataclass(eq=False)
class _Waiter:
    priority: int
    since: float
    seq: int
    fut: asyncio.Future


class GpuScheduler:
    def __init__(self, age: float = AGE_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.age, self._clock = age, clock
        self._owned = False
        self._waiting: list[_Waiter] = []
        self._seq = itertools.count()

    @property
    def owned(self) -> bool:
        return self._owned

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def _rank(self, w: _Waiter, now: float) -> tuple[int, int]:
        return w.priority - int((now - w.since) // self.age), w.seq

    async def acquire(self, priority: int) -> None:
        if not self._owned:
            self._owned = True
            return
        w = _Waiter(priority, self._clock(), next(self._seq), asyncio.get_running_loop().create_future())
        self._waiting.append(w)
        try:
            await w.fut
        except asyncio.CancelledError:
            if w.fut.done() and not w.fut.cancelled():
                self.release()  # it was handed over just as the waiter was cancelled: hand it on
            elif w in self._waiting:
                self._waiting.remove(w)
            raise

    def release(self) -> None:
        """Hand the GPU to the waiter ranked first (its priority, raised by its wait), else free it."""
        now = self._clock()
        while self._waiting:
            w = min(self._waiting, key=lambda x: self._rank(x, now))
            self._waiting.remove(w)
            if not w.fut.done():  # a cancelled waiter's future is done: skip it
                w.fut.set_result(None)
                return
        self._owned = False

    @asynccontextmanager
    async def hold(self, priority: int) -> AsyncIterator[None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()
