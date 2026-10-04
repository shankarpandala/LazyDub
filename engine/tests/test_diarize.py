"""Whole-file diarization (OFFLINE-RENDER §2.3) on a stand-in for pyannote's pipeline: no model, no torch. The pipeline
calls its hook as pyannote 4.0.7's `apply` does: progress per batch of segmentation, the segmentation itself (where the
memory guard counts what VBx would cluster), then progress per batch of embeddings, the embeddings, clustering (which
has no hook) and the discrete diarization."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from maata_engine.backends import torch_common as tc
from maata_engine.backends.base import COARSE_STEP, Cancelled
from maata_engine.backends.mock import MockDiarizer

WINDOW = 10.0  # s: community-1's segmentation window


class Annotation:
    def __init__(self, tracks: list[tuple[str, float, float]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for k, (label, a, b) in enumerate(self.tracks):
            yield SimpleNamespace(start=a, end=b), k, label

    def labels(self) -> list[str]:
        return sorted({label for label, _, _ in self.tracks})


def segmentation(chunks: int, pairs_per_chunk: int, frames: int = 10) -> np.ndarray:
    """(chunks, frames, 3) binary activity: in each chunk the first `pairs_per_chunk` speakers each talk alone for 3 of its
    10 frames (enough for VBx to keep them); the others are silent."""
    data = np.zeros((chunks, frames, 3), np.float32)
    for k in range(pairs_per_chunk):
        data[:, 3 * k:3 * k + 3, k] = 1.0
    return data


class FakePipeline:
    def __init__(self, pairs: list[int], wait: threading.Event | None = None, clustering=None) -> None:
        self._segmentation = SimpleNamespace(step=0.1 * WINDOW, duration=WINDOW)
        self.pairs = pairs      # speakers talking alone per chunk, per call
        self.calls: list[dict] = []
        self.wait = wait        # blocks the embeddings step until set
        self.embedding = threading.Event()  # set once it is computing embeddings
        self.clustering = clustering  # called where pyannote clusters, between two hook calls

    def __call__(self, file: dict, *, hook, num_speakers=None, min_speakers=None, max_speakers=None):
        self.calls.append({"step": self._segmentation.step, "num_speakers": num_speakers, "min_speakers": min_speakers,
                           "max_speakers": max_speakers, "samples": file["waveform"].shape[-1]})
        hook = _with_file(hook)
        chunks = 40
        hook("segmentation", None, completed=0, total=chunks)
        hook("segmentation", None, completed=chunks, total=chunks)
        hook("segmentation", SimpleNamespace(data=segmentation(chunks, self.pairs[len(self.calls) - 1])))
        hook("speaker_counting", None)
        hook("embeddings", None, completed=0, total=2)
        self.embedding.set()
        if self.wait is not None:
            self.wait.wait(5)
        hook("embeddings", None, completed=1, total=2)
        hook("embeddings", None, completed=2, total=2)
        hook("embeddings", np.zeros((chunks, 3, 4)))
        if self.clustering is not None:
            self.clustering()
        hook("discrete_diarization", None)
        ann = Annotation([("SPEAKER_00", 0.5, 4.0), ("SPEAKER_01", 4.0, 9.5)])
        return SimpleNamespace(speaker_diarization=ann, exclusive_speaker_diarization=ann,
                               speaker_embeddings=np.eye(2, 4, dtype=np.float32))


def _with_file(hook):
    """pyannote binds `file=` to the hook (Pipeline.setup_hook)."""
    return lambda *a, **kw: hook(*a, file={"uri": "x"}, **kw)


def diarizer(pipeline: FakePipeline) -> tc.PyannoteDiarizer:
    d = object.__new__(tc.PyannoteDiarizer)  # no model load
    d.pipeline = pipeline
    d._torch = SimpleNamespace(from_numpy=lambda a: SimpleNamespace(unsqueeze=lambda i: a[None], shape=a.shape))
    return d


AUDIO = np.zeros(10 * 16_000, np.float32)


def test_retained_counts_the_pairs_vbx_keeps():
    assert tc._retained(segmentation(40, 3)) == 120
    data = segmentation(5, 2)
    data[:, 0:2, 1] = 1.0      # speaker 1 overlaps speaker 0 there: those frames aren't clean for either
    assert tc._retained(data) == 5  # speaker 0 keeps 1 of 10 clean frames (< 20 %), speaker 1 still 3


def test_under_the_budget_one_run_at_the_pipelines_own_step(monkeypatch):
    monkeypatch.setattr(tc, "DIAR_CLUSTER_BUDGET", 8.4 * 120 ** 2)  # exactly 120 embeddings fit
    shares: list[float] = []
    p = FakePipeline([3])
    block = diarizer(p).diarize(AUDIO, min_speakers=1, max_speakers=6, progress=shares.append)
    assert [c["step"] for c in p.calls] == [1.0]
    assert (p.calls[0]["min_speakers"], p.calls[0]["max_speakers"], p.calls[0]["num_speakers"]) == (1, 6, None)
    assert block.step == pytest.approx(0.1) and (block.start, block.end) == (0.0, 10.0) and block.embeddings == 120
    assert [(t.speaker, t.start, t.end) for t in block.exclusive] == [("SPEAKER_00", 0.5, 4.0), ("SPEAKER_01", 4.0, 9.5)]
    assert set(block.centroids) == {"SPEAKER_00", "SPEAKER_01"}
    assert shares == pytest.approx([0.0, 0.4, 0.4, 0.65, 0.9])  # segmentation 0-40 %, embeddings 40-90 %


def test_the_memory_guard_stops_before_embeddings_and_runs_again_at_the_coarse_step(monkeypatch):
    monkeypatch.setattr(tc, "DIAR_CLUSTER_BUDGET", 8.4 * 119 ** 2)  # 120 embeddings are one too many
    p = FakePipeline([3, 2])
    shares: list[float] = []
    block = diarizer(p).diarize(AUDIO, num_speakers=2, progress=shares.append)
    assert [c["step"] for c in p.calls] == [1.0, COARSE_STEP * WINDOW]  # the 2 s step: half the windows
    assert all(c["num_speakers"] == 2 for c in p.calls)
    assert shares[:3] == pytest.approx([0.0, 0.4, 0.0])  # the first run stopped before any embedding
    assert block.step == pytest.approx(COARSE_STEP) and block.embeddings == 80  # what the second run clustered
    assert p._segmentation.step == 1.0  # the pipeline's own step is back


def test_a_forced_coarse_step_runs_once_even_over_budget(monkeypatch):
    monkeypatch.setattr(tc, "DIAR_CLUSTER_BUDGET", 1.0)
    p = FakePipeline([3])
    block = diarizer(p).diarize(AUDIO, step=COARSE_STEP)
    assert [c["step"] for c in p.calls] == [COARSE_STEP * WINDOW] and block.step == pytest.approx(COARSE_STEP)
    assert p._segmentation.step == 1.0


def test_a_set_cancel_event_aborts_the_run():
    stop = threading.Event()
    p = FakePipeline([3], wait=threading.Event())
    out: list = []

    def run() -> None:
        try:
            out.append(diarizer(p).diarize(AUDIO, cancel=stop))
        except Exception as e:  # noqa: BLE001
            out.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    assert p.embedding.wait(5)
    stop.set()           # while the pipeline is computing embeddings
    p.wait.set()
    worker.join(5)
    assert not worker.is_alive() and isinstance(out[0], Cancelled)
    assert len(p.calls) == 1 and p._segmentation.step == 1.0
    with pytest.raises(Cancelled):  # set before it starts: it stops at the first hook
        diarizer(FakePipeline([3])).diarize(AUDIO, cancel=stop)


def test_a_pause_during_clustering_lets_the_run_finish():
    """pyannote's clustering has no hook: a pause that arrives during it would only be seen at the hook after it, when
    the whole run is done. Its result is kept (the render writes diarization.json, then pauses), not thrown away."""
    stop = threading.Event()
    p = FakePipeline([3], clustering=stop.set)
    block = diarizer(p).diarize(AUDIO, cancel=stop)
    assert len(p.calls) == 1 and [t.speaker for t in block.exclusive] == ["SPEAKER_00", "SPEAKER_01"]


def test_single_speaker_and_mock_diarize_the_whole_file():
    one = tc.SingleSpeakerDiarizer().diarize(np.zeros(16_000 * 30, np.float32), min_speakers=1, max_speakers=6)
    assert [(t.speaker, t.start, t.end) for t in one.exclusive] == [("S1", 0.0, 30.0)] and not one.centroids
    mock = MockDiarizer()
    auto = mock.diarize(np.zeros(16_000 * 120, np.float32))
    assert {t.speaker for t in auto.exclusive} == {"A", "B", "C"}
    assert [(t.start, t.end) for t in auto.exclusive if t.speaker == "C"] == [MockDiarizer.BLIP]
    assert float(auto.centroids["C"] @ auto.centroids["A"]) > 0.9
    three = mock.diarize(np.zeros(16_000 * 120, np.float32), num_speakers=3, step=COARSE_STEP)
    assert [t.speaker for t in three.exclusive][:4] == ["A", "B", "C", "A"] and three.step == COARSE_STEP
    assert mock.calls == 2 and mock.step == COARSE_STEP
    with pytest.raises(Cancelled):
        mock.diarize(np.zeros(16_000, np.float32), cancel=_set())


def _set() -> threading.Event:
    ev = threading.Event()
    ev.set()
    return ev
