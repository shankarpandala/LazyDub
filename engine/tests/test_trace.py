"""units.jsonl (ARCHITECTURE §7 step 0): per-block diarization and ASR events, and per-unit cost and text fields, with a
stable set of keys whether or not the TTS reports its internal timings."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np
import pytest

from maata_engine.backends.base import SR_ANALYSIS, Backend
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTranscriber, MockTTS
from maata_engine.gpu import BACKGROUND, FRONTIER, URGENT, VOICE, GpuScheduler
from maata_engine.resolve import DemoResolver
from maata_engine.session import Session, VoiceCost
from maata_engine.text.scene_prompt import PROMPT_HASH

UNIT_KEYS = {"lock_wait_s", "takes", "synth_s", "t3_s", "t3_tokens", "t3_steps", "t3_ms_per_token", "flow_s", "cfm_steps",
             "render_s", "retakes",
             "translate_ready_at", "translate_s", "model", "prompt_hash", "cache_hit", "wall_s", "audio_s", "tier",
             "tier_first", "tiers", "speech_s", "scene", "full_aksharas", "full_pred_aksharas", "flags", "provisional",
             "breaks", "anchors", "cut_off", "music", "lint", "lint_version", "takes_n", "take_failures", "gpu_priority",
             "lead_s", "r", "said", "parts", "rate_cap", "hard_breaks", "speech", "voiced", "anchor_errors",
             "end_error_s", "overlap_speech"}  # the timing fields (§3.10)
ASR_KEYS = {"redecoded", "recovered", "rejected", "punctuated", "edge_guesses", "uncovered", "low_confidence",
            "repeats"}  # the ASR guards (§3.4)


@dataclass
class FakeTake:
    samples: np.ndarray
    t3_s: float = 0.02
    t3_tokens: int = 40
    t3_steps: int = 40
    flow_s: float = 0.01
    cfm_steps: int = 6

    @property
    def seconds(self) -> float:
        return len(self.samples) / MockTTS.sample_rate


class MelTTS(MockTTS):
    """The mock voice, split into a mel take plus vocoding like Chatterbox, reporting fixed T3 and S3Gen timings."""

    def synthesize_mel(self, text, voice, language="te", max_seconds=None) -> FakeTake:
        return FakeTake(self.synthesize(text, voice, language, max_seconds))

    def vocode(self, take: FakeTake, rate: float = 1.0) -> np.ndarray:
        return take.samples[: round(len(take.samples) / rate)]


async def _dub(tmp_path, tts, units: int = 3) -> list[dict]:
    async def sj(_: dict) -> None: ...

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, tts)
    s = Session(backend, DemoResolver(), tmp_path, sj, sb)
    await s.open("https://youtu.be/dQw4w9WgXcQ")
    trace = tmp_path / "dQw4w9WgXcQ" / "units.jsonl"
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            if trace.exists() and trace.read_text().count('"event": "unit"') >= units:
                break
    finally:
        await s.close()
    return [json.loads(line) for line in trace.read_text().splitlines()]


@pytest.mark.parametrize("tts", [MockTTS(), MelTTS()], ids=["samples", "mel"])
async def test_units_jsonl_records_block_and_unit_costs(tmp_path, tts):
    events = await _dub(tmp_path, tts)
    by = {k: [e for e in events if e["event"] == k] for k in ("diar", "asr", "clone", "unit")}
    assert all(by.values()), {k: len(v) for k, v in by.items()}
    for e in by["diar"] + by["asr"]:
        assert e["b"] > e["a"] and e["run_s"] >= 0 and e["lock_wait_s"] >= 0 and e["priority"] in (FRONTIER, BACKGROUND)
    assert all(e["turns"] > 0 and e["speakers"] > 0 for e in by["diar"])
    assert any(e["words"] > 0 for e in by["asr"])  # a short tail chunk may have none
    assert all(ASR_KEYS <= e.keys() for e in by["asr"])
    assert all(e["build_s"] >= 0 and e["calibrate_s"] >= 0 and e["lock_wait_s"] >= 0 for e in by["clone"])
    # the calibration (three sentences where the TTS gives mel takes; none for the plain mock) and the pace it set
    assert all(e["calibration_takes"] == (3 if isinstance(tts, MelTTS) else 0) and e["pace"] > 0 and e["overhead_s"] >= 0
               for e in by["clone"])
    heard = set()
    for u in by["unit"]:
        assert UNIT_KEYS <= u.keys()
        assert u["takes"] >= 1 and u["lock_wait_s"] >= 0 and u["synth_s"] >= 0 and u["render_s"] >= 0
        assert u["takes_n"] == 1 and u["take_failures"] == [] and u["gpu_priority"] in (URGENT, VOICE)  # no batching TTS
        assert u["translate_ready_at"] <= u["t"] and u["translate_s"] >= 0
        assert u["model"] == "mock" and u["prompt_hash"] == PROMPT_HASH
        # The demo says six lines over and over: a line can be served from the line cache only once said before.
        assert isinstance(u["cache_hit"], bool) and (not u["cache_hit"] or (u["source"], u["speaker"]) in heard)
        heard.add((u["source"], u["speaker"]))
        if isinstance(tts, MelTTS):
            assert u["t3_tokens"] == u["t3_steps"] == 40 * u["takes"] and u["t3_ms_per_token"] == 0.5
            assert u["cfm_steps"] == 6 and u["retakes"] == 0
            assert u["t3_s"] == pytest.approx(0.02 * u["takes"]) and u["flow_s"] == pytest.approx(0.01 * u["takes"])
        else:  # the TTS doesn't say: null, not missing
            assert u["t3_s"] is u["t3_tokens"] is u["t3_steps"] is u["t3_ms_per_token"] is u["flow_s"] is None
            assert u["cfm_steps"] is None


async def test_lock_wait_counts_only_the_wait():
    gpu, cost = GpuScheduler(), VoiceCost()

    async def holder() -> None:
        async with gpu.hold(BACKGROUND):
            await asyncio.sleep(0.1)

    task = asyncio.create_task(holder())
    await asyncio.sleep(0)  # let the holder take the GPU
    async with cost.hold(gpu, URGENT):
        await asyncio.sleep(0.05)  # work while holding it isn't waiting
    await task
    assert 0.08 <= cost.lock_wait_s < 0.15


async def test_a_clone_on_the_voicing_path_counts_its_lock_waits(tmp_path, monkeypatch):
    """A speaker cloned when their first line is voiced: the unit waited for the GPU lock during the clone too."""
    async def sj(_: dict) -> None: ...

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, MelTTS())
    s = Session(backend, DemoResolver(), tmp_path, sj, sb)
    s._dir, s.audio = tmp_path, np.zeros(10 * SR_ANALYSIS, np.float32)
    monkeypatch.setattr(s.registry, "best_span", lambda *a, **k: None)  # the stitched reference, from one 6 s clip
    monkeypatch.setattr(s.registry, "reference_clips", lambda *a, **k: [(0.0, 6.0)])

    async def holder() -> None:
        async with s.gpu.hold(BACKGROUND):
            await asyncio.sleep(0.1)

    task = asyncio.create_task(holder())
    await asyncio.sleep(0)  # the diarizer, say, has the GPU
    cost = VoiceCost()
    _, kind, key = await s._voice_for("S1", cost)
    await task
    assert kind.value == "cloned" and key == s.voices["S1"].key and key.voice == "S1" and key.reference
    assert 0.08 <= cost.lock_wait_s < 0.3 and cost.takes == 0  # the calibration take isn't one of the unit's
    clone = [json.loads(line) for line in (tmp_path / "units.jsonl").read_text().splitlines()][-1]
    assert clone["event"] == "clone" and clone["lock_wait_s"] == round(cost.lock_wait_s, 3)


def test_chatterbox_take_times_t3_and_flow_separately():
    """The real `synthesize_mel` and flow, with stand-ins for the model's parts (no weights, CPU): T3 is timed when the
    take is made, S3Gen's flow when it is vocoded."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("chatterbox.mtl_tts")
    from types import SimpleNamespace

    from maata_engine.backends.torch_common import ChatterboxTeluguTTS, ChatterboxVoice

    def t3(tt, max_new, cfg, n, seeds):
        time.sleep(0.03)
        return [(torch.arange(10, 22), False)], 16  # 12 tokens, its end, and the steps to the next check

    def flow(speech, ref_dict, n_cfm_timesteps, finalize):
        time.sleep(0.02)
        return torch.zeros(1, 80, 2 * speech.shape[-1])

    tts = object.__new__(ChatterboxTeluguTTS)  # skip loading weights
    tts._torch, tts.cfm_steps, tts.cfg_weight, tts._t3_tokens = torch, 6, 0.5, t3
    tts.model = SimpleNamespace(
        device="cpu", conds=None, t3=SimpleNamespace(hp=SimpleNamespace(start_text_token=1, stop_text_token=2)),
        tokenizer=SimpleNamespace(text_to_tokens=lambda text, language_id: torch.ones(1, 5, dtype=torch.long)),
        s3gen=SimpleNamespace(flow_inference=flow))
    take = tts.synthesize_mel("ఒక చిన్న వాక్యం", ChatterboxVoice(SimpleNamespace(gen={})))
    assert (take.n_tokens, take.t3_tokens, take.t3_steps, take.cfm_steps, take.capped) == (12, 12, 16, 6, False)
    assert 0.03 <= take.t3_s < 0.2 and take.flow_s == 0.0
    tts._flow(take)
    assert 0.02 <= take.flow_s < 0.2


async def test_the_flow_run_when_a_take_is_vocoded_counts_as_synthesis_not_rendering(tmp_path):
    """S3Gen's flow runs when the take voiced is vocoded: it goes in `synth_s` (and `flow_s`), and `render_s` stays the
    vocoder alone, as when every take was flowed as it was made."""
    class FlowTTS(MelTTS):
        def vocode(self, take: FakeTake, rate: float = 1.0) -> np.ndarray:
            time.sleep(0.04)
            take.flow_s += 0.04  # the flow, timed as torch_common times it
            time.sleep(0.01)     # the vocoder
            return take.samples

    async def sj(_: dict) -> None: ...

    async def sb(_: bytes) -> None: ...

    s = Session(Backend("mock", "cpu", MockTranscriber(), MockDiarizer(), MockSceneTranslator, FlowTTS()), DemoResolver(),
                tmp_path, sj, sb)
    cost = VoiceCost(synth_s=0.2)
    await s._render(FakeTake(np.zeros(10, np.float32), flow_s=0.0), 1.0, cost)
    assert cost.flow_s == pytest.approx(0.04) and cost.synth_s == pytest.approx(0.24)
    assert 0.005 <= cost.render_s < 0.035


def test_costs_add_up_over_takes():
    cost = VoiceCost()
    cost.add_take(FakeTake(np.zeros(10, np.float32)), 0.5)
    cost.add_take(FakeTake(np.zeros(10, np.float32), t3_s=0.06, t3_tokens=20, t3_steps=20), 0.25)
    f = cost.fields()
    assert (f["takes"], f["synth_s"], f["t3_tokens"], f["t3_steps"], f["cfm_steps"]) == (2, 0.75, 60, 60, 6)
    assert f["t3_s"] == 0.08 and f["t3_ms_per_token"] == pytest.approx(1.333, abs=1e-3) and f["flow_s"] == 0.02
    # Two takes of one batched decode: 40 steps (to the check after the later end) in 0.06 s, its time shared out. The
    # steps count once, so the ms per token is the decode's per step, as for one take; per token sampled it would halve.
    batch = VoiceCost()
    batch.add_takes([FakeTake(np.zeros(10, np.float32), t3_s=0.03, t3_tokens=31, t3_steps=40),
                     FakeTake(np.zeros(10, np.float32), t3_s=0.03, t3_tokens=37, t3_steps=40)], 0.1)
    f = batch.fields()
    assert (f["takes"], f["t3_tokens"], f["t3_steps"]) == (2, 68, 40) and f["t3_ms_per_token"] == pytest.approx(1.5)
    plain = VoiceCost()
    plain.add_take(np.zeros(10, np.float32), 0.5)
    assert plain.fields()["t3_s"] is None and plain.fields()["t3_ms_per_token"] is None
