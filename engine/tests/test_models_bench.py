import hashlib
import json
import wave

import numpy as np

from maata_engine.bench import main as bench_main
from maata_engine.models import FileSpec, lfs_pointer_oid, load_lock, verify
from maata_engine.timing.isochrony import TimingSettings

_HEX = set("0123456789abcdef")


def test_lock_is_pinned_and_complete():
    lock = load_lock()
    ids = {m["id"] for m in lock["models"]}
    assert {"whisper-large-v3-turbo-mlx", "pyannote-community-1", "chatterbox-telugu"} <= ids
    # No local LLM (ADR-019): text goes through the Claude CLI, so nothing here translates.
    assert not [m for m in lock["models"] if "translator" in m["role"] or "gemma" in m["id"]]
    for m in lock["models"]:
        assert len(m["revision"]) == 40  # a commit, never a branch
        assert m["files"] and all(f["size"] > 0 for f in m["files"])
        for f in m["files"]:
            # Every file carries a real hex digest (HF masks some gated sha256s as "****").
            pins = [(k, f[k]) for k in ("sha256", "git_oid", "lfs_oid") if k in f]
            assert pins, f
            assert all(set(v) <= _HEX and len(v) == (64 if k == "sha256" else 40) for k, v in pins), f
        assert m["license"]


def test_verify_lfs_pointer_oid(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"weights")
    oid = lfs_pointer_oid(hashlib.sha256(b"weights").hexdigest(), 7)
    assert verify(p, FileSpec("w.bin", 7, None, None, oid))
    assert not verify(p, FileSpec("w.bin", 7, None, None, "0" * 40))
    # Matches a real pin: whisper-large-v3-turbo's weights at the locked commit (sha256 → HF blob id).
    assert lfs_pointer_oid("951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6", 1613977612) == \
        "26b12d9fb292a7bee72610f6e7e2ec84225feab7"


def test_verify_sha256_and_git_oid(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"maata")
    sha = hashlib.sha256(b"maata").hexdigest()
    git = hashlib.sha1(b"blob 5\0maata").hexdigest()
    assert verify(p, FileSpec("f.bin", 5, sha, None))
    assert verify(p, FileSpec("f.bin", 5, None, git))
    assert not verify(p, FileSpec("f.bin", 5, "0" * 64, None))
    assert not verify(p, FileSpec("f.bin", 6, sha, None))


def test_bench_pipeline_on_local_file_with_mock_backend(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    sr = 16000
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.random.default_rng(0).standard_normal(sr * 70) * 300).astype("<i2").tobytes())
    # The baseline run: timed as before timing v2, measured the same way.
    bench_main(["--cache", str(tmp_path), "--models", str(tmp_path), "pipeline", str(wav), "--backend", "mock",
                "--timing", "v1"])
    out = json.loads(capsys.readouterr().out)
    assert out["timing_mode"] == "v1"
    assert out["backend"] == "mock" and out["units"] > 0 and out["skipped"] == 0
    assert out["translator"] == "MockSceneTranslator" and out["tts_script"] == "telugu" and out["latin_ratio"] == 0
    assert out["speedup_max"] <= TimingSettings().speed_cap + 1e-9
    assert out["throughput_x_realtime"] > 1
    assert out["calibration_seconds"] == 0  # the plain mock TTS gives no mel takes to calibrate from
    dub = out["dub"]  # every line reviewed (the mock finds each complete), and filled against its speech time
    assert dub["lines"] == out["units"] and dub["coverage"] == {"C": 1.0, "m": 0.0, "P": 0.0, "E": 0.0,
                                                                "other_tier": 0.0, "unreviewed": 0.0}
    assert dub["retranslations_kept"] == 0 and dub["speech_fill"] > 0 and 0 <= dub["required_rate_in_band_share"] <= 1
    rt = out["realtime"]  # one take a line: the mock TTS doesn't batch takes
    assert rt["takes_n"] == {"1": out["units"]} and rt["take_failures"] == {} and rt["voicer_lock_wait_p95_s"] >= 0
    timing = out["timing"]  # §3.10, from units.jsonl: every line voiced, measured at speech level, the guards held
    assert timing["lines"] == out["units"] and timing["speech_lines"] == out["units"]
    assert timing["silent_s_per_min"] is not None and timing["end_error_p50"] is not None
    assert timing["guards"]["lag_p95"]["ok"] is not None and timing["guards"]["rate_step_p90"]["ok"] is None  # no baseline
    assert out["planner"]["lines"] == out["units"] and "end_error_p50" in out["planner"]
    # timing v2 (the default) measured against it: the targets that are relative to it say whether they are met
    (tmp_path / "base.json").write_text(json.dumps(out))
    bench_main(["--cache", str(tmp_path), "--models", str(tmp_path), "pipeline", str(wav), "--backend", "mock",
                "--baseline", str(tmp_path / "base.json")])
    v2 = json.loads(capsys.readouterr().out)
    assert v2["timing_mode"] == "v2" and v2["timing"]["speech_lines"] == v2["units"] > 0
    again = v2["timing"]["guards"]
    assert again["rate_step_p90"]["limit"] == timing["rate_step_p90"] and again["rate_step_p90"]["ok"] is not None
    for k in ("end_early_1s_share", "long_end_error_p50"):
        assert again[k]["limit"] == (None if timing[k] is None else abs(timing[k]) / 2)


def test_realtime_metrics_are_the_voicers_gpu_wait_p95_takes_per_line_and_take_failures():
    from maata_engine.bench import realtime_metrics

    units = [{"event": "unit", "lock_wait_s": w / 10, "takes_n": 1 + (w % 2), "take_failures": ["cap"] if w == 3 else [],
              "retakes": int(w == 3)} for w in range(20)] + [{"event": "asr", "lock_wait_s": 99.0}]
    got = realtime_metrics(units)
    assert got["voicer_lock_wait_p95_s"] == 1.8  # the 19th of 20 (nearest rank)
    assert got["takes_n"] == {"1": 10, "2": 10} and got["take_failures"] == {"cap": 1} and got["retakes"] == 1
    assert realtime_metrics([])["voicer_lock_wait_p95_s"] is None


def test_dub_metrics_are_coverage_shares_speech_fill_and_the_required_rate_band():
    from maata_engine.backends.base import Coverage, LineResult, Wording
    from maata_engine.bench import dub_metrics
    from maata_engine.session import Chunk, UnitState
    from maata_engine.timing.planner import Plan
    from maata_engine.types import SourceUnit

    def line(i, speech, take, rate, cov, tier="full"):
        st = UnitState(SourceUnit(i, "S1", 10.0 * i, 10.0 * i + 5.0, "A line."), None, Chunk(0.0, 60.0), speech_s=speech,
                       voiced=True, tier=tier)
        st.line = LineResult(i, {"full": Wording("ఒక మాట అని."), "concise": Wording("ఒక మాట.")}, coverage=cov)
        st.plan, st.take_s = Plan(i, 10.0 * i, rate, take / rate, 0.0), take
        return st

    skipped = UnitState(SourceUnit(9, "S1", 90.0, 95.0, "Gone."), None, Chunk(0.0, 60.0), speech_s=4.0, voiced=True)
    got = dub_metrics([line(0, 4.0, 4.0, 1.0, Coverage("C", tier="full")),                            # 1.0: in band
                       line(1, 4.0, 3.2, 1.0, Coverage("m", tier="full")),                            # 0.8: under it
                       line(2, 4.0, 4.8, 1.2, Coverage("C", tier="full", by="validators", first="P")),  # 1.2: in band
                       line(3, 4.0, 6.0, 1.2, None),                                                  # 1.5: over it
                       # reviewed on `full`, voiced on `concise` (a fit, or a shorter re-synthesis): not its class
                       line(4, 4.0, 4.0, 1.0, Coverage("C", tier="full"), tier="concise"), skipped])
    assert got["lines"] == 5 and got["retranslations_kept"] == 1
    assert got["coverage"] == {"C": 0.4, "m": 0.2, "P": 0.0, "E": 0.0, "other_tier": 0.2, "unreviewed": 0.2}
    assert got["coverage_reviewed"] == {"C": 0.2, "m": 0.2, "P": 0.2, "E": 0.0, "other_tier": 0.2, "unreviewed": 0.2}
    assert got["speech_fill"] == round((4.0 + 3.2 + 4.0 + 5.0 + 4.0) / 20.0, 3) and got["speech_fill_median"] == 1.0
    assert got["required_rate_in_band_share"] == 0.6
    assert dub_metrics([])["speech_fill"] is None


def test_timing_metrics_are_the_speech_level_yardsticks_of_the_lines_as_finally_voiced():
    from maata_engine.bench import timing_metrics

    def unit(i, spk, start, end, lag, rate, voiced, freeze=0.0, said="whole", errs=()):
        return {"event": "unit", "id": i, "speaker": spk, "start": start, "end": end, "lag": lag, "audio_rate": rate,
                "freeze": freeze, "said": said, "speech": [[start, end]], "voiced": voiced, "anchor_errors": list(errs)}

    # Line 2's provisional take was joined pieces with a freeze; the rephrase that replaced it is said whole, unfrozen.
    events = [unit(0, "a", 0.0, 10.0, 0.0, 1.0, [[0.0, 9.0]], freeze=0.3),
              unit(1, "b", 12.0, 20.0, 0.2, 1.1, [[12.2, 15.0], [16.0, 20.5]], said="pieces", errs=[0.1]),
              unit(2, "a", 22.0, 30.0, 0.0, 1.2, [[22.0, 29.0]], freeze=0.6, said="joined"),
              {"event": "rephrase", "id": 2, "outcome": "replaced", "voiced": [[22.0, 30.0]], "anchor_errors": [],
               "said": "whole", "freeze": 0.0},
              {"event": "rephrase", "id": 1, "outcome": "late"},
              {"event": "skipped", "id": 3, "start": 40.0, "end": 60.0, "speech": [[40.0, 60.0]]},
              {"event": "asr", "a": 0.0, "b": 60.0}]
    got = timing_metrics(events)
    assert (got["lines"], got["skipped"]) == (3, 1) and got["said"] == {"whole": 2, "pieces": 1, "joined": 0, "anchored": 0}
    assert got["rate_step_p90"] == 0.2 and got["freeze_s_per_10min"] == 3.0        # a's 1.0 -> 1.2; line 0's 0.3 s in 60 s
    assert got["end_error_p50"] == 0.0 and got["anchors"] == 1                     # -1.0, +0.5, and the rephrase's 0.0
    # silent: line 0's last second, line 1's 0.2 s at its start and 1 s between its pieces, all 20 s of the skipped line
    assert got["silent_s_per_min"] == round((1.0 + 1.2 + 20.0) / (46.0 / 60.0), 4)
    assert got["guards"]["silent_s_per_min"]["ok"] is False and got["guards"]["freeze_s_per_10min"]["ok"] is False
    assert got["guards"]["lag_p95"]["ok"] is True and got["guards"]["rate_p90"]["ok"] is False
    empty = timing_metrics([])
    assert empty["lines"] == 0 and empty["lag_p95"] is None and empty["guards"]["lag_p95"]["ok"] is None


def test_bench_pipeline_can_swap_in_the_mock_translator_and_latin_tts_input(tmp_path, capsys, monkeypatch):
    """--translator mock runs any backend's pipeline offline; --tts-script latin feeds the TTS the Latin rebuild."""
    import maata_engine.bench as bench
    from maata_engine.backends.claude_translator import ClaudeTranslator
    from maata_engine.backends.mock import MockSceneTranslator, make_mock_backend

    def claude_backend(name, models_dir):
        b = make_mock_backend()
        b.translator = ClaudeTranslator  # as on apple and cuda: this would call Claude
        return b

    monkeypatch.setattr(bench, "load_backend", claude_backend)
    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.random.default_rng(1).standard_normal(16000 * 40) * 300).astype("<i2").tobytes())
    bench_main(["--cache", str(tmp_path), "--models", str(tmp_path), "pipeline", str(wav), "--translator", "mock",
                "--tts-script", "latin"])
    out = json.loads(capsys.readouterr().out)
    assert out["translator"] == MockSceneTranslator.__name__ and out["tts_script"] == "latin"
    assert out["units"] > 0 and out["latin_ratio"] > 0  # the demo lines' English words, said in Latin script
