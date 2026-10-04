import hashlib
import json
import sys
from pathlib import Path

import pytest

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


def test_the_separator_is_pinned_for_the_apple_backend_only():
    """ADR-020: the background sound's separator, fetched by `maata-bench fetch --backend apple` like every model."""
    from maata_engine.models import models_for

    sep = next(m for m in load_lock()["models"] if m["role"] == "separation")
    assert sep["id"] == "mel-roformer-kim-vocal-2-mlx" and sep["repo"] == "mlx-community/mel-roformer-kim-vocal-2-mlx"
    assert sep["revision"] == "64cbfcb004e39430e5f584552c05949440ec39ce" and not sep["gated"]
    assert {f["path"]: f.get("sha256") or f.get("git_oid") for f in sep["files"]} == {
        "config.json": "4e7f9ed2cfcb2b3232166a5c803adf20d6816031",
        "model.safetensors": "312c38e5b698f8dfaa4d6064e8f79010744825828917871a9d22673a43eb7fe5"}
    assert sep["total_bytes"] == sum(f["size"] for f in sep["files"]) == 833 + 456_483_463
    assert sep in models_for("apple") and sep not in models_for("cuda")


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


def file_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bench_pipeline_dubs_a_local_video_to_an_mp4_with_the_mock_backend(tmp_path, capsys):
    """`maata-bench pipeline FILE` runs a render job on a local video (`LocalResolver`) to an MP4 and prints the render,
    dub, timing and export blocks; the input file is read in place and left byte-identical."""
    av = pytest.importorskip("av")
    from maata_engine.resolve import synth_video

    video = tmp_path / "talk.mp4"
    synth_video(video, 70.0)
    before = file_digest(video)
    out_dir = tmp_path / "out"
    bench_main(["--cache", str(tmp_path / "cache"), "--models", str(tmp_path), "pipeline", str(video), "--backend", "mock",
                "--out", str(out_dir)])
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "mock" and out["units"] > 0 and out["skipped"] == 0
    assert out["audio_seconds"] == pytest.approx(70.0, abs=0.05)  # (the AAC's declared duration)
    assert out["translator"] == "MockSceneTranslator" and out["tts_script"] == "telugu" and out["latin_ratio"] == 0
    assert out["speedup_max"] <= TimingSettings().speed_cap + 1e-9 and out["throughput_x_realtime"] > 1
    assert out["calibration_seconds"] == 0  # the plain mock TTS gives no mel takes to calibrate from
    render = out["render"]  # per stage: its seconds and GPU seconds, from this run's stage events
    assert set(render["stages"]) == {"fetch", "speakers", "transcript", "units", "voices", "brief", "separate",
                                     "translate", "voice_lines", "finish", "video", "export"}
    assert all(s["seconds"] >= 0 and s["gpu_s"] >= 0 and s["cached"] is False for s in render["stages"].values())
    assert render["gpu_s_per_video_s"] == round(render["gpu_s"] / out["audio_seconds"], 4) and render["peak_rss_bytes"] > 0
    assert render["claude"] == {}  # the mock translator never runs the Claude CLI
    dub = out["dub"]  # every line reviewed (the mock finds each complete), and filled against its speech time
    assert dub["lines"] == out["units"] and dub["coverage"] == {"C": 1.0, "m": 0.0, "P": 0.0, "E": 0.0,
                                                                "other_tier": 0.0, "unreviewed": 0.0}
    timing = out["timing"]  # §3.10, from units.jsonl's final lines
    assert timing["lines"] == out["units"] and timing["guards"]["lag_p95"]["ok"] is not None
    assert out["planner"]["lines"] == out["units"]
    export = out["export"]  # the MP4: in --out, the video copied, the Telugu audio and both subtitle tracks
    path = out_dir / "talk (Telugu).mp4"
    assert export["path"] == str(path) and export["bytes"] == path.stat().st_size and export["video_seconds"] == pytest.approx(70.0, abs=0.1)
    assert export["loudness"]["I"] == pytest.approx(-16.0, abs=0.5) and export["video"]["codec"] == "h264"
    assert not export["video"]["owned"] and export["video"]["file"] == str(video.resolve())
    with av.open(str(path)) as c:
        assert [s.type for s in c.streams] == ["video", "audio", "subtitle", "subtitle"]
    assert file_digest(video) == before  # read in place, never changed (and never deleted)
    # measured against a committed run's JSON: the targets relative to it say whether they are met
    (tmp_path / "base.json").write_text(json.dumps(out))
    bench_main(["--cache", str(tmp_path / "cache"), "--models", str(tmp_path), "pipeline", str(video), "--backend", "mock",
                "--out", str(out_dir), "--baseline", str(tmp_path / "base.json")])
    again = json.loads(capsys.readouterr().out)
    assert again["timing"]["guards"]["rate_step_p90"]["limit"] == timing["rate_step_p90"]
    assert again["export"]["path"] == str(path) and file_digest(video) == before


def test_each_bench_run_is_a_clean_job_on_its_own_file(tmp_path, capsys):
    """Two files benched one after the other in one cache: the second run dubs and saves the second file (not the job
    the first left, served from disk), every stage measured again; by default the MP4 goes to <cache>/out."""
    pytest.importorskip("av")
    from maata_engine.resolve import synth_video

    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    synth_video(first, 30.0)
    synth_video(second, 45.0)
    runs = []
    for video in (first, second):
        bench_main(["--cache", str(tmp_path / "cache"), "--models", str(tmp_path), "pipeline", str(video),
                    "--backend", "mock"])
        runs.append(json.loads(capsys.readouterr().out))
    a, b = runs
    assert a["audio_seconds"] == pytest.approx(30.0, abs=0.05) and b["audio_seconds"] == pytest.approx(45.0, abs=0.05)
    assert b["file"] == "second.mp4" and b["export"]["video"]["file"] == str(second.resolve())
    assert b["export"]["path"] == str(tmp_path / "cache" / "out" / "second (Telugu).mp4")
    assert b["export"]["video_seconds"] == pytest.approx(45.0, abs=0.1)
    assert not any(s["cached"] for s in b["render"]["stages"].values())  # nothing served from the first run's job
    assert Path(a["export"]["path"]).is_file()  # the first file's MP4 stays


def test_the_bench_wants_a_video_before_any_model_loads(tmp_path, monkeypatch):
    import wave

    import maata_engine.bench as bench

    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\0\0" * 16_000)
    monkeypatch.setattr(bench, "load_backend", lambda *a: pytest.fail("a model loaded"))
    with pytest.raises(SystemExit, match="clip.wav can't be dubbed to a video: clip.wav has no video stream"):
        bench_main(["--cache", str(tmp_path / "cache"), "--models", str(tmp_path), "pipeline", str(clip)])


def test_render_metrics_are_stage_times_gpu_per_video_second_and_claude_by_call_type(monkeypatch):
    from types import SimpleNamespace

    from maata_engine.bench import render_metrics

    events = [{"event": "stage", "key": "speakers", "seconds": 40.0, "gpu_s": 30.0, "cached": False},
              {"event": "stage", "key": "separate", "seconds": 12.0, "gpu_s": 0.0, "cached": False, "blocks": 3,
               "found": 0},
              {"event": "stage", "key": "transcript", "seconds": 90.0, "gpu_s": 80.0, "cached": False},
              {"event": "stage", "key": "fetch", "seconds": 0.1, "gpu_s": 0.0, "cached": True},
              {"event": "claude", "call": "scene", "input_tokens": 100, "cache_read_tokens": 900, "output_tokens": 600},
              {"event": "claude", "call": "scene", "input_tokens": 50, "cache_read_tokens": None, "output_tokens": 400},
              {"event": "claude", "call": "review", "input_tokens": 10, "cache_read_tokens": 0, "output_tokens": 160},
              {"event": "unit", "id": 0}]
    got = render_metrics(events, 1000.0)
    assert got["stages"]["fetch"] == {"seconds": 0.1, "gpu_s": 0.0, "cached": True}
    assert got["gpu_s"] == 110.0 and got["gpu_s_per_video_s"] == 0.11 and got["peak_rss_bytes"] > 0
    assert got["claude"] == {"scene": {"calls": 2, "input_tokens": 150, "cache_read_tokens": 900, "output_tokens": 1000},
                             "review": {"calls": 1, "input_tokens": 10, "cache_read_tokens": 0, "output_tokens": 160}}
    assert got["stages"]["separate"] == {"seconds": 12.0, "gpu_s": 0.0, "cached": False, "blocks": 3}  # 4 s a block
    assert render_metrics([], 0.0)["gpu_s_per_video_s"] is None
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(get_peak_memory=lambda: 7_000_000_000))
    assert render_metrics([], 0.0)["mlx_peak_bytes"] == 7_000_000_000  # Whisper's and the separator's, on MLX
    monkeypatch.delitem(sys.modules, "mlx.core")
    assert render_metrics([], 0.0)["mlx_peak_bytes"] is None  # nothing ran on MLX (the mock, CUDA)


def test_dub_metrics_are_coverage_shares_speech_fill_and_the_required_rate_band():
    from maata_engine.backends.base import Coverage, LineResult, Wording
    from maata_engine.bench import dub_metrics
    from maata_engine.dubber import UnitState
    from maata_engine.timing.planner import Plan
    from maata_engine.types import SourceUnit

    def line(i, speech, take, rate, cov, tier="full"):
        st = UnitState(SourceUnit(i, "S1", 10.0 * i, 10.0 * i + 5.0, "A line."), None, speech_s=speech, voiced=True,
                       tier=tier)
        st.line = LineResult(i, {"full": Wording("ఒక మాట అని."), "concise": Wording("ఒక మాట.")}, coverage=cov)
        st.plan, st.take_s = Plan(i, 10.0 * i, rate, take / rate, 0.0), take
        return st

    skipped = UnitState(SourceUnit(9, "S1", 90.0, 95.0, "Gone."), None, speech_s=4.0, voiced=True)
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
    pytest.importorskip("av")
    import maata_engine.bench as bench
    from maata_engine.backends.claude_translator import ClaudeTranslator
    from maata_engine.backends.mock import MockSceneTranslator, make_mock_backend
    from maata_engine.resolve import synth_video

    def claude_backend(name, models_dir):
        b = make_mock_backend()
        b.translator = ClaudeTranslator  # as on apple and cuda: this would call Claude
        return b

    monkeypatch.setattr(bench, "load_backend", claude_backend)
    video = tmp_path / "clip.mp4"
    synth_video(video, 40.0)
    bench_main(["--cache", str(tmp_path / "cache"), "--models", str(tmp_path), "pipeline", str(video), "--translator",
                "mock", "--tts-script", "latin", "--out", str(tmp_path / "out")])
    out = json.loads(capsys.readouterr().out)
    assert out["translator"] == MockSceneTranslator.__name__ and out["tts_script"] == "latin"
    assert out["units"] > 0 and out["latin_ratio"] > 0  # the demo lines' English words, said in Latin script
