"""verify-mac.sh's measurements (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §10 step 6, §12), on media made here:

    python verify_mac.py video WORK AIFF...       the test video: colour bars with one white frame at MARK s, and audio
                                                  that is the `say` speech stem (the AIFFs, in turn) over a music stem
                                                  (synthetic chords and percussion, or MAATA_VERIFY_MUSIC checked by
                                                  MAATA_VERIFY_MUSIC_SHA256) with a 1 kHz beep at MARK; both stems kept
    python verify_mac.py separation WORK MODELS   the separator on the test mix: seconds per block, MLX peak memory, the
                                                  vocals' SI-SDR against the speech stem, bf16 against float32 on one
                                                  block, and the English left in the bed over the speech turns
    python verify_mac.py export WORK MP4 MANIFEST the dubbed MP4: its streams, faststart, the copied video, loudness, the
                                                  A/V offset (the white frame against the beep in the Telugu track) and
                                                  the PerTh watermark under the Telugu lines
    python verify_mac.py proxy WORK               an HTTPS proxy on 127.0.0.1 that tunnels only to Anthropic's hosts
                                                  (the Claude CLI's way out of the bench's sandbox); WORK/proxy.port
    python verify_mac.py combine WORK OUT         the steps' JSON, the network log and the checks, in one file OUT

Run from engine/ with `uv run --no-sync python ../scripts/verify_mac.py`. Each step writes WORK/<step>.json and prints it.
Everything but the proxy is local: no network, no Claude. The music is synthetic unless a local CC0/CC-BY file is
given."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

SR = 44_100                 # the separator's, the mix's and the test video's audio rate
MARK = 10.0                 # s: the white frame and the start of the beep
BEEP = (1000.0, 0.3)        # Hz and s of the beep (an effect: the separator leaves it in the bed)
FPS, SIZE = 10, (320, 180)  # the test card, as the demo's
SPEECH_DB, MUSIC_DB = -20.0, -26.0  # RMS dBFS of the speech over its turns and of the music: speech above the music
SI_SDR_FLOOR = 8.0          # dB: the vocals' SI-SDR below this fails the run (§12)
DTYPE_MARGIN = 0.1          # dB: bf16 losing more than this against float32 means SEP_DTYPE should be float32 (§2.14)
ENGLISH_MAX_DB = float(os.environ.get("MAATA_VERIFY_ENGLISH_MAX_DB", "-20"))  # the English left in the bed, at most
LEAK_NFFT, LEAK_HOP = 2048, 441  # the separator's STFT (Kim Vocal 2), where its mask acts: the English left is read there
LEAK_TILE = 16              # turn frames (0.16 s) of one frequency bin the bed is fitted over (english_left)
AV_MAX_MS = 20.0            # |A/V offset| above this fails (well under what a viewer notices)
LUFS, LUFS_TOL, TP_MAX = -16.0, 0.5, -1.0  # the export's loudness targets (§2.15)


def _decode(path: Path) -> np.ndarray:
    """A file's audio, stereo float32 at SR ([2, n])."""
    from maata_engine.resolve import decode_stereo

    return np.concatenate(list(decode_stereo(path, SR)), axis=1)


def _db(x: float) -> float:
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def _rms_to(x: np.ndarray, db: float, mask: np.ndarray | None = None) -> np.ndarray:
    """`x` scaled so its RMS (over `mask`'s samples) is `db` dBFS."""
    sel = x[..., mask] if mask is not None else x
    rms = float(np.sqrt(np.mean(np.square(sel, dtype=np.float64))))
    return (x * (10 ** (db / 20) / rms)).astype(np.float32) if rms > 0 else x


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR (dB) of `est` against `ref` (same shape, any channels: flattened)."""
    est, ref = est.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    target = (est @ ref) / (ref @ ref) * ref
    err = float((est - target) @ (est - target))
    return _db(float(target @ target) / err) if err > 0 else float("inf")


# ---- the test video ----------------------------------------------------------------------------------------------------
def _music(n: int) -> np.ndarray:
    """Synthetic music [2, n]: a four-chord loop (each note with three harmonics, none within 100 Hz of the beep's
    1 kHz), a kick on every beat and soft hats on the off-beats, with the stereo field a little wide."""
    t = np.arange(n) / SR
    chords = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63), (196.0, 233.08, 293.66)]
    beat = 0.5  # 120 bpm
    out = np.zeros((2, n), np.float32)
    bar = 4 * beat
    for k in range(int(math.ceil(n / SR / bar))):
        a, b = int(k * bar * SR), min(int((k + 1) * bar * SR), n)
        if a >= n:
            break
        tt = t[a:b] - k * bar
        env = np.minimum(1.0, tt / 0.05) * np.exp(-tt / 3.0)
        tone = np.zeros(b - a)
        for f0 in chords[k % 4]:
            for h, g in ((1, 1.0), (2, 0.3), (3, 0.12)):
                if abs(f0 * h - BEEP[0]) > 100.0:
                    tone += g * np.sin(2 * np.pi * f0 * h * tt)
        out[0, a:b] += 0.8 * env * tone
        out[1, a:b] += 0.8 * env * np.roll(tone, 40)
    rng = np.random.default_rng(7)
    for k in range(int(n / SR / beat)):
        a = int(k * beat * SR)
        m = min(int(0.25 * SR), n - a)
        tt = np.arange(m) / SR
        kick = np.sin(2 * np.pi * (50 + 60 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.12)
        out[:, a:a + m] += 1.5 * kick
        h = int((k * beat + beat / 2) * SR)
        if h + 2000 < n:
            noise = np.diff(rng.standard_normal(2001)) * np.exp(-np.arange(2000) / 300.0)  # high-passed: a hat
            out[:, h:h + 2000] += 0.15 * noise
    return out


def _music_file(path: Path, sha256: str, n: int) -> np.ndarray:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256.lower():
        raise SystemExit(f"{path.name}: sha256 {digest}, not {sha256}")
    x = _decode(path)
    return np.tile(x, (1, -(-n // x.shape[1])))[:, :n]


def make_video(work: Path, aiffs: list[Path], seconds: float = 120.0) -> dict:
    """stems.npz (speech, background, the speech turns) and test.mp4 in `work`: the say lines in turn from 1 s, 0.7 s
    apart with a 3 s music-only gap after every fourth, none within 1.5 s of MARK, until `seconds`."""
    import av

    from maata_engine.export import aac

    if not 60.0 <= seconds <= 180.0:
        raise SystemExit("the test video is 1-3 minutes")
    n = int(seconds * SR)
    lines = [np.mean(_decode(p), axis=0) for p in aiffs]
    speech = np.zeros(n, np.float32)
    turns: list[list[float]] = []
    t, k = 1.0, 0
    while True:
        x = lines[k % len(lines)]
        d = len(x) / SR
        if t < MARK + 1.5 and t + d > MARK - 1.5:
            t = MARK + 1.5
        if t + d > seconds - 1.0:
            break
        a = int(round(t * SR))
        speech[a:a + len(x)] += x
        turns.append([round(t, 3), round(t + d, 3)])
        t += d + (3.0 if k % 4 == 3 else 0.7)
        k += 1
    on = np.zeros(n, bool)
    for a, b in turns:
        on[int(a * SR):int(b * SR)] = True
    speech = np.stack([speech, speech])
    speech = _rms_to(speech, SPEECH_DB, on)
    music_env = os.environ.get("MAATA_VERIFY_MUSIC")
    if music_env:
        music = _music_file(Path(music_env).expanduser(), os.environ.get("MAATA_VERIFY_MUSIC_SHA256", ""), n)
        source = {"file": Path(music_env).name, "sha256": os.environ["MAATA_VERIFY_MUSIC_SHA256"].lower()}
    else:
        music, source = _music(n), {"synthetic": "chords, kick and hats"}
    music = _rms_to(music, MUSIC_DB)
    hole = (np.arange(n) / SR - MARK)  # the music dips around the beep, so its onset is the beep's alone
    music *= np.clip(np.abs(hole) / 0.5 - 0.5, 0.0, 1.0).astype(np.float32)
    a, m = int(MARK * SR), int(BEEP[1] * SR)
    beep = 0.25 * np.sin(2 * np.pi * BEEP[0] * np.arange(m) / SR).astype(np.float32)
    background = music.copy()
    background[:, a:a + m] += beep
    mix = speech + background
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        speech, background, mix = (x * (0.98 / peak) for x in (speech, background, mix))
    np.savez(work / "stems.npz", speech=speech, background=background, turns=np.asarray(turns, np.float64))
    w, h = SIZE
    bars = np.array([180, 168, 145, 133, 63, 51, 28], np.uint8)[np.arange(w) * 7 // w]
    with av.open(str(work / "test.mp4"), "w") as out:
        v = out.add_stream("libx264", rate=FPS)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        au = out.add_stream(aac(), rate=SR)
        au.layout = "stereo"
        step = SR // FPS
        for f in range(int(seconds * FPS)):
            y = np.full((h, w), 235, np.uint8) if f == int(round(MARK * FPS)) else np.broadcast_to(np.roll(bars, 2 * f), (h, w))
            uv = np.full((h // 2, w // 2), 128, np.uint8)
            frame = av.VideoFrame.from_ndarray(np.concatenate([y.ravel(), uv.ravel(), uv.ravel()]).reshape(-1, w),
                                               format="yuv420p")
            frame.pts = f
            for p in v.encode(frame):
                out.mux(p)
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(mix[:, f * step:(f + 1) * step]), format="fltp",
                                            layout="stereo")
            af.sample_rate, af.pts, af.time_base = SR, f * step, Fraction(1, SR)
            for p in au.encode(af):
                out.mux(p)
        for s in (v, au):
            for p in s.encode(None):
                out.mux(p)
    return {"seconds": seconds, "lines": len(turns), "speech_turns_s": round(sum(b - a for a, b in turns), 1),
            "mark": MARK, "beep_hz": BEEP[0], "speech_db": SPEECH_DB, "music_db": MUSIC_DB, "music": source,
            "say_voices": os.environ.get("MAATA_VERIFY_VOICES", "").split()}


# ---- the separator on the test mix -------------------------------------------------------------------------------------
def _turn_mask(turns: np.ndarray, n: int) -> np.ndarray:
    on = np.zeros(n, bool)
    for a, b in turns:
        on[int(a * SR):int(b * SR)] = True
    return on


def _spectra(x: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Mono `x`'s Hann-windowed spectra on the separator's grid, one frame centred on each sample of `at` [frames, bins]."""
    pad = np.pad(x.astype(np.float64), LEAK_NFFT // 2)
    return np.fft.rfft(pad[at[:, None] + np.arange(LEAK_NFFT)] * np.hanning(LEAK_NFFT + 1)[:-1], axis=1)


def english_left(bed: np.ndarray, speech: np.ndarray, music: np.ndarray, on: np.ndarray) -> dict:
    """What of the English is still in the bed over the speech turns (`on`), in dB against the speech there. On the
    separator's STFT grid, each frequency bin's turn frames are taken LEAK_TILE at a time and the bed fitted there as
    a·speech + b·music (complex least squares): the English left is Σ|a|²·|speech|² (|a| at most 1), so a leak the
    separator's mask filtered ((1 − mask)·speech: a band of it, a time-varying share) counts at its energy, where one
    gain over all the turns would read it at half its dB. What is neither (the separator's artefacts) is reported the
    same way, and the music's kept share as an amplitude gain."""
    at = np.flatnonzero(on[::LEAK_HOP]) * LEAK_HOP
    at = at[:len(at) // LEAK_TILE * LEAK_TILE]
    left = rest = s_energy = kept = m_energy = 0.0
    for ch in range(bed.shape[0]):
        B, S, M = (_spectra(x[ch], at).reshape(-1, LEAK_TILE, LEAK_NFFT // 2 + 1) for x in (bed, speech, music))
        ss, mm = (np.sum(np.abs(X) ** 2, axis=1) for X in (S, M))
        sm, sb, mb = (np.sum(np.conj(X) * Y, axis=1) for X, Y in ((S, M), (S, B), (M, B)))
        lam = 1e-9 * (ss + mm) + 1e-30  # a tile with no speech or no music in a bin stays solvable
        det = (ss + lam) * (mm + lam) - np.abs(sm) ** 2
        a = (sb * (mm + lam) - sm * mb) / det
        b = ((ss + lam) * mb - np.conj(sm) * sb) / det
        left += float(np.sum(np.minimum(np.abs(a), 1.0) ** 2 * ss))
        rest += float(np.sum(np.abs(B - a[:, None] * S - b[:, None] * M) ** 2))
        kept += float(np.sum(np.abs(b) ** 2 * mm))
        s_energy += float(ss.sum())
        m_energy += float(mm.sum())
    return {"english_left_db": round(_db(left / s_energy), 2), "artefacts_db": round(_db(rest / s_energy), 2),
            "background_gain": round(math.sqrt(kept / m_energy), 3) if m_energy else None}


def separation(work: Path, sep, pair: dict | None = None) -> dict:
    """The separator `sep` (the backend's at SEP_DTYPE, `vocals([b, 2, chunk]) -> vocals`) over stems.npz's mixture
    block by block, as the separate stage runs it (`separate.block_bed`): seconds per block, the vocals' SI-SDR against
    the speech stem, and the English left in the bed over the speech turns. With `pair` (the same model as
    {"bfloat16": ..., "float32": ...}), the block with the most speech separated by both."""
    from maata_engine import separate

    z = np.load(work / "stems.npz")
    speech, background, turns = z["speech"], z["background"], z["turns"]
    mix = (speech + background).astype(np.float32)
    n = mix.shape[1]
    peak = None
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except ImportError:
        mx = None
    beds, per_block = [], []
    for k in range(separate.block_count(n)):
        t0 = time.perf_counter()
        bed, _ = separate.block_bed(k, n, mix, 0, sep.vocals, sr=SR, chunk=sep.chunk)
        per_block.append(round(time.perf_counter() - t0, 2))
        beds.append(bed)
    if mx is not None:
        peak = int(mx.get_peak_memory())
    bed = np.concatenate(beds, axis=1)
    vocals = mix - bed
    on = _turn_mask(turns, n)
    out = {"blocks": len(per_block), "block_seconds": per_block,
           "seconds_per_block": round(float(np.mean(per_block)), 2), "mlx_peak_bytes": peak,
           "dtype": getattr(sep, "dtype", None), "si_sdr_db": round(si_sdr(vocals, speech), 2),
           **english_left(bed, speech, background, on)}
    out["si_sdr_ok"] = out["si_sdr_db"] >= SI_SDR_FLOOR
    out["english_ok"] = out["english_left_db"] <= ENGLISH_MAX_DB
    if pair is not None:
        k = max(range(len(beds)), key=lambda j: on[j * separate.block_samples():(j + 1) * separate.block_samples()].sum())
        a, b = separate.block_range(k, n)
        got = {}
        for name in ("bfloat16", "float32"):
            s = pair[name]
            t0 = time.perf_counter()
            blk, _ = separate.block_bed(k, n, mix, 0, s.vocals, sr=SR, chunk=s.chunk)
            got[name] = {"si_sdr_db": round(si_sdr(mix[:, a:b] - blk, speech[:, a:b]), 2),
                         "seconds": round(time.perf_counter() - t0, 2)}
        loss = got["float32"]["si_sdr_db"] - got["bfloat16"]["si_sdr_db"]
        out["dtype_check"] = {"block": k, **got, "bf16_loss_db": round(loss, 2),
                              "sep_dtype": "float32" if loss > DTYPE_MARGIN else "bfloat16"}
    out.update(floor_db=SI_SDR_FLOOR, english_max_db=ENGLISH_MAX_DB)
    return out


# ---- the dubbed MP4 ----------------------------------------------------------------------------------------------------
def boxes(path: Path) -> list[str]:
    """The MP4's top-level boxes, in order."""
    out = []
    with path.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        at = 0
        while at + 8 <= size:
            f.seek(at)
            n, kind = int.from_bytes(f.read(4), "big"), f.read(4).decode("latin-1")
            if n == 1:
                n = int.from_bytes(f.read(8), "big")
            out.append(kind)
            if n < 8:
                break
            at += n
    return out


def streams(path: Path) -> list[dict]:
    import av

    out = []
    with av.open(str(path)) as c:
        for s in c.streams:
            d = {"index": s.index, "type": s.type, "codec": s.codec_context.name,
                 "language": s.metadata.get("language"), "title": s.metadata.get("title") or s.metadata.get("handler_name"),
                 "default": bool(s.disposition & av.stream.Disposition.default)}
            if s.type == "audio":
                d.update(sample_rate=s.codec_context.sample_rate, channels=s.codec_context.channels)
            elif s.type == "video":
                d.update(width=s.codec_context.width, height=s.codec_context.height)
            out.append(d)
    return out


def _video_packets(path: Path) -> list[str]:
    import av

    with av.open(str(path)) as c:
        return [hashlib.sha256(bytes(p)).hexdigest() for p in c.demux(c.streams.video[0]) if p.size]


def white_frame(path: Path) -> float | None:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        h = s.codec_context.height
        return next((float(f.time) for f in c.decode(s) if f.to_ndarray(format="yuv420p")[:h].mean() > 220), None)


def beep_onset(x: np.ndarray, near: float, hz: float = BEEP[0], window: float = 0.02) -> float | None:
    """When the `hz` beep starts in mono `x` (SR), within 1 s of `near`: a centred `window` s average of x·e^(-iωt)
    (a band about 1/window wide) first over half its peak there. None when nothing stands out."""
    a, b = max(0, int((near - 1.0) * SR)), min(len(x), int((near + 1.0) * SR))
    seg = x[a:b].astype(np.float64)
    t = (np.arange(len(seg)) + a) / SR
    w = max(1, int(round(window * SR)))
    mag = np.abs(np.convolve(seg * np.exp(-2j * np.pi * hz * t), np.ones(w) / w, mode="same"))
    if not len(mag) or mag.max() < 4 * np.median(mag):
        return None
    return (a + int(np.argmax(mag > mag.max() / 2))) / SR


def watermark(x: np.ndarray, spans: list[tuple[float, float]]) -> float | None:
    """resemble-perth's detector over `spans` of mono `x` (SR): its mean confidence (0: none, 1: watermarked)."""
    try:
        import perth
    except ImportError:
        return None
    det = perth.PerthImplicitWatermarker()
    got = [det.get_watermark(x[int(a * SR):int(b * SR)], sample_rate=SR, round=False) for a, b in spans if b - a >= 1.0]
    return round(float(np.mean(np.concatenate([np.ravel(g) for g in got]))), 3) if got else None


def check_export(work: Path, mp4: Path, manifest: Path) -> dict:
    """The dubbed MP4 against the test video: one H.264 video copied packet for packet, one Telugu AAC (default), two
    mov_text tracks (tel, eng), moov before mdat; its loudness; the A/V offset (the white frame against the beep's
    onset in the Telugu track, where the bed carries it); the PerTh watermark under the Telugu lines, against the
    source's audio as the control."""
    from maata_engine import export

    got = streams(mp4)
    kinds = [s["type"] for s in got]
    audio = [s for s in got if s["type"] == "audio"]
    subs = [s for s in got if s["type"] == "subtitle"]
    top = boxes(mp4)
    copied = _video_packets(mp4) == _video_packets(work / "test.mp4")
    layout_ok = (kinds.count("video") == 1 and [s["codec"] for s in got if s["type"] == "video"] == ["h264"] and copied
                 and len(audio) == 1 and audio[0]["codec"] == "aac" and audio[0]["language"] == "tel"
                 and audio[0]["default"] and audio[0]["sample_rate"] == SR and audio[0]["channels"] == 2
                 and sorted((s["codec"], s["language"]) for s in subs) == [("mov_text", "eng"), ("mov_text", "tel")]
                 and "moov" in top and "mdat" in top and top.index("moov") < top.index("mdat"))
    loud = export._measure(mp4)
    loud_ok = abs(loud["I"] - LUFS) <= LUFS_TOL and loud["TP"] is not None and loud["TP"] <= TP_MAX
    dub = np.mean(_decode(mp4), axis=0)
    flash = white_frame(mp4)
    onset = beep_onset(dub, MARK)
    offset = None if flash is None or onset is None else round(1000.0 * (onset - flash), 2)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    spans = [(x["start"], x["start"] + x["audioWall"]) for x in m["lines"]]
    source = np.mean(_decode(work / "test.mp4"), axis=0)
    out = {"streams": got, "boxes": top[:6], "video_copied": copied, "layout_ok": layout_ok, "loudness": loud,
           "loudness_ok": loud_ok, "flash_s": flash, "beep_onset_s": onset, "av_offset_ms": offset,
           "av_ok": offset is not None and abs(offset) <= AV_MAX_MS,
           "watermark": {"telugu_lines": watermark(dub, spans), "source_control": watermark(source, spans)},
           "lines": len(m["lines"]), "skipped": len(m["skipped"])}
    wm = out["watermark"]
    out["watermark_ok"] = None if wm["telugu_lines"] is None else (wm["telugu_lines"] >= 0.5 > (wm["source_control"] or 0))
    return out


# ---- the bench's one way out: the Claude CLI to Anthropic ---------------------------------------------------------------
ALLOWED = (".anthropic.com", "claude.ai")  # host suffixes the proxy tunnels to (port 443 only)


def allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == s.lstrip(".") or host.endswith(s if s.startswith(".") else "." + s) for s in ALLOWED)


async def proxy(work: Path) -> None:
    """An HTTP CONNECT proxy on 127.0.0.1 (its port in WORK/proxy.port) that tunnels to `allowed` hosts on 443 and
    refuses everything else; each request goes to WORK/proxy.log as {host, port, tunnelled}. The sandboxed bench reaches
    the network only through it (HTTPS_PROXY), so only the Claude CLI's calls to Anthropic leave the Mac."""
    import asyncio

    log = (work / "proxy.log").open("a", encoding="utf-8")

    async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while data := await r.read(65536):
                w.write(data)
                await w.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            w.close()

    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            head = await r.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            w.close()
            return
        method, target, *_ = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ") + ["", ""]
        host, _, port = target.rpartition(":") if method == "CONNECT" else (target, "", "")
        ok = method == "CONNECT" and port == "443" and allowed(host)
        log.write(json.dumps({"host": host or target, "port": port, "tunnelled": ok}) + "\n")
        log.flush()
        if not ok:
            w.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await w.drain()
            w.close()
            return
        try:
            ur, uw = await asyncio.open_connection(host, 443)
        except OSError:
            w.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            await w.drain()
            w.close()
            return
        w.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await w.drain()
        await asyncio.gather(pipe(r, uw), pipe(ur, w))

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    (work / "proxy.port").write_text(str(server.sockets[0].getsockname()[1]))
    async with server:
        await server.serve_forever()


def combine(work: Path, dest: Path) -> int:
    """One results file from the steps' JSON (video, separation, pipeline, export), the proxy's log and the checks.
    Returns 1 when a check failed, else 0."""
    def read(name: str) -> dict | None:
        p = work / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    sep, exp, pipe = read("separation"), read("export"), read("pipeline")
    log = work / "proxy.log"
    reqs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.is_file() else []
    checks = {"si_sdr": sep and sep["si_sdr_ok"], "english_left": sep and sep["english_ok"],
              "pipeline_done": pipe is not None, "layout": exp and exp["layout_ok"], "loudness": exp and exp["loudness_ok"],
              "av_offset": exp and exp["av_ok"], "watermark": exp and exp["watermark_ok"]}
    out = {"stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "video": read("video"), "separation": sep,
           "pipeline": pipe, "export": exp,
           # (what the proxy refused never left the Mac: the sandbox lets the bench reach only the proxy)
           "network": {"sandbox": "deny network-outbound but 127.0.0.1", "proxy_allows": list(ALLOWED),
                       "tunnelled": sorted({r["host"] for r in reqs if r["tunnelled"]}),
                       "refused": sorted({f"{r['host']}:{r['port']}" for r in reqs if not r["tunnelled"]})},
           "checks": {k: bool(v) for k, v in checks.items()},
           "sep_dtype": (sep or {}).get("dtype_check", {}).get("sep_dtype")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["checks"], indent=2))
    return 0 if all(out["checks"].values()) else 1


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("video", "separation", "export", "proxy", "combine"):
        print(__doc__, file=sys.stderr)
        return 2
    step, work = argv[0], Path(argv[1])
    work.mkdir(parents=True, exist_ok=True)
    if step == "proxy":
        import asyncio

        asyncio.run(proxy(work))
        return 0
    if step == "combine":
        return combine(work, Path(argv[2]))
    if step == "video":
        got = make_video(work, [Path(p) for p in argv[2:]], float(os.environ.get("MAATA_VERIFY_SECONDS", "120")))
    elif step == "separation":
        from maata_engine.backends.apple import MLXMelRoFormerSeparator

        folder = Path(argv[2]).expanduser() / "mel-roformer-kim-vocal-2-mlx"
        sep = MLXMelRoFormerSeparator(folder)  # SEP_DTYPE, as the separate stage runs it
        if sep.missing():
            raise SystemExit(f"the separator's model isn't in {folder}: run `uv run maata-bench fetch --backend apple`")
        pair = {d: sep if sep.dtype == d else MLXMelRoFormerSeparator(folder, d) for d in ("bfloat16", "float32")}
        got = separation(work, sep, pair)
    else:
        got = check_export(work, Path(argv[2]), Path(argv[3]))
    (work / f"{step}.json").write_text(json.dumps(got, indent=2), encoding="utf-8")
    print(json.dumps(got, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
