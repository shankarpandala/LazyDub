"""The dubbed MP4 (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §2.17), muxed with PyAV only (no ffmpeg binary): the
video stream copied packet for packet, the Telugu mix (`mix.py`: the voices over the ducked background sound) as AAC,
and the two subtitle tracks (`subtitles.py`) as
mov_text, interleaved per 10 s block into <folder>/.<name>.part.mp4, then put in place under its name. It runs in a worker
thread and stops at the next block when its cancel event is set, deleting the partial file.

Clocks: a dub time t (the analysis audio's, whose sample 0 is the source audio's first decoded sample, `source.audioStart`
in its stream) plays at t + audioStart - videoStart in the file, whose 0 is the video stream's first frame."""

from __future__ import annotations

import contextlib
import math
import os
import re
import struct
import threading
import unicodedata
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from . import mix, subtitles
from .backends.base import Cancelled
from .subtitles import Cue

EXPORT_VERSION = 2       # bumped whenever how the file is made changes (§12): 2, the background sound under the voices
AUDIO_BITRATE = 192_000  # AAC-LC, 44.1 kHz stereo
AAC_OPTIONS: dict[str, str] = {}  # the encoder's own options: none, so AudioToolbox runs at its best quality
OUTPUT_PER_HOUR = 0.1e9  # bytes of the output's volume per hour of video, beside the video's own size (§2.17)
NAME_BYTES = 150         # the title's share of a file name, in UTF-8 bytes
FINISHING = "Finishing the file…"  # the stage's extra while faststart rewrites the file (it can't be paused)
CHECKING = "Checking audio quality…"  # the encoded AAC's loudness and true peak, with per-block pause checks
VP9 = "QuickTime can't play VP9; open it in IINA or VLC."
NO_BED = "No background sound: the file has the Telugu voices only."
# A minimal ASS header: without one, a mov_text encoder doesn't open (`avcodec_open2` fails).
SUB_HEADER = (b"[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n[V4+ Styles]\n"
              b"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
              b"Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
              b"MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,"
              b"10,10,10,0\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f\x7f]')

Progress = Callable[[float, float, "str | None"], None]  # (seconds done, seconds in all, the stage's extra)


def aac() -> str:
    """Apple's AudioToolbox AAC encoder where PyAV has it (macOS), else FFmpeg's."""
    import av

    return "aac_at" if "aac_at" in av.codecs_available else "aac"


# ---- the file's name --------------------------------------------------------------------------------------------------
def sanitize(title: str, fallback: str) -> str:
    """A title as a file name's start: `/ \\ : * ? " < > |` and control characters become spaces, runs of spaces
    collapse, leading dots go, at most NAME_BYTES of UTF-8 (cut at a character); `fallback` when nothing is left."""
    name = re.sub(r"\s+", " ", _UNSAFE.sub(" ", unicodedata.normalize("NFC", title))).strip().lstrip(".").strip()
    data = name.encode("utf-8")[:NAME_BYTES]
    name = data.decode("utf-8", errors="ignore").strip()
    return name or fallback


def output_name(title: str, fallback: str, stop: float | None) -> str:
    """`<title> (Telugu).mp4`, or for a preview `<title> (Telugu, first 15 min).mp4`."""
    label = "Telugu" if stop is None else f"Telugu, first {max(1, round(stop / 60))} min"
    return f"{sanitize(title, fallback)} ({label}).mp4"


def output_path(folder: Path, name: str, own: Path | None) -> Path:
    """Where the file goes: the job's own earlier output (`own`: the caller has checked it is still the file the job
    wrote) when it is in `folder` under `name` or a numbered form of it, replaced; else `name` in `folder`, or with
    " (2)", " (3)", ... while a file has the name (anyone else's: never replaced)."""
    stem, suffix = os.path.splitext(name)
    if own is not None and own.parent.resolve() == folder.resolve() \
            and re.fullmatch(re.escape(stem) + r"( \(\d+\))?" + re.escape(suffix), own.name):
        return own
    for n in range(1, 10_000):
        path = folder / (name if n == 1 else f"{stem} ({n}){suffix}")
        if not path.exists():
            return path
    raise FileExistsError(f"Too many files named like {name} in {folder}.")


def part_path(dest: Path) -> Path:
    """The file being written: `.<name>.part.mp4` beside where it goes, so a half-written file never looks finished."""
    return dest.with_name(f".{dest.stem}.part.mp4")


# ---- the video stream -------------------------------------------------------------------------------------------------
def probe(path: Path) -> dict:
    """The video stream's codec, size, frame rate, start time (s) and duration (s), read with PyAV."""
    import av

    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError(f"{path.name} has no video stream.")
        s = c.streams.video[0]
        tb = s.time_base
        start = float((s.start_time or 0) * tb)
        duration = float(s.duration * tb) if s.duration else float(c.duration or 0) / 1e6 - start
        rate = s.average_rate or s.guessed_rate
        return {"codec": s.codec_context.codec.canonical_name, "width": s.codec_context.width,
                "height": s.codec_context.height, "fps": round(float(rate), 3) if rate else None,
                "start": round(start, 6), "duration": round(duration, 6)}


def _packets(path: Path) -> Iterator[tuple[object, object]]:
    """The video stream's packets in decode order (no flush packets), with the stream."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for p in c.demux(s):
            if p.size:
                yield p, s


def keyframe_at(path: Path, at: float) -> tuple[int, float] | None:
    """The first keyframe, in decode order, whose time (from the stream's start) is `at` s or later: its place among the
    packets and its time; None when there is none (the cut would be past the end)."""
    for n, (p, s) in enumerate(_packets(path)):
        if p.is_keyframe and p.pts is not None and float((p.pts - (s.start_time or 0)) * s.time_base) >= at - 1e-6:
            return n, float((p.pts - (s.start_time or 0)) * s.time_base)
    return None


# ---- the export -------------------------------------------------------------------------------------------------------
def export(dest: Path, *, video: Path, lines: Sequence[mix.Line], voice_sr: int, telugu: Sequence[Cue],
           english: Sequence[Cue], audio_start: float, stop: float | None, metadata: dict[str, str],
           bed: mix.Background | None = None, cancel: threading.Event | None = None,
           progress: Progress | None = None) -> dict:
    """Write the dubbed MP4 to `dest`, through `part_path(dest)`. `lines`, `telugu`, `english`, `bed` (the background
    sound; None: the voices alone, with a warning) and `stop` (a preview's stop point; None: the whole video) are on the
    dub's clock. The bed sits under the voices at the balance `mix.bed_gain` gives (after each speaker's gain), ducked
    by its envelope. A preview ends at the first keyframe at or after
    max(stop, the end of the last line's audio), so no line is cut mid-word and the copied video needs no re-encode.
    The audio spans the video (or the preview), and a last line that runs past the video's end. Its loudness is two
    passes over the mix: one measures it, the other sets it to MIX_LUFS, limits it and encodes it; then a decode of the
    AAC written measures what the file plays (`loudness`). Returns {bytes, seconds, loudness, warning}. A set `cancel`
    stops it at the next block (`Cancelled`); a pause or a failure deletes the partial file. Also returns `bedGain`, the
    bed's balance in dB (None without a bed)."""
    import av

    part = part_path(dest)
    info = probe(video)
    off = audio_start - info["start"]
    lines = [replace(x, start=x.start + off) for x in lines]
    ends = [x.start + x.samples / voice_sr for x in lines]
    cut = None
    if stop is not None:
        cut = keyframe_at(video, max([stop + off] + ends))
    end = cut[1] if cut is not None else max([info["duration"]] + ends)
    warnings = ([VP9] if info["codec"] == "vp9" else []) + ([NO_BED] if bed is None else [])
    tracks = [subtitles.samples([replace(c, start=c.start + off, end=c.end + off) for c in cues], end)
              for cues in (telugu, english)]

    def report(done: float, extra: str | None = None) -> None:
        if progress is not None:
            progress(round(done, 1), round(end, 1), extra)

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled("export paused")

    def measured(seconds: float) -> None:
        # AAC padding may extend past `end`. Keep the displayed count below 100% until the meter has flushed too.
        report(min(2 * end / 3 + min(seconds, end) / 3, max(0.0, round(end, 1) - 0.1)), CHECKING)

    try:
        # Pass 1: each speaker's gain, the bed's balance, then the mix's integrated loudness (a third of the progress).
        gains = mix.speaker_gains(lines, voice_sr)
        bed_db = mix.bed_gain(lines, gains, bed.vocals_ms) if bed is not None else None
        under = bed.on(off, bed_db) if bed is not None else None
        meter, done = mix.Loudness(mix.MIX_SR, 2), 0
        for block in mix.Mix(lines, end, voice_sr, gains, under).blocks():
            check()
            meter.add(block)
            done += block.shape[1]
            report(end * done / max(1, round(end * mix.MIX_SR)) / 3)
        gain = mix.mix_gain(meter.result()["I"])
        # Pass 2: mastered, encoded and muxed beside the copied video and the cues, block by block.
        with av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            _mux(out, video, mix.master(mix.Mix(lines, end, voice_sr, gains, under).blocks(), gain), tracks, cut, end,
                 metadata, check, report)
            report(2 * end / 3, FINISHING)
        check()
        _subtitles_off(part)
        report(2 * end / 3, CHECKING)
        loudness = _measure(part, check=check, progress=measured)
        if loudness["TP"] is not None and loudness["TP"] > mix.TP_CEIL:
            warnings.append(f"The loudest peak is {loudness['TP']:.1f} dBTP, over {mix.TP_CEIL:.0f} dBTP.")
        size = part.stat().st_size
        check()  # a stopped export never puts a file in place (the job would not know it as its own)
        report(end, CHECKING)
        check()  # progress callbacks may themselves request a pause
        os.replace(part, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    return {"bytes": size, "seconds": round(end, 3), "loudness": loudness, "warning": " ".join(warnings) or None,
            "bedGain": None if bed_db is None else round(bed_db, 2)}


def _mux(out, video: Path, audio: Iterator[np.ndarray], tracks: list[list[tuple[int, int, bytes]]],
         cut: tuple[int, float] | None, end: float, metadata: dict[str, str], check: Callable[[], None],
         report: Callable[..., None]) -> None:
    """The streams in order (the video copied, the AAC, Telugu then English mov_text) and, per block of the mix, the
    video packets that start decoding before the block's end, the block's AAC packets and the cues that start in it."""
    import av

    out.metadata.update(metadata)
    packets = _packets(video)
    first = next(packets, None)
    if first is None:
        raise ValueError(f"{video.name} has no video frames.")
    vin = first[1]
    v = out.add_stream_from_template(vin, opaque=True)  # a stream copy: no encoder (AV1 decodes as libdav1d)
    a = out.add_stream(aac(), rate=mix.MIX_SR, options=dict(AAC_OPTIONS))
    a.layout, a.bit_rate = "stereo", AUDIO_BITRATE
    a.metadata["language"] = "tel"
    a.disposition = av.stream.Disposition.default
    subs = []
    for lang, title in (("tel", "Telugu"), ("eng", "English")):
        s = out.add_stream("mov_text")
        s.codec_context.subtitle_header = SUB_HEADER
        s.time_base = Fraction(1, 1000)
        s.metadata.update({"language": lang, "title": title, "handler_name": title})
        s.disposition = av.stream.Disposition(0)  # neither shows until picked in the player's menu (M12)
        subs.append(s)
    shift = vin.start_time or 0
    tb = vin.time_base
    pending: list = [first]
    n = 0  # packets copied, in decode order

    def copy_video(until: float) -> None:
        nonlocal n
        while True:
            got = pending.pop() if pending else next(packets, None)
            if got is None or (cut is not None and n >= cut[0]):
                return
            p = got[0]
            dts = p.dts if p.dts is not None else p.pts
            if float((dts - shift) * tb) >= until:
                pending.append(got)
                return
            p.pts = None if p.pts is None else p.pts - shift
            p.dts = None if p.dts is None else p.dts - shift
            p.stream = v
            out.mux(p)
            n += 1

    cues = [0, 0]

    def copy_cues(until_ms: int) -> None:
        for k, (s, track) in enumerate(zip(subs, tracks)):
            while cues[k] < len(track) and track[cues[k]][0] < until_ms:
                start, duration, data = track[cues[k]]
                p = av.Packet(data)
                p.pts = p.dts = start
                p.duration, p.time_base, p.stream = duration, Fraction(1, 1000), s
                out.mux(p)
                cues[k] += 1

    try:
        _blocks(out, a, audio, copy_video, copy_cues, end, check, report)
    finally:
        packets.close()  # (a preview's cut leaves the source open otherwise)


def _blocks(out, a, audio: Iterator[np.ndarray], copy_video: Callable[[float], None], copy_cues: Callable[[int], None],
            end: float, check: Callable[[], None], report: Callable[..., None]) -> None:
    import av

    done = 0
    for block in audio:
        check()
        until = (done + block.shape[1]) / mix.MIX_SR
        copy_video(until)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block, np.float32), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = mix.MIX_SR, done, Fraction(1, mix.MIX_SR)
        for p in a.encode(frame):
            out.mux(p)
        copy_cues(round(until * 1000))
        done += block.shape[1]
        report(end / 3 + until / 3)
    for p in a.encode(None):
        out.mux(p)
    copy_video(math.inf)
    copy_cues(2 ** 62)


def _boxes(f, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """The MP4 boxes in [start, end) of the open file `f`: (type, where its body starts, where it ends)."""
    out, pos = [], start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < head:
            break
        out.append((kind, pos + head, pos + size))
        pos += size
    return out


def _subtitles_off(path: Path) -> None:
    """Neither subtitle track shows until picked in the player's menu (§2.16, M12): libavformat's mp4 muxer enables the
    first track of each type when none is `default` (its `enable_tracks`), and QuickTime shows an enabled subtitle track
    at once, so the `enabled` flag of each subtitle track's header (tkhd) is cleared in place, in the moov faststart put
    at the front. Each export does the same, so an export made again stays the same bytes."""
    with open(path, "r+b") as f:
        moov = [(a, b) for kind, a, b in _boxes(f, 0, f.seek(0, os.SEEK_END)) if kind == b"moov"]
        for trak, a, b in _boxes(f, *moov[0]) if moov else ():
            inner = {kind: (x, y) for kind, x, y in _boxes(f, a, b)} if trak == b"trak" else {}
            hdlr = {kind: x for kind, x, _ in _boxes(f, *inner[b"mdia"])}.get(b"hdlr") if b"mdia" in inner else None
            if hdlr is None or b"tkhd" not in inner:
                continue
            f.seek(hdlr + 8)  # after the handler box's version, flags and pre_defined: its type
            if f.read(4) in (b"sbtl", b"text", b"subt"):
                at = inner[b"tkhd"][0] + 3  # the header's flags, last byte: 1 is "enabled"
                f.seek(at)
                flags = f.read(1)[0]
                f.seek(at)
                f.write(bytes([flags & ~1]))


def _measure(path: Path, *, check: Callable[[], None] | None = None,
             progress: Callable[[float], None] | None = None) -> dict:
    """{"I", "TP", "LRA"} of the file's AAC, decoded (AAC adds its own overshoot to the peaks), fed to the meter in
    BLOCK s chunks. `progress` receives seconds measured; `check` may cancel at each block boundary."""
    import av

    if check is not None:
        check()
    meter = mix.Loudness(mix.MIX_SR, 2, peak=True)
    measured = 0

    def add(frames: list[np.ndarray], n: int) -> None:
        nonlocal measured
        if check is not None:
            check()
        meter.add(np.concatenate(frames, axis=1))
        measured += n
        if progress is not None:
            progress(measured / mix.MIX_SR)
        if check is not None:
            check()

    with av.open(str(path)) as c:
        frames, n = [], 0
        for frame in c.decode(c.streams.audio[0]):
            frames.append(frame.to_ndarray().reshape(2, -1))
            n += frame.samples
            if n >= mix.BLOCK * mix.MIX_SR:
                add(frames, n)
                frames, n = [], 0
        if frames:
            add(frames, n)
    got = meter.result()
    return {"I": round(got["I"], 2), "TP": got["TP"], "LRA": round(got["LRA"], 2)}
