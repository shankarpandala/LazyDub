"""The dubbed MP4's two soft subtitle tracks (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §2.16), pure: Telugu at the
dub's times (each voiced line's wording over its audio) and English at the source's (every transcript unit over its
words), shaped into cues of at most SUB_ROWS rows a speaker, overlapping cues merged into one sample, and written as
tx3g (mov_text) samples. Times are seconds on whatever clock the caller gives (the export's: the output's)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .text.akshara import count_units

SUB_ROWS = 2      # rows a speaker's cue may have
SUB_ROW = 42      # characters (English) or aksharas (Telugu) a row may have
SUB_MIN = 0.7     # s a cue lasts at least, where the next cue allows
DASH = "– "       # in front of each speaker's text while two or more show at once

Width = Callable[[str], float]  # a word's width; a row's is its words' and one for each space between them


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    speaker: str
    text: str     # its rows, joined by "\n"


def width_en(word: str) -> float:
    """An English word's width: its characters."""
    return float(len(word))


def width_te(word: str) -> float:
    """A Telugu word's width: its aksharas (a conjunct is up to three code points, `text/akshara`)."""
    return count_units(word)


def _row(ws: Sequence[float]) -> float:
    return sum(ws) + max(len(ws) - 1, 0)


def rows(words: Sequence[str], width: Width) -> list[str] | None:
    """`words` laid out as one row when they fit SUB_ROW, else two broken at the space nearest the middle of those where
    both rows fit; None when no break fits (a single word too wide is a row of its own anyway)."""
    ws = [width(w) for w in words]
    whole = _row(ws)
    if whole <= SUB_ROW or len(words) == 1:
        return [" ".join(words)]
    fits = [(abs(_row(ws[:k]) - whole / 2), k) for k in range(1, len(words))
            if _row(ws[:k]) <= SUB_ROW and _row(ws[k:]) <= SUB_ROW]
    if not fits:
        return None
    _, k = min(fits)
    return [" ".join(words[:k]), " ".join(words[k:])]


def pieces(words: Sequence[str], width: Width) -> list[list[str]]:
    """`words` split at word boundaries into consecutive pieces that each fit SUB_ROWS rows: each piece takes as many
    words as fit."""
    out: list[list[str]] = []
    for w in words:
        if out and rows(out[-1] + [w], width) is not None:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def telugu_cues(lines: Iterable[dict], sample_rate: int, offset: float = 0.0) -> list[Cue]:
    """The Telugu cues: each voiced line's wording (the manifest's `telugu`, Telugu script, the wording voiced) over its
    audio, [start, start + samples / sample_rate] + `offset`, split into pieces each timed by its share of the line's
    aksharas."""
    out: list[Cue] = []
    for line in lines:
        words = str(line.get("telugu") or "").split()
        if not words:
            continue
        start = float(line["start"]) + offset
        span = int(line["samples"]) / sample_rate
        parts = pieces(words, width_te)
        weights = [max(count_units(" ".join(p)), 0.0) for p in parts]
        if sum(weights) <= 0:
            weights = [float(len(" ".join(p))) for p in parts]
        total, t = sum(weights), start
        for p, wt in zip(parts, weights):
            end = t + span * wt / total
            out.append(Cue(t, end, str(line["speaker"]), "\n".join(rows(p, width_te) or [" ".join(p)])))
            t = end
    return out


def english_cues(units: Iterable[tuple[str, Sequence[tuple[str, float, float]]]], offset: float = 0.0) -> list[Cue]:
    """The English cues: every transcript unit, (speaker, its words as (text, start, end)), over its words' times +
    `offset`, split into pieces each timed by its own words."""
    out: list[Cue] = []
    for speaker, words in units:
        words = [(str(t), float(a), float(b)) for t, a, b in words if str(t).strip()]
        k = 0
        for p in pieces([t for t, _, _ in words], width_en):
            got = words[k:k + len(p)]
            k += len(p)
            out.append(Cue(got[0][1] + offset, got[-1][2] + offset, speaker, "\n".join(rows(p, width_en) or [" ".join(p)])))
    return out


def shape(cues: Iterable[Cue], stop: float | None = None, floor: float | None = None) -> list[Cue]:
    """The cues in time order, each lasting at least SUB_MIN s where the next cue allows (up to its start), with `stop`
    (a preview's end) none past it, and with `floor` (the file's 0) none before it: a cue that began earlier shows from
    it, one that ended by then is left out."""
    if floor is not None:
        cues = [Cue(max(c.start, floor), c.end, c.speaker, c.text) for c in cues if c.end > floor]
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    out = []
    for i, c in enumerate(cues):
        end = c.end
        if end - c.start < SUB_MIN:
            nxt = next((x.start for x in cues[i + 1:] if x.start > c.start), None)
            end = max(end, c.start + SUB_MIN if nxt is None else min(c.start + SUB_MIN, nxt))
        if stop is not None:
            end = min(end, stop)
        if end > c.start:
            out.append(Cue(c.start, end, c.speaker, c.text))
    return out


def merge(cues: Sequence[Cue]) -> list[tuple[float, float, str]]:
    """One sample at a time (tx3g shows one): time is cut at every cue's start and end, and each piece shows every cue
    active in it, one block per speaker (in the order they began), each block prefixed DASH when there are two or
    more. A piece the same as the one before it, and touching it, extends it."""
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    times = sorted({t for c in cues for t in (c.start, c.end)})
    out: list[tuple[float, float, str]] = []
    active: list[Cue] = []
    k = 0
    for a, b in zip(times, times[1:]):
        active = [c for c in active if c.end > a]
        while k < len(cues) and cues[k].start <= a:
            if cues[k].end > a:
                active.append(cues[k])
            k += 1
        if not active:
            continue
        blocks: dict[str, list[str]] = {}
        for c in active:
            blocks.setdefault(c.speaker, []).append(c.text)
        texts = ["\n".join(v) for v in blocks.values()]
        text = "\n".join(DASH + t for t in texts) if len(texts) > 1 else texts[0]
        if out and out[-1][2] == text and out[-1][1] == a:
            out[-1] = (out[-1][0], b, text)
        else:
            out.append((a, b, text))
    return out


def tx3g(text: str) -> bytes:
    """A tx3g (mov_text) sample: the text's UTF-8 length as 2 bytes big-endian, then the text."""
    data = text.encode("utf-8")
    return struct.pack(">H", len(data)) + data


def samples(cues: Iterable[Cue], stop: float | None = None) -> list[tuple[int, int, bytes]]:
    """A track's samples, (start ms, duration ms, tx3g bytes), from its cues on the file's clock: shaped (none before
    the file's 0, where the video starts after the audio: a sample before it would put the whole track out of time),
    merged; the empty samples between them are libavformat's to write."""
    out = []
    for a, b, text in merge(shape(cues, stop, floor=0.0)):
        start, end = round(a * 1000), round(b * 1000)
        if end > start:
            out.append((start, end - start, tx3g(text)))
    return out
