"""Segmenter (§6.2): turn timed words + speaker turns into dubbing units.

Split at sentence ends, speaker changes and pauses > ~300 ms; aim for 1.5–12 s; merge
fragments; drop short backchannels ("yeah", "mm-hmm") unless they carry meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import SourceUnit, SpeakerTurn, TimedWord

_SENTENCE_END = (".", "?", "!", "…", "।")
_CLAUSE_END = (",", ";", ":", "—", "-")
_BACKCHANNELS = {"yeah", "yes", "yep", "mm", "mhm", "mm-hmm", "uh-huh", "right", "okay", "ok", "hmm", "uh", "um", "ah", "oh", "wow"}


@dataclass(frozen=True, slots=True)
class SegmenterSettings:
    pause_split: float = 0.30
    min_len: float = 1.5
    max_len: float = 12.0
    backchannel_max: float = 0.6


def speaker_at(turns: list[SpeakerTurn], t: float) -> str:
    best, best_overlap = "S1", -1.0
    for turn in turns:
        if turn.start <= t <= turn.end:
            ov = turn.end - turn.start
            if ov > best_overlap:  # dominant (longest) turn wins on overlap
                best, best_overlap = turn.speaker, ov
    return best


def _is_backchannel(words: list[TimedWord], s: SegmenterSettings) -> bool:
    if not words or words[-1].end - words[0].start > s.backchannel_max:
        return False
    return all(w.text.lower().strip(".,!?") in _BACKCHANNELS for w in words)


def segment(words: list[TimedWord], turns: list[SpeakerTurn], s: SegmenterSettings = SegmenterSettings()) -> list[SourceUnit]:
    groups: list[tuple[str, list[TimedWord]]] = []
    cur: list[TimedWord] = []
    cur_spk = ""
    for w in words:
        spk = speaker_at(turns, (w.start + w.end) / 2)
        if cur:
            prev = cur[-1]
            dur = w.end - cur[0].start
            hard = spk != cur_spk or dur > s.max_len
            soft = prev.text.endswith(_SENTENCE_END) or (w.start - prev.end) > s.pause_split
            if hard or (soft and prev.end - cur[0].start >= s.min_len):
                groups.append((cur_spk, cur))
                cur = []
            elif dur > s.max_len * 0.75 and prev.text.endswith(_CLAUSE_END):
                groups.append((cur_spk, cur))
                cur = []
        if not cur:
            cur_spk = spk
        cur.append(w)
    if cur:
        groups.append((cur_spk, cur))

    # Merge short fragments into a neighbour of the same speaker when the gap is small.
    merged: list[tuple[str, list[TimedWord]]] = []
    for spk, ws in groups:
        if (
            merged
            and merged[-1][0] == spk
            and (ws[-1].end - ws[0].start < s.min_len or merged[-1][1][-1].end - merged[-1][1][0].start < s.min_len)
            and ws[0].start - merged[-1][1][-1].end <= 1.0
            and ws[-1].end - merged[-1][1][0].start <= s.max_len
            and not merged[-1][1][-1].text.endswith(_SENTENCE_END[1:3])
        ):
            merged[-1] = (spk, merged[-1][1] + ws)
        else:
            merged.append((spk, ws))

    units: list[SourceUnit] = []
    for spk, ws in merged:
        if _is_backchannel(ws, s):
            continue
        units.append(
            SourceUnit(id=len(units), speaker=spk, start=ws[0].start, end=ws[-1].end, text=" ".join(w.text for w in ws).strip(), words=ws)
        )
    return units
