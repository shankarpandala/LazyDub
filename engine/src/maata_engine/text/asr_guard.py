"""Guards on the English ASR (ARCHITECTURE §3.4; docs/research/dubbing-2026-09/10-asr-and-qa.md R4).

- `uncovered`: diarized speech that got no words, which the transcriber decodes again and the trace logs, so dropped
  speech is seen rather than silently undubbed.
- `low_confidence` and `repeats`: stretches Whisper is unsure of, and phrases repeated back to back (its decoding loop,
  or a sung refrain).
- `music_like`: the interim music and singing flag (until an audio-event tagger is approved, decision D15).
- `vet_redecode`: which of the words a second decode of uncovered speech found may join the transcript.

Apart from `vet_redecode`, they only report; none of them drops, adds or changes a word. The thresholds are starting
points, not tuned on fixtures.

Pure logic (ADR-005).
"""

from __future__ import annotations

import re
import statistics

from ..types import SourceUnit, SpeakerTurn, TimedWord

UNCOVERED_MIN = 0.8   # s of diarized speech with no word in it that counts as dropped
LOW_PROB, LOW_RUN = 0.4, 3  # this many words in a row, each below this probability, are a low-confidence cluster
REPEAT_MAX_N, REPEAT_WORDS = 4, 6  # a phrase of up to 4 words said again back to back, over at least 6 words in all
MUSIC_MIN, MUSIC_CONF = 8.0, 0.6  # a sung-looking unit is this long, with its median word probability below this
NO_SPEECH = 0.6       # a re-decoded segment Whisper thinks likelier than this to hold no speech isn't trusted

Span = tuple[float, float]
_NOT_WORD = re.compile(r"[^\w']+")


def _core(text: str) -> str:
    return _NOT_WORD.sub("", text.lower())


def _merge(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


def uncovered(words: list[TimedWord], speech: list[Span], min_gap: float = UNCOVERED_MIN) -> list[Span]:
    """The stretches of `speech` (diarized speech spans, on the words' clock) that no word's [start, end] touches and
    that last more than `min_gap` s, in time order."""
    ws = sorted((w.start, w.end) for w in words)
    out: list[Span] = []
    for a, b in _merge(speech):
        t = a
        for s, e in ws:
            if e <= t:
                continue
            if s >= b:
                break
            if s - t > min_gap:
                out.append((t, s))
            t = max(t, e)
        if b - t > min_gap:
            out.append((t, b))
    return out


def low_confidence(words: list[TimedWord], below: float = LOW_PROB, run: int = LOW_RUN) -> list[Span]:
    """Spans of at least `run` consecutive words whose probability is below `below`."""
    out: list[Span] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].confidence < below:
            j += 1
        if j - i >= run:
            out.append((words[i].start, words[j - 1].end))
        i = max(j, i + 1)
    return out


def repeats(words: list[TimedWord], max_n: int = REPEAT_MAX_N, min_words: int = REPEAT_WORDS) -> list[Span]:
    """Spans where a phrase of up to `max_n` words is said again straight after itself, the repeats covering at least
    `min_words` words ("no no no no no no", a line looped by the decoder). Words compare lowercased, without marks."""
    cores = [_core(w.text) for w in words]
    out: list[Span] = []
    i = 0
    while i < len(cores):
        best = 0
        for n in range(1, max_n + 1):
            gram = cores[i:i + n]
            if len(gram) < n or not all(gram):
                continue
            times = 1
            while cores[i + times * n:i + (times + 1) * n] == gram:
                times += 1
            if times >= 2 and times * n >= min_words:
                best = max(best, times * n)
        if best:
            out.append((words[i].start, words[i + best - 1].end))
            i += best
        else:
            i += 1
    return out


def music_like(unit: SourceUnit, turns: list[SpeakerTurn]) -> bool:
    """Whether a unit looks sung rather than spoken (§3.4's interim heuristic): it is long, Whisper is unsure of its
    words, a phrase repeats back to back, and the diarizer (`turns`) hears no other speaker inside it. For the report
    only: a flagged line is still dubbed."""
    if unit.end - unit.start < MUSIC_MIN or not unit.words:
        return False
    if statistics.median(w.confidence for w in unit.words) >= MUSIC_CONF or not repeats(unit.words):
        return False
    return {t.speaker for t in turns if t.end > unit.start and t.start < unit.end} <= {unit.speaker}


def vet_redecode(found: list[TimedWord], no_speech: list[float], words: list[TimedWord], gap: Span,
                 pad: float) -> tuple[list[TimedWord], list[TimedWord]]:
    """Split what a second decode of `gap` found into the words that may join `words` (the first pass, in time order)
    and those that may not. `found`: the words inside the gap, in time order; `no_speech`: each one's segment's
    no-speech probability; `pad`: the s of audio decoded on each side of the gap.

    The clip is short and padded to 30 s, where Whisper invents text most readily ("Thank you." over a laugh), and its
    hallucination guard can't act in so little audio. So the clip is taken whole or not at all: not when Whisper
    thought any of it likelier silence than speech (over NO_SPEECH), was unsure of it (a median word probability under
    LOW_PROB) or looped (see `repeats`). A clip taken still loses a first (last) word that is the first pass's word
    just before (after) the gap, within `pad`: that word, heard again in the padding."""
    if not found:
        return [], []
    if max(no_speech) > NO_SPEECH or statistics.median(w.confidence for w in found) < LOW_PROB or repeats(found):
        return [], list(found)
    a, b = gap
    before = [w for w in words if a - pad <= w.end <= a]
    after = [w for w in words if b <= w.start <= b + pad]
    kept, dropped = list(found), []
    if before and _core(kept[0].text) == _core(before[-1].text):
        dropped.append(kept.pop(0))
    if kept and after and _core(kept[-1].text) == _core(after[0].text):
        dropped.append(kept.pop())
    return kept, dropped
