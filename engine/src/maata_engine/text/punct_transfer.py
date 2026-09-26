"""Copy punctuation and casing from a punctuation-prompted Whisper pass onto the unprompted one (ARCHITECTURE §3.4).

Whisper large-v3-turbo drops punctuation and casing over long stretches, and the segmenter needs them to find sentence
ends (docs/research/dubbing-2026-09/gap-3.md E1). Prompting Whisper with punctuated text brings them back but also drops
whole passages of speech (gap-3 E2), so the prompted pass is never the transcript. Its words are aligned to the
unprompted pass's by `difflib`, and each matched word gives only its first letter's case and its trailing sentence or
clause marks. Every word, timestamp and confidence of the unprompted pass is kept, and no word is added. The prompted
pass runs per 30 s window, and where a window is cut into speech it guesses a sentence edge; `unguess` drops that guess.

Pure logic (ADR-005).
"""

from __future__ import annotations

import difflib
import re

from ..types import TimedWord

MARKS = ".,?!;:…"          # trailing marks the prompted pass may give (the ones the segmenter reads)
_CORE = re.compile(r"[^\w']+")


def _core(text: str) -> str:
    """What the two passes are aligned on: the word lowercased, without punctuation ("Don't," -> "don't")."""
    return _CORE.sub("", text.replace("’", "'").lower())


def _first_letter(text: str) -> int | None:
    return next((i for i, ch in enumerate(text) if ch.isalpha()), None)


def _cased(word: str, like: str) -> str:
    """`word` with its first letter in the case of `like`'s. A word's own capital stays when it is always written so:
    the pronoun I (and I'm, I'll) and acronyms (AI, GPU)."""
    i, j = _first_letter(word), _first_letter(like)
    if i is None or j is None:
        return word
    ch, rest = word[i], word[i:]
    acronym = sum(c.isalpha() for c in rest) >= 2 and rest.upper() == rest
    if like[j].isupper():
        ch = ch.upper()
    elif like[j].islower() and not (acronym or _core(word).split("'")[0] == "i"):
        ch = ch.lower()
    return word[:i] + ch + word[i + 1:]


def _marked(word: str, like: str) -> str:
    """`word` ending in `like`'s trailing marks when it has any (they replace `word`'s own); else unchanged."""
    marks = like[len(like.rstrip(MARKS)):]
    return word.rstrip(MARKS) + marks if marks else word


def transfer(words: list[TimedWord], styled: list[TimedWord], max_shift: float = 1.0) -> list[TimedWord]:
    """`words` (the unprompted pass) with the casing and trailing marks of the words of `styled` (a punctuation-prompted
    pass over the same audio, on the same clock) aligned to them. A matched pair whose midpoints lie more than
    `max_shift` s apart is a chance match between passes that diverged, and copies nothing. Returns a new list of the
    same words."""
    out = list(words)
    a, b = [_core(w.text) for w in words], [_core(w.text) for w in styled]
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            w, p = words[block.a + k], styled[block.b + k]
            if not a[block.a + k] or abs((w.start + w.end) - (p.start + p.end)) / 2 > max_shift:
                continue
            text = _marked(_cased(w.text, p.text), p.text)
            if text != w.text:
                out[block.a + k] = TimedWord(text, w.start, w.end, w.confidence)
    return out


def unguess(words: list[TimedWord], start: float | None, end: float | None,
            edge: float = 1.0) -> tuple[list[TimedWord], int]:
    """One prompted window's words (in time order) without its guesses at the cuts into speech that bound it: Whisper
    ends a window cut mid-sentence with a mark and opens the next with a capital. So the last word's trailing marks go
    when it ends within `edge` s of `end`, and the first word is left out (it would lend its capital) when it is
    capitalised, starts within `edge` of `start` and isn't the pronoun I. None: that side is the audio's own edge, not a
    cut. Everything else stays, a real sentence end near a cut included. Returns the words and how many guesses went."""
    out, n = list(words), 0
    if out and end is not None and end - out[-1].end <= edge:
        w = out[-1]
        bare = w.text.rstrip(MARKS)
        if bare != w.text:
            out[-1], n = TimedWord(bare, w.start, w.end, w.confidence), n + 1
    if out and start is not None and out[0].start - start <= edge:
        text = out[0].text
        i = _first_letter(text)
        if i is not None and text[i].isupper() and _core(text).split("'")[0] != "i":
            out, n = out[1:], n + 1
    return out, n
