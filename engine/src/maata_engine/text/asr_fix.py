"""Repair Whisper's split numbers in ASR output (architect S2).

Whisper sometimes emits a space inside a number, so its words read "17" ".72", "3" ",000" or "50" "%".
Joined, that is "17 .72": the translator reads two numbers and the dub says them. This rejoins such
pieces, conservatively. A word joins the one before only when it starts with the separator itself
(no space after it) and the result is an unambiguous number:

- "17 .72"  -> "17.72"    a decimal point, after a whole number
- "3 ,000"  -> "3,000"    a thousands group of three digits (or Indian "1 ,00,000" -> "1,00,000")
- "10 :30"  -> "10:30"    minutes or seconds: two digits
- "50 %"    -> "50%"
- "$ 5"     -> "$5"       a lone currency sign before a number

A space after the separator ("in 2017. 72 people", "items 3, 400") is how sentences and lists are
written, so it is left alone, as is anything not number-like ("A4 .5", "2017 ,000", "1.2 .3").

Pure logic (ADR-005).
"""

from __future__ import annotations

import re

from ..types import TimedWord

_PRE = r"[(\[\"'“‘+\-−~$₹€£¥]*"                 # opening marks, signs and currency before a number
_INT = rf"^{_PRE}(?:\d+|\d{{1,3}}(?:,\d{{2,3}})+)$"   # 17, 3,000, 1,00,000
_LEFT = {
    "decimal": re.compile(_INT),
    "group": re.compile(rf"^{_PRE}\d{{1,3}}(?:,\d{{2,3}})*$"),
    "clock": re.compile(rf"^{_PRE}\d{{1,2}}(?::\d{{2}})?$"),
    "percent": re.compile(rf"^{_PRE}\d+(?:[.,]\d+)*$"),
}
_RIGHT = {
    "decimal": re.compile(r"^\.\d"),
    "group": re.compile(r"^,(?:\d{2},)*\d{3}(?!\d)"),
    "clock": re.compile(r"^:\d{2}(?!\d)"),
    "percent": re.compile(r"^%"),
}
_CURRENCY = re.compile(r"^[(\[\"'“‘]*[$₹€£¥]$")
_SPACE = re.compile(r"(\s+)")


def joins(left: str, right: str) -> bool:
    """Whether ASR word `right` is the split-off rest of the number ending word `left`."""
    if _CURRENCY.match(left):
        return right[:1].isdigit()
    return any(_RIGHT[k].match(right) and _LEFT[k].match(left) for k in _RIGHT)


def fix_text(text: str) -> str:
    """`text` with split numbers rejoined across a run of spaces or tabs; everything else unchanged."""
    parts = _SPACE.split(text)
    out: list[str] = []
    cur = parts[0]
    for gap, word in zip(parts[1::2], parts[2::2]):
        if "\n" not in gap and joins(cur, word):
            cur += word
        else:
            out += [cur, gap]
            cur = word
    out.append(cur)
    return "".join(out)


def fix_words(words: list[TimedWord]) -> list[TimedWord]:
    """Timed ASR words with split numbers rejoined: a joined word spans its pieces and keeps the lowest
    confidence among them. Returns a new list; the input is not changed."""
    out: list[TimedWord] = []
    for w in words:
        if out and joins(out[-1].text, w.text):
            prev = out[-1]
            out[-1] = TimedWord(prev.text + w.text, prev.start, max(prev.end, w.end), min(prev.confidence, w.confidence))
        else:
            out.append(w)
    return out
