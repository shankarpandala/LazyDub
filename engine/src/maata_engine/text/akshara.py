"""Speaking-length units (spec §6.4, §6.6; docs/research/dubbing-2026-09/ARCHITECTURE.md §4.5).

A dub line is Telugu script only, English words included (the `spoken` field), so its length is counted in aksharas,
with explicit rules rather than grapheme clusters so word-final pollu and ZWNJ half-forms can be weighted (Swift-style
grapheme clustering counts them as full clusters). Digits are counted through their spoken Telugu form. English in Latin
letters is not a dub line's length: the Latin rebuild of a line (`tenglish.latin_spoken`) is measured by the Telugu-script
line it comes from, which predicts its duration better too (research gap-2 C). English source lines are counted in
syllables (`mixed_units`), for the prior on how long their Telugu will be (ARCHITECTURE §4.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize_te import normalize_telugu

VIRAMA = "్"
ZWJ, ZWNJ = "‍", "‌"


def _is_consonant(ch: str) -> bool:
    o = ord(ch)
    return 0x0C15 <= o <= 0x0C39 or 0x0C58 <= o <= 0x0C5A or o == 0x0C5D


def _is_indep_vowel(ch: str) -> bool:
    o = ord(ch)
    return 0x0C05 <= o <= 0x0C14 or o in (0x0C60, 0x0C61)


@dataclass(frozen=True, slots=True)
class AksharaConfig:
    pollu_weight: float = 0.5  # word-final dead consonant (e.g. న్) or ZWNJ half-form


def count_telugu(text: str, cfg: AksharaConfig = AksharaConfig()) -> float:
    count = 0.0
    n = len(text)
    for i, ch in enumerate(text):
        if _is_indep_vowel(ch):
            count += 1
        elif _is_consonant(ch):
            j = i - 1
            if j >= 0 and text[j] == ZWJ:
                j -= 1
            if not (j >= 0 and text[j] == VIRAMA):
                count += 1
        elif ch == VIRAMA:
            k = i + 1
            if k < n and text[k] == ZWJ:
                k += 1
            if not (k < n and _is_consonant(text[k])):
                count -= 1 - cfg.pollu_weight
    return max(count, 0.0)


_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def english_syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    groups = len(_VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    elif w.endswith(("es", "ed")) and not w.endswith(("ted", "ded", "ses", "zes", "ces", "ges", "xes")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def count_units(text: str, cfg: AksharaConfig = AksharaConfig()) -> float:
    """Speaking-length units of a Telugu-script line: its aksharas, digits spoken in Telugu. Latin letters count 0."""
    return count_telugu(normalize_telugu(text), cfg)


def mixed_units(text: str, cfg: AksharaConfig = AksharaConfig()) -> float:
    """Speaking-length units of text with English in Latin letters: English syllables plus Telugu aksharas, digits spoken
    in Telugu. What an English source line is measured in for the prior on its Telugu length (ARCHITECTURE §4.3)."""
    normalized = normalize_telugu(text)
    return count_telugu(normalized, cfg) + sum(english_syllables(w) for w in _WORD.findall(normalized))
