"""Deterministic Telugu text normalizer (spec §6.4): numbers, currency and percentages as spoken words.

The translator is asked to write numbers as words; this catches anything it misses.
Colloquial spoken forms are used. The maintainer should review these tables (see tests).
"""

from __future__ import annotations

import re

_UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పందొమ్మిది",
]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
_TE_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")


def _below_100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    t, r = divmod(n, 10)
    return _TENS[t] if r == 0 else f"{_TENS[t]} {_UNITS[r]}"


def _scaled(n: int, unit: int, one: str, plural: str, joiner: str, rest_fn) -> str:
    q, r = divmod(n, unit)
    head = one if q == 1 else f"{number_to_telugu(q)} {plural}"
    if r == 0:
        return head
    head = one if q == 1 else f"{number_to_telugu(q)} {joiner}"
    return f"{head} {rest_fn(r)}"


def number_to_telugu(n: int) -> str:
    """Spoken Telugu for a non-negative integer (Indian grouping: వేలు, లక్షలు, కోట్లు)."""
    if n < 0:
        return "మైనస్ " + number_to_telugu(-n)
    if n < 100:
        return _below_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1:
            return "వంద" if r == 0 else f"నూట {_below_100(r)}"
        return f"{_UNITS[h]} వందలు" if r == 0 else f"{_UNITS[h]} వందల {_below_100(r)}"
    if n < 100_000:
        return _scaled(n, 1000, "వెయ్యి", "వేలు", "వేల", number_to_telugu)
    if n < 10_000_000:
        return _scaled(n, 100_000, "లక్ష", "లక్షలు", "లక్షల", number_to_telugu)
    return _scaled(n, 10_000_000, "కోటి", "కోట్లు", "కోట్ల", number_to_telugu)


def _decimal(int_part: str, frac: str) -> str:
    digits = " ".join(_UNITS[int(d)] for d in frac)
    return f"{number_to_telugu(int(int_part))} పాయింట్ {digits}"


_CURRENCY_BEFORE = {"₹": "రూపాయలు", "rs.": "రూపాయలు", "rs": "రూపాయలు", "inr": "రూపాయలు", "$": "డాలర్లు", "usd": "డాలర్లు", "€": "యూరోలు", "£": "పౌండ్లు"}
_NUM = r"(\d[\d,]*)(?:\.(\d+))?"
_RE_CURRENCY = re.compile(r"(₹|\$|€|£|\b(?:rs\.?|inr|usd)\s?)\s?" + _NUM, re.IGNORECASE)
_RE_PERCENT = re.compile(_NUM + r"\s?%")
_RE_NUMBER = re.compile(_NUM)


def _num_words(int_part: str, frac: str | None) -> str:
    int_part = int_part.replace(",", "")
    return _decimal(int_part, frac) if frac else number_to_telugu(int(int_part))


def normalize_telugu(text: str) -> str:
    """Rewrite digits, currency and percentages as spoken Telugu words."""
    text = text.translate(_TE_DIGITS)

    def currency(m: re.Match[str]) -> str:
        sym = m.group(1).strip().lower()
        return f"{_num_words(m.group(2), m.group(3))} {_CURRENCY_BEFORE.get(sym, _CURRENCY_BEFORE.get(sym.rstrip('.'), ''))}".strip()

    text = _RE_CURRENCY.sub(currency, text)
    text = _RE_PERCENT.sub(lambda m: f"{_num_words(m.group(1), m.group(2))} శాతం", text)
    text = _RE_NUMBER.sub(lambda m: _num_words(m.group(1), m.group(2)), text)
    return re.sub(r"\s{2,}", " ", text).strip()
