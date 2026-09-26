"""Deterministic checks on Claude's scene replies (docs/research/dubbing-2026-09/ARCHITECTURE.md §4.4).

Pure: no model, no I/O, so they run in cloud CI too. Three severities:
- reject: the line can't be used and is asked for again: a requested id missing or given twice, or a `full` wording whose
  `spoken` breaks the Telugu-script contract (Latin letters or romanised Telugu, digits, ZWNJ/ZWJ, brackets, symbols,
  invisible characters, a letter of another script such as a Cyrillic a (U+0430) inside a word, or no Telugu at all);
- repair: the line is kept without the broken part, and a flag says what went: an `english` entry that doesn't point at
  a Telugu word or isn't an English word, an emphasis index out of range, a shorter or fuller wording that breaks the
  script contract or the akshara order, pieces that don't join back to `full`;
- flag: suspect but kept: a glossary term not in its fixed spelling, a negation or a question lost or added, a number
  not said.
Flags travel with the line into units.jsonl; they never block a line on their own. The coverage review (Claude) reads
only the English and the Telugu; the meaning flags class a re-translation, which is never reviewed again (qa/coverage.py).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..backends.base import EMOTIONS, ENERGIES, TIERS, Delivery, GlossaryEntry, LineResult, LineSpec, Wording
from ..text.akshara import count_telugu
from ..text.normalize_te import number_to_telugu

PUNCT = frozenset(".,?!;:…'\"‘’“”-–—")  # all `spoken` may carry besides Telugu letters and spaces
BRACKETS = frozenset("()[]{}<>")
JOINERS = frozenset("\u200c\u200d")

# Romanised Telugu: frequent function words and verb forms, and endings no English word has. Checked on lowercase
# Latin words only, so names (Hindi, Peru) pass.
ROMAN_TELUGU = frozenset("""
    nenu nuvvu meeru memu manam vaadu vaallu aame adi idi avunu kaadu kadu ledu ledhu undi unnadi unnaru unnanu
    chesi chesthe chestha chesta chesanu chesaru cheyandi cheyali cheyyi cheyyandi kani kaani kuda kooda chala chaala
    baga baaga emiti enti endi ekkada eppudu ela enduku ippudu appudu ikkada akkada andi sare saare inka mari kotha
    telusu teliyadu kada kadha kadaa ante antey okka rendu moodu nundi nunchi tho lo ki ku ga gaa ra raa
""".split())
_ROMAN_END = re.compile(r"(?:andi|indi|aaru|unnaru|unnanu|tunnaa?|tunnaru|tunnadu|isthe|esthe|chesi|cheyyi)$")
_LATIN_RUN = re.compile(r"[^\W\d_]+")

_NEG_EN = re.compile(r"\b(?:not|never|no|nobody|nothing|none|nowhere|neither|nor|cannot|without)\b|n['’]t\b", re.I)
# Telugu negative morphology, per word: లేదు/లేడు/లేరు/లేను/... (రాలేడు too), కాదు/కావు, వద్దు and a verb's -ొద్దు
# (చేయొద్దు, but not పొద్దు or పొద్దున్నే), కూడదు, -కుండా, and the negative -అదు/-ీదు endings (రాదు, తెలీదు), but not
# ముందు or హద్దు (ం or ్ before దు), and not లేదా ("or"). A bare masculine -డు (రాడు) is left out: most -డు words aren't
# negative (వాడు, ఎప్పుడు).
_NEG_TE = re.compile(r"లేద(?:ు|ండి)|లే(?:డు|రు|ను|ము|వు|ని|కుండా|కపో)|కాద(?:ు|ండి|ా)|కావు|కాకుండా|కాకపో|వద్దు|"
                     r"^.{2,}ొద్ద(?:ు|ండి)$|కూడదు|కుండా|(?<![ంఁ్])దు$")
_TE_PUNCT = "".join(PUNCT)

_EN_DIGITS = re.compile(r"(?<![A-Za-z\d.,])(?<![A-Za-z]-)\d[\d,]*(?:\.\d+)?(?![\d,.]?\d|st\b|nd\b|rd\b|th\b|[A-Za-z])")
_EN_WORDS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen".split())}
_EN_WORDS.update({w: 10 * n for n, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})
_EN_WORDS.update(hundred=100, thousand=1000, lakh=100_000, million=1_000_000, crore=10_000_000, billion=10 ** 9)
_EN_WORDS.pop("one")  # "the one", "one of": a pronoun far more often than a number
_EN_NUMBER_WORD = re.compile(r"\b(" + "|".join(sorted(_EN_WORDS, key=len, reverse=True)) + r")\b", re.I)


@dataclass(slots=True)
class Checked:
    """One reply, checked against the lines asked for."""

    lines: dict[int, LineResult] = field(default_factory=dict)
    rejected: dict[int, list[str]] = field(default_factory=dict)  # requested ids unusable or absent -> why
    unexpected: list[object] = field(default_factory=list)       # ids nobody asked for (context lines translated too)
    glossary_additions: list[GlossaryEntry] = field(default_factory=list)


# ---- script -------------------------------------------------------------------------------------------------------
def _script(ch: str) -> str:
    return unicodedata.name(ch, "UNKNOWN").split()[0]  # LATIN, CYRILLIC, TELUGU, DEVANAGARI, ...


def is_telugu_letter(ch: str) -> bool:
    o = ord(ch)
    return 0x0C05 <= o <= 0x0C39 or 0x0C58 <= o <= 0x0C61


def romanised_telugu(word: str) -> bool:
    return word.islower() and (word in ROMAN_TELUGU or bool(_ROMAN_END.search(word)))


def script_problems(spoken: str) -> list[str]:
    """What breaks the Telugu-script contract in `spoken`, worded for the retry message; [] when it holds."""
    kinds: dict[str, list[str]] = {}
    for word in spoken.split():
        found: set[str] = set()
        for ch in word:
            cat = unicodedata.category(ch)
            if ch in JOINERS:
                found.add("ZWNJ/ZWJ in")
            elif ch in PUNCT:
                continue
            elif ch in BRACKETS:
                found.add("brackets in")
            elif cat[0] == "N":
                found.add("digits in")
            elif 0x0C00 <= ord(ch) <= 0x0C7F:
                if cat[0] == "S":
                    found.add("symbols in")
            elif cat[0] in "LM":
                found.add("Latin letters in" if _script(ch) == "LATIN" else f"{_script(ch).title()} letters in")
            elif cat[0] == "C":
                found.add("invisible characters in")
            else:
                found.add("symbols in")
        if "Latin letters in" in found and any(romanised_telugu(r) for r in _LATIN_RUN.findall(word)):
            found.discard("Latin letters in")
            found.add("romanised Telugu in")
        for kind in found:
            kinds.setdefault(kind, []).append(word)
    problems = [f"{kind} {', '.join(words)}" for kind, words in sorted(kinds.items())]
    if not any(is_telugu_letter(ch) for ch in spoken):
        problems.append("no Telugu")
    return problems


def _english_word_problem(en: object) -> str | None:
    if not isinstance(en, str) or not en.strip():
        return "is empty"
    letters = [ch for ch in en if unicodedata.category(ch)[0] == "L"]
    other = sorted({_script(ch).title() for ch in letters if _script(ch) != "LATIN"})
    if other:
        return f"has {', '.join(other)} letters"
    if not letters:
        return "has no letters"
    if any(romanised_telugu(w) for w in _LATIN_RUN.findall(en)):
        return "is romanised Telugu"
    return None


def _english(raw: object, words: list[str]) -> tuple[tuple[tuple[int, str], ...], list[str]]:
    """The valid entries of an `english` map, and a flag per entry dropped."""
    out: dict[int, str] = {}
    flags: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        i, en = (item.get("i"), item.get("en")) if isinstance(item, dict) else (None, None)
        if isinstance(i, float) and i.is_integer():
            i = int(i)
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(words) \
                or not any(is_telugu_letter(ch) for ch in words[i]):
            flags.append(f"english: {i!r} is not a word of the line")
        elif i in out:
            flags.append(f"english: word {i} listed twice")
        elif why := _english_word_problem(en):
            flags.append(f"english: {en!r} {why}")
        else:
            out[i] = en.strip()
    return tuple(sorted(out.items())), flags


def wording(raw: object) -> tuple[Wording | None, list[str], list[str]]:
    """(the wording, what rejects it, what was repaired). Whitespace is normalised to single spaces."""
    if not isinstance(raw, dict) or not isinstance(raw.get("spoken"), str):
        return None, ["not a wording"], []
    spoken = " ".join(raw["spoken"].split())
    if problems := script_problems(spoken):
        return None, problems, []
    english, flags = _english(raw.get("english"), spoken.split())
    return Wording(spoken, english), [], flags


# ---- the line --------------------------------------------------------------------------------------------------------
def _ordered(tiers: dict[str, Wording]) -> tuple[dict[str, Wording], list[str]]:
    """Keep `full`, then each shorter tier only if shorter than the one kept above it, and `fuller` only if longer."""
    n = {k: count_telugu(w.spoken) for k, w in tiers.items()}
    keep, flags, bound = {"full": tiers["full"]}, [], n["full"]
    for name in ("concise", "very_concise"):
        if name not in tiers:
            continue
        if n[name] < bound:
            keep[name], bound = tiers[name], n[name]
        else:
            flags.append(f"{name} dropped: {n[name]:g} aksharas, not shorter than {bound:g}")
    if "fuller" in tiers:
        if n["fuller"] > n["full"]:
            keep["fuller"] = tiers["fuller"]
        else:
            flags.append(f"fuller dropped: {n['fuller']:g} aksharas, not longer than {n['full']:g}")
    return {k: keep[k] for k in TIERS if k in keep}, flags


def _pieces(raw: object, spec: LineSpec, full: Wording) -> tuple[tuple[str, ...], list[str]]:
    if not raw:
        return (), []
    pieces = tuple(" ".join(p.split()) for p in raw if isinstance(p, str)) if isinstance(raw, list) else ()
    if not spec.breaks:
        why = "the line has no breaks"
    elif not pieces or len(pieces) != len(raw) or not all(pieces):
        why = "not a list of wordings"
    elif len(pieces) > len(spec.breaks) + 1:
        why = f"{len(pieces)} pieces for {len(spec.breaks)} breaks"
    elif " ".join(pieces) != full.spoken:
        why = "they don't join back to full"
    else:
        return pieces, []
    return (), [f"pieces dropped: {why}"]


def _delivery(raw: object, n_words: int) -> tuple[Delivery, list[str]]:
    d = raw if isinstance(raw, dict) else {}
    flags = [f"delivery: {k} {d.get(k)!r} unknown" for k, allowed in (("emotion", EMOTIONS), ("energy", ENERGIES))
             if k in d and d[k] not in allowed]
    emphasis = []
    for i in d.get("emphasis") or []:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n_words:
            emphasis.append(i)
        else:
            flags.append(f"delivery: emphasis {i!r} is not a word of the line")
    return Delivery(d.get("emotion") if d.get("emotion") in EMOTIONS else "neutral",
                    d.get("energy") if d.get("energy") in ENERGIES else "mid",
                    d.get("question") is True, tuple(dict.fromkeys(emphasis))), flags


def check_line(raw: Mapping, spec: LineSpec, glossary: Mapping[str, str] = {}) -> tuple[LineResult | None, list[str]]:
    """One line of a reply (or of the line cache): (the line, []) when usable, else (None, why)."""
    full, problems, flags = wording(raw.get("full"))
    if full is None:
        return None, [f"full: {p}" for p in problems]
    tiers = {"full": full}
    flags = [f"full {f}" for f in flags]
    for name in TIERS:
        if name == "full" or name not in raw:
            continue
        w, problems, repaired = wording(raw[name])
        if w is None:
            flags.append(f"{name} dropped: {problems[0]}")
        else:
            tiers[name] = w
            flags += [f"{name} {f}" for f in repaired]
    tiers, dropped = _ordered(tiers)
    flags += dropped
    flags += [f"{name} missing" for name in spec.want if name not in tiers]
    pieces, why = _pieces(raw.get("pieces"), spec, full)
    flags += why
    moved = tuple(m for m in raw.get("moved") or [] if isinstance(m, str) and m.strip()) if pieces else ()
    delivery, why = _delivery(raw.get("delivery"), len(full.spoken.split()))
    flags += why
    flags += suspect(spec.en, full, glossary)
    return LineResult(spec.id, tiers, delivery, pieces, moved, raw.get("unfinished") is True, tuple(flags)), []


def check_reply(data: object, specs: Sequence[LineSpec], glossary: Mapping[str, str] = {}) -> Checked:
    """A reply against the lines asked for: every id exactly once, each line checked by `check_line`."""
    by_id = {s.id: s for s in specs}
    got: dict[int, list[Mapping]] = {}
    out = Checked()
    for raw in (data.get("lines") if isinstance(data, dict) else None) or []:
        rid = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(rid, float) and rid.is_integer():
            rid = int(rid)
        if isinstance(rid, int) and not isinstance(rid, bool) and rid in by_id:
            got.setdefault(rid, []).append(raw)
        else:
            out.unexpected.append(rid)
    for s in specs:
        copies = got.get(s.id, [])
        if not copies:
            out.rejected[s.id] = ["missing from the reply"]
        elif len(copies) > 1:
            out.rejected[s.id] = [f"returned {len(copies)} times"]
        else:
            line, why = check_line(copies[0], s, glossary)
            if line is None:
                out.rejected[s.id] = why
            else:
                out.lines[s.id] = line
    out.glossary_additions = glossary_entries(data.get("glossary_additions") if isinstance(data, dict) else None)
    return out


def glossary_entries(raw: object) -> list[GlossaryEntry]:
    """The usable glossary items of a reply or a brief: a term, and a spelling that keeps the script contract."""
    out: dict[str, GlossaryEntry] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        term, spoken = item.get("term"), item.get("spoken")
        if not isinstance(term, str) or not term.strip() or not isinstance(spoken, str) or script_problems(spoken):
            continue
        note = item.get("note") if isinstance(item.get("note"), str) else ""
        out.setdefault(term.strip().lower(), GlossaryEntry(term.strip(), " ".join(spoken.split()),
                                                           item.get("keep_english") is not False, note.strip()))
    return list(out.values())


# ---- flags -----------------------------------------------------------------------------------------------------------
def negations(en: str, spoken: str) -> tuple[int, int]:
    """(English negations, Telugu words with negative morphology): a heuristic count."""
    words = (w.strip(_TE_PUNCT) for w in spoken.split())
    te = sum(1 for w in words if _NEG_TE.search(w) and not w.startswith("ఐదు"))
    return len(_NEG_EN.findall(en)), te


def _stem(word: str) -> str:
    """A Telugu numeral without its final vowel sign, so inflected forms match (రెండు -> రెండ: రెండేళ్ళు)."""
    return word[:-1] if word and word[-1] in "ుి" else word


def unsaid_numbers(en: str, full: Wording) -> list[str]:
    """Numbers in the English (digits, and number words from two up) that the Telugu doesn't seem to say: neither as a
    Telugu numeral (its first word, inflected or not) nor as an English word in the map."""
    words = full.spoken.split()
    english = " ".join(e for _, e in full.english).lower()
    found: list[tuple[str, int]] = []
    for m in _EN_DIGITS.finditer(en):
        text = m.group(0).replace(",", "")
        found.append((m.group(0), int(float(text))))
    found += [(m.group(1), _EN_WORDS[m.group(1).lower()]) for m in _EN_NUMBER_WORD.finditer(en)]
    missing = []
    for said, value in found:
        stems = {_stem(number_to_telugu(value).split()[0])}
        stems |= {"వేల"} if value == 1000 else {"ఒక"} if value == 1 else set()  # వెయ్యి/వేల; ఒకటి/ఒక before a noun
        if not any(w.startswith(tuple(stems)) for w in words) and said.lower() not in english:
            missing.append(said)
    return list(dict.fromkeys(missing))


def meaning_flags(en: str, w: Wording) -> dict[str, str]:
    """The meaning errors the heuristics can see in a wording, by kind (the coverage review's E kinds): a negation or a
    question lost or added, a number not said."""
    flags = {}
    n_en, n_te = negations(en, w.spoken)
    if (n_en > 0) != (n_te > 0):
        flags["negation"] = f"negation: {n_en} in the English, {n_te} in the Telugu"
    if ("?" in en) != ("?" in w.spoken):
        flags["question"] = "question: " + ("lost" if "?" in en else "added")
    if missing := unsaid_numbers(en, w):
        flags["number"] = f"numbers not said: {', '.join(missing)}"
    return flags


def suspect(en: str, full: Wording, glossary: Mapping[str, str] = {}) -> list[str]:
    """The heuristic flags for a line: glossary spellings, negation and question parity, numbers."""
    flags = []
    for term, spelling in glossary.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", en, re.I) and spelling not in full.spoken:
            flags.append(f"glossary: {term} not written {spelling}")
    return flags + list(meaning_flags(en, full).values())
