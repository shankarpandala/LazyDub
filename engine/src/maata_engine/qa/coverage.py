"""The coverage review's deterministic side (docs/research/dubbing-2026-09/ARCHITECTURE.md §4.6; research gap-4 E5).

Claude classes every line of a scene on the wording chosen for it (the review call, with its own system prompt and schema
in text/scene_prompt.py): C complete, m a minor drop (an intensifier, a hedge, a filler, a backchannel), P a content
phrase or clause missing, E a meaning error. A P or E line gets one re-translation from its English, with the missing
words named; Claude never reviews that one again. The checks here class it instead, and the better of the two wordings
by class is kept, the reviewed one on a tie. Pure: no model, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..backends.base import Coverage, Wording
from ..text.akshara import count_units
from .validators import meaning_flags

CLASSES = ("C", "m", "P", "E")  # best first
REDO = frozenset("PE")           # classes that get a re-translation
ERRORS = ("negation", "number", "name", "question", "addition", "other")


def _strings(raw: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(x.strip() for x in raw if isinstance(x, str) and x.strip())) if isinstance(raw, list) else ()


def check_review(data: object, ids: Sequence[int]) -> tuple[dict[int, Coverage], dict[int, str]]:
    """A review reply against the ids asked for: (a class for each id answered exactly once, why each other id has
    none). Ids nobody asked for are ignored. An E comes with its kind ("other" when the reply names none); the other
    classes carry none."""
    got: dict[int, list[Mapping]] = {}
    for raw in (data.get("lines") if isinstance(data, dict) else None) or []:
        rid = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(rid, float) and rid.is_integer():
            rid = int(rid)
        if isinstance(rid, int) and not isinstance(rid, bool) and rid in ids:
            got.setdefault(rid, []).append(raw)
    out: dict[int, Coverage] = {}
    why: dict[int, str] = {}
    for i in ids:
        copies = got.get(i, [])
        if len(copies) != 1:
            why[i] = f"returned {len(copies)} times" if copies else "missing from the review"
            continue
        raw = copies[0]
        cls = raw.get("class")
        if cls not in CLASSES:
            why[i] = f"class {cls!r} unknown"
            continue
        error = (raw.get("error") if raw.get("error") in ERRORS else "other") if cls == "E" else None
        out[i] = Coverage(cls, _strings(raw.get("missing")), _strings(raw.get("added")), error)
    return out, why


def finding(c: Coverage) -> tuple[str, ...]:
    """What a re-translation of an E line is told about the wording it replaces (it never sees that wording)."""
    if c.cls != "E":
        return ()
    added = f"; it added what the English doesn't say: {', '.join(c.added)}" if c.added else ""
    return (f"a review found a meaning error ({c.error}) in the earlier Telugu{added}",)


def redo_class(en: str, before: Coverage, reviewed: Wording, new: Wording, tier: str = "full") -> Coverage:
    """The class the checks give a re-translation's wording on `tier`, set against the line's own wording on that tier
    (`reviewed`, the one the review classed when the re-translation has its tier). E when a negation, a question or a
    number doesn't match the English, unless `reviewed` has the same mismatch and the review found no such error there:
    a flag both carry is the heuristic missing the English, and says nothing of which wording is better. P, for a P
    line, while it is no longer in aksharas than `reviewed` (the phrase can't be back). Else C. `first` keeps the
    review's class of the wording it would replace."""
    was = meaning_flags(en, reviewed)
    flags = [k for k in meaning_flags(en, new) if k not in was or (before.cls == "E" and before.error == k)]
    if flags:
        return Coverage("E", error=flags[0], tier=tier, by="validators", first=before.cls)
    if before.cls == "P" and count_units(new.spoken) <= count_units(reviewed.spoken):
        return Coverage("P", before.missing, tier=tier, by="validators", first=before.cls)
    return Coverage("C", tier=tier, by="validators", first=before.cls)


def rank(c: Coverage | None) -> int:
    """Lower is better; an unreviewed line ranks last."""
    return CLASSES.index(c.cls) if c is not None else len(CLASSES)


def voiced_class(c: Coverage | None, tier: str | None, first: bool = False) -> str:
    """A voiced line's class as the metrics count it (§7 step 4): its class (with `first`, the review's own, before a
    re-translation replaced the wording) only when it is the class of the wording voiced, on `tier`; "other_tier" when
    it is another wording's (the pick moved after the review: a fit, the voice's pace, a shorter re-synthesis);
    "unreviewed" when it has none."""
    if c is None:
        return "unreviewed"
    if c.tier != tier:
        return "other_tier"
    return (c.first or c.cls) if first and c.by == "validators" else c.cls


def coverage_json(c: Coverage | None) -> dict | None:
    """A class as the line cache and units.jsonl keep it."""
    if c is None:
        return None
    return {"class": c.cls, "by": c.by, "tier": c.tier, "first": c.first, "missing": list(c.missing),
            "added": list(c.added), "error": c.error}


def coverage_from(raw: object) -> Coverage | None:
    """A stored class back (a bare class string from an older cache row too); None when there is none."""
    if isinstance(raw, str):
        raw = {"class": raw}
    if not isinstance(raw, dict) or raw.get("class") not in CLASSES:
        return None
    first = raw.get("first") if raw.get("first") in CLASSES else None
    return Coverage(raw["class"], _strings(raw.get("missing")), _strings(raw.get("added")),
                    raw.get("error") if raw.get("error") in ERRORS else None,
                    raw.get("tier") if isinstance(raw.get("tier"), str) else None,
                    "validators" if raw.get("by") == "validators" else "review", first)
