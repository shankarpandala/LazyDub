"""The coverage review's deterministic side (ARCHITECTURE §4.6): reading a review reply, classing a re-translation, and
keeping a class in the line cache. All English and Telugu here is original test text."""

from __future__ import annotations

import pytest

from maata_engine.backends.base import Coverage, LineResult, Wording
from maata_engine.qa.coverage import (CLASSES, approved_full, check_review, check_review_fallbacks, coverage_from,
                                     coverage_json, finding, rank, redo_class, source_allows_full_fallback, voiced_class)


def v(id_, cls, missing=(), added=(), error="none") -> dict:
    return {"id": id_, "class": cls, "missing": list(missing), "added": list(added), "error": error}


def test_a_reply_gives_each_id_asked_for_its_class():
    got, why = check_review({"lines": [v(1, "C"), v(2, "m", ["really"]), v(3, "P", [" over the hill ", ""]),
                                       v(4, "E", added=["at night"], error="addition"), v(99, "C")]}, [1, 2, 3, 4])
    assert why == {}
    assert got == {1: Coverage("C"), 2: Coverage("m", ("really",)), 3: Coverage("P", ("over the hill",)),
                   4: Coverage("E", (), ("at night",), "addition")}


def test_an_e_names_its_kind_and_only_an_e_has_one():
    got, _ = check_review({"lines": [v(1, "E", error="none"), v(2, "C", error="negation"), v(3, "E", error="number")]},
                          [1, 2, 3])
    assert (got[1].error, got[2].error, got[3].error) == ("other", None, "number")


def test_ids_missing_twice_or_unclassed_have_no_class():
    got, why = check_review({"lines": [v(1, "C"), v(2, "C"), v(2, "P"), {"id": 3, "class": "Z"}, v(4.0, "m")]},
                            [1, 2, 3, 4, 5])
    assert set(got) == {1, 4} and got[4].cls == "m"
    assert why == {2: "returned 2 times", 3: "class 'Z' unknown", 5: "missing from the review"}
    assert check_review(None, [1]) == ({}, {1: "missing from the review"})


def test_an_e_retranslation_is_told_what_was_wrong_a_p_one_only_the_missing_words():
    assert finding(Coverage("P", ("the hill",))) == ()
    (said,) = finding(Coverage("E", (), ("at night",), "addition"))
    assert "meaning error (addition)" in said and "at night" in said


UP = "గాలిపటం పైకి వెళ్తుంది."


@pytest.mark.parametrize("en", [
    "The replacement may cost another fifty rupees.",
    "The replacement might cost more.", "The replacement could cost more.",
    "This can fail during an upload.", "The second upload should finish first.",
    "The second upload would finish first.", "Perhaps the second upload is done.",
    "Maybe the second upload is done.", "The upload probably finished.",
    "The upload possibly finished.", "The upload is likely complete.",
    "The upload is unlikely to finish.", "Apparently the upload finished.",
    "The upload reportedly finished.", "The upload seems complete.", "The upload appears complete.",
    "The upload seemed complete.", "The upload appeared complete.",
    "It takes approximately five minutes.", "It takes roughly five minutes.",
    "The estimated wait is five minutes.", "Wait about five minutes.",
    "The warning is rare.", "This warning RARELY appears.", "The upload usually finishes first.",
    "The upload sometimes finishes first.", "The upload often finishes first.",
    "The upload is almost complete.", "The upload is nearly complete.",
])
def test_qualified_sources_keep_normal_correction_instead_of_the_full_shortcut(en):
    assert not source_allows_full_fallback(en)


@pytest.mark.parametrize("en", [
    "Restart the router only after both uploads finish.",
    "Keep three copies and send the original on Friday.",
    "Put the canvas behind the mayor's chair.",  # whole words: can/may fragments are not modals
])
def test_unqualified_sources_remain_eligible_for_explicit_semantic_and_timing_checks(en):
    assert source_allows_full_fallback(en)


def test_full_shortcut_exclusion_does_not_change_general_meaning_flags():
    from maata_engine.qa.validators import meaning_flags

    en = "The replacement may cost another fifty rupees."
    missing_uncertainty = Wording("మార్చడానికి ఇంకో యాభై రూపాయలు అవుతుంది.")
    assert not source_allows_full_fallback(en)
    # The existing number/negation/question checks are intentionally unchanged; this new guard has narrower scope.
    assert meaning_flags(en, missing_uncertainty) == {}


@pytest.mark.parametrize("en,before,reviewed,new,cls,error", [
    # a fresh semantic C is necessary even when a correction says more
    ("The kite climbs over the hill.", Coverage("P", ("over the hill",)), UP, "గాలిపటం కొండ మీదుగా పైకి వెళ్తుంది.",
     "C", None),
    # an explicit fresh P remains P, independently of length
    ("The kite climbs over the hill.", Coverage("P", ("over the hill",)), UP, UP, "P", None),
    # a negation, a question or a number the reviewed wording said and the new one doesn't: E, whatever the review said
    ("The kite never comes down.", Coverage("P", ("never",)), "గాలిపటం ఎప్పుడూ కిందికి రాదు.",
     "గాలిపటం ఎప్పుడూ కిందికి వస్తుంది, కొండ మీదుగా.", "E", "negation"),
    # ... or the very error the review found, still there
    ("Did the kite come down?", Coverage("E", error="question"), "గాలిపటం కిందికి వచ్చిందా.", "గాలిపటం కిందికి వచ్చింది.",
     "E", "question"),
    ("Two kites came down.", Coverage("E", error="number"), "గాలిపటాలు కిందికి వచ్చాయి.", "గాలిపటాలు కిందికి వచ్చాయి.",
     "E", "number"),
    # a corrected number plus a fresh semantic C can replace the E
    ("Two kites came down.", Coverage("E", error="number"), "గాలిపటాలు కిందికి వచ్చాయి.",
     "రెండు గాలిపటాలు కిందికి వచ్చాయి.", "C", None),
    # a flag both wordings carry, where the review found no such error ("a pair" for "two", which the number check
    # can't match): the heuristic missing the English, no evidence against the one that restores the phrase
    ("Two kites came down slowly.", Coverage("P", ("slowly",)), "జంట గాలిపటాలు కిందికి వచ్చాయి.",
     "జంట గాలిపటాలు నెమ్మదిగా కిందికి వచ్చాయి.", "C", None),
])
def test_the_checks_class_a_retranslation(en, before, reviewed, new, cls, error):
    got = redo_class(en, before, Wording(reviewed), Wording(new), semantic=Coverage(cls, error=error))
    assert (got.cls, got.error, got.by, got.first, got.tier) == (cls, error, "validators" if error else "review", before.cls, "full")


def test_a_retranslation_is_classed_on_the_tier_it_is_set_against():
    got = redo_class("The kite climbs over the hill.", Coverage("P", ("over the hill",), tier="concise"),
                     Wording(UP), Wording(UP), "concise", Coverage("P", ("over the hill",)))
    assert (got.cls, got.tier, got.missing) == ("P", "concise", ("over the hill",))


def test_better_means_a_lower_rank_and_unreviewed_ranks_last():
    assert [rank(Coverage(c)) for c in CLASSES] == [0, 1, 2, 3] and rank(None) == 4


@pytest.mark.parametrize("c", [Coverage("C", tier="full"), Coverage("P", ("over the hill",), tier="concise"),
                               Coverage("E", (), ("at night",), "addition", "full"),
                               Coverage("C", tier="full", by="validators", first="P")])
def test_a_class_round_trips_through_the_line_cache(c):
    assert coverage_from(coverage_json(c)) == c


def test_an_older_or_broken_stored_class():
    assert coverage_from("m") == Coverage("m")  # a bare class, as the wave-1 hook stored it
    assert coverage_from(None) is None and coverage_from({"class": "Q"}) is None and coverage_json(None) is None
    assert coverage_from({"class": "E", "error": "bogus", "by": "someone", "first": "Z"}) == Coverage("E")


@pytest.mark.parametrize("first", ["P", "E"])
def test_approved_full_marker_round_trips_and_reports_actual_reviewed_class(first):
    c = Coverage("C", tier="full", by="review", first=first)
    line = LineResult(1, {"full": Wording(UP)}, coverage=coverage_from(coverage_json(c)))
    assert approved_full(line)
    assert voiced_class(line.coverage, "full", first=True) == "C"
    assert voiced_class(line.coverage, "concise") == "other_tier"


@pytest.mark.parametrize("c", [None, Coverage("C", tier="full"), Coverage("m", tier="full", first="P"),
                              Coverage("C", tier="concise", first="P"),
                              Coverage("C", tier="full", by="validators", first="P"),
                              Coverage("C", tier="full", first="m")])
def test_only_explicit_semantic_full_approval_is_pinned(c):
    assert not approved_full(LineResult(1, {"full": Wording(UP)}, coverage=c))
    assert not approved_full(LineResult(1, {}, coverage=Coverage("C", tier="full", first="P")))


@pytest.mark.parametrize("fallback", [None, {}, "C", {"class": "C"},
    {"class": "m", "missing": [], "added": [], "error": "none"},
    {"class": "C", "missing": ["maybe"], "added": [], "error": "none"},
    {"class": "C", "missing": [], "added": ["always"], "error": "none"},
    {"class": "C", "missing": [], "added": [], "error": "number"},
    {"class": "C", "missing": [], "added": [], "error": "none", "extra": True}])
def test_missing_or_contradictory_full_verdict_never_authorizes_reuse(fallback):
    row = {**v(1, "P", ["maybe"]), "fallback": fallback}
    assert check_review_fallbacks({"lines": [row]}, [1]) == set()
    assert check_review({"lines": [row]}, [1])[0][1].cls == "P"  # main verdict remains usable


def test_fallback_verdict_must_be_unique_and_offered():
    row = {**v(1, "P"), "fallback": {"class": "C", "missing": [], "added": [], "error": "none"}}
    assert check_review_fallbacks({"lines": [row]}, [1]) == {1}
    assert check_review_fallbacks({"lines": [row]}, [2]) == set()
    assert check_review_fallbacks({"lines": [row, row]}, [1]) == set()
    assert check_review_fallbacks({"lines": [{**row, "id": True}]}, [1]) == set()
    assert check_review_fallbacks({"lines": None}, [1]) == set()


@pytest.mark.parametrize("new", [UP, "గాలిపటం కొండ మీదుగా నెమ్మదిగా చాలా పైకి వెళ్తుంది."])
def test_correction_length_never_supplies_missing_semantic_approval(new):
    assert redo_class("The kite climbs over the hill.", Coverage("P"), Wording(UP), Wording(new)) is None


def test_shorter_complete_correction_uses_semantic_verdict_not_character_length():
    old = Wording("గాలిపటం ఇప్పుడు ఇక్కడ చాలా నెమ్మదిగా పైకి వెళ్తుంది.")
    new = Wording("గాలిపటం కొండ దాటింది.")
    assert len(new.spoken) < len(old.spoken)
    got = redo_class("The kite crossed the hill.", Coverage("P", ("crossed the hill",)), old, new,
                     semantic=Coverage("C"))
    assert got == Coverage("C", tier="full", by="review", first="P")


def test_longer_incomplete_correction_keeps_fresh_missing_content_verdict():
    new = Wording("గాలిపటం ఇప్పుడు చాలా నెమ్మదిగా పైకి వెళ్తుంది.")
    verdict = Coverage("P", ("over the hill",))
    got = redo_class("The kite climbs over the hill.", verdict, Wording(UP), new, semantic=verdict)
    assert got == Coverage("P", ("over the hill",), tier="full", by="review", first="P")


@pytest.mark.parametrize("verdict", [Coverage("C", missing=("may",)), Coverage("C", added=("always",)),
                                    Coverage("C", error="negation")])
def test_contradictory_complete_verdict_never_approves_correction(verdict):
    assert redo_class("The kite crossed the hill.", Coverage("P"), Wording(UP), Wording("గాలిపటం కొండ దాటింది."),
                      semantic=verdict) is None
