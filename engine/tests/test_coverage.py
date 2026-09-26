"""The coverage review's deterministic side (ARCHITECTURE §4.6): reading a review reply, classing a re-translation, and
keeping a class in the line cache. All English and Telugu here is original test text."""

from __future__ import annotations

import pytest

from maata_engine.backends.base import Coverage, Wording
from maata_engine.qa.coverage import (CLASSES, check_review, coverage_from, coverage_json, finding, rank, redo_class)


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


@pytest.mark.parametrize("en,before,reviewed,new,cls,error", [
    # a P line whose re-translation says more, and flags nothing: C
    ("The kite climbs over the hill.", Coverage("P", ("over the hill",)), UP, "గాలిపటం కొండ మీదుగా పైకి వెళ్తుంది.",
     "C", None),
    # ... no longer than the wording that lacked the phrase: the phrase can't be back
    ("The kite climbs over the hill.", Coverage("P", ("over the hill",)), UP, UP, "P", None),
    # a negation, a question or a number the reviewed wording said and the new one doesn't: E, whatever the review said
    ("The kite never comes down.", Coverage("P", ("never",)), "గాలిపటం ఎప్పుడూ కిందికి రాదు.",
     "గాలిపటం ఎప్పుడూ కిందికి వస్తుంది, కొండ మీదుగా.", "E", "negation"),
    # ... or the very error the review found, still there
    ("Did the kite come down?", Coverage("E", error="question"), "గాలిపటం కిందికి వచ్చిందా.", "గాలిపటం కిందికి వచ్చింది.",
     "E", "question"),
    ("Two kites came down.", Coverage("E", error="number"), "గాలిపటాలు కిందికి వచ్చాయి.", "గాలిపటాలు కిందికి వచ్చాయి.",
     "E", "number"),
    # an E line whose re-translation passes the checks: C (the checks can't see names or additions)
    ("Two kites came down.", Coverage("E", error="number"), "గాలిపటాలు కిందికి వచ్చాయి.",
     "రెండు గాలిపటాలు కిందికి వచ్చాయి.", "C", None),
    # a flag both wordings carry, where the review found no such error ("a pair" for "two", which the number check
    # can't match): the heuristic missing the English, no evidence against the one that restores the phrase
    ("Two kites came down slowly.", Coverage("P", ("slowly",)), "జంట గాలిపటాలు కిందికి వచ్చాయి.",
     "జంట గాలిపటాలు నెమ్మదిగా కిందికి వచ్చాయి.", "C", None),
])
def test_the_checks_class_a_retranslation(en, before, reviewed, new, cls, error):
    got = redo_class(en, before, Wording(reviewed), Wording(new))
    assert (got.cls, got.error, got.by, got.first, got.tier) == (cls, error, "validators", before.cls, "full")


def test_a_retranslation_is_classed_on_the_tier_it_is_set_against():
    got = redo_class("The kite climbs over the hill.", Coverage("P", ("over the hill",), tier="concise"),
                     Wording(UP), Wording(UP), "concise")
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
