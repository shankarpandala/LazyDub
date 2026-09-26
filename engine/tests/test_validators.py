"""The deterministic checks on Claude's scene replies (ARCHITECTURE §4.4). All Telugu here is original test text."""

from __future__ import annotations

import pytest

from maata_engine.backends.base import LineSpec, Wording
from maata_engine.qa.validators import (check_line, check_reply, glossary_entries, negations, script_problems, suspect,
                                        unsaid_numbers)


def w(spoken: str, *english: tuple[int, str]) -> dict:
    return {"spoken": spoken, "english": [{"i": i, "en": en} for i, en in english]}


def spec(i: int = 1, en: str = "The shop opens early.", want: tuple[str, ...] = ("full",), **kw) -> LineSpec:
    return LineSpec(i, "S1", en, 0.0, 3.0, 2.8, 14.0, want, **kw)


def line(i: int = 1, full: dict | None = None, **kw) -> dict:
    return {"id": i, "full": full or w("షాప్ పొద్దున్నే తెరుస్తారు.", (0, "shop")), "delivery": {"emotion": "neutral", "energy": "mid"},
            **kw}


# ---- the script contract --------------------------------------------------------------------------------------------
def test_clean_telugu_passes():
    assert script_problems("ముందు సెట్టింగ్స్ ఓపెన్ చేయండి, తర్వాత “సరే” అనండి… సరేనా?") == []


@pytest.mark.parametrize("spoken,problem", [
    ("ముందు settings ఓపెన్ చేయండి.", "Latin letters in settings"),
    ("ఇప్పుడు cheyandi అని చెప్పు.", "romanised Telugu in cheyandi"),
    ("nenu వస్తా.", "romanised Telugu in nenu"),
    ("మొత్తం 40 రూపాయలు.", "digits in 40"),
    ("మొత్తం ౪౦ రూపాయలు.", "digits in ౪౦"),
    ("ఫోన్\u200cలో చూడండి.", "ZWNJ/ZWJ in ఫోన్\u200cలో"),
    ("క\u200dష్టం.", "ZWNJ/ZWJ in"),
    ("ఇది (అంటే అది) చూడండి.", "brackets in (అంటే"),
    ("డేటా పెరిగింది 50%.", "symbols in 50%."),
    ("ఇ\u00adది.", "invisible characters in"),
    ("ఇది नमस्ते చూడండి.", "Devanagari letters in नमस्ते"),
    ("మెనూ", None),
])
def test_script_violations_are_named_word_by_word(spoken, problem):
    problems = script_problems(spoken)
    if problem is None:
        assert problems == []
    else:
        assert any(p.startswith(problem) for p in problems), problems


def test_a_cyrillic_letter_hidden_inside_a_word_is_caught():
    # A live reply on 2026-09-24 had a Cyrillic a (U+0430) inside a word: it looks right and would reach the TTS.
    assert script_problems("ఇది క్ల\u0430స్ మాట.") == ["Cyrillic letters in క్ల\u0430స్"]
    assert script_problems("ఇది క్లాస్ మాట.") == []


def test_no_telugu_at_all_is_a_problem():
    assert script_problems("… ?") == ["no Telugu"]
    assert "no Telugu" in script_problems("hello")


def test_a_full_wording_that_breaks_the_script_rejects_the_line():
    got, why = check_line(line(full=w("ఈ shop పొద్దున్నే తెరుస్తారు.")), spec())
    assert got is None and why == ["full: Latin letters in shop"]
    got, why = check_line({"id": 1, "delivery": {}}, spec())
    assert got is None and why == ["full: not a wording"]


# ---- the english map -------------------------------------------------------------------------------------------------
def test_english_entries_must_point_at_a_telugu_word_and_be_english():
    full = w("షాప్ పొద్దున్నే తెరుస్తారు — సరే.", (0, "shop"), (9, "early"), (3, "dash"), (0, "store"), (1, "\u0435arly"),
             (2, "cheyandi"), (4, ""))
    got, why = check_line(line(full=full), spec())
    assert why == [] and got.full.english == ((0, "shop"),)
    assert got.flags == ("full english: 9 is not a word of the line", "full english: 3 is not a word of the line",
                         "full english: word 0 listed twice", "full english: '\u0435arly' has Cyrillic letters",
                         "full english: 'cheyandi' is romanised Telugu", "full english: '' is empty")


def test_english_names_with_digits_capitals_and_symbols_pass():
    full = w("జీపీటీ ఫోర్ తో సీ ప్లస్ ప్లస్ రాయొచ్చు.", (0, "GPT-4"), (2, "C++"), (1, "four"))
    got, _ = check_line(line(full=full), spec(en="You can write C++ with GPT-4."))
    assert got.full.english == ((0, "GPT-4"), (1, "four"), (2, "C++")) and got.flags == ()


# ---- tiers -----------------------------------------------------------------------------------------------------------
def test_tiers_are_kept_in_strict_akshara_order_and_bad_ones_dropped():
    raw = line(full=w("షాప్ పొద్దున్నే తెరుస్తారు అండి.", (0, "shop")),
               concise=w("షాప్ పొద్దున్నే తెరుస్తారు.", (0, "shop")),
               very_concise=w("షాప్ పొద్దున్నే తెరుస్తారు.", (0, "shop")),     # not shorter than concise
               fuller=w("షాప్ పొద్దున్నే."))                                    # not longer than full
    got, why = check_line(raw, spec(want=("full", "concise", "very_concise", "fuller")))
    assert why == [] and list(got.tiers) == ["concise", "full"]
    assert "very_concise dropped: 8.5 aksharas, not shorter than 8.5" in got.flags
    assert "fuller dropped: 4.5 aksharas, not longer than 10.5" in got.flags
    assert "very_concise missing" in got.flags and "fuller missing" in got.flags


def test_a_shorter_tier_that_breaks_the_script_is_dropped_not_the_line():
    raw = line(concise=w("shop తెరుస్తారు."))
    got, why = check_line(raw, spec(want=("full", "concise")))
    assert why == [] and list(got.tiers) == ["full"]
    assert got.flags == ("concise dropped: Latin letters in shop", "concise missing")


def test_whitespace_is_normalised():
    got, _ = check_line(line(full=w("  షాప్   పొద్దున్నే\nతెరుస్తారు. ", (0, "shop"))), spec())
    assert got.full.spoken == "షాప్ పొద్దున్నే తెరుస్తారు."


# ---- pieces ----------------------------------------------------------------------------------------------------------
FULL = "షాప్ పొద్దున్నే తెరుస్తారు, కానీ సాయంత్రం తొందరగా మూసేస్తారు."


@pytest.mark.parametrize("pieces,breaks,kept,flag", [
    (["షాప్ పొద్దున్నే తెరుస్తారు,", "కానీ సాయంత్రం తొందరగా మూసేస్తారు."], (1.4,), True, None),
    (["షాప్ పొద్దున్నే  తెరుస్తారు, ", "కానీ సాయంత్రం తొందరగా మూసేస్తారు."], (1.4,), True, None),
    (["షాప్ పొద్దున్నే తెరుస్తారు,", "కానీ సాయంత్రం మూసేస్తారు."], (1.4,), False, "they don't join back to full"),
    (["షాప్", "పొద్దున్నే తెరుస్తారు,", "కానీ సాయంత్రం తొందరగా మూసేస్తారు."], (1.4,), False, "3 pieces for 1 breaks"),
    (["షాప్ పొద్దున్నే తెరుస్తారు,", "కానీ సాయంత్రం తొందరగా మూసేస్తారు."], (), False, "the line has no breaks"),
    (["షాప్ పొద్దున్నే తెరుస్తారు,", ""], (1.4,), False, "not a list of wordings"),
])
def test_pieces_must_join_back_to_full_within_the_break_count(pieces, breaks, kept, flag):
    raw = line(full=w(FULL, (0, "shop")), pieces=pieces, moved=["early"])
    got, _ = check_line(raw, spec(en="The shop opens early, but it closes early too.", breaks=breaks))
    assert bool(got.pieces) == kept
    assert got.moved == (("early",) if kept else ())
    if flag:
        assert f"pieces dropped: {flag}" in got.flags


# ---- ids -------------------------------------------------------------------------------------------------------------
def test_every_requested_id_exactly_once():
    specs = [spec(1), spec(2, "The bus is late."), spec(3, "It is raining.")]
    reply = {"lines": [line(1), line(2, w("బస్సు లేట్.", (1, "late"))), line(2, w("బస్సు లేట్.", (1, "late"))),
                       line(9, w("ఇంకొకటి.")), line(3.0, w("వాన పడుతోంది."))],
             "glossary_additions": [{"term": "shop", "spoken": "షాప్"}]}
    out = check_reply(reply, specs)
    assert set(out.lines) == {1, 3}
    assert out.rejected == {2: ["returned 2 times"]}
    assert out.unexpected == [9]
    assert [g.term for g in out.glossary_additions] == ["shop"]
    assert check_reply({"lines": []}, specs[:1]).rejected == {1: ["missing from the reply"]}
    assert check_reply(None, specs[:1]).rejected == {1: ["missing from the reply"]}


def test_glossary_additions_must_keep_the_script_contract():
    got = glossary_entries([{"term": "shop", "spoken": "షాప్"}, {"term": "bus", "spoken": "bus"},
                            {"term": " ", "spoken": "ఏదో"}, {"term": "Shop", "spoken": "షాపు"},
                            {"term": "fare", "spoken": "చార్జీ", "keep_english": False, "note": "not ఫేర్"}, "junk"])
    assert [(g.term, g.spoken, g.keep_english, g.note) for g in got] == [("shop", "షాప్", True, ""),
                                                                         ("fare", "చార్జీ", False, "not ఫేర్")]


# ---- delivery --------------------------------------------------------------------------------------------------------
def test_delivery_is_repaired_not_rejected():
    raw = line(delivery={"emotion": "bored", "energy": "mid", "question": True, "emphasis": [0, 7, 0, True]})
    got, why = check_line(raw, spec())
    assert why == [] and got.delivery.emotion == "neutral" and got.delivery.question is True
    assert got.delivery.emphasis == (0,)
    assert "delivery: emotion 'bored' unknown" in got.flags and "delivery: emphasis 7 is not a word of the line" in got.flags


# ---- flags -----------------------------------------------------------------------------------------------------------
def test_a_glossary_term_must_keep_its_fixed_spelling():
    full = Wording("బండి పొద్దున్నే వస్తుంది.")
    assert suspect("The van comes early.", full, {"van": "వ్యాన్"}) == ["glossary: van not written వ్యాన్"]
    assert suspect("The van comes early.", Wording("వ్యాన్ పొద్దున్నే వస్తుంది."), {"van": "వ్యాన్"}) == []
    assert suspect("Advantage comes early.", full, {"van": "వ్యాన్"}) == []  # not the word van


@pytest.mark.parametrize("en,te,counts", [
    ("I'm not tired.", "నాకు అలసట లేదు.", (1, 1)),
    ("I'm not tired.", "నేను బాగా అలసిపోయా.", (1, 0)),                      # negation lost
    ("It could mean anything.", "దానికి అర్థం ఏమీ కాదు.", (0, 1)),          # negation added
    ("Don't go there.", "అక్కడికి వెళ్ళొద్దు.", (1, 1)),
    ("Come in the morning.", "పొద్దున్నే రండి, పొద్దు పోకముందే.", (0, 0)),         # పొద్దు is morning
    ("She never knew.", "ఆమెకి ఎప్పుడూ తెలీదు.", (1, 1)),
    ("He can't come.", "అతను రాలేడు, రాడు కూడా.", (1, 1)),                  # లేడు counts; a bare -డు (రాడు) doesn't
    ("He isn't at home.", "అతను ఇంట్లో లేడు.", (1, 1)),
    ("He came when it rained.", "వాన పడినప్పుడు వాడు వచ్చాడు.", (0, 0)),     # -డు words that aren't negative
    ("Before the rain.", "వానకి ముందు.", (0, 0)),                            # ముందు is not a negation
    ("Tea or coffee?", "టీ కావాలా లేదా కాఫీ?", (0, 0)),                     # లేదా is "or"
    ("Five people came.", "ఐదు మంది వచ్చారు.", (0, 0)),                      # ఐదు is five
    ("Nobody knew it, without a doubt.", "ఎవరికీ తెలియకుండా, సందేహం లేకుండా.", (2, 2)),
])
def test_negation_counts(en, te, counts):
    assert negations(en, te) == counts


def test_negation_and_question_parity_are_flags():
    assert "negation: 1 in the English, 0 in the Telugu" in suspect("I'm not tired.", Wording("నేను అలసిపోయా."))
    assert "negation: 0 in the English, 1 in the Telugu" in suspect("I'm tired.", Wording("నేను అలసిపోలేదు."))
    assert "question: lost" in suspect("Are you tired?", Wording("మీరు అలసిపోయారు."))
    assert "question: added" in suspect("You are tired.", Wording("మీరు అలసిపోయారా?"))


@pytest.mark.parametrize("en,full,missing", [
    ("It costs 40 rupees.", Wording("దాని ధర నలభై రూపాయలు."), []),
    ("It costs 40 rupees.", Wording("దాని ధర కొంచెం ఎక్కువ."), ["40"]),
    ("It took two years.", Wording("రెండేళ్ళు పట్టింది."), []),                     # inflected numeral
    ("It grew 60% in 3.5 years.", Wording("మూడున్నర ఏళ్ళలో అరవై శాతం పెరిగింది."), []),
    ("About 1,500 people.", Wording("వెయ్యిన్నర మంది."), []),
    ("Two thousand people.", Wording("రెండు వేల మంది."), []),
    ("Ten percent off.", Wording("టెన్ పర్సెంట్ ఆఫ్.", ((0, "ten"), (1, "percent"), (2, "off"))), []),  # said in English
    ("The 21st time, with GPT-4 on an M5.", Wording("ఇరవై ఒకటో సారి."), []),      # ordinals and names aren't numbers
    ("Twenty-five of them.", Wording("వాటిలో ఇరవై."), ["five"]),
    ("One of them.", Wording("వాటిలో ఒకటి."), []),
    ("Give me 1 minute.", Wording("నాకు ఒక నిమిషం ఇవ్వండి."), []),             # ఒక before a noun is 1
    ("Give me 1 minute.", Wording("నాకు కొంచెం టైం ఇవ్వండి.", ((2, "time"),)), ["1"]),
])
def test_numbers_must_be_said(en, full, missing):
    assert unsaid_numbers(en, full) == missing
    assert any(f.startswith("numbers not said") for f in suspect(en, full)) == bool(missing)
