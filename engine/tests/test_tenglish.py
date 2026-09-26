"""The English-word lint, the Latin rebuild of a Telugu-script line from its English map, and the code-mixing index.
Every sentence below is original."""

import re
from pathlib import Path

import pytest

from maata_engine.text import tenglish as t

QUOTA = re.compile(r"\bthird\b|\d+\s*%|\bpercent(age)?\s+of\b|\bquota\b|\bratio\b.*\btarget\b", re.I)


def test_no_english_quota_anywhere_in_the_module():
    source = Path(t.__file__).read_text(encoding="utf-8")
    assert not QUOTA.search(source.replace("ratio (", "")), "a quota crept in"
    assert not re.search(r"\bthird\b|30\s*%", source)


# ---- the English-word map --------------------------------------------------------------------------------

@pytest.mark.parametrize("spoken,english,latin", [
    ("సెట్టింగ్స్ ఓపెన్ చేసి, వైఫై ఆఫ్ చేయండి.", [(0, "settings"), (1, "open"), (3, "Wi-Fi"), (4, "off")],
     "settings open చేసి, Wi-Fi off చేయండి."),
    ("పోయిన వీకెండ్ నెట్ఫ్లిక్స్లో ఏం చూశారు?", [(1, "weekend"), (2, "Netflix")], "పోయిన weekend Netflix లో ఏం చూశారు?"),
    ("మా రౌటర్ని రీస్టార్ట్ చేశా.", [(1, "router"), (2, "restart")], "మా router ని restart చేశా."),
    ("\"ఫోన్,\" అని అడిగాడు.", [(0, "phone")], "\"phone,\" అని అడిగాడు."),
    ("ఫ్లో బాగుంది, హలో చెప్పండి.", [(0, "flow"), (2, "hello")], "flow బాగుంది, hello చెప్పండి."),  # not a case ending
    ("ఇంట్లో అందరూ బాగున్నారు.", [], "ఇంట్లో అందరూ బాగున్నారు."),
    ("ఒక మాట.", [(5, "stray"), (0, " ")], "ఒక మాట."),  # indices past the line and empty words are ignored
])
def test_latin_spoken_rebuilds_english_words_in_latin(spoken, english, latin):
    assert t.latin_spoken(spoken, english) == latin


def test_the_lint_knows_english_words_from_the_map_not_from_latin_letters():
    source = "They got the best results."
    spoken = "ద బెస్ట్ రిజల్ట్స్ వచ్చాయి."  # "the" said in English, written in Telugu script like every English word
    english = [(0, "the"), (1, "best"), (2, "results")]
    assert t.lint(spoken, english, source) == ["English grammar words (say these in Telugu): the",
                                               "English phrase left untranslated: the best results"]
    assert t.lint(spoken, [], source) == []  # nothing mapped: nothing English
    # Latin letters in `spoken` break the script contract (the validators reject the line); they aren't English words
    assert t.lint("the బెస్ట్ రిజల్ట్స్ వచ్చాయి.", [(1, "best"), (2, "results")], source) == []


def test_the_lint_reads_mapped_words_with_their_case_endings():
    source = "Our sales improved a lot on Netflix."
    spoken = "నెట్ఫ్లిక్స్లో మా సేల్స్ బాగా ఇంప్రూవ్డ్ అయ్యాయి."
    english = [(0, "Netflix"), (2, "sales"), (4, "improved")]
    assert t.lint(spoken, english, source) == [
        "inflected English before చేయు/అవు (use a Telugu verb or the bare stem): improved"]
    better = "నెట్ఫ్లిక్స్లో మా సేల్స్ బాగా ఇంప్రూవ్ అయ్యాయి."
    assert t.lint(better, [(0, "Netflix"), (2, "sales"), (4, "improve")], source) == []
    # an untranslated clause spread over mapped words, with a case ending closing it
    clause = "మీకు యూ వాంట్ మోర్ కదా?"
    assert t.lint(clause, [(1, "you"), (2, "want"), (3, "more")], "") == [
        "English grammar words (say these in Telugu): you, more", "English phrase left untranslated: you want more"]


def test_the_lint_measures_and_compares_the_telugu_script_line():
    spoken = "మనం మెషిన్ లెర్నింగ్ నేర్చుకుందాం."  # 11 aksharas, its English words' included
    english = [(1, "machine"), (2, "learning")]
    assert t.lint(spoken, english, "", target_units=11) == []
    assert t.lint(spoken, english, "", target_units=5) == ["too long: 11 aksharas for a target of about 5"]
    # an echo of the line before, whose English words are in Telugu script too
    prev = ("Let us learn machine learning.", spoken)
    again = "మనం మెషిన్ లెర్నింగ్ నేర్చుకుందాం, సరేనా?"
    assert t.lint(again, english, "Then we can build models.", context=[prev]) == [t.ECHO]


def test_cmi_is_the_minority_languages_share():
    assert t.cmi([]) == 0.0
    assert t.cmi([("ఇంట్లో అందరూ బాగున్నారు.", [])]) == 0.0
    assert t.cmi([("సెట్టింగ్స్ ఓపెన్ చేయండి.", [(0, "settings"), (1, "open")])]) == pytest.approx(100 / 3)
    lines = [("ఒకటి రెండు మూడు నాలుగు", [(0, "one")]), ("ఐదు ఆరు", [(1, "six"), (1, "six")])]
    assert t.cmi(lines) == pytest.approx(100 * 2 / 6)  # a repeated index counts once


# ---- names in the English -------------------------------------------------------------------------------

@pytest.mark.parametrize("source,terms", [
    ("I tried the new Pixel camera with gcam.", ["Pixel"]),
    ("Have you watched The Morning Show on Apple TV?", ["The Morning Show", "Apple TV"]),
    ("GPT-4 beat GPT-3.5 at NASA on my M5 Pro.", ["GPT-4", "GPT-3.5", "NASA", "M5 Pro"]),
    ("I bought an iPhone 17 Pro Max.", ["iPhone 17 Pro Max"]),
    ("Game of Thrones is long.", ["Game of Thrones"]),
    ("The Lord of the Rings is long.", ["Lord of the Rings"]),
    ("My friend Ramesh loves Google and Apple.", ["Ramesh", "Google", "Apple"]),
    ("We ate at McDonald's.", ["McDonald's"]),
    ("But I think it's wrong.", []),
    ("So I went there.", []),
    ("You Know What, forget it.", []),
    ("On Monday my Mom called.", []),
    ("Oh my God, this is great.", []),
    ("Netflix raised prices.", []),  # a lone sentence-initial word can't be told from "Yesterday"
])
def test_source_keyterms(source, terms):
    assert t.source_keyterms(source) == terms


# ---- lint ----------------------------------------------------------------------------------------------
# The cases below are written as Latin-script Tenglish, for reading; `mapped` turns one into what the lint reads: the
# Telugu-script line, each English word a stand-in (punctuation kept), and the English map naming those words.
STAND_IN = "పదం"
_EDGE_PUNCT = ".,!?;:\"'“”‘’()"


def mapped(line: str) -> tuple[str, list[tuple[int, str]]]:
    words, english = [], []
    for i, word in enumerate(line.split()):
        core = word.strip(_EDGE_PUNCT)
        if core and re.search(r"[A-Za-zÀ-ɏ]", core):
            lead = word.index(core)
            words.append(word[:lead] + STAND_IN + word[lead + len(core):])
            english.append((i, core))
        else:
            words.append(word)
    return " ".join(words), english


def test_mapped_writes_english_words_as_stand_ins():
    assert mapped("\"Don't,\" మా friend's phone పోయింది.") == (
        "\"పదం,\" మా పదం పదం పోయింది.", [(0, "Don't"), (2, "friend's"), (3, "phone")])
    assert mapped("ఇంట్లో అందరూ బాగున్నారు.") == ("ఇంట్లో అందరూ బాగున్నారు.", [])


def lint_of(line, source="", **kw):
    return t.lint(*mapped(line), source, **kw)


def flags_for(line, source="", **kw):
    return " | ".join(lint_of(line, source, **kw))


def grammar_words(line, source="", **kw):
    flag = next((f for f in lint_of(line, source, **kw) if f.startswith("English grammar words")), "")
    return flag.split(": ", 1)[1].split(", ") if flag else []


@pytest.mark.parametrize("line,source,word", [
    ("వాళ్ళకి the best results వచ్చాయి.", "They got the best results.", "the"),
    ("I think ఇది సరైనదే.", "I think this is right.", "I"),
    ("ఇంకా more practice కావాలి.", "You need more practice.", "more"),
    ("Don't అలా చేయకండి.", "Don't do that.", "Don't"),
])
def test_lint_flags_english_grammar_words(line, source, word):
    assert f"English grammar words (say these in Telugu): {word}" in flags_for(line, source)


@pytest.mark.parametrize("line", [
    "By the way, రేపు కలుద్దాం.", "Thank you అండీ, మళ్ళీ కలుద్దాం.", "Of course, నేను వస్తా.", "No problem, తర్వాత చూద్దాం.",
    "At least ఒక్కసారైనా try చేయండి.", "ముందు account లో log in చేయండి.", "US లో IT jobs ఎక్కువ.", "Plan A పని చేయలేదు.",
    "నిన్న shopping చేశాం.", "ఆ news విని excited అయ్యా.", "మా sales బాగా improve అయ్యాయి.",
])
def test_lint_passes_natural_lines(line):
    assert lint_of(line, "") == []


@pytest.mark.parametrize("line,source", [
    ("For example, మీ phone చూడండి.", "For example, look at your phone."),
    ("In fact, అది నిజమే.", "In fact, it's true."),
    ("Step by step చెప్తా.", "I'll explain it step by step."),
    ("ఆయన ఎప్పుడూ on time వస్తారు.", "He always comes on time."),
    ("ఇప్పుడు work from home చేస్తున్నా.", "I'm working from home now."),
    ("Oh my God, ఇది చాలా బాగుంది.", "Oh my God, this is great."),
    ("Trust me, ఇది పనిచేస్తుంది.", "Trust me, this works."),
    ("దీనికి will power కావాలి.", "You need will power for this."),
    ("A to Z అన్నీ చెప్తా.", "I'll tell you everything from A to Z."),
    ("McDonald's లో తిన్నాం.", "We ate at McDonald's."),
    ("Domino's pizza order చేశా.", "I ordered a Domino's pizza."),
    ("ఐదు o'clock కి వస్తా.", "I'll come at five o'clock."),
    ("Subscribers two hundred అయ్యారు.", "We now have two hundred subscribers."),
    ("మా cousin wedding అయిపోయింది.", "My cousin's wedding is over."),
    ("Opening అయ్యింది.", "The opening happened."),
    ("ఆయన machine learning engineer.", "He is a machine learning engineer."),
    ("మా customer support team చాలా fast.", "Our customer support team is very fast."),
    ("Deep learning model train చేశాం.", "We trained a deep learning model."),
    ("Google and Apple రెండూ అదే అన్నాయి.", "Both Google and Apple said the same."),
    ("Game of Thrones చూశారా?", "Game of Thrones is on tonight."),
])
def test_lint_passes_set_phrases_brands_and_noun_compounds(line, source):
    assert lint_of(line, source) == []


@pytest.mark.parametrize("line,source,words", [
    ("But I think అది తప్పు.", "But I think it's wrong.", ["I"]),
    ("So I అక్కడికి వెళ్ళా.", "So I went there.", ["I"]),
    ("And I అలా అనుకున్నా.", "And I thought so.", ["And", "I"]),
    ("Okay. So I మళ్ళీ try చేశా.", "Okay. So I tried again.", ["I"]),
    ("You Know What, అది వదిలేయ్.", "You Know What, forget it.", ["You", "What"]),
    ("In India లో ఇది ఎక్కువ.", "In India this is common.", ["In"]),
])
def test_lint_sees_grammar_words_that_open_a_sentence(line, source, words):
    assert grammar_words(line, source) == words


@pytest.mark.parametrize("line,word", [
    ("It's చాలా simple.", "It's"), ("Let's మొదలుపెడదాం.", "Let's"), ("మా friend's phone పోయింది.", "friend's"),
    ("I'm ready అయ్యా.", "I'm"), ("Don't అలా చేయకండి.", "Don't"),
])
def test_lint_still_flags_contractions_and_possessives(line, word):
    assert word in grammar_words(line)


def test_lint_masks_the_sources_own_terms_and_keeps_acronyms_exact():
    assert lint_of("కొత్త Pixel లో photos బాగా వస్తాయి.", "Photos come out well on the new Pixel.") == []
    assert grammar_words("నేను it లో పని చేస్తా.", "I work in IT.", keyterms=["IT"]) == ["it"]
    assert lint_of("నేను IT లో పని చేస్తా.", "I work in IT.", keyterms=["IT"]) == []


def test_lint_flags_untranslated_clauses_but_not_noun_compounds():
    assert "English phrase left untranslated: you want more" in flags_for("నాకు తెలుసు you want more కదా.")
    assert "English phrase left untranslated" not in flags_for("మా customer support team చాలా fast.")


def test_lint_flags_inflected_english_before_a_light_verb():
    assert "improved" in flags_for("మా sales బాగా improved అయ్యాయి.", "Our sales improved a lot.")
    assert "explaining" in flags_for("ఇప్పుడు ఇది explaining చేస్తా.", "Now I'll explain this.")
    assert "cancelled" in flags_for("Flight cancelled అయ్యింది.", "The flight got cancelled.")
    assert lint_of("మా sales బాగా improve అయ్యాయి.", "Our sales improved a lot.") == []
    assert lint_of("Flight cancel అయ్యింది.", "The flight got cancelled.") == []


def test_lint_leaves_ing_topics_and_adjectives_alone():
    assert lint_of("Reading అయితే నాకు చాలా ఇష్టం.", "As for reading, I love it.") == []
    assert lint_of("అది నిజంగా amazing అయిన చోటు.", "That is a truly amazing place.") == []


def test_lint_flags_everyday_verbs_said_in_english():
    assert "everyday verbs said in English (use the Telugu verb): think" in flags_for("చాలామంది అలా think చేస్తారు.")
    assert "remember" in flags_for("ఆ రోజు నాకు బాగా remember అవుతుంది.")


def test_lint_flags_untranslated_phrases_but_not_numbers_or_verb_stems():
    assert "English phrase left untranslated: really cool new features" in flags_for("ఇందులో really cool new features ఉన్నాయి.")
    assert lint_of("మొదట్లో ten percent off ఉంటుంది.", "") == []
    assert lint_of("తర్వాత automatic updates off చేయండి.", "") == []


def test_lint_brackets_scripts_and_bookish_words():
    assert flags_for("ఇది చాలా ముఖ్యం (important).") == "brackets or glosses"
    assert flags_for("ఇది बहुत ముఖ్యం.").startswith("characters from another script:")
    assert flags_for("మనకి time మరియు energy రెండూ కావాలి.") == "bookish words: మరియు"
    assert lint_of("మనకి time మరియు energy రెండూ కావాలి.", "", style="formal") == []
    assert "యొక్క" in flags_for("ఈ plan యొక్క అసలు ఉద్దేశం వేరు.")


def test_lint_catches_an_echo_of_a_context_line_unless_the_source_repeats_too():
    prev = ("We should leave early tomorrow.", "రేపు పొద్దున్నే తొందరగా బయల్దేరదాం, సరేనా?")
    echo = "రేపు పొద్దున్నే తొందరగా బయల్దేరదాం, సరేనా?"
    assert "repeats an earlier line" in flags_for(echo, "Then we can reach by noon.", context=[prev])
    assert "repeats an earlier line" in flags_for(echo, "Then we can reach by noon.", context=[prev[1]])
    assert lint_of(echo, "We should leave early tomorrow, okay?", context=[prev]) == []
    assert lint_of("మధ్యాహ్నానికల్లా చేరిపోతాం.", "Then we can reach by noon.", context=[prev]) == []


def test_lint_length_window():
    line = "ఇదంతా మొదలవ్వకముందు మీరేం చేసేవారు?"  # 16 aksharas
    assert lint_of(line, "", target_units=16) == []
    assert lint_of(line, "", target_units=9) == []
    assert flags_for(line, target_units=7).startswith("too long: 16 aksharas for a target of about 7")
    assert flags_for(line, target_units=41).startswith("too short: 16 aksharas")


def test_too_short_is_only_a_flag():
    target = 10.1  # "I don't know." drawn out over 1.6 s, plus a borrowed half second
    assert lint_of("తెలీదు.", "I don't know.", target_units=target) == ["too short: 3 aksharas for a target of about 10"]


def test_lint_keeps_names_from_keyterms_and_the_source():
    line = "The Morning Show లో ఈ interview వచ్చింది."
    assert "The" in flags_for(line, "This interview aired on a TV programme.")
    assert lint_of(line, "This interview aired on a TV programme.", keyterms=["The Morning Show"]) == []
    assert lint_of(line, "This interview aired on The Morning Show.") == []


@pytest.mark.parametrize("raw", [
    "దీని ధర €50 మాత్రమే.", "£20 ఇచ్చా.", "బయట 40°C ఉంది.", "Pokémon cards కొన్నా.", "José తో మాట్లాడా.", "Café లో కలుద్దాం.",
])
def test_lint_keeps_symbols_and_accented_names(raw):
    assert "characters from another script" not in " | ".join(t.lint(raw, [], ""))


def test_lint_flags_other_scripts():
    assert flags_for("ఇది ა ముఖ్యం 好 సరే.") == "characters from another script: ა, 好"
