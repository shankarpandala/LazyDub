import pytest

from maata_engine.text.akshara import count_telugu, count_units, english_syllables, mixed_units
from maata_engine.text.normalize_te import normalize_telugu, number_to_telugu


@pytest.mark.parametrize(
    "word,expected",
    [
        ("క్షమించండి", 4),   # క్ష · మిం · చం · డి
        ("స్త్రీ", 1),
        ("అన్నం", 2),
        ("ప్రపంచం", 3),
        ("సత్యం", 2),
        ("నమస్కారం", 4),
        ("మాట", 2),
        ("వెళ్ళాన్", 2.5),  # word-final pollu weighted 0.5
    ],
)
def test_telugu_aksharas(word, expected):
    assert count_telugu(word) == expected


def test_zwnj_half_form_is_weighted():
    assert count_telugu("కార్‌యం") == 2.5


@pytest.mark.parametrize("word,n", [("machine", 2), ("learning", 2), ("video", 2), ("code", 1), ("the", 1), ("table", 2), ("computer", 3)])
def test_english_syllables(word, n):
    assert english_syllables(word) == n


def test_a_dub_line_is_counted_in_telugu_script_only():
    # English is written in Telugu script (ARCHITECTURE §3.8): its aksharas are its length, and Latin letters count 0
    assert count_units("మనం మెషిన్ లెర్నింగ్ నేర్చుకుందాం") == 2 + 2.5 + 2.5 + 4  # మె·షి·న్ and లె·ర్నిం·గ్
    assert count_units("మనం machine learning నేర్చుకుందాం") == 2 + 4
    assert count_units("3 రోజులు") == count_units("మూడు రోజులు") == 5  # digits as they are said


def test_mixed_units_count_latin_english_in_syllables():
    # "machine learning" counts 4 syllables, not 15 letters; an English source line is measured this way
    assert mixed_units("మనం machine learning నేర్చుకుందాం") == 2 + 4 + 4
    assert mixed_units("Please bring the tickets.") == 1 + 1 + 1 + 2
    assert mixed_units("It costs 5 dollars.") == 1 + 1 + 2 + 2  # the digit as its spoken Telugu (ఐదు)


@pytest.mark.parametrize(
    "n,words",
    [
        (0, "సున్నా"), (7, "ఏడు"), (15, "పదిహేను"), (21, "ఇరవై ఒకటి"), (100, "వంద"), (105, "నూట ఐదు"),
        (250, "రెండు వందల యాభై"), (1000, "వెయ్యి"), (2024, "రెండు వేల ఇరవై నాలుగు"),
        (100000, "లక్ష"), (250000, "రెండు లక్షల యాభై వేలు"), (10000000, "కోటి"),
    ],
)
def test_numbers(n, words):
    assert number_to_telugu(n) == words


def test_normalizer_currency_percent_digits():
    assert normalize_telugu("₹500 కి") == "ఐదు వందలు రూపాయలు కి"
    assert normalize_telugu("50% మంది") == "యాభై శాతం మంది"
    assert normalize_telugu("౩ పుస్తకాలు") == "మూడు పుస్తకాలు"
    assert normalize_telugu("3.5 GB") == "మూడు పాయింట్ ఐదు GB"
    assert normalize_telugu("1,000 users") == "వెయ్యి users"
