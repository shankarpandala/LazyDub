import pytest
from hypothesis import given, strategies as st

from maata_engine.text.asr_fix import fix_text, fix_words, joins
from maata_engine.types import TimedWord


@pytest.mark.parametrize("raw, fixed", [
    ("The loss came to 17 .72 percent last year.", "The loss came to 17.72 percent last year."),
    ("We sold 3 ,000 copies.", "We sold 3,000 copies."),
    ("That is 1 ,000 ,000 rows.", "That is 1,000,000 rows."),
    ("It cost 1 ,00,000 rupees.", "It cost 1,00,000 rupees."),
    ("Nearly 50 % of users, maybe 50 %.", "Nearly 50% of users, maybe 50%."),
    ("A rise of 3,000 .50 dollars, or 17 .72 %.", "A rise of 3,000.50 dollars, or 17.72%."),
    ("We meet at 10 :30 tomorrow.", "We meet at 10:30 tomorrow."),
    ("It lasted 1:05 :09 in total.", "It lasted 1:05:09 in total."),
    ("It costs $ 5 ,000 now.", "It costs $5,000 now."),
    ("Prices (₹ 250) went up.", "Prices (₹250) went up."),
    ("A drop of -0 .5 degrees.", "A drop of -0.5 degrees."),
])
def test_split_numbers_are_rejoined(raw, fixed):
    assert fix_text(raw) == fixed


@pytest.mark.parametrize("text", [
    "It happened in 2017. 72 people came.",       # a sentence end, then a number
    "Pick items 3, 400 and 12.",                   # a list
    "Page 2017 ,000 is odd.",                      # four digits cannot take a thousands group
    "It was 3 ,00 or so.",                         # a group needs three digits
    "Paper size A4 .5 is not a thing.",            # not a number on the left
    "Version 1.2 .3 shipped.",                     # a decimal after a decimal
    "At 10 :3 or 100 :30 nothing changes.",        # clock parts are two digits
    "The score was 3 . 2 in the end.",             # a bare point
    "Costs $ and cents.",
    "Up by five % only.",                          # a number word is not digits
])
def test_everything_else_is_left_alone(text):
    assert fix_text(text) == text


def test_whitespace_is_kept_except_inside_a_rejoined_number():
    assert fix_text("  17 .72\tand\n3 ,000  ") == "  17.72\tand\n3,000  "
    assert fix_text("17\t.72") == "17.72"
    assert fix_text("3\n,000") == "3\n,000"        # a line break is not a split number
    assert fix_text("") == "" and fix_text("   ") == "   "


def test_joins():
    assert joins("17", ".72") and joins("3", ",000.") and joins("50", "%,") and joins("$", "5")
    assert not joins("17.", "72") and not joins("", ".5") and not joins("$", "five") and not joins("x", "%")


def test_fix_words_merges_times_and_keeps_the_lowest_confidence():
    ws = [TimedWord("about", 0.0, 0.3), TimedWord("17", 0.4, 0.7, 0.9), TimedWord(".72", 0.7, 1.1, 0.6),
          TimedWord("%", 1.1, 1.3, 0.8), TimedWord("of", 1.4, 1.5), TimedWord("it.", 1.5, 1.8)]
    out = fix_words(ws)
    assert out == [TimedWord("about", 0.0, 0.3), TimedWord("17.72%", 0.4, 1.3, 0.6), TimedWord("of", 1.4, 1.5),
                   TimedWord("it.", 1.5, 1.8)]
    assert len(ws) == 6                            # the input is not changed
    assert fix_words([]) == [] and fix_words(ws[:1]) == ws[:1]


_pieces = st.lists(st.sampled_from(["17", ".72", "3", ",000", "%", "$", "10", ":30", "2017.", "72", "the", "A4", ".5",
                                    ",00", "1.2", "and,", "5"]), max_size=12)


@given(_pieces)
def test_fix_words_matches_fix_text_and_never_loses_characters(texts):
    ws = [TimedWord(x, i, i + 0.5) for i, x in enumerate(texts)]
    out = fix_words(ws)
    assert " ".join(w.text for w in out) == fix_text(" ".join(texts))
    assert "".join(w.text for w in out) == "".join(texts)
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))
    once = fix_text(" ".join(texts))
    assert fix_text(once) == once
