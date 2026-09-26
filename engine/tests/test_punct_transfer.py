"""Punctuation transfer (ARCHITECTURE §3.4): the prompted pass lends casing and trailing marks, never words. All text here
is original test text."""

from __future__ import annotations

from maata_engine.text.punct_transfer import transfer, unguess
from maata_engine.types import TimedWord


def timed(text: str, t: float = 0.0, word: float = 0.3, gap: float = 0.1, conf: float = 0.9) -> list[TimedWord]:
    out = []
    for x in text.split():
        out.append(TimedWord(x, round(t, 3), round(t + word, 3), conf))
        t += word + gap
    return out


def texts(ws: list[TimedWord]) -> str:
    return " ".join(w.text for w in ws)


def test_marks_and_casing_come_over_and_everything_else_stays():
    plain = timed("so we packed the van early and then the rain started nobody expected it")
    styled = timed("So we packed the van early, and then the rain started. Nobody expected it.", t=0.05)
    got = transfer(plain, styled)
    assert texts(got) == "So we packed the van early, and then the rain started. Nobody expected it."
    assert [(w.start, w.end, w.confidence) for w in got] == [(w.start, w.end, w.confidence) for w in plain]
    assert texts(plain) == "so we packed the van early and then the rain started nobody expected it"  # not changed


def test_no_word_is_ever_taken_from_the_prompted_pass():
    """The prompted pass drops a passage and invents a closing line: the transcript keeps every word it heard and gains
    none."""
    plain = timed("first we test the brakes then we check the lights and finally we sign the form")
    styled = (timed("First, we test the brakes.") + timed("and finally, we sign the form.", t=4.4)
              + timed("Thanks for watching.", t=7.0))
    got = transfer(plain, styled)
    assert [w.text.strip(".,").lower() for w in got] == [w.text for w in plain]
    assert texts(got) == "First, we test the brakes. then we check the lights and finally, we sign the form."


def test_a_chance_match_far_away_copies_nothing():
    plain = timed("the gate opens at nine")
    styled = timed("The gate opens at nine.", t=30.0)                   # the same words, 30 s later
    assert texts(transfer(plain, styled)) == "the gate opens at nine"
    assert texts(transfer(plain, styled, max_shift=40.0)) == "The gate opens at nine."


def test_casing_keeps_i_and_acronyms_and_lowers_a_stray_capital():
    plain = timed("Then I'm sure the GPU and I will manage")
    styled = timed("then i'm sure the gpu and i will manage.")
    assert texts(transfer(plain, styled)) == "then I'm sure the GPU and I will manage."


def test_marks_replace_the_words_own_but_keep_what_is_not_a_mark():
    plain = timed("prices rose 5% right. okay, fine")
    styled = timed("Prices rose 5%, right? Okay fine.")
    assert texts(transfer(plain, styled)) == "Prices rose 5%, right? Okay, fine."
    assert texts(transfer(timed("about 50% more"), timed("About 50, more."))) == "About 50%, more."  # its % stays
    # A curly apostrophe still aligns.
    assert texts(transfer(timed("we don’t know"), timed("We don't know."))) == "We don’t know."


def test_empty_passes():
    plain = timed("hello there")
    assert transfer(plain, []) == plain and transfer([], timed("Hello there.")) == []


def test_only_the_guess_at_a_window_cut_is_left_out():
    """A window cut into speech at 30 s ends with a guessed mark and the next opens with a guessed capital; a real mark a
    word earlier, and anything further from the cut, stays."""
    tail = timed("the tide came in fast. Then we", t=27.2)             # "we" ends 29.9 s, "fast." 29.1 s
    tail[-1] = TimedWord("we.", tail[-1].start, tail[-1].end)
    got, n = unguess(tail, None, 30.0)
    assert texts(got) == "the tide came in fast. Then we" and n == 1
    head = timed("Ran for the steps.", t=30.1)
    got, n = unguess(head, 30.0, None)
    assert texts(got) == "for the steps." and n == 1
    assert unguess(timed("I ran for the steps.", t=30.1), 30.0, None) == (timed("I ran for the steps.", t=30.1), 0)
    assert unguess(timed("Ran for the steps.", t=31.2), 30.0, None)[1] == 0      # 1.2 s from the cut: not a guess
    assert unguess(head, None, None) == (head, 0)                                # the audio's own edges
    both = [TimedWord("Yes.", 30.2, 30.5)]
    assert unguess(both, 30.0, 31.0) == ([], 2)
