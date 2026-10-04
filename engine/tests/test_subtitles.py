"""The two soft subtitle tracks' cues (OFFLINE-RENDER §2.16): rows of 42 characters (English) or aksharas (Telugu), at
most 2 a speaker, long text split at word boundaries and timed by aksharas (Telugu) or by its words (English), a
minimum duration where the next cue allows, a preview's clip, overlapping cues merged, and the tx3g sample layout. All
test text is original."""

from __future__ import annotations

import struct

import pytest

from maata_engine.subtitles import (DASH, SUB_MIN, SUB_ROW, Cue, english_cues, merge, pieces, rows, samples, shape,
                                    telugu_cues, tx3g, width_en, width_te)
from maata_engine.text.akshara import count_units


def test_a_short_line_is_one_row_and_a_longer_one_breaks_at_the_space_nearest_the_middle():
    assert rows("The garden gate was open.".split(), width_en) == ["The garden gate was open."]
    text = "Every morning the baker opens the shop and sweeps the street in front of it."
    got = rows(text.split(), width_en)
    assert got is not None and len(got) == 2 and " ".join(got) == text
    assert all(len(r) <= SUB_ROW for r in got)
    # the break nearest the middle of all the breaks where both rows fit
    best = min((abs(len(" ".join(text.split()[:k])) - len(text) / 2), k) for k in range(1, len(text.split())))
    assert got[0] == " ".join(text.split()[:best[1]])
    assert rows(("word " * 30).split(), width_en) is None  # 149 characters: no break leaves two rows of 42


def test_long_english_is_split_into_cues_timed_by_their_own_words():
    text = ("Every morning the baker opens the shop early and sweeps the street in front of it, then he sets out the "
            "bread on wooden shelves and waits for the first neighbours to come in from the cold.")
    words = [(w, 10.0 + 0.4 * k, 10.0 + 0.4 * k + 0.3) for k, w in enumerate(text.split())]
    cues = english_cues([("S1", words)])
    assert len(cues) >= 2 and " ".join(c.text.replace("\n", " ") for c in cues) == text
    k = 0
    for c in cues:
        n = len(c.text.split())
        assert (c.start, c.end) == (words[k][1], words[k + n - 1][2])  # its own first and last word
        assert c.text.count("\n") <= 1 and all(len(r) <= SUB_ROW for r in c.text.split("\n"))
        k += n
    first = english_cues([("S1", words[:3])], offset=0.5)[0]
    assert (first.start, first.end, first.text) == (10.5, pytest.approx(11.6), "Every morning the")


def test_telugu_rows_are_counted_in_aksharas_and_pieces_timed_by_their_share():
    # 30 words of 2 aksharas: 89 units with the spaces, too wide for one row of 42, in two cues
    word = "మాట"
    assert width_te(word) == 2 == count_units(word) and len(word) == 3  # three code points, two aksharas
    line = {"speaker": "S2", "start": 4.0, "samples": 24_000 * 12, "telugu": " ".join([word] * 30)}
    cues = telugu_cues([line], 24_000)
    assert [len(c.text.split()) for c in cues] == [28, 2]  # 28 words: 2 rows of 14 (27 units each)
    assert all(count_units(r) + r.count(" ") <= SUB_ROW for c in cues for r in c.text.split("\n"))
    assert cues[0].start == 4.0 and cues[-1].end == pytest.approx(16.0)
    assert cues[0].end == pytest.approx(4.0 + 12.0 * 28 / 30) == cues[1].start  # by their share of the aksharas
    assert len(cues[0].text) > SUB_ROW * 2 // 2  # code points would make the row half as wide on screen
    assert telugu_cues([{**line, "telugu": ""}], 24_000) == []
    assert pieces(["ఒక"], width_te) == [["ఒక"]]


def test_a_short_cue_lasts_the_minimum_where_the_next_allows_and_a_preview_clips():
    cues = [Cue(1.0, 1.2, "S1", "a"), Cue(1.5, 1.6, "S1", "b"), Cue(5.0, 5.1, "S2", "c")]
    got = shape(cues)
    assert [(c.start, c.end) for c in got] == [(1.0, 1.5), (1.5, 1.5 + SUB_MIN), (5.0, 5.0 + SUB_MIN)]
    clipped = shape(cues + [Cue(9.0, 12.0, "S1", "d"), Cue(10.0, 11.0, "S2", "e")], stop=10.0)
    assert [(c.text, c.start, c.end) for c in clipped][-1] == ("d", 9.0, 10.0)  # cut at the end; "e" starts at it


def test_overlapping_cues_are_merged_one_block_per_speaker():
    long = Cue(0.0, 6.0, "S1", "A long line that keeps going\nwith a second row")
    short = Cue(2.0, 3.0, "S2", "Right!")
    got = merge([long, short])
    assert got == [(0.0, 2.0, long.text), (2.0, 3.0, f"{DASH}{long.text}\n{DASH}Right!".replace(
        "\nwith", "\nwith")), (3.0, 6.0, long.text)]
    # each block's first row has the dash; the speaker's second row doesn't
    assert got[1][2].split("\n") == [DASH + "A long line that keeps going", "with a second row", DASH + "Right!"]
    # a gap between cues is left to the muxer (an empty sample); identical touching pieces join
    assert merge([Cue(0, 1, "S1", "x"), Cue(1, 2, "S1", "x"), Cue(3, 4, "S2", "y")]) == [(0, 2, "x"), (3, 4, "y")]


def test_tx3g_samples_are_a_big_endian_length_then_utf8():
    assert tx3g("తెలుగు") == struct.pack(">H", 18) + "తెలుగు".encode()
    got = samples([Cue(1.0, 2.5, "S1", "Hello"), Cue(2.0, 3.0, "S2", "Hi")])
    assert got == [(1000, 1000, tx3g("Hello")), (2000, 500, tx3g(f"{DASH}Hello\n{DASH}Hi")), (2500, 500, tx3g("Hi"))]
    assert samples([Cue(1.0, 2.5, "S1", "Hello")], stop=1.2) == [(1000, 200, tx3g("Hello"))]


def test_nothing_is_before_the_files_start():
    # the video starts after the audio: shifted onto the file's clock, early cues fall before its 0
    cues = [Cue(-0.4, -0.1, "S1", "gone"), Cue(-0.3, 0.2, "S2", "late"), Cue(2.5, 3.5, "S1", "kept")]
    got = shape(cues, floor=0.0)
    assert [(c.text, c.start, c.end) for c in got] == [("late", 0.0, SUB_MIN), ("kept", 2.5, 3.5)]
    assert samples(cues) == [(0, round(SUB_MIN * 1000), tx3g("late")), (2500, 1000, tx3g("kept"))]
