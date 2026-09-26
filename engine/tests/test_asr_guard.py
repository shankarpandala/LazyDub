"""The ASR guards (ARCHITECTURE §3.4, report 10 R4): uncovered diarized speech, low-confidence clusters, phrases repeated
back to back, the interim music flag, and the vetting of a re-decode's words. All text here is original test text."""

from __future__ import annotations

from maata_engine.text.asr_guard import low_confidence, music_like, repeats, uncovered, vet_redecode
from maata_engine.types import SourceUnit, SpeakerTurn, TimedWord


def timed(text: str, t: float = 0.0, word: float = 0.3, gap: float = 0.1, conf: float = 0.9) -> list[TimedWord]:
    out = []
    for x in text.split():
        out.append(TimedWord(x, round(t, 3), round(t + word, 3), conf))
        t += word + gap
    return out


def test_uncovered_finds_diarized_speech_with_no_words():
    words = timed("we left early", t=0.0) + timed("and got there by noon", t=3.0)  # 0-1.1 s, 3.0-4.9 s
    assert uncovered(words, [(0.0, 6.0)]) == [(1.1, 3.0), (4.9, 6.0)]
    assert uncovered(words, [(0.0, 5.5)]) == [(1.1, 3.0)]                  # 0.6 s at the end: under 0.8 s
    assert uncovered(words, [(0.0, 2.0), (1.5, 4.0)]) == [(1.1, 3.0)]      # overlapping turns count once
    assert uncovered(words, [(8.0, 9.5)]) == [(8.0, 9.5)] and uncovered([], [(0.0, 0.5)]) == []
    assert uncovered(words, [(0.0, 6.0)], min_gap=2.0) == []


def test_low_confidence_clusters_need_three_in_a_row():
    words = timed("the shop on the corner sells fresh bread")
    for k in (1, 2, 3, 6):
        words[k] = TimedWord(words[k].text, words[k].start, words[k].end, 0.2)
    assert low_confidence(words) == [(words[1].start, words[3].end)]
    assert low_confidence(words, run=1) == [(words[1].start, words[3].end), (words[6].start, words[6].end)]


def test_repeats_are_phrases_said_again_back_to_back():
    assert repeats(timed("no no no no no no")) == [(0.0, 2.3)]
    assert repeats(timed("No, no. No no, no no!")) == [(0.0, 2.3)]      # marks and case don't matter
    loop = timed("we will see we will see we will see then")
    assert repeats(loop) == [(0.0, loop[8].end)]
    assert repeats(timed("tick tock tick tock tick tock")) == [(0.0, 2.3)]
    assert repeats(timed("very very good")) == [] and repeats(timed("I know I know")) == []  # ordinary emphasis
    assert repeats(timed("the cat and the dog and the cat")) == []       # repeated, but not back to back
    assert repeats(timed("one two three"), min_words=2) == [] and repeats(timed("go go"), min_words=2) == [(0.0, 0.7)]


def unit(text: str, conf: float, t: float = 0.0, word: float = 0.5, speaker: str = "S1") -> SourceUnit:
    ws = timed(text, t, word, 0.1, conf)
    return SourceUnit(0, speaker, ws[0].start, ws[-1].end, text, ws)


def test_music_flag_needs_all_four_signs():
    sung = "la la la la la la the kettle is on the stove again la la la la la la"   # 19 words, 11.3 s
    one_voice = [SpeakerTurn("S1", 0.0, 12.0)]
    assert music_like(unit(sung, 0.3), one_voice)
    assert music_like(unit(sung, 0.3), [])                               # no diarized turn at all: no change either
    assert not music_like(unit(sung, 0.9), one_voice)                    # Whisper is sure of the words
    assert not music_like(unit(sung, 0.3), one_voice + [SpeakerTurn("S2", 5.0, 6.0)])  # someone else speaks inside
    assert not music_like(unit("la la la la la la now", 0.3), one_voice)  # 4.1 s: too short
    assert not music_like(unit("the kettle is on the stove and the bread is almost ready for the table now", 0.3),
                          one_voice)                                     # no repeated phrase


def test_a_redecode_is_taken_whole_or_not_at_all_and_never_echoes_its_neighbours():
    first = timed("we crossed the old", t=0.0) + timed("before dark", t=3.0)  # the gap: 1.5-3.0 s
    gap, pad = (1.5, 3.0), 0.3
    found = timed("stone bridge", t=1.8)
    assert vet_redecode(found, [0.1, 0.1], first, gap, pad) == (found, [])
    assert vet_redecode(timed("stone bridge", t=1.8, conf=0.3), [0.1, 0.1], first, gap, pad)[0] == []  # unsure
    assert vet_redecode(found, [0.1, 0.7], first, gap, pad) == ([], found)                    # likelier silence
    looped = timed("so so so so so so", t=1.55, word=0.15, gap=0.05)
    assert vet_redecode(looped, [0.1] * 6, first, gap, pad) == ([], looped)                    # a decoding loop
    # "old" and "before", heard again in the padding at the gap's edges, are not new words.
    echoes = timed("old stone bridge before", t=1.5, word=0.3, gap=0.05)
    assert vet_redecode(echoes, [0.1] * 4, first, gap, pad) == (echoes[1:3], [echoes[0], echoes[3]])
    # A repeat of a neighbour further away than the padding is a word said again.
    far = timed("we crossed the old", t=0.0, gap=0.0)[:3] + timed("old", t=0.9)          # "old" ends 1.2 s
    assert vet_redecode(timed("old stone", t=1.6), [0.1, 0.1], far, (1.6, 3.0), pad)[1] == []
    assert vet_redecode([], [], first, gap, pad) == ([], [])
