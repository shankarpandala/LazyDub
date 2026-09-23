from maata_engine.segment import segment
from maata_engine.types import SpeakerTurn, TimedWord


def w(text, s, e):
    return TimedWord(text, s, e)


def test_splits_on_sentence_speaker_and_pause_and_drops_backchannel():
    words = [
        w("Hello", 0.0, 0.4), w("everyone,", 0.45, 0.9), w("welcome", 0.95, 1.4), w("back.", 1.45, 1.9),
        w("Today", 2.0, 2.4), w("we", 2.45, 2.6), w("learn", 2.65, 3.0), w("Swift.", 3.05, 3.6),
        w("Yeah.", 3.9, 4.2),
        w("Great", 5.0, 5.4), w("question", 5.45, 6.0), w("from", 6.05, 6.3), w("you.", 6.35, 6.8),
    ]
    turns = [SpeakerTurn("A", 0, 3.7), SpeakerTurn("B", 3.8, 4.3), SpeakerTurn("A", 4.9, 7)]
    units = segment(words, turns)
    assert [u.speaker for u in units] == ["A", "A", "A"]
    assert units[0].text == "Hello everyone, welcome back."
    assert all(u.end - u.start <= 12 for u in units)
