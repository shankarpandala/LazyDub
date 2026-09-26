"""MLXWhisper's guards (ARCHITECTURE §3.4) against a stand-in for mlx_whisper: no model, no GPU. The punctuation pass runs
per 30 s window with the style prompt and lends only marks and casing, never its guesses at the window cuts; diarized
speech with no words is decoded again on its own, and what that finds is vetted before it joins; the hallucination
threshold is set on every call. All text here is original test text."""

from __future__ import annotations

import numpy as np

from maata_engine.backends.apple import HALLUCINATION_SILENCE, STYLE_PROMPT, MLXWhisper
from maata_engine.backends.base import SR_ANALYSIS

SR = SR_ANALYSIS


def words(text: str, t: float, step: float = 0.5) -> list[tuple[str, float, float]]:
    return [(x, round(t + i * step, 3), round(t + i * step + 0.4, 3)) for i, x in enumerate(text.split())]


PLAIN = (words("the ferry was late again", 1.0) + words("so we waited by the pier", 18.0)
         + words("then the captain waved us on board", 22.5) + words("and we kept talking", 28.1)
         + words("about the trip", 30.1) + words("we sat near the window", 35.0) + words("it was calm", 61.5))
# The prompted pass guesses a sentence edge at each cut between its 30 s windows ("talking." | "About"): not lent.
STYLED = (words("The ferry was late again.", 1.0) + words("So we waited by the pier.", 18.0)
          + words("Then the captain waved us on board,", 22.5) + words("and we kept talking.", 28.1)
          + words("About the trip.", 30.1) + words("We sat near the window.", 35.0)
          + words("Please like and subscribe.", 45.0) + words("It was calm.", 61.5))
HIDDEN = words("with our bags", 21.0)            # what a second look between the two sentences finds


class FakeWhisper:
    """mlx_whisper.transcribe as MLXWhisper calls it. The audio is a ramp, so its first sample says where a window
    starts. Unprompted: `plain`; prompted: `styled`; a clip: `hidden` inside it, at probability `sure` in a segment of
    no-speech probability `quiet`. Every call's options are kept."""

    def __init__(self, language: str = "en", plain=PLAIN, styled=STYLED, hidden=HIDDEN, sure=0.9, quiet=0.1) -> None:
        self.language, self.calls = language, []
        self.plain, self.styled, self.hidden, self.sure, self.quiet = plain, styled, hidden, sure, quiet

    def transcribe(self, audio, **kw):
        offset, seconds = round(float(audio[0]), 2), len(audio) / SR
        self.calls.append({**kw, "offset": offset, "seconds": round(seconds, 2)})
        prob, quiet = 0.9, 0.0
        if kw.get("clip_timestamps"):
            a, b = kw["clip_timestamps"]
            got, prob, quiet = [w for w in self.hidden if a <= w[1] and w[2] <= b], self.sure, self.quiet
        elif kw.get("initial_prompt"):
            got = [(x, s - offset, e - offset) for x, s, e in self.styled if offset <= s and e <= offset + seconds]
        else:
            got = self.plain
        seg = {"words": [{"word": " " + x, "start": s, "end": e, "probability": prob} for x, s, e in got],
               "no_speech_prob": quiet}
        return {"segments": [seg], "language": kw["language"] or self.language}


def whisper(language: str = "en", **fake) -> MLXWhisper:
    m = object.__new__(MLXWhisper)  # no model load
    m._mw, m.model_dir = FakeWhisper(language, **fake), "/models/whisper"
    return m


AUDIO = (np.arange(65 * SR, dtype=np.float64) / SR).astype(np.float32)


def test_punctuation_pass_per_window_lends_only_marks_and_casing():
    m = whisper()
    tr = m.transcribe(AUDIO)
    first, *styled = m._mw.calls
    assert first.get("initial_prompt") is None and first["condition_on_previous_text"] is False
    assert all(c["hallucination_silence_threshold"] == HALLUCINATION_SILENCE and c["word_timestamps"] for c in m._mw.calls)
    assert [(c["offset"], c["seconds"], c["initial_prompt"]) for c in styled] == [
        (0.0, 30.0, STYLE_PROMPT), (30.0, 30.0, STYLE_PROMPT), (60.0, 5.0, STYLE_PROMPT)]
    assert all(c["language"] == "en" for c in styled)                   # detected once, then passed on
    assert " ".join(w.text for w in tr.words) == ("The ferry was late again. So we waited by the pier. Then the captain "
                                                  "waved us on board, and we kept talking about the trip. We sat near "
                                                  "the window. It was calm.")
    assert [(w.start, w.end) for w in tr.words] == [(s, e) for _, s, e in PLAIN]  # "Please like and subscribe." isn't
    assert tr.punctuated and tr.gaps == [] and tr.recovered == 0 and tr.language == "en"
    assert tr.edge_guesses == 2                                          # "talking." and "About" at the 30 s cut


def test_a_real_sentence_end_near_a_window_cut_is_still_lent():
    """Only the guess itself goes: the last word's mark before the cut and the first word's capital after it."""
    plain = [("we", 27.0, 27.3), ("packed", 27.4, 27.8), ("the", 27.9, 28.1), ("car", 28.2, 29.1),
             ("then", 29.2, 29.5), ("we", 29.5, 29.9), ("drove", 30.1, 30.5), ("off", 30.6, 31.0), ("slowly", 31.1, 31.6)]
    styled = [("We", 27.0, 27.3), ("packed", 27.4, 27.8), ("the", 27.9, 28.1), ("car.", 28.2, 29.1),
              ("Then", 29.2, 29.5), ("we.", 29.5, 29.9), ("Drove", 30.1, 30.5), ("off", 30.6, 31.0),
              ("slowly.", 31.1, 31.6)]
    tr = whisper(plain=plain, styled=styled).transcribe(AUDIO)
    assert " ".join(w.text for w in tr.words) == "We packed the car. Then we drove off slowly."
    assert tr.edge_guesses == 2


def test_diarized_speech_with_no_words_is_decoded_again():
    m = whisper()
    tr = m.transcribe(AUDIO, speech=[(0.5, 3.5), (17.8, 25.9)])        # 20.9-22.5 s has speech but no words
    clips = [[round(x, 2) for x in c["clip_timestamps"]] for c in m._mw.calls if c.get("clip_timestamps")]
    assert tr.gaps == [(20.9, 22.5)] and clips == [[20.6, 22.8]] and tr.recovered == 3
    assert [w.text for w in tr.words][10:16] == ["pier.", "with", "our", "bags", "Then", "the"]
    assert all(a.start <= b.start for a, b in zip(tr.words, tr.words[1:]))


def test_redecoded_words_are_vetted_before_they_join():
    """A re-decode Whisper was unsure of, thought silent, or looped on adds nothing; a neighbour heard again in the
    clip's padding isn't added twice."""
    speech = [(0.5, 3.5), (17.8, 25.9)]                                  # the gap: 20.9-22.5 s, between "pier" and "then"
    for fake in ({"sure": 0.2}, {"quiet": 0.8}, {"hidden": words("go on go on go on", 20.95, 0.25)}):
        tr = whisper(**fake).transcribe(AUDIO, speech=speech)
        n = len(fake.get("hidden", HIDDEN))
        assert tr.gaps == [(20.9, 22.5)] and tr.recovered == 0 and tr.rejected == n, fake
        assert [w.text for w in tr.words][10:12] == ["pier.", "Then"]
    echoes = [("pier", 20.8, 21.1)] + words("with our bags", 21.1, 0.35) + [("then", 22.2, 22.6)]
    tr = whisper(hidden=echoes).transcribe(AUDIO, speech=speech)
    assert tr.recovered == 3 and tr.rejected == 2
    assert [w.text for w in tr.words][10:15] == ["pier.", "with", "our", "bags", "Then"]


def test_no_punctuation_pass_for_other_languages():
    m = whisper("hi")
    tr = m.transcribe(AUDIO)
    assert len(m._mw.calls) == 1 and not tr.punctuated and tr.words[0].text == "the"
