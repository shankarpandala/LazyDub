from hypothesis import given, settings, strategies as st

from maata_engine.segment import SegmenterSettings, _resplit, anchors, breaks, merge_fragments, segment, speaker_at
from maata_engine.timing.isochrony import TimingSettings
from maata_engine.types import SourceUnit, SpeakerTurn, TimedWord

S = SegmenterSettings()


def w(text, s, e):
    return TimedWord(text, s, e)


def run(texts, t=0.0, word=0.3, gap=0.1):
    """Evenly timed words from a list of texts, starting at t."""
    out = []
    for x in texts:
        out.append(w(x, t, t + word))
        t += word + gap
    return out


def test_splits_on_sentence_speaker_and_pause_and_drops_backchannel():
    words = [
        w("Hello", 0.0, 0.4), w("everyone,", 0.45, 0.9), w("welcome", 0.95, 1.4), w("back", 1.45, 1.8), w("again.", 1.85, 2.2),
        w("Today", 2.3, 2.8), w("we", 2.85, 3.1), w("learn", 3.15, 3.6), w("Swift.", 3.65, 4.4),
        w("Yeah.", 4.6, 4.9),
        w("Great", 5.6, 6.0), w("question", 6.05, 6.6), w("from", 6.65, 6.9), w("you.", 6.95, 7.4),
    ]
    turns = [SpeakerTurn("A", 0, 4.45), SpeakerTurn("B", 4.5, 5.0), SpeakerTurn("A", 5.5, 8)]
    units = segment(words, turns)
    assert [u.speaker for u in units] == ["A", "A", "A"]
    assert [u.text for u in units] == ["Hello everyone, welcome back again.", "Today we learn Swift.", "Great question from you."]


def test_short_sentence_joins_the_next_instead_of_standing_alone():
    words = run(["Right.", "So", "the", "model", "runs", "locally."])
    assert [u.text for u in segment(words, [])] == ["Right. So the model runs locally."]


def test_short_pause_mid_sentence_does_not_split():
    words = run(["We", "tried"]) + run(["three", "different", "models", "today."], t=1.3)  # 0.6 s pause, no comma
    assert len(segment(words, [])) == 1


def test_a_long_pause_splits_after_complete_speech_but_is_a_break_inside_a_sentence():
    a = run(["When", "we", "started", "our", "little", "company,"])  # 0–2.3 s
    b = run(["nobody", "believed", "us", "at", "all", "really"], t=3.1)  # 0.8 s after a comma: same sentence
    c = run(["and", "then", "things", "changed."], t=7.0)  # 1.6 s pause, then the sentence goes on: a break
    (u,) = segment(a + b + c, [])
    assert u.words == a + b + c and u.breaks == [12] and u.anchors == [6] and not u.cut_off
    # After complete speech the same pause splits: a capital opens the next sentence, with or without a mark before it.
    c[0] = w("And", c[0].start, c[0].end)
    units = segment(a + b + c, [])
    assert [x.words for x in units] == [a + b, c] and units[0].breaks == []
    b[-1] = w("really.", b[-1].start, b[-1].end)
    assert [x.words for x in segment(a + b + c, [])] == [a + b, c]
    c[0] = w("and", c[0].start, c[0].end)                                # "." before a lowercase word runs on
    assert [x.words for x in segment(a + b + c, [])] == [a + b + c]


def test_sentence_that_fits_is_not_split_at_a_clause_pause():
    a = run(["I", "think", "the", "real", "reason,", "honestly,"])  # 0–2.3 s
    b = run(["we", "never", "shipped", "that", "thing", "was", "fear."], t=3.1)  # 0.8 s after the comma
    assert [u.text for u in segment(a + b, [])] == ["I think the real reason, honestly, we never shipped that thing was fear."]


def test_over_max_len_splits_at_last_clause_mark():
    head = run(["word"] * 20 + ["okay,"] + ["word"] * 6)  # the comma ends at 8.3 s
    tail = run(["word"] * 30 + ["done."], t=head[-1].end + 0.1)
    units = segment(head + tail, [])
    assert units[0].text.endswith("okay,") and units[0].end == head[20].end
    assert all(u.end - u.start <= S.max_len for u in units)


def test_over_max_len_without_marks_splits_at_longest_pause():
    a = run(["word"] * 30)  # 11.9 s
    b = run(["word"] * 30, t=a[-1].end + 1.2)  # 1.2 s: longest pause, below long_pause
    units = segment(a + b, [])
    assert [u.words for u in units] == [a, b]


def test_gap_word_keeps_previous_speaker():
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 6.0, 9.0)]
    words = run(["I", "think", "that", "this", "is"]) + run(["really", "good."], t=2.6) + run(["Sure.", "I", "agree", "completely."], t=6.1)
    units = segment(words, turns)
    assert [(u.speaker, u.text) for u in units] == [("A", "I think that this is really good."), ("B", "Sure. I agree completely.")]


def test_speaker_at():
    turns = [SpeakerTurn("A", 0, 5), SpeakerTurn("B", 4, 12), SpeakerTurn("A", 20, 22)]
    assert speaker_at(turns, 4.5) == "B"                  # overlap: the longest turn wins
    assert speaker_at(turns, 12.4, prev="A") == "B"       # within 0.5 s of a turn
    assert speaker_at(turns, 19.6, prev="B") == "A"
    assert speaker_at(turns, 15.0, prev="A") == "A"       # in a gap: the previous word's speaker
    assert speaker_at(turns, 15.0) == "B"                 # a window's first word: the nearest turn at any distance
    assert speaker_at([], 1.0, prev="B") == "B"
    assert speaker_at([], 1.0) == "S1"


def test_window_starting_in_a_diarization_hole_takes_the_next_turn():
    words = run("and that is why we moved everything on device.".split())
    turns = [SpeakerTurn("S2", 0.9, 4.0)]  # nothing from 0 to 0.9 s
    assert [(u.speaker, u.text) for u in segment(words, turns)] == [("S2", "and that is why we moved everything on device.")]


def test_short_diarization_flip_inside_a_sentence_is_smoothed():
    words = run("so the thing we really wanted was a model that runs locally.".split())
    turns = [SpeakerTurn("S1", 0, 1.95), SpeakerTurn("S2", 2.0, 2.35), SpeakerTurn("S1", 2.4, 6)]  # "wanted" flips to S2
    assert [(u.speaker, u.text) for u in segment(words, turns)] == [("S1", "so the thing we really wanted was a model that runs locally.")]


def test_short_run_between_one_speakers_words_is_smoothed_even_across_sentence_ends():
    """A run of <= 3 words between one speaker's words is a flip, even when it is a sentence of its own."""
    words = run(["So", "the", "model"]) + [w("No.", 1.2, 1.5)] + run(["runs", "locally."], t=1.6)
    turns = [SpeakerTurn("A", 0, 1.15), SpeakerTurn("B", 1.15, 1.55), SpeakerTurn("A", 1.55, 3)]
    assert [(u.speaker, u.text) for u in segment(words, turns)] == [("A", "So the model No. runs locally.")]
    a = run(["We", "shipped", "it", "in", "spring."])                    # 0–1.9 s
    b = run(["It", "was", "late."], t=2.0, word=0.5)                     # 3 words over 1.7 s: still a flip
    c = run(["Then", "we", "fixed", "the", "installer."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, b[-1].end + 0.05), SpeakerTurn("A", b[-1].end + 0.05, 9)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "We shipped it in spring. It was late. Then we fixed the installer.")]


def test_run_under_a_second_is_smoothed_whatever_its_word_count():
    a = run(["The", "first", "build"])                                   # 0–1.1 s
    b = run(["was", "a", "lot", "slower"], t=1.2, word=0.15, gap=0.05)   # 4 quick words in 0.75 s
    c = run(["than", "we", "hoped."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, 1.15), SpeakerTurn("B", 1.15, b[-1].end + 0.05), SpeakerTurn("A", b[-1].end + 0.05, 9)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [("A", "The first build was a lot slower than we hoped.")]


def test_longer_run_of_the_other_speaker_is_its_own_turn():
    a = run(["The", "first", "build", "was", "slow."])                   # 0–1.9 s
    b = run(["Did", "you", "profile", "it?"], t=2.0)                     # 4 words over 1.5 s
    c = run(["Not", "at", "first,", "no."], t=b[-1].end + 0.2)
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, b[-1].end + 0.1), SpeakerTurn("A", b[-1].end + 0.1, 9)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "The first build was slow."), ("B", "Did you profile it?"), ("A", "Not at first, no.")]


def test_answer_to_a_question_is_not_smoothed():
    a = run(["Does", "it", "need", "a", "network?"])                     # 0–1.9 s
    b = run(["No,", "never."], t=2.0)
    c = run(["Great,", "that", "keeps", "it", "private."], t=b[-1].end + 0.2)
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, b[-1].end + 0.1), SpeakerTurn("A", b[-1].end + 0.1, 9)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "Does it need a network?"), ("B", "No, never."), ("A", "Great, that keeps it private.")]


def test_short_turn_set_off_by_long_pauses_is_not_smoothed():
    a = run(["That", "was", "the", "whole", "demo."])                    # 0–1.9 s
    b = run(["Very", "nice."], t=3.6)                                    # 1.7 s after, 1.6 s before the next
    c = run(["Next", "we", "will", "look", "at", "pricing."], t=b[-1].end + 1.6)
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 3.5, 4.4), SpeakerTurn("A", 5.9, 12)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "That was the whole demo."), ("B", "Very nice."), ("A", "Next we will look at pricing.")]


def test_real_turn_split_at_a_sentence_end_rejoins():
    """A short real turn that a sentence end splits (its first sentence reaches min_len) is one unit again: its pieces
    are fragments of the same speaker, not turns of their own."""
    a = run(["That", "was", "the", "whole", "demo."])                    # 0–1.9 s
    b = [w("Great", 3.6, 4.4), w("demo.", 4.8, 5.7), w("Wow.", 6.9, 7.1)]  # 2.1 s to the ".", then 1.2 s before "Wow."
    c = run(["Next", "we", "will", "look", "at", "pricing."], t=8.7)
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 3.5, 7.2), SpeakerTurn("A", 8.6, 12)]
    units = segment(a + b + c, turns)
    assert [(u.speaker, u.text) for u in units] == [
        ("A", "That was the whole demo."), ("B", "Great demo. Wow."), ("A", "Next we will look at pricing.")]
    assert units[1].breaks == [2]


def test_quick_run_is_kept_where_the_turns_really_overlap():
    a = run(["So", "the", "plan", "is"])                                 # 0–1.5 s
    b = run(["wait", "what", "about", "memory"], t=1.6, word=0.15, gap=0.05)  # 0.75 s of crosstalk
    c = run(["to", "ship", "it", "on", "Friday."], t=b[-1].end + 0.1)
    excl = [SpeakerTurn("A", 0, 1.55), SpeakerTurn("B", 1.55, b[-1].end + 0.05), SpeakerTurn("A", b[-1].end + 0.05, 9)]
    both = [SpeakerTurn("A", 0, 9), SpeakerTurn("B", 1.55, b[-1].end + 0.05)]  # A never stopped talking
    assert [u.speaker for u in segment(a + b + c, excl, overlaps=both)] == ["A", "B", "A"]
    assert [u.speaker for u in segment(a + b + c, excl)] == ["A"]        # no overlap evidence: a flip


def test_short_crosstalk_stays_with_its_speaker():
    """A run of <= 3 words is short enough to be a fragment, but crosstalk is a turn of its own: it is not
    spliced into the flanking speaker's sentence, and it takes none of their words."""
    a = run(["So", "the", "plan", "is"])                                 # 0–1.5 s
    b = run(["hold", "on", "there"], t=1.6, word=0.2, gap=0.05)          # 0.7 s of crosstalk
    c = run(["to", "ship", "it", "on", "Friday."], t=b[-1].end + 0.1)
    excl = [SpeakerTurn("A", 0, 1.55), SpeakerTurn("B", 1.55, b[-1].end + 0.05), SpeakerTurn("A", b[-1].end + 0.05, 9)]
    both = [SpeakerTurn("A", 0, 9), SpeakerTurn("B", 1.55, b[-1].end + 0.05)]  # A never stopped talking
    assert [(u.speaker, u.text) for u in segment(a + b + c, excl, overlaps=both)] == [
        ("A", "So the plan is"), ("B", "hold on there"), ("A", "to ship it on Friday.")]
    a = run(["So", "the"])                                               # A's side of the window is a fragment too
    b = run(["hold", "on", "there"], t=0.8, word=0.2, gap=0.05)
    c = run(["plan", "is", "to", "ship", "it", "on", "Friday."], t=b[-1].end + 0.1)
    excl = [SpeakerTurn("A", 0, 0.75), SpeakerTurn("B", 0.75, b[-1].end + 0.05), SpeakerTurn("A", b[-1].end + 0.05, 9)]
    both = [SpeakerTurn("A", 0, 9), SpeakerTurn("B", 0.75, b[-1].end + 0.05)]
    assert [u.speaker for u in segment(a + b + c, excl, overlaps=both)] == ["A", "B", "A"]


def test_flip_after_a_rhetorical_or_tag_question_is_smoothed():
    """Words right after a question are an answer only when they are a complete utterance. Ones that read
    straight on into the flanking speaker's sentence are a flip."""
    a = run(["What", "does", "local", "mean", "here?"])                  # 0–1.9 s
    b = run(["It", "means"], t=a[-1].end + 0.15)                         # flipped to B
    c = run(["the", "model", "never", "leaves", "your", "laptop."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [u.speaker for u in segment(a + b + c, turns)] == ["A"]
    a = run(["It", "runs", "offline,", "right?"])
    b = run(["So"], t=a[-1].end + 0.15)
    c = run(["we", "shipped", "it", "on", "Friday."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [("A", "It runs offline, right? So we shipped it on Friday.")]
    # A backchannel word reading on after the question goes the same way (kept, not dropped).
    b = run(["yeah", "so"], t=a[-1].end + 0.15)
    c = run(["we", "shipped", "it", "on", "Friday."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "It runs offline, right? yeah so we shipped it on Friday.")]
    # With a turn-taking gap before a new sentence, the same short run is an answer.
    b = run(["It", "does"], t=a[-1].end + 0.4)
    c = run(["We", "shipped", "it", "on", "Friday."], t=b[-1].end + 0.4)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "It runs offline, right?"), ("B", "It does"), ("A", "We shipped it on Friday.")]


def test_short_question_answered_by_the_flanking_speaker_is_kept():
    a = run(["I", "moved", "the", "whole", "team", "to", "Pune."])
    b = run(["Really?"], t=a[-1].end + 0.4)
    c = run(["Yes,", "back", "in", "March."], t=b[-1].end + 0.4)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "I moved the whole team to Pune."), ("B", "Really?"), ("A", "Yes, back in March.")]
    # Between two other speakers too, and the answer after it is not handed to the asker.
    turns[2] = SpeakerTurn("C", c[0].start - 0.05, 20)
    c = run(["Yes."], t=b[-1].end + 0.4)
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "I moved the whole team to Pune."), ("B", "Really?"), ("C", "Yes.")]
    # No turn-taking gap before the flanking speaker goes on: a flip of A's own rhetorical question.
    b = run(["Really?"], t=a[-1].end + 0.4)
    c = run(["Yes,", "back", "in", "March."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", c[0].start - 0.05, 20)]
    assert [u.speaker for u in segment(a + b + c, turns)] == ["A"]


def test_backchannel_word_inside_a_sentence_is_a_flip_not_dropped():
    for text, k in (("At the corner you turn right and park.", 5), ("Then he said yes to the offer.", 3)):
        words = run(text.split())
        turns = [SpeakerTurn("A", 0, words[k].start - 0.02), SpeakerTurn("B", words[k].start - 0.02, words[k].end + 0.02),
                 SpeakerTurn("A", words[k].end + 0.02, 9)]
        assert [(u.speaker, u.text) for u in segment(words, turns)] == [("A", text)]
    # A filler is still dropped, and so is a word set off by a pause or where both talk at once.
    words = run("and the real reason is".split()) + run(["mm-hmm"], t=2.0, word=0.3) + run("that it is private.".split(), t=2.4)
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, 2.35), SpeakerTurn("A", 2.35, 9)]
    assert [u.text for u in segment(words, turns)] == ["and the real reason is that it is private."]
    words[5] = w("yeah", 2.0, 2.3)
    both = [SpeakerTurn("A", 0, 9), SpeakerTurn("B", 1.95, 2.35)]
    assert [u.text for u in segment(words, turns, overlaps=both)] == ["and the real reason is that it is private."]
    assert [u.text for u in segment(words, turns)] == ["and the real reason is yeah that it is private."]


def test_turn_change_mid_sentence_moves_to_the_sentence_end():
    """The diarizer starts B's turn a word early: A's last word goes back to A."""
    a = run(["We", "run", "the", "whole", "model"])
    b = run(["locally."], t=a[-1].end + 0.1) + run(["Did", "you", "profile", "it", "first", "though?"], t=a[-1].end + 0.8)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", a[-1].end + 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b, turns)] == [
        ("A", "We run the whole model locally."), ("B", "Did you profile it first though?")]
    # …or a word late: B's first words, after A's sentence end, go to B.
    a = run(["We", "run", "the", "whole", "model", "locally.", "Did", "you"])
    b = run(["profile", "it", "first", "though?"], t=a[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", a[-1].end + 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b, turns)] == [
        ("A", "We run the whole model locally."), ("B", "Did you profile it first though?")]
    # A capitalised word, or a turn-taking gap, is a real turn change: nothing moves.
    a = run(["So", "the", "thing", "is"])
    b = run(["Wait.", "Let", "me", "ask", "you", "something."], t=a[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", a[-1].end + 0.05, 20)]
    assert [u.text for u in segment(a + b, turns)] == ["So the thing is", "Wait. Let me ask you something."]
    # Nor does a word next to another speaker's filler, which is still dropped.
    a = run(["We", "shipped", "it", "in", "May.", "And", "then"])
    b = [w("mm-hmm", a[-1].end + 0.1, a[-1].end + 0.4)]
    c = run(["the", "installer", "broke", "again."], t=b[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", a[-1].end + 0.05, b[-1].end + 0.05),
             SpeakerTurn("A", b[-1].end + 0.05, 20)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "We shipped it in May. And then the installer broke again.")]


def test_period_ends_a_sentence_only_before_a_capital():
    for text in ("Every single morning we start at 9 a.m. with a short call to plan the day for the whole team.",
                 "Most of our users are in the U.S. and they pay in dollars every single month."):
        assert [u.text for u in segment(run(text.split()), [])] == [text]
    words = run("Every morning we start at 9 a.m. Then the whole team joins a short call.".split())
    assert [u.text for u in segment(words, [])] == ["Every morning we start at 9 a.m.", "Then the whole team joins a short call."]
    words = run(["It", "was", "all", "going", "really", "fine…"]) + run(["and", "then", "it", "broke."], t=3.2)
    assert [u.text for u in segment(words, [])] == ["It was all going really fine… and then it broke."]
    assert anchors(words) == [6]                                         # still a marked breath: an anchor


def test_alternating_flips_go_to_the_speaker_holding_the_floor():
    words = run("so what we really wanted from the start was a model that runs on the laptop itself.".split())
    # B, A, B flicker over words 5–10 ("from the start was a model") inside A's sentence.
    b1, a1, b2 = (words[5].start, words[6].end), (words[7].start, words[8].end), (words[9].start, words[10].end)
    turns = [SpeakerTurn("A", 0, b1[0] - 0.02), SpeakerTurn("B", *b1), SpeakerTurn("A", *a1), SpeakerTurn("B", *b2),
             SpeakerTurn("A", b2[1] + 0.02, 12)]
    units = segment(words, turns)
    assert [(u.speaker, len(u.words)) for u in units] == [("A", len(words))]


# -- fragments --------------------------------------------------------------------------------

def test_fragment_at_the_window_start_joins_the_sentence_it_begins():
    b = run(["So", "the"])                                               # labelled B, then A goes on
    a = run(["model", "runs", "on", "the", "laptop."], t=0.8)
    turns = [SpeakerTurn("B", 0, 0.75), SpeakerTurn("A", 0.75, 4)]
    assert [(u.speaker, u.text) for u in segment(b + a, turns)] == [("A", "So the model runs on the laptop.")]


def test_fragment_at_the_window_end_joins_the_sentence_it_ends():
    a = run(["We", "run", "the", "whole", "model"])
    b = run(["locally."], t=2.0)
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, 3)]
    assert [(u.speaker, u.text) for u in segment(a + b, turns)] == [("A", "We run the whole model locally.")]


def test_fragment_between_two_other_speakers_goes_where_it_reads_on():
    a = run(["and", "the", "answer", "is"])                              # A ends mid-sentence…
    b = run(["local."], t=1.6)                                           # …B's one word completes it…
    c = run(["Right,", "and", "that", "saves", "money."], t=2.0)         # …then C starts a new sentence
    turns = [SpeakerTurn("A", 0, 1.55), SpeakerTurn("B", 1.55, 1.95), SpeakerTurn("C", 1.95, 5)]
    assert [(u.speaker, u.text) for u in segment(a + b + c, turns)] == [
        ("A", "and the answer is local."), ("C", "Right, and that saves money.")]


def test_fragment_followed_by_a_long_pause_stands_alone():
    a = run(["That", "is", "how", "it", "went."])                        # 0–1.9 s
    b = run(["Very", "interesting."], t=2.0) + run(["So", "let", "us", "begin."], t=4.4)  # 1.6 s pause inside B
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 1.95, 6)]
    assert [(u.speaker, u.text) for u in segment(a + b, turns)] == [
        ("A", "That is how it went."), ("B", "Very interesting."), ("B", "So let us begin.")]


def test_stub_left_by_a_max_len_cut_is_rebalanced_mid_sentence():
    """A run-on of 13 s, then another speaker: the max_len cut would leave a one-word stub."""
    a = run(["word"] * 32 + ["now."])                                    # 0–13.1 s, no mark before the end
    b = run(["Is", "that", "the", "whole", "list?"], t=a[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, a[-1].end + 0.05), SpeakerTurn("B", a[-1].end + 0.05, 20)]
    units = segment(a + b, turns)
    assert _content(units) == a + b and [u.speaker for u in units] == ["A", "A", "B"]
    assert all(len(u.words) > S.fragment_words and u.end - u.start <= S.max_len for u in units)


def test_stub_sentence_moves_to_an_earlier_sentence_end_not_mid_sentence():
    one = run(["The", "installer", "was", "fixed."])                     # 0–1.5 s: too short to split off alone
    two = run(["Word"] + ["word"] * 24 + ["then."], t=1.6)               # 1.6–11.9 s
    stub = run(["That", "is", "all."], t=two[-1].end + 0.1)              # ends at 13.1 s: past max_len from 0
    after = run(["Thanks", "for", "coming", "on", "today."], t=stub[-1].end + 0.1)
    turns = [SpeakerTurn("A", 0, stub[-1].end + 0.05), SpeakerTurn("B", stub[-1].end + 0.05, 30)]
    units = segment(one + two + stub + after, turns)
    assert [u.words for u in units] == [one, two + stub, after]         # re-cut at "fixed.", not inside `two`
    # With no earlier sentence end to move to, the short sentence stays whole on its own.
    one = run(["The", "installer", "was", "fixed"])
    units = segment(one + two + stub + after, turns)
    assert [u.words for u in units] == [one + two, stub, after]


def test_anchors_mark_pauses_at_clause_and_sentence_marks():
    ws = run(["When", "it", "started,"]) + run(["nobody", "cared", "much."], t=2.0) + run(["Then", "they", "did"], t=3.5)
    ws += run(["really."], t=ws[-1].end + 0.9)                           # a pause, but not at a mark
    assert anchors(ws) == [3, 6]                                         # 0.9 s after "started,"; 0.4 s after "much."
    assert anchors(ws, SegmenterSettings(anchor_pause=0.7)) == [3]
    assert S.anchor_pause == 0.3                                         # Amazon's phrase pause (ARCHITECTURE §3.4)
    assert anchors(ws[:1]) == [] and anchors([]) == []


# -- sentence units: breaks, cut-offs, max_len cut points (ARCHITECTURE §3.4) ---------------------------------------

def test_a_long_pause_with_nothing_to_show_a_new_sentence_is_a_break():
    """With no mark before a pause of long_pause or more, a number or "I" (never lowercase) shows no new sentence, and
    nothing does after a word no sentence ends on: the sentence stays one unit, with a break. A capital that opens a
    sentence still splits."""
    a = run(["and", "the", "answer", "to", "the", "whole", "riddle", "was"])  # 0–3.1 s
    for tail in (["42."], ["I", "think,", "roughly", "twelve."], ["seven", "hundred."]):
        b = run(tail, t=a[-1].end + 1.6)
        (u,) = segment(a + b, [])
        assert u.words == a + b and u.breaks == [8], tail
    b = run(["Nobody", "guessed", "it."], t=a[-1].end + 1.6)
    assert [x.words for x in segment(a + b, [])] == [a, b]
    c = run(["we", "handed", "the", "spare", "keys", "to", "the"])
    d = run(["Rao", "family", "next", "door."], t=c[-1].end + 1.6)
    (u,) = segment(c + d, [])
    assert u.words == c + d and u.breaks == [7]


def test_units_carry_their_breaks_and_anchors():
    """A pause of 1 s or more inside a sentence is a break in the unit, marked or not; 0.3 s at a mark is an anchor."""
    a = run(["If", "the", "bus", "is", "late,"])                         # 0–1.9 s
    b = run(["we", "walk", "to", "the", "old", "bridge"], t=2.3)         # 0.4 s after the comma: an anchor only
    c = run(["and", "wait", "there", "for", "the", "others."], t=b[-1].end + 1.0)  # 1.0 s, no mark: a break only
    (u,) = segment(a + b + c, [])
    assert u.anchors == [5] and u.breaks == [11]
    assert breaks(u.words) == [11] and breaks(u.words, SegmenterSettings(break_pause=1.1)) == []


def test_max_len_never_cuts_inside_a_name_or_a_number():
    """With no mark to cut at, the longest pause is taken, unless it falls inside a name or a spoken number."""
    for pair in (("Blue", "Ridge"), ("twenty", "five"), ("3", "million")):
        head = run(["word"] * 12)                                        # 0–4.7 s
        other = run(["word"] * 8, t=head[-1].end + 0.5)                  # the next longest pause: 0.5 s
        tied = run([pair[0]], t=other[-1].end + 0.1) + run([pair[1]], t=other[-1].end + 1.3)  # 0.9 s inside the pair
        tail = run(["word"] * 12, t=tied[-1].end + 0.1)
        units = segment(head + other + tied + tail, [])
        assert [x.words[0] for x in units][:2] == [head[0], other[0]], pair
        assert all(x.end - x.start <= S.max_len for x in units)
    # Between a number and a plain word the pause is fair game.
    head = run(["word"] * 12)
    other = run(["word"] * 8, t=head[-1].end + 0.5)
    tied = run(["twenty"], t=other[-1].end + 0.1) + run(["word"], t=other[-1].end + 1.3)
    units = segment(head + other + tied + run(["word"] * 12, t=tied[-1].end + 0.1), [])
    assert units[1].words[0] == tied[1]


def test_a_resplit_never_cuts_inside_a_name():
    ws = run(["word"] * 6) + run(["word"] * 5, t=2.9)                   # a 0.6 s pause after the sixth word
    ws += run(["New"], t=4.9) + run(["Delhi"], t=6.1) + run(["word"] * 5, t=6.5)  # 0.9 s inside the name
    assert _resplit(ws, S, sentences_only=False) == 6


def test_a_speaker_cut_off_mid_sentence_is_marked():
    """Trailing off on a word no sentence ends on, or talked over, the unit is left unfinished. A missing mark alone is
    no interruption: a quick reply that opens its own sentence leaves an unpunctuated question complete."""
    def pair(first, second, gap, talk_over=False):
        a = run(first.split())
        b = run(second.split(), t=a[-1].end + gap)
        turns = [SpeakerTurn("A", 0, a[-1].end + gap / 2), SpeakerTurn("B", a[-1].end + gap / 2, 30)]
        # The diarizer's turns with crosstalk kept: B starts inside A's last word.
        both = [SpeakerTurn("A", 0, a[-1].end), SpeakerTurn("B", a[-1].start + 0.1, 30)] if talk_over else None
        return [(u.speaker, u.cut_off) for u in segment(a + b, turns, overlaps=both)]

    assert pair("So what I was hoping we could do", "Sorry, can I jump in here?", 0.1, talk_over=True) == [
        ("A", True), ("B", False)]
    assert pair("So what I was hoping we could do", "Sorry, can I jump in here?", 0.1) == [("A", False), ("B", False)]
    assert pair("We could paint the whole fence, but", "No, let us wait for spring.", 0.8) == [("A", True), ("B", False)]
    assert pair("It rained on the fence all week, so", "Then we wait for spring.", 0.4) == [("A", True), ("B", False)]
    assert pair("We could paint it again I think so", "Then we wait for spring.", 0.4) == [("A", False), ("B", False)]
    for gap in (0.1, 0.15, 0.25, 0.8):
        assert pair("What time does the ferry leave", "It leaves at noon.", gap) == [("A", False), ("B", False)], gap
    assert pair("What time does the ferry leave", "um it leaves at noon.", 0.1) == [("A", False), ("B", False)]
    assert pair("We could paint the whole fence.", "No, let us wait for spring.", 0.1) == [("A", False), ("B", False)]
    # A zero-gap flip mid-sentence, more than a fragment from any sentence end, with a lowercase continuation (§3.3):
    # without a way to tell whether it is the same voice, the first part is left unfinished rather than completed.
    assert pair("the reason we built the whole", "thing on the device was privacy.", 0.05) == [("A", True), ("B", False)]


def test_a_backchannel_is_never_cut_off():
    """An answer left standing alone (max_len keeps it from both neighbours) is complete, however quickly the other
    speaker goes on."""
    a = run(["Word"] + ["word"] * 28 + ["start?"])                       # 0–11.9 s
    b = run(["yeah"], t=a[-1].end + 0.4)
    c = run(["Great"] + ["word"] * 28 + ["done."], t=b[-1].end + 0.1)   # 11.9 s: with "yeah", over max_len
    turns = [SpeakerTurn("B", 0, a[-1].end + 0.2), SpeakerTurn("A", b[0].start - 0.05, b[-1].end + 0.05),
             SpeakerTurn("B", c[0].start - 0.05, 40)]
    units = segment(a + b + c, turns)
    assert [(u.speaker, u.words, u.cut_off) for u in units] == [("B", a, False), ("A", b, False), ("B", c, False)]


def test_a_sentence_split_by_a_dropped_backchannel_after_a_comma_rejoins():
    a = run(["And", "the", "real", "reason,", "honestly,"])              # ends 1.9 s
    b = run(["is", "that", "it", "stays", "private."], t=3.3)            # 1.4 s later, "yeah" from B in between
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 2.2, 2.7), SpeakerTurn("A", 3.2, 9)]
    (u,) = segment(a + [w("yeah", 2.3, 2.6)] + b, turns)
    assert u.words == a + b and u.breaks == [5] and u.anchors == [5]


def _unit(words, speaker="S1", cut_off=False):
    return SourceUnit(0, speaker, words[0].start, words[-1].end, " ".join(x.text for x in words), words,
                      breaks=breaks(words), anchors=anchors(words), cut_off=cut_off)


def test_merge_fragments_keeps_breaks_and_anchors_and_marks_a_long_join():
    a = run(["The", "first", "thing,"]) + run(["and", "the", "part", "that"], t=2.3)   # a break at 3 (1.2 s)
    b = run(["I", "keep", "coming", "back", "to,"], t=a[-1].end + 1.2) + run(["is", "the", "cost."], t=7.3)
    (m,) = merge_fragments([_unit(a), _unit(b, cut_off=True)], max_gap=1.5)
    assert m.words == a + b and m.breaks == [3, 7] and m.anchors == [3, 12] and m.cut_off
    assert (m.breaks, m.anchors) == (breaks(m.words), anchors(m.words))
    # A short join at a clause mark is an anchor, not a break.
    b = run(["I", "keep", "coming", "back", "to."], t=a[-1].end + 0.4)
    a[-1] = w("that,", a[-1].start, a[-1].end)
    (m,) = merge_fragments([_unit(a), _unit(b)])
    assert m.breaks == [3] and m.anchors == [3, 7] and not m.cut_off


def test_fragment_of_one_speaker_joins_their_next_sentence_across_a_pause():
    words = run(["Right."]) + run(["The", "next", "part", "is", "pricing."], t=1.6)  # 1.3 s pause, over merge_gap
    assert [u.text for u in segment(words, [])] == ["Right. The next part is pricing."]


def test_backchannel_answering_a_question_is_kept():
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 2.1, 2.6)]
    words = run(["Is", "it", "running", "locally?"]) + [w("Yes.", 2.2, 2.5)]
    assert [u.text for u in segment(words, turns)] == ["Is it running locally?", "Yes."]


def test_answer_opening_a_window_is_kept():
    """The previous chunk ended on a question (chunks end after "?"), so "Yes." opens this one."""
    words = [w("Yes.", 0.0, 0.3)] + run(["Great,", "so", "how", "does", "it", "work", "then?"], t=0.8)
    turns = [SpeakerTurn("A", 0, 0.4), SpeakerTurn("B", 0.7, 4)]

    def texts(**kw):
        return [u.text for u in segment(words, turns, **kw)]

    assert texts(prev_text="Does it run locally?") == ["Yes.", "Great, so how does it work then?"]
    assert texts() == ["Yes.", "Great, so how does it work then?"]  # context unknown: a yes-like word stays
    assert texts(prev_text="That is how it went.") == ["Great, so how does it work then?"]  # an interjection
    assert texts(prev_text="") == ["Great, so how does it work then?"]


def test_own_short_sentence_before_a_long_one_is_kept():
    words = run(["Okay."]) + run(["Word"] + ["word"] * 59 + ["done."], t=0.4)  # "Okay." then a 24 s sentence
    units = segment(words, [])
    assert _content(units) == words and units[0].words[0].text == "Okay."
    assert all(u.end - u.start <= S.max_len for u in units)


def test_reply_before_a_run_on_is_kept_after_the_other_speaker():
    b = run(["That", "is", "how", "it", "went."])  # 0–1.9 s
    a = run(["Okay."], t=2.3) + run(["Word"] + ["word"] * 59 + ["done."], t=2.7)  # then A's 24 s run-on
    turns = [SpeakerTurn("B", 0, 2.0), SpeakerTurn("A", 2.2, 40)]
    units = segment(b + a, turns)  # "Okay." may stand alone (the run-on is cut), but A goes on: not an interjection
    assert _content(units) == b + a and units[1].speaker == "A" and units[1].words[0].text == "Okay."
    assert all(u.end - u.start <= S.max_len for u in units)


def test_sentence_interrupted_by_a_backchannel_rejoins():
    turns = [SpeakerTurn("A", 0, 1.95), SpeakerTurn("B", 2.0, 2.4), SpeakerTurn("A", 2.45, 6)]
    words = run(["And", "the", "real", "reason", "is"]) + [w("mm-hmm", 2.05, 2.35)] + run(["that", "it", "is", "private."], t=2.5)
    assert [(u.speaker, u.text) for u in segment(words, turns)] == [("A", "And the real reason is that it is private.")]


def test_sentence_rejoins_across_a_backchannel_in_a_longer_pause():
    a1 = run(["And", "the", "real", "reason", "is"])  # ends 1.9 s
    a2 = run(["that", "it", "is", "private."], t=3.1)  # 1.2 s later: under long_pause, so no split without the "yeah"
    turns = [SpeakerTurn("A", 0, 2.0), SpeakerTurn("B", 2.1, 2.7), SpeakerTurn("A", 2.9, 6)]
    expected = [("A", "And the real reason is that it is private.")]
    assert [(u.speaker, u.text) for u in segment(a1 + [w("yeah", 2.2, 2.6)] + a2, turns)] == expected
    assert [(u.speaker, u.text) for u in segment(a1 + a2, [SpeakerTurn("A", 0, 6)])] == expected


def test_max_len_leaves_telugu_room_under_the_synth_cap():
    """Past the TTS cap the synthesizer stops and the line's tail is never spoken. A max_len unit's
    Telugu must be able to run to 2x its slot (1.6x of video after the 1.25x speed-up) under it."""
    from maata_engine.session import MAX_LINE_SECONDS

    assert (S.max_len + TimingSettings().borrow_max) * 2.0 <= MAX_LINE_SECONDS


# Property tests ---------------------------------------------------------------

def _content(units):
    return [x for u in units for x in u.words]


def _check_lossless(words, units):
    """Units are in order, disjoint and cover every word except dropped backchannels."""
    kept = _content(units)
    assert kept == [x for x in words if x in kept]  # order preserved, nothing duplicated
    dropped = [x for x in words if x not in kept]
    assert all(x.text.lower().strip(".,!?") in {"yeah", "mm-hmm"} for x in dropped)
    for u in units:
        assert u.words and u.start == u.words[0].start and u.end == u.words[-1].end


@st.composite
def sentences(draw):
    """Single-speaker speech: sentences of 1–30 words that each fit in max_len.

    Inside a sentence pauses stay under long_pause, also after commas; sentence gaps are 0–3 s.
    """
    words, t = [], 0.0
    for _ in range(draw(st.integers(1, 25))):
        n = draw(st.integers(1, 30))
        start = t
        for j in range(n):
            d = draw(st.floats(0.08, 0.6))
            if t + d - start > S.max_len:
                break
            last = j == n - 1 or t + d + S.long_pause + 0.6 - start > S.max_len
            text = ("Word" if j == 0 else "word") + (draw(st.sampled_from([".", "?", "!", "…"])) if last
                                                      else draw(st.sampled_from(["", "", "", ","])))
            words.append(w(text, t, t + d))
            t += d
            if last:
                break
            t += draw(st.floats(0.0, S.long_pause - 0.01))
        if not words[-1].text.endswith((".", "?", "!", "…")):
            words[-1] = w(words[-1].text.rstrip(",") + ".", words[-1].start, words[-1].end)
        t += draw(st.floats(0.0, 3.0))
    return words


@settings(max_examples=300, deadline=None)
@given(sentences())
def test_never_splits_mid_sentence_when_sentences_fit(words):
    units = segment(words, [SpeakerTurn("A", 0, words[-1].end)])
    _check_lossless(words, units)
    assert _content(units) == words
    for u in units:
        assert u.words[-1].text.endswith((".", "?", "!", "…"))  # every boundary is a sentence end
        assert u.end - u.start <= S.max_len + 1e-9


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.floats(0.05, 1.0), st.floats(0.0, 1.49)), min_size=1, max_size=400))
def test_long_unpunctuated_runs_obey_max_len(spec):
    """Whisper can emit minutes without punctuation; pauses must still keep units ≤ max_len."""
    words, t = [], 0.0
    for d, gap in spec:
        words.append(w("word", t, t + d))
        t += d + gap
    units = segment(words, [])
    assert _content(units) == words
    assert all(u.end - u.start <= S.max_len + 1e-9 for u in units)
    _check_no_stranded_fragments(units)                                  # every cut is a max_len cut: no stubs


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.floats(0.05, 1.0), st.floats(0.0, 3.0)), min_size=1, max_size=300))
def test_pauses_inside_a_sentence_are_breaks_not_splits(spec):
    """Lowercase speech with no marks is never split at a pause, however long: every cut is a max_len cut (the two
    units could not be one), and every pause of break_pause or more inside a unit is one of its breaks."""
    words, t = [], 0.0
    for d, gap in spec:
        words.append(w("word", t, t + d))
        t += d + gap
    units = segment(words, [])
    assert _content(units) == words
    assert all(u.end - u.start <= S.max_len + 1e-9 for u in units)
    assert all(b.end - a.start > S.max_len for a, b in zip(units, units[1:]))
    for u in units:
        gaps = [u.words[k].start - u.words[k - 1].end for k in range(1, len(u.words))]
        assert u.breaks == [k + 1 for k, g in enumerate(gaps) if g >= S.break_pause]
        assert u.anchors == [] and not u.cut_off


turn_lists = st.lists(st.tuples(st.sampled_from("ABC"), st.floats(0.0, 6.0), st.floats(0.2, 8.0)), max_size=12)


def _raw_speakers(words, turns):
    raw, prev = [], None
    for x in words:
        prev = speaker_at(turns, (x.start + x.end) / 2, prev)
        raw.append(prev)
    return raw


def _check_voices(words, turns, units):
    """Only short runs (a flip, or a fragment merged into a neighbour) are voiced by someone else, and the
    ends of a long run: up to 3 words that finish or start the neighbour's sentence at a turn change."""
    raw = _raw_speakers(words, turns)
    by_word = {id(x): u.speaker for u in units for x in u.words}
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and raw[j + 1] == raw[i]:
            j += 1
        run_ = words[i:j + 1]
        if len(run_) > S.flip_words and run_[-1].end - run_[0].start >= S.flip_max:
            moved = {k for k, x in enumerate(run_) if by_word.get(id(x), raw[i]) != raw[i]}
            head = next(k for k in range(len(run_) + 1) if k not in moved)          # moved words at its start…
            tail = next(k for k in range(len(run_) + 1) if len(run_) - 1 - k not in moved)  # …and at its end
            assert head <= S.fragment_words and tail <= S.fragment_words
            assert moved == set(range(head)) | set(range(len(run_) - tail, len(run_)))
        i = j + 1


def _ends(x, nxt=None):
    """x ends a sentence before nxt (None: after the last word). A "." or "…" does only before a word that
    is not lowercase."""
    t = x.text
    if t.endswith(("?", "!")):
        return True
    return t.endswith((".", "…")) and (nxt is None or not nxt.text[:1].islower())


def _opens(x):
    return not x.text[:1].islower()


def _complete(u, v):
    """Unit u is a complete utterance before unit v (None: u is the last): it ends a sentence, or v opens
    a new one after a turn-taking gap."""
    return v is None or _ends(u.words[-1], v.words[0]) or (v.start - u.end >= S.turn_gap and _opens(v.words[0]))


def _qa_turn(units, k, prev_text):
    """Unit k is a question-and-answer turn: a complete answer to another speaker's question (at the window
    start, to `prev_text`; unknown (None): a yes-like word), or a question that starts a sentence and that
    another speaker answers after a turn-taking gap."""
    u = units[k]
    nxt = units[k + 1] if k + 1 < len(units) else None
    if k:
        asked = units[k - 1].speaker != u.speaker and units[k - 1].words[-1].text.endswith("?")
    elif prev_text is None:
        asked = all(x.text.lower().strip(".,!?") in {"yeah", "yes", "yep", "mhm", "mm-hmm", "uh-huh", "right", "okay", "ok"}
                    for x in u.words)
    else:
        asked = prev_text.endswith("?")
    if asked and _complete(u, nxt):
        return True
    if nxt is None or nxt.speaker == u.speaker or not u.words[-1].text.endswith("?") or nxt.start - u.end < S.turn_gap:
        return False
    if not k:
        return _opens(u.words[0])
    before = units[k - 1]
    return _ends(before.words[-1], u.words[0]) or (u.start - before.end >= S.turn_gap and _opens(u.words[0]))


def _resplittable(ws, sentences_only):
    """Whether ws can be cut into two parts of more than a fragment each, both within max_len (with
    `sentences_only`, at a sentence end)."""
    fw = S.fragment_words
    return any(ws[k - 1].end - ws[0].start <= S.max_len + 1e-9 and ws[-1].end - ws[k].start <= S.max_len + 1e-9
               and (_ends(ws[k - 1], ws[k]) or not sentences_only) for k in range(fw + 1, len(ws) - fw))


def _check_no_stranded_fragments(units, prev_text=None):
    """A unit of <= 3 words stands alone only when it is a question-and-answer turn (see _qa_turn) or no
    neighbour can take it: its own speaker's neighbour (within long_pause) can neither absorb it within
    max_len nor be re-split with it (keeping a sentence-end boundary at a sentence end); or, squeezed
    between other speech, no neighbour that is not another speaker's question-and-answer turn fits."""
    for k, u in enumerate(units):
        if len(u.words) > S.fragment_words or _qa_turn(units, k, prev_text):
            continue
        near = []
        for i in (k - 1, k + 1):
            if 0 <= i < len(units):
                v = units[i]
                gap = max(v.start - u.end, u.start - v.end)
                near.append((v, gap, max(u.end, v.end) - min(u.start, v.start) <= S.max_len + 1e-9, _qa_turn(units, i, prev_text)))
        own = [(v, gap, fits) for v, gap, fits, _ in near if v.speaker == u.speaker and gap < S.long_pause]
        if own:
            assert not any(fits for _, _, fits in own)                   # the speaker goes on, but max_len is in the way…
            for v, _, _ in own:                                          # …and no re-split frees the stub
                first, second = (v, u) if v.start < u.start else (u, v)
                assert not _resplittable(first.words + second.words, sentences_only=_ends(first.words[-1], second.words[0]))
        elif near and all(gap < S.long_pause for _, gap, _, _ in near):
            assert not any(fits and not qa for _, _, fits, qa in near)


@settings(max_examples=300, deadline=None)
@given(turn_lists, st.lists(st.tuples(st.sampled_from(["so", "So", "the", "model,", "runs.", "yeah", "mm-hmm", "why?"]),
                                       st.floats(0.05, 0.8), st.floats(0.0, 2.5)), min_size=1, max_size=120))
def test_speakers_lossless_and_bounded(turn_spec, word_spec):
    turns, t = [], 0.0
    for spk, gap, length in turn_spec:
        turns.append(SpeakerTurn(spk, t + gap, t + gap + length))
        t += gap + length
    words, t = [], 0.0
    for text, d, gap in word_spec:
        words.append(w(text, t, t + d))
        t += d + gap
    units = segment(words, turns)
    _check_lossless(words, units)
    assert all(u.end - u.start <= S.max_len + 1e-9 for u in units)
    assert all(a.end <= b.start for a, b in zip(units, units[1:]))
    # A word far from every turn takes the previous word's speaker (at the very start, the nearest turn's).
    prev = None
    for x in words:
        mid = (x.start + x.end) / 2
        spk = speaker_at(turns, mid, prev)
        if all(min(abs(mid - tt.start), abs(mid - tt.end)) > S.turn_snap and not tt.start <= mid <= tt.end for tt in turns):
            nearest = min(turns, key=lambda tt: min(abs(mid - tt.start), abs(mid - tt.end)), default=None)
            assert spk == (prev or (nearest.speaker if nearest else "S1"))
        prev = spk
    _check_voices(words, turns, units)
    _check_no_stranded_fragments(units)
    _check_marks(units)


def _check_marks(units):
    """Breaks and anchors are those of each unit's own words, also once fragments are joined, and only a unit that ends
    without a sentence mark just before another speaker's is cut off."""
    for u, v in zip(units, units[1:] + [None]):
        assert (u.breaks, u.anchors) == (breaks(u.words), anchors(u.words))
        if u.cut_off:
            assert v is not None and v.speaker != u.speaker and not u.words[-1].text.endswith((".", "?", "!", "…"))
    for u in merge_fragments(units):
        assert (u.breaks, u.anchors) == (breaks(u.words), anchors(u.words))


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.sampled_from("AB"), st.integers(1, 3)), min_size=3, max_size=40),
       st.sampled_from(["", "", ".", "?"]), st.integers(0, 2**16))
def test_flicker_between_speakers_is_smoothed_within_bounds(spec, punct, seed):
    """Dense back-and-forth of 1–3 word runs with no pauses (the worst diarizer output): every word is
    kept, no long run changes voice, and no fragment is stranded."""
    import random

    rng = random.Random(seed)
    words, turns, t = [], [], 0.0
    for spk, n in spec:
        start = t
        for _ in range(n):
            d = rng.uniform(0.1, 0.4)
            words.append(w("word" + (punct if rng.random() < 0.3 else ""), t, t + d))
            t += d + rng.uniform(0.0, 0.2)
        turns.append(SpeakerTurn(spk, start - 0.01, t - 0.01))
    units = segment(words, turns)
    assert _content(units) == words
    assert all(u.end - u.start <= S.max_len + 1e-9 for u in units)
    _check_voices(words, turns, units)
    _check_no_stranded_fragments(units)
    _check_marks(units)


def test_merge_fragments_joins_a_sentence_split_across_units():
    from maata_engine.segment import merge_fragments
    from maata_engine.types import SourceUnit

    a = SourceUnit(0, "S1", 0.0, 3.0, "They get their first salary, and within a week")
    b = SourceUnit(1, "S1", 3.2, 6.0, "half of it is gone.")
    c = SourceUnit(2, "S1", 6.3, 8.0, "Here's a rule.")
    out = merge_fragments([a, b, c])
    assert [u.text for u in out] == ["They get their first salary, and within a week half of it is gone.", "Here's a rule."]
    assert out[0].start == 0.0 and out[0].end == 6.0


def test_merge_fragments_keeps_speakers_long_gaps_and_the_length_limit():
    from maata_engine.segment import merge_fragments
    from maata_engine.types import SourceUnit

    other = [SourceUnit(0, "S1", 0.0, 2.0, "So what you are saying"), SourceUnit(1, "S2", 2.1, 3.0, "is true.")]
    assert len(merge_fragments(other)) == 2
    gap = [SourceUnit(0, "S1", 0.0, 2.0, "Well"), SourceUnit(1, "S1", 3.5, 4.0, "anyway.")]
    assert len(merge_fragments(gap)) == 2
    long = [SourceUnit(0, "S1", 0.0, 12.0, "and then"), SourceUnit(1, "S1", 12.2, 22.0, "we left.")]
    assert len(merge_fragments(long, max_len=20.0)) == 2
