"""Segmenter (§6.2; ARCHITECTURE §3.4): turn timed words + speaker turns into dubbing units, one per sentence.

Units are sentence-complete wherever the audio allows (the ASR's punctuation comes from text/punct_transfer.py). A word
ending in "." ends a sentence only before a word that may open one (Whisper capitalises sentence starts), so "9 a.m.
with" runs on.

Speaker labels are smoothed first: a run of another speaker that is short (≤ 3 words or < 1 s)
between one speaker's words is a diarization flip (a laugh, a mislabelled word) and takes the flanking
speaker, unless it is a real turn (see _real_turn): a question-and-answer turn, genuine crosstalk, or a
turn set off by long pauses. A backchannel there is left to be dropped below, unless the flanking
sentence runs straight through it ("turn right and park"). Then a turn change that falls mid-sentence
moves to the nearest sentence end at most 3 words away, so the first speaker's last word is not voiced
by the next (see _boundaries).

Then split on a speaker change; split softly only at a sentence end once the unit is ≥ min_len, or at
a pause ≥ long_pause after complete speech. A pause inside a sentence never splits it: one of break_pause or more is
a break inside the unit (`SourceUnit.breaks`). A unit that would pass max_len splits at its last sentence end, else
its last clause mark, else (a run with no marks) its longest pause, never inside a name or a number and never
mid-word. Fragments merge into a neighbour of the same speaker. Short backchannels ("yeah", "mm-hmm") interjected by
the other speaker are dropped unless
they answer a question; no other word is ever dropped. Last, a unit of ≤ 3 words that is not a turn of
its own (see _stands: a real turn kept above, or a question-and-answer turn) merges into a neighbour:
its own speaker's when that speaker goes on, else (squeezed between other speakers' speech) the
neighbour it reads on with, never another speaker's turn of its own. A piece of a real turn that a sentence end split
off still rejoins the rest of that turn. Where max_len keeps it from its
own speaker's neighbour (a stub left by a max_len cut), the two are re-split so both halves are longer
than a fragment, moving a mid-sentence cut elsewhere mid-sentence or a sentence-end cut to another
sentence end. Only a long pause after complete speech, or max_len with no such re-split, leaves one standing alone.

A unit stays one translation unit (Telugu is verb-final, so it is not cut at English pauses). `breaks` mark its hard
timing boundaries, and `anchors` where it breathes: pauses ≥ anchor_pause at a clause or sentence mark, where the dub
may re-sync (soft; ARCHITECTURE §3.10). A unit that ends mid-sentence as another speaker comes in is `cut_off` when
something shows an interruption (see _cut_off), not merely because its mark is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import SourceUnit, SpeakerTurn, TimedWord

_SENTENCE_END = (".", "?", "!", "…", "।")
_CLAUSE_END = (",", ";", ":", "—", "-")
_CLOSERS = "\"')]}»”’"
# Words ending in "." that come before a capitalised word without ending a sentence (titles, "e.g."). Words that
# end sentences as often ("a.m.", "U.S.", "Inc.", "No.") are not listed: sentence_end looks at the next word.
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "mx.", "dr.", "prof.", "st.", "vs.", "jr.", "sr.", "e.g.", "i.e.", "approx.",
                  "cf.", "fig.", "gen.", "gov.", "sen.", "capt.", "lt.", "sgt."}
_BACKCHANNELS = {"yeah", "yes", "yep", "mm", "mhm", "mm-hmm", "uh-huh", "right", "okay", "ok", "hmm", "uh", "um", "ah", "oh", "wow"}
_FILLERS = {"mm", "mhm", "mm-hmm", "uh-huh", "hmm", "uh", "um"}  # backchannels that are never part of a sentence
_ANSWERS = {"yeah", "yes", "yep", "mhm", "mm-hmm", "uh-huh", "right", "okay", "ok"}  # backchannels that can answer a question
# Words no sentence ends on: a speaker who stops on one before another comes in was cut off ("…and", "…the").
_DANGLING = {"and", "but", "or", "because", "the", "a", "an", "if", "than", "my", "your", "our", "their"}
# Words of a spoken number: a cut never falls between two of them ("twenty | five", "3 | million").
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                 "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                 "billion", "trillion", "lakh", "lakhs", "crore", "crores", "point", "percent", "dozen"}

Group = tuple[str, list[TimedWord]]


@dataclass(frozen=True, slots=True)
class SegmenterSettings:
    min_len: float = 2.0          # a sentence end splits only once the unit is this long
    # Hard ceiling; over it, split at the best mark or pause. Kept so a max_len unit's Telugu
    # (budget ≈ max_len + borrow) can run to 2x its slot under the TTS cap (session.MAX_LINE_SECONDS).
    max_len: float = 12.0
    backchannel_max: float = 0.6
    long_pause: float = 1.5       # s of pause that splits after complete speech (inside a sentence it is a break)
    break_pause: float = 1.0      # s of pause inside a unit that is a hard timing break (SourceUnit.breaks)
    merge_gap: float = 1.0        # fragments merge across at most this much silence
    turn_snap: float = 0.5        # a word in a diarization gap takes a turn this close
    # A run of another speaker between one speaker's words, at most flip_words words or under flip_max s, is a
    # diarization flip unless it is a real turn (see _real_turn).
    flip_words: int = 3
    flip_max: float = 1.0
    overlap_min: float = 0.1      # s of overlapping turns inside a short run that make it genuine crosstalk
    turn_gap: float = 0.3         # s of pause that can be a turn change (shorter ones are within one speaker's speech)
    fragment_words: int = 3       # a unit this short merges into a neighbour unless it answers a question
    # s of pause at a clause or sentence mark inside a unit that is a (soft) timing anchor: Amazon's 300 ms phrase
    # pause (report 08 F2). If lines average more than about 3 anchors, raise it to 0.4-0.5 s (ARCHITECTURE §3.4).
    anchor_pause: float = 0.3


def speaker_at(turns: list[SpeakerTurn], t: float, prev: str | None = None, snap: float = 0.5) -> str:
    """The turn containing t (the longest, if several); else the nearest turn within `snap` s;
    else `prev`, the previous word's speaker; else (a window's first word) the nearest turn at
    any distance; "S1" only when there are no turns. Ties go to `prev`."""
    best, best_len = None, -1.0
    near, near_d = None, float("inf")
    for turn in turns:
        if turn.start <= t <= turn.end:
            ln = turn.end - turn.start
            if ln > best_len or (ln == best_len and turn.speaker == prev):
                best, best_len = turn.speaker, ln
            continue
        d = turn.start - t if t < turn.start else t - turn.end
        if d < near_d or (d == near_d and turn.speaker == prev):
            near, near_d = turn.speaker, d
    for spk in (best, near if near_d <= snap else None, prev, near):
        if spk is not None:
            return spk
    return "S1"


def _bare(text: str) -> str:
    return text.rstrip(_CLOSERS)


def _opens(text: str) -> bool:
    """Whether a word may open a sentence: its first letter or digit is not lowercase (Whisper capitalises
    sentence starts). A word with neither counts."""
    for ch in text:
        if ch.isalnum():
            return not ch.islower()
    return True


def sentence_end(text: str, nxt: str | None = None) -> bool:
    """Whether the word `text` ends a sentence, `nxt` being the next word's text (None after the last word).
    A word ending in "." or "…" does only when the next word may open a sentence, so "at 9 a.m. with" and
    "in the U.S. and" run on; a known abbreviation never does."""
    t = _bare(text)
    if not t.endswith(_SENTENCE_END) or t.lower() in _ABBREVIATIONS:
        return False
    return nxt is None or not t.endswith((".", "…")) or _opens(nxt)


def _clause_end(text: str) -> bool:
    return _bare(text).endswith(_CLAUSE_END)


def _marked(text: str) -> bool:
    """The word ends with any sentence or clause mark."""
    return _bare(text).endswith(_SENTENCE_END + _CLAUSE_END)


def _question(text: str) -> bool:
    return _bare(text).endswith("?")


def _numeric(text: str) -> bool:
    t = text.lower().strip(".,!?;:\"'()$%₹€£")
    return any(ch.isdigit() for ch in t) or t in _NUMBER_WORDS


def _capital(text: str) -> bool:
    return next((ch for ch in text if ch.isalnum()), "").isupper()


def _bound(a: TimedWord, b: TimedWord) -> bool:
    """Whether no cut may fall between the words a and b: with no mark between them, they are parts of one name (both
    capitalised: "New York") or one number ("twenty five", "3 million")."""
    if _marked(a.text):
        return False
    return (_numeric(a.text) and _numeric(b.text)) or (_capital(a.text) and _capital(b.text))


def _complete(last: TimedWord, nxt: TimedWord | None, s: SegmenterSettings) -> bool:
    """Whether speech ending with `last` is a complete utterance rather than the start of a sentence that
    `nxt` (the next word; None after the last) goes on with: `last` ends a sentence, or `nxt` may open a
    new one after a turn-taking gap."""
    if nxt is None:
        return True
    return sentence_end(last.text, nxt.text) or (nxt.start - last.end >= s.turn_gap and _opens(nxt.text))


def _fresh(prev: TimedWord | None, first: TimedWord, s: SegmenterSettings) -> bool:
    """Whether speech starting with `first` starts a sentence rather than going on with `prev` (the word
    before; None before the first): `prev` ends a sentence, or `first` may open one after a turn-taking gap
    (with no word before, it may open one)."""
    if prev is None:
        return _opens(first.text)
    return sentence_end(prev.text, first.text) or (first.start - prev.end >= s.turn_gap and _opens(first.text))


def _span(ws: list[TimedWord]) -> float:
    return ws[-1].end - ws[0].start


def _word(w: TimedWord) -> str:
    return w.text.lower().strip(".,!?")


def _is_backchannel(words: list[TimedWord], s: SegmenterSettings) -> bool:
    if not words or _span(words) > s.backchannel_max:
        return False
    return all(_word(w) in _BACKCHANNELS for w in words)


def _runs(labels: list[str]) -> list[tuple[int, int]]:
    """Maximal runs of equal labels, as inclusive (first, last) index pairs."""
    out: list[tuple[int, int]] = []
    for k, lab in enumerate(labels):
        if out and labels[out[-1][0]] == lab:
            out[-1] = (out[-1][0], k)
        else:
            out.append((k, k))
    return out


def _run_of(labels: list[str]) -> list[tuple[int, int]]:
    """For each index, the (first, last) indices of its run of equal labels."""
    return [r for r in _runs(labels) for _ in range(r[0], r[1] + 1)]


def _crosstalk(overlaps: list[SpeakerTurn], spk: str, other: str, a: float, b: float, s: SegmenterSettings) -> bool:
    """Whether the turns in `overlaps` show spk and other talking at once for at least overlap_min s inside [a, b]."""
    live = [t for t in overlaps if t.start < b and t.end > a]
    return any(m.speaker == spk and o.speaker == other and min(m.end, o.end, b) - max(m.start, o.start, a) >= s.overlap_min
               for m in live for o in live)


def _real_turn(words: list[TimedWord], i: int, j: int, spk: str, flank: str, overlaps: list[SpeakerTurn],
               s: SegmenterSettings) -> str:
    """What the short run words[i..j] of `spk`, between two words of `flank`, is.

    "turn": a real turn, which stays a unit of its own (see _stands). The overlapping turns show both
    people talking at once there (crosstalk); or long pauses set it off on both sides; or it answers the
    question just asked and is a complete utterance, not the start of the flanking speaker's next sentence
    ("…mean here? It means the model…" is a flip); or it is a question, starting a sentence, that the
    flanking speaker answers after a turn-taking gap.
    "backchannel": left to _drop_backchannels (dropped, or kept as an answer). A filler ("mm-hmm") always
    is; a word like "right" or "yes" only when something sets it off from the flanking speaker's sentence
    (a mark before or on it, a turn-taking gap, crosstalk), else that sentence runs straight through it
    ("turn right and park") and it is a flip.
    "flip": a diarization flip; it takes the flanking speaker.
    """
    run, before, after = words[i:j + 1], words[i - 1], words[j + 1]
    a, b = run[0].start, run[-1].end
    crosstalk = _crosstalk(overlaps, spk, flank, a, b, s)
    if _is_backchannel(run, s):
        through = not (crosstalk or _marked(before.text) or _marked(run[-1].text)
                       or a - before.end >= s.turn_gap or after.start - b >= s.turn_gap)
        return "flip" if through and not all(_word(w) in _FILLERS for w in run) else "backchannel"
    if crosstalk or (a - before.end >= s.long_pause and after.start - b >= s.long_pause):
        return "turn"
    if _question(before.text) and _complete(run[-1], after, s):
        return "turn"  # it answers the question just asked
    if _question(run[-1].text) and after.start - b >= s.turn_gap and _fresh(before, run[0], s):
        return "turn"  # it asks a question the flanking speaker answers
    return "flip"


def _speakers(words: list[TimedWord], turns: list[SpeakerTurn], overlaps: list[SpeakerTurn],
              s: SegmenterSettings) -> tuple[list[str], set[int]]:
    """Each word's speaker, with diarization flips smoothed out, and the ids of the words in short runs kept
    as real turns.

    A run of another speaker between two words of one speaker, ≤ flip_words words or < flip_max s,
    takes the flanking speaker when _real_turn says it is a flip. Runs are relabelled one at a time,
    shortest first (then the one with the most flanking words), so alternating flips all go to the
    speaker holding the floor.
    """
    labels: list[str] = []
    spk = None
    for w in words:
        spk = speaker_at(turns, (w.start + w.end) / 2, spk, s.turn_snap)
        labels.append(spk)
    real: dict[tuple[int, int, str, str], str] = {}
    while True:
        runs = _runs(labels)
        best, best_key = None, None
        for k in range(1, len(runs) - 1):
            i, j = runs[k]
            flank = labels[i - 1]
            run = words[i:j + 1]
            if labels[j + 1] != flank or (len(run) > s.flip_words and _span(run) >= s.flip_max):
                continue
            key = (i, j, labels[i], flank)
            if key not in real:
                real[key] = _real_turn(words, i, j, labels[i], flank, overlaps, s)
            if real[key] != "flip":
                continue
            flanking = (i - runs[k - 1][0]) + (runs[k + 1][1] - j)
            rank = (_span(run), len(run), -flanking, i)
            if best_key is None or rank < best_key:
                best, best_key = (i, j, flank), rank
        if best is None:
            break
        i, j, flank = best
        labels[i:j + 1] = [flank] * (j + 1 - i)
    kept: set[int] = set()
    runs = _runs(labels)
    for k in range(1, len(runs) - 1):
        i, j = runs[k]
        if labels[j + 1] == labels[i - 1] and real.get((i, j, labels[i], labels[i - 1])) == "turn":
            kept.update(id(w) for w in words[i:j + 1])
    return labels, kept


def _boundaries(words: list[TimedWord], labels: list[str], kept: set[int], s: SegmenterSettings) -> None:
    """Move each turn change that falls mid-sentence to the nearest sentence end at most fragment_words
    words away (labels are changed in place).

    The diarizer often starts the next speaker's turn a word or two early or late. A change is mid-sentence
    when the word before it does not end a sentence, the word after it is lowercase (Whisper capitalises a
    new sentence) and no turn-taking gap lies between them. Then either the next speaker's first sentence
    goes back to the first speaker, when it ends within fragment_words words ("…the whole model | locally.
    Did you…"), or the first speaker's words after their last sentence end go to the next speaker, when
    there are at most fragment_words of them; whichever moves fewer words, the first on a tie. The speaker
    giving words up must keep more than a fragment in that run (else the fragment rules place the rest).
    A run kept as a real turn, or a backchannel left for _drop_backchannels, is never changed.
    """
    fw = s.fragment_words
    runs = _run_of(labels)
    for i in range(1, len(words)):
        prev, first = words[i - 1], words[i]
        a_spk, b_spk = labels[i - 1], labels[i]
        if (a_spk == b_spk or sentence_end(prev.text, first.text) or _opens(first.text)
                or first.start - prev.end >= s.turn_gap or id(prev) in kept or id(first) in kept):
            continue
        b_end, a_start = runs[i][1], runs[i - 1][0]
        if _is_backchannel(words[a_start:i], s) or _is_backchannel(words[i:b_end + 1], s):
            continue
        moves = []
        for e in range(i, min(i + fw, b_end + 1)):        # words[i..e] go back to a_spk
            if sentence_end(words[e].text, words[e + 1].text if e + 1 < len(words) else None):
                if b_end - e > fw:
                    moves.append((e + 1 - i, 0, i, e, a_spk))
                break
        for lo in range(i - 1, max(i - 1 - fw, a_start), -1):  # words[lo..i-1] go to b_spk
            if sentence_end(words[lo - 1].text, words[lo].text):
                if lo - a_start > fw:
                    moves.append((i - lo, 1, lo, i - 1, b_spk))
                break
        if moves:
            _, _, lo, hi, spk = min(moves)
            labels[lo:hi + 1] = [spk] * (hi + 1 - lo)
            runs = _run_of(labels)


def _long_break(prev: TimedWord, w: TimedWord, s: SegmenterSettings) -> bool:
    """Whether a pause between prev and w ends the unit: it is at least long_pause, after complete speech: prev ends a
    sentence, or w's capital shows a new one. A number or the pronoun I (never lowercase) shows nothing, and nothing
    does after a word no sentence ends on. One inside a sentence ("and the answer was … 42.", "…to the … Priya")
    does not split it: the sentence stays one unit, with a break."""
    if w.start - prev.end < s.long_pause:
        return False
    if sentence_end(prev.text, w.text):
        return True
    return _capital(w.text) and _word(w).replace("’", "'").split("'")[0] != "i" and _word(prev) not in _DANGLING


def _soft_split(cur: list[TimedWord], w: TimedWord, s: SegmenterSettings) -> bool:
    prev = cur[-1]
    return _long_break(prev, w, s) or (sentence_end(prev.text, w.text) and prev.end - cur[0].start >= s.min_len)


def _cut(ws: list[TimedWord], nxt: TimedWord, s: SegmenterSettings) -> int:
    """Split point k (head = ws[:k]) for a group that `nxt` would push past max_len.

    The last sentence end wins, even with a short head: the sentence after it may still fit
    whole, and a stranded "Okay." is kept by _drop_backchannels. Then the last clause mark, so a sentence is cut only
    between clauses (never between a verb and its object while a comma is there to cut at). Only a run with no mark is
    cut at its longest pause (to 10 ms; latest on a tie), never inside a name or a number (see _bound). Both prefer
    heads ≥ min_len. Cutting before `nxt` (k = len(ws)) is allowed.
    """
    ks = range(1, len(ws) + 1)
    after = ws[1:] + [nxt]
    for k in reversed(ks):
        if sentence_end(ws[k - 1].text, after[k - 1].text):
            return k
    ok = [k for k in ks if ws[k - 1].end - ws[0].start >= s.min_len] or list(ks)
    for k in reversed(ok):
        if _clause_end(ws[k - 1].text):
            return k
    free = [k for k in ok if not _bound(ws[k - 1], after[k - 1])] or ok
    return max(free, key=lambda k: (round(after[k - 1].start - ws[k - 1].end, 2), k))


def _group(words: list[TimedWord], labels: list[str], s: SegmenterSettings) -> list[Group]:
    groups: list[Group] = []
    cur: list[TimedWord] = []
    cur_spk = ""
    for w, spk in zip(words, labels):
        if cur and (spk != cur_spk or _soft_split(cur, w, s)):
            groups.append((cur_spk, cur))
            cur = []
        while cur and w.end - cur[0].start > s.max_len:
            k = _cut(cur, w, s)
            groups.append((cur_spk, cur[:k]))
            cur = cur[k:]
        if not cur:
            cur_spk = spk
        cur.append(w)
    if cur:
        groups.append((cur_spk, cur))
    return groups


def _mergeable(a: Group, b: Group, s: SegmenterSettings) -> bool:
    (sa, wa), (sb, wb) = a, b
    gap = wb[0].start - wa[-1].end
    if sa != sb or wb[-1].end - wa[0].start > s.max_len:
        return False
    last = wa[-1].text
    if not sentence_end(last, wb[0].text):
        # A mid-sentence fragment, e.g. cut by the other speaker's dropped "yeah": rejoin across any
        # gap _group wouldn't have split at (the gap includes the interjection itself).
        return not _long_break(wa[-1], wb[0], s)
    return gap <= s.merge_gap and (_span(wa) < s.min_len or _span(wb) < s.min_len) and not _bare(last).endswith(("?", "!"))


def _merge(groups: list[Group], s: SegmenterSettings) -> list[Group]:
    out: list[Group] = []
    for g in groups:
        if out and _mergeable(out[-1], g, s):
            out[-1] = (g[0], out[-1][1] + g[1])
        else:
            out.append(g)
    return out


def _drop_backchannels(groups: list[Group], s: SegmenterSettings, prev_text: str | None) -> list[Group]:
    """Drop interjections, judged by the group before (else after); a speaker's own "Okay."
    before their next sentence stays, and so does an answer to a question. At the window start
    the question is `prev_text`; when that is unknown (None), a yes-like word is kept."""
    kept: list[Group] = []
    for i, (spk, ws) in enumerate(groups):
        before = groups[i - 1] if i else None
        after = groups[i + 1] if i + 1 < len(groups) else None
        if after is not None and after[0] == spk and after[1][0].start - ws[-1].end < s.long_pause:
            kept.append((spk, ws))  # the speaker goes on: their own lead-in, not an interjection
            continue
        other = before or after
        q = before[1][-1].text if before else prev_text
        answers = all(_word(x) in _ANSWERS for x in ws) if q is None else _bare(q).endswith("?")
        interjection = not answers and (other is None or other[0] != spk)
        if not (interjection and _is_backchannel(ws, s)):
            kept.append((spk, ws))
    return kept


def _answers(groups: list[Group], k: int, s: SegmenterSettings, prev_text: str | None) -> bool:
    """Whether group k answers a question: another speaker's group just before it ends with "?" (at the
    window start the question is `prev_text`; when that is unknown (None), a yes-like group counts), and it
    is a complete utterance, not the start of a sentence that the next group goes on with (see _complete)."""
    spk, ws = groups[k]
    if k:
        before, bws = groups[k - 1]
        asked = before != spk and _question(bws[-1].text)
    elif prev_text is None:
        asked = all(_word(x) in _ANSWERS for x in ws)
    else:
        asked = _question(prev_text)
    return asked and _complete(ws[-1], groups[k + 1][1][0] if k + 1 < len(groups) else None, s)


def _asks(groups: list[Group], k: int, s: SegmenterSettings) -> bool:
    """Whether group k is a question that another speaker answers: it starts a sentence (see _fresh), ends
    with "?", and the next group is another speaker's, after a turn-taking gap."""
    spk, ws = groups[k]
    if k + 1 >= len(groups) or groups[k + 1][0] == spk or not _question(ws[-1].text):
        return False
    return (groups[k + 1][1][0].start - ws[-1].end >= s.turn_gap
            and _fresh(groups[k - 1][1][-1] if k else None, ws[0], s))


def _stands(groups: list[Group], k: int, s: SegmenterSettings, prev_text: str | None, kept: set[int]) -> bool:
    """Whether group k is a turn of its own, which stays a unit however short: a run _speakers kept as a
    real turn, or a question-and-answer turn (see _answers and _asks)."""
    return (all(id(x) in kept for x in groups[k][1]) or _answers(groups, k, s, prev_text)
            or _asks(groups, k, s))


def _host(groups: list[Group], k: int, s: SegmenterSettings, prev_text: str | None, kept: set[int]) -> int | None:
    """The neighbour that fragment k merges into, or None when it stands alone.

    The speaker's own neighbour (within long_pause) when there is one: the speaker goes on, so their
    words stay theirs, even when max_len then leaves the fragment alone. Else, when the fragment is
    squeezed between other speech (every gap < long_pause), the other speaker's neighbour: it is a
    split-off piece of that speech. Another speaker's turn of its own (see _stands) takes no one else's
    words. Between candidates, the one the fragment reads on with wins (it ends mid-sentence into the next
    group, or the previous group ends mid-sentence into it), then the shorter gap.

    A question-and-answer turn stands alone. A piece of a run kept as a real turn still rejoins its own
    speaker (the rest of that turn, split at a sentence end), but no other speaker takes it.
    """
    spk, ws = groups[k]
    if len(ws) > s.fragment_words or _answers(groups, k, s, prev_text) or _asks(groups, k, s):
        return None
    turn = all(id(x) in kept for x in ws)
    near: list[tuple[int, float]] = []
    for j in (k - 1, k + 1):
        if 0 <= j < len(groups):
            other = groups[j][1]
            near.append((j, ws[0].start - other[-1].end if j < k else other[0].start - ws[-1].end))
    own = [(j, gap) for j, gap in near if groups[j][0] == spk and gap < s.long_pause]
    if own:
        pool = own
    elif not turn and near and all(gap < s.long_pause for _, gap in near):
        pool = [(j, gap) for j, gap in near if not _stands(groups, j, s, prev_text, kept)]
    else:
        return None

    def reads_on(j: int) -> bool:
        head, tail = (groups[j][1], ws) if j < k else (ws, groups[j][1])
        return not sentence_end(head[-1].text, tail[0].text)

    fits = [(not reads_on(j), gap, j) for j, gap in pool
            if max(ws[-1].end, groups[j][1][-1].end) - min(ws[0].start, groups[j][1][0].start) <= s.max_len]
    return min(fits)[2] if fits else None


def _absorb_fragments(groups: list[Group], s: SegmenterSettings, prev_text: str | None, kept: set[int]) -> list[Group]:
    """Merge every fragment (≤ fragment_words words) that has a host (see _host) into it."""
    out = list(groups)
    k = 0
    while k < len(out):
        j = _host(out, k, s, prev_text, kept)
        if j is None:
            k += 1
            continue
        lo = min(j, k)
        out[lo:lo + 2] = [(out[j][0], out[lo][1] + out[lo + 1][1])]
        k = max(lo - 1, 0)  # the grown group's neighbours may have a new host now
    return out


def _resplit(ws: list[TimedWord], s: SegmenterSettings, sentences_only: bool) -> int | None:
    """Split point k (head = ws[:k]) leaving both parts longer than a fragment and within max_len, or None.

    With `sentences_only`, only a sentence end qualifies (the latest). Otherwise, as in _cut: the
    latest sentence end, then the latest clause mark, then the longest pause not inside a name or a number, the last
    two preferring parts ≥ min_len.
    """
    fw = s.fragment_words
    ks = [k for k in range(fw + 1, len(ws) - fw)
          if ws[k - 1].end - ws[0].start <= s.max_len and ws[-1].end - ws[k].start <= s.max_len]
    ends = [k for k in ks if sentence_end(ws[k - 1].text, ws[k].text)]
    if ends or sentences_only:
        return ends[-1] if ends else None
    if not ks:
        return None
    ok = [k for k in ks if ws[k - 1].end - ws[0].start >= s.min_len and ws[-1].end - ws[k].start >= s.min_len] or ks
    clauses = [k for k in ok if _clause_end(ws[k - 1].text)]
    if clauses:
        return clauses[-1]
    free = [k for k in ok if not _bound(ws[k - 1], ws[k])] or ok
    return max(free, key=lambda k: (round(ws[k].start - ws[k - 1].end, 2), k))


def _rebalance(groups: list[Group], s: SegmenterSettings, prev_text: str | None, kept: set[int]) -> list[Group]:
    """Re-split a fragment with its own speaker's neighbour when max_len keeps them apart.

    Such a stub is what a max_len cut leaves when the speaker's run ends a word or three later. The pair
    (within long_pause of each other) is cut again so both parts are longer than a fragment: anywhere
    (see _resplit) when the old cut was mid-sentence, else only at another sentence end, so a sentence
    that was whole stays whole. A turn of its own is left as it is (see _stands).
    """
    out = list(groups)
    for k in range(len(out) - 1):
        (sa, wa), (sb, wb) = out[k], out[k + 1]
        if sa != sb or wb[0].start - wa[-1].end >= s.long_pause or wb[-1].end - wa[0].start <= s.max_len:
            continue
        if not any(len(out[i][1]) <= s.fragment_words and not _stands(out, i, s, prev_text, kept) for i in (k, k + 1)):
            continue
        ws = wa + wb
        cut = _resplit(ws, s, sentences_only=sentence_end(wa[-1].text, wb[0].text))
        if cut is not None:
            out[k], out[k + 1] = (sa, ws[:cut]), (sb, ws[cut:])
    return out


def _settle_fragments(groups: list[Group], s: SegmenterSettings, prev_text: str | None, kept: set[int]) -> list[Group]:
    """Absorb, re-split and merge until nothing changes. Each pass that changes anything removes a group
    or a fragment, so this ends."""
    while True:
        nxt = _merge(_rebalance(_absorb_fragments(groups, s, prev_text, kept), s, prev_text, kept), s)
        if nxt == groups:
            return nxt
        groups = nxt


def anchors(words: list[TimedWord], s: SegmenterSettings = SegmenterSettings()) -> list[int]:
    """Timing anchors inside one unit's words: indices k where the dub may re-sync to words[k].start.

    An anchor is a pause of at least anchor_pause after a clause or sentence mark (any "." or "…" too,
    even one the sentence runs on after: it still marks a breath). The unit is still translated and voiced
    whole; an anchor only tells the timeline where the English breathes, so a line running late may catch
    up there if the Telugu breaks near it. It is never forced to.
    """
    return [k for k in range(1, len(words))
            if words[k].start - words[k - 1].end >= s.anchor_pause
            and (sentence_end(words[k - 1].text) or _clause_end(words[k - 1].text))]


def breaks(words: list[TimedWord], s: SegmenterSettings = SegmenterSettings()) -> list[int]:
    """Hard breaks inside one unit's words: indices k where a pause of at least break_pause comes before words[k],
    marked or not. The sentence is still translated whole; the timeline splits its dub there, at a point where the
    Telugu pauses naturally (ARCHITECTURE §3.10)."""
    return [k for k in range(1, len(words)) if words[k].start - words[k - 1].end >= s.break_pause]


def _dangles(ws: list[TimedWord]) -> bool:
    """Whether speech ending with ws[-1] stops on a word no sentence ends on: one of _DANGLING, or a "so" that opens its
    clause (alone, or after a mark: "…, so"), unlike the "so" of "I think so"."""
    last = _word(ws[-1])
    return last in _DANGLING or (last == "so" and (len(ws) == 1 or _marked(ws[-2].text)))


def _cut_off(g: Group, nxt: Group | None, overlaps: list[SpeakerTurn], s: SegmenterSettings) -> bool:
    """Whether the speech of group g is cut off by the next group, another speaker's. It ends mid-sentence, and:
    - it stops on a word no sentence ends on ("…and", "…, so") before the other speaker comes in (see _dangles);
    - or the other speaker goes on with the sentence, in lowercase, with no turn-taking gap (a mid-sentence flip the
      diarizer could not settle, ARCHITECTURE §3.3);
    - or `overlaps` show the two talking at once as the turn changes (they talk over it).
    A quick answer to a question left unpunctuated ("What time does it leave | It leaves at noon.") is not cut off,
    and a backchannel never is."""
    spk, ws = g
    last = ws[-1]
    if nxt is None or nxt[0] == spk or sentence_end(last.text) or all(_word(w) in _BACKCHANNELS for w in ws):
        return False
    first = nxt[1][0]
    gap = first.start - last.end
    if gap < s.long_pause and _dangles(ws):
        return True
    if gap < s.turn_gap and not _opens(first.text) and _word(first) not in _FILLERS:
        return True
    return _crosstalk(overlaps, spk, nxt[0], last.start, first.end, s)


def segment(words: list[TimedWord], turns: list[SpeakerTurn], s: SegmenterSettings = SegmenterSettings(),
            *, prev_text: str | None = None, overlaps: list[SpeakerTurn] | None = None) -> list[SourceUnit]:
    """Dubbing units for one window. `prev_text` is the source text just before the window (the
    previous unit's), so a "Yes." opening the window is recognised as an answer; "" at the video start.

    `turns` label the words (exclusive diarization is best). `overlaps` are the diarizer's turns with
    crosstalk kept (e.g. SpeakerRegistry.turns_in(a, b, exclusive=False)); a short run of another
    speaker keeps its label, and stays a unit of its own, where they overlap the flanking speaker's.
    Defaults to `turns`.
    """
    overlaps = turns if overlaps is None else overlaps
    labels, kept = _speakers(words, turns, overlaps, s)
    _boundaries(words, labels, kept, s)
    groups = _merge(_group(words, labels, s), s)
    # Merge again once interjections are gone, so a sentence split by one rejoins.
    groups = _merge(_drop_backchannels(groups, s, prev_text), s)
    groups = _settle_fragments(groups, s, prev_text, kept)
    return [
        SourceUnit(id=i, speaker=spk, start=ws[0].start, end=ws[-1].end, text=" ".join(w.text for w in ws).strip(),
                   words=ws, breaks=breaks(ws, s), anchors=anchors(ws, s),
                   cut_off=_cut_off((spk, ws), groups[i + 1] if i + 1 < len(groups) else None, overlaps, s))
        for i, (spk, ws) in enumerate(groups)
    ]


def merge_fragments(units: list[SourceUnit], max_len: float = 20.0, max_gap: float = 1.0,
                    s: SegmenterSettings = SegmenterSettings()) -> list[SourceUnit]:
    """Join a unit that ends mid-sentence with the same speaker's next unit, so every sentence is translated and dubbed
    whole. Translating half a sentence made the model invent the rest (the maintainer's podcast, 2026-09-24); the
    timeline planner fits the joined line over the combined span. Stops at `max_len` s so the TTS cap still holds.

    Once units are sentence-complete this is a safety net (ARCHITECTURE §3.4). The joined unit keeps both parts' breaks
    and anchors, and the join itself is a break when its pause is at least break_pause (an anchor too, where
    `anchors` would put one). It is cut off when its second part is."""
    out: list[SourceUnit] = []
    for u in units:
        if out:
            p = out[-1]
            if (p.speaker == u.speaker and not p.text.rstrip("\"'”’)").endswith(_SENTENCE_END)
                    and u.start - p.end <= max_gap and u.end - p.start <= max_len):
                n = len(p.words)
                seam = [n] if n and u.words else []  # the join, as an index into the joined words
                out[-1] = SourceUnit(
                    p.id, p.speaker, p.start, u.end, f"{p.text} {u.text}".strip(), list(p.words) + list(u.words),
                    breaks=p.breaks + (seam if u.start - p.end >= s.break_pause else []) + [n + k for k in u.breaks],
                    anchors=p.anchors + (seam if seam and anchors([p.words[-1], u.words[0]], s) else [])
                    + [n + k for k in u.anchors],
                    cut_off=u.cut_off)
                continue
        out.append(u)
    return out
