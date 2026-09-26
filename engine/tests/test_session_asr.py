"""The listening stage's side of sentence units (ARCHITECTURE §3.4): a chunk's cut searches the pad for a sentence end,
the ASR guards reach units.jsonl, a sung-looking line is flagged but still dubbed, and a line's breaks and cut-off reach
Claude. No models and no Claude: the demo backend. All English here is original test text."""

from __future__ import annotations

import asyncio
import json
import time

from maata_engine.backends.base import Backend, Transcript
from maata_engine.backends.mock import MockDiarizer, MockSceneTranslator, MockTTS
from maata_engine.resolve import DemoResolver
from maata_engine.segment import anchors, breaks
from maata_engine.session import Chunk, Session, UnitState
from maata_engine.types import SourceUnit, TimedWord


def timed(text: str, t: float, word: float = 0.3, gap: float = 0.1, conf: float = 0.9) -> list[TimedWord]:
    out = []
    for x in text.split():
        out.append(TimedWord(x, round(t, 3), round(t + word, 3), conf))
        t += word + gap
    return out


def test_the_chunk_cut_searches_the_pad_for_a_sentence_end():
    b = 60.0
    run_on = timed("and that is how the whole thing started", 55.0)          # ends 58.1 s, no mark
    ws = run_on + timed("back then. Nobody knew", 59.9)                      # "then." ends 60.6 s, in the pad
    assert Session._sentence_cut(ws, b, last=False) == 60.6
    # A sentence end that is the audio's last word may be where the audio stopped, not the speaker: not taken.
    assert Session._sentence_cut(run_on + timed("back then.", 59.9), b, last=False) == 58.1
    opened = timed("It began here.", 50.0) + timed("That is how the whole thing started", 55.0) + ws[-4:]
    assert Session._sentence_cut(opened, b, last=False) == 51.1                         # one inside still wins
    assert Session._sentence_cut(ws, b, last=True) == ws[-1].end


def test_the_chunk_cut_reads_the_next_word_like_the_segmenter():
    """An abbreviation's "." before a lowercase word, or a title's, is no sentence end, in the pad or inside the chunk:
    a cut there would start the next chunk mid-sentence, where merge_fragments can't mend it."""
    b = 60.0
    run_on = timed("and that is how the whole thing started", 55.0)          # ends 58.1 s, no mark
    pad = timed("at 9 a.m. with the whole crew. Nobody knew", 59.9)          # "a.m." ends 61.0 s, "crew." 62.6 s
    assert Session._sentence_cut(run_on + pad, b, last=False) == 62.6
    pad = timed("with Dr. Rao about it. Then", 59.9)                        # "Dr." ends 60.6 s, "it." 61.8 s
    assert Session._sentence_cut(run_on + pad, b, last=False) == 61.8
    inside = timed("the shop opens at nine a.m. on most weekdays", 52.0)     # "a.m." ends 54.3 s
    late = timed("and the staff stay until late", 57.0) + timed("most nights. Then", 59.9)  # "nights." ends 60.6 s
    assert Session._sentence_cut(inside + late, b, last=False) == 60.6


def session(tmp_path, transcriber) -> tuple[Session, list[dict]]:
    msgs: list[dict] = []

    async def sj(m: dict) -> None:
        msgs.append(m)

    async def sb(_: bytes) -> None: ...

    backend = Backend("mock", "cpu", transcriber, MockDiarizer(), MockSceneTranslator, MockTTS())
    return Session(backend, DemoResolver(), tmp_path, sj, sb, prepass=30.0), msgs


def test_a_lines_breaks_and_cut_off_reach_claude(tmp_path):
    s, _ = session(tmp_path, None)
    ws = timed("When the rain stopped,", 10.0) + timed("we walked home slowly", 12.9)  # 1.4 s after "stopped,"
    u = SourceUnit(0, "S1", ws[0].start, ws[-1].end, " ".join(w.text for w in ws), ws, breaks=breaks(ws),
                   anchors=anchors(ws), cut_off=True)
    spec = s._spec(UnitState(u, None, Chunk(10.0, 20.0), speech_s=3.5))
    assert spec.breaks == (11.5,) and spec.cut_off                        # where the pause starts
    plain = SourceUnit(1, "S1", ws[0].start, ws[3].end, "When the rain stopped,", ws[:4])
    spec = s._spec(UnitState(plain, None, Chunk(10.0, 20.0), speech_s=1.5))
    assert spec.breaks == () and not spec.cut_off


SUNG = "la la la la la la the kettle is on the stove again la la la la la la."


class ScriptedTranscriber:
    """The first chunk's words (it starts at 0 s, so its clock is video time), as if the guards had re-decoded one
    stretch; nothing after. Keeps the speech spans each call was given."""

    def __init__(self) -> None:
        self.speech: list[list[tuple[float, float]]] = []

    def transcribe(self, audio, language=None, speech=()) -> Transcript:
        self.speech.append(list(speech))
        if len(self.speech) > 1:
            return Transcript([], "en")
        words = timed(SUNG, 0.6, 0.4, 0.1, conf=0.3) + timed("We made tea after that.", 14.5)
        return Transcript(words, "en", gaps=[(10.5, 11.5)], recovered=2, rejected=1, punctuated=True, edge_guesses=2)


async def test_asr_guards_and_the_music_flag_reach_the_trace(tmp_path):
    tr = ScriptedTranscriber()
    s, _ = session(tmp_path, tr)
    await s.open("https://youtu.be/dQw4w9WgXcQ")
    path = tmp_path / "dQw4w9WgXcQ" / "units.jsonl"
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            if path.exists() and path.read_text().count('"event": "unit"') >= 2:
                break
    finally:
        await s.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert tr.speech[0][0] == (0.0, 14.0) and tr.speech[0][-1][1] == 65.0  # the diarized turns, on the audio's clock
    asr = next(e for e in events if e["event"] == "asr")
    assert (asr["redecoded"], asr["recovered"], asr["rejected"]) == ([[10.5, 11.5]], 2, 1)
    assert asr["punctuated"] is True and asr["edge_guesses"] == 2
    assert asr["uncovered"] == [[10.0, 14.5]]                            # between the song and the next line
    assert asr["low_confidence"] == [[0.6, 10.0]] and asr["repeats"] == [[0.6, 3.5], [7.1, 10.0]]
    units = {e["source"]: e for e in events if e["event"] == "unit"}
    assert units[SUNG]["music"] is True and units["We made tea after that."]["music"] is False  # flagged, still voiced
    assert not any(e["event"] == "skipped" for e in events)
