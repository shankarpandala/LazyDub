"""The Claude prompts for scene translation (docs/research/dubbing-2026-09/ARCHITECTURE.md §4.3, §4.4).

One system prompt serves the scene, fit, re-translate and rephrase calls of a video, and it is byte-identical for the
whole video (a new brief is the only change), so separate `claude -p` processes in one fixed working directory share the
prompt cache. The call type goes in the user message. Brief and review calls have their own system prompts.

The system prompt: role; style guide (colloquial, and the formal style that replaced the retired local formal model);
the Telugu-script contract; the two-way length contract; original examples written to that contract; the video's brief;
the JSON contract. The schemas are draft-07 without `format`, and `claude_cli.check_schema` must accept them, so the
local validator checks everything the CLI does. Pure: no model, no I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ..backends.base import EMOTIONS, ENERGIES, Brief, GlossaryEntry, LineSpec, SceneRequest, VideoMeta, Wording

# Bump SHOTS_VERSION whenever the prompt, an example or a schema changes; PROMPT_HASH follows the content and keys the
# per-line cache, so a change never serves lines made under the old prompt.
SHOTS_VERSION = "scene-v2"  # v2: a fit copies "current" into "full"
DESCRIPTION_MAX = 2000  # characters of the video description placed in the brief

ROLE = (
    "You dub English speech into spoken Telugu: the words a fluent, educated Telugu speaker says aloud on a Telugu "
    "YouTube explainer or podcast. A Telugu text-to-speech voice says your Telugu over the original video, so every "
    "line must be speakable exactly as written."
)

STYLE = """STYLE (colloquial; the default)
- Sishta vyavaharika: standard spoken Telugu of the central dialect, neutral between Andhra and Telangana. Never grandhika words such as యొక్క, తద్వారా, మరియు, కావున.
- No English quota. Say a word in English only where educated Telugu speakers really say it in English: names, brands, technical and modern terms, and established loans (ఫోన్, టైం, జాబ్, ఫ్రెండ్). Many lines need no English at all; the topic decides.
- An English verb is the bare English stem followed by చేయు or అవు (రీస్టార్ట్ చేయండి, లేట్ అయింది), never an -ed or -ing form. Everyday verbs stay Telugu.
- Drop pronouns the verb already marks. Use spoken verb forms (చెప్తా, చేస్తున్నా, చేయొచ్చు, ఏంటి).
- Translate fillers and discourse markers by their meaning (so → అంటే or అసలు, like → అంటే, you know → కదా); never carry them over in English.
- Use the address forms the brief gives for each speaker and for the audience: మీరు to the audience and to anyone addressed politely, నువ్వు only between people the brief or the English shows to be close.
- An idiom, pun or joke becomes a Telugu equivalent or a plain paraphrase, never a word-for-word calque.
- Numbers are spoken Telugu words (నలభై ఐదు, రెండు వేల ఇరవై ఆరు). An acronym is spelled as it is said (ఏఐ, జీపీయూ).
- Keep every fact, name, number, negation and question. Add nothing: no glosses, notes or brackets, and no Hindi or Urdu words.

STYLE (formal; only when the message says "style": "formal")
- Clear standard Telugu for a general audience, the way a news reader or a lecturer says it: complete sentences, మీరు throughout, no slang.
- A Telugu word wherever a common one exists (పరిశోధన, not రీసెర్చ్); English only for names, brands and technical terms that have no common Telugu word.
- Still spoken Telugu, never grandhika. The same faithfulness rules: every fact, name, number, negation and question, nothing added."""

SCRIPT = """SCRIPT CONTRACT
- Every "spoken" field is Telugu script only: no Latin letters, no digits (Telugu digits included), no zero-width joiners or non-joiners, no brackets, and no symbols such as % or $. Punctuation: . , ? ! ; : … - and quotes.
- Write each English word the way Telugu speakers write it in Telugu script (సెట్టింగ్స్, నెట్ఫ్లిక్స్). A Telugu ending may join it (నెట్ఫ్లిక్స్లో) or follow as its own word (ఫ్రెండ్ కి).
- List every English word, and every name or brand normally written in Latin script, in "english": "i" is the 0-based index of its word in "spoken" split on spaces (punctuation stays with its word), and "en" is that word in Latin script, as the Telugu line says it (ఎంజాయ్ → enjoy).
- Words Telugu has fully absorbed (బస్సు, సినిమా, గ్లాసు) and Indian names (హైదరాబాద్) are Telugu words: don't list them.
- Never write Telugu words in Latin script."""

LENGTH = """LENGTH CONTRACT
- Each line's "target_aksharas" is computed locally from the speaker's speech time and the voice's pace. Use it only to judge how much the shorter wordings must cut; don't count aksharas yourself.
- "full" is the natural, complete line, with no length pressure. It is always given.
- Give the other wordings only when "want" names them.
- "concise" and "very_concise" say the same thing in fewer aksharas. They cut only fillers, hedges, repetitions and pronouns the verb marks, and use shorter spoken forms; every fact, name, number, negation and question stays. "concise" is clearly shorter than "full", and "very_concise" clearly shorter than "concise".
- "fuller" restores what the speaker actually said that "full" smoothed away: hedges, repetitions, discourse markers, full verb forms, pronouns. It adds no fact and no filler the speaker didn't say, and is longer than "full"."""

CONTRACT = """JSON CONTRACT
The message is JSON; "call" says what to do.
- "scene": dub every line in "lines". "context_before" (earlier lines, with the Telugu used for them when there is one), "context_done" (lines of this scene already dubbed) and "context_after_en" (the lines that follow) are context only: never dub them and never return their ids.
- "fit": each line's "current" is the wording chosen for it, and "overflow_aksharas" is how far it runs over its slot (negative: under). Give new wordings, closer to the slot, for the tiers in "want". A fit never changes the natural line: "full" is "current" copied unchanged.
- "retranslate": a review found content missing, and "missing" names the English words. Translate the English afresh so their meaning is there; don't rework any earlier Telugu.
- "rephrase": the Telugu voice couldn't say "failing" cleanly, and "finding" says what went wrong. Give a simpler "full" with the same meaning (plainer words, another common form of a hard term) and the tiers in "want".
- "problems" on a line lists what was wrong with your last answer for it: fix exactly that.
- "glossary_recent" holds spellings decided in earlier scenes: use them.
Reply with one JSON object: {"lines": [...], "glossary_additions": [...]}.
- Every requested id exactly once, in any order.
- "full" always; other tiers only when "want" names them.
- "pieces" only for a line with "breaks": at most one piece more than there are breaks, split where Telugu naturally pauses. Restructure so each piece can be said on its own; the pieces joined with single spaces must be exactly full.spoken. Content that had to move to another piece is listed, in English, in "moved".
- "unfinished": true for a "cut_off" line: keep it unfinished (a non-finite form ending in …), never complete the thought.
- "delivery": how to say it: "emotion" and "energy", "question": true for a question, and "emphasis": the indices of stressed words in full.spoken.
- "glossary_additions": names and terms first used in this reply that later scenes must spell the same way, with the Telugu-script spelling used ("spoken") and "keep_english": true when it is said in English."""


def _w(spoken: str, *english: tuple[int, str]) -> dict:
    return {"spoken": spoken, "english": [{"i": i, "en": en} for i, en in english]}


def _d(emotion: str = "neutral", energy: str = "mid", question: bool = False, *emphasis: int) -> dict:
    out: dict = {"emotion": emotion, "energy": energy}
    if question:
        out["question"] = True
    if emphasis:
        out["emphasis"] = list(emphasis)
    return out


# Original examples written for Maata (none from any video), from no English at all to technical lines; English words
# are spelled the way Telugu speakers write them and listed in the english map. Each is (English, fields of the input
# line besides id, want and en, the reply line without its id). Tests check every one against the validators.
# MAINTAINER REVIEW: please read these as a native speaker (word choice, loan spellings, verb forms, the formal pair)
# before the step-1 bake-off; SHOTS_VERSION goes up with any change.
SHOTS: list[tuple[str, dict, dict]] = [
    ("My father worked two jobs so that we could go to school.", {},
     {"full": _w("మేము బడికి వెళ్ళాలని మా నాన్న రెండు ఉద్యోగాలు చేశారు."), "delivery": _d("serious")}),
    ("So, have you ever noticed how time just flies when you're, like, really enjoying something?", {},
     {"full": _w("అసలు ఎప్పుడైనా గమనించారా, అంటే ఏదైనా బాగా ఎంజాయ్ చేస్తున్నప్పుడు టైం ఎంత తొందరగా గడిచిపోతుందో?",
                 (6, "enjoy"), (8, "time")),
      "delivery": _d("happy", "mid", True)}),
    ("Bro, you promised you'd come. I waited for an hour!", {},
     {"full": _w("రేయ్, వస్తానని మాటిచ్చావ్ కదా. గంటసేపు ఎదురుచూశా!"), "delivery": _d("angry", "high")}),
    ("Open the settings, turn off Wi-Fi, and restart the router.", {},
     {"full": _w("సెట్టింగ్స్ ఓపెన్ చేసి, వైఫై ఆఫ్ చేసి, రౌటర్ని రీస్టార్ట్ చేయండి.",
                 (0, "settings"), (1, "open"), (3, "Wi-Fi"), (4, "off"), (6, "router"), (7, "restart")),
      "delivery": _d()}),
    ("So basically what happened was, the bus was late, and then it started raining, and by the time I got there, the "
     "shop had already closed.", {},
     {"full": _w("అంటే ఏమైందంటే, బస్సు లేట్ అయింది, తర్వాత వాన మొదలైంది, నేను అక్కడికి చేరేసరికి షాప్ అప్పటికే మూసేశారు.",
                 (3, "late"), (11, "shop")),
      "concise": _w("బస్సు లేట్ అయింది, వాన కూడా మొదలైంది, నేను చేరేసరికి షాప్ మూసేశారు.", (1, "late"), (8, "shop")),
      "very_concise": _w("బస్సు లేట్, పైగా వాన, వెళ్ళేసరికి షాప్ మూసేశారు.", (1, "late"), (5, "shop")),
      "delivery": _d("sad")}),
    ("Prices went up from 40 rupees to 65 in just two years. That's more than 60 percent.", {},
     {"full": _w("కేవలం రెండేళ్ళలో ధర నలభై రూపాయల నుంచి అరవై ఐదుకి పెరిగింది. అంటే అరవై శాతం కంటే ఎక్కువ."),
      "delivery": _d("surprised", "mid", False, 10, 11)}),
    ("AI models need a GPU with a lot of memory, which is why they're so expensive to run.", {},
     {"full": _w("ఏఐ మోడల్స్ కి చాలా మెమరీ ఉన్న జీపీయూ కావాలి, అందుకే వాటిని రన్ చేయడానికి అంత ఖర్చవుతుంది.",
                 (0, "AI"), (1, "models"), (4, "memory"), (6, "GPU"), (10, "run")),
      "delivery": _d()}),
    ("I never said it would be easy. I said it would be worth it.", {},
     {"full": _w("ఈజీగా ఉంటుందని నేనెప్పుడూ చెప్పలేదు. వర్త్ అవుతుందని చెప్పా.", (0, "easy"), (4, "worth")),
      "delivery": _d("serious", "mid", False, 2, 4)}),
    ("Don't worry, it's not rocket science.", {},
     {"full": _w("కంగారు పడకండి, ఇదేమీ బ్రహ్మవిద్య కాదు."), "delivery": _d("happy")}),
    ("If this video helped you, share it with a friend who needs it.", {},
     {"full": _w("ఈ వీడియో మీకు ఉపయోగపడితే, ఇది అవసరమైన ఫ్రెండ్ కి షేర్ చేయండి.", (1, "video"), (6, "friend"), (8, "share")),
      "delivery": _d("happy")}),
    ("Once the tests pass, push your code to GitHub and open a pull request.", {},
     {"full": _w("టెస్టులు పాస్ అయ్యాక, కోడ్ని గిట్హబ్ లోకి పుష్ చేసి, పుల్ రిక్వెస్ట్ ఓపెన్ చేయండి.",
                 (0, "tests"), (1, "pass"), (3, "code"), (4, "GitHub"), (6, "push"), (8, "pull"), (9, "request"),
                 (10, "open")),
      "delivery": _d()}),
    ("Yeah, I mean, it's fine. It works.", {},
     {"full": _w("అవును, బాగానే ఉంది. పని చేస్తుంది."),
      "fuller": _w("అవును, అంటే, బాగానే ఉంది. అది పని చేస్తుంది."),
      "delivery": _d("neutral", "low")}),
    ("So what I was trying to say is that the", {"cut_off": True},
     {"full": _w("అంటే, నేను చెప్పాలనుకున్నది ఏంటంటే…"), "unfinished": True, "delivery": _d()}),
    ("Your brain is only about two percent of your body weight, but it uses around twenty percent of its energy.", {},
     {"full": _w("మన మెదడు శరీర బరువులో దాదాపు రెండు శాతమే ఉంటుంది, కానీ మొత్తం ఎనర్జీలో ఇరవై శాతం దాకా అదే వాడుకుంటుంది.",
                 (10, "energy")),
      "delivery": _d("surprised")}),
    ("When I moved to Hyderabad for my first job, I didn't know anyone, and honestly, it was really lonely.",
     {"start": 11.2, "end": 18.9, "breaks": [15.3]},
     {"full": _w("ఫస్ట్ జాబ్ కోసం హైదరాబాద్ వచ్చినప్పుడు నాకు ఎవరూ తెలీదు, నిజం చెప్పాలంటే చాలా ఒంటరిగా అనిపించింది.",
                 (0, "first"), (1, "job")),
      "pieces": ["ఫస్ట్ జాబ్ కోసం హైదరాబాద్ వచ్చినప్పుడు నాకు ఎవరూ తెలీదు,", "నిజం చెప్పాలంటే చాలా ఒంటరిగా అనిపించింది."],
      "delivery": _d("sad", "low")}),
    ("Researchers found that people who slept less than six hours made more mistakes the next day.", {},
     {"full": _w("ఆరు గంటల కంటే తక్కువ పడుకున్నవాళ్ళు మర్నాడు ఎక్కువ తప్పులు చేశారని రీసెర్చ్ లో తేలింది.", (9, "research")),
      "delivery": _d()}),
    ("Researchers found that people who slept less than six hours made more mistakes the next day.", {"style": "formal"},
     {"full": _w("ఆరు గంటల కంటే తక్కువ నిద్రపోయినవారు మరుసటి రోజు ఎక్కువ తప్పులు చేశారని పరిశోధకులు కనుగొన్నారు."),
      "delivery": _d("serious")}),
    ("Did you watch the new Marvel movie on Netflix last weekend?", {},
     {"full": _w("పోయిన వీకెండ్ నెట్ఫ్లిక్స్లో కొత్త మార్వెల్ సినిమా చూశారా?", (1, "weekend"), (2, "Netflix"), (4, "Marvel")),
      "delivery": _d("happy", "high", True)}),
]

_TIER_ORDER = ("full", "fuller", "concise", "very_concise")  # the order `want` lists them in the examples


def _json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def shot_lines() -> list[tuple[dict, dict, str]]:
    """The examples as (input line, reply line, style), numbered from 1."""
    out = []
    for k, (en, extra, reply) in enumerate(SHOTS, 1):
        fields = {f: v for f, v in extra.items() if f != "style"}
        line = {"id": k, **fields, "want": [t for t in _TIER_ORDER if t in reply], "en": en}
        out.append((line, {"id": k, **reply}, extra.get("style", "colloquial")))
    return out


def _examples() -> str:
    parts = ["EXAMPLES (each: one line of \"lines\", then that line of the reply)"]
    for line, reply, style in shot_lines():
        parts.append(("in (formal style): " if style == "formal" else "in:  ") + _json(line))
        parts.append("out: " + _json(reply))
    return "\n".join(parts)


SYSTEM_HEAD = "\n\n".join([ROLE, STYLE, SCRIPT, LENGTH, _examples()])
SYSTEM_TAIL = CONTRACT

# ---- schemas (draft-07, no `format`) -------------------------------------------------------------------------------
_WORDING = {"type": "object", "additionalProperties": False, "required": ["spoken", "english"],
            "properties": {"spoken": {"type": "string", "minLength": 1},
                           "english": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False, "required": ["i", "en"],
                               "properties": {"i": {"type": "integer", "minimum": 0}, "en": {"type": "string"}}}}}}

# §4.4, shared by the scene, fit, re-translate and rephrase calls (one schema, one prompt: one cache per model).
SCENE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["lines"],
    "definitions": {"wording": _WORDING},
    "properties": {
        "lines": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "full", "delivery"],
            "properties": {
                "id": {"type": "integer"},
                "full": {"$ref": "#/definitions/wording"},
                "fuller": {"$ref": "#/definitions/wording"},
                "concise": {"$ref": "#/definitions/wording"},
                "very_concise": {"$ref": "#/definitions/wording"},
                "pieces": {"type": "array", "items": {"type": "string"}},
                "moved": {"type": "array", "items": {"type": "string"}},
                "unfinished": {"type": "boolean"},
                "delivery": {"type": "object", "additionalProperties": False, "required": ["emotion", "energy"],
                             "properties": {"emotion": {"enum": list(EMOTIONS)}, "energy": {"enum": list(ENERGIES)},
                                            "question": {"type": "boolean"},
                                            "emphasis": {"type": "array", "items": {"type": "integer"}}}}}}},
        "glossary_additions": {"type": "array", "items": {
            "type": "object", "required": ["term", "spoken"],
            "properties": {"term": {"type": "string"}, "spoken": {"type": "string"},
                           "keep_english": {"type": "boolean"}}}}}}


BRIEF_HEADER = ("VIDEO BRIEF (version {}). Follow its glossary spellings, each speaker's gender (for verb agreement) and "
                "address forms, and its numbers convention. Apply its ASR fixes only to the words listed.")


def system_prompt(brief: Brief) -> str:
    """The scene system prompt for this brief: byte-identical for equal briefs."""
    return "\n\n".join([SYSTEM_HEAD, brief_block(brief), SYSTEM_TAIL])


def brief_block(brief: Brief) -> str:
    return BRIEF_HEADER.format(brief.version) + "\n" + _json(brief_dict(brief))


def brief_dict(brief: Brief) -> dict:
    m = brief.meta
    video: dict = {"title": m.title, "channel": m.channel}
    if m.description:
        video["description"] = m.description[:DESCRIPTION_MAX]
    if m.chapters:
        video["chapters"] = [{"start": round(t, 1), "title": s} for t, s in m.chapters]
    if m.tags:
        video["tags"] = list(m.tags)
    if m.talk_shares:
        video["speakers"] = [{"id": s, "talk_share": round(x, 2)} for s, x in m.talk_shares]
    out: dict = {"video": video}
    if brief.version < 1:
        return out
    made = {
        "topic": brief.topic, "register": brief.register,
        "speakers": [{"id": s.id, "name": s.name, "gender": s.gender, "role": s.role, "audience": s.audience,
                      "address": [{"to": to, "form": form} for to, form in s.address]} for s in brief.speakers],
        "glossary": [_glossary_item(g, note=True) for g in brief.glossary],
        "entities": list(brief.entities), "idioms": list(brief.idioms), "numbers": brief.numbers,
        "asr_fixes": [{"heard": h, "meant": m} for h, m in brief.asr_fixes],
    }
    out.update({k: v for k, v in made.items() if v})
    return out


def _glossary_item(g: GlossaryEntry, note: bool = False) -> dict:
    item: dict = {"term": g.term, "spoken": g.spoken, "keep_english": g.keep_english}
    if note and g.note:
        item["note"] = g.note
    return item


def _line(s: LineSpec, call: str) -> dict:
    d: dict = {"id": s.id, "speaker": s.speaker}
    if s.to:
        d["to"] = s.to
    d.update(start=round(s.start, 2), end=round(s.end, 2), speech_s=round(s.speech_s, 2),
             target_aksharas=round(s.target_aksharas), want=list(s.want), breaks=[round(b, 2) for b in s.breaks],
             cut_off=s.cut_off)
    if s.delivery_hint:
        d["delivery_hint"] = {k: round(v, 2) for k, v in s.delivery_hint.items()}
    d["en"] = s.en
    if call == "fit":
        d["current"] = s.current
        d["overflow_aksharas"] = None if s.overflow is None else round(s.overflow)
    elif call == "retranslate":
        d["missing"] = list(s.missing)
    elif call == "rephrase":
        d["failing"] = s.current
        d["finding"] = s.finding or {}
    if s.problems:
        d["problems"] = list(s.problems)
    return d


def user_message(req: SceneRequest, style: str = "colloquial", glossary_recent: Sequence[GlossaryEntry] = (),
                 done: Sequence[tuple[LineSpec, str]] = ()) -> str:
    """The user message of §4.3 for one call. `done` holds the scene's lines already translated (from the line cache or
    an earlier attempt) with their Telugu: context only, like `context_before`."""
    msg: dict = {"scene": req.scene, "call": req.call, "style": style,
                 "glossary_recent": [{"term": g.term, "spoken": g.spoken} for g in glossary_recent],
                 "context_before": [{"en": en, "te": te} if te else {"en": en} for en, te in req.context_before]}
    if done:
        msg["context_done"] = [{"id": s.id, "start": round(s.start, 2), "en": s.en, "te": te} for s, te in done]
    msg["context_after_en"] = list(req.context_after_en)
    msg["lines"] = [_line(s, req.call) for s in req.lines]
    return _json(msg)


# ---- the brief call ------------------------------------------------------------------------------------------------
BRIEF_SYSTEM = f"""You prepare the brief that guides dubbing an English YouTube video into spoken Telugu. The brief is read before every line is dubbed, so keep it short and certain.
The message is JSON: the video's metadata, each speaker's share of talk time, and the English transcript so far as {{"speaker", "en"}} lines. "previous", when present, is the brief so far: then return only what is new or has changed, and leave the rest out (an empty string or list means no change).
Reply with one JSON object:
- "topic": one sentence.
- "register": how formal the video is and how the speakers talk to the audience and to each other.
- "speakers": per speaker id, "name" (only if said), "gender" (male, female or unknown; Telugu verbs agree with it), "role", "audience" (polite for మీరు, familiar for నువ్వు) and "address" ({{"to", "form"}} for each other speaker they talk to).
- "glossary": names, brands and terms that recur or could be spelled two ways. "term" as the English has it; "keep_english": true when Telugu speakers say it in English, and then "spoken" is how they write it in Telugu script; otherwise "spoken" is the fixed Telugu rendering. "spoken" is Telugu script only: no Latin, digits or zero-width characters. "note": how to say it, if not obvious.
- "entities": other names of people, places and organisations.
- "idioms": idioms, puns and jokes in the English, quoted, that need a Telugu equivalent.
- "numbers": the convention for numbers, money and years (lakhs and crores or millions, and how years are said).
- "asr_fixes": only transcription errors you are sure of, as {{"heard", "meant"}}; never a guess."""

BRIEF_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["topic", "register", "speakers", "glossary"],
    "properties": {
        "topic": {"type": "string"},
        "register": {"type": "string"},
        "speakers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id"],
            "properties": {"id": {"type": "string"}, "name": {"type": "string"},
                           "gender": {"enum": ["male", "female", "unknown"]}, "role": {"type": "string"},
                           "audience": {"enum": ["polite", "familiar"]},
                           "address": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False, "required": ["to", "form"],
                               "properties": {"to": {"type": "string"},
                                              "form": {"enum": ["polite", "familiar"]}}}}}}},
        "glossary": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["term", "spoken", "keep_english"],
            "properties": {"term": {"type": "string", "minLength": 1}, "spoken": {"type": "string", "minLength": 1},
                           "keep_english": {"type": "boolean"}, "note": {"type": "string"}}}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "idioms": {"type": "array", "items": {"type": "string"}},
        "numbers": {"type": "string"},
        "asr_fixes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["heard", "meant"],
            "properties": {"heard": {"type": "string"}, "meant": {"type": "string"}}}}}}


def brief_message(meta: VideoMeta, transcript: Sequence[tuple[str, str]], previous: Brief | None = None) -> str:
    """The brief call's message: the video's metadata, the (speaker, English) transcript and the brief so far."""
    msg: dict = {"call": "brief", **brief_dict(Brief(0, meta)),
                 "transcript": [{"speaker": s, "en": en} for s, en in transcript]}
    if previous is not None and previous.version >= 1:
        msg["previous"] = {k: v for k, v in brief_dict(previous).items() if k != "video"}
    return _json(msg)


# ---- the coverage review (§4.2, §4.6) ------------------------------------------------------------------------------
# Its own system prompt and schema, byte-identical for every video (no brief: it judges meaning, not spellings), so its
# calls share one prompt cache on the review model.
REVIEW_SYSTEM = """You check a Telugu dub against its English, line by line, for meaning only: never style, word choice, spelling or length.
The message is JSON. "lines" holds the lines to check, each {"id", "en", "te"}: the English and the Telugu said for it. "context_before" (earlier lines, with their Telugu when there is one) and "context_after_en" (the lines that follow) are context only: never return them.
The Telugu is spoken Telugu in Telugu script, and the English words it keeps are spelled in Telugu script too (సెట్టింగ్స్ is "settings"). A line with "cut_off": true was interrupted in the English, and its Telugu is unfinished on purpose.
Reply with every id in "lines" exactly once, with "class":
- "C": complete: every fact, name, number, negation and question of the English is there, and nothing is added;
- "m": a minor drop only: an intensifier, a hedge, a filler or discourse marker, "okay", a trailing backchannel;
- "P": a content phrase or clause of the English is missing;
- "E": a meaning error: a negation lost or added, a number, name or question changed, or content that isn't in the English.
With it: "missing", the English words (as the English has them) whose meaning is not in the Telugu, empty for "C"; "added", Telugu content that isn't in the English, said in English; "error", the kind of an "E" error, else "none"."""

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["lines"],
    "properties": {"lines": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "class", "missing", "added", "error"],
        "properties": {"id": {"type": "integer"}, "class": {"enum": ["C", "m", "P", "E"]},
                       "missing": {"type": "array", "items": {"type": "string"}},
                       "added": {"type": "array", "items": {"type": "string"}},
                       "error": {"enum": ["none", "negation", "number", "name", "question", "addition", "other"]}}}}}}


def review_message(req: SceneRequest, lines: Sequence[tuple[LineSpec, Wording]]) -> str:
    """The review call's message: each line's English and the Telugu chosen for it, with the scene's context."""
    msg: dict = {"scene": req.scene, "call": "review",
                 "context_before": [{"en": en, "te": te} if te else {"en": en} for en, te in req.context_before],
                 "lines": [{"id": s.id, "en": s.en, "te": w.spoken, **({"cut_off": True} if s.cut_off else {})}
                           for s, w in lines],
                 "context_after_en": list(req.context_after_en)}
    return _json(msg)


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def prompt_hash() -> str:
    """A short hash of everything fixed that decides a scene line: the system prompt around the brief, the examples,
    the schema and the user-message layout. The brief is left out on purpose: lines made under brief v1 stay valid
    under v2 (§4.7)."""
    probe = SceneRequest(0, (LineSpec(0, "S1", "<en>", 0.0, 1.0, 1.0, 5.0, ("full", "concise"), "S2", (0.5,), True,
                                      {"energy_rel": 1.0}, "<current>", 1.0, ("<missing>",), {"cer": 0.0},
                                      ("<problem>",)),),
                         "scene", (("<en>", "<te>"), ("<en>", None)), ("<next>",))
    layouts = [user_message(SceneRequest(0, probe.lines, call, probe.context_before, probe.context_after_en), "formal",
                            [GlossaryEntry("<term>", "<spoken>")], [(probe.lines[0], "<te>")])
               for call in ("scene", "fit", "retranslate", "rephrase")]
    return _digest(SHOTS_VERSION, SYSTEM_HEAD, BRIEF_HEADER, SYSTEM_TAIL, SCENE_SCHEMA, layouts)


def brief_hash() -> str:
    return _digest(BRIEF_SYSTEM, BRIEF_SCHEMA)


def review_hash() -> str:
    probe = SceneRequest(0, (), "review", (("<en>", "<te>"), ("<en>", None)), ("<next>",))
    layout = review_message(probe, [(LineSpec(0, "S1", "<en>", 0.0, 1.0, 1.0, 5.0, cut_off=True), Wording("<te>"))])
    return _digest(REVIEW_SYSTEM, REVIEW_SCHEMA, layout)


PROMPT_HASH = prompt_hash()
BRIEF_HASH = brief_hash()
REVIEW_HASH = review_hash()
