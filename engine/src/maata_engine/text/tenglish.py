"""Code-mixed Telugu text: the English-word lint, the Latin rebuild of a Telugu-script line, and code-mixing measures.

A dub line is Telugu script plus a map of its English words (the scene contract, ARCHITECTURE §4.3). `latin_spoken`
rebuilds the line with those words in Latin script: the TTS input of the maintainer's Latin A/B (decision D6). `lint`
knows a line's English words from the same map, never from Latin letters (ARCHITECTURE §3.6), so its checks for English
said where Telugu speakers wouldn't (grammar words, inflected verbs, untranslated phrases) read what each English word
is. `latin_ratio` and `cmi` are logged measures only, never targets: no amount of English is ever asked for. Pure Python.
Every example sentence in the tests is original, written for Maata.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from .akshara import count_telugu, count_units

# Bump whenever a lint rule changes; logged with the lint's flags.
LINT_VERSION = "lint-v3"

_L = "A-Za-zÀ-ÖØ-öø-ɏ"  # Latin letters, accented ones included (Pokémon, José)
_LATIN = re.compile(rf"[{_L}][{_L}'’-]*")
_TELUGU = re.compile(r"[ఀ-౿]+")


def _contains(term: str, text: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I))


# Letters of another script are a decoding glitch (a stray Georgian or Devanagari letter, a CJK character or
# fullwidth punctuation), and so are control, invisible and private-use characters and emoji. Kept: Telugu with
# ZWNJ/ZWJ, ASCII, Latin with its accents (Pokémon, José), typographic punctuation, and currency, letterlike and
# math symbols (€, £, °, ℃, ½, −), which carry numbers and names the line must keep.
_STRAY = re.compile(r"[^\u0c00-\u0c7f\u200c\u200d\x20-\x7e\u00a0-\u036f\u1e00-\u1eff\u2010-\u2027\u2030-\u205e"
                    r"\u20a0-\u22ff\s]")


def latin_ratio(text: str) -> float:
    """Share of words written in Latin script, over Latin + Telugu words. A logged measure, never a target."""
    lat = len(_LATIN.findall(text))
    tel = len(_TELUGU.findall(text))
    return lat / (lat + tel) if lat + tel else 0.0


# ---- the English-word map -------------------------------------------------------------------------------
# Case endings and postpositions a line may join to an English word (నెట్ఫ్లిక్స్లో, రౌటర్ని). The Latin rebuild says
# them as a word of their own after the English (Netflix లో), as Latin-script Tenglish writes them. Longest first.
_ENDINGS = ("లోని", "నుంచి", "వల్ల", "లో", "కి", "కు", "ని", "ను", "తో", "గా")


def _latin_word(word: str, en: str) -> str:
    i, j = 0, len(word)
    while i < j and unicodedata.category(word[i])[0] == "P":  # quotes and punctuation around the word stay
        i += 1
    while j > i and unicodedata.category(word[j - 1])[0] == "P":
        j -= 1
    lead, core, trail = word[:i], word[i:j], word[j:]
    for end in _ENDINGS:  # a stem of at least 1.5 aksharas, so హలో or ఫ్లో keep their last syllable
        if core.endswith(end) and count_telugu(core[: -len(end)]) >= 1.5:
            return f"{lead}{en} {end}{trail}"
    return f"{lead}{en}{trail}"


def latin_spoken(spoken: str, english: Iterable[tuple[int, str]]) -> str:
    """`spoken` with each word the English map lists written as its Latin form, punctuation kept. An index past the
    line's words is ignored (the validators drop those)."""
    words = spoken.split()
    for i, en in english:
        if 0 <= i < len(words) and en.strip():
            words[i] = _latin_word(words[i], en.strip())
    return " ".join(words)


def cmi(lines: Iterable[tuple[str, Sequence[tuple[int, str]]]]) -> float:
    """Code-Mixing Index (Das and Gambäck) over Telugu-script lines and their English maps, in percent: 100 x the
    minority language's share of the words. Logged per scene, never enforced (ARCHITECTURE §4.5)."""
    words = english = 0
    for spoken, mapped in lines:
        words += len(spoken.split())
        english += len({i for i, _ in mapped})
    return 100.0 * min(english, words - english) / words if words else 0.0


# ---- lint -----------------------------------------------------------------------------------------------
# Heuristic checks for the ways a spoken line goes wrong, run on the Telugu-script line and its English map. A flag is
# logged with the line (units.jsonl) and never blocks it. Word lists are kept small. "Too short" says little on its own:
# a short reply said slowly is fine.

# Set phrases Telugu speakers say in English as they are.
ALLOWED_PHRASES = (
    "by the way", "of course", "at least", "no problem", "thank you", "all the best", "excuse me", "for example",
    "in fact", "in case", "just in case", "as usual", "step by step", "one by one", "day by day", "on time",
    "on the way", "on the spot", "oh my god", "trust me", "believe me", "work from home", "will power", "must watch",
    "before and after", "at the end of the day", "as soon as possible", "a to z",
)

_GRAMMAR = frozenset("""
    a an the
    of in on at to for with from by about into onto under after before between through during without within
    against among upon towards toward across behind beyond than
    is are was were be been being am do does did have has had will would shall should can could may might must
    i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its itself
    we us our ours ourselves they them their theirs themselves
    this that these those what which who whom whose when where why how
    some any many much more most less few fewer all every each both several lot lots another
    and or because if although though unless whether while not
""".split())
_PARTICLES = frozenset({"in", "on"})  # as part of a phrasal-verb stem before చేయు/అవు: log in చేయండి

# Everyday verbs Telugu speakers say in Telugu (architecture §5), with their common English forms.
_BASIC_VERBS = frozenset("""
    think thinks thought understand understands understood see sees saw seen eat eats ate eaten drink drinks drank
    go goes went gone come comes came learn learns learnt remember remembers forget forgets forgot forgotten
    increase increases decrease decreases change changes know knows knew known say says said tell tells told
    give gives gave given take takes took taken want wants believe believes speak speaks spoke spoken
    sleep sleeps slept cry cries laugh laughs
""".split())

# -ed and -ing words that are fine before చేయు/అవు: states (excited అయ్యా) and nouns (shopping చేశాం, wedding అయింది).
_ED_OK = frozenset("""
    excited bored tired confused shocked surprised married engaged settled frustrated depressed stressed relaxed
    interested disappointed impressed satisfied convinced selected placed qualified scared worried attached addicted
    irritated disturbed exhausted motivated inspired involved focused committed dedicated retired locked
""".split())
_ING_OK = frozenset("""
    shopping meeting training coding programming planning booking parking cooking marketing editing trading banking
    gaming streaming trending shooting dubbing recording dancing singing swimming jogging boxing painting driving
    cycling trekking camping fasting dieting networking branding testing coaching counselling counseling posting
    loading buffering charging processing running pending matching missing screening boring interesting confusing
    amazing exciting shocking surprising annoying inspiring promising outstanding upcoming ongoing existing
    wedding opening ending evening morning building feeling warning ceiling clothing sibling pudding
""".split())
_ING_BASE = frozenset("thing bring sting swing cling fling sling spring string nothing something everything anything".split())
_ED_BASE = frozenset("shed embed sled".split())

# Spoken as English numbers ("ten percent off"), they don't make a phrase untranslated.
_NUMBER_WORDS = frozenset("""
    zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
    eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion lakh
    lakhs crore crores percent point
""".split())
_TITLE_FN = frozenset({"of", "the", "and", "a", "an", "in", "on", "for", "to", "at", "by", "with", "de"})
# Words that open a sentence without opening a name: capitalised there, but "So Google" is not a name.
_STARTERS = frozenset("""
    so but and or also actually basically honestly literally obviously seriously anyway anyways now then well okay ok
    yeah yes no oh hey hi hello look listen see please maybe just even only still again right sure wait today
    yesterday tomorrow tonight here there first second next last finally everyone everybody nobody somebody someone
""".split())
# Capitalised words that are not names or product terms to keep in English.
_NOT_TERMS = frozenset("""
    monday tuesday wednesday thursday friday saturday sunday january february march april june july august
    september october november december god mom mum dad sir madam mr mrs ms dr ok okay hi hello hey oh yes no yeah
""".split())

_BOOKISH = ("యొక్క", "తద్వారా", "మరియు", "కావున")
_LIGHT_VERB = re.compile(r"చే(?:య|స|శ|ద్ద)|చెయ్య|అవు|అవ్వ|అయ్య|అయి|ఐంది|ఐపో")
_CONTRACTION = re.compile(r"(?:n['’]t|['’](?:m|re|ll|ve|d))$", re.I)  # don't, I'm, you're, we'll, I've, she'd
_CLITIC_S = re.compile(r"['’]s$", re.I)
_CLITIC_BASES = _GRAMMAR | {"let", "there", "here"}  # it's, that's, let's, there's
_BRACKETS = re.compile(r"[()\[\]{}]")
_TOKEN = re.compile(rf"[{_L}][{_L}'’-]*|[ఀ-౿\u200c\u200d]+|\d+(?:[.,:]\d+)*|\S")
# Source words keep their digits and inner capitals together (M5, GPT-3.5, iPhone, McDonald's).
_SRC_TOKEN = re.compile(rf"[{_L}][{_L}0-9]*(?:[-'’][{_L}0-9]+|\.\d+)*|\d+(?:[.,:]\d+)*|\S")
_LAT_START = re.compile(rf"[{_L}]")
_SENTENCE_END = frozenset(".?!…")
ECHO = "repeats an earlier line instead of translating this one"
TOO_SHORT = "too short"

Token = tuple[str, str]  # (kind, text); kind is lat, tel, num, punct, en (a mapped English word) or name (masked)
NameWord = tuple[str, bool, bool, bool, int]  # (word, has a capital, can anchor a name, starts a sentence, offset)


def _tokens(text: str) -> list[Token]:
    out = []
    for tok in _TOKEN.findall(text):
        c = tok[0]
        kind = ("lat" if _LAT_START.match(c) else "tel" if "ఀ" <= c <= "౿" or c in "\u200c\u200d"
                else "num" if c.isdigit() else "punct")
        out.append((kind, tok))
    return out


_ALLOWED = [tuple(t for _, t in _tokens(p)) for p in ALLOWED_PHRASES]


def _mapped_tokens(spoken: str, english: Iterable[tuple[int, str]]) -> list[Token]:
    """The line's tokens, each word the English map lists said as its English (kind "en") with a case ending joined to it
    as a Telugu token after it, as `latin_spoken` writes it. Only the map makes a word English: Latin letters written in
    `spoken` (which the validators reject) are left to the script checks."""
    mapped = {i: en.strip() for i, en in english if en.strip()}
    out: list[Token] = []
    for i, word in enumerate(spoken.split()):
        if i in mapped:
            out += [("en" if kind == "lat" else kind, tok) for kind, tok in _tokens(_latin_word(word, mapped[i]))]
        else:
            out += _tokens(word)
    return out


def _grammatical(low: str) -> bool:
    """A grammar word, a contraction (don't, I'm) or a pronoun with 's (it's, that's, let's)."""
    if low in _GRAMMAR or _CONTRACTION.search(low):
        return True
    return bool(_CLITIC_S.search(low)) and low[:-2] in _CLITIC_BASES


def _name_runs(source: str) -> list[list[NameWord]]:
    """Runs of capitalised words, with the small words of titles and numbers between them, in an English line.

    Any word with a capital can anchor a name, except 'I' and a grammar or filler word that opens a sentence
    (But I, So Google, The …); names are the spans between anchors (`_name_spans`).
    """
    runs: list[list[NameWord]] = []
    run: list[NameWord] = []
    initial = True
    for m in _SRC_TOKEN.finditer(source):
        tok = m.group(0)
        low = tok.lower()
        if run and tok[0].isdigit():  # iPhone 17 Pro
            run.append((tok, False, False, False, m.start()))
            continue
        if _LAT_START.match(tok):
            cap = any(c.isupper() for c in tok)
            if (cap or low in _TITLE_FN) and low != "i" and not low.startswith(("i'", "i’")):
                anchor = cap and not (initial and (_grammatical(low) or low in _STARTERS))
                run.append((tok, cap, anchor, initial, m.start()))
            elif run:
                runs.append(run)
                run = []
            initial = False
            continue
        if run:
            runs.append(run)
            run = []
        if tok in _SENTENCE_END:
            initial = True
        elif tok[0].isdigit() or tok in ",;:":
            initial = False
    return runs + [run] if run else runs


def _name_spans(run: Sequence[NameWord]) -> list[tuple[int, int]]:
    """Spans between two anchors that hold at least two capitalised words other than grammar words
    (The Morning Show, Google and Apple; not You Know What)."""
    anchors = [i for i, w in enumerate(run) if w[2]]
    content = [i for i in anchors if not _grammatical(run[i][0].lower())]
    return [(a, b) for a in anchors for b in anchors if b > a and sum(a <= i <= b for i in content) >= 2]


def _source_names(source: str) -> set[tuple[str, ...]]:
    """Title-case names of the source (The Morning Show, Google and Apple, Game of Thrones) and every span between two
    of their capitalised words: the line may keep these as they are (matched in exact case)."""
    names: set[tuple[str, ...]] = set()
    for run in _name_runs(source):
        for a, b in _name_spans(run):
            names.add(tuple(t for _, t in _tokens(" ".join(w[0] for w in run[a:b + 1]))))
    return names


def source_keyterms(text: str) -> list[str]:
    """Names and product terms of an English line, in order, to keep as written (architecture §S4, like ElevenLabs'
    keyterms).

    Title-case names (Google Maps, The Morning Show, Game of Thrones), a capitalised word mid-sentence (Pixel),
    acronyms (GPU) and words with inner capitals or digits (iPhone, M5, GPT-4). A lowercase word other than 'of'
    or 'the' splits a name, so "Google and Apple" gives two. 'I', a lone word that opens a sentence and everyday
    capitalised words (days, months, God, Mom) are not terms; lowercase product names need the glossary.
    """
    found: list[tuple[int, str]] = []
    for run in _name_runs(text):
        parts: list[list[NameWord]] = [[]]
        for w in run:
            if w[1] or w[0][0].isdigit() or w[0].lower() in ("of", "the", "de"):
                parts[-1].append(w)
            elif parts[-1]:
                parts.append([])
        for part in parts:
            if spans := _name_spans(part):
                a, b = min(s for s, _ in spans), max(e for _, e in spans)
                found.append((part[a][4], " ".join(w[0] for w in part[a:b + 1])))
            else:
                found += [(w[4], w[0]) for w in part if w[2] and not w[3] and len(w[0]) >= 2 and not _grammatical(
                    w[0].lower()) and w[0].lower() not in _NOT_TERMS and w[0].lower() not in _BASIC_VERBS]
    for m in _SRC_TOKEN.finditer(text):
        tok = m.group(0)
        if _LAT_START.match(tok) and len(tok) >= 2 and tok.lower() not in _NOT_TERMS and (
                re.fullmatch(r"[A-Z]{2,}s?", tok) or re.search(r"[a-z][A-Z]|\d", tok)):
            found.append((m.start(), tok))
    terms = list(dict.fromkeys(term for _, term in sorted(found)))
    return [t for t in terms if not any(t != other and _contains(t, other) for other in terms)]


def _mask(toks: list[Token], ci: Iterable[tuple[str, ...]], cs: Iterable[tuple[str, ...]]) -> list[Token]:
    """Collapse allowed phrases and key terms (any case) and source names (exact case) into single name tokens."""
    phrases = sorted([(p, False) for p in ci if p] + [(p, True) for p in cs if p], key=lambda x: -len(x[0]))
    out, i = [], 0
    while i < len(toks):
        for ph, exact in phrases:
            span = toks[i:i + len(ph)]
            if tuple(t if exact else t.lower() for _, t in span) == ph and any(k == "en" for k, _ in span):
                out.append(("name", " ".join(t for _, t in span)))
                i += len(ph)
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def _is_light_verb(tok: Token) -> bool:
    return tok[0] == "tel" and bool(_LIGHT_VERB.match(tok[1]))


def _is_grammar(tok: str, nxt: Token, before_lv: bool) -> bool:
    low = tok.lower()
    if _CONTRACTION.search(low):
        return True
    if _CLITIC_S.search(low):  # a name keeps its 's (McDonald's, Domino's); it's, that's and a friend's don't
        return not tok[0].isupper() or low[:-2] in _CLITIC_BASES
    if low not in _GRAMMAR:
        return False
    if low in _PARTICLES and before_lv:
        return False
    if low in ("a", "an") and nxt[0] not in ("en", "name"):  # Plan A, Vitamin A
        return False
    return not (low == "may" and tok[0] == "M")  # the month


def _inflected(low: str) -> bool:
    if low in _ED_OK or low in _ING_OK or low in _NUMBER_WORDS:  # hundred
        return False
    if low.endswith("ing"):
        return len(low) >= 5 and low not in _ING_BASE
    if low.endswith("ed"):
        return len(low) >= 4 and not low.endswith("eed") and low not in _ED_BASE
    return False


def _run_length(run: list[Token], stem_follows: bool) -> int:
    n = sum(1 for kind, tok in run if kind == "name" or tok.lower() not in _NUMBER_WORDS)
    if stem_follows and run and run[-1][0] == "en" and run[-1][1].lower() not in _NUMBER_WORDS:
        n -= 1  # the verb stem of "automatic updates off చేసి"
    return n


def _untranslated(run: list[Token], stem_follows: bool) -> bool:
    """English words in a row that read as an untranslated clause: three or more with a grammar word or an everyday
    verb in them, or four or more of any kind. A noun compound (machine learning engineer) is how the term is said."""
    clause = any(kind == "en" and (_grammatical(tok.lower()) or tok.lower() in _BASIC_VERBS) for kind, tok in run)
    return _run_length(run, stem_follows) >= (3 if clause else 4)


def _words(text: str) -> list[str]:
    return [t.lower() for k, t in _tokens(text) if k in ("lat", "tel", "num")]


def _overlap(a: list[str], b: list[str], n: int = 2) -> float:
    """Share of a's word n-grams that also occur in b."""
    grams = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    other = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(grams & other) / len(grams) if grams else 0.0


def _issues(spoken: str, english: Iterable[tuple[int, str]], source: str, context: Sequence[tuple[str, str] | str],
            target_units: float | None, keyterms: Iterable[str], style: str) -> list[tuple[str, list[str]]]:
    terms = [tuple(t for _, t in _tokens(k)) for k in keyterms]
    # A one-word acronym that reads as a grammar word in lower case (IT, US) only masks itself, in capitals.
    exact = {p for p in terms if len(p) == 1 and p[0].isupper() and p[0].lower() in _GRAMMAR}
    ci = _ALLOWED + [tuple(w.lower() for w in p) for p in terms if p not in exact]
    cs = _source_names(source) | exact | {tuple(t for _, t in _tokens(k)) for k in source_keyterms(source)}
    toks = _mask(_mapped_tokens(spoken, english), ci, cs)
    grammar, inflected, basic, runs = [], [], [], []
    run: list[Token] = []
    for i, tok in enumerate(toks):
        kind, text = tok
        nxt = toks[i + 1] if i + 1 < len(toks) else ("", "")
        before_lv = _is_light_verb(nxt)
        if kind in ("en", "name"):
            run.append(tok)
        else:
            if _untranslated(run, _is_light_verb(tok)):
                runs.append(" ".join(t for _, t in run))
            run = []
        if kind != "en" or (len(text) >= 2 and text.isupper()):  # masked phrases and acronyms (US, IT) pass
            continue
        if _is_grammar(text, nxt, before_lv):
            grammar.append(text)
        elif before_lv and _inflected(text.lower()) and not (text.lower().endswith("ing") and nxt[1].startswith("అయితే")):
            inflected.append(text)  # "reading అయితే" is a topic ("as for reading"), not a verb
        elif before_lv and text.lower() in _BASIC_VERBS:
            basic.append(text)
    if _untranslated(run, False):
        runs.append(" ".join(t for _, t in run))

    issues: list[tuple[str, list[str]]] = [
        ("English grammar words (say these in Telugu)", grammar),
        ("inflected English before చేయు/అవు (use a Telugu verb or the bare stem)", inflected),
        ("everyday verbs said in English (use the Telugu verb)", basic),
        ("English phrase left untranslated", runs),
    ]
    issues = [(label, list(dict.fromkeys(items))) for label, items in issues if items]
    if _BRACKETS.search(spoken):
        issues.append(("brackets or glosses", []))
    if stray := sorted(set(_STRAY.findall(spoken))):
        issues.append(("characters from another script", stray))
    words = [t for k, t in toks if k == "tel"]
    if style != "formal" and (bookish := [w for w in _BOOKISH if w in words]):
        issues.append(("bookish words", bookish))
    mine, src = _words(spoken), _words(source)
    if len(mine) >= 4:
        for item in context:
            ctx_src, ctx_te = ("", item) if isinstance(item, str) else item
            if ctx_te and _overlap(mine, _words(ctx_te)) > 0.6 and not (ctx_src and _overlap(src, _words(ctx_src)) > 0.6):
                issues.append((ECHO, []))
                break
    if target_units and target_units > 0:
        units = count_units(spoken)
        if units > 2 * target_units:
            issues.append((f"too long: {round(units)} aksharas for a target of about {round(target_units)}", []))
        elif units < 0.4 * target_units:
            issues.append((f"{TOO_SHORT}: {round(units)} aksharas for a target of about {round(target_units)}", []))
    return issues


def lint(spoken: str, english: Iterable[tuple[int, str]], source: str, context: Sequence[tuple[str, str] | str] = (),
         target_units: float | None = None, *, keyterms: Iterable[str] = (), style: str = "colloquial") -> list[str]:
    """Flags for a dub line, each naming the offending words; [] when the line looks natural.

    `spoken` is the Telugu-script line and `english` its map of English words (word index, Latin form): the words the
    map lists are the line's English. `context` holds earlier (source, Telugu-script line) pairs or bare lines;
    `keyterms` are names and terms kept as written (the source's own names and product terms are always kept). Its
    length is `spoken` counted in aksharas. Pure: no model, no I/O.
    """
    return [f"{label}: {', '.join(items)}" if items else label
            for label, items in _issues(spoken, list(english), source, context, target_units, keyterms, style)]
