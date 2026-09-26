# Gap 3: the translation and timing unit for verb-final Telugu

Researched 2026-09-24. This builds on `sota_ranking.md` and reports 01, 02, 04, 05, 08, 09 and 10, and corrects them where noted.

Labels used below:
- **DISCLOSED:** a company or its authors state it.
- **MEASURED:** a number from a paper, or from my own run on the M5 Pro.
- **INFERRED:** my reasoning.

**Limits on this pass:**
- WebSearch was exhausted, so papers were found through known IDs and ACL Anthology listings, then read in full from their PDFs.
- The `claude` CLI on this machine is not signed in (`claude auth status` returned `loggedIn: false`), so the Claude-arm A/B could not run (see E8).

---

## Question

What is the right translation and timing unit for verb-final Telugu when an English sentence spans more than one unit?

Four positions conflict:
- **Report 01 (F22):** units are sentence-complete.
- **Reports 02 (R1) and 05 (R3):** merge fragments, then split the Telugu back at clause boundaries.
- **Report 08 (R1):** anchor Telugu phrases to English phrase onsets.

The choice sets three things:
- the Claude call contract;
- whether the segmenter merges across pauses and speaker flips;
- whether the planner lets a sentence's dub drift across an English pause.

## Answer

1. **The translation unit is the complete sentence. The problem in the trace is mostly an ASR bug, not a segmentation-design question.**
   - **The bug:** in the 20:06 session, 17 of the 18 units that end mid-sentence sit inside stretches where Whisper large-v3-turbo emitted lowercase text with no punctuation. That is about 37% of speech time.
   - **How the segmenter fails there:** it relies on punctuation, so it falls back to cutting at `max_len` (12 s), at speaker changes and at the 60 s ASR-chunk boundary.
   - **Replay:** I replayed Maata's exact chunking and segmenter on the podcast. It reproduced the same 18 units.
   - **Fix, measured in the replay:**
     - Copy punctuation and casing from a second, punctuation-prompted Whisper pass onto the unprompted pass's words, keeping every original word and timestamp.
     - Non-final units fall from 18 of 101 (18%) to 2 of 119 (2%). Verified in two runs.
     - The obvious one-line alternative, just passing Whisper a punctuated `initial_prompt`, also fixes punctuation. But it silently dropped 14–23 s of speech in all three runs, so it must not be used on its own.

2. **The Claude contract stays id-keyed, one id per sentence-complete unit** (report 01 R1).
   - A unit carries internal break markers only in the rare case where one sentence has to span a hard timing boundary. Split-back is the exception, not the default.

3. **Timing: the sentence's dub may drift across English pauses inside the sentence** (off-screen tolerance). The only exception is a hard boundary. Hard boundaries are:
   - an internal pause of about 1.0 s or more;
   - a sentence longer than the line cap;
   - a real interruption by the other speaker.

   At a hard boundary, the planner splits the dub at a pause that falls naturally in the **Telugu** (fluency-based), not at the Telugu phrase that carries English phrase k's content (content-based).
   - This amends report 08 R1: anchor the **pauses**, not the **content**.
   - Why: Amazon's aligner chose a fluency-based criterion, ESTsoft's LLM pause placement follows the target language's structure (Korean included), and order-preserving translation into an SOV language costs naturalness (NAIST).

4. **No published dubbing study with an SOV or verb-final target tests split-back.** Several premises of the question need correcting (see Corrections):
   - Sony's EN→HI work is sentence- or utterance-level.
   - Microsoft's LSST translates Korean **into** English.
   - IWSLT 2025 had no dubbing track.
   - The IWSLT 2024 dubbing baselines that did split back at commas scored far below the dubbing system, though that comparison is confounded.
   - The closest evidence comes from simultaneous interpretation into Japanese. Order-preserving output is achievable with sentence splitting, connectives and demonstratives, including by LLMs, but it sounds less natural.

5. **When a split is needed, Telugu can mostly take it, provided the break sits at a clause mark.**
   - This is my annotation of 35 boundaries (E7), not native-speaker verified.
   - 18 of 24 valid comma or clause-mark boundaries in long punctuated sentences keep Telugu order with no restructuring.
   - 9 of 10 of the arbitrary `max_len` cuts that caused the trace fragments would need clefts, re-segmentation or a moved negation.

---

## Why the four reports disagree, and how to reconcile them

| Report | Claim | Verdict |
|---|---|---|
| 01 F22 | Units are sentence-complete | **Refuted in practice.** That is the segmenter's design intent (`segment.py` docstring), but 18 of 98 units in session 20:06, 53 of 135 in 13:44 and 1 of 38 in 18:31 end mid-sentence. 01 read the code, not the trace. |
| 02 R1, 05 R3 | Merge into sentences, then split the Telugu back at clause boundaries | **Merge: confirmed. Split-back as the default: not supported.** Google's patent aligns whole sentences and discloses no split-back of MT output. Split-back is useful only at hard boundaries. |
| 08 R1 | Anchor Telugu phrases to English phrase onsets (pause ≥ 0.3 s) | **Amend.** Keep pause anchoring, but place the breaks where Telugu breaks naturally. Don't expect Telugu phrase k to carry English phrase k's content. On this podcast, pauses of 0.7 s or more occur inside only 3% of sentences. |
| 08 OQ6 | Are early endings and "incomplete" lines caused by dropped content? | **Partly answered.** Translating fragments produced 6 serious meaning errors in 13 true mid-sentence cuts (E3). The trace does not decide how much fragments contribute to early endings. |

---

## Evidence

### E1. Why each fragment was cut, in the trace and in an exact replay (MEASURED)

**Trace source:** `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`, the podcast fixture, with three sessions on 2026-09-24.

| Session | Units | Units whose English ends without . ? ! … | Share of unit time in unpunctuated ASR text |
|---|---|---|---|
| 13:44 (ADR-016 build, older segmenter) | 135 | 53 | 38% (247 of 647 s) |
| 18:31 | 38 | 1 | — |
| 20:06 (current `segment.py`, `merge_fragments`) | 98 | 18 | 37% (234 of 633 s), 35% of words |

"Unpunctuated" means a unit of 5 or more words with under 3 punctuation marks per 100 words. In session 20:06, 17 of the 18 non-final units lie in such text or are under 5 words.

**Chunk boundaries:** taken from the 20:06 engine log (`scratchpad/app5.log`), for example `heard 544-604 s`.

**Classification of the 18 non-final units in session 20:06** (my reading of text, speaker, gap and chunk; unit ids as in the trace):

| Cause | Units | Count |
|---|---|---|
| Not actually a fragment: a complete utterance with its final punctuation missing. Four are questions at a real turn change; one ends before a 4.4 s pause | 39, 40, 68, 81, 84 | 5 |
| `max_len` 12 s cut inside unpunctuated text. All run-on word gaps are 0.0 s, so the "longest pause" tie-break cuts at the last word | 41, 46, 57, 58, 73, 74, 82, 83, 97 | 9 |
| 60 s ASR-chunk boundary with no sentence end in the window. `_sentence_cut` falls back to the last word | 85 | 1 |
| Speaker change mid-sentence: probably a diarization error in the cold open | 1 | 1 |
| Speaker change: a real interruption or trail-off ("…but", then the host; a lone "so", then the host) | 54, 67 | 2 |
| Pause of 1.5 s or more splitting a real sentence | — | 0 |

**Exact replay** (`scratchpad/gap3/resegment.py`):
- **Method:**
  - The same `mlx_whisper` 0.4.3 call as `backends/apple.py`: `condition_on_previous_text=False`, word timestamps.
  - The same 60 s chunks with a 0.3 s lead and a 5 s pad.
  - The same `_sentence_cut`, `fix_words`, `segment()` and `merge_fragments()`.
  - Speaker turns approximated from the 20:06 units.
- **Result:** exactly the same 18 non-final units with the same spans (5.06–8.08, 207.04–219.02, 663.04–674.88, and so on).
- **What that shows:** the fragments are deterministic, and they follow from the ASR output.

### E2. The fix: punctuation, without losing words (MEASURED, M5 Pro, two runs each)

**Test 1, three chunks re-transcribed** (`gap3/asr_punct.py`). Punctuation marks per 100 words:

| Chunk | Engine settings | Punctuated `initial_prompt` (first 30 s window) | Prompt on each 30 s window |
|---|---|---|---|
| 440.8–501.8 s | 1.8 | 15.9 | 20.2 |
| 544.3–608.8 s | 0.0 | 9.9 | 18.5 |

**Why only the first window:** in the locked mlx-whisper 0.4.3, `condition_on_previous_text=False` resets the prompt after the first 30 s window (`transcribe.py:255-261, 534`). So an `initial_prompt` affects only the first window of each call. Test 3's winning method therefore prompts each 30 s window separately.

**Why prompting works:** OpenAI's Whisper prompting guide (Cookbook, added 2023-06-27) shows that Whisper copies the prompt's style, and that unpunctuated or lowercase prompts give unpunctuated or lowercase output.

**Test 2, full replay of 0–675 s:**

| ASR variant | Units | Non-final | Punct./100 words | Words kept vs engine |
|---|---|---|---|---|
| Engine today | 101 | 18 (18%) | 9.7 | — |
| Prompt, first window only | 107 | 3 (3%) | 15.4 | **drops 134.3–148.7 s** (the subscribe appeal) |
| Prompt, each 30 s window | 119 | 3 (3%) | 16.6 | **drops the same passage.** A separate run dropped 130.7–153.8 s |
| **Two-pass transfer**: unprompted words and timestamps, plus casing and punctuation copied from the prompted pass by `difflib` alignment | 119 | **2 (2%)** | 17.1 | all words kept (1,972). Identical in two runs |

- **The two remaining fragments:**
  - the probable diarization flip (unit 1);
  - one `max_len` cut in a long run-on question.
- **Cost:** the prompted pass took 2.3–4.6 s per 65 s chunk on the M5 Pro (the engine pass takes 2.1–3.0 s). That is affordable with a 5–10 min lookahead.
- **Prompt leakage:** none. The prompt text never appeared in any transcript.
- **Sentence ends:** Google's patent also inserts missing sentence ends from gaps and capitalisation (US 2020/0404386, description). That supports repairing sentence ends before translating.

### E3. What translating fragments did to meaning (MEASURED from the trace; the Telugu reading is mine)

Session 20:06's Telugu came from the per-line local translator (ADR-018). ADR-018 had removed the "next line" note, so a fragment was translated without seeing how the sentence ends. The `merge_fragments` docstring records the maintainer's own finding that this made the model invent the rest.

Of the 13 true mid-sentence cuts:

- **6 serious meaning errors:**
  - **Unit 73** ("…I'm not | motivated…"): the Telugu for unit 74 says he thinks he *is* motivated. The negation is lost and the meaning inverted.
  - **Unit 82** ("…could mean a billionaire who | looks like…"): the Telugu adds a negation ("superhuman is *not* about billionaires").
  - **Unit 57:** a statement became a question.
  - **Unit 46:** "minds and | machines" became "only on minds".
  - **Unit 58:** an invented clause.
  - **Unit 1:** the causal link was broken, and "visible" became "close".
- **2 minor errors.**
- **5 acceptable.**
- **Unit 85** ("…people that have | achieved…") shows the other side: the translator split it into two Telugu sentences ("I worked with amazing people. They achieved…"). That is exactly the interpreter's segmentation strategy (E6).

This bears directly on two complaints, "translation very bad" and "sentences incomplete or skipped". The sample is small (13), from one video and one local model. Claude may do better on fragments if it sees the next line, but the right fix is not to create the fragments.

### E4. After the fix: how often a sentence really spans a timing boundary (MEASURED)

Punctuated ASR of 0–680 s (prompt on each window; used only for sentence statistics): 160 sentences.

| Measure | Value |
|---|---|
| Sentence duration | median 2.46 s, p90 8.26 s, max 24.0 s |
| Longer than 12 s | 6 (4%) |
| Longer than 20 s | 1 (1%) |
| Internal pause ≥ 0.3 s | in 22 sentences (14%). 26 pauses, 20 of them at a comma or clause mark |
| Internal pause ≥ 0.7 s (today's `anchor_pause`) | 5 sentences (3%) |
| Internal pause ≥ 1.0 s | 3 sentences (2%) |
| Internal pause ≥ 1.5 s (today's `long_pause`) | 1 sentence (1%) |

**Caveats:**
- Whisper DTW gaps have about 120 ms MAE and compress pauses (`sota_ranking.md`). A forced aligner or VAD would find somewhat more.
- This is one podcast; lectures and vlogs may pause more.

**Consequence (INFERRED):** once punctuation is fixed, the drift-vs-anchor question affects roughly 5–7% of sentences on talk content, not a third of them.

### E5. What published systems do about the unit (DISCLOSED / MEASURED)

**Google / YouTube: US 2020/0404386 A1** (filed 2018-02-26, published 2020-12-24), granted as **US 11,582,527 B2** (2023-02-14). Text re-read 2026-09-24.
- **Claim 7 (MT path):** caption fragments are joined into caption **sentences** by punctuation, the sentences are machine-translated, each translated sentence gets the time interval of its source sentence, and alignment is done per sentence.
- **Claim 5 (human translations exist):** fragment-level timing is used only to estimate intervals, and translated fragments are re-joined into sentences before alignment.
- **Split-back:** no split-back of MT output is disclosed. Fit is handled by audio rate, video rate, and silence between lines.
- **Confirms:** the task's reading.

**Amazon prosodic alignment.**
- **Federico et al. 2020** (arXiv 2001.06785 v3, 2020-02-02; IWSLT 2020):
  - Translates the whole sentence, then finds k breakpoints in the **target** that best match the source's k speech–pause segments, scored by duration match times the plausibility of a break at that point.
  - The authors say they use a fluency-based rather than a content-based criterion (contrast with Öktem 2019).
  - Italian native listeners found the result less fluent (−10.93 MUSHRA with MT changes, report 08 F6).
- **Virkar et al. 2022** (arXiv 2204.02530, 2022-04-06; Interspeech 2022):
  - Adds a cross-lingual semantic-match feature.
  - Relaxes boundaries for off-screen sentences, using whole inter-phrase and inter-sentence gaps.
  - Runs TTS **on the entire sentence**, then force-aligns to get segment durations.
- **Scope:** all EN→FR/IT/DE/ES, so all SVO targets.

**ESTsoft, Won et al., EMNLP 2025 System Demonstrations, pp. 515–521** (2025-11; product perso.ai). **This is the only dubbing system found with a verb-final (Korean) target.**
- Translates sentence triplets as whole sentences, with duration-based length control.
- Then an LLM step places pauses. The authors state that copying source pause positions yields awkward segments because each language has its own structure, so the LLM puts the pause at a suitable boundary in the **translation**, and TTS inserts it there.
- Speech overlap EN→KO 0.898 vs 0.845 for plain GPT-4o. COMET 0.838 vs 0.852.
- The pause step has no separate ablation.
- Synthesis is ElevenLabs.

**Sony Research India, EN→HI (Hindi is SOV).** All three are sentence- or utterance-level. None handles a sentence spanning units, and none splits back.
- Mhaskar et al., NAACL 2024 Findings (arXiv 2403.15469, 2024-03-20): isometric NMT on BPCC and FLoRes sentence pairs and movie subtitle lines.
- Dasare et al., ICASSP 2026 (arXiv 2603.28717): MELD utterances translated by an LLM, F5-TTS, **global** time-stretch per clip.
- DubWise, Interspeech 2024 (arXiv 2406.08802, 2024-06-13): duration control inside TTS per utterance.

**Microsoft LSST** (arXiv 2506.00740 v1, 2025-05-31): the model is **(ES, KO)→EN**. Korean is the *source*, English the target, and the evaluation uses FLEURS single-sentence utterances. It is no evidence about an SOV target.

**IWSLT 2024 Findings §7** (ACL Anthology 2024.iwslt-1.1, August 2024). Dubbing was scored on EN→ZH.
- **The organizers' baselines split back:** they built dubs for three offline ST systems by splitting each sentence and its translation at commas and full stops, aligning the pieces with Vecalign and LASER-2, projecting source timestamps, and synthesising with Polly under a max-duration cap plus silence padding.
- **Result:** speech overlap 0.279–0.304 and human score 3.2–3.5, against 0.698 and 3.9 for the one dubbing system (report 04 F8).
- **Confound:** generic, non-length-controlled translations and a forced duration cap. This is weak evidence against naive split-back, not a clean test.

**IWSLT 2025 Findings** (ACL Anthology 2025.iwslt-1.44):
- Seven tasks: offline, simultaneous (EN→DE/JA/ZH), subtitling, compression, low-resource, Indic (EN↔HI/BN/TA speech-to-text) and instruction-following.
- **No dubbing track**, so it has no evidence on this question.

**Practice:**
- **ElevenLabs** (bring-your-own transcript, fetched 2026-09-24, report 01 F6) advises breaking sentences at any pause of 1 s or more, because segment boundaries set timing.
  - That is consistent with "split only at long pauses".
  - How ElevenLabs then translates a sentence split across segments is not disclosed.
- **pyVideoTrans** (@d63d27ae, 2026-09-23, report 09 A3) forbids merging and translates each block alone. Its Hindi prompt admits the verb-final conflict.
- **VideoLingo** translates LLM-split pieces with neighbouring context (report 09 A2).

**Human dubbing:** Brannon et al., TACL 2023 (arXiv 2212.12137):
- Dubs should respect perceptible pauses within a turn (citing Miggiani 2019).
- Human dubbers break timing rather than change their speaking rate. Mean overlap is 0.658, and 0.662 off-screen (report 08 F3).

### E6. Evidence on verb-final targets from simultaneous interpretation (DISCLOSED)

- **He, Boyd-Graber and Daumé III, NAACL 2016, pp. 971–976** (English↔Japanese interpretation corpus):
  - Interpreters keep source order by *segmentation*: breaking a sentence into smaller sentences joined by conjunctions or relative-clause glue, instead of concatenating chunk translations.
  - They also use *passivisation*.
  - Interpreters agree on segmentation 73.7% of the time.
- **Sakai, Makinae, Kamigaito and Watanabe (NAIST), arXiv 2404.12299 (2024-04-18):**
  - Chunk-wise monotonic translation (CWMT) for EN→JA keeps source chunk order.
  - An LLM (GPT-4) can produce it by chunking at clauses, phrases, relativisers and conjunctions, translating each chunk, and joining them only with demonstratives, conjunctions, punctuation and sentence splits.
  - The authors say plain concatenation of chunk translations is unnatural. Offline translation reorders over long distances to stay natural, while CWMT keeps order at some cost to naturalness.
  - SiMT models trained on this data kept offline-level BLEU/COMET at lower latency. This is a model metric, not a listening test.
- **Telugu typology** (WALS Online, Dryer & Haspelmath eds., CC-BY-4.0, datapoints fetched 2026-09-24, source Lisker 1963):
  - order of subject, object and verb: **SOV**;
  - relative clause: **before the noun**;
  - adverbial subordinator: **Mixed**.
- **INFERRED:** "Mixed" fits spoken Telugu's clause-initial connectives (ఎందుకంటే, అయితే, అంటే) next to suffixal ones (-తే, -ప్పుడు). The initial connectives are what make order-preserving splits possible.

### E7. How each split boundary behaves in Telugu (INFERRED; my annotation, not native-verified)

**Items:** 26 items and 35 slot boundaries (`gap3/items.json`):
- the 8 trace groups where a sentence crosses units: 10 boundaries at the engine's real cuts;
- the 18 punctuated sentences longer than 8 s, cut at clause marks near equal time shares (as a clause-boundary split-back or phrase-anchor planner would): 25 boundaries.

**Rubric:** is Telugu order preserved at the boundary without restructuring?
- **Yes:** between sentences or list items; after a fronted "if" or "when" clause; subject | predicate; a trailing clause introduced by అంటే.
- **Restructuring needed:**
  - verb | object ("…study | how we can…") needs a cleft (చేసేది … ఏంటంటే);
  - noun | relative clause ("…people that have | achieved…") needs a new sentence with a demonstrative;
  - negation | predicate ("…I'm not | motivated") means moving the negated verb;
  - main | purpose clause needs a connective (అప్పుడే, దాని వల్ల).

| Boundary source | Order-compatible | Needs restructuring | Invalid |
|---|---|---|---|
| Engine `max_len`/chunk cuts in unpunctuated text (10) | 1 | 9 (2 clefts, 3 segmentation/apposition, 3 negation or predicate moves, 1 verb move) | 0 |
| Clause-mark cuts in punctuated long sentences (25) | 18 | 6 | 1 (inside a name: "Elon \| Musk") |

- **Reading (INFERRED):** split-back into Telugu is cheap when the cut sits at a comma or clause mark, and costly when it sits at an arbitrary word. With punctuation fixed, the segmenter only cuts at marks, so the two designs converge.
- **Needs checking:** some rescues rely on spoken Telugu's post-verbal afterthoughts (right-dislocation). I could not source that for Telugu; the maintainer should check it as a native speaker.

### E8. The blind A/B that was asked for: not run, harness ready

- **What blocked it:** the Claude CLI here is not signed in (`loggedIn: false`, CLI 2.1.201). Signing in is the maintainer's action. A listening A/B also needs his ears.
- **What is ready:**
  - `gap3/run_arms.py {U|A|B}` runs three contracts on the 26 items through the engine's own `ClaudeCLI`, with the `_SPOKEN` style prompt:
    - **U:** id-keyed per unit, the report 01 R1 contract as used today;
    - **A:** whole sentence over the merged span, drift allowed;
    - **B:** split-back with slot markers, explicit permission to restructure, and a `moved` list.
  - `gap3/judge.py` does blind, label-shuffled scoring (fidelity 1–5, naturalness 1–5, error list, best) with a different model.
  - Both are syntax-checked but never run.
- **Note:** CLI 2.1.201 maps `opus` to Opus 4.8 and ignores `--json-schema` (report 05). Update the CLI first.

---

## Corrections to earlier reports and to the task's premises

1. **01 F22 is refuted on the trace** (E1). "Units are sentence-complete" holds only when the ASR punctuates.
2. **"Microsoft LSST, KO" is not an SOV-target study.** It is (ES, KO)→EN (arXiv 2506.00740, §3.1). Reports 02 MS4 and 04 F10 give the Korean numbers without the direction, and they should read KO→EN.
3. **"IWSLT 2024/2025 dubbing tracks":** 2025 had none. 2024 evaluated EN→ZH, plus one DE→EN submission that was not scored.
4. **Sony EN→HI (2403.15469, 2603.28717) and DubWise** are sentence- or utterance-level, and none of them splits back.
5. **08 R1 is amended:** anchor pauses by fluency-based Telugu break placement (Federico 2020's criterion; Won 2025's LLM pause step), not by content.
   - 08 R1's `anchor_pause` of 0.3 s would add break opportunities in 14% of sentences on this podcast. Only about 2% have a pause of 1 s or more that is worth a hard split.
6. **02 R1 / 05 R3 are narrowed:** merge, yes; split-back only at hard boundaries (Answer 3).
7. **New warning:** don't fix punctuation with a Whisper `initial_prompt` alone. It dropped 14–23 s of speech in three runs (E2). That would turn fragments into skipped sentences.

## Confidence

| Claim | Confidence | Basis |
|---|---|---|
| The trace's fragments come mainly from Whisper's lost punctuation and casing | **High** | Exact replay reproduced all 18; 17 of 18 lie in unpunctuated text; checked twice |
| Two-pass punctuation transfer cuts non-final units from 18% to 2% with no words lost | **High on this fixture**, medium in general | Two identical runs; one video |
| A prompt-only Whisper fix drops speech | **High on this fixture** | Three runs, two different drop spans |
| The translation unit should be the full sentence | **High** | Every SOV-target system is sentence-level; Google's patent; 6 of 13 fragments mistranslated |
| Default drift inside a sentence; split only at a pause of about 1 s or more, the line cap, or an interruption | **Medium** | Brannon, Virkar and ElevenLabs's 1 s rule; the pause statistics. The 1.0 s threshold is inferred |
| Fluency-based, not content-based, break placement for Telugu | **Medium** | Federico 2020, Won 2025 (Korean) and NAIST CWMT; no Telugu test |
| Telugu boundary-type tallies (E7) | **Medium–low** | One annotator, not a native speaker |
| No dubbing study with an SOV target tests split-back | **Medium** | Checked Sony ×3, LSST, IWSLT 2024 and 2025, the patent and Won 2025; no broad web search was possible |

---

## Implications for Maata

**1. Fix the ASR first (`backends/apple.py`, `session._frontend`).** Impact high, effort S–M, no approval needed.
- Run a second `mlx_whisper` pass per 30 s window with a punctuated style prompt.
- Copy casing and trailing punctuation onto the unprompted pass's words by sequence alignment. Never take words from the prompted pass. A reference implementation is in `gap3/transfer.py`.
- Add a coverage guard: words per second against VAD speech per 30 s window. The ASR guard in report 10 R4 also catches prompt-style drops.
- In `_sentence_cut`, when no sentence end falls by `b`, search into `CHUNK_PAD` before falling back to the last word. That removes the chunk-boundary fragment (unit 85).
- **Alternative (INFERRED):** a Claude text pass that returns sentence-break word indices over the unchanged words. It uses no new ASR compute but spends subscription calls.

**2. Claude call contract: one id per sentence-complete unit** (report 01 R1, scene-batched).
- **Per unit, send:**
  - id, speaker, start, end;
  - budget in aksharas;
  - `breaks`: English break times, present only when the unit spans a hard boundary;
  - `cut_off: true` when the other speaker interrupts.
- **Reply shape:**
  - `{id, te}` for normal units;
  - `{id, te, pieces:[…], moved:[…]}` for units with breaks. The pieces must join back to `te`.
- **Wording of the break request:** "Put breaks where Telugu naturally pauses; restructure (split into sentences, ఎందుకంటే/అంటే/అయితే, demonstratives, clefts) so each piece can be said on its own; content may shift between pieces if it must, but list it."
- **For `cut_off` units:** keep the utterance unfinished (a non-finite form plus "…"). Never complete it, and never render a lone "so" as a content word.
- **Deterministic checks:**
  - piece count ≤ break count + 1;
  - numbers, names and negations preserved (a negation-count check would have caught units 73 and 82);
  - no piece forcing more than the 1.2× rate cap into its slot. If one does, drop the pieces and voice `te` as one drifting line.

**3. Segmenter (`segment.py`).**
- Keep one translation unit per sentence.
- A pause of about 1.0 s or more inside a sentence becomes a **break inside the unit**, not a new translation unit.
- A turn change with zero gap, where the next word continues the sentence in lowercase (unit 1), should be checked against both sides' speaker embeddings. Merge if they match; otherwise mark `cut_off`.
- When `max_len` must cut, cut only at clause marks. Never cut inside a name or number, and never between a verb and its object when a comma exists.
- Test `long_pause` at 1.0 s against 1.5 s only through the A/B (report 01 R6).

**4. Planner (`timing/planner.py`, ADR-017).**
- The default placement is the whole sentence over its span, drifting across pauses under 1 s within ADR-017's lag limits. Human dubbers and Amazon's off-screen relaxation both accept this.
- At breaks, synthesise the whole sentence once, as Amazon does. Find the Telugu pause (silence at the Telugu comma, or `torchaudio.functional.forced_align`, already locked per report 10). Stretch or insert silence there to meet the English pause.
- Report 08 R1's 0.3 s anchors become optional soft break opportunities, taken only when a Telugu break falls near the same time share, scored as in Federico 2020 (duration match × break plausibility).
- Record this as an amendment to ADR-017, which the maintainer sees.

**5. Run the A/B once the CLI is signed in and updated.**
- **Text stage:** `run_arms.py U`, `A`, `B`, then `judge.py`.
- **Listening stage:** 20–30 clips of 20–40 s, drawn from sentences with internal pauses of 0.7 s or more, sentences over 12 s, and interruptions. Arms A and B use the same TTS and planner. The maintainer rates meaning, naturalness and sync blind, 0–100, in randomised order.
- **Decision rule:** adopt B at hard boundaries only if it wins on sync without losing more than 5 points on naturalness. Otherwise keep A everywhere.

## Open questions

1. Does the two-pass transfer hold on other videos: fast talkers, music beds, accents? Measure the non-final share and VAD coverage on 3–5 more fixtures.
2. Would a natively punctuating ASR make the second pass unnecessary? Candidates are parakeet-tdt-0.6b-v2 and Qwen3-ASR, per `sota_ranking.md`, and their punctuation on this podcast is untested.
3. Does spoken Telugu accept post-verbal afterthoughts naturally enough to rescue verb | complement breaks? This needs the maintainer's judgement.
4. Is 1.0 s the right hard-break threshold for Telugu viewers watching a talking face? No Telugu or Indic perception data were found.
5. How often do real interruptions (`cut_off`) occur in multi-speaker content, and should they overlap as ElevenLabs allows?

## Sources (date = published, or fetched)

**Maata trace and logs:**
- `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`: sessions 2026-09-24 13:44:57, 18:31:17 and 20:06:51, analysed 2026-09-24.
- `scratchpad/app5.log`: 20:06 run, 2026-09-24.

**Maata repo, read 2026-09-24 (uncommitted working tree):**
- `engine/src/maata_engine/segment.py` (mtime 18:55), `session.py` (19:25), `backends/apple.py`, `claude_cli.py`, `text/tenglish.py`, `text/asr_fix.py`, `types.py`;
- `docs/DECISIONS.md` ADR-017 and ADR-018;
- `engine/.venv/.../mlx_whisper/transcribe.py` (mlx-whisper 0.4.3).

**My runs (M5 Pro, 2026-09-24), all in `scratchpad/gap3/`:**
- `asr_punct.py`, `resegment.py`, `transfer.py` (run twice; `transfer_run3.txt`), `pauses.py`;
- `items.json`, `run_arms.py`, `judge.py`.

**Papers:**
- Federico et al., "From Speech-to-Speech Translation to Automatic Dubbing", IWSLT 2020. arXiv 2001.06785 v3 (2020-02-02).
- Virkar et al., "Prosodic Alignment for Off-screen Automatic Dubbing", Interspeech 2022. arXiv 2204.02530 (2022-04-06).
- Brannon, Virkar, Thompson, "Dubbing in Practice", TACL 11 (2023). arXiv 2212.12137 (2022-12-23).
- Won, Jeong, Choi, Kim (ESTsoft), EMNLP 2025 System Demonstrations, pp. 515–521 (2025-11). https://aclanthology.org/2025.emnlp-demos.37.pdf, fetched 2026-09-24.
- Chadha, Subramanian et al. (Microsoft), "Length Aware Speech Translation for Video Dubbing". arXiv 2506.00740 v1 (2025-05-31).
- Mhaskar et al. (Sony Research India), NAACL 2024 Findings. arXiv 2403.15469 v1 (2024-03-20), fetched 2026-09-24.
- Dasare, Shah, Gudmalwar, Wasnik (Sony Research India), ICASSP 2026. arXiv 2603.28717, fetched 2026-09-24.
- Sahipjohn et al. (Sony Research India), DubWise, Interspeech 2024. arXiv 2406.08802 (2024-06-13).
- Findings of the IWSLT 2024 Evaluation Campaign, §7 (2024-08). https://aclanthology.org/2024.iwslt-1.1.pdf, fetched 2026-09-24.
- Findings of the IWSLT 2025 Evaluation Campaign. https://aclanthology.org/2025.iwslt-1.44.pdf, fetched 2026-09-24.
- Sakai, Makinae, Kamigaito, Watanabe (NAIST), "Simultaneous Interpretation Corpus Construction by Large Language Models in Distant Language Pair". arXiv 2404.12299 (2024-04-18).
- He, Boyd-Graber, Daumé III, "Interpretese vs. Translationese", NAACL 2016, pp. 971–976 (2016-06). https://aclanthology.org/N16-1111.pdf

**Patent:**
- Google LLC, US 2020/0404386 A1: filed 2018-02-26, published 2020-12-24, granted as US 11,582,527 B2 on 2023-02-14.
- Text via freepatentsonline, fetched 2026-09-24 (`dub_research/fpo_text.txt`).

**Other:**
- OpenAI Cookbook, "Whisper prompting guide": added 2023-06-27 (openai-cookbook commit #551), fetched 2026-09-24.
- WALS Online, Telugu datapoints 81A, 90A and 94A (Dryer & Haspelmath eds., CC-BY-4.0), fetched 2026-09-24.
- ElevenLabs bring-your-own-transcript guide, fetched 2026-09-24 (via report 01 F6).
- pyVideoTrans @d63d27ae (2026-09-23) and VideoLingo @9bc30202 (via report 09).
