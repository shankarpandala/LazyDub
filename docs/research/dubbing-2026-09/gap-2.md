# Gap 2: which script should English words use at the chatterbox-telugu input?

Research date: 2026-09-24. All measurements were made on the M5 Pro today. Labels used below:
- **MEASURED**: I ran it on local files.
- **DISCLOSED**: a vendor or author says it.
- **INFERRED**: my own reasoning.

## Question

1. Should English words reach the Telugu TTS in Latin script, as `text/tenglish.py` asks for today, or in Telugu script (ఆఫీస్)?
2. Can chatterbox-telugu pronounce Latin-script English at all?

## Answer

**Send English words to the TTS in Telugu script.** Keep a Latin-script copy only for captions and for the lint. One Telugu-script "spoken" field from Claude then does four jobs:
- TTS input;
- the reference for the IndicConformer QA check (report 10 R2);
- the text the akshara budget is counted on;
- the V3 retrain convention.

The retrain needs no new code-mixed Latin data.

**Can the model read Latin-script English? Yes, but it loses more words.** It says most Latin-script English words, because the base model's Latin embeddings and English ability survive the LoRA. But:
- **Loss rate:** it loses about 1 English word in 5, against about 1 in 8 when the same words are written in Telugu script.
- **Line length:** lines come out slightly longer and their length is harder to predict.
- **Telugu words are unaffected:** only the English words themselves get worse.

**Why the model behaves this way.** It learned code-mixing almost entirely in Telugu script:
- Of its 34,450 training utterances, 104 contain any Latin text. All of them are FLEURS proper nouns and acronyms.
- At least 12.5 % contain common English loanwords written in Telugu script.

**What this settles, and what it doesn't.** Part of the maintainer's "unnatural English/Telugu mix" complaint is therefore a TTS input problem that no prompt wording fixes. Whether Telugu-script English also *sounds* more natural to a native listener isn't measured yet. The blind A/B kit below is ready. One check (below) suggests Latin-script input gives slightly more English-sounding words, though the difference is not significant, and a listener may prefer that on some lines.

## Evidence

### A. What chatterbox-telugu was trained on (MEASURED, primary files)

Sources:
- the fork `~/projects/chatterbox` at f30b0d1 (commit dated 2026-06-06), which is the commit pinned in `engine/.venv` `direct_url.json`;
- its training log `train_10k_20260605_174226.log` (2026-06-05);
- the feature files T3 actually trained on, `runs/te/features/index.jsonl` (34,450 rows).

I decoded every `text_tokens` array back to text (`scratchpad/gap2_manifest_scan.py`, `gap2_trainset_loans.py`).

**A1. The training mix**

| Source | Utterances | Hours (speech tokens / 25 per s, after the 14 s crop) |
|---|---|---|
| IndicVoices-R Telugu | 33,287 | 66.52 |
| FLEURS te_in | 1,163 | 2.96 |
| **Total** | **34,450** | **69.47** |

The model card says about 5 h of FLEURS. The trained set holds 2.96 h, from 1,163 of the 2,302 FLEURS train rows.

**A2. Latin-script text in the training data**
- 104 of 34,450 utterances (0.30 %) contain any Latin text.
- All 104 are FLEURS rows; none come from IndicVoices-R.
- That is 199 Latin words in total, 0.019 % of text tokens.
- They are names, brands and acronyms: xinhua, aol, fbi, gdp, nba, "utah jazz", "george w bush" and so on.
- There are no everyday English words inside Telugu grammar (the "phone out ఇస్తే" pattern) in Latin script anywhere.

**A3. English loanwords in Telugu script are common in the training data**
- 4,295 utterances (12.5 %) contain at least one of 52 common loanwords in an exact or suffixed form, 5,805 occurrences in all. The top ones are ఆర్డర్ 503, ఫోన్ 452, టైం 391, స్కూల్ 336, ఫస్ట్ 326 and ఆన్లైన్ 270.
- 626 of them sit directly before a light verb (చేయు, అవు or పెట్టు), for example ఆర్డర్ చేస్తా. This is exactly the Tenglish grammar the prompt asks for, but in Telugu script.
- 12.5 % is a lower bound, because only 52 words were listed.
- FLEURS translators did the same with loans: ఇంటర్నెట్ సర్వీస్ ప్రొవైడర్, బిజినెస్ లేదా ఫస్ట్-క్లాస్ సీట్ల డిస్కౌంట్.

**A4. Tokenizer**
- Telugu was added as 66 single-codepoint tokens plus `[te]` (ids 2454–2520).
- The Latin BPE pieces are the base model's (ids below 2454).
- `MTLTokenizer.encode` lowercases and NFKD-normalises everything. So "IQ" reaches T3 as `i`,`q` and cannot be told apart from a word.
- The fork's comment says Latin runs are "left untouched so the model keeps its English ability".
- The log confirms `text_emb`/`text_head` were resized 2454→2521 with "overlap copied, new rows fresh". The LoRA (rank 16, q/k/v/o/gate/up/down, 16.47M trainable) covered 10k steps × 16 accumulation, about 4.6 epochs.

**A5. The code-switching claim has no valid number behind it**
- The only code-switch figure in the fork is `runs/eval_heldout.log`. It is 3 demo sentences scored with `openai/whisper-small`, which transcribes Telugu into the wrong script (CER 0.83–20 on Telugu). The fork's own `rescore_heldout.py` docstring says so.
- The later vasista22 rescore left no summary in `rescore.log`.
- So the model card's "code-switched Tenglish" support is untested. It is also contradicted by A2.

**A6. The full IndicVoices-R Telugu corpus has no Latin at all**
- The corpus is all 185 cached shards of snapshot 5f4495c9: 47,208 rows, 134.0 h (`gap2_ivr_full.py`).
- It has **0 Latin letters and 0 bracket characters** in any of `text`, `verbatim` or `normalized`.
- This extends the maintainer's 12-shard spot check to the whole corpus.

**A7. FLEURS te_in** (HF snapshot 70bb2e84; paper arXiv 2205.12446, v1 2022-05-25: FLoRes-101 sentences read aloud)
- Share of rows with any Latin word: 11.2 % of train (258/2,302), 17.7 % of validation and 14.2 % of test.
- Latin words make up 1.6–2.1 % of all words.
- They are proper nouns, brands and acronyms only; the translators wrote loans in Telugu script (A3).

### B. The corpus conventions behind that data

**B1. IndicVoices writes loanwords in native script (DISCLOSED).** Source: IndicVoices, arXiv 2403.01926v1, 2024-03-04 (only version); local PDF, appendix pp. 48–49, "Rules for dealing with loan words". The page's Indic examples didn't survive text extraction; the rule text did.
- Borrowed English words and English named entities are transcribed verbatim in the native script.
- When such a word is uninflected, the English spelling is added in square brackets, e.g. "order (native) → (native) [order]", "[amazon]", "[first]".
- Acronyms spoken letter by letter are written with spaces, e.g. "N R I", with bracketed letters.
- Numbers and units said in English are written as said, in the target script.
- Borrowed words with native suffixes keep them ("calendar-on", "Friendo").
- This answers the 2026-09-24 note that the HTML fetch lacked the English-word rule.

**B2. IndicVoices-R keeps the verbatim text, without the English spellings.**
- DISCLOSED (arXiv 2409.05356v2, 2024-10-07, NeurIPS 2024 Datasets and Benchmarks; pipeline section, "Transcript" and "Step 6: Post-Processing"): IV-R uses "the verbatim version of the transcript" and applies no text normalisation. It also drops utterances where the two transcription levels differ by more than 5 % CER.
- MEASURED (A6): the released Telugu text carries none of the bracketed English spellings.
- INFERRED: they were stripped or belong only to the other transcription level. Either way, what T3 saw is native script only.

### C. Listening-free test: 50 lines rendered both ways (MEASURED)

**Setup**
- **Lines:** 50 Tenglish lines from the maintainer's podcast trace (`~/Library/Caches/maata/4Vz6L8B73i4/units.jsonl`, the current contract, latin_ratio 0.17–0.67, mean 0.35). The obviously broken lines were lightly cleaned.
- **Two arms:** each line was written twice, identical word for word.
  - **L:** English in Latin script, as today.
  - **T:** English spelled in Telugu script the way Telugu speakers write it, e.g. ఎంప్లాయీస్, ఐక్యూ. I wrote these forms myself, as Claude would.
  - File: `scratchpad/gap2/pairs.py`.
- **Voices:** the podcast's two real speakers, built exactly as ADR-017 builds them:
  - S1 (guest): best single span, averaged identity, own-English prompt, cfg 0.5.
  - S2 (host): the stitched reference.
- **Rendering:** 2 seeds per line and voice, with the same seed for both arms. That makes 400 takes through the production `ChatterboxTeluguTTS` path (bf16 T3, 10 CFM steps, no speed-up), 2,254 s wall time.
- **Scoring:**
  - The reference is the T-script line for both arms, which is the script normalisation.
  - CER uses no spaces and NFC. Only punctuation and symbols are removed, because Telugu vowel signs are kept.
  - A Levenshtein alignment splits the errors into English-word spans and native-Telugu spans.
  - Bootstrap 95 % CIs are resampled over lines. Scripts: `render.py`, `asr.py`, `score.py`, `score2.py`, `en_check.py`.
- **Checker:** vasista22/whisper-telugu-small@717212f9.
  - Apache-2.0, created 2022-12-20, already in the local HF cache.
  - Its floor on 52 human FLEURS te test sentences is corpus CER 7.6 % (median 4.0 %). That floor is optimistic, because it was trained on FLEURS train and dev.
  - It writes English words in Telugu script, like IndicConformer. Neither arm's transcript ever contained Latin.
- **What I could not run:** IndicConformer and SraVaani are not in the local cache (refs only). Downloading a gated model needs the maintainer's approval, so both are still to run.

**Results (L = Latin-script English, T = Telugu-script English)**

| Measure (200 paired takes, 50 lines) | L | T | L − T, 95 % CI |
|---|---|---|---|
| Whole-line CER | 10.0 % | 7.1 % | **+2.8 pts [+1.3, +4.5]** |
| CER on English-word spans | 19.8 % | 12.5 % | **+7.4 pts [+4.1, +11.0]** |
| CER on native Telugu spans | 4.1 % | 4.1 % | 0.0 [−1.0, +1.0] |
| English words lost (≥ 50 % of characters wrong) | 20.8 % (145/696) | 12.1 % | **+8.8 pts [+4.5, +13.3]** |
| Native words lost | 3.9 % | 4.1 % | −0.2 [−1.5, +1.3] |
| English words recognised by English-mode Whisper large-v3-turbo (the pipeline's own MLX model) | 74.0 % | 68.4 % | +5.6 [−0.6, +12.1] (not significant) |
| Line duration | 4.41 s | 4.26 s | +0.15 s [+0.07, +0.22] (+3.4 %) |
| Speaker similarity (pyannote, same as voice_e3) | 0.559 | 0.551 | +0.009 [0.000, +0.017] |
| Takes with CER > 50 % | 2 | 0 | |
| Token-cap hits | 0 | 0 | |

**Checks that the result holds**
- The English-span gap has the same sign for both voices (S1 +7.0, S2 +7.7) and both seeds (+8.8, +5.9).
- It holds without outliers (+6.7).
- Telugu-script input is not uniformly better: in 11 of 200 pairs T lost an English word that L kept, for example రీచ్ and అచీవ్.
- Typical L failures are a skipped or garbled English word with the audio not shorter:
  - "favor" missing;
  - "work" heard as ఒప్పు or ఓట్;
  - "employees … IQ lower" dropped by S1;
  - "billionaire brain కి broke person brain కి" lost.

**Akshara counting**
- Fitting duration = a + units/rate per arm:

  | Audio | Counting English in Latin (English-syllable rule) | Counting the Telugu-script form (pure akshara rule) |
  |---|---|---|
  | L arm | R² 0.675, mean error 12.1 % | R² 0.714, 11.2 % |
  | T arm | R² 0.744, 10.7 % | R² 0.774, 10.2 % |

- The Telugu-script count predicts duration better on both arms.
- It gives 9 % more units per line than the current count (1,216.5 vs 1,116.0 over the 50 lines).

**What these numbers are and aren't**
- They are an ASR proxy for intelligibility, from one checker family (Whisper) plus one English-mode cross-check. They are not a naturalness judgement.
- The English-mode row hints that Latin input comes out a little more English-sounding; T comes out more Telugu-sounding and more reliably spoken.
- Which one a Telugu listener prefers is open.

### D. External evidence

**D1. Sarvam's rule and Sarvam's own sample disagree, for different reasons (DISCLOSED).**
- **The rule:** the TTS best-practices page (undated, fetched 2026-09-24 twice) says "Write English words in English script, Hindi words in Devanagari". Its examples are Hindi only. It also warns that romanised Indic "significantly degrades output quality".
- **The sample:** Bulbul V3's Telugu cloning sample (blog dated 2026-02-05; local copy plus a live fetch today) writes ప్లీజ్ కన్ఫర్మ్ and పేమెంట్ in Telugu script, and keeps only "Okay" and "7th November" in Latin. The live fetch's summariser mislabelled which words were Latin, so the sample string itself is the evidence.
- **Reading (INFERRED):** Sarvam's rule describes a model trained on Latin-script code-mixing. It does not transfer to chatterbox-telugu, whose training data is the opposite (A2, A3).

**D2. The QA recogniser writes English in Telugu script.** IndicConformer transcribes English words in Telugu script, keeping 0 % in Latin. Source: third-party svanita-0.6b card, fetched 2026-09-24, cited in report 10 A7; self-reported. A Telugu-script spoken form therefore matches the QA recogniser with no mapping step.

**D3. Upstream Chatterbox has no code-switching support.** resemble-ai/chatterbox issue #346 (opened 2025-11-07, still open, read via `gh` today): the multilingual model takes a single `language_id`, and embedded foreign words get the main language's phonetics. User reports only (anecdotal).

**D4. Resemble's V3 post is silent on code-switching.** The Chatterbox Multilingual V3 post (2026-06-10) never mentions code-switching or mixed-script input.

**D5. Older research: map mixed text into the voice's own language.** Sitaram, Rallabandi, Rijhwani and Black, "Experiments with Cross-lingual Systems for Synthesis of Code-Mixed Text", SSW9 2016 (isca-archive.org, abstract read today). They mapped mixed Hindi–English text into the phonetic space of the language the voice was recorded in. Listeners strongly preferred the Hindi-target systems.
- This predates neural TTS and is directional only.
- It matches writing English in the script the Telugu model was trained on.

**D6. What I could not search.** The web-search budget for this session ran out before a wider academic sweep, so there is no survey of 2025–2026 code-mixed Indic TTS papers here. arXiv API queries returned nothing that tests input script for Indic code-mixed TTS.

### E. Side findings

**E1. ZWNJ reaches the TTS but was never trained on.**
- 0 of 34,450 training utterances contain ZWNJ (token 2050), yet `tenglish._STRAY` lets U+200C/U+200D through to the TTS.
- The earlier `scratchpad/tts_ab.py` pair 4 (మైండ్‌సెట్) also carried ZWNJ.
- Fix: strip ZWNJ/ZWJ from the TTS form.

**E2. `docs/SPEC.md:221` is wrong.** It says Latin-script English is what "chatterbox-telugu is trained to speak". A2 contradicts that.

**E3. The current contract hard-codes Latin in several places.**
- `text/tenglish.py:60` (formal style) and `:332` (condense) ask for English in Latin.
- The colloquial shots model Latin.
- The `Length:` note counts "an English syllable" as one akshara.
- `latin_ratio` and `lint` detect English by Latin letters.

**E4. The training used about 70 % of the available data.** 33,287 of the 47,208 cached IndicVoices-R Telugu rows were used.

## Corrections to earlier reports

- **Report 03 R4 and question 3:** this is not an optional A/B. The TTS side is now tested (C), and Telugu script wins on intelligibility with a CI that excludes zero. Only the naturalness listening remains.
- **Report 06 B12 / model card:** "~5 h FLEURS" is 2.96 h in the trained set. The code-switching support has no valid measurement (A5).
- **The maintainer's framing, "the [te] T3 probably never saw Latin text":** it is nearly right. It saw Latin in 0.30 % of utterances, all FLEURS names and acronyms, and never everyday English inside Telugu grammar.
- **Report 10 R2:** the proposed extra "spoken-form copy" should become the primary TTS field, not a QA-only field.

## Confidence

- **High:** the training data is Telugu-script-only for everyday English words (A1–A6, measured on the exact feature files and the full corpus). Two independent passes over the same files agree, and the full-corpus scan agrees with the maintainer's 12-shard scan.
- **Moderate–high:** Telugu-script input is more intelligible to a Telugu recogniser and makes duration more predictable (C). The effect is paired, its CI excludes zero, and it holds for both voices and seeds. But there is one checker family, and the Telugu-script spellings were mine.
- **Low to moderate: which arm sounds more natural to a native listener.** It is not measured. The English-mode check hints that Latin gives more English-sounding words, but that difference is not significant.

## Implications for Maata

1. **Change Claude's output contract.**
   - Claude returns a Telugu-script `spoken` line per tier: normal, concise and very concise.
   - It also returns a small `english` map from word index to English word, e.g. `{"4": "favor"}`.
   - `spoken` feeds TTS, the IndicConformer reference and the akshara budget.
   - Captions, `latin_ratio` and `lint` are rebuilt from `spoken` plus the map. This avoids asking Claude for two full copies that could drift apart.
   - A pure-logic validator should reject:
     - any Latin letter in `spoken`;
     - ZWNJ/ZWJ in `spoken`;
     - a map index that points at a native word.
   - Acronyms are spelled as said (ఐక్యూ, ఏఐ). Names Claude can't transliterate should trigger one retry, not a Latin fallback.
   - Update the prompt and shots (`tenglish.py:60, :332` and `SHOTS_COLLOQUIAL`), `SPEC.md:221` and an ADR.
2. **Akshara counting.**
   - Count `spoken` with `count_telugu` only. The English-syllable heuristic becomes a fallback for stray Latin.
   - Expect about 9 % more units per line, and re-seed each voice's rate.
3. **QA (report 10 R2).**
   - The same `spoken` field is the CER reference. Use per-word tolerance, because checker spellings vary (ఎంప్లాయూస్ vs ఎంప్లాయీస్).
   - Calibrate thresholds on the A/B ratings.
   - Run IndicConformer on these 400 takes once its download is approved: `gap2/asr.py` swaps in with a different loader.
4. **V3 retrain data plan (06 R6).**
   - No Latin code-mixed data is needed for a Telugu-script contract. IndicVoices-R already carries loans in Telugu script (A3). Use all 47,208 rows rather than 33,287.
   - Transliterate the 104 Latin FLEURS rows to Telugu script, or drop them, so `[te]` stays single-script.
   - **If Latin input is ever wanted instead,** that is when code-mixed data becomes necessary. Latin-script English inside Telugu grammar would have to be added, for example from the bracketed English spellings in raw IndicVoices transcripts (B1). IndicVoices' own licence is unverified here.
5. **Listening test.** The blind kit is ready:
   - clips: `scratchpad/gap2/ab/` (50 items, X/Y wavs);
   - rating sheet: `sheet.csv`;
   - answer key: `scratchpad/gap2/ab_key.json` (don't open it before rating).

   Suggested rule: adopt Telugu script unless the native listener prefers Latin on clearly more than half of the items.

## Files

All paths are under `/private/tmp/claude-501/-Users-shankarpandala-projects-LazyDub/96304fd6-abc3-4ba1-bb31-785d21e94c6b/scratchpad/`:
- `gap2/pairs.py`
- `gap2/render.py`, `gap2/takes.jsonl`, `gap2/wav/` (400 takes)
- `gap2/asr.py`, `gap2/hyps.jsonl`, `gap2/fleurs_hyps.jsonl`
- `gap2/score.py`, `gap2/score2.py`, `gap2/scored.json`
- `gap2/en_check.py`, `gap2/en_hyps.jsonl`
- `gap2/make_ab.py`, `gap2/ab/`, `gap2/ab_key.json`
- `gap2_manifest_scan.py`, `gap2_trainset_loans.py`, `gap2_ivr_full.py`, `gap2_fleurs.py`
