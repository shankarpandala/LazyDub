# Gap 4: Why do Maata's dubs underfill their slots, and which length policy follows?

Written 2026-09-24. It builds on `sota_ranking.md` and reports 03, 04, 05, 06, 08 and 09 in this folder, and does not repeat them. Everything below was re-derived from the maintainer's trace, the engine log, the repo, a fresh on-device measurement and primary sources. Scripts and raw outputs are in `scratchpad/gap4/`.

Labels:
- **MEASURED:** a number computed here from Maata's own data, or from a study's data.
- **DISCLOSED:** stated by a company or by a paper's authors.
- **INFERRED:** reasoning, not evidence.

The web-search budget for this session was already spent, so primary pages were fetched directly (WebFetch, curl, the arXiv HTML and PDF, the Hugging Face resolve URLs).

---

## Question

The ADR-017 smoke run filled 0.78x of its slots, and the 20:06 trace played dub audio for 0.87 of the source time. Lines of 8 s or more ended a median 1.98 s early. Aksharas per English syllable fell from 1.57 to 1.19 between two sessions on the same day. Three causes are possible:
- dropped content;
- budgets that are too tight;
- a cloned voice that is faster than assumed.

Which is it? And which length policy follows: a condense-only or a two-way translation contract, whether ADR-017's 1.0 audio-rate floor changes, and whether a coverage check has to come first? The reports disagree on the rate prior (05 R2 says about 3.4 aksharas/s; 08 measured about 5.6–5.7) and on lengthening (04 R1 says never; 08 R2 says to restore targets on every line; 03 R1, 09 R4 and 08 R5 say to slow the audio).

## Short answer

1. **It is mostly compact but faithful Telugu against source spans that contain pauses. Dropped content is a real but secondary cause. Budgets and the voice's rate are not the cause.** MEASURED on the 20:06 session (98 lines, 0–675 s):
   - A line-by-line coverage judgment found 71 lines complete, 16 with a minor drop (an intensifier, a hedge or a discourse marker), 8 with a content phrase or clause missing, and 3 with meaning errors unrelated to length.
   - Restoring every dropped word would raise the fill from 0.870 to about 0.91. Dropped content therefore explains about 30 % of the 82 s of net underfill at most. Complete lines alone account for 49 s of it (60 %).
   - The lowest-fill lines are complete lines where the English is slow and emphatic (2.0–2.5 syllables/s over the span). 20 of the 27 lines of 8 s or more are complete, and 20 of the 27 are run-on ASR chunks that start mid-sentence, in lowercase.
2. **"Budgets too tight" is refuted for the current build.**
   - Since ADR-018, normal lines are translated with no length note at all (`tenglish.plain_messages`).
   - Only 2 of 98 lines had any budget-driven wording change, and both were re-syntheses that shortened an overrunning line.
   - Against a correctly calibrated budget, the median line already uses 0.89 of its akshara allowance.
3. **"The cloned voice is faster than assumed" is refuted as a cause of the 1.57 → 1.19 fall.** The voices ran at about the same pace in both sessions: 5.83 units per logged second at 13:44 (6.56 on lines played unscaled), against 5.71 natural at 20:06.
   - The engine's calibration takes logged 5.4–5.8 aksharas/s on the ADR-017 references.
   - A fresh re-measurement today on the current references gives the same (see E3).
   - `DEFAULT_RATE = 5.5` is within about 5 % of all three.
   - The 3.2–3.5 figures in `tts_pace.json` and ADR-017's "3.43 before" come from **one stitched guest reference cut from a 5-minute excerpt**. No production session ever ran that slowly.
   - **05 D4/R2 (lower the prior to about 3.4) is wrong and should not be adopted.** Under that prior, 93 of 98 lines would exceed their length target, by a median of 1.51x.
4. **The fall from 1.57 to 1.19 is a translator and prompt change.** INFERRED from MEASURED evidence: the 13:44 session was the ADR-016 build, the Qwen3.5-9B era, and the 20:06 session is Gemma 3 12B with the plain prompt.
   - On the identical 37 bake-off lines, Qwen3.5-9B writes 1.60 aksharas/syllable and Gemma 3 12B writes 1.38.
   - In the 5 dropped-clause windows checked (E5), the 13:44 session had rendered the missing content, but in the English-heavy, calqued style the maintainer rejected.
5. **Policy that follows:**
   - **(a) Put a coverage check first.** It must exist before any timing or length-target work, because every length lever measured so far trades content for fit. In ADR-018 the length note raised major meaning errors from 5 % to 13.5 %. In HOMURA, a separate length-rewrite pass is where omissions appear.
   - **(b) Make the translation contract two-way but asymmetric.** Completeness comes first. Add a *fuller* tier, one that restores what the speaker actually said and adds no new facts, next to the concise tiers. Select by predicted duration inside a two-sided band. This is not condense-only.
   - **(c) Keep the audio-rate floor at 1.0 for now.** The underfill is a text and pause-placement problem. Slowing speech is the least tolerated direction. The voice's own `cfg_weight` already moves its pace: cfg 0.3 is 6–9 % slower than 0.5 on today's references (E3).
   - **(d) Keep `DEFAULT_RATE` ≈ 5.5–5.8.** Calibrate each voice after its settings are fixed, and re-calibrate whenever cfg, exaggeration or the reference changes.

## Evidence

### E1. The fill identity, per session (MEASURED)

The natural fill is (aksharas per English syllable) × (source syllables/s over the span) ÷ (voice units/s):
- 20:06 session: 1.218 × 4.13 ÷ 5.71 = **0.881**. The trace's own natural audio/span is 0.882. The in-vocoder speed-up (mean 1.017) takes it to the played 0.870.
- 13:44 session (ADR-016 build): 1.563 × 4.11 ÷ 5.83 = 1.10. Played 0.989, at a mean rate of 1.136.
- 18:31 session (ADR-017 build, 38 lines): 1.134 × 4.50 ÷ 5.45 = 0.94. Played 0.916.

Between 13:44 and 20:06 the source rate and the voice rate barely moved (4.11 → 4.13 and 5.83 → 5.71). The length ratio carries the whole change.

Source: `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`, sessions opened 2026-09-24 13:44:57, 18:31:17 and 20:06:51 (local time), analysed 2026-09-24 (`gap4/trace_stats.py`).
- Counting: `count_units` (the engine's akshara counter) for Telugu and the engine's `english_syllables` for English.
- `audio_s` is the natural duration. Verified by checking that the rendered `pcm/<id>.npy` length equals `audio_s / audio_rate` on the 20:06 session.
- The 13:44 build is not in git, so its `audio_s` semantics could not be re-read. The voice-rate conclusion therefore uses only lines played at rate 1.00, which are unambiguous: 41 lines at 6.56 units/s.

**The ADR-017 smoke run's 0.78** (`scratchpad/smoke/localbench0/units.jsonl`, 2026-09-24 16:17, `podcast_90s.wav`, 15 lines, one speaker). MEASURED:
- The 0.78 is natural audio ÷ `target_s`, which includes borrowed silence (0.776). Against the source span it is 0.826 natural and 0.819 played.
- Aksharas/syllable 1.42, source 3.69 syllables/s, voice 6.33 units/s.
- Reading the pairs, 6 of the 15 lines carry text belonging to a neighbouring line: one line's own content is replaced by the next line's, and another repeats its neighbour. This is the "next line" note failure that ADR-018 later removed.
- That run therefore cannot separate the causes, and its 0.78 should not be used as a calibration point.

### E2. The voice's pace: three independent measurements agree on about 5.5–6, and the 3.4 is one outlier reference (MEASURED)

| Source | Guest / S1 | Host / S2 | Reference |
|---|---|---|---|
| Trace, natural units/s, 20:06 | 5.67 | 5.86 | ADR-017 single clip + 60 s identity |
| Trace, 18:31 | 5.46 | 5.42 | ADR-017 |
| Trace, 13:44 (lines at rate 1.00, both speakers) | 6.56 | | ADR-016 stitched 11.2 s |
| `engine.log` calibration take, 18:32 | 5.7 | 5.4 | ADR-017 |
| `engine.log` calibration take, 20:08 | 5.8 | 5.8 | ADR-017 |
| Fresh re-measurement today, cfg 0.5, exaggeration 0.5 (E3) | calibration 5.97, trace lines 5.63 | calibration 5.93, trace lines 6.02 | ADR-017, rebuilt from the same audio |
| `tts_pace.json` (13:19) and `voice_e3.json` "a-current" (14:55) | **3.18–3.54 / 3.43** | 5.49 | stitched reference from `podcast_600_900.wav`; the log shows "19.2 s from 2 clips" |

- **Reading:**
  - The slow figure belongs to one reference built from a different 5-minute excerpt.
  - The same speaker's stitched reference in the real 13:44 session ran at about 5.8–6.6.
  - Pace therefore depends strongly on which reference audio is chosen. That argues for per-voice calibration, not for any global prior.
  - ADR-017's "fixed its pace (3.4 → 5.1)" is true of that experiment, but the slowness it fixed never occurred in a production session.
- **05 D4 read `tts_pace.json` correctly but generalised it.** Its own caveat (guest only, 10 lines) was the warning sign.
- **Simulation** (`gap4/prior_sim.py`: the session's calibration take plus online RLS updates, replayed over the 20:06 lines):
  - With the 5.5 prior, the estimate is 5.89 after calibration, and 8 of 98 lines are predicted to overrun (and would request concise wordings).
  - With a 3.4 prior, the estimate is only 4.25 after calibration, because the RLS prior is weak and one take moves it only part way. 18 of 98 lines are then flagged, and the early lines are the worst (predicted/actual up to 1.63).

### E3. Fresh re-measurement: pace on the current references by cfg and exaggeration (MEASURED today)

**Method:** `gap4/pace_now.py`, on the M5 Pro, offline, with the pinned local models.
- It repeats the session's pre-pass on `audio.m4a` 0–600 s: pyannote community-1 in 3-minute blocks, 15 s overlap.
- It builds each voice as `Session._build_voice` does: best single span ≥ 8 s for S3Gen, up to 60 s of clean clips for T3 identity, T3 prompted with the speaker's own English, first minute avoided.
- Every condition voices the same text: the session's calibration sentence plus 8 Telugu lines from the 20:06 trace.
- Durations are `synthesize_mel` natural seconds, as the session observes them.

"calib" is what the session's calibration take would log. "lines" is aksharas per natural second over the 8 trace lines.

**Result** (`gap4/pace_now.json`, 2026-09-24 22:00–22:32 IST). Aksharas per natural second over the 8 trace lines. Seed 0, one take per line per condition. S1 is the guest (394 s of speech in the first 10 min) and S2 the host (160 s):

| cfg \ exaggeration | S1 guest: 0.5 / 0.7 / 0.9 | S2 host: 0.5 / 0.7 / 0.9 |
|---|---|---|
| 0.3 | 5.15 / 5.32 / 5.25 (mean 5.24) | 5.50 / 5.72 / 5.90 (mean 5.71) |
| **0.5 (current)** | **5.63** / 5.78 / 5.88 (mean 5.76) | **6.02** / 6.16 / 6.12 (mean 6.10) |
| 0.7 | 5.95 / 6.00 / 6.15 (mean 6.03) | 6.22 / 6.26 / 6.30 (mean 6.26) |
| ADR-016 stitched reference, 0.5 / 0.5 | 5.61 | 5.74 |

- **The current setting reproduces production.** At 0.5 / 0.5 the voices run at 5.63 and 6.02, against 5.67 and 5.86 in the 20:06 trace (within 3 %).
- **cfg_weight is the lever.**
  - Going from 0.5 to 0.3 slows the voice by 9 % (guest) and 6 % (host).
  - Going to 0.7 speeds it up by 5 % and 3 %.
  - The direction matches the Chatterbox README.
- **Exaggeration is a small lever.** Going from 0.5 to 0.9 speeds the voice up by about 3 % for both speakers (means over cfg: 5.58 → 5.76 and 5.91 → 6.11). The direction matches the README; the size is small.
- **The ADR-016 stitched reference, rebuilt from the session's own first 10 minutes, is not slow** (5.61 and 5.74). This confirms that 3.4 belonged to the excerpt-specific reference in E2.
- **A single calibration sentence is noisy.** Across the 18 conditions, the calibration sentence's rate differs from the 8-line rate by a factor of 0.87 to 1.18, while the 8-line rate varies smoothly with the settings. One sentence is not enough to seed a budget (MEASURED; see implication 1).
- **Coupling with 06 R2 (MEASURED):** the voice-match sweep over cfg ∈ {0, 0.25, 0.5} would shift every length budget by roughly −6 % to −15 %. The cfg 0 end is extrapolated and was not measured here. Exaggeration ∈ {0.4, 0.5, 0.7} would shift it by only about ±2 %.

**Other sources on the same question:**
- **README (DISCLOSED):** the Chatterbox README says higher exaggeration tends to speed speech up, that lowering cfg_weight gives slower, more deliberate pacing, and that about 0.3 helps fast-speaking references. resemble-ai/chatterbox `README.md` on master, last changed 2026-07-21 (commit 5de7a54a), fetched 2026-09-24.
- **Older sweep (MEASURED, on the slow excerpt reference):** `tts_pace.json` shows cfg 0.3 → 3.18–3.20, cfg 0.5 → 3.37–3.47 and cfg 0.7 → 3.54. That is about +11 % from cfg 0.3 to 0.7, while exaggeration 0.5 → 0.9 at cfg 0.5 moved it by only ±2 %.

### E4. Budgets are not binding in the current build (MEASURED from code and trace)

- **Normal lines get no length target.** `MLXChatTranslator.translate()` uses `tenglish.plain_messages`, which carries no length note. ADR-018 removed the note because, with lint retries, it raised major meaning errors to 13.5 % against 5 % for the plain prompt, in blind judging of 37 lines. `target_units` reaches the model only through `translate_candidates`, which runs when `_fits()` predicts an overrun, and through the re-synthesis path.
- **In the trace**, `telugu_first == telugu` on 96 of 98 lines. The 2 changed lines were re-syntheses that shortened an overrun.
- **The budget formula** is `target_units = target_s × rate × 0.95`, where `target_s` is the span plus up to 0.5 s of borrowed silence (target/span = 1.031).
  - At the calibrated 5.8, the median line uses 0.89 of its allowance, and 28 of 98 lines exceed it.
  - At 5.5: 0.94, and 36 lines.
  - At 05's 3.4: **1.51, and 93 lines** (88 by more than 10 %).
- **So** the budget is currently slack, not tight. If 08 R2's "length target on every line" were adopted with a 3.4 prior, it would ask almost every line to compress by a third (INFERRED from these counts).

### E5. Coverage judgment of all 98 source/Telugu pairs (MEASURED; Claude as the judge)

**Method:** every pair of the 20:06 session was read by this Claude session against its English, and each line was classed.
- **C:** complete.
- **m:** a minor drop, such as an intensifier ("really", "a lot of"), a hedge, "okay", or a trailing backchannel.
- **P:** a content phrase or clause missing, with the dropped English syllables estimated.
- **E:** a meaning error not caused by length.

The labels are in `gap4/coverage.py`. This is the "Claude coverage and back-translation judge" asked for, run in-session rather than through `claude -p`, so no quota was used and no dependency added.

| Class | Lines | Fill (aggregate) | Median fill | Aksharas/syllable | Net underfill |
|---|---|---|---|---|---|
| C complete | 71 | 0.894 | 0.92 | 1.26 | 49.1 s |
| m minor drop | 16 | 0.871 | 0.94 | 1.15 | 11.6 s |
| P clause/phrase dropped | 8 (ids 4, 5, 27, 37, 56, 60, 69, 91) | 0.695 | 0.76 | 0.92 | 18.3 s |
| E meaning error | 3 (ids 17 number misread, 48 unrelated sentence, 82 negation added) | 0.836 | 0.78 | 1.67 | 3.4 s |

- **Restoring the dropped content:** about 71 English syllables (P) and 45 (m), at the session's 1.22 aksharas/syllable and 5.7 units/s, add about 15.2 s and 9.6 s of audio. The fill would rise from 0.870 to **about 0.909**.
- **Long lines (≥ 8 s):** 20 C, 3 m, 3 P, 1 E. Fill 0.817, against 0.935 for shorter lines. Source syllable rate: 4.03 for long lines, 4.26 for short.
- **The 15 lowest-fill lines** include the "belief" passage (ids 94–97; fill 0.51–0.70). It is complete, but the English runs at 2.0–2.5 syllables/s because the speaker pauses for emphasis. Only 4 of the 15 are P.
- **Correlations across lines:** fill vs source syllables/s r = 0.25; fill vs aksharas/syllable r = 0.20. Neither dominates. Per-line fill is noisy because spans include pauses (08 F1's caveat).
- **Second check:** for each P window, the 13:44 session (a different model and segmentation) did render the missing content, for example the table and the "different things" list in id 5, and the "everyone wants to be superhuman" clause in id 91. So the content is really in the audio; it is not an ASR artifact. The 13:44 renderings kept it as English calques, which is the style the maintainer rejected.
- **Other "incomplete" signals (MEASURED; the effect on perception is INFERRED):**
  - 25 of 98 English units start lowercase, meaning they continue a sentence, and 18 don't end in terminal punctuation.
  - 20 of the 27 long lines are run-on ASR chunks that start mid-sentence.
  - Telugu is verb-final, so a sentence split across units is rendered either as two restructured halves or as a dangling half. Either can sound "incomplete" even when nothing is dropped. This supports 05 R3 (translate sentences, then spread the Telugu over the slots).
- **Limits:** one judge in one pass; the syllable estimates for dropped phrases are approximate; there is no second human rater.

### E6. What a faithful Telugu length looks like (MEASURED from public data and the local bake-offs)

**FLEURS** (Conneau et al., arXiv 2205.12446, 2022-05-25; CC-BY-4.0): the `te_in` and `en_us` dev and test transcript TSVs, fetched 2026-09-24 from huggingface.co/datasets/google/fleurs. 442 FLORES sentences are read by native speakers in both languages. Professional, faithful, formal written register, with 1 % Latin script:
- **1.62 aksharas per English syllable** (median 1.63).
- **The Telugu reading lasts 1.14x the English reading** (median 1.15).
- Fitted articulation: Telugu 5.91 aksharas/s, English 4.54 syllables/s. Durations include the recordings' edge silence, which the fit's intercept absorbs.

**Split into code-mixed and Telugu parts** (`gap4/split_ratio.py`). The "Telugu-part ratio" counts Telugu aksharas per English syllable not kept in English:

| Text | Aksharas/syllable | Latin share of units | Telugu-part ratio |
|---|---|---|---|
| FLEURS human, formal | 1.62 | 1 % | 1.63 |
| TranslateGemma 4B on podcast lines (ADR-015 run) | 1.71 | 0 % | 1.71 |
| TranslateGemma 12B, bake-off lines | 1.90 | 0 % | 1.90 |
| Qwen3.5-9B, bake-off lines | 1.60 | 6 % | 1.66 |
| Gemma 3 12B plain, bake-off lines (scripted English) | 1.38 | 16 % | 1.49 |
| Trace 13:44 (ADR-016 build) | 1.56 | 27 % | 1.98 |
| Trace 20:06 (Gemma 3 12B plain) | 1.22 | 19 % | **1.29** |

- **Reading (INFERRED from these MEASURED ratios):**
  - The 20:06 Telugu is about 20 % shorter per unit of meaning than formal human translation, and about 13 % shorter than the same model's output on clean scripted English.
  - Some of that is legitimate: colloquial spoken forms are shorter than the written standard, English terms stay at 1:1 syllables, and the prompt's "use the shorter phrasing when equally natural" rule.
  - Some of it is the P and m drops in E5.
- **Disfluency is not the explanation here.** Whisper transcripts carry few fillers; a heuristic count puts them at 2 % of source syllables.
- **What this implies for fill (INFERRED):** a faithful formal Telugu line read at native pace runs about 14 % longer than the English, so it would overfill. Maata's compact spoken Tenglish, on a clone that articulates about 10 % faster than the FLEURS readers (about 6.5 against 5.9 aksharas/s), underfills by 12 %. The right target lies between, and wording is the lever that moves it without touching the rate.

### E7. What primary sources say about the length policy

- **HOMURA / Sand-Glass** (Cui et al., Bilibili; arXiv 2601.10187 v3, 2026-09-03; read from the arXiv HTML). DISCLOSED and MEASURED:
  - **The budget is a two-sided interval.** The upper bound is fixed; the lower bound is relaxed for short utterances. It is not condense-only.
  - **Table 17** (Zh→En, synthesized with one fixed TTS voice against the original timestamps): median spoken duration ÷ budget is 1.32 for unconstrained LLMs, 1.13 for prompt compression, **0.87 for Best-of-N** graded variants, 1.10 for pipeline post-editing and 1.02 for HOMURA. The authors say Best-of-N fits only by systematically undershooting and leaves slots noticeably underfilled.
  - **For Maata (INFERRED):** the number matches Maata's 0.87, but the cause differs. In the 20:06 session the candidate path touched only 2 lines, and the underfill comes from the natural lines. The lesson is prospective: moving to 05 R1's graded variants with a "most complete that fits" rule would lock this undershoot in.
  - **The "cliff" pattern:** neighbouring prompted variants jump across the target interval (for example ratios 1.45 → 0.93 → 0.55), so a fixed set of shorter tiers cannot cover the moderate range where dubbing works.
  - **Two-stage translate-then-rewrite** shows drops in back-translation fidelity, which the authors attribute to "hallucinated omissions".
- **Lakew et al., "Machine Translation Verbosity Control for Automatic Dubbing"** (Amazon; arXiv 2110.03847, 2021-10-08; PDF read): the goal is within ±10 % of the source length, a two-sided band. The authors note that depending on the language pair either shortening or lengthening is needed.
- **Brannon, Virkar and Thompson, "Dubbing in Practice"** (TACL 11, 2023; arXiv 2212.12137 v1, 2022-12-23; PDF text re-read). MEASURED on 319.57 h of Amazon human dubs:
  - Human dubs are non-isometric in both directions: Spanish 30 % of lines ≥10 % longer and 42 % ≥10 % shorter; German 53 % and 16 %.
  - Duration follows content (r = 0.523), not rate (r = 0.163), and dub rate varies less than the source's.
  - Reference-free quality (comet-qe, prism-src) shows no meaningful on-screen/off-screen difference. So professional dubbers do not measurably trade translation quality for timing.
- **ElevenLabs Dubbing Studio docs** (fetched 2026-09-24, undated). DISCLOSED: the default Fixed Generations keep a clip's duration regardless of its text, so speech can speed up or slow down significantly. The page's own example says a short text in a long clip comes out sounding slow and stretched. Dynamic Generations fit the clip to the text instead, at the cost of sync. So ElevenLabs' default fills slots by rate in both directions, and the documented way out is to change the text or switch to Dynamic Generations (the second half is INFERRED from the page's advice).
- **Pérez, García and Villegas, QoMEX 2019** (UPM record; the table was re-read from the saved text, confirming 08 F13). MEASURED: MOS 4.14 at 0.90x against 4.37 at 1.10x (4.64 at 1.00). Any speed G > 1 rated better than 1/G.
- **Roberts & Paliwal, JASA 2020** (arXiv 2006.00848; per 08 F14): of an inverse pair, the slower is rated lower. Their ratios nearest Maata's range (0.78, 0.83) are slow-downs well outside what is proposed here.
- **ADR-018 Round 2** (Maata's own blind judging, 2026-09-24). MEASURED: the app's old prompt with a length note and lint retries gave 13.5 % major meaning errors against 5 % for the plain prompt. The average length barely changed (1.36 against 1.38 aksharas/syllable on the same 37 lines, computed here). The note cost meaning without buying length control.

---

## Resolving the contradictions between the reports

| Claim | Verdict | Why |
|---|---|---|
| 05 D4/R2: lower `DEFAULT_RATE` to about 3.4 | **Refuted** | The 3.4 is one excerpt's stitched guest reference (E2). The production voices, calibration takes and today's re-measurement give 5.4–6.0. It would put 93 of 98 lines over budget (E4). Keep 05 R2's *other* bullet: calibrate each voice from 2–3 sentences and seed the overhead as well. |
| 04 R1: never lengthen to fill a slot | **Partly right** | Right that padding with new words is harmful (ADR-018; HOMURA's rewrite omissions). Wrong as a blanket rule: human dubs run at least 10 % longer on 30–53 % of lines (Brannon), Amazon and HOMURA use two-sided targets, and in Maata up to 30 % of the underfill is content the speaker actually said and the dub left out. "Fuller" must mean *restore*, not *pad*. |
| 08 R2: a length target on every line, with long variants | **Right in direction; order and form need changing** | Adopt it only after the coverage gate (E5) exists. Use the target as a *selection* criterion over tiers generated in one call, not as a note on the line (ADR-018). Derive it from the calibrated rate, never from 3.4. |
| 03 R1(b), 09 R4, 08 R5: slow the audio to 0.85–0.92 | **Not now** | The underfill is textual and pause-placement (E5). The worst lines are emphatic pauses, where 0.9x moves the fill from 0.51 to about 0.57. Slowing is the less tolerated direction (Pérez; Roberts & Paliwal; ElevenLabs' own caution about over-slowed clips). Revisit 08 R5's 0.92 only as a blind A/B after the tiers and anchors. |
| 03 R1(c): a "match timing" arm with fuller phrasing and no new facts | **Adopt as the fuller tier** | This is the two-way tier, gated by coverage and additions checks. |
| 08 R1: anchors inside long lines | **Supported, and it becomes the main fix for long lines** | 20 of 27 long lines are complete. Their silence is pauses gathered at the end, not missing words. |

## Answers to the three decisions

1. **Translation contract: two-way tiers, completeness first. Not condense-only.**
   - In one `claude -p` call per scene (05 R1), return for each id: `full` (the natural, complete line), `fuller`, `concise` and `very_concise`.
     - `fuller` keeps every hedge, repetition and discourse marker the speaker actually used, uses natural spoken expansions (for example అంటే, కదా, the full verb forms), and adds no fact that isn't in the English.
     - Ask for `fuller` only when `full` is predicted to fill less than about 0.85 of a speech-dense slot.
   - Choose deterministically:
     - Take the most complete candidate whose predicted natural duration falls in a band of about [0.85, 1.10] × `target_s`.
     - If none does, take the closest from above only if the planner can absorb it within 1.2x and 0.6 s of lag; otherwise take the closest from below.
     - Measure fill against VAD speech time inside the span, not the raw span, so emphatic pauses aren't "filled" (08 F1 caveat).
2. **Audio-rate floor: keep 1.0.**
   - Treat long-line underfill with anchors and inserted pauses (08 R1) and short-line underfill with wording (the tiers).
   - Treat the voice's pace as a voice setting: cfg_weight and the reference choice, swept in 06 R2 and blind-judged for naturalness, not a planner knob.
   - An A/B at 0.92–0.95 on speech-dense lines is reasonable once the above is measured (08 R5). 0.85 is not.
3. **Coverage check first: yes, before any timing or length-target change.**
   - An id-keyed Claude judge over each scene returns, per line: missing content (with the English words), additions, and meaning errors (negation, numbers, hallucination). It back-translates only flagged lines.
   - A line with a missing clause gets one re-translation from its English with the missing words named, never a rewrite of the Telugu.
   - Log per-session coverage (share of lines C/m/P/E) next to fill in `maata-bench` and `verify-mac.sh`.
   - Why first:
     - It is the direct fix for "sentences incomplete".
     - It catches E-class errors (ids 48, 82) that no timing work touches.
     - It is the only guard against the next length lever trading content for fit, as the ADR-018 length note did.

## Implications for Maata (ordered)

1. **Keep `DEFAULT_RATE` at 5.5, or move it to 5.8 (the median of the logged calibration takes). Never 3.4.**
   - Calibrate from 3 fixed sentences of different lengths, so rate and overhead separate. One sentence misestimates the multi-line rate by −13 % to +18 % (E3).
   - Set the per-voice model directly from them rather than as one weak RLS update.
   - Key the estimator on (voice, cfg, exaggeration, reference hash), so a settings change re-calibrates.
   - Effort S. No approval needed. Record it against ADR-017.
2. **Build the coverage judge on the Claude CLI**, with the E5 classes, and run it on the 20:06 trace as its first regression fixture. The labels in `gap4/coverage.py` serve as the gold set. Effort M; it uses the maintainer's quota. Record it with the no-local-LLM ADR.
3. **Two-way tiers with band selection** (decision 1). Replace "most complete that fits" with the band rule, which avoids HOMURA's systematic undershoot of about 0.87. Effort S–M. Record it in ADR-018's successor. Blind-judge `full` against `fuller` for padding.
4. **Anchors and pause placement inside long and run-on lines** (08 R1), plus sentence-level translation before splitting (05 R3). Effort M. Needs an ADR, since it changes ADR-017.
5. **Run the voice-match sweep (06 R2) before tuning any fill numbers.** cfg alone moves pace by 3–9 % between 0.3 and 0.7 (E3), which shifts every budget by the same amount. Because budgets are derived from the calibrated rate, they follow automatically once calibration runs after the settings are fixed. Fill targets and bake-offs should still be re-measured after the sweep.
6. **Audio floor:** leave it at 1.0. Schedule a blind 0.92–0.95 A/B only after steps 2–4.

## Confidence

- **High:**
  - The voice-rate facts (three independent sources agree; E2 and E3).
  - Budgets are not binding in the current build (code plus trace).
  - The fall from 1.57 to 1.19 is a translation-length change, not a voice change.
  - 05 R2's 3.4 prior is wrong.
- **Medium:**
  - The split of the underfill: about 30 % dropped content at most, the rest compact but faithful Telugu plus pauses. It rests on one Claude judge pass and approximate syllable counts for dropped phrases.
  - The claim that the 13:44 build was Qwen3.5-9B (INFERRED from ADR-015 and ADR-018; the build is not in git and the log doesn't name the model).
- **Medium–low:**
  - The fill band [0.85, 1.10] (INFERRED from HOMURA, Lakew and Brannon, all in other language pairs).
  - Whether a Claude `fuller` tier stays free of padding for Telugu. Untested: it needs the blind A/B.
- **Low:** any claim about how natural conversational Telugu pace sounds. FLEURS is read, formal speech, and no Telugu conversational-rate study was reachable without web search.

## Open questions

1. How much of each long span is speech? A speech VAD or aligner on the vocal stem would let fill be scored against speech time (08 R3 and R8).
2. Does a Claude `fuller` tier restore dropped content without adding filler, as judged blind by native listeners?
3. After the 06 R2 sweep, which cfg does the maintainer prefer, and what pace does it give? That fixes the per-voice budgets.
4. Does moving from line-level to sentence-level translation (05 R3) by itself remove most P-class drops, which cluster on long, run-on or split lines?

## Sources (date = published or fetched)

- **Maata data** (read 2026-09-24):
  - `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl` (sessions 2026-09-24 13:44:57, 18:31:17, 20:06:51) and `pcm/*.npy`.
  - `~/Library/Logs/Maata/engine.log` (calibration lines 2026-09-24 18:32:17, 18:32:39, 20:08:15, 20:08:32).
  - `scratchpad/tts_pace.{py,json,log}` (2026-09-24 13:19), `scratchpad/voice_e3.{py,json}` (2026-09-24 14:55), and the bake-off outputs `scratchpad/mt_*.json` and `round*/mt_*.json`.
- **Maata repo** (working tree, read 2026-09-24): `engine/src/maata_engine/session.py` (`_translator`, `_dub`, `_calibrate`, `_build_voice`, `FIT_TOLERANCE`, `CALIBRATION_TE`), `timing/duration.py` (`DEFAULT_RATE = 5.5`), `timing/planner.py` (`target_seconds`), `text/tenglish.py` (`plain_messages`, `candidates_ask`), `backends/apple.py`, `backends/torch_common.py`, `docs/DECISIONS.md` (ADR-016, ADR-017, ADR-018).
- **New measurements:** `scratchpad/gap4/` (2026-09-24): `trace_stats.py`, `rate1.py`, `prior_sim.py`, `coverage.py`, `align.py`, `split_ratio.py`, `fleurs.py`, `pace_now.py` / `pace_now.json` / `pace_now.log`.
- **FLEURS:** Conneau et al., arXiv 2205.12446 (2022-05-25). Transcript TSVs `data/{te_in,en_us}/{dev,test}.tsv` from https://huggingface.co/datasets/google/fleurs (fetched 2026-09-24; CC-BY-4.0).
- **HOMURA:** Cui, Yu, Shi, Shi, Li (Bilibili), arXiv 2601.10187 v3 (2026-09-03), https://arxiv.org/html/2601.10187v3: §4 and §4.4, Appendix I (Observations 1–2), Appendix O Table 17.
- **Lakew et al.:** Lakew, Federico, Wang, Hoang, Virkar, Barra-Chicote, Enyedi, arXiv 2110.03847 (2021-10-08), https://arxiv.org/abs/2110.03847: §3 and §4.1.
- **Brannon et al.:** Brannon, Virkar, Thompson, TACL 11:419–435 (2023), arXiv 2212.12137 v1 (2022-12-23): §4.2, §4.3, §4.5.
- **ElevenLabs:** Dubbing Studio docs, https://elevenlabs.io/docs/eleven-creative/products/dubbing/dubbing-studio (fetched 2026-09-24, undated).
- **Chatterbox:** resemble-ai/chatterbox `README.md`, master, last changed 2026-07-21 (5de7a54a), fetched 2026-09-24.
- **Pérez et al.:** Pérez, García, Villegas, "Subjective Assessment of Adaptive Media Playout for Video Streaming", QoMEX 2019 (2019-06-05 to 07), https://oa.upm.es/64246/ (Table re-read 2026-09-24).
- **Roberts & Paliwal:** JASA 2020, arXiv 2006.00848 v2 (2020-07-16), as verified in 08 F14.
- **Sister reports** (2026-09-24): `05-llm-translation-claude-cli.md` (D4, R1–R3, C1), `08-timing-sync.md` (F1, F13, F14, R1, R2, R5), `04-academic.md` (R1), `03-other-commercial.md` (R1), `09-open-source-pipelines.md` (R4), `06-voice-tts.md` (R2), `sota_ranking.md`.
