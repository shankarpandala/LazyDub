# Gap 5: what "nowhere near the real voices" consists of, and the realistic EN→TE ceiling

Written 2026-09-24. It builds on `../sota_ranking.md` and reports 01, 04, 06 and 10, and corrects them where noted.

**Two kinds of evidence:**
1. **Primary sources**, listed with dates.
2. **A new scratch ablation**, run today on the maintainer's M5 Pro (24 GB). It used the same podcast excerpt, speakers, held-out split, 10 Telugu lines and WeSpeaker model as ADR-017's E3 experiment.
   - Files:
     - Scripts: `../gap5_ablate.py` and `../gap5_duration.py`.
     - Results: `../gap5_ablation.json` and `../gap5_duration.json`.
     - Listening files: `../gap5/*.wav`.
   - **These numbers are not committed.** Under CLAUDE.md they may not be quoted in an ADR until they are re-run and committed under `docs/spikes/results/`.

**Labels used below:**
- **DISCLOSED:** stated by a source.
- **MEASURED:** from today's scratch run.
- **INFERRED:** my reasoning.

---

## Question

What does "nowhere near the actual voices" consist of? The candidates:
- **Timbre:** S3Gen's reference and x-vector.
- **English accent or prosody:** carried in through T3's prompt and speaker embedding.
- **A compressed pitch range, or the wrong pace.**

And what WeSpeaker ceiling can EN→TE clones realistically reach? The effort lane and the metric that gates the bake-off depend on the answer.

## Answer, in brief

1. **The yardstick behind "host 0.645 = 86% of the 0.746 ceiling" is mis-specified.** Three separate faults (MEASURED; the code was read):
   - **0.746 is the guest's ceiling, not the host's.** It is the guest's mean clip-to-clip cosine from `voice_sim.py`, which I reproduced exactly (0.746). The host's own clip-to-clip ceiling is 0.658.
   - **0.746 is on a different basis from the take scores.** E3 scores each take against a centroid of held-out clips, and that basis runs higher:
     - Guest 9 s clips against the centroid: 0.811–0.816.
     - Host clips against the centroid: 0.704.
   - **The host's 0.645 is inflated by leakage.** It came from E3's "a-current" stitched reference, and **4.4 s of that 11.7 s reference is 67% of the 6.5 s of held-out audio it was scored against.**
     - Without leakage the host scores 0.56–0.57 (E3's b-closest 0.562; today's base arm 0.567).
     - The host has only 13.8 s of clean speech in the 5-minute excerpt, so every host number is fragile.
   - **Score depends heavily on clip length** (Carbonneau et al. 2025 found the same). The guest's real English clips score:

     | Clip length | 2 s | 3 s | 5 s | 9 s |
     |---|---|---|---|---|
     | Cosine to her centroid | 0.577 | 0.656 | 0.732 | 0.811 |

     Takes average about 5 s, so they must be compared with real clips of about 5 s.

2. **For the guest, the loss decomposes into two nearly equal parts: the acoustic stack and the language change. T3 conditioning barely moves the metric** (MEASURED, 9 s windows, WeSpeaker ResNet34):

   | Stage | Cosine to held-out English centroid | Change |
   |---|---|---|
   | Real English, 9 s | 0.811 | — |
   | Her own real English re-synthesised through S3Gen only (real tokens, E3 timbre span) | 0.678 | −0.133 (acoustic stack plus channel) |
   | Same voice, TTS of **English** text | 0.678 | 0.000 (T3 costs nothing in English) |
   | Same voice, TTS of **Telugu** text (the production path) | 0.506 | −0.172 (language) |
   | Swap only T3's prompt to a native Telugu clip | 0.480 | −0.026 |
   | Swap only T3's speaker embedding to a native clip | 0.477 | −0.029 |
   | cfg 0 / 0.3 / 0.5 | 0.495 / 0.503 / 0.506 | ≤0.011, within noise |
   | Exaggeration 0.7 with cfg 0.3 | 0.478 | −0.028 |
   | Swap only S3Gen's reference to a native clip | 0.193 | −0.313 (falls to the native speaker's own score, 0.191) |

   **The metric is almost entirely S3Gen's timbre plus a language penalty.**
   - That penalty is the same size that the literature finds for real bilingual speakers (see Evidence, E3).
   - ADR-017's native-prompt test could therefore not have shown a benefit: its metric cannot see what a prompt changes.

3. **A compressed pitch range is refuted for Maata's clones on gross statistics** (MEASURED).

   | Speaker | Semitone SD, English | Semitone SD, dub | p10–p90, English | p10–p90, dub |
   |---|---|---|---|---|
   | Guest | 3.25 | 3.48 | 7.2 st | 7.9 st |
   | Host | 3.24 | 3.44 | 7.7 st | 7.3 st |

   - Native IndicVoices-R extempore Telugu is much flatter: 2.05 st female, 1.84 st male. The fine-tune has not taken on that flatness.
   - ElevenLabs' 40%-narrower Telugu pitch range (PSP pilot) does not carry over to Maata's clones.
   - **What does differ:**
     - **Mean pitch:** both dubs sit about 1.1 semitones below the speaker's English. Guest 192 vs 205 Hz; host 110 vs 117 Hz; per take −2.5 to +0.6 st.
     - **Syllable-nucleus rate:** guest 4.3/s vs 3.8/s in English; host 4.0/s vs 2.8/s. Native Telugu runs about 4.1–4.4/s. This rough estimator counts more syllables per second in Telugu anyway.
   - Rhythm and contour shape are not measured yet.

4. **WeSpeaker is nearly blind to T3's prosody** (MEASURED, and consistent with Carbonneau et al.).
   - Swapping the host's T3 speaker embedding for a flat-speaking native Telugu male's cut the dub's pitch variability by 38%: 3.44 → 2.12 st SD, p10–p90 7.3 → 4.2 st.
   - Similarity barely moved: 0.567 → 0.557.
   - So a clone can get flatter or less "him" in delivery without WeSpeaker noticing, and humans do notice (Bakkouche et al. 2025).

5. **Realistic ceiling (INFERRED, low–medium confidence).** At 9 s, for the guest:
   - **Telugu output with the current Chatterbox acoustic stack: about 0.51.** She is already there. Her Telugu is 75% of the stack's English 0.678, and a real bilingual on a WavLM verifier keeps about 79% across languages (EMIME via VALL-E X). So little of the drop is fixable within this stack.
   - **With a near-perfect timbre stack: about 0.60–0.64** (0.811 × 0.75–0.79).
   - So on WeSpeaker the headroom is about +0.1, and it must come from the S3Gen/acoustic side.
   - The host cannot be assessed until a longer excerpt gives at least 30 s of held-out clean speech.

6. **What the maintainer probably hears (INFERRED).** The metric cannot separate these; ranked by evidence:
   - **(a) Timbre and channel fidelity of S3Gen.** Measured: −16% even for his own real English.
   - **(b) Telugu speaking style that is not the speaker's own.** Pace in syllables per second, a mean pitch about 1 st low, and rhythm. The rhythm part is plausible but not yet measured.
   - **(c) Accent.** Unmeasured. The T3 prompt and embedding carry it, but WeSpeaker cannot score it.
   - **(d) Channel mismatch.** A clean vocoder against podcast room acoustics; report 07's EQ and room matching is the lever.
   - Humans recognise real bilinguals across languages well: SMOS 4.91/5 on EMIME, where the verifier gave only 0.58. **The language change by itself should not make a clone sound "nowhere near."** Things WeSpeaker under-weights should.

**So:**
- **Effort:**
  - The WeSpeaker-addressable effort goes to the acoustic/timbre lane.
  - The perceptual effort goes to prosody and accent, gated by listeners.
- **The bake-off gate:**
  - Primary: a blind "sounds like him/her" test plus native-listener accent and naturalness ratings.
  - Guards: duration-matched WeSpeaker as a fraction of each speaker's own ceiling, and CER.
- **Cost lanes:** see "Implication for Maata" below.

---

## Evidence

### E1. Maata's own measurement: what ADR-017 actually recorded (MEASURED and code read, 2026-09-24)

**The raw numbers exist, just not in the repo.**
- `voice_e3.py`, `voice_e3.json` and `voice_e3.log` are in the scratchpad.
- `docs/spikes/` does not exist at all, so ADR-017's reference to `docs/spikes/results/` points nowhere.

**This closes 01 U7 and 06 B6.** ADR-017's voice-match comparison used:
- **Metric:** the pyannote community-1 embedding (WeSpeaker ResNet34-LM, `window="whole"`), cosine to a centroid of every other reference clip (`clips[1::2]`), never used for conditioning.
- **Sample:** 10 Telugu lines, one seed.
- **Conditions:**
  - "b-closest": English prompt, cfg 0.5.
  - "c-balanced": native prompt **and** cfg 0.3.
  - "d-natural": native prompt **and** native T3 embedding **and** cfg 0.3.

So the native-prompt test changed two or three factors at once, on one seed, with a metric that cannot see accent. Today's one-factor arm (native T3 prompt only, cfg 0.5) cost 0.021–0.026 for the guest and 0.007 for the host: small and within noise for the host.

**Reference clean audio in the 5-minute excerpt (`podcast_600_900.wav`), per `reference_clips(target=60, max_clip=10)`:**

| Speaker | Clips | Held out | Pool |
|---|---|---|---|
| Guest (S1) | 7 × about 9 s | 27.6 s | 36.7 s |
| Host (S2) | 4 clips (4.7, 4.4, 2.6, 2.1 s) | 6.5 s | 7.3 s |

**The host's E3 "a-current" reference** (`reference_clips(target=10)`, 11.7 s) contains 4.4 s of the host's held-out clips, 67% of them. The guest's contains none.

**Ceilings** (real vs real, WeSpeaker):

| Basis | Guest | Host |
|---|---|---|
| Clip-to-clip mean (`voice_sim.py` basis, which gives the "0.746") | **0.746** (reproduced) | 0.658 |
| Pool clip to held-out centroid (the basis takes are scored on) | 0.816 | 0.704 |
| Real crops to centroid, 2 / 3 / 4 / 5 / 6 / 9 s | 0.577 / 0.656 / 0.692 / 0.732 / 0.765 / 0.811 | 0.659 / 0.735 / 0.767 (n = 4 / 2 / 1) |
| Floor: guest centroid vs host centroid | 0.112 | 0.112 |

**Reproducibility.** Today's base arm (b-closest settings, 2 seeds × 10 lines) gave 0.462 per take for the guest (E3: 0.466) and 0.567 for the host (E3 b-closest: 0.562). The seed-to-seed spread on the same line averaged 0.052 (guest) and 0.041 (host).

### E2. One-factor ablation and the decomposition (MEASURED, 2026-09-24)

**Setup:**
- Chatterbox-Telugu fp32 on MPS, T3 in bf16, 10 CFM steps.
- 2 seeds × 10 lines per arm; each take scored against the E3 held-out centroid.
- Scored two ways: per take (about 5 s each), and in 9 s windows made by joining consecutive takes with the gaps removed. Windows cancel the length confound.

| Arm | Guest, per take | Guest, 9 s | Host, per take | Host, 9 s |
|---|---|---|---|---|
| Real English (ceiling, same basis) | 0.732 at 5 s | 0.811 | about 0.77 at 4 s (n = 1) | n/a |
| S3Gen resynthesis of own English | 0.673 (9 s clips) | 0.678 | 0.539 (one 2.6 s clip) | n/a |
| Base: timbre span, pooled embedding, English prompt, cfg 0.5 | 0.462 | 0.506 | 0.567 | 0.620 |
| Base voice, English text | 0.625 | 0.678 | 0.571 | 0.637 |
| cfg 0.3 | 0.457 | 0.503 | 0.569 | 0.603 |
| cfg 0 | 0.474 | 0.495 | 0.564 | 0.585 |
| Exaggeration 0.7, cfg 0.3 | 0.443 | 0.478 | 0.572 | 0.596 |
| T3 prompt → native clip | 0.441 | 0.480 | 0.560 | 0.602 |
| T3 speaker embedding → native clip | 0.431 | 0.477 | 0.557 | 0.604 |
| S3Gen reference → native clip | 0.189 | 0.193 | 0.362 | 0.374 |
| That native clip vs the speaker's centroid | 0.191 | — | 0.274 | — |

"Per take" is averaged over 20 takes (2 seeds); the "9 s" columns use seed 0 only.

**Readings:**
- **S3Gen's reference sets the WeSpeaker identity.** Replacing it drops the score to the native speaker's own level.
- **T3's prompt and speaker embedding move WeSpeaker by at most 0.03.**
- **cfg changes pace and pitch spread, not similarity:**
  - cfg 0 made lines 21% (guest) and 27% (host) longer.
  - It widened the semitone SD to 3.94 (guest) and 4.53 (host).
  - For timing that is the wrong direction.
- **The language change costs the guest 0.17 at 9 s but the host almost nothing (0.02).** The host's centroid rests on 6.5 s of audio, so that contrast is not reliable. Whether the host's English is Indian-accented, which could shrink the gap, is unknown.
- **Real English tokens carry some identity.** S3Gen resynthesis of the speaker's real English with a native timbre still scored 0.358 (guest) and 0.385 (host), 0.11–0.17 above the native floor. T3-generated Telugu tokens carried almost none: 0.189 against 0.191 for the guest.
  - It is unresolved whether that token-borne part is the speaker's rhythm and prosody or just the English language matching an English centroid. No arm separated the two.

**Selection bias** (pooled over 120 Telugu takes per speaker):
- Take similarity correlates with take length: r = 0.31 (guest), 0.24 (host).
- With pitch variability it correlates inconsistently: r = −0.20 (guest), +0.15 (host).
- **Picking the best of N takes by WeSpeaker therefore favours longer takes. Score candidates at matched length.**

### E3. The cross-language penalty exists for real people, and humans overcome it far better than verifiers

**Buitrago & Hernando, "Disentangling Speaker and Language Effects in Cross-Lingual Speaker Verification for Iberian Languages"** (arXiv 2607.01161, 2026-07-01; html fetched 2026-09-24).
- **DISCLOSED:**
  - Bilingual same-speaker sets: es-ca (300 speakers), es-eu (64), es-pt (40), es-gl (21). HuBERT-based verification.
  - The authors conclude that speaker variability explains only part of the degradation, and language mismatch is the main driver.
  - Results are given as relative-degradation and AUC matrices, not cosines. This corrects 10 C5's implied cosine reading.
- **INFERRED:** it supports the direction of the penalty, not its size.

**Zhang et al., VALL-E X** (arXiv 2303.03926v1, 2023-03-07, Table 4 and §5.4; PDF text read 2026-09-24).
- **DISCLOSED:**
  - On EMIME's bilingual Chinese/English speakers, a WavLM-based verifier scored the same real speaker's two languages (tgt vs src) at 0.58 ± 0.09.
  - Human similarity MOS between the source prompt and the ground-truth target was 4.91 on a 5-point scale.
- **INFERRED:**
  - On a WavLM-SV scale, real same-language pairs score about 0.73–0.76 (the seed-tts-eval "Human" rows; 06 D3; different data).
  - So real bilinguals keep about 0.58/0.73 ≈ 79% across languages. Maata's within-model Telugu/English ratio for the guest is 0.506/0.678 = 75%.
  - The penalty is close to what a real bilingual would show, so it is mostly not fixable.
  - Humans still hear the same person, so the penalty is not what makes a clone sound unlike the speaker.

**Winters, Levi & Pisoni, JASA 2008** (doi 10.1121/1.2913046, abstract via PubMed E-utilities, fetched 2026-09-24).
- **DISCLOSED:** listeners can generalise a bilingual talker's identity across languages using language-independent cues. Their same/different judgements also shift with whether the two languages match, and they lean on language-dependent cues in familiar languages.
- **INFERRED:** a native Telugu listener who knows the speaker from English partly judges "same voice" on language-dependent habits, which a clone has to earn.

**Ordin & Mennen, JSLHR 2017** (doi 10.1044/2016_jslhr-s-16-0315) and **Mennen, Schaeffler & Docherty, JASA 2012** (doi 10.1121/1.3681950). Crossref metadata and abstracts, fetched 2026-09-24.
- **DISCLOSED:**
  - Bilinguals, especially women, use different F0 ranges in their two languages.
  - English and German speakers differ in F0 span.
- **INFERRED:** the right Telugu pitch target lies between the speaker's English and Telugu norms. "Match the English F0 SD exactly" is not the goal, and an unexplained constant offset (the −1.1 st here) is still worth testing.

### E4. What speaker embeddings see and miss, and what listeners use

**Carbonneau, van Niekerk, Seuté, Letendre, Kamper & Zaïdi, "Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis"** (arXiv 2507.02176, 2025-07-02; accepted at SSW13, the Interspeech 2025 synthesis workshop; abstract and HTML fetched 2026-09-24).
- **DISCLOSED:**
  - Across seven verification embeddings (GE2E, x-vector, ResNet-TDNN, ECAPA and WavLM variants), static spectral traits are well predicted from the embedding: mean pitch, HNR, shimmer, α-ratio.
  - Dynamic traits are poorly predicted: speech rate, voiced/unvoiced durations, pitch variation, loudness variation.
  - Duration is a confound: shorter files score lower. The mitigation is matched lengths, or synthesising the same texts as the real corpus.
  - Equalisation strongly affects GE2E.
  - They propose U3D, a language-agnostic rhythm distance over unit durations.
- **Correction:** the exact R² values are only in their Fig. 1. The figures in my first fetch were not in the text, so none are quoted here.
- **INFERRED:**
  - Chatterbox's T3 VoiceEncoder is a GE2E-style 3-layer LSTM (its `similarity_weight` and `similarity_bias` parameters, read in the installed package). Pooling it over 60 s of podcast audio also pools channel.
  - The host embedding-swap arm (E2) is a Maata-specific demonstration: pitch variability −38%, similarity −0.01.

**Bakkouche, McGhee, Lau, Cooper, Luo, Rees, Alter, Post & Schwarz, "Finding the Human Voice in AI"** (Interspeech 2025, ISCA Archive bakkouche25, August 2025; page fetched 2026-09-24; details via 04 F22).
- **DISCLOSED:**
  - Human speech with its F0 variation cut to 30% was rated no more similar than StyleTTS-2 or XTTS-v2 clones.
  - ElevenLabs was not distinguishable from human.
  - The authors conclude that prosody is key to perceived naturalness and similarity.

**Mou et al., "Dynamic Prosody Prediction in LLM-based TTS for Improving Speaker Similarity"** (arXiv 2606.15267, 2026-06-13; abstract fetched 2026-09-24).
- **DISCLOSED:** predicting per-syllable prosody from the speech already synthesised improved speaker similarity on three datasets. The abstract gives no numbers.
- **INFERRED:** style-specific prosody is a recognised lever on similarity in LLM-style TTS such as Chatterbox's T3.

**Menta, PSP** (arXiv 2604.25476v1, 2026-04-28; HTML fetched 2026-09-24; also 01 F17).
- **DISCLOSED:**
  - ElevenLabs Telugu log-F0 range 0.87 vs 1.44 native, nPVI 92 vs 107. These are 10-utterance pilots by a single author with a competing product.
  - With Chatterbox (a Praxy LoRA), exaggeration 0.7, temperature 0.6 and min_p 0.1 was the winning setting.
- **INFERRED:** this was a commercial model's failure. Maata's dubs are not compressed (E2 and the table in the summary), so this candidate is refuted for Maata. The nPVI/rhythm half remains untested.

### E5. Mechanism: where Chatterbox puts identity, accent and prosody

**Installed package** (`engine/.venv/.../chatterbox`, read 2026-09-24).
- **S3Gen** (`s3gen.py`, `embed_ref`) conditions on:
  - a 24 kHz prompt mel of at most 10 s (longer triggers a warning);
  - a CAMPPlus x-vector from the same clip;
  - that clip's S3 tokens.
- **T3** (`T3Cond`) takes:
  - a VoiceEncoder speaker embedding (Maata averages it over up to 60 s);
  - 6 s of prompt speech tokens;
  - `emotion_adv`.
- Maata's `prepare_voice_parts` wires these as ADR-017 describes (`torch_common.py`). The S3Gen x-vector comes only from the single timbre span, never pooled.

**Du, Wang, Chen et al., CosyVoice 2** (arXiv 2412.10117v3, 2024-12-25; HTML fetched twice, 2026-09-24).
- **DISCLOSED:** CosyVoice 2 dropped the utterance-level speaker vector from its text-speech LM. The authors found it also carries language and paralanguage information, which harms prosody naturalness and cross-lingual ability.
- The flow-matching stage keeps the speaker embedding, tokens and prompt features.
- **INFERRED:** Chatterbox's T3 keeps such a vector. Maata feeds it English-only audio, which is a plausible path for English prosody and accent into Telugu. E2 shows the vector does steer prosody (the host arm).

**Resemble AI, Chatterbox README** (raw GitHub, undated, fetched 2026-09-24).
- **DISCLOSED:** a reference in a language other than the language tag can make the output inherit the reference's accent; setting cfg_weight to 0 is suggested. The README also gives the cfg 0.3 advice for fast speakers and the cfg 0.3 / exaggeration 0.7 advice for expressive speech, and says higher exaggeration speeds speech up.
- **MEASURED today:** on Chatterbox-Telugu, cfg 0 left WeSpeaker unchanged and slowed speech by 21–27%. Whether it reduces accent needs native listeners.

**ElevenLabs Dubbing docs** (`/docs/overview/capabilities/dubbing`, undated, fetched 2026-09-24).
- **DISCLOSED:** the cloning-strength range is 0–10, default 7. Higher values favour resemblance but can sound less natural in phonetically distant languages and carry over more of the source accent.
- **INFERRED:** the industry leader treats similarity and accent as a trade-off tuned by ear, not as one score.

**Community reports** (GitHub issues, resemble-ai/chatterbox, fetched 2026-09-24; anecdotal, low weight):
- #311 "Polish language has strong English accent" (2025-10-04).
- #551 "V3 doesn't appear to work as well as V2" (2026-08-18, Danish).
- **INFERRED:** accent carry-over in the multilingual model is a known user complaint. V3 is not a guaranteed improvement.

### E6. How much best-of-N can buy on WeSpeaker (MEASURED, cross-checked)

- **Per-line seed spread** is 0.04–0.05 (E1), and take SD within an arm is 0.04–0.06.
- **Normal order statistics** put the expected best of 3 at +0.85 SD, about +0.03–0.05, and the best of 32 at about +2 SD, about +0.1.
- **Meta's Text-Audiobox** (arXiv 2609.03992, submitted 2026-09-03; numbers per 04 F15) reported similarity rising 0.66 → 0.74 at N = 32 (+0.08), consistent with that estimate.
- **INFERRED:** best-of-3 is a small, real timbre-lane lever, provided candidates are scored at matched length (E2's length bias).

---

## Confidence

| Claim | Confidence | Why |
|---|---|---|
| The 0.746 ceiling is the guest's clip-to-clip mean, not the host's and not the scoring basis | High | Code read. Reproduced to 3 decimals |
| The host's 0.645 includes leakage (67% of held-out audio in the reference), and the host has only 13.8 s of clean speech | High | Computed from the same registry calls E3 made |
| Duration strongly confounds the score | High | Literature (Carbonneau 2025) plus a monotone local curve with n = 4–32 crops per length |
| S3Gen's reference dominates WeSpeaker; T3 prompt, embedding and cfg move it ≤0.03 | Medium–high | Two speakers, 20 takes per arm, effects far larger (0.27–0.31) than noise (about 0.012 SE), but one excerpt |
| Guest decomposition: stack −0.13, language −0.17 at 9 s | Medium | One speaker, 3 resynthesis clips, 10 lines |
| Host decomposition | Low | Centroid from 6.5 s; resynthesis on one 2.6 s clip |
| No pitch-range compression; mean pitch about 1.1 st low | Medium–low | pyin on 20 takes per arm and 4–7 English clips. The syllable-nucleus rate is a rough estimator |
| The language penalty is mostly irreducible (about 75–79% retention) | Low–medium | Cross-dataset inference (EMIME, WavLM-SV, zh/en) plus one within-model ratio. Needs the bilingual recording |
| Humans perceive the clone gap through dimensions WeSpeaker under-weights | Medium | Bakkouche 2025, Carbonneau 2025, EMIME SMOS 4.91, and today's host embedding swap. Not yet a Maata listening result |

**Key claims verified twice:**
- **0.746:** code plus reproduction.
- **E3 guest score:** reproduced (0.466 → 0.462 per take, 0.467 when re-scored from E3's WAVs).
- **Carbonneau:** abstract and HTML. The R² figures from the first fetch were dropped as unverifiable.
- **CosyVoice 2 quote:** two fetches.
- **VALL-E X numbers:** PDF text and abstract page.
- **PSP:** HTML and 01 F17.
- **Bakkouche:** ISCA page and 04 F22.

---

## Implication for Maata

### 1. Fix the yardstick before any voice bake-off. Effort S, no approval.
- **Score at matched length.** Join takes into 9 s windows and compare with each speaker's own 9 s real-clip-to-centroid ceiling. Or crop real clips to each take's length. Report similarity as a fraction of that speaker's ceiling.
- **No shared audio.** Conditioning audio and scoring audio must never overlap; assert it in code.
- **Enough audio.** At least 30 s of held-out clean speech per speaker, which means the full podcast rather than a 5-minute excerpt. At least 20 lines × 2 seeds per arm.
- **Retire the host's 0.645 and the "86% of 0.746" framing.** On the corrected basis:
  - The guest's production voice is about 62–65% of her same-language ceiling.
  - The host's is about 75–80% of a ceiling measured on almost no audio.
- **Commit** scripts, JSON and the WAV manifest under `docs/spikes/results/<machine>/`. ADR-017's citation points at a directory that does not exist.

### 2. Record the bilingual ceiling. Effort S, no approval, maintainer's time. (06 R1, 10 R6)
- The maintainer reads matched passages: about 10 × 9 s in English and about 10 × 9 s in natural spoken Telugu.
- Measure English-vs-English, Telugu-vs-English-centroid and clone-vs-English-centroid in 9 s windows.
- This one number decides whether the ~0.1 of timbre headroom estimated above is real, or whether the Telugu clone is already at the bilingual ceiling.

### 3. The WeSpeaker-addressable effort goes to the timbre/acoustic lane. Effort S–M, no approval.
- **Test the S3Gen and vocoder options under the fixed yardstick:**
  - Choose the S3Gen span by cosine to the centroid, not only by cleanliness (06 R3).
  - Pool S3Gen's CAMPPlus x-vector over the clean clips. Today it comes from the one span; T3's embedding is already pooled.
  - Try the `s3gen_v3` vocoder (sota arm 1).
- **Resynthesis as a stack gauge.** Re-synthesising the speaker's own English is a cheap direct measure of the stack: 0.678 against 0.811 real. Any S3Gen change should raise it before being tried on Telugu.
- **Match the output channel** with EQ, room and loudness against the source (07). WeSpeaker compares clean vocoder output with podcast audio, and listeners hear the room too.
- **Best-of-3 takes,** scored at matched length (E6): about +0.03–0.05 expected.

### 4. The perceptual effort goes to prosody and accent, gated by listeners, not WeSpeaker. Effort S–M, no approval.
- **cfg sweep: done here for similarity.** 0, 0.3 and 0.5 are equal on WeSpeaker. Choose by native-listener accent and naturalness ratings, and by the timing planner, because cfg 0 lengthens lines 21–27%. Wire `cloneStrength` only after that listening test.
- **Mean pitch:** A/B a +1 semitone post-shift, or choose the take nearest the speaker's median F0. Both dubs sit about 1.1 st low. Cheap and specific.
- **Pace and rhythm:**
  - Add syllable rate relative to each language's norm (04 F17: Seamless SRD_norm).
  - Add a rhythm distance: U3D (Carbonneau) or nPVI (PSP).
  - These run on every take in numpy; they diagnose, they don't gate.
- **Per-line T3 prompt (06 R5, 01 R4) and the V3 T3 retrain (06 R6)** are prosody and accent levers, not similarity levers: T3 moves WeSpeaker by ≤0.03.
  - Judge them with listeners and CER.
  - Guard each with a pitch-variability check, because T3's speaker embedding alone halved the host's pitch spread without moving WeSpeaker.
- **Re-test native Telugu prompts** one factor at a time at cfg 0.5 and cfg 0. Their WeSpeaker cost is ≤0.03, and ADR-017's rejection was made on a metric that cannot see what they change (accent).

### 5. Defer per-speaker RVC (06 R7).
- It would attack the ~16% stack loss. But a converter trained on each speaker's English audio would plausibly raise WeSpeaker partly by pulling Telugu toward English phonetics, i.e. by gaming the language component.
- Post-hoc conversion also has a record of lost intelligibility (04 F17; kNN-VC in 06 F7).
- Consider it only after item 3 is exhausted, and only with CER and native-accent gates plus approval for the new dependency.

### 6. What gates the voice bake-off
- **Primary:** a blind listening test. MUSHRA-style "sounds like him/her" against the speaker's real English, with the maintainer's own Telugu recording as a hidden reference. A separate native-listener rating covers Telugu naturalness and accent.
- **Guards:**
  - Duration-matched WeSpeaker as a fraction of each speaker's ceiling: no regression beyond about 0.02 (the seed noise).
  - Telugu CER.
- **Diagnostics, not gates:** F0 median offset (st), semitone-SD ratio, syllable-rate ratio against Telugu norms, energy range, rhythm distance.
- The listening material for the first round already exists in `../gap5/`, 8 arms × 2 speakers. It includes resynthesis and English-text references.

### Cost lanes, re-ranked by this diagnosis
1. **Small:** fix the yardstick; the bilingual recording; the +1 st pitch test; cfg decided by ear (the similarity sweep is done).
2. **Medium:** S3Gen span selection and x-vector pooling, `s3gen_v3`, channel matching, best-of-3 at matched length, per-line prompts gated by listeners.
3. **Large:** the V3 T3 retrain. It is a prosody and accent lever, so only after listeners show T3-side deficits.
4. **New dependency:** RVC, last.

## Corrections to the earlier reports
- **Task framing and 06 sweep F9:** "host 0.645 = 86% of the 0.746 ceiling" is wrong three ways: another speaker's ceiling, a different basis, and leakage (E1).
- **10 C5:** arXiv 2607.01161 reports relative-degradation and AUC figures, not cosines. It supports the direction of the language penalty, not its size (E3).
- **01 U7 and 06 B6, now resolved:**
  - ADR-017 used WeSpeaker similarity to a held-out centroid, with native prompts at cfg 0.3, 10 lines and 1 seed.
  - Its "did not help" conclusion is confounded, and the metric cannot see the intended benefit (E1, E2).
- **04 F22 inference** ("match each clone's pitch range too"): Maata's clones are not range-compressed; the measurable prosodic offsets are mean pitch and syllable rate (E2).
- **06 new finding 6** (similarity selection may favour English-accented takes): still plausible but untested. The measured bias is toward longer takes (E2).

## Still open
1. The bilingual ceiling for the maintainer's own voice (item 2).
2. Whether the token-borne identity in real English (+0.11–0.17 above the floor) is speaker rhythm or just the English language. One more arm settles it: another speaker's English tokens with a native timbre, scored against this speaker.
3. Why the host showed no language penalty. It needs the full-length podcast.
4. Listener results for cfg 0, the native prompt, exaggeration 0.7 and the +1 st shift.
