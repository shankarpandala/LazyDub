# 04: Academic automatic-dubbing research, 2018–2026 (verified)

Verified 2026-09-24. The source draft is `04-academic.sweep.md`, which has 25 findings, plus the researcher's JSON summary.

**How it was checked:**
- Every number a recommendation depends on was re-read in the paper text. The texts are in `scratchpad/acad/*.txt` and `scratchpad/dub_research/*.txt`, extracted from the arXiv and ACL PDFs.
- Titles, authors, dates and venue comments come from the arXiv API (`scratchpad/acad/arxiv_meta.xml`, fetched 2026-09-24), the ISCA Archive, the ACL Anthology and CVF Open Access.
- Licences come from the Hugging Face model API and `gh api`.
- Repo claims were read in `docs/DECISIONS.md`, `engine/src/maata_engine/session.py` and `timing/planner.py`.
- The web-search budget ran out during this pass. After that, sources were fetched directly: arXiv API, ISCA/ACL/CVF pages, Hugging Face and GitHub.

**Verdict markers:**
- **[confirmed]**: matches the primary source.
- **[corrected]**: the substance holds, but a number, attribution, venue or scope was wrong. The correct fact is given.
- **[unverifiable]**: could not be confirmed from a primary source.

Refuted sub-claims are listed separately. Each finding says whether it is disclosed (measured or stated in the paper) or inferred (the researcher's or verifier's application to Maata).

---

## Summary

**Tally:**
- Findings: 18 confirmed, 7 corrected, 0 refuted outright.
- Side claims (licences, repo facts, "not found" claims, venues): 6 confirmed, 4 unverifiable.
- Four sub-claims inside corrected findings were wrong; they are listed under **Refuted sub-claims**.
- The verifier added 5 findings the draft missed (N1–N5).

**What still stands, and it points one way for an audio-only YouTube dub:**
- **Meaning and a steady, natural speaking rate come first.** Timing is fixed mostly in the wording, not by squeezing the audio.
  - Professional dubbers keep their rate steadier than the source and give up timing before meaning (F3).
  - Forced alignment that makes the rate uneven lowered native ratings (F1).
  - Near-perfect isochrony sounds robotic (F10).
  - For off-screen speech, borrowing silence across neighbouring gaps wins clearly (F2).
- **Character-length targets don't buy timing** (F4).
- **What works is a duration budget in the target voice's units plus choosing among wordings** (F7, F19, F24). Two additions sharpen this for the Claude CLI plan:
  - N1: prompted frontier LLMs, Claude 3.5 Sonnet included, controlled duration worse than small fine-tuned systems. The budget must be computed locally, given to Claude as a number, and the result measured locally. Claude's own sense of length can't be trusted.
  - Asking an LLM for "simple" wordings shortened lines, but cost a little meaning and TTS intelligibility (corrected F9).
- **Best-of-N TTS takes are the cheapest lever on voice similarity and dropped words** (F15, Meta 2026). The rule is to filter by speaker similarity, then pick the lowest back-transcription error.
- **Perceived similarity depends on intonation, not only timbre** (F22). In a new large Indic study that includes Telugu, expressiveness and intelligibility drove listener preference (N3).
- **Evaluation:** UTMOS and NISQA don't predict human judgements (F16). Equal-weight metric averages track humans poorly (F21). COMETKiwi and XCOMET are CC-BY-NC-SA-4.0 and can't be used. `Unbabel/wmt22-comet-da` is Apache-2.0 (N4).

**Key corrections:**
1. **F5 (pause markers in MT):** the draft's "48.4 vs 30.8 / 61.4 vs 25.8" figures compare translate-then-align against length control, not the pause-marker system.
   - Pause markers vs length control is 51.7 vs 30.1 (De) and 60.9 vs 22.0 (Fr).
   - Once the relaxation step (silence borrowing) was added, pause markers lost their edge. The better-translating system won on French (45.0 vs 35.1, p<0.01) and tied on German.
   - Maata's planner already borrows silence, so pause markers drop to a low-priority experiment.
2. **F23 (background and "studio-clean" speech):** in HoliDubber's zero-shot table, listeners gave the studio-clean FunCineForge the **highest** MOS. HoliDubber led only in its joint speech-plus-ambience mode, and only by small margins. "Over-smoothed" is the authors' reading of UTMOS, not a listener result. The HW-TSC long-form system's source separation was never evaluated.
   - The only measured support for adding background is Federico 2020: +10.34 MUSHRA for non-native listeners, and a non-significant +1.05 for native listeners.
3. **F11 (speed-up hurts similarity):** Microsoft changed the rate by scaling phoneme durations inside a zero-shot TTS, not by speeding up finished speech. Applying it to Maata's post-hoc mel resampling is an inference and must be measured.
   - N2 adds that non-uniform (per-phoneme) stretching beats uniform stretching at 1.1–1.4x (Amazon, ICASSP 2022).
4. **F9 (HW-TSC pause-aware):**
   - The numbers are from the DE→EN subtask (NLLB + LLM post-editor + MeloTTS + OpenVoice), not the EN→ZH track in F8.
   - The "simple" variant lost COMET (89.29 → 88.01) and raised CER.
   - Only the unconstrained post-edit kept meaning.
5. **Venues and dates:**
   - Chronopoulou et al. 2023 is an arXiv preprint. It is absent from the Interspeech 2023 proceedings and cited only as arXiv by later papers.
   - VoiceCraft-Dub is ICCV 2025, not CVPR 2025.
   - InstructDubber is AAAI 2026, and Dasare et al. is ICASSP 2026.
   - F25 energy correlations are 0.174–0.183, not 0.176–0.186.

---

## Findings

### F1 [confirmed] Forcing phrase-level prosodic alignment lowered native listeners' ratings; adding background and reverb helped non-native listeners
- Source: Federico, Enyedi, Barra-Chicote et al. (Amazon), "From Speech-to-Speech Translation to Automatic Dubbing", IWSLT 2020 (ACL Anthology 2020.iwslt-1.31). arXiv 2001.06785, v1 2020-01-19, v3 2020-02-02.
- **Disclosed:**
  - EN→IT TED clips, MUSHRA, 14 listeners (5 Italian, 9 non-Italian), 657 ratings, mixed-effects models (Table 2).
  - A→B (length-controlled MT plus prosodic alignment): non-Italian +1.14 (n.s.), Italian −10.93 (p<0.01).
  - B→C (U-Net background plus re-reverberation from a blind RT60 estimate): non-Italian +10.34 (p<0.05), Italian +1.05 (n.s.).
  - Italian listeners' comments blamed a speaking rate that was too slow, too fast or uneven.
  - Duration was fitted by spline-resizing the mel before the vocoder. The authors observed, with no numbers given, that this sounded better than time-stretching.
- Repo check: ADR-017 also speeds up by resampling the mel before the vocoder (CosyVoice's method). Maata already uses this family of method, not time-stretching. The draft JSON's wording implied the opposite.
- **Inferred for Maata:** uneven rate is the failure to avoid.

### F2 [confirmed] For off-screen speech, relaxing alignment globally across inter-phrase and inter-sentence pauses clearly beats strict per-sentence isochrony
- Source: Virkar, Federico, Enyedi, Barra-Chicote (Amazon), "Prosodic Alignment for Off-screen Automatic Dubbing", Interspeech 2022 (ISCA Archive virkar22). arXiv 2204.02530, 2022-04-06.
- **Disclosed:**
  - Pauses are silences of at least 300 ms.
  - Off-screen score (eq. 7): 1 if the target rate r ≤ 1, 2 − r for 1 < r ≤ 2, and 0 above 2.
  - 20 native raters. EN→FR/IT/DE/ES on MuST-C (TED) and 3 YouTube vlogs, 15 four-sentence clips per domain.
  - ON/OFF won 36.7–68.7% of comparisons vs 15.3–36.3% for ISO, all p<0.01. Example: IT YouTube (D1) 6.16 vs 5.03.
  - Smoothness rose 9.9–28.3% relative (MuST-C D1).
  - The human evaluation used post-edited translations, so it isolates timing.
- **Inferred:** a muted-player dub of podcasts or explainers is effectively off-screen dubbing.

### F3 [confirmed] Professional dubbers keep speaking rate steadier than the source and break timing instead; character-length matching is a poor proxy for timing
- Source: Brannon, Virkar, Thompson, "Dubbing in Practice", TACL vol. 11, pp. 419–435 (2023). arXiv 2212.12137, 2022-12-23.
- **Disclosed:**
  - 319.57 h from 54 titles.
  - Speech-time overlap: mean 0.658, median 0.731. On-screen 0.684 vs off-screen 0.662 (+3.3%).
  - Character-length ratio vs overlap: r = 0.279.
  - Standard deviation of dub speaking rate is lower than the source's: ES 1.25 vs 1.47 w/s, DE 1.26 vs 1.46 (percentile bootstrap, p<1e-10).
  - No on-screen vs off-screen difference in COMET-QE or Prism.
  - Line-level source→dub correlations: rate r = 0.439 (0.584 for lines of at least 1 s), mean pitch 0.792, pitch std 0.429.
  - About 12.4% of on-screen speech time has matching visemes.
- **Inferred:** one steady per-speaker rate, nudged per line by the source line's rate; accept some drift; never trade meaning for fit.

### F4 [corrected: venue] Isometric MT is no more isochronous than standard MT once both go through the same TTS; jointly predicting durations raises speech overlap from 0.53 to 0.92 at a BLEU cost
- Source: Chronopoulou, Thompson, Mathur, Virkar, Lakew, Federico (AWS AI Labs; LMU Munich), "Jointly Optimizing Translations and Speech Timing to Improve Isochrony in Automatic Dubbing", arXiv 2302.12979, 2023-02-25.
- **Correction:** it is **not** in the Interspeech 2023 proceedings (ISCA Archive index checked 2026-09-24). Pal et al. 2023, IWSLT 2024 Findings and Won et al. 2025 cite it only as arXiv, so it is a preprint. It is AWS work; no Apple paper on this topic was found (see unverifiable S7).
- **Disclosed, numbers confirmed (Table 2, CoVoST-2 De→En):**
  - StdMT: BLEU 35.6 / speech overlap 0.53.
  - IsoMT: 35.3 / 0.53.
  - Txt2Phn: 33.3 / 0.52.
  - Joint phonemes-plus-durations model: 30.6 / 0.92 without noise, 32.4 / 0.81 with σ = 0.1 noise.
  - test91 (human-read): StdMT 38.6 / 0.54 → joint model 27.2 / 0.84.
  - The authors limit the isometric-MT point to "in isolation, without prosodic alignment" and say isometric MT might still interact well with alignment.
- **Inferred:** character or word targets won't buy timing for Maata; duration in the voice's units will.

### F5 [corrected: numbers and conclusion] Pause markers in the MT text beat translate-then-align only without relaxation; aggressive length control drops content
- Source: Tam, Lakew, Virkar, Mathur, Federico (AWS), "Isochrony-Aware Neural Machine Translation for Automatic Dubbing", Interspeech 2022. arXiv 2112.08548, v2 2022-07-08.
- **Disclosed (Table 4, 40 native annotators, 1,000 judgements per comparison, EN→DE/FR, 50 single-sentence videos):**
  - MT+[pause] (B) vs MT+PA (A): De 41.0 vs 32.0 (p<0.01); Fr 38.2 vs 36.9 (n.s.).
  - **Correction:** B vs MT+LC (C) is De 51.7 vs 30.1 and Fr 60.9 vs 22.0. The draft's 48.4/30.8 and 61.4/25.8 are A vs C.
  - MT+LC had the best smoothness but far fewer acceptable translations, because it dropped tokens.
  - **Correction to the takeaway:** once the relaxation mechanism was added to both systems, pause markers lost their edge.
    - D′ (Lakew isometric MT + full PA + relaxation) beat B′ on French, 45.0 vs 35.1 (p<0.01), and tied on German, 37.4 vs 37.5.
    - The authors attribute this to D translating better, and conclude that translation quality and speaking rate are both necessary.
- **Inferred:** Maata's planner already borrows silence (a relaxation), so the evidence says to invest in translation quality before pause markers.

### F6 [confirmed] Prompted LLMs shorten reliably only with extreme-short examples plus matching instructions; sampling several outputs, filtering by length and choosing by quality estimation gives the best trade-off
- Source: Javorský, Bojar, Yvon, "Prompting LLMs: Length Control for Isometric Machine Translation", IWSLT 2025 (arXiv comment). arXiv 2506.04855, 2025-06-05.
- **Disclosed:**
  - 8 quantized open LLMs via Ollama (Llama 3, Gemma 2, Qwen 2, Mistral/Mixtral). EN→DE/FR/ES.
  - Isometric examples lead models to ignore the constraint. Going from 5 to 10 to 20 shots gives only marginal gains.
  - Final setup: 10 outputs from 10 "Tiny" 10-shot sets, keep those within ±10% of the source character count, pick by COMETKiwi.
  - About half of sentences already comply with an unconstrained prompt.
  - Authors' caveats: 10 outputs "may not be feasible in practice"; length is measured in characters; no downstream dubbing evaluation.
- Side claim: COMETKiwi licence [confirmed]. `Unbabel/wmt22-cometkiwi-da` @1ad78519 and `wmt23-cometkiwi-da-xl` are CC-BY-NC-SA-4.0; `XCOMET-XL` too (HF API, 2026-09-24). Not usable.

### F7 [confirmed, with added context] Duration-based LLM translation lifts speech overlap about 16% over plain GPT-4o at a small COMET cost
- Source: Won, Jeong, Choi, Kim (ESTsoft), "End-to-End Multilingual Automatic Dubbing via Duration-based Translation with Large Language Models", EMNLP 2025 System Demonstrations, pp. 515–521, November 2025 (ACL Anthology 2025.emnlp-demos.37).
- **Disclosed:**
  - Algorithm 1:
    - Predict a target phoneme count from the source duration.
    - Translate; while the count misses by more than δ, ask the LLM for 3 shorter or longer candidates.
    - Keep candidates that move closer, and pick by COMET against the unconstrained translation used as a pseudo-reference.
    - Finally the LLM aligns pauses.
  - Average speech overlap, DT / PT / plain GPT-4o: →en 0.930 / 0.891 / 0.787; →es 0.900 / 0.883 / 0.797; →ko 0.915 / 0.871 / 0.788.
  - COMET cost is 0.016–0.042 (for example →en 0.816 vs 0.850).
  - MOS on 8 clips with 30 raters: 6.57 vs 5.83 for an unnamed proprietary system (weak evidence).
- **Added context the draft missed:** synthesis was **ElevenLabs TTS**, followed by speech-to-speech conversion to restore the speaker's timbre. The voice reference was chosen by UTMOS within the interquartile range. Only the translation loop is analogous to Maata; the voice side is a commercial API.
- **Inferred:** closest published analogue to Maata's Claude-CLI translation loop. For Maata, only shorten (lengthening adds filler).

### F8 [confirmed, with context] IWSLT 2024: the timing-aware dubbing system reached human-dub-level overlap and was judged more natural; generic translations squeezed into slots were hard to understand
- Source: Findings of the IWSLT 2024 Evaluation Campaign, §7, August 2024 (ACL Anthology 2024.iwslt-1.1).
- **Disclosed:**
  - EN→ZH, five 10-minute ITV clips.
  - HWTSC-Dubbing speech overlap 0.698 vs 0.279–0.304 for offline ST systems (Table 10).
  - PEAVS 3.05 vs 1.28–1.33 (original 3.82).
  - Human score (1–6 scale, first 20 sentences per clip): 3.9 vs 3.2–3.5.
  - The organizers note the offline systems had to speed synthesis up, which hurt intelligibility.
- **Context:**
  - HWTSC-Dubbing was the only dubbing submission.
  - The organizers dubbed the offline baselines with Amazon Polly's standard voice under a max-duration flag, and padded short output with silence.

### F9 [corrected: task and variant costs] LLM post-editing for length, reranking by back-transcription CER and sync, and splitting at pauses only when CER holds
- Source: Li, Guo, Zhang et al. (Huawei TSC), "Pause-Aware Automatic Dubbing using LLM and Voice Cloning", IWSLT 2024 pp. 60–64 (ACL Anthology 2024.iwslt-1.2).
- **Correction (context):** these results are the **DE→EN** subtask (CoVoST-2; NLLB-1.3B fine-tuned; LLM post-editor; MeloTTS + OpenVoice; CER by wav2vec2-base). They are not the EN→ZH track scored in F8. That separate entry used diarization, MMDenseNet source separation and edge-tts, with no analysis.
- **Disclosed (Table 2):**
  - "Complex"/"simple" were chosen because they were more stable than "longer"/"shorter".
  - The unconstrained LLM post-edit kept COMET (89.42 vs 89.29).
  - **Correction:** "simple" dropped COMET to 88.01 and raised CER (subset1 5.68 → 6.35; subset2 3.93 → 4.47). "Complex" collapsed BLEU to 19.67 and COMET to 84.08. So the variants do **not** fully keep meaning.
  - Pause-aware split: LSE-D 13.88 → 12.27, but CER 3.93 → 4.47. The combined "split only if CER doesn't worsen" rule gave CER 3.59.
  - Reranking 4 candidates by LSE-D and CER: CER 4.62, LSE-D 10.75 (subset1).
  - Speed was allowed from 0.75x to 2.5x.
  - OpenVoice VC raised CER: removing it gave 4.08 vs 4.62.
  - SeamlessExpressive's non-English speech did not sound native to the team.

### F10 [confirmed] Tightly time-constrained MT reaches near-perfect overlap but sounds more robotic and repeats end words; noisy durations (σ ≈ 0.1) were the best compromise
- Source: Pal, Thompson, Virkar, Mathur, Chronopoulou, Federico (AWS), Interspeech 2023 (ISCA Archive pal23). arXiv 2305.13204, 2023-05-22.
- **Disclosed:**
  - Counters without noise: overlap 0.9887.
  - With noise 0.1: BLEU 35.6 vs 35.8, overlap 0.8649 (above human dubs).
  - The crowd evaluation pilot was too noisy to use. Qualitative: tight models are a little robotic and sometimes repeat a final word.

### F11 [corrected: scope] Moving TTS rate away from 1x lowers speaker similarity and intelligibility, in a model that re-generates at the new durations
- Source: Eskimez, Wang, Thakker et al. (Microsoft), "Total-Duration-Aware Duration Modeling for Text-to-Speech Systems", Interspeech 2024 (ISCA Archive eskimez24). arXiv 2406.04281, 2024-06-06.
- **Disclosed:**
  - Baseline regression + length regulator: similarity fell as the rate moved from 1x.
  - At 2x, WER went from 2.7 to 15.1; 1x similarity was 0.805.
  - FM+LR at 2x: WER 26.2 and similarity 0.603, vs 8.9 and 0.680 for the TDA variant.
  - Listeners (15 native speakers, 30 samples at 2x) preferred TDA 58.7% for intelligibility and 53.9% for similarity.
  - Slower rates slightly lowered similarity. Values at 1.1–1.25x appear only in a figure.
- **Correction (scope):** rate was changed by linearly scaling predicted phoneme durations before a zero-shot audio model generated speech, not by speeding up finished speech. Carrying the result over to Maata's post-hoc mel resampling is an inference and must be measured on Maata's voices.

### F12 [confirmed] Phrase-level cross-lingual prosody transfer beat utterance-level transfer and closed 23.2% of the gap to human dubbing
- Source: Swiatkowski, Wang, Babianski et al. (Amazon), Interspeech 2023 (ISCA swiatkowski23). arXiv 2306.11662, v2 2023-06-21. Companion arXiv 2306.11658, 2023-06-20.
- **Disclosed:**
  - EN→ES (Castilian), MUSHRA with video, 25 bilingual raters, 100 utterances.
  - Phrase VAE 74.31 / 66.56 vs utterance-level 69.58 / 63.21; recording 83.75 / 87.22.
  - +6.2% MUSHRA overall.
  - WER 0.101 vs 0.098. Without length regularization, WER on the shortest 25% of phrases is 0.229 vs 0.161.
  - Amazon's in-house TTS, trained on parallel dubbing data.
- **Inferred:** try per-line prosody conditioning from the same line's English audio (untested on Chatterbox).

### F13 [confirmed] Hard timing constraints inside the TTS make it drop words when the text is too long and repeat words when it is too short
- Source: Pérez-González-de-Martos, Lux, Elizarova et al. (AppTek), "Not Quite My Tempo", accepted at Interspeech 2026. arXiv 2609.26486, 2026-09-22.
- **Disclosed:**
  - VAD-mask-conditioned flow-matching DiT.
  - Silero VAD frame accuracy 72.69% → 96.21% (LibriTTS model) and 71.35% → 91.59% (multilingual).
  - MOS (40 Prolific raters, multilingual model): placement 3.82 → 3.73, prosody 3.81 → 3.68. Not significant (Mann-Whitney p>0.05).
  - WER (internal ASR): 8.5% → 13.9% (LibriTTS), 6.3% → 8.3% (multilingual).
  - Qualitative: omissions when the translation exceeds the time budget, repetitions when it is too short, reordering near punctuation.
- **Repo side claim [confirmed]:** `session.py:_take` caps generation at `min(30 s, max(4 s, target_seconds × speed_cap × 2))`. `speed_cap` defaults to 1.2 (clamped 1.0–1.25), and `target_seconds` is the source span plus a small borrow of the following silence (`timing/planner.py:93`). So the cap is about 2.4x the target.
- **Inferred:** Maata's cap truncates rather than compresses, but the lesson is the same. Treat a take that hits the cap without EOS as failed.

### F14 [confirmed] LLM dubbing translations beat human subtitles on accuracy but fall short on vividness; per-line duration control of multi-line dialogue needed fine-tuning
- Source: Cui, Huang, Wang et al. (Alibaba DME / HUST), "Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization", ACL 2025 (Long), pp. 4524–4546. arXiv 2508.08550, 2025-08-12.
- **Disclosed:**
  - 20-line dialogue blocks, β = 0.5, about 10k lines.
  - Format validity: full fine-tuning 81–90%; LoRA 99.7–99.8% (Table 5).
  - Pairwise human evaluation by 4 professional translators per direction (zh→en, es→zh). LLM translations win on accuracy but lose on vividness, because LLMs lack the scene and emotion cues from audio and video.
- Repo side claim [confirmed]: ADR-018 Round 1 records that windowed JSON translation shifted lines when a sentence spanned two of them.
- See N1 for the draft's missed Claude result from the same paper.

### F15 [confirmed] Meta's September 2026 dubbing TTS: generate N takes, keep those with speaker similarity at least 75% of the best, pick the lowest ASR WER; at N=32 similarity rose from 0.66 to 0.74 and WER fell from 4.05% to 2.20%
- Source: Chen, Hwang, Inoue et al. (FAIR at Meta), "Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis". arXiv 2609.03992 v1 2026-09-03; paper dated 2026-09-04.
- **Disclosed:**
  - 3B flow-matching DiT on 25 Hz latents at 48 kHz, 480k h of pretraining.
  - Dubbing SFT on about 2k h monolingual plus 50 h EN↔ES.
  - Reranking uses WavLM-SV similarity and Whisper-large-v3 WER; gains grow steadily with N.
  - Side-by-side against Meta's internal dubbing model (internal benchmark, 100 En→Es + 100 Es→En, scale −3 to 3): prosody similarity +0.33/+0.34, voice similarity +0.29/+0.36, naturalness +0.39/+0.45, shareability +0.40/+0.38.
  - Cross-lingual SFT: WER 3.16 → 2.72, but similarity 0.73 → 0.60.
  - No release of weights is mentioned.
- **Inferred:** the recipe carries over to Chatterbox's sampled T3. It needs a Telugu ASR (see S2).

### F16 [confirmed, with caveat] Human judgements of podcast-domain speech translation track translation quality and length compliance; UTMOS and NISQA fail
- Source: Koudounas, Futami, Jodelet, Take et al. (Sony / CMU), "Benchmarking Speech-to-Speech Translation Models" (COMPASS), under submission. arXiv 2606.03241, 2026-06-02.
- **Disclosed:**
  - 46 metrics, 1,248 configurations, 3 native annotators per clip.
  - Systems include Voxtral + Chatterbox, and Whisper + Gemma + CosyVoice3.
  - Translation metrics ρ = 0.82 in podcasts. Isometry metrics predict podcast naturalness and overall quality (ρ 0.95–1.00). Prosody and timing metrics ρ = 0.91 in dubbing.
  - UTMOS/NISQA near zero or negative; NISQA vs emotional preservation ρ = −0.90.
  - Ground truth won 89% of EN→X podcast clips.
- **Caveats:**
  - The "podcasts" are EuroParl Multimedia Centre clips (six 30 s clips per direction per pair), not real podcasts.
  - The correlations are over 4–5 systems and saturate at ±1.

### F17 [confirmed] Compare speaking rates relative to each language's mean; post-hoc VC raises similarity but costs intelligibility; an intermediate text is key
- **Seamless (Meta), arXiv 2312.05187, 2023-12-08:**
  - SRD_norm normalises each language's rate by that language's mean (eq. 25).
  - Fine-tuning on the high- vs low-prosody-alignment split, eng–X (Table 25): rate correlation 0.49 → 0.63, pause 0.10 → 0.31, ASR-BLEU 32.57 → 33.48.
- **Dub-S2ST (Choi, Kim, Chung), EMNLP 2025 Findings, arXiv 2505.20899, v2 2025-12-29:**
  - CosyVoice VC: ASR-BLEU 23.88 → 23.09, SIM 0.036 → 0.315 (Table 8).
- **TransVIP (Le et al.), NeurIPS 2024, arXiv 2405.17809, v3 2024-10-31:**
  - The textless baseline repeated and hallucinated; the authors conclude an intermediate text is crucial.
- **Inferred:** set per-line rate relative to the speaker's own mean; prefer reranking over voice conversion.

### F18 [corrected: venues] The lip-video movie-dubbing TTS family is same-language, benchmarked on English V2C/Chem/GRID/LRS3, and not usable for Maata; two ideas carry over
- **Correct metadata (arXiv API, CVF 2026-09-24):**
  - EmoDubber: CVPR 2025, 2412.08988 v3 2025-04-25. Uses positive/negative guidance for emotion intensity.
  - InstructDubber: **AAAI 2026**, 2512.17154, 2025-12-19. Qwen2.5-7B-Instruct analyses emotion instructions.
  - VoiceCraft-Dub: **ICCV 2025** (not CVPR), 2504.02386, 2025-04-03.
  - CoSyncDiT: 2604.12292, 2026-04-14.
  - HoliDubber: 2606.09098, v1 2026-06-08, v2 2026-06-15.
  - DeepDubber-V1: 2503.23660, 2025-03-31.
  - DiFlowDubber: **CVPR 2026 Findings**, 2603.14267.
  - FlowDubber: 2505.01263, v2 2025-08-25. ACM MM 2025 is unverifiable (S9).
- **Newer, same family, not usable:** CineDub (ACM MM 2026, 2608.15734), SyncVoice (2512.05126), TBDub (2609.06144, visual), JUST-DUB-IT (2601.22143).
- PS-TTS (ICPR 2026, 2604.09111) does LM paraphrasing for isochrony plus vowel-DTW lip sync. The isochrony half repeats F7/F19.
- "Dub-S1" [confirmed not found]: arXiv API search on 2026-09-24 returned 0 hits.
- **Inferred:** use per-line emotion and rate labels (InstructDubber-style) to drive TTS controls, and adapt delivery to scene type (DeepDubber).

### F19 [confirmed] Short/normal/long variants in one pass, chosen with the target voice's duration model, improved perceived sync at about equal BLEU
- Source: Chadha, Subramanian, Joshi, Bansal et al. (Microsoft), "Length Aware Speech Translation for Video Dubbing", Interspeech 2025. arXiv 2506.00740, 2025-05-31.
- **Disclosed:**
  - End-to-end ST with length tokens.
  - Durations estimated with the on-device LeanSpeech TTS duration model.
  - Speech-rate compliance (±20%): ES 49.06 → 57.04 (+16.3%), KO 62.02 → 74.34 (+19.9%).
  - BLEU 23.23 vs 23.66 (ES), 23.41 vs 22.57 (KO).
  - Sync MOS +0.34 (ES) and +0.65 (KO). Latency +4.3%.

### F20 [confirmed] English→Indic conversational translations often overshoot the source's phoneme count
- Source: Mhaskar, Shah, Zaki, Gudmalwar et al. (Sony Research India), NAACL 2024 Findings. arXiv 2403.15469, 2024-03-20.
- **Disclosed:**
  - EN→HI. IndicTrans2 share within ±20% of the source phoneme count: FLoRes 72.72%, BPCC-General 81.25%, BPCC-Conversational 46.5%, movies 36.77%.
  - LLaMA2-7B about 0.3–0.5%.
  - RL-NMT: conversational 83.23% at BLEU 24.55 vs 28.55.
- **Inferred:** for Telugu, aksharas are the closest unit. Expect more than half of conversational lines to need the concise path.

### F21 [confirmed; venue added] Human ratings of AI-dubbed EN↔HI clips are driven mostly by the audio; equal-weight metric averages track humans poorly
- Source: Dasare, Shah, Gudmalwar, Wasnik (Sony Research India), **ICASSP 2026**. arXiv 2603.28717, v2 2026-04-24.
- **Disclosed:**
  - 12k clips (MELD EN→HI, M2H2 HI→EN).
  - LLM translator (written "Gemini-9B") prompted with speaker and emotion tags, F5-TTS, global time-stretch.
  - 30 raters, 1,350 ratings, α = 0.82.
  - Predicting MOS: audio PCC 0.68, text 0.34, video 0.05, all three 0.76.
  - Network trained on an equal-weight proxy MOS: 0.22. With active-learned weights: 0.68 (Table 5).

### F22 [confirmed] Flattening pitch movement alone makes real speech sound as unlike the speaker as mid-tier clones
- Source: Bakkouche, McGhee, Lau et al., "Finding the Human Voice in AI", Interspeech 2025 (ISCA Archive bakkouche25), August 2025.
- **Disclosed:**
  - MUSHRA, 28 naturalness and 27 similarity listeners.
  - Two SSBE speakers and one General American speaker; clones from 15 s references; 3 s stimuli.
  - Similarity: ElevenLabs was not different from human (p = .097). The 30%-F0 human condition was not different from StyleTTS-2 (p = 1.000) or XTTS-v2 (p = .053).
- **Inferred:** WeSpeaker cosine is necessary but not sufficient; match each clone's pitch range too.

### F23 [corrected] Acoustic realism: measured support is thinner than the draft said
- **Disclosed:**
  - Federico 2020 (F1): background plus reverb gave +10.34 MUSHRA (p<0.05) for non-native listeners only. Native listeners +1.05 (n.s.).
  - **Correction (HoliDubber, 2606.09098):** in zero-shot mode (Table 1), the studio-clean FunCineForge got the **highest** MOS: 3.91 on VoxCeleb2 and 4.06 on CelebV-Dub, vs HoliDubber's 3.83 and 3.96.
  - HoliDubber led only in text-prompted joint speech-plus-ambience mode, and narrowly: 3.92 vs 3.86 and 3.99 vs 3.96.
  - The claim that speech cleaner than the scene is a failure is the authors' reading of UTMOS, not a listener result.
  - **Correction (HW-TSC 2024):** the EN→ZH entry's source separation was not evaluated.
- **Inferred:** a background bed is plausible but weakly evidenced. It is worth an A/B test, not a rule change on its own.

### F24 [confirmed] Duration-aware translation methods converge: model speech duration, carry the time remaining into generation, and select among candidates
- **Sources:**
  - VideoDubber (Wu et al.), AAAI 2023, arXiv 2211.16934, v2 2023-12-05. Uses duration-aware positional embeddings for the time left, and AdaSpeech 4 zero-shot TTS.
  - Effendi, Virkar, Barra-Chicote, Federico, ICASSP 2022, pp. 8037–8041. The full text was read in this pass; see N2.
  - Rao et al., IWSLT 2023 (ACL Anthology 2023.iwslt-1.9). Reranked candidates by speech overlap, per HW-TSC 2024's own description.
  - Yousefi et al., arXiv 2411.07387, 2024-11-11 (Sino-Tibetan → Indo-European).

### F25 [corrected: minor number] In professional dubs, word-aligned pitch and energy contours correlate weakly across languages; duration and phoneme count correlate strongly
- Source: Xie, Ulgen, Son, Sisman, Koehn (JHU), **EMNLP 2026 Findings**. arXiv 2608.27848, 2026-08-28.
- **Disclosed:**
  - Professionally dubbed TV episodes, EN↔DE/ES/FR.
  - Pitch mean r 0.229–0.246.
  - **Energy 0.174–0.183** (draft said 0.176–0.186).
  - Removing nouns lowers energy correlation by 0.036–0.040.
  - Phoneme count r 0.871–0.885; duration 0.884–0.891.
- **Inferred:** transfer utterance-level traits and content-word emphasis, not contours.

---

## Findings added by the verifier

### N1 [confirmed] Prompted frontier LLMs, Claude 3.5 Sonnet included, control per-line duration worse than small fine-tuned systems
- Source: Cui et al., ACL 2025 (F14), Table 2 and §5.2.1.
- **Disclosed:**
  - GPT-3.5, GPT-4o and Claude 3.5 Sonnet with prompt engineering improved on the gold reference.
  - They failed to match the fine-tuned AutoDubbing and VideoDubber baselines on duration consistency.
  - The authors conclude that LLMs lack a sense of text duration and need explicit duration information.
- **Inferred for Maata:** compute the akshara budget locally, pass it to Claude as a number, and measure each returned wording locally with the voice's calibrated rate. Never rely on Claude's own judgement of length.

### N2 [confirmed] Non-uniform (per-phoneme) duration scaling beats uniform scaling at 1.1–1.4x and 0.6–0.9x
- Source: Effendi, Virkar, Barra-Chicote, Federico (Amazon), "Duration Modeling of Neural TTS for Automatic Dubbing", ICASSP 2022, pp. 8037–8041. Full text in `dub_research/dur_tts.txt`.
- **Disclosed:**
  - Each phoneme's duration is shifted in proportion to its predicted standard deviation, instead of uniform stretching.
  - 20 native raters and 1,000 ratings per language.
  - Win improvements: fast speech en-it +172.8%, en-es +82.5%; slow speech +61.3%, +28.9%. All p<0.01.
  - Larger gains at the rate extremes.
- **Inferred:** Maata's mel resampling is uniform. A cheap approximation is to compress a take's internal silences and pauses before speeding up speech frames. This needs a listening test.

### N3 [confirmed] Large Indic TTS preference study, including Telugu and code-mixed text: expressiveness and intelligibility drive preference
- Source: Anand, Ashwin, et al. (IIT Madras / AI4Bharat / Josh Talks), "Preferences of a Voice-First Nation". arXiv 2604.21481, v1 2026-04-23, v2 2026-06-23. Read from the local arXiv HTML copy (`dub_research/vfn.txt`).
- **Disclosed:**
  - 5,357 sentences in 10 Indian languages including Telugu, with a code-mixed subset.
  - 7 TTS systems, over 120K pairwise comparisons from over 1,900 native raters.
  - SHAP analysis: expressiveness and intelligibility are the strongest predictors of overall preference, then liveliness and voice quality.
- This is TTS, not dubbing. It is the closest Telugu-inclusive perception evidence found, and it partly answers the draft's "no Telugu perception study" question.

### N4 [confirmed licence; untested for Telugu] An Apache-2.0 meaning scorer exists
- `Unbabel/wmt22-comet-da` @2760a223 is Apache-2.0 and not gated (HF API, 2026-09-24). COMETKiwi and XCOMET are CC-BY-NC-SA-4.0.
- Won et al. (F7) used reference-based COMET with the unconstrained translation as a pseudo-reference to rank shortened candidates.
- **Inferred:** Maata could do the same with Claude's unconstrained Telugu. This needs a new dependency and approval, and Telugu reliability is unmeasured. A Claude back-translation judge needs no new dependency.

### N5 [confirmed: exists; abstract only] Scene-level TTS evaluation
- Source: SceneTTS-Bench, arXiv 2609.26255, 2026-08-13, submitted to the ACM MM 2026 Dataset Track.
- Measures timbre consistency across turns, emotional expressiveness and **rhythm coherence under segmented long-form synthesis**. That last one is Maata's line-by-line case.
- Only the abstract was read.

---

## Side claims checked

| # | Claim | Verdict | Evidence (date) |
|---|---|---|---|
| S1 | COMETKiwi is CC-BY-NC-SA-4.0 | confirmed | HF API wmt22-cometkiwi-da @1ad78519 (2026-09-24) |
| S2 | IndicConformer-600M is MIT, gated, trust_remote_code | confirmed | HF API @e9b71b36: licence mit, gated auto, `auto_map` → `model_onnx.py`, ONNX assets with a Telugu head `joint_post_net_te.onnx` (2026-09-24). Runs on onnxruntime, a new dependency. |
| S3 | `session.py` cap ≈ 2.4x slot | confirmed | `session.py:534`, `planner.py:93` |
| S4 | ADR-018 windowed arm shifted lines | confirmed | `docs/DECISIONS.md` ADR-018 Round 1 |
| S5 | Maata speeds up by mel resampling, not time-stretch | confirmed | ADR-017 |
| S6 | "Dub-S1" not findable | confirmed | arXiv API search, 0 hits (2026-09-24) |
| S7 | No Apple paper on joint translation and timing | unverifiable | Web-search budget exhausted; the arXiv title search found none. The joint MT+timing work found is AWS. |
| S8 | Seamless AutoPCP licence | unverifiable | stopes code is MIT (`gh api`); the AutoPCP-multilingual-v2 checkpoint (fbaipublicfiles zip) states no licence |
| S9 | FlowDubber at ACM MM 2025 | unverifiable | arXiv has no venue comment |
| S10 | V2C papers "use data from commercial films" | unverifiable | Not checked per paper |

## Refuted sub-claims (inside corrected findings)

- Pause markers beat length control by "48.4 vs 30.8 / 61.4 vs 25.8". Those are MT+PA vs MT+LC (F5).
- Pause markers beat translate-then-align in general. Not once relaxation is applied (F5).
- "Studio-clean speech is a known failure," as a listener finding. HoliDubber's own zero-shot MOS favoured the studio-clean system (F23).
- The HW-TSC "simple"/"complex" variants keep meaning. Only the unconstrained post-edit did (F9).
- Venue errors: Chronopoulou 2023 was not at Interspeech 2023 (F4); VoiceCraft-Dub was not at CVPR 2025 (F18).

---

## Recommendations for Maata (updated for the corrections)

1. **Duration-budgeted translation through the Claude CLI.** Impact high, effort M, no new dependency.
   - **Change:**
     - For every line, compute an akshara budget locally from `timing/isochrony.py` (`slot_budget`, `target_units`) and the cloned voice's measured aksharas/s.
     - Put the number in the prompt.
     - Only for lines predicted to overrun, ask in the same call for 2–3 graded "more concise / simpler" wordings.
     - **Measure each returned wording locally.** Pick by fit first, then meaning, using a Claude back-translation judge.
     - Never lengthen to fill a slot.
   - **Why:** F4, F6, F7, F19, F20, F24, and N1 (LLMs can't judge duration themselves).
   - **Watch:**
     - "Simple" wordings cost a little meaning and intelligibility (F9), so gate them.
     - Subscription rate limits for multi-candidate calls.
   - Optional: `wmt22-comet-da` (Apache-2.0) as a pseudo-reference ranker (N4). This needs a new dependency and approval.

2. **Keep speaking rate near 1.0 and set it relative to the speaker, not the slot.** Impact high, effort S, no approval.
   - **Change:**
     - Per-line target = voice rate × clamp(source line rate / speaker mean rate, about 0.85–1.15).
     - Use mel speed-up only as a final trim: prefer ≤1.1x, keep 1.2x as the cap.
     - Before speeding speech frames, first compress a take's internal pauses (N2, inference).
     - Measure WeSpeaker similarity and back-transcription CER at 1.0, 1.1 and 1.2x on the maintainer's voices. The published loss (F11) came from in-model duration scaling, so Maata's post-hoc numbers are unknown.
   - **Why:** F1, F2, F3, F11 (corrected), F17, N2.

3. **Best-of-N TTS takes.** Impact high, effort M.
   - **Change:**
     - 2–3 seeds per line, more for short lines.
     - Drop takes below about 0.75× the best WeSpeaker similarity.
     - Choose the lowest Telugu back-transcription CER.
     - A take that hits the generation cap without EOS counts as failed.
   - **Why:** F15, F9, F13.
   - **Approval:** needs a Telugu ASR good enough to rank takes; ADR-014 found Whisper-te too noisy.
     - Candidate: ai4bharat/indic-conformer-600m-multilingual (MIT, gated, remote code, ONNX/onnxruntime). New dependency, pin and ADR.
     - The ~8 GB freed by dropping the local translator gives room for it and for extra takes.

4. **Per-speaker prosody calibration and per-line expressiveness.** Impact high, effort M, no approval.
   - **Change:**
     - Measure each speaker's F0 range (semitone std), energy range and rate from their English.
     - Tune Chatterbox exaggeration and cfg per speaker to match.
     - A/B a per-line T3 prompt from the same line's English audio, keeping the S3Gen timbre reference fixed.
     - Have Claude tag emotion and emphasised content words from the transcript plus locally measured features.
   - **Why:** F3, F12, F18, F22, F25, N3.

5. **Evaluation harness before the next bake-off.** Impact high, effort M.
   - **Per line:** CER, WeSpeaker similarity, duration and chars-per-second ratio, onset lag, applied rate, F0-range ratio, and meaning via a Claude judge.
   - **Per scene:** rhythm coherence across consecutive lines (N5).
   - **Human sheet:** accuracy, naturalness, similarity, prosody/rhythm, timing, emotion and overall, with weights fit to the maintainer's ratings.
   - Never gate on UTMOS or NISQA. Never use COMETKiwi or XCOMET (NC).
   - **Why:** F16, F21, F15, N3, N4, N5.
   - **Approval:** none if Claude is the judge.

6. **Target human-level synchrony, not perfect isochrony.** Impact medium, effort S, no approval.
   - Aim for speech overlap of about 0.65–0.75 with small onset lag. Prefer small drift over rate changes or condensing.
   - **Why:** F3, F8, F10.

7. **Background audio A/B, downgraded after the F23 correction.** Impact medium/uncertain, effort M.
   - Compare a dry dub, source-matched reverb plus low room tone, and (only if the maintainer relaxes the rule) a separated background stem.
   - **Why:** the only measured support is Federico 2020, and only for non-native listeners.
   - **Approval:** the stem option needs the "original audio never played" rule relaxed plus a separation model (a new dependency). Reverb and room tone need neither.

8. **Pause markers inside long lines, downgraded to optional.** Impact low, effort M, no approval.
   - Pass English pauses of at least 300 ms to Claude, and insert silences only where CER doesn't worsen.
   - **Why:** F5 (corrected): no advantage once silence borrowing is used. F9: splits can hurt intelligibility.

9. **Things to avoid:**
   - Character or word isometric targets (F4).
   - Per-phrase alignment that makes the rate uneven (F1).
   - Post-hoc voice conversion for similarity (F9, F17).
   - Tight generation caps (F13).
   - LLM fine-tuning methods such as SSPO (F14).
   - Lip-video movie-dubbing models (F18).
   - Trusting an LLM's own sense of length (N1).

## Relation to `sota_ranking.md`

- Section 1 of `sota_ranking.md` (local translator shortlist: Gemma 4 12B, Hy-MT2, Bodhan, IndicTrans2) is **superseded** by today's decision: no local LLMs, with translation through the Claude CLI. Its roughly 8 GB slot is freed.
- Nothing in this academic sweep contradicts its ASR, diarization or TTS rankings.
- The best-of-N recipe (F15) and the prosody calibration (F22) apply to whichever Chatterbox-Telugu variant wins the TTS bake-off (S3Gen v3 swap or v3 T3 retrain).
- `sota_ranking.md` has no Telugu back-transcription ASR, which Recommendation 3 needs. IndicConformer-600M (S2) is the candidate to add as a QA-stage arm.

## Open questions

1. Is IndicConformer-600M's Telugu CER on Chatterbox output low enough to rank takes? Its gate and remote-code loader need approval.
2. How much WeSpeaker and perceived similarity does Chatterbox-Telugu lose per 0.1x of post-hoc mel speed-up, and does compressing pauses first help (N2)?
3. Does a per-line T3 prompt from the line's own English audio improve prosody without hurting timbre or Telugu pronunciation?
4. There is still no Telugu (or Dravidian) *dubbing* perception study. N3 covers Indic TTS preference only. Telugu viewers' weighting of sync vs voice vs naturalness is unknown.
5. How reliable are Claude's condensed Telugu wordings on meaning? Measure with a Claude back-translation judge, and optionally `wmt22-comet-da` (N4).
6. What are the Claude CLI's throughput and subscription rate limits for per-line multi-candidate calls with 5–10 minutes of lookahead?
7. What is the maintainer's ruling on background audio (dry / room tone plus reverb / separated stem), given the weaker evidence after the F23 correction?
8. What is the AutoPCP checkpoint licence (S8)? Is FlowDubber's venue ACM MM 2025 (S9)? Is there really no Apple paper (S7)?
