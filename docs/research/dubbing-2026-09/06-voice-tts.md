# 06 Voice cloning and expressive Telugu TTS, local on Apple Silicon: verified report

Verified 2026-09-24 against primary sources: vendor docs, model cards, licence files, arXiv papers (re-read or re-fetched) and Maata's own code. This report builds on `../sota_ranking.md` section 4 and on the researcher's sweep (`06-voice-tts.sweep.md`), and corrects both where needed. No model was run.

**Limits of this check.** The web-search budget was already used up, so claims that only a search could confirm are marked unverifiable. The GitHub REST API was rate-limited, so licences were read from the raw LICENSE files and commit dates from the repos' commit feeds.

**Verdict key.**
- **[confirmed]:** matches the primary source.
- **[corrected]:** partly wrong. The correct fact and its source are given.
- **[unverifiable]:** could not be confirmed today.
- Refuted claims would go in their own list; none were found.

## Summary

**Bottom line (still holds).** No permissively licensed, Telugu-native, zero-shot voice cloner beats the current family (Chatterbox Multilingual plus the maintainer's Telugu fine-tune). The newer permissive models lack Telugu:
- Chatterbox V3
- Qwen3-TTS
- VoxCPM2
- CosyVoice 3
- IndexTTS-2.5
- MOSS-TTS v1.5

The strong alternatives that do cover Telugu, or that clone best, fail the licence policy:
- OmniVoice, F5-TTS, MaskGCT and Vevo are non-commercial.
- IndexTTS uses a custom licence.
- Seed-VC is GPL-3.0.
- snorTTS-Indic sits on a Llama-3.2 base.

Gains will come from how Maata conditions, selects and retrains Chatterbox-Telugu.

**The verification makes staying on Chatterbox a stronger choice.** The IWSLT 2026 paper, which the researcher cited, also benchmarks the candidates cross-lingually. Every system cloned English reference voices into Arabic, French and Chinese, scored with ECAPA speaker similarity on a 4-speaker subset. Chatterbox (0.680 / 0.619 / 0.653) beat both Qwen3-TTS (French 0.533, Chinese 0.522) and VoxCPM2 (0.607 / 0.575 / 0.569). Those are the two models the researcher ranked as the "strongest permissive cloners", but their vendor numbers are same-language. OmniVoice (non-commercial) led at 0.703 / 0.753 / 0.702.

**Counts** (58 checked claims): 40 confirmed, 11 corrected, 0 refuted, 7 unverifiable.

**Key corrections:**
1. **Singlish study.** The fine-tune gains are real, but the study adapts a model to one accent within one language. "Accent similarity" there means sounding more like Singlish, the accent of the fine-tuning data. For Maata, the transferable number is the unseen-speaker gain of +0.069 in speaker similarity. Word error rate also rose, from 11.48 % to 15.39 %.
2. **IWSLT 2026 best-of-N.** The 0.5(1−CER)+0.5·SIM selection was used to build a distillation training set from three teacher models. It was not reranking at inference time. The per-language numbers were mixed across two test subsets. On the full set, LoRA fine-tuning lowered Arabic similarity (0.734 → 0.726).
3. **VoxCPM2 speed on a Mac.** "RTF about 1.76 on an M4 Pro" is the llama.cpp-omni Q8_0 build on Metal, not MLX. VoxCPM2's MLX and MPS speed is unmeasured.
4. **CosyVoice 3 emotion numbers.** Mis-stated. The correct figures for CosyVoice3-1.5B are 0.64–0.86 when the text carries the emotion and 0.44–0.64 when the text is neutral. The conclusion holds: the models take emotion mostly from the text.
5. **Smaller corrections:**
   - X-Voice's synthetic prompts total 10k h, not 10.5k.
   - Chatterbox-Turbo's decoder is one step per the README; the code default is 2 steps.
   - RVC's English README recommends at least 10 min of speech.
   - The kNN-VC leakage quote comes from arXiv 2506.09709, not 2409.17387.
   - Gnani.ai's Vachana TTS is API/on-prem only, not open weights.
6. **Maata's code.** The UI already has a "cloneStrength" setting (closest / balanced / natural) that reaches the engine but changes nothing. Voice building hard-codes cfg 0.5 and uses the first 6 s of the timbre span as the T3 prompt.

## Findings

### A. How ElevenLabs and YouTube handle cloned voices

**A1 [confirmed] ElevenLabs Dubbing Studio (v1): track clone vs clip clone.**
- **What it offers:**
  - A **track clone** is built from all of a speaker's clips. The help centre warns it can be a little unstable when the voice varies.
  - A **clip clone** is built from one specific clip.
  - "Create Voice from Selection" saves a clip's voice for reuse elsewhere.
  - Regenerations are Fixed by default (they keep the clip's duration) or Dynamic.
  - Studio is in maintenance mode: critical fixes only, no new features.
- **Nuance the sweep missed:** the help centre's practical advice is to find one strong clip, clone it, and assign that clone to the whole track for consistency. That is close to Maata's "best single span" (ADR-017).
- Sources: elevenlabs.io/docs/eleven-creative/products/dubbing/dubbing-studio; the help-centre page "What is the difference between a track clone and a clip clone" (both undated, fetched 2026-09-24).

**A2 [confirmed] ElevenLabs Dubbing v2 (alpha) cloning strength.**
- It runs from 0 to 10, default 7.
- Higher values favour similarity to the original speaker but can sound less natural across phonetically distant languages. Lower values trade resemblance for natural delivery in the target language.
- v2 also:
  - claims to preserve emotion, timing and tone;
  - handles up to 32 speakers per file;
  - keeps the original background audio.
- Sources: elevenlabs.io/docs/eleven-creative/products/dubbing and elevenlabs.io/docs/overview/capabilities/dubbing (fetched 2026-09-24).
- The cited changelog (docs/changelog/2026/8/10) announces Dubbing v2 in the API. It says nothing about cloning strength.

**A3 [unverifiable] "Third-party write-ups say a higher cloning strength carries over more of the original accent."** No source was given, and the search budget was exhausted. ElevenLabs' own wording covers only naturalness.

**A4 [confirmed] YouTube auto-dubbing (blog, 2026-02-04).**
- Expressive Speech covers 8 languages: en, fr, de, hi, id, it, pt, es. Telugu is not one of them.
- Auto-dubbing as a whole covers 27 languages.
- Lip sync is still a pilot.
- The post makes no voice-cloning claim.
- Source: blog.youtube/news-and-events/youtube-auto-dubbing-expressive-speech/ (2026-02-04).

**A5 [confirmed] ElevenLabs Instant Voice Cloning guidance.**
- About 1–2 min of clean audio; beyond 3 min adds little.
- Level: −23 to −18 dB RMS, with a true peak of −3 dB.
- One speaker, with no reverb or noise.
- The clone imitates everything it hears, including speed, accent, breathing, noise and mouth clicks.
- For hard accents, use Professional Voice Cloning. A clone's accent can't be changed after it is created.
- Source: elevenlabs.io/docs/eleven-creative/voices/voice-cloning/instant-voice-cloning (undated, fetched 2026-09-24).

### B. Chatterbox internals and Maata's current setup

**B1 [confirmed] Chatterbox Multilingual V3 (released 2026-06-10, MIT).**
- **Model:** the same 0.5B Llama T3 as v2. Training data grew from 25.6k to 36.7k hours.
- **Languages:** the README lists 23 base languages; the launch blog lists 25, adding cs and vi.
- **Language Pack:** six single-language fine-tunes: zh-cmn, es-mx-latam, pt-br, es-es, pt-pt and hi.
- There is no Telugu, Tamil or Bengali.
- Sources:
  - github.com/resemble-ai/chatterbox README (fetched 2026-09-24).
  - resemble.ai/resources/chatterbox-multilingual-v3-… (2026-06-10).
  - HF ResembleAI/chatterbox @5bb1f6ee (2026-06-10), whose files include `t3_mtl23ls_v3.safetensors` and `s3gen_v3.safetensors`.

**B2 [confirmed, strengthened] V3's speaker-similarity claim is unmeasured.** The README claims better voice identity and accent preservation across languages. The launch blog says its evaluation does not measure prosody, speaker similarity or expressive delivery, and that an extended suite is in progress. Source: the same blog (2026-06-10).

**B3 [confirmed] Resemble's README tip on accent leakage.**
- **The tip:** if the reference clip's language differs from the language tag, the output may inherit the reference's accent; set `cfg_weight` to 0 to mitigate it.
- **Other README advice:**
  - Defaults are exaggeration 0.5 and cfg 0.5.
  - Use cfg around 0.3 for fast speakers.
  - For expressive speech, use cfg around 0.3 with exaggeration of 0.7 or more.
- Source: the README (fetched 2026-09-24).

**B4 [confirmed] Chatterbox's guidance acts on the text only.**
- In `T3.prepare_input_embeds`, the unconditional row zeroes only the text embedding (`text_emb[1].zero_()`).
- The conditioning block is shared by both rows:
  - the speaker embedding;
  - the 6 s speech-prompt tokens, passed through a Perceiver resampler;
  - `emotion_adv`, the exaggeration value.
- Maata's `_t3_tokens` repeats `cond + w*(cond − uncond)`.
- So the English prompt's accent and prosody are present in both branches. The only inference levers are the prompt tokens, the speaker embedding, exaggeration and w.
- Sources: `src/chatterbox/models/t3/t3.py` and `t3/modules/cond_enc.py` on master (read 2026-09-24); `engine/src/maata_engine/backends/torch_common.py` lines 205–230.

**B5 [confirmed] Maata's voice conditioning today** (read from the repo, 2026-09-24).
- `session._build_voice` calls `prepare_voice_parts(timbre, pool, timbre, SR, 0.5)`:
  - S3Gen's timbre comes from the speaker's best clean span, at most 10 s.
  - T3's speaker embedding is Chatterbox's VoiceEncoder averaged over up to 60 s of clean clips.
  - T3's prompt is the first 6 s of that same timbre span.
  - cfg is hard-coded at 0.5.
- The host voice (similarity 0.645) takes the older stitched-reference path (`prepare_voice`, then Chatterbox `prepare_conditionals`), per ADR-017's table.

**B6 [unverifiable] "ADR-017 tested cfg 0.3 only with native-Telugu prompts."** ADR-017 records that native Telugu prompt clips ("balanced" / "natural") were measured and rejected. Neither the repo nor docs/ record which cfg was used. The main point, that cfg 0 with the English prompt was never tested, is consistent with the code (B5).

**B7 [confirmed] Chatterbox's VC model is the same S3Gen stage.** `ChatterboxVC.generate` does three steps:
1. It tokenises the source at 16 kHz with S3Tokenizer.
2. It runs `S3Gen.inference` against a reference of at most 10 s of the target voice, loaded from the same `s3gen.safetensors`.
3. It applies the PerTh watermark.

Running it after TTS cannot add timbre that S3Gen didn't already get. Source: `src/chatterbox/vc.py` (read 2026-09-24).

**B8 [confirmed] S3Gen lineage and swappable acoustic files.**
- `s3gen.py` is marked as modified from CosyVoice. It uses `CausalMaskedDiffWithXvec`, CAMPPlus and HiFT, with `S3Tokenizer("speech_tokenizer_v2_25hz")`.
- Resemble ships `s3gen_v3.safetensors` (ResembleAI/chatterbox @5bb1f6ee) and `s3gen_meanflow.safetensors` (ResembleAI/chatterbox-turbo @749d1c1a, 2025-12-15, MIT).
- The Hindi pack repo (ResembleAI/Chatterbox-Multilingual-hi @82ca7127) ships `t3_hi` together with `s3gen_v3`. This is a new data point for which vocoder to pair with a V3-era T3.
- Sources: s3gen.py (read 2026-09-24); HF API file listings (2026-09-24).

**B9 [corrected] "2-step meanflow S3Gen."** The code defaults to 2 CFM steps when `meanflow=True`, against 10 otherwise (`n_cfm_timesteps or (2 if self.meanflow else 10)`). The README describes Turbo's distilled decoder as going from 10 steps to one. Treat 1 or 2 steps as a setting to A/B. Sources: s3gen.py; the README (2026-09-24).

**B10 [unverifiable] CosyVoice2-0.5B's flow plus HiFT (Apache-2.0, HF lastModified 2026-05-31) as a drop-in for S3Gen.** The tokenizer family matches, but compatibility is an inference that no source tests.

**B11 [confirmed] Chatterbox-Turbo and Chatterbox-Nano.** Turbo is 350M, English-only and has paralinguistic tags. Nano is 110M, shares Turbo's architecture and runs on CPU, about 3× real time on 8 cores. Neither is multilingual, so neither can replace the Telugu model. Source: the README (2026-09-24).

**B12 [confirmed] The incumbent shankarpandala/chatterbox-telugu** (HF, created 2026-06-06, sha d4341468, CC-BY-4.0).
- **Base and training:**
  - Base `t3_mtl23ls_v2`.
  - A rank-16 LoRA merged into the weights, plus a retrained text embedding and head.
  - 10k steps on about 34.5k Telugu clips from FLEURS (about 5 h) and IndicVoices-R.
  - SLR66 was excluded for licence reasons. There is no Rasa data.
- **Unchanged from the base:** the acoustic stack (S3Gen, HiFi-GAN, VoiceEncoder).
- No SIM or CER numbers are reported.
- Source: the model card and HF API (2026-09-24).

### C. Training evidence

**C1 [confirmed] Singlish study numbers** ("Singlish, Can or Not?", arXiv 2607.23027, published 2026-07-25).
- **Setup:**
  - A full fine-tune of Chatterbox's T3 only; S3Gen and VoiceEncoder stay frozen.
  - The vocabulary grows to 2454 tokens; 100 epochs = 73,600 steps.
  - Data: 55.5 h from 50 IMDA NSC speakers.
- **Speaker similarity** (ECAPA, against held-out clips of the same speaker):
  - Seen speakers: 0.6036 → 0.6949.
  - Unseen speakers: 0.5364 → 0.6055.
- **Accent similarity** (CommonAccent ECAPA):
  - Seen speakers: 0.5114 → 0.6376.
  - Unseen speakers: 0.4697 → 0.5617.
- **CosyVoice 3 in the same study:**
  - Fine-tuning its LLM helped accent the most.
  - Fine-tuning its flow gave no noticeable change.
  - Fine-tuning its vocoder degraded quality.
  - A learned per-speaker index gave the best match similarity, 0.7383, but only for seen speakers and with the lowest prompt similarity.
- **UTMOS ranking:** real Singlish scored lowest (2.50) and off-the-shelf synthetic speech highest (3.34).
- **Inference settings:** cfg 0.5, exaggeration 0.4, temperature 0.75.
- Source: arXiv 2607.23027 (full text re-read 2026-09-24).

**C2 [corrected] What the Singlish study implies for Maata.** The report says a T3 fine-tune "raised … accent similarity by about 0.09–0.13", presented as a gain Maata can expect.
- **Why that framing is wrong:** the study is same-language (English to English). "Accent similarity" measures how Singlish the output sounds; the fine-tune taught T3 the accent of its training data.
- **The Maata analogue:** training on Telugu data should make the output sound more like native Telugu, which is what Maata wants. The study does not show that fine-tuning reduces cross-lingual English-accent leakage.
- **Transferable estimate:** the unseen-speaker similarity gain, +0.069.
- **Cost:** fine-tuning raised WER from 11.48 % to 15.39 % (toward real accented speech), so Telugu CER must be watched.

**C3 [confirmed] X-Voice (arXiv 2605.05611, v1 2026-05-07; F5-TTS based, 30 languages, no Indian language).**
- With a text-level language ID alone the model still leaked accent. A dual-level language injection (time level plus text level) fixed it.
- Joint guidance: w 2.5 → 4.0 cut WER from 8.85 to 8.62 but lowered similarity from 0.693 to 0.672.
- Decoupled guidance with warm-up reached WER 8.20 at similarity 0.685.
- The authors conclude that a conservative joint CFG scale preserves timbre best.
- The DiT was initialised from F5-TTS v1 Base, which was trained on Emilia, so the weights likely inherit a non-commercial licence.
- Source: arXiv 2605.05611 (text re-read 2026-09-24).

**C4 [corrected] X-Voice's synthetic prompts total "10.5k h".** The paper synthesises about 10k h of speaker-consistent prompts from a 30k-h high-fidelity subset. Stage 2 then fine-tunes with the prompt text masked. The method follows "Cross-Lingual F5-TTS 2". Source: as C3.

**C5 [confirmed] Rasa (ai4bharat, CC-BY-4.0, INTERSPEECH 2024, gated with automatic approval).**
- Telugu: 27.28 h female and 24.98 h male, from 2 speakers.
- Emotions: the six Ekman emotions plus neutral.
- Styles: commands, conversation, news and narration.
- Source: huggingface.co/datasets/ai4bharat/Rasa (HF lastModified 2026-06-06, fetched 2026-09-24).

**C6 [confirmed] IndicVoices-R (ai4bharat, CC-BY-4.0, NeurIPS 2024 Datasets & Benchmarks).**
- 1,704 h from 10,496 speakers in 22 languages, at 48 kHz; 93.25 % extempore speech.
- Metadata: speaker ID, gender and age group.
- It includes a zero-, few- and many-shot speaker-similarity (S-SIM) benchmark.
- Source: huggingface.co/datasets/ai4bharat/indicvoices_r (fetched 2026-09-24).

**C7 [unverifiable] "The incumbent used only part of IndicVoices-R."** The card gives 34.5k clips in total but not IndicVoices-R's Telugu size, which the gated card didn't expose.

### D. Cross-lingual similarity, selection and evaluation

**D1 [confirmed] CosyVoice 3 cross-lingual similarity (arXiv 2505.17589, Table 8).**
- CosyVoice3-1.5B: en2zh SS 66.9 (WER 8.01) and zh2en SS 66.4 (WER 4.32).
- The 0.5B model scores 67.4 and 67.8.
- Source: arXiv 2505.17589 (re-read 2026-09-24).

**D2 [confirmed] Language mismatch drives cross-lingual speaker-verification loss** (arXiv 2607.01161, 2026-07-01). The study's languages are Iberian, not Indic. Source: the arXiv HTML (2026-09-24).

**D3 [confirmed] Seed-TTS-eval similarity scale** (Fun-CosyVoice3-0.5B-2512 card, HF sha 29e01c4e).

| System | test-en SIM | test-zh SIM |
|---|---|---|
| Human | 73.4 | 75.5 |
| Fun-CosyVoice3-0.5B-2512 | 71.8 | 78.0 |
| IndexTTS2 | 70.6 | 76.5 |
| VoxCPM (v1) | 72.9 | 77.2 |
| CosyVoice2 | 65.9 | 75.7 |
| F5-TTS | 64.7 | 74.1 |

The scorer is seed-tts-eval's WavLM-large fine-tuned for speaker verification. Maata's WeSpeaker scores are on a different model, so any comparison is loose.

**D4 [corrected] IWSLT 2026 cross-lingual similarity numbers** ("One Voice, Many Tongues", arXiv 2604.26136, v2 2026-06-25; 20 s VAD-trimmed English references; ECAPA similarity).
- **Report's figures:** "OmniVoice baseline 0.703 / 0.753 / 0.702, after LoRA 0.726 / 0.760 / 0.732".
- **Correct figures:** 0.703 / 0.753 / 0.702 are from the 4-speaker subset. On the full blind set:
  - OmniVoice baseline: 0.734 ar, 0.753 fr, 0.719 zh.
  - After LoRA: 0.726, 0.760, 0.732.
  - Arabic similarity fell after fine-tuning.
- The rough "0.70–0.76 cross-lingual" range still holds.

**D5 [corrected] Best-of-N reranking described as "standard cross-lingual cloning practice".**
- **What the paper did:** generated candidates from three teachers (OmniVoice, VoxCPM, Chatterbox) and kept the best by S = 0.5(1−CER)+0.5·SIM, with CER from Whisper large-v3 and SIM from ECAPA. The goal was a synthetic dataset for per-language rsLoRA training (400 steps).
- **Selection shares:** OmniVoice won 73.4 % (ar), 76.2 % (fr) and 68 % (zh). Chatterbox won 23.8 % of French and 0 % of Arabic and Chinese.
- **What it did not do:** pick takes at inference time. Using the same score per line at inference is a reasonable adaptation, not established practice.
- Source: as D4.

**D6 [confirmed] OmniVoice weights are non-commercial.** The code is Apache-2.0. The model card says the pre-trained model is CC-BY-NC because of its training data, such as Emilia. Source: HF k2-fsa/OmniVoice @c5fdb5cc (2026-07-03), README fetched 2026-09-24.

**D7 [corrected] CosyVoice 3 emotion-cloning numbers** (Table 9).
- **Report's figures:** "0.86–0.98 when the text carries emotion, 0.44–0.64 when neutral".
- **Correct figures for CosyVoice3-1.5B** (happy / sad / angry):
  - Emotion in the text: 0.86 / 0.64 / 0.72.
  - Neutral text: 0.64 / 0.44 / 0.48.
- **With DiffRO-EMO post-training:** 0.98 / 0.68 / 0.84, against 0.98 / 0.50 / 0.68.
- The paper's conclusion stands: TTS systems infer emotion mainly from the text.

**D8 [confirmed] Phrase-level cross-lingual prosody transfer** (Swiatkowski et al., arXiv 2306.11662, 2023-06-20). It scored 6.2 % higher on MUSHRA than utterance-level global prosody transfer and closed 23.2 % of the gap to expressive human dubbing.

**D9 [confirmed] Emotion and duration post-training of CosyVoice2 (Gao et al., arXiv 2609.11523, 2026-09-10).**
- Joint-RL model: ASR error 4.13, speaker similarity 0.7945, duration A15 54.64 %.
- Boundary MOS 4.64, against 3.34 for concatenated CosyVoice3 and 3.99 for concatenated IndexTTS2.
- No code or weights release is mentioned.
- Plain CosyVoice3's speaker similarity was higher, 0.8059.

**D10 [confirmed] PS-TTS (arXiv 2604.09111, v2 2026-05-02).** It achieves isochrony by having a language model paraphrase the translation, then matches vowels by DTW. It beat voice actors on objective metrics for ko↔en dubbing.

**D11 [confirmed] VoxCPM2 separates identity from prosody** (arXiv 2606.06928, 2026-06-05, Table 4). Seed-TTS test-EN similarity:

| Conditioning | SIM |
|---|---|
| Continuation prefix only | 77.7 |
| Isolated reference only | 75.3 |
| Both | 79.5 |

Reference-only gave the best intelligibility on hard Chinese, because it gives the model more freedom over prosody. The authors say cross-lingual quality still depends on data imbalance.

**D12 [confirmed] Phrase-Localized Language-Contrastive Guidance** (arXiv 2609.01016, 2026-09-01, paper CC-BY-4.0).
- Tested on OmniVoice across 12 code-switch directions, none of them Indian.
- Mixed error rate fell from 0.564 to 0.445.
- Language accuracy on the embedded phrases rose from 0.233 to 0.518.
- Speaker similarity moved from 0.973 to 0.969.
- 75.5 % of 488 native-listener ratings preferred it.
- It is designed for non-autoregressive speech language models, so it is untested on Chatterbox's autoregressive T3.

**D13 [confirmed] Reference-free MOS predictors on clean TTS** (arXiv 2609.13150, published 2026-07-02).
- The study tested 13 predictor families (33 variants).
- On clean commercial TTS they collapse to chance. UTMOSv2 reached 0.528 pairwise accuracy against a human ceiling of 0.764, and simply preferring the longer clip tied it.
- **Nuance the sweep missed:**
  - The calibrated composite, the strongest evaluator tested, also reached only 0.52 on that clean set.
  - Optimising a single score caused reward hacking: the metric improved while held-out judges and human listeners got worse.

**D14 [confirmed] Similarity tools and licences.**
- seed-tts-eval scores similarity with WavLM-large fine-tuned for speaker verification. Its UniSpeech checkpoint is CC-BY-SA-3.0: fine for an offline benchmark, but under the policy it is "ask".
- WeSpeaker is Apache-2.0.
- UTMOSv2 is MIT (HF tag; LICENSE read 2026-09-24).

### E. Alternative models and licences

**E1 [confirmed] Licences of the non-commercial or custom models** (HF API, 2026-09-24).

| Model | Licence | HF sha / date |
|---|---|---|
| SWivid/F5-TTS | CC-BY-NC-4.0 | 84e5a410, 2025-03-21 |
| amphion/MaskGCT | CC-BY-NC-4.0 | — |
| amphion/Vevo | CC-BY-NC-4.0 | — |
| amphion/Vevo1.5 | CC-BY-NC-ND-4.0 | — |
| IndexTeam/IndexTTS-2.5 | "other" (custom) | c39ce5ba, created 2026-08-10 |

IndexTTS-2.5's languages are zh, en, ja, es and ar.

**E2 [confirmed, extended] IndexTTS licence** (bilibili Model Use License, LICENSE in index-tts/index-tts, read 2026-09-24).
- It is royalty-free.
- A separate licence is needed above 100M monthly active users or RMB 1B annual revenue.
- Fine-tunes, LoRAs and quantisations count as Derivative Works.
- Recipients downstream must be bound by the same terms.
- **Missed by the sweep:** the licence forbids using IndexTTS or its derivatives to improve any other AI model, except IndexTTS itself or non-commercial models. So IndexTTS cannot even serve as a teacher or prompt synthesiser for Chatterbox-Telugu.

**E3 [confirmed] MOSS-TTS-v1.5** (OpenMOSS, Apache-2.0, sha cdd3b911, 2026-05-26). It uses the 8B MossTTSDelay API and offers zero-shot cloning plus token-level duration control (`tokens=325`). Its 31 languages include hi but not te. Source: the HF card and API.

**E4 [unverifiable] "CosyVoice3 offers only speed and instruct control, not a hard duration target."** Not re-checked today.

**E5 [confirmed] Qwen3-TTS** (Apache-2.0; 12Hz-1.7B-Base sha fd4b2543, 2026-01-23).
- **Languages:** 10 (zh, en, ja, ko, de, fr, ru, pt, es, it); no Telugu.
- **Base models:** do 3-second cloning and are offered for fine-tuning.
- **Vendor same-language speaker similarity**, 12Hz-1.7B against ElevenLabs:

| Language | Qwen3-TTS 12Hz-1.7B | ElevenLabs |
|---|---|---|
| en | 0.775 | 0.613 |
| de | 0.775 | 0.614 |
| es | 0.814 | 0.615 |
| fr | 0.714 | 0.535 |
| ru | 0.792 | 0.676 |

- Source: the HF card (fetched 2026-09-24).

**E6 [confirmed] aguken-ai Qwen3-TTS Indic LoRAs** (Apache-2.0, sha 5450c7fc, 2026-07-02).
- 30 adapters (15 Indic languages × 2 genders), built on Qwen3-TTS-12Hz-0.6B-Base.
- LoRA rank 16 and alpha 32, learning rate 2e-6, up to 500 Rasa clips per language and gender.
- `telugu_female` and `telugu_male` each include a learned speaker embedding. These are single voices, not zero-shot cloners, and no quality numbers are given.

**E7 [confirmed] VoxCPM2** (Apache-2.0, HF sha 32279eff, 2026-08-18; released 2026-04).
- **Model and languages:** 2B parameters. 30 languages, including hi but not te. The README invites testing or fine-tuning for unlisted languages.
- **Fine-tuning:** SFT and LoRA are supported from as little as 5–10 min of audio.
- **Cloning tip:** passing the same clip as both `reference_wav_path` and `prompt_wav_path` gives maximum similarity.
- **Vendor similarity:** Hindi 85.6 against ElevenLabs 73.0 on MiniMax-MLS.
- **Speed and memory:** on an RTX 4090 in PyTorch, RTF 0.30 using about 8 GB.
- Sources: the README and arXiv 2606.06928 (2026-09-24).

**E8 [corrected] VoxCPM2 "RTF about 1.76 on an M4 Pro" as its Apple-Silicon speed.** The README gives RTF about 1.76 for the llama.cpp-omni Q8_0 GGUF build on Metal on an M4 Pro. Speed through mlx-audio or MPS is not published. `sota_ranking.md` attaches the figure to mlx-community/VoxCPM2-4bit, which is also wrong.

**E9 [corrected] "Qwen3-TTS and VoxCPM2 are the strongest permissive zero-shot cloners."**
- **Where the claim holds:** on vendor same-language tables.
- **Where it doesn't:** Maata's case is English reference to foreign-language output. In the independent IWSLT 2026 benchmark (blindset-4, ECAPA):

| System | Arabic | French | Chinese |
|---|---|---|---|
| Chatterbox | 0.680 | 0.619 | 0.653 |
| VoxCPM2 | 0.607 | 0.575 | 0.569 |
| Qwen3-TTS | no Arabic | 0.533 | 0.522 |
| OmniVoice (non-commercial) | 0.703 | 0.753 | 0.702 |

- The sample is small (4 speakers, automatic metrics only), but it is the only head-to-head English-to-X comparison found.
- Source: arXiv 2604.26136, Table 2.

**E10 [confirmed] mlx-audio (MIT) ports** Chatterbox v2 and v3, Qwen3-TTS, VoxCPM2 (bf16, 8-bit, 4-bit), MOSS-TTS and MOSS-TTS-Nano, and OmniVoice (non-commercial). mlx-community/chatterbox-multilingual-v3 is MIT (sha 03565773, 2026-07-31). mlx-audio's pyproject requires `transformers>=5.14.0`, which clashes with Maata's transformers==5.2.0 pin. Sources: the mlx-audio README and pyproject; HF API (2026-09-24).

**E11 [confirmed] snorTTS-Indic-v0** (HF sha c771394e, 2026-07-13).
- It is tagged Apache-2.0, but it is LLaMA-3.2-3B plus SNAC 24 kHz.
- Its base, canopylabs/3b-hi-pretrain-research_release, is tagged `license:llama3.2`.
- It claims cloning, code-switching and cross-lingual cloning in 9 Indic languages, including te.
- Under the licence policy it needs the maintainer's approval.

**E12 [unverifiable] Other HF Telugu TTS repos** (higgs-telugu LoRAs, orpheus-telugu QLoRAs, pocket-tts-telugu, IndicF5 derivatives, OpenBible-Telugu). Not re-checked today.

**E13 [corrected] "Gnani.ai announced a 12-language Indic cloner; open weights unconfirmed."** The product is Vachana TTS, launched 2026-02-19. It clones from under 10 s of audio in 12 languages, Telugu included. It is offered via API with on-prem deployment and is not open source, so it can't be used in Maata. Source: businesstoday.in (2026-02-19, fetched 2026-09-24).

### F. Post-hoc voice conversion

**F1 [confirmed] Seed-VC and WavLM licences.**
- Seed-VC is GPL-3.0: reject.
- microsoft/UniSpeech (the WavLM checkpoints) is CC-BY-SA-3.0: share-alike but not non-commercial, so "ask", not "never".
- Sources: the LICENSE files (read 2026-09-24).

**F2 [confirmed] MeanVC2** (ASLP-lab, Apache-2.0; HF sha 39cdd195, 2026-08-03; arXiv 2606.09050, 2026-06-08).
- About 18M parameters; the Vocos vocoder outputs 16 kHz.
- Its speaker encoder is ECAPA-TDNN on WavLM.
- The README says FunASR models auto-download from ModelScope on first use, which breaks the no-auto-download rule.

**F3 [confirmed] The RVC stack is permissive.**

| Component | Licence | Latest date |
|---|---|---|
| RVC-WebUI | MIT | last commit 2026-08-04 |
| lj1995/VoiceConversionWebUI weights | MIT | sha e6d0c1a1, 2026-08-01 |
| ContentVec | MIT | — |
| RMVPE | Apache-2.0 | — |
| Applio | MIT | last commit 2026-09-23 |

- RVC's base model was pre-trained on nearly 50 h of VCTK.
- RVC uses top-1 retrieval to reduce tone leakage.
- **Missed by the sweep:** Applio now says it will no longer receive frequent updates. Its official build also carries its own Terms of Use on top of MIT.

**F4 [corrected] "RVC's stated goal is a good model from 10 minutes or less."** The English README recommends at least 10 min of low-noise speech. Maata has that much per podcast speaker, but not for short videos. Source: `docs/en/README.en.md` (read 2026-09-24).

**F5 [confirmed] OpenVoice** is MIT, with its last commit on 2025-04-19.

**F6 [unverifiable] "OpenVoice V2 is weak."** The only support is a secondary review, and no primary benchmark was found.

**F7 [corrected] kNN-VC leaks the reference language.**
- **Attribution:** the leakage statement comes from arXiv 2506.09709 (2025-06-11), not from 2409.17387. That paper's example is /r/ changing from rolled to guttural.
- **Numbers:** its table gives kNN-VC WER 96.66 and CER 63.34, the worst in the table. It also gives kNN-VC the highest similarity (96.5).
- **Lesson:** similarity can rise while intelligibility and phonetics collapse. The language pair was not re-verified.

## New findings the researcher missed

1. **Chatterbox leads the permissive models cross-lingually** (E9). In the only head-to-head English-reference benchmark found (IWSLT 2026), it beat Qwen3-TTS and VoxCPM2. This demotes the "reserve arm" further.
2. **Maata already has the "voice match strength" plumbing, but it does nothing.**
   - `cloneStrength` (closest / balanced / natural) exists in `app/src/lib/settings.ts` and `voice.ts`, `engine/src/maata_engine/server.py:88` and `session.py:114`.
   - The session only stores it. `_build_voice` hard-codes cfg 0.5 and reuses the timbre span as the T3 prompt (`session.py:314`).
   - Recommendation R9 is therefore about wiring up an existing control, not building a new one.
3. **ElevenLabs Dubbing v2 supports Telugu (te).** A maintainer-made ElevenLabs Telugu dub of the test podcast could serve as an external listening reference for the benchmark. Using it is the maintainer's call: it is a cloud service and must never be in the product path.
4. **V3's "accent preservation" wording is a risk for Maata** (B2). If V3 keeps the reference's accent more strongly, an English prompt could leak more English accent into Telugu. Resemble publishes no measurement either way.
5. **The IndexTTS licence also bars using it as a teacher** (E2).
6. **Selecting on similarity alone can be gamed** (D13, F7):
   - Single-score optimisation causes reward hacking.
   - kNN-VC had the highest similarity alongside unusable WER.
   - Speaker-verification embeddings shift with language (D2), so picking the highest-similarity take may favour takes with more English accent. This is an inference; the per-line score must include CER and a listening spot-check.
7. **The Singlish fine-tune raised WER** (C2). A Telugu retrain needs a CER gate as well as a similarity gate.
8. **Gnani Vachana is API-only** (E13). One open question is closed.
9. **Resemble pairs the Hindi pack's T3 with `s3gen_v3`** (B8). That is useful for choosing the vocoder for a V3-based Telugu retrain.

## Refuted
None. Every checked claim was confirmed, corrected or left unverifiable.

## Corrections to `../sota_ranking.md` section 4
- **VoxCPM2 speed:** the "RTF about 1.76 on an M4 Pro" belongs to llama.cpp-omni Q8_0 on Metal, not to mlx-community/VoxCPM2-4bit. Its MLX speed is unmeasured (E8).
- **MeanVC2 and kNN-VC:** they were "rejected: WavLM is CC-BY-SA-3.0". Under the policy CC-BY-SA is "ask", not "never". The practical verdict stays reject, for 16 kHz output, auto-download (MeanVC2) and phonetic leakage (kNN-VC) (F1, F2, F7).
- **VoxCPM2 as a fallback:** its fallback status should note that it scored below Chatterbox cross-lingually in IWSLT 2026 (E9).
- **Vocoder pairing:** sota arm 2 says Resemble's general V3 demo pairs the v3 T3 with the v2 vocoder. The Hindi pack instead ships its T3 with `s3gen_v3` (B8). Test both pairings, as sota already proposes.
- **Two additions:**
  - Chatterbox's VC is S3Gen and adds nothing (B7).
  - `s3gen_meanflow` (MIT) is a speed arm at 1 or 2 steps (B8, B9).

## Recommendations for Maata (updated for the corrections)

**R1. Build a voice benchmark before changing any model.** Effort S. No approval is needed for similarity scoring.
- **Clips:**
  - Lines per speaker from the podcast.
  - Self-recorded bilingual clips of the maintainer (English and Telugu), to measure the true English-to-Telugu same-speaker ceiling (D1–D4).
- **Metrics:**
  - WeSpeaker similarity (Apache-2.0).
  - ECAPA similarity, for comparability with IWSLT and the Singlish study. The model's licence must be checked before adding it.
  - Telugu CER.
  - A blind native-listener A/B for accent and "sounds like him".
- **Rules:**
  - Combine the metrics into one score and check it against listening. Never gate on UTMOS, and never optimise a single score (D13).
- **Optional external reference:** the maintainer can make an ElevenLabs Dubbing v2 Telugu dub (new finding 3). It is for listening only and needs the maintainer's decision.

**R2. Sweep the inference settings and wire up the existing `cloneStrength` control.** Effort S. No approval needed.
- **Settings:**
  - cfg_weight ∈ {0, 0.25, 0.5}.
  - T3 prompt ∈ {current (first 6 s of the timbre span), a different clean English span, the matching source line, none}.
  - Exaggeration ∈ {0.4, 0.5, 0.7}.
- **Why:** Resemble's cfg 0 advice (B3) has never been tested with an English prompt (B5, B6). CFG leaves the prompt in both branches (B4), and X-Voice shows a guidance trade-off between WER and similarity (C3).
- **Then:** map the winning settings to closest / balanced / natural (new finding 2). Default toward natural Telugu, as ElevenLabs' strength dial implies (A2).

**R3. Build identity from more audio.** Effort S–M. No approval needed.
- Average S3Gen's CAMPPlus x-vector over the same clean pool that T3's embedding already uses. Today S3Gen sees only one span of at most 10 s (B5).
- Choose the S3Gen span by similarity to the speaker's centroid, not only by cleanliness.
- Aim for 1–2 min of identity audio (A5).
- Also re-test the host, who still uses the stitched path.

**R4. Generate 2–3 takes per line within the 5–10 min lookahead and keep the best.** Effort M. Similarity-only selection needs no approval; a Telugu ASR for CER does.
- Score with a composite, such as 0.5(1−CER)+0.5·SIM, plus an accent or listening spot-check.
- This adapts the IWSLT selection score to inference. The paper itself used it only to build training data (D5).
- Guard against similarity gaming (new finding 6).
- Pay for the extra takes with the GPU time freed by moving translation to the Claude CLI and with a 1- or 2-step `s3gen_meanflow` (B8, B9).

**R5. Per-line "clip clone" prosody.** Effort M. No approval needed.
- For emotional or emphatic lines, give T3 the matching source segment as its prosody prompt. Keep identity (the speaker embedding and the S3Gen reference) at the track level.
- Gate each line on CER and accent, and fall back to the track prompt when it fails.
- **Evidence:** D7, D8, D11. ElevenLabs' own advice favours one strong clip for consistency (A1), so identity stays single-source.

**R6. Retrain Chatterbox-Telugu on the V3 T3 as a full fine-tune.** Effort L. No new licence (MIT, CC-BY-4.0). Needs an NVIDIA or rented GPU, an ADR and a new pin.
- **Data:** all of IndicVoices-R Telugu, plus Rasa Telugu for emotion, plus FLEURS.
- **Cross-lingual prompts:** optionally condition on English prompts of the same voice, following X-Voice's synthetic-prompt method (C3, C4). Make them only with licence-clean models such as Chatterbox itself, never IndexTTS (E2).
- **Frozen:** S3Gen and the vocoder.
- **Test:** both `s3gen.pt` and `s3gen_v3` (B8).
- **Realistic expectation:**
  - Unseen-speaker similarity: up to about +0.07 (C2).
  - Accent: more native-sounding Telugu.
  - WER: watch it (C2).
  - V3's own similarity claim is unmeasured (B2).

**R7. Optional: per-speaker RVC or Applio conversion trained on each speaker's own English.** Effort M–L. Needs approval for a new dependency and an on-device training step.
- Train on at least 10 min per speaker (F4).
- Keep index_rate low to limit English phonetics.
- Watermark after conversion.
- Note Applio's maintenance status (F3).
- How well it works across languages is unmeasured.

**R8. Reserve arm, demoted: a Telugu fine-tune of Qwen3-TTS or VoxCPM2.** Effort L. Needs approval for a new runtime and a training run.
- **When:** only if R6 stalls and a cross-lingual benchmark shows them ahead of Chatterbox. Today's evidence points the other way (E9).
- **Speed:** VoxCPM2's MLX speed is unknown (E8).
- **Dependency clash:** mlx-audio clashes with the transformers pin (E10).

**R9. Wire up the existing `cloneStrength` setting** (folded into R2). Effort S. No approval needed.

**R10. Do not pursue:**
- Seed-VC (GPL-3.0).
- OmniVoice, F5-TTS, MaskGCT and Vevo (non-commercial).
- IndexTTS2/2.5: a custom licence, no Telugu, and it cannot be used as a teacher.
- Gnani Vachana (API-only).
- Chatterbox VC (a no-op).
- MeanVC2 and kNN-VC (16 kHz output, auto-download, phonetic leakage).
- OpenVoice v2 (unproven).
- snorTTS-Indic, unless the maintainer accepts the Llama-3.2 terms.

## Open questions
1. What is the true English-to-Telugu same-speaker similarity ceiling? It needs bilingual self-recordings.
2. Does cfg 0 cut English accent on the Telugu fine-tune, and at what cost in similarity and CER?
3. Does picking takes by similarity favour English-accented takes? Measure with listeners.
4. Does V3's "accent preservation" mean more reference-accent carry-over? How much does a V3 re-fine-tune gain?
5. Which vocoder pairs best with a V3-era Telugu T3: `s3gen.pt`, `s3gen_v3` or `s3gen_meanflow`? And at 1 step or 2?
6. Is the CosyVoice2 flow plus HiFT (Apache-2.0) a working drop-in for S3Gen?
7. How long does RVC or Applio training take on the M5 Pro, what artefacts does it add across languages, and does the watermark survive?
8. Which local Telugu ASR should compute CER, and under what licence? Whisper-te was too noisy (ADR-014).
9. Will the maintainer accept an ElevenLabs Telugu dub as an offline listening reference?
10. How good is snorTTS-Indic, and will the maintainer accept the Llama-3.2 terms?
11. Does keeping background audio or room tone raise perceived similarity? This ties to the maintainer's open question about original audio.
12. Unchecked today:
    - RVC timing on Apple Silicon.
    - Whether denoising the reference raises similarity.
    - The HF Telugu repos in E12.
    - The ECAPA licence for R1.
