# 07 Audio realism: separation, background, acoustics and mixing (verified, 2026-09-24)

This is an adversarial check of the researcher's sweep (`07-audio-realism.sweep.md`), done on 2026-09-24. It builds on `sota_ranking.md`. That file has no audio-mixing stage, so nothing in it needed correcting here.

**How the claims were checked:**
- Every claim that a recommendation depends on was re-checked against a primary source: papers (arXiv or ar5iv full text), official docs, the Hugging Face and GitHub APIs, PyPI, the local macOS 27.0 SDK headers and the repo itself.
- No model was run and no M5 Pro number was produced.
- The web-search budget for this session was already used up (200/200), so checking relied on direct page fetches and APIs. Where a fact could only have been found by search, it is marked unverifiable.

**Verdict markers:**
- [confirmed]: the primary source says it.
- [corrected]: the claim was partly wrong. The correct fact and its source are given.
- [unverifiable]: this session could not confirm it.
- Refuted claims are listed at the end.
- **Disclosed** means a company or author states it. **Inferred** means it is our reasoning.

## Summary

1. **Pacing and translation come before the acoustic bed.** The only controlled dubbing test is Amazon's 2020 study, and it was stronger against the acoustic bed than the sweep reported.
   - Adding the separated *original* background plus re-reverb raised naturalness for non-native listeners (+10.34 MUSHRA, B to C, p<0.05). Even there, the A vs C difference (+11.01) was not significant.
   - Native (Italian) listeners ranked the plain baseline A best, because the prosodic alignment added in B made the speaking rate unnatural.
   - The measured gain came from the original background (the equivalent of our option C). It does not come from synthetic room tone.
2. **Professional practice keeps the music-and-effects track and matches the dub to the original.**
   - ElevenLabs retains the background by default.
   - Netflix mixes the dub at levels similar to the original version and leaves the M&E unaltered.
   - iZotope Dialogue Match matches EQ, reverb and a *synthesised* ambience.
   - All confirmed.
3. **Under the rule "original audio never played", the best option is still synthetic room tone plus loudness, EQ and reverb matching.** It is cheap, needs no new dependency, and is robust to pause, seek and rate changes.
   - Its evidence is indirect: the telephony comfort-noise rationale (RFC 3389) and ADR practice (iZotope's synthesised Ambience).
   - No listening test of dubbing has measured it.
4. **Playing a separated background bed has real costs.**
   - The YouTube API Developer Policies (last updated 2026-09-14) forbid it: III.I.7 bars separating, isolating or modifying the audio of YouTube content, and its example bars applying alternate audio tracks.
   - English residue would remain in the bed.
   - The original speakers' laughs would leak into the effects stem.
   - The policies also bear on the core product. The sweep missed III.I.14, which bars using any technology other than YouTube API Services to access any portion of YouTube audiovisual content (yt-dlp), and III.E.6 (scraping). This is for the maintainer and legal review, not an engineering call.
5. **Cleaning the cloning reference must be tested, not assumed.**
   - One 2026 paper found that Sidon-enhanced prompts raised quality but lowered speaker similarity (F5-TTS SECS 0.35 to 0.28, not 0.355 as the sweep said).
   - Licence-clean candidates on the Mac: AUSoundIsolation (a system audio unit), Kim Mel-Band RoFormer (MIT, MLX), MossFormer2_SE_48K (Apache-2.0, MLX) and DeepFilterNet (MIT/Apache-2.0, MLX).
6. **Laughs and sighs can be carried over by synthesis.**
   - CED (Apache-2.0) detects AudioSet Laughter, Chuckle, Sigh, Gasp, Breathing, Cough, Throat clearing, Sniff and Crying.
   - Chatterbox-Turbo and Nano (MIT, English) have matching tags: `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[cough]`, `[clear throat]`, `[sniff]`, `[crying]` and others.
   - Multilingual V2/V3 have no tags.

**Counts** (23 report findings plus sub-claims):
- Confirmed: 17
- Corrected: 6
- Unverifiable: 3
- Refuted: 0

---

## Findings

### A. Evidence on what makes a dub sound real

**A1. Amazon: background plus re-reverb helps non-native listeners; native listeners were dominated by pacing. [corrected]**

Source: Federico et al., "From Speech-to-Speech Translation to Automatic Dubbing", arXiv 2001.06785 (2020-01, ICASSP 2020); full text read via ar5iv on 2026-09-24.

Confirmed:
- **Audio rendering:**
  - A U-Net with soft ratio masks separates foreground from background. It was trained on 360 h of speech and 120 h of background.
  - RT60 is estimated blind by maximum likelihood (Löllmann et al.).
  - A synthetic RIR from Habets' RIR generator is convolved with the dub, which is then remixed with the extracted background.
- **Test:** MUSHRA with 14 listeners (5 Italian, 9 non-Italian) and 657 ratings on 24 clips of 10–15 s from 6 MuST-C TED talks, English to Italian.
- **Non-Italian listeners:** B to C was +10.34 (47.74 to 58.08, p<0.05). C had median rank 1.0, and C beat A 66% of the time against 52% for B.
- **Italian listeners:** B to C was +1.05, not significant. They blamed a speaking rate that was "too slow, too fast, or too uneven".

Corrections:
- For native listeners, adding prosodic alignment (A to B) *lowered* naturalness by 10.93 (p<0.01), and the baseline A had the best median rank (1.0). C beat A only 31% of the time.
- For non-native listeners, A vs C was +11.01 but *not significant*.
- The background in C was the original's separated background. The study is therefore direct evidence for a separated bed (option C), not for synthetic room tone (option B).

Kind: measured.

**A2. Line-level loudness and pitch of professional dubs track the source line. [confirmed]**

Source: Brannon, Virkar and Thompson, "Dubbing in Practice", arXiv 2212.12137 (2022-12-23; TACL vol. 11, 2023).

- The study covers 319.57 h from 54 professionally dubbed titles.
- Source to dub correlations:
  - energy mean: r=0.381;
  - pitch mean: r=0.792;
  - speaking rate: r=0.439 on all lines, 0.584 for lines of 1 s or more.
- Adding source-line energy to a speaker-only model raises r² from 0.184 to 0.268. Every increase is significant at α=10⁻⁶ by F-test.
- Word-aligned pairs correlate 0.08–0.11 more than unaligned pairs.

Kind: measured.

**A3. ElevenLabs Dubbing keeps the original background; dropping it is an advanced option. [confirmed]**

Sources: https://elevenlabs.io/docs/overview/capabilities/dubbing and https://elevenlabs.io/docs/api-reference/dubbing/create. Both pages are undated; fetched 2026-09-24.

- The docs say dubbing retains the original background audio, and list "Keep background audio" as a feature.
- `drop_background_audio` is "an advanced setting", suggested for speeches or monologues.
- `foreground_audio_file` and `background_audio_file` are "for use only with csv input".
- Dubbing Studio "is in maintenance mode and receives critical bug fixes only".
- The separation model is not disclosed.

Kind: disclosed.

**A4. YouTube does not document how auto-dubbing handles the background; a competitor says the reused background sounds distorted. [confirmed]**

Sources:
- YouTube Help answer 15569972 (undated; fetched 2026-09-24).
- BeMultilingual blog, 2026-01-23. The publisher is a human-dubbing company, so it is biased.

Details:
- The help page says only that dubs "might contain errors due to … background noise". It excludes videos longer than 120 min, videos with no speech or only music, and speech that is too fast.
- BeMultilingual says YouTube reuses the original background and that the result is usually "very distorted".
- Inferred: listeners notice separation artefacts in a bed.

Kind: disclosed, plus a biased opinion.

**A5. Dialogue-vs-rest is the easy split: about 15 dB SDR on real film. [confirmed; the consequence drawn from it is inferred]**

Source: Uhlich et al., SDX23 Cinematic Demixing, arXiv 2308.06981 (2023-08).

- The CDXDB23 test set is 11 Sony Pictures films: 156 clips, about 28.7 min.
- Leaderboard B (any training data):

| Team | Global SDR | Dialogue | Effects | Music |
|---|---|---|---|---|
| JusperLee | 8.181 | 14.619 | 3.958 | 5.966 |
| Audioshake | 8.077 | 14.963 | 4.034 | 5.234 |
| ZFTurbo | 7.630 | 14.734 | 3.323 | 4.834 |

- Leaderboard A (DnR-only training): the best global SDR was about 4.35 (aim-less), with dialogue 7.98.
- Inferred: a mix-minus-dialogue bed would carry dialogue-estimate error at roughly 15 dB below the speech level, and some of that error is English speech.

Kind: measured, with the consequence inferred.

**A6. Separators trained on read speech put laughter and screams into the effects stem. [confirmed]**

Source: Hasumi and Fujita, "DnR-nonverbal", arXiv 2506.02499 (v1 2025-06-03; Interspeech 2025); dataset on Zenodo 15470640.

- Emotional voices such as laughter and screams are "more easily separated as an effect, not speech".
- Consequence for Maata: a separated bed would replay the original speakers' laughs, and cleaning a reference could strip them.

Kind: measured.

### B. Separation and enhancement models (licences, runtimes)

**B1. BandIt Plus (a permissive dialogue/music/effects separator) with a published score. [corrected]**

Source: ZFTurbo MSST `docs/pretrained_models.md` (fetched 2026-09-24); the repo is MIT and was last pushed 2026-09-09.

Confirmed:
- `model_bandit_plus_dnr_sdr_11.47.chpt`, release v.1.0.3, scores a DnR test average of 11.50 (speech 15.64, music 9.18, effects 9.69).
- Vocals on Multisong from the same page:

| Model | Vocals SDR |
|---|---|
| BS PolarFormer | 11.00 |
| Kim MelBand | 10.98 |
| BS-RoFormer viperx | 10.87 |
| MDX23C | 10.17 |
| HTDemucs4, MVSep fine-tune | 8.78 |
| HTDemucs4 FT, official | 8.38 |

- SCNet XL IHF scores a MUSDB18 test average of 10.08.

Corrections:
- **Wrong record:** DnR **v2** is Zenodo **6949108** (version 2.0, CC-BY-4.0; it fixes v1's annotation errors and grows the test set from 652 to 973 mixtures). Zenodo 5574713 is v1.0.
- **Licence caveat:** DnR's CC-BY-4.0 label hides mixed clip licences. FSD50K (Zenodo 4060432) says each clip keeps its Freesound licence, "some forbidding further commercial reuse" (CC-BY-NC). So BandIt Plus falls under the same data-inheritance ruling (#8 in `sota_ranking.md`) as the MUSDB-trained models.
- **No licence on the weights:** the release weights carry no separate licence.
- **Not tested on real audio:** DnR is a synthetic test set.
- **No MLX runtime found:** see B4. BandIt runs through MSST on PyTorch.

**B2. Kim Mel-Band RoFormer is now MIT and has a parity-tested MLX port. [confirmed]**

Sources: Hugging Face API commit list for `KimberleyJSN/melbandroformer` and the `mlx-community/mel-roformer-kim-vocal-2-mlx` model card, both fetched 2026-09-24.

- **Licence history:**
  - 2024-08-06: initial upload, no licence.
  - 2025-06-17 (f45f9e3d): GPL-3.0.
  - 2026-04-22 (ac9b0614): MIT, the current licence.
- **MLX port:**
  - MIT, current sha 64cbfcb0 (last modified 2026-05-01).
  - `model.safetensors` is 456,483,463 B in bf16, about 228M parameters, at 44.1 kHz stereo.
  - Converted 2026-04-25 with xocialize/mlx-audio @8380ab8. PyTorch-to-MLX parity is 66.08 dB SDR.
  - The card says it is a music vocal model and not validated for general-purpose separation.
- **Data caveat:** MUSDB18 is "academic purposes only", and 46 tracks (CC BY-NC-SA 4.0) plus 2 (CC BY-NC-SA 3.0) are non-commercial (sigsep.github.io, fetched 2026-09-24). Kim Vocal 2's training data is undisclosed.

**B3. macOS includes a voice-isolation audio unit; the Audio Mix unit cannot take YouTube's stereo. [confirmed]**

Sources: local SDK `MacOSX27.0.sdk` headers `AUComponent.h` and `AudioUnitParameters.h`; WWDC25 session 251, "Enhance your app's audio recording capabilities" (2025-06).

- **AUSoundIsolation** (`'vois'`), macOS 13+:
  - `WetDryMixPercent` runs 0–100 (default 100).
  - `SoundToIsolate` has `HighQualityVoice` = 0 (macOS 15+) and `Voice` = 1 (the default).
  - QuietNow (MIT) says Apple Music Sing is built on it.
- **AUAudioMix** (`'amix'`), macOS 26+:
  - It offers Cinematic, Studio and In-Frame styles, plus foreground-only and background-only stems.
  - WWDC25 says its input is four channels of FOA spatial audio plus `SpatialAudioMixMetadata`, computed when an iPhone spatial recording stops. It can't take YouTube stereo.
- Quality against RoFormer models is unmeasured, and both units are Mac-only.

**B4. MLX runtimes for separators. [corrected]**

Sources: `ssmall256/mlx-audio-separator` README and `models.json` (MIT, pushed 2026-09-23); `Blaizzy/mlx-audio` `pyproject.toml` (MIT, pushed 2026-09-24); PyPI `demucs`.

Confirmed:
- **mlx-audio-separator** runs Roformer, MDXC, MDX, VR and Demucs.
  - Validation snapshot 2026-02-24..26: 163/163 models ok.
  - Speedup against PyTorch audio-separator on an M4 mini: 1.40× htdemucs_ft, 2.16× BS-RoFormer ep317, 2.50× Mel-Band instv7n, 1.53× MDX-Inst_HQ_3; median 1.847×.
  - Converted Demucs weights are cached under `~/.cache/mlx-audio-separator`.
- **mlx-audio** declares `transformers>=5.14.0`, which conflicts with the 5.2.0 pin. Its `sts/models` includes mel_roformer, deepfilternet, mossformer2_se, sam_audio and dialogue_sidon.
- **Demucs:** facebookresearch/demucs is archived (MIT). demucs 4.1.0 appeared on PyPI on 2026-07-11, from the adefossez fork. It is a PyPI release, not a GitHub release.
- **GUI speed claim:** the mlx-audio-separator GUI README says a 3–4 minute song "typically separates in well under a minute". This is anecdotal, and the GUI needs macOS 15.

Corrections:
- "MLX runtimes exist for every major separator family" is overstated. The mlx-audio-separator catalogue has no BandIt, DnR or dialogue/music/effects model and no SCNet. The only permissive dialogue/music/effects separator with a score (B1) therefore has no MLX path found.
- The same catalogue also ships non-commercial weights (the Sucial dereverb-echo models). Any use needs a per-model allowlist, not the catalogue.

**B5. Enhancing the reference raises quality but can lower speaker similarity. [corrected (numbers)]**

Source: Giraldo et al., arXiv 2602.05770 (2026-02-05), WildSpoof 2026 TTS track, on in-the-wild TITW data.

- Sidon-enhanced prompts:
  - F5-TTS: UTMOS 3.51 to 3.89, DNSMOS 3.11 to 3.31, WER 0.08 to 0.07.
  - StyleTTS2: UTMOS 3.37 to 3.97.
  - The paper says speaker similarity scores "are degraded".
- Prompt length: moving from 5.5 s to 7.7 s raised F5-TTS SECS from 0.24 to 0.35, while UTMOS fell from 3.62 to 3.51.
- The authors attribute HierSpeech++'s intelligibility loss to MP-SENet's content-preservation issues.
- **Correction:** F5-TTS SECS is **0.35 to 0.28** (not 0.355), and StyleTTS2 is 0.19 to 0.18.
- Small study; Chatterbox was not tested.

Kind: measured.

**B6. Sidon is MIT and has a CoreML port, but it re-synthesises speech. [confirmed]**

Sources: arXiv 2509.17052 (v1 2025-09-21, v3 2026-01-26); HF `sarulab-speech/sidon-v0.1` @b3b02d8b; HF `seanll95/sidon-coreml` @edf5a3b6.

- **Model:** MIT, 2025-12-15. A fine-tuned w2v-BERT 2.0 feature predictor followed by a vocoder, trained on FLEURS-R and LibriTTS-R. The paper claims quality comparable to Miipher and speeds up to 500× real time on a GPU.
- **CoreML port:** MIT, 2026-09-16.
  - 16 kHz in, 48 kHz out.
  - fp16 encoder 468 MB, decoder 101 MB.
  - 11.2× real time on an iPhone warm, 1.7 dB below the GPU version.
  - The card says it "re-synthesises speech from w2v-BERT semantic features rather than repairing" the audio.
- **DialogueSidon** @d43d7478 is CC-BY-NC-4.0: never use it.

**B7. Enhancement and dereverb licences. [confirmed]**

Sources: HF and GitHub APIs, 2026-09-24.

Permissive:
- **DeepFilterNet:** the repo ships LICENSE-MIT and LICENSE-APACHE (last pushed 2024-10-17). MLX weights `mlx-community/DeepFilterNet-mlx` @220d5dfb (MIT, 2026-03-11): v1, v2 and v3 at about 7–9 MB each, 48 kHz, with stateful streaming for v2 and v3 in mlx-audio.
- **ClearerVoice-Studio:** Apache-2.0. `alibabasglab/MossFormer2_SE_48K` @eff8c979 is Apache-2.0. Its MLX port `starkdmi/MossFormer2_SE_48K_MLX` @ccd0ded0 (Apache-2.0) ships fp32, fp16, int8, int6 and int4, and auto-chunks audio over 60 s.
- **Resemble Enhance** (HF @4e3510ce), **VoiceFixer** and **MP-SENet:** all MIT.

Not permissive:
- **anvuew** `dereverb_mel_band_roformer` @cef05ad2, `dereverb_bs_roformer` @bd5c6b55 and `dereverb_room` @0b85f5b8: GPL-3.0, which needs a ruling.
- **Sucial** `Dereverb-Echo_Mel_Band_Roformer` @9c83bf27: CC-BY-NC-SA-4.0, never usable.
- **facebook/sam-audio-large** @5f2cd3a9: licence `other` / `sam-license`, gated with manual approval. The terms were not readable without a login, so ask.

### C. Professional mixing practice and acoustic matching

**C1. The Netflix Mixing Style Guide for Dubbed Content. [confirmed]**

Source: partnerhelp.netflixstudios.com article 360017944573 (undated; fetched 2026-09-24).

- Mix the dub at levels similar to the original version and leave the M&E unaltered.
- Use more than one reverb at a time, and keep track of the treatments used for each location.
- EQ and compression should "match the quality of OV production dialogue (or better)". The guide does not ask for recording flaws to be copied.
- Efforts, breaths and screams should match the original and not be over-emphasised.
- Mix "as if it were recorded on-set", following perspective.

**C2. The Netflix Sound Mix Specification v1.6: -27 LKFS ±2 LU dialogue-gated, true peak -2 dBFS. [unverifiable]**

The old partnerhelp URL now redirects (301) to `studiopartner.netflix.net/studio/branded-sound-mix-spec-and-best-practices`, a JavaScript page that returned no text. With search exhausted, the numbers could not be re-read. Nothing in Maata depends on them.

**C3. iZotope Dialogue Match: EQ, reverb and ambience matching for replacement dialogue. [confirmed; one sub-claim unverifiable]**

Sources:
- izotope.com, "The Technology Behind the Reverb Module in Dialogue Match", 2019-11-05.
- docs.izotope.com Dialogue Match help, Ambience page.
- izotope.com, "How to Use Dialogue Match". Fetched 2026-09-24.

Details:
- **Reverb module:** a neural "Regression Net", trained on reverberant dialogue, sets Exponential Audio reverb parameters (reflection density, decay time, colour, size). In a MUSHRA test against an expert engineer, p=0.24.
- **Reference length:** the Reverb module wants a reference of at least three seconds, and the network was trained on dialogue of at least 3 s. That is a Reverb-module tip, not a product-wide rule; the how-to says the Capture clip can be "any length".
- **Ambience module:** it matches "the room tone and noise floor" and has a level for "synthesized Ambience processing". This confirms that the ambience is synthesised, not copied.
- **Status:** the product is no longer sold, and support runs "up through September 24, 2026".
- **Unverifiable:** "EQ matching comes from Ozone".

**C4. Telephony comfort noise is the precedent for synthetic room tone. [confirmed]**

Source: RFC 3389 (IETF, September 2002).

- Complete silence can feel unnatural and make a call seem dropped.
- The comfort-noise payload carries only a noise level (0 to -127 dBov) and optional reflection coefficients of an all-pole spectral model.

**C5. Blind room-acoustics estimation (Rec-RIR). [confirmed]**

Sources: Wang and Li, arXiv 2509.15628 (v1 2025-09-19, v3 2026-06-26; Interspeech 2026); GitHub `Audio-WestlakeU/Rec-RIR` (MIT, pushed 2026-08-10, `ckpt/epoch35.tar`).

- **Model:** 3.1M parameters, 35.2 GMAC/s, 16 kHz, RIRs up to 0.96 s.
- **Results on SimACE** (WSJ0 speech with measured ACE RIRs, 20 dB SNR):
  - RT60 MAE 0.069 s (ρ=0.994);
  - DRR MAE 0.684 dB;
  - C50 MAE 0.858 dB;
  - better than FiNS, BUDDy and both VINP variants.
- **Training data:** 200 h from DNS Challenge, VCTK and **EARS**. EARS's LICENSE file is "Attribution-NonCommercial 4.0 International", so the checkpoint needs a ruling or a retrain.
- **No-model alternative:** `pyroomacoustics.experimental.measure_rt60(h)` needs a known impulse response. A blind estimate must be DSP, as Amazon used (Löllmann), or a model.

**C6. Zero-shot TTS entangles the prompt's room with the voice. [confirmed (paper); the Chatterbox consequence is inferred]**

Source: Lu, Wang, Ai, Du, Ling and Yamagishi, DAIEN-TTS, arXiv 2608.03011 (2026-08-04; submitted to IEEE TASLP).

- The paper says current zero-shot TTS systems "either strip away or entangle the acoustic environment with speaker characteristics".
- It is built on F5-TTS, whose weights (`SWivid/F5-TTS`) are CC-BY-NC-4.0, so it is a research direction only.
- Inferred for Maata: ADR-017's S3Gen timbre span (the best clean 8–10 s span) colours the dub. Cleaning only the T3 embedding clips could help identity without drying the voice.

**C7. YouTube turns loud videos down and never turns quiet ones up; the -14 LUFS reference. [corrected]**

Source: Ian Shepherd, Production Advice, "YouTube Stats for Nerds", 2017-09-29.

- Confirmed:
  - YouTube does not turn quieter videos up.
  - "Content loudness" in Stats for nerds is the difference between YouTube's loudness estimate and its reference playback level.
- Correction: this source does *not* give -14 LUFS. It tells readers the point is not to aim for -14 LUFS.
- The -14 LUFS figure is **unverifiable** this session, so no Maata target should be hard-coded to it.

Kind: third-party measurement.

### D. Non-verbal carry-over

**D1. Chatterbox-Turbo and Nano have paralinguistic tags; Multilingual does not. [confirmed]**

Sources: resemble-ai/chatterbox README (MIT, repo pushed 2026-07-21); HF `ResembleAI/chatterbox-turbo` @749d1c1a (MIT, 2025-12-15) and its `added_tokens.json`; HF `ResembleAI/chatterbox-nano` @71ccd1d0 (MIT, 2026-07-21).

- **Models:** Turbo has 350M parameters and is English-only. Nano has 110M parameters, shares Turbo's architecture, is loaded with `nano=True`, and runs "3x faster than realtime on 8 CPU cores".
- **Tag tokens:**
  - Non-verbal: `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[cough]`, `[clear throat]`, `[sniff]`, `[crying]`, `[groan]`, `[shush]`, `[whispering]`.
  - Emotion and style: `[angry]`, `[happy]`, `[sarcastic]`, `[surprised]`, `[fear]`, `[dramatic]`, `[narration]`, `[advertisement]`.
- **Multilingual V3** is described without tags.
- **Unverified:** whether a Turbo clone matches the Multilingual clone of the same reference.

**D2. Taggers that locate non-verbal events. [confirmed]**

Sources: HF `mispeech/ced-base` @db3e14a8 and `ced-small` @06bb40c5 (Apache-2.0, 2026-03-30); `MIT/ast-finetuned-audioset-10-10-0.4593` @f826b80d (BSD-3-Clause); NVV-Locator, arXiv 2609.09940 (v1 2026-09-09, v2 2026-09-23).

- CED's 527 labels include Laughter, Chuckle, Giggle, Snicker, Belly laugh, Sigh, Breathing, Gasp, Cough, Throat clearing, Sniff and Crying. These map almost one-to-one onto Turbo's tags. The labels are clip-level.
- NVV-Locator covers 26 categories, with micro-F1 71.0% and a boundary MAE of 59.6 ms. Its weights and licence are not stated.

### E. Platform constraints (YouTube)

**E1. The YouTube API Services Developer Policies. [corrected: the sweep's clauses are confirmed, the ToS date was wrong, and key clauses were missed]**

Source: https://developers.google.com/youtube/terms/developer-policies ("Last updated 2026-09-14"; fetched 2026-09-24). Numbering was taken from the page's list structure.

Confirmed:
- **III.I.6:** do not "modify, build upon, or block any portion or functionality of a YouTube player".
- **III.I.7:** do not "separate, isolate, or modify the audio or video components" of YouTube content. Its example: "you must not apply alternate audio tracks to videos".
- **III.I.8:** do not promote audio or video components separately.
- **III.I.9:** no background player.
- **III.E.1.a:** do not download, cache or store YouTube audiovisual content without YouTube's prior written approval.
- The policies name "The YouTube IFrame Player API service" as a YouTube API service, so the embed is covered.

Missed by the sweep:
- **III.I.14:** do not "use any technology other than YouTube API Services to access or retrieve API Data, including to access any portion of any YouTube audiovisual content". This bears directly on the yt-dlp fetch.
- **III.E.6:** no scraping of YouTube Applications, and no obtaining scraped YouTube content.
- **III.I.1:** do not act as a substitute for YouTube Applications.

YouTube Terms of Service (fetched 2026-09-24): the US version in force is **effective 2023-12-15**, not 2022-01-05.
- It bars accessing or downloading content "except … as expressly authorized by the Service".
- It bars accessing the Service "using any automated means".

This is not legal advice.

**E2. The IFrame Player API volume and rate controls. [confirmed]**

Source: https://developers.google.com/youtube/iframe_api_reference ("Last updated 2026-09-15").

- `mute()`, `unMute()` and `isMuted()`.
- `setVolume()` takes an integer from 0 to 100, and `getVolume()` returns the volume even while muted.
- `setPlaybackRate(suggestedRate)` rounds unsupported values, and `onPlaybackRateChange` fires on changes.
- `onAutoplayBlocked` fires when the browser blocks scripted playback.

### F. Repo facts (read 2026-09-24, not modified) [confirmed]

- **Reference normalisation:** `engine/src/maata_engine/backends/torch_common.py` `_loudness_normalise(target_dbfs=-20.0)` trims silence, RMS-normalises and caps the peak at 0.95. It is applied to the S3Gen timbre span (cut to 10 s), the prompt and the embedding clips.
- **Locked audio dependencies:** `engine/uv.lock` locks pyloudnorm 0.2.0 (pulled in by chatterbox-tts), librosa 0.11.0, soundfile 0.14.0, scipy and numpy, and transformers 5.2.0.
- **pyloudnorm's scope:** it implements BS.1770-4 integrated loudness with standard gating. It has **no true-peak meter** (only sample `peak()` normalisation) and **no dialogue gating**. This was missed by the sweep and affects R2.
- **The Chatterbox fork:** the installed `chatterbox/tts_turbo.py` has `norm_loudness(target_lufs=-27)`.
- **Playback:** `app/src/lib/sync.ts` uses one `AudioContext` and one `GainNode`, and changes volume with `setTargetAtTime(…, 0.02)`.
- **Spec and ADRs:**
  - SPEC §6.3 says to evaluate Sidon or HTDemucs cleaning and keep it "only if it measurably improves speaker similarity".
  - SPEC §6.5 lists trim, short fades and consistent loudness.
  - Amendment 01 says the original audio is never played.
  - ADR-017 takes the timbre from the best single clean span of 8–10 s.
- **Web Audio:** MDN (last modified 2025-07-28) documents `ConvolverNode` convolution reverb as Baseline, widely available since July 2015. A reverb or room tone could therefore also run client-side.

---

## Options for background audio under "original audio never played"

**A. Dry dub (status quo).**
- Leaves digital silence between lines and dead air in music-only stretches.
- Each line brings its own noise floor, inherited from the S3Gen reference, which switches on and off (inferred).

**B. Synthetic room tone plus acoustic matching (no original samples played).**
- Recommended now.
- The evidence is indirect: C3 (synthesised ambience in ADR practice) and C4 (comfort noise). A1 does not test it.

**C. A separated music-and-effects bed (needs a rule change).**
- Gives the most realism, and A1 is direct evidence for it.
- Costs:
  - III.I.7 (E1);
  - English residue about 15 dB down (A5, inferred);
  - the original speakers' laughs in the effects stem (A6);
  - the bed must follow pause, seek and rate;
  - separation artefacts are disliked (A4).
- Hybrid C′: play the bed only where nobody speaks, ducked and gated.

**D. Unmute YouTube's own player in long stretches with no speech.**
- Uses the player through its own API (E2), so there is no separation and no downloaded audio is played.
- It still plays original audio, so it needs a rule change.
- Volume steps are coarse integers, latency is unknown, and there is a risk of missed speech.

**E. Voice-over with the original ducked underneath.** Not dubbing; not recommended.

**F. A generated (text-to-audio) background.** Not faithful to the original; licences were not re-checked; not recommended.

---

## Recommendations for Maata (updated for the corrections)

**R1. Fix pacing and translation first, then run realism experiments.**
- Impact: high. Effort: none (ordering only).
- Why: in A1, native listeners ranked the unrendered baseline best because of pacing.

**R2. Match loudness to the original.**
- Impact: high. Effort: S. Approval: a minor ADR declaring pyloudnorm (MIT, already locked) a direct dependency.
- What to do:
  - Measure each speaker's integrated loudness over their **own diarised speech segments**. This is our own dialogue gate, because pyloudnorm has none.
  - Set the dub to that level.
  - Add per-line gain of about ±3 dB that follows the source line's energy relative to the speaker's mean (A2).
  - Limit true peak to about -1 dBTP with a 4× oversampled peak check (`scipy.signal.resample_poly`), since pyloudnorm has no true-peak meter.
  - Do **not** hard-code -14 LUFS (C7 is unverified). Make the playback reference a setting, and follow YouTube's known behaviour: turn loud originals down, never quiet ones up.

**R3. Add a synthetic room-tone bed (option B).**
- Impact: medium to high. Effort: S. Approval: the maintainer must confirm that statistics derived from the original do not count as "playing original audio". No new dependency is needed.
- What to do:
  - For each scene, estimate a noise level and an LPC or third-octave spectral envelope from the original's non-speech gaps, using the existing diarisation and VAD.
  - Generate shaped noise from those statistics, never from original samples.
  - Play it on its own GainNode in `sync.ts`, about 20–35 dB below dialogue. It keeps running through pauses and seeks, and rate changes do not affect it.
  - Label it as unproven in listening tests, and test it in R8.

**R4. Match each speaker's sound to the original.**
- Impact: medium. Effort: M. Approval: none if done in numpy or scipy.
- What to do:
  - Match the long-term spectrum with EQ that is smoothed and capped at ±6 dB. Match the original's quality "or better" (C1), not its flaws.
  - Add light re-reverb only where the blind RT60 estimate is above about 0.3 s. Estimate RT60 and DRR with DSP (Löllmann-style), build a synthetic exponential-decay RIR in numpy, and convolve in the engine or with a Web Audio `ConvolverNode`.
- Avoid Rec-RIR's checkpoint (EARS is non-commercial) unless it is retrained or the maintainer rules on the data. pyroomacoustics can't do blind estimation, so it adds nothing here.

**R5. Run the reference-cleaning bake-off that SPEC §6.3 already asks for.**
- Impact: medium (the complaint is voice similarity: guest 0.466, host 0.645). Effort: M.
- Approval needed: new model pins and vendored MLX modules. mlx-audio can't be installed as-is, because it needs transformers>=5.14.
- Arms:
  - raw (the control);
  - AUSoundIsolation HighQualityVoice (macOS 15+, a system unit, run through a small Swift helper or pyobjc offline render);
  - Kim Mel-Band RoFormer (MIT, MLX);
  - MossFormer2_SE_48K (Apache-2.0, MLX);
  - DeepFilterNet3 (MIT/Apache-2.0, MLX).
- Apply each arm (a) to the T3 embedding clips only, and (b) also to the S3Gen timbre span.
- Adopt a cleaner only if WeSpeaker similarity rises, because B5 shows a drop from 0.35 to 0.28.
- Try longer and cleaner spans first: prompt length helped in B5.
- Excluded:
  - Sidon, for identity, because it re-synthesises.
  - DialogueSidon and Sucial, which are non-commercial.
  - The anvuew models (GPL-3.0) and SAM-Audio (custom licence), which need a ruling.
- Pick weights from a per-model allowlist, never from the mlx-audio-separator catalogue (B4).

**R6. Put the background decision (A–F) to the maintainer with the corrected evidence.**
- Impact: high. Effort: M. Approval: a product rule change plus legal review.
- A1 is the only controlled evidence, and it favours the original bed (C), not room tone. C, however, collides with III.I.7.
- If the rule is relaxed, prefer D (the player's own audio in speech-free stretches) over C.

**R7. Carry over non-verbal sounds by synthesis.**
- Impact: medium. Effort: M. Approval: model pins for CED and Chatterbox-Turbo or Nano, plus a similarity check between the Turbo and Multilingual clones.
- What to do:
  - Detect events with CED (Apache-2.0).
  - Have the Claude translation keep markers such as `(laughs)` in position.
  - Render the matching Turbo or Nano tag (`[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[sniff]`, …) from the same reference, placed at the original time and loudness-matched.

**R8. Tidy the Web Audio mix.**
- Impact: medium. Effort: S. Approval: none.
- What to do:
  - 5–10 ms fades on every unit.
  - 30–80 ms equal-power crossfades where units overlap.
  - All gain changes as ramps or `setTargetAtTime`.
  - The bed follows the player's state.
  - One master limiter (a `DynamicsCompressorNode` or a baked limiter).

**R9. Measure the result.**
- Impact: medium. Effort: M.
- Add to maata-bench:
  - loudness difference per speaker (LU);
  - long-term-spectrum distance;
  - RT60 difference;
  - WeSpeaker similarity against both the raw and the cleaned original.
- Then run a blind test with native Telugu listeners (dry vs room tone vs room tone plus matching) after the pacing fixes land. Only committed JSON from the M5 Pro counts.

**R10. Escalate the YouTube terms question as a product and legal item.**
- Impact: high. Effort: S. Owner: the maintainer and legal counsel.
- III.I.7 ("must not apply alternate audio tracks"), III.I.14 (no non-API technology to access audiovisual content), III.E.1.a (no downloading or caching) and III.E.6 (no scraping) apply to the core design of a muted embed with a yt-dlp-fed dub, not only to a background bed.

---

## Open questions

1. Which separators do ElevenLabs and YouTube use, and how bad is YouTube's reused background under controlled tests? Neither is disclosed, and A4's only evidence is biased.
2. What are the real-time factor and peak memory on the M5 Pro for Kim Mel-Band, MossFormer2_SE_48K, DeepFilterNet3 and AUSoundIsolation? Is there any MLX or Core ML path for BandIt Plus?
3. How does AUSoundIsolation HighQualityVoice compare with RoFormer models on podcast speech, and can the Python engine render it offline (pyobjc manual rendering, or a small signed Swift helper)?
4. Does cleaning the reference raise or lower Chatterbox-telugu similarity? The only evidence (F5-TTS, StyleTTS2) shows a drop.
5. Does room tone plus matching measurably help native Telugu listeners once pacing is fixed? No study has tested synthetic room tone in dubbing.
6. Licences still open:
   - BandIt Plus weights, which have no licence file;
   - data inheritance (DnR/FSD50K NC clips, MUSDB18 NC tracks, EARS NC) under ruling #8;
   - Kim Vocal 2's undisclosed training data;
   - the SAM License terms (gated, not read).
7. How accurate are blind RT60 and DRR estimates on real YouTube speech over music beds?
8. Does the maintainer count statistics derived from the original (noise profile, loudness, spectrum, RT60) as compatible with "original audio never played"?
9. Are IFrame `setVolume` ramps (integer steps, unknown latency) smooth enough for option D, and does the Tauri webview block unmuted scripted playback (`onAutoplayBlocked`)?
10. Is a Chatterbox-Turbo or Nano clone close enough to the Multilingual clone for spliced laughs to sound like the same person?
11. Three claims stayed unverifiable, because web search was exhausted: the Netflix Sound Mix Spec v1.6 numbers, YouTube's -14 LUFS reference, and "EQ from Ozone". Newer 2026 dialogue separators may also have been missed.

## Refuted

None. Every error found was partial and is listed under [corrected] above:
- A1: the interpretation of native listeners and what the study tested.
- B1: the DnR record and the NC inheritance through FSD50K.
- B4: "every separator family" has an MLX port.
- B5: 0.355 should be 0.35.
- C7: -14 LUFS was misattributed to Production Advice.
- E1: the ToS date and the missed clauses.

## Sources (all fetched or queried 2026-09-24 unless a date is given)

**Papers**
- arXiv 2001.06785, Amazon automatic dubbing (2020-01), https://arxiv.org/abs/2001.06785, full text read via ar5iv.
- arXiv 2212.12137, Dubbing in Practice (2022-12-23), https://arxiv.org/abs/2212.12137.
- arXiv 2308.06981, SDX23 CDX (2023-08), https://arxiv.org/abs/2308.06981.
- arXiv 2506.02499, DnR-nonverbal (2025-06-03), https://arxiv.org/abs/2506.02499.
- arXiv 2602.05770, WildSpoof prompt enhancement (2026-02-05), https://arxiv.org/abs/2602.05770.
- arXiv 2509.17052, Sidon (2025-09-21, v3 2026-01-26), https://arxiv.org/abs/2509.17052.
- arXiv 2608.03011, DAIEN-TTS (2026-08-04), https://arxiv.org/abs/2608.03011.
- arXiv 2509.15628, Rec-RIR (2025-09-19, v3 2026-06-26), https://arxiv.org/abs/2509.15628.
- arXiv 2609.09940, NVV-Locator (2026-09-09, v2 2026-09-23), https://arxiv.org/abs/2609.09940.

**Company docs and pages**
- ElevenLabs Dubbing docs and API (undated), https://elevenlabs.io/docs/overview/capabilities/dubbing and https://elevenlabs.io/docs/api-reference/dubbing/create.
- YouTube Help 15569972 (undated), https://support.google.com/youtube/answer/15569972.
- BeMultilingual (2026-01-23), https://www.bemultilingual.ca/blog/youtube-auto-dubbing.
- Netflix Mixing Style Guide for Dubbed Content (undated), https://partnerhelp.netflixstudios.com/hc/en-us/articles/360017944573.
- iZotope reverb-module article (2019-11-05), https://www.izotope.com/en/learn/the-technology-behind-the-reverb-module.html.
- iZotope Dialogue Match help, Ambience page, https://docs.izotope.com/dialogue-match/en/ambience/index.html.
- iZotope, How to Use Dialogue Match, https://www.izotope.com/en/learn/how-to-use-dialogue-match.
- WWDC25 session 251 (2025-06), https://developer.apple.com/videos/play/wwdc2025/251/.
- YouTube API Services Developer Policies (2026-09-14), https://developers.google.com/youtube/terms/developer-policies.
- YouTube IFrame Player API (2026-09-15), https://developers.google.com/youtube/iframe_api_reference.
- YouTube Terms of Service (effective 2023-12-15), https://www.youtube.com/static?template=terms.
- MDN ConvolverNode (2025-07-28), https://developer.mozilla.org/en-US/docs/Web/API/ConvolverNode.

**Standards and third-party measurements**
- RFC 3389 (2002-09), https://datatracker.ietf.org/doc/html/rfc3389.
- Production Advice (2017-09-29), https://productionadvice.co.uk/stats-for-nerds/.

**Datasets**
- Zenodo 5574713, DnR v1 (2021-10-17), https://zenodo.org/records/5574713.
- Zenodo 6949108, DnR v2, https://zenodo.org/records/6949108.
- Zenodo 4060432, FSD50K, https://zenodo.org/records/4060432.
- MUSDB18 licence page, https://sigsep.github.io/datasets/musdb.html.
- EARS licence file, https://github.com/facebookresearch/ears_dataset/blob/main/LICENSE.

**Hugging Face API**, with revisions as cited above:
- KimberleyJSN/melbandroformer
- mlx-community/mel-roformer-kim-vocal-2-mlx
- sarulab-speech/sidon-v0.1
- seanll95/sidon-coreml
- sarulab-speech/DialogueSidon
- mlx-community/DeepFilterNet-mlx
- starkdmi/MossFormer2_SE_48K_MLX
- alibabasglab/MossFormer2_SE_48K
- anvuew/dereverb_mel_band_roformer, anvuew/dereverb_bs_roformer and anvuew/dereverb_room
- Sucial/Dereverb-Echo_Mel_Band_Roformer
- facebook/sam-audio-large
- ResembleAI/chatterbox-turbo, ResembleAI/chatterbox-nano and ResembleAI/chatterbox
- mispeech/ced-base and mispeech/ced-small
- MIT/ast-finetuned-audioset-10-10-0.4593
- ResembleAI/resemble-enhance
- SWivid/F5-TTS

**GitHub API**:
- Blaizzy/mlx-audio
- ssmall256/mlx-audio-separator
- fdebkowski/mlx-audio-separator-gui
- ZFTurbo/Music-Source-Separation-Training
- facebookresearch/demucs and adefossez/demucs
- Rikorose/DeepFilterNet
- modelscope/ClearerVoice-Studio
- resemble-ai/resemble-enhance
- haoheliu/voicefixer
- yxlu-0102/MP-SENet
- Audio-WestlakeU/Rec-RIR
- facebookresearch/ears_dataset
- spotlightishere/QuietNow
- LCAV/pyroomacoustics
- csteinmetz1/pyloudnorm
- resemble-ai/chatterbox
- darius522/dnr-utils

**Other**
- PyPI: demucs 4.1.0 (2026-07-11).
- Local files: `MacOSX27.0.sdk` AudioToolbox headers, and the LazyDub repo files named in section F.
