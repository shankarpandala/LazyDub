# 02: How YouTube/Google, Meta, Microsoft and Amazon dub: verified report

Verified 2026-09-24. This is an adversarial check of sweep 02 (`02-youtube-google-meta.sweep.md`), run against the prior model sweep (`../sota_ranking.md`).

**Method:**
- Every claim that a recommendation depends on was re-checked against a primary source: live fetches of the help pages, blogs, docs and licence files, and full-text greps of the arXiv PDFs.
- Where the repo was involved, I read the code. Nothing was changed.
- The web-search budget ran out partway through, so a few secondary claims could not be re-found. They are marked [unverifiable].

**Verdict markers:**
- [confirmed]: the source says it.
- [corrected]: the fact is fixed, with its source.
- [unverifiable]: I could not confirm it from a primary source.
- Refuted claims are listed at the end.

**Kinds:**
- DISCLOSED: the company or paper states it.
- REPORTED: press or a third party states it.
- INFERRED: our reasoning.

## Summary

1. **Every system that discloses its design is a cascade**, and even Google's end-to-end translator is trained on data a cascade produced.
   - Aloud/YouTube, Google's dubbing patent, Azure video translation and Amazon's research pipelines all disclose the chain: speech-to-text, then translation (Azure adds an LLM rewrite), then TTS, then remixing.
   - Google's end-to-end Meet/Pixel translator is trained on data built that way.
   - **Meta does not disclose its architecture.** The sweep's claim that all four companies converge is overstated for Meta [corrected].
2. **YouTube dubs English into Telugu, but not with Expressive Speech** [confirmed].
   - The only Telugu pair with Expressive Speech is Telugu into English.
   - Meta (Reels, Telugu since 2026-01-16) and Azure (Telugu personal-voice target) ship voice-cloned Telugu, but only in the cloud.
   - Google's new Gemini 3.8 TTS is cloud-only too, and its voice replication is **not offered in India** (new finding).
3. **Timing: keep the speaking rate steady and fit time by changing the words.**
   - Amazon's measured evidence points this way (Virkar 2022, Brannon 2023, Microsoft LSST 2025).
   - Correction: professional dubbers **do** give up lip sync, and often exact timing. What they refuse to give up is a steady speaking rate and translation quality [corrected].
4. **Room rendering: its measured gain is narrower than the sweep implied** [corrected].
   - In Amazon 2020 (IWSLT 2020, not ICASSP), adding the background and reverb gave +10.34 MUSHRA only for listeners who don't speak Italian.
   - Native Italian listeners gained +1.05, which is not significant.
   - It is still worth doing, but it is not proven to be the biggest lever for native Telugu viewers.
5. **Prosody transfer per line** is the best-evidenced way to carry expressiveness over [confirmed].
   - Amazon's phrase-level prosody transfer scored +6.2% MUSHRA over a single global reference.
   - AudioPaLM conditions on a 3 s sample taken from the input utterance itself.
   - Maata reuses one prompt clip per speaker. The Chatterbox README warns that a reference clip in a different language can pass on its accent, and suggests `cfg_weight=0` to reduce that (new finding). This is the main risk to test.
6. **Repo check: the sweep's video-rate recommendation is already built** [refuted as a recommendation].
   - The app already reads `getAvailablePlaybackRates()` and sends it to the engine (`session.set_player_rates`).
   - ADR-017 has retired video slow-down as the normal path.

Counts (claim units below): 31 confirmed, 5 corrected, 1 refuted, 6 unverifiable.

## Findings

### YouTube / Google

**Y1 [confirmed] YouTube auto-dubbing covers English to Telugu, without Expressive Speech. DISCLOSED.**
- Source: YouTube Help, "Use automatic dubbing", https://support.google.com/youtube/answer/15569972 (live fetch 2026-09-24; the page shows no last-updated date).
- Targets for English sources: Arabic, Bengali, Dutch, French\*, German\*, Hebrew, Hindi\*, Indonesian\*, Italian\*, Japanese, Korean, Malayalam, Polish, Portuguese\*, Punjabi, Russian, Spanish\*, Tamil, Telugu, Ukrainian. The asterisk marks expressive speech.
- A Telugu source is dubbed into English\* only.
- The page says expressive speech replicates the original's pitch and intonation, and that it "won't always be used".
- Videos are ineligible if they are longer than 120 min, or if the speech is too fast, "which would result in an unlistenable, sped-up dub".
- Among the known issues it lists: "difficulty matching the voice used for dubbing to the original voice".
- The page says nothing about keeping background audio or music.
- What voice YouTube uses for non-expressive Telugu is not disclosed. Calling it a "plain synthetic voice" is INFERRED.

**Y2 [confirmed] On 2026-02-04 YouTube opened auto-dubbing to everyone. DISCLOSED.**
- Source: YouTube Blog, Chandralekha Motati (PM), https://blog.youtube/news-and-events/youtube-auto-dubbing-expressive-speech/ (2026-02-04).
- 27 languages.
- Expressive Speech in 8: EN, FR, DE, HI, ID, IT, PT, ES. The post says it helps capture the creator's original emotion and energy.
- More than 6M daily viewers watched at least 10 min of auto-dubbed content in December 2025.
- A smart filter skips music videos and silent vlogs.
- A lip-sync pilot is running.
- The blog post itself does not mention DeepMind or voice cloning.

**Y3 [confirmed] Slator quotes YouTube's dubbing lead on what Expressive Speech carries over. DISCLOSED quote, REPORTED statistic.**
- Source: Slator, https://slator.com/youtube-multilingual-expressive-speech-ai-dubbing/ (2026-02-06).
- Buddhika Kottahachchi, Head of Product for YouTube Dubbing, says the system carries pitch, intonation and energy from the original into the dub.
- It was built with Google DeepMind.
- Dubbed videos average about 75% of the original-language view duration.
- Slator frames the release as answering earlier criticism that the dubs sounded artificial. The sweep's quote of "robotic" appears in heise (Y12), not in this fetch.

**Y4 [confirmed, interested source] sync.so says YouTube uses generated voices and hasn't shipped lip sync. REPORTED.**
- Source: https://sync.so/blog/youtube-auto-dubbing (2026-07-02).
- It says YouTube uses generated voices, not the creator's cloned voice.
- It says lip sync "has not shipped" as of July 2026. YouTube Help lists lip sync as experimental early access for select channels.
- sync.so sells a competing lip-sync and voice-cloning product, so treat it as an interested party.

**Y5 [confirmed] Aloud (Google Area 120) was a cascade. DISCLOSED.**
- Source: https://blog.google/technology/area-120/aloud/ (2022-03-09).
- It combined "audio separation, machine translation and speech synthesis".
- Creators supplied subtitles or transcripts.
- It started with Spanish and Portuguese, with Hindi and Bahasa Indonesia "coming soon".
- Creators had to label dubs as synthetic.

**Y6 [unverifiable] "Aloud did not preserve the speaker's voice."** The post doesn't say either way.

**Y7 [confirmed] Google's dubbing patent. DISCLOSED.**
- Source: US 2020/0404386 A1, application 16/975,696, filed 2018-02-26, published 2020-12-24, assignee Google LLC, https://www.freepatentsonline.com/y2020/0404386.html (fetched 2026-09-24).
- Caption fragments are joined into sentences: consecutive fragments by punctuation, by short or no gaps, with speaker IDs carried along.
- Machine translation then runs on the **caption sentences** (claim 7).
- If a dubbed segment doesn't fit its time, the patent adjusts the playback speed either of the translated audio (claim 10) or of the video segment (claim 11).
- Mixing options:
  - lower the original's volume;
  - "digitally erase the voice ... while retaining other sounds such as music or background noises";
  - or replace the original audio.
- In a short gap between two dubbed lines (example: ½ s), it inserts silence rather than original audio.
- Each speaker ID can get a unique synthetic voice. Speaker IDs come from face tracking, visual speech classification and voice ID.

**Y8 [corrected] The patent is granted.** The sweep's open question said grant status was unknown.
- It issued as **US 11,582,527 B2 on 2023-02-14**: same title, same application 16/975,696, assignee Google LLC.
- Source: https://www.freepatentsonline.com/11582527.html (fetched 2026-09-24).

**Y9 [unverifiable] A continuation, US 12,114,048.** The fetch failed (connection reset).

**Y10 [confirmed] Google's real-time speech-to-speech translator. DISCLOSED.**
- Source: Google Research blog, Misiunas & Ablavatski (Google DeepMind), https://research.google/blog/real-time-speech-to-speech-translation/ (2025-11-19).
- Architecture:
  - built on AudioLM and SpectroStream;
  - 16 tokens per 100 ms;
  - the encoder sees the preceding 10 s;
  - about 2 s of delay;
  - int8/int4 quantisation.
- Languages: English to and from ES, DE, FR, IT and PT. Hindi is described as "promising".
- Training data comes from a pipeline of ASR, forced alignment, MT, validation, and a custom TTS "preserving the voice characteristics", followed by a final alignment. Samples that fail the delay limits are filtered out.
- Nuance the sweep missed: Meet runs it **on servers**. Only Pixel 10 runs it on the device.

**Y11 [confirmed] Google research models carry voice and prosody over by conditioning on the source speech. DISCLOSED.**
- AudioPaLM, arXiv 2306.12925 (2023-06-22): voice conditioning is "a 3-second long voice sample" taken from the original input speech, as audio tokens and SoundStream tokens. It is repeated if the input is shorter than 3 s.
- Translatotron 2, arXiv 2107.08661 (v5 2022-05-17, ICML 2022):
  - up to +15.5 BLEU over Translatotron 1;
  - keeps each speaker's voice across speaker turns without speaker segmentation;
  - is designed to "mitigate potential misuse of voice cloning".
- Translatotron 3, arXiv 2305.17547 (v2 2024-01-16):
  - unsupervised;
  - +18.14 BLEU over a cascade on synthesized Unpaired-Conversational data;
  - retains pauses, speaking rate and speaker identity.

**Y12 [confirmed] Viewers complained about YouTube's early auto-dubs. REPORTED.**
- Source: heise online, https://www.heise.de/en/news/YouTube-auto-dubbing-Robotic-translation-annoys-users-and-cannot-be-turned-off-10315551.html (dated 2025-03-14 in the sweep; my fetch did not show the date).
- Complaints: a "tin can" voice; dubs "sometimes accelerated to 1.5 times the original speed"; stiff sentences and mistranslations; no setting for viewers to turn dubbing off.

**Y13 [confirmed] The IFrame API treats setPlaybackRate as a suggestion. DISCLOSED.**
- Source: YouTube IFrame Player API reference, https://developers.google.com/youtube/iframe_api_reference (last updated 2026-09-15).
- An unsupported value is rounded "down to the nearest supported value in the direction of 1".
- The app must listen for `onPlaybackRateChange`.
- `getAvailablePlaybackRates()` returns the video's supported set, which always includes 1.
- The saved copy of the reference documents no method for choosing an audio track (new, see N4).

**Y14 [confirmed] Gemini 3.8 Flash TTS and Flash-Lite TTS are cloud-only. DISCLOSED.**
- Sources: Google blog, https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-text-to-speech/ (2026-09-23); MarkTechPost (2026-09-23).
- More than 100 languages and over 2,000 voices. Telugu is not named.
- Delivery is directed with stage directions and tags such as `<laughs>`, `<sigh>` and `<gasp>`, plus backchannels.
- Flash-Lite is aimed at "dubbing".
- Voice replication needs a 30 s sample (MarkTechPost) and a verbal consent recording that matches the reference speaker (official blog).
- Output carries SynthID and C2PA marks.
- API only, with no open weights.
- **New:** voice replication is "not available in Illinois, Texas, EEA, UK, Switzerland, and India".

**Y15 [unverifiable] "Made on YouTube 2026" (2026-09-23) announced real-time auto-dubbing for Live, with a pilot in early 2027.** The guessed URLs returned 404 and the search budget was exhausted. Not load-bearing.

**Y16 [unverifiable] YouTube lip sync covers 5 languages up to 1080p, using a custom stack that "modifies the pixels".** This is from the sweep's press summary and was not re-found.

### Meta

**M1 [confirmed] Meta AI Translations for Reels launched on 2025-08-19. DISCLOSED/REPORTED.**
- Source: TechCrunch (2025-08-19).
- English and Spanish.
- The voice uses the sound and tone of the creator's own voice.
- Lip sync is optional.
- At most two speakers, who must not talk over each other.
- Meta advises facing the camera and keeping background noise and music low.
- Facebook creators need at least 1,000 followers.
- Creators preview before posting, and viewers see a "Translated with Meta AI" label.

**M2 [confirmed] Hindi and Portuguese were added on 2025-10-09, translating in both directions.** Source: about.fb.com (2025-10-09).

**M3 [confirmed] Telugu was added.**
- Source: about.fb.com, "Instagram empowers creators to go global with local voice translations" (post dated 2026-01-16, announced 2025-11-28).
- Bengali, Tamil, Telugu, Kannada and Marathi were added, "rolling out to everyone today".
- The voice "preserves the sound and tone of your voice".

**M4 [unverifiable] Telugu dubs are English to and from Telugu.** The Telugu post doesn't list the direction pairs, so they could be Hindi-centred. The two-speaker limit comes from the English/Spanish launch and is not restated. The architecture is not disclosed anywhere.

**M5 [confirmed] SeamlessExpressive and SeamlessM4T v2 are excluded as non-commercial. DISCLOSED.**
- Source: HF model card facebook/seamless-expressive (fetched 2026-09-24).
  - Prosody UnitY2 carries phrase-level speech rate and pauses. PRETSSEL keeps utterance-level vocal style.
  - Languages: EN, FR, DE, IT, ZH, ES.
  - English to Spanish: ASR-BLEU 42.92, vocal-style similarity 0.274, AutoPCP 3.183, pause 0.508, rate 0.675.
  - The model is gated.
- Source: GitHub SEAMLESS_LICENSE (fetched 2026-09-24).
  - The licence grants use "solely for Noncommercial Research Uses".
  - SeamlessM4T v1/v2 and SeamlessStreaming are CC-BY-NC 4.0.
  - Only non-generative parts are MIT: the code, the w2v-BERT 2.0 encoder, and the aligner and toxicity tools.

### Microsoft

**MS1 [confirmed] Azure video translation is a cascade with an LLM rewrite. DISCLOSED.**
- Sources:
  - Learn overview (ms.date 2025-10-21, updated 2026-06-05).
  - How-to (ms.date 2025-12-19, API 2026-03-01).
- Core features: dialogue extraction and transcription, then MT plus "large language model (LLM) reformulation" (gender-aware), then TTS, then subtitles.
- Voices:
  - `PlatformVoice` picks the closest standard voice.
  - `PersonalVoice` gives "high-quality voice replication in a few seconds"; you must apply for access.
- Parameters: `speakerCount`, `subtitleMaxCharCountPerSegment` (30 if unsure), `enableLipSync`, `enableVideoSpeedAdjustment`, `adjustWebvttAlignment`.
- Editing: iterations use editable WebVTT/JSON with speakerId, gender, sourceLocaleText, translatedText and ssmlProperties.
- Overlapping speakers can produce segments whose voice is "unidentified".
- Limits: mp4 under 5 GB and shorter than 4 hours.

**MS2 [confirmed] Azure supports Telugu as a personal-voice target.**
- Source: Learn language-support page, video translation table (ms.date 2026-09-09, fetched 2026-09-24).
- For te-IN, all five columns are Yes: standard voice source and target, personal voice source and target, custom-lexicon phoneme target.

**MS3 [unverifiable] "Azure keeps the background under the dub."** The overview says only that the "original speech [is] replaced with the translated speech" after dialogue extraction. Keeping the background is a reasonable INFERENCE, not a disclosure.

**MS4 [confirmed] Microsoft's length-aware speech translation. MEASURED.**
- Source: arXiv 2506.00740, Chadha, Subramanian et al., Microsoft (2025-05-31).
- A phoneme-based model generates short, normal and long candidates, "tailored for real-time, on-device video dubbing".
- Length-aware beam search makes all three in a single pass, adding about 4.3% latency.
- The chosen speaker's duration model picks the candidate.
- Speech-rate compliance (within 20%): ES 49.06 to 57.04 (+16% relative); KO 62.02 to 74.34 (+20%).
- BLEU: ES 23.66 to 23.23; KO 22.57 to 23.41.
- Synchronisation MOS: +0.34 ES, +0.65 KO.

**MS5 [corrected] Microsoft's affiliation on VideoDubber is DISCLOSED, not inferred.**
- Source: arXiv 2211.16934 (2022-11-30), AAAI 2023.
- Authors are listed from Microsoft Research Asia, Microsoft Azure Speech and Microsoft Azure Translation. The first author is from Renmin University.
- It steers MT decoding by predicted speech duration.

**MS6 [confirmed] Azure Personal Voice DragonV2.1Neural. DISCLOSED via press.**
- Source: The Register (2025-07-31).
- Needs "just a few seconds" of speech, and works in 100+ languages.
- Microsoft claims "more realistic and stable prosody" and better pronunciation.
- It requires the speaker's consent and disclosure of synthetic content, and adds a watermark.
- Cloud only.

### Amazon

**A1 [corrected] Federico et al., "From Speech-to-Speech Translation to Automatic Dubbing". MEASURED.**
- Source: arXiv 2001.06785 (v3 2020-02-02).
- **Venue correction:** the sweep called it ICASSP 2020. Later Amazon papers cite it as **IWSLT 2020**.
- Pipeline:
  - length-token MT (t1=0.95, t2=1.05; the shortest type gives an average length ratio of 0.97);
  - prosodic alignment;
  - mel spline interpolation, which gave "better quality than traditional time-stretching";
  - U-Net foreground/background separation;
  - blind reverberation-time estimation plus a synthetic room impulse response.
- Evaluation: 657 ratings from 14 listeners (5 Italian, 9 non-Italian) on 24 clips.
- Results:
  - Adding MT length control and alignment: Italians −10.93 (significant). The speaking rate was "too slow, too fast, or too uneven".
  - Adding rendering: **non-Italian** listeners +10.34 (significant); **Italian** listeners +1.05 (not significant).
- **Correction to how the sweep used it:** the "biggest naturalness gain" was measured on listeners who don't understand the target language. It is not established for native listeners.

**A2 [confirmed] Virkar et al., "Prosodic Alignment for Off-screen Automatic Dubbing". MEASURED.**
- Source: arXiv 2204.02530 (2022-04-06), Amazon AI.
- For off-screen sentences, the aligner may use "the entire available inter-phrase and inter-sentence intervals".
- Rate score: 1 if r≤1, 2−r if 1<r≤2, 0 if r>2.
- Human evaluation: 20 native speakers gave 600 scores. Clips came from 20 TED talks and 3 YouTube vlogs, English to FR/IT/DE/ES.
- On/off-screen alignment beat isochrone alignment on win rate: +107.7–139.1% (MuST-C) and +110–297.1% (YouTube), p<0.01.
- A mixed-effects model found smoothness "the most impactful metric".
- Caveat the sweep missed: the human evaluation used **post-edited** translations, so the gains isolate timing.

**A3 [corrected] Brannon, Virkar, Thompson, "Dubbing in Practice" (TACL). MEASURED.**
- Source: arXiv 2212.12137 (2022-12-23), MIT Media Lab and AWS AI Labs.
- Corpus: 319.57 h from 54 professionally dubbed titles, English to DE and ES.
- Duration ratio correlates with word-length ratio at r=0.523, but with the dub's speaking rate at only r=0.163.
- Character-length ratio correlates with time overlap at only r=0.279.
- The paper finds substantial influence of the source audio beyond the words (emphasis, emotion).
- **Correction:** the sweep said dubbers "rarely sacrifice translation quality or lip sync". The paper says they show "less respect for isochrony and especially lip sync" than assumed, while being "surprisingly unwilling to vary speaking rates or sacrifice translation quality".
- Off-screen lines have a mean overlap fraction of 0.662, against 0.684 on-screen. Human dubs let timing drift.

**A4 [confirmed] Effendi et al., "Duration modeling of neural TTS for automatic dubbing". MEASURED.**
- Source: Amazon AI, ICASSP 2022 (amazon.science).
- Non-isoelastic (Gaussian per-phoneme) duration scaling beat uniform scaling on 50 sentences. Win rates:

| Pair | Speed | Uneven | Uniform |
|---|---|---|---|
| EN→IT | slow | 34.2 | 21.2 |
| EN→IT | fast | 40.1 | 14.7 |
| EN→ES | slow | 37.9 | 29.4 |
| EN→ES | fast | 44.9 | 24.6 |

- All at p<0.01.
- The duration model also made prosodic-alignment training about 100× faster.
- It works inside the TTS, which has explicit phoneme durations. Maata's post-hoc warping has none.

**A5 [confirmed] Chronopoulou et al., "Jointly Optimizing Translations and Speech Timing". MEASURED.**
- Source: arXiv 2302.12979 (2023-02-25), AWS AI Labs and LMU Munich.
- "isometric MT (in isolation, without prosodic alignment) is no more isometric than standard MT when both are fed into the same TTS model".
- The joint model gains 55% relative speech overlap. It costs 2.7% BLEU against the Txt2Phn baseline and 9% against standard MT.

**A6 [confirmed] Swiatkowski et al., phrase-level cross-lingual prosody transfer. MEASURED.**
- Source: arXiv 2306.11662 (2023-06-21 v2), Amazon Science.
- A prosodic phrase is "a continuous segment of speech separated by silence regions".
- It scored +6.2% MUSHRA over utterance-level global transfer and closed 23.2% of the gap to expressive human dubbing, with no loss of intelligibility.
- Companion paper arXiv 2306.11658: a noise module separates noise from prosody and closes 11.2% of the gap.

**A7 [confirmed] Goncalves et al., AVS2S (arXiv 2412.16530, Amazon).**
- LSE-D 10.67, a 9.2% reduction.
- It keeps the original video and deliberately does not mimic the speaker's voice.
- Not applicable to Maata.

**A8 [confirmed] Prime Video's AI-aided dubbing pilot. DISCLOSED.**
- Source: https://www.aboutamazon.com/news/entertainment/prime-video-ai-dubbing-english-spanish (2025-03-05).
- 12 licensed titles, English and Latin American Spanish, only titles with no existing dub.
- A "hybrid approach" in which localization professionals do QC.
- Quote from Raf Soltanovich (VP Technology).

**A9 [confirmed] Amazon removed "AI beta" English anime dubs after a backlash. REPORTED.**
- Source: Engadget (2025-12-03).
- Titles: Banana Fish, No Game No Life, Vinland Saga.
- Critics called them "completely devoid of any emotion or convincing intonation in dramatic moments". NAVA called them "AI slop".

**A10 [confirmed] Prime Video lip sync. REPORTED.**
- Source: Unite.AI (2026-09-09).
- It debuted on Maxton Hall S1–2, for **human-performed** dubs, using "AI and VFX technologies".

### Maata repo (read-only, 2026-09-24)

**R1 [confirmed] Voice preparation.**
- `engine/src/maata_engine/backends/torch_common.py:156` `prepare_voice_parts` (ADR-017):
  - an S3Gen timbre clip of at most 10 s;
  - a speaker embedding averaged over the clips;
  - a 6 s T3 prompt, which its docstring says carries pace, prosody and accent.
- `session.py:314` passes the **same timbre span as the prompt**, with `cfg_weight` 0.5.
- `exaggeration` is one global value (default 0.5).

**R2 [confirmed] Timing planner.**
- `timing/planner.py` is a rolling horizon with a one-line lookahead.
- Parameters: `speed_cap` 1.2, `lead_max` 0.3, `max_lag` 0.6, `w_smooth` 4.0, freeze budget 1 s per 60 s. It cites Brannon 2023.
- The video plays at 1.0×. ADR-017 retired the ADR-016 cascade of speed-up, video slow-down and freeze as the normal path.

**R3 [confirmed] Translation candidates.**
- `text/tenglish.py` already defines three tiers (normal, concise, very_concise), sends the last two lines as context, and marks the next line as context only.
- `session.py:485` requests the tiers only after the first translation doesn't fit.
- `claude_cli.py` holds the `claude -p` backend.

### New findings (missed by the sweep)

- **N1 [confirmed] The Chatterbox README warns about cross-language references.**
  - Source: README of resemble-ai/chatterbox at master (fetched 2026-09-24).
  - If the reference clip's language differs from the language tag, the output "may inherit the accent of the reference clip's language". To mitigate this, set `cfg_weight` to 0.
  - Higher `exaggeration` "tends to speed up speech". A lower `cfg_weight` (about 0.3) slows the pacing.
  - This bears directly on per-line English prompts and on per-line exaggeration changes (Rec 2 and Rec 7). Every tag changes the duration that the planner must fit.
- **N2 [confirmed] Maata already probes the IFrame playback rates.**
  - `app/src/App.svelte` sends `getAvailablePlaybackRates()` to the engine.
  - `session.set_player_rates` keeps only the real rates below 1.0 as the allowed slow-downs.
  - This makes the sweep's recommendation 8 redundant (see Refuted).
- **N3 [confirmed] Gemini TTS voice replication is excluded in India** (Y14). Even setting aside the no-cloud rule, it could not serve Maata's maintainer or users there.
- **N4 [confirmed, from the saved copy] The IFrame API documents no audio-track selection.** A YouTube Telugu auto-dub can only serve as a listening reference on youtube.com itself, not inside Maata, and must never be downloaded.
- **N5 [confirmed] Patent granted** as US 11,582,527 B2 (Y8).

## Refuted

- **The premise of sweep recommendation 8** ("remove any assumption that setPlaybackRate(0.95) takes effect; probe getAvailablePlaybackRates at runtime"). The repo already probes the rates and restricts slow-downs to them (N2), and the planner no longer slows the video in the normal path (R2).

## Recommendations for Maata (updated for the corrections)

Ranked by expected effect on the maintainer's complaints. None needs a new dependency or model.

1. **Translate whole sentences, then split them back into units.** Impact high, effort M, no approval needed.
   - Merge consecutive units into full sentences: same speaker, sentence punctuation, gaps under about 0.5 s.
   - Send each sentence to Claude with unit boundaries marked, plus the previous and next sentences as context.
   - Claude returns the Telugu split at the same boundaries.
   - Why: Google's patent translates caption *sentences* formed from fragments (Y7, claim 7). This targets the "incomplete/skipped sentences" and "translation very bad" complaints; the engine currently translates 12–20-word units.
2. **Per-line prosody prompt, with a cross-language guard.** Impact high, effort M, no approval needed.
   - Build the T3 prompt tokens from the source audio of the line being dubbed (or a 3–6 s window around it).
   - Keep the S3Gen timbre clip and the averaged embedding from the clean per-speaker span.
   - Fall back to the global prompt for lines under about 1.5 s, or lines that are overlapped or noisy.
   - Sweep `cfg_weight` over {0, 0.3, 0.5} together with the prompt choice, because Resemble says `cfg_weight=0` reduces accent carried over from a reference in another language (N1).
   - Measure WeSpeaker similarity (against the current guest 0.466 and host 0.645), an intelligibility ratio and blind listening.
   - Why: phrase-level prosody transfer +6.2% MUSHRA (A6); AudioPaLM conditions on 3 s of the input utterance itself (Y11); YouTube's Expressive Speech carries pitch, intonation and energy (Y3); human dubs carry emotion beyond the words (A3).
3. **Plan timing over a window.** Impact high, effort M, no approval needed.
   - Widen the planner from a one-line lookahead to a window of about 8 lines or 30 s. This is feasible because the engine works 5–10 minutes ahead.
   - Let slack flow across all gaps, and minimise rate changes between neighbouring lines under the 1.2× cap.
   - Why: every Maata line is effectively off-screen. Off-screen alignment beat isochrone alignment by 107–297% in win rate, and smoothness was the strongest predictor of human scores (A2). Human dubbers let timing drift rather than change their rate (A3). This targets "pace not in sync" and "not smooth".
4. **Always ask for the three length tiers in the first Claude call.** Impact medium, effort S, no approval needed.
   - Pick a tier by predicted duration (aksharas × the speaker's measured rate) before any speed-up.
   - Never target character counts.
   - Why: Microsoft's one-pass tiers gave +16–20% relative rate compliance with BLEU flat (MS4). Character control alone didn't help timing (A5, A3). Today the tiers cost a second call on misfit (R3).
   - Cost: more subscription tokens per line.
5. **Room-matched rendering. Downgraded from "biggest lever".** Impact medium (unproven for native listeners), effort M.
   - Add blind reverberation-time estimation, a matched synthetic impulse response on the dub, and a low room-tone bed. Test in a blind A/B with Telugu-speaking listeners.
   - Why: +10.34 MUSHRA, but only for listeners who don't speak the target language; for natives +1.05, not significant (A1). Google's patent and Aloud keep the background (Y5, Y7).
   - Approval: reverb and room tone need none. Playing a separated background stem changes the "original audio is never played" rule, needs a source-separation model with an acceptable licence, and needs a check against YouTube's terms. Both are the maintainer's call.
6. **Uneven time compression.** Impact medium, effort M, no approval needed.
   - Before uniform mel interpolation, shorten silences inside the line and steady vowel frames first.
   - Why: Amazon's uneven scaling beat uniform scaling, 40.1 vs 14.7 and 44.9 vs 24.6 when speeding up (A4). That result came from a TTS with explicit phoneme durations, so check by listening.
7. **Per-line delivery tags.** Impact medium, effort S, no approval needed.
   - Claude labels each line's emotion, energy and pace. Map the labels to bounded per-line `exaggeration` and `cfg_weight` values.
   - Re-predict duration after mapping, because `exaggeration` changes the pace (N1).
   - Why: flat emotion is the public failure of AI dubs (A9, Y12), and Gemini TTS is directed line by line (Y14). This is INFERENCE, so run a blind A/B.
8. **Automatic dubbing metrics in maata-bench.** Impact medium, effort S, no approval needed.
   - Add smoothness, intelligibility ratio and speech overlap. Amazon found these predict human preference (A2).
   - Where a test video has YouTube's own English-to-Telugu dub, listen to it on youtube.com as the non-expressive baseline to beat (Y1, N4). Never download it.
9. **Record exclusions in the ADRs.**
   - Meta SeamlessExpressive, SeamlessM4T v1/v2 and SeamlessStreaming: non-commercial licences (M5).
   - Gemini 3.8 TTS, Azure Personal Voice and Meta Reels: cloud only; Gemini voice replication is also unavailable in India (Y14, MS1, MS6, M1).
- **Dropped:** sweep recommendation 8 (IFrame rate probing), because it is already implemented (N2).

## Open questions

1. Does YouTube's Expressive Speech clone the creator's timbre, or put the source prosody onto stock voices? YouTube hasn't said. sync.so, a competitor, says stock voices.
2. Which TTS does YouTube use for non-expressive English-to-Telugu dubs? Which direction pairs and architecture does Meta use for Telugu Reels? Neither is disclosed (M4).
3. Is US 12,114,048 a continuation of US 11,582,527? This is unverified.
4. Did Made on YouTube 2026 announce live auto-dubbing? This is unverified (Y15).
5. Does a per-line English T3 prompt lower WeSpeaker similarity or add English accent to the Telugu? At which `cfg_weight`? This needs a Maata A/B (Rec 2, N1).
6. Does Amazon's gain from uneven scaling carry over to post-hoc mel warping of Chatterbox output? (A4)
7. Does room rendering help **native** Telugu listeners, once timing disfluencies are fixed? Amazon's native-listener result was not significant (A1).
8. Would a separated background stem violate the "original audio is never played" rule as the maintainer means it, or YouTube's terms? This is a maintainer or legal decision.
9. Are Azure's Telugu personal voice and Meta's Telugu Reels good enough to serve as quality references? No Telugu evaluation of either is published.

## Verification tally

| Verdict | Count | IDs |
|---|---|---|
| confirmed | 31 | Y1, Y2, Y3, Y4, Y5, Y7, Y10, Y11a (AudioPaLM), Y11b (Translatotron 2), Y11c (Translatotron 3), Y12, Y13, Y14, M1, M2, M3, M5, MS1, MS2, MS4, MS6, A2, A4, A5, A6, A7, A8, A9, A10, R1, R2/R3 (repo) |
| corrected | 5 | Y8 (patent granted), A1 (venue and listener group), A3 (lip sync is sacrificed), MS5 (affiliation disclosed), cross-company convergence ("all four", but Meta's design is undisclosed) |
| refuted | 1 | recommendation 8 premise (rates already probed, N2) |
| unverifiable | 6 | Y6, Y9, Y15, Y16, M4, MS3 |
