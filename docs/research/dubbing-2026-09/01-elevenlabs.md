# 01: How ElevenLabs dubs, and what Maata can take from it (verified 2026-09-24)

**Scope.** This is an adversarial check of the researcher's sweep (`01-elevenlabs.sweep.md`, F1–F20, R1–R11) and of the report built from it.

**How each claim was checked:**
- Against live primary sources fetched today, 2026-09-24:
  - ElevenLabs docs as `.md`, saved in `dub_research/el_verify/`;
  - blog, landing and insights pages;
  - changelog entries;
  - the arXiv HTML of the PSP paper.
- Against WebFetch reads of third-party pages: Turing Post, LinkedIn, gregpreece.com, The Elec and Sarvam.
- Against the Maata repo: `engine/src/maata_engine/segment.py`, `session.py`, `backends/torch_common.py`, the pinned Chatterbox package, `app/src/lib/voice.ts` and `docs/DECISIONS.md` (ADR-017, ADR-018).

**Search limits.** The session's WebSearch budget was already used up, so every check went through direct fetches. A patent search wasn't possible: Google Patents returned 503 and Justia showed a bot wall, which I did not bypass.

**Verdict markers:**
- [confirmed]: the primary source says it.
- [corrected]: partly wrong; the correct fact and its source are given.
- [unverifiable]: no primary source could settle it.
- Refuted claims are listed at the end.

**Disclosed vs inferred.** "Disclosed" means ElevenLabs states it. "Inferred" means it is our reading of what they state.

---

## Summary

**What ElevenLabs runs:**
- It runs two dubbing models:
  - **Dubbing v2**: Alpha, the default, fully automatic, no in-app editor. The UI launched 2026-05-28 and the API 2026-08-10.
  - **Legacy v1**: the only route to Dubbing Studio, which is in maintenance mode.
- v2 is marketed as an "audio-to-audio model" with "sync-aware translation".

**v2's own API docs show a hybrid, not end-to-end speech-to-speech:**
- The source is transcribed once into segments: id, text, speaker, start, end.
- Every language is translated from that master transcript as plain text per segment.
- The generator then voices each translation inside the source segment's own time span. The docs say translations "are rendered to fit their segment's time span". The sweep missed this line.

**What the generator is.** It conditions on the original performance: tone, pacing, emotion and voice. ElevenLabs has not disclosed whether it is speech-to-speech or TTS with a prosody reference. That part stays unverifiable.

**Four corrections that matter for Maata:**
1. **CEO interview (Turing Post, 2025-04-12).** The "speech-to-text, LLM, text-to-speech" description and the line that the three-step setup is still recommended are about real-time translation and voice agents, not dubbing. The dubbing parts hold:
   - Spanish runs 30–40% longer than English when spoken.
   - The Lex Fridman dubs took 1–2 weeks.
   - Translation was the hardest part, with QA by external partners.
2. **PSP paper (arXiv 2604.25476).** The "Telugu reference" clips that improved a Chatterbox LoRA's Telugu prosody were synthetic outputs of Sarvam Bulbul and Cartesia Sonic-3, not real Telugu speakers. A Telugu reference also swaps in a different voice. For Maata, whose speakers speak English, this trade-off costs identity.
3. **Speed setting.** The 0.7–1.2x range is ElevenLabs' general text-to-speech speed setting. It is not a documented dubbing limit. How far Dubbing v2 speeds speech up is not disclosed.
4. **Cloning audio.** The 1–2 minutes of audio for Instant Voice Cloning is ElevenLabs' guidance for its standalone cloning feature. How much audio automatic dubbing clones from is not disclosed.
   - Chatterbox cannot use a long reference anyway: S3Gen truncates it to 10 s and T3's prompt to 6 s.
   - Only the speaker embedding can pool longer audio, and Maata already pools up to 60 s.

**New findings from the repo:**
- **The voice-match setting does nothing.** Maata already ships an ElevenLabs-style voice-match setting ("Closest to original", "Balanced", "Most natural Telugu" in `app/src/lib/voice.ts`). The engine stores it and never reads it (`session.py:114` is its only use). So "add a similarity setting" (report R5) is refuted; the real task is to make the existing one work.
- **Segmentation is by time, not word count.** `segment.py` splits on seconds, pauses and sentence ends: `max_len` 12 s, `long_pause` 1.5 s, and re-sync anchors at 0.7 s pauses. The "12–20 words" description and the report's "never split on word count" advice are out of date.

**Counts:** 21 confirmed, 4 corrected, 3 refuted, 9 unverifiable.

---

## Findings

### A. Product surface and API (disclosed)

**F1 [confirmed]: two models, limits and outputs.**
- **Models:** Dubbing v2 Alpha is the default. Dubbing Studio is v1-only and in maintenance mode: critical bug fixes only, and "no new feature work is planned".
- **Size limits:**
  - App: 1 GB and 180 min.
  - API: 3 GiB.
  - Studio: 1 GB and 45 min.
  - One help-centre FAQ gives 2 GB and 180 min for the v2 website; the docs disagree, see U6.
- **Speakers:** up to 32 unique speakers per file. Speakers are detected "even with overlapping speech".
- **Concurrency:** 3 concurrent jobs on self-serve plans, 10 on Enterprise, counted per model.
- **Outputs:**
  - v2 returns "a single lossless audio file" (FLAC via `outputs.lossless_audio`).
  - Studio exports per-speaker WAV stems, AAF, SRT and CSV.
- Realtime or live dubbing is "not currently available".
- The v2 app also accepts YouTube and TikTok URLs.
- Sources (all fetched 2026-09-24, undated): elevenlabs.io/docs/overview/capabilities/dubbing; /docs/eleven-creative/products/dubbing/dubbing-studio; /docs/eleven-creative/products/dubbing (FAQ); /docs/help-center/product/dubbing/what-is-dubbing.

**F2 [confirmed]: v2's only voice setting is `cloning_strength`.**
- 0–10, default 7, shown as "Speaker similarity" in the app.
- ElevenLabs states the trade-off itself:
  - Higher values sound more like the speaker but can sound less natural in phonetically distant languages.
  - They "can also carry over more of the original accent".
  - Lower values give more natural target-language delivery.
- It is set per language target in `voice_settings`, "applied to every speaker in this language", not per speaker.
- Sources: /docs/overview/capabilities/dubbing; /docs/api-reference/dubbing/language-targets/create-language-target (fetched 2026-09-24).

**F3 [confirmed, caveat added]: the v2 API is project-based.**
- `POST /v1/dubbing/project` takes:
  - `file` or `source_url`;
  - `source_language` (auto-detected if omitted);
  - `model_id` (`dubbing_v1` or `dubbing_v2`, default v2, fixed at creation);
  - `keyterms`: up to 1,000 terms of at most 50 characters and 5 words each, to "bias transcription and translation";
  - `webhook_ids` (at most 3), `target_language`;
  - `transcript`: JSON segments.
- `POST .../language` takes `target_language`, `voice_settings.cloning_strength` and `translations`, which skip machine translation.
- **Revisions:**
  - The project `revision` bumps on every source edit.
  - Each language has `revision` and `output_revision`, plus a `stale` status.
  - `regenerate` re-synthesizes "only the edited regions".
  - Each target gets a free regeneration allowance equal to the source's duration.
- A `voices_not_permitted` warning means a replacement voice was used for that speaker.
- The v1-only settings are gone from v2: `num_speakers`, `drop_background_audio`, `watermark`, `start_time`/`end_time`, `disable_voice_cloning`, `target_accent`.
- **Caveat the report under-stated:** supplying a transcript or translations, editing transcripts and regenerating are all **Enterprise-only** on v2. Plain create-and-download works on every plan.
- Sources: /docs/api-reference/dubbing/create-project (example timestamp 2026-07-03); /docs/api-reference/dubbing/target-transcript/regenerate-target; /docs/eleven-api/guides/how-to/dubbing/refine-and-regenerate; changelog 2026-08-10 (all fetched 2026-09-24).

**F4 [confirmed, missed by the sweep]: the v2 data model is text per segment, voiced into the source span.**
- **Segment fields:**
  - Source segment: `id`, `text`, `speaker_id`, `start_s`, `end_s`, `external_id`.
  - Target segment: the same span plus `source_text` and `translation`.
  - No prosody, speed or emotion fields are exposed. Target segments keep the source's start and end times.
- **What the bring-your-own-transcript guide says:**
  - Translations "are rendered to fit their segment's time span".
  - A translation much longer or shorter than the original changes the pacing of the dubbed speech, so aim for comparable spoken length.
  - Segment timings and text are used directly to time and voice the dub.
- This is the most transferable disclosure: the generator absorbs length mismatch as a pace change, and the translator's job is to deliver comparable spoken length.
- Sources: /docs/api-reference/dubbing/source-transcript/get-source-transcript; /docs/api-reference/dubbing/target-transcript/get-target-transcript; /docs/eleven-api/guides/how-to/dubbing/bring-your-own-transcript (fetched 2026-09-24).

**F5 [confirmed]: the legacy v1 API shows a vocal stem and a background stem.**
- `drop_background_audio`: an advanced setting that "can improve dub quality" for "speeches or monologues".
- `num_speakers`: 0 auto-detects.
- `disable_voice_cloning`: uses a similar Voice Library voice instead.
- `target_accent` [Experimental]: steers voice choice and the translation's dialect.
- `mode=manual`: takes a CSV plus separate `foreground_audio_file` and `background_audio_file`; "production use is strongly discouraged".
- A 2025-05-12 endpoint returns the top 10 library voices similar to a speaker.
- The stem model is an **inference**. v2 supports it: its docs say "The original background audio is retained".
- Sources: /docs/api-reference/legacy/dubbing/create; changelog 2025-05-12 (fetched 2026-09-24).

**F6 [confirmed]: segmentation rule (bring-your-own transcript).**
- Segments run 0.1–25 s.
- One speaker's segments must not overlap; different speakers' segments may overlap for simultaneous speech.
- Advice: prefer shorter segments, and break sentences at any pause of one second or longer, because segment boundaries set the dub's timing.
- Source: /docs/eleven-api/guides/how-to/dubbing/bring-your-own-transcript (fetched 2026-09-24).

**F7 [confirmed]: Dubbing Studio (v1) timing and voice options.**
- **Timing:**
  - Regenerations default to "Fixed Generations": the clip keeps its duration, so speech can speed up or slow down "significantly".
  - "Dynamic Generations" fit the clip to its text but "could affect sync".
  - Stale audio can be regenerated in bulk.
  - Clips can be split and merged.
  - "Clip History" lets you pick the best of earlier takes.
- **Voices:**
  - Track clone (default): one clone from all of a speaker's clips. It may be "a bit more unstable" if the voice changes drastically.
  - Clip clone: built from one clip.
  - A Voice Library voice.
- Tip from the docs: clone a clip you like and assign it to the whole track.
- Sources: /docs/eleven-creative/products/dubbing/dubbing-studio; help-centre "track clone vs clip clone" (fetched 2026-09-24).

### B. Architecture: disclosed vs inferred

**F8 [confirmed; the hybrid reading is inference, now better supported].**
- **Disclosed marketing:**
  - The launch blog (Imogen Mulliner and Jakub Lichman, 2026-05-28, updated 2026-09-06) says v2 "conditions directly on the original performance". It also says its sync-aware translation "aligns starts, stops, and pacing automatically".
  - The landing page calls it "An Audio-to-Audio Model".
  - The Elec (Hyun-Seon Park, 2026-08-11) reports "ElevenLabs said" it is direct speech-to-speech and names 92 languages.
- **Disclosed mechanics:**
  - The API blog (Jack Limebear, 2026-08-06, updated 2026-09-22) says "The source is transcribed once and every language translates from it". It calls the source transcript "the single source of truth".
  - The model "translates with timing in mind", with no video manipulation or lip sync.
  - The same blog self-estimates "80-90% of human quality" (method not given; see U3).
- **Inference:** a cascade-shaped hybrid:
  1. ASR and diarization into segments;
  2. LLM or MT translation per segment, with duration awareness;
  3. a generator conditioned on the target text plus the source segment's audio, for voice and prosody, rendered to fit the source span (F4);
  4. a remix over the kept background.
- `cloning_strength` probably scales how strongly the source audio conditions the generator. The accent leakage at high values fits that reading. The generator type is still unknown (U4).
- Sources: elevenlabs.io/blog/introducing-dubbing-v2; elevenlabs.io/blog/ai-dubbing-api; elevenlabs.io/dubbing-studio (landing, fetched 2026-09-24); thelec.net idxno=12934.

**F9 [confirmed]: ElevenLabs' five-step description.**
1. Transcription and analysis: speaker identity, timing, vocal characteristics, emotion.
2. Translation that preserves intent, emotion, style and meaning, and swaps idioms.
3. Voice cloning ("In some workflows" an existing target-language voice is used instead).
4. Speech generation.
5. Synchronization: timing and speed are "slightly modified" where necessary.
- Source: elevenlabs.io/insights/what-is-ai-dubbing (Jack Limebear; datePublished 2026-07-27 in the page metadata).

**F10 [corrected]: CEO interview (Turing Post, "Inference with Mati Staniszewski", 2025-04-12).**
- **Confirmed for dubbing:**
  - English to Spanish runs 30–40% longer when spoken, which is hard to compress into the same time.
  - The Lex Fridman dubs took "anywhere from a week to two weeks".
  - Translation was the hardest part. ElevenLabs worked with another team, did "QA and QC" on translation, and had external partners review it.
- **Correction:**
  - The "speech-to-text, then LLM, then text-to-speech" walkthrough answers a question about real-time translation (a "Babel fish" device) and conversational AI.
  - The recommendation to keep the three-step setup because it is more stable and easier to monitor is about voice agents.
  - Neither statement describes ElevenLabs' dubbing pipeline. The report's "keeps the cascade for enterprise dubbing for control and safety" is refuted (see Refuted).
- Source: https://www.turingpost.com/p/mati (2025-04-12; read three times via WebFetch on 2026-09-24).

**F11 [confirmed]: changelog components.**
- 2025-02-25: automatic trimming of overlapping clips "to ensure clean audio tracks for each speaker".
- 2025-03-03: "Dubbing Studio now uses Scribe by default for speech recognition".
- 2025-04-14: the render endpoint "automatically handles missing transcriptions or translations".
- 2025-09-01: `should_normalize_volume` added to render.
- 2025-11-21:
  - JSON transcripts with word- and character-level timing;
  - a `DubbingModel` enum with `dubbing_v2` and `dubbing_v3`.
- 2026-04-20: v1 resource-editing endpoints deprecated.
- 2026-08-10: Dubbing v2 API.
- 2026-08-24: SDK v2.65.0 adds "realtime Dubbing message types". No product is announced.
- Sources: elevenlabs.io/docs/changelog/{2025/2/25, 2025/3/3, 2025/4/14, 2025/9/1, 2025/11/21, 2026/4/20, 2026/8/10, 2026/8/24} (re-fetched 2026-09-24).

**F12 [confirmed]: history.**
- **v1 launch (blog 2023-10-10):**
  - It transfers "emotions and intonation from original audio" to the dub.
  - It uses an in-house method to detect different speakers and remove background noise.
  - It covers 20+ languages.
- **Dubbing Studio (X post 2024-01-23, read via the api.fxtwitter.com mirror):**
  - It detects and labels speakers and creates an editable script.
  - Users can update translations, change timing and regenerate.
  - The post cites 29 languages.
- Sources: elevenlabs.io/blog/elevenlabs-launches-voice-translation-tool-to-break-down-language-barriers-for-content; x.com/elevenlabsio/status/1749863738570690692.

**F13 [confirmed; the reuse is unverifiable]: standalone components.**
- **Scribe v2:**
  - 90+ languages;
  - keyterm prompting (up to 1,000);
  - "Speaker diarization, up to 32 speakers";
  - dynamic audio tagging;
  - word timestamps with `spacing` tokens.
- **Voice Isolator:** "Not specifically optimized for isolating vocals from music".
- **Forced Alignment:** the 29 Multilingual v2 languages. No Telugu, and no diarization.
- **Inference only (U1):** v2 reuses Scribe, based on the shared 32-speaker limit.
- Sources: /docs/overview/models; /docs/overview/capabilities/speech-to-text; /voice-isolator; /forced-alignment (fetched 2026-09-24).

### C. Voice cloning and TTS controls

**F14 [corrected]: cloning guidance.**
- **Confirmed:**
  - The help-centre tips give "Instant Voice Cloning: 1 - 2 minutes of good audio" and Professional Voice Cloning 30–180 minutes.
  - More than 2–3 minutes "will yield little improvement" and can hurt stability.
  - Prefer consistent tone and quality over runtime.
  - The Instant Voice Cloning (IVC) product page says to record at least 1 min and avoid more than 3.
  - IVC is "few-shot adaptation", with the sample used as a conditioning signal.
  - A clone from calm narration may differ when asked for strong emotion.
  - Pacing is "highly influenced by the audio used to create the voice", so use longer, continuous samples.
  - Professional cloning fine-tunes for about 3–6 h.
- **Corrections:**
  1. The warning that fragmented or overlapping turns make speaker separation unreliable is in the docs. But the harm they describe, separation artefacts becoming training data, is stated for Professional Voice Cloning, not IVC.
  2. This is guidance for ElevenLabs' standalone cloning. **How much audio automatic dubbing clones from is not disclosed** (U5).
  3. It does not transfer directly to Chatterbox. The pinned package truncates the S3Gen reference to 10 s (`DEC_COND_LEN = 10 * S3GEN_SR`) and T3's prompt tokens to 6 s (`ENC_COND_LEN`); see F24.
- **Confirmed Maata facts:** the guest's pace changed from 3.43 to 5.11 aksharas/s when the reference changed (ADR-017).
- Sources: /docs/help-center/product/voices/voice-cloning/are-there-any-tips-to-get-good-quality-cloned-voices; …/how-many-voice-samples…; /docs/eleven-creative/voices/voice-cloning/instant-voice-cloning; /professional-voice-cloning; /docs/eleven-api/concepts/voice-cloning; /docs/overview/capabilities/text-to-speech/best-practices (all fetched 2026-09-24).

**F15 [corrected]: Eleven v3 and TTS controls.**
- **Confirmed:**
  - Eleven v3 covers 70+ languages including Telugu (tel). Multilingual v2 (29) and Flash v2.5 (32) do not include Telugu.
  - v3 stability presets:
    - Creative ("prone to hallucinations");
    - Natural ("Closest to the original voice recording");
    - Robust.
  - No SSML breaks on v3; capitalization adds emphasis; IPA gives 80–90% consistency.
  - Audio tags include [whispers], [sarcastic], [strong X accent], [overlapping] and [interrupting].
  - The UI "Enhance" button runs an LLM with a published prompt that adds tags while "STRICTLY" preserving the text.
  - Normalizing numbers is advised in the LLM prompt or in code.
  - Professional clones are not fully optimized for v3.
- **Correction:** the 0.7–1.2 speed setting is the general TTS voice setting, available in Text to Speech, Studio and the Agents Platform. It is not a v3-specific or dubbing limit. Maata's 1.2x cap is its own choice, not ElevenLabs' dubbing limit.
- Sources: /docs/overview/models; /docs/overview/capabilities/text-to-speech/best-practices (fetched 2026-09-24).

### D. Telugu, quality evidence, human services

**F16 [confirmed]: Telugu support.**
- Telugu (`te`) is in the Dubbing v2 table with no dialect variants. Only en, es, fr, pt, ar-EG and zh-TW have dialects.
- `te` is also in the Dubbing v1 list, which "supports the same languages as the Eleven v3 model".
- ElevenLabs publishes no Telugu quality statement.
- Source: /docs/overview/capabilities/dubbing (fetched 2026-09-24).

**F17 [corrected]: PSP benchmark (arXiv 2604.25476v1, 2026-04-28).**
- **Author:** single author, V. P. T. Menta of Praxel Ventures, who builds the competing Praxy Voice.
- **Limits:** 10-utterance pilots, and the author abstains from claims of significance.
- **Confirmed Telugu numbers:**
  - Retroflex collapse: ElevenLabs v3 40%; Sarvam and Parler 33%; Cartesia 50%.
  - FAD: Sarvam 250.4, Parler 325.0, ElevenLabs 328.9, Cartesia 458.1.
  - PSD: ElevenLabs 154.4, against Sarvam 11.1 and Parler 10.4.
  - ElevenLabs had a pitch range 40% narrower than natives (log-F0 range 0.87 vs 1.44) and nPVI 92 vs 107, heard as "flat, non-expressive".
  - Chatterbox settings: exaggeration 0.7, temperature 0.6, min_p 0.1 (defaults 0.5, 0.8, 0.05); cfg_weight 0.7 was "mid".
  - Digits in Telugu text produced garbage until written out as Telugu words.
- **Correction:**
  - The 8–9 s "Telugu references" given to Praxy R6 (a LoRA on Chatterbox's T3, not Maata's fine-tune) were **synthetic clips from Sarvam Bulbul and Cartesia Sonic-3**, not real Telugu speakers.
  - They cut retroflex collapse from 40% to 26.7% (Sarvam reference) or 33% (Cartesia reference), and PSD from 61.7 to 13.1 or 26.5.
  - Using such a reference also changes the voice to the reference's voice. That is the similarity-versus-nativeness trade-off of F2, at its "natural" extreme.
- Source: https://arxiv.org/abs/2604.25476 and /html/2604.25476v1 (fetched 2026-09-24).

**F18 [confirmed]: Josh Talks blind A/B test.** It covered 11 languages, and "ElevenLabs v3 alpha leads on audio quality". No Telugu-specific result was reported. It is reported by Sarvam, a competitor.
- Source: https://www.sarvam.ai/blogs/bulbul-v3 (2026-02-05).

**F19 [confirmed; low confidence, read via a WebFetch summary]: user failure reports on the v2 launch post.**
- A Polish user: literal, nonsense phrases, and v2 "compresses some parts to match the timings". The sweep missed this; it suggests sync-aware translation can drop content.
- A Latvian→English user: later sections were left in Latvian, about 64% completed.
- A Spanish user: Latin American and Spain Spanish were treated as the same.
- Source: linkedin.com/posts/matiii_today-we-are-launching-a-revolutionary-new-activity-7465835349643202562-O-dn (about 2026-05-28; fetched 2026-09-24).

**F20 [confirmed]: the only independent hands-on review found.**
- Greg Preece, 2026-07-24, tested French, Italian and Japanese and cites the recommended strength of 7.
- He praises emotion, pacing and tone transfer. v2 delivers audio only: you merge and sync it yourself, and there is no lip sync.
- Source: gregpreece.com/articles/elevenlabs-dubbing-v2-review-tutorial.

**F21 [confirmed, with nuance]: ElevenLabs Productions (human-in-the-loop).**
- **Steps it lists:**
  - a native speaker reviews the translation and edits it for naturalness;
  - human voice selection;
  - for pacing, "we regenerate segments or edit the transcription";
  - a QC checklist.
- **Terms:** delivery in 7 business days; lip sync not guaranteed.
- **Nuance:**
  - The launch blog says Productions pairs human translators with Dubbing v2, which handles speech generation and synchronization.
  - The Productions FAQ says you can refine the result in Dubbing Studio, a v1 tool. The docs are not fully consistent.
- Sources: /docs/eleven-creative/services/productions/dubbing (fetched 2026-09-24); blog/introducing-dubbing-v2.

### E. Maata repo facts the recommendations depend on (checked 2026-09-24)

**F22 [confirmed]: segmenter settings.** In `segment.py` `SegmenterSettings`:
- `min_len` 2.0 s, `max_len` 12.0 s;
- `long_pause` 1.5 s ("splits anywhere");
- `anchor_pause` 0.7 s: in-unit re-sync anchors at clause or sentence marks.

Units are sentence-complete. A unit stays one translation unit because Telugu is verb-final. Splitting is by time, pauses and sentence ends, never by word count.

**F23 [confirmed, new]: the voice-match setting has no effect.** The UI has a three-level setting, `CLONE_STRENGTH_OPTIONS` in `app/src/lib/voice.ts`: closest, balanced and natural. The server passes it to `Session`, and `session.py:114` stores it as `self.clone_strength`. No other engine code reads it (checked with grep over `engine/src/maata_engine`).

**F24 [confirmed, new]: Chatterbox reference limits.** The pinned `chatterbox/mtl_tts.py` sets `ENC_COND_LEN = 6 * S3_SR` and `DEC_COND_LEN = 10 * S3GEN_SR`. Maata's `torch_common.py` slices timbre to 10 s and prompt tokens to 6 s. Only the VoiceEncoder embedding can pool more audio, and ADR-017 pools up to 60 s.

**F25 [confirmed]: ADR-017 evidence.**
- Guest: similarity 0.400 → 0.466 and pace 3.43 → 5.11 aksharas/s.
- Host: stitched reference stays best at 0.645.
- T3 is prompted with the speaker's own English at cfg 0.5.
- The speed cap is 1.2x; lag is at most 0.6 s; up to 0.3 s early start.

---

## Refuted

1. **"ElevenLabs keeps the three-step cascade for enterprise dubbing, for control and safety" (report F9 and the summary).** Refuted. In the Turing Post interview this remark is about conversational voice agents, and the STT→LLM→TTS walkthrough is about real-time translation, not dubbing (F10).
2. **"Add a user-facing similarity setting" (report R5; implies Maata has none).** Refuted. Maata already ships a three-level voice-match setting modelled on `cloning_strength`. It is a no-op in the engine (F23).
3. **"Split at pauses and sentence ends, never on word count alone" (report R6; implies Maata splits on word count).** Refuted. `segment.py` splits by seconds, pauses and sentence ends. The "≤12–20 words" description is out of date (F22).

## Unverifiable

- **U1.** That Dubbing v2 reuses Scribe for ASR and diarization. It is inferred only from the matching 32-speaker limit. Studio has used Scribe since 2025-03-03.
- **U2.** That the `dubbing_v3` enum value (changelog 2025-11-21) is today's public "Dubbing v2".
- **U3.** The "80–90% of human quality" self-estimate. No method was given.
- **U4.** What Dubbing v2's generator is (speech-to-speech conditioned on target text, or TTS with a per-segment prosody reference), which LLM translates, and what duration targets it uses.
- **U5.** How much of each speaker's audio automatic dubbing clones from, and the maximum speed-up or stretch v2 applies.
- **U6.** The v2 app upload limit. The capabilities page says 1 GB; the help-centre FAQ says 2 GB.
- **U7.** That ADR-017 rejected native Telugu prompt clips "on similarity alone". The ADR only says they "were measured and did not help", and no spike results are committed.
- **U8.** Whether ElevenLabs holds dubbing patents. None could be searched: Google Patents returned 503, Justia showed a bot wall, and WebSearch was exhausted.
- **U9.** Whether the realtime Dubbing SDK types (2026-08-24) mean a live-dubbing product is coming.

---

## Recommendations for Maata (updated for the corrections)

### R1 (high impact, M effort, no approval needed): sync-aware translation with Claude from the first pass

**What to change:**
- Send `claude -p` one scene at a time: 2–5 min of the 5–10 min lookahead, as JSON.
- Each item carries the segment id, speaker, start, end, slot seconds and a Telugu akshara budget derived from that voice's measured pace. Keep units whole, since Telugu is verb-final.
- Ask for spoken Telugu that fits each budget. Ask for per-id output plus an explicit list of any content dropped.
- Keep the current three-wording request as the fallback for lines that still overrun.

**Why:** ElevenLabs translates "with timing in mind" from one master transcript, and says its generator renders each translation into its segment span, so comparable spoken length is the translator's job (F4, F8). Its CEO calls translation under duration limits the hardest part (F10).

**Caveat:** users report that v2's compression drops content (F19). R1 must ship together with R2.

### R2 (high, M, no approval): coverage and review gate

**What to change:**
- Hard-check that every segment id returns non-empty text in Telugu script.
- Run a second Claude review call that flags omissions, literal or bookish wording, register problems and dropped content, then retranslate the flagged lines.
- After synthesis, compare the audio duration with the expected aksharas to catch Chatterbox truncation.
- Log a pass/fail status per line.

**Why:** ElevenLabs' human service relies on native-speaker review plus a checklist (F21), and automatic v2 still produces literal and incomplete dubs (F19). These are the same failures the maintainer reports.

### R3 (high, M, no new dependency): make the existing voice-match setting real, then improve references

**What to change:**
1. Wire closest, balanced and natural (F23) to actual parameters: cfg_weight, exaggeration and the reference choice. Label it like ElevenLabs' trade-off (F2).
2. Choose the S3Gen 10 s span nearest the speaker's WeSpeaker centroid, not only the cleanest span. Chatterbox cannot use more than 10 s (F24), so ElevenLabs' 1–2 min guidance turns into better span selection plus embedding pooling. Trial pooling the T3 embedding over up to 120 s against the current 60 s.
3. Synthesize 2–3 takes per line and keep the best by WeSpeaker similarity plus duration fit. The memory freed from the local LLM (about 8 GB) pays for this. This is ElevenLabs' "Clip History" and Productions' regeneration (F7, F21).
4. Score speaker similarity and a native listener's naturalness rating separately (F2, F17).

### R4 (medium, M, experiment): per-line performance conditioning

**What to change:**
- Keep the speaker's timbre span for S3Gen.
- For each line, drive exaggeration, or the T3 prompt when the line's own clean source audio is 3 s or longer, from that line's delivery. Alternatively, use an energy or emotion label that Claude adds per line (R8).
- Include the PSP sampling settings (exaggeration 0.7, temperature 0.6, min_p 0.1) as one arm.
- Judge with a blind A/B test.

**Why:** v2's claimed gain is conditioning on each line's original performance (F8). Studio's clip clone is the manual version (F7).

### R5 (medium, S): retest Telugu-language prompts, correctly framed

**What to change:** make the "natural" end of R3's setting test a native Telugu prompt for T3 while keeping S3Gen timbre from the speaker. Cross it with cfg_weight 0, 0.3 and 0.5. Score native-listener naturalness and accent, and WeSpeaker similarity.

**Why, corrected:** PSP's gains came from synthetic commercial Telugu clips, which also replaced the voice (F17). Expect nativeness to rise and similarity to fall. ADR-017 did not record which metric it used (U7).

**Needs approval:** the source of any Telugu prompt clip must be licence-clean (CC0 or CC-BY, or self-recorded). A clip generated by commercial TTS would breach the no-cloud-AI rule.

### R6 (medium, S): align segmentation with ElevenLabs' rule, by A/B test rather than by fiat

**What to change:**
- Maata already splits by time and sentence (F22). Test `long_pause` at 1.0 s (ElevenLabs' threshold, F6) against today's 1.5 s. Measure onset lag p95, the freeze rate and translation completeness.
- Keep the 0.7 s anchors.
- Let different speakers' lines overlap where the source overlaps, as ElevenLabs allows (F6).

### R7 (medium, S): keyterms

**What to change:** build a per-video list of names, brands and technical terms from the yt-dlp title, description and chapters plus the proper nouns in the ASR output. Pass it to the ASR prompt and to Claude as a keep-in-English or transliterate glossary.

**Why:** ElevenLabs' `keyterms` biases both transcription and translation (F3).

### R8 (low–medium, S): the ElevenLabs "Enhance" pattern and text normalisation

**What to change:**
- Have Claude add per-line delivery labels (energy, emotion, question) without changing the text, and map them to TTS parameters.
- Strip bracketed non-verbal tokens before synthesis.
- Expand numbers into Telugu words.

**Why:** ElevenLabs uses an LLM for tags and advises normalising text (F15), and PSP saw digits break Chatterbox (F17).

### R9 (medium, S; needs maintainer decision): background audio

**What to change:** keep the dry dub (dub voice only) as the default for talk content, with loudness normalisation and optional synthetic room tone.

**Why:** ElevenLabs keeps the background by default, but ships `drop_background_audio` because dropping it "can improve dub quality" for speeches and monologues (F5).

**Needs approval:** a separated background stem would require relaxing "the original audio is never played", plus a check that no English speech leaks through.

### R10 (medium, S): timing repair order

**What to change:** make the order reword, then regenerate a take, then speed up.

**Why:** Productions fixes pacing by regenerating or editing (F21). ElevenLabs describes only "slightly modified" speed (F9). Maata's 1.2x cap is its own choice, not an ElevenLabs dubbing limit (F15 correction).

### R11 (low, S; needs approval: cloud upload and cost, benchmark only, never shipped): external bar

**What to change:** dub 2–3 min of the maintainer's own content with Dubbing v2 (te, strength 7) in the app. That works on any plan; API editing is Enterprise-only (F3). Blind-compare it with Maata on meaning, naturalness, similarity and sync.

**Expectations:** ElevenLabs' Telugu TTS prosody measured flat in a small, conflicted pilot (F17), and users report literal translations (F19). The v2 per-minute price was not verified.

### R12 (low, M): per-segment revision and stale state

**What to change:** track a revision and stale state per segment, so an edit or a retranslation from R2 re-synthesizes only that line, as the v2 API does (F3).

---

## Open questions

1. What exactly is v2's generator, and which LLM does its translation (U4)? Does v2 condense translations to fit, and so drop content (F19)?
2. What is v2's maximum time-stretch, and how does it handle laughter, non-verbals and overlapping speech in the output (U5)?
3. How much of each speaker's audio does automatic dubbing clone from (U5)?
4. What does the `dubbing_v3` enum mean, and is a live-dubbing product coming (U2, U9)?
5. There is still no independent Telugu evaluation of Dubbing v2. The only Telugu data is PSP's 10-utterance pilot on Eleven v3 TTS, by a conflicted author.
6. For Maata: can a 2–5 min scene per `claude -p` call meet the lookahead budget given CLI latency and subscription rate limits? This was not measured.
7. Patents are unsearched (U8).

## Source index (all fetched 2026-09-24 unless dated)

- **ElevenLabs docs** (undated pages): overview/capabilities/dubbing; eleven-creative/products/dubbing; …/dubbing/dubbing-studio; api-reference/dubbing/{create-project, language-targets/create-language-target, source-transcript/get-source-transcript, target-transcript/get-target-transcript, target-transcript/regenerate-target}; api-reference/legacy/dubbing/create; eleven-api/guides/how-to/dubbing/{bring-your-own-transcript, refine-and-regenerate}; eleven-api/concepts/voice-cloning; eleven-creative/voices/voice-cloning/{instant-voice-cloning, professional-voice-cloning}; help-center voice-cloning and dubbing FAQs; overview/models; overview/capabilities/{text-to-speech/best-practices, speech-to-text, voice-isolator, forced-alignment}; eleven-creative/services/productions/dubbing.
- **ElevenLabs changelog:** 2025-02-25, 2025-03-03, 2025-04-14, 2025-05-12, 2025-09-01, 2025-11-21, 2026-04-20, 2026-08-10, 2026-08-24.
- **ElevenLabs blog and web pages:**
  - Introducing Dubbing v2 (2026-05-28, updated 2026-09-06);
  - AI dubbing API (2026-08-06, updated 2026-09-22);
  - What is AI dubbing? (2026-07-27);
  - AI Dubbing launch (2023-10-10);
  - Dubbing v2 landing page (undated);
  - X post, Dubbing Studio (2024-01-23).
- **Third party:**
  - Turing Post interview (2025-04-12);
  - The Elec (2026-08-11);
  - Greg Preece (2026-07-24);
  - LinkedIn launch post and comments (about 2026-05-28);
  - PSP, arXiv 2604.25476v1 (2026-04-28);
  - Sarvam Bulbul V3 blog (2026-02-05).
- **Maata repo:**
  - `engine/src/maata_engine/segment.py`, `session.py`, `server.py`, `backends/torch_common.py`;
  - `engine/.venv/.../chatterbox/mtl_tts.py`;
  - `app/src/lib/voice.ts`, `settings.ts`;
  - `docs/DECISIONS.md` (ADR-017, ADR-018).
