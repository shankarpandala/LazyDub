# 03 Other commercial dubbing products (especially Indian languages): verified report

Verified 2026-09-24 against primary sources. The draft checked is `03-other-commercial.sweep.md` in this folder, along with the researcher's JSON summary of 29 findings. It builds on `../sota_ranking.md` for models and does not redo that work.

**How this was checked.** The session's WebSearch budget was already used up (200 of 200), so every check went through WebFetch, curl (including the `.md` endpoints on docs sites), the Hugging Face API, the GitHub API (`gh`), local copies of pages and papers, and the LazyDub repo at HEAD.

**What the tags mean.**
- DISCLOSED: the company says it.
- MEASURED: a third party measured it, or I read it in code or an API.
- INFERENCE: my reading, not stated anywhere.

**Verdicts.**
- [confirmed]: the claim matches the primary source.
- [corrected]: part of the claim was wrong; the correct fact and its source are given.
- [unverifiable]: I could not confirm it, so it should not be relied on.
- Refuted claims are listed at the end.

## Summary

The researcher's main lessons mostly stand. Five of them change the recommendations.

1. **Maata already fits the text to time at translation.** The report said Maata "does not yet budget each line's length at translation time". That is refuted by the repo.
   - `session.py` passes `target_units` = slot seconds × the voice's measured aksharas/s × 0.95. The aksharas are counted in code (`count_units`).
   - `tenglish.py` adds "Length: about N aksharas" to the prompt, and asks for normal, concise and very concise wordings when a line is predicted to overrun.
   - The real gap against Descript is the other direction: short lines. Maata deliberately never lengthens text ("never add words to reach the length"), and never slows audio below 1.0× (`planner.py` rate is in [1.0, 1.2]). The ADR-017 smoke run had Telugu at 0.78× of its slots.
   - Descript's listening test says audio slowed by 10% still sounds natural. Descript's "Match timing" also lengthens phrasing.
2. **The IFrame rate list is not the real limit.**
   - The API reference (updated 2026-09-15) does say unsupported rates are rounded toward 1.
   - But the timing sweep's own probe found something else in Chromium 152. `getAvailablePlaybackRates()` listed only 0.25 steps, yet `setPlaybackRate(0.9)` (and 0.8 to 1.3) read back exactly and fired `onPlaybackRateChange`.
   - So "if it lists only 0.75 and 1, drop the idea" is wrong. What needs testing is the readback and whether frames actually advance at that rate, in the Tauri WKWebView.
3. **Sarvam's own Bulbul V3 study says ElevenLabs v3 alpha leads on full-band quality.** Bulbul wins only at 8 kHz telephony. So there is no contradiction with the independent AI4Bharat/IITM/Josh Talks leaderboard (Gemini 2.5 Pro TTS, then ElevenLabs v3 ≈ Sonic 3, then Bulbul). Josh Talks ran both studies.
4. **Chitralekha's automatic path does not cut sentences.** It speeds up an overrunning line by the full ratio with no cap (ffmpeg `atempo`), then trims only the residue. Its failure mode is "chipmunk" speech, not truncation. This still supports Maata's 1.2× cap.
5. **Sarvam documents more than the report found.**
   - A Dubbing API (Beta) covering 12 languages, with per-speaker cloning, at ₹40/min.
   - An editor with separate **Dubbed / Original / Background** volume tracks. Its best-practice guide warns that muting background audio entirely "can make a dub feel sterile".
   - A style selector (Auto / Formal / "Urban colloquial"). The guide says a flat or overly formal dub should be fixed through style before blaming the voice model.
   - Bulbul v3 and Saaras v3 can now be self-hosted, but only on AWS SageMaker. Weights are still closed and nothing runs on-device.

Other points that still hold:
- Premium tiers are the same model plus a human pass.
- Indic vendors treat register, script, spoken form and gender as translation inputs.
- Expressiveness and intelligibility drive Indic listener preference.
- None of the strong Indic or dubbing vendors ship usable open weights.

No independent end-to-end evaluation of AI dubbing into Telugu exists. Descript's dubbing does not support Telugu at all (Hindi and Tamil only).

**Counts** (finding-level verdicts, plus sub-claims that were refuted or couldn't be checked): 18 confirmed, 11 corrected, 5 refuted, 10 unverifiable.

## Verified findings

### Sarvam (India)

**F1. Sarvam Dub puts duration control inside speech generation.** [confirmed] DISCLOSED.
- The blog says systems that speed up or slow down audio afterwards distort rhythm. With Sarvam Dub, "You specify the target duration upfront."
- Speaker similarity is ECAPA-TDNN cosine on Sarvam's own set: 64 speakers, 10 Indian languages plus English, more than 700 samples per system, same-language and cross-lingual. Results appear only as a chart.
- Mann Ki Baat is dubbed monthly into 11 languages, with a Telugu sample on the page. Union Budget 2026 was dubbed live into Kannada and Hindi.
- A 6.6× latency cut came from ONNX tracing, selective quantisation, PTQ on in-distribution calibration data, and caching. Code-mixing is described as "the norm".
- Source: https://www.sarvam.ai/blogs/sarvam-dub (datePublished 2026-02-01, dateModified 2026-09-14; saved copy `../sdub.html`). BusinessToday coverage, 2026-02-01/02.

**F2. The Sarvam Dub product is a four-stage cascade.** [confirmed] DISCLOSED. Added: the API.
- Product page stages: Saaras STT with timestamps, then Sarvam Translation keeping "timing cues needed for sync", then Bulbul V3 with per-line emotion, pitch and pace, then sync and export.
- Coverage is 11 languages including Telugu, with 35+ voices. Price starts at ₹30 per 10K characters, and a 90-minute film costs under ₹2,500.
- **Added:** the docs describe a Dubbing API (Beta).
  - 12 Indian target languages, per-speaker voice cloning and translation tone control.
  - It exports video, audio and SRT from one asynchronous job.
  - Price is ₹40/min on Starter, doubling to ₹80/min with `editor_flow`.
- Sources:
  - https://www.sarvam.ai/text-to-speech/dubbing (undated, fetched 2026-09-24)
  - https://docs.sarvam.ai/api/api-guides-tutorials/dubbing/overview (fetched 2026-09-24; copy `sv_api_api-guides-tutorials_dubbing_overview.md`)

**F3. Bulbul V3 is an LLM-based TTS with a vendor-run preference test.** [corrected] DISCLOSED.
- Confirmed:
  - It is "built on an LLM" that infers emphasis, pauses, tone and pacing.
  - It supports voice cloning.
  - The Josh Talks blind A/B test used 50–70 annotators and about 2,000 votes per language, over 20,000 votes in total.
  - There is no Telugu-specific number.
- Corrected:
  - The test covered **11 languages** with **500+ annotators**.
  - Sarvam calls Josh Talks "independent third-party"; that Sarvam commissioned it is an inference.
  - Result: "ElevenLabs v3 alpha leads on audio quality" in full-band. Bulbul beats Sonic-3 and the rest, and is top only at 8 kHz telephony.
  - The Telugu cloning sample writes English loans in **Telugu script** (ప్లీజ్ కన్ఫర్మ్, పేమెంట్). Only "Okay" and "7th November" stay in Latin.
- The weights are closed (see F20).
- Source: https://www.sarvam.ai/blogs/bulbul-v3 (datePublished 2026-02-05, dateModified 2026-09-14; copy `bulbul.txt`).

**F4. Sarvam's translation API takes register, script, spoken form and speaker gender as inputs.** [confirmed] DISCLOSED.
- Four modes:
  - Formal: "pure language forms".
  - Classic-Colloquial: "Balanced mix of languages, slightly informal".
  - Modern-Colloquial: "Casual and direct style with mixed language".
  - Code-Mixed: English inside the target sentence, `mayura:v1` only.
- `output_script` is roman, fully-native or spoken-form-in-native.
- `speaker_gender` (Male/Female) gives "gender-specific translations". `numerals_format` is international or native.
- `mayura:v1` supports te-IN with a 1,000-character cap. `sarvam-translate:v1` is formal only.
- Sources: https://docs.sarvam.ai/api/api-guides-tutorials/text-processing/translation and https://docs.sarvam.ai/api-reference-docs/models/mayura (undated, fetched 2026-09-24; copies `sv_translation.md`, `sv_mayura.md`).

**F5. Sarvam's TTS writing guidance.** [confirmed] DISCLOSED. One sub-claim is [unverifiable].
- Write English words in English script and Indic words in native script. Romanised Indic input "significantly degrades output quality".
- Punctuation controls pauses: comma is short, full stop medium, ellipsis a hesitation. Split long sentences. There is no SSML.
- Pace: 0.8–0.9 relaxed, 1.0 natural default, 1.1 slightly brisk, 1.2–1.5 fast.
- Saaras v3 STT has a `codemix` mode that keeps English words in Latin.
- The claim that normalisation of English words and numbers "always runs" in v3 is [unverifiable]. The docs say to set `enable_preprocessing=true` when text has many abbreviations or digits. They also say the language code feeds a pre-TTS normalisation model.
- The rule to keep English in Latin script conflicts with Sarvam's own Telugu sample (F3), so for Telugu it is not settled.
- Maata's `tenglish.py` keeps English in Latin script and logs `latin_ratio` without targeting it. Confirmed in the repo.
- Sources: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/best-practices, https://docs.sarvam.ai/api-reference-docs/building-for-india, and https://docs.sarvam.ai/api/getting-started/models/bulbul (undated, fetched 2026-09-24).

**F13. Sarvam Studio.** [corrected] DISCLOSED, vendor-run evaluation.
- Confirmed:
  - Transcription, speaker separation, translation and speech generation are automatic, and users can edit and regenerate segments.
  - Domain experts judged about 280 head-to-head comparisons against ElevenLabs, YouTube Dub and Rask.
  - Average similarity was 0.88, with no Telugu breakdown.
- Corrected:
  - The "custom glossaries, terminology, and style guidelines" belong to Sarvam's **document** translation product, not to dubbing.
  - The Studio post does not name the similarity encoder; ECAPA is inferred from F1.
- Source: https://www.sarvam.ai/blogs/sarvam-studio (2026-02-12).

**N1 (missed). Sarvam's dubbing best-practice guide.** [confirmed] DISCLOSED.
- The style dropdown is Auto / Formal / "Urban colloquial". The guide says a dub whose tone "feels flat or overly formal" should get a different style before the voice model is blamed.
- Other advice:
  - Get the speaker count right before upload, rounding up when unsure.
  - Review multi-speaker dubs speaker by speaker.
  - Spot-check names, brands and technical terms.
  - Regenerate per block, and use Find & Replace for a recurring mistranslation.
- The editor has three volume sliders: **Dubbed**, **Original** and **Background**. The guide says to keep some background, because muting it entirely "can make a dub feel sterile".
- Sources: https://docs.sarvam.ai/creative-dubbing-best-practices and https://docs.sarvam.ai/creative-dubbing-edit (fetched 2026-09-24).

### Descript (with OpenAI)

**F6. Descript translates to a syllable budget, not by stretching audio.** [confirmed] DISCLOSED, with vendor-run measurements.
- Unnatural pace was the "number one complaint".
- The method:
  - Chunks follow sentence boundaries, pauses and speaking patterns.
  - The model counts source syllables, and language-specific speaking rates set the target syllable count.
  - The prompt optimises duration and meaning together, with surrounding chunks as context.
  - Earlier models could not count syllables reliably; GPT-5-series models could.
- Listening test: slowed by 10% or sped up by 20% "generally still sounded natural".
- Results:
  - The share of segments inside that window rose from 40–60% to 73–83% depending on language.
  - Duration adherence rose 13–43 percentage points, and dubbed exports rose 15% in 30 days.
  - An LLM judge rated 85.5% of segments 4 or 5 out of 5, with a deliberately lower meaning bar than for captions.
- Date: the page is undated. The secondary summary (byline 2026-03-06, metadata 2026-03-30) calls it "OpenAI's March 2026 case study".
- Sources: https://openai.com/index/descript/ (local copy `descript_jina.md`) and https://www.gend.co/blog/descript-multilingual-dubbing-openai-models.

**F7. Descript's "Match timing" fits lines in both directions.** [confirmed] DISCLOSED.
- The option "may use additional descriptive words or different phrasing to achieve the right duration" and takes longer to process.
- Auto uses timing match for dubs.
- Direct translation leaves audio "slightly sped up or slowed down".
- Source: https://help.descript.com/hc/en-us/articles/37202655823629 (undated, fetched 2026-09-24).

**N2 (missed; corrects sweep F16). Descript's dubbing does not support Telugu.** [corrected] DISCLOSED.
- The stock-voice dubbing table lists **28 languages**, not 30+. They include Hindi and Tamil but **not Telugu**.
- "Recommended voices designed specifically for dubbing" cover **13 languages**, not 12, on Business and Enterprise only.
- Editing a translation requires Regenerate. There is also a "do not translate" list.
- Source: https://help.descript.com/hc/en-us/articles/37194900295821 (`.md` copy, fetched 2026-09-24).

### Western dubbing platforms

**F8. HeyGen retimes the video by default, and its pricier tier has better translation.** [confirmed] DISCLOSED. One sub-claim is [unverifiable].
- Dynamic Duration lets segments stretch or compress "by up to ±20%". It is enabled by default and "may affect the total length of your video". The API parameter `enable_dynamic_duration` defaults to true.
- Speed costs 6 credits/min with "Adequate" translation. Precision costs 10 credits/min with "Context- and Gender-Aware" translation and high lip-sync. Audio-only costs 4 credits/min.
- Proofread (Review & Edit) is on Pro, Business, Team and Enterprise plans.
- Sound effects are "always removed". Music is kept unless Remove Background Sound is on. Only one person should speak at a time.
- The brand glossary holds pronunciations, do-not-translate terms and forced translations.
- [unverifiable]: "the API docs say to turn it off only for frame-exact timing" is not in the current docs. HeyGen's Telugu support is also unverified, because the language list sits behind the API.
- Sources (undated, fetched 2026-09-24; copies `hg_*.md`):
  - https://help.heygen.com/en/articles/10029081
  - https://developers.heygen.com/docs/video-translation-precision
  - https://developers.heygen.com/docs/video-translate

**F9. Synthesia's default mode also changes the video's speed.** [corrected] DISCLOSED.
- Confirmed:
  - Adaptive (the default) "Adjusts video playback to better match the length of the translated voiceover" and is recommended for instructional content.
  - Original keeps the video speed and adjusts only the voiceover.
  - Speakers are auto-detected and cloned, and can be reassigned.
  - Review Transcript is free, lip sync costs 240 credits per minute, and only spoken audio is translated.
  - 134 languages; Telugu was not seen in the fetched text.
- Corrected: "Dubbing 2.0 released 2026-07-15" is refuted as sourced. The cited feisworld article is dated 2025-11-26 and gives no such date.
- Source: https://help.synthesia.io/en/articles/10054222 ("updated over 3 weeks ago" at fetch on 2026-09-24).

**F16. ElevenLabs Productions is the automatic dubber plus a human pass.** [corrected] DISCLOSED.
- The five human dubbing steps are confirmed:
  - transcription with speaker allocation;
  - native-speaker translation review "to make the voice-over sound as natural as possible";
  - voice selection;
  - pacing, by regenerating segments or editing the transcription;
  - a QC checklist.
- Delivery takes about 7 business days.
- Corrected: "Pricing starts at $2.00 per minute" covers Productions as a whole (dubbing, captions and subtitles, transcripts, audiobooks). The dubbing docs say dubbing prices are a custom quote from sales.
- Sources: https://elevenlabs.io/blog/introducing-productions-human-edited-content-done-for-you (2025-09-15) and https://elevenlabs.io/docs/services/productions/dubbing (undated, fetched 2026-09-24).

**F17. Deepdub.** [confirmed] DISCLOSED.
- eTTS 2.0 is described as "a multimodal Large Language Model" covering 130+ languages, with Accent Control to keep or adapt accents (2024-02-29).
- Deepdub GO is for post-production editors; the white-glove service is run by in-house staff.
- Phantom X 3.2 (2026-03-10):
  - Zero-shot cloning from about 1 s of possibly noisy audio.
  - Emotion styles layered within a line.
  - "Key Names and Phrases (KNP)" for consistent pronunciation **and** translation of recurring names and terms.
- No Indian languages are named, and there is no mention of open weights.
- Sources: prnewswire 302075611 (2024-02-29) and 302709376 (2026-03-10).

**F18. Papercup was absorbed by RWS.** [confirmed] DISCLOSED. One sub-claim is [unverifiable].
- RWS acquired Papercup's IP. CEO Jesse Shemen and much of the team left for Scale AI.
- RWS says its "hybrid approach puts humans in the loop to optimise tone, pacing and accuracy". It has 1,800 in-house specialists and a network of 40,000+.
- TechCrunch (2022-06-09) confirms that native speakers do QC.
- "Studio quality at one-fifth of the cost" is not in TechCrunch [unverifiable].
- Sources: https://businesscloud.co.uk/news/rws-scoops-up-papercup-ip-after-staff-exodus/ (2025-06-26) and https://techcrunch.com/2022/06/09/papercup-raises-20m-for-ai-that-automatically-dubs-videos/.

**F19. CAMB.AI MARS8.** [confirmed] DISCLOSED plus HF API.
- Report dated 2026-01-20. Models: Flash 600M, Pro 600M, Instruct 1.2B and Nano 50M.
- Speaker similarity 0.87 (WavLM-base-sv) and 0.71 (CAM++) from references averaging 2.3 s. 70% of the MAMBA benchmark is cross-lingual.
- Flash CER is 5.67%. There are no Indic results, and BOLI is "coming soon". The architecture is not disclosed.
- The only CAMB-AI repo on Hugging Face is MARS5-TTS, licensed **AGPL-3.0** and created 2024-06-07.
- Sources: https://www.camb.ai/blog-post/mars8-technical-report and the HF API (2026-09-24).

**F21. Rask.ai.** [confirmed] DISCLOSED. One sub-claim is [unverifiable].
- The llm-info page (updated 2026-04-10) lists:
  - translation prompting;
  - terminology dictionaries;
  - enterprise human-in-the-loop;
  - AI script adjustment;
  - multi-speaker detection;
  - 135+ languages, with cloning in 32.
- There is no architecture disclosure, and Telugu is not mentioned.
- The third-party review is by a **competitor**, ESTsoft's Perso Dubbing (2026-09-08). It confirms lip-sync uses 3× minutes, and plans at $60, $150 and $750 a month.
- "Expressiveness / Similarity / Accent dials" are not in the review [unverifiable].
- Sources: https://www.rask.ai/llm-info and https://perso.ai/blog/rask-ai-dubbing-review-2026-features-pricing-how-it-compares.

**F22. Panjaya BodyTalk.** [confirmed] DISCLOSED. One sub-claim is [unverifiable].
- It "automatically adjusts translations and aligns the rhythm and tone of the dubbed audio with the pauses, breaths, gestures and other body movements".
- It uses third-party translation and TTS with in-house lip-sync, in 29 languages, with a human-in-the-loop tool planned. TED is a customer. The $9.5M seed round closed in September 2024.
- The "doubled completion rates" claim and the Slator page date were not seen [unverifiable].
- Source: https://slator.com/ai-dubbing-start-up-panjaya-ai-bags-9m-funding/.

**F23. Visual dubbing (Flawless, NeuralGarage) uses human voices.** [confirmed] DISCLOSED.
- TrueSync is "audio-agnostic" and "does not generate, modify, or synthesize audio".
- THR India (updated 2026-05-12): NeuralGarage VisualDub was used on War 2 so Hindi-shot scenes look spoken in Telugu, with human voices.
- The Variety article on Coolie redirected to a paywall [unverifiable].
- Sources: https://flawlessai.com/localization-and-dubbing and https://www.hollywoodreporterindia.com/features/insight/exclusive-the-ai-revolution-hollywood-is-fearing-is-already-here-in-india.

### Independent evaluations and open-source tools

**F12. The independent Indic TTS leaderboard.** [corrected] MEASURED.
- The paper is arXiv 2604.21481 v2 (23 June 2026), by IIT Madras, AI4Bharat and Josh Talks.
  - 5,357 sentences in 10 languages including Telugu.
  - 1,915 raters and 120K comparisons, scored with Bradley-Terry.
- Scores:

  | System | Score | Win rate |
  |---|---|---|
  | Gemini 2.5 Pro TTS | 1128.53 | |
  | ElevenLabs v3 | 1056.28 | |
  | Sonic 3 | 1050.83 | |
  | Bulbul v3 Beta | 1021.91 | |
  | MiniMax Speech 2.8 HD | 993.94 | |
  | GPT-4o-mini TTS | 942.76 | |
  | IndicF5 | 805.75 | 19% |

  - Gemini is first in 9 of 10 languages and all 16 domains.
  - On the code-mixed subset: Gemini 1135.45, Sonic 3 1054.74, ElevenLabs v3 1054.00.
- SHAP: expressiveness and intelligibility predict preference most, then liveliness and voice quality. Hallucinations and noise matter less once basic robustness is met.
- Default voices were used, with no cloning and no style conditioning.
- Not every system covered every language. Language counts: ElevenLabs 9, Sonic 3 8, Bulbul 9, MiniMax 2.
- Corrected: this does **not** contradict Sarvam's own study (see F3), and Josh Talks co-ran both.
- Source: local `vfn.txt` (arXiv HTML v2).

**F26. AI4Bharat open speech data includes Telugu.** [confirmed] HF API, 2026-09-24.

| Dataset | Commit | Licence | Gating | Telugu | Last modified |
|---|---|---|---|---|---|
| SpeechArenaBench | fdd26e85 | MIT | automatic | yes | 2026-06-19 |
| Rasa | 632f55c7 | CC-BY-4.0 | automatic | yes (expressive; RASA-test used in F12) | 2026-06-06 |
| IndicVoices | c96f9088 | CC-BY-4.0 | | telugu config | 2026-06-15 |
| indicvoices_r | 5f4495c9 | CC-BY-4.0 | | yes | 2025-03-06 (not 2026) |

**F15. Chitralekha (AI4Bharat) speeds up overrunning lines without a cap.** [corrected] Blog DISCLOSED; code MEASURED.
- Confirmed:
  - The roles are Transcript, Translation and Voice Over Editors and Reviewers, plus a Universal Editor.
  - The models are IndicASR, IndicTrans, IndicTTS and IndicXlit.
  - Machine voice-over is available "only for a single speaker video".
  - The backend is MIT; bc2c16a3 is a 2026-02-13 merge of PR #1074 (per gh).
  - `background-music-api` runs Spleeter 2-stems on 60 s chunks and overlays the accompaniment under the dub.
- Corrected, from `backend/voiceover/utils.py` `adjust_audio`:
  - A short line is padded with silence.
  - A long line gets ffmpeg `atempo` at the **full** ratio when above 1.009, uncapped. The image is python:3.8 with apt ffmpeg, where atempo accepts values above 2.
  - Then `-t original_time` trims only the residue.
  - The automatic failure is uncapped speed-up, not cut sentences.
- Sources: https://ai4bharat.iitm.ac.in/blog/chitralekha (2024-01-02) and https://github.com/AI4Bharat/Chitralekha-Backend @bc2c16a3 (local copy `chb/`).

### Indian dubbing and speech market

**F14. Dubverse.** [corrected] DISCLOSED.
- The interview is confirmed and dated **2022-07-22** (Slator):
  - A 5-minute video takes about 30 s to process and then gets "about a 15-minute review".
  - Dubverse hired a translator who had worked with Google and Netflix.
  - "Emotion is the biggest hurdle".
  - Cloning then needed "one hour of data" for a "90, 95% match".
- Corrected, on the Telugu product page:
  - Six named Telugu voices: Shaan, Sunidhi, Sumahi, Nara, Lakshanika and Lehan.
  - "9 different Indian native languages", and 120+ voices across 30+ languages.
  - The page does **not** mention code-mixed scripts or cloning, and still says "private beta", so it is stale.
- Sources: https://slator.com/dubverse-anuja-dhawan-on-ai-dubbing-for-youtubers-tiktokers-content-creators/ and https://dubverse.ai/online-video-dubbing-telugu/ (fetched 2026-09-24).

**F20. No Indian vendor's TTS or dubbing model runs locally.** [corrected] HF API plus docs.
- Sarvam's Hugging Face models:
  - LLMs: sarvam-m, 30b and 105b, Apache-2.0.
  - sarvam-translate: GPL-3.0.
  - shuka-1: Llama 3 licence. OpenHathi: Llama 2 licence. sarvam-1: no licence tag.
  - No TTS, STT or dubbing weights.
- gnani-ai has one ASR model (Apache-2.0, 2026-05-29). Navana has none.
- Gnani Vachana (BusinessToday, 2026-02-19) clones from under 10 s in 12 languages including Telugu.
- Corrected:
  - Vachana also offers **on-premises** deployment for regulated sectors. The article does not mention code-mixing.
  - Sarvam Bulbul v3 and Saaras v3 can be **self-hosted on AWS SageMaker** through the Marketplace (docs, "August 2026").
  - Neither is open weights or on-device, so the conclusion for Maata stands.
- Sources: HF API (2026-09-24), https://docs.sarvam.ai/api/self-hosted/introduction, and https://www.businesstoday.in/technology/story/gnaniai-launches-zero-shot-voice-cloning-tts-model-for-12-indic-languages-516896-2026-02-19.

**F24. Telugu viewers complain about suspected AI dubs on JioHotstar.** [confirmed] Press. Confidence is low that AI was actually used.
- binged.com (2026-04-11) names the Telugu dubs of Daredevil: Born Again S2 and The Hunting Party S2.
  - Complaints: flat delivery, lip-sync mismatch, robotic or "unreal" translation.
  - JioHotstar has not commented.
- "Jio Voice Print" was announced at Reliance's 48th AGM (India TV, 2025-08-30).
- THR India (2026-05-12):
  - About 20,000 freelance voice artists work in India.
  - M.G. Srinivas cloned Shiva Rajkumar's voice from Kannada into three languages for Ghost, and later co-founded AI Samhitha.
  - **Added:** a voice artist estimates 70–80% of brand voices in major Indian ads have been replaced by AI.

**F25. Pocket FM.** [corrected] Press (TechCrunch, 2026-09-10).
- Confirmed:
  - AI powers 93% of the catalogue and 99% of new content, at about 80× lower cost.
  - Revenue run rate is $500M.
  - Its writing and TTS models are proprietary, trained on production data and listener-engagement signals. Humans still create story concepts.
- Corrected:
  - "Started with ElevenLabs in 2024" is not in this article [unverifiable here].
  - The **US is about 70% of revenue**, so this is not evidence about Indian-language or Telugu listeners.
  - "Largest Indian-language audio business" overstates what the source says.

**F29. The remaining public Telugu dubbing comparisons are vendor marketing.** [confirmed]
- TrueFan AI (2026-01-14): 15 raters, 5 each for Hindi, Tamil and Telugu, using LSE-D, timing offset and MOS. TrueFan's own Studio wins its table, and the "94%" code-switching figure is undefined.
- Speechify lists Hindi and Telugu but discloses no method.

### Maata-specific checks and cross-vendor synthesis

**F10. The IFrame API rounds unsupported playback rates.** [corrected] Doc DISCLOSED; probe MEASURED by the timing sweep.
- The doc text is confirmed:
  - It lists 0.25, 0.5, 1, 1.5 and 2 as typical rates.
  - An unsupported rate is rounded "down to the nearest supported value in the direction of 1".
  - Code should respond to `onPlaybackRateChange`.
  - Source: https://developers.google.com/youtube/iframe_api_reference (last updated 2026-09-15; copy `iframe_api.txt`).
- Corrected consequence, from the timing sweep's probe in Chromium 152 with the youtube-nocookie embed (`08-timing-sync.sweep.md` F18):
  - The list was `[0.25…2 in 0.25 steps]`, yet 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15 and 1.3 were accepted, read back exactly, and fired the change event.
  - It is unverified whether frames actually advanced at 0.9×, and WKWebView (Tauri) was not tested.
- ADR-008 (line 61) already contained a rule for "nearest available IFrame rate at or above 0.85×".
- Probe page: `iframe_probe/index.html`.

**F11. Maata's current timing approach.** [corrected] INFERENCE plus a repo check.
- The disclosed strategies are:
  - fit the text (Descript);
  - fit the model (Sarvam);
  - retime the video (HeyGen, Synthesia).
- Maata already fits the text at translation time. The pieces:
  - `session.py` sets target_units = slot × voice rate × 0.95.
  - `tenglish.py` adds a length note and requests tiers on predicted overrun; "too short" is only logged.
  - One concise re-synthesis is allowed for up to 10% of lines.
  - Speed-up is capped at 1.2× and freezes are metered.
- What it does not do:
  - lengthen short lines, or slow audio below 1.0×;
  - slow the video (ADR-017: "The video is never slowed down").
- The 1.2× cap matches Descript's natural-speed limit.
- Source: LazyDub HEAD (`engine/src/maata_engine/session.py:479`, `text/tenglish.py:49–51, 388, 706–726`, `timing/planner.py`, `docs/DECISIONS.md` ADR-017).

**F27. The steps that make dubbing publishable are mostly text and editorial work.** [confirmed] INFERENCE.
- The disclosed levers:
  - Context and gender-aware translation: HeyGen Precision; Sarvam's `speaker_gender`.
  - Glossaries and pronunciation lists: HeyGen's brand glossary (pronunciations included); Deepdub KNP; Rask dictionaries; Descript's do-not-translate list; Sarvam's Find & Replace.
  - Native-speaker review: ElevenLabs Productions, Dubverse, RWS, Chitralekha.
  - Regenerating segments: ElevenLabs, Descript, Sarvam, HeyGen, Synthesia.
- None of these needs a local model.
- On Telugu honorifics depending on the speakers' relationship: this is linguistic inference and was not source-checked.

**F28. What products do with background audio.** [confirmed] DISCLOSED.
- HeyGen always drops sound effects and keeps music.
- Chitralekha overlays a Spleeter accompaniment stem.
- **Added:** Sarvam's editor mixes Dubbed, Original and Background tracks and warns against a "sterile" fully muted background (N1).
- Rask's handling is undocumented.

## Refuted

1. **"Sarvam's commissioned test claims Bulbul beats ElevenLabs, contradicting the independent leaderboard."** Sarvam's own post says ElevenLabs v3 alpha leads on full-band audio quality (F3).
2. **"Synthesia Dubbing 2.0 was released 2026-07-15 (feisworld)."** The cited article is dated 2025-11-26 and gives no such date (F9).
3. **"Chitralekha's automatic path hard-trims overrunning speech, so it cuts sentences."** It compresses the whole line with uncapped atempo and trims only the residue (F15).
4. **"Maata does not yet budget each line's length at translation time."** It does, one-way, counted in code (F11).
5. **"If getAvailablePlaybackRates() lists only 0.75 and 1, a 0.9× video slow-down is impossible."** In Chromium the player accepted 0.9 although it listed only 0.25 steps (F10). WKWebView is untested.

## Unverifiable (do not rely on)

1. HeyGen API docs advising to disable Dynamic Duration only "for frame-exact timing".
2. HeyGen's Telugu support.
3. Papercup's "studio quality at one-fifth the cost".
4. Rask's Expressiveness, Similarity and Accent cloning dials.
5. Panjaya/TED "doubled completion rates".
6. The Variety Coolie details (paywall).
7. Pocket FM "started with ElevenLabs in 2024".
8. Gnani Vachana code-mixing.
9. Bulbul v3 normalisation "always on".
10. The encoder behind Sarvam Studio's 0.88 (ECAPA is assumed).

## Recommendations for Maata (updated for the corrections)

**R1. Keep the existing per-line akshara budget when translation moves to the Claude CLI, and close the short-line gap.** Impact high, effort S–M.
- **What.** Port `target_units`, `count_units` and the three-tier request unchanged into the Claude prompt. Counting stays in code, never done by the LLM. Then add what Descript does that Maata doesn't:
  - (a) Log a KPI: the share of lines whose required audio rate falls inside [0.9, 1.2]. Descript reached 73–83%. Today ADR-017 logs only mean and max rate, plus a 0.78 slot fill.
  - (b) Let the planner slow a short line's audio to about 0.9×, using the same mel interpolation, before leaving silence. Today the rate is in [1.0, 1.2] and `isochrony.min_audio_rate` is 0.95.
  - (c) Blind A/B a "match timing" prompt arm. It may use fuller phrasing (never new facts) for lines under about 0.8 of their slot. Compare it with today's no-padding rule, which exists to avoid padding.
- **Why.** Pace is the maintainer's top complaint and Descript's. Maata already has the condensing half; the evidence says the other half is where it differs.
- **Approval.** (b) changes the ADR-017 timing rules, so it needs an ADR. No new dependency.

**R2. Add an episode-brief pass.** Impact high, effort S.
- **What.** One Claude call per 5–10 minute window returns JSON:
  - a topic summary;
  - each speaker's role, gender and relationship, which sets Telugu address forms;
  - a glossary of names and terms, each marked keep-in-English or with a fixed Telugu form, plus pronunciation notes for TTS;
  - a register decision.
- Persist the glossary per channel.
- **Why.** It matches HeyGen Precision's context and gender awareness, HeyGen's glossary (with pronunciations), Deepdub KNP, Descript's do-not-translate list, Rask dictionaries, and Sarvam's `speaker_gender` and Find & Replace.

**R3. Add a Claude reviewer pass on top of the existing `lint()`.** Impact high, effort M.
- **What.** Review windows of about 10 lines against the English, looking for dropped clauses, skipped sentences, meaning errors, bookish words, forced English and length. Score each line 1–5 like Descript's judge; a line below 4 gets one re-translation.
- **Why.** It replaces the human linguist pass that every premium tier sells: ElevenLabs Productions, Dubverse, RWS, Chitralekha.
- **Watch.** Claude subscription rate limits against the 5–10 minute look-ahead.

**R4. Make register and script explicit, versioned inputs.** Impact medium, effort S.
- **What.** Name the style, for example Sarvam's "Urban colloquial" / modern-colloquial, and blind A/B it against a code-mixed arm.
- **Test English script for TTS.** Maata's prompt keeps English in Latin, but Sarvam's own Telugu sample writes loans in Telugu script (ప్లీజ్, పేమెంట్). A/B Latin against Telugu-script English as chatterbox-telugu input for intelligibility.
- **Why.** Sarvam's guide says a flat or overly formal dub should be fixed through style before blaming the voice model.

**R5. Reconsider "the video is never slowed down" (ADR-017) as an Adaptive/Original setting like Synthesia's.** Impact medium, effort S.
- **Why.** HeyGen (±20%, on by default) and Synthesia (Adaptive by default) both retime the video. Since the original audio is never played, a slow-down to about 0.9× is only visual.
- **Test first**, in the Tauri WKWebView on the M5 Pro, using `iframe_probe/index.html`:
  - does `setPlaybackRate(0.9)` read back as 0.9?
  - does `getCurrentTime()` then advance at about 0.9× wall time?
- Don't gate on the listed rates.
- **Approval.** Yes: it reverses an ADR-017 rule.

**R6. Add a "this line is wrong" hotkey.** Impact medium, effort M. UI only.
- It re-translates the line (through the reviewer), re-synthesises it ahead of the playhead, saves the fix to the channel glossary, and logs the flag.
- This is the edit-and-regenerate loop every product has: HeyGen Proofread, Synthesia Review Transcript, Sarvam per-block regenerate, ElevenLabs, Descript.

**R7. Keep "never trim audio" and the 1.2× cap.** Low effort. Document the reasons:
- Chitralekha's uncapped atempo shows what happens without a cap.
- Descript's listening test puts natural speed-up at up to 20%.
- Pair this with R1(b) for the slow-down side.

**R8. Background audio: give the maintainer the evidence.** Impact medium, effort M.
- **Evidence.** Industry practice is a separated background stem under the dub:
  - HeyGen keeps music and drops sound effects.
  - Chitralekha overlays a Spleeter accompaniment.
  - Sarvam's editor mixes a Background track and warns that muting it makes a dub "sterile".
- **If he relaxes the rule:** play only an accompaniment stem, ducked, never a stem that can leak speech.
- **Otherwise:** use synthetic room tone.
- **Approval.** Yes: it relaxes a hard constraint, and a separation model is a new dependency needing a licence check and pinning.

**R9. Use the AI4Bharat evaluation method.** Impact medium, effort S.
- Run blind pairwise tests scored with Bradley-Terry, rating intelligibility, expressiveness, liveliness and voice quality.
- Use SpeechArenaBench's Telugu native and code-mixed sentences (MIT, automatic gate) as a fixed TTS regression set.
- Compare speaker similarity only within one encoder. Sarvam's 0.88 (ECAPA, assumed) and CAMB's 0.87 (WavLM) are not comparable with Maata's WeSpeaker 0.466 and 0.645.

**R10. Items for the TTS sweep.** Impact medium, effort L.
- (a) Duration-conditioned TTS is Sarvam's long-term fix. The open options are licence-blocked:
  - IndexTTS2/2.5 use the custom bilibili licence, with a >100M MAU / RMB 1bn clause.
  - F5-TTS base weights are CC-BY-NC-4.0.
  - Corrected: IndicF5 is tagged MIT, but its data inheritance is an open question per `sota_ranking.md`, not a settled "non-commercial".
  - So R1 stays the practical route.
- (b) ai4bharat/Rasa (CC-BY-4.0, Telugu, expressive) is candidate data for adding expressiveness to the chatterbox-telugu fine-tune. Expressiveness is the top driver of Indic preference (F12), and flat emotion is the first Telugu complaint about suspected AI dubs (F24).
- **Approval.** The maintainer must accept the Rasa gate himself.

## Open questions

1. In the Tauri WKWebView on the M5 Pro, does `setPlaybackRate(0.9)` take effect with frames advancing at 0.9×, or is it silently rounded to 1?
2. Should Maata ever lengthen text for short lines (Descript's Match timing), or only slow audio to about 0.9× (R1b)? This needs a blind A/B with the maintainer.
3. Does chatterbox-telugu pronounce English better written in Latin or in Telugu script? Sarvam's docs and its own Telugu sample disagree.
4. Can the Claude subscription sustain per-line translation, the episode brief and the reviewer inside a 5–10 minute look-ahead? Throughput and rate limits are unmeasured.
5. No independent end-to-end evaluation of AI dubbing into Telugu exists. Is a Sarvam Dub output usable as an offline reference in a bake-off? It is a cloud product, so this needs the maintainer's ruling under "no cloud AI".
6. Sarvam's similarity numbers exist only as charts. Is the 0.88 same-language, cross-lingual or a blend, and which encoder produced it? There is no Telugu split.
7. HeyGen, Rask and Synthesia do not disclose their translation LLM or voice engine, and do not clearly list Telugu.
8. Background-audio policy (dry dub, room tone, or a separated music stem) is the maintainer's decision on a hard constraint.
