# 08 Timing and synchronization: verified report

Verified 2026-09-24. Scope: how Maata should place Telugu speech against a muted YouTube video it cannot alter, on the M5 Pro. This checks the researcher's sweep (`08-timing-sync.sweep.md`) against primary sources, the repo and the maintainer's own trace. It builds on `sota_ranking.md`.

How it was checked:
- **Trace:** I recomputed every trace number from `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl` and ran an energy VAD on the cached source audio.
- **Code:** I read the relevant files and ADRs.
- **Papers:** I re-read each one from its PDF text (arXiv, ISCA, ACL, UPM, JoSTrans, IJoC).
- **Vendor pages:** I re-fetched the vendor docs, the Help Center, the licences and PyPI metadata.
- **IFrame probe:** I re-ran it.

The session's WebSearch budget was already spent, so I used WebFetch, curl and the saved copies.

Labels:
- **DISCLOSED:** the company or authors state it.
- **MEASURED:** a number from a study or from our trace.
- **INFERRED:** reasoning, not evidence.

**Verdict counts, researcher's findings:** 14 confirmed, 8 corrected, 1 unverifiable, 1 refuted sub-claim. I also added 10 findings (A1–A10).

---

## Summary

- **Onsets are fixed; ends and long lines are not.** On the maintainer's podcast (latest session, 98 lines, 675 s):
  - The ADR-017 planner meets its onset goals: rate mean 1.017 (max 1.20), onset lag p95 about +0.22 s, no freezes.
  - Dubs end early: median −0.39 s, and a median −1.98 s on lines of 8 s or more.
  - The dub is silent for 8.2 s per minute while a source line is active. A new energy-VAD check finds that 6–7 s/min of that is real audio activity, not pauses inside lines.
  - Maata's segmenter computes intra-line anchors, but nothing uses them.
- **Speed cap 1.2x: keep.** It matches the ±20 % band in VideoDubber and in Microsoft's LSST. ElevenLabs' 0.7–1.2 speed range is a TTS voice setting, not a dubbing rule. Google's dubbing patent gives about ±10 % audio-rate changes as its worked examples.
- **Lag up to 0.6 s: keep.** It is in line with human dubbing practice: overlap 0.66, and dubbers keep their rate and let timing slip.
- **Early starts: make them cost more.** Viewers notice early audio about twice as fast as late audio. This is inference from same-language lip-sync studies.
- **Video slow-down: less promising than the sweep said.** The IFrame API docs (updated 2026-09-15) say an unsupported rate is rounded toward 1. I reproduced the researcher's probe: fine rates read back exactly, but only on a player that never started playing. The playout study also shows that talking-head content is the least tolerant of speed changes.
- **Four corrections that change recommendations:**
  - **Pause markers (Tam 2022) are not clearly the best method.** Once boundary relaxation was added, a length-controlled MT plus prosodic alignment matched or beat them.
  - **Federico 2020's background-audio gain was significant only for non-native listeners.** For native listeners the gain was not significant, and hard phrase alignment was rated significantly worse than baseline.
  - **The 500–600 ms pause peak is from read speech.** Spontaneous speech peaks near 430 ms.
  - **Google's patent does describe its fitting method** (the sweep said its numbers were unknown). It changes audio rate within per-language limits, inserts extra pauses between grouped lines, and falls back to changing the video rate.

---

## Findings (researcher's), with verdicts

### F1 [confirmed] On the maintainer's podcast, onsets meet ADR-017's targets, but dubs end early, especially on long lines. MEASURED.
- **Source:** trace `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`, the session opened 2026-09-24 20:06:51. It has 98 units covering 0–674.88 s. I recomputed everything.
- **Planner:**
  - Rate: mean 1.017, p90 1.073, max 1.20.
  - Onset lag: p50 0.00 s, p95 +0.22 s, min −0.204 s, max +0.62 s.
  - 0 freezes, 0 overdraft, 2 re-syntheses.
- **End error (dub end minus source end):**
  - Median −0.39 s, p25 −1.33 s.
  - p5 is −3.59 s interpolated or −4.42 s nearest-rank; the researcher used nearest-rank.
  - 44.9 % of lines end more than 0.5 s early, and 32.7 % more than 1 s early.
  - The 27 lines of 8 s or more end a median 1.98 s early (p10 −4.54 s interpolated, −4.68 s nearest-rank).
- **Played time:** `audio_s` is the natural, pre-rate duration (session.py:603), so played time is `audio_s / rate`. The dub plays for 0.870 of the source-span time.
- **Coverage:**
  - The dub is silent while a source span is active for 92.6 s, which is 8.2 s/min.
  - The dub plays over source silence for 0.92 s/min.
  - Mean per-line IoU is 0.844.
- **Spans:** p50 4.71 s, p90 12.14 s, max 19.58 s. The researcher's 4.66 s and 12.08 s differ only by percentile method.
- **Speaking rates:** with my own heuristic counter I get a median of 1.19 Telugu aksharas per English syllable, about 4.2 English syllables/s and about 5.7 TTS aksharas/s. These match the claimed 1.19, 4.27 and 5.61 within counter differences.
- **Earlier same-day session (13:44, ADR-016 build):**
  - Rate mean 1.136.
  - Audio/span p90 2.20 (claimed 2.17).
  - Aksharas per syllable 1.57 (claimed 1.59).
- **Caveat the researcher missed:** the 0.844 IoU is computed on unit spans, which include pauses, and not on VAD speech. It is therefore not comparable with Brannon's 0.658 human-dub figure, which is speech-level. Do not read "0.844 > 0.658" as better than human.

### F2 [confirmed] Intra-line anchors are computed but never used, and their threshold is too high. MEASURED (code).
- `engine/src/maata_engine/segment.py:523` defines `anchors()`, and line 68 sets `anchor_pause = 0.7`.
- No module in the engine calls `anchors()`.
- `SourceUnit` (`types.py:25`) has only `words` and no anchors field.
- On the threshold, see F17 for the corrected pause statistics.

### F3 [confirmed] Human dubbers keep their speaking rate steady and let timing slip. MEASURED.
- **Source:** Brannon, Virkar and Thompson, "Dubbing in Practice", TACL vol. 11, pp. 419–435 (2023); arXiv 2212.12137 v1, 2022-12-23.
- **Corpus:** 319.57 h from 54 titles.
- **Overlap (intersection over union of speech):** mean 0.658, median 0.731; 4.3 % of lines have zero overlap.
- **On-screen vs off-screen:** 0.684 vs 0.662 overall (+3.3 %). Excluding animation, 0.690 vs 0.656.
- **Rate variability:** the dub's speaking-rate SD is lower than the source's: Spanish 1.25 vs 1.47 w/s, German 1.26 vs 1.46 w/s.
- **What drives duration:** the duration ratio correlates with the content ratio (r = 0.523), not with dub rate (r = 0.163).
- **Conclusion:** the authors conclude that dubbers would rather break timing than vary their rate.
- **Isometry:** it is a weak proxy for overlap (r = 0.279).
- **Pauses:** dubs should respect perceptible pauses within a turn (citing Miggiani 2019).

### F4 [confirmed] Amazon lets on-screen boundaries move by fractions of 300 ms, and lets off-screen lines use whole gaps. MEASURED.
- **Source:** Virkar, Federico, Enyedi and Barra-Chicote, "Prosodic Alignment for Off-screen Automatic Dubbing", Interspeech 2022 (ISCA archive; arXiv 2204.02530, 2022-04-06).
- **Pause:** at least 300 ms of silence.
- **On-screen:** each boundary may shift by δ ∈ {0, ±¼, ±½, ±¾, ±1} × 300 ms.
- **Off-screen:** global dynamic-programming relaxation over the whole inter-phrase and inter-sentence gaps.
- **Off-screen rate score:** 1 if r ≤ 1; 2 − r for 1 < r ≤ 2; 0 if r > 2.
- **Results on YouTube vlog clips (on/off vs isochrone-only):** Wins +110 % to +297.1 %, Score +2.8 % to +22.5 %, mostly p < 0.01.

### F5 [corrected] Pause markers in the MT text beat MT followed by prosodic alignment, but they are not clearly the best method once relaxation is added. MEASURED.
- **Source:** Tam, Lakew, Virkar, Mathur and Federico, "Isochrony-Aware NMT for Automatic Dubbing", Interspeech 2022 (arXiv 2112.08548 v2, 2022-07-08).
- **Confirmed:** a pause is 300 ms of silence, and the paper concludes that injecting pause markers into the text is the best approach.
- **Correction, from its own dubbed-video human evaluation (Table 4; 40 native raters, 50 single-sentence videos):**
  - MT+[pause] beat MT+PA: German Wins +28.1 % (p < 0.01), French +3.5 % (not significant).
  - With Amazon's relaxation applied to both, D′ (Lakew's verbosity-controlled MT plus full PA) beat B′ (pause markers) on French (+28.2 %, p < 0.01) and tied on German.
  - On automatic Acceptability, the Lakew+PA cascade scored best.
- **Scope:** the evidence comes from fine-tuned NMT for En–De and En–Fr, not from prompting an LLM. Carrying `[pause]` through a Claude prompt is INFERRED to work and is untested.

### F6 [corrected] Mel resizing before the vocoder sounded better than time-stretching. Hard phrase alignment hurt native listeners, and the background-audio gain was significant only for non-natives. MEASURED.
- **Source:** Federico et al., "From Speech-to-Speech Translation to Automatic Dubbing", IWSLT 2020 (ACL Anthology 2020.iwslt-1.31; arXiv 2001.06785 v3, 2020-02-02).
- **Confirmed:**
  - Spline mel resizing is described as an empirical observation, with no table.
  - 657 MUSHRA ratings from 14 listeners (5 Italian, 9 non-Italian) on 24 clips.
  - Non-Italian listeners preferred C over A 66 % of the time vs B over A 52 %, a 27 % relative gain.
  - Italian listeners complained that the rate was too slow, too fast or too uneven.
- **Correction:**
  - Italian (native) listeners rated system B (better MT plus prosodic alignment) significantly worse than baseline: −10.93, the largest effect in the study.
  - For natives, adding background and reverb (B to C) was +1.05, not significant (27 % to 31 % preference, 15 % relative).
  - The background gain therefore rests on non-native listeners.

### F7 [confirmed] CosyVoice's speed control is linear mel interpolation before the HiFT vocoder, non-streaming only. DISCLOSED (code).
- **Source:** `FunAudioLLM/CosyVoice` main, `cosyvoice/cli/model.py` (fetched 2026-09-24).
  - `token2wav` in CosyVoiceModel, CosyVoice2Model and CosyVoice3Model calls `F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')`.
  - An assert limits speed changes to non-streaming mode.
- **Maata:** `backends/torch_common.py:264` `vocode()` does the same. It has no floor on rate; only the planner clamps to [1.0, 1.2].

### F8 [confirmed] ElevenLabs Dubbing Studio fixes each clip's duration by default and varies speed to fit, and publishes no lead or lag numbers. DISCLOSED.
- **Dubbing Studio docs** (fetched 2026-09-24, undated):
  - Fixed Generations are the default and can make speech speed up or slow down significantly.
  - Dynamic Generations may affect sync and timing.
  - Clips have handles, split and merge, and separate speaker tracks.
  - No numeric timing limits are stated.
- **Speed control** (Agents/TTS docs, fetched 2026-09-24): range 0.7–1.2, default 1.0; values outside that range are unsupported. This is a TTS voice setting, not a dubbing timing rule.
- **Dubbing overview** (fetched 2026-09-24): up to 32 speakers, detects overlapping speech, keeps the original background.
- **Conclusion:** no ElevenLabs document gives a 0.3 s early / 0.6 s late rule, so ADR-017's "ElevenLabs-style" numbers are Maata's own.

### F9 [corrected] YouTube refuses videos whose speech is too fast. Google's patent discloses an audio-rate-then-pauses-then-video-rate method with about 10 % example changes. DISCLOSED.
- **YouTube Help, "Use automatic dubbing"** (fetched 2026-09-24):
  - A video is ineligible if its speech is too fast for a listenable sped-up dub.
  - The maximum length is 120 minutes.
  - Telugu is a target for English originals.
  - Lip sync is experimental early access.
- **Expressive Speech:** English→Telugu is listed without the asterisk that marks it.
  - The asterisked targets for English originals are French, German, Hindi, Indonesian, Italian, Portuguese and Spanish.
  - The YouTube Blog (2026-02-04, Chandralekha Motati) gives 27 languages, with Expressive Speech in 8 including English.
- **Correction, the patent:** granted as US 11,582,527 B2 (2023-02-14; application 16/975,696, filed 2018-02-26; published as US 2020/0404386 A1 on 2020-12-24; freepatentsonline, fetched 2026-09-24). The sweep said its numbers were unknown. The published text gives:
  - Worked examples of about 10 % speed-up and slow-down (2.5 s to 2.2 s; 1.95 s to 2.2 s).
  - Configured minimum and maximum playback-speed thresholds, which may be language-specific. No values are given.
  - Grouping lines that are close in time and adding extra pauses between them, so speech need not be slowed much.
  - Changing the video rate (claim 11) via metadata that tells the client player when to speed up or slow down.
- **Caveat:** a patent shows what Google protected, not what YouTube ships (INFERRED).

### F10 [confirmed] Sarvam Dub builds duration control into generation and publishes no numbers. DISCLOSED (vendor claim).
- **Source:** https://www.sarvam.ai/blogs/sarvam-dub (2026-02-01).
- The target duration is set up front, and Sarvam says time-stretched speech sounds mechanical.
- It gives the example of a 2 s English phrase taking about 3 s in Hindi.
- It publishes no MOS or duration-accuracy numbers.
- Telugu is shown. The product is closed.

### F11 [confirmed] Early audio is detected at about half the offset of late audio. MEASURED (same-language lip sync).
- **ITU-R BT.1359:** detectability +45 ms (audio leads) / −125 ms (audio lags); acceptability +90 / −185 ms.
  - Secondary source: Randy Hoffner, TV Technology, 2003-05-14. The sweep said "unknown date".
- **Dixon & Spitz (1980), Perception 9:** a speech narrator was detected at about 131 ms audio lead vs 258 ms audio lag.
  - Source: Eg & Behne, Frontiers in Psychology 6, 2015-06-02, PMC4451240. They also restate the asymmetry.
  - The sweep's "51 ms/s drift" detail is unverifiable here; the source says only that the asynchrony was introduced gradually.
- **Applying it to Maata (INFERRED):** for a Telugu dub the words never match the lips, so only mouth onsets and offsets are cues.

### F12 [confirmed] Voice-over lets the original lead and lets the translation run off the ends, and voice-over viewers watch much like viewers of the original. MEASURED and DISCLOSED practice.
- **Sepielak, IJoC 10 (2016), pp. 1054–1073:**
  - Voice-over isochrony has the translation start after the original's onset and end earlier.
  - Luyken et al. (1991) let the original be heard for several seconds at the start.
  - Grigaravičiūtė & Gottlieb (1999) found Lithuanian voices running up to a couple of seconds past the Danish lines.
- **Flis, Sikorski and Szarkowska, JoSTrans 33 (2020), doi 10.26034/cm.jostrans.2020.548:**
  - 35 Polish viewers of voiced-over *Casablanca* did not avoid mouths.
  - Their gaze resembled that of English viewers of the original.
  - Romero-Fresco's (2016) Spanish dub viewers spent 95 % of the time on eyes and 5 % on mouths, vs 76 % and 24 % for English viewers.
- **For Maata:** this supports tolerance of lag for off-screen voices. There is no direct evidence for faces on screen (INFERRED).

### F13 [confirmed] Changing whole-clip speed by 10 % is measurably worse than reference, and slowing hurts more than speeding up. MEASURED.
- **Source:** Pérez, García and Villegas, "Subjective Assessment of Adaptive Media Playout for Video Streaming", QoMEX 2019, Berlin, 5–7 June 2019 (UPM record).
- **Design:** 50 viewers, 15 sources, 10 s clips with the speed changed after 4 s. Audio and video were changed together (ScaleTempo). Scored on the DCR scale.
- **MOS by speed:** 0.67 → 1.72; 0.80 → 2.84; 0.90 → 4.14; 1.00 → 4.64; 1.10 → 4.37; 1.25 → 3.45; 1.50 → 2.24.
- **Authors' conclusions:**
  - A maximum of 10 % is "much safer".
  - Even ±10 % differs significantly from the reference.
  - Any speed G > 1 rates better than 1/G.
- **Precision:** at ±10 % the average sits between imperceptible and perceptible-but-not-annoying, so "noticed on average" slightly overstates it.
- **Added (the sweep missed it):** content with people speaking was less resilient than animation or content without speech. That argues for caution with video slow-down on a podcast.

### F14 [corrected] In time-scale modification, speeding up rates better than the inverse slow-down, and WSOLA is preferred for voice when speeding up. MEASURED.
- **Source:** Roberts & Paliwal, "A Time-Scale Modification Dataset with Subjective Quality Labels", JASA 2020 (arXiv 2006.00848 v2, 2020-07-16).
- **Confirmed:**
  - 5,280 files at 10 ratios.
  - Of an inverse pair (for example 0.5 vs 2), the slower is rated lower.
  - Over all sources, WSOLA beats the phase vocoder for speed-up, and the phase vocoder beats WSOLA for slow-down.
  - WSOLA is preferred for voice, with minor differences.
- **Correction:**
  - For voice, the slow-down preference is "less clear". Methods rate similarly for 0.6 < β < 1, and the phase vocoder clearly wins only below 0.6.
  - The tested ratios nearest Maata's range are 0.78, 0.83, 0.996 and 1.38. None falls in 1.05–1.2, so applying the result there is extrapolation.
  - Maata's planner uses mel interpolation, not WSOLA; WSOLA only serves the user's own speed setting.

### F15 [confirmed] About ±20 % between dub and source duration is treated as the tolerable band. DISCLOSED metric and MEASURED.
- **VideoDubber** (Wu et al., AAAI 2023, arXiv 2211.16934, v1 2022-11-30):
  - Its user study made p = 0.2 the reasonable interval; 1.4x speed-up or 0.6x slow-down caused large inconsistency.
  - The study's own numbers are not reported.
- **Microsoft LSST** (Chadha, Subramanian et al., arXiv 2506.00740 v1, 2025-05-31):
  - Speech Rate Compliance uses a 20 % threshold.
  - Length-aware beam search produces short, normal and long variants in one pass, adding about 4.3 % latency.
  - SRC improves 16.3 % (ES) and 19.9 % (KO) relative; sync MOS improves 0.34 (ES).
  - It is a trained phoneme-level speech-translation model (ES/KO→EN), not a prompt.

### F16 [confirmed] Professional Indian-language read speech runs at 6–8 syllables/s. MEASURED.
- **Source:** Sathiyamoorthy et al. (IIT Madras), arXiv 2410.14197 (2024-10-18, submitted to ICASSP 2025).
- **Findings:**
  - It gives 6–8 syllables/s as standard for professional speakers.
  - Three Kannada male datasets measured 7.70 ± 0.95, 7.71 ± 1.14 and 8.32 ± 1.05 syllables/s.
- **Maata:** its TTS runs at about 5.6–5.7 aksharas/s, so 1.2x gives about 6.8. An akshara is not exactly a syllable, and no Telugu conversational figure was found.

### F17 [corrected] Silent pauses fall into brief, medium and long groups, but the medium peak in spontaneous speech is about 430 ms, not 500–600 ms. MEASURED.
- **Source:** Campione & Véronis, Speech Prosody 2002.
- **Confirmed:**
  - About 6,000 pauses.
  - Lognormal categories: brief (under 200 ms), medium (200–1000 ms), long (over 1000 ms, spontaneous speech only).
  - Pauses under 200 ms are hard to tell from occlusives.
- **Correction:**
  - The 500–600 ms medium peak comes from read speech (about 4.5 h in 5 languages).
  - Spontaneous speech (about 1 h, French only) peaks at about 78 ms, 426 ms and 1585 ms (long-pause weight 0.14).
  - For a podcast, the relevant medium peak is therefore about 0.43 s, which is even further below Maata's 0.7 s anchor threshold.
- **From `sota_ranking.md`, not re-checked:** Whisper DTW word timing has about 117–121 ms MAE.

### F18 [unverifiable] The embedded player accepts fine-grained rates such as 0.9 and 0.95. MEASURED, inconclusive.
- **Re-run:** I re-ran the probe (`iframe_probe/probe2.html`, timer-based sampling) in the Claude Browser pane (Chromium 152, macOS).
  - The pane was hidden, as it was for the researcher, and the player stayed unstarted (state −1). It never played.
  - `getAvailablePlaybackRates()` returned [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2].
  - `setPlaybackRate` with 0.9, 0.95, 1.1, 0.85, 0.75, 1.05 and 1.2 was read back exactly, and `onPlaybackRateChange` fired each time.
- **Docs conflict:** the IFrame API reference (last updated 2026-09-15) says a non-supported rate is rounded down to the nearest supported value toward 1. Under that rule 0.9 becomes 1.0.
- **Result:** readback on an unstarted player does not show what rate frames play at. Whether the rate is honoured while playing in WKWebView is unknown, so this needs S5 with a visible window.

### F19 [confirmed] Stalls cost viewers, as rebuffering does. MEASURED.
- **Source:** Krishnan & Sitaraman, "Video Stream Quality Impacts Viewer Behavior", ACM IMC 2012 (Boston, November 2012).
- **Data:** 23 M views from 6.7 M viewers. I decoded these figures from the PDF's substituted-font abstract.
- **Findings:**
  - Viewers start abandoning when startup takes more than 2 s, and each extra second adds 5.8 % abandonment.
  - A rebuffer equal to 1 % of the video's duration means 5 % less play time.
  - A failure makes a viewer 2.32 % less likely to revisit within a week.
- **For Maata (INFERRED):** its freeze budget of 1 s per 60 s would be a 1.7 % stall ratio if fully spent.

### F20 [confirmed] Rubber Band is GPL or commercial; Signalsmith Stretch is MIT. DISCLOSED.
- Rubber Band: GPL-2.0-or-later or a paid commercial licence (breakfastquay.com, fetched 2026-09-24).
- pyrubberband 0.4.0: ISC (2024-09-30), a wrapper around the command-line tool.
- Signalsmith Stretch: `LICENSE.txt` is MIT (© 2022 Geraint Luff / Signalsmith Audio).
- python-stretch 0.3.1: MIT (PyPI, 2025-02-14).
- audiotsm 0.1.2: MIT (2017-09-21).
- pytsmod 0.3.8: GPL-3.0.

### F21 [corrected] The only open autoregressive TTS with duration control uses a restrictive licence, and its `duration_factor` belongs to IndexTTS-2.5, not IndexTTS2. DISCLOSED.
- **Source:** the index-tts README and LICENSE (fetched 2026-09-24; the LICENSE is identical to the saved copy).
- **IndexTTS2 (2025-09-08):** claims precise duration control, but the feature is not enabled in that release.
- **IndexTTS-2.5 (2026-08-10):**
  - Adds `duration_factor` (0.5–2.0).
  - Supports Chinese, English, Japanese, Spanish and Arabic. It has no Telugu.
- **Licence, the bilibili Model Use License:**
  - An extra-licence clause applies above 100 M MAU or RMB 1 bn revenue.
  - It bans using the model to improve other AI models except its own derivatives or non-commercial ones.
  - It is not on Maata's approved list.
- **DubWise** (Sony Research India, Interspeech 2024, arXiv 2406.08802, 2024-06-13): adds a duration loss to an XTTS-based autoregressive TTS, which requires training.
- **Chatterbox:** `ChatterboxMultilingualTTS.generate()` has no duration or speed argument. I checked the installed package.

### F22 [corrected] WebKit reports `AudioContext.outputLatency` on Apple platforms, and Bluetooth adds about 0.18 s. DISCLOSED and MEASURED.
- **WebKit commit 420f89a0** (Jean-Yves Avenard, 2025-01-14): implemented for Cocoa platforms; other platforms return 0.
- **Added:** MDN browser-compat-data lists `outputLatency` from Safari 18.4.
- **jamieonkeys.dev (2022-07-01):** Firefox on macOS reported 0.178 s with AirPods.
- **Correction:** the post gives built-in speakers at 0.025 s (after first reporting 0). It contains no "wired 0.015 s" figure.
- **Maata:** `app/src/lib/sync.ts:252` already subtracts `outputLatency + baseLatency`.

### F23 [corrected] Maata's timing parameters fit the evidence, with refinements. INFERRED.
- **Speed cap 1.2:** supported by VideoDubber's ±20 %, LSST's 20 % and Virkar's penalty above 1.0.
  - The ElevenLabs 1.2 is a TTS setting, not a dubbing limit. ElevenLabs Fixed Generations may speed up "significantly", with no disclosed cap.
  - Google's patent uses about 10 % examples.
- **Lag 0.6 s and early start 0.3 s:** as in the sweep, with the asymmetry caveat (F11).
- **Video slow-down as a pre-freeze step:** weaker than the sweep said (F18, F13).
- **The real gap:** end alignment and alignment inside long lines, as the sweep said.

---

## Added findings (missed by the researcher)

- **A1 [confirmed] Google's patent groups nearby lines and inserts pauses before resorting to slowing speech or changing the video rate.** DISCLOSED (see F9 for the source). This is the same order Maata should use: fill with wording and pauses, keep rate changes small, and change the video only as a last step.
- **A2 [confirmed] The IFrame API documents a rounding rule toward 1 for unsupported rates.** DISCLOSED (reference last updated 2026-09-15). It contradicts the sweep's "no rounding rule" (see Refuted).
- **A3 [confirmed] Speech content is the least tolerant of playout-speed change.** MEASURED. Pérez et al. 2019 found animation and content without speech the most resilient, and rhythmic music the least. A podcast face is closer to the sensitive end.
- **A4 [confirmed] Isometric MT alone does not give isochrony.** MEASURED.
  - Source: Chronopoulou et al., "Jointly Optimizing Translations and Speech Timing…", arXiv 2302.12979 (2023-02-25).
  - Length-controlled MT fed to the same TTS was no more isometric than standard MT.
  - A joint model gained 55 % relative speech overlap.
  - For Maata: choose wordings by predicted TTS duration (the per-voice aksharas/s estimator), not by character count.
- **A5 [unverifiable in this pass] Non-uniform, per-phoneme duration scaling beats uniform scaling.** MEASURED. Source: Effendi, Virkar, Barra-Chicote and Federico, ICASSP 2022. I confirmed the authors and venue; the sister report 04-academic confirmed the win-rate table (for example EN→IT fast 40.1 vs 14.7). The method needs a TTS with explicit phoneme durations, which Chatterbox does not have.
- **A6 [confirmed] Maata's `vocode()` already handles rates below 1.0.** MEASURED (code). A mild audio slow-down therefore needs only a planner change, not new DSP (`backends/torch_common.py:264`).
- **A7 [confirmed] The researcher's upper-bound silence is mostly real audio.** MEASURED, low–medium confidence.
  - Method: an energy VAD (30 ms frames, three thresholds, 150 ms gap fill) on the first 680 s of `audio.m4a`.
  - Of the 92.7 s where a source line is active but no dub plays, 68–81 s has signal energy: 6.1–7.2 s/min.
  - Caveat: an energy VAD cannot tell speech from music.
- **A8 [confirmed] ElevenLabs Dubbing v2 lists Telugu (`te`).** DISCLOSED (dubbing overview, fetched 2026-09-24). YouTube's English→Telugu dubs are available but without Expressive Speech (F9).
- **A9 [confirmed] ADR-008 does not literally assume 0.25-step rates, and ADR-017 forbids video slow-down.**
  - ADR-008 says to use the nearest available IFrame rate at or above 0.85x.
  - phase-0 D7 says the player has a fixed set of rates.
  - ADR-017 says the video is never slowed. Any video-rate step needs an ADR change.
- **A10 [confirmed] ADR-018 removed the length note from normal lines.** Concise variants are requested only when a line is predicted to overrun. This explains why lines are never asked to fill their slot (see R2).

## Refuted

- **"The IFrame docs list no rounding rule."** They do: an unsupported `suggestedRate` is rounded down to the nearest supported value toward 1 (IFrame API reference, updated 2026-09-15).

---

## Recommendations for Maata (updated for the corrections)

1. **Align inside long lines at phrase level, softly.**
   - **Change:**
     - Store `anchors` on `SourceUnit`.
     - Lower `anchor_pause` from 0.7 s to about 0.3 s. Amazon's definition is 300 ms, and the spontaneous medium-pause peak is about 0.43 s.
     - Place each Telugu phrase near its English phrase onset with soft relaxation (±0.3 s on-screen, whole gaps when the face is off-screen or the line is long).
     - Fill leftover time with inserted silence, not per-phrase rate jumps, and keep the smoothness cost.
     - Ask Claude to keep `[pause]` markers at natural Telugu clause breaks, but don't depend on them: run the planner's relaxation either way.
   - **Why:** F1 and F2; Virkar (F4); Federico's native listeners punished uneven, hard-forced rate (F6); Tam's markers did not beat a length-controlled cascade once relaxation was added (F5); Google's patent inserts pauses (A1).
   - Impact high, effort M.
   - **Approval:** it amends ADR-017's timing design, so record it as an ADR the maintainer sees.
2. **Size each line to fill its slot, and choose by predicted duration.**
   - **Change:**
     - Restore a length target on every line: slot × the voice's measured aksharas/s × about 0.95–1.0, as a target, not a quota.
     - Ask for short, normal and long variants in one `claude -p` call.
     - Choose with `planner.evaluate()` using the duration estimator, not character counts.
   - **Why:** played time is 0.87 of source time (F1); ADR-018 dropped the hint (A10); LSST's one-pass variants (F15); isometric MT alone doesn't help (A4).
   - Impact high, effort S. No approval needed; log it in ADR-018.
3. **Measure what the maintainer hears.**
   - **Change:** add these to `maata-bench`, `verify-mac.sh` and ADR-017's reopen criteria:
     - the end-error distribution;
     - speech-level overlap (VAD or aligner, not unit spans);
     - seconds per minute of source speech with no dub;
     - anchor onset error.
   - **Why:** F1 and A7. The current criteria check onsets and freezes only.
   - Impact high, effort S. No approval needed.
4. **Make the onset cost asymmetric.**
   - **Change:** split `w_lag` into `w_early` (about 2x) and `w_late`. Keep `lead_max` at 0.3 s, and spend it only into real silence.
   - **Why:** F11. The thresholds come from same-language lip sync, so applying them here is an inference.
   - Impact medium, effort S. No approval needed.
5. **Allow a mild audio slow-down (to about 0.92x), after trying pause insertion and a longer wording.**
   - **Change:** apply it to lines predicted to end more than 0.5 s early. `vocode()` already supports it (A6).
   - **Why:** F13 and F14; Google's patent uses about 10 % examples and prefers adding pauses first (F9, A1).
   - Impact medium, effort S.
   - **Approval:** yes, because ADR-017 floors the rate at 1.0. Confirm with a blind listening A/B against pause insertion alone.
6. **Video slow-down: move it to "investigate only".**
   - **Change:** measure in S5, in the Tauri WKWebView with the window visible and the video playing. Record `getCurrentTime()` slope against wall time at 0.9 and 0.95 set via `setPlaybackRate`, and check whether `onPlaybackRateChange` reports 1.0.
   - **Why:** the docs say unsupported rates round toward 1 (A2); the probe proved nothing while unstarted (F18); talking heads are the least tolerant (A3).
   - Impact low–medium, effort M.
   - **Approval:** yes, because ADR-017 forbids video slow-down.
7. **Keep the 1.2x cap, and add an absolute ceiling of about 7.5 aksharas/s.**
   - **Change:** limit rate to min(1.2, 7.5 / voice rate), and keep `w_smooth`.
   - **Why:** F15 and F16. The 7.5 figure is INFERRED from Kannada read-speech rates, so check it against native Telugu podcasts.
   - Impact low, effort S. No approval needed.
8. **Find pauses from accurate timings.**
   - **Change:** use a forced aligner (MFA or Qwen3-ForcedAligner, per `sota_ranking.md`) or a VAD on the vocal stem, with a 300 ms threshold, instead of Whisper DTW gaps (about 120 ms MAE).
   - Impact medium, effort M.
   - **Approval:** yes, for the same dependencies listed in `sota_ranking.md`.
9. **Stretch method: keep in-vocoder mel interpolation. Don't ship Rubber Band (GPL).**
   - **Change:** if a classic-stretch A/B is ever wanted, use Signalsmith Stretch via python-stretch (MIT).
   - Impact low, effort S.
   - **Approval:** only if python-stretch is added.
10. **Calibrate output latency.**
    - **Change:**
      - Keep `outputLatency` (reported on macOS from Safari 18.4 / WebKit 2025-01).
      - Add a per-output-device user offset; Bluetooth is about 0.18 s.
      - Add an explicit offset for Linux WebKitGTK, which returns 0.
    - Impact medium, effort S. No approval needed.
11. **Batch Claude CLI translation with look-ahead.**
    - **Change:**
      - Send several consecutive lines with their pause markers per `claude -p` call.
      - Keep the translator at least 60 s ahead.
      - Measure per-call latency before changing ADR-016's prepare-ahead formula.
      - Add a start-up watchdog.
    - **Why:** startup delay drives abandonment (F19).
    - Impact medium, effort M. No approval needed; it uses his quota.
12. **Background audio: present it as an open decision with weaker evidence.**
    - The only controlled gain was among non-native listeners (F6).
    - ElevenLabs keeps the background by default (F8). Google's patent includes erasing the voice while keeping music and background (sister report 02).
    - Cheapest first step within the current rule: synthetic room tone. Playing a separated stem needs the maintainer to relax the hard rule and to approve a separator model.

---

## Open questions

1. Does the embedded player actually play at 0.9–0.95x in WKWebView, or does it round to 1.0 as documented? What are `getCurrentTime()`'s update cadence and jitter there (S5)?
2. How noticeable is slowing a muted video of a talking face while the dub audio stays at normal speed? Every playout study changed audio and video together.
3. How much early or late onset do viewers tolerate when a face is on screen but the dub is in another language?
4. What is the natural conversational Telugu rate in aksharas/s? It could be measured from native Telugu podcasts.
5. How much of the 6–7 s/min of uncovered source audio is speech rather than music? This needs a speech VAD or aligner.
6. Are early endings caused by dropped content (the "incomplete sentences" complaint) or by compact but faithful Telugu? Aksharas per syllable fell from 1.57 to 1.19 between the day's sessions.
7. Do `[pause]` markers survive a Claude prompt reliably, and does adding them hurt Telugu naturalness? This needs a blind test.
8. What minimum and maximum speed thresholds does YouTube actually configure for Telugu? The patent gives none.
9. What is the per-call latency of `claude -p` on the M5 Pro, and how many lines per call keep quality?
10. Does Chatterbox output at 0.92x via mel interpolation sound natural against pause insertion alone?

## Sources (date = published or fetched)

- Maata trace `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl` and `audio.m4a` (session 2026-09-24 20:06), analysed 2026-09-24.
- Maata repo files, read 2026-09-24:
  - `engine/src/maata_engine/segment.py`, `types.py`, `session.py`, `timing/planner.py`, `backends/torch_common.py`
  - `app/src/lib/sync.ts`, `app/src/lib/player.ts`
  - `docs/DECISIONS.md` (ADR-008, ADR-017, ADR-018), `docs/plans/phase-0.md`
- Brannon, Virkar, Thompson, TACL 11:419–435 (2023), https://arxiv.org/abs/2212.12137 (v1 2022-12-23).
- Virkar et al., Interspeech 2022, https://arxiv.org/abs/2204.02530 (2022-04-06).
- Tam et al., Interspeech 2022, https://arxiv.org/abs/2112.08548 (v2 2022-07-08).
- Federico et al., IWSLT 2020, https://arxiv.org/abs/2001.06785 (v3 2020-02-02); https://aclanthology.org/2020.iwslt-1.31/
- Chronopoulou et al., https://arxiv.org/abs/2302.12979 (2023-02-25).
- Effendi et al., ICASSP 2022, https://www.amazon.science/publications/duration-modeling-of-neural-tts-for-automatic-dubbing (fetched 2026-09-24).
- Wu et al., VideoDubber, AAAI 2023, https://arxiv.org/abs/2211.16934 (v1 2022-11-30).
- Chadha, Subramanian et al. (Microsoft), https://arxiv.org/abs/2506.00740 (2025-05-31).
- Roberts & Paliwal, JASA 2020, https://arxiv.org/abs/2006.00848 (v2 2020-07-16).
- Pérez, García, Villegas, QoMEX 2019, https://oa.upm.es/64246/ (conference 2019-06-05 to 07).
- Sathiyamoorthy et al., https://arxiv.org/abs/2410.14197 (2024-10-18).
- Campione & Véronis, Speech Prosody 2002, https://www.isca-archive.org/speechprosody_2002/campione02_speechprosody.pdf (2002-04).
- Krishnan & Sitaraman, ACM IMC 2012, https://people.cs.umass.edu/~ramesh/Site/HOME_files/imc208-krishnan.pdf (2012-11).
- Hoffner, "A/V Synchronization: How Bad Is Bad?", TV Technology, https://www.tvtechnology.com/opinions/av-synchronization-how-bad-is-bad (2003-05-14).
- Eg & Behne, Front. Psychol. 6, https://pmc.ncbi.nlm.nih.gov/articles/PMC4451240/ (2015-06-02); Dixon & Spitz, Perception 9 (1980), doi 10.1068/p090719.
- Sepielak, IJoC 10:1054–1073 (2016), https://ijoc.org/index.php/ijoc/article/download/3559/1578/0
- Flis, Sikorski, Szarkowska, JoSTrans 33 (2020), https://doi.org/10.26034/cm.jostrans.2020.548
- ElevenLabs docs, fetched 2026-09-24, undated:
  - Dubbing Studio: https://elevenlabs.io/docs/eleven-creative/products/dubbing/dubbing-studio
  - Speed control: https://elevenlabs.io/docs/eleven-agents/customization/voice/speed-control
  - Dubbing overview: https://elevenlabs.io/docs/overview/capabilities/dubbing
- YouTube Help, "Use automatic dubbing", https://support.google.com/youtube/answer/15569972 (fetched 2026-09-24).
- YouTube Blog, https://blog.youtube/news-and-events/youtube-auto-dubbing-expressive-speech/ (2026-02-04).
- Google patent US 2020/0404386 A1 (2020-12-24), granted as US 11,582,527 B2 (2023-02-14): https://www.freepatentsonline.com/y2020/0404386.html (fetched 2026-09-24).
- YouTube IFrame Player API reference, https://developers.google.com/youtube/iframe_api_reference (last updated 2026-09-15).
- IFrame probe re-run: `dub_research/iframe_probe/probe2.html`, Chromium 152 (2026-09-24).
- Sarvam Dub, https://www.sarvam.ai/blogs/sarvam-dub (2026-02-01).
- CosyVoice `cosyvoice/cli/model.py`, https://github.com/FunAudioLLM/CosyVoice (main, fetched 2026-09-24).
- Rubber Band licence, https://breakfastquay.com/rubberband/license.html (fetched 2026-09-24).
- Signalsmith Stretch, https://github.com/Signalsmith-Audio/signalsmith-stretch (LICENSE.txt, fetched 2026-09-24).
- PyPI (fetched 2026-09-24): python-stretch 0.3.1 (2025-02-14), audiotsm 0.1.2 (2017-09-21), pytsmod 0.3.8, pyrubberband 0.4.0 (2024-09-30).
- IndexTTS README and LICENSE, https://github.com/index-tts/index-tts (fetched 2026-09-24).
- DubWise, Interspeech 2024, https://arxiv.org/abs/2406.08802 (2024-06-13).
- WebKit commit 420f89a0, https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg224878.html (2025-01-14).
- MDN browser-compat-data `api/AudioContext.json` (fetched 2026-09-24).
- jamieonkeys.dev, https://www.jamieonkeys.dev/posts/web-audio-api-output-latency/ (2022-07-01).
- Sister reports in this folder: `sota_ranking.md` (Whisper timing MAE), `01-elevenlabs.md` / `02-youtube-google-meta.md` / `04-academic.md` drafts (patent grant and mixing options, Effendi table).
