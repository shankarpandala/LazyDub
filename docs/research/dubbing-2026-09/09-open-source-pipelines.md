# 09 Open-source dubbing pipelines: verified report

Verified 2026-09-24. I re-checked the researcher's report against primary sources myself:
- the code of all 16 repos, fetched through the GitHub API at the pinned commits;
- the GitHub issues and PRs;
- the Chatterbox source;
- the YouTube IFrame API reference;
- the Maata repo (read-only);
- one live browser measurement.

Source files are cached under `scratchpad/v09/`, and issue text under `scratchpad/v09/issues/`. The web-search budget was already used up, so I verified through the GitHub API, the Hugging Face API, WebFetch of official docs, and the browser.

Verdict markers:
- **[confirmed]** matches the primary source.
- **[corrected]** is partly wrong; the correct fact and its source are given.
- **[unverifiable]** couldn't be confirmed (for example, one-off self-reported measurements).
- Refuted claims are listed at the end.
- **[new]** marks a finding the researcher missed. I checked each one against the source cited.

## Summary

The researcher's reading of the open-source pipelines is mostly accurate: 27 claims confirmed, 6 corrected, 2 refuted, 4 unverifiable. Four corrections change what Maata should do.

1. **The IFrame player does accept fine playback rates.** The official reference says unsupported rates are rounded toward 1, and `getAvailablePlaybackRates()` lists only 0.25 steps. But in a live test on 2026-09-24 (Chromium 152, youtube-nocookie embed), `setPlaybackRate(0.85, 0.9, 0.95, 1.1, 1.15, 1.3)` was accepted and read back exactly, and the `<video>` element's `playbackRate` became 0.9. So gentle per-line video slow-down (0.90–0.95) is probably possible. It still has to be confirmed in the Tauri WKWebView.
2. **Maata already builds its voice references the way the report recommends.** ADR-017 (`session._build_voice`) takes the timbre from the speaker's best single clean clip (≤10 s), averages the speaker embedding over up to 60 s of clean clips, and prompts T3 with that clip. This moved guest similarity from 0.400 to 0.466. The guest SIM of 0.466 the report calls a gap is the number *after* this change. The recommendation to "average the voice-encoder embedding over 30–60 s" is therefore not new.
3. **"Keep full bandwidth / no 16 kHz step before S3Gen" does not apply to Chatterbox.** S3Gen's reference mel has `fmax = 8000` Hz, and its tokenizer, voice encoder and x-vector all run at 16 kHz. A clean 16 kHz reference loses essentially nothing. Only a lossy, low-bitrate step that cuts below about 8 kHz would matter, like VideoLingo's 32 kbps MP3.
4. **Most speed caps in the open-source pipelines are soft or absent.** VideoLingo's max of 1.4 only triggers a text trim, and chunk speed is uncapped. KrillinAI's SpeedMax of 1.30 only logs a warning. pyVideoTrans defaults to 100. So the pipelines do not "converge on a 1.2× cap". The 1.2× figure is VoiceStudio's audio-only threshold and VideoLingo's "accept" level. This strengthens the report's main diagnosis: uncapped compression causes the "suddenly very fast" complaints.

The recommendations that survive:
- scene-batched, id-keyed Claude calls that return length tiers, plus a per-video brief;
- underrun fill;
- a round-trip ASR check with a Telugu-capable ASR. Maata's Whisper can't do it: FLEURS CER is 82.65 zero-shot, per the BuzzASR card. MIT candidates exist.
- a separation stem only with a permissively licensed weight. The BS-RoFormer ep_317 weight has no declared licence; Demucs is MIT.

## Verified findings

### A. Translation passes and segmentation

**A1 [confirmed] VideoLingo runs three LLM stages with neighbouring context and fails loudly.**
- Stages: summary and terms (a 2-sentence theme, fewer than 15 terms), then a faithful pass ("direct"), then an expressiveness pass ("reflect" + "free", check conciseness, casual language for tutorials).
- Context: each chunk sees the last 3 lines of the previous chunk and the first 2 of the next (`_4_2_translate.py`). Chunks hold at most 600 characters or 10 lines (L57).
- Line-count or key mismatch: 3 retries, then `ValueError`; it never pads (`translate_lines.py` L26-40).
- Note: the reflect pass is gated by config (`reflect_translate: true` by default).
- VoiceStudio re-implements the same pattern (`translation_quality.py`):
  - a THEME plus up to 30 TERM lines, cached by a sha256 transcript hash;
  - a critique pass, then a polish pass ("same length or shorter", "never add content", never switch language), at temperature 0.2.
- Sources: VideoLingo `core/prompts.py`, `translate_lines.py`, `_4_2_translate.py` @9bc30202 (2026-09-23); VoiceStudio `backend/services/translation_quality.py` @324eb9a6 (2026-09-24).

**A2 [corrected] What VideoLingo translates, and what its "align" step does.**
- Confirmed:
  - spaCy splitting, then an LLM split of sentences over 20 tokens into ceil(n/20) parts, choosing between 2 alternatives (`_3_2_split_meaning.py` L83-108);
  - `max_split_length: 20`, with the config comment "below 18 … above 22" (`config.yaml` L61-62);
  - the align prompt may rewrite (`prompts.py` L276).
- Correction: the report says translation "is done at sentence level and never fragment by fragment". In fact:
  - The translation *unit* is each LLM-split piece of up to about 20 tokens, so a long sentence becomes several pieces. The pieces are translated as numbered lines inside a chunk of up to 10 lines, with neighbouring context (`_4_2_translate.py` reads `split_by_meaning.txt`).
  - The align step (`_5_split_sub.py`) splits lines that are too long for *display subtitles*.
  - The "remerged" output of that step (`_5_REMERGED`) is what feeds the dub (`_6_gen_sub.py` L160-163).
- The #202 claims are confirmed:
  - the maintainer (2024-10-27) said only Claude could reliably reason about and output sensible splits for complex scenes;
  - a user (2024-10-28) reported frequent JSON errors with local qwen2.5 32B, and 72B basically fine. This is anecdotal and about 2024 models.
- Sources: VideoLingo `_3_2_split_meaning.py`, `_5_split_sub.py`, `_6_gen_sub.py`, `core/utils/models.py` @9bc30202; issue #202 (opened 2024-10-27).

**A3 [confirmed] pyVideoTrans enforces block-local ("zero-shift") translation, and its Telugu add-on asks for spoken Telugu.**
- `videotrans/prompts/srt/chatgpt.txt` says: NEVER MERGE; ZERO-SHIFT FRAGMENT TRANSLATION; translate only the words physically in each block; ellipsis bridging; keep the syllable/character count proportional to the duration; write for the ear.
- `language_prompts/te.txt` (162 B) asks for natural spoken Telugu (వాడుక భాష), dropped subject pronouns and spoken contractions.
- `hi.txt` admits the conflict: it uses flexible word order to avoid moving verbs or nouns between blocks.
- Inference (unchanged): block-local translation works against verb-final Telugu; Maata should not adopt it.
- Source: pyVideoTrans @d63d27ae (2026-09-23).

**A4 [confirmed] Linly-Dubbing translates line by line with a rolling history, and uses a CJK-only validator.**
- Line-by-line translation with `history[-30:]` (15 pairs) (`step030_translation.py` L265-313).
- The validator rejects output longer than 0.75× the source or containing words like "translate" (L55-94).
- The summary-input part is corrected in C4.
- Source: Linly-Dubbing @5677191e (2025-03-05, dormant).

**A5 [confirmed] YouDub-webui runs a preprocess pass and returns an `audio_mode` per line.**
- The preprocess pass over metadata and the full transcript returns:
  - a 3–5 sentence summary;
  - hotwords (GPU/API/Transformer kept as-is);
  - only high-confidence ASR corrections.
- Each sentence comes back as JSON `{"dst", "audio_mode": "tts"|"original"}`. Wordless laughs, screams and so on keep the original audio.
- An optional content-only rule drops fillers that carry no information.
- Source: `backend/app/adapters/_translate_prompts.py` @0f6c7593 (2026-09-21), Apache-2.0.

**A6 [confirmed] mazinger gives translation visual and metadata context.**
- An LLM picks thumbnail timestamps, and ffmpeg extracts those frames.
- A "Describe" pass produces the content analysis.
- Optional ASR review in batches of 30 with 6 overlapping.
- Translation in batches of 24 with 8 overlapping.
- Word budget: duration × 2.0 words/s × 0.80.
- LLM re-segmentation: merge fragments, split entries over 84 characters or 4.0 s.
- Whisper `initial_prompt` built from YouTube metadata.
- Source: `docs/pipeline.md` @e7595760 (2026-09-05), MIT.

**A7 [confirmed] KrillinAI validates each translation and retries.**
- 3 retries. It rejects: empty output; output identical to the source (unless proper nouns or numbers); the wrong script (checks only CJK/JP/EN; other languages pass); a length ratio outside 0.3–3×.
- Each retry adds the failed attempt to the prompt.
- Context is 3 sentences before and 3 after.
- Source: `runtime/krillinai/internal/service/translate.go` L228-310 @f96cdf19 (2026-09-24).

**A8 [confirmed] SoniTranslate batch translation falls back to line by line on a count mismatch.**
- Lines are joined with ` ||||| ` into chunks of up to 2,000 characters. If the returned count differs, it translates line by line (Google-translator batch path).
- Source: `soni_translate/translate_segments.py` L160-190 @70b6390a (2026-08-29).

**A9 [confirmed] Ariel asks Gemini for per-utterance speaking and translation instructions.**
- One Gemini call on the whole video (GCS URI) at temperature 0.2 returns:
  - speakers with name, gender and a Gemini-TTS voice;
  - the transcript with timestamps;
  - per-utterance `speaking_instructions` (tone, pacing, rhythm, pitch, …) and `translation_instructions` (formality, jargon, cultural context);
  - the translation.
- The editor colours overlapping utterances and accepts per-line voice and translate instructions.
- It is cloud-only, so it serves Maata as a design idea only.
- Sources: `transcribe.py`, `docs/user_manual.md` @28ec8a92 (2026-09-23), Apache-2.0.

### B. Timing and fitting

**B1 [confirmed] pyVideoTrans absorbs the gap before speeding up, and its default audio cap is effectively none.**
- Each line's end is moved to the next line's start (`_rate.py` L350-356). The doc's example drops from 1.75× to 1.4× (`docs/Synchronize.md` L182-186).
- `BOTH_MODE_AUDIO_ONLY_THRESHOLD = 1.2` (L221). Above it, audio speed-up and video slow-down each take half the difference (L418-431).
- `max_audio_speed_rate` defaults to 100 (L260). `max_video_pts_rate` is 10.
- Rubber Band is preferred and atempo is the fallback.
- `_concat_audio_aligned` lets an overflow push the timeline (L579-605).
- #869 (2025-08-06): the maintainer says v3.74 capped speed at 3×, the cap was removed because it caused A/V desync, and he advises setting `max_audio_speed_rate` back to 3.
- #1190 (2026-08-22): quality "drops sharply" when accelerating.
- Source: pyVideoTrans @d63d27ae (2026-09-23), GPL-3.0.

**B2 [confirmed] SoniTranslate stretches each segment against its own slot.**
- The ratio is TTS length ÷ slot.
- Optional regulation, only when the ratio is 1.3 or more: borrow 50% of the gap if the next speaker is the same, 70% otherwise, with a floor of 1.2.
- Then: capped at `max_accelerate_audio`; ratios from 0.8 to 1.15 snap to 1.0; 0.79 or less becomes atempo 0.8. That last case is a *slow-down fill* the report didn't point out.
- Defaults (`app_rvc.py` L406-414): max_accelerate 2.1, original audio at 0.25, dub at 1.80, avoid_overlap False.
- #19 (2024-02-17): a user heard speech "like x10". The maintainer blamed imprecise Whisper timestamps and suggested lowering the then-default 1.9 to 1.5 or 1.0.
- Source: `text_to_speech.py` L1089-1206 @70b6390a.

**B3 [confirmed] open-dubbing trims each clip's tail silence and prefers native re-synthesis.**
- It removes trailing silence from each TTS clip.
- Speed = ceil(dub/slot × 10)/10, measured against the start of the next utterance.
- Hard `MAX_SPEED = 1.3`.
- It re-synthesizes at that speed when the voice supports it, otherwise uses FFmpeg.
- `--update` re-dubs from edited metadata.
- Sources: `open_dubbing/text_to_speech.py` L158-375, `DOCUMENTATION.md` @bff534d3 (2025-07-08, dormant).

**B4 [confirmed] KrillinAI writes a machine-readable dubbing plan and report.**
- `Report` holds warnings, failed_indexes, max_speed_factor and rewrite_count.
- It rewrites a line once with the LLM when the estimate exceeds the cue plus 1.5 s. The prompt asks for more natural, shorter, voice-over-friendly wording, with no new facts, fitting %.1f s.
- The estimator has no Telugu profile, so Telugu uses a heuristic: runes/8.5 + 0.12 s per word.
- `Calibrate()` (an EMA of 0.7 old and 0.3 of a target clamped to 0.5–1.5) is defined but never called in the dubbing package.
- #176 (2025-04-27): speech too fast. A 2025-04-30 reply says a better duration solution is in development. A 2025-07-07 user says only slowing the video, or both, feels better.
- #109 (2025-04-12): same topic.
- Source: `dubbing/*.go` @f96cdf19.

**B5 [confirmed] VoiceStudio Smart Fit.**
- It is a clean-room reimplementation "from a published description".
- FitParams: audio-only up to 1.2; audio cap 1.5; video cap 2.0; gap guard 0.05 s; `min_audio_rate` 0.85; `UNDERRUN_TOLERANCE` 0.95; hard audio cap 1.8.
- Beyond 1.2×, audio = min(√need, 1.5).
- Source: `backend/services/fit_planner.py` @324eb9a6 (2026-09-24), AGPL-3.0.

**B6 [unverifiable] VoiceStudio's own measurements.**
- "8.8 s of holes across 18.7 s of speech", and separated background residue at about 37% of the original's energy.
- Both appear only in code comments, from one run. I could not reproduce them.

**B7 [confirmed] VoiceStudio predicts fit before synthesis and reports overflows.**
- The duration planner:
  - calibrates on the median characters per second of this job's clips (needs at least 3 samples, each at least 0.4 s and 4 characters);
  - lets a line borrow at most 3.0 s from the gap;
  - labels each line fits, tight or impossible, and sends impossible lines to `condense_for_slot`.
- `dub_timing_overflow` raises an explicit "Speech exceeds the fitting limits…" error instead of cutting audio.
- The default strategy is `concise`, and the default `voice_match` is `per_line`.
- `dub_qc.py` scores drift at 0.5 and keeps the generated text authoritative.
- Sources: `duration_planner.py`, `dub_qc.py`, `api/routers/dub_generate.py` L695-699 and L1790 @324eb9a6.

**B8 [confirmed] youtube-auto-dub makes translation length-aware.**
- NLLB returns 6 beams / 6 candidates. It keeps the longest candidate within window × cps (default 15), otherwise the shortest.
- Sync: trims at -40 dB, speed-up capped at 1.4, slow-down floored at 0.85 when the clip is under 85% of its window, and overlays each clip at its original start (no ripple).
- Sources: `src/ytdub/stages/translate/nllb.py`, `stages/synchronize.py`, README @631ee274 (2026-09-05), MIT.

**B9 [unverifiable] youtube-auto-dub's self-reported gains.**
- Over-compressed segments fell from about 86% to 56%, on one EN→IT clip.
- Segments needing stretch fell from 6/6 to 1/4, on one 32 s clip.
- The self-test reports 0.00 s drift and 0.888 cosine, using XTTS IT→EN on a generated clip.
- All are single-clip self-reports.

**B10 [confirmed] YouDub-webui sets one global pace for the whole dub.**
- Base factor = sum(desired)/sum(TTS) × 0.99, clamped to [0.8, 1.2]. A per-line factor is then clamped to [0.9, 1.1].
- Start = max(original start, end of the previous clip).
- Source: `backend/app/adapters/audio.py` @0f6c7593.

**B11 [confirmed] ThioJoe's tool supports two-pass synthesis.**
- `two_pass_voice_synth` (off by default) re-synthesizes through the cloud TTS API's own rate control.
- `combine_subtitles_max_chars = 200`; `speech_rate_goal = Auto`.
- The ffmpeg-vs-Rubber Band preference is the author's anecdote.
- Source: `config.ini` @8360469f (2026-05-11), AGPL-3.0.

**B12 [confirmed] A Claude Code dubbing skill makes punctuation re-segmentation mandatory and fills underruns in order.**
- Re-segmenting at punctuation is mandatory, with 3–8 s per cue: under about 1.5 s is choppy, over about 10 s usually hides a missed break.
- Underrun levers, in order:
  1. native TTS rate at -12%;
  2. `atempo = max(0.82, tts/target)` when slack is over 0.5 s;
  3. expand the text in the worst cues.
- Sample 3–4 voice and rate combinations before the full run.
- The original audio is kept as a bed at 0.15–0.25 (default 0.18).
- Source: jianshuo/claude-skills @b2690f5b (2026-08-20), MIT.

### C. Where skipped or incomplete lines come from

**C1 [confirmed] Silent failure paths in open-source dubbers.**
- **VideoLingo** (`tts_main.py` L27-33, L68-80):
  - empty or 1-character text becomes 100 ms of silence;
  - a zero-length result after 3 tries also becomes silence;
  - an exception on the third try *raises* instead.
- **pyVideoTrans**:
  - in plain-text mode, lines the LLM didn't return are padded with "" (`translator/_base.py` L114-118);
  - SRT mode doesn't check counts at all.
- **ViDubb** (`inference.py` L505-541):
  - text cut to 50 words plus "…";
  - XTTS run at speed=2, then 350 ms cut from every clip;
  - a line with θ < 0.44 (needing more than 2.27× speed) becomes silence;
  - an ffmpeg failure also becomes silence.
- **Voice-Pro**: a failed line becomes silence for its slot, and an overrun pushes the next line (`abus_tts_f5.py` L178-216).
- **Ariel**: empty TTS returns 0.0 (`generate_audio.py` L99-115).
- **SoniTranslate #127** (2024-11-21): output incomplete with no error.
- Sources: repos at the pinned commits above.

**C2 [new] Two more silent-fallback paths.**
- pyVideoTrans replaces a missing TTS file with a silent placeholder for the whole slot (`_rate.py` L357-363 and L588-592).
- KrillinAI uses the *source-language text* as the target when both batch and single translation fail (`translate.go` L615-620). The line would then be dubbed in English.

**C3 [confirmed] Energy-gate trimming eats final syllables.**
- pyVideoTrans #1012 (2026-04-17): the last 2–3 characters are swallowed.
- Contributor zwyin (2026-05-19) traced it to `remove_silence_wav()`: -40 dB threshold, 180 ms tail padding, on by default through `remove_dubb_silence`. Suggested fix: 300 ms of padding and a -50 dB threshold.
- youtube-auto-dub also trims both ends at -40 dB.
- **Maata check [new]:** Maata does not trim TTS output. It trims only *reference* clips (`librosa.effects.trim(top_db=35)`, `backends/torch_common.py` L103).

**C4 [corrected] Linly-Dubbing's timing and summary inputs.**
- Confirmed: stretch clamped to [0.6, 1.1] (up to about 1.67× faster, or about 10% slower), and start = max(start, end of the previous clip), so drift ripples forward.
- Corrections (`step040_tts.py` L26-44 and L92-107; `step030_translation.py` L147-162):
  - The slot is the line's *original length*, clipped at the next line's **end**. It never extends into the gap.
  - The stretched audio is then truncated to the target length.
  - The summary uses title + uploader + up to 2,000 transcript characters. The description is not used, and tags are used only when the summary is translated.

**C5 [confirmed] Drift comes from concatenation and from ASR onsets.**
- pyVideoTrans #1007 (2026-04-05): the dub started at 0:01 instead of 0:10; fixed by a patch on 2026-04-06.
- VideoLingo #154 (2024-10-14): a line started early after a long pause. The maintainer blamed WhisperX alignment limits.
- VoiceStudio's `onset_align.py` snaps starts forward only, on the separated vocal track: 20 ms frames, 10% threshold, 160 ms sustained within 300 ms, minimum shift 0.15 s, jumps over 1.5 s only across near-silence.
- I read VoiceStudio #280 (2026-06-04: the dub starts at 0 s while the speaker starts at 2–3 s) and #963 (2026-07-05: footsteps taken as speech onset) directly. The researcher could cite them only through code comments.

**C6 [confirmed] Sudden fast lines are a common complaint.**
- VideoLingo #453 (2025-05-21, open): a commenter on 2025-07-24 traces it to compressing each target line into its source time.
- SoniTranslate #19, pyVideoTrans #869 and #1190, KrillinAI #176 and #109, as above.
- VoiceStudio's comment that above about 1.8× speech becomes "garbled" is a design note.
- Calling it the *most* common complaint is the researcher's judgment over the issues sampled.

### D. Voice references and identity

**D1 [confirmed] How Chatterbox Multilingual uses the reference.**
- `ENC_COND_LEN = 6·S3_SR`: T3 prompt tokens come from the first 6 s.
- `DEC_COND_LEN = 10·S3GEN_SR`: S3Gen takes the first 10 s.
- The voice encoder sees the whole clip.
- Source: `src/chatterbox/mtl_tts.py` L156-157 and L253-276 @5de7a54a (2026-07-21), MIT.

**D2 [new] More of Chatterbox's conditioning internals.**
- The voice encoder trims silence (top_db 20) and averages partial windows. `VoiceEncoder.embeds_from_wavs(..., as_spk=True)` / `utt_to_spk_embed` averages several clips into one speaker embedding.
- S3Gen `embed_ref` resamples any reference to 24 kHz for an 80-band mel with **fmax 8000 Hz**. Its x-vector and tokenizer use 16 kHz.
- Consequence: nothing in Chatterbox's conditioning uses audio above 8 kHz.
- Sources: `models/voice_encoder/voice_encoder.py` L246-274; `models/s3gen/s3gen.py` L118-155; `models/s3gen/utils/mel.py` L25-37 @5de7a54a.

**D3 [new] Maata already implements the reference strategy the report recommends (ADR-017).**
- `session._build_voice` builds three parts:
  - timbre = the best single clean clip, ≤10 s, skipping the cold open;
  - identity = the voice-encoder embedding averaged over up to 60 s of clean clips (`prepare_voice_parts`);
  - T3 prompt = the first 6 s of the timbre clip.
- It falls back to a stitched reference when no single clip reaches 8 s.
- Measured: guest similarity 0.400 → 0.466, and pace 3.43 → 5.11 aksharas/s. The host stays on the stitched reference at 0.645.
- Sources: `engine/src/maata_engine/session.py` L301-321; `backends/torch_common.py` L155-190; `docs/DECISIONS.md` ADR-017 (table at L172-173).

**D4 [confirmed] SoniTranslate cleans its voice reference.**
- It takes the first speaker segment lasting 7–12 s (skipping the first one if there are more than 4), trims 1 s from each end, and runs UVR MDX main-vocals plus dereverb. Output is mono, 22.05 kHz, s16. Fallback: the first segment clamped to 2–9 s.
- Voice-conversion references: spans of 3–18 s; `max_segments` 10 in code, 3 in the UI; FreeVC by default.
- Source: `text_to_speech.py` L445-585 and L1230-1330 @70b6390a.

**D5 [confirmed] Other pipelines' reference choices.**
- mazinger's auto-clone takes the contiguous 20–60 s window with the most words (at least 20), 16 kHz mono (`mazinger/profiles.py`, `MIN/MAX_CLONE_DURATION`, `MIN_CLONE_WORDS`).
- YouDub clones each line from its own vocal slice (-80/+160 ms). Fallback: the speaker's first slice of at least 1,200 ms. VoxCPM runs with cfg 2.0 and 10 timesteps, retrying bad cases up to 3 times.

**D6 [confirmed] Per-line vs per-speaker references.**
- Per line: pyVideoTrans `clone-{i}.wav` (`_stage_dubbing.py` L88-90), VideoLingo `_9_refer_audio.py`, YouDub, and VoiceStudio `per_line` (its default).
- Per speaker: SoniTranslate, Linly, mazinger, VoiceStudio `consistent`, and Maata.
- The hybrid idea (per-speaker S3Gen and voice-encoder conditioning, with T3 prompt tokens from the line's own audio) remains an **untested inference**.

**D7 [confirmed] Identity and accent failures users report.**
- pyVideoTrans #1123 (2026-05-28; maintainer reply 2026-05-29): every engine carries the accent across languages; the only fix is prompting.
- pyVideoTrans #1182 (2026-08-18): the second line comes out in "another unknown voice".
- VideoLingo #279 (2024-11-22; maintainer 2024-12-04): GPT-SoVITS has requirements on the reference audio.
- SoniTranslate #120 (2024-11-04): an Edge-TTS *multilingual preset* voice switches language for one word. This is not a cloned voice.
- VoiceStudio #280 also reports a strong source accent in cloned Spanish dubs.

**D8 [confirmed] VideoLingo's reference bandwidth measurement.**
- v3.0 lowered extraction to 16 kHz / 32 kbps. White-noise -10 dB cutoff: about 7.2 kHz, against about 15.3 kHz at 32 kHz / 128 kbps.
- PR #611 (by the maintainer, merged 2026-09-14) restored 32 kHz / 128 kbps and the 44.1 kHz Fish reference. It states the listening benefit is unverified.
- PR #610 (by contributor doomsday616, merged 2026-09-14) made the loudness gain ignore low-level frames, capped the peak, and restored AAC to 192 kbps.
- #533 (2026-03-12) is still open. The maintainer (2026-09-14) says the line-by-line feel comes from per-line synthesis, hard-silence joins and per-chunk atempo, and won't be fixed now.
- For how this applies to Maata, see D2 and the Refuted list.

### E. Player, background audio and licences

**E1 [corrected] The YouTube IFrame player and fine playback rates.**
- The documentation wording is confirmed (reference last updated 2026-09-15):
  - `setPlaybackRate` does not guarantee a change;
  - an unsupported value is rounded down toward 1;
  - cueing or loading a video resets the rate to 1;
  - apps must listen for `onPlaybackRateChange`.
- Correction to "can't be reproduced unless the player lists finer rates" [new, measured by me on 2026-09-24]:
  - Setup: Chromium 152 (Claude Browser pane), `youtube-nocookie.com/embed/aqz-KE-bpKQ`, the embed's `movie_player`.
  - `getAvailablePlaybackRates()` returned [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2].
  - Yet `setPlaybackRate` with 0.9, 0.95, 1.1, 1.15, 0.85 and 1.3 each read back exactly.
  - `document.querySelector('video').playbackRate` was 0.9.
  - The pane was hidden, so frames did not advance.
  - This matches sibling sweep 08 (F18).
- Not verified: WKWebView (Tauri on macOS), and frames actually advancing at 0.9×.
- Maata's `app/src/lib/player.ts` L90 already reads `getAvailablePlaybackRates()`.
- ADR-008 still assumes the nearest *listed* rate at or above 0.85×.

**E2 [confirmed] The usual open-source default is to keep a background bed.**
- SoniTranslate: original at 0.25 (with an optional sidechain-ducking mix [new], `app_rvc.py` L1082).
- claude-skills: 0.18.
- mazinger: Demucs background at 0.15, with LUFS matched to the original.
- VideoLingo and Linly: separated background.
- Ariel: BS-RoFormer background.
- YouDub: the original audio for non-verbal sounds.
- That Maata's rule makes gaps more audible is an inference.

**E3 [confirmed] Repo licences and dates.** (GitHub API, 2026-09-24)

| Licence | Repos |
|---|---|
| Apache-2.0 | VideoLingo (18.5k stars), KrillinAI, SoniTranslate, Linly (dormant since 2025-03-05), open-dubbing (dormant since 2025-07-08), ViDubb, Ariel, YouDub-webui |
| MIT | mazinger, youtube-auto-dub, claude-skills, Chatterbox |
| GPL-3.0 | pyVideoTrans, Voice-Pro |
| AGPL-3.0 | VoiceStudio (created 2026-04-09, 35.1k stars), ThioJoe |

GPL and AGPL code informs ideas only; none of it can be copied into Apache-2.0 Maata.

**E4 [confirmed] The BS-RoFormer ep_317 weight licence is not established.**
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` is a release asset of TRvlvr/model_repo, tag `all_public_uvr_models`, uploaded 2024-03-29.
- That repo declares **no licence**. The audio-separator code is MIT.
- Under Maata's policy this weight needs explicit maintainer approval. Treat it as all-rights-reserved.
- Demucs is MIT (facebookresearch/demucs, archived 2024-04-24).

### F. Maata code facts and Telugu ASR

**F1 [confirmed] The report's statements about Maata's code.**
- The planner rate stays within [1.0, `speed_cap`=1.2]. Other limits: lead 0.3 s, max lag 0.6 s, freeze budget 1 s per 60 s. There is no underrun fill.
- `RESYNTH_SHARE` = 0.10.
- The duration estimator is akshara-based, using Huber-weighted RLS per voice.
- Translation is one request per unit with a `target_units` budget. `translate_candidates` is called only if the first result doesn't fit.
- There is no round-trip ASR check.
- Sources: `timing/planner.py`, `timing/duration.py`, `session.py` L468-510 and L560-581.

**F2 [new] What happens when Maata fails on a line.**
- A failed translation or synthesis calls `_skip`. That sends `unit_skipped` to the UI; it is not silent.
- There is no retry, except the re-translation when a line echoes an earlier one.
- `ClaudeCLI` makes one sealed `claude -p` call per request and supports `--json-schema`. The flag exists in the installed CLI's help.
- Sources: `session.py` L608-613; `claude_cli.py`.

**F3 [new] Telugu ASR for a round-trip QC check.**
- Whisper-large-v3 zero-shot Telugu CER is 82.65 on FLEURS (BuzzASR/telugu card, vendor-reported, repo updated 2026-09-22). So Maata's existing Whisper can't do the check; this matches ADR-014's "too noisy".
- MIT candidates (Hugging Face API):
  - ai4bharat/indic-conformer-600m-multilingual (sha e9b71b369c, gated=auto);
  - BuzzASR/telugu (sha 598f495e, a Whisper-large-v3 fine-tune; vendor-reported FLEURS CER 13.53 / WER 47.06).
- Maata writes English words in Latin script, so CER needs script normalisation before it is meaningful (sibling sweep 10, F16).

**F4 [unverifiable] "Maata's planner and estimator already beat every open-source planner."**
- This is a design comparison; no measurement compares them head to head.
- The design is more principled than the static tables and EMAs I read. VoiceStudio's median characters-per-second calibration is the closest.

## Corrections to the report's summary

- **"Audio-only speed-up is capped at about 1.2×" [corrected].**
  - VideoLingo: max 1.4 is only the trim trigger; chunk speed is uncapped (`process_chunk` else-branch).
  - KrillinAI: SpeedMax 1.30 only logs a warning (`fit.go` L36-45).
  - SoniTranslate 2.1, youtube-auto-dub 1.4, open-dubbing 1.3, pyVideoTrans 100.
  - Only VoiceStudio uses 1.2 as its audio-only threshold.
- **VideoLingo merge limit [corrected].** Merging stops at 3 lines (the `merge_count == 2` early return) even though `MAX_MERGE_COUNT = 5` (`_8_2_dub_chunks.py`).
- **VideoLingo Telugu estimate [corrected].** No regex matches Telugu script, so `_detect_language` returns 'en'. Each Telugu word then goes through the English syllable counter with a floor of 1: about 0.225 s per word. The underestimate is the same as the report says; the path differs.
- **Chatterbox voice reference [corrected].** The report lists the voice reference as a Maata gap. It is mostly already done (D3). Only the hybrid per-line T3 prompt is untested.

## Refuted

1. **"Keep full bandwidth, with no 16 kHz step before S3Gen" would help Chatterbox similarity.** S3Gen's reference mel stops at 8 kHz, and the tokenizer, voice encoder and x-vector run at 16 kHz (D2). A clean 16 kHz reference removes essentially nothing Chatterbox uses. VideoLingo's problem came from a lossy 32 kbps MP3 cutting at about 7.2 kHz, not from 16 kHz sampling as such.
2. **The voice-encoder embedding averaged over 30–60 s, with the cleanest 6–10 s first, is a gap in Maata.** ADR-017 already does both and measured guest 0.400 → 0.466 (D3).

## Recommendations for Maata (updated)

1. **Scene-batched, id-keyed Claude calls that return length tiers (high impact, M effort).**
   - One `claude -p --json-schema` call per 20–30 units, with 6–8 units of overlapping context.
   - Each unit returns `{id, full, short, shortest}`.
   - Validate by id and retry only the missing ids. Never pad; send `unit_skipped` only after the retry.
   - The planner then picks the longest tier that fits.
   - Caveat: ADR-018 found that a "next line" note made the local model translate the next line too. Id-keyed JSON should prevent that, but test it in the blind bake-off.
   - Record it as an ADR, since the prompt contract changes.
2. **A per-video brief (high impact, S effort).**
   - Input: yt-dlp title, description, tags and chapters, plus the full transcript.
   - Output: theme and register, speaker roles, a glossary deciding English vs Telugu for recurring terms, and high-confidence ASR corrections only.
   - Cache it by transcript hash and inject it into every scene call. This is what VideoLingo, YouDub, VoiceStudio and mazinger do.
3. **Critique-then-polish only on lines the Tenglish lint flags or that need condensing (medium, S).** Instructions: same length or shorter, add nothing, keep the brief's term decisions.
4. **Underrun fill (medium, S; planner ADR).** Let the rate drop to about 0.85–0.9 when a dub fills less than about 90–95% of its window.
5. **Revisit video-side fitting (medium, S).**
   - Run the probe (`dub_research/iframe_probe/index.html`) inside the Tauri WKWebView with the window visible.
   - If 0.90–0.95 plays smoothly there, update ADR-008. A gentle, metered video slow-down could replace some freezes, since fine rates are accepted even though they aren't listed.
   - Keep the freeze as the last resort.
6. **Round-trip ASR check of the synthesized Telugu (high, M; needs approval).**
   - Candidate: IndicConformer-600M (MIT, gated auto) or BuzzASR/telugu (MIT). Whisper can't do it (F3).
   - Normalise script before computing CER, and flag at a drift of about 0.5.
   - A cheap first step with no new model: flag takes whose duration is far below the estimator's prediction (for example under 0.6×), or that hit the T3 token cap. This catches truncation. It is my inference and untested.
7. **Tail trimming (low priority).** Maata doesn't trim TTS output today. Keep it that way, and never add a -40 dB gate or fixed-millisecond cuts.
8. **Voice references (high impact on the complaint, M).** Drop the bandwidth and averaging items; they don't apply (see Refuted). What remains:
   - an A/B test of the hybrid: per-speaker S3Gen and voice-encoder conditioning, with the T3 prompt from the line's own audio;
   - for the host, whose single clips are under 8 s: build the averaged embedding (`as_spk`) over many clips, while keeping the stitched timbre.
   - Bench both with WeSpeaker SIM and a blind listening test.
9. **Separation (medium, M; needs approval).** If it is approved, prefer Demucs (MIT). The BS-RoFormer ep_317 weight has no declared licence (E4). This also decides the background-stem option.
10. **Per-speaker loudness over speech-only frames (low, S).**
11. **Session report in the UI (medium, S).** Show skipped ids, lines re-synthesized shorter, overdraft and freeze totals, maximum rate, and QC flags. Most of these are already in the planner stats and the `unit_skipped` events.
12. **Background audio decision for the maintainer (needs approval; relaxes a hard constraint):**
   - (a) dry dub plus underrun fill;
   - (b) a separated background stem at about 0.15;
   - (c) the original at 0.15–0.25 under the dub;
   - (d) the original audio for non-verbal sounds only.

## Open questions

- Does WKWebView (Tauri on macOS) honour `setPlaybackRate(0.9)` with frames advancing smoothly and `onPlaybackRateChange` firing? Chromium does, measured with a hidden pane.
- What is the measured Telugu/English duration ratio for chatterbox-telugu on the reference podcasts: does overrun or underrun dominate?
- Does a hybrid T3 prompt (the line's own audio) or a multi-clip `as_spk` embedding for the host raise WeSpeaker SIM? A bilingual same-speaker ceiling is needed to interpret 0.466 and 0.645.
- How does IndicConformer-600M, or BuzzASR/telugu, do on Maata's Tenglish output after script normalisation, and what memory does it need alongside the resident models?
- What does scene batching with three tiers plus a brief cost per hour of video in Claude CLI time and usage limit?
- Is there a permissively licensed vocal-separation weight newer than Demucs v4 (MIT, archived) that Maata could pin?

## Sources

GitHub, at pinned commits (commit dates from the GitHub API):
- Huanshere/VideoLingo @9bc30202 (2026-09-23); issues #154, #202, #279, #453, #533; PRs #610, #611 (2026-09-14)
- jianchang512/pyvideotrans @d63d27ae (2026-09-23); issues #869, #1007, #1012, #1123, #1182, #1190
- R3gm/SoniTranslate @70b6390a (2026-08-29); issues #19, #120, #127
- krillinai/KrillinAI @f96cdf19 (2026-09-24); issues #109, #176
- Softcatala/open-dubbing @bff534d3 (2025-07-08)
- google-marketing-solutions/ariel @28ec8a92 (2026-09-23)
- Kedreamix/Linly-Dubbing @5677191e (2025-03-05)
- medahmedkrichen/ViDubb @6a90ea25 (2025-07-23)
- abus-aikorea/voice-pro @7231384f (2026-07-13)
- debpalash/VoiceStudio @324eb9a6 (2026-09-24); issues #280, #963
- resemble-ai/chatterbox @5de7a54a (2026-07-21)
- liuzhao1225/YouDub-webui @0f6c7593 (2026-09-21)
- ThioJoe/Auto-Synced-Translated-Dubs @8360469f (2026-05-11)
- mazzasaverio/youtube-auto-dub @631ee274 (2026-09-05)
- bakrianoo/mazinger @e7595760 (2026-09-05)
- jianshuo/claude-skills @b2690f5b (2026-08-20)
- TRvlvr/model_repo, release `all_public_uvr_models` (asset uploaded 2024-03-29)
- facebookresearch/demucs (MIT, archived 2024-04-24)

Documentation and model cards:
- YouTube IFrame Player API reference, https://developers.google.com/youtube/iframe_api_reference (last updated 2026-09-15)
- Hugging Face API: ai4bharat/indic-conformer-600m-multilingual (2026-02-07); BuzzASR/telugu card (2026-09-22)

Maata repo (read-only, 2026-09-24):
- `engine/src/maata_engine/{session.py, timing/planner.py, timing/duration.py, backends/torch_common.py, claude_cli.py}`
- `app/src/lib/player.ts`
- `docs/DECISIONS.md` (ADR-008, ADR-014, ADR-017, ADR-018)

Measurement: my own IFrame playback-rate probe, 2026-09-24, Chromium 152, via the Claude Browser pane.
