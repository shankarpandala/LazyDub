# Maata: target architecture for a realistic Telugu dub

Date: 2026-09-24. Author: architect pass over the verified research in this folder, written for the maintainer. Revised the same day after an independent review; Appendix C lists what changed and which review points were rejected.

**Scope and inputs.** This document builds on and corrects:
- `../sota_ranking.md`, the model sweep;
- the ten verified reports `01`–`10`;
- the six gap studies `gap-1`–`gap-6`;
- the engine at HEAD, read-only (`session.py`, `backends/{apple,torch_common,base}.py`, `timing/planner.py`, `segment.py`, `speakers.py`, `text/tenglish.py`, `claude_cli.py`, `server.py`, `timing/duration.py`);
- `docs/DECISIONS.md` ADR-013 to ADR-018.

Two live checks were made for this pass:
- The Claude Code CHANGELOG top entry is 2.1.281 (raw GitHub, fetched 2026-09-24). 2.1.280 added `claude-opus-5-5`.
- The saved CLI reference (`scratchpad/cc_docs/cli-reference.md`, fetched 2026-09-24) documents a `--restricted` mode (v2.1.248+) that none of the reports used.

**Labels.**
- **[D] disclosed:** a company, paper or document states it.
- **[M] measured:** a number from a study, or from a run on the M5 Pro.
- **[I] inferred:** reasoning.

**Citations.** Evidence is cited by report and finding id, for example `[01 F4]` or `[gap-3 E2]`. The primary sources and their dates are in those reports; Appendix A lists the load-bearing ones with dates.

**Measurement rule (CLAUDE.md).** Every M5 Pro number below from `gap-1` to `gap-5` comes from **scratch runs that are not committed**. They set direction here. They may not be quoted in an ADR as results until `verify-mac.sh` or `maata-bench` re-runs them and the JSON is committed under `docs/spikes/results/<machine>/`. ADR-017 cites that directory, but it doesn't exist yet [gap-5 E1].

---

## 1. Executive summary

### 1.1 What realistic dubbing requires, in order of impact for a native Telugu viewer

1. **Complete, faithful, conversational Telugu.**
   - ElevenLabs' CEO calls translation the hardest part [01 F10].
   - Every premium tier sells a native-speaker translation review [01 F21, 03 F16].
   - In Maata's own trace:
     - translating sentence fragments inverted or invented meaning in 6 of 13 cases [gap-3 E3];
     - 8 of 98 lines dropped a clause [gap-4 E5];
     - a length note on the prompt raised major meaning errors from 5% to 13.5% (ADR-018).
   - The maintainer's complaints "translation very bad", "incomplete or skipped sentences" and "bookish" all sit here.
2. **A steady, natural speaking rate, with timing that slips rather than jumps.**
   - For native listeners, adding length-controlled MT plus prosodic alignment together (system A→B) lowered ratings by 10.93 MUSHRA (p < 0.01). They blamed a rate that was too slow, too fast or uneven [04 F1; 02 A1]. The study did not isolate alignment alone.
   - Smoothness was the most impactful metric in off-screen dubbing [02 A2].
   - Professional dubbers keep their rate steadier than the source and let timing slip [04 F3].
   - This is the "pace not in sync" and "not smooth" complaint.
3. **Uninterrupted playback.**
   - Today's pipeline runs at 0.59× real time, so any video longer than about 20 minutes must stall [gap-1].
   - Taking translation off the GPU is the single largest fix: it was 1.13 of the 1.70 GPU-s spent per video second [gap-1 §1].
4. **An intelligible, native-sounding Telugu voice.** Three measured levers:
   - Telugu-script input for English words cuts English-word loss from 20.8% to 12.1% [gap-2 C].
   - Best-of-N takes with ASR selection [04 F15].
   - QA against a Telugu recogniser that works; zero-shot Whisper does not, at CER 82.65 [10 A1].
5. **A voice that sounds like the speaker.**
   - WeSpeaker similarity is almost entirely S3Gen timbre plus a language penalty [gap-5 E2].
   - What listeners also hear (prosody, mean pitch about 1.1 st low, pace, accent) is invisible to WeSpeaker [gap-5 E4].
   - So similarity work has two lanes: an acoustic lane measured by WeSpeaker, and a perceptual lane gated by listeners.
6. **Acoustic consistency: loudness, room, EQ, and no dead silence.**
   - This is real, but the only controlled test found a non-significant gain for native listeners (+1.05 MUSHRA), against +10.34 for non-natives [04 F1, 07 A1].
   - It comes last, and never at the cost of items 1–3.

### 1.2 The eight biggest changes

| # | Change | Why | Where |
|---|---|---|---|
| 1 | **Fix ASR punctuation before segmenting.** A second, punctuation-prompted Whisper pass copies casing and punctuation onto the unprompted words. It takes no words from the prompted pass. | 18% → 2% of units end mid-sentence, with no words lost [gap-3 E2, M, scratch]. A prompt-only fix drops 14–23 s of speech [gap-3 E2]. | `backends/apple.py`, new `text/punct_transfer.py`, `session._sentence_cut` |
| 2 | **Scene-batched Claude translation, gated by a blind bake-off and a usage run before Gemma is removed.** A per-video brief (glossary, speakers, register), sentence-complete id-keyed units, and `--json-schema` output. Each line returns a **Telugu-script `spoken` field** plus an English-word map, and two-way length tiers (`full`, plus `fuller` / `concise` / `very_concise` only where needed), chosen locally by the calibrated duration. Sonnet 5 is the default; Opus 5.5 only if it wins by a stated margin. Validated lines are cached, so re-watching spends nothing. The voicer never waits on Claude. | Google translates caption *sentences* [02 Y7]. HOMURA [05 C1] and Microsoft LSST [02 MS4] support graded variants in one call. Telugu script is what the TTS was trained on [gap-2]. Budgets must be computed locally [04 N1]. No public EN→TE score exists for any Claude model [05 B2]. | new `backends/claude_translator.py`, `text/scene_prompt.py`; `session._translator` rewritten; inline Claude call removed from `session._dub` |
| 3 | **Coverage and meaning gate before any length or timing tuning.** Deterministic checks (ids, numbers, negations, glossary), a Claude scene review (complete / minor drop / phrase dropped / meaning error), and re-translation from the English with the missing words named. | Every length lever measured so far traded content for fit [gap-4]. This replaces the human linguist pass [03 F27]. | new `qa/coverage.py`, `session` |
| 4 | **Shape the pipeline for real time.** Translation off the GPU and running up to the lookahead; a priority GPU scheduler (voicer first when the lead is short); N=2 takes in one batched T3 decode; QA in a separate CPU process with capped threads, so it doesn't slow T3's CPU-bound decode loop; a pacing formula in wall-clock terms; a throughput governor; a seek policy. | 0.59× → about 1.2× on the podcast (1.4× only if the measured basis was 10 CFM steps and 6 steps is accepted) [gap-1 §4, M+I] | `server.py` lock → scheduler, `session`, `torch_common._t3_tokens`, new `qa/worker.py`, `app/src/lib/buffer.ts` |
| 5 | **Per-take QA and selection.** IndicConformer-600M, run in its own CPU process, scores CER against the `spoken` reference, with CTC forced alignment per word. At N ≤ 3, the pick is by CER plus duration fit. WeSpeaker similarity is scored at matched length and used only to reject outliers. Retake, then an asynchronous rephrase, then flag. | Whisper can't check Telugu [10 A1]. Best-of-N [04 F15]. Similarity must be scored at matched length [gap-5 E2]. | new `qa/take.py`, `session._dub` |
| 6 | **Timing v2.** Place whole sentences and let them drift across pauses under 1 s. Split only at hard breaks, at a pause that is natural in the Telugu. Early starts cost about 2× late ones. A windowed lookahead. Underfill is fixed by wording and inserted pauses, not by slowing audio. Keep the 1.2× cap and the 1.0 floor. | End error median −0.39 s, long lines −1.98 s; 8.2 s/min of silent dub while the speaker talks [08 F1]. Only 30% of the underfill is dropped content [gap-4]. | `timing/planner.py`, `segment.py`, `types.py` |
| 7 | **Voice lane, re-aimed by the gap-5 diagnosis, run in parallel from the start.** First fix the yardstick (matched length, no leakage, per-speaker ceiling, bilingual ceiling). The acoustic lane: every speaker built through one path (single clip or stitched timbre, plus a pooled x-vector and T3 embedding), S3Gen span chosen by cosine to the centroid, a `s3gen_v3` A/B. The perceptual lane: a +1 semitone mean-pitch test, cfg fixed per voice by ear. The V3 T3 retrain on Telugu-script data comes later. | The host's 0.645 includes leakage, and 0.746 is the guest's ceiling [gap-5 E1]. S3Gen sets WeSpeaker identity, and T3 moves it by at most 0.03 [gap-5 E2]. The host is on the stitched path, which no earlier change reached [06 B5, R3]. | `session._build_voice`, `speakers.best_span(score=…)`, `torch_common.prepare_voice_parts`, `bench.py` |
| 8 | **Mix like a dub, not a TTS stream.** Per-speaker loudness matched to the source (own dialogue gate), per-line ±3 dB following source energy, fades, crossfades on overlaps, a master limiter, and synthetic room tone (option B, pending the maintainer's decision). | Netflix mixes the dub at the original's level [07 C1]. Fully muting reads as "sterile" [03 N1]. | `app/src/lib/sync.ts`, engine loudness pass |

**Not changed:**
- the English ASR model, until a bake-off;
- diarization, until a bake-off plus a licence ruling;
- the TTS family: Chatterbox-Telugu stays. No permissive Telugu cloner beats it cross-lingually [06 E9].
- no post-hoc voice conversion;
- the IFrame video is never slowed in the normal path.

### 1.3 Corrections to the brief's premises and to `sota_ranking.md`

| Premise | Correct fact | Source |
|---|---|---|
| "Segmentation into short units (≤12–20 words)" | `segment.py` splits by **time**: `max_len` 12 s, `long_pause` 1.5 s, sentence ends, `min_len` 2 s. Then `session` calls `merge_fragments(max_len=20, max_gap=1.0)`, which joins mid-sentence fragments into lines of up to 20 s; the TTS cap is `MAX_LINE_SECONDS = 30`. Units are meant to be sentence-complete, but 18% weren't, because Whisper dropped punctuation on about 37% of the speech. | [01 F22], [gap-3 E1], `session.py:56, 402`, `segment.py:558` |
| Guest 0.466 / host 0.645 against a 0.746 ceiling | 0.746 is the **guest's** clip-to-clip ceiling. The host's is 0.658, on another basis. The host's 0.645 includes leakage: 4.4 s of its reference is 67% of the held-out audio. Leakage-free, the host scores about 0.56–0.57. Takes must be compared at matched length. | [gap-5 E1] |
| Voice pace 3.4 aksharas/s (report 05 D4/R2) | Refuted. Production voices run at 5.4–6.0 under today's count (trace, calibration takes, re-measurement). 3.4 came from one excerpt's stitched reference. `DEFAULT_RATE` (5.5 today) is right for today's count; under the Telugu-script count, which gives about 9% more units, the prior becomes about 6.0–6.3 (§3.7). | [gap-4 E2, E3; gap-2 C] |
| "Budgets too tight cause underfill" | Normal lines get no length note since ADR-018. Underfill is mostly compact but complete Telugu against spans that contain pauses, plus about 30% dropped content. | [gap-4 E4, E5] |
| YouTube dubs English→Telugu with Expressive Speech (report 05 C7) | Telugu is a target **without** Expressive Speech. Only Telugu→English is expressive. | [02 Y1], [08 F9] |
| `sota_ranking.md` §1 (local translator shortlist) | **Superseded** by today's no-local-LLM decision. About 8 GB and 1.13 GPU-s per video-s are freed. | [04 relation], [gap-1] |
| `sota_ranking.md` §4: VoxCPM2 "RTF about 1.76 on M4 Pro" | That figure is the llama.cpp-omni Q8_0 build on Metal. MLX speed is unmeasured. VoxCPM2 also scored below Chatterbox cross-lingually. | [06 E8, E9] |
| `sota_ranking.md` §4: MeanVC2 / kNN-VC rejected because WavLM is CC-BY-SA | CC-BY-SA is "ask", not "never". They stay rejected for other reasons: 16 kHz output, auto-download, phonetic leakage. | [06 F1, F2, F7] |
| `sota_ranking.md` §4: vocoder pairing | Resemble's Hindi pack ships its T3 with `s3gen_v3`, so test both pairings. Chatterbox VC is S3Gen again and adds nothing. | [06 B7, B8] |
| `sota_ranking.md`: missing stage | It has no Telugu QA recogniser. Add IndicConformer-600M (MIT) as the take selector, with a different-family model for reporting. | [10 A2, B2] |
| "Opus" in `claude_cli.py` | On the installed CLI 2.1.201, `opus` resolves to **Opus 4.8**. Opus 5.5 needs 2.1.280 or later, and the server rejects older clients. 2.1.201 also silently ignores an invalid `--json-schema`. | [05 A1, A12] |
| "The engine makes no network calls except yt-dlp", "no cloud AI" | Contradicted by today's decision (Claude CLI for text). Needs an ADR and a CLAUDE.md amendment. | [gap-6 §8] |

---

## 2. How ElevenLabs, YouTube and the best others actually do it

Only verified facts are listed. Dates are publication dates, or fetch dates for undated pages. [R] marks third-party reporting.

### 2.1 ElevenLabs

**Product [D]:**
- Dubbing v2 (Alpha) is the default: the UI launched 2026-05-28, the API 2026-08-10.
- Dubbing Studio (v1) is in maintenance mode.
- Up to 32 speakers per file, with overlapping speech detected.
- Telugu (`te`) is supported, with no dialect variants. ElevenLabs publishes no Telugu quality claim.
- Realtime dubbing is "not currently available". v2 returns one lossless file, so it renders offline before anything plays.
- Sources: ElevenLabs dubbing docs, fetched 2026-09-24 [01 F1, F16; gap-1 §6].

**Data model [D].**
- The source is transcribed once into segments (id, text, speaker, start, end). Every language translates from that master transcript as text per segment.
- Target segments keep the source's start and end.
- The bring-your-own-transcript guide says translations are rendered to fit their segment's time span, and that a much longer or shorter translation changes the dub's pacing, so aim for comparable spoken length.
- Segmentation guidance: segments of 0.1–25 s; break sentences at pauses of 1 s or more; different speakers' segments may overlap.
- Source: API docs and guide, fetched 2026-09-24 [01 F4, F6].

**Generator [D marketing / I mechanism].**
- Marketed as "audio-to-audio" and "sync-aware translation" that conditions on the original performance (blog 2026-05-28, updated 2026-09-06).
- **[I]:** a cascade-shaped hybrid:
  - ASR and diarization;
  - LLM translation per segment with timing in mind;
  - a generator conditioned on the target text plus the source segment's audio;
  - a remix over the retained background.
- Undisclosed: the generator type, the LLM, and the maximum stretch [01 F8, U4, U5].
- The API blog (2026-08-06, updated 2026-09-22) self-estimates "80–90% of human quality" with no method [01 U3].

**Voice [D].**
- `cloning_strength` runs 0–10, default 7, per language target. Higher values sound more like the speaker but can be less natural in phonetically distant languages, and can "carry over more of the original accent" [01 F2].
- Studio offers a track clone (default, from all clips; can be unstable when the voice varies) and a clip clone. Its advice: clone one strong clip and use it for the whole track [06 A1].
- IVC guidance (standalone product, not dubbing): 1–2 min of clean audio [01 F14].

**Timing [D].**
- Studio's default Fixed Generations keep the clip duration, so speech can speed up or slow down "significantly". Dynamic Generations fit the clip to the text at the cost of sync [01 F7; gap-4 E7].
- The 0.7–1.2 speed range is a general TTS voice setting, **not** a dubbing limit [01 F15; 08 F8].
- No lead or lag numbers are published. Maata's 0.3 s / 0.6 s are Maata's own [08 F8].

**Background [D].** The original background is retained by default. v1 had `drop_background_audio`, which "can improve dub quality" for speeches and monologues [01 F5; 07 A3].

**Quality process [D].** Productions adds:
- native-speaker translation review;
- human voice selection;
- pacing fixed by regenerating segments or editing text;
- a QC checklist;
- about 7 business days per job.

Sources [01 F21; 10 E1]. The CEO (Turing Post, 2025-04-12) said translation was the hardest part, that Spanish runs 30–40% longer, and that the Lex Fridman dubs took 1–2 weeks with external QA [01 F10].

**Failures reported [R, low confidence].** Users on the v2 launch post report literal translations and "compresses some parts to match the timings", meaning content was dropped [01 F19].

**Telugu TTS [M, conflicted].** In a 10-utterance PSP pilot by a competitor, Eleven v3 Telugu had a pitch range 40% narrower than natives [01 F17]. Maata's clones do *not* show that compression [gap-5 E2].

### 2.2 YouTube and Google

**Product [D].**
- Auto-dubbing covers English→Telugu, but Expressive Speech covers only 8 languages. Telugu is expressive only as a source dubbed into English.
- Ineligible: videos over 120 min, and speech "too fast" to dub without an unlistenable speed-up.
- Known issues YouTube lists include proper nouns, idioms, jargon, and matching the dub voice to the original.
- Sources: YouTube Help 15569972, fetched 2026-09-24; blog 2026-02-04 [02 Y1, Y2; 10 E2].

**Expressive Speech [D quote / R].** It carries pitch, intonation and energy, and was built with DeepMind (Buddhika Kottahachchi via Slator, 2026-02-06) [02 Y3]. Whether it clones the creator's timbre is undisclosed. sync.so, a competitor, says YouTube uses generated voices [02 Y4].

**Early failures [R].** Dubs sped up to 1.5×, a "tin can" voice, stiff sentences (heise, 2025-03-14) [02 Y12].

**Patent [D]: US 2020/0404386 A1, granted as US 11,582,527 B2 on 2023-02-14.**
- Caption fragments are joined into **sentences**, the sentences are machine-translated, and each keeps its source interval.
- Fit order: audio rate (worked examples of about ±10%), then grouping nearby lines and inserting pauses, then changing the video rate.
- Mixing options: lower the original, erase the voice while keeping music, or replace the audio.
- Sources [02 Y7, Y8; 08 F9, A1; gap-3 E5].

**Research [D].**
- Aloud (2022-03-09) was a cascade with audio separation.
- The real-time S2ST translator (2025-11-19) is trained on data a cascade produced (ASR, alignment, MT, voice-preserving TTS).
- AudioPaLM conditions on a 3 s sample of the input utterance.
- Sources [02 Y5, Y10, Y11].

**Scheduling [D].** Dubs are generated at upload, then published or held for creator review [gap-1 §6]. Like ElevenLabs, this is offline rendering.

### 2.3 The best others

| Who | What they disclose | Tag, source |
|---|---|---|
| **Sarvam Dub** | Four-stage cascade: Saaras STT → translation keeping timing cues → Bulbul V3 with per-line emotion, pitch and pace → sync. The target duration is set **up front in generation**, because post-hoc stretching "distorts rhythm". A style selector (Auto / Formal / "Urban colloquial"): fix a flat or formal dub through style before blaming the voice. The editor mixes Dubbed / Original / Background, and warns a fully muted background feels "sterile". Its Telugu sample writes English loans in **Telugu script**. | [D] blog 2026-02-01 (modified 2026-09-14), docs fetched 2026-09-24 [03 F1–F5, N1] |
| **Descript + OpenAI** | "Unnatural pace" was the top complaint. It translates to a **syllable budget** from language-specific rates, with neighbouring chunks as context. Slowing 10% or speeding 20% still sounded natural. Segments inside that window rose from 40–60% to 73–83%. An LLM judge scores meaning. "Match timing" can use fuller phrasing. No Telugu. | [D, vendor-measured] OpenAI case study about 2026-03 [03 F6, F7, N2] |
| **HeyGen / Synthesia** | Both retime the video by default (HeyGen ±20% "Dynamic Duration"; Synthesia "Adaptive"). HeyGen's Precision tier adds context- and gender-aware translation and a brand glossary with pronunciations. | [D] help docs fetched 2026-09-24 [03 F8, F9] |
| **Microsoft** | Azure video translation is a cascade with LLM reformulation. LSST (arXiv 2506.00740, 2025-05-31) makes short, normal and long variants in one pass and picks one with the voice's duration model: speech-rate compliance +16–20% relative, BLEU flat, sync MOS +0.34/+0.65. Direction is (ES,KO)→EN, so it is not an SOV-target study. | [D/M] [02 MS1, MS4; gap-3 corr. 2] |
| **Amazon research** | Non-uniform per-phoneme scaling beats uniform scaling [02 A4]. Phrase-level prosody transfer scored +6.2% MUSHRA [02 A6]. Off-screen relaxation beats isochrony [02 A2]. Isometric MT alone doesn't buy timing [02 A5]. Human dubs keep rate steady and overlap only 0.66 [04 F3]. | [M] 2020–2023 |
| **Meta** | The Text-Audiobox dubbing TTS generates N takes, filters to speaker similarity ≥ 75% of the best, then picks the lowest ASR WER. At N=32, similarity rose 0.66 → 0.74 and WER fell 4.05% → 2.20%. Reels translations include Telugu (since 2026-01-16), cloud only, architecture undisclosed. | [M] arXiv 2609.03992 (2026-09-03) [04 F15]; [D] [02 M3] |
| **ESTsoft (perso.ai)** | The only dubbing system found with a verb-final target (Korean). It translates **whole sentences** with duration-based length control, then an LLM places pauses where the **target language** breaks naturally. | [M] EMNLP 2025 demos [04 F7; gap-3 E5] |
| **Bilibili HOMURA** | Unconstrained LLMs land in budget only 22–25% of the time. Graded variants in one call reach 62–68%, but systematically undershoot (median 0.87 of budget). A separate translate-then-rewrite pass is where omissions appear. | [M] arXiv 2601.10187 v3 (2026-09-03) [05 C1; gap-4 E7] |

### 2.4 What the leaders converge on, and what Maata takes

| Converged practice | Status [D unless noted] | Maata today | Target |
|---|---|---|---|
| Cascade: ASR/diarization → LLM translation with timing awareness → cloned TTS → sync → remix | Everyone who discloses | Yes | Yes |
| Translate **sentences** from one master transcript, with context and glossary | Google, ElevenLabs, ESTsoft, HeyGen, Sarvam | Line by line, fragments when ASR loses punctuation | Scene-batched sentences plus a brief (§4) |
| Length control **inside** translation (budget or variants), chosen by the voice's duration | Descript, LSST, ESTsoft, HOMURA | One-way, and only on predicted overrun | Two-way tiers, local band selection |
| Duration control inside generation | Sarvam, Meta (Audiobox) | Not possible with Chatterbox | Text tiers + take choice + ≤1.2× mel (cfg is a fixed voice setting, not a per-line lever) |
| Small rate changes; fix the rest with pauses and wording; video retime as a fallback | Google patent, HeyGen, Synthesia | Rate ≤1.2×, freezes, no video retime | Same, plus pause insertion and wording; video retime only after a WKWebView probe |
| Per-segment conditioning on the original performance | ElevenLabs v2, YouTube Expressive | One global prompt per speaker | Per-line delivery labels; per-line T3 prompt as an A/B |
| A user-facing similarity-vs-naturalness dial | ElevenLabs | UI exists; the engine ignores it (`session.py:114`) | Wire it after the listening test |
| Human linguist review plus regeneration | ElevenLabs Productions, Dubverse, RWS | None | Claude coverage review plus a "this line is wrong" hotkey |
| Keep the background | ElevenLabs, HeyGen (music), Sarvam | Dry dub | Synthetic room tone (option B), pending the maintainer (§6) |
| Render offline before viewing | ElevenLabs, YouTube | Streaming lookahead | Streaming stays (the user wants playback within 5–10 min), with a "prepare whole video" mode [gap-1] |

---

## 3. Target pipeline, stage by stage

### 3.0 Shape

```
                 ┌── QA worker process (spawned, ≤ 4 threads) ──┐   ┌──── off-device (Claude CLI, 2–3 concurrent) ────┐
yt-dlp ─► audio ─┤ IndicConformer QA, CTC align, pyin/DSP,      │   │ brief v1/v2 · scene translation · fit ·          │
 + metadata      │ loudness, EQ and room statistics             │   │ coverage review · async rephrase (deadline)      │
                 └──────────────────────────────────────────────┘   └──────────────────────────────────────────────────┘
                              ▲                                                   ▲   │ validated lines → line cache
GPU scheduler (one owner at a time; priority: voicer when lead < 60 s):         │   ▼
  diarize (blocks) ─► ASR pass 1 + punct pass 2 ─► sentence units ─► music flag ─► scenes
                                                                       │
                                     translated units ─► TTS (2 takes, batched T3) ─► QA select ─► plan ─► DSP ─► watermark
                                          ▲ (the voicer never awaits Claude;                                        │
                                          │  a rephrase replaces a provisional take if it arrives in time)          │
                                                          WebSocket ─► Svelte SyncEngine (muted IFrame, mix, room tone)
```

**Unit lifecycle** (`UnitState` gains fields):

```
heard → music? (skipped_music) → scened → translated (tiers, spoken, en_map, delivery)
      → covered (C / m / P / E, retranslated if P/E)
      → chosen (tier) → takes → qa (cer, sim, dur, cap)
      → planned → voiced | voiced_provisional (rephrase pending, deadline) | flagged | skipped
```

Every state change goes to `units.jsonl`, with `cfm_steps`, `model`, `prompt_hash` and `cache_hit` on every voiced or translated unit.

### 3.1 Ingest

- **Approach.**
  - yt-dlp stays: pinned, `--no-remote-components`, never `-U`.
  - `ResolvedVideo` also carries the description, chapters and tags, which feed the brief and glossary [01 R7; 09 R2; mazinger 09 A6].
  - Keep the cached audio only as long as the session needs it [gap-6 Y12].
- **Change.**
  - `resolve.py` returns the metadata.
  - `session.open` passes it to the brief builder.
- **Evidence.** ElevenLabs' `keyterms` bias both transcription and translation [01 F3]. The open-source pipelines build a video brief from metadata [09 A1, A5, A6].
- **Caveat.** The whole ingest path is the main YouTube-terms exposure (§6, §9).

### 3.2 Separation and enhancement

- **Approach.**
  - **No separation for playback.**
  - Enhancement appears only as bake-off arms for the *cloning reference*, adopted only if duration-matched WeSpeaker similarity rises. One 2026 study found enhanced prompts *lowered* similarity (F5-TTS SECS 0.35 → 0.28) [07 B5].
  - Speech activity for the timing metrics comes from pyannote's own turns. No new VAD.

| Arm | Weight licence | Training-data licence | Runtime | Size | M5 Pro speed |
|---|---|---|---|---|---|
| Raw (control) | — | — | — | — | — |
| AUSoundIsolation HighQualityVoice | macOS system unit | not applicable (OS component used under the macOS licence) | small signed Swift helper, or pyobjc offline render | 0 | unmeasured |
| MossFormer2_SE_48K (`starkdmi/…_MLX` @ccd0ded0) | Apache-2.0 | **unchecked**: must be read from the ClearerVoice-Studio training recipe and recorded before adoption | MLX | small | unmeasured |
| DeepFilterNet3 (`mlx-community/DeepFilterNet-mlx` @220d5dfb) | MIT / Apache-2.0 | **unchecked**: DNS Challenge sets (mixed per-source licences) and other corpora named by the DeepFilterNet papers; each component licence must be recorded before adoption | MLX | about 9 MB | unmeasured |

- **Training-data rule** (the same inheritance test the TTS side applies, §3.8): no enhancement, separation or QA model is adopted until its pin ADR records the training corpora and their licences. A non-commercial corpus goes to the D16 ruling. The bake-off itself may run before that, because it produces only measurements.
- **Excluded:**
  - Sidon, for identity (it re-synthesises);
  - DialogueSidon and Sucial (non-commercial);
  - anvuew dereverb (GPL-3.0);
  - SAM-Audio (custom licence);
  - the BS-RoFormer ep_317 weight (no licence);
  - BandIt Plus (no weight licence; DnR/FSD50K NC inheritance) [07 B1–B7; 09 E4];
  - **Kim Mel-Band RoFormer** (`mlx-community/mel-roformer-kim-vocal-2-mlx` @64cbfcb0). The weights are MIT, but the training data is undisclosed, and vocal separators of this kind are typically trained on MUSDB18: academic use only, 48 tracks under CC BY-NC-SA. The card also says it is a music-vocal model not validated for general separation [07 B2]. It moves under D16 as an "ask". Nothing in the plan depends on it.
- **Change.** A bake-off only: `maata-bench voice` gains an `--clean {raw,ausi,mossformer,dfn}` arm for the T3 embedding clips alone, and for the S3Gen span too.
- **Approvals.** Vendoring MLX modules. mlx-audio itself needs transformers ≥ 5.14, which clashes with the 5.2.0 pin [07 B4; 06 E10].

### 3.3 Diarization

- **Approach now.** Keep pyannote community-1 @3533c8c (CC-BY-4.0, 0.03 GB). It measured 0.036–0.044 GPU-s per video-s on the M5 Pro [gap-1 §1, scratch]. It stays the fallback in every case.
- **Fixes, no new dependency.**
  - Flag crosstalk units where turns minus exclusive turns exceed 30% [10 R4].
  - A zero-gap speaker flip mid-sentence, with a lowercase continuation, is checked with both sides' embeddings. Merge the two if they match; otherwise mark `cut_off` [gap-3 §3].
  - Diarize enough audio to give every speaker at least 30 s of clean held-out speech for scoring. The host has only 13.8 s in a 5-minute excerpt [gap-5 E1].
- **Bake-off candidate:**
  - `nvidia/Nemotron-3-Diarization` MLX BF16 @59ed2dbf (0.199 GB): DIHARD III 12.73 vs 20.2 (cross-source), up to 8 speakers, streaming speaker cache.
  - Paired with `Wespeaker/…redimnet2-B6-LM` @e34354de (0.05 GB, VoxCeleb1-O EER 0.276 vs 0.797) for registry linking, since Nemotron outputs no embeddings [sota §3].
  - Runtime: the vendored MIT mlx-audio module from 03a4d99. `SpeakerRegistry` thresholds must be recalibrated for 192-d embeddings.
- **Licence.** Nemotron's OpenMDW-1.1 is permissive but not on the list, so it needs approval. Fallback: Sortformer v2 (CC-BY-4.0, ≤4 speakers) on NeMo-Speech.cpp.
- **Acceptance.** DER on the podcast plus two multi-speaker fixtures. Speaker-link errors across 3-minute blocks are zero on fixtures.

### 3.4 ASR and sentence segmentation

**ASR model now.**
- Whisper large-v3-turbo MLX (1.6 GB), about 0.03–0.05 GPU-s per video-s [gap-1].
- Add three things to `MLXWhisper.transcribe` (`backends/apple.py:32`):
  1. **Punctuation transfer** [gap-3 E2, M scratch, verified twice]:
     - a second pass per 30 s window with a punctuated style prompt;
     - `difflib` alignment copies casing and trailing punctuation onto the unprompted words;
     - it **never takes words** from the prompted pass. Reference implementation: `scratchpad/gap3/transfer.py`.
     - Cost: 2.3–4.6 s per 65 s chunk.
  2. `hallucination_silence_threshold` ≈ 2 s. It is supported in the locked mlx-whisper 0.4.3 and applies because `word_timestamps=True` [10 B6].
  3. **A coverage guard.** Diarized speech with no ASR words for more than 0.8 s is logged and re-decoded with `clip_timestamps`. Clusters of low-probability words and repeated n-grams are flagged. This also catches prompt-style drops [10 R4].
- **Never** use Whisper `initial_prompt` as the text source. It dropped 14–23 s of speech in three runs [gap-3 E2].
- Keyterm spellings come from the Claude brief (high-confidence corrections only, the YouDub pattern [09 A5]), not from a Whisper prompt.

**`session._sentence_cut`:** when no sentence end falls by `b`, search into `CHUNK_PAD` before falling back to the last word [gap-3 §1].

**Segmenter (`segment.py`, `types.SourceUnit`):**
- One translation unit per sentence.
- A pause of 1.0 s or more inside a sentence becomes a **break inside the unit** (`SourceUnit.breaks`), not a new unit.
- `max_len` may cut only at clause marks: never inside a name or number, and never between a verb and its object when a comma exists.
- **Anchors.** Today `segment.anchors()` (`segment.py:523`) exists but nothing calls it, and `SourceUnit` has no field for its result [08 F2]. Add `SourceUnit.anchors` and fill it in `segment()`. **The threshold changes:** `anchor_pause` goes from 0.7 s to 0.3 s, because §3.10 uses anchors only as *soft* opportunities scored against Telugu break plausibility. Re-check the anchor count per line after the change; if lines average more than about 3 anchors, raise it to 0.4–0.5 s.
- Add `cut_off` for real interruptions.
- **How this fits `merge_fragments` and the TTS cap.** Once units are sentence-complete, `merge_fragments(max_len=20, max_gap=1.0)` has little left to join. Keep it as the safety net for the residual non-final units (target ≤ 3%), with two changes:
  - it may join only a unit whose text ends without a sentence mark to the same speaker's next unit, as today;
  - the joined line keeps both parts' `breaks` and `anchors`, and records the join point as a break if the gap is ≥ 1.0 s.
  - The order of ceilings stays: the segmenter's clause-mark-only `max_len` (12 s) < the merged-line limit (20 s) < `MAX_LINE_SECONDS` (30 s), so a merged line's Telugu can still run past its slot without hitting the TTS cap. A sentence longer than 20 s is cut at its best clause mark before merging, never mid-clause.
- On this podcast, only 2% of sentences have an internal pause of 1 s or more [gap-3 E4].

**Music and sung content (new check, in the frontend).**
- **Problem.** Whisper transcribes lyrics, and the pipeline would dub them. YouTube's own auto-dub skips music videos with a filter [02 Y2]. The loudness gate and the room-tone statistics (§3.11) also assume that non-speech gaps are ambience, not a music bed.
- **Approach.** Tag each 1 s frame with an audio-event classifier. CED (Apache-2.0, already a §3.5 candidate) is an AudioSet tagger [07 D1], and AudioSet's label set includes Music and Singing [I: its per-class accuracy on podcasts is unmeasured]. It needs a pin and approval (D15).
  - A unit whose speech is mostly Singing is marked `skipped_music`: shown in the UI as "music, not dubbed", never voiced, and counted in the session report.
  - A unit over a music bed stays dubbed; the flag only removes its gaps from the room-tone and loudness statistics.
  - Until CED is approved: a heuristic flag (a long unit with low word confidence, repeated n-grams and no diarized turn change) that marks lines for the report only.
- **Test fixture.** A CC0 or CC-BY clip with a sung intro and a music bed under speech, per the CLAUDE.md media rule.
- **Acceptance.** Sung lines on the fixture are not voiced; speech over the bed is voiced; no speech line is lost to the flag.

**Timestamps.**
- Whisper DTW has about 117–121 ms MAE. That matters for pause detection and end-error metrics, not for translation [sota §2; 08 R8].
- Bake-off pairs:
  - **Qwen3-ASR-1.7B 8-bit** (Apache-2.0, 2.47 GB). Short-form WER 4.31 vs 6.36. Punctuation unverified. About 7–11× RT (anecdotal). Needs vendored mlx-audio.
  - **parakeet-tdt-0.6b-v2**, cast to bf16 at about 1.25 GB (CC-BY-4.0, native punctuation, word timestamps at about 75 ms MAE, fastest).
  - **Aligner:** MFA 3.4.2 on the CPU (english_mfa CC-BY-4.0 / MIT, 20–34 ms MAE, needs conda approval) or Qwen3-ForcedAligner-0.6B 8-bit (Apache-2.0, 1.28 GB, second GPU pass).
  - Whisper stays the control. A natively punctuating ASR may make the second pass unnecessary [gap-3 OQ2].
- Qwen3-ASR has no Telugu, which doesn't matter for the English side [10 A6].

**Acceptance:**
- non-final units ≤ 3% on the podcast plus 3–5 more fixtures (fast talker, music bed, lecture, multi-speaker);
- words kept ≥ 99.5% of the single-pass count;
- 0 prompt leakage.

### 3.5 Emotion and prosody analysis

- **Approach: local DSP plus Claude labels. No emotion model yet.**
  - **Per speaker** (numpy, and librosa `pyin`; librosa 0.11 is already locked):
    - F0 median and semitone SD;
    - energy range;
    - syllable rate over diarized speech.
  - **Per line:**
    - the source line's energy and syllable rate relative to that speaker's mean;
    - its pitch-mean offset;
    - its internal pauses.
  - These go to Claude as numbers, next to the text.
  - Claude returns per line: `delivery = {emotion ∈ neutral|happy|sad|angry|surprised|serious, energy ∈ low|mid|high, question: bool, emphasis: [word indices]}`. This is the ElevenLabs "Enhance" pattern [01 F15, R8] and InstructDubber-style [04 F18].
  - **Mapping:** bounded per-line Chatterbox `exaggeration` ∈ [0.4, 0.7], starting from the speaker's calibrated default. Energy maps to a per-line ±3 dB mix gain (§3.11).
  - Higher exaggeration speeds speech up by about 3% [gap-4 E3, M scratch], so the duration estimate must be re-predicted after mapping.
  - **Exaggeration is baked into the cached voice today.** `prepare_voice_parts` builds `T3Cond(emotion_adv=self.exaggeration)` and the stitched path calls `prepare_conditionals(exaggeration=…)`, once per speaker (`torch_common.py:150, 188`). Per-line values therefore need a `synthesize_mel(…, exaggeration=None, prompt_tokens=None)` override that copies the cached `T3Cond` and replaces only `emotion_adv` for that call. That is a cheap tensor swap with no GPU work.
  - **The per-line T3 prompt A/B** (the line's own audio as the prompt) is not cheap: it also needs an S3-tokenizer pass on the GPU per line, about one short forward pass per line [I, unmeasured]. It is counted in §5.2 and only runs in the A/B until it wins.
- **Why:**
  - Brannon: source→dub line correlations are pitch mean 0.792, energy 0.381, rate 0.439 (0.584 for lines ≥ 1 s) [07 A2].
  - Word-level contours transfer weakly (0.17–0.25), so transfer *utterance-level* traits and emphasis, not contours [04 F25].
  - TTS takes emotion mainly from the text [06 D7]. That makes Claude's text-side labels the cheap lever.
  - Emotion-embedding cosine doesn't match perception, so never gate on it [10 C4].
- **Deferred, needs approval:**
  - emotion2vec+ (FunASR licence with forfeiture and self-updating terms);
  - Emotion2Vec-S (Apache-2.0, but needs fairseq plus a trained head);
  - 3loi (WavLM CC-BY-SA base, MSP-Podcast academic data) [10 C1].
  - Non-verbal carry-over (CED Apache-2.0 plus Chatterbox-Turbo tags, [07 D1, D2]) is low priority. Turbo is English-only and a different clone.
- **Acceptance.** A blind A/B (fixed 0.5 vs mapped) with ≥ 60% preference, no CER regression, and pace change accounted for in the planner.

### 3.6 Translation with the Claude CLI

Summary here; the full design is §4.
- **Model:** chosen per call type by the step-1 blind bake-off (§7.1). **Sonnet 5 is the default and wins ties**; Opus 5.5 (`claude-opus-5-5`) is chosen only if it beats Sonnet by the stated margin, because it draws more of the plan and has its own, tighter weekly limit [05 A4]. The other model is the `--fallback-model`.
- **Unit:** sentence-complete units in 60–150 s scenes (≤ 30 s for the first scene after start or a seek).
- **Output:** id-keyed `--json-schema` JSON with a Telugu-script `spoken` field and tiers.
- **Around it:** a brief/glossary, a coverage review, deterministic validators, and a per-line cache of validated output (§4.7).
- **Code changes:**
  - A new `backends/claude_translator.py` implements a `SceneTranslator` protocol (new in `backends/base.py`).
  - `session._translator` is rewritten to batch scenes, run 2–3 calls concurrently **without any GPU lock**, and translate up to the lookahead horizon. `TRANSLATE_AHEAD = 60` existed only because of GPU sharing.
  - **The voicer never awaits Claude.** Today `session._dub` calls `translator.translate_candidates` inline, holding the shared GPU lock, when a line needs to be shorter (`session.py:566–574`). That call is removed. A rephrase request goes to the translator's queue instead, and the voicer ships a provisional take (§3.13, §4.9).
  - `MLXChatTranslator`, `MLXTranslateGemma` and `StyledTranslator` are removed from `backends/apple.py`, and `LlamaCppTranslator` from `backends/cuda.py`. **Only after** step 1's bake-off and usage run pass (§7).
  - The "formal" style becomes a style section in the Claude prompt.
  - `claude_cli.py` is hardened (§4.9).
  - `text/tenglish.lint` is kept, but detects English through the English-word map instead of Latin letters [gap-2 E3].
- **Evidence:** §4.

### 3.7 Voice building

**Approach (ADR-017 structure kept, re-aimed by gap-5):**
1. **Yardstick first** (`bench.py`, a new `maata-bench voice`):
   - score takes in 9 s joined windows against each speaker's own 9 s real-clip-to-centroid ceiling, and report the fraction;
   - assert in code that conditioning audio and scoring audio never overlap;
   - use at least 30 s of held-out clean speech per speaker and at least 20 lines × 2 seeds per arm [gap-5 §1].
2. **Bilingual ceiling.** The maintainer records about 10 × 9 s of English and about 10 × 9 s of natural spoken Telugu. This single number decides whether any timbre headroom (about +0.1 estimated) exists [gap-5 §2].
3. **Acoustic lane** (WeSpeaker-addressable):
   - **One build path for every speaker.** Today a speaker without an 8 s clean clip (`SINGLE_CLIP_MIN`) falls back to the stitched reference: `prepare_voice` → Chatterbox `prepare_conditionals` (`session._build_voice`, `session.py:301–322`). That path gets no pooled T3 embedding and none of the changes below. The fixture's host, with 13.8 s of clean speech in the excerpt, is on it [06 B5, R3; gap-5 E1]. Route every speaker through `prepare_voice_parts` instead: the timbre input is the best single clip when one is ≥ 8 s, otherwise the stitched reference (trimmed to 10 s); the T3 embedding and the S3Gen x-vector are pooled over all clean clips either way. Keep the old stitched path as a bench arm only.
   - Choose the S3Gen span by cosine to the speaker centroid. `SpeakerRegistry.best_span` already accepts a `score` callback: pass the pyannote-embedding cosine.
   - **Pool S3Gen's CAMPPlus x-vector** over the same clean clips as T3's embedding. Today it comes from the single span.
   - A/B the identity pool at 60 s vs 120 s.
   - Re-synthesising the speaker's own English is the stack gauge: 0.678 vs 0.811 real [gap-5 E2].
4. **Perceptual lane** (listener-gated):
   - A **+1 semitone mean-pitch** post-shift, or picking the take nearest the speaker's median F0. Both dubs sit about 1.1 st low [gap-5 §3].
   - **cfg is a per-voice setting, fixed once.** cfg ∈ {0, 0.3, 0.5} is equal on WeSpeaker, but cfg 0 lengthens lines 21–27% and widens pitch spread [gap-5 E2], and cfg also governs accent carry-over [06 B3; 02 N1]. So it is chosen per voice by the blind listening test and then calibrated. It is **never** changed line by line to fit a slot [gap-4 decision 2].
   - Re-test native Telugu T3 prompts one factor at a time. ADR-017's rejection changed 2–3 factors at once on a metric blind to accent [gap-5 E1].
5. **Calibration.**
   - Three fixed Telugu sentences of different lengths per voice set rate *and* overhead directly. One sentence misestimates by −13% to +18% [gap-4 E3].
   - **The sentences follow the new script contract.** Today's `CALIBRATION_TE` (`session.py:62`) contains Latin "friends" and "share". Latin words are counted by syllables and were lost 20.8% of the time [gap-2 C], which biases the measured rate. Replace it with three Telugu-script sentences that include English loans written in Telugu script (for example ఫ్రెండ్స్, షేర్). Record the change against ADR-017.
   - Key `DurationEstimator` by (voice, cfg, exaggeration, reference hash).
   - **Rescale the prior.** `DEFAULT_RATE` (5.5 aksharas/s today, `timing/duration.py:13`) is right for today's count. The Telugu-script count gives about 9% more units per line [gap-2 C], so the prior becomes about **6.0–6.3** in the same change. The prior matters only until a voice's calibration lands.
6. **Wire `cloneStrength`.** closest / balanced / natural map to (cfg, prompt source, exaggeration) *after* step 4's listening test.
   - The engine stores it at `session.py:114` and never reads it [01 F23; 06 new 2].
   - The default leans toward natural Telugu, as ElevenLabs' dial implies [06 A2].

**Changes:**
- `session._build_voice` (`session.py:301`): one build path for all speakers; score callback; pooled x-vector; per-settings calibration.
- `torch_common.prepare_voice_parts` (`:156`): accept a stitched timbre; average the x-vector over clips, then replace `gen["embedding"]`.
- `session._calibrate`: three Telugu-script sentences.
- `timing/duration.py`: keyed models; `DEFAULT_RATE` rescaled with the count change.

**Memory and speed:** unchanged. Embedding extraction on the MPS is negligible (inferred).

**Acceptance:**
- similarity as a fraction of each speaker's ceiling +0.05 at matched length, with no CER regression, **for both the guest and the host** (the host scored on ≥ 30 s of held-out speech from the full podcast);
- blind MUSHRA "sounds like him or her" +10 with the maintainer's own Telugu recording as the hidden reference.

### 3.8 TTS with duration control

**Model now.**
- `shankarpandala/chatterbox-telugu` @d4341468 (model card CC-BY-4.0; base Chatterbox MIT).
- T3 in bf16, S3Gen fp32, about 2.2–3.2 GB resident.
- One take costs about 0.46 GPU-s per video-s on the podcast; T3's autoregressive decode is 56% of it [gap-1 §1, M scratch].

**Input contract: Telugu-script `spoken` text.**
- English words are spelled the way Telugu speakers write them (ఆర్డర్, ఐక్యూ).
- No Latin, ZWNJ/ZWJ or digits.
- Numbers are written as spoken Telugu words [01 F17, digits broke Chatterbox].
- Why:
  - The model's training set has Latin text in 0.30% of utterances (names and acronyms only), and Telugu-script loans in at least 12.5%.
  - Measured on 50 lines × 2 voices × 2 seeds: English-word loss 20.8% → 12.1%, whole-line CER 10.0% → 7.1%, duration more predictable, native spans unchanged [gap-2, M scratch].
  - Naturalness is not yet rated. The blind kit is ready in `scratchpad/gap2/ab/`.
- `SPEC.md:221` ("Latin is what the model is trained to speak") is wrong [gap-2 E2].
- The translator emits this contract from step 1. Until the maintainer's blind listen (D6), the TTS can be fed either form from the same output: the `english` map rebuilds the Latin-English version, so the listen needs no second translation.

**Duration control** (Chatterbox has no duration argument [08 F21]), in this order:
1. Text tiers chosen by predicted duration (§4.5).
2. Choosing among N takes by duration fit.
3. Mel interpolation at 1.0–1.2× in `vocode`: CosyVoice's method, already in `torch_common.py:264`.

cfg is **not** on this list. It is fixed per voice by ear (§3.7 step 4), and the duration estimator is calibrated at that value. Changing it per line would make a speaker's accent and pitch spread vary from line to line (cfg 0.3 is 6–9% slower than 0.5, and cfg 0 is 21–27% slower with a wider pitch spread) [gap-4 E3, decision 2; gap-5 E2].

**Watermark order.** `vocode` applies the PerTh watermark as its last step (`torch_common.py:282`), and ADR-014 keeps it. This plan adds engine-side processing after vocoding: the +1 st shift (§3.7), pause insertion and silence compression (§3.10), EQ, re-reverb and gain (§3.11). **Move `apply_watermark` to the end of the engine's DSP chain**, just before the PCM is cached and sent. Then check detection with the perth detector on the engine output and on an offline render of the app's Web Audio chain (fades, gain ramps, limiter), and record both in an ADR-014 amendment.

**Take generation:**
- **N=2 takes in one batched T3 decode.** Widen `_t3_tokens` from 2 rows to 2N rows (N conditional plus N unconditional). The ratio is about 1.2× one take for N=2, and 1.3–1.4× for N=3 [gap-1 §2, M scratch, medium-low confidence: heavy swap during the run].
- Remove the per-step host sync (`int(nxt) == eos` every step). Check EOS every k steps and trim.
- S3Gen still runs per take.
- **A take that hits the token cap without EOS is a failed take.** Today `_t3_tokens` only logs it (`torch_common.py:239`) [04 F13]. The cap in `_take` (`session.py:534`) is about 2.4× the target.

**Speed arms (MIT, need a fetch approval):**
- `s3gen_meanflow` at 1–2 steps (ResembleAI/chatterbox-turbo @749d1c1a);
- ADR-014's 6-step CFM, which is waiting on the maintainer's listening check;
- `s3gen_v3.safetensors` (@5bb1f6ee, about 0.53 GB fp16) as a fidelity arm.
- The fork's loader needs a filename parameter [sota §4; 06 B8, B9].

**Medium-term: re-train on the V3 T3.** `t3_mtl23ls_v3` @e2d6902d, MIT, same tensor shapes.
- Arms: full fine-tune and LoRA, each with both `s3gen.pt` and `s3gen_v3`.
- Data:
  - all 47,208 IndicVoices-R Telugu rows (today uses 33,287);
  - Rasa Telugu for emotion (CC-BY-4.0, gated, auto);
  - FLEURS, with its 104 Latin rows transliterated or dropped so `[te]` stays single-script;
  - a "fullcase" tokenizer; ZWNJ stripped [gap-2 A6, E1, E4; 06 R6; 03 R10].
- Expectation: unseen-speaker similarity up to about +0.07 [06 C2], a more native accent, and CER must be watched (the Singlish fine-tune raised WER 11.48 → 15.39).
- V3's "accent preservation" claim is unmeasured and could mean more English-accent carry-over [06 new 4].
- Needs an NVIDIA or rented GPU.

**Fallbacks** only if the V3 retrain stalls: VoxCPM2 or Qwen3-TTS Telugu fine-tunes. Both lost to Chatterbox cross-lingually [06 E9].

**Rejected:**
- OmniVoice, F5-TTS, MaskGCT, Vevo, IndicF5 family (NC or unclear inheritance);
- IndexTTS 2/2.5 (custom licence; it also cannot be used as a teacher);
- Seed-VC (GPL) [06 E1, E2].

### 3.9 Voice conversion

**None.**
- Chatterbox VC is the same S3Gen stage, so it adds no timbre [06 B7].
- Post-hoc VC raised similarity but cost intelligibility in the literature [04 F9, F17].
- kNN-VC had the highest similarity alongside unusable WER [06 F7].
- A converter trained on each speaker's English would plausibly raise WeSpeaker by pulling Telugu toward English phonetics, which games the metric [gap-5 §5].
- RVC/Applio (MIT) is deferred until the acoustic lane (§3.7) is exhausted, and would then need CER and accent gates plus approval.

### 3.10 Timing and placement

**Placement (`timing/planner.py`; amends ADR-017):**
1. **Place the whole sentence over its span.** It may drift across internal pauses under 1 s within the existing lag limits (0.6 s, or 1.0 s before a long pause) [gap-3 §4; 04 F2].
2. **At a hard break** (`SourceUnit.breaks`):
   - synthesise the sentence once;
   - find the Telugu pause (silence at a Telugu comma, or `torchaudio.functional.forced_align`, already locked);
   - insert silence there to meet the English pause.
   - Breaks are placed by **Telugu fluency, not English content** [gap-3 §3; Federico's fluency criterion, ESTsoft's LLM pause step, NAIST CWMT].
   - Claude's `pieces` (§4.4) supply the break points.
3. **Soft anchors** (0.3 s pauses) are used only when a Telugu break falls near the same time share, scored as duration match × break plausibility [gap-3 §4].
4. **Asymmetric onset cost.**
   - Split `w_lag` into `w_early` ≈ 2 × `w_late`, and spend `lead_max` 0.3 s only into real silence [08 R4].
   - The asymmetry comes from same-language lip-sync studies (early audio is detected at about half the offset of late audio), so it is inferred for dubs [08 F11].
5. **Windowed lookahead.** Evaluate about 8 lines or 30 s instead of one line, so slack flows across gaps and rate changes between neighbours are minimised. The engine is minutes ahead, so this is free [02 rec 3].
6. **Rate:**
   - Keep `speed_cap` 1.2. It matches VideoDubber's and LSST's ±20% and Descript's natural window [08 F15; 03 F6].
   - Add an absolute ceiling of min(1.2, 7.5 aksharas/s ÷ voice rate). The 7.5 figure is inferred from Kannada read-speech rates, so check it on native Telugu podcasts [08 R7].
   - **Keep the 1.0 floor.** Underfill is textual and pause-driven. Slowing is the less tolerated direction (MOS 4.14 at 0.90× vs 4.37 at 1.10× [08 F13]) [gap-4 decision 2].
   - An A/B at 0.92–0.95 on speech-dense lines only comes later.
7. **Uneven compression.** Before speeding speech frames, compress the take's internal silences, within a floor of 120 ms each. Amazon found non-uniform scaling beats uniform at 1.1–1.4× [04 N2]. Chatterbox has no phoneme durations, so this is an A/B.
8. **Freezes** stay a metered last resort (0.6 s per event, 1 s per minute).
   - **Video slow-down** stays out of the normal path. First probe `setPlaybackRate(0.9)` in the Tauri WKWebView with the window visible and the video playing, and record the `getCurrentTime()` slope.
   - The docs say unsupported rates round toward 1, and the Chromium probe only read back values on an unstarted player [08 F18; 09 E1].
   - Talking-head content is the least tolerant of speed change [08 A3].
9. **Interruptions** (`cut_off`): the unfinished Telugu stays unfinished, and different speakers may overlap as far as the source did. The planner already allows that overlap.

**Metrics** (added to `maata-bench`, `verify-mac.sh` and ADR-017's reopen criteria):
- **speech-level end error:** the dub's last voiced frame minus the end of the source's last pyannote speech turn inside the unit, not the unit span's end. That is consistent with §4.3/§4.5, which size lines to speech time, so a line that ends early only because the source paused at its end isn't counted as an error;
- **speech-level** overlap from pyannote turns, not unit spans. The 0.844 IoU is not comparable with Brannon's 0.658 [08 F1];
- seconds per minute of source speech with no dub (today 8.2);
- anchor onset error;
- smoothness: already computed as `rate_step_p90` and `rate_step_ok_share` in `planner.stats()` (`planner.py:237`). Report it; no new metric is needed;
- the intelligibility ratio (1 − CER at the applied rate) ÷ (1 − CER at 1.0) for lines above 1.1× [10 R5].

**Acceptance.** Today's build already meets two of the earlier targets: onset lag p95 is +0.22 s and rate p90 is 1.073 [08 F1]. They stay as **regression guards**, and the improvement targets are measured against the committed step-0 baseline on the same fixtures.
- Regression guards: onset lag p95 ≤ 0.3 s; rate p90 ≤ 1.1; `rate_step_p90` not worse than baseline; freezes ≤ 1 s per 10 min.
- Improvement targets:
  - silent-while-speaking ≤ 3 s per minute (baseline 8.2);
  - speech-level end error: median within ±0.3 s, and the share of lines ending more than 1 s early at speech level at least halved against baseline;
  - the long-line (> 8 s) speech-level end-error median at least halved against baseline (unit-level baseline −1.98 s [08 F1]);
  - a blind sync rating better than the baseline build, ≥ 60% preference.

### 3.11 Mixing and acoustics

**Engine (numpy/scipy/pyloudnorm, all locked):**
- **Per-speaker loudness.** Measure integrated loudness over the speaker's own diarised speech, as a home-made dialogue gate (pyloudnorm has none), and match the dub to it.
  - Per-line gain of ±3 dB follows the source line's energy relative to the speaker's mean [07 R2; Brannon energy r 0.381].
  - True peak ≤ −1 dBTP via a 4× oversampled check (pyloudnorm has no true-peak meter [07 F]).
  - Never hard-code −14 LUFS; that figure is unverified [07 C7].
- **Spectral match.** A long-term spectrum EQ toward the source, smoothed and capped at ±6 dB. Match the original's quality "or better" (Netflix [07 C1]), not its flaws.
- **Light re-reverb** only where a blind DSP RT60 estimate exceeds about 0.3 s: a synthetic exponential RIR in numpy. Rec-RIR's checkpoint is excluded because EARS is non-commercial [07 C5].
- **Room-tone statistics** (option B, §6): the noise level and an LPC or third-octave envelope of each scene's non-speech gaps. These are *statistics only*, never samples. Gaps flagged as music or singing (§3.4) are excluded.
- The PerTh watermark is applied after all of the above (§3.8).

**App (`app/src/lib/sync.ts`, Web Audio):**
- 5–10 ms fades on every unit;
- 30–80 ms equal-power crossfades where units overlap;
- all gain changes as `setTargetAtTime` ramps;
- one master limiter (`DynamicsCompressorNode`);
- a room-tone node (shaped noise, 20–35 dB below dialogue) that keeps running through pauses and seeks [07 R3, R8].

**Evidence.** Industry keeps a bed or matches the ambience (ElevenLabs, HeyGen, Sarvam, Netflix, iZotope's synthesised Ambience) [07 A3, C1, C3; 03 N1]. The measured benefit for native listeners is not significant [07 A1]. So these are cheap polish steps, and the room tone is A/B-gated.

**Acceptance:**
- per-speaker loudness difference ≤ 1 LU;
- long-term-spectrum distance down against the dry dub;
- a blind A/B (dry vs room tone vs room tone plus matching) with native listeners, only *after* the timing fixes land.

### 3.12 Playback

The muted IFrame and `SyncEngine` scheduling stay, with these changes:
- **Pacing (`app/src/lib/buffer.ts`, `session._send_ready`).**
  - The wait before play is lead ÷ r in wall-clock time.
  - The engine keeps working past the lookahead while paused when r < 1.25.
  - Today a 480 s lead at r = 0.59 runs out after about 19.5 min [gap-1 corr. 2–3].
  - Add a "prepare the whole video" mode for long videos. PCM is already cached per unit on disk (`pcm/<id>.npy`, float16).
  - **Memory in whole-video mode.** `_evict_raw` evicts only units more than 60 s *behind* the playhead (`session.py:657`). Voicing a 2.9 h video ahead would keep about 1 GB of 24 kHz float32 PCM in `self.raw`, and about as much again as decoded `AudioBuffer`s in the WebView. Evict *ahead* too: keep only the window [playhead − 60 s, playhead + lookahead] in `self.raw` and in the app, and reload the rest from disk on demand (the `_resend_near` path already does this after a seek).
- **Seek into untranslated video** (new; today `seek()` resets the ASR and diarization cursors and the planner, `session.py:176`):
  1. **Cancel or deprioritise Claude work outside the new window.** Queued scene calls whose scenes end before the seek point or start beyond the lookahead horizon are dropped. In-flight calls for those scenes get SIGINT; their partial usage is lost, but they stop holding 2–3 concurrency slots. Lines already validated stay in the line cache (§4.7), so seeking back costs nothing.
  2. **A short first scene.** The first scene after the seek point is cut at the first sentence end at or after 20 s and no later than 30 s, like the start-up scene. Normal 60–150 s scenes follow.
  3. **Context without a translated past.** When the previous scene isn't translated, `context_before` carries its last 3 English lines only, with no Telugu. The brief supplies global context.
  4. **Priority.** Diarization and ASR at the new frontier take GPU priority 2 (§5.3). The voicer takes priority 1 as soon as the first line is translated, and the throughput governor holds it at N=1 until the lead is back at target.
  5. Speakers not yet cloned use today's preset fallback until their clone lands.
  - **Acceptance:** time from a seek into unprocessed video to the first dubbed audio ≤ 60 s at p95 over ≥ 10 scripted seeks on the podcast, with speakers already cloned. `verify-mac.sh` runs the seeks and commits the numbers.
- **Output latency.** Keep `outputLatency` (WebKit reports it on macOS) and add a per-device user offset (Bluetooth about 0.18 s) [08 R10].
- **Compliance under any scope** [gap-6 §3]:
  - **Referer.** Required Minimum Functionality asks desktop WebView clients to send `https://<app id>` as the Referer [gap-6 Y3]. The Referer comes from the page's origin, and per ADR-006 the engine serves the UI from `http://127.0.0.1:<port>`, so this is **not a `player.ts` edit**. It means serving the UI from Tauri's custom protocol (`https://<app id>`-style origin), which changes ADR-006, the CSP and the WebSocket origin check. ADR-006 chose loopback because a custom scheme risked player errors 152/153. So: first a probe that embeds the player from the custom-protocol origin in the Tauri WKWebView and plays a video, then an ADR-006 amendment. It is a shell and serving change (`app/src-tauri`, `server.py` origin checks), not a UI one;
  - no overlays on the player;
  - volume and caption controls stay usable;
  - the player stays at 200 × 200 px or larger;
  - a privacy notice that transcripts go to Anthropic through the user's own CLI.
- **UI:**
  - a session report (skipped ids, music lines not dubbed, lines re-synthesised shorter, provisional takes not replaced in time, QA flags, freezes, maximum rate, Claude usage share for the session);
  - a **"this line is wrong" hotkey** that re-translates through the reviewer, re-synthesises ahead of the playhead, saves the fix to the channel glossary and logs the flag [03 R6];
  - clear Claude-state messages: not signed in, usage limit with its reset time, CLI too old, and **"Claude Code didn't start. If Claude Code is open in another window on the same version, it can block Maata (known issue #91987); retrying in N s"** when the startup watchdog fires (§4.9).

### 3.13 QA loop

**Where QA runs: a separate worker process, not threads in the engine.**
- **Why.** T3 decode is limited by per-step overhead, not memory bandwidth: 13–20 ms per step against about 3 ms of memory traffic, with a host sync every step [gap-1 §2]. So TTS speed depends on the engine's CPU dispatch thread. onnxruntime spreads over every core by default, and IndicConformer's RNNT greedy loop, CTC alignment, `pyin` and loudness code are Python that would contend for the GIL with the T3 loop if run through `asyncio.to_thread`.
- **Design.** `qa/worker.py`: one `spawn`-started worker process (`ProcessPoolExecutor(max_workers=1)`), holding the ONNX session with `intra_op_num_threads=4` and `inter_op_num_threads=1`, and a lowered scheduling priority (`os.nice`). It runs IndicConformer, CTC alignment (a small numpy Viterbi, or torchaudio's `forced_align` if importing torch there is cheap enough), `pyin` and the other DSP, and loudness, EQ and room statistics. Audio goes in as float32 arrays; results come back as small dicts. WeSpeaker similarity stays in the engine, on the GPU under the scheduler (it is tiny).
- The 2–3 `claude` Node processes are already separate processes and mostly wait on the network.
- **Acceptance (steps 5 and 6).** With QA running on every take *and* 3 Claude calls in flight, T3 ms per token and end-to-end r stay within 10% of the idle figures on the same fixture. Otherwise fall back to QA on the chosen take only, plus flagged pairs.

**Per take:**
- **Selector:** `ai4bharat/indic-conformer-600m-multilingual` @e9b71b36.
  - MIT, gated (auto); the maintainer hasn't accepted the gate yet [10 A2].
  - **A new runtime dependency on the Mac.** onnxruntime appears in `uv.lock` (1.30.0) only as a dependency of faster-whisper in the `cuda` extra; it is not in the `apple` extra and not installed in `engine/.venv` [04 S2]. (Report 10's "already locked" is wrong for the Mac.) The model card installs `onnxruntime==1.20.1` [10 A2], plus `onnx==1.20.1` and `onnxruntime-gpu` (the reviewer's read of the card, 2026-09-24), while the lock already has onnx 1.23.0 through s3tokenizer. So onnxruntime must be added to the `apple` extra with approval (D9), pinned in `uv.lock`, and tested with the pinned onnx.
  - **No remote code.** The card loads with `AutoModel.from_pretrained(…, trust_remote_code=True)`, which fetches code at runtime and breaks "never let a library download on its own". Vendor its `model_onnx.py` (MIT) into `engine/src/maata_engine/qa/indicconformer.py` at the pinned commit, list the ONNX files with sha256 in `models.lock.json`, and load them from local paths with `HF_HUB_OFFLINE=1`. A test loads the model with the network blocked.
  - Third-party Kathbath Telugu CER is 3.1. It writes English words in Telugu script (0% kept in Latin), which matches the `spoken` field directly [10 A7; gap-2 D2].
  - Its RNNT head gives the transcript; its CTC head plus forced alignment gives per-word scores [10 B4].
  - Its training corpora and their licences are **unchecked here**; they are recorded in the pin ADR before adoption, as for every QA and enhancement model (§3.2).
- **Checks:**
  - CER against `spoken`, after NFC normalisation and removal of zero-width characters and punctuation, with per-word tolerance for loan-spelling variants;
  - skipped words and audio after the last word;
  - WeSpeaker similarity (pyannote's own embedding, no new model) scored at matched length;
  - natural duration vs the estimate (flag under 0.6×);
  - token-cap hits;
  - the take's aksharas/s vs the voice calibration.
- **Selection** [04 F15; gap-5 E1, E2; 10 R6]:
  - Meta's "drop takes below 0.75 × the best similarity" was tuned at N=32 on WavLM-SV. At N=2 with WeSpeaker, the per-line seed spread is only 0.04–0.05, so that filter would almost never fire, and best-of-2 buys only about +0.02 in similarity.
  - So at **N ≤ 3 the pick is by CER plus duration fit** (best average rank). Similarity is only an **outlier guard**: reject a take more than 2 SD below the speaker's running median similarity (at matched length), per 10 R6.
  - Never optimise one score [06 D13].

**Retake ladder** (the voicer never waits on Claude):
1. N=2 batched takes.
2. If both fail: a third seed.
3. If still failing: **ship the best take as provisional and flag it**, and queue an asynchronous Claude rephrase (a simpler wording or another form of the term) through the translator's queue, with a deadline: the new take must be voiced at least 60 s before the playhead reaches the line. If it arrives in time *and* its take passes QA *and* fits the span already planned for the provisional take, it replaces the provisional audio (the same replace path as the "this line is wrong" hotkey). Otherwise the provisional take stays, flagged.
4. Never skip silently.

This extends `RESYNTH_SHARE` (10%) to a QA budget governed by throughput (§5).

**Per scene (Claude):** the coverage review of §4.6.

**Reporting only, never in the product path** (a different family avoids same-family selection inflation, which recovers 2–3× more oracle headroom than cross-family pairs [10 B1]):
- **BuzzASR/telugu** @598f495e: MIT, a Whisper-large-v3 fine-tune (FLEURS CER 13.53, vendor), converted to MLX and run on the pinned mlx-whisper. **Bench only** (about 3.1 GB on the GPU); if it is ever used on flagged lines in the app, it is loaded on demand and unloaded after. Its training corpora and their licences are recorded in its pin ADR before use (the CV25 vs FLEURS gap suggests Common Voice is among them [10 A1]);
- or Omnilingual CTC v2 (Apache-2.0, needs fairseq2).
- SraVaani and svanita are not cross-family [10 B2].

**Never gate on:** UTMOS, NISQA (NC weights, runtime download), DNSMOS (unpinned download), or emotion-embedding cosine.
- **IndicMOS** (CC-BY-4.0) may flag the bottom 5% within a voice only [10 C3, R8].
- **Distill-MOS stays "ask"**: its package is MIT with bundled weights, but its `xls_r_sqa` dependency's licence is unchecked and its card names research as the primary use [10 C2]. Use IndicMOS alone until that is resolved.

**Acceptance for take QA:** every take with CER > 50% or a token-cap hit that ships is **flagged in the UI and counted** (100% of them); their share is reported per session, with a target of ≤ 1% of lines on the podcast.

**Human protocol:**
- a MUSHRA-DG-style sheet on fixed 60–90 s clips;
- blind A/B win rates between builds;
- Bradley-Terry scoring for multi-arm rounds;
- SpeechArenaBench Telugu sentences (MIT) as a fixed TTS regression set;
- ratings committed, and used to calibrate CER and similarity thresholds [10 R9; 03 R9].

---

## 4. Translation design with the Claude CLI

### 4.1 Unit of translation

- **Line unit.** The sentence-complete unit from §3.4, id-keyed. Evidence:
  - every SOV-target system found translates sentences [gap-3 E5];
  - Google translates caption sentences [02 Y7];
  - fragments produced 6 serious errors in 13 cases [gap-3 E3];
  - Telugu is verb-final.
- **Call unit: a scene.**
  - Contiguous units covering at most 150 s of video and at most 30 units, cut at a speaker turn or a pause of 1 s or more.
  - **The first scene is 20–30 s** (cut at the first sentence end at or after 20 s), at video start and after a seek into untranslated video (§3.12), so first audio isn't held up [gap-1 §5]. The second scene is about 60 s; normal scenes follow.
  - Context:
    - the previous scene's last 3 lines (English plus the *final chosen* Telugu) as read-only context; English only when the previous scene isn't translated (after a seek);
    - the next scene's first 2 English lines, marked "context only, never translate".
  - Id-keyed JSON prevents the "translated the next line too" failure of ADR-018 Round 1, which came from windowed free text. This is confirmed by bake-off, not assumed [09 R1].
- **Split-back** only for units with `breaks` (about 2% on talk content): Claude returns `pieces` at natural Telugu breaks, and a `moved` list when content crosses pieces [gap-3 §2].

### 4.2 Call graph per video

"Scene model" below means the model the step-1 bake-off picks for that call type: **Sonnet 5 by default**, Opus 5.5 only if it wins by the margin in §7.1.

| Call | When | Model / effort | Input | Output |
|---|---|---|---|---|
| **Brief v0** (no call) | At open | — | title, channel, description, chapters, tags, and the pre-pass speaker count and talk shares, placed as-is in the system prompt | lets scene 1 start without waiting for brief v1 |
| **Brief v1** | After the 10-min pre-pass transcript, in the background | scene model / medium | title, channel, description, chapters, tags, English transcript so far with speaker ids and talk shares | topic; register decision; speakers (display name, gender, role, how they address each other and the audience: మీరు/నువ్వు); glossary (term → keep-in-English with a Telugu-script spelling, or a fixed Telugu rendering, plus a pronunciation note); named entities; idioms, puns and jokes flagged; numbers convention; high-confidence ASR corrections only. Swapped in at the next scene boundary |
| **Brief v2** | After the full-video ASR finishes in the background | same | full transcript | a diff of v1. Swapped in at a scene boundary; one prompt-cache write |
| **Scene translation** | Per scene, 2–3 concurrent, only for lines not in the line cache (§4.7) | scene model / medium (bake-off) | §4.3 | §4.4 |
| **Fit** | Only when a scene still has lines outside the band after selection (before voicing) | scene model / low (bake-off) | those ids, the measured overflow or underflow in aksharas, the English | new tiers for those ids only |
| **Coverage review** | Per scene, after selection | Sonnet 5 / high, or Opus 5.5 / low (bake-off). If the scene model is Opus, Sonnet reviewing also spreads usage across the per-family weekly limits and may decorrelate errors [05 A4, inferred] | English plus chosen Telugu per id | per id: class C / m / P / E, missing English words, additions, error type |
| **Re-translate** | Ids classed P or E | scene model / medium | English, the missing words named, the brief | new `full` (+ tiers). Never a rewrite of the Telugu: rewrite passes are where omissions appear [05 C1; gap-4 E7] |
| **Rephrase** (async) | Ids whose takes failed TTS QA twice, or ran far over after synthesis; queued by the voicer, never awaited | scene model / low | English, the failing `spoken`, the QA finding (skipped words, CER, overrun in aksharas), the deadline | a simpler `full` and tiers for those ids; dropped if the deadline has passed |

**Channel persistence.** The glossary and speaker register are cached per channel. The brief is cached by transcript hash [09 R2]. Validated lines are cached per video (§4.7).

### 4.3 Prompt structure

**System prompt: byte-identical for the whole video**, so separate `-p` processes in one fixed working directory share the prompt cache (1-hour TTL within plan usage) [05 A8]. It contains:
1. **Role:** dubbing English speech into spoken Telugu, as said aloud on Telugu YouTube explainers and podcasts.
2. **Style guide** [05 D3, R4; 03 F4, N1; ADR-017]:
   - sishta vyāvahārika: standard spoken Telugu from the central dialect, neutral between Andhra and Telangana;
   - no grāndhika vocabulary (యొక్క, తద్వారా, మరియు);
   - no English quota: English only where educated speakers really say it (names, brands, technical and modern terms, established loans);
   - English verbs as the bare stem plus చేయు/అవు (the maintainer to confirm as a native speaker);
   - drop pronouns the verb marks;
   - spoken verb forms;
   - fillers translated by meaning;
   - the register pair from the brief;
   - idioms and puns become a Telugu equivalent or a paraphrase, never a calque [05 C8];
   - numbers as spoken Telugu words;
   - acronyms as said (ఏఐ);
   - **every fact, name, number, negation and question kept; nothing added; no glosses or brackets.**
3. **Script contract** [gap-2]: `spoken` is Telugu script only. Every English word is spelled the way Telugu speakers write it, and listed in `english` with its word index and Latin form.
4. **Length contract** [04 N1; 05 C2; gap-4]:
   - each line has a target in aksharas, computed locally; Claude's own sense of length is not trusted;
   - `full` is the natural complete line, with no length pressure;
   - `concise` and `very_concise` cut only fillers, hedges, repetitions and pronouns the verb marks;
   - `fuller` restores what the speaker actually said (hedges, repetitions, discourse markers, full verb forms) and **adds no fact**.
   - Examples for the short tiers must be visibly shorter than their English [05 C2].
5. **About 16 original example pairs**, rewritten to the Telugu-script contract. Today's `SHOTS_COLLOQUIAL` uses Latin. Versioned as `SHOTS_VERSION` and `PROMPT_HASH`, reviewed by the maintainer.
6. **The video's brief** (glossary, speakers, register). It changes only at brief swaps (v0 → v1 → v2), so it belongs in the cached system prompt. **Glossary additions from scenes do not go here**: they travel in each later scene's user message (`glossary_recent`) until the next brief swap folds them in. Otherwise every merge would rewrite the cache.
7. **The JSON contract** in words, plus `--json-schema` enforcement.

The `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` split applies only to direct API calls, so it is not relied on here [05 A8].

**What is cached, and what is not.** The cache is per model and per identical prefix [05 A8]. So:
- The scene, fit, re-translate and rephrase calls share one system prompt, with the call type stated in the user message, so they can share one cache per model.
- Brief calls and review calls have their own system prompts and caches. A review on a different model from the scene calls can't read the scene calls' cache.
- Whether `--json-schema` enters the cached prefix is unknown; if it does, each schema is a separate cache.
- Budget one cache write per (model, system-prompt version, schema) per hour of TTL, plus one per brief swap. The step-1 usage run measures cache-read against cache-creation tokens per call type before any cost or quota estimate relies on caching.

**User message per scene:**

```json
{"scene": 7, "call": "scene", "glossary_recent": [{"term": "...", "spoken": "..."}],
 "context_before": [{"en": "...", "te": "..."}], "context_after_en": ["...", "..."],
 "lines": [{"id": 412, "speaker": "S2", "to": "S1", "start": 603.1, "end": 609.8, "speech_s": 5.9,
            "target_aksharas": 33, "want": ["full", "concise"], "breaks": [], "cut_off": false,
            "delivery_hint": {"energy_rel": 1.3, "rate_rel": 0.9, "pitch_rel_st": 1.2},
            "en": "..."}]}
```

**How the fields are filled:**
- `target_aksharas` = speech-time seconds (VAD from diarization, not the raw span) × the voice's calibrated rate × 1.0.
- `want` is set locally *before* the call:
  - Predict `full` from the English: syllables × a ratio *k* of aksharas per English syllable.
  - **k is a weak prior, updated online.** The 20:06 trace's 1.22 is too low as a starting value: that Gemma output was about 20% shorter per unit of meaning than faithful human translation, partly because of P and m drops, and it counted Latin English by syllables [gap-4 E1, E6]. The Telugu-script contract adds about 9% to unit counts [gap-2 C]. Faithful formal FLEURS translation runs at 1.62 [gap-4 E6]. So start at **k ≈ 1.4** (inferred, between those bounds), then replace it per speaker with the running median of measured `full` aksharas ÷ English syllables from each validated scene. Re-derive the starting value from the step-1 bake-off output.
  - Ask for `concise` and `very_concise` when the prediction exceeds 1.10 × target.
  - Ask for `fuller` when it falls below 0.85 of a speech-dense slot.
  - This keeps output lean [gap-1 §5]. A prior that runs low shows up as more fit calls; `units.jsonl` logs predicted vs actual `full` length, so that is visible.

### 4.4 Structured output (`--json-schema`, draft-07, no `format` keyword)

```json
{"type":"object","additionalProperties":false,"required":["lines"],
 "definitions":{"wording":{"type":"object","additionalProperties":false,"required":["spoken","english"],
   "properties":{"spoken":{"type":"string","minLength":1},
                 "english":{"type":"array","items":{"type":"object","additionalProperties":false,
                   "required":["i","en"],"properties":{"i":{"type":"integer","minimum":0},"en":{"type":"string"}}}}}}},
 "properties":{
  "lines":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["id","full","delivery"],
    "properties":{
      "id":{"type":"integer"},
      "full":{"$ref":"#/definitions/wording"},
      "fuller":{"$ref":"#/definitions/wording"},
      "concise":{"$ref":"#/definitions/wording"},
      "very_concise":{"$ref":"#/definitions/wording"},
      "pieces":{"type":"array","items":{"type":"string"}},
      "moved":{"type":"array","items":{"type":"string"}},
      "unfinished":{"type":"boolean"},
      "delivery":{"type":"object","additionalProperties":false,"required":["emotion","energy"],
        "properties":{"emotion":{"enum":["neutral","happy","sad","angry","surprised","serious"]},
                      "energy":{"enum":["low","mid","high"]},"question":{"type":"boolean"},
                      "emphasis":{"type":"array","items":{"type":"integer"}}}}}}},
  "glossary_additions":{"type":"array","items":{"type":"object","required":["term","spoken"],
    "properties":{"term":{"type":"string"},"spoken":{"type":"string"},"keep_english":{"type":"boolean"}}}}}}
```

**Deterministic validators** (a pure module, unit-tested, in cloud CI too):
- every requested id appears exactly once;
- `spoken` has Telugu script, and no Latin, digits, ZWNJ/ZWJ or brackets;
- `english[i]` points at a real token;
- the tiers are strictly ordered in aksharas;
- `pieces` join back to `full.spoken`, with piece count ≤ breaks + 1;
- glossary terms appear in their fixed spelling;
- negation count parity: English not / never / no vs Telugu negative morphology, as a heuristic flag, which would have caught units 73 and 82 [gap-3 §2];
- numbers in the English are present as spoken numerals (flag only).

Also: treat `success` without `structured_output` as a failure; drop `claude_cli._json_in`'s unvalidated fallback, or validate its result locally [05 A12]. Captions, `latin_ratio`, lint and CMI are rebuilt from `spoken` plus `english`, so there is one source of truth [gap-2 §1].

### 4.5 Timing budgets and selection (local, deterministic)

- For each candidate, predicted natural duration = the keyed `DurationEstimator` on `spoken`. `count_units` counts Telugu script only, which gives about 9% more units than today's count, so each voice is re-seeded [gap-2 C].
- **Band rule** [gap-4 decision 1]:
  - take the most complete candidate whose prediction lies in [0.85, 1.10] × speech-time target;
  - otherwise the closest from above, if `planner.evaluate()` absorbs it within 1.2× and the lag limit;
  - otherwise the closest from below.
- This avoids HOMURA's "most complete that fits" undershoot (median 0.87) [gap-4 E7].
- Fill is measured against **speech time**, so emphatic pauses are not "filled" [08 F1 caveat].
- **Register and code-mixing monitoring:** log CMI per scene. Warn only well outside about 20–40%; the reference sets are CoSTA's Telugu podcast 32.14% and IndicVoices 25.5%. Never enforce a quota [05 D1, R4].

### 4.6 Glossary, speaker consistency and coverage

- The brief fixes each speaker's gender and role (Telugu verb agreement) and the address form per pair. It is cached per channel. `glossary_additions` from scenes are held in `glossary_recent` (sent in each later scene's user message) and merged into the brief only at the next brief swap, with conflicts resolved to the brief (§4.3).
- **Coverage review** (Sonnet 5): per id, class C / m / P / E [gap-4 E5], the missing English words, additions, and negation / number / hallucination errors.
  - P and E lines get one re-translation with the missing words named.
  - The 20:06 trace's labels (`scratchpad/gap4/coverage.py`) are the first regression fixture.
- **No generic "refine everything" pass.** Refinement mostly improves fluency, not adequacy, and pulls output toward the refiner's style [05 C4, R8].

### 4.7 Persistent vs one-shot process

- **One sealed one-shot `claude -p` per call** (today's model), run through `asyncio.to_thread` under a semaphore of 2–3.
- One **fixed** empty working directory per engine run (not `mkdtemp` per instance) and a byte-identical system prompt, for cross-process cache hits [05 A8].
- A warm `--input-format stream-json` process saves roughly 1.5 s per call (an anecdote) but accumulates history. Adopt it only if the measured spawn overhead matters, and recycle it per scene [05 R5; 05 A9]. With about 70–90 calls per video-hour, the saving is about 2 minutes, which is not worth the risk now.

**Line cache (re-watches spend nothing).**
- Every line that passes the §4.4 validators is stored in `<cache>/<video_id>/lines.jsonl`, keyed by (video id, hash of the unit's English text and speaker, `PROMPT_HASH`, model). The entry holds the tiers, `english` map, `delivery`, pieces, coverage class and the brief version it was made under.
- Keying per line rather than per scene means a different scene cut (after a seek, or a different start) still hits.
- Before a scene call, cached lines are removed from `lines` and passed only as context. A scene with every line cached makes no call. Tiers that a new calibration asks for and the cache lacks go to one fit call.
- A `PROMPT_HASH` or model change invalidates the cache by construction. Brief v2 does not; lines made under v1 are reused.
- Like the dub PCM, it lives in the per-video cache directory and is cleared with it.

### 4.8 Model, effort, latency and usage budget

**Model** [05 B1, A7; 05 new 3]:
- **Default `--model claude-sonnet-5`, with `--fallback-model claude-opus-5-5`.** If the bake-off picks Opus 5.5 for a call type (it needs CLI ≥ 2.1.280), the two swap. The fallback covers overload and unavailability, not usage limits.
- On a model-specific weekly limit ("You've hit your Opus limit" or the Sonnet equivalent), re-run the scene with the other model; on a session or weekly limit, pause (§4.9).
- **Never** Fable (it bills credits in `-p` without asking) and **never** fast mode (credits-only on Pro/Max).
- Opus 5.5 cannot run with thinking off, and its default effort is medium. M-GATE reports that reasoning reliably helps translation [05 B2].
- There is no public EN→TE score for any Claude model, so the **step-1 bake-off decides**, and a tie goes to Sonnet 5 (§7.1).

**Budget per hour of video (inferred; to be measured and committed):**

| Item | Estimate | Basis |
|---|---|---|
| Calls | about 70–90 on a first viewing: brief 2, scenes about 25, fit about 10–25, review about 25, re-translate and rephrase about 5–15. **About 0 on a re-watch** (line cache) | 150 s scenes |
| Input tokens | about 8–12k static (style + examples + brief) per call. Cached reads only within the same model, system prompt and (possibly) schema, so budget about 4–8 cache writes per video-hour (scene and review prompts on one or two models, plus brief swaps) and treat the rest as reads. About 2–4k uncached scene content per call. So roughly 0.6–1.0M cache reads plus 0.05–0.1M cache writes plus 0.2–0.35M uncached | [05 A8]; Claude's Telugu tokenization is unpublished; GPT-4's Telugu premium is 8.34× [05 corr.] |
| Output tokens (before thinking) | lean 29k–106k (translation) + about 10–20k (review) | [gap-1 §5] |
| Thinking | unknown, possibly the same order as the output | cannot be disabled on Opus 5.5 |
| Serial decode time | lean 7–25 min per video-hour at about 71 tok/s, + about 6 s floor × calls ≈ 7–9 min; **about 5–15 min of wall time with 2–3 concurrent calls**, i.e. about 0.1–0.25 s per video-s | [gap-1 §5]; the 71 tok/s is inferred from "> 30% faster than Opus 5" |
| API-price equivalent, for scale only | about $3–8 per video-hour on Opus 5.5 ($4 / $20 per MTok, cache reads $0.20); about half on Sonnet 5 ($2 / $10) | [05 B1]; a subscription draws plan limits, not dollars. Unverified until the usage run measures cache reads vs writes |
| Plan limits | not published. Five-hour session plus weekly limit, with per-family Opus and Sonnet weekly limits; "raised" on 2026-09-22 by an unstated amount | [05 A4] |

**Measure first (part of step 1's acceptance, §7):**
- a real 1-hour run, with nothing else using the plan during it;
- log input, cache-read, cache-creation (`ephemeral_1h_input_tokens`) and output tokens per call type and model, TTFT and seconds per call;
- read `rate_limit_event` (`utilization` when present, `rateLimitType`, `status`, `overageStatus`) from `--output-format stream-json --verbose`. `utilization` is optional and was absent in the one raw event quoted in #78476 [05 A5], so also record the plan-usage meter the maintainer sees (Claude Code or claude.ai) before and after the run;
- commit the JSON.

**Pass/fail on usage.** The maintainer sets two thresholds from his plan and his weekly viewing: the share of a **5-hour window** and the share of the **weekly windows** (overall, and the Opus or Sonnet window for the chosen model) that one video-hour may use. A worked rule: if he dubs 5 video-hours a week and wants half his weekly allowance left for development, one video-hour may use at most 10% of each weekly window. Step 1 passes only if the measured shares are under both thresholds. If not: shorter output (lean tiers), Sonnet for more call types, larger scenes, or fewer review calls, re-measured.

This also settles lean vs rich output, the concurrency and the scene size [05 R5; gap-1 §6].

### 4.9 Failure handling and sealing

**Version gate at startup.**
- Parse `claude --version` (of the binary Maata actually runs, §4.9 isolation): require ≥ 2.1.280, since Opus 5.5 is either the scene model or the fallback. `--restricted`, if adopted, needs ≥ 2.1.248; schema validation needs ≥ 2.1.205.
- Otherwise show "run `claude update`". Keep `DISABLE_AUTOUPDATER`.
- Log the resolved model from `modelUsage` [05 A1, R5].

**Error classes** (fixes the regexes at `claude_cli.py:60-61`) [05 A10]:

| Message | Meaning | Handling |
|---|---|---|
| `hit your (session\|weekly\|opus\|sonnet) limit · resets …` | usage limit | pause translation, show the reset time. On an Opus-only limit, switch the scene to Sonnet 5 |
| `temporarily limiting requests (not your usage limit)` | transient | back off exponentially. Today it is misread as a quota |
| `Not logged in`, `Failed to authenticate`, `OAuth … expired\|revoked` | not signed in | UI: run `claude auth login` |
| `does not support this model`, `claude_code_version_too_old` | CLI too old | UI: run `claude update` |
| `success` without `structured_output`, schema failure | bad output | retry the missing ids once; then per-line calls; then `unit_skipped` shown in the UI. Never pad, never fall back to English |

**The #91987 lock hang is the default case, not an edge case.** The maintainer develops with Claude Code on this machine, and D3 has him `claude update` that same install. In #91987 (open when checked with `gh` on 2026-09-24, last updated 2026-09-04), `claude -p` hangs indefinitely at startup when an interactive session on the **same CLI version** holds the version-directory lock, and an immediate retry hangs the same way [05 A9]. A watchdog alone would turn his normal workflow into a stream of failed scene calls, and translation has no fallback (D2).
- **Mitigation to evaluate before step 1 ships** (05 open question 9; needs its own ADR, D3):
  1. **A separately pinned CLI for Maata.** Install a second, unmodified Claude Code at a pinned version in a Maata-owned prefix, with its auto-updater disabled, and have the engine run only that binary. The collision needs both processes on the same version, so the engine checks at startup that Maata's version differs from the `claude` on `PATH`, and warns if they match. Sign-in stays the CLI's own flow. Whether this counts as "bundling" for distribution is one of the D18 questions; in personal scope it is his own install.
  2. **If (1) doesn't avoid the lock** (for example if the lock lives outside the version directory): test a separate configuration directory for Maata's CLI, which means one more `claude auth login` in that directory [I, untested].
  3. If neither works: step 1 ships with the watchdog plus a clear UI message, and the maintainer closes interactive sessions while watching. Recorded as a known limitation.
- **Acceptance (step 1):** with an interactive Claude Code session open on the maintainer's usual version, run **50 consecutive scene calls: 0 hangs**.

**Startup watchdog** (still needed for other stalls).
- No stream event within 60–90 s → SIGINT, then SIGTERM, then SIGKILL.
- **Back off rather than retrying immediately** (#91987 comment).
- When it fires, the UI names the likely cause: "Claude Code didn't start. If Claude Code is open in another window on the same version, it can block Maata (known issue #91987). Retrying in N s." (§3.12).
- A roughly 405 s once-per-session stall (#83859) has been reported on macOS with tools; Maata uses none, so whether it applies is unverified [05 A9; gap-1 corr. 7].

**The voicer never awaits Claude.**
- The only Claude consumer on the voicing path today is the inline `translate_candidates` call in `session._dub` (`session.py:566–574`), which runs while holding the shared GPU lock. It is removed.
- A line that needs a rephrase after synthesis is voiced with its best take (provisional, flagged), and a rephrase request is queued to the translator with a deadline: the replacement must be voiced ≥ 60 s before the playhead reaches the line (§3.13). A late rephrase is dropped; its usage is logged.
- Under a Claude call's roughly 6 s floor plus thinking and output time, usage-limit pauses or watchdog stalls, the voicer keeps voicing whatever is translated.
- **Test:** with every Claude call artificially delayed by 90 s (a fake CLI in the test suite), the voicer's `lock_wait_s` and throughput on already-translated lines are unchanged within noise, and no voicer task awaits a translator future.

**Lookahead as the buffer.** The translator runs up to the lookahead horizon, so a slow or failed call delays only lines minutes away. Pacing (§3.12) holds playback if the dub truly runs out. After a seek, calls outside the new window are cancelled (§3.12).

**Sealing:**
- **Flags:** `-p --output-format stream-json --verbose --model <scene model> --fallback-model <the other model> --system-prompt <ours> --tools "" --safe-mode --strict-mcp-config --no-session-persistence --disable-slash-commands --settings '{"fastMode":false}' --json-schema <schema>`.
  - **New, to test:** `--restricted` (≥ 2.1.248). It loads only managed settings and `--settings`, removes command-running tools and WebFetch, and confines file tools. That would also neutralise any persisted user `fastMode`.
  - Verify on the M5 Pro that subscription sign-in still works with it. The CLI reference, fetched 2026-09-24, doesn't say either way.
- **Never** `--bare`: it never reads OAuth, so it shuts out subscription users, and restricting auth methods conflicts with Anthropic's conditions [05 A6; gap-6 A1].
- **Environment:** `DISABLE_TELEMETRY`, `DISABLE_ERROR_REPORTING`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_AUTOUPDATER`, `DISABLE_FEEDBACK_COMMAND`; strip `CLAUDECODE`. Already mostly in `claude_cli.QUIET_ENV`.
- The prompt goes on stdin. **Only text leaves the machine; audio never does.**
- Never read `~/.claude`, the keychain, `CLAUDE_CODE_OAUTH_TOKEN` or `setup-token` output.
- A one-time notice that transcript text goes to Anthropic under the user's plan, linking the data-privacy setting (training on/off; retention 5 years vs 30 days) [05 A11].
- `--max-budget-usd` applies only to API-key users. Offer it as a guard when a user signs the CLI in with their own key.

### 4.10 Policy status of a subscription login from an app

**Quoted from Claude Code "Legal and compliance"** (code.claude.com/docs/en/legal-and-compliance, undated; the product section first appears in a Wayback capture of 2026-08-30 [gap-6 A1]; re-read from the saved copy fetched 2026-09-24):
- For products that run Claude Code: "Customers may not pay for, resell, or intermediate Claude usage". Each end user authenticates with their own API key, subscription credentials or cloud credential. The binary must be unmodified, with no authentication method removed or restricted, and the Commercial Terms must be accepted.
- On developers: products "should use API key authentication", and developers "may not collect, store, or intermediate Claude.ai credentials or session tokens".
- The carve-out: this does not prevent an end user "signing in to the unmodified Claude Code binary with their own Claude subscription".
- On limits: Pro and Max limits "assume ordinary, individual usage of Claude Code and the Agent SDK". Enforcement may come without notice.

**Billing.** The planned move of `claude -p` and Agent SDK use onto API-rate credits is **paused**: support article 15036540, updated 2026-06-16, gives no new date [05 A3].

**Recommended stance** [gap-6 §4–5]:
1. **Now (maintainer's personal use):** compliant. He signs in with `claude auth login` on his own unmodified CLI; Maata spawns it and handles no tokens.
2. **Before any distribution:**
   - the maintainer accepts the Commercial Terms;
   - Maata detects the user's installed `claude`, never bundles or patches it, and directs sign-in to the CLI's own flow;
   - a user's own API key stays the user's choice;
   - no Claude or Anthropic branding in Maata's name or logo (plain-text "runs Claude Code" is allowed);
   - ask Anthropic sales in writing: (a) does a free open-source app that spawns the user's CLI need the Commercial Terms; (b) is batch translation of long transcripts "ordinary, individual usage"; (c) will the paused billing change apply?
3. **Never spend usage credits or overage by default.** Pause with a message instead, unless the maintainer opts in.

---

## 5. Memory and throughput on 24 GB

### 5.1 Resident set

File sizes unless noted. Peaks are from `sota_ranking.md`. The runtime overheads are inferred.

| Component | Where | Today | Target default |
|---|---|---|---|
| Gemma 3 12B 4-bit translator | MLX | **8.03 GB** + KV | **0** (removed) |
| TranslateGemma 4B (formal, on first use) | MLX | about 2–3 GB when used | 0 (formal style moves to Claude) |
| Whisper large-v3-turbo (both passes share weights) | MLX | 1.6 GB (peak about 2 GB on 65 s chunks) | same |
| pyannote community-1 (segmentation + WeSpeaker ResNet34) | MPS | 0.03 GB | same; its embedding also serves QA similarity |
| Chatterbox-Telugu (T3 bf16, S3Gen fp32, voice encoder, tokenizer) | MPS | about 2.2–3.2 GB | same; `s3gen_v3` would *replace* S3Gen (0.53 fp16 / 1.06 fp32) |
| Batched-take KV (0.5B T3, 2N = 4 rows) | MPS | — | small, under 0.3 GB (inferred) |
| IndicConformer-600M (ONNX) + QA worker runtime | CPU, own process | — | about 2.4 GB fp32 at runtime (inferred from 600M parameters; repo 2.56 GB) + about 0.3–0.5 GB for the worker's Python, onnxruntime and numpy (inferred) |
| Engine runtime (Python, torch, MLX, buffers, 16 kHz float32 analysis audio) | — | about 2–3 GB, of which the analysis audio is about 230 MB per video-hour (667 MB for the 2.9 h fixture, decoded whole by `load_audio`) | same |
| Dub PCM held in `self.raw` (24 kHz float32, about 0.35 GB per dubbed hour) | — | up to about 1 GB on a 2.9 h video voiced ahead, because `_evict_raw` evicts only behind the playhead | ≤ about 0.1 GB: a window of [playhead − 60 s, playhead + lookahead], the rest reloaded from disk (§3.12) |
| Decoded `AudioBuffer`s in the WebView | — | about the same as `self.raw` | the same window, ≤ about 0.1 GB |
| Claude CLI processes (2–3 concurrent) | CPU | 0 (unused) | about 0.2–0.4 GB each (inferred) |
| Tauri WebView + YouTube player (excluding dub buffers) | — | about 1–2 GB (inferred) | same |
| **Maata total** | | **about 17–20 GB, plus up to 2 GB of dub PCM on long videos** | **about 11–13.5 GB** |

**Optional arms**, each replacing its stage while benched:
- Qwen3-ASR 8-bit: 2.47 GB, peaks 3.3–4.1;
- Qwen3-ForcedAligner: 1.28 GB, or MFA on the CPU at 0.25;
- Nemotron: 0.2 GB, + ReDimNet 0.05;
- BuzzASR: about 3.1 GB, **bench only**; if ever used on flagged lines in the app, loaded on demand and unloaded after, never resident.

Worst case in the product (Qwen3-ASR at its long-form peak plus the Qwen3 aligner plus the QA worker, with dub PCM windowed): about 16–17 GB. A bench run that adds BuzzASR is about 19–20 GB and is run with the app closed.

**Why the headroom matters.**
- In the gap-1 bench, heavy system swap (9.1 → 16.9 GB used) slowed T3 1.6–3.2× per token [gap-1 §3, M scratch].
- Keep Maata's default configuration at or below about 13.5 GB resident. The optional ASR arms stay bench-only until their memory is measured in the app.
- `verify-mac.sh` records `vm.swapusage`, the load average and thermal state with every run.
- A resident-set check at startup warns when free memory is under about 6 GB.

### 5.2 GPU time per video second

Podcast fixture; built from gap-1's measured parts (scratch) plus inference.

**Basis: the CFM step count of the measured 0.46 is unresolved.** gap-1 labels the 20:06 one-take cost "S3Gen 6 steps". But the engine defaults to 10 steps unless `MAATA_CFM_STEPS` is set (`backends/apple.py:214–215`), nothing in the shell or `scripts/` sets it, and `engine.log` doesn't record the value. So the 20:06 run most likely used **10 steps**, and gap-1's label is probably wrong; it can't be proven from the logs. This table is stated on that 10-step basis:
- the "as measured" column is the planning figure;
- the 6-step saving is counted only once, in its own column, and only holds if the basis is confirmed;
- if the run turns out to have used 6 steps, the "as measured" column already includes that saving, and the 6-step column must be dropped;
- from step 0 on, every voiced unit logs `cfm_steps` in `units.jsonl`, so this can't recur.

| Stage | Today | Target: N=2 batched, as measured (most likely 10 CFM steps) | Target: N=2 batched, 6-step CFM or meanflow (only if the basis was 10) | Throughput-governed floor: N=1, pair only on QA failure |
|---|---|---|---|---|
| Diarization | 0.04–0.06 | same | same | same |
| ASR pass 1 | 0.03–0.05 | same | same | same |
| ASR punctuation pass 2 | — | +0.035–0.07 | same | same |
| Music/singing tagging (CED, if approved) | — | +about 0.01 (inferred, small model) | same | same |
| Translation | **1.13** | **0** | 0 | 0 |
| TTS (incl. about 15% QA retakes, similarity) | 0.46 (1 take) | about 0.68 | about 0.56 (inferred: 6 steps saves about 0.4 s per line and take, ADR-014) | about 0.60 (scenario F) |
| Per-line T3 prompt tokenisation (A/B arm only, §3.5) | — | +about 0.01–0.02 while the arm runs (inferred: one S3-tokenizer pass per line at 0.145 lines per video-s) | same | same |
| **Total** | **1.70 (r = 0.59)** | **about 0.84 (r ≈ 1.2)** | **about 0.72 (r ≈ 1.4)** | **about 0.77 (r ≈ 1.3)** |

The QA worker's CPU time is not in these GPU figures, but it can slow T3's CPU-bound decode loop; §3.13's contention test guards that.

**Start-up** needs r ≥ 0.92 (60-min video, 5-min wait) to 0.97 (the 2.9 h fixture) [gap-1 §4]. The target is r ≥ 1.25, to absorb swap, long lines, seeks and retakes. So:
- **Default:** N=2 batched, with 6-step CFM if the maintainer's ADR-014 listening check accepts it; otherwise meanflow once it is benched.
- **Until then**, a **throughput governor** in `session`:
  - It measures r and the lead from `Throughput` / `ReadyRanges`.
  - When the lead is under target and r < 1.1, it drops to N=1 with pairs only on QA failure, and limits retakes.
  - When the lead is comfortable, it raises to N=2 or 3 (N=3 for short lines).
  - After a seek into unprocessed video, it holds N=1 until the lead is back at target (§3.12).
  - It stays on after the default is settled, since swap and thermal state vary.
- **Never** three serial takes: scenario C, r = 0.70 [gap-1 §4].

### 5.3 Scheduling now that translation has left the GPU

**GPU.**
- Keep **one GPU owner at a time**. Concurrent MLX and MPS was measured slower: TTS RTF 0.70 → 2.79–3.99 beside an LLM (ADR-016). Without the LLM, the case for concurrency is weaker still.
- Replace the FIFO `asyncio.Lock` (`server.py:57`) with a small priority scheduler:
  1. the voicer, when the lead is under 60 s;
  2. ASR and diarization the translator is waiting on (the scene frontier);
  3. the voicer in normal operation;
  4. background: full-video ASR for brief v2, calibration, the diarization pre-pass beyond the scene frontier.
- Today one voicer line can queue behind every other stage up to four times [gap-1 §6].
- Instrument `lock_wait_s`.

**CPU: the QA worker process, not a thread pool** (§3.13):
- IndicConformer QA and CTC alignment;
- DSP prosody features (`pyin`);
- loudness, EQ and room statistics.
- It runs in one spawned process with onnxruntime capped at 4 intra-op threads and a lowered priority, so it doesn't compete with the T3 decode loop's dispatch thread or the engine's GIL. JSON validation is cheap and stays in the engine.
- The M5 Pro has 15–18 cores. The QA CPU cost is unmeasured: a FLOP estimate gives 0.1–0.5 s per 5 s take [gap-1 §4]. Per-take QA on every take is locked in only if the contention test passes: T3 ms per token and end-to-end r within 10% of idle with QA and 3 Claude calls running. Otherwise QA runs on the chosen take only and on flagged pairs.

**Claude:**
- a separate pipeline of 2–3 subprocesses, with no GPU and no lock;
- it runs up to the lookahead horizon (replacing `TRANSLATE_AHEAD = 60`) as soon as each scene's ASR and diarization are done;
- the frontend and diarizer must stay ahead of it: ASR is about 0.04 s/s, so that is easy;
- the voicer never awaits it (§4.9).

**Start-up path (serial where it must be; inferred times):**
1. yt-dlp fetches the whole audio track and `load_audio` decodes it (unmeasured; depends on the link and the video length);
2. pre-pass diarization of 10 min (about 25–30 s of GPU);
3. ASR of the first 20–30 s scene (a few seconds);
4. **in parallel:** scene 1 on Claude with brief v0 (metadata only, no call needed; §4.2), about 15–60 s; and, on the GPU, clone plus calibration at about 15–25 s per speaker (three sentences). Scene 1 no longer waits for brief v1;
5. the first line voiced (a few seconds);
6. in the background: ASR of the rest of the pre-pass window, then brief v1, swapped in at a scene boundary.

Expect about **2–5 minutes to first dubbed audio** for a two-speaker podcast, dominated by the download, the 10-minute pre-pass and cloning. `verify-mac.sh` records time to first dubbed audio and each step's share, and step 1 requires **≤ 5 min on the podcast** (the maintainer accepts a 5–10 minute wait). If it fails, the first levers are a shorter diarization pre-pass before the first clone and streaming decode of the first minutes of audio.

---

## 6. The background-audio decision (the maintainer decides)

### 6.1 Evidence, as corrected by the verifiers

- **Federico 2020** is the only controlled dubbing test [04 F1; 07 A1; gap-6 C]:
  - the *original separated background* plus re-reverb: +10.34 MUSHRA for non-native listeners (p < 0.05);
  - +1.05 (not significant) for **native** listeners. For them, the earlier step of adding length-controlled MT plus prosodic alignment (uneven rate) had lowered ratings by 10.93;
  - it tests option C, not synthetic room tone.
- **Industry practice** keeps a bed:
  - ElevenLabs by default [01 F5];
  - HeyGen keeps music and drops effects [03 F8];
  - Sarvam warns a muted background feels "sterile" [03 N1];
  - Chitralekha overlays a Spleeter stem [03 F15];
  - Netflix leaves the M&E unaltered [07 C1];
  - iZotope's Ambience module *synthesises* room tone [07 C3].
- **HoliDubber:** listeners gave the studio-clean system the *highest* zero-shot MOS [04 F23].
- **Separation costs:**
  - dialogue residue about 15 dB down, some of it English speech [07 A5];
  - the original speakers' laughs leak into the effects stem [07 A6];
  - YouTube's reused beds are reported as distorted (a biased source) [07 A4].
- **YouTube API Developer Policies** (last updated 2026-09-14):
  - III.I.7 forbids separating, isolating or modifying audio, and its example says "you must not apply alternate audio tracks to videos";
  - III.I.16 forbids translating API content;
  - III.I.14 and III.E.1.a bar non-API access and downloading.
  - A plain reading covers **Maata's core design**, not just a bed. A separated bed is the textbook III.I.7 case [gap-6 §1–2, Y2]. This is not legal advice.

### 6.2 Options

| Option | Plays original samples? | Realism evidence | Terms exposure beyond the core design | Engineering | Verdict |
|---|---|---|---|---|---|
| A. Dry dub (today) | No | none; digital silence between lines, and each line brings its own noise floor | none | done | baseline |
| **B. Synthetic room tone + loudness, EQ and reverb matching** | **No.** Only statistics of the original (noise level, spectral envelope, RT60) | indirect: comfort noise (RFC 3389), iZotope synthesised ambience; no dubbing listening test | none | S; numpy/scipy + Web Audio, no new dependency | **Recommended**, built off by default until the maintainer's ruling (§6.3) |
| C. Separated music-and-effects bed | Yes (processed) | Federico +10.34 for non-natives only | highest (III.I.7 "separate, isolate") | M; separator pin (Demucs MIT is the clean candidate), gating, leak checks | Not recommended in either scope |
| C′. Bed only where nobody speaks, ducked and gated | Yes | partial | high | M | Only if C is ever cleared |
| D. Unmute the YouTube player in long speech-free stretches (music intros, montages) | Yes, YouTube's own stream through its own player | plausible (intros no longer silent) | lower than C: documented `unMute` / `setVolume`, nothing separated. Still breaks the maintainer's rule | S–M; integer volume ramps, risk of missed speech | Second choice, **personal scope only**, if the rule is relaxed |
| E. Voice-over with the original ducked | Yes | not dubbing | high | S | No |
| F. Generated text-to-audio background | No | unfaithful | none | M | No |

### 6.3 Recommendation

- **Build B now** (§3.11), behind a setting that **defaults to off** until the maintainer rules, in the ADR, that *statistics derived from* the original do not count as "playing original audio" (a CLAUDE.md hard constraint). Only then may the default flip, and only if the A/B favours it. **Evaluate it by a blind native-listener A/B (A vs B vs B plus matching) only after the timing fixes land**, because pacing dominates native judgements [07 R1].
- **Do not build C** for either scope.
- If the maintainer relaxes the rule for personal use only, prefer **D** over C, and limit it to stretches with no diarized speech for at least 3 s, with a VAD guard.
- Record the decision in an ADR together with the scope decision (§8, D1).

---

## 7. Ranked roadmap (realism gain per unit of effort)

Each step is recorded as an ADR when it changes a decision. Metrics are logged to `units.jsonl`, rolled up by `maata-bench`, and committed from `verify-mac.sh`.

| # | Step | Effort | Gain | Acceptance tests / metrics |
|---|---|---|---|---|
| 0 | **Instrument and fix the yardsticks.** Per-unit `lock_wait_s`, `t3_s`, `t3_tokens`, `t3_ms_per_token`, `flow_s`, `cfm_steps`, `takes`, `qa_cpu_s`, `cer`, `sim`, `translate_ready_at`. Per-block diar/ASR events; per-Claude-call usage events; the session GPU-seconds split, the lead over time, stalls, swap, thermal state, time to first dubbed audio. Voice yardstick at matched length, no leakage, with per-speaker ceilings. Timing metrics at speech level (§3.10) [gap-1 §7; gap-5 §1; 08 R3] | S | enables everything | A committed `verify-mac.sh` JSON from the M5 Pro with every field present; the host yardstick uses ≥ 30 s of held-out audio; this JSON is the baseline for every later "vs baseline" target |
| 1 | **Claude CLI translator, scene-batched, with the script contract** (§4, including the Telugu-script `spoken` field and English-word map of §4.3–4.4), replacing the Gemma translator **only after its gates pass**; CLI isolation for #91987 (§4.9); voicer decoupled from Claude; line cache; time-to-first-audio path (§5.3) | M–L | throughput 0.59× → about 1.8× with 1 take (scenario B); removes the 8 GB model | **Quality gate:** a blind translation bake-off (§7.1; ADR-018 rubric, ≥ 37 lines and preferably 60–100, 3 blind raters, label-shuffled): overall ≥ Gemma 3 plain + 0.5, major meaning errors ≤ 5%, naturalness ≥ 2.5 of 3. **Usage gate:** a 1-hour run whose 5-hour-window and weekly-window shares per video-hour are under the maintainer's thresholds (§4.8), with cache reads vs writes per call type committed. **Reliability:** 50 consecutive scene calls with an interactive Claude Code open on the usual version: 0 hangs; 0 missing ids over the 1-hour run; error classes unit-tested against the documented messages; with Claude calls delayed 90 s, voicer `lock_wait_s` and throughput unchanged. **Speed:** r ≥ 1.5 on the podcast with N=1; time to first dubbed audio ≤ 5 min; a 60-min video plays with no stall after starting within 5 min. A re-watch makes 0 scene calls. The Gemma code, models and extras are removed in the same change once these pass (Appendix B) |
| 2 | **Punctuation transfer + sentence segmentation + music flag** (§3.4) | S–M | removes the fragment failures behind "incomplete" and meaning inversions; stops dubbing lyrics | non-final units ≤ 3% on 4+ fixtures; words kept ≥ 99.5%; VAD coverage guard with no unexplained gaps > 0.8 s; on the music fixture, sung lines not voiced and no speech line lost |
| 3 | **Brief and glossary refinements + style guide + 3-sentence Telugu-script calibration + rescaled prior** (§4.2, §4.3, §4.6, §3.7.5) | S–M | intelligibility of English words; register; consistent names | English-word loss ≤ 13% (checker IndicConformer, or vasista22 whisper-te-small until it is approved); duration-estimate error ≤ 10% MAPE; name and term consistency across scenes ≥ 98% on the glossary terms; a second blind round shows no regression against step 1 |
| 4 | **Coverage gate + two-way tiers with band selection** (§4.5, §4.6) | M | fixes skipped or incomplete content and underfill without padding | coverage over the 1-hour run: C + m ≥ 95%, P ≤ 2%, E ≤ 1%; speech-time fill 0.90–1.00; share of lines whose required rate is in [0.9, 1.2] ≥ 75% (Descript 73–83%); a blind check that `fuller` adds no filler |
| 5 | **Real-time shape.** Priority GPU scheduler, batched N=2, throughput governor, pacing formula fix, keep working past the lookahead while paused, seek policy, whole-video mode with windowed PCM (§5, §3.12) | M | smooth playback on long videos | r ≥ 1.25 on an idle M5 Pro (N=2) or ≥ 1.3 (governed); the 2.9 h fixture starts within 10 min with 0 stalls; voicer `lock_wait_s` p95 ≤ 2 s; seek into unprocessed video → first dubbed audio ≤ 60 s at p95; resident set ≤ 13.5 GB through a 2.9 h run |
| 6 | **Take QA** (IndicConformer in the QA worker + CTC + similarity guard + cap/duration checks), retake ladder with asynchronous rephrase, UI flags (§3.13) | M | dropped or garbled words in audio; fewer bad lines | **contention test:** T3 ms per token and end-to-end r within 10% of idle with QA on every take and 3 Claude calls in flight, else QA on the chosen take only; every shipped take with CER > 50% or a cap hit is flagged and counted (share ≤ 1% of lines); per-line back-transcription CER on `spoken` against thresholds calibrated on the maintainer's MUSHRA-DG ratings; QA CPU ≤ 0.3 s per take; IndicConformer loads with the network blocked |
| 7 | **Timing v2** (§3.10) | M | pace and sync, the maintainer's top complaint after translation | regression guards: lag p95 ≤ 0.3 s, rate p90 ≤ 1.1, `rate_step_p90` not worse; improvements vs the step-0 baseline: silent-while-speaking ≤ 3 s/min (from 8.2), speech-level end-error median within ±0.3 s, long-line speech-level end error at least halved; blind sync rating vs the step-4 build ≥ 60% preference |
| 8 | **Voice lane, parallel from step 0** (§3.7): one build path for every speaker (the host included), bilingual ceiling, span by centroid, pooled x-vector, `s3gen_v3` A/B, +1 st test, cfg fixed per voice by ear, wire `cloneStrength` | S–M | "voices not close" | for guest **and** host: similarity fraction of ceiling +0.05 at matched length, no CER regression; blind MUSHRA "sounds like him/her" +10 vs baseline with a hidden reference; accent rating by a native listener not worse |
| 9 | **Mix** (§3.11): loudness, fades, limiter, EQ; room tone per §6 (off by default until the ruling); watermark moved to the end of the DSP chain | S | polish; no dead air | per-speaker difference ≤ 1 LU; no clicks at unit joins (automated discontinuity check); perth detection passes on the final engine output and on an offline render of the app chain; native blind A/B: A vs B vs B plus matching |
| 10 | **Per-line delivery** (§3.5): Claude labels mapped to exaggeration through a per-call `T3Cond` override; A/B the per-line T3 prompt from the line's own audio (≥ 3 s, clean) | M | expressiveness, the top driver of Indic TTS preference [03 F12] | blind A/B ≥ 60% win; CER not worse; pitch-variability ratio within 0.8–1.25 of the speaker's English |
| 11 | **Model upgrades by bake-off** (§7.1) | M–L | state of the art per stage | per bake-off rules |
| 12 | **V3 T3 retrain** on Telugu-script data + Rasa (§3.8) | L | accent, prosody, unseen-speaker similarity | unseen-speaker similarity +0.05 on IndicVoices-R S-SIM; CER ≤ incumbent + 1 pt; native accent and naturalness blind win |

**Why this order.**
- Step 1 now carries its own quality, usage and reliability gates. Translation quality and plan usage are the two things nothing else can compensate for, no public EN→TE score exists for any Claude model [05 B2], and the Gemma path is removed only when those gates pass. The script contract is part of step 1 because the step-1 schema already requires it.
- Steps 2–4 address the other complaints with the strongest evidence (completeness, fragments, underfill) at the lowest cost.
- Step 5 is what makes step 6's extra synthesis affordable.
- Timing (7) comes after the text is complete, because underfill is mostly a text problem [gap-4].
- **Voice (8) is a parallel lane.** Its S-effort items (the bilingual recording, one build path, span by centroid, pooled x-vector, `s3gen_v3` A/B, +1 st test, cfg by ear) depend only on the step-0 yardstick, so they start right after step 0 alongside steps 1–4. Their benches still run one at a time on an idle machine. Only the V3 retrain (12) waits.
- Acoustic polish (9) comes last for native listeners [07 R1].

### 7.1 Bake-offs to run

| Bake-off | Arms | Metric and decision rule |
|---|---|---|
| **Translation (Claude)**, part of step 1 | Sonnet 5 medium; Sonnet 5 high; Opus 5.5 low; Opus 5.5 medium. Same prompts, scene contract and schema (Telugu-script `spoken` plus English map). Gemma 3 12B plain as the historical anchor from saved outputs. YouTube's own EN→TE auto-dub (non-expressive) as a listen-only reference on youtube.com, never downloaded [02 N4] | ADR-018 rubric (meaning, grammar, naturalness, English use; 3 blind raters, label-shuffled) on ≥ 37 lines, preferably 60–100; fit rate in band; coverage classes; tokens, plan-usage share and seconds per scene. Choose per call type (scene, fit, review). **Sonnet 5 is the default and wins ties.** Opus 5.5 is chosen only if it is better by a stated margin, with the other rubric metrics not worse and its usage share within the maintainer's threshold: at least 3 fewer lines with a major error on the set (about 5 points on 60 lines), or mean naturalness higher by ≥ 0.2 on the 3-point scale. Between Sonnet efforts, the lower effort wins ties |
| **Unit and contract** | U (id per unit, today); A (whole sentence, drift); B (split-back at breaks with `pieces`) on the 26 items in `scratchpad/gap3/items.json` (harness `run_arms.py`, `judge.py`) [gap-3 §5] | Adopt B at hard breaks only if it wins on sync without losing more than 5 naturalness points |
| **TTS input script** | Latin vs Telugu-script English, 50 lines, blind (kit `scratchpad/gap2/ab/`) | Adopt Telugu script unless the native listener prefers Latin on clearly more than half of the items [gap-2 §5] |
| **Voice settings** | cfg {0, 0.3, 0.5} (one value chosen per voice, then fixed) × prompt {English span, native clip, line's own audio} × exaggeration {0.5, 0.7}, one factor at a time; one build path vs today's stitched path for short-reference speakers (the host); +1 st shift; `s3gen.pt` vs `s3gen_v3`; 10 vs 6 CFM steps vs meanflow 1–2 | Primary: blind "sounds like" plus native accent and naturalness. Guards: similarity fraction (no regression beyond 0.02, the seed noise), CER, duration. Speed arms must not lose on blind listening |
| **Takes** | N ∈ {1, 2, 3}, batched vs serial, on the maintainer's cloned voices, idle machine, alternating order | GPU-s per line; do batched takes match serial ones on CER and similarity? |
| **Telugu QA recogniser** | IndicConformer CTC/RNNT, SraVaani, svanita, BuzzASR, Omnilingual CTC, on 50–100 Chatterbox Tenglish takes plus the gap-2 400 takes | agreement with the maintainer's MUSHRA-DG error counts; CPU RTF on the M5 Pro |
| **English ASR + aligner** | Whisper turbo (+ punctuation transfer) vs Qwen3-ASR 8-bit vs parakeet-v2 bf16; aligner MFA vs Qwen3-FA | WER on 3 fixtures with hand transcripts; non-final unit share; pause-boundary MAE vs a hand-marked set; GPU-s/s |
| **Diarization** | community-1 vs Nemotron-3 (+ ReDimNet2 linking) | DER and cross-block link errors on the podcast plus 2 multi-speaker fixtures |
| **Reference cleaning** | raw vs AUSoundIsolation vs MossFormer2 vs DeepFilterNet3, applied (a) to T3 clips only and (b) to the S3Gen span too (Kim Mel-Band only if D16 clears it) | adopt only if duration-matched similarity rises and listening doesn't regress, and the arm's training-data licences are recorded (§3.2) |
| **Claude CLI isolation** (#91987), part of step 1 | Maata on the user's `claude` vs a separately pinned CLI vs a separate config directory, each with an interactive session open on the usual version | 50 consecutive scene calls, 0 hangs; sign-in still the CLI's own flow |
| **QA contention**, part of steps 5–6 | T3 idle vs T3 with the QA worker on every take plus 3 Claude calls; worker threads 2 / 4 / 8 | T3 ms per token and end-to-end r within 10% of idle; pick the largest thread count that passes |
| **Background** | A vs B vs B plus matching (plus D if relaxed) | native blind A/B after step 7 |
| **External bar (optional)** | ElevenLabs Dubbing v2 (te, strength 7) on 2–3 min of the maintainer's *own* content, made by the maintainer in the ElevenLabs app | blind comparison on meaning, naturalness, similarity and sync. Cloud, benchmark only, never in the product path [01 R11] |

---

## 8. Decisions that need the maintainer's approval

| # | Decision | Recommendation |
|---|---|---|
| D1 | **Scope and YouTube terms.** The core design (muted embed + yt-dlp audio + synced dub) conflicts on a plain reading with Developer Policies III.I.7, III.I.14, III.I.16, III.E.1.a and III.E.6, and with the YouTube ToS. "Must not" is defined as absolute, and there is no personal-use exception [gap-6] | Declare Maata a **personal tool** for now. That suspends the signed-installer plan, which is both Amendment 01 §3.7 and a **CLAUDE.md hard constraint ("Signed installers per OS")**, so the CLAUDE.md change goes in the same ADR (with D2) and brief counsel before any distribution. Apply the compliance fixes (Referer, no overlays, privacy notice) regardless. Consider a local-file mode for distribution. Not legal advice |
| D2 | **Amend CLAUDE.md, the spec and the amendment** ("no cloud AI", "no network except yt-dlp") for text-only Claude CLI use; `sota_ranking.md` §1 becomes reference only; there is no offline translation fallback | Approve, in an ADR recording the no-local-LLM decision, text-only egress, the privacy notice, and the data-privacy setting link The same ADR retires the local translators everywhere: `models.lock.json` entries (`gemma-3-12b-it-mlx`, `translategemma-4b-it-4bit-mlx`), the `mlx-lm` (apple) and `llama-cpp-python` (cuda) extras, the fetch lists in `setup-mac.sh` / `verify-mac.sh`, and the CUDA backend's `LlamaCppTranslator`, once step 1's gates pass |
| D3 | **Update the Claude CLI** to ≥ 2.1.280 (a configuration change on his machine) and enforce a version floor in the engine; test `--restricted`. **Plus the #91987 mitigation** (§4.9): a separately pinned, unmodified Claude Code install for Maata, at a version different from his interactive one, auto-update off (05 open question 9); fallback, a separate config directory | Approve the update and the floor (pin the minimum, since server-side floors move). Approve evaluating the separate install, recorded in its own pinning ADR; it must pass the 50-call no-hang test before step 1 ships. Personal scope only until D18 answers whether it counts as bundling |
| D4 | **Usage credits and overage** | Never by default; pause with a message. Opt-in only |
| D5 | **Background audio rule** (§6) | Rule on whether statistics derived from the original are compatible with "original audio never played". Until then option B is built but **off by default**. No C. D only if relaxed, in personal scope |
| D6 | **Telugu-script TTS input contract** (changes `SPEC.md:221`, the prompt and shots) | Approve, pending his blind listen on the ready kit |
| D7 | **Planner ADR amendments to ADR-017**: sentence drift, hard breaks at ≥ 1 s with Telugu-fluency placement, asymmetric onset cost, windowed lookahead, 7.5 aksharas/s ceiling, internal-pause compression; later an A/B of a 0.92–0.95 floor and the WKWebView video-rate probe | Approve the first set; defer the floor and video rate until after measurement |
| D8 | **Extra synthesis budget** beyond `RESYNTH_SHARE` 10% for QA retakes, governed by throughput | Approve |
| D9 | **IndicConformer-600M** (MIT, gated: he must accept the HF gate himself) as the QA selector. This brings **onnxruntime as a new Mac dependency**: today it is in `uv.lock` only through faster-whisper in the `cuda` extra and is not installed in `engine/.venv`. The card's `onnx==1.20.1` pin conflicts with the locked onnx 1.23.0 (via s3tokenizer), and its loader uses `trust_remote_code`. **BuzzASR/telugu** (MIT) as a bench-only reporting model | Approve onnxruntime in the `apple` extra, pinned in `uv.lock` and tested with the locked onnx. Vendor the model's `model_onnx.py` instead of `trust_remote_code`; list the ONNX files with sha256 in `models.lock.json`; load with `HF_HUB_OFFLINE=1`; add a network-blocked load test. Record both models' training corpora and licences in their pin ADRs |
| D10 | **ADR-014's 6-step CFM** listening check; fetch `s3gen_meanflow` and `s3gen_v3` (MIT) | Listen and decide 6 steps; approve the fetches |
| D11 | **ASR bake-off dependencies**: vendored mlx-audio modules (transformers conflict), parakeet (CC-BY-4.0 attribution), MFA via conda or Qwen3-FA; the Cohere gate if that arm is wanted | Approve the vendoring route (one decision covers ASR, aligner and diarization). MFA only if conda is acceptable |
| D12 | **Nemotron-3-Diarization** under OpenMDW-1.1; **ReDimNet2** licence ADR (MIT / Apache / CC-BY stated three ways) | Approve OpenMDW (permissive, commercial use allowed) for the bake-off; never the `-preview` evaluation-only weights |
| D13 | **V3 T3 retrain**: rented NVIDIA compute; the Rasa gate (CC-BY-4.0); all IndicVoices-R rows; FLEURS Latin rows transliterated | Approve after steps 6–8 show T3-side deficits |
| D14 | **Reference-cleaning arms** (AUSoundIsolation, MossFormer2, DeepFilterNet3): vendored MLX modules; a small Swift helper for AUSoundIsolation. Training-data licences recorded per arm before adoption | Approve for the bake-off only |
| D15 | **CED (Apache-2.0)** as the music and singing tagger (§3.4), pinned; **optional:** emotion models (FunASR licence / fairseq); CED + Chatterbox-Turbo for non-verbals; RVC/Applio | Approve the CED pin for the music flag (heuristic flag meanwhile). Defer the three optional items |
| D16 | **Rulings on training-data inheritance** (IndicTTS, Expresso, Emilia, EARS, FSD50K, MUSDB NC clips), now including **Kim Mel-Band RoFormer** (MIT weights, undisclosed training data, probably MUSDB18) as an "ask" | Keep them blocked; nothing in the plan depends on them |
| D17 | **ElevenLabs external benchmark** of his own content (cloud) | Optional; his call |
| D18 | **Anthropic written questions** before distribution (§4.10), now also: does a separately installed, unmodified CLI for Maata count as bundling? | Send them when distribution is on the table |
| D19 | **Distill-MOS** as a flagger: "ask" until the `xls_r_sqa` dependency's licence is checked (its card names research as the primary use) | Use IndicMOS (CC-BY-4.0) alone meanwhile |
| D20 | **Usage thresholds** (§4.8): the share of a 5-hour window and of the weekly windows one video-hour may use | He sets both from his plan and weekly viewing; step 1 is measured against them |
| D21 | **Referer compliance** (§3.12): serve the UI from the Tauri custom-protocol origin instead of loopback, amending ADR-006, the CSP and the WebSocket origin check | Approve a probe first (player errors 152/153 were ADR-006's reason for loopback); amend ADR-006 only if the probe plays |

---

## 9. Risks and open questions

**Risks, highest first:**
1. **Platform terms (D1).** The whole product rests on a pattern YouTube's policies appear to forbid. The background choice doesn't change that. The size of the enforcement risk for a personal tool is not estimable from sources [gap-6].
2. **Claude as a dependency:**
   - unpublished plan limits, and a paused (not cancelled) billing change;
   - transient throttles;
   - the #91987 lock hang when an interactive session runs on the same CLI version. This is the maintainer's normal workflow, not an edge case, and translation has no fallback;
   - plan usage shared with his own development work;
   - the unverified #83859 stall;
   - a future `--bare` default for `-p` that could break subscription sign-in [05 A6, A9].
   - Mitigations: a separately pinned CLI for Maata with a 50-call no-hang gate (§4.9, D3); a voicer that never waits on Claude; lookahead buffering; error classes; a watchdog with a UI message naming the cause; the other-model fallback; the line cache; usage thresholds (D20); token logging; a version floor.
3. **Claude's Telugu quality and cost are unmeasured.**
   - There is no public EN→TE score [05 B2].
   - Telugu tokenization could make rich output slower than real time on one stream [gap-1 §5].
   - The bake-off and a 1-hour usage run are now step 1's own acceptance gates (§7), so the Gemma path is removed only after they pass.
4. **Throughput margin.**
   - The shaped pipeline is about 1.2–1.4× on paper [gap-1].
   - Swap can cost 1.6–3.2× [gap-1 §3], and the IndicConformer CPU cost is a FLOP estimate.
   - T3 decode is CPU-overhead-bound [gap-1 §2], so CPU work beside it (QA, DSP, the `claude` processes) can slow TTS even though it uses no GPU time.
   - The CFM step count behind the measured 0.46 GPU-s/s is unresolved (§5.2), so the 1.4× figure may double-count a saving.
   - Mitigations: the governor, the QA worker process with capped threads and its contention test, a memory ceiling, `cfm_steps` logging, measurement.
5. **The voice ceiling may be close.** The guest is already at about 75% of the stack's English score, in line with real bilinguals' cross-language retention (about 79%) [gap-5 E3]. "Nowhere near" may be mostly prosody and accent, which only listeners and the V3 retrain can move. The maintainer's ears are the gate, so the listening sessions need his time.
6. **Overfitting to one podcast.** Nearly every number comes from one two-speaker podcast. Lectures, vlogs, music beds, fast talkers and 3+ speakers are needed before tuning thresholds.
7. **V3 "accent preservation"** could increase English-accent leakage [06 new 4]. Users report V3 underperforming V2 in Danish (#551, anecdotal) [gap-5 E5].
8. **Telugu-script input** could sound less English on English words than a listener wants. This is unmeasured, and the kit is ready [gap-2].
9. **Privacy.** Transcripts go to Anthropic. Whether they train models depends on the user's setting. The notice and link are required.
10. **Scratch numbers.** None of the gap studies' M5 Pro numbers are committed yet (CLAUDE.md rule).

**Open questions:**
- Does `--restricted` keep subscription sign-in? Does `--settings '{"fastMode":false}'` override a persisted value under `--safe-mode`? (Test on the M5 Pro.)
- What are Sonnet 5's and Opus 5.5's tokens per Telugu character, its TTFT and output speed at low and medium effort, and the subscription concurrency limit?
- IndicConformer on synthetic Tenglish takes: its CER thresholds, its CPU RTF, and whether it runs on the onnxruntime version pinned for the Mac alongside the locked onnx 1.23.0.
- Does a separately pinned CLI (or a separate config directory) avoid the #91987 lock? Where does the lock live?
- Did the 20:06 run use 10 or 6 CFM steps (§5.2)?
- Does the PerTh watermark survive the engine DSP and the app's Web Audio chain?
- How well does CED separate singing from speech over a music bed on YouTube content?
- The bilingual same-speaker ceiling for the maintainer. Why did the host show no language penalty? (It needs the full-length podcast.)
- Does `fuller` stay free of filler for Telugu? Does sentence-level translation alone remove most P-class drops [gap-4 OQ2, OQ4]?
- Does spoken Telugu accept post-verbal afterthoughts naturally enough to rescue verb | complement breaks [gap-3 OQ3]? (Native judgement.)
- Is the 1.0 s hard-break threshold right for Telugu viewers watching a face [gap-3 OQ4]? Does the early/late asymmetry hold for dubbed speech?
- Does WKWebView honour `setPlaybackRate(0.9)` while playing [08 F18]?
- Does room tone help native Telugu listeners once pacing is fixed? No study exists [07 OQ5].
- Does the natural conversational Telugu rate in aksharas/s support the 7.5 ceiling [08 OQ4]?
- Would Anthropic treat a distributed Maata as ordinary individual usage [gap-6 §5]?

---

## Appendix A. Load-bearing sources, with dates

All were fetched or read on 2026-09-24 unless another date is given. Full lists are in the cited reports.

**ElevenLabs:**
- dubbing capabilities and API docs (undated) [01];
- Introducing Dubbing v2 (2026-05-28, updated 2026-09-06);
- AI dubbing API blog (2026-08-06, updated 2026-09-22);
- What is AI dubbing (2026-07-27);
- changelog 2026-08-10;
- Turing Post interview (2025-04-12);
- Greg Preece review (2026-07-24).

**YouTube / Google:**
- Help 15569972 (undated);
- YouTube Blog (2026-02-04);
- Slator (2026-02-06);
- heise (2025-03-14);
- patent US 2020/0404386 A1 (2020-12-24), granted as US 11,582,527 B2 (2023-02-14);
- Aloud (2022-03-09);
- Google Research S2ST (2025-11-19);
- IFrame API reference (updated 2026-09-15);
- API Services ToS, Developer Policies and Required Minimum Functionality (updated 2026-09-14);
- YouTube ToS (effective 2023-12-15) [02, 08, gap-6].

**Other vendors:**
- Sarvam Dub (2026-02-01, modified 2026-09-14);
- Bulbul V3 (2026-02-05);
- Sarvam docs (undated);
- Descript / OpenAI case study (about 2026-03);
- HeyGen and Synthesia help (undated);
- Azure video translation (updated 2026-06-05) [03, 02].

**Research:**
- Federico et al., arXiv 2001.06785 (IWSLT 2020);
- Virkar et al., 2204.02530 (Interspeech 2022);
- Brannon et al., 2212.12137 (TACL 2023);
- Effendi et al. (ICASSP 2022);
- Swiatkowski et al., 2306.11662 (2023-06-21);
- Chronopoulou et al., 2302.12979 (2023-02-25);
- Chadha et al. (LSST), 2506.00740 (2025-05-31);
- Won et al. (EMNLP 2025 demos, 2025-11);
- HOMURA, 2601.10187 v3 (2026-09-03);
- Cui et al., ACL 2025, 2508.08550;
- Text-Audiobox, 2609.03992 (2026-09-03);
- Indic TTS preferences, 2604.21481 v2 (2026-06-23);
- IWSLT 2026 voice cloning, 2604.26136 v2 (2026-06-25);
- Singlish, 2607.23027 (2026-07-25);
- Carbonneau et al., 2507.02176 (2025-07-02);
- VALL-E X, 2303.03926 (2023-03-07);
- CosyVoice 2, 2412.10117 (2024-12-25);
- Pérez et al. (QoMEX 2019);
- CoSTA, 2406.10993 (2024-06-16);
- Javorský et al., 2506.04855 (2025-06-05);
- Tan et al., 2605.13368 (2026-05-13) [04, 05, 06, 08, gap-3, gap-4, gap-5].

**Models (Hugging Face API, 2026-09-24):**
- chatterbox @5bb1f6ee / @e2d6902d (2026-06-10);
- chatterbox-turbo @749d1c1a;
- chatterbox-telugu @d4341468;
- indic-conformer-600m @e9b71b36 (2026-02-07);
- BuzzASR/telugu @598f495e (2026-09-22);
- Nemotron-3-Diarization MLX @59ed2dbf (2026-09-23);
- Qwen3-ASR-1.7B @7278e1e7;
- parakeet-tdt-0.6b-v2 @8ae15530 (MLX);
- Rasa @632f55c7 (2026-06-06);
- IndicVoices-R @5f4495c9 [sota, 06, 10].

**Anthropic:**
- Claude Code legal and compliance (undated; product section first captured 2026-08-30);
- CLI reference (saved 2026-09-24);
- CHANGELOG top entry 2.1.281 (live, 2026-09-24);
- models overview and Opus 5.5 page (2026-09-22);
- support 15036540 (updated 2026-06-16);
- Consumer Terms (effective 2025-10-08);
- Commercial Terms (effective 2025-06-17) [05, gap-6].

**Maata data (M5 Pro, 2026-09-24, scratch, not committed):**
- trace `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl` (sessions 13:44, 18:31, 20:06);
- `engine.log`;
- `scratchpad/gap1_batch_takes.*`, `gap2/`, `gap3/`, `gap4/`, `gap5_*`.

## Appendix B. File-level change list

| File | Change |
|---|---|
| `engine/src/maata_engine/backends/base.py` | Add `SceneRequest`, `LineSpec`, `SceneResult` and a `SceneTranslator` protocol; drop the local-LLM `Translator.condense` and `translate_candidates` |
| `backends/claude_translator.py` (new) | brief / scene / fit / review / re-translate / async rephrase calls; schema; validators; concurrency semaphore; cancellation on seek; per-line cache (`lines.jsonl`); usage logging per call type and model |
| `claude_cli.py` | version gate; error classes; stream-json `rate_limit_event`; watchdog with a cause-naming status; `--settings` fastMode off; `--restricted` (after test); model `claude-sonnet-5` default with the other model as fallback (swapped if the bake-off picks Opus); path to a separately pinned CLI and the version-differs check (after the D3 ADR); one fixed workdir; strict schema handling |
| `backends/apple.py` | remove `MLXChatTranslator`, `MLXTranslateGemma`, `StyledTranslator` (after step 1's gates); add the punctuation-transfer pass, the hallucination threshold and the coverage re-decode in `MLXWhisper.transcribe`; pass `cfm_steps` into the trace |
| `backends/cuda.py` | remove `LlamaCppTranslator` and wire `ClaudeTranslator` (after step 1's gates) |
| `engine/models.lock.json` | remove `gemma-3-12b-it-mlx` and `translategemma-4b-it-4bit-mlx`; add IndicConformer's ONNX files with sha256 (D9), later CED (D15) |
| `engine/pyproject.toml`, `uv.lock` | remove `mlx-lm` from `apple` and `llama-cpp-python` from `cuda`; add `onnxruntime` to `apple`, pinned (D9); CED's runtime if approved |
| `scripts/setup-mac.sh`, `scripts/verify-mac.sh` | fetch lists follow the new lock (no local LLM); verify also records time to first audio, seek-to-audio, usage shares, `cfm_steps`, T3 ms per token idle vs under QA and Claude load, swap and thermal state |
| `text/punct_transfer.py` (new), `text/scene_prompt.py` (new) | from `gap3/transfer.py`; the Claude system prompt (shared by scene, fit, re-translate and rephrase calls), examples and schema (versioned as `PROMPT_HASH`) |
| `text/tenglish.py`, `text/akshara.py` | lint on the `english` map; drop Latin assumptions (`:60`, `:332`, shots); count `spoken` only |
| `segment.py`, `types.py` | sentence units; `breaks`, `anchors` (filled by calling `anchors()`, `anchor_pause` 0.7 → 0.3 s), `cut_off` on `SourceUnit`; clause-mark-only `max_len` cuts; `merge_fragments` keeps breaks and anchors and records the join as a break; music/singing flag |
| `session.py` | scene batching; translator off the GPU, up to the horizon; **inline `translate_candidates` removed from `_dub`**, provisional takes and deadline-bound replacement; unit lifecycle fields; band selection with an online *k* prior; coverage gate; take QA and retake ladder; throughput governor; seek policy (short first scene, cancellation, N=1 until the lead recovers); 3-sentence Telugu-script `CALIBRATION_TE`; one voice-build path for all speakers with centroid score and pooled x-vector; `_sentence_cut` pad search; `_evict_raw` evicts ahead of the playhead too; wire `clone_strength` |
| `server.py` | priority GPU scheduler replacing the shared `asyncio.Lock`; Claude health (version, auth, isolation) on `hello`; origin checks if D21 moves the UI origin |
| `backends/torch_common.py` | batched 2N-row `_t3_tokens`; fewer host syncs; cap-hit returned as failure; `prepare_voice_parts` accepts a stitched timbre and pools the x-vector; **`synthesize_mel(…, exaggeration=None, prompt_tokens=None)`** overriding the cached `T3Cond` per call; watermark split out of `vocode` into a final `watermark()` step; S3Gen filename parameter; CFM steps / meanflow |
| `timing/planner.py`, `timing/duration.py` | sentence drift, breaks, asymmetric cost, windowed lookahead, akshara ceiling, pause compression; estimator keyed by voice settings, directly seeded; `DEFAULT_RATE` 5.5 → about 6.0–6.3 with the count change |
| `qa/coverage.py`, `qa/take.py`, `qa/mix.py`, `qa/worker.py`, `qa/indicconformer.py` (new) | deterministic checks; the spawned QA worker (onnxruntime capped at 4 intra-op threads); vendored IndicConformer loader (from its `model_onnx.py`, no remote code); CTC alignment; CER-plus-duration selection with a similarity outlier guard; loudness, EQ, reverb and room statistics |
| `bench.py` | voice yardstick (guest and host); timing, coverage and QA metrics; Claude usage; the contention and CLI-isolation benches; BuzzASR reporting (bench only) |
| `app/src/lib/{sync.ts,buffer.ts,voice.ts}` | fades, crossfades, limiter, room-tone node (off by default); pacing formula; replacing a provisional unit's audio; windowed `AudioBuffer` retention; `cloneStrength` labels; session report; "this line is wrong" hotkey; Claude status messages |
| `app/src-tauri` | serving the UI from the custom-protocol origin for the Referer, CSP and WebSocket origin (D21, after the probe) |
| `docs/` | ADRs for D1–D21 (including the CLAUDE.md amendments for Claude text egress and the suspended signed installers, the pinned CLI, the watermark order under ADR-014, the calibration change under ADR-017, and ADR-006 if D21 passes); `SPEC.md:221`; commit the spike JSON under `docs/spikes/results/<machine>/` |

## Appendix C. Review changes (2026-09-24)

An independent reviewer raised 9 major and 19 minor issues. The code claims were re-checked against HEAD before editing: `session.py:566–574`, `:62`, `:176`, `:301–322`, `:657`; `backends/cuda.py` (`LlamaCppTranslator`); `segment.py:523, 558`; `planner.py:237`; `torch_common.py:150, 188, 282`; `apple.py:214–215`; `uv.lock` (onnxruntime only via faster-whisper; onnx 1.23.0 via s3tokenizer); `models.lock.json`. All were confirmed.

**Rejected outright: none.** Every issue was accepted. Six were adopted in a different form than proposed, for these reasons:

| Issue | What was proposed | What the revision does instead, and why |
|---|---|---|
| Voicer blocking on Claude (§4.9) | park the line and move on; ship the best take only if the rephrase misses its deadline | Ship the best take **as a provisional take at once**, and replace it if the rephrase arrives ≥ 60 s ahead and fits the planned span. A parked line has no audio at all if the rephrase is late and the voicer is busy, which would create the very gap the rule is meant to prevent. |
| Usage tie rule (§7.1) | reverse the tie rule, with "a stated margin" | Reversed, with the margin stated as ≥ 3 fewer major-error lines on the set, or +0.2 naturalness. A margin in percentage points is below the resolution of a 37-line set (one line is 2.7 points), so the set is recommended at 60–100 lines. |
| Scene cache key (§4.7) | cache validated scene JSON per (video, unit text hash, `PROMPT_HASH`, model) | Cached **per line** with that key, so a different scene cut after a seek or a new start still hits. |
| GPU budget basis (§5.2) | resolve the CFM step count from the 20:06 run's environment or log | It can't be resolved: `engine.log` doesn't record it. The table is restated on the most likely basis (10 steps: the engine default, with no override in the shell or scripts), the 6-step column is marked conditional, and `cfm_steps` is logged from step 0. |
| Kim Mel-Band (§3.2) | drop it, or move it under D16 | Moved under D16 as an "ask", out of the bake-off arms. It costs nothing to keep the question open, and nothing depends on it. |
| Music and singing (§3.4) | leave music and sung units undubbed | Only **sung** units are left undubbed. Speech over a music bed is still dubbed; the flag only removes those gaps from the room-tone and loudness statistics. CED needs a pin approval (D15), so a report-only heuristic covers the interim. |

**What changed, by issue:**
- **Majors.**
  - #91987 lock hang: a separately pinned CLI for Maata (D3, own ADR), a 50-call no-hang gate in step 1, and a UI message naming the cause (§4.9, §3.12).
  - CPU contention: QA moved to a spawned worker capped at 4 threads, plus a contention test in steps 5–6 (§3.13, §5.3).
  - IndicConformer runtime: onnxruntime is a new Mac dependency; vendored loader instead of remote code; network-blocked load test (§3.13, D9).
  - Voicer never awaits Claude: the inline `translate_candidates` is removed, and rephrases are async with a deadline and a 90 s-delay test (§3.6, §3.13, §4.9).
  - Roadmap order: the bake-off, the usage run and the script contract moved into step 1, before the Gemma path is removed (§7).
  - Usage pass/fail thresholds, Sonnet as the default, and the line cache (§4.8, §4.7, D20).
  - cfg removed from the per-line duration ladder (§3.7, §3.8).
  - Kim Mel-Band removed from the arms, and training-data licences recorded per model (§3.2, §3.13).
  - Seek policy with a ≤ 60 s p95 seek-to-audio target (§3.12, §4.1).
- **Minors.**
  - Start-up: brief v0 lets scene 1 start without brief v1; a 2–5 min estimate and a ≤ 5 min gate (§5.3, §4.2).
  - Caching: per-model caches, glossary additions in the user message, cache writes budgeted (§4.3, §4.8).
  - The *k* prior starts at about 1.4 and is updated online (§4.3).
  - `DEFAULT_RATE` rescaled to about 6.0–6.3, with Telugu-script calibration sentences (§3.7).
  - One voice-build path for the host (§3.7).
  - A per-call `T3Cond` override, with per-line prompt tokenisation counted (§3.5, §5.2).
  - Code statements corrected: `LlamaCppTranslator`, anchors, `anchor_pause` 0.7 → 0.3 s, `merge_fragments`, `rate_step_p90` (§1.3, §3.4, §3.10, Appendix B).
  - Local-LLM cleanup across the lock, the extras, the scripts and CUDA (D2, Appendix B).
  - Memory: BuzzASR is bench-only, and dub PCM is windowed (§5.1, §3.12).
  - Timing: regression guards plus improvement targets, and speech-level end error (§3.10).
  - Watermark applied last, with a detection check (§3.8).
  - Referer re-scoped as a shell change with ADR-006 (§3.12, D21).
  - The signed-installer hard constraint is named in D1; option B is off by default (§6.3, D5).
  - Distill-MOS is "ask", and BuzzASR's training data is recorded (§3.13, D19).
  - Music and singing flag (§3.4).
  - The voice lane runs in parallel after step 0 (§7).
  - The Federico wording is corrected (§1.1, §6.1).
