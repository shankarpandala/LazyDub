# Phase 0 plan: spikes

Status: **proposed; waiting for your go-ahead** (spec §0). No code has been written.

- Research behind the plan: `docs/research/2026-09-23-dependency-survey.md`.
- How every number is measured: `phase-0-methods.md`.

---

## To approve

**Schedule.** 18 working days at most, counted from go-ahead. Each box is a cap: when time runs out, the spike reports what it has and drops its *cut first* items.

| Days | Work |
|---|---|
| 1–2 | Groundwork |
| 3–4 | S1 Streams |
| 5–7 | S5 Sync |
| 8–11 | S4 TTS (the patch and converter come first; S2 and the co-run need them) |
| 12–14 | S2 Front-end |
| 15–17 | S3 Translation |
| 18 | The §10 composition and the report |

**Decisions needed before I start**

| # | Decision | Recommendation |
|---|---|---|
| D1 | **Xcode version.** 27.0 went GA on 2026-09-14. 26.4+ needs Tahoe 26.2, and 27 needs Tahoe 26.6. | **Xcode 26.6** if the Mac runs Tahoe 26.2+ (it's also the `macos-26` CI default); otherwise the newest 26.x the Mac can run. Pin mlx-swift 0.31.4 either way. The alternative is 27.0 on the `xcode-27` preview runner, but no upstream project tests it. |
| D2 | **A speech-swift fork for Chatterbox-Telugu.** | You create `shankarpandala/speech-swift` and add it to the session, and I push a patch branch. <br>**Phase 0 patch:** vocab size from `config.json`, the tokenizer fix, a language set derived from the tokenizer, cancellation, a seedable RNG, and an opt-in load path that keeps fp16 (experimental). <br>**Deferred to Phase 3:** voice conditioning that persists to disk. <br>Upstream PRs are drafted for your review and opened only after you approve. |
| D3 | **Where the S2–S4 engine adapters live.** The spec says spikes are throwaway. | Keep them in the real packages, marked *provisional*, so `maata-bench` has something to run. Phase 1 may rewrite them, and the adapters for losing engines are deleted after the report. The alternative is to keep them in `Spikes/` and rewrite them in Phase 1. |
| D4 | **Dependencies the spec doesn't name.** | **WhisperKit** (MIT): an S2 word-timing candidate, **yes**. <br>**SpeechLanguageID** (VoxLingua107; licence not yet checked): yes, if the licence check I do first comes back permissive. <br>**Nemotron** streaming ASR (openmdw-1.1; licence not yet reviewed): I summarise the licence for you first. If it's acceptable, it becomes an S2 candidate for Hindi word timing and a second Telugu CER judge. **Omnilingual stays the primary judge** (spec §11). |
| D5 | **Training-data overlap.** chatterbox-telugu was trained on FLEURS and IndicVoices-R, and S4 wants Telugu speech for calibration and upper bounds. | Tell me which splits and speakers were used. I'll use only held-out speakers and sentences, or about 10 min of self-recorded Telugu. |
| D6 | **NC engines.** OmniVoice (CC-BY-NC + Boson) and Indic-Mio (NC training data). | A capped comparison of at most half a day. If one clearly beats chatterbox-telugu, I'll propose an opt-in download behind its own licence acceptance (§3.6). Otherwise, drop it. |
| D7 | **Rate-control acceptance margin at 1.2×**, measured against the 1.0× render. | CER rises by at most 1 point absolute, WeSpeaker similarity drops by at most 0.02, and your blind listening doesn't reject it. The cheapest method that passes wins. |
| D8 | **Drop HLS from §6.1's video list?** `setRate(_:time:atHostTime:)`, which the sync design relies on, isn't supported for HLS. | Yes, unless S1 finds HLS is the only option that works. |
| D9 | **A dev-only Python script** (HF `tokenizers`) that produces golden token ids for the tokenizer tests. The research already generated the ids; they aren't committed. | Yes. Commit it with the golden file under `tools/dev-python/`. It is never built into the app. |

**Inputs I need from you** (by day, counted from go-ahead)

| Input | Format | Amount | Needed by | If it's late |
|---|---|---|---|---|
| Video list | Approve or edit my draft `TEST_VIDEOS.md` (the §11 categories plus edge cases) | ~15 URLs | day 2 | I use my draft |
| S3 sentence set | Edit my draft of English lines covering numbers, dates, currency, names, tech terms, idioms, and short and long lines | 50 | day 4 | My draft as-is |
| S5 capture session | A 240 fps iPhone (or a photodiode + mic rig), with built-in speakers, wired headphones and AirPods | ~1 h | day 7 | Internal metric only, marked *unvalidated* |
| Held-out Telugu | Your answer to D5, or self-recorded speech (2+ speakers) | ~10 min | day 8 | FLEURS test split, flagged |
| S4 listening | A blind sheet: 3 engines × 20 lines, plus rate-control renders | ~110 clips, ~1.5 h | day 12 | Automatic metrics only |
| S2 fixtures | WAV + RTTM + plain transcript. No word timings: those come from AMI's word annotations. | 3 English clips × ~5 min (interview, 3+ speakers, music bed) + 1 Hindi | day 12 | AMI and LibriSpeech only; Hindi gets WER only |
| S3 scoring | Adequacy and naturalness, each 1–5, for 4 systems × 50 lines, plus 40 condensed lines | ~240 ratings, ~2 h | day 17 | Automatic metrics only, and no quality recommendation |

**Defaults I'll use unless you object**

- **Naming:** the product is **Maata**, with bundle-ID prefix `io.github.shankarpandala`.
- **Public datasets:** AMI, LibriSpeech, LibriTTS-R and FLEURS (all CC-BY 4.0), with small attributed excerpts committed.
- **TranslateGemma** comes from the mlx-community mirrors for the spikes. They lack the Gemma Notice file, so mirror vs. `google/*` with a token flow is decided after S3.
- **yt-dlp:** S1 benchmarks it, because §12 requires the comparison. Whether to *ship* `yt-dlp_macos` (a frozen Python binary, ad-hoc signed) is decided after S1.
- **S1's Gatekeeper test** uses ad-hoc signing and is marked provisional, unless you give me a Developer ID Team ID.
- **Standard-preset numbers** are budget-capped on the 24 GB Mac and marked provisional, unless a 16 GB Mac is available.
- **Watermark:** the Swift port applies no PerTh watermark. This is recorded in `MODELS.md`; whether to port PerTh or drop the claim is decided after S4.
- **CAM++:** WeSpeaker reconciles speakers in S2. CAM++ exists only inside Chatterbox's S3Gen, and it's compared in Phase 3, when the threshold is tuned.

---

## Goals

1. Choose the default for every stage, plus the resolver, ingest and sync approaches, based on M5 Pro measurements.
2. Show whether §10 can be met with those choices, for Recommended (24 GB) and, provisionally, Standard (16 GB).
3. Deliver the Telugu Chatterbox MLX bundle and the speech-swift patch that loads it.

## What the research changed

| The spec assumed | What's actually true |
|---|---|
| speech-swift's Chatterbox port just needs converted weights | It rejects Telugu: a hard-coded vocab and language gate, and a tokenizer that turns most Telugu into `[UNK]`. A fix is verified in Swift. Conversion is a key rename. |
| WhisperASR gives word timestamps | It gives text only, and can drop text silently. |
| TranslateGemma covers the §6.4 request | Its template only translates, and mlx-swift-lm skips its RoPE scaling. |
| OmniVoice is Apache-2.0 | Weights are CC-BY-NC, and the codec is under Boson's licence. |
| Xcode 27 is in beta | It went GA on 2026-09-14. |
| Two independent resolvers | Both rely on essentially one YouTube client. |

## Groundwork (days 1–2)

- **Project:** XcodeGen `project.yml` (Swift 6 mode, macOS 15, arm64), with `Package.resolved` committed and pins recorded as ADRs.
- **Packages:**
  - `MaataCore`: types, the §5 protocols, and `AksharaCounter` (Telugu clusters plus an English syllable heuristic).
  - `Diagnostics`: result schema, memory and timing probes, and WER/CER/DER scoring.
  - `StreamResolving`: the URL parser, with §11 tests.
  - `Tools/maata-convert-chatterbox`.
- **`maata-bench`:** `fetch` pulls models from `models.lock.json`, pinned by commit and sha256. All model runs load from local paths with outbound network denied, so no library downloads on its own.
- **CI:** `macos-26` runner (build plus MLX-free tests).

## Spikes

### S1 Streams (days 3–4)

- **Candidates:**
  - YouTubeKit (`methods: [.local]` only);
  - yt-dlp + Deno.
- **Measure:**
  - Resolution success and latency.
  - A distinct error for each out-of-scope case: live, premiere, members-only, age-restricted, private, DRM.
  - In-scope edge cases: made-for-kids, ended live, 4K, Shorts, and every URL form.
  - Time to the first 60 s of decoded audio (**≤ 2 s**, reported by video length up to ≥ 2 h), plus its alignment to the video timeline (≤ 1 ms).
  - **1080p avc1 video-only playback with ≥ 20 random seeks**, tried in §6.1's order.
  - URL expiry and refresh.
  - Network audit: only YouTube and Google video hosts may appear.
- **Must deliver:** YouTubeKit results, 1080p playback with seeking, the ingest method, and expiry.
- **Cut first:** the resource-loader variant, and the yt-dlp Gatekeeper test.
- **Decides:** the default and fallback resolver, the video strategy, and the ingest method.

### S5 Sync (days 5–7)

- **Build:**
  - A 60 fps test video with no audio, carrying a frame code and a flash patch.
  - Synthetic dub units: chirps on AVAudioEngine, placed on flash frames.
  - A SyncEngine prototype:
    - start and seek aligned by host time;
    - edits run *at their moment*;
    - re-anchoring on every timebase and stall event.
- **Measure:**
  - Audible start error per unit (p50 / p95 / max) and drift, across steady playback, rate changes, slow-downs, freezes, seeks, stalls and pause.
  - Each run with and without `sourceClock`, on each output device.
  - User speed: offline re-render vs. live TimePitch. The unit that's already playing is never left running at 1.0× past its slot.
  - Ground truth from the capture session.
- **Must deliver:** built-in speaker results, with and without `sourceClock`, and one validated session.
- **Cut first:** AirPods, the muxed variant, and the live-TimePitch alternative.
- **Decides:** `sourceClock` or not, the edit mechanism, user-speed handling, and the production drift metric.

### S4 TTS (days 8–11)

- **Build:**
  - The D2 patch.
  - `maata-convert-chatterbox`, which produces the full offline bundle with its licence files.
  - Adapters for Chatterbox-Telugu, plus the capped D6 comparison.
- **Measure:**
  - RTF and time to first audio for ~2 / 5 / 10 s lines.
  - Load time, and memory in fp32 and fp16.
  - Round-trip CER, using Omnilingual calibrated on held-out Telugu.
  - Speaker similarity: English reference → Telugu output, against a Telugu upper bound and a preset baseline.
  - Tenglish: CER on the Telugu spans, plus English-term intelligibility from the blind sheet.
  - Rate control by the D7 rule. Candidates: native controls (Chatterbox `cfgWeight`/`exaggeration`, OmniVoice `duration:`), TimePitch, and WSOLA.
- **Must deliver:** the Telugu bundle loading in Swift, with Chatterbox-Telugu fully measured, plus samples.
- **Cut first:** Indic-Mio, then WSOLA, then OmniVoice.
- **Decides:** the default engine, the rate-control method, and fp16 vs. fp32.

### S2 Front-end (days 12–14)

- **Candidates:**
  - Diarization: Sortformer vs. Community-1 + VBx (Core ML).
  - Word timing:
    - (a) Whisper + Qwen3ForcedAligner (11 languages, not Hindi);
    - (b) Whisper + our own DTW over alignment heads;
    - (c) WhisperKit;
    - (d) Qwen3-ASR + the aligner;
    - (e) Nemotron, if D4 allows.
  - Language ID: Whisper vs. SpeechLanguageID.
- **Measure:**
  - DER, speaker count and WER.
  - Word start and end error, and bias.
  - Pause detection.
  - `s_u`/`e_u` error after a draft Segmenter.
  - Dropped words.
  - RTF and memory.
  - Where Core ML places each model, and contention with Chatterbox running on the GPU.
- **Must deliver:** English results for the diarizers and for (a) and (b).
- **Cut first:** (d), (e), Hindi word timing, and SpeechLanguageID.
- **Decides:** the diarizer, the word-timing strategy for each language, and language ID.

### S3 Translation (days 15–17)

- **Candidates:**
  - TranslateGemma 4B and 12B, with the loader fixes applied in our adapter at load time: no fork, pinned files unmodified;
  - Gemma 4 E4B or Qwen3-4B for JSON and condensing;
  - MADLAD-400 3B for quality, latency and memory only.
- **Pipelines:**
  - (i) TranslateGemma + an instruct model;
  - (ii) an instruct model alone;
  - (iii) MADLAD + an instruct condenser.
- **Measure:**
  - Latency and memory.
  - JSON validity.
  - Length accuracy against 100 / 85 / 70 % targets, split into pure-Telugu and code-mixed lines.
  - Condense hit rate.
  - Your scores.
  - An A/B of the RoPE fix.
- **Must deliver:** TranslateGemma 4B + instruct, and MADLAD.
- **Cut first:** the 12B and 8-bit variants, and pipeline (iii).
- **Decides:** one model vs. a pair, and the models for Recommended and for Standard.

## Checking §10 (day 18)

The report has a "Phase 0 vs §10" table. Every row is labelled *composed, not end-to-end*.

- **Paste → start:**
  - Warm: S1 resolve and the first 60 s, plus S2 on the first window, plus S3 and S4 for units 1–2.
  - Cold: the same, plus each model's load time.
- **Throughput:** the combined S3 + S4 GPU queue must stay ≤ 40 s per video-minute, measured with S2 contention included.
- **Peak memory:** one `maata-bench corun` in a single process, for Recommended (**14.4 GB**) and Standard (**9.6 GB**).
- **Start error:** from S5.
- **Added video time:** not measurable until Phase 4.

**Memory estimates** (weight sizes; not measured):

| Preset | Configuration | Estimate | Budget |
|---|---|---|---|
| Recommended | 4B path | 11.7–13.7 GB, depending on the condenser | 14.4 GB |
| Recommended | 12B + condenser | 16.2–18.1 GB | 14.4 GB |
| Standard | (ii) one 4B 4-bit model | ≈ 8.1 GB | 9.6 GB |
| Standard | (iii) with fp16 Chatterbox | ≈ 10.0 GB (≈ 9.0 GB without the aligner) | 9.6 GB |

So 12B is probably High Quality only. The Standard candidates are (ii), and (iii) without the aligner.

## Risks

1. **YouTube.** Both resolvers depend on one client, and made-for-kids videos are expected to fail. That's inferred from source; S1 measures it.
2. **Word timing** needs a second model or new code, and Hindi has no aligner.
3. **Memory.** 12B doesn't fit at 24 GB. The 4B path may have as little as 0.7 GB of headroom, and Standard is tight.
4. **Licences.** Several HF labels are wrong, so the manifest carries licence strings we've verified ourselves.
5. **Toolchain.** speech-swift's CI builds only with Xcode 16.4. mlx-swift's CI Xcode is unknown, so no upstream CI is known to cover Xcode 26.x or 27.
6. **Your inputs** are on the critical path, especially the video list, the capture session and the S3 scoring.

## Deliverables

- `docs/SPIKES.md`, where every number is backed by committed JSON, with one recommendation per spike.
- An ADR for every pin and default.
- The §10 table.
- Listening samples.
- The Telugu bundle, for you to publish.
- Draft upstream PRs.
- A `MODELS.md` stub.

Then I stop.
