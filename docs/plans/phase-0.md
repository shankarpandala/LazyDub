# Phase 0 plan: spikes

Status: **proposed; waiting for the maintainer's go-ahead** (spec §0). Nothing is built yet.

- The survey this plan builds on: `docs/research/2026-09-23-dependency-survey.md`.
- How every number gets measured (memory, cold vs. warm, sync mechanics, offline enforcement): `phase-0-methods.md`.

## Goals

1. Pick the default engine for each stage, and the resolver, ingest and sync approaches, with measurements from the M5 Pro.
2. Show whether the §10 targets can be met with those choices, for the Recommended preset (24 GB) and the Standard preset (16 GB).
3. Produce the Telugu Chatterbox MLX bundle and the speech-swift patch that loads it.

Everything is built and measured on the reference Mac, from a local Claude Code session. The cloud container can't run Xcode, Metal or the Neural Engine, and YouTube blocks it.

## What the research changed

| Spec assumption | What's actually true | Consequence |
|---|---|---|
| speech-swift's Chatterbox only needs converted weights | Its hard-coded vocab and language gate reject Telugu, and its tokenizer turns most Telugu into `[UNK]`. A fix is verified in Swift. | A small speech-swift patch (Q2). Converting the weights is only a key rename. |
| WhisperASR gives word timestamps | It returns text only, and it can silently drop text. | S2 compares strategies for word timing. |
| TranslateGemma covers the whole §6.4 request | Its template only translates. mlx-swift-lm skips its RoPE scaling. | S3 tests TranslateGemma alone and paired with an instruct model. |
| OmniVoice is Apache-2.0 | Its weights are CC-BY-NC, and its codec uses Boson's license. | Comparison only (Q6). |
| Build with stable Xcode 26, since 27 is beta | Xcode 27.0 went GA on 2026-09-14. | Name the exact Xcode and pin mlx-swift to 0.31.4 (Q1). |
| The two resolvers are independent fallbacks | Both depend on essentially one YouTube client, and `yt-dlp_macos` is a frozen Python binary. | Q4. |

## Groundwork (1 day)

- **Project setup.**
  - `project.yml` for XcodeGen 2.46.0: `SWIFT_VERSION: 6.0`, complete strict concurrency, macOS 15, arm64 only.
  - `Package.resolved` is committed through a `.gitignore` exception inside the generated project, and I'll check that `xcodegen generate` preserves it. CI and the bench scripts build with `-onlyUsePackageVersionsFromResolvedFile`.
- **Packages I keep.** These are the real Phase 1 homes, so no bench code depends on throwaway code:
  - **`MaataCore`:** types, the engine protocols from §5, and the Telugu `AksharaCounter`.
  - **`Diagnostics`:** the result JSON schema, memory and timing probes, WER/CER/DER scoring, and percentiles.
  - **`StreamResolving`:** the URL / video-ID / `t=` parser, with §11 unit tests.
  - **`SpeechFrontEnd`, `Translation`, `Synthesis`:** the S2–S4 engine adapters. Adapters for engines that lose are deleted after the report.
  - **`Tools/maata-convert-chatterbox`:** the Swift CLI that builds the Telugu bundle.
- **`maata-bench`.** One subcommand each for `fetch`, `asr`, `diarize`, `align`, `translate` and `tts`, plus `corun` for co-residency and contention runs.
  - It ships with `mlx-swift_Cmlx.bundle` beside the binary.
  - The S1 and S5 prototypes need AVPlayer, so they're throwaway apps under `Spikes/` that write the same JSON.
- **Models stay offline.**
  - `models.lock.json` pins repo, commit, file, size and sha256 for every model and tool.
  - `maata-bench fetch` downloads and verifies them.
  - Every spike loads only from local paths, and S2–S4 runs execute with outbound network denied (details in methods §1).
  - No library downloads anything on its own, not even temporarily.
- **ADRs** in `docs/DECISIONS.md` pin:
  - speech-swift (fork) commit;
  - mlx-swift **exact 0.31.4**;
  - mlx-swift-lm **exact 3.31.4**, declared directly because S3 links MLXLLM;
  - YouTubeKit 0.4.9;
  - yt-dlp and Deno by sha256;
  - XcodeGen;
  - the Xcode version.
- **CI** on GitHub `macos-26` with a pinned Xcode: generate, build, and run the MLX-free tests. No model tests.

## Spikes

Timeboxes are caps: when one runs out, the spike reports what it has. S1 and S5 need no input from you, so they go first.

### S1: Streams (2 days)

**Build**

- A YouTubeKit resolver that only ever passes `methods: [.local]`.
- A yt-dlp + Deno resolver, **only if Q4 approves it**.
- A throwaway AVPlayer app.

**Measure**, on `TEST_VIDEOS.md` plus out-of-scope cases (live, upcoming premiere, members-only, age-restricted, made-for-kids, private) and every URL form:

- **Resolution:** success rate, and cold and warm latency.
- **Errors:** whether each out-of-scope case produces its own distinct error.
- **Stream details:** client and itags seen, `clen`, `expire`, and whether `pot` is present.
- **Audio:** time from resolve to the first 60 s of decoded PCM (**target ≤ 2 s, reported by video length up to ≥ 2 h**).
  - The first sample's alignment to the video timeline (AAC priming, edit lists), with a pass mark of ≤ 1 ms.
  - Full-file fetch in ranges.
- **Video**, in §6.1's order: direct video-only MP4, then HLS (*see Q11*), then a resource-loader cache, then muxed with the audio track disabled.
  - Time to first frame, seek latency, and stalls.
- **Expiry:** URL lifetime and refresh.
- **Network audit:** every host the app and its subprocesses contact.

**Decides:** the default and fallback resolver, the video strategy, and the ingest method.

### S2: Front-end (3 days)

**Candidates**

- **Diarization:** Sortformer or Community-1 + VBx, both Core ML. Speakers are reconciled across windows with WeSpeaker.
- **Word timing:**
  - (a) WhisperASR + Qwen3ForcedAligner. The aligner covers 11 languages including English, but not Hindi or Telugu.
  - (b) WhisperASR + our own DTW over the bundle's alignment heads, which works in every Whisper language.
  - (c) WhisperKit, *if Q3 allows*.
  - (d) Qwen3-ASR + the aligner.
- **Language ID:** Whisper's first-window guess, or SpeechLanguageID (*Q3*).

**Measure**, per language (English and Hindi), on your fixtures, and on AMI/LibriSpeech until they arrive (*Q8*):

- **Accuracy:** DER, speaker-count accuracy, and WER.
- **Word timing:** start and end error (median, p95, signed bias), detection of pauses over 300 ms, `s_u`/`e_u` error after a draft Segmenter, and the dropped-word rate.
- **Speed and memory:** RTF, first-load / cold / warm time, and peak memory.
- **Hardware placement:** where Core ML actually runs each model.
- **Contention:** RTF while Chatterbox is synthesizing on the GPU.

**Decides:** the diarizer, the word-timing strategy for each source language, and the language-ID method.

### S3: Translation (3 days)

**Candidates**

- **TranslateGemma 4B and 12B.** mlx-community 4-bit mirrors, plus 8-bit for 4B. Two loader fixes first: RoPE scaling, and `<end_of_turn>` as a stop token.
- **Instruct models** for JSON and condensing: Gemma 4 E4B and Qwen3-4B.
- **MADLAD-400 3B**, scored on quality, latency and memory only; it can't produce JSON or aim for a length.

**Pipelines**

- (i) TranslateGemma for translation, with an instruct model for condensing and JSON.
- (ii) An instruct model alone.
- (iii) MADLAD with an instruct condenser. This is the Standard-preset candidate.

**Measure**

- **Latency:** per unit (TTFT, total, p50/p95), and tokens/s.
- **Memory:** peak for each model alone and for each pair.
- **JSON validity:** before and after the retry.
- **Length accuracy:** `|aksharas − target| / target`, with targets set at 100 / 85 / 70 % of the unconstrained output. Reported separately for pure Telugu and code-mixed lines.
- **Condensing:** hit rate.
- **Quality:** your 1–5 scores on the ~50 sentences.
- **The RoPE fix:** an A/B test.

**AksharaCounter**

- Split text into script runs.
- Telugu runs count grapheme clusters, with rules for word-final pollu and ZWNJ.
- Latin runs use an English syllable heuristic ("machine learning" → 4).
- Digits count through their spoken form.
- Grapheme segmentation is verified on Linux Swift 5.10.1–6.4 and gets re-checked on macOS here.

**Decides:** whether translation and condensing use one model or a pair, plus the models and quantization for Recommended and for Standard.

### S4: TTS (4 days)

**Build**

- The **speech-swift patch** (Q2). It fixes the vocab, tokenizer and language set; makes voice conditioning cacheable to disk; and adds cancellation and a seedable RNG. Golden token-id tests come from Q5.
- **`maata-convert-chatterbox`**, which produces the full offline bundle with its licence files. See the methods doc §7 for the attribution and change notice.
- **Adapters** for Chatterbox-Telugu, Indic-Mio (WavLM pinned locally) and OmniVoice (the last two for comparison, Q6).
- **Rate control**, since Chatterbox has none: AVAudioUnitTimePitch in offline mode vs. WSOLA, at 1.0 / 1.1 / 1.2×.

**Measure**

- **Speed:** RTF and time to first audio for lines of about 2, 5 and 10 s; conditioning time; load time.
- **Memory:** peak, with fp32 and fp16 in memory.
- **Round-trip CER:** using a Telugu ASR judge, calibrated first on FLEURS.
- **Cross-lingual speaker similarity:** English reference → Telugu output, compared with a Telugu→Telugu upper bound and a preset-voice baseline.
- **Tenglish:**
  - CER on the Telugu spans, with Latin-script spans masked out;
  - English-term intelligibility, marked by you in a blind listening sheet;
  - `[UNK]`, truncation and repetition counts.
- **Rate-control quality:** CER and similarity drop at 1.2×, and render cost. The rule is decided in advance: take the cheapest method that stays within the agreed margin.

**Also produces** blind, engine-anonymised listening samples (CC-BY or self-recorded references only).

**Decides:** the default TTS engine, the rate-control method, fp16 vs. fp32, and how voice conditioning is persisted.

### S5: Sync (3 days)

**Build**

- **Test media:** a generated 60 fps test video with a frame code, a flash patch and chirp bursts. It is fragmented MP4 served through a custom-scheme resource loader, with a muxed variant whose audio track is disabled.
- **SyncEngine prototype:**
  - Start and seek use `preroll` + `setRate(_:time:atHostTime:)` at a future host time.
  - Timeline edits run *at their moment* from a host-clock timer, because that API re-anchors immediately and doesn't support HLS.
  - It re-anchors on every timebase, stall and rate notification.
  - A user speed change re-renders the pending units offline; the real-time audio graph always stays at 1.0×.

**Measure**, as audible host-time error per unit:

- **Internal:** a per-unit metric from rendered samples, compensated for output-device latency.
- **External:** validated against **ground truth** captured on a physical rig or a 240 fps camera (*Q12*).

**Scenarios**

- 10 min of steady playback;
- user rates from 0.75× to 1.5×;
- slow-downs and freezes;
- 30 seeks;
- injected stalls;
- pause/play.

Each scenario runs with and without `sourceClock` set to the device's audio clock, on built-in speakers, wired output and AirPods.

**Report:** p50 / p95 / max and drift over time. Targets: p95 ≤ 50 ms and max ≤ 120 ms.

**Decides:** `sourceClock` or not, the start/seek and edit mechanism, and the production drift metric.

## Checking the §10 targets

The final report has a "Phase 0 vs §10" table. Each row is labelled *composed, not end-to-end*:

- **Paste → start,** warm and cold: S1 resolve and first 60 s, then S2 on the first window, then S3 + S4 for units 1–2. Model load times are included.
- **Throughput:** the combined cost of the S3 and S4 GPU queue must stay ≤ 40 s per minute of video. The S2 contention run counts toward this.
- **Peak memory:** one run loads the candidate set in a single process (`maata-bench corun`), for Recommended (**14.4 GB** budget) and for Standard (**9.6 GB**). The Standard run happens on a 16 GB Mac if one is available (Q12).
- **Start error:** from S5.
- **Added video time:** can't be measured until Phase 4.

Estimate only (weight sizes, with Chatterbox upcast to fp32; not measured):

- The 4B path is about **11.7–13.7 GB**, depending on the condenser.
- 12B plus a condenser is about **16.2–18.1 GB**.
- 12B alone is about **13.9 GB**.

So 12B probably belongs in High Quality, and Standard needs something like pipeline (iii) with Chatterbox kept in fp16.

## Risks

1. **YouTube:** both resolvers depend on one client. Made-for-kids videos and itags that exist only on the web client are expected to fail; this is inferred from source, and S1 measures it.
2. **Word timing:** it needs a second model or new code, and Hindi has no aligner.
3. **Memory:** 12B at 24 GB; any LLM pair at 16 GB.
4. **Licenses:** several labels on Hugging Face are wrong, so the manifest carries verified strings.
5. **Toolchain:** no upstream CI covers any Xcode 26.x or 27; speech-swift's CI uses Xcode 16.4.
6. **speech-swift:** it compiles in Swift 5 mode, isn't Sendable, and its online paths aren't pinned. We wrap it in actors and use only its offline paths.
7. **Your inputs** are on the critical path for S2–S4.

## Decisions needed from you

| # | Question | Recommendation |
|---|---|---|
| Q1 | Which Xcode? 27.0 is GA; 26.4+ needs Tahoe 26.2 and 27 needs Tahoe 26.6. | Choose based on the Mac's macOS and a version CI can pin (`macos-26` has 26.0.1–26.6). Pin mlx-swift 0.31.4 either way. |
| Q2 | Fork speech-swift, or build an adapter? | **Fork under your account.** The minimal patch goes upstream as PRs, and we pin the fork until they're merged. An adapter can't add disk-cached conditioning or cancellation. |
| Q3 | Allow these extras the spec doesn't name? **WhisperKit** (MIT) as an S2 candidate; **SpeechLanguageID** (VoxLingua107, licence to be verified); **Nemotron** streaming ASR (openmdw-1.1, licence to be reviewed) as a fallback and as a Telugu CER judge. | Yes to all three, as candidates only. |
| Q4 | `yt-dlp_macos` is a frozen Python interpreter. Is it acceptable under §3.2, given §6.1? | Yes: it does no inference, and it's pinned and verified. Without it, S1 reports on YouTubeKit alone and lays out options. |
| Q5 | Golden token ids: can I use a dev-only Python script with HF `tokenizers`, or will you supply them? | Either works. |
| Q6 | OmniVoice (NC and Boson) and Indic-Mio (NC data): comparison only? | Yes. |
| Q7 | The Swift port adds no PerTh watermark, but your model card promises one. Port PerTh (MIT), or change the claim? | Decide after S4. `MODELS.md` records the finding either way. |
| Q8 | May the spikes use AMI, LibriSpeech, LibriTTS-R and FLEURS (CC-BY 4.0), with small attributed excerpts committed? | Yes. |
| Q9 | Inputs I need: TEST_VIDEOS URLs; labelled fixtures, including one Hindi; the ~50 S3 sentences; listening sessions. | Shall I draft the sentence set for you to edit? |
| Q10 | Should the modules, bundle ID and cache paths use `Maata` or `LazyDub`? | `Maata`. |
| Q11 | Apple doesn't support `setRate(_:time:atHostTime:)` on HLS, so HLS video would break the sync design. Can HLS come off §6.1's list? | Yes, unless S1 finds HLS is the only option. |
| Q12 | For S5 ground truth, do you have a 240 fps iPhone, or a USB audio interface with a photodiode and mic? Is a 16 GB Mac available for Standard numbers? | The iPhone is enough. |
| Q13 | The mlx-community TranslateGemma mirrors ship without the Gemma Notice file. Use them for the spikes, and decide on mirrors vs. `google/*` with the token flow after S3? | Yes. |

## Deliverables

- `docs/SPIKES.md` with a recommendation per spike. Every number in it is backed by committed JSON.
- ADRs for every pin and default.
- The §10 table.
- Listening samples.
- The Telugu bundle, ready for you to publish.
- Upstream speech-swift PRs.
- A `MODELS.md` stub.

Then I stop.
