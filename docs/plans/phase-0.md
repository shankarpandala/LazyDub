# Phase 0 plan: spikes

Status: **proposed; waiting for the maintainer's go-ahead** (spec §0). Nothing in this plan has been built yet.

The inputs are `docs/SPEC.md` and `docs/research/2026-09-23-dependency-survey.md`, the verified dependency survey. The survey changed several assumptions in the spec; section 1 lists them.

## 0. Where this runs

- **The Mac runs everything.** Every spike builds, runs and is measured on the reference M5 Pro (24 GB), from a Claude Code session on that machine (the Desktop app or `claude remote-control`).
- **The cloud container runs nothing that counts.** It has no Xcode, Metal or Neural Engine, and YouTube serves it a bot wall. It can only do research and MLX-free pure-Swift work.
- **Recorded with every result:** macOS build, Xcode build, power source, and whether other apps were running.
- **First step in the local session:** record `sw_vers`, `xcodebuild -version` for each installed Xcode, `xcode-select -p`, and free disk space.

## 1. What the research changed

| Spec assumption | What's actually true (see the survey) | Effect on Phase 0 |
|---|---|---|
| speech-swift's Chatterbox port just needs converted weights | The loader hard-codes vocab 2352 and the language gate lacks `te`. Its tokenizer can't read the Telugu JSON, and when it can, it turns most Telugu into `[UNK]`. The fix is verified in Swift (3007/3007 fuzz strings match HF). | S4 needs a speech-swift patch (fork or adapter; see Q2). The weight conversion itself is a key rename on one safetensors file, so the Swift converter is small. |
| WhisperASR gives word timestamps | It gives text only. It also silently drops the rest of a 30 s window after a word repeats 3×, and detects language only from the first 30 s. | S2 becomes "ASR + a word-timing strategy". Candidates: Qwen3ForcedAligner (English only, not Hindi), our own DTW over the Whisper bundle's alignment heads, or WhisperKit (Q3). |
| TranslateGemma handles the whole §6.4 request | Its strict template is translate-only. mlx-swift-lm drops its ×8 linear RoPE scaling on global layers and has no EOS id. | S3 fixes the loader first, then tests TranslateGemma alone vs. TranslateGemma + an instruct model for JSON, length targets and condensing. |
| OmniVoice is Apache-2.0 | Weights are CC-BY-NC, and the codec is under the Boson Higgs Audio license. Indic-Mio has NC training data and an ambiguous WavLM license. | S4 evaluates both for comparison only. Neither goes in a shipped preset without an explicit decision (Q6). |
| "Build with stable Xcode 26; 27 is beta" | Xcode 27.0 went GA on 2026-09-14. Xcode 26.4+ resolves mlx-swift 0.31.6 (Swift 6.3 + a CudaBuild plugin prompt). | Pin mlx-swift exact 0.31.4 and the Xcode version explicitly (Q1). |
| YouTubeKit and yt-dlp are independent fallbacks | Both depend on essentially one PO-token-free client (visionOS). `yt-dlp_macos` is a frozen Python interpreter, and Deno isn't bundled. | S1 measures both against the same videos. Q4 asks whether `yt-dlp_macos` is acceptable under §3.2. |
| The Swift port watermarks like Python | It doesn't. The Telugu model card says every output carries PerTh. | Q7. |

## 2. Groundwork (timebox: 1 day)

This is the minimum that makes the spikes reproducible. Anything under `Spikes/` is throwaway; everything else is kept.

- **`project.yml` (XcodeGen 2.46.0).**
  - `SWIFT_VERSION: 6.0` set explicitly (XcodeGen's preset defaults to 5.0), complete strict concurrency, deployment target macOS 15.0, arm64 only.
  - Targets: a placeholder `Maata` app and a `maata-bench` CLI.
  - Not included: `-undefined dynamic_lookup`.
- **Local packages under `Packages/`:**
  - **`MaataCore`:** shared types (`VideoTime`, `AudioClip`, `SourceUnit`, `DubUnit`, `Speaker`, `LanguageCode`), the §5 engine protocols, and the preliminary Telugu `AksharaCounter` needed by S3 (see §3.3). MLX-free, and `swift test`-able on macOS and Linux.
  - **`Diagnostics`:** the bench result schema (JSON), stage timers, a peak-memory sampler, WER/CER/DER scoring, and percentile and histogram helpers. MLX-free.
- **`maata-bench`:**
  - Subcommands `stream | asr | diarize | align | translate | tts | sync`. Each writes one JSON result, with fields: tool git SHA, machine info, engine, model id + revision, input hash, cold/warm flag, per-stage wall time, RTF, time to first output, peak `phys_footprint`, MLX peak memory, plus spike-specific fields.
  - Ships with `mlx-swift_Cmlx.bundle` beside the binary; a CLI target doesn't embed SwiftPM resources.
- **Pins recorded as ADRs** in `docs/DECISIONS.md`:
  - speech-swift commit (or fork commit), mlx-swift **exact 0.31.4**, mlx-swift-lm 3.31.4 (transitive), YouTubeKit 0.4.9, yt-dlp 2026.08.19 + Deno 2.9.7 (checksums), XcodeGen 2.46.0, and the Xcode version.
  - `Package.resolved` is committed.
- **CI** (`.github/workflows/ci.yml`, `macos-26` runner, pinned Xcode): `xcodegen generate`, build, and MLX-free unit tests. No model tests (7 GB runners; their virtual ANE can't run stateful Core ML models).
- **Docs:** `docs/SPIKES.md` (results) and `docs/TEST_VIDEOS.md` (a template with the §11 categories, for you to fill in); `CLAUDE.md` updated with real commands.

Spike code lives in `Spikes/S1…S5/` with its own XcodeGen spec, and is deleted or archived after the report.

## 3. Spikes

Each spike stops at its timebox and reports what it has. The timeboxes are caps, not estimates, and spikes that don't wait on your inputs run first.

### S1: Streams (timebox: 2 days)

**Build**

- `StreamResolver` implementations behind the §5 protocol:
  - **YouTubeKit:** always `methods: [.local]`. One `YouTube` instance per extraction, inside an actor, with extractions serialised. Our own video-ID/`t=` parser.
  - **yt-dlp:** `yt-dlp_macos` + Deno from a dev tools folder, checksums (and the yt-dlp GPG signature) verified. Flags: `--no-remote-components --ignore-config --no-plugin-dirs --no-js-runtimes --js-runtimes deno:<path> --cache-dir <ours> -J`, never `-U`.
- A minimal AVPlayer test app.

**Measure** on every `TEST_VIDEOS.md` entry (see Q9), plus edge cases: made-for-kids, age-restricted, ended live, private/deleted, > 1 h, 4K.

- **Resolution:** success rate and resolve latency, cold and warm (JS cache).
- **Stream properties:** for each resolved stream, client `c=`, itags and codecs, `clen`, `expire`, and whether `pot` is present.
- **Audio fetch:** full itag-140 fetch in 10 MiB ranges (throughput, mid-file 403s), and **time from resolve to the first 60 s of audio decoded to PCM (target ≤ 2 s)**. Candidates, fastest first:
  - (a) a byte-range fetch of the init segment plus the first fragments, then AVAssetReader on the partial file;
  - (b) AVFragmentedAsset;
  - (c) our own fMP4 fragment reader feeding AudioConverter.
- **Video playback:** a 1080p avc1 video-only stream in AVPlayer, played directly vs. through an `AVAssetResourceLoaderDelegate` chunker. Time to first frame, seek latency (20 random seeks), stalls. Fallback: HLS (where offered), then muxed itag 18 with the audio track disabled.
- **Expiry:** URL lifetime, behaviour after expiry and after a network change, and re-resolve plus byte-range resume.
- **Network audit:** every host contacted, via a logging `URLProtocol` for our own sessions plus `nettop` for subprocesses. It must be only YouTube and Google video hosts.

**Decides:** default resolver and fallback, video strategy, and audio ingest strategy (ADRs).

### S2: Front-end (timebox: 3 days)

**Candidates**

- **Diarization:** Sortformer streaming (Core ML) vs. Community-1 + VBx (Core ML), using 60 s windows with 5 s overlap. Cross-window speaker reconciliation uses WeSpeaker cosine similarity.
- **ASR + word timing:**
  - (a) speech-swift WhisperASR + Qwen3ForcedAligner (English);
  - (b) WhisperASR + our own DTW over the bundle's alignment heads (every Whisper language, including Hindi);
  - (c) WhisperKit with `wordTimestamps` (**only if Q3 approves it**);
  - (d) Qwen3-ASR + aligner as a secondary option.
  - Nemotron streaming (word timings for en/hi/te) only if all of the above fall short, since its license needs review.
- **Language ID:** Whisper's first-window detection vs. SpeechLanguageID (VoxLingua107) over several windows.

**Measure**

- DER (collar 0.25 s, and no collar) and speaker-count accuracy.
- WER, normalised.
- Word-start error vs. labels (median / p95).
- RTF and time to first 60 s window, cold and warm.
- Peak memory.
- Compute placement: which units Core ML actually uses, checked once per model in Instruments.
- The known Whisper failure modes: the repeat cutoff, 30 s window seams, and language-ID misses on intros.

**Data**

- Your labelled fixtures (Q9).
- Until they arrive: AMI meeting-corpus excerpts (CC-BY 4.0, with reference diarization and words) and LibriSpeech (CC-BY 4.0), if Q8 approves.
- Scoring code (DER/WER/CER) is written in Swift in `Diagnostics`, with unit tests.

### S3: Translation (timebox: 3 days)

**Candidates**

- TranslateGemma 4B and 12B (mlx-community 4-bit, plus 8-bit for 4B) through mlx-swift-lm, after two fixes:
  - inject `rope_scaling: {type: linear, factor: 8}`;
  - set explicit stop tokens.
- Condenser / JSON model: Gemma 4 and Qwen3-4B instruct.
- MADLAD-400 3B (speech-swift, int4 and int8) as the non-LLM baseline.

**Build**

- A prompt harness implementing the §6.4 request (context of the previous 3 units, glossary, target aksharas, JSON output with one retry, then plain fallback) and the §6.4 condense call (2–3 candidates).
- Two pipelines:
  - (i) TranslateGemma in its own template plus the instruct model for condense/JSON;
  - (ii) the instruct model alone.

**Measure**

- Latency per unit (time to first token, total; p50 / p95) and tokens/s.
- Peak memory, for each model alone and for each translate-plus-condense pair loaded together.
- JSON validity before and after the retry.
- Length accuracy: the distribution of `|aksharas − target| / target`, and the condense hit rate (≤ N aksharas).
- Your 1–5 scores (adequacy, natural spoken Telugu, Tenglish handling) on the ~50-sentence set (Q9).
- An A/B of the RoPE fix itself, so we know it matters.

**The akshara counter** is preliminary, in `MaataCore`: grapheme clusters minus spaces and punctuation, with explicit rules for word-final pollu (`న్` counts 0.5 → configurable) and ZWNJ half-forms.

- The research verified that Swift 5.7+ treats conjunct + matra as one `Character` (GB9c), on Linux toolchains.
- S3 re-runs the same tests on macOS, because segmentation there comes from the OS's `libswiftCore`.

### S4: TTS (timebox: 4 days)

**Build**

1. **The speech-swift patch** (per Q2):
   - `T3Config` from `config.json`'s `vocab_size`.
   - Tokenizer: added tokens matched on Unicode scalars, literal `[SPACE]` replacement, both merge formats accepted.
   - The language set derived from the `[xx]` tokens present in the tokenizer, so `te` works without hard-coding.
   - A public `ChatterboxS3GenRef` init, so voice conditioning can be cached to disk.
   - A cancellation check and a seedable RNG in the T3 sampling loop.
   - Each item gets a golden test against HF `tokenizers` ids (Q5).
2. **`maata-convert-chatterbox`:** a Swift dev CLI, not shipped in the app.
   - Inputs: `t3_mtl_te.safetensors` and the aufklarer MLX bundle.
   - Output: a complete offline bundle: `model.safetensors` (VE + S3Gen + renamed Telugu T3), `conformer.safetensors`, `s3_tokenizer.safetensors`, `tokenizer.json`, `Cangjie5_TC.json`, `config.json`, and a README with CC-BY-4.0 and MIT attribution.
   - You publish it on Hugging Face.
3. **Adapters** for chatterbox-telugu (primary), Indic-Mio and OmniVoice behind `VoiceCloningTTS`, with in-memory conditioning caches.
4. **Rate control:** Chatterbox has none, so compare AVAudioUnitTimePitch in offline manual rendering against WSOLA at 1.0 / 1.1 / 1.2×. OmniVoice's `duration:` parameter is tested as a native alternative.

**Measure**

- RTF and time to first audio per unit, bucketed by line length (≈2 / 5 / 10 s).
- Voice-conditioning preparation time.
- Peak memory. The loader upcasts to fp32; we'll measure whether keeping fp16 is worth a further patch.
- Round-trip CER through a Telugu ASR judge. The judge (Omnilingual CTC 1B/3B, or Nemotron te-IN if approved) is first calibrated on FLEURS Telugu (CC-BY 4.0), so its own error floor is known.
- Cross-lingual speaker similarity: English reference → Telugu output via WeSpeaker cosine, against an upper bound (Telugu reference → Telugu) and a preset-voice baseline.
- Tenglish sentences (30, drawn from the S3 set).
- Failure rates: truncation at `maxNewTokens`, repetition, `[UNK]` counts.
- Listening samples in `docs/spikes/s4-samples/`, generated only from self-recorded or CC-BY references (LibriTTS-R if Q8 approves).

### S5: Sync (timebox: 3 days)

**Build:** a small macOS app.

- **Test media:** a generated test video (CC0, made with AVAssetWriter) with a visible frame counter.
- **Synthetic dub units:** tone bursts with known onsets, scheduled on AVAudioEngine.
- **Planning:** units are planned in video time and converted to host time through the player item's timebase.
- **Start, seek and timeline edits:**
  - Start and seek via `preroll(atRate:)` + `setRate(_:time:atHostTime:)`, so video and audio start on the same future host time.
  - Timeline edits (0.85× slow-down, freeze-frame ≤ 1.5 s) are scheduled with `setRate(_:time:atHostTime:)`, and the audio is re-anchored at each edit.

**Measure** actual vs. planned start of every unit, two ways:

- **Internal (becomes the production metric):** the output render callback's host time + sample offset + the device's presentation latency, mapped into video time through the item timebase.
- **External (ground truth, used once to validate the internal metric):** a ScreenCaptureKit capture of the app's frames and audio on the host clock, analysed offline (frame counter vs. tone onsets).

**Scenarios:** 10-minute steady playback; user rates 0.75–1.5× (dub time-stretched); timeline slow-downs and freezes; 30 seeks; injected stalls (the resource loader delays bytes); play/pause cycles.

**Conditions:**

- With and without `AVPlayer.sourceClock` = the output device's audio clock.
- Output devices: built-in speakers, wired headphones, and AirPods. Bluetooth latency is large, and the metric must account for it.

**Report:** p50 / p95 / max start error per scenario, and drift over time. Targets: p95 ≤ 50 ms, max ≤ 120 ms; the Phase 1 gate is p95 ≤ 80 ms.

## 4. Measurement rules (all spikes)

- **Setup:** the M5 Pro on AC power, no other heavy apps. Each configuration gets one warm-up and then ≥ 5 runs. Report p50 / p95 / max, plus n.
- **Cold vs. warm:**
  - *Cold* = a fresh process with Core ML models already compiled, and whatever MLX JIT kernel compilation happens in that process (only 9 kernels are precompiled).
  - *Warm* = the same process, second run onward.
- **Memory:**
  - Peak = the maximum `phys_footprint` from `task_info`, sampled every 50 ms, cross-checked once per spike with `footprint`.
  - MLX's own peak counter is reported alongside.
  - Budget: 60% of 24 GB = 14.4 GB.
- **Storage:** raw JSON under `docs/spikes/results/<spike>/`, summarised in `docs/SPIKES.md`. No number goes into a doc unless a committed JSON backs it.
- **Unmet targets:** a missed target is reported as measured, with options (spec §0).

**Memory estimate for the Recommended preset** (file sizes, *not measured*; S2–S4 replace it):

| Model | Estimated memory |
|---|---|
| Whisper turbo | 1.6 GB |
| Aligner | ~0.5 GB |
| Diarization + VAD | < 0.4 GB |
| Chatterbox, fp32 in memory | ~3.9 GB |
| TranslateGemma 4B 4-bit | 2.2 GB |
| (alternatively) TranslateGemma 12B 4-bit | 6.7 GB |
| A separate 4B instruct condenser | ~2.3 GB |
| App + AVPlayer | ~1 GB |

That puts the 4B path near 12 GB and the 12B path near 16.5 GB, over the 14.4 GB budget before KV caches. Expect 12B to be High Quality only, unless it can also do the condensing.

## 5. Modules touched

- **Kept:** `project.yml`, `Packages/MaataCore`, `Packages/Diagnostics`, `maata-bench`, CI, `docs/DECISIONS.md`, `docs/SPIKES.md`, `docs/TEST_VIDEOS.md`, `CLAUDE.md`.
- **Throwaway:** `Spikes/S1…S5`.
- **Outside this repo:** the speech-swift fork/patch (Q2) and the published Telugu MLX bundle (you publish it).

## 6. Risks

1. **YouTube.** Both resolvers depend on one client that needs no PO token; made-for-kids videos and `.web`-only itags already fail. Mitigation: keep both resolvers, and make a yt-dlp pin bump a manifest update rather than an app release.
2. **Word timing** now needs a second model or new code (see S2). Hindi source has no aligner at all today.
3. **Memory:** the 12B translator likely doesn't fit alongside everything else at 24 GB (the estimate in §4).
4. **Licenses:** OmniVoice (NC), Indic-Mio (NC data, WavLM SA?), Sortformer (NVIDIA OML), Gemma ToU. The manifest must carry verified license strings, because several Hugging Face labels are wrong.
5. **Toolchain drift:** Xcode 27 is GA but untested upstream; 26.4+ changes the mlx-swift resolution.
6. **speech-swift is Swift 5 mode and tracks `main`.** We pin a commit, wrap it in actors, and never use its online download paths (no revision pinning; 110 s offline retry loop).
7. **Your inputs are on the critical path:** S2 fixtures, the S3 sentences, and S4 listening.

## 7. Decisions needed from you

| # | Question | My recommendation |
|---|---|---|
| Q1 | Xcode: the spec says "stable Xcode 26", but 27.0 went GA on 2026-09-14. Which exact version do we build with? | The newest installed **26.x**, named in an ADR, with mlx-swift pinned **exact 0.31.4**. Revisit Xcode 27 after Phase 1, once upstream CI covers it. |
| Q2 | Chatterbox Telugu: fork speech-swift or write an adapter? | **Fork under your account** (`shankarpandala/speech-swift`) holding the minimal patch set from S4; each item goes upstream as a PR, and we pin the fork commit until it merges. An adapter-only route would copy ~150 lines of private loader code and can't add disk-cached conditioning or cancellation. I'll need the fork added to the session. |
| Q3 | Approve **WhisperKit** (MIT, v1.1.0) as an S2 candidate? The spec doesn't name it. | Yes, as a candidate only. It gives word timestamps and language ID for every Whisper language, including Hindi. |
| Q4 | `yt-dlp_macos` is a PyInstaller-frozen **Python interpreter**. Is it acceptable under §3.2, given that §6.1 sanctions yt-dlp for resolution and fetch only? | Yes: it does no inference, runs as a subprocess, and is pinned and verified. Without it, YouTubeKit is a single point of failure. |
| Q5 | May a dev-only Python script (outside the app, never shipped) generate golden token ids with HF `tokenizers`? Or will you supply goldens from your training environment? | Either works; the Python script is faster to start with. |
| Q6 | OmniVoice (CC-BY-NC + Boson) and Indic-Mio (NC training data): evaluate for comparison only? | Yes. Neither ships in a preset unless you decide otherwise after S4. |
| Q7 | Watermarking: the Swift port applies none, but your model card says every output is watermarked. Port PerTh (MIT, ~37 MB net) to MLX, or drop the claim for Swift output? | Decide after S4; S4 can measure PerTh's cost if you want it ported. |
| Q8 | May the spikes use AMI, LibriSpeech, LibriTTS-R and FLEURS (all CC-BY 4.0), with small attributed excerpts committed as fixtures? | Yes. It unblocks S2–S4 before your fixtures arrive. |
| Q9 | Your inputs: `TEST_VIDEOS.md` URLs; labelled diarization/ASR fixtures (RTTM + word-timed transcripts, including one Hindi clip); the ~50-sentence S3 set; listening sessions. | Shall I draft the S3 sentence set for you to edit? It would cover numbers, dates, currency, names, tech terms, idioms, and short and long lines. |
| Q10 | Name: the repo is `LazyDub`, the spec says `Maata`. Which one goes in module prefixes, the bundle ID and cache paths? | `Maata` (spec), unless you prefer otherwise; renaming later touches everything. |

## 8. Deliverables at the end of Phase 0

- `docs/SPIKES.md`: every measurement, backed by committed JSON, with a recommendation per spike.
- ADRs in `docs/DECISIONS.md` for every pin and every default chosen.
- S4 listening samples.
- The Telugu Chatterbox bundle, ready for you to publish.
- Upstream PRs drafted for the speech-swift patches.
- A report against the §10 targets, where Phase 0 can measure them. Then stop.
