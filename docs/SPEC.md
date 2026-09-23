# Maata — Build Spec for Claude Code

> **Amended 2026-09-23:** the product is now a cross-platform hybrid desktop app (Tauri + local inference engine, NVIDIA and Apple Silicon, YouTube embedded player). `SPEC-AMENDMENT-01.md` overrides §2, §3.2–3.5, §3.7, §4, §5, §6.7, §10 and §12 where they conflict.

Working name: **Maata** (మాట, Telugu for "word" or "speech"). Rename freely. This file is the source of truth for the project; read all of it before doing anything.

## 0. How we work

You are the lead engineer building a native macOS app. I'm the maintainer: an ML engineer who built the Telugu TTS model this app depends on, so skip ML basics and be thorough about Swift and Apple-platform trade-offs.

We work in phases (§12). For every phase:

1. Post a short plan — goals, modules you'll touch, risks, what you'll measure — and wait for my go-ahead.
2. Build it, with tests.
3. End with a report: what's done, measurements against the targets in §10, deviations, open questions. Then stop.

Ground rules:

- The hard constraints in §3 are non-negotiable. If one looks impossible, stop and lay out options. Never route around one, not even temporarily.
- The library and model notes in this spec were researched in September 2026 and will drift. Before using any dependency, read its current README and source, then pin an exact tag or commit. Record pins and significant choices as ADRs in `docs/DECISIONS.md`.
- Ask before adding any dependency this spec doesn't name.
- Never invent measurements. If a target is missed, report the real number and your options. Don't quietly relax a target or skip an acceptance check.
- Keep `main` building, commit in small logical steps, and keep `CLAUDE.md` current with build and test commands, the module map, and conventions.

## 1. Product

Maata plays YouTube videos dubbed into Telugu, entirely on the Mac. The user pastes a link and presses Play. Within seconds the video starts with its original audio muted, and each person in it speaks Telugu in an AI clone of their own voice, timed to what's on screen. Speech recognition, speaker separation, translation and voice synthesis all run locally on Apple Silicon, using models the user downloads once on first launch.

Experience principles, in priority order:

1. It feels in sync: each Telugu line starts when the original line started.
2. The voices sound like the original speakers speaking natural Telugu.
3. It starts fast, and never stalls without saying why.
4. It's honest: the UI always shows that the voices are AI-generated and what the app is doing ("Preparing Telugu audio…").

## 2. Scope

**In v1:**

- Public YouTube videos (regular and Shorts) by URL, in any source language, dubbed into Telugu. The source language is detected automatically; English is the main tuning target.
- Multi-speaker videos, with per-speaker voice cloning and a preset-voice fallback.
- Play, pause, seek, playback speed, volume, and resume position per video.
- First-run model download and management.
- An architecture where adding a target language (Hindi, Tamil, Kannada…) means adding a language pack and model entries, not rewriting the pipeline.

**Not in v1:**

- Live streams and premieres.
- Private, members-only, age-restricted or DRM content.
- Playlists.
- Exporting dubbed audio or video.
- Viewer subtitles (a debug transcript is fine).
- Lip-sync.
- Any cloud fallback.
- Intel Macs and iOS.

## 3. Hard constraints

1. **On-device inference only.** No cloud AI APIs, no remote inference, no telemetry or analytics. The network is used only for:
   - YouTube stream resolution and media fetch;
   - downloading models and tools from the pinned sources in the manifest;
   - an optional update check the user can turn off.
2. **Native runtime only.** Swift 6 and Apple frameworks, with ML running through MLX (mlx-swift) or Core ML. No Python, PyTorch, ONNX Runtime or sidecar inference processes at runtime. Dev-time conversion tools are a separate case (§6.5).
3. **Models are never bundled.** The app ships a model manifest (metadata only); users download weights on first run through the Model Manager (§7). Libraries must never download on their own: point each library's cache directory at our storage and run it in offline mode (speech-swift supports both).
4. **Native playback.** Video plays through AVFoundation (AVPlayer). The original audio is never played.
5. **Platform.** macOS 15 or later (speech-swift's floor), Apple Silicon only. Minimum 16 GB RAM (warn below that); 24 GB recommended.
6. **Open source.** App code is Apache-2.0 unless I say otherwise. Every model shows its own license at download and requires acceptance.
7. **Distribution.** GitHub Releases, as a Developer ID–signed, notarized DMG with Hardened Runtime. Not the Mac App Store: the app fetches YouTube media directly, which App Review doesn't allow.

## 4. Environment and conventions

- **Machines and toolchain:**
  - The dev and reference machine is an M5 Pro with 24 GB.
  - Xcode 26 (stable) and the Xcode 27 beta are both installed. Build with stable Xcode 26, selected through `xcode-select` or `DEVELOPER_DIR`.
  - The deployment target is macOS 15.
- **Project setup:** generate the project with XcodeGen (`project.yml`) so it's reproducible and never hand-edited. Feature code lives in local Swift packages. Ask before switching to Tuist or a checked-in project.
- **MLX builds:** MLX needs its compiled Metal shader library at runtime, so build and run through `xcodebuild`.
  - If you use `swift build` or `swift test` on packages that touch MLX, follow speech-swift's metallib instructions.
  - Keep MLX-free packages testable with plain `swift test`.
- **Code conventions:**
  - Swift 6 language mode with complete strict concurrency.
  - Actors for stateful services, `@Observable` models for UI state, and structured concurrency everywhere.
  - No force-unwraps outside tests.
  - One OSLog category per module.
  - Typed errors per module that map to user-facing messages.
  - No third-party UI frameworks.

## 5. Architecture

The core idea: fetch the audio ahead of playback, process it ahead of the playhead, and plan everything on a timeline measured in video time. The video clock is the master. The video only yields when the timeline deliberately slows or freezes it (§6.6).

```
YouTube URL
 └─ StreamResolver ─► video-only stream ────────────────────────────► AVPlayer (muted)
                  └─► audio-only stream ─► MediaIngest (progressive fetch + decode)
                                              │ 16 kHz mono PCM, look-ahead windows
                                              ▼
                                   VAD + diarization ─► SpeakerRegistry ─► voice references
                                              │                            + conditioning cache
                                              ▼
                                   ASR (word timestamps) ─► Segmenter (dubbing units)
                                              ▼
                                   Translator (duration-aware, context + glossary)
                                              ▼
                                   Voice-cloning TTS ─► DurationEstimator + IsochronyFitter
                                              ▼
                                   DubTimeline (audio placements + video rate/freeze edits)
                                              ▼
                                   SyncEngine: AVAudioEngine scheduled against AVPlayer time
```

Modules (local SPM packages; the names are suggestions):

| Package | Responsibility |
|---|---|
| MaataCore | Shared types (VideoTime, AudioClip, SourceUnit, DubUnit, Speaker), protocols, settings, LanguagePack |
| StreamResolving | URL parsing, stream resolution, expiry and refresh |
| MediaIngest | Progressive audio fetch, decoding, resampling, windowing |
| SpeechFrontEnd | VAD, diarization, ASR, alignment, segmentation, SpeakerRegistry |
| Translation | Translator and condenser, prompts, glossary, JSON validation, TextNormalizer |
| Synthesis | TTS adapters, voice-conditioning cache, loudness, time-stretch |
| Timing | DurationEstimator, IsochronyFitter, DubTimeline |
| Playback | AVPlayer wrapper, SyncEngine, timeline-edit execution, drift metrics |
| Orchestration | SessionController, priority Scheduler, memory governor |
| ModelManager | Manifest, downloads, verification, licenses, presets |
| Storage | Per-video cache, LRU eviction, persistence |
| Diagnostics | Logging, metrics, debug-HUD data, the `maata-bench` CLI |

Put every model behind a protocol so engines can be swapped based on the Phase 0 results. A starting sketch:

```swift
protocol StreamResolver: Sendable {
    // Returns a video-only playable stream, an audio-only stream, metadata and expiry.
    func resolve(_ url: URL) async throws -> ResolvedVideo
}
protocol SpeakerSegmenter: Sendable {          // VAD + diarization over one window
    func segment(_ window: AudioWindow) async throws -> [SpeakerTurn]
}
protocol Transcriber: Sendable {
    func transcribe(_ window: AudioWindow, language: LanguageCode?) async throws -> [TimedWord]
}
protocol Translator: Sendable {
    func translate(_ request: TranslationRequest) async throws -> TranslationResult
    func condense(_ request: CondenseRequest) async throws -> [String]
}
protocol VoiceCloningTTS: Sendable {
    func prepareVoice(references: [AudioClip]) async throws -> VoiceConditioning   // cacheable
    func synthesize(_ text: String, voice: VoiceConditioning,
                    language: LanguageCode, rate: Double) async throws -> AudioClip
}
protocol DurationEstimator: Sendable {
    func estimate(_ text: String, voice: VoiceID) -> Duration
    func observe(_ text: String, voice: VoiceID, actual: Duration)
}
```

**Hardware.** Run the Core ML stages (VAD, diarization, ASR) on the Neural Engine while the MLX stages (translation, TTS) run on the GPU. To start, push heavy MLX work through a single priority queue to avoid GPU contention and memory spikes; relax that only if measurements show it's safe.

**LanguagePack.** Everything language-specific lives here:

- each engine's language codes and tags;
- the TextNormalizer and the syllable counter;
- translation style presets;
- the fallback voice bank;
- the default speaking-rate model.

Telugu (`te`) is the first pack.

## 6. Pipeline details

### 6.1 Stream resolution and media

- Implement two StreamResolvers behind the protocol:
  - **YouTubeKit** — pure Swift, extraction runs locally. Its optional remote fallback sends requests to a third-party server, so it must stay off.
  - **yt-dlp + Deno** — yt-dlp now needs an external JavaScript runtime for full YouTube support. Both binaries would be pinned, checksummed tools downloaded by the Model Manager, and run as subprocesses only for resolution and fetch.

  Spike S1 picks the default. The other stays available as a fallback in Settings.
- **Video:** the best AVPlayer-playable video-only stream, up to 1080p by default (configurable). Test these in S1, in this order of preference:
  - a direct video-only MP4;
  - HLS, where YouTube offers it;
  - a local cache backed by a resource loader;
  - as a last resort, a progressive muxed MP4 with its audio disabled.
- **Audio:** the audio-only AAC (m4a) stream, used only for processing.
  - Fetch it faster than real time, and start processing before the download finishes (S1 chooses between AVFragmentedAsset and byte-range chunks).
  - Re-resolve on a 403 or expiry, and resume by byte range.
- **URLs:** accept `watch?v=`, `youtu.be/`, `/shorts/` and `/embed/`, including `t=` start offsets. Out-of-scope content fails fast with a specific, clear message.
- As soon as a valid URL is pasted, start resolving and processing. Play shows progress, and becomes available once the start conditions in §6.7 are met.

### 6.2 Speech front-end

- Decode to 16 kHz mono Float32 for analysis. Keep the original-rate audio for voice references.
- **Diarization:** work in look-ahead windows, starting at 60 s with 5 s of overlap.
  - Candidates: Sortformer (Core ML, incremental streaming with stable speaker IDs), and Pyannote Community-1 with VBx.
  - Reconcile speaker IDs across windows using speaker embeddings (WeSpeaker or CAM++), with the cosine threshold tuned in Phase 3.
- **ASR:** produces word timestamps. Detect the source language once per video, with a manual override in the UI.
  - Candidates: WhisperASR (Whisper Large-v3 Turbo on Core ML) as the default, with Qwen3-ASR as the alternative.
  - Add a forced aligner if timestamps are poor.
- **Segmenter:** builds dubbing units from words.
  - Split at sentence ends, speaker changes, and pauses longer than about 300 ms. Aim for 1.5–12 s per unit.
  - Never split mid-clause; merge fragments.
  - Skip backchannels under about 0.6 s ("yeah", "mm-hmm") unless they carry meaning.

### 6.3 Speakers and voice references

- For each speaker, collect 6–15 s of clean, single-speaker speech from the look-ahead window: no overlap, high VAD confidence, and little music or noise.
- Evaluate cleaning references with a speech-restoration model (Sidon) or vocal separation (HTDemucs). Keep a cleaner only if it measurably improves speaker similarity.
- Compute voice conditioning once per speaker, and cache it.
- Until a speaker has enough reference audio:
  - use a partial reference if it passes a quality threshold;
  - otherwise use a preset Telugu voice matched on pitch range.
  - Switch to the cloned voice only at a unit boundary, and at most once per speaker.
- When speech overlaps, dub the dominant speaker.

### 6.4 Translation

- Each request carries:
  - the unit's text and speaker;
  - the previous three units, in both source and Telugu, for coherence;
  - a per-video glossary of names, brands and technical terms, built automatically and kept consistent;
  - a target length in aksharas, derived from the unit's time budget and the voice's measured speaking rate (§6.6).
- **Default style:** natural spoken Telugu, the way a native speaker talks on YouTube, not bookish prose. Keep widely used English technical terms in Latin script (code-mixed "Tenglish"), which chatterbox-telugu is trained to speak. Offer a "more formal Telugu" style in Settings.
- Write numbers, dates, units and currency out as spoken Telugu words. A deterministic TextNormalizer catches anything the model misses.
- Output is strict JSON. Validate it; retry once on a parse failure, then fall back to a plain translation.
- **Condense:** rewrite an existing Telugu line to at most N aksharas while keeping its meaning. Return two or three candidates and pick the one closest to the target.
- Candidates for S3:
  - **TranslateGemma 4B and 12B** (Gemma 3–based). It uses a strict chat template that requires source and target language codes, and it accepts about 2K input tokens. Google's Hugging Face repos are gated behind the Gemma terms, so either handle the license and token flow or use a compliant mirror. Verify Telugu support and quality yourself.
  - **A general instruct model** (a Gemma 4 or Qwen3 variant) for condensing, if TranslateGemma condenses poorly outside its template.
  - **MADLAD-400 3B** as a non-LLM baseline.
  - Run them through mlx-swift-lm or the backends speech-swift already provides.

### 6.5 Voice-cloning TTS

- **Primary: chatterbox-telugu** (`shankarpandala/chatterbox-telugu`), my Telugu fine-tune of Chatterbox-Multilingual (checkpoint `t3_mtl23ls_v2`). What changed from the base model:
  - Only the T3 text side: LoRA merged into the weights, plus a retrained text embedding and head.
  - The grapheme tokenizer was extended with Telugu script and a `[te]` tag (vocabulary of 2521).
  - The acoustic stack is unchanged.
  - Zero-shot cloning works from a 6–15 s reference, and it handles code-switched Telugu–English.
- speech-swift already ships a Swift/MLX port of Chatterbox-Multilingual. So the job is converting the Telugu T3 weights and tokenizer into the bundle format that port expects and reusing its acoustic stack — not writing a new port.
  - If its loader hard-codes the vocabulary or language list, extend it through an adapter or a fork, and propose a clean upstream PR.
  - Conversion is a dev-time step. Prefer a Swift tool. A Python script is acceptable only as an offline maintainer script that lives outside the app, and only with my approval.
  - The output is an MLX bundle I publish on Hugging Face. The Model Manager downloads it like any other model.
- **Alternatives for S4:** OmniVoice (MLX, Apache-2.0, 600+ languages, zero-shot cloning) and Indic-Mio. Verify Telugu quality for both before trusting either.
- **Rate control:** prefer the engine's native speaking-rate control if quality holds. Otherwise use a high-quality time-stretch (AVAudioUnitTimePitch in offline manual rendering, or WSOLA). The default cap is 1.2×.
- **Post-processing:** trim leading and trailing silence, add short fades, and keep loudness consistent across units and speakers.
- The Python Chatterbox pipeline adds Resemble's PerTh watermark to every output. Check whether the Swift port does, and document the answer in `MODELS.md`.

### 6.6 Timing: fitting Telugu into the original timeline

This is the heart of the product. Build it as pure logic, and test it heavily.

Definitions for each unit `u`:

- `s_u`, `e_u`: when the original speech starts and ends, in video time.
- `gap_u`: the silence between `e_u` and the next unit's start.
- `borrow_u = clamp(gap_u − 0.15 s, 0, 0.8 s)`: how much of that silence the unit may use.
- `B_u = (e_u − s_u) + borrow_u`: the unit's time budget.
- `d_u`: how long the synthesized Telugu line lasts.

Apply these steps in order, and stop as soon as `d_u ≤ B_u`:

```
1. Length-targeted translation: target_aksharas = B_u × voice_rate × 0.95
2. Condense (only if overflow > 5%): up to 2 rewrites toward the new target,
   re-synthesize, keep the best
3. Speed up: rate = min(d_u / B_u, speedCap = 1.2)
4. Yield video time for the remaining overflow Δ:
     a. slow the video across the unit (rate ≥ 0.85)
     b. if that's not enough, freeze-frame at the unit's end (≤ 1.5 s per event)
   Budget: added video time ≤ 4 s per rolling minute.
   If the budget is used up, go back to step 2 with a harder target.
```

Invariants (property-test every one):

- Every dub unit is planned to start exactly at `s_u` in video time. Execution error is a metric (§10), and drift never accumulates.
- No speech is cut mid-word, and no two dub units overlap.
- Lines shorter than their slot are never slowed below 0.95×; pad them with silence instead.
- All timeline edits are planned into the DubTimeline ahead of the playhead, never improvised during playback.
- Every threshold above is a setting, with these values as defaults.

**DurationEstimator.** Start with an aksharas-per-second model for each voice, and count English words with a syllable heuristic. Refine it online after every synthesized unit, using a robust running regression.

Swift's grapheme clustering may already treat Telugu conjuncts as single clusters under the Unicode 15.1 rules. Verify that with tests before relying on it, and write explicit rules if it doesn't hold.

### 6.7 Playback and sync

- **Video:** AVPlayer in an AVPlayerView, with a scrubber that shows how much dub audio is ready. Load no audio track, and fully mute the muxed fallback stream.
- **Dub audio:** AVAudioEngine with scheduled buffers.
  - Compute each unit's host start time from its video time.
  - Re-anchor on every rate change, seek, stall, and play/pause.
  - Evaluate pointing AVPlayer's `sourceClock` at the audio device clock, so both run on one clock.
- **Timeline edits** — rate changes and freeze-frames — run at exact video times, coordinated with the audio schedule.
- **Buffering** uses hysteresis, and every value is tunable:
  - Start playback when the first two units are ready and the projected lead is at least 8 s.
  - If the lead drops below 2 s, pause at the next unit boundary and show "Preparing Telugu audio…".
  - Resume at a 12 s lead.
- **Seek:** re-prioritize work to `[seek, seek + 60 s]`, reuse cached units, and resume once the start conditions are met again.
- **User playback speed (0.75–1.5×):** change the video rate and time-stretch the dub audio to match.
- **Metrics:** record the planned vs. actual start of every unit. These feed the debug HUD and `maata-bench`.

### 6.8 Orchestration and resources

- Each video gets its own SessionController.
- **Scheduler:**
  - Prioritize work by how far ahead of the playhead it is.
  - Bound concurrency per stage.
  - Apply backpressure: by default, don't work more than 5 minutes ahead.
- **Cancellation:** a seek, or closing the video, promptly cancels work that's no longer needed.
- **Memory:**
  - Choose model presets based on RAM.
  - On memory-pressure events, shrink the look-ahead and unload idle models.
  - Target peak memory at or below 60% of physical RAM.
- **Warm models:** keep models loaded between videos, and preload them at launch or when a URL is pasted.
- **Failures:**
  - If one unit fails, play silence for it, show a small "line skipped" marker, log it, and keep going. Never un-mute the original audio.
  - If a stage keeps failing, show a clear error state with a Retry button.

## 7. Model Manager and first run

- **Manifest:** ships in the app, versioned, metadata only. Each entry has:
  - an id and role (asr, diarization, vad, embedding, translator, condenser, tts, restoration, tool);
  - the engine it binds to;
  - its source repo, with a pinned revision (commit hash);
  - its files, with SHA-256 and sizes;
  - its license id and URL, and whether acceptance is required;
  - whether it's gated;
  - its minimum RAM tier and the presets it belongs to.
- **Presets:** Standard (16 GB), Recommended (24 GB+), and High Quality (48 GB+, only if it earns its place). Recommend one based on detected RAM and free disk space.
- **Onboarding:**
  1. Hardware check.
  2. Choose a preset, with the total download size shown.
  3. A license screen for each model, with View License and Accept.
  4. Download through a background URLSession, with per-file progress and pause/resume that survives restarts.
  5. Verify each file's SHA-256, then move it into place atomically.
  6. Show "Optimizing models for your Mac…" while Core ML compiles on first load.
  7. Run a 10-second self-test that synthesizes one Telugu sentence.
- **Storage:** defaults to `~/Library/Application Support/Maata/Models`, and the location can be changed. Offer Reveal in Finder, per-model delete, re-verify, and a short "why this model" note for each.
- **Hugging Face:** an optional user token, stored in the Keychain, for gated repos. Never log it.
- **Tools:** if the yt-dlp resolver is chosen, pin and verify its binaries exactly like models. Handle quarantine and Gatekeeper for downloaded executables, and only check for updates when the user asks.

## 8. Storage and cache

- Each video gets a cache folder at `~/Library/Caches/Maata/<videoID>/`. It holds the transcript, translations, speaker registry, voice conditioning, rendered unit audio, and the timeline.
- Key every artifact by the model versions and settings it depends on. Changing the translator or the style then regenerates only what depends on it.
- The cache is capped at 5 GB by default, with LRU eviction.
- Voice data expires automatically after 14 days.
- Settings has a "Clear all" button.
- A video that's already cached starts instantly.

## 9. UI

- **Main window:**
  - A URL field that accepts paste and drag-and-drop.
  - A target-language picker, with Telugu as the only option for now.
  - The player, with the dub-ready ranges drawn on the scrubber.
  - A status chip, e.g. "Preparing Telugu audio… 6 s".
  - An "AI dub" badge that's always visible.
  - A speaker panel showing each detected speaker, whether it has a cloned or preset voice, and a per-speaker "use preset voice" toggle.
- **Settings:**
  - Models and Storage.
  - Timing: the speed-up cap, whether slow-downs and freeze-frames are allowed, and the maximum pause.
  - Translation style.
  - Voice: clone speakers (default) or always use preset voices.
  - Network: the update check.
  - Advanced: a debug HUD showing per-stage latency, lead, and drift, plus a debug transcript drawer.
- **Keyboard:** space to play or pause, the arrow keys to seek ±5 s, and ⌘L to focus the URL field.
- **Accessibility:** full VoiceOver labels and complete keyboard navigation.
- **Localization:** keep all strings in a String Catalog so a Telugu UI can follow later.

## 10. Targets

Measure these on the reference machine (M5 Pro, 24 GB) with the Recommended preset. Report Standard-preset numbers on 16 GB machines too, but don't block on them.

| Metric | Target |
|---|---|
| Paste → playback can start (models warm) | p50 ≤ 10 s, p95 ≤ 20 s |
| Paste → playback can start (models cold, already compiled) | ≤ 25 s |
| Sustained pipeline throughput | ≥ 1.5× real time |
| Dub-unit start error vs. plan | p95 ≤ 50 ms, max ≤ 120 ms (excluding seeks) |
| Video time added by slow-downs and freezes | ≤ 2 s per minute, averaged over the test set |
| Speed-up applied | ≤ 1.2× (default cap) |
| Peak memory | ≤ 60% of physical RAM |
| Voice quality | Report speaker similarity and Telugu round-trip CER per model; my listening scores break ties |

## 11. Testing

- **Unit tests:**
  - IsochronyFitter and DubTimeline, with property-based tests of the §6.6 invariants.
  - Scheduler priority and cancellation.
  - The akshara counter and DurationEstimator.
  - TextNormalizer (Telugu numbers, dates, currency).
  - Manifest parsing and checksum verification.
  - URL parsing.
  - Cache keys and eviction.
- **Integration tests:** run the pipeline on local fixture audio, with no network. Check segment boundaries, speaker counts, and timeline validity against golden files.
- **Automatic quality checks:**
  - Round-trip intelligibility: transcribe the Telugu output with a Telugu-capable ASR (for example Omnilingual ASR) and compute CER against the intended text.
  - Speaker similarity: cosine similarity between speaker embeddings of the reference and the dub.
- **`maata-bench` CLI:** runs the pipeline on a local file with a chosen preset and prints JSON: per-stage RTF, time to first audio, peak memory, and a drift histogram. Use it in Phase 0, then as a regression check.
- **End-to-end tests** (manual or opt-in, since they need the network): a `docs/TEST_VIDEOS.md` list that I'll curate. It should cover a single-speaker lecture, a two-person interview, a fast talker, a music-heavy vlog, a video over one hour, a Hindi-source video, and a Short.
- **Test media:** only self-recorded or CC0/CC-BY clips go in the repo. Never commit downloaded YouTube media.
- **CI:** GitHub Actions on a macOS runner builds the app and runs the unit tests. Tests that need models are tagged and skipped in CI.

## 12. Phases and acceptance criteria

**Phase 0 — Spikes (time-boxed; report and stop).** Each spike is a small throwaway target or CLI. Results go into `docs/SPIKES.md` — measurements, audio samples, and a recommendation per spike — with the resulting decisions as ADRs.

- **S1 — Streams:**
  - Compare YouTubeKit (local-only) with yt-dlp + Deno across the test list.
  - Play a 1080p video-only stream in AVPlayer, with seeking.
  - Get the first 60 s of audio decodable within 2 s of resolving, whatever the video's length.
  - Measure how stream URLs expire and refresh.
- **S2 — Front-end:** compare the diarization and ASR candidates on accuracy (DER and WER on fixtures I'll label), timestamp quality, RTF, and memory.
- **S3 — Translation:**
  - Compare TranslateGemma 4B and 12B, an instruct model for condensing, and MADLAD-400.
  - Use a set of about 50 English→Telugu sentences that I'll score.
  - Measure latency, memory, JSON reliability, and how well each model hits a length target.
- **S4 — TTS:**
  - Load chatterbox-telugu through speech-swift's Chatterbox port, converting the weights and tokenizer.
  - Compare it with OmniVoice and Indic-Mio on RTF, memory, round-trip CER, and cross-lingual speaker similarity (English reference → Telugu output).
  - Test Tenglish handling, and produce samples for me to listen to.
- **S5 — Sync:**
  - Prototype AVPlayer plus AVAudioEngine with synthetic units.
  - Measure drift across rate changes, freeze-frames, seeks, and stalls.
  - Compare results with and without `sourceClock`.

**Phase 1 — Walking skeleton.** A URL goes in and dubbed playback comes out. Use one preset voice, no diarization, start-aligned units, and speed-up only. Load models from a dev path.
- *Accept when:*
  - a 10-minute English single-speaker video plays end to end in Telugu;
  - seeking works;
  - dub-unit start error is at or below 80 ms at p95;
  - three videos in a row play with no crashes.

**Phase 2 — Model Manager and onboarding** (§7), including offline mode for every library.
- *Accept when:*
  - a fresh macOS user account gets from first launch to a dubbed video using only the in-app flow;
  - a download interrupted halfway resumes;
  - a corrupted file is detected and downloaded again;
  - with the network blocked after setup, everything except YouTube fetching still works, and YouTube fetching fails gracefully.

**Phase 3 — Speakers and cloning** (§6.2–6.3).
- *Accept when:*
  - a two-person interview gets two distinct cloned voices;
  - speaker counts are correct on my labeled fixtures;
  - switching from a fallback voice to a clone happens only at unit boundaries;
  - speaker similarity beats the preset-voice baseline by a margin we agree on in the Phase 3 plan.

**Phase 4 — Timing and translation quality** (§6.4, §6.6). Build the full cascade, timeline edits, the glossary and context, and online DurationEstimator calibration.
- *Accept when:*
  - every invariant holds under property tests;
  - the §10 targets for added video time and start error are met on the test set;
  - I rate the translations acceptable on the S3 set.

**Phase 5 — Robustness and performance.** Cover seeks, stalls, URL expiry, memory pressure, videos over an hour, cache reuse, cancellation, and error states.
- *Accept when:*
  - the §10 targets are met on the reference machine;
  - Standard-preset numbers are reported;
  - a 2-hour soak test runs with no leaks and no growth in drift.

**Phase 6 — Release.** Polish Settings and accessibility, and write the docs: README, CONTRIBUTING, LICENSE, MODELS.md, and PRIVACY.md. Add CI and a release workflow that produces a signed, notarized DMG; I'll provide the secrets. Sparkle updates are optional — ask first.
- *Accept when:*
  - the notarized DMG installs and runs on a second Mac;
  - the README quick start works exactly as written.

## 13. Responsible use

The app dubs what the speaker actually said, for private viewing, so v1 has:

- no export of dubbed media, cloned voices, or reference clips;
- no feature that speaks arbitrary text in a detected speaker's voice;
- voice data that stays local, is cleared along with the cache, and expires as described in §8;
- the "AI dub" badge always on screen, and a first-run explanation that the voices are AI-generated.

The README must say that Maata is not affiliated with YouTube or Google, that it's meant for personal viewing, and that users are responsible for following YouTube's terms and their local law. It should also encourage people to support creators on YouTube itself.

## 14. References

Checked in September 2026. Verify before relying on any of these.

- speech-swift (Swift, MLX and Core ML speech toolkit; Apache-2.0): https://github.com/soniqo/speech-swift
- chatterbox-telugu: https://huggingface.co/shankarpandala/chatterbox-telugu
- OmniVoice MLX bundle: https://huggingface.co/aufklarer/OmniVoice-MLX-fp16
- YouTubeKit: https://github.com/alexeichhorn/YouTubeKit
- yt-dlp's JavaScript-runtime requirement: https://github.com/yt-dlp/yt-dlp/issues/15012
- TranslateGemma model card and chat template: https://huggingface.co/google/translategemma-27b-it
- mlx-swift and mlx-swift-lm (ml-explore)
