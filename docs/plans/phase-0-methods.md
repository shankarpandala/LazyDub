# Phase 0 measurement methods

This is the companion to `phase-0.md`. It says how each number is produced, so results are trustworthy and repeatable.

- API names are as documented. Any marked *confirm at pin* are checked against the pinned SDK or package before use.
- Everything here is a method. No results yet.

## 1. Offline models (spec §3.3)

**Pinning and fetching**

- `models.lock.json` has one entry per file: `repo`, `commit`, `path`, `size`, `sha256`, `license`. It uses the same shape as the §7 manifest.
- `maata-bench fetch` downloads `https://huggingface.co/<repo>/resolve/<commit>/<path>` with URLSession, checks the sha256, and moves the file into `~/MaataDev/Models/` atomically.
- Tools (yt-dlp, Deno) are pinned the same way. The yt-dlp `SHA2-256SUMS.sig` is checked once, when the version is pinned, on the maintainer's machine. The verified sha256 goes into the ADR. Spikes and the app only compare sha256; they never run GnuPG.

**Loading from local paths only**

| Engine | How it loads |
|---|---|
| Chatterbox | `fromPretrained(localDir:s3TokenizerWeights:conformerWeights:)`, which never touches the network |
| Community-1, MADLAD | `fromLocal(directory:)` |
| Whisper, Sortformer, Silero, WeSpeaker, Omnilingual, Qwen3-ASR / aligner | `cacheDir:` pointing at our folder, plus `offlineMode: true`. Qwen3 folder names must keep their `0.6B` / `4bit` markers, because the architecture is read from the model id. |
| mlx-swift-lm models | the local-directory `ModelConfiguration` (*confirm at pin*) |
| WhisperKit (if approved) | `modelFolder` / `tokenizerFolder`, with downloading disabled |
| Indic-Mio | `fromBundle` for the main model, plus `INDIC_MIO_WAVLM_BUNDLE` set to a pinned local WavLM bundle. Without it, voice cloning downloads WavLM from `main`. |
| Gemma 4 / Qwen3 chat (speech-swift `Qwen3Chat`) | `fromDirectory(_:)` (*confirm at pin*) |
| OmniVoice | `fromBundle` (its `fromPretrained` has no `cacheDir`) |
| SpeechLanguageID, Nemotron (if D4 allows) | their `cacheDir` + `offlineMode: true` loaders (*confirm at pin*) |

`fetch` writes each model into exactly the directory its loader receives. Loaders that take a `cacheDir` get the exact upstream model id, because some of them derive paths and variants from it.

**Enforcement**

- Every S2–S4 bench run executes under `sandbox-exec` with outbound network denied.
- A blocked connection doesn't always surface: speech-swift's online path retries, then falls back to its cache. So after each run, the unified log is scanned for Sandbox `deny(1) network-outbound` entries from the bench PID, and any hit fails the run.
- The result JSON records `offline: true`.

## 2. Run hygiene (every result)

- **Recorded fields:**
  - machine model and RAM; `sw_vers`; Xcode build;
  - git SHA of the tool;
  - `ProcessInfo.thermalState` at start and end;
  - `isLowPowerModeEnabled`; `pmset -g` power mode;
  - on AC power or not;
  - the QoS class the stage ran at;
  - the `Memory.cacheLimit` setting.
- **Execution:**
  - Measured work runs at a fixed QoS matching the planned scheduler (`.userInitiated` for look-ahead work).
  - S5 runs inside `ProcessInfo.beginActivity([.userInitiated, .latencyCritical, .idleSystemSleepDisabled])`.
  - Runs that reach `.serious` thermal state are thrown out.
- **Reporting:** at least 5 runs per configuration; p50 / p95 / max, plus n. Raw JSON is committed under `docs/spikes/results/<spike>/`.

## 3. Memory

- **Per-process peak.** Recorded as `proc_pid_rusage(getpid(), RUSAGE_INFO_V6)`, using `ri_lifetime_max_phys_footprint`.
  - If the SDK provides them, also `ri_neural_footprint` and `ri_lifetime_max_neural_footprint`.
  - Every "model alone", "pair" and "co-run" configuration runs in its own new process.
  - A 50 ms `phys_footprint` sampler runs for timeline plots only.
- **System-level used memory.** Taken from `host_statistics64(HOST_VM_INFO64)`: (internal − purgeable + wired + compressor pages) × page size, minus a baseline taken before the run.
  - This catches Core ML / Neural Engine memory charged to `aned` or to IOSurfaces instead of our process.
  - Cross-checked once per Core ML model with `sudo footprint -p aned`.
- **MLX.**
  - Set `Memory.cacheLimit` explicitly.
  - At every stage boundary, call `Memory.clearCache()` and `Memory.resetPeakMemory()`.
  - Report `Memory.peakMemory`, `Memory.activeMemory` and `Memory.cacheMemory` (*confirm names at mlx-swift 0.31.4*).
- **Budgets.** 60% of physical RAM: 14.4 GB at 24 GB, 9.6 GB at 16 GB.

## 4. Load states

| State | Definition | How often |
|---|---|---|
| First load | The first load after the model is installed at its final path. Core ML's per-device Neural Engine specialisation is still pending, and Metal's shader cache is empty. | Once per model, and again after any OS update |
| Cold | A new process, with operating-system caches already populated. Each process counts as one run. | ≥ 5 processes |
| Warm | The same process, from the second run on | ≥ 5 runs |

- Model load time and first-prediction time are recorded separately.
- **Moving a compiled model.** Check whether moving an `.mlmodelc` re-triggers specialisation. If it does, an ADR records that the Model Manager's "Optimizing models…" step runs after its atomic move.

## 5. Core ML placement and contention

For each Core ML model, the result records:

- the `computeUnits` actually passed;
- an `MLComputePlan` summary: the operation count on the Neural Engine, GPU and CPU, and the share weighted by `estimatedCost(of:)`.

`.cpuAndNeuralEngine` is tested explicitly. Instruments' Core ML template confirms runtime dispatch once per model.

**Contention run (`maata-bench corun`).** The Core ML front-end processes a 60 s window while MLX Chatterbox synthesises. Both RTFs are reported next to their solo values.

## 6. S1 details

**Audio ingest**

- **Fetching:**
  - One request covers bytes 0 through `indexRange.end` (the init segment plus `sidx`).
  - Then fetch exactly the fragments covering [0, 60 s], using the `sidx`.
  - Record fragment durations and sizes.
- **For each candidate** (partial file + AVAssetReader, AVFragmentedAsset, own fMP4 reader + AudioConverter with the `esds` magic cookie), report:
  1. time to 60 s of PCM;
  2. the offset of sample 0 against a reference decode of the whole file, by cross-correlation, in samples;
  3. the seam error between consecutive windows;
  4. steady-state cost per window.
- **Alignment check.** Once per video, the itag-140 decode is cross-correlated with the audio of muxed itag 18 decoded by AVAssetReader, which applies edit lists. The offset must be ≤ 1 ms. Only the offsets are committed, never media.

**Video**

- Target: 1080p avc1 video-only, with ≥ 20 random seeks, trying the strategies in §6.1's order.
- `automaticallyWaitsToMinimizeStalling = false`, the same setting S5 needs.
- **Direct URL:** `AVURLAssetHTTPUserAgentKey` is set to the resolving client's User-Agent, and each 403 is logged with its cause.
- **Resource-loader cache:** uses a custom scheme (`maata-media://`), fills `contentInformationRequest` (content type, `clen`, byte-range support), and serves ranges from our URLSession.
- **HLS:** for each client, record whether `hlsManifestUrl` appears on non-live videos.

**Out-of-scope content** (live, upcoming premiere, members-only, age-restricted, private, DRM). For each class, record the error each resolver returns and whether it maps to a distinct §6.1 message.

**In-scope edge cases** (made-for-kids, ended live, 4K, Shorts). Failures here are defects, reported under Risk 1.

**yt-dlp** (benchmarked in S1 as §12 requires; whether to ship it is decided after S1)

- **Environment:** `DENO_NO_UPDATE_CHECK=1`, `DENO_DIR` and `TMPDIR` inside our container.
- **Flags:** `--no-remote-components --ignore-config --no-plugin-dirs --no-js-runtimes --js-runtimes deno:<path> --cache-dir <ours> -J`, never `-U`.
- **Launch test:** both binaries are started from a signed, Hardened-Runtime host app, twice:
  - as downloaded by URLSession (no quarantine xattr);
  - with `com.apple.quarantine` set.
  - Record what runs and which entitlements were needed. None of this has been verified on macOS yet.

**Network audit**

Pass criterion: only `youtube.com`, `www.youtube.com`, `*.googlevideo.com` and `i.ytimg.com` (thumbnails) may appear; any other host fails S1.

- `nettop -p <app pid>` is meant to cover AVFoundation's own requests, which a URLProtocol can't see. Check that googlevideo connections really do show up under the app's PID.
- A system-wide `tcpdump` capture of DNS and TLS SNI runs as a backstop for the whole session.
- A URLProtocol log covers our own sessions.
- `nettop` also covers the subprocesses.
- Deno's upgrade host must never appear.

## 7. S4 details

**Telugu bundle licence files**

- **CC-BY-4.0** attribution for the Telugu T3 and tokenizer: author, FLEURS, IndicVoices-R. Plus a statement of changes:
  - key rename `tfmr.` → `tfmr.model.` with the `t3.` prefix;
  - the fp16 cast, if applied;
  - the tokenizer renamed to `tokenizer.json`, with its merges rewritten.
- **MIT ©** Resemble AI for VE, S3Gen, the conformer and `Cangjie5_TC.json`.
- **Apache-2.0** licence text and notice for `s3_tokenizer.safetensors` (CosyVoice2 S3TokenizerV2).

**Rate control: candidates and rule**

- Native controls first (§6.5):
  - a Chatterbox `cfgWeight` × `exaggeration` sweep, measuring the change in duration and in CER;
  - OmniVoice `duration:`, if it's in the D6 comparison.
- Then time-stretching: TimePitch and WSOLA.
- **Acceptance (D7):** at 1.2× against the 1.0× render, CER rises by ≤ 1 point absolute, WeSpeaker similarity drops by ≤ 0.02, and the blind listening sheet doesn't reject it. The cheapest method that passes wins.
- The 1.1× and 1.2× renders go into the blind listening sheet.

**Offline rate change** with `AVAudioUnitTimePitch` in manual rendering mode (`enableManualRenderingMode(.offline, …)`)

- Append `latency + tailTime` of silence to the input.
- Render `input / rate + latency` frames and drop the first `latency`.
- Assert:
  - the output length equals input / rate, ± 1 ms;
  - the onset shift at 1.0× is ≤ 1 ms, by cross-correlation.
- Sweep overlap over {8, 32}, with peak locking on.
- WSOLA gets the same length and onset checks.

**CER judge**

- Omnilingual ASR is the primary judge (spec §11). Nemotron te-IN is the second judge, if D4 allows it.
- Calibrate on **held-out** Telugu (D5), meaning speakers and sentences chatterbox-telugu didn't train on, before trusting it. Report its own CER floor. The Telugu→Telugu similarity upper bound and the S4 test lines use held-out material too.
- For Tenglish lines, compute CER on Telugu spans only, with Latin spans masked in both reference and hypothesis.
- English terms are rated in the blind listening sheet.

**Speaker similarity** is WeSpeaker cosine (a single embedding backend, since the MLX and Core ML embeddings don't mix) for:

- English reference → Telugu output;
- Telugu → Telugu, as the upper bound;
- a preset voice, as the baseline.

## 7b. S3 details

**TranslateGemma loader fixes**

- Applied in our adapter at load time, with no mlx-swift-lm fork, and the pinned files are left unmodified.
- The adapter builds an overlay directory: a generated `config.json` (the pinned one plus `rope_scaling: {type: linear, factor: 8}` in `text_config`), with the pinned weights and tokenizer symlinked. If mlx-swift-lm accepts an in-memory config override at the pin, that's used instead.
- `<end_of_turn>` is added through `extraEOSTokens`.

**Length targets.** For each sentence, targets are 100 / 85 / 70 % of the akshara count of that model's own unconstrained translation. Lines with real timings also get B_u × a provisional rate, replaced by S4's measured aksharas/s.

## 8. S5 details

**Latency compensation**

- **Total output latency** = (`kAudioDevicePropertyLatency` + `kAudioStreamPropertyLatency` of the active output stream + `kAudioDevicePropertySafetyOffset`) / sample rate.
  - These are read for the device from `kAudioOutputUnitProperty_CurrentDevice` on `engine.outputNode.audioUnit`.
  - `AVAudioOutputNode.presentationLatency` is logged alongside.
  - Everything is re-read on `AVAudioEngineConfigurationChange`.
  - The ground-truth rig decides which formula matches each device.
- **Video latency** per display comes from the rig.
- **The scheduler subtracts both** when it converts a unit's start into a host time.

**Planned time**

- The DubTimeline's piecewise-linear mapping from video time to host time, re-anchored to the live timebase at each edit that actually executes.
- `CMSyncConvertTime(item.timebase → host clock)` is only a cross-check, because it's wrong across edits that are still pending.

**Actual time**

- Taken from rendered samples: an `AVAudioSourceNode`, or a post-render notify on the output unit, records each unit's first-sample `mHostTime` and frame offset into a lock-free ring.
- It is never derived from the requested `AVAudioTime`, which would be circular.
- `error = audible host time − (planned host time of s_u + video latency)`.

**Re-anchoring** listens for:

- `kCMTimebaseNotification_EffectiveRateChanged` and `kCMTimebaseNotification_TimeJumped` on `playerItem.timebase`;
- `AVPlayerItem.playbackStalledNotification`;
- `AVPlayer.rateDidChangeNotification`, including its reason key.

On each event, the affected player nodes (one per unit, pooled) are stopped and flushed, and then rescheduled. For stalls, record the time from detection to re-anchor, and the milliseconds of dub audio that play past a video stall.

**Edits**

- Start and seek: seek at rate 0, wait for `preroll(atRate:)` to report true, then call `setRate(_:time:atHostTime:)` with a *future* host time.
- Slow-downs and freezes:
  - Run from a `.strict` `DispatchSourceTimer` at the edit's host time, calling `setRate(newRate, time: T_edit, atHostTime: ≈ now)`.
  - Log the timebase time before and after, the discontinuity, and how late the timer fired.
  - Preroll during freezes, and measure resume latency.
- `AVPlayerView.controlsStyle = .none`, so every rate change goes through the SyncEngine.

**Test media and dub units.** The test video is 60 fps with no audio track, and carries a binary frame code and a flash patch. The dub units are synthetic chirp bursts scheduled on AVAudioEngine at planned `s_u` times that fall on flash frames. The video's own audio is never played (§3.4).

**User speed.** Two approaches are compared.

- **(a) Offline re-render:** the real-time graph stays at 1.0×. Units that haven't played yet are re-rendered with the §7 method and rescheduled for the new video rate. The unit already playing is re-rendered from its current word boundary, or cut there. It is never left running at 1.0× past its slot, which would overlap the next unit.
- **(b) Live TimePitch in the graph:** scheduled in node sample time through the rate.
- **Measured for both:** time until the dub follows the new rate, and the start error of the unit that was playing.

**`sourceClock`.** `CMAudioDeviceClockCreate` for the engine's current output device UID, rebuilt when the route changes.

**Ground truth** (the S5 capture session in the plan's inputs table)

- **Preferred:** a photodiode on the flash patch and a mic at the transducer, both recorded into one USB audio interface. The wired output is taken electrically; for AirPods, the mic is coupled to the earbud.
- **Cheaper fallback:** a calibrated 240 fps iPhone slow-motion recording, synced with a clapper.
- The rig runs once per output device and display, for start, one seek, one rate change and one freeze. It calibrates the latency constants and validates the internal metric.
- ScreenCaptureKit is used only as an optional check before the output device, never as ground truth: it can't see device or display latency.

## 9. Scoring definitions

- **DER:** md-eval-style, reported both with a 0.25 s collar and with no collar, overlap included. Implemented in `Diagnostics` with unit tests.
- **WER / CER:** Levenshtein distance after a documented normalisation (case, punctuation, numerals) that's shared by every engine.
- **Word timing:** matched words only, plus a dropped-word rate: aligned words ÷ reference words.
