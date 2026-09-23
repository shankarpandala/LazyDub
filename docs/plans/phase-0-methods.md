# Phase 0 measurement methods (cross-platform)

This is the companion to `phase-0.md`. It says how each number is produced. Everything here is a method, not a result. API names are marked *confirm at pin* where they must be checked against the pinned version.

## 1. Offline models and engine

- **Pinning.** `models.lock.json` has one entry per file: `repo`, `commit`, `path`, `size`, `sha256`, `license`. The Python environment is pinned the same way: a python-build-standalone archive plus a `uv.lock` with hashes, in separate CUDA and MPS variants.
- **Fetching.** `maata-bench fetch` downloads `https://huggingface.co/<repo>/resolve/<commit>/<path>`, verifies the sha256, and moves each file into place atomically.
- **Loading.** Every library loads from local paths only. The engine sets:
  - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
  - `HF_HOME`, `TORCH_HOME` and `XDG_CACHE_HOME` pointing into our storage;
  - pyannote, WhisperX and faster-whisper pointed at local model directories.
- **Enforcement.** S2–S4 bench runs execute with outbound network blocked. On Linux that's a network namespace (`unshare -n`). On macOS it's `sandbox-exec` denying network-outbound. On Windows, a firewall rule for the bench executable. Any connection attempt fails the run, and the result records `offline: true`.
- **yt-dlp.** Flags: `--no-remote-components --ignore-config --no-plugin-dirs --js-runtimes deno:<path> --cache-dir <ours>`, never `-U`. It runs with `DENO_NO_UPDATE_CHECK=1`.

## 2. Run hygiene

- **Recorded with every result:**
  - OS and version; CPU/GPU model; RAM and VRAM;
  - GPU driver and CUDA version (NVIDIA), or the macOS version (Apple);
  - Python, torch, llama.cpp and other library versions, taken from the lockfile;
  - the tool's git SHA;
  - power source;
  - GPU clocks and temperature at start and end (NVML on NVIDIA), or `thermalState` via a small helper on macOS.
- **Runs:** at least 5 per configuration; report p50 / p95 / max and n. Runs that hit thermal throttling are discarded.
- **Storage:** raw JSON goes in `docs/spikes/results/<spike>/<machine>/`.

## 3. Memory

- **NVIDIA:**
  - The VRAM peak comes from `torch.cuda.max_memory_allocated` / `max_memory_reserved` per stage, plus NVML's per-process used memory sampled every 50 ms. Both are needed because llama.cpp allocates outside torch.
  - The system-RAM peak is the process-tree peak RSS (`psutil`, children included).
- **Apple Silicon:**
  - The process-tree peak physical footprint (`proc_pid_rusage` `ri_lifetime_max_phys_footprint`, via a small helper, for each process).
  - `torch.mps.driver_allocated_memory()` and `mx.get_peak_memory()` where they apply (*confirm at pin*).
- **Configurations:** each one ("model alone", "pair", "co-run") runs in a fresh process.
- **Budgets:**
  - Apple: 60% of unified memory.
  - NVIDIA: VRAM tier, with 10% headroom; system RAM at 60%.

## 4. Load states

| State | Definition |
|---|---|
| First load | Right after install: the CUDA kernel cache / MPS shader cache is empty, and the torch.compile cache (if used) is empty |
| Cold | A new engine process with OS caches populated; each process counts as one run |
| Warm | Same process, second run onward |

Model load time and first-inference time are recorded separately.

## 5. S5 sync on the IFrame player

- **The player's clock:**
  - Sample `player.getCurrentTime()` at 60 Hz alongside `performance.now()`, and record the step size, jitter and update cadence.
  - Record `getAvailablePlaybackRates()`.
  - Record the latency from `pauseVideo()` / `playVideo()` / `seekTo()` to the resulting state change.
- **Mapping:**
  - Keep a linear fit of video time against `performance.now()`, re-anchored on `onStateChange`, `onPlaybackRateChange` and seeks.
  - Convert to `AudioContext.currentTime` via `getOutputTimestamp()`.
  - Schedule each unit with `AudioBufferSourceNode.start(when)`, subtracting `AudioContext.outputLatency` where the webview reports it (*check per webview*).
- **Actual onset:** an `AudioWorklet` on the output path records the context frame at which each unit's first non-zero sample renders, converted back to `performance.now()` time.
- **Ground truth:** a 240 fps phone recording of the screen (a flash patch in the test video) and the speaker. The analysis measures the flash edge against the chirp onset. It runs once per OS webview and output device, and calibrates the internal metric.
- **Scenarios:** 10 min steady, 20 seeks, pause/play, each available user rate, and throttled-network buffering. Each runs in WebView2, WKWebView and WebKitGTK.
- **Report:** audible start error p50 / p95 / max, and drift over time.

## 6. S1 details

- **Resolution:** success, latency, formats seen (`itag`, `clen`, `expire`), and whether a PO token was needed. Results cover out-of-scope classes (live, premiere, members-only, age-restricted, private, DRM) and in-scope edge cases (made-for-kids, ended live, Shorts, long videos).
- **Audio ingest:** fetch the init segment plus the fragments covering [0, 60 s] (using `sidx`), then decode (PyAV / ffmpeg libs, *pin*). Report the time to PCM, and the sample-0 offset against a full-file decode (must be ≤ 1 ms).
- **Timeline alignment:** check that the offset between the decoded audio and the IFrame player's `getCurrentTime()` timeline is constant, measured once per test video in the S5 harness.
- **Network audit:** a system-wide DNS/SNI capture. Only `youtube.com`, `*.googlevideo.com`, `*.ytimg.com` and `*.google.com` player assets may appear during playback.

## 7. S4 details

- **CER judge:** Omnilingual ASR, calibrated on held-out Telugu first; report its own error floor. For Tenglish lines, compute CER on the Telugu spans only, with Latin spans masked out; English terms are rated in the blind listening sheet.
- **Speaker similarity:** WeSpeaker (or ECAPA) cosine similarity for three cases: English reference → Telugu output; Telugu → Telugu as the upper bound; a preset voice as the baseline.
- **Rate control:** candidates are native `cfg_weight` / `exaggeration` first, then time-stretch (a librosa phase vocoder vs. WSOLA vs. Rubber Band, *licence check*). Acceptance at 1.2×, against the 1.0× render: CER rises by at most 1 point absolute, similarity drops by at most 0.02, and blind listening doesn't reject it. The cheapest method that passes wins.
- **Watermark:** check whether the PerTh watermark is detectable on output, and measure its latency cost.

## 8. Scoring

- **DER:** md-eval style, with a 0.25 s collar and with no collar, overlap included.
- **WER / CER:** after a shared, documented normalisation.
- **Word timing:** start and end error (median, p95, signed bias) on matched words, plus the dropped-word rate.
- **Length accuracy:** `|aksharas − target| / target`, with targets at 100 / 85 / 70 % of the unconstrained output, reported separately for pure Telugu and code-mixed lines.
