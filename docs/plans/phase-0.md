# Phase 0 plan: spikes (cross-platform)

Status: **proposed; waiting for your go-ahead** (spec §0). No code has been written.

This plan replaces the earlier native-macOS plan, following `docs/SPEC-AMENDMENT-01.md`: a Tauri desktop app with a web UI and a local inference engine, on-device only, on NVIDIA and Apple Silicon, with YouTube's embedded player.

- Methods: `phase-0-methods.md`.
- Research still valid from the macOS survey: models, licences, TranslateGemma facts and YouTube client behaviour (`docs/research/`). The speech-swift / Swift / Xcode findings no longer apply.

---

## To approve

**Schedule.** 18 working days at most, counted from go-ahead. Each box is a cap: when time runs out, the spike reports what it has and drops its *cut first* items.

| Days | Work |
|---|---|
| 1–2 | Groundwork and dependency re-survey |
| 3–5 | S5 Sync on the IFrame player (the riskiest spike, so it goes first) |
| 6–7 | S1 Streams and embedding |
| 8–10 | S4 TTS |
| 11–13 | S2 Front-end |
| 14–16 | S3 Translation |
| 17–18 | S6 Packaging, the §10 composition, and the report |

**Decisions needed before I start**

| # | Decision | Recommendation |
|---|---|---|
| D1 | **Engine language and runtime.** | **Python 3.12 + PyTorch** in a sidecar process. chatterbox-telugu, pyannote, faster-whisper / WhisperX and yt-dlp are all Python-native. Your own patched Chatterbox pipeline runs unchanged, PerTh watermark included. The alternative is a Rust/C++ engine (ONNX Runtime, whisper.cpp, llama.cpp), which is leaner but means porting Chatterbox. |
| D2 | **How the engine reaches users** (it's large: PyTorch + CUDA is multi-GB). | Ship a small installer. On first run, the Model Manager downloads a **pinned, checksummed Python environment**: python-build-standalone plus a `uv` lockfile, CUDA or MPS variant. This follows the §7 rules. The alternative is a PyInstaller-frozen engine in each installer, which gives huge per-OS artifacts and slow CI. |
| D3 | **Apple Silicon backend.** | PyTorch **MPS** for Chatterbox and pyannote. **MLX** (`mlx-whisper`, `mlx-lm`) where it's faster: ASR and the LLM. S2–S4 decide per stage. |
| D4 | **The LLM runtime for translation.** | **llama.cpp** (GGUF, CUDA and Metal) through `llama-cpp-python`: one runtime on both platforms. `mlx-lm` is an Apple alternative, and `transformers` the fallback. |
| D5 | **UI stack in the webview.** | **TypeScript + Svelte + Vite**: a small runtime with no heavy framework. Sync uses the Web Audio API against the IFrame clock. The alternative is React. |
| D6 | **The NVIDIA reference machine.** Spec §10 needs real numbers from one. | Name the machine: GPU model and VRAM, OS, driver. Without it, only Apple numbers are real, and CUDA gets build and smoke checks only. |
| D7 | **Sync targets on the IFrame player.** It exposes a coarse clock (`getCurrentTime()` over postMessage), a fixed set of playback rates, and pause-only freezes. The §10 start-error target (p95 ≤ 50 ms) and §6.6's 0.85× slow-down may not be reachable. | Measure first in S5. If the targets can't be met, I'll bring you the real numbers and options (for example, rates restricted to what the player allows, or a local-player fallback) before changing any target. |
| D8 | **Training-data overlap.** chatterbox-telugu was trained on FLEURS and IndicVoices-R. | Tell me which splits and speakers were used; S4 evaluates only on held-out speakers and sentences. |
| D9 | **NC engines** (OmniVoice, Indic-Mio). | A capped comparison of at most half a day. If one clearly wins, it becomes an opt-in download with its own licence acceptance. |
| D10 | **Linux scope.** | **Ubuntu 22.04 / 24.04 x64** with NVIDIA driver ≥ 550 as the supported target. AppImage + deb. Other distros are best-effort. |

**Inputs I need from you**

| Input | Amount | Needed by | If late |
|---|---|---|---|
| Answers to D1–D10 | — | day 1 | I proceed with the recommendations |
| Access to the NVIDIA machine (D6): run the bench there, or a Claude Code session on it | a few hours in total | day 8 | CUDA numbers marked *not measured* |
| Approve my draft `TEST_VIDEOS.md` | ~15 URLs | day 2 | My draft |
| S5 capture session: a 240 fps phone recording of screen + speaker, on the Mac and the NVIDIA PC | ~1 h each | day 5 | Internal metric only, marked *unvalidated* |
| Held-out Telugu (D8), or self-recorded speech | ~10 min | day 8 | FLEURS test split, flagged |
| S4 blind listening | ~110 clips, ~1.5 h | day 11 | Automatic metrics only |
| S2 fixtures: WAV + RTTM + transcript | 3 English clips of ~5 min + 1 Hindi | day 11 | AMI / LibriSpeech |
| S3 sentence edits (day 4) and scoring | 50 lines; ~240 ratings, ~2 h | day 16 | My draft set; automatic metrics |

**Defaults I'll use unless you object:** product name **Maata**; public CC-BY datasets (AMI, LibriSpeech, LibriTTS-R, FLEURS) with attributed excerpts; TranslateGemma from ungated mirrors for the spikes; Omnilingual ASR as the Telugu CER judge.

---

## Architecture being tested

```
Tauri shell (Rust)            Webview (TypeScript)                   Engine sidecar (Python, localhost)
- window, installer, updates  - URL field, UI, status                 - yt-dlp: resolve + fetch the audio stream
- spawns and supervises       - YouTube IFrame player (muted)         - VAD / diarization / ASR / alignment
  the engine                  - SyncEngine: Web Audio scheduled       - translation (llama.cpp)
- Model Manager: pinned         against the IFrame clock              - TTS (chatterbox-telugu)
  downloads + sha256          - dub-ready ranges, speaker panel       - DubTimeline planning
- per-video cache, storage                                             streams dub units + timeline
                                                                       over a local WebSocket
```

- **Engine interface:**
  - A local WebSocket on 127.0.0.1, with a random per-launch token passed by the shell. No other interface.
  - The engine never makes network calls except yt-dlp fetching YouTube; models load only from local paths, with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **The same Python engine** also runs headless as `maata-bench` for measurements.

## Spikes

### S5 Sync on the IFrame player (days 3–5) — riskiest, so it goes first

- **Build:** a Tauri window running the IFrame API with a muted test video (our own upload, or an unlisted CC0 test video with a frame code and flash patch), and dub units as Web Audio chirps. The SyncEngine maps video time to `AudioContext` time by extrapolating `getCurrentTime()` samples against `performance.now()`, and re-anchors on state changes, rate changes and seeks.
- **Measure:**
  - `getCurrentTime()` resolution and jitter.
  - Audible start error (p50 / p95 / max) and drift, for steady playback, seeks, pause/play, user rates and buffering stalls, in each OS webview (WebView2, WKWebView, WebKitGTK).
  - Which playback rates are available, and how fast pause and resume take effect: this decides §6.6's step 4.
  - Ground truth from the capture session.
- **Must deliver:** the Mac and Windows numbers, and the achievable edit mechanics.
- **Decides:** the sync design, and whether D7's targets hold or need a proposal.

### S1 Streams and embedding (days 6–7)

- **Check:** whether the IFrame player works inside Tauri's webview on each OS. Embeds need a valid origin, so a custom-scheme origin may be refused; the fallback is serving the UI from a loopback HTTP origin.
- **Measure:**
  - yt-dlp + Deno (pinned; `--no-remote-components`) resolution success and latency across `TEST_VIDEOS.md`.
  - A distinct error for each out-of-scope case.
  - Time to the first 60 s of decoded audio (≤ 2 s), and its alignment to the player's timeline.
  - URL expiry and refresh.
  - Network audit: only YouTube and Google hosts.
- **Decides:** the embedding origin setup, and the audio ingest method.

### S4 TTS (days 8–10)

- **Build:** chatterbox-telugu in the engine, using your patched pipeline, on CUDA and MPS, plus the capped D9 comparison.
- **Measure:**
  - RTF and time to first audio for ~2 / 5 / 10 s lines.
  - VRAM / unified-memory peak, with fp16 and bf16 where supported.
  - Round-trip CER on held-out Telugu.
  - Speaker similarity, English reference → Telugu output.
  - Tenglish handling.
  - Rate control: native `cfg_weight` / `exaggeration` vs. time-stretch, with the acceptance rule in the methods doc.
  - Whether the PerTh watermark is present, and what it costs.
- **Decides:** the TTS configuration per backend, and the rate-control method.

### S2 Front-end (days 11–13)

- **Candidates:**
  - Diarization: pyannote community-1, vs. NVIDIA Sortformer on CUDA if its licence is OK.
  - ASR and word timing: faster-whisper (CUDA) and mlx-whisper (Apple) with word timestamps, vs. WhisperX forced alignment (which covers Hindi).
  - Language ID: Whisper's own detection.
- **Measure:** DER, speaker count, WER, word start/end error and bias, pause detection, dropped words, RTF, and memory, per backend.
- **Decides:** the front-end stack for each backend.

### S3 Translation (days 14–16)

- **Candidates:**
  - TranslateGemma 4B and 12B as GGUF via llama.cpp (with correct RoPE scaling and stop tokens verified);
  - an instruct model (Gemma 4 / Qwen3-4B) for JSON and condensing;
  - MADLAD-400 3B as the baseline.
- **Measure:** latency, memory, JSON validity, length accuracy (with a Telugu + English-syllable akshara counter), condense hit rate, and your scores.
- **Decides:** one model or a pair, and the quantisation per memory tier.

### S6 Packaging (days 17–18, with the report)

- Build installers for all three OSes in CI.
- Test the first-run engine download (D2): size, time, resume, checksum failure.
- Code signing: a notarized DMG on macOS; a signed installer on Windows (a cert is needed later); an AppImage on Linux.
- Check that the sidecar starts under macOS hardened runtime and Windows Defender / SmartScreen.

## Checking §10

The report has a "Phase 0 vs §10" table **per reference machine**. Every row is labelled *composed, not end-to-end*.

- **Paste → start:** warm and cold.
- **Throughput:** ≥ 1.5× real time.
- **Peak memory:** from one co-run of all stages. On Apple, against 60% of unified memory. On NVIDIA, against VRAM and system RAM separately.
- **Start error:** from S5.

Memory presets are now tiered by **VRAM** on NVIDIA (proposal: 8 / 12 / 24 GB) and by unified memory on Apple (16 / 24 GB). S2–S4 set the actual numbers.

## Risks

1. **Sync precision on the IFrame clock** may miss the §10 start-error target, and §6.6's fine-grained slow-downs may be impossible. S5 goes first to find out.
2. **YouTube embedding inside a desktop webview** may be refused (origin/referrer errors). Mitigation: a loopback HTTP origin.
3. **Engine size and first-run download:** CUDA PyTorch runs to several GB. Mitigation: pinned downloads that can resume.
4. **Three OSes × two backends** multiply testing, and CUDA drivers vary. Mitigation: a single supported Linux target (D10) and minimum driver versions.
5. **Resolution still depends on YouTube's client quirks:** yt-dlp is the only resolver now. Its version pins must be updatable through the manifest without an app release.
6. **Licences:** the licence findings from the earlier survey still apply (OmniVoice NC, Sortformer OML, Gemma ToU, WeSpeaker CC-BY).

## Deliverables

- `docs/SPIKES.md`, where every number is backed by committed JSON.
- ADRs for every pin and default.
- The §10 tables per machine.
- Listening samples.
- CI building installers for the three OSes.
- A `MODELS.md` stub.

Then I stop.
