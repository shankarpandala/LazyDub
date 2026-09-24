# Architecture decision records

The maintainer delegated decisions on 2026-09-23 with one overriding goal: **run at best performance on the MacBook Pro M5 Pro (24 GB), with a very good-looking interface.** Every ADR below is judged against that goal first. NVIDIA/CUDA support stays in the design, but it is secondary.

Each ADR records the decision, why it was made, and what would reopen it. Versions are pinned in the lockfiles (`engine/uv.lock`, `app/package-lock.json`, `app/src-tauri/Cargo.lock`) once they are generated on the reference Mac.

---

## ADR-001 — Hybrid desktop app: Tauri 2 shell + Svelte UI + Python engine
- **Decision:** a Tauri 2 shell (Rust) hosts a webview UI written in TypeScript + Svelte 5 + Vite. A local Python inference engine runs as a sidecar.
- **Why:**
  - Tauri uses the system WebKit on macOS, so the shell costs little memory (~30 MB, against ~150 MB+ for Electron). That leaves RAM for models on a 24 GB machine.
  - Svelte compiles to small, fast DOM code and makes a custom, polished design easy without a heavy UI kit.
- **Reopen if:** YouTube embedding fails in WKWebView in a way the loopback origin (ADR-006) can't fix.

## ADR-002 — Engine in Python, Apple Silicon first
- **Decision:** the engine is Python 3.11–3.12. There are two backends selected at runtime:
  - **`apple`:** MLX + PyTorch MPS. This is the primary, tuned backend.
  - **`cuda`:** PyTorch CUDA + CTranslate2 + llama.cpp. Secondary.
- **Why:** chatterbox-telugu, pyannote, Whisper toolchains and yt-dlp are all Python-native. MLX gives the best Apple Silicon throughput for Whisper and LLMs.
- **Reopen if:** first-run install size or startup time misses §10 on the M5 Pro.

## ADR-003 — Per-stage engines on the M5 Pro

| Stage | Apple (primary) | CUDA (secondary) | Why for the M5 Pro |
|---|---|---|---|
| ASR + word timestamps | `mlx-whisper`, large-v3-turbo, `word_timestamps=True` | `faster-whisper` | MLX is the fastest Whisper on Apple GPUs. DTW word timing covers every Whisper language, Hindi included. |
| VAD | Silero (torch CPU) | same | Tiny; the CPU is fine and leaves the GPU free |
| Diarization | pyannote community-1 on MPS | pyannote on CUDA | Best open DER; CC-BY-4.0 |
| Translation | `mlx-lm` + TranslateGemma 4B 4-bit | `llama-cpp-python`, GGUF | ~2.2 GB. The 12B model breaks the 24 GB × 60 % budget next to TTS. |
| Condense / JSON | `mlx-lm` + Qwen3-4B-Instruct 4-bit, loaded on demand | llama.cpp | Only needed when a line overflows |
| TTS | chatterbox-telugu, PyTorch **MPS**, fp16 | PyTorch CUDA | The maintainer's own pipeline, unchanged, PerTh watermark included |

- **Memory plan:** the steady-state resident set is Whisper + diarization + TranslateGemma + Chatterbox, about 8–9 GB by weight sizes. The condenser loads lazily. Measured in Phase 0 (S2–S4).
- **Reopen if:** a measurement shows a stage misses its RTF share of the 1.5× real-time budget.

## ADR-004 — The GPU is serialised, the front-end is pipelined
- **Decision:** one asyncio priority queue owns the GPU (MLX/MPS work: ASR, translation, TTS), with priority set by distance to the playhead. The CPU-side work (fetching, decoding, VAD) runs in parallel threads.
- **Why:** Apple unified memory and a single GPU; running MLX and MPS concurrently causes contention and memory spikes (spec §5).

## ADR-005 — Timing core is pure Python with property tests
- **Decision:** the §6.6 isochrony cascade, DubTimeline, DurationEstimator, AksharaCounter and Telugu TextNormalizer live in `maata_engine.timing` / `maata_engine.text`. They have no ML imports and are tested with `hypothesis`.
- **Why:** the spec calls this the heart of the product. Pure code runs in CI on any OS.

## ADR-006 — YouTube IFrame player served from a loopback HTTP origin
- **Decision:** the engine serves the built UI at `http://127.0.0.1:<port>/`, and the Tauri window loads that URL. The UI embeds `https://www.youtube-nocookie.com/embed/…` through the IFrame API, muted.
- **Why:** YouTube's player refuses embeds without a valid HTTP origin/referrer, and a custom `tauri://` scheme risks error 152/153. The nocookie domain is the privacy-friendlier embed.
- **Reopen if:** S1 shows `tauri://` works everywhere, which would drop one moving part.

## ADR-007 — Engine ⇄ UI protocol
- **Decision:** a single WebSocket at `/ws?token=…`.
  - JSON messages carry control and events.
  - Dub audio is sent as binary frames: a 16-byte header (unit id u32, sample rate u32, sample count u32, reserved) followed by float32 PCM.
  - The token is random per launch, created by the shell and passed to both sides.
- **Why:** one connection and no per-unit HTTP requests. PCM decodes straight into `AudioBuffer`s.

## ADR-008 — Sync: Web Audio scheduled against an extrapolated player clock
- **Decision:**
  - `VideoClock` fits video time against `performance.now()` from `getCurrentTime()` samples, re-anchoring on state, rate and seek events.
  - The `SyncEngine` schedules each unit with `AudioBufferSourceNode.start(when)` at the context time mapped from its `s_u`, corrected by `outputLatency`.
  - Freezes are `pauseVideo()`. Slow-downs use the nearest available IFrame rate at or above 0.85×; if none fits, the cascade falls back to condense.
- **Why:** these are the only clocks available in a webview. S5 measures how accurate they are.

## ADR-009 — Visual design
- **Decision:** a dark, cinematic theme by default, with a light theme too.
  - A warm turmeric-to-vermilion accent gradient, taken from the Telugu cultural palette.
  - Glass panels over a subtly animated backdrop.
  - Inter for Latin text and Noto Sans Telugu for Telugu, both self-hosted. No network fonts, because the app runs offline.
  - The "AI dub" badge is always visible.
  - Motion respects `prefers-reduced-motion`.
- **Why:** it's the maintainer's explicit goal, and the design matches the app's content (video first, chrome recedes).

## ADR-010 — Stream resolution: yt-dlp inside the engine
- **Decision:** yt-dlp runs as a Python library in the engine, pinned, with `remote_components` disabled and no self-update. Deno is its pinned JS runtime.
- **Why:** yt-dlp is the most actively maintained resolver. YouTubeKit was Swift-only and no longer fits.

## ADR-011 — Engine distribution
- **Decision:**
  - The installer carries the shell and UI only.
  - On first run, the Model Manager fetches a pinned python-build-standalone runtime plus the `uv.lock`-pinned wheels for the detected backend, then the models. Everything is verified by sha256.
  - In development, `uv run` from `engine/` is used directly.
- **Why:** it keeps installers small and follows spec §3.3 / §7.

## ADR-012 — Linux target
- **Decision:** Ubuntu 22.04 / 24.04 x64 with NVIDIA driver ≥ 550 is supported. Other distros are best-effort.
