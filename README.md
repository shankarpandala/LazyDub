# Maata · మాట

**Any YouTube video, spoken in Telugu — in the speaker's own voice. On your Mac.**

Maata plays a YouTube video with its original audio muted, while every person in it speaks Telugu in an AI clone of their own voice, timed to what's on screen. Speech recognition, speaker separation, translation and voice synthesis all run **locally** — tuned first for Apple Silicon (MLX + Metal), with NVIDIA GPUs supported too. Nothing you watch leaves your machine.

> The Telugu voices are AI-generated. The app always shows an **AI dub** badge.

## Quick start (Apple Silicon Mac)

Requirements: macOS 14+, Apple Silicon, 16 GB memory (24 GB recommended), ~20 GB free disk, and `uv`, Node 22, Rust (`rustup`) and Deno (`brew install deno`).

```bash
git clone https://github.com/shankarpandala/LazyDub && cd LazyDub
export HF_TOKEN=…            # once, after accepting pyannote/speaker-diarization-community-1's terms on Hugging Face
export CHATTERBOX_SRC=…      # the patched Chatterbox that supports Telugu (`te`)
./scripts/setup-mac.sh       # engine, ~9.4 GB of pinned models, UI
cd app && npm run tauri dev  # launch Maata
```

Paste a YouTube link, press Play.

### Try the interface without models

```bash
cd app && npm ci && npm run build
cd ../engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765
# open http://127.0.0.1:8765/?token=demo
```

The demo engine runs the full pipeline (timing, sync, UI) with synthetic audio and no network; the app labels it **Demo engine**.

## How it works

```
Tauri shell ──spawns──► Python engine (127.0.0.1, per-launch token)
   │                      yt-dlp → Whisper (MLX) → pyannote (MPS) → TranslateGemma (MLX)
   │                      → chatterbox-telugu (MPS) → isochrony fitter → dub units
   ▼
Web UI: YouTube player (muted) + Web Audio sync engine, scheduled against the player clock
```

- `engine/` — the inference engine and `maata-bench`. The timing core (§6.6 isochrony cascade, DubTimeline, akshara counting, Telugu normalisation) is pure Python with property tests.
- `app/` — the Svelte UI and the Tauri shell.
- `docs/` — spec, amendment, plans, decisions (ADRs) and research.

## Measure on your Mac

```bash
cd engine
uv run maata-bench pipeline path/to/clip.wav --backend apple   # JSON: throughput, first audio, speed-ups, memory
```

## Privacy and responsible use

- All inference happens on your device. The network is used only for YouTube, pinned model downloads, and an optional update check.
- There is no export of dubbed media, cloned voices or reference clips, and no way to make a speaker say arbitrary text. Voice data stays in the local cache.
- Maata is **not affiliated with YouTube or Google**. It is for personal viewing; you are responsible for following YouTube's Terms of Service and your local laws. Please support creators on YouTube itself.

## Licence

App code: Apache-2.0. Each model has its own licence, shown before download (see `engine/models.lock.json`).
