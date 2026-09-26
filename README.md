# Maata · మాట

**Any YouTube video, spoken in Telugu — in the speaker's own voice. On your Mac.**

Maata plays a YouTube video with its original audio muted, while every person in it speaks Telugu in an AI clone of their own voice, timed to what's on screen. Speech recognition, speaker separation and voice synthesis run **locally** — tuned first for Apple Silicon (MLX + Metal), with NVIDIA GPUs supported too — and audio never leaves your machine. Translation goes through your own [Claude Code](https://claude.com/claude-code), signed in with your own plan: only the English transcript, as text, is sent.

> The Telugu voices are AI-generated. The app always shows an **AI dub** badge.

## Quick start (Apple Silicon Mac)

Requirements: macOS 14+, Apple Silicon, 16 GB memory (24 GB recommended), ~20 GB free disk, and `uv`, Node 22.12+, Rust (`rustup`), Deno (`brew install deno`) and Claude Code, signed in (`claude auth login`).

```bash
git clone https://github.com/shankarpandala/LazyDub && cd LazyDub
export HF_TOKEN=…            # optional: after accepting pyannote/speaker-diarization-community-1's terms on Hugging Face
./scripts/setup-mac.sh       # engine (incl. the pinned Telugu Chatterbox), ~4.9 GB of pinned models, UI
cd app && npm run tauri dev  # launch Maata
```

Paste a YouTube link, press Play.

### Try the interface without models

```bash
cd app && npm ci && npm run build
cd ../engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765
# open http://127.0.0.1:8765/?token=demo
```

The demo engine runs the full pipeline (timing, sync, UI) with synthetic audio, a stand-in translator and no network; the app labels it **Demo engine**.

## How it works

```
Tauri shell ──spawns──► Python engine (127.0.0.1, per-launch token)
   │                      yt-dlp → speaker pre-pass: pyannote (MPS), every voice cloned before the first line
   │                      → Whisper (MLX), sentence units → scenes → your Claude Code (text only, up to 3 calls at once)
   │                      → chatterbox-telugu (MPS) → timeline planner (never trims) → dub units, up to 10 min ahead
   ▼
Web UI: YouTube player (muted) + Web Audio sync engine, scheduled against the player clock
```

Every local model is open source (MIT, Apache-2.0 or CC-BY-4.0); see `engine/models.lock.json`. No translation model is downloaded (ADR-019).

Logs: `~/Library/Logs/Maata/engine.log` has one line per dubbed sentence, and `~/Library/Caches/Maata/<video id>/units.jsonl` has the full per-line trace (source, Telugu, timing, voice).

- `engine/` — the inference engine and `maata-bench`. The timing core (§6.6 isochrony cascade, DubTimeline, akshara counting, Telugu normalisation) is pure Python with property tests.
- `app/` — the Svelte UI and the Tauri shell.
- `docs/` — spec, amendment, plans, decisions (ADRs) and research.

## Measure on your Mac

```bash
cd engine
uv run maata-bench pipeline path/to/clip.wav --backend apple   # JSON: throughput, first audio, speed-ups, memory
uv run maata-bench pipeline path/to/clip.wav --backend apple --translator mock   # the same, offline, without Claude
```

## Privacy and responsible use

- Speech recognition, diarization and voice synthesis happen on your device, and audio never leaves it. The network is used only for YouTube, pinned model downloads, an optional update check, and your Claude Code translating the transcript. Library telemetry that would otherwise phone home (pyannote.audio's usage metrics, Hugging Face Hub's) is switched off by the engine.
- Translation sends the video's English transcript, as text, to Anthropic through your own Claude Code and plan; it counts toward that plan's usage. Whether Anthropic may use it to improve its models, and how long it is kept, follow your account's privacy settings on claude.ai. The app says so once, before the first video.
- There is no export of dubbed media, cloned voices or reference clips, and no way to make a speaker say arbitrary text. Voice data stays in the local cache.
- Maata is **not affiliated with YouTube or Google**. It is for personal viewing; you are responsible for following YouTube's Terms of Service and your local laws. Please support creators on YouTube itself.

## Licence

App code: Apache-2.0. Each model has its own licence, shown before download (see `engine/models.lock.json`).
