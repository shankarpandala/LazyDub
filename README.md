# Maata · మాట

**Any YouTube video, spoken in Telugu — in the speaker's own voice. On your Mac.**

Maata dubs a whole YouTube video into Telugu in the background and saves it as an MP4: every person in it speaks Telugu in an AI clone of their own voice, timed to what's on screen, over the video's own music and sound effects, with Telugu and English subtitles you can switch on in the player. Speech recognition, speaker separation, music separation and voice synthesis run **locally** — tuned first for Apple Silicon (MLX + Metal), with NVIDIA GPUs supported too — and audio never leaves your machine. Translation goes through your own [Claude Code](https://claude.com/claude-code), signed in with your own plan: only the English transcript, as text, is sent.

> The Telugu voices are AI-generated. The app always shows an **AI dub** badge.

## Quick start (Apple Silicon Mac)

Requirements: macOS 14+, Apple Silicon, 16 GB memory (24 GB recommended), ~20 GB free disk, and `uv`, Node 22.12+, Rust (`rustup`), Deno (`brew install deno`) and Claude Code, signed in (`claude auth login`).

```bash
git clone https://github.com/shankarpandala/LazyDub && cd LazyDub
export HF_TOKEN=…            # optional: after accepting pyannote/speaker-diarization-community-1's terms on Hugging Face
./scripts/setup-mac.sh       # engine (incl. the pinned Telugu Chatterbox), ~4.9 GB of pinned models, UI
cd app && npm run tauri dev  # launch Maata
```

Paste a YouTube link and press **Dub**. Maata works through the whole video (or its first 15 minutes, as a preview) and saves `<title> (Telugu).mp4` in `~/Movies/Maata` (Settings changes the folder). It keeps going with the window closed; quitting pauses it, and the next launch continues. Dubs queue one after another, and the Library shows each one's progress and, when it's done, the file.

### Try the interface without models

```bash
cd app && npm ci && npm run build
cd ../engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765
# open http://127.0.0.1:8765/?token=demo
```

Paste any YouTube link and press **Dub**: the demo engine runs the whole job (speakers, sentences, voices, background sound, translation, timing, the MP4) on a synthetic test video, with stand-in models, a stand-in translator and no network, and saves the MP4 in the output folder. The app labels it **Demo engine**. Add `--config-dir <folder>` to keep its settings apart from the app's.

## How it works

```
Tauri shell ──spawns──► Python engine (127.0.0.1, per-launch token): one dub job at a time, the rest queued
   │                      yt-dlp audio → pyannote (MPS) on the whole file → Whisper (MLX) → sentence units
   │                      → every speaker's voice cloned (chatterbox-telugu) + a video brief from the whole transcript
   │                      → music & effects separated (Mel-Band RoFormer, MLX) │ scenes → your Claude Code (text only)
   │                      → Telugu takes (MPS) on a timeline planner (never trims) → the mix → MP4 with subtitles
   ▼
Web UI: the Library, New dub, each job's progress, its speaker check and Settings
```

The music separation runs on Apple Silicon; on an NVIDIA machine the MP4 has the Telugu voices alone, for now.

Every local model is open source (MIT, Apache-2.0 or CC-BY-4.0); see `engine/models.lock.json`. No translation model is downloaded (ADR-019).

Logs: `~/Library/Logs/Maata/engine.log` has one line per dubbed sentence, and `~/Library/Caches/Maata/<video id>/units.jsonl` has the full per-line trace (source, Telugu, timing, voice). Each job's work is kept in `~/Library/Caches/Maata/<video id>/render/`, so a pause or a restart continues where it stopped.

- `engine/` — the inference engine and `maata-bench`. The timing core (§6.6 isochrony cascade, DubTimeline, akshara counting, Telugu normalisation) is pure Python with property tests.
- `app/` — the Svelte UI and the Tauri shell.
- `docs/` — spec, amendment, plans, decisions (ADRs) and research.

## Measure on your Mac

```bash
cd engine
uv run maata-bench pipeline path/to/video.mp4 --backend apple   # dubs it to an MP4; JSON: per-stage times, memory
uv run maata-bench pipeline path/to/video.mp4 --backend apple --translator mock   # the same, offline, without Claude
cd .. && ./scripts/verify-mac.sh   # the reference check: a synthesised speech + music video, separation, A/V, loudness
```

## Privacy and responsible use

- Speech recognition, diarization and voice synthesis happen on your device, and audio never leaves it. The network is used only for YouTube, pinned model downloads, an optional update check, and your Claude Code translating the transcript. Library telemetry that would otherwise phone home (pyannote.audio's usage metrics, Hugging Face Hub's) is switched off by the engine.
- Translation sends the video's English transcript, as text, to Anthropic through your own Claude Code and plan; it counts toward that plan's usage. Whether Anthropic may use it to improve its models, and how long it is kept, follow your account's privacy settings on claude.ai. The app says so once, before the first video.
- The dubbed MP4 is saved in your output folder for your own viewing. Cloned voices and reference clips are never exported, and there is no way to make a speaker say arbitrary text. Voice data stays in the local cache.
- Maata is **not affiliated with YouTube or Google**. It is for personal viewing; you are responsible for following YouTube's Terms of Service and your local laws. Please support creators on YouTube itself.

## Licence

App code: Apache-2.0. Each model has its own licence, shown before download (see `engine/models.lock.json`).
