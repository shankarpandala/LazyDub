# Maata · మాట

**YouTube videos, dubbed into Telugu on your Mac.**

Maata dubs a video in the background and saves an MP4 with Telugu audio, the original music and effects, and selectable Telugu and English subtitles. Speech recognition, speaker detection, music separation and speech synthesis run locally. Translation sends transcript, translation and video-context text to OpenAI through your signed-in [Codex CLI](https://developers.openai.com/codex/cli/); audio stays on your device.

On Apple Silicon, speech uses **OmniVoice with Male and Female output profiles**. Maata estimates a suitable profile from clean source speech, or you can choose it for each speaker. Ambiguous speech requires a manual choice before synthesis. These are AI-generated voices: a matching profile does not reproduce the source speaker's identity, and voice character can vary between sentences. The app labels its output **AI dub**.

## Install on an Apple Silicon Mac

Requirements: **macOS 14 or later**, an **Apple Silicon (arm64) Mac**, **16 GB memory** minimum (**24 GB recommended**), and **30 GB free disk recommended** before setup. Allow at least 20 GiB (about 21.5 GB); longer videos need additional working and output space.

1. Download the Apple Silicon DMG from [GitHub Releases](https://github.com/shankarpandala/LazyDub/releases).
2. Open it and drag **Maata.app** to **Applications**.
3. Quit Maata if it is running. Install the [uv](https://docs.astral.sh/uv/getting-started/installation/), [Deno](https://docs.deno.com/runtime/getting_started/installation/) and [Codex CLI](https://developers.openai.com/codex/cli/) prerequisites, install Python with `uv python install 3.12`, then run **Setup Maata.command** from the mounted disk image.
4. Complete `codex login` yourself, if needed. Setup does not sign you in.
5. When setup finishes, launch **Maata** from Applications.

The DMG includes the app, setup command and an Applications shortcut. **Node.js and Rust are not required to use this installer.** First setup needs an internet connection to install the pinned Python environments and download approximately **8.58 GB of verified model files**; model weights are not bundled in the DMG.

This release is **ad-hoc signed and not notarized by Apple**. If macOS blocks it, review the release/source first, then follow Apple's **System Settings → Privacy & Security → Open Anyway** process only if you trust the download. See the [installation guide](docs/release-installation.md) for prerequisites, optional speaker-model access, storage locations and troubleshooting.

Paste a YouTube link and press **Dub**. The Library shows queued jobs and progress, and completed videos are saved to `~/Movies/Maata` by default. Settings can change that folder. Closing the window leaves work running; quitting stops the engine gracefully. Explicitly paused or cancelled jobs remain stopped until you resume them.

## Voices and quality

Each speaker has an **Auto**, **Male** or **Female** profile choice. Auto uses conservative pitch evidence from clean diarized speech; it is a vocal-range estimate, not a gender-identity classifier. If there is insufficient evidence, choose Male or Female in the speaker panel and resume. Calibration and audio caches follow the resolved profile, so changing one speaker's profile does not relabel another profile's saved audio.

Translations use GPT-6 Luna (`gpt-6-luna`) at low reasoning effort for the brief, scene translation, wording adjustments and meaning reviews. Corrective wording requires a fresh meaning review. Timing can use shorter reviewed wording and bounded time stretching; it does not cut speech to a target duration. These checks do not establish that every line is complete or naturally pronounced. Native Telugu listening and representative long-video quality evaluation remain ongoing.

## Build from source

Development additionally requires Node.js 22.12+ and Rust through `rustup`, plus the macOS development tools needed by Tauri. Keep uv, Deno and Codex CLI available on your PATH.

```bash
git clone https://github.com/shankarpandala/LazyDub
cd LazyDub
./scripts/setup-mac.sh
codex login
cd app && npm run tauri dev
```

The developer setup installs the engine, its separate OmniVoice runtime, pinned models and UI dependencies. Optional multi-speaker diarization requires access to the gated pyannote model; see the [installation guide](docs/release-installation.md#optional-multi-speaker-model). The source also contains an NVIDIA backend with a different audio stack; the packaged macOS release described here targets Apple Silicon.

To build the installer from a prepared checkout, run `./scripts/build-installer.sh`. It builds the app and writes `dist/Maata-0.1.0-macos-arm64.dmg` with `dist/SHA256SUMS`. The build prepares a checksummed engine-source payload; it does not bundle model weights. End users still run the included setup command.

### Try the interface without models

```bash
cd app && npm ci && npm run build
cd ../engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765
# Open http://127.0.0.1:8765/?token=demo
```

Paste any YouTube link. The demo uses a synthetic test video, stand-in models and an offline translator, and labels itself **Demo engine**. Add `--config-dir <folder>` to keep its settings apart from the app's.

## How it works

```text
Tauri shell → local Python engine: one job at a time, with a queue
  YouTube audio → whole-file speaker detection → Whisper transcript → sentences
  → source-matched Male/Female profiles + per-profile calibration
  → full-video brief and scene translation/review through Codex CLI (text only)
  → local music/effects separation + OmniVoice Telugu speech
  → timing planner → mix → MP4 with Telugu/English subtitles
```

OmniVoice runs in a persistent, isolated Python environment with pinned model and runtime versions. It does not condition its production voices on English reference audio. PerTh watermarking follows timing changes. Chatterbox remains available as an explicit legacy rollback; it is not the Apple release's default voice engine.

- `engine/` — Python engine, timing/text logic, backends and `maata-bench`.
- `app/` — Svelte UI and Tauri shell.
- `docs/` — specification, decisions, research and recorded evaluations.

Per-job work is stored in `~/Library/Caches/Maata/<video id>/render/`. The adjacent `units.jsonl` trace contains source text, Telugu text and timing details; treat it as private when sharing diagnostics.

## Measure on your Mac

```bash
cd engine
uv run maata-bench pipeline path/to/video.mp4 --backend apple
uv run maata-bench pipeline path/to/video.mp4 --backend apple --translator mock
cd .. && ./scripts/verify-mac.sh
```

The second benchmark keeps translation offline with a stand-in translator. The verification script checks a synthetic speech/music video, separation, audio/video timing and loudness. These checks are not a listening-quality guarantee or a promise of dubbing speed on every video.

## Privacy and licences

Audio inference stays on your device. Network access is used for YouTube, setup/model downloads and text translation through Codex. Translation uses your signed-in account and its usage limits; your account's applicable data controls and retention policies apply. The engine disables supported model-library telemetry.

App code is **Apache-2.0**. Model licences differ: OmniVoice model weights are listed as **CC-BY-NC** by the upstream model card, while its code is Apache-2.0. Other model licences and exact revisions are recorded in [engine/models.lock.json](engine/models.lock.json). They are not all permissively licensed software, and the app's licence does not replace model terms.

Maata is for personal viewing and is not affiliated with YouTube or Google. Follow the source service's terms and applicable law, and support creators on YouTube.
