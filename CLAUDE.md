# CLAUDE.md

Maata (working name; the repo is `LazyDub`) is a cross-platform desktop app that plays YouTube videos dubbed into Telugu, with all AI inference on the user's device.

## Read first

1. `docs/SPEC.md` together with `docs/SPEC-AMENDMENT-01.md`. The amendment overrides the spec's native-macOS constraints: the app is now a Tauri shell with a web UI plus a local Python inference engine, on NVIDIA (Windows/Linux) and Apple Silicon, using YouTube's embedded player.
2. `docs/plans/phase-0.md`: the current plan, decisions D1–D10, inputs and defaults. `docs/plans/phase-0-methods.md` says how numbers are measured.
3. `docs/research/`: a survey from the earlier native-macOS plan. Its model, licence, TranslateGemma and YouTube-client findings still apply; its speech-swift, Swift and Xcode findings don't.

## Current status

- The maintainer delegated decisions (goal: best performance on the M5 Pro 24 GB and a very good-looking UI). They are recorded as ADRs in `docs/DECISIONS.md`.
- Built: the engine (pure timing/text core, Apple/CUDA/mock backends, session pipeline, WebSocket server, pinned model fetcher, `maata-bench`), the Svelte UI and the Tauri shell.
- Runs end to end on the M5 Pro (2026-09-24): YouTube link → Telugu dub in the Tauri app, real models. Until pyannote's HF terms are accepted, diarization falls back to one speaker (ADR-013).
- Not yet measured on the M5 Pro: committed real-model numbers. Run `./scripts/verify-mac.sh` and commit its JSON.

## Where work runs

- **Reference machines:** the M5 Pro (24 GB, Apple Silicon), plus an NVIDIA machine named in D6.
- **Measurements:** only numbers produced on a reference machine and backed by committed JSON are ever reported.
- **Cloud (Linux, no GPU) sessions:** usable for the UI, the Rust shell, pure-logic Python/TS and CI config. They can't measure performance or reach YouTube (bot wall).

## Commands

- Engine tests: `cd engine && uv sync --group dev && uv run pytest -q`
- UI checks: `cd app && npm ci && npx svelte-check && npx vitest run && npm run build`
- Shell compile: `cd app/src-tauri && cargo check` (Linux needs libwebkit2gtk-4.1-dev)
- Demo (no models): `cd engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765`, then open `http://127.0.0.1:8765/?token=demo`
- Mac setup and launch: `./scripts/setup-mac.sh` (and `claude auth login` once: translation runs through the Claude CLI), then `cd app && npm run tauri dev`
- Fetch models: `uv run maata-bench fetch --backend apple`; bench: `uv run maata-bench pipeline FILE --backend apple` (translates through the Claude CLI; `--translator mock` runs it offline)
- Verify on the M5 Pro (real pipeline metrics + UI screenshot into `docs/spikes/results/<machine>/`): `./scripts/verify-mac.sh [YOUTUBE_URL]`, then review and commit the results

## Layout (planned)

- `app/`: the Tauri shell (Rust) and the web UI (TypeScript).
- `engine/`: the Python inference sidecar and `maata-bench`.
- `docs/`: spec, amendment, plans, ADRs (`DECISIONS.md`) and spike results.

## Conventions

- **Dependencies:**
  - Pin exact versions or commits: `uv.lock` for Python, `Cargo.lock` and the npm lockfile committed.
  - Record pins as ADRs.
  - Ask before adding any dependency the spec, the amendment or an approved decision doesn't name.
- **Models:** never let a library download on its own.
  - Fetch only through `models.lock.json` (pinned commit and sha256).
  - Load from local paths with `HF_HUB_OFFLINE=1`.
  - Run model benches with outbound network blocked.
- **Engine interface:** a loopback WebSocket, using a per-launch token from the shell. The engine makes no network calls except yt-dlp fetching from YouTube and the `claude` CLI for translation.
- **yt-dlp:** pinned; `--no-remote-components`; never `-U`.
- **Test media:** only self-recorded or CC0/CC-BY. Never commit downloaded YouTube media.

## Hard constraints

- On-device inference for all audio: speech recognition, diarization, TTS and voice cloning run locally, and audio never leaves the machine. No telemetry.
- Translation is the one exception (maintainer, 2026-09-24): no local LLMs. English→Telugu text goes through the `claude` CLI on the maintainer's own subscription, run headless and sealed off (no tools, hooks, plugins or MCP servers). Only transcript text is sent.
- Model licences (maintainer, 2026-09-24): prefer open-source weights (MIT, Apache-2.0, BSD, CC-BY/CC0). The maintainer has also accepted the Gemma Terms of Use, so TranslateGemma is allowed. Ask before any other restrictive licence (Llama/community licences), and never use non-commercial terms (CC-BY-NC).
- No bundled model weights; the engine runtime is downloaded pinned and checksummed.
- The YouTube player is always muted, and the original audio is never played.
- Apache-2.0 app code, with every model's licence shown at download.
- Signed installers per OS.

If a constraint looks impossible, stop and lay out options.
