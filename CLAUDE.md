# CLAUDE.md

Maata (working name; the repo is `LazyDub`) is a cross-platform desktop app that plays YouTube videos dubbed into Telugu, with all AI inference on the user's device.

## Read first

1. `docs/SPEC.md` together with `docs/SPEC-AMENDMENT-01.md`. The amendment overrides the spec's native-macOS constraints: the app is now a Tauri shell with a web UI plus a local Python inference engine, on NVIDIA (Windows/Linux) and Apple Silicon, using YouTube's embedded player.
2. `docs/plans/phase-0.md`: the current plan, decisions D1–D10, inputs and defaults. `docs/plans/phase-0-methods.md` says how numbers are measured.
3. `docs/research/`: a survey from the earlier native-macOS plan. Its model, licence, TranslateGemma and YouTube-client findings still apply; its speech-swift, Swift and Xcode findings don't.

## Current status

- **Phase 0 (spikes): the plan is written and is waiting for the maintainer's go-ahead.** No code yet.
- Per spec §0, each phase runs: plan, go-ahead, build with tests, report, stop.

## Where work runs

- **Reference machines:** the M5 Pro (24 GB, Apple Silicon), plus an NVIDIA machine named in D6.
- **Measurements:** only numbers produced on a reference machine and backed by committed JSON are ever reported.
- **Cloud (Linux, no GPU) sessions:** usable for the UI, the Rust shell, pure-logic Python/TS and CI config. They can't measure performance or reach YouTube (bot wall).

## Commands

To be filled in when the Phase 0 groundwork lands.

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
- **Engine interface:** a loopback WebSocket, using a per-launch token from the shell. The engine makes no network calls except yt-dlp fetching from YouTube.
- **yt-dlp:** pinned; `--no-remote-components`; never `-U`.
- **Test media:** only self-recorded or CC0/CC-BY. Never commit downloaded YouTube media.

## Hard constraints

- On-device inference only: no cloud AI, no telemetry.
- No bundled model weights; the engine runtime is downloaded pinned and checksummed.
- The YouTube player is always muted, and the original audio is never played.
- Apache-2.0 app code, with every model's licence shown at download.
- Signed installers per OS.

If a constraint looks impossible, stop and lay out options.
