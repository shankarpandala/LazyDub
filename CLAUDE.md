# CLAUDE.md

Maata (working name; the repo is `LazyDub`) is a native macOS app that plays YouTube videos dubbed into Telugu, entirely on-device.

## Read first

1. `docs/SPEC.md`: the source of truth. Read all of it before doing anything.
2. `docs/plans/phase-0.md`: the current phase plan and the open decisions (Q1–Q13). `docs/plans/phase-0-methods.md` says how every number is measured.
3. `docs/research/2026-09-23-dependency-survey.md`: researched facts about speech-swift, YouTubeKit, yt-dlp, the models and the toolchain. Each item's verification level is in `docs/research/evidence/`.

## Current status

- **Phase 0 (spikes): the plan is written and is waiting for the maintainer's go-ahead.** No code yet.
- Per spec §0, each phase runs: plan, then go-ahead, then build with tests, then report, then stop.

## Where work runs

- **Build, run and measure on the reference Mac:** M5 Pro, 24 GB, with the Xcode named in an ADR. Standard-preset (16 GB) numbers come from a named 16 GB Mac, if the maintainer provides one.
- **Cloud (Linux) sessions** can't build the app, run MLX or Core ML, or reach YouTube (bot wall). Use them only for research and MLX-free pure-Swift packages.
- **Never report a measurement** that wasn't produced on the reference Mac (or the named 16 GB Mac) and backed by a committed JSON result.

## Commands

To be filled in when the Phase 0 groundwork lands. Planned:

- `xcodegen generate`: regenerate `Maata.xcodeproj` from `project.yml`. Never hand-edit the project.
- `xcodebuild -scheme Maata -destination 'platform=macOS,arch=arm64' -onlyUsePackageVersionsFromResolvedFile build`: xcodebuild bundles MLX's `default.metallib` automatically. For `swift build` / `swift test` on MLX packages, use speech-swift's `scripts/build_mlx_metallib.sh` (spec §4).
- `swift test --package-path Packages/<MLX-free package>`: pure-logic packages.
- `maata-bench <stage> …`: prints JSON; ships with `mlx-swift_Cmlx.bundle` beside the binary.

## Module map (spec §5; planned)

MaataCore · StreamResolving · MediaIngest · SpeechFrontEnd · Translation · Synthesis · Timing · Playback · Orchestration · ModelManager · Storage · Diagnostics. Each is a local SwiftPM package under `Packages/`, and spikes live in `Spikes/`, which is throwaway.

## Conventions

- **Language:** Swift 6 language mode with complete strict concurrency. XcodeGen defaults to Swift 5, so set `SWIFT_VERSION: 6.0` explicitly.
- **Concurrency:**
  - Actors for stateful services, `@Observable` for UI state, and structured concurrency.
  - speech-swift and YouTubeKit are Swift 5-mode and non-Sendable. Wrap them in actors, and serialise heavy MLX work.
- **Safety and errors:** no force-unwraps outside tests; one OSLog category per module; typed errors per module that map to user-facing messages.
- **Dependencies:**
  - Pin exact tags or commits, and record each as an ADR in `docs/DECISIONS.md`.
  - Ask the maintainer before adding any dependency the spec doesn't name.
  - Commit `Package.resolved`.
- **Linking:** never copy `-Wl,-undefined,dynamic_lookup` from speech-swift's example projects.
- **Models:** never let a library download on its own.
  - Fetch only through `maata-bench fetch` from `models.lock.json` (pinned commit and sha256).
  - Load only from local paths: speech-swift's local-directory loaders, or `offlineMode: true` with our cache directory. Its online path tracks `main` with no revision pinning.
  - Set `INDIC_MIO_WAVLM_BUNDLE` whenever Indic-Mio is used.
  - Run model benches with outbound network denied (see `docs/plans/phase-0-methods.md` §1).
- **YouTubeKit:** always pass `methods: [.local]`. The `.remote` method turns the user's Mac into an HTTP proxy for a third-party server.
- **Test media:** only self-recorded or CC0/CC-BY. Never commit downloaded YouTube media.

## Hard constraints (spec §3, non-negotiable)

- On-device inference only.
- Native runtime only: MLX or Core ML, no Python, PyTorch or ONNX at runtime.
- No bundled model weights.
- AVPlayer only, and the original audio is never played.
- macOS 15+, Apple Silicon only.
- Apache-2.0 app code, with every model's license shown at download.
- Distribution as a notarized DMG.

If a constraint looks impossible, stop and lay out options.
