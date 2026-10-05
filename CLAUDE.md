# CLAUDE.md

Maata (working name; the repo is `LazyDub`) is a cross-platform desktop app that dubs a whole YouTube video into Telugu in the background and saves the dubbed video as an MP4 (maintainer, 2026-10-03). All audio inference runs on the user's device.

## Read first

1. `docs/SPEC.md` together with `docs/SPEC-AMENDMENT-01.md`. The amendment overrides the spec's native-macOS constraints: the app is now a Tauri shell with a web UI plus a local Python inference engine, on NVIDIA (Windows/Linux) and Apple Silicon, using YouTube's embedded player. ADR-021 supersedes its embedded player: nothing plays in the app.
2. `docs/plans/phase-0.md`: the current plan, decisions D1–D10, inputs and defaults. `docs/plans/phase-0-methods.md` says how numbers are measured.
3. `docs/research/`: a survey from the earlier native-macOS plan. Its model, licence, TranslateGemma and YouTube-client findings still apply; its speech-swift, Swift and Xcode findings don't.

## Current status

- The maintainer delegated decisions (goal: best performance on the M5 Pro 24 GB and a very good-looking UI). They are recorded as ADRs in `docs/DECISIONS.md`.
- Built (ADR-021, 2026-10-04):
  - the engine: the pure timing/text core; Apple, CUDA and mock backends; the background render job; the WebSocket server with its job queue; the pinned model fetcher; `maata-bench`;
  - the background render job is `RenderJob` on `Dubber`: whole-file speakers, voices, brief, translation, the dub loop, the separator, the mix and the MP4 export;
  - the Svelte UI (Library, New dub, Job, Settings) and the Tauri shell;
  - the streaming player and `Session` are gone.
- Runs end to end on the M5 Pro (2026-10-04): a YouTube link becomes a dubbed MP4, with real models. An 8.2-minute talk took 9.8 minutes.
- Speed/quality pass (ADR-022, 2026-10-05): vectorized separator mask merge; duration-aware complete first translations;
  priority admission for queued Claude calls; selected-waveform checks with bounded retries; synthesis-versioned take
  restoration; corrected export wall-time traces. Models, precision and CFM defaults are unchanged.
- Translation latency (ADR-023): scenes now hold at most 30 seconds / 6 complete lines, preserving the whole-video
  brief, context, models and meaning reviews. Queued rephrases recheck the synthesis budget before spending it.
  `scripts/bench_scene_pipeline.py` compares explicit scene limits on fresh whole-video runs;
  `scripts/bench_translation_scenes.py` also checks context-dependent original text without loading audio models.
- Text model (ADR-025, maintainer request): all text calls use the signed-in Codex CLI with `gpt-6-luna` at low
  reasoning effort, including meaning reviews. This replaces Haiku and has no Claude fallback.
  Earlier Claude speed and quality measurements do not establish Luna's performance or Telugu quality.
- Long-queue monitoring (ADR-026): English substitutions are anchored to exact Telugu words and retained only
  for the tier approved by meaning review, which checks the actual Latin TTS input too. Cache restoration and
  voice calibration follow the corrected input. Completed outputs remain intact.
  Run-scoped stage and GPU-operation traces distinguish new work from cached take costs; inspect them with
  `engine/.venv/bin/python scripts/monitor-queue.py --out /tmp/maata-queue.json`.
  Streaming mix allocation is reduced without changing PCM; export progress includes the cancellable final
  audio-quality pass. Lane weights remain an internal experiment until measured against the balanced baseline.
- Reproducible component measurements and a real-model integration smoke live in `docs/spikes/results/`.
  `engine/.venv/bin/python scripts/bench_inference_components.py --component separator` reruns the paired offline
  separator benchmark. `engine/.venv/bin/python scripts/check-translation.py --out /tmp/maata-translation.json`
  checks the production prompt through the isolated Codex CLI on original text. The integration smoke covers MP4,
  subtitles, loudness, synchronization and watermark; it is not a paired whole-video speed or listening evaluation.
- Still owed: native Telugu listening, representative long-video throughput, and the complete `./scripts/verify-mac.sh`
  acceptance run (including separation/leakage quality and UI inspection).
  The latest original-audio smoke passes export checks but has excess uncovered source speech from early endings;
  final heuristic coverage must not be presented as independent semantic approval.

## Where work runs

- **Reference machines:** the M5 Pro (24 GB, Apple Silicon), plus an NVIDIA machine named in D6.
- **Measurements:** only numbers produced on a reference machine and backed by committed JSON are ever reported.
- **Cloud (Linux, no GPU) sessions:** usable for the UI, the Rust shell, pure-logic Python/TS and CI config. They can't measure performance or reach YouTube (bot wall).

## Commands

- Engine tests: `cd engine && uv sync --group dev && uv run pytest -q`
- UI checks: `cd app && npm ci && npx svelte-check && npx vitest run && npm run build`
- Shell compile: `cd app/src-tauri && cargo check` (Linux needs libwebkit2gtk-4.1-dev)
- Demo (no models):
  - Start: `cd engine && uv run maata-engine --backend mock --demo --ui ../app/dist --token demo --port 8765` (add `--config-dir <dir>` to keep its settings apart).
  - Open `http://127.0.0.1:8765/?token=demo`, paste any YouTube link and press Dub.
  - The dubbed MP4 lands in the output folder: `~/Movies/Maata`, unless Settings says otherwise.
- Mac setup and launch: `./scripts/setup-mac.sh` (and `codex login` once: translation runs through Codex), then `cd app && npm run tauri dev`
- Fetch models: `uv run maata-bench fetch --backend apple`.
- Bench: `uv run maata-bench pipeline FILE --backend apple`.
  - FILE is a local video. It is dubbed to an MP4 in `--out`, by default `<cache>/out`.
  - Translation goes through Codex CLI; `--translator mock` runs offline.
- Verify on the M5 Pro: `./scripts/verify-mac.sh` (it takes no URL), then review and commit the results.
  - It makes a speech + music video, then checks separation, runs the bench under a network sandbox that allows the Codex CLI's OpenAI traffic, checks the MP4 and takes a Library screenshot.
  - The results go into `docs/spikes/results/<machine>/`.
- Check YouTube once: `./scripts/check-youtube-video.sh URL` shows which video format the engine gets.

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
- **Engine interface:** a loopback WebSocket, using a per-launch token from the shell. The engine makes no network calls except yt-dlp fetching from YouTube and the `codex` CLI for translation.
- **yt-dlp:** pinned; `--no-remote-components`; never `-U`.
- **Test media:** only self-recorded or CC0/CC-BY. Never commit downloaded YouTube media.

## Hard constraints

- On-device inference for all audio: speech recognition, diarization, TTS and voice cloning run locally, and audio never leaves the machine. No telemetry.
- Translation is the one exception (maintainer, 2026-10-05, ADR-025): no local LLMs. English→Telugu text goes through the signed-in `codex` CLI on the maintainer's plan, headless with user configuration, hooks, plugins and action tools disabled. Transcript, translation and video-context text is sent; all audio inference stays local.
- The translation target is the everyday spoken Telugu of the general public (maintainer, 2026-10-03), with English words where people naturally use them, never bookish. Everything else stays on the device.
- Model licences (maintainer, 2026-10-03): they don't constrain choices. Maata is for the maintainer's personal viewing only, so pick the best model that runs locally, whatever its licence. Weights are still pinned and loaded offline.
- No bundled model weights; the engine runtime is downloaded pinned and checksummed.
- The output is a dubbed video file (maintainer, 2026-10-03). The original English speech is removed by the on-device separator; a faint residue can remain where no Telugu covers it. Only the original music and effects are kept, separated on the device by a local source-separation model and mixed under the Telugu voices. The file has one audio track (Telugu) and two soft subtitle tracks (Telugu, English). It is for personal viewing.
- Apache-2.0 app code, with every model's licence shown at download.
- Signed installers per OS.

If a constraint looks impossible, stop and lay out options.
