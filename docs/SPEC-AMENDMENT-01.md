# Spec amendment 01: cross-platform hybrid app (2026-09-23)

The maintainer changed the product from a native macOS app to a hybrid desktop app that runs on any major OS. This amendment **overrides** the sections of `SPEC.md` named below. Everything else in `SPEC.md` still applies: product, experience principles, pipeline design (§6), timing (§6.6), Model Manager (§7), cache (§8), responsible use (§13).

## Decisions (from the maintainer)

| Topic | Decision |
|---|---|
| Architecture | **Desktop shell with a web UI plus a local inference engine.** The shell is Tauri, and the engine runs as a local sidecar process. |
| Inference | **On-device only**, unchanged. No cloud AI, no remote inference, no telemetry. |
| Hardware | **NVIDIA GPUs (CUDA)** on Windows and Linux, and **Apple Silicon** on macOS. CPU-only, AMD and Intel GPUs are out of scope for v1. |
| Video playback | **YouTube's official embedded (IFrame) player, muted.** Dub audio is synced to its clock. The audio stream is still fetched separately, for processing only. |

## Sections replaced

- **§0 and §1 wording:** "a native macOS app" and "on the Mac" become "a desktop app for Windows, Linux and macOS". The trade-offs to be thorough about are now cross-platform desktop, GPU runtimes and packaging, not Swift specifically.
- **§2 "Not in v1":** "Intel Macs and iOS" becomes "CPU-only machines, AMD/Intel GPUs, Intel Macs, mobile".
- **§3.2 Native runtime:** replaced. The UI is web technology in a Tauri webview. The inference engine is a local process, and Python (PyTorch and friends) is allowed in it. Each backend uses the platform's accelerator: CUDA on NVIDIA; MPS or MLX on Apple Silicon. Nothing runs in the cloud.
- **§3.3 Models are never bundled:** unchanged. The engine runtime itself (a Python environment and its wheels) may also be downloaded on first run rather than bundled, under the same pinning and checksum rules.
- **§3.4 Native playback:** replaced by the IFrame player, which is always muted. The original audio is never played.
- **§3.5 Platform:** Windows 10/11 x64 and Linux x64 with an NVIDIA GPU; macOS 14+ on Apple Silicon. Minimum memory tiers are set in Phase 0.
- **§3.7 Distribution:** GitHub Releases with signed installers per OS: a notarized DMG on macOS, a signed MSI/NSIS installer on Windows, and an AppImage/deb on Linux.
- **§4 Environment and conventions:** replaced by the plan's toolchain section. There are two reference machines: the M5 Pro (24 GB) and an NVIDIA machine to be named.
- **§5 Architecture:** same pipeline and module responsibilities. The modules split between the engine (Python) and the UI/sync layer (TypeScript in the webview), with the Tauri core (Rust) handling process management, downloads and storage.
- **§6.7 Playback and sync:** sync runs against the IFrame player's clock. Timeline edits (§6.6 step 4) are limited to what the IFrame API supports, which is established in Phase 0.
- **§10 Targets:** measured on both reference machines. Start-error targets may need revising once S5 shows what the IFrame clock allows; they are not relaxed silently.
- **§12 Phase 0:** replaced by `docs/plans/phase-0.md`.
