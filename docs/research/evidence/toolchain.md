# Toolchain claims: verification (2026-09-23)

Checked against primary sources only: fresh clones (paths are relative to the scratchpad directory), swift.org JSON, Apple release-notes markdown, the HF API, and actions/runner-images.
I could not run a build here: this is a Linux x86_64 box with no Xcode. So "it builds" is only as strong as the evidence cited.

Clones used:
- `verify-tc-mlx-swift` (mlx-swift, tag 0.31.6 = 0bb916c, 2026-07-01, fetched without blobs, full history)
- `verify-tc-mlx` (mlx submodule pin ce45c52)
- `verify-tc-mlx-swift-lm` (tag 3.31.4 = bd4b743, 2026-06-29, plus origin/main)
- `verify-tc-xcodegen` (tag 2.46.0)
- `verify-tc-runner-images` (main 2026-09-23)
- `verify-tc-swiftpm` (swift-package-manager main, plus tags swift-6.2.3/6.3/6.3.3-RELEASE)
- `verify-tc-speech-swift` (fetched without blobs, for history)
- the existing `speech-swift` clone (c4c2fab)
- `verify-tc-mlx-lm` (Python reference)
- `verify-tc-swift-transformers` (tag 1.3.4)
- `verify-tc-hf/*` (HF configs and headers)

---

## X1 - mlx-swift latest is 0.31.x (~2026-07) and REQUIRES Swift 6.3 - **PARTLY**

- The latest tag is **0.31.6**, commit 0bb916c dated 2026-07-01 (`git ls-remote --tags`; `git log -1`). main (9019419, 2026-09-17) is 22 commits further on and has no newer tag.
- `Package.swift:1` at 0.31.6 is `// swift-tools-version: 6.3;(experimentalCGen)`.
- Tools version at each tag (`git show <tag>:Package.swift | head -1`):
  - 0.29.1: 5.10
  - 0.30.1 through **0.31.4**: 5.12
  - **0.31.5 and 0.31.6**: 6.3;(experimentalCGen)
  - The change came in commit e23ae6b "Allowing SPM to compile on Linux with CUDA (#413)", 2026-06-17.
- `experimentalCGen` is a SwiftPM 6.3 feature. See swiftpm `Sources/PackageModel/ToolsVersion.swift:93-105` (`self >= .v6_3 && ...`) and `Sources/PackageLoading/ToolsVersionParser.swift:375`. It is present in the swift-6.3-RELEASE tag and absent in swift-6.2.3-RELEASE.
- The README says nothing about Swift 6.3 or an Xcode version. There is no `#if compiler(>=6.3)`; the only checks are `#if swift(>=5.10)` in `Source/MLX/Device.swift:95` and `Memory.swift:68`.
- **Correction:** only 0.31.5 and 0.31.6 need Swift 6.3; 0.31.1 through 0.31.4 do not.
  - SwiftPM's resolver drops versions whose tools version is incompatible and falls back to older ones (`Workspace/PackageContainer/SourceControlPackageContainer.swift:164-176,421-436`; `PubGrubPackageContainer.swift:167-175`).
  - So a Swift 6.2 or 6.1 toolchain quietly resolves **mlx-swift 0.31.4**. That still meets mlx-swift-lm 3.31.4's `.upToNextMinor(from: "0.31.4")` (mlx-swift-lm `Package.swift:39`).
  - Only mlx-swift-lm **main** (tools 6.2, `.upToNextMinor(from: "0.31.6")`) needs 6.3.
- **New risk:** 0.31.5 and 0.31.6 attach the `CudaBuild` build-tool plugin to `Cmlx` on every platform (`Package.swift:292-294`).
  - On macOS it does nothing (`Plugins/CudaBuild/plugin.swift:190-195`), but Xcode still asks you to trust the plugin.
  - Without `-skipPackagePluginValidation`, headless `xcodebuild` fails at "Validate plug-in 'CudaBuild' in package 'mlx-swift'". A third-party PR (Dagmawi-Y/VoiceInk#1, 2026-09-20) reports exactly this.
  - Commit 3a24fa3 on main makes the plugin Linux-only (2026-08-14), but that fix is **not in any tag**.

## X2 - Which Xcode 26.x ships Swift 6.3? Can stable Xcode 26 build these? Is Xcode 27 still beta? - **PARTLY; the Xcode 27 premise is wrong**

- From swift.org `api/v1/install/releases.json`:

  | Swift | Date | Xcode |
  |---|---|---|
  | 6.2 | 2025-09-15 | 26 |
  | 6.2.1 | | 26.1 |
  | 6.2.3 | | 26.2 |
  | 6.2.4 | | 26.3 |
  | **6.3** | **2026-03-24** | **26.4** |
  | 6.3.1 | | 26.4.1 |
  | 6.3.2 | | 26.5 |
  | 6.3.3 | 2026-06-29 | 26.6 |
  | **6.4.0** | **2026-09-14** | **27.0** (xcode_release=true) |

- Apple release notes (`developer.apple.com/documentation/xcode-release-notes/xcode-<v>-release-notes.md`):
  - Xcode 26: "includes Swift 6.2 ... requires macOS Sequoia 15.6 or later".
  - Xcode 26.3: "includes Swift 6.2.3 ... Sequoia 15.6".
  - **Xcode 26.4: "includes Swift 6.3 ... requires a Mac running macOS Tahoe 26.2 or later"**.
  - Xcode 26.6: "includes Swift 6.3 ... Tahoe 26.2".
  - **Xcode 27: "includes Swift 6.4 ... requires a Mac running macOS Tahoe 26.6 or later"**. The page is titled Release Notes, not beta.
- xcodereleases.com agrees: 27.0 released 2026-09-14 (27A266a), 27.1 and 27.2 are in beta, and 26.6 was released 2026-06-25.
- **Correction:** Xcode 27 is **not beta**. It went GA on 2026-09-14, so "stable Xcode" now means 27.0 / Swift 6.4, which neither upstream tests against. The newest 26.x is 26.6 (Swift 6.3.3).
- What Xcode 26.x resolves to:
  - **Xcode 26.4-26.6** meet mlx-swift 0.31.6's tools-version 6.3, and the resolver picks 0.31.6. That needs `-skipPackagePluginValidation` in CI.
  - **Xcode 26.0-26.3** (Swift 6.2.x) fall back to mlx-swift 0.31.4.
  - On a Mac still running macOS 15, Xcode 26.3 is the ceiling.
- speech-swift HEAD's own CI runs on `macos-15` with the default Xcode 16.4, i.e. Swift 6.1 (`.github/workflows/tests.yml:15`; `release.yml:18,28` explicitly selects `Xcode_16.4.app`). It is green at c4c2fab (GitHub Actions page), so it demonstrably builds against **mlx-swift 0.31.4**.
- speech-swift HEAD + mlx-swift 0.31.6 + Xcode 26.6 / 27 is **not exercised upstream: unverifiable**.

## X3 - mlx-swift-lm 3.31.4: Swift 6 mode, gemma3/gemma4/qwen3; TranslateGemma loadable? - **PARTLY (structurally loadable; two fidelity gaps)**

- 3.31.4 is the latest tag (bd4b743, 2026-06-29). `Package.swift:1` is `swift-tools-version: 6.1` and there is no `swiftLanguageModes`. Tools ≥ 6 defaults to language mode 6 (swiftpm `ToolsVersion.swift` `swiftLanguageVersion`). **Confirmed.**
- The registry in `Libraries/MLXLLM/LLMModelFactory.swift:33-47` has:
  - `gemma3` and `gemma3_text`, both pointing to `Gemma3TextModel`
  - `gemma3n`, `gemma4`, `gemma4_unified`, `gemma4_text`
  - `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5*`
  - The matching model files exist: `Gemma3Text.swift`, `Gemma4.swift`, `Gemma4Text.swift`, `Qwen3.swift`.
- mlx-community conversions: `translategemma-{4b,12b,27b}-it-{4bit,8bit}`, plus 27b-bf16 and 6bit variants.
  - Their `config.json` has top-level **`model_type: "gemma3"`** (`Gemma3ForConditionalGeneration`) and **`text_config.model_type: "gemma3_text"`**.
  - `Gemma3TextConfiguration.init(from:)` reads the nested `text_config` (`Gemma3Text.swift:84-121`).
  - `sanitize` extracts the `language_model` subtree (`:362-372`).
  - The 4b-4bit safetensors header has **only** `language_model.*` keys: 922 tensors, embed [262208, 320]. File size is 2,183,295,977 bytes.
  - So the load path matches. It has not been executed.
- **Gap 1 (RoPE):** the TranslateGemma config has `rope_scaling: null` and puts the scaling in `rope_parameters.full_attention = {factor: 8.0, rope_type: linear}`.
  - Gemma3Text reads only `rope_scaling` (`:77,119-120,172-176`), and there is no `rope_parameters` in Gemma3Text on 3.31.4 or main.
  - Result: global layers run **without the ×8 linear scaling**. Python mlx-lm `gemma3_text.py:32,69` has the same gap.
  - Workaround: inject `rope_scaling: {"type":"linear","factor":8.0}` into `text_config`. `initializeRope` supports `linear` (`MLXLMCommon/RoPEUtils.swift:410-432`).
- **Gap 2 (stop token):** neither `config.json` nor `generation_config.json` has `eos_token_id`, and the tokenizer eos is `<eos>`.
  - Stop IDs come only from the config, the tokenizer eos, and `extraEOSTokens` (`MLXLMCommon/Evaluate.swift:1121-1130`).
  - The registry's Gemma-3 entries add `extraEOSTokens: ["<end_of_turn>"]` (`LLMModelFactory.swift:168-198`); TranslateGemma isn't registered.
  - So the app must set `extraEOSTokens: ["<end_of_turn>"]` itself.
- The chat template is in a separate `chat_template.jinja`, which swift-transformers 1.3.4 loads (`Sources/Hub/Hub.swift:224,277`). It needs structured user content (`source_lang_code`, `target_lang_code`, `text`); `te` / `te-IN` map to Telugu (template lines 530-531, 601-611).

## X4 - speech-swift metallib instructions; Xcode/XcodeGen guidance; does xcodebuild auto-build default.metallib? - **PARTLY**

- speech-swift:
  - `Makefile:5-14`: `swift build` followed by `./scripts/build_mlx_metallib.sh release|debug`.
  - The script compiles every non-`*_nax` `.metal` file in `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels` into `.build/<cfg>/mlx.metallib`, with `-std=metal3.2 -mmacosx-version-min=15.0 -fno-fast-math` (`scripts/build_mlx_metallib.sh:47-78,114-136`).
  - It also copies the metallib into the test bundles (`:139-149`) and tells you to run `xcodebuild -downloadComponent MetalToolchain`.
  - README:314 says "Swift 6+, Xcode 16+ (with Metal Toolchain)"; README:327 and :553 cover the "Failed to load the default metallib" error.
- Xcode guidance:
  - The only Xcode-project examples are **iOS XcodeGen** specs: `Examples/iOSEchoDemo/project.yml` and `Examples/iOSBenchmark/project.yml`, with `packages: Qwen3Speech: path: ../..`.
  - Neither has a metallib step. Both set an unexplained `OTHER_LDFLAGS: -Wl,-undefined,dynamic_lookup` (added in 2f9eba9, with no rationale in the commit).
  - For macOS, PersonaPlexDemo's README:84-130 shows hand-assembling a `.app` and copying `mlx.metallib` into `Contents/MacOS`.
  - There is no macOS XcodeGen app example.
- mlx-swift:
  - README:85-104 says "SwiftPM (command line) cannot build the Metal shaders ... `xcodebuild` can".
  - In `Package.swift:202-203`: `SWIFTPM_BUNDLE="mlx-swift_Cmlx"` and `METAL_PATH="default.metallib"`.
  - The 9 `.metal` files in `Source/Cmlx/mlx-generated/metal` are not excluded, so Xcode compiles them into `mlx-swift_Cmlx.bundle/default.metallib`.
  - Load order in mlx `backend/metal/device.cpp:137-172`: colocated `mlx.metallib`, then `Resources/mlx.metallib`, then `<bundle>/mlx-swift_Cmlx.bundle/default.metallib` via `mainBundle` or `allBundles` resourceURL, then `Resources/default`, then CWD `default.metallib`.
  - mlx-swift PR #430 (still **open**) adds SwiftPM-CLI metallib support and "skips its command under Xcode, where Metal resources are already built".
- **Caveat:** only those 9 kernels (arg_reduce, conv, gemv, layer_norm, random, rms_norm, rope, sdpa, steel_attention) are ahead-of-time.
  - Cmlx builds in JIT mode (`nojit_kernels.cpp` is excluded at `Package.swift:267`; `tools/update-mlx.sh` uses `-DMLX_METAL_JIT=ON`).
  - Every other kernel is compiled from source at first use (`device.cpp:686-703`, `jit_kernels.cpp`).
- **Answer:** yes, xcodebuild bundles `default.metallib` automatically, provided the Metal Toolchain component is installed. This is supported by the README and the code; I did not run it.
  - For a **CLI tool** target, `mlx-swift_Cmlx.bundle` sits next to the binary in build products and has to ship with it.

## X5 - XcodeGen: latest release; local packages, Swift 6, macOS app + CLI - **CONFIRMED (with a default to override)**

- The latest tag is **2.46.0**: commit "Update to 2.46.0", 2026-07-16. The release asset `releases/download/2.46.0/xcodegen.zip` returns HTTP 200, and Homebrew stable is 2.46.0.
- Local packages: `Docs/ProjectSpec.md:1261-1290` (`packages: X: path:`; traits are new in 2.46.0).
- Product types include `application` and `tool` (`ProjectSpec.md:415,435`).
- `buildToolPlugins` is supported (`:398,774-793`). The default `projectFormat` is `xcode16_0` (`:136`).
- **Watch out:** `SettingPresets/base.yml:50` sets `SWIFT_VERSION: '5.0'`. Swift 6 mode needs an explicit `SWIFT_VERSION: "6.0"` setting (build settings are free-form). The same preset also sets `MTL_FAST_MATH: YES`, which applies only to the project's own `.metal` files.

## X6 - GitHub Actions macOS arm64 images, Xcodes, RAM - **CONFIRMED / corrected details**

From runner-images README:30-36 and `images/macos/*-Readme.md`, image versions 2026-09:

| Label | OS | Xcodes (default marked) | Swift ≥ 6.3? |
|---|---|---|---|
| `macos-15` / `-xlarge` | 15.7.9 | 26.3, 26.2, 26.1.1, 26.0.1, **16.4 (default)**, 16.3-16.0 | **No** (max 26.3 = 6.2.x) |
| `macos-26` = **`macos-latest`** / `-xlarge` | 26.6.2 | **26.6 (default)**, 26.5, 26.4.1, 26.3-26.0.1 | **Yes** |
| `xcode-27` / `-xlarge` (public preview) | macOS 27.0 | 27.0 (27A266a) only | 6.4 |
| `macos-14` (deprecated) | 14.8.9 | 16.2, 16.1, 15.4 (default)... | No |

- The images run `xcodebuild -downloadComponent MetalToolchain` for every Xcode ≥ 26 (`images/macos/scripts/build/Install-Xcode.ps1:38-41`, `scripts/helpers/Xcode.Installer.psm1:108-118`).
- Hardware (docs.github.com, github-hosted-runners):
  - Standard arm64 (`macos-14/15/26/latest`, `xcode-27`): **3 CPU (M1), 7 GB RAM, 14 GB SSD**, no nested virtualization.
  - `-xlarge`: M2, 5 CPU, 8-core GPU, **14 GB RAM**, 14 GB SSD.
- speech-swift's workflows note that the virtualized runners have no usable Neural Engine and that their paravirtual GPU hangs on CoreML stateful loads, so they force `cpuOnly`.

## Extra risks for the plan
1. Pin mlx-swift deliberately.
   - Either `exact: "0.31.4"`, which matches speech-swift CI, drops the Swift 6.3 requirement, and avoids the CudaBuild plugin prompt.
   - Or accept 0.31.6 and add `-skipPackagePluginValidation` (plus `-skipMacroValidation` if MLXHuggingFace is linked) to every `xcodebuild`.
2. CI must use `runs-on: macos-26`, or `macos-latest` pinned to Xcode 26.6. `macos-15` cannot reach Swift 6.3.
3. Local dev Macs need macOS Tahoe 26.2+ for Xcode 26.4-26.6, and 26.6+ for Xcode 27.
4. TranslateGemma via mlx-swift-lm needs two config patches: rope ×8 linear and `<end_of_turn>` as EOS.
5. The speech-swift package is tools 5.10, so its targets compile in Swift 5 mode. The app in Swift 6 mode imports modules that aren't Sendable-audited.
6. speech-swift's own XcodeGen specs use `-undefined dynamic_lookup`. Don't copy that blindly; it can hide missing symbols until runtime.
