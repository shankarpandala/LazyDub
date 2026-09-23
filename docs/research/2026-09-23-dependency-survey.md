# Dependency survey, 2026-09-23

This survey checks the spec's September 2026 notes against current primary sources, before any dependency is pinned. The load-bearing claims were checked by a second, adversarial pass against source code at the commit named, or against Hugging Face API/tree data; that evidence, with file:line citations, is in `evidence/`. A few supporting details come only from the first-pass survey. `evidence/first-pass-citations.md` lists them with the citations a later review spot-checked. macOS-only behaviour is marked *unverified on macOS*.

**Scope and caveats**

- Nothing here was built or run on a Mac, and there are no performance numbers in this document.
- Sizes are file sizes from the Hub. Memory figures are estimates, flagged as such. Real numbers come from the Phase 0 spikes on the M5 Pro.
- Swift behaviour (tokenizer, grapheme clusters) was run on Linux Swift 5.10.1 / 6.0.3 / 6.2.4 / 6.4. It still has to be confirmed on macOS.

## Snapshot of versions seen

| Dependency | Latest seen | Notes |
|---|---|---|
| speech-swift | `c4c2fab` (2026-09-20, v0.0.27 + 23 commits), Apache-2.0 | swift-tools 5.10, **Swift 5 language mode**, macOS 15 / iOS 18. README says to depend on `branch: main`; we must pin a commit. |
| mlx-swift | 0.31.6 (2026-07-01) | 0.31.5+ need Swift 6.3 (tools 6.3) and attach a CudaBuild build-tool plugin to Cmlx. 0.31.1–0.31.4 are tools 5.12. |
| mlx-swift-lm | 3.31.4 (2026-06-29) | speech-swift pins it `exact: 3.31.4`, which pins mlx-swift `upToNextMinor(0.31.4)`. Has Gemma 3, Gemma 4, Qwen3. |
| YouTubeKit | 0.4.9 / `e5b7d03` (2026-08-18), MIT | swift-tools 5.9 (Swift 5 mode), no SwiftPM deps. Bundles yt-dlp's EJS solver (Unlicense), meriyah (ISC), astring (MIT). |
| yt-dlp | 2026.08.19 | `yt-dlp_macos`: universal2 PyInstaller onefile (~35 MB), ad-hoc signed, not notarized. SHA2-256SUMS + GPG `.sig`. |
| Deno | 2.9.7 | `deno-aarch64-apple-darwin.zip` + `.sha256sum`. yt-dlp needs Deno ≥ 2.3.0. |
| XcodeGen | 2.46.0 (2026-07-16) | Base preset defaults `SWIFT_VERSION` to 5.0; set 6.0 explicitly. |
| WhisperKit | v1.1.0, MIT | **Not named in the spec.** Only speech-swift's benchmark executable uses it; apps don't get it transitively. |
| Xcode | 26.6 (Swift 6.3.3, 2026-06-25) is the newest 26.x. **Xcode 27.0 (Swift 6.4) went GA on 2026-09-14.** | The spec says "Xcode 27 beta". Xcode 26.4+ requires macOS Tahoe 26.2+. |

## 1. Chatterbox Telugu in speech-swift (S4)

**Verdict:** speech-swift can't load `shankarpandala/chatterbox-telugu` as shipped. The fixes are small and well understood, and none needs Python or a new port.

### What the Telugu repo contains

The repo is at sha `d4341468`, CC-BY-4.0, not gated.

- **T3:** `t3_mtl_te.safetensors`, F32, 2.14 GB, 292 tensors (536,126,464 params).
  - Its keys and shapes are identical to upstream `t3_mtl23ls_v2`, except `text_emb` and `text_head`, which are `[2521,1024]` against v2's `[2454,1024]`.
  - Base-language rows were also retrained: row 0 differs from v2 by 2.6e-2.
- **Unchanged Resemble files:** `s3gen.pt`, `ve.pt`, `conds.pt` and `Cangjie5_TC.json` are byte-identical to `ResembleAI/chatterbox` (same LFS sha256).
- **Tokenizer:** `grapheme_mtl_merged_expanded_v1.json`, with `config.json` `{vocab_size: 2521}`.
  - `model.vocab` is the base's 2454 entries.
  - Telugu is 66 single-codepoint *added tokens* (ids 2454–2519), and `[te]` = 2520.
  - Merges are stored as `["a","b"]` pairs; newer `tokenizers` versions write them that way.

### What speech-swift expects

speech-swift was checked at `c4c2fab`.

- **Vocab size is hard-coded.** `T3Config.textTokensDictSize = 2352` (`T3Conditioning.swift:8`). The loader builds `ChatterboxT3()` with that default and loads with `verify: .all`, which includes a shape check (`ChatterboxTTSModel.swift:201-202`). `config.json` is never parsed.
- **Language gate.** `MTLTokenizer.supportedLanguages` is an immutable set of 23 codes without `te`, and `clone()` throws `unsupportedLanguage` (`ChatterboxTTSModel.swift:409-412`). `encode(_:languageId:)` itself has no gate.
- **Tokenizer loading.** The tokenizer loads a file literally named `tokenizer.json`, decodes merges as `[String]` (pair-format merges throw `malformedTokenizerJSON`), and requires `Cangjie5_TC.json`; a `[]` placeholder is enough.
- **The Telugu tokenizer bug.** Added tokens are matched per Swift `Character`, which is a grapheme cluster (`MTLTokenizer.swift:348-377`). Under Unicode 15.1 GB9c, a whole conjunct plus its vowel sign is one `Character`, so almost no Telugu cluster equals a single-codepoint added token, and each scalar falls through to `[UNK]`. Examples in real Swift:
  - `నమస్కారం, మీరు ఎలా ఉన్నారు?` gives 18 of 28 ids as `[UNK]`.
  - `ఈ video లో మనం machine learning గురించి నేర్చుకుందాం.` gives 23 of 45.
  - The HF `tokenizers` reference gives 0 `[UNK]` for both.
- **The verified fix.**
  - Match added tokens on `unicodeScalars`, and replace spaces with `options: .literal`, or at the scalar level.
  - The first alone misses cases where a space or `]` is followed by a combining mark. The two together match the HF reference on 8/8 hand-written strings and **3007/3007 fuzz strings** on Swift 6.0.3, 6.2.4 and 6.4.
  - On 5.10.1, 4 fuzz strings containing the rare nukta (U+0C3C) differ, because of older ICU NFKD ordering.
  - Moving the Telugu tokens into `model.vocab` is *not* enough on its own: 2210/3007.
- **Weights are upcast to fp32.** The loader converts every tensor to fp32 on load (`ChatterboxTTSModel.swift:180`). Runtime memory is therefore about the fp32 size whatever the storage dtype. Estimate for Chatterbox with the Telugu T3, from the header parameter counts: ~3.2 GB of weights (T3 2.14 + S3Gen 0.42 + S3 tokenizer 0.49, already F32 + conformer 0.14), before KV cache and activations.
- **No watermark.** A grep for `perth` or `watermark` finds no watermark code. Upstream Python applies PerTh (`mtl_tts.py:175,354`). The Telugu model card's "every output carries PerTh" holds only for Python.
- **Missing controls.**
  - No speaking-rate, duration or streaming controls.
  - The knobs are `exaggeration, cfgWeight, temperature, topP, minP, repetitionPenalty, maxNewTokens (1000 ≈ 40 s, truncates silently), memoryOptions`.
  - Sampling uses unseeded `Float.random`, so outputs are not reproducible.
  - `clone()` is synchronous and can't be cancelled.
- **Voice conditioning isn't cached.** It is recomputed on every `clone()`.
  - All the building blocks are public (`s3gen.embedRef`, `voiceEncoder.embed`, `s3gen.tokenizer.encode`, `t3.inference`, `s3gen.synthesize`), so it can be cached in memory.
  - `ChatterboxS3GenRef` has no public init, so it **can't be persisted to disk** without a patch.
- **Swift 6 friction.** `ChatterboxTTSModel` is a non-Sendable class, so it needs an actor wrapper and serialized MLX work.

### Conversion needed

- **Only T3 needs converting:** `dst = "t3." + (src.hasPrefix("tfmr.") ? "tfmr.model." + rest : src)`, plus an optional fp16 cast.
  - T3 is Linear, Embedding and Llama only, so there are no transposes.
  - The source is already safetensors, so a **Swift** CLI using mlx-swift's safetensors I/O can do it, with no pickle reading. Because the loader upcasts anyway, the rename could even happen in memory at load time.
- **Everything else comes from `aufklarer/Chatterbox-Multilingual-MLX-fp16`** (`0a062e0`, MIT, 1.85 GB): VE, S3Gen, `conformer.safetensors` and `s3_tokenizer.safetensors`.
  - Its S3Gen is the v2 `s3gen`, which matches the Telugu repo.
  - Its own T3 is converted from the older `t3_23lang` (2352 rows) but ships the 2454-id tokenizer. `€` and `♭` index past the embedding. That doesn't affect a Telugu bundle.
- **Avoid a second download.** Pass the bundle's own `s3_tokenizer.safetensors` to the `localDir` loader; it is byte-identical to `mlx-community/S3TokenizerV2` and saves a 495 MB second download.
- **Complete offline Telugu bundle:**
  - `model.safetensors` (VE + S3Gen from the bundle, plus the renamed Telugu T3);
  - `conformer.safetensors`;
  - `s3_tokenizer.safetensors`;
  - the fixed `tokenizer.json`;
  - `Cangjie5_TC.json`;
  - `config.json`.

### Python goldens

The model card's Python usage doesn't run on upstream chatterbox `5de7a54`: it lacks `te` and has a hard-coded 2454 vocab. Golden reference audio would need the maintainer's patched environment. Golden *token ids* only need the HF `tokenizers` library.

## 2. ASR, word timestamps, language ID (S2)

**Verdict:** the spec's default, speech-swift's `WhisperASR`, gives **no word timestamps**, and no speech-swift module gives both English word timestamps and acoustic language ID. S2 has to compare timing strategies, not just ASR models.

- **WhisperASR** (Core ML, `aufklarer/Whisper-Large-v3-Turbo-CoreML`, 1.63 GB):
  - **Timestamps:** it returns text only. The bundle's decoder exports `alignment_heads_weights` and `generation_config.json` lists `alignment_heads`, so DTW word timing is possible but not implemented.
  - **Language ID:** detected once from the first 30 s window, falling back to English on failure.
  - **Silent text loss:** decoding of a 30 s window stops when a word repeats three times (`maxRepeatedWordRun = 2`; "no, no, no" triggers it), and windows are fixed 30 s with no overlap.
- **Qwen3ASR** (MLX/Core ML): no word timestamps, and the detected language is discarded. Its upstream 30-language list includes Hindi but **not Telugu**. For inputs over 15 s it auto-enables a no-repeat 3-gram filter.
- **Qwen3ForcedAligner** (`aufklarer/Qwen3-ForcedAligner-0.6B-4bit`):
  - **Output:** word timestamps at 80 ms resolution.
  - **Languages:** English yes, **Hindi no**, Telugu no. The `language` argument only picks word splitting.
  - **Length limits:** reliable to ~270 s per call on TED-Ed, but collapses at 60–220 s on out-of-domain audio. The Core ML version is limited to 30 s / 768 prompt tokens.
  - **Silent failures:** `alignLong` can silently drop trailing words. A music intro pulls the first word toward 0 s, so trim with VAD first.
- **NemotronStreamingASR** (Core ML INT8): emission-aligned word timings (80 ms frames), streaming API only. It supports `en-US`, `hi-IN` and `te-IN`. License `openmdw-1.1` still needs review. `transcribeStream` swallows errors.
- **SpeechLanguageID:** SpeechBrain ECAPA VoxLingua107 (MLX/Core ML), a standalone acoustic language ID.
- **WhisperKit** (MIT): has `wordTimestamps` and language detection, and can read the same Whisper bundle layout. **Needs approval** as a new dependency.
- **Omnilingual ASR** (Telugu round-trip CER judge):
  - Telugu was among its training languages. The CTC model ignores `language`, and the language-conditioned LLM variant isn't ported.
  - The upstream Telugu CER of 6.5 is for the 7B LLM-ASR, not the 300M CTC default.
  - Input over 40 s must be segmented first.
  - The judge itself must be calibrated on real Telugu speech (e.g. FLEURS te) before its CER is trusted.
- **Diarization**, all inside speech-swift's SpeechVAD product:
  - Sortformer Core ML with streaming `push`/`finish` and stable IDs;
  - Community-1 Core ML + VBx, with a 256-d centroid per speaker;
  - Pyannote segmentation + WeSpeaker (MLX);
  - `WeSpeakerModel.embed`, 256-d. MLX and Core ML embeddings aren't interchangeable.
- **VAD:** Silero v6.2.1 (MLX/Core ML), Pyannote, FireRedVAD.
- **Other available modules:** Sidon speech restoration (Core ML, 48 kHz) and HTDemucs-FT separation (MLX, 320 MB). FluidAudio adds nothing we need.

## 3. Translation (S3)

- **TranslateGemma:**
  - **Gating and mirrors:** `google/translategemma-{4b,12b,27b}-it` are gated (manual approval). The mlx-community conversions are ungated: `translategemma-4b-it-4bit` (2.22 GB), `-8bit` (4.16 GB), `-12b-it-4bit` (6.66 GB), `-8bit` (12.54 GB), `-27b-it-4bit` (15.23 GB).
  - **Terms:** the Gemma Terms of Use and Prohibited Use Policy still apply, so show them and require acceptance.
  - **Telugu:** `te` and `te-IN` are in the chat template, and WMT24++ includes te_IN.
  - **Template:** the user turn is a one-item list `{type: "text", source_lang_code, target_lang_code, text}`.
  - **Context:** input is ~2K tokens.
- **mlx-swift-lm 3.31.4** should load these conversions as `gemma3` / `gemma3_text`. The configs and key layout match on reading, but this hasn't been run. There are two fidelity gaps to fix before measuring quality:
  1. **RoPE scaling is ignored.** TranslateGemma's config puts RoPE scaling in `rope_parameters.full_attention = {rope_type: linear, factor: 8}` with `rope_scaling: null`, and `Gemma3Text` reads only `rope_scaling`. Global-attention layers therefore run **without** the ×8 linear scaling. Python mlx-lm has the same gap. Fix by injecting `rope_scaling`.
  2. **No EOS token id.** TranslateGemma's configs set no `eos_token_id`, and it isn't in mlx-swift-lm's registry. mlx-swift-lm already stops on the tokenizer's `<eos>`, so `<end_of_turn>` must be added via `extraEOSTokens`.
- **Template mismatch:** TranslateGemma's strict template does plain translation. The §6.4 request needs context, glossary, a target length and JSON output, which doesn't fit the template. Expect a split: TranslateGemma translates, and an instruct model condenses, emits JSON and follows the length target. S3 measures whether one model can do both.
- **Instruct candidates:**
  - speech-swift `Qwen3Chat` has Qwen3.5 0.8B, Qwen3 4B and a hand-written Gemma 4 port (`Gemma4Chat`).
  - mlx-swift-lm has Gemma 4 and Qwen3.
  - There is no TranslateGemma or Gemma 3 support in speech-swift itself.
- **MADLAD-400 3B:** `aufklarer/MADLAD400-3B-MT-MLX` (`e442b3cd`, Apache-2.0, ungated), 1.84 GB int4 / 3.31 GB int8. API `translate(_:to:)`.

## 4. Streams (S1)

**Verdict:** both resolvers now depend on essentially **one** YouTube client that needs no PO token: visionOS. A single YouTube change could break both at once. YouTubeKit was broken from about 2026-03-07 to 03-16; whether yt-dlp broke over the same window wasn't checked.

- **YouTubeKit** (0.4.9):
  - **Local by default:** `methods` default to `[.local]` on macOS. With `[.local]`, no code path contacts a non-YouTube host.
  - **The `.remote` method is dangerous:** it opens a WebSocket to `remote-production.youtubekit.dev`, sends the video ID and app ID, and lets the server make the client perform **arbitrary HTTP requests** (server-chosen URL, method, headers and body) and return full responses. Always pass `methods: [.local]` explicitly.
  - **Clients:** `[.visionOS, .web]` plus conditional fallbacks. `.web` streams need a GVS PO token (which yt-dlp drops without one) and YouTubeKit has no PO-token support, so `.web`-only itags can 403. Callers can't tell which client a `Stream` came from, so probe each URL with a range GET.
  - **Coverage gaps:** "made for kids" videos aren't available with the visionOS client. SABR-only and unknown itags are dropped silently. HLS comes only through `livestreams`.
  - **Missing public fields:** `contentLength`, real width/height/fps and expiry aren't public; parse `clen`, `dur` and `expire` from the googlevideo URL query. Streams can be matched by itag via `stream(withITag:)`.
  - **Concurrency:** it isn't Sendable, and has unsynchronised static and instance caches. Use one instance per extraction inside an actor, and serialise extractions.
  - **Signature solving** evaluates meriyah plus the multi-MB player JS in a fresh JavaScriptCore VM, about twice per video.
  - **URL parsing:** its video-ID regex is loose and `YouTube(url:)` silently uses `""` on failure, so parse IDs ourselves.
- **yt-dlp + Deno:**
  - **Solver scripts:** `yt-dlp_macos` bundles the EJS solver scripts. Remote component download is off by default and blocked by `--no-remote-components`.
  - **JavaScript runtime:** Deno is the only runtime enabled by default and is **not bundled**. Without it, yt-dlp falls back to a deprecated visionOS-only mode.
  - **Flags:** `--no-remote-components --ignore-config --no-plugin-dirs --no-js-runtimes --js-runtimes deno:<path> --no-cache-dir` (or `--cache-dir` inside our container), and never `-U`.
  - **Binary format:** `yt-dlp_macos` is a PyInstaller-frozen **Python interpreter** that extracts `Python.framework` to `$TMPDIR`.
  - **Packaging** (*unverified on macOS*: S1 checks this):
    - Downloading it at runtime (not embedding it) avoids re-signing it inside our notarized bundle.
    - An ad-hoc-signed arm64 binary without a quarantine attribute runs; a quarantined one is blocked.
    - A non-sandboxed app doesn't quarantine its URLSession downloads unless it opts in.
    - App Sandbox is effectively ruled out on this path, because child processes inherit the sandbox.
- **Other YouTube behaviour:**
  - Googlevideo URLs expire after ~6 h and are IP-bound. yt-dlp fetches in 10 MiB ranges.
  - This container gets YouTube's bot wall. All S1 work needs a residential IP, which means the Mac.

## 5. Toolchain

- **Xcode choice.** "Stable Xcode 26" is ambiguous now that Xcode 27.0 is GA.
  - speech-swift's CI builds only with Xcode 16.4 / Swift 6.1, against mlx-swift 0.31.4. mlx-swift's self-hosted CI uses an unknown Xcode ("Xcode-latest"). So no upstream CI is *known* to cover speech-swift with Xcode 26.x or 27.
  - Xcode 26.4–26.6 resolve mlx-swift 0.31.6 (Swift 6.3). That needs `-skipPackagePluginValidation` in CI and triggers a "Trust & Enable" prompt for the CudaBuild plugin.
  - Xcode 26.0–26.3 fall back to mlx-swift 0.31.4.
- **Recommendation:** pin **mlx-swift exact 0.31.4** in the app. It's the mlx-swift version speech-swift's CI (Xcode 16.4) resolves. It avoids mlx-swift 0.31.6's Swift 6.3 requirement and the plugin prompt. Xcode 26.4+ still needs Tahoe 26.2 whatever mlx-swift version is pinned. It satisfies both speech-swift (`from: 0.30.0`) and mlx-swift-lm (`upToNextMinor 0.31.4`). Commit `Package.resolved`.
- **Metal kernels:**
  - Under `xcodebuild`, `default.metallib` lands in `mlx-swift_Cmlx.bundle` automatically; the Metal Toolchain component must be installed.
  - Only 9 kernels are precompiled; the rest are JIT-compiled from source on first use. This affects the cold-start target.
  - A CLI target doesn't embed SwiftPM resource bundles, so `maata-bench` must ship `mlx-swift_Cmlx.bundle` beside its binary or live inside the `.app`.
- **Don't copy `-Wl,-undefined,dynamic_lookup`** from speech-swift's iOS XcodeGen examples: it turns link errors into runtime crashes.
- **GitHub hosted runners:**
  - `macos-26` offers Xcode 26.0.1–26.6 (default 26.6); `macos-15` tops out at 26.3.
  - Standard arm64 runners have 7 GB RAM, 3 vCPU and 14 GB disk, and their virtualised GPU/ANE can't run stateful Core ML models.
  - So hosted CI does build and unit tests only. Model tests need a self-hosted Apple Silicon Mac.
- **speech-swift dependency graph:** it resolves its whole graph: swift-transformers, Hummingbird, MCP SDK, swift-nio, swift-syntax, WhisperKit, and the SpeechCore.xcframework binary. Only the products we import get built and linked.

## 6. Offline guarantees in speech-swift

- **No global switch.** Offline mode is per call (`offlineMode: true`), and there is no global switch or `HF_HUB_OFFLINE`.
- **Online mode risks:**
  - It re-downloads files whose size changed on `main`, with **no revision pinning**.
  - With no network, it retries for [5, 15, 30, 60] s before using the cache, which is a cold-start hazard.
  - **Rule for the app:** only local-directory loaders or `offlineMode: true` with our `cacheDir`. The Model Manager does every download, from pinned revisions.
- **Per-model offline loading:**
  - **Chatterbox:** `fromPretrained(localDir:s3TokenizerWeights:conformerWeights:)` makes no network calls. Its async path ignores `cacheDir` for the S3 tokenizer.
  - **Community-1 and MADLAD:** `fromLocal(directory:)`.
  - **Whisper:** only via `fromPretrained(cacheDir:offlineMode:)`, because its init is private.
  - **Qwen3ASR and ForcedAligner:** these pick architecture and bit width from the **modelId string**, so local folder names must carry `0.6B`, `4bit` and similar markers, or the wrong config loads silently.
  - **Indic-Mio:** voice cloning downloads WavLM from the Hub, unpinned, unless `INDIC_MIO_WAVLM_BUNDLE` points to a local bundle (`IndicMioTTSModel.swift:234-241`).

## 7. Licenses: what the manifest must say

Hugging Face labels are wrong in several places. The manifest must carry **verified** license strings, not copied tags.

| Model | Show | Constraint |
|---|---|---|
| chatterbox-telugu T3 + tokenizer | CC-BY-4.0, plus attribution to the author, FLEURS and IndicVoices-R | Attribution. We become a redistributor with CC-BY duties if we re-host a converted bundle. |
| Resemble VE / S3Gen / conds; aufklarer Chatterbox bundle | MIT | Notice |
| S3TokenizerV2 (mlx-community mirror, unlabelled) | Apache-2.0 (upstream CosyVoice2) | Notice |
| TranslateGemma (mlx-community) | Gemma Terms of Use + Prohibited Use Policy | Require acceptance. The mirrors ship no Gemma Notice file. |
| MADLAD-400 3B | Apache-2.0 | none |
| Whisper large-v3 turbo (Core ML) | MIT | none |
| Sortformer (aufklarer, from v2.1) | **NVIDIA Open Model License** + NVIDIA notice (the conversion's CC-BY-4.0 label is incomplete) | Notice. v2 (not 2.1) is CC-BY-4.0; v1 is CC-BY-NC, so avoid v1. |
| Pyannote Community-1 (aufklarer mirror) | CC-BY-4.0 | Attribution |
| WeSpeaker | **CC-BY-4.0** (labelled MIT) | Attribution |
| **OmniVoice** | **CC-BY-NC** (weights, per the upstream card since 2026-07-03) + **Boson Higgs Audio 2 community license** (codec) | **Non-commercial.** "Built with Higgs Materials…" display, no use of outputs to train other LLMs, and an extra licence above 100,000 annual active users. The speech-swift README's "Apache-2.0" is stale. |
| **Indic-Mio** | Apache-2.0 label | Training data includes `ylacombe/expresso` (**CC-BY-NC-4.0**) and possibly Emilia (NC). Its WavLM dependency may be **CC-BY-SA-3.0** (UniSpeech LICENSE), not MIT. |
| Omnilingual ASR | Apache-2.0 | none |
| Nemotron streaming ASR | openmdw-1.1 | Needs review |
| SpeechLanguageID (`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-{MLX,CoreML}`) | **not yet checked** | Check the SpeechBrain model's and VoxLingua107's data licences before S2 uses it |
| YouTubeKit + bundled JS | MIT, Unlicense (ejs), ISC (meriyah), MIT (astring) | Include notices |
