# Checking the chatterbox-loader claims

**Sources checked:** speech-swift `c4c2fab` (2026-09-20); upstream chatterbox `5de7a54`; mlx-swift HEAD and tag 0.31.4; mlx-swift-lm tag 3.31.4; swift-transformers 1.3.4; SwiftPM `swift-6.1.2-RELEASE`; the Hugging Face API, with safetensors headers read through HTTP range requests.

**Runtime test:** I compiled speech-swift's `MTLTokenizer.swift` with Swift 6.1.2 on Linux. I made only two changes: I removed `import NaturalLanguage` and stubbed out the two Chinese and Japanese helper functions, which no other language uses. The harness is in `verify-swift/tokharness`, and the test data is in `verify-swift/fixtures`.

## Verdict table

| Claim | Verdict | What changes |
|---|---|---|
| C1 | confirmed, with a note | Loading does fail. But `T3Config` and `ChatterboxT3.init(_:)` are public, so an app can build a T3 with 2521 rows without forking. The Telugu checkpoint also uses different key names. |
| C2 | confirmed, with a note | Only `clone()` checks the language. `MTLTokenizer.encode` and `encodeStrict` accept any `languageId`. |
| C3 | confirmed | Tested at runtime. A file containing just `[]` satisfies the check. |
| C4 | confirmed | The only matches are dictionary words in the MagpieTTS pronunciation files. |
| C5 | confirmed | The model also has no seed control, and sampling uses `Float.random`. |
| C6 | confirmed, with a note | Conditioning can be cached in memory. `ChatterboxS3GenRef` has no public init, so it cannot be saved to disk and rebuilt. |
| C7 | partly right | The S3 tokenizer part is right. Offline loading from local folders works for most models, but nothing guarantees zero network use: IndicMio's WavLM encoder always goes online, and online mode checks the Hub API on every load. |
| C8 | confirmed | Weights are [2352,1024] and tokenizer.json has 2454 entries. The T3 matches the older v1 checkpoint, but the tokenizer is the newer v2 one. |
| C9 | confirmed, with a note | Every package dependency is resolved and fetched, plus the SpeechCore binary. Only mlx-swift and swift-transformers are compiled. |

## C1: text vocabulary size is hard-coded, and loading is strict

- `Sources/ChatterboxTTS/T3Conditioning.swift:7-8` declares `public struct T3Config { public var textTokensDictSize = 2352`.
- `ChatterboxTTSModel.swift:201-202` runs `let t3 = ChatterboxT3()` and then `t3.update(..., verify: .all)`.
- In mlx-swift `Source/MLXNN/Module.swift:397`, `.all` has rawValue -1, which turns on every check. Lines 476-480 throw `mismatchedSize` when shapes differ.
- `config.json` is only named in the download lists (`ChatterboxTTSModel.swift:274`, `:289`). It is never parsed.

**Missed:** the Telugu checkpoint also uses different key names.
- `shankarpandala/chatterbox-telugu/t3_mtl_te.safetensors` has 292 F32 tensors. The keys have no `t3.` prefix and use `tfmr.layers.*`, `tfmr.embed_tokens` and `tfmr.norm`.
- speech-swift expects `tfmr.model.*` (`T3Model.swift:6-8`, `T3.swift:157-159`).
- The key set is the same as upstream `t3_mtl23ls_v2`. The only shape differences are `text_emb` and `text_head`, which are [2521,1024] here and [2454,1024] in v2.

**The claim understates one thing: no fork is needed.** Everything required is public:
- `T3Config`
- `ChatterboxT3.init(_ cfg:)` (`T3Model.swift:25`)
- `ChatterboxTTSModel.init(tokenizer:voiceEncoder:t3:s3gen:)` (`:117`)
- every component initializer and `loadWeights`

So an app can build `T3Config` with `textTokensDictSize = 2521`, rename `tfmr.X` to `tfmr.model.X`, and load with `verify: .all`.

## C2: supported languages and the check in `clone()`

- `MTLTokenizer.swift:67-71` declares `public static let supportedLanguages` with 23 codes and no `te`. The runtime printed `count = 23, contains te: false`.
- `ChatterboxTTSModel.swift:409-412` throws `unsupportedLanguage`.
- However, `MTLTokenizer.encode` and `encodeStrict` (`:121`, `:147`) do not check the language. The public pieces `t3.inference` (`T3Model.swift:69`) and `s3gen.synthesize` (`ChatterboxS3Gen.swift:197`) let an app skip `clone()` entirely.

## C3: Cangjie5_TC.json is required

- `MTLTokenizer.swift:106` calls `loadCangjieMap` for every language.
- `:173-176` throws `missingCangjieMap`.
- At runtime, the folder without the file threw `missingCangjieMap`. A file containing `[]` loaded correctly.

## C4: no PerTh or watermark code

A case-insensitive grep over the whole repo found only `Sources/MagpieTTS/Resources/cmudict_ipa_{en,de}.txt` (dictionary words). By contrast, upstream `mtl_tts.py:175,354` does apply the watermark. The Telugu model card also says "every output carries … PerTh", which will not be true for Swift output.

## C5: no speed control and no streaming

- `clone()` (`ChatterboxTTSModel.swift:394-407`) takes these settings: `exaggeration` 0.5, `maxNewTokens` 1000, `temperature` 0.8, `topP` 1.0, `minP` 0.05, `repetitionPenalty` 1.2, `cfgWeight` 0.5, and `memoryOptions`.
- It returns the whole `[Float]` at once. There is no speed, rate, duration or streaming option.
- `MatchaCFM.solve(nTimesteps:temperature:)` is public, but `flowMel` never passes those values through.
- T3 sampling uses `Float.random` with no seed (`T3Model.swift:174`). The S3Gen noise uses a fixed seed of 0 (`MatchaCFM.swift:579`).

## C6: voice conditioning is recomputed on every call

`ChatterboxTTSModel.swift:414-439` recomputes the conditioning each time. These parts are public, so an app can cache it:

| Part | Where |
|---|---|
| `tokenizer`, `voiceEncoder`, `t3`, `s3gen` properties | `ChatterboxTTSModel.swift:94-97` |
| `embedRef` | `ChatterboxS3Gen.swift:119` |
| `embed(samples:)` | `VoiceEncoder.swift:88` |
| `S3TokenizerV2.encode` | `S3TokenizerV2.swift:333` |
| `s3gen.tokenizer` | `ChatterboxS3Gen.swift:42` |

`ChatterboxS3GenRef` (`ChatterboxS3Gen.swift:28-36`) has public `let` fields but no public init. It can be cached in memory but cannot be saved to disk and rebuilt.

## C7: cache directory and offline loading

**The S3 tokenizer ignores `cacheDir`.** `ChatterboxTTSModel.swift:268-269` calls `getCacheDirectory(for: s3TokenizerModelId, cacheDirName: "chatterbox-s3-tokenizer")` without `basePath`. `QWEN3_CACHE_DIR` moves it, and so does the legacy `QWEN3_ASR_CACHE_DIR` (`HuggingFaceDownloader.swift:739-752`).

Two ways around it:
- The synchronous `fromPretrained(localDir:s3TokenizerWeights:conformerWeights:)` (`:149-254`) never touches the network.
- The bundle's own `s3_tokenizer.safetensors` is identical to the separate `mlx-community/S3TokenizerV2` file (same LFS sha256 `928726bc…`, 494,868,984 bytes). The download step's `*.safetensors` glob fetches it and then never uses it, so 495 MB is downloaded twice.

**Offline mode works per call, but there is no global switch.**
- With `offlineMode: true`, all three download functions return before touching the network (`HuggingFaceDownloader.swift:148-155`, `248-255`, `336-346`).
- `HF_HUB_OFFLINE` is not supported.
- In online mode, `downloadWeights` fetches the `/api/models/<id>/tree/main` listing on every load (`HuggingFaceRepoManifest.swift:84-105`). It re-downloads any file whose size has changed, because `main` is not pinned.
- When offline, it retries after 5, 15, 30 and 60 seconds (`:503`) before falling back to the cache. That is at least 110 s.

**Loading from a local folder, model by model:**

| Model | How to load it | Network use |
|---|---|---|
| Chatterbox | `fromPretrained(localDir:...)` | none |
| Whisper | `fromPretrained(cacheDir:offlineMode:)`. The init is private. | none when the bundle is complete (`WhisperASR.swift:64,90`). Otherwise a direct `URLSession` fetch of the tokenizer (`:182-227`), which obeys `offlineMode`. |
| Qwen3ASR | `fromPretrained(modelId:cacheDir:offlineMode:true)` | Size and bit width come from the modelId string, not the config (`Qwen3ASR.swift:1194-1215`, `1298-1299`), so the string must still be passed. |
| ForcedAligner | same as Qwen3ASR | Variant comes from `quantize_config.json` or the modelId (`ForcedAligner.swift:18-27`, `474-483`). |
| Community-1 | `fromLocal(directory:)` | none |
| MADLAD | `fromLocal(directory:)` | none. swift-transformers `from(modelFolder:)` is local only. |
| Sortformer, Silero, WeSpeaker, Omnilingual | `cacheDir` plus `offlineMode: true` | none |
| IndicMio | `fromBundle` to load | Voice cloning (`extractGlobalEmbedding`) calls `IndicMioWavLMFeatureModel.fromPretrained()` with default settings (`IndicMioTTSModel.swift:229-243`). It goes online unless the `INDIC_MIO_WAVLM_BUNDLE` environment variable is set. |

## C8: the default bundle's shapes don't match its tokenizer

- The default model is `aufklarer/Chatterbox-Multilingual-MLX-fp16` (`ChatterboxTTSModel.swift:132`), at sha `0a062e0f`.
- The safetensors header gives `t3.text_emb.weight` and `t3.text_head.weight` as F16 [2352,1024].
- `tokenizer.json` has 2454 vocabulary entries (ids 0-2453) and 118 added tokens.
- `config.json` says `vocab_size: 2454`, which is also wrong.

Where the pieces come from:
- The T3 shape matches ResembleAI's `t3_23lang.safetensors` (v1).
- The tokenizer is byte-identical (git blob `d27fb3f2`) to ResembleAI's `grapheme_mtl_merged_expanded_v1.json`, the 2454-entry v2/v3 tokenizer.

What breaks: after NFKD normalization, `€` and `♭` still map to ids 2352 and 2430. The runtime returned id 2352 for "5 €". That id is past the end of the embedding table.

## C9: Package.swift and what gets built

- `// swift-tools-version: 5.10` (line 1).
- No `swiftSettings` and no `swiftLanguageVersions`.
- Platforms are macOS 15.0 and iOS 18.0 (lines 6-9).
- Package dependencies are on lines 230-248:
  - mlx-swift `from: 0.30.0`
  - mlx-swift-lm `exact: 3.31.4`, which pins mlx-swift to `.upToNextMinor(0.31.4)` and swift-syntax to 602..<604
  - swift-argument-parser
  - MCP swift-sdk `exact 0.12.1`
  - swift-system
  - swift-transformers `from 1.1.6`
  - Hummingbird 2.5..<2.17 and hummingbird-websocket
  - swift-websocket 1.5..<1.6
  - WhisperKit `from 1.0.0`

**What an app with the four products gets:**

- **Resolved and fetched: everything.** SwiftPM 6.1.2 is built without `ENABLE_TARGET_BASED_DEPENDENCY_RESOLUTION` (`Manifest.swift:181-221`). For a dependency package, that code keeps every package any of its products needs, not just the products you use. Each of the ten dependencies is reachable from some speech-swift product. For example, WhisperKit comes in through `asr-bench`, Hummingbird through `speech-server`, and MCP through `speech`.
- **The SpeechCore binary is downloaded too.** `Workspace+BinaryArtifacts.swift` `parseArtifacts` collects every binary target from every dependency manifest, so `CSpeechCore` (speech-core v0.0.14 xcframework) is downloaded even though it is unused.
- **Built and linked:**
  - the speech-swift targets AudioCommon, MLXCommon, ChatterboxTTS, SpeechVAD, WhisperASR and Qwen3ASR
  - from mlx-swift: MLX, MLXNN, MLXFast, MLXFFT, and the Cmlx C++ library
  - from swift-transformers: Hub and Tokenizers, which bring in swift-jinja, swift-huggingface (with EventSource and swift-crypto), swift-collections, swift-crypto and yyjson
- **Not built:** mlx-swift-lm, swift-syntax, Hummingbird and NIO, MCP, WhisperKit, ArgumentParser, swift-system.

## Other findings the claims missed

1. **This blocks Telugu. The Chatterbox tokenizer in speech-swift cannot tokenize the Telugu fine-tune.**
   - The Telugu `grapheme_mtl_merged_expanded_v1.json` stores merges as pairs (`[["t","h"],…]`). `MTLTokenizer` decodes them as `[String]` (`MTLTokenizer.swift:77-89`), so loading throws `malformedTokenizerJSON`. This was seen at runtime.
   - The Telugu letters are *added tokens* (ids 2454-2520, `[te]` = 2520), not part of the main vocabulary. `tokenize()` matches added tokens by Swift `Character` (a whole grapheme cluster, `:348-377`).
   - Swift treats "స్కా" as one Character. So after I converted the merges, "నమస్కారం, మీరు ఎలా ఉన్నారు?" gave 18 of 28 tokens as `[UNK]`. The HF `tokenizers` library gives 0 `[UNK]` for the same text.
   - The app needs its own tokenizer that matches on Unicode scalars.
2. The Telugu fine-tune keeps the base acoustic stack unchanged. Its `s3gen.pt` and `ve.pt` have the same sha256 as `ResembleAI/chatterbox` (`9b9ff07e…`, `4b16d836…`), so their weights can be reused. The T3 is F32, 2.1 GB, and has to be converted.
3. The loader converts the whole bundle to fp32 (`ChatterboxTTSModel.swift:180`). Memory use is about twice the fp16 file size.
4. `ChatterboxTTSModel` is not `Sendable`, and `clone()` is synchronous and cannot be cancelled. A Swift 6 app needs an actor or `@unchecked` wrapper around it.
5. `maxNewTokens` of 1000 is about 40 s of speech. Longer lines are cut off silently.
6. Downloads always track `main` with no revision pinning, so upstream changes to a model repo reach users without notice.
