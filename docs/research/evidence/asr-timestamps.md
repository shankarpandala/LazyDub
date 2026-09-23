# Verification: ASR and timestamp claims (speech-swift)

Checked 2026-09-23. Sources are primary only: the speech-swift clone (soniqo/speech-swift at c4c2fab, 2026-09-20), WhisperKit v1.1.0 cloned into `verify-WhisperKit` (commit 1e2a1637), the upstream facebookresearch/omnilingual-asr cloned into `verify-omnilingual-asr` (81f51e22), the Hugging Face API, tree, raw and safetensors headers, and the upstream Qwen/Qwen3-ForcedAligner-0.6B model card (`verify-qwen3fa-readme.md`).
Paths below are relative to `scratchpad/speech-swift/` unless another location is given.

## Summary table

| ID | Verdict | One-line result |
|---|---|---|
| A1 | confirmed | WhisperASR decodes with no timestamps and suppresses every special token. Acoustic language ID works through `transcribeWithLanguageAsync` and `transcribeWithLanguage`, but it only looks at the first 30 s window. |
| A2 | partly | `Qwen3ASRModel` returns a plain String, and the protocol returns `language == nil`. The model does detect the language, but the code strips it out. The same module's `StreamingASR` does return VAD **segment** start and end times. |
| A3 | partly | The class, the default repo, the 80 ms resolution, the ~270 s figure and `alignLong` all check out. The ~270 s figure is an empirical comment, not a guarantee. `alignLong` can silently drop words. The Core ML aligner has a 30 s limit and no `alignLong`. **Languages: 11 upstream (zh, en, yue, fr, de, it, ja, ko, pt, ru, es). English yes, Hindi NO, Telugu NO.** |
| A4 | partly | WhisperKit (`from: 1.0.0`, currently resolves to v1.1.0, MIT) is used **only** by the `AsrBenchmark` executable. It is not re-exported, so an app gets nothing from it transitively. WhisperKit itself does support `wordTimestamps` and language detection. |
| A5 | partly | Among the listed modules, only NemotronStreamingASR returns word timings, and they are emission-aligned and come only from the streaming API. MOSS returns segment timestamps. None of these modules also reports an acoustically detected language. |
| A6 | partly | `tel_Telu` is in Meta's catalog, but the port is the language-agnostic CTC model, which ignores `language`. Core ML ships only 300M INT8 (10 s default, 5 s alt). MLX ships 300M, 1B, 3B and 7B at 4 or 8 bit (MLX default is 300M 4-bit). There is a hard 40 s cap per call. |

---

## A1. WhisperASR: no word timestamps; detects language. **CONFIRMED**

Evidence:
- The default repo is set at `Sources/WhisperASR/WhisperASR.swift:6`: `defaultModelId = "aufklarer/Whisper-Large-v3-Turbo-CoreML"`.
  - The Hugging Face repo exists: license mit, sha a8e93b20, lastModified 2026-07-01.
  - It is a re-staged argmaxinc/whisperkit-coreml `openai_whisper-large-v3-v20240930_turbo` (its `manifest.json` has `"source_repo": "argmaxinc/whisperkit-coreml"`, `"format": "coreml-whisperkit"`).
- Decoding starts from the no-timestamps token at `Sources/WhisperASR/WhisperCoreMLRuntime.swift:156`: `var nextToken = generationConfig.noTimestampsToken`.
- `WhisperCoreMLRuntime.swift:257` suppresses every token `>= specialTokenBegin`, and `WhisperGenerationConfig.swift:52` sets `specialTokenBegin = eos (50257)`. That range includes all timestamp tokens, so none can be emitted.
- The result struct holds only text and language: `WhisperTranscription { text, language }` (`WhisperCoreMLRuntime.swift:5-8`).
- The docs say so too:
  - `docs/models/whisper-asr.md:41` lists "Word timestamps and timestamp-token decoding" under "Not yet implemented".
  - `docs/inference/whisper-asr-inference.md:110` says "No word timestamps yet."
- Language ID:
  - `WhisperASR.swift:144-158` defines `transcribeWithLanguageAsync(audio:sampleRate:language:) -> TranscriptionResult(text, language, confidence: 0)`.
  - `WhisperASR.swift:171` has the synchronous `transcribeWithLanguage`.
  - `WhisperCoreMLRuntime.swift:113-128` does detection as one decoder step from SOT followed by an argmax over the language tokens. If that fails it falls back to English.

Caveats:
- Language is detected only on the first 30 s chunk and then reused as the hint for every later chunk (`WhisperCoreMLRuntime.swift:63,68-70`).
- The returned value is a Whisper code such as `"en"`, not `"english"`.
- The bundle *can* support word timestamps, but the runtime does not use this:
  - The decoder exports `alignment_heads_weights`: the output schema of `TextDecoder.mlmodelc/metadata.json` is `[logits, key_cache_updates, value_cache_updates, alignment_heads_weights]`.
  - `generation_config.json` has `alignment_heads`.
  - So DTW word timing could be added, or WhisperKit could be run on the same files.

## A2. Qwen3ASR: no timestamps; language returned as nil. **PARTLY**

Evidence:
- `Qwen3ASRModel.transcribe(...) -> String` (`Sources/Qwen3ASR/Qwen3ASR.swift:269-400`).
- The protocol conformance implements only `transcribe` (`Sources/Qwen3ASR/Qwen3ASR+Protocols.swift:5-10`). `transcribeWithLanguage` therefore falls back to the default `TranscriptionResult(text:)`, which has `language: nil` (`Sources/AudioCommon/Protocols.swift:216-219`).
- The same applies to `CoreMLASRModel` (`CoreMLASRModel.swift:359-370`).
- The model itself does auto-detect ("Without a language hint, the model auto-detects and prepends 'language XX'", `Qwen3ASR.swift:453-455`). The decoder throws that away: `Qwen3ASR.swift:516-519` removes everything before `<asr_text>`.

Correction:
- The Qwen3ASR module does have timestamped output. `StreamingASR` yields `TranscriptionSegment { text, startTime, endTime }` built from Silero VAD segments (`StreamingASR.swift:7-21, 60-140`).
- Those times are segment-level, not word-level.
- Word-level timing comes from the separate `Qwen3ForcedAligner` in the same module.

Note: Qwen3-ASR's language list (30 languages) includes Hindi but **not Telugu** (upstream card, `verify-qwen3fa-readme.md:39`).

## A3. Qwen3ForcedAligner. **PARTLY**

Confirmed:
- `public class Qwen3ForcedAligner` (`ForcedAligner.swift:61`) has the default `fromPretrained(modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit")` (`ForcedAligner.swift:451`).
  - The Hugging Face repo exists: sha f0e9f12a, apache-2.0.
  - Its `model.safetensors` is 978,674,048 B, lfs sha256 8187bcb2ab9046cbb274559523d21f60249410ccc561682bc5860b07101c5568.
- The resolution is 80 ms:
  - `Configuration.swift:146-147` sets `classifyNum = 5000` and `timestampSegmentTime = 0.08`.
  - The safetensors header has `thinker.lm_head.weight` with shape [5000, 1024].
- The ~270 s figure comes from a comment in `ForcedAligner.swift:92-98` and from `docs/inference/forced-aligner.md:47,141`.
- `alignLong` is at `ForcedAligner.swift:109-197`.

Corrections:
- The ~270 s figure was "observed on TED-Ed material". The same file says the classify head collapses at **60-220 s** on narrowband or out-of-domain audio (`ForcedAligner.swift:117-119`). Upstream claims "up to 5 minutes" (`verify-qwen3fa-readme.md:22`).
- `alignLong` can return **fewer words than it was given**. It exits with the rest of the words still unaligned in these cases:
  - saturation starts at word 0 (`:159-164`)
  - the remaining audio is shorter than 5 s (`:175`)
  - the split index runs past the word list (`:172,180`)
  - more than 10 passes (`:193`)
- `alignLong` exists only on the MLX aligner. `CoreMLForcedAligner` (default `aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16`) differs:
  - Its Hugging Face `config.json` has `fixed_mel_frames: 3000` (30 s) and `text_fixed_t: 768`.
  - It throws if either limit is exceeded (`CoreMLForcedAligner.swift:203-209, 455-461`).
  - It has no long-audio path.

**Languages:**
- The upstream card lists 11: Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish (`verify-qwen3fa-readme.md:40`).
- The aufklarer card metadata lists 9 (en, zh, ja, ko, de, fr, es, it, ru).
- **English is supported. Hindi and Telugu are not.**
- speech-swift's `language` argument only picks the word-splitting strategy (`TextPreprocessing.swift:98-114`). It never reaches the model prompt: `buildInputIds` ignores `language` (`ForcedAligner.swift:392-433`).
- So passing "Hindi" or "Telugu" runs without error, but it is outside what the model was trained on.

## A4. WhisperKit dependency. **PARTLY (facts)**

- The dependency is declared at `Package.swift:247-248`: `.package(url: "https://github.com/argmaxinc/WhisperKit", from: "1.0.0")`, with the comment "retained for benchmark comparison against the native WhisperASR runtime".
- Only one target uses it: the `AsrBenchmark` **executable** (`Package.swift:783-797`, product `asr-bench` at `:217`). The only import is `Sources/AsrBenchmark/EngineWhisperKit.swift:2`.
- The `WhisperASR` target depends only on `AudioCommon` (`Package.swift:706-711`).
- There is no `@_exported` anywhere in `Sources/`.
- So an app that links speech-swift library products **does not** get WhisperKit symbols or its word-timestamp capability. Using it would need a direct dependency.
- I did not verify whether SwiftPM or Xcode still clones or pins WhisperKit while resolving the graph. Either way, it is not linked.
- WhisperKit tags (from `git ls-remote`):
  - The latest is **v1.1.0** (1e2a1637, 2026-08-06). HEAD is 3111602, ahead of it.
  - `from: 1.0.0` resolves to v1.1.0 today.
- License: **MIT** ("Copyright (c) 2024 argmax, inc.").
- In v1.1.0 the manifest's package name is `argmax-oss-swift`, with products ArgmaxOSS, WhisperKit, TTSKit and SpeakerKit. It requires macOS 13+.
- Word timestamps:
  - `DecodingOptions.wordTimestamps` defaults to false (`Sources/WhisperKit/Core/Configurations.swift:175,204`).
  - It is implemented in `TranscribeTask.swift:197-214`, which needs decoder `alignmentWeights`.
  - The result type is `WordTiming{word,tokens,start,end,probability}` (`Models.swift:622`).
- Language detection:
  - `DecodingOptions.detectLanguage` (`Configurations.swift:172`) defaults to `!usePrefillPrompt`, which is **false** with the default options (`:229`).
  - The standalone APIs are `detectLanguage(audioPath:)` and `detectLangauge(audioArray:)` (the second name has a typo) at `WhisperKit.swift:527,540`.

## A5. Other ASR modules: timestamps and language ID. **PARTLY**

| Module (default repo) | Word or token timestamps | Language ID |
|---|---|---|
| ParakeetASR (`aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s`) | None. The TDT decoder returns only (tokens, logprobs, confidence) (`TDTGreedyDecoder.swift:64,158`). Per-word **confidence** only. | Not acoustic: runs `NLLanguageRecognizer` on the output text and maps unknown languages to "english" (`ParakeetASR+Protocols.swift:16-47`). |
| ParakeetStreamingASR (`aufklarer/Parakeet-EOU-120M-CoreML-INT8`) | None | Text-based NL detection (`+Protocols.swift:17-21`) |
| CanaryASR (`aufklarer/Canary-180M-Flash-CoreML`) | None | Echoes the requested language (en, de, es, fr) (`CanaryASR+Protocols.swift:15-18`, `CanaryASR.swift:163-166`) |
| VoxtralASR | None | Echoes the hint, or nil (`VoxtralASR.swift:63-68`) |
| NemotronStreamingASR (`aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8`) | **Yes: emission-aligned `TimedWord`s at 80 ms frames** (8 × 160 / 16 kHz), cumulative, with a lag caveat (`Protocols.swift:169-184` in AudioCommon, `StreamingSession.swift:160-167`). Available only through `transcribeStream`, `createSession`, `pushAudio` or `StreamingRecognitionUpdate.words`; `transcribeAudio` returns a String (`NemotronStreamingASR.swift:158-183`). | Has an "auto" prompt slot (`languages.json` autoSlot 101) but reports only the requested language, or nil (`+Protocols.swift:16-28`) |
| OmnilingualASR | None (CTC frames are not exposed) | None (ignores `language`) |
| CohereTranscribeASR | None. The prompt always uses `<|notimestamp|>`, because `buildPromptTokens(language:)` is called without `useTimestamps` (`CohereTranscribe.swift:190`, `CohereTranscribeTokenizer.swift:51-66`). | Defaults to "en" and echoes it (`CohereTranscribe.swift:72-79`) |
| MossTranscribe | **Segment-level** start and end times plus speaker labels (`MossTypes.swift:121-140`), not word-level | None (default protocol, so nil) |

Answer: no module in the list gives English word timestamps together with automatic language ID. Two workable combinations:
- WhisperASR (for language ID) plus Qwen3ForcedAligner (for word timings).
- Nemotron streaming (for word timings) plus the separate `SpeechLanguageID` module. That module is SpeechBrain ECAPA VoxLingua107, with `aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-{MLX,CoreML}` (`SpeechLanguageIdentifier.swift:15-17`).

## A6. Omnilingual ASR and Telugu. **PARTLY**

- Telugu coverage:
  - Upstream `src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py:1412` lists `"tel_Telu"`.
  - speech-swift's `docs/models/omnilingual-asr.md:1777-1782` lists it too.
  - But speech-swift ports only the **CTC** variant, which is language-agnostic and ignores `language` (`OmnilingualASR.swift:19-21`; `MLX/OmnilingualMLXModel.swift:15-17`). You cannot condition it on `tel_Telu`.
- Core ML model (`OmnilingualASRModel`):
  - The default is `aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s` (`OmnilingualASR.swift:30`).
  - The alternative is the 5 s window `aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8` (`:34`).
  - The Core ML build exists only at 300M: `...-1B-CoreML-INT8-10s` returns HTTP 401.
- MLX model (`OmnilingualASRMLXModel`):
  - Published at `aufklarer/Omnilingual-ASR-CTC-{300M,1B,3B,7B}-MLX-{4bit,8bit}`, and all 8 repos exist (apache-2.0) (`OmnilingualMLXConfig.swift:14-17,106-108`).
  - The default is 300M 4-bit (`OmnilingualMLXModel.swift:57-58`).
- There is a hard cap of 40 s per call, and it throws above that (`OmnilingualASR.swift:43,194`).
- No timestamps and no language ID.
- The upstream Telugu CER of 6.5 applies to the **7B LLM-ASR** model (`per_language_results_table_7B_llm_asr.csv:1413`), not to the 300M CTC.

---

## Extra findings and risks

1. **Whisper repeated-word guard truncates silently.** With `maxRepeatedWordRun = 2` (`WhisperCoreMLRuntime.swift:24,177-181,273-292`), decoding of a 30 s window stops as soon as the same word appears 3 times in a row. The check lower-cases and strips punctuation, so "no, no, no" triggers it. The rest of that window is lost without any error.
2. **Whisper's chunking is crude.** Windows are fixed at 30 s with no overlap (`:59-72`), so words get cut at the boundaries. There is also no temperature fallback.
3. **`alignLong` word-index mismatch.** `plateauStart` indexes the preprocessed words, where punctuation-only tokens are merged and Han characters are split. The remaining text, however, is re-split on `" "` (`ForcedAligner.swift:179-181`). Text containing standalone tokens such as "-" or "—" will re-align from the wrong word. Normalize the text first, and check the word count of the output.
4. **The aligner stamps the first word near 0 s when there is leading non-speech** such as a music intro (`docs/inference/forced-aligner.md:162`). This is common on YouTube, so trim with VAD first.
5. **Qwen3ASR auto-enables a no-repeat 3-gram filter on inputs longer than 15 s** (`Qwen3ASR.swift:48,86-94`), which can suppress legitimate repetition. The default `maxTokens` of 448 truncates long inputs.
6. **Nemotron supports `te-IN` (slot 40) and `hi`.**
   - Its license is `openmdw-1.1` according to the Hugging Face card, so it needs checking.
   - `transcribeStream` swallows errors: it calls `continuation.finish()` without throwing (`NemotronStreamingASR.swift:141`).
7. **Package requirements:** speech-swift uses `swift-tools-version: 5.10` and requires macOS 15.0 or later (`Package.swift:1,6-8`).
8. **Language-code formats differ between modules.** Whisper returns "en", while Parakeet returns "english", which is the format the `TranscriptionResult` doc describes. Normalize these codes in the app.
