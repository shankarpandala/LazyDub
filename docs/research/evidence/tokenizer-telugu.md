# Verification: tokenizer-telugu claim cluster

Verifier run 2026-09-23. I used primary sources only: HF API/tree JSON, the raw tokenizer file, safetensors headers fetched with HTTP range requests, speech-swift source at `c4c2fab`, chatterbox source at `5de7a54`, and Swift stdlib source by tag. I also ran real Swift toolchains on Linux x86_64 (Ubuntu 24.04).

Toolchains, from download.swift.org ubuntu24.04 tarballs:

- **Swift 6.4 (swift-6.4-RELEASE):** the primary toolchain.
- **Swift 5.10.1, 6.0.3 and 6.2.4:** used for cross-checks.

The Python reference used `tokenizers` 0.23.2 in a scratch venv.

All artifacts are under `scratchpad/verify_swift/`:

| File | Purpose |
|---|---|
| `main.swift` | Standalone extraction of the tokenizer, with no MLX dependency |
| `graphemes.swift` | Grapheme cluster counts (T4) |
| `hf_ref.py` | HF tokenizers reference |
| `texts.json` | 8 hand-written test strings |
| `fuzz.json` | 3007 random Telugu/English/punctuation/ZWJ strings |
| `te_strmerges.json` | Telugu tokenizer with merges rewritten as `"a b"` |
| `te_fixA.json` | Fix (a): Telugu tokens moved into `model.vocab` |
| `*_out.txt`, `*fuzz*.txt` | Raw outputs |

---

## T1: Telugu tokenizer JSON structure. Verdict: CONFIRMED (with one precision)

The file is `grapheme_mtl_merged_expanded_v1.json` in `shankarpandala/chatterbox-telugu` at repo sha `d4341468c1738ea669ef096baccecc68cc8ba9f3`. Its tree API oid is `281f4bde…`. My fresh download has the same `git hash-object`, and it is byte-identical to the local `te_tok.json`.

- **Telugu is only in `added_tokens`.** There are 66 Telugu `added_tokens`, and every one is a single codepoint. Their ids run 2454..2519 with no gaps. All have `special:false, normalized:true`.
  - `model.vocab` contains 0 Telugu keys.
  - `model.merges` contains 0 Telugu merges.
- **`[te]` is id 2520.** It is an *added token* only; it is not in `model.vocab`.
- **"Total vocab 2521" needs a precision.** `model.vocab` has **2454** entries (ids 0..2453). It is identical to the `model.vocab` in `ResembleAI/chatterbox` `grapheme_mtl_merged_expanded_v1.json`.
  - The Telugu file adds 67 new added tokens (66 Telugu plus `[te]`).
  - The union of ids is 0..2520 with no gaps, which is 2521 ids.
  - `config.json` says `"vocab_size": 2521`.
  - The safetensors header of `t3_mtl_te.safetensors` has `text_emb.weight F32 [2521,1024]` and `text_head.weight F32 [2521,1024]`.
- **Merges are pairs.** There are 265 merges, all stored as 2-element lists; the first is `["t","h"]`. Joined with a space, they are identical to the base file's 265 `"t h"`-style strings. The file is the base tokenizer plus 67 added tokens, re-saved by a newer `tokenizers` that writes merges as pairs.
- **Things the claim does not mention:**
  - **34 assigned Telugu codepoints are missing and map to [UNK] even in the reference.** These include Telugu digits 1-9 (digit 0, U+0C66, *is* present), U+0C00, U+0C04, U+0C0C, U+0C34, U+0C3C (nukta), U+0C3D (avagraha), U+0C44, U+0C55, U+0C58-5A, U+0C5D, U+0C60-63, and U+0C77-7F.
  - **Id 2513 (U+0C48 ై) can never be produced.** NFKD, which both the reference and speech-swift apply, splits it into U+0C46 U+0C56 (ids 2511 and 2518).

## T2: MTLTokenizer rejects pair-format merges. Verdict: CONFIRMED (ran it in Swift)

- **Source:**
  - `Sources/ChatterboxTTS/MTLTokenizer.swift:78-81` declares `let merges: [String]`.
  - `:95-97` has `guard let tj = try? JSONDecoder().decode(...) else { throw ChatterboxTokenizerError.malformedTokenizerJSON }`.
  - `:104` keys the rank table by the raw string.
  - `:407` looks up `symbols[i] + " " + symbols[i+1]`.
- **Swift 6.4 run on the raw Telugu JSON.** The decoder fails with `DecodingError.typeMismatch: Expected value of type String. Path: model.merges[0]`, which becomes `malformedTokenizerJSON`. Swift 5.10.1, 6.0.3 and 6.2.4 give the same result. The base Resemble JSON, which uses `"a b"` strings, loads fine.
- **Other blockers on the same load path:**
  - `MTLTokenizer.swift:94` hard-codes the file name `tokenizer.json`, but the repo ships `grapheme_mtl_merged_expanded_v1.json`.
  - `:106,169-177` requires `Cangjie5_TC.json`. The Telugu repo does include it.
  - `MTLTokenizer.supportedLanguages` (`:67-71`) lacks `te`, and `ChatterboxTTSModel.swift:410-412` throws `unsupportedLanguage` for it.

## T3: Grapheme-cluster matching turns Telugu into [UNK]. Verdict: CONFIRMED in real Swift (claim is slightly understated)

**Method.** I copied `TokenizerJSON`, `init`, `encode` (generic path), `tokenize` (`:348-380`), `whitespaceSplit` (`:384-391`) and `bpe` (`:395-417`) verbatim into `verify_swift/main.swift`. It has no MLX or NaturalLanguage imports. For the original variant, I rewrote merges as `"a b"` strings so the JSON would load at all.

**Reference.** The reference is `tokenizers.Tokenizer.from_file(te json)`. It applies the same preprocessing as chatterbox `tokenizer.py:268-305`: `lower()`, then NFKD, then prefix `[te]`, then replace `' '` with `[SPACE]`, then `.encode().ids`.

**Swift 6.4, original `MTLTokenizer` logic:**

| text | ids (Swift) | [UNK] | Telugu tokens correct | HF reference |
|---|---|---|---|---|
| `నమస్కారం, మీరు ఎలా ఉన్నారు?` | `2520,2489,2494,1,1,1,1,1,1,7,2,1,1,1,1,2,2464,1,1,2,2461,1,1,1,1,1,1,13` | **18 / 28** | 4 of 22 | `2520,2489,2494,2503,2517,2470,2505,2496,2455,7,2,2494,2507,2496,2508,2,2464,2498,2505,2,2461,2489,2517,2489,2505,2496,2508,13` (0 UNK) |
| `ఈ video లో మనం machine learning గురించి నేర్చుకుందాం.` | `2520,2460,2,35,208,28,2,1,1,2,2494,1,1,2,26,14,71,203,2,64,59,27,52,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,9` | **23 / 45** | 2 of 25 | `2520,2460,2,35,208,28,2,2498,2515,2,2494,2489,2455,2,26,14,71,203,2,64,59,27,52,2,2472,2508,2496,2506,2455,2475,2506,2,2489,2512,2496,2517,2475,2508,2470,2508,2455,2487,2505,2455,9` (0 UNK) |

- **Mechanism.** In `tokenize()`, `Array(s)` splits the string into grapheme clusters. Only a bare consonant or independent vowel that no combining mark follows (న, మ, ఎ, ఉ, ఈ) equals a single-codepoint added token.
  - Every other cluster goes to the buffer. `bpe()` then splits it into scalars, and Telugu scalars are not in `model.vocab`, so each one becomes `unkId` (1).
  - The number of ids is preserved, but the content is destroyed. English and punctuation ids are unaffected.
- **The claim understates the damage.** Because of GB9c, a whole conjunct plus its vowel sign is a single Character, for example `<0C38 0C4D 0C15 0C3E>` for "స్కా". So the failure covers more than consonant+vowel-sign and consonant+virama pairs.
- **Every toolchain agrees.** Swift 5.10.1, 6.0.3 and 6.2.4 give the same ids for all 8 hand-written texts.

### Proposed fixes, measured against the HF reference

| variant | 8 hand-written texts (incl. both required) | 3007-string fuzz |
|---|---|---|
| original (Characters) | 0/8 | 235/3007 |
| (a) Telugu tokens moved to `model.vocab`, same ids | **8/8** | 2210/3007 |
| (b) added-token matching on `unicodeScalars` | **8/8** | 2561/3007 |
| (a)+(b) | 8/8 | 2561/3007 |
| **(b) + `replacingOccurrences(of:" ", with:"[SPACE]", options: .literal)`** | 8/8 | **3007/3007** |
| (b) + scalar-level space replacement | 8/8 | **3007/3007** |
| (a) + scalar-level space replacement | 8/8 | 2210/3007 |

The HF reference gives identical ids whether it loads the original JSON or `te_fixA.json` (3007/3007), so the converted JSON is safe as a Python golden too.

Every fuzz mismatch for (a) and (b) involves a space, or the start of the text, directly followed by a combining mark: a dependent vowel sign, virama, anusvara, length mark, or ZWJ/ZWNJ.

- **(a) fails on added tokens followed by a mark.** `"]"` plus the mark is a single Character, so the `[SPACE]` or `[te]` added token no longer matches. It is then BPE'd letter by letter as `303,295,292,277,279,281,305`.
- **(b) alone fails in the space replacement.** `MTLTokenizer.swift:140` calls `replacingOccurrences(of:" ", with:"[SPACE]")` with no options. That goes through NSString/CF non-literal search, which will not match a space that forms a composed sequence with the following mark. The `[SPACE]` token is silently dropped.
  - I verified this on Linux, in `swift-corelibs-foundation` `NSStringAPI.swift:907-927` and `NSString.swift:1215-1228`.
  - macOS uses the same CF non-literal search semantics, but I did not run it on macOS.

Well-formed Telugu text never contains this pattern. It can appear in ASR, MT or LLM output, or in text with stray marks.

Recommendation: use fix (b) together with a literal or scalar-level space replacement, then golden-test against the HF reference.

On Swift 5.10.1 only, 4 of the 3007 fuzz strings still differ, and all 4 contain U+0C3C (nukta, added in Unicode 14). The cause is NFKD canonical reordering in the older corelibs ICU. It does not matter in practice.

## T4: Swift grapheme segmentation of Telugu conjuncts (GB9c). Verdict: CONFIRMED (Swift 6.4, cross-checked on 5.10.1, 6.0.3 and 6.2.4)

These results are from `swift-6.4-RELEASE` on x86_64 Linux. Swift 5.10.1, 6.0.3 and 6.2.4 produced byte-identical output, and each binary links its own `libswiftCore.so`.

| word | Characters | scalars | clusters |
|---|---|---|---|
| క్షమించండి | **4** | 10 | క్ష · మిం · చం · డి |
| స్త్రీ | **1** | 6 | స్త్రీ |
| అన్నం | **2** | 5 | అ · న్నం |
| ప్రపంచం | **3** | 7 | ప్ర · పం · చం |
| సత్యం | **2** | 5 | స · త్యం |
| నమస్కారం | **4** | 8 | న · మ · స్కా · రం |
| మాట | **2** | 3 | మా · ట |
| వెళ్ళాన్ (word-final virama) | **3** | 8 | వె · ళ్ళా · న్ |
| extra: నేర్చుకుందాం | 4 | 12 | నే · ర్చు · కుం · దాం |
| extra: కార్‌యం (ZWNJ) | 3 | 7 | కా · ర్‌ · యం |
| extra: ర్‍క (ZWJ) | 1 | 4 | ర్‍క |

- **Stdlib source.**
  - Today, GB9c is implemented with InCB properties. The Telugu virama U+0C4D is listed as `InCB=Linker` in `stdlib/public/core/StringGraphemeBreaking.swift:439-454` (main `b57d5b9`), and the rule itself is at `:808-845`.
  - Tags `swift-6.1-RELEASE` and `swift-6.2-RELEASE` use the same InCB code.
  - From `swift-5.7-RELEASE` through `swift-6.0-RELEASE`, an earlier "linking consonant + virama" version was used. Its `_isVirama` includes `0xC4D`.
- **Using this as an akshara counter.** It works for conjuncts and for trailing anusvara or visarga. There are three caveats:
  - A word-final pollu (`న్`) counts as its own cluster, so వెళ్ళాన్ counts 3 against 2 spoken syllables.
  - A ZWNJ half-form counts as its own cluster.
  - Punctuation and spaces are Characters too, so filter them out before counting.
- **macOS caveat.** On macOS, `Character` segmentation comes from the OS-bundled `libswiftCore`, so it depends on the user's macOS version, not on Xcode. Any macOS whose runtime is Swift 5.7 or later should behave like this, but I did not run it on macOS.

---

## Extra findings and risks

1. **speech-swift's own MLX bundle has mismatched weights and tokenizer.**
   - `aufklarer/Chatterbox-Multilingual-MLX-fp16@0a062e0` `model.safetensors` has `t3.text_emb.weight F16 [2352,1024]`, which matches `t3_23lang.safetensors` (2352).
   - Its `tokenizer.json` is byte-identical (blob `d27fb3f`) to the 2454-id `grapheme_mtl_merged_expanded_v1.json`, which pairs with `t3_mtl23ls_v2` (`[2454,1024]`).
   - `config.json` claims `vocab_size 2454`.
   - `T3Config.textTokensDictSize = 2352` is hard-coded (`T3Conditioning.swift:8`), and `ChatterboxTTSModel.swift:201-202` builds `ChatterboxT3()` with the default and `verify: .all`.
   - The Telugu T3 is `[2521,1024]` and is derived from v2, so it will fail the shape check unless the config is parameterised. It also needs its own MLX conversion.
   - Ids 2352..2453 (102 tokens) exceed the bundled embedding.
2. **The model card's Python usage does not run on upstream chatterbox `5de7a54`.**
   - `mtl_tts.py:31-55` `SUPPORTED_LANGUAGES` lacks `te`, and `:293` raises `ValueError`.
   - `:204` builds `T3(T3Config.multilingual())` with 2454 rows (`t3_config.py:39-41`), so the strict `load_state_dict` of the 2521-row checkpoint would fail.
   - Dev-time Python goldens therefore need a patched chatterbox.
3. **The Telugu T3 is F32.** It is 2.14 GB (`t3_mtl_te.safetensors`, 292 tensors), so it needs conversion or quantisation for the app.
4. **Load order in speech-swift.** `ChatterboxTTSModel.fromPretrained` loads the tokenizer (`:167`) *before* the weights. Every tokenizer blocker above (T2 plus the file name) fails the whole model load.
