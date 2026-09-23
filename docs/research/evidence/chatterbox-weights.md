# Verify: chatterbox-weights cluster (W1–W6)

Checked 2026-09-23 against primary sources only: HF API JSON, safetensors headers fetched with HTTP range requests, small tensor slices fetched by range (values compared), torch `.pt` zip archives read by range (pickle parsed with a stub unpickler, no torch), speech-swift source at `c4c2fab` (2026-09-20), mlx-swift `9019419`, resemble-ai/chatterbox `5de7a54`.
Working files: `scratchpad/vcw/` (`*_tree.json`, `*_hdr`/`*.json` headers, `hdr.py`, `cmp*.py`, `vals*.py`, `ptrange.py`, `swtok/`).

| Claim | Verdict |
|---|---|
| W1 Telugu repo contents / license / not gated | **partly** (the numbers are right; the file list is incomplete; the license is mixed) |
| W2 T3 header matches v2 except text_emb/text_head | **confirmed** (and values show it was fine-tuned from v2) |
| W3 s3gen.pt / ve.pt / conds.pt byte-identical | **confirmed** |
| W4 default bundle has ve/s3gen, so only T3 needs converting | **partly**: the key mapping is right, but speech-swift cannot load a 2521-vocab T3 or "te" text without code changes. The bundle T3 is `t3_23lang`, not v2 |
| W5 S3 tokenizer fetched separately and downloaded twice | **confirmed** |
| W6 pure-Swift converter, no pickles needed | **confirmed** (with caveats) |

---

## W1: `shankarpandala/chatterbox-telugu`: PARTLY

Evidence: `https://huggingface.co/api/models/shankarpandala/chatterbox-telugu` and `/tree/main`
- sha `d4341468c1738ea669ef096baccecc68cc8ba9f3`, lastModified 2026-06-06T03:00:44Z, `gated:false`, `private:false`, `cardData.license: cc-by-4.0`. The repo has two commits: `c7b9c7c` initial and `d434146` "Add Chatterbox Telugu (te) fine-tune".
- `t3_mtl_te.safetensors`: 2,144,538,616 B, lfs sha256 `9f2065c5…07da9`. The header has 292 tensors, all `F32`, and **536,126,464 params**. There is no `__metadata__`.
- `s3gen.pt` 1,057,165,844 B · `ve.pt` 5,698,626 B · `conds.pt` 107,374 B.
- Correction: the "tokenizer JSON" is **`grapheme_mtl_merged_expanded_v1.json`** (89,649 B). It is a BPE file whose ids run to 2520: the base `model.vocab` has 2454 entries and there are 185 `added_tokens`, 67 of them new. The 66 Telugu codepoints are ids 2454–2519 and `[te]` is id 2520. The repo also contains `Cangjie5_TC.json` (git oid identical to Resemble's), `config.json` (`base_t3: t3_mtl23ls_v2.safetensors`, `vocab_size: 2521`) and `README.md`.
- License nuance: the card is CC-BY-4.0 (attribution is required, both for the model and for FLEURS / IndicVoices-R). The README says `s3gen.pt`, `ve.pt`, `conds.pt` and `Cangjie5_TC.json` stay **MIT** (© Resemble AI).

## W2: T3 header vs `ResembleAI/chatterbox` `t3_mtl23ls_v2.safetensors`: CONFIRMED

The two headers were fetched by range (both are 32,752 B) and compared with `vcw/cmp.py`:
- The key sets are identical: 292 keys on each side, and none is unique to either file.
- Every dtype is F32. The only shape differences are `text_emb.weight` and `text_head.weight`, at [2521,1024] vs [2454,1024].
- Values: `speech_emb`, `speech_head`, `cond_enc.*`, `*_pos_emb`, all RMSNorm weights and `tfmr.norm` are **bit-identical to v2** (max diff 0.0). The tfmr Linear weights differ by about 1e-2, consistent with a merged LoRA. v3 differs everywhere, so the fine-tune really is from v2 (`vcw/vals4.py`).
- Even base rows of `text_emb` changed (row 0 diff 2.6e-2), so the "English and 23-language ability preserved" claim is unverified.

## W3: s3gen.pt / ve.pt / conds.pt byte-identical: CONFIRMED

The LFS sha256 values from both `/tree/main` listings match:
- `s3gen.pt` `9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3`
- `ve.pt` `4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1`
- `conds.pt` `6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e`. I also downloaded this 107 KB file and its sha256sum matches.
- Note: Resemble also publishes `s3gen_v3.pt` and `s3gen_v3.safetensors`, whose HiFi-GAN (`mel2wav.*`) weights differ. The Telugu repo ships the **v2** `s3gen.pt`.

## W4: default speech-swift bundle, and "only T3 needs converting": PARTLY

**Bundle identity.** `ChatterboxTTSModel.swift:132` sets `defaultModelId = "aufklarer/Chatterbox-Multilingual-MLX-fp16"`. The HF bundle is at sha `0a062e0f`, license MIT. Its files are `model.safetensors` (1,286,967,446 B), `conformer.safetensors` (68 MB), `s3_tokenizer.safetensors` (494,868,984 B), `tokenizer.json`, `Cangjie5_TC.json`, `config.json` and `s3_tokenizer_config.json`.

**The bundle already has converted ve and s3gen: TRUE.**
- The `model.safetensors` header has 2297 F16 tensors: 13 `ve.*`, 292 `t3.*` and 1992 `s3gen.*`.
- `ve.*` and `s3gen.*` are restructured, so they are not a 1:1 mapping of the source. Examples:
  - `ve.lstm.layers.N.{Wx,Wh,bias}`, where `bias = bias_ih + bias_hh` (verified numerically).
  - `flow.decoder.estimator.down_blocks_0.resnet.block1.conv.conv.*`.
  - Conv kernels are transposed to [O,K,I].
- Numerically, the bundle's `s3gen.mel2wav.*` matches `s3gen.pt` (v2) and not `s3gen_v3`. For example, `mel2wav.conv_pre.bias` differs from v2 by 5e-5 and from v3 by 1.0. The bundle's `ve.*` matches `ve.pt` to fp16 precision. This was checked by range-reading the `.pt` zips (`vcw/ptrange.py`, `probe_pt2.py`, `probe_ve.py`).
- So the bundle's acoustic stack is the same stack the Telugu repo redistributes.

**Correction: the bundle's T3 is NOT `t3_mtl23ls_v2`.**
- `t3.text_emb.weight` and `t3.text_head.weight` are **[2352,1024]**. Values match **`t3_23lang.safetensors`** to fp16 rounding (≤3e-5 on rows 0 and 1 of eight tensors) and differ from v2 by about 1e-2 (`vcw/vals.py`).
- speech-swift hardcodes the same size: `T3Conditioning.swift:8` sets `textTokensDictSize = 2352`.
- The bundle's `config.json` claims `vocab_size: 2454`, and `tokenizer.json` is byte-identical to Resemble's `grapheme_mtl_merged_expanded_v1.json`: 2454 ids, same git oid `d27fb3f2`. That is a pre-existing mismatch between tokenizer and model. After NFKD, only `€` and `♭` can produce ids ≥ 2352, which would index out of range in MLX.

**Exact T3 key transform.** Checked against the bundle header with `vcw/cmp_bundle.py`: 0 missing keys, 0 extra keys, and 0 shape differences apart from the vocab rows.
```
dst = "t3." + (src.hasPrefix("tfmr.") ? "tfmr.model." + src.dropFirst(5) : src)
value = value.asType(.float16)        // optional, see W6
```
- No transpose or reshape is needed for any T3 tensor. Rows 0 and 1 of square Linear weights (`q_proj`, `perceiver.attn.to_q`), `pre_attention_query` [1,32,1024], `speech_pos_emb`, `speech_head`, `down_proj` and `spkr_enc` [1024,256] all match the source in its native layout.
- `tfmr.embed_tokens.weight` [8,1024] is kept as a vestigial stub (`T3.swift:155-163`).
- The loader strips `t3.` (`ChatterboxTTSModel.swift:183`). Module keys are `tfmr.model.*` (`T3Model.swift:5-21`).
- `emotion_adv_fc` [1024,1], the conv-free perceiver and `speech_emb`/`speech_head` [8194,1024] map as-is.

**The plan's claim that "only T3 needs converting" is wrong in practice.** speech-swift at `c4c2fab` blocks a Telugu T3 in three places:
1. `ChatterboxTTSModel.swift:201-202` calls `ChatterboxT3()` with the default `T3Config` (2352) and then `update(..., verify: .all)`. The `.all` option includes `.shapeMismatch` (mlx-swift `Module.swift:395-397, 476`). A [2521,1024] `text_emb` therefore throws `mismatchedSize`, and so would v2's 2454.
2. `ChatterboxTTSModel.swift:410` checks `MTLTokenizer.supportedLanguages`. That set, at `MTLTokenizer.swift:67-71`, has no `"te"`, so `clone(... languageId:"te")` throws `unsupportedLanguage`.
3. `MTLTokenizer` cannot read the Telugu tokenizer JSON. I tested this by compiling the real `MTLTokenizer.swift` with Swift 6.2.4 on Linux, with only the Chinese and Japanese helpers stubbed (`vcw/swtok/`):
   - **a.** `MTLTokenizer.swift:80` declares `merges: [String]`. The Telugu file stores merges as `[["t","h"],…]` (the newer tokenizers format), so loading throws `malformedTokenizerJSON`.
   - **b.** After converting the merges, added tokens are still matched on Swift `Character`, which is a grapheme cluster (`:349`, `:365`). The Telugu letters exist only as single-codepoint `added_tokens`, not in `model.vocab`. As a result, every consonant followed by a vowel sign or virama becomes `[UNK]`=1. For example, "నమస్కారం, ఈ రోజు…" encodes to `[2520,2489,2494,1,1,1,1,1,1,7,…]`, while HF tokenizers gives `[2520,2489,2494,2503,2517,2470,2505,2496,2455,7,…]`.
   - **Tested fix:** move the 66 Telugu single-codepoint added tokens into `model.vocab` and write merges as `"a b"` strings. Swift output then matches Python exactly on 3 sentences, including NFKD `ై` (U+0C48) → U+0C46 + U+0C56.
   - Patching Swift to match added tokens on unicode scalars would also work.

To load the Telugu model you must either fork or patch speech-swift, or write your own loader. For the second route, the public API is enough to reimplement `fromPretrained(localDir:)` and `clone()`:
- `T3Config` has public vars, and `ChatterboxT3(_:)`, `ChatterboxVoiceEncoder()`, `CAMPPlus`, `S3GenVocoder`, `S3GenConformer`, `MatchaCFM`, `S3TokenizerV2`, `ChatterboxS3Gen.init`, `embedRef`, `synthesize` and `t3.inference` are all public.
- The private helpers `conformerBlockWeights` and `s3TokenizerWeights` need to be copied, and `dropInvalidTokens` is internal.

## W5: S3 tokenizer and offline file set: CONFIRMED

- The tokenizer is fetched separately. `ChatterboxTTSModel.swift:134` sets `s3TokenizerModelId = "mlx-community/S3TokenizerV2"`, and `:269` puts it in cache dir `chatterbox-s3-tokenizer`. `:293-298` downloads it when `model.safetensors` is missing there, and `:304` loads it from that path.
- **It is downloaded twice.** The bundle download (`:286-291`) passes no `.safetensors` in `additionalFiles`, so `HuggingFaceDownloader.downloadWeights` adds the `*.safetensors` glob (`HuggingFaceDownloader.swift:157-163`). That glob also pulls the bundle's own `s3_tokenizer.safetensors`, which the loader never opens.
- The two copies are the same file: sha256 `928726bc1f206a613d36b8f49e297eae9c5593a21bf9b92ddfe2c23f85eb92cc`, 494,868,984 B each. A first run therefore downloads about 990 MB for this one component. The two copies land in `~/Library/Caches/qwen3-speech/models/aufklarer/…` and `~/Library/Caches/chatterbox-s3-tokenizer/models/mlx-community/S3TokenizerV2/`. The `QWEN3_CACHE_DIR` variable overrides the root (`:739-752`).
- Files the loader actually opens in `fromPretrained(localDir:s3TokenizerWeights:conformerWeights:)`:
  1. `<bundle>/model.safetensors`, via `MLX.loadArrays` (`:171`)
  2. `<bundle>/conformer.safetensors`, if present (`:163`). If it is missing, the conformer blocks stay at random init under a relaxed verify (`:223-231`), so treat it as required.
  3. `<bundle>/tokenizer.json` (`MTLTokenizer.swift:94`)
  4. `<bundle>/Cangjie5_TC.json` (`MTLTokenizer.swift:173`). It is required even for non-Chinese text, because a missing file throws `missingCangjieMap`.
  5. The S3 tokenizer safetensors at any path (`:320`). Accepted inputs are the bundle's `s3_tokenizer.safetensors`, the mlx-community `model.safetensors`, or Resemble's `s3gen.safetensors`, whose `tokenizer.*` keys are converted in place.
- `config.json` is never read; its existence is only checked on the async path (`:272-282`). `s3_tokenizer_config.json` is never read.
- A complete offline Telugu bundle therefore needs `model.safetensors` (with the Telugu T3), `conformer.safetensors`, the fixed Telugu `tokenizer.json`, `Cangjie5_TC.json`, an S3 tokenizer safetensors, and `config.json` if you use the async API. Pass the bundle's own `s3_tokenizer.safetensors` as `s3TokenizerWeights` to avoid the second download.

## W6: pure-Swift conversion, and whether pickles are needed: CONFIRMED (with caveats)

- mlx-swift provides `loadArrays(url:)`, which reads safetensors (`Source/MLX/IO.swift:125`), and `save(arrays:metadata:url:)`, which writes safetensors (`:61-85`). The Telugu T3 is plain F32 safetensors, so a converter needs only a rename, `asType(.float16)` and a save. No torch is involved.
- **No pickle is needed:**
  - `ve.*` and `s3gen.*` come from the MLX bundle, which I verified numerically equal to `ve.pt` and `s3gen.pt`.
  - `conds.pt` holds only the built-in default voice: t3 `speaker_emb` [1,256], `cond_prompt_speech_tokens` [1,150], `emotion_adv`; gen `prompt_token` [1,157], `prompt_feat` [1,314,80], `embedding` [1,192]. The Swift `clone()` always takes reference audio and never reads conds.
- Caveat: the loader upcasts everything to fp32 (`ChatterboxTTSModel.swift:180`). An fp16 T3 therefore saves disk space (about 1.07 GB instead of 2.14 GB) but not RAM, since T3 still takes about 2.14 GB at runtime. For the same reason you could skip the offline converter and rename the keys in memory at load time.
