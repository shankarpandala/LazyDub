# License verification: LazyDub model manifest (cluster "licenses")

Verified 2026-09-23 from primary sources only: the HF API (`/api/models/<repo>`, `/tree/main`, `/commits/main`, raw files at pinned revisions), upstream LICENSE files, and the source trees of speech-swift (HEAD c4c2fab), k2-fsa/OmniVoice (08be0b4), wenet-e2e/wespeaker (9fecd6c), facebookresearch/omnilingual-asr (81f51e2), microsoft/UniSpeech (6112826) and microsoft/unilm (31c5b90). The NVIDIA Open Model License and Gemma Terms of Use pages were fetched from their official URLs. Working files are in `scratchpad/vl/`.

## Verdicts

| ID | Verdict | Summary |
|---|---|---|
| L1 | **partly** | Weights are CC-BY-NC and the codec is under the Boson Higgs Audio 2 Community License: both confirmed. Telugu is supported. The "CC-BY-NC" text exists only on the HF card (added 2026-07-03). The GitHub repo says Apache-2.0 for code and nothing about weights, and the HF metadata now carries no license field. |
| L2 | **partly** | Apache-2.0, not gated, Telugu listed, and expresso CC-BY-NC-4.0: all confirmed. The ~1.24 GB figure is the upstream repo (1,241,227,372 B). The bundle speech-swift actually downloads is 1.50 GB, plus 0.38 GB of WavLM for voice cloning. |
| L3 | **confirmed** (with nuances) | The google repos are gated (API `gated: "manual"`, although the prompt says "Requests are processed immediately"). The 6 mlx-community 4/8-bit repos plus 27b-bf16 are ungated. TranslateGemma is in the Gemma ToU Appendix. te/te-IN are in the template. The card states a 2K-token input context. |
| L4 | **confirmed** | chatterbox-telugu is CC-BY-4.0 and not gated. ResembleAI/chatterbox is MIT. aufklarer/Chatterbox-Multilingual-MLX-fp16 is labelled MIT. Its companion mlx-community/S3TokenizerV2 has **no** license label. |
| L5 | **partly** | Only **v2.1** is under the NVIDIA OML; **v2 is CC-BY-4.0**. The speech-swift Sortformer conversion (base v2.1) is labelled CC-BY-4.0 but ships no copy of the OML and no Notice file, so the relabel is incomplete. community-1: the mirror is ungated and CC-BY-4.0 with a LICENSE file (legit). WeSpeaker conversions are labelled **MIT**, but the upstream weights are **CC-BY-4.0**, so the label is wrong. |
| L6 | **confirmed** | MADLAD MLX: Apache-2.0, ungated, `<2te>` = token id 139. All 10 aufklarer Omnilingual conversions: Apache-2.0, ungated. Upstream facebook models are Apache-2.0. `tel_Telu` is in lang_ids. |

## Evidence by claim

### L1 OmniVoice
- `aufklarer/OmniVoice-MLX-fp16`: HF API tags `license:apache-2.0`, `gated: False`, sha bec00bc, created 2026-06-24. `aufklarer/OmniVoice-MLX-int8` is also `apache-2.0` and ungated.
- `k2-fsa/OmniVoice` (sha c5fdb5c) `README.md:781-783`: "Our code is released under the Apache 2.0 License. The pre-trained model is licensed under the CC-BY-NC due to constraints from its training data (e.g., Emilia)." No CC version is given. The current `cardData` has **no `license` key**.
- History: the card at revision `999c332499` (2026-05-07) had `license: apache-2.0` in its YAML (line 651) and no CC-BY-NC text. Commit `c5fdb5ccb1` (2026-07-03) removed that YAML line and added the section above. The aufklarer conversion (2026-06-24) therefore copied the label as it stood then; that label is now stale.
- GitHub `k2-fsa/OmniVoice` @08be0b4: `LICENSE` is Apache-2.0, `pyproject.toml:10` is `license = "Apache-2.0"`, and the README says nothing about the weights' license.
- Codec: `k2-fsa/OmniVoice/audio_tokenizer/LICENSE:1` is "BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT". `audio_tokenizer/model.safetensors` sha256 `fe7c5e87…` is **byte-identical** to `bosonai/higgs-audio-v2-tokenizer/model.safetensors`, which is labelled `license:other`. The aufklarer bundles re-host this codec (`audio_tokenizer/model.safetensors`, 402,864,450 B in fp16) **without the LICENSE file**.
- Boson terms that bind users of the app:
  - §1.b.i: "prominently display 'Built with Higgs Materials licensed from Boson AI USA, Inc. … Meta Llama 3 …'" for "a product or service that uses any of them".
  - §1.b.iv: the Llama 3 Acceptable Use Policy is incorporated.
  - §1.b.v: outputs must not be used to improve other LLMs.
  - §2: an expanded license is needed above 100,000 annual active users.
  - Line 16: the license is accepted "by using".
- Telugu is supported: HF language tag `te`; `docs/languages.md:561` "| 552 | Telugu | te | tel | 230.21 |" (hours of training data); `omnivoice/utils/lang_map.py:581`.
- speech-swift `README.md:67` still advertises OmniVoice as "Apache-2.0", which is stale.

### L2 Indic-Mio
- `SPRINGLab/Indic-Mio`: `license:apache-2.0`, `gated: False`, sha 25feace. Datasets are `ai4bharat/Rasa` (CC-BY-4.0, gated auto), `mythicinfinity/libritts_r` (CC-BY-4.0) and `ylacombe/expresso` (**`license:cc-by-nc-4.0`**, verified through `/api/datasets`). Telugu is in `language`. `README.md:146` also names IndicTTS, Syspin and SPICOR, which are not in the YAML and whose licenses I did not verify.
- Size: repo total 1,241,227,372 B, of which `model.safetensors` is 1,217,825,224 B.
- `aufklarer/Indic-Mio-MLX-fp16`: `apache-2.0`, ungated, 1,500,252,547 B. It bundles MioCodec (MIT) and its `soniqo_manifest.json` says `"license_posture": "commercial-safe"`, which ignores the expresso NC data. Its language list is 14 languages, `te` included. Its validation was Hindi only. Sample rate is 24 kHz, while the upstream card says "44kHz".
- Upstream `Aratako/MioCodec-25Hz-24kHz` is MIT and lists `amphion/Emilia-Dataset`; its table shows it used Emilia-**YODAS** (CC-BY-4.0). `Aratako/MioTTS-0.6B` is Apache-2.0 and also lists Emilia without saying which subset.
- Voice-cloning companion `aufklarer/WavLM-Base-Plus-MLX-fp16` is labelled MIT ("following … microsoft/unilm"). The card of upstream `microsoft/wavlm-base-plus` (no license tag) says "The official license can be found [here](…/UniSpeech/blob/main/LICENSE)". That UniSpeech `LICENSE:1` is "**Attribution-ShareAlike 3.0 Unported**", while unilm `LICENSE` is MIT. The license is therefore **ambiguous: CC-BY-SA-3.0 or MIT**.

### L3 TranslateGemma
- `google/translategemma-{4b,12b,27b}-it`: `license:gemma`, API `gated: "manual"`. The gate prompt reads "…review and agree to Google's usage license… Requests are processed immediately." Unauthenticated `resolve/main/config.json` returns **401**. `google/gemma-3-4b-it` shows the same `manual` value.
- Ungated MLX conversions (all `license:gemma`, `gated: False`, unauthenticated config.json returns 200, made with mlx-lm 0.29.1, text-only `gemma3`):

  | Repo | Size |
  |---|---|
  | `mlx-community/translategemma-4b-it-4bit` | 2.22 GB |
  | `mlx-community/translategemma-4b-it-8bit` | 4.16 GB |
  | `mlx-community/translategemma-12b-it-4bit` | 6.66 GB |
  | `mlx-community/translategemma-12b-it-8bit` | 12.54 GB |
  | `mlx-community/translategemma-27b-it-4bit` | 15.23 GB |
  | `mlx-community/translategemma-27b-it-8bit` | 28.74 GB |
  | `mlx-community/translategemma-27b-it-bf16` | – |

  Their READMEs copy the `extra_gated_*` YAML, but it has no effect. They ship no Gemma "Notice" file.
- Gemma ToU (ai.google.dev/gemma/terms, last modified April 1, 2026): the Appendix lists "TranslateGemma". §3.1 requires a "Notice" file reading "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms", plus the §3.2 use restrictions (Prohibited Use Policy), on redistribution.
- Card: "Total input context of 2K tokens". WMT24++ (55 langs) includes `en-te_IN` (dataset `google/wmt24pp` config). The card prose never names Telugu; `te`/`te-IN` appear only in the template (`chat_template.jinja:530-531`, 581 codes).
- Template structure (minimal): the user message `content` must be a list with exactly one item `{type: "text"|"image", source_lang_code, target_lang_code, text}`. Codes are normalised with `_` → `-` and looked up in `languages[...]` to build "You are a professional {src} ({src_code}) to {tgt} ({tgt_code}) translator…". An unknown code raises an error.

### L4 Chatterbox
- `shankarpandala/chatterbox-telugu`: `license:cc-by-4.0`, `gated: False` (config.json returns 200 unauthenticated), sha d434146. Per its card, the T3 is trained on FLEURS (CC-BY-4.0) and IndicVoices-R (CC-BY-4.0, HF gated auto). `s3gen.pt`, `ve.pt` and `conds.pt` are redistributed from Resemble under MIT.
- `ResembleAI/chatterbox`: `license:mit`. GitHub `LICENSE` reads "MIT License, Copyright (c) 2025 Resemble AI".
- `aufklarer/Chatterbox-Multilingual-MLX-fp16`: `license:mit`, ungated. Its runtime companion `mlx-community/S3TokenizerV2` has **no license tag**; upstream `FunAudioLLM/CosyVoice2-0.5B` is Apache-2.0.

### L5 Diarization
- `nvidia/diar_streaming_sortformer_4spk-v2`: **`license:cc-by-4.0`** (`README.md:541-543`).
- `nvidia/diar_streaming_sortformer_4spk-v2.1`: `license:other`, `nvidia-open-model-license` (`README.md:597-599`). This was unchanged by today's (2026-09-23) README edits.
- `nvidia/diar_sortformer_4spk-v1` is CC-BY-NC-4.0 and should be avoided.
- speech-swift `Sources/SpeechVAD/SortformerDiarizer.swift:26` uses `aufklarer/Sortformer-Diarization-CoreML`. That repo has `license:cc-by-4.0`, `base_model: nvidia/diar_streaming_sortformer_4spk-v2.1`, and README text: "Upstream license: NVIDIA Open Model License (weights); this conversion is published under CC-BY-4.0 with attribution to NVIDIA." Its files are models, configs and README only, with no LICENSE or NOTICE.
- NVIDIA OML (Last Modified Oct 24, 2025), Redistribution: you "must give any other recipients of the Model a copy of this Agreement and include … 'Licensed by NVIDIA Corporation under the NVIDIA Open Model License'". You "may provide additional or different license terms … for any such Derivative Models as a whole, provided Your use… of the Model otherwise complies". So relabelling is permitted in principle, but these conditions are not met. The OML also has a Guardrail-termination clause, and NVIDIA may update the terms.
- `aufklarer/Ultra-Sortformer-Diarization-CoreML` (`SortformerDiarizer.swift:31`) is labelled apache-2.0. It is built on `devsy0117/ultra_diar_streaming_sortformer_8spk_v1` (apache-2.0 label, base v2.1 under OML), which was trained on the Korean AI Hub corpus. Its license chain is the same problem, only worse.
- `pyannote/speaker-diarization-community-1`: `license:cc-by-4.0`, `gated: "auto"`. The gate collects "Company/university" and "Use case" and consents to marketing email. The unauthenticated config returns 401.
- speech-swift `Community1DiarizationPipeline.swift:18` downloads `aufklarer/Pyannote-Community-1-CoreML`: cc-by-4.0, `gated: False` (200), includes `LICENSE` (CC BY 4.0 notice plus source revision 3533c8c). CC-BY permits this ungated mirror.
- WeSpeaker:
  - `pyannote/wespeaker-voxceleb-resnet34-LM` and `Wespeaker/wespeaker-voxceleb-resnet34-LM` are both **cc-by-4.0**.
  - wespeaker `docs/pretrained.md:19-23`: pretrained VoxCeleb models follow CC BY 4.0. The code `LICENSE` is Apache-2.0.
  - `aufklarer/WeSpeaker-ResNet34-LM-{MLX,CoreML}` (`WeSpeaker.swift:56,59`) are labelled **mit**. The MLX card says "original WeSpeaker model is released under the MIT License", linking a file that is Apache-2.0. That statement is wrong.
- `aufklarer/Pyannote-Segmentation-MLX` is MIT and ungated. Upstream `pyannote/segmentation-3.0` is MIT with gate `auto`. These are consistent.

### L6 MADLAD / Omnilingual
- `MADLADTranslator.swift:25` has default `aufklarer/MADLAD400-3B-MT-MLX`: apache-2.0, ungated, int4 1.84 GB and int8 3.31 GB.
- Upstream `google/madlad400-3b-mt` is apache-2.0, ungated, and includes `te` among 419 language tags. The card says "Primary intended users: Research community", which is not a license term. `int4/tokenizer.json` contains `<2te>` at id 139.
- Omnilingual: `aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8{,-10s}` and `…-{300M,1B,3B,7B}-MLX-{4bit,8bit}` are all apache-2.0 and ungated. Upstream `facebook/omniASR-CTC-{300M,1B,3B,7B}` are apache-2.0. The GitHub LICENSE is Apache-2.0. `lang_ids.py:1412` has `"tel_Telu"`.

## What the app must show, per model

| Model (repo the app downloads) | Display | Accept step? | Gated? | NC / attribution |
|---|---|---|---|---|
| aufklarer/OmniVoice-MLX-fp16 / -int8 | **CC-BY-NC (weights, per k2-fsa card) + Boson Higgs Audio 2 Community License (codec) incl. Llama 3 license/AUP**; do NOT show the HF "apache-2.0" label | Yes (Boson terms bind on use; recommend click-through) | No | **Non-commercial**; CC attribution; the "Built with Higgs Materials…" string must be displayed; 100k AAU cap; no training other LLMs on outputs |
| aufklarer/Indic-Mio-MLX-fp16 | Apache-2.0 (Indic-Mio) + MIT (MioCodec) | No | No | Attribution. Data caveat: expresso is CC-BY-NC-4.0 |
| aufklarer/WavLM-Base-Plus-MLX-fp16 (Indic-Mio voice cloning) | Ambiguous: HF card points to CC-BY-SA-3.0, unilm is MIT | No | No | Possible ShareAlike and attribution |
| mlx-community/translategemma-*-it-{4bit,8bit} | Gemma Terms of Use + Prohibited Use Policy | Yes (recommended; the upstream gate requires it) | No (upstream google/* gated) | Use restrictions; Notice file if ever re-hosted |
| shankarpandala/chatterbox-telugu | CC-BY-4.0 (T3) + MIT © Resemble AI (s3gen/ve/conds) | No | No | Attribution (author, FLEURS, IndicVoices-R, Resemble) |
| aufklarer/Chatterbox-Multilingual-MLX-fp16 + mlx-community/S3TokenizerV2 | MIT + Apache-2.0 (CosyVoice2 upstream; mirror unlabelled) | No | No | Attribution |
| aufklarer/Sortformer-Diarization-CoreML | **NVIDIA Open Model License** (upstream v2.1) with the notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"; conversion labelled CC-BY-4.0 | Implied by use | No | Attribution / Notice; commercial OK |
| aufklarer/Pyannote-Community-1-CoreML | CC-BY-4.0 | No | No (upstream gated auto) | Attribution to pyannote |
| aufklarer/WeSpeaker-ResNet34-LM-{MLX,CoreML} | **CC-BY-4.0** (not MIT) | No | No | Attribution to WeSpeaker / VoxCeleb |
| aufklarer/Pyannote-Segmentation-MLX | MIT | No | No (upstream gated auto) | Notice |
| aufklarer/MADLAD400-3B-MT-MLX | Apache-2.0 | No | No | Notice |
| aufklarer/Omnilingual-ASR-CTC-* | Apache-2.0 | No | No | Notice |

## Conflicts with a free Apache-2.0 app that ships only manifest metadata

- No license here forbids redistributing repo ids, license names, URLs or hashes.
- The real risk is the manifest **mis-stating** licenses. Copying HF labels would present:
  - OmniVoice as Apache-2.0 (wrong)
  - WeSpeaker as MIT (wrong)
  - Sortformer as CC-BY-4.0 (it is the OML)
  - WavLM as MIT (ambiguous)

  Hard-code verified license strings instead.
- OmniVoice is non-commercial. Dubbing and uploading monetised YouTube videos is commercial use, so do not make it a default engine, or gate it behind an NC acknowledgement.
- If LazyDub ever **re-hosts** converted weights, redistribution duties apply. That is likely for chatterbox-telugu, which ships `.pt` pickles and no `model.safetensors`/`tokenizer.json`, while speech-swift's loader expects those files (`ChatterboxTTSModel.swift:155,273-275`). The duties are:
  - CC-BY attribution plus an indication of changes
  - an MIT notice
  - for Gemma: a Notice file, a copy of the ToU, and the §3.2 restrictions
  - for the OML: a copy of the Agreement and a Notice file
  - for Boson: attribution, and the "Higgs Audio 2" name prefix for derived models
