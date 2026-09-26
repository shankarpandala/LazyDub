# 10 — Recognition for dubbing and automatic dub QA (verified, 2026-09-24)

This is desk research. Nothing was run on the M5 Pro and no model was downloaded. It builds on `scratchpad/sota_ranking.md` and does not redo its English ASR, aligner or diarization shortlist. The researcher's draft (`10-asr-and-qa.sweep.md`, 26 findings) was checked claim by claim against primary sources: arXiv PDFs and metadata, Hugging Face API and model cards, GitHub repos through the authenticated `gh` CLI, vendor help pages and the LazyDub repo at HEAD. The web-search budget was exhausted, so sources were fetched directly.

Verdict markers:
- **[confirmed]**: I re-checked it against the primary source.
- **[corrected]**: the claim was partly wrong. The correct fact and its source are given.
- **[unverifiable]**: I could not confirm it.

Refuted sub-claims are listed separately. Unless a note says otherwise, "disclosed" means the company or authors published it, and "inference" means it is my reasoning or the researcher's.

## Summary

1. **Whisper can't check Telugu [confirmed].** Zero-shot Whisper large-v3 scores a Telugu CER of 82.65 on FLEURS and 103.03 on Common Voice 25. That explains ADR-014's "Whisper-te back-transcription CER was too noisy".
2. **Best open Telugu recognisers [confirmed].** SraVaani-1.0 (MIT) averages 25.1 WER over six Telugu test sets and IndicConformer-600M (MIT) averages 26.4, both ahead of Gemini 3 Flash (28.9) and Sarvam Saaras v3 (28.7). A third-party table confirms IndicConformer on clean read Telugu: Kathbath CER 3.1. Synthetic dub takes are clean read-like speech, so that result is the relevant one.
3. **Key correction: the proposed "second family" isn't a second family.** The Best-of-N paper's families are Whisper, wav2vec 2.0 and HuBERT. SraVaani and IndicConformer are both Conformer-style CTC/transducer models trained on overlapping public Indic corpora. For independent reporting, use a Whisper-lineage Telugu model or a wav2vec2-lineage one:
   - BuzzASR/telugu (MIT, a whisper-large-v3 fine-tune that can run on the already-pinned mlx-whisper after conversion);
   - Meta Omnilingual CTC v2 (Apache-2.0, but it needs fairseq2).
4. **Script normalisation is required [confirmed], and a finding the researcher missed shows how.** An independent model card measured IndicConformer keeping 0% of English words in Latin script: it writes "order" as ఆర్డర్. Maata's lines keep English in Latin. So compare in Telugu script after mapping each English word to its spoken Telugu-script form. Claude, already in the pipeline, can emit that form. A deterministic ISO-15919 romanisation won't match loanword spellings (inference). New model: svanita-0.6b (CC-BY-4.0, released 2026-09-19) transcribes Telugu with English kept in Latin script, but it is less accurate than IndicConformer.
5. **Automatic quality predictors are weak signals [confirmed].**
   - Off-the-shelf MOS predictors correlate at 0.08–0.29 (utterance-level Spearman) with human ratings on LIMMITS'24 Indic cross-lingual cloning. IndicMOS reaches 0.56, but only in-domain.
   - Emotion-embedding cosine doesn't match human perception.
   - Cross-lingual speaker-similarity loss comes mainly from the language change, so Maata's same-language ceiling of 0.746 overstates what EN→TE clones can reach.
6. **ElevenLabs and YouTube disclose no automatic QA [confirmed].** Their QA is human review plus regeneration.
   - **Correction:** ElevenLabs' instability warning is about the default *track* clone when a voice varies a lot, not about clip clones.
   - YouTube also admits trouble matching the dubbing voice to the original voice.
7. **Human dubs are only partly time-aligned [confirmed].** The median share of overlapping speech time per line is 0.731.
   - **Correction:** the corpus is 319.57 h of *English originals* with German and Spanish dubs where available (43.2 h and 118.7 h).
8. **Maata's ASR call has no omission or hallucination guards [confirmed in the repo].** The locked mlx-whisper already supports `hallucination_silence_threshold` and `clip_timestamps`, and the locked torchaudio 2.11.0 has `forced_align`. Per-word CTC scoring therefore needs no new dependency.

## Repo facts (LazyDub HEAD, 2026-09-24) — all [confirmed]

- `docs/DECISIONS.md` ADR-014 says: "Whisper-te back-transcription CER was too noisy to rank the variants".
- `engine/src/maata_engine/backends/apple.py:33-35` calls `mlx_whisper.transcribe(word_timestamps=True, condition_on_previous_text=False, verbose=None)`.
  - `hallucination_silence_threshold` and `clip_timestamps` are left at their defaults, and there is no VAD.
  - The per-word `probability` is stored in `TimedWord` but never used.
  - The locked mlx_whisper supports both parameters (`transcribe.py:76-77`), and the threshold applies only when `word_timestamps=True`, which Maata sets.
- `backends/torch_common.py:124` defaults `exaggeration` to 0.5 once per backend instance, and `:188` passes `emotion_adv = self.exaggeration`.
- `text/tenglish.py:60` (formal style) asks for English terms "written in Latin script". The default colloquial style keeps English words too, and `tenglish.py:375` logs the Latin-script share. Latin-script English is therefore the house convention.
- `speakers.py` keeps both `turns` (which may overlap, i.e. crosstalk) and `exclusive` (pyannote exclusive diarization).
- `session.py:58` sets `RESYNTH_SHARE = 0.10`. `:566` allows a shorter re-synthesis on at most max(3, 10 %) of voiced lines. ADR-016 gives every voice a calibration take that seeds its speaking rate.
- `claude_cli.py` makes the `claude` CLI the only text model and states that audio never goes out.
- `engine/uv.lock` pins onnxruntime 1.30.0, torchmetrics 1.9.0, torchaudio 2.11.0 (`torchaudio.functional.forced_align` is present), transformers 5.2.0 and librosa 0.11.0.
- The voice-similarity yardstick (`scratchpad/voice_sim.py`) uses pyannote community-1 WeSpeaker ResNet34 cosine. ADR-017 records guest 0.466 and host 0.645.

## Findings, with verdicts

### A. Telugu recognisers (the checker for the dub)

**A1. Zero-shot Whisper large-v3 can't be used to check Telugu. [confirmed]**
- **BuzzASR/telugu card** (HF sha 598f495e, lastModified 2026-09-22, MIT, 3.10 GB, base whisper-large-v3):
  - Whisper large-v3 zero-shot: normalised CER 82.65 on FLEURS and 103.03 on CV25.
  - BuzzASR fine-tune: FLEURS CER 13.53 / WER 47.06, CV25 CER 1.22 / WER 8.21.
- **Paper:** arXiv 2609.09554, published 2026-09-09. The arXiv comment says "Accepted at EMNLP 2026"; the card says Findings.
- **Caveat (inference):** CV25 CER 1.22 against FLEURS 13.53 suggests Common Voice sentence overlap between training and test. Trust FLEURS more.
- **Sources:** https://huggingface.co/BuzzASR/telugu (2026-09-22); https://arxiv.org/abs/2609.09554 (2026-09-09).

**A2. SraVaani-1.0 and IndicConformer-600M are the best open Telugu recognisers. [confirmed]**
- **SraVaani paper, Table 3.** Telugu over 6 datasets and 17.77 h of test audio, normalised WER:
  - Gemini 3 Flash 28.9
  - Saaras v3 28.7
  - IndicConformer-600M (RNNT decoding) 26.4
  - SraVaani-1.0 25.1
  - The numbers come from the SraVaani authors (IISc/ARTPARK), who are an interested party.
- **SraVaani card:** the README is readable even though the repo is gated.
  - About 430M parameters, FastConformer with a hybrid TDT-CTC decoder (paper: 17 layers, d_model 1024).
  - Stored as fp16 TorchScript, about 900 MB (API: 0.909 GB). MIT, gated (auto), trust_remote_code, HF sha f5dd5358 (2026-09-02).
  - Telugu per set: FLEURS 23.1, IndicTTS 20.9, Kathbath 21.3, MUCS 26.4, RESPIN 25.4, Vaani 33.5.
  - The card lists `en` among its languages (Vaani English WER 14.8).
  - The sample code targets only cuda or cpu. The official ONNX export is hosted on Google Drive, which a pinned `models.lock.json` fetch can't use unless the file is re-hosted with a checksum.
- **IndicConformer card:**
  - MIT, 600M parameters, hybrid CTC+RNNT, 22 languages, gated (auto), HF sha e9b71b36 (2026-02-07), 2.56 GB.
  - Files: `assets/ctc_decoder.onnx` (0.023 GB) and `joint_post_net_te.onnx`.
  - Its README installs `onnxruntime==1.20.1` plus transformers and torchaudio and uses trust_remote_code. Maata pins onnxruntime 1.30.0, and whether they are compatible is unverified.
  - It requires an explicit language code.
- **Repo gates (verified 2026-09-24):** the maintainer's HF login gets "Access denied" on both repos, so he has not yet accepted either gate.
- **Sources:** https://arxiv.org/abs/2608.08235 (2026-08-08, rev. 2026-08-12); https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 (2026-09-02); https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual (2026-02-07).

**A3. Telugu Whisper fine-tunes score worse than the Conformers. [corrected: sample size]**
- vasista22/whisper-telugu-large-v2 (Apache-2.0) scores WER 0.329 on FLEURS (n=100), 0.483 on CV25 and 0.420 on IndicVoices (n=100). **Correction:** the CV25 Telugu set has n=86, not 100 (Praxel Table V).
- ARTPARK-IISc/whisper-large-v3-vaani-telugu (Apache-2.0, sha 9525b3e0, 2026-04-02, 6.18 GB) publishes no numbers [confirmed].
- The Praxy LoRA wins on entity-dense audio: entity hit rate 0.473 against 0.027 for vasista22 [confirmed]. It costs 6.6 points of WER on FLEURS-Te.
- **Sources:** https://arxiv.org/abs/2605.03073 (2026-05-04); HF API (2026-09-24).

**A4. Meta Omnilingual ASR. [confirmed]**
- Apache-2.0. The per-language CSV gives `tel_Telu` 229.3 training hours and CER 6.5 for the 7B LLM-ASR v1, on Meta's own evaluation mix.
- Sizes from the README (December 2025 update):
  - CTC-300M v2: 1.3 GiB, about 2 GiB of memory, RTF 0.001.
  - CTC-1B v2: 3.7 GiB, about 3 GiB.
  - LLM-7B: 30 GiB, about 17 GiB.
- CTC and LLM inference accepts only audio under 40 s. December 2025 added `LLM_Unlimited_*_v2` variants without that cap. Maata's 3–5 s lines are unaffected either way.
- It needs fairseq2 and libsndfile (`brew install libsndfile`). The v2 weights are on dl.fbaipublicfiles.com. facebook/omniASR-CTC-1B on HF was created 2025-11-27.
- **Added (inference):** the CTC models are wav2vec2-lineage, which makes them a true second family for Telugu reporting.
- **Sources:** https://github.com/facebookresearch/omnilingual-asr (README, fetched 2026-09-24); per_language_results_table_7B_llm_asr.csv.

**A5. Indic TTS research measures intelligibility with IndicConformer and speaker similarity with WavLM. [confirmed]**
- In the IndicF5 paper ("Phir Hera Fairy", AI4Bharat), WER comes from IndicConformer on the IN11 test set and similarity from WavLM embeddings.
- The MOS test used 134 listeners, about 12 native speakers per language, each rating at least 30 utterances.
- **Added:** WavLM's official licence (microsoft/UniSpeech LICENSE) is CC-BY-SA-3.0, so WavLM-based similarity is not on the pre-approved list.
- **Sources:** https://arxiv.org/abs/2505.20693 (2025-05-27); https://github.com/microsoft/UniSpeech/blob/main/LICENSE.

**A6. Qwen3-ASR has no Telugu. [confirmed]**
- The card lists 30 languages plus 22 Chinese dialects, including Hindi but not Telugu. Qwen3-ForcedAligner covers 11 languages.
- parakeet-tdt-0.6b-v2 outputs punctuation and capitalisation natively.
- The Canary-Qwen-2.5B card (CC-BY-4.0) says its training skewed it towards verbatim transcripts that keep disfluencies.
- The English-side choice in sota_ranking.md is unaffected.
- **Sources:** https://huggingface.co/Qwen/Qwen3-ASR-1.7B; https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2; nvidia/canary-qwen-2.5b card (all fetched 2026-09-24).

**A7 (missed by the researcher). svanita-0.6b: a Telugu recogniser that keeps English in Latin script, and third-party numbers for IndicConformer.**
- **Disclosed on the card [confirmed as stated]:**
  - prasadvittaldev/svanita-0.6b, CC-BY-4.0, sha 795b39ac, created 2026-09-19, updated 2026-09-24.
  - Base nvidia/parakeet-tdt-0.6b-v3: FastConformer (24 layers) with a TDT decoder, about 622M parameters, a new 4,097-piece tokenizer.
  - Trained on 451 h of Telugu and 685 h of Hindi from IndicVoices and Kathbath.
  - Output is Telugu script with English words in Latin, lowercase, no punctuation. Runs at 10–12× real time on a desktop CPU in fp32.
  - The card warns that int8 quantisation cost about 30 WER points and that its ONNX export was not numerically faithful.
- **Card evaluation, self-reported by one developer [unverifiable independently].**
  - IndicVoices Telugu, conversational speech (1,499 clips, 153 held-out speakers):
    - IndicConformer RNNT: 27.4 WER (either script), 39.1 WER when English must be in Latin, CER 22.1, **0 %** of English words kept in Latin.
    - Svanita v0.1 (Telugu-only branch): 36.5 WER, 36.9 with Latin required, CER 16.4, 76 % of English kept in Latin.
    - Svanita v0.2: 37.7 WER, 74 % kept in Latin.
    - vasista22 large-v2: 44.0 WER.
  - Kathbath Telugu read speech: IndicConformer RNNT 20.9 WER / 3.1 CER; svanita v0.1 36.3 / 8.9.
- **Why it matters (inference):**
  - It answers the draft's open question: IndicConformer writes English loanwords in Telugu script.
  - On clean read speech, the closest proxy for TTS takes, IndicConformer is far more accurate.
  - Svanita is a candidate for a Tenglish-aware second opinion, but it is also Conformer-lineage and all its numbers are self-reported.
- **Source:** https://huggingface.co/prasadvittaldev/svanita-0.6b (2026-09-24).

**A8 (missed). Other new Telugu recognisers found and rejected or parked.**
- **bharatgenai/Shrutam-2** (updated 2026-09-15): the card says "BharatGen non-commercial license". It is also LLM-based. **Reject.** [confirmed]
- **bodhan-ai/indic-transcribe-core and -flex** (2026-09-17 and 2026-09-23): canary-1b-v2 base, 9.78 GB repo, licence tag "other", gated. The card couldn't be read without accepting the gate. **Park.** Any use needs a licence ruling.
- **ARTPARK-IISc/Vaani-FastConformer-Telugu:** MIT, 1.75 GB, 2026-04-02, no numbers read.
- **Source:** HF API (2026-09-24).

### B. How to use recognisers for QA

**B1. Picking the best of N takes with the same recogniser family used for evaluation inflates the results. [confirmed]**
- **Study:** Yu & Kang (KAIST), ICML 2026 workshop. F5-TTS on LibriSpeech-PC, English only.
- **Findings:**
  - Same-family selector/evaluator pairs recover 2–3× more oracle headroom than cross-family pairs.
  - The preferred selector changes between Whisper, wav2vec 2.0 and HuBERT evaluators.
  - Cross-family rank averaging and max-rank ensembles gave a mean WER of 1.61 % at N=10, 12 % below the baseline. At that N, rank-avg ties the single distil-v3 selector.
  - SIM-o changed by at most ±0.0006 and UTMOS by at most ±0.005.
- **Source:** https://arxiv.org/abs/2607.08256 (2026-07-09, rev. 2026-08-05).

**B2. Corrected implication: SraVaani is not an independent family from IndicConformer. [corrected, inference]**
- The draft's summary (point 3) and recommendation R1 call SraVaani an "independent family" to IndicConformer. Both are Conformer-style hybrid CTC/transducer ASR, fine-tuned on overlapping public Indic corpora (Kathbath, FLEURS, IndicVoices-family, MUCS and so on).
- The paper's families are Whisper, wav2vec 2.0 and HuBERT, and its coupling is at the "identity or lineage" level.
- **Recommendation:** report with a Whisper-lineage Telugu model (BuzzASR/telugu, MIT) or a wav2vec2-lineage one (Omnilingual CTC v2, Apache-2.0).
- **Sources:** as B1 and A2.

**B3. Code-mixed back-transcription needs script normalisation before computing CER. [confirmed; the method is corrected]**
- Maata writes English words in Latin script, and IndicConformer returns them in Telugu script (A7), so plain CER counts correctly spoken words as errors.
- **SN-WER** (Pattnayak, ACL 2026 MeLLM) transliterates both sides into one canonical script:
  - the transliterator (IAST, ITRANS or ICU) changes results by less than 0.002;
  - it removes 67 % of artificial romanisation-induced WER inflation;
  - it covers hi, bn, ta, or and gu, **not Telugu**, and targets monolingual benchmarks rather than code-switching.
- **PIER** scores only tagged embedded-language words (arXiv 2501.09512, 2025-01-16; repo Apache-2.0, created 2024-09-05).
- **Correction (inference):** a letter-for-letter romanisation doesn't make "phone" match ఫోన్. The reference side needs each English word in the Telugu-script spelling a Telugu speaker would use.
- **Sources:** https://arxiv.org/abs/2606.02548 (2026-06-01); https://github.com/enesyugan/PIER-CodeSwitching-Evaluation.

**B4. CTC scoring against the intended text pinpoints skipped or garbled words. [confirmed for the method; applying it to Telugu TTS is inference]**
- GOP-SA / GOP-SF (Cao et al., accepted to IEEE TASLP) lets CTC-trained models score pronunciation without segmenting the audio first. It was validated on English learner speech (CMU Kids, speechocean762).
- For Maata, a simpler variant is enough: CTC forced alignment of the intended token sequence to IndicConformer's CTC output, giving a score per word. `torchaudio.functional.forced_align` is present in the locked torchaudio 2.11.0.
- **Source:** https://arxiv.org/abs/2507.16838 (2025-07-18).

**B5. Joining ASR words to speakers works best with exclusive diarization, which Maata already uses; overlapped speech is tracked but unused. [confirmed; blog date unverifiable]**
- The pyannote blog says speech-to-text models "often struggle with overlapped speech and short backchannels", and introduces exclusive mode to handle that. The date shown on the page is unreliable (2026-09-21/22 appear in it). The HF model repo was last modified 2025-09-29 (sha 3533c8cf).
- Downstream flags for crosstalk lines are absent (inference).
- **Source:** https://www.pyannote.ai/blog/community-1 (fetched 2026-09-24).

**B6. Maata's ASR call has no omission or hallucination guards. [confirmed in the repo; the literature is confirmed with nuances]**
- **Whisper-CD** (Interspeech 2026): training-free contrastive decoding. It cuts WER by up to 24.3 points on CORAAL and generates tokens 48 % faster than beam search. Tested on English long-form audio.
- **Calm-Whisper** (Interspeech 2025): 3 of 20 heads account for more than 75 % of non-speech hallucinations on UrbanSound. **Nuance:** it fine-tunes those heads (it is not training-free), and it was studied on large-v3, not turbo.
- **[unverifiable]:** whether ASR omissions cause the "skipped sentences" complaint. That needs a coverage log.
- **Sources:** https://arxiv.org/abs/2603.06193 (2026-03-06, rev. 2026-06-22); https://arxiv.org/abs/2505.12969 (2025-05-19).

### C. Automatic predictors (MOS, emotion, prosody, similarity)

**C1. Licences for emotion-recognition models. [corrected: one extra constraint]**
- **emotion2vec_plus_large** (about 300M parameters, 9 classes, HF sha 6c303ba9, 2024-06-24, 1.95 GB) is under the FunASR Model License v1.1. [confirmed]
  - It allows use and modification with attribution.
  - §4.2: "unjustified denigration" leads to automatic forfeiture of the licence.
  - §6: revised terms take effect automatically.
- **ASLP-lab/Emotion2Vec-S:** Apache-2.0, 1.13 GB, an SSL representation model that needs fairseq and a head. [confirmed]
- **3loi SER-Odyssey WavLM Multi-Attributes:** arousal/valence/dominance, MIT tag, 1.27 GB file, trained on MSP-Podcast. **Correction:** it is a WavLM fine-tune, and WavLM's licence is CC-BY-SA-3.0 (share-alike), on top of the MSP-Podcast academic-licence caveat.
- **audeering w2v2 MSP-dim:** CC-BY-NC-SA-4.0. **Reject.** [confirmed]
- **Chatterbox:** one emotion control, `exaggeration`, default 0.5. The README says higher exaggeration tends to speed speech up, and lower `cfg_weight` compensates. [confirmed]
- **Sources:** HF API and cards (2026-09-24); https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE; resemble-ai/chatterbox README.

**C2. Licences for MOS predictors. [corrected]**
- UTMOSv2 is MIT for both code and weights (HF sha 506474f2, lastModified 2025-05-10). [confirmed] Its code loads an SSL backbone through `transformers.AutoModel.from_pretrained` and the five fold weights from HF, so all of it must be pre-fetched and pinned.
- NISQA weights are CC-BY-NC-SA-4.0. **Reject.** [confirmed] **Added:** torchmetrics' NISQA wrapper downloads those weights at runtime (`functional/audio/nisqa.py`), so never call it.
- torchmetrics 1.9.0 DNSMOS downloads its ONNX files at runtime from `raw.githubusercontent.com/microsoft/DNS-Challenge/master`, which is unpinned. [confirmed] The DNS-Challenge repo is CC-BY-4.0.
- Audiobox Aesthetics is CC-BY-4.0, with a 0.415 GB safetensors file (repo 0.83 GB with the .pt). [confirmed]
- VERSA is Apache-2.0 (pushed 2026-09-20), but each bundled metric has its own licence. [confirmed]
- **Correction on Distill-MOS:**
  - Its LICENSE file is MIT; the GitHub API reports NOASSERTION.
  - Its weights (`distillmos/weights/distill_mos_v7.pt`) ship **inside the package** and are loaded locally with `weights_only=True`. Pinning the package version pins the weights; nothing is downloaded.
  - It depends on the separate `xls_r_sqa` package, whose licence I didn't check.
  - The card says its primary use is research.
- **Sources:** GitHub via gh (2026-09-24); engine `.venv` torchmetrics source.

**C3. Off-the-shelf MOS predictors barely track the quality of Indic cross-lingual cloning. [confirmed, with nuance]**
- **IndicMOS** (Interspeech 2024), utterance-level Spearman correlation with human ratings:

  | Predictor | LIMMITS'24 (TTS+VC, cross-lingual cloning) | LIMMITS'23 (plain TTS) |
  |---|---|---|
  | torchaudio-squim | 0.086 | 0.575 |
  | DNSMOS | 0.078 | 0.520 |
  | SSL-MOS | 0.285 | 0.636 |
  | IndicW2V+CER+LID (IndicMOS) | 0.562 | — |

  - The Telugu-only IndicW2V figure on L24 is 0.521.
  - **Nuance:** the IndicMOS 0.562 model was trained on L24's own training split, so it is in-domain. Expect less on Chatterbox Tenglish.
  - Weights: SYSPIN/IndicMOS, CC-BY-4.0, 0.38 GB, sha 8270effb. [confirmed]
- **Nuance on arXiv 2406.08911:** the model is UTMOS22, not UTMOSv2. "Zero-shot" there means TTS in an unseen language without fine-tuning data, not zero-shot voice cloning. The finding that it "consistently predicts the highest MOS" for that audio is confirmed.
- **Sources:** https://www.isca-archive.org/interspeech_2024/udupa24b_interspeech.pdf (2024-09); https://arxiv.org/abs/2406.08911 (2024-06-13).

**C4. Emotion-embedding cosine doesn't match human perception. [confirmed]**
- Tsai, Lin, Chou, Hsu, Hsu, Chen, Narayanan and Lee (Interspeech 2026) show that these latent spaces classify emotion well but are "unsuitable for zero-shot similarity evaluation". Differences in words and speaker swamp the emotion signal.
- **Source:** https://arxiv.org/abs/2604.26347 (2026-04-29, rev. 2026-07-22).

**C5. Language mismatch, more than speaker change, drives cross-lingual verification loss. [confirmed; the implication for Maata is inference]**
- Buitrago & Hernando (submitted to IberSPEECH 2026) used a bilingual same-speaker set for five Iberian languages with a HuBERT-based verifier.
- For Maata, the 0.746 same-language ceiling overstates what EN→TE clones can reach.
- **Source:** https://arxiv.org/abs/2607.01161 (2026-07-01).

**C6. In cross-lingual cloning, intelligibility and speaker similarity trade off. [confirmed, but MOSS figures unverifiable]**
- **Paper:** Ahtasam, Jamaluddin and Nadeem, IWSLT 2026. English ACL-60/60 references cloned into ar, de, ja, fr, ru and zh. Content scored with Whisper large-v3, similarity with ECAPA-TDNN cosine.
- The paper reports "a clear inverse relationship" between error rate and similarity.
- Mean error / similarity:
  - Qwen3-TTS 0.16 / 0.50
  - VoxCPM2 0.28 / 0.70
  - CosyVoice3 0.18 / 0.69
  - **[unverifiable]:** MOSS-TTS 0.35 / 0.52. The text says only that it had the highest error rate, and those figures appear only in charts.
- The IWSLT 2026 track scores content with ASR, similarity with a speaker-embedding cosine, and objective quality, and allows only open-source models. [confirmed]
- **Sources:** https://aclanthology.org/2026.iwslt-1.12.pdf (2026-07-03); https://iwslt.org/2026/voice-cloning.

**C7. Meta's AutoPCP (cross-language prosody comparison) exists, but its checkpoint has no declared licence. [confirmed]**
- The stopes code is MIT. The README downloads `AutoPCP-multilingual-v2.zip` from dl.fbaipublicfiles.com and states no licence. It was used in the Seamless paper.
- **Sources:** https://github.com/facebookresearch/stopes/tree/main/stopes/eval/auto_pcp (fetched 2026-09-24); https://arxiv.org/abs/2312.05187 (2023-12-08).

**C8. Training-free contrastive decoding for LM-based TTS (ECCD). [confirmed; transfer to Chatterbox is unverifiable]**
- ECCD uses the same speech LM with and without text conditioning. Tested on CosyVoice2, CosyVoice3, Llasa and GLM-TTS:
  - WER/CER down by up to 55.6 % on SeedTTS-Eval;
  - better in 24 of 25 CV3-Eval multilingual settings;
  - CMOS +0.644.
- The paper is marked "work in progress".
- Not tested on Chatterbox or Telugu.
- **Source:** https://arxiv.org/abs/2608.00722 (2026-08-01).

### D. Timing metrics and human evaluation from dubbing research

**D1. Human dubs are only partly time-aligned. [corrected: corpus description]**
- **Paper:** Brannon, Virkar and Thompson (TACL 2023).
- **Correction:** the corpus is 319.57 h of **English-original** Prime Video content (54 shows, 674 episodes), with professional German (43.2 h) and Spanish (118.7 h) dubs where available. Much of the analysis uses a 35.68 h subset that has both dubs.
- Source and dub line durations correlate at r=0.877.
- Overlap fraction (the time both speak divided by the time either speaks, per line): mean 0.658, median 0.731, zero for 4.3 % of lines. On-screen lines score 0.684 and off-screen 0.662.
- The authors argue that vocal naturalness and translation quality matter more than isometry or lip-sync.
- **Source:** https://arxiv.org/abs/2212.12137 (2022-12-23).

**D2. Amazon's automatic dubbing metrics. [confirmed]**
- **Smoothness:** speaking-rate stability across contiguous segments of a clip.
- **Fluency.**
- **Intelligibility:** I(F) = (1 − WER of the prosody-aligned TTS) / (1 − WER of the TTS without alignment) (arXiv 2204.02530).
- **Tam et al.:**
  - Metrics: segmentation accuracy, PhraseLC and MT labels (Acceptable / Fixable / Wrong).
  - Human test: 40 native annotators, each rating 25 of 50 videos, 1,000 judgements per comparison. Raters first watch a post-edited reference dub, then score two systems from 0 to 10, and the result is reported as a percentage of wins (arXiv 2112.08548).
- **IsoChronoMeter:** isochrony estimated from TTS duration predictors, needing no gold data.
- **Sources:** https://arxiv.org/abs/2204.02530 (2022-04-06); https://arxiv.org/abs/2112.08548 (2021-12, Interspeech 2022); https://arxiv.org/abs/2410.11127 (2024-10-14).

**D3. Amazon's MUSHRA test for dubbed video. [confirmed]**
- Federico et al.: 0–100 ratings with the original-language clip as a hidden reference.
- 24 TED clips of 10–15 s. 657 ratings from 14 volunteers, only 5 of whom were Italian, the target language.
- The MT was screened beforehand on a 3-point scale.
- **Source:** https://arxiv.org/abs/2001.06785 (2020-01-19).

**D4. MUSHRA-NMR and MUSHRA-DG for Indic TTS. [corrected: dataset scope]**
- **Paper:** Varadhan et al., TMLR 05/2025. 492 listeners.
- **Reference-matching bias:** Tamil raters preferred StyleTTS2 over the human reference (CMOS 0.24).
- **Rater rejection** shifts scores but not rankings.
- **Anchor:** the Hindi anchor scored 70.81.
- **MUSHRA-NMR:** Spearman above 0.95 with just 20 utterances or 40 listeners; adding listeners helps more than adding utterances.
- **MUSHRA-DG:**
  - Raters count mild and severe mispronunciations, unnatural pauses or speed changes, digital artefacts, sudden energy fluctuations and word skips.
  - They also rate liveliness, voice quality and rhythm.
- **Correction:** MANGO's 246,000 ratings are Hindi (127,500) and Tamil (118,500) only. English was a separate generalisation check with 30 raters.
- The rated systems were FastSpeech2, VITS and StyleTTS2, not LLM-based cloners, so thresholds won't transfer directly.
- MANGO is CC-BY-4.0 (sha 4626e0c7, 2025-05-13).
- **Sources:** https://arxiv.org/abs/2411.12719 (2024-11-19, TMLR 2025); https://huggingface.co/datasets/ai4bharat/MANGO.

### E. What ElevenLabs and YouTube disclose

**E1. ElevenLabs discloses no automatic QA; its QA is human review plus regeneration. [corrected: clone-stability wording]**
- **Dubbing Studio docs** (undated, fetched 2026-09-24):
  - Voice options: Clip clone ("a unique voice clone for each clip"), Track clone ("a single voice clone for the whole track"), and Voice Library voices.
  - The track clone is the default.
  - **Correction:** the page warns that the *track* clone "might create a voice that is a bit more unstable" if the voice changes drastically. Clip clones capture a specific clip's tone.
  - Regeneration defaults to Fixed Generations, which keep clip duration. Dynamic follows text length and can break sync.
  - "Stale audio" flags clips that need regenerating. Clip History lets you pick a better earlier take.
- **Productions:** "each dub goes through a checklist to ensure accuracy, consistency, and natural delivery", a native speaker reviews the translation, segments are regenerated for pacing, and delivery aims for 7 business days.
- **Sources:** https://elevenlabs.io/docs/eleven-creative/products/dubbing/dubbing-studio; https://elevenlabs.io/docs/services/productions/dubbing (both fetched 2026-09-24).

**E2. YouTube automatic dubbing. [confirmed, plus one addition]**
- Creators can "Publish manually" to review dubs first, and can publish, unpublish or delete dubs.
- YouTube admits errors from "mispronunciations, accents, dialects, or background noise" and with "proper nouns, idioms, and jargon".
- **Addition:** it also admits "difficulty matching the voice used for dubbing to the original voice".
- Telugu appears only as an original language dubbed into English, with expressive speech support.
- **Source:** https://support.google.com/youtube/answer/15569972 (fetched 2026-09-24).

### F. Where judging runs under the new rules

**F1. Text judging goes to the Claude CLI, audio judging stays with small local models. [corrected]**
- Claude can judge text: completeness and meaning of English against Telugu, and the intended Telugu against the back-transcript.
- **Correction:** the draft calls the audio-LLM judge studies "local LLMs". arXiv 2607.07985 actually tests **Gemini cloud models** (2.5 Flash, 3.5 Flash, 3.1 Pro) as audio judges. ParaPairAudioBench (Interspeech 2026) finds that audio-LLM judges trail humans by 32 points on paralinguistic comparisons and calibrate poorly.
- Sending dub audio to a cloud judge would break `claude_cli.py`'s rule that audio never leaves the machine. A local audio LLM would break the "no local LLMs" decision.
- That Claude can't take audio input through the CLI is my inference; I didn't verify it for 2026.
- **Sources:** https://arxiv.org/abs/2607.07985 (2026-07-08); https://arxiv.org/abs/2606.24648 (2026-06-23); https://arxiv.org/abs/2605.04505 (2026-05-06).

## Refuted (sub-claims; their parent findings are counted as corrected)

- "Distill-MOS downloads its weights automatically, so they must be pre-fetched." **Refuted:** the weights ship inside the package and load from a local path (C2).
- "An ElevenLabs clip clone can be unstable if the voice changes a lot." **Refuted:** the docs attach that warning to the default *track* clone (E1).

## Recommendations for Maata (updated for the corrections)

**R1. Telugu checker: select with IndicConformer, report with a model from another family.**
- **Change:**
  - Use IndicConformer-600M (MIT) on the CPU with the locked onnxruntime as the per-take *selector*. Use its CTC head for per-word alignment and its RNNT head for the transcript.
  - For *reporting*, add one model from a different family:
    - BuzzASR/telugu (MIT, whisper-large-v3 lineage), converted to MLX and run through the already-pinned mlx-whisper, so no new runtime; or
    - Omnilingual CTC-300M or 1B v2 (Apache-2.0, wav2vec2 lineage), which needs fairseq2.
  - Keep SraVaani (MIT) and svanita (CC-BY-4.0) as bench candidates only. They aren't cross-family.
- **Why:** Whisper zero-shot scores 82.65 CER on Telugu. IndicConformer is strongest on clean read speech (third-party: Kathbath CER 3.1). Scoring and reporting with the same lineage inflates gains (B1, B2).
- **Impact:** high. **Effort:** M.
- **Needs approval:**
  - New model pins, each recorded as an ADR.
  - The maintainer must accept the HF gates himself; both currently return access denied.
  - IndicConformer's README pins onnxruntime 1.20.1 against the locked 1.30.0; check compatibility.
  - Omnilingual would add fairseq2, a new dependency.

**R2. A pure-logic `qa` module.**
- **Change:**
  1. **Reference in Telugu script.** When Claude translates a line, it also returns a "spoken-form" copy in which each English word is spelled in Telugu script the way Telugu speakers write it, for example "order" → ఆర్డర్. CER is computed between that copy and the IndicConformer transcript after NFC normalisation, removing zero-width characters and stripping punctuation.
  2. **Per-word scores.** CTC forced alignment with `torchaudio.functional.forced_align` (already in the locked torchaudio 2.11.0) of the spoken-form tokens against the take. It flags the weakest word, a skipped word, or audio after the last word.
  3. **Rate check.** Compare the take's speaking rate with the voice's calibration-take rate in aksharas per second (ADR-016).
- **Why:** plain CER counts correctly spoken English words as errors, and a romanised comparison won't match loanword spellings (B3, A7). Per-word scores locate the fault (B4).
- **Impact:** high. **Effort:** M.
- **Needs approval:** none. No new dependency; the spoken-form field is a prompt change.

**R3. A QA gate on every line inside the 5–10 minute look-ahead.**
- **Change:** extend the existing retake path (`RESYNTH_SHARE`, `session.py:566`).
  1. When a take fails the CER, per-word or tail check, resynthesise with a new seed, up to 3 candidates.
  2. Keep the take with the best average rank across the IndicConformer CER and the speaker-similarity score.
  3. If it still fails, have Claude rephrase the line (simpler wording, or another form of the English term) and synthesise once more.
  4. Write per-line metrics to a JSON log and flag failing lines in the UI, like ElevenLabs' stale-audio flag and clip history.
- **Why:** it targets "incomplete or skipped sentences" on the output side and replaces the human QA pass that ElevenLabs Productions and YouTube's manual review rely on (E1, E2).
- **Impact:** high. **Effort:** M.
- **Needs approval:** the maintainer's sign-off on the extra synthesis budget beyond the 10 % cap.

**R4. Harden the English recognition step.**
- **Change:**
  - Set `hallucination_silence_threshold`, starting around 2 s and tuned on fixtures.
  - Log diarized speech that got no ASR words for more than about 0.8 s, and re-decode those spans with `clip_timestamps`.
  - Flag clusters of low-probability words and repeated n-grams.
  - Flag units whose source overlap (turns minus exclusive) exceeds about 30 %.
- **Why:** there are no guards today (B6), and the coverage log would show whether ASR omissions contribute to the skipped-sentence complaint, which is unverified.
- **Impact:** medium. **Effort:** S.
- **Needs approval:** none.

**R5. Log timing QA for every line.**
- **Change:** record for every line:
  - source/dub overlap fraction, with human dubs' median of 0.73 as a reference, not a target to force (D1);
  - onset lag and applied speed;
  - Smoothness, the rate change between adjacent lines;
  - Amazon's intelligibility ratio for lines sped past 1.1×, **corrected** to (1 − CER at the applied speed) / (1 − CER of the same take at 1.0×) (D2).
- **Why:** it makes "pace not in sync" measurable and checks that mel-interpolation speed-up doesn't cost intelligibility.
- **Impact:** medium. **Effort:** S.
- **Needs approval:** none.

**R6. Change how speaker similarity is judged.**
- **Change:**
  - Measure a bilingual same-speaker ceiling (the maintainer reading matched English and Telugu).
  - Report similarity as a fraction of that ceiling.
  - Flag takes more than 2 SD below the speaker's running median.
  - Always gate similarity together with CER.
- **Why:** language mismatch is the main cause of cross-lingual similarity loss (C5), and similarity trades off against intelligibility (C6).
- **Impact:** medium. **Effort:** S.
- **Needs approval:** none. Use the existing WeSpeaker model, not WavLM (CC-BY-SA-3.0).

**R7. Per-line emotion control, tested blind before adopting.**
- **Change:**
  - Run emotion recognition on each English unit.
  - Map arousal (or the predicted label) to Chatterbox `exaggeration` within about 0.35–0.8. The README says higher values speed speech up, so the timing planner must see the change.
  - Optionally pick an emotion-matched English prompt clip from the same speaker.
  - Blind A/B test it against the fixed 0.5 before adopting it.
  - Never gate on emotion-embedding cosine (C4).
- **Impact:** medium. **Effort:** M.
- **Needs approval:** yes, a licence ruling for one of:
  - emotion2vec+ (FunASR licence: attribution, forfeiture for denigration, terms that change automatically);
  - Emotion2Vec-S (Apache-2.0) plus a head we train, which adds fairseq;
  - 3loi WavLM A/V (MIT tag, but a CC-BY-SA-3.0 WavLM base and MSP-Podcast academic data; the weakest of the three on licensing).

**R8. MOS predictors only as outlier flags.**
- **Change:** flag only the bottom 5 % within each voice; never use an absolute threshold or a ranking.
  - **Preferred:** Distill-MOS (MIT, weights bundled in the package; check the `xls_r_sqa` licence) or IndicMOS (CC-BY-4.0).
  - **Also possible:** UTMOSv2 (MIT; pre-fetch its SSL backbone and fold weights).
  - **Never:** NISQA (non-commercial weights, and torchmetrics fetches them), or torchmetrics DNSMOS (unpinned runtime download).
- **Why:** off-the-shelf predictors correlate at 0.08–0.29 on Indic cross-lingual cloning (C3).
- **Impact:** low. **Effort:** S.
- **Needs approval:** a new dependency or model pin, with an ADR.

**R9. Human test protocol for each release.**
- **Change:**
  - A MUSHRA-DG-style scoresheet on fixed 60–90 s clips, counting mispronunciations, unnatural pauses or speed changes, artefacts, energy jumps, skipped words and sync complaints, and rating liveliness, voice similarity and Telugu naturalness from 0 to 100.
  - Blind A/B win rates between builds.
  - MUSHRA-NMR for rounds with several systems.
  - Commit the ratings and calibrate R2/R3 thresholds on them, because MANGO's systems and languages differ from Maata's (D4).
- **Impact:** medium. **Effort:** S.
- **Needs approval:** none.

**R10. Contrastive decoding in the T3 sampler, once QA can measure it.**
- **Change:** prototype ECCD in the engine's copy of the T3 sampler (`torch_common.py`), reusing the existing CFG text/no-text pass (C8).
- **Why:** it is a code-only change, but its benefit on Chatterbox and Telugu is unproven.
- **Impact:** medium. **Effort:** M.
- **Needs approval:** an ADR.

## Open questions

1. Which Telugu recogniser is most robust on synthetic Chatterbox Tenglish takes: IndicConformer (CTC and RNNT), SraVaani, svanita v0.1, BuzzASR-te or Omnilingual CTC? Every published number is on human speech. This needs a local bench of 50–100 takes on the M5 Pro, scored against the maintainer's MUSHRA-DG ratings.
2. Does IndicConformer's remote code run under onnxruntime 1.30.0 and transformers 5.2.0? How fast is it on the M5 Pro CPU for a 3–5 s line? Does SraVaani's fp16 TorchScript run on the CPU or MPS on macOS?
3. Can Claude reliably produce the Telugu-script spoken form of English words that IndicConformer will emit? For example, ఫోన్ vs ఫోను. Or is an in-repo exception list needed?
4. What CER and per-word thresholds separate good takes from bad ones? They need calibration against human ratings.
5. What is the bilingual same-speaker WeSpeaker ceiling for EN→TE?
6. What licence covers Meta's AutoPCP-multilingual-v2 checkpoint? None is declared. And the `xls_r_sqa` dependency of Distill-MOS?
7. What licence covers bodhan-ai's indic-transcribe models? It is gated and tagged "other".
8. Does per-line exaggeration help, or does it destabilise pace? This needs a blind A/B.
9. Are ASR omissions part of the skipped-sentence complaint? R4's coverage log will answer this.

## Verdict tally (claims checked)

- **Confirmed: 19.** A1, A2, A4, A5, A6, B1, B3 (the finding), B4, B5 (content), B6 (repo and literature), C3, C4, C5, C6 (the three systems), C7, C8 (the paper), D2, D3, E2.
- **Corrected: 9.** A3 (sample size), B2 (independent family), C1 (3loi WavLM licence), C2 (Distill-MOS weights), D1 (corpus description), D4 (MANGO scope), E1 (track vs clip clone instability), F1 (audio judges are cloud Gemini, not local), and the R5 intelligibility-ratio formula.
- **Refuted sub-claims: 2.** Listed above.
- **Unverifiable: 5.**
  - pyannote blog date;
  - MOSS-TTS mean figures;
  - whether ASR omissions cause skipped sentences;
  - whether ECCD transfers to Chatterbox or Telugu;
  - whether GOP/CTC scoring works on Telugu TTS.
- **Added findings the researcher missed:**
  - A7: svanita, and third-party IndicConformer numbers showing 0 % English kept in Latin.
  - A8: Shrutam-2 non-commercial, bodhan gated.
  - The WavLM CC-BY-SA-3.0 licence.
  - torchmetrics NISQA runtime download.
  - Omnilingual `LLM_Unlimited` variants and CTC as a wav2vec2-lineage evaluator.
  - The locked torchaudio `forced_align`.
  - Maata's existing retake path.
  - The HF gates are not yet accepted.
