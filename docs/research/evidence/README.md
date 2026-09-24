# Verification evidence (2026-09-23)

These notes are the raw evidence behind `../2026-09-23-dependency-survey.md`. Each file records one adversarial verification pass: an agent tried to refute a group of research claims using only primary sources. Those sources were upstream source code at pinned commits, Hugging Face API and tree JSON, safetensors headers read with HTTP range requests, and official release assets.

- They were produced in a Linux cloud container. Nothing here was measured on a Mac, and no performance numbers come from these notes.
- Paths such as `scratchpad/...` refer to that container's scratch directory. The clones, scripts and fixtures there were not committed. Re-derive anything you need from the pinned commits named in each file.
- Swift results (tokenizer and grapheme segmentation) came from Linux Swift toolchains 5.10.1, 6.0.3, 6.2.4 and 6.4. macOS behaviour still has to be confirmed on the reference Mac.

| File | Covers |
|---|---|
| `chatterbox-loader.md` | speech-swift ChatterboxTTS loader: vocab size, language gate, offline loading, watermark, controls |
| `chatterbox-weights.md` | chatterbox-telugu weights vs. upstream and vs. the MLX bundle; conversion mapping |
| `tokenizer-telugu.md` | Telugu tokenizer failure in the Swift port, verified fixes, grapheme segmentation (akshara counting) |
| `asr-timestamps.md` | Word timestamps and language ID across speech-swift ASR modules; forced aligner limits |
| `licenses.md` | Licenses, gating and redistribution duties for every candidate model |
| `toolchain.md` | Xcode / Swift / mlx-swift / XcodeGen / GitHub runner compatibility |
| `streams.md` | YouTubeKit and yt-dlp + Deno behaviour, hosts contacted, PO tokens, packaging |
| `first-pass-citations.md` | Supporting details from the first-pass survey, with citations and how far each was checked |
