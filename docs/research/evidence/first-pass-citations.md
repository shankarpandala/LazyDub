# First-pass details and their citations

The dependency survey relies on a few supporting details that were not in any of the seven adversarial verification clusters. This file lists them with their citations and says how far each was checked.

**Check levels**

- **re-checked**: the line was read again at the pinned commit while writing this file.
- **reviewer**: the fact-consistency review pass confirmed it against source. It was not re-read here.
- **first-pass**: only the initial survey agent saw it. Treat it as unverified until it's used.
- **unverified on macOS**: the claim is about macOS runtime behaviour, which no Linux container can check.

## speech-swift (`c4c2fab`)

| Detail | Citation | Check |
|---|---|---|
| Sortformer streaming session exposes `push(audio:)` / `finish()` | `Sources/SpeechVAD/SortformerStreamingSession.swift:105,116` | re-checked |
| Community-1 speaker embedding is 256-d | `Sources/SpeechVAD/Community1Configuration.swift:17` | re-checked |
| WeSpeaker embedding is 256-d | `Sources/SpeechVAD/WeSpeaker.swift:62` | re-checked |
| MLX and Core ML WeSpeaker embeddings are not interchangeable | `docs/inference/speaker-diarization.md:130` | reviewer |
| Silero VAD default is v6.2.1 (MLX) | `Sources/SpeechVAD/SileroVAD.swift:72` | re-checked |
| Gemma 4 chat default `aufklarer/gemma-4-E4B-it-MLX-4bit` | `Sources/Qwen3Chat/Gemma4Chat.swift:51` | re-checked |
| Qwen3 dense chat default `aufklarer/Qwen3-4B-Instruct-2507-MLX-5bit` | `Sources/Qwen3Chat/Qwen3DenseChat.swift:49` | re-checked |
| MADLAD `translate(...)` API | `Sources/MADLADTranslation/MADLADTranslator.swift:139` | re-checked |
| MADLAD repo `aufklarer/MADLAD400-3B-MT-MLX` at `e442b3cd`, Apache-2.0 | HF API | first-pass |
| Sidon restoration outputs 48 kHz; HTDemucs-FT MLX is 320 MB | module sources / HF tree | first-pass |
| No TranslateGemma or Gemma 3 support in speech-swift | repo-wide search | reviewer |

## YouTubeKit (`e5b7d03`)

| Detail | Citation | Check |
|---|---|---|
| Loose video-ID regex `(?:v=\|\/)([0-9A-Za-z_-]{11})` | `Sources/YouTubeKit/Extraction.swift:16-17` | re-checked |
| `YouTube(url:)` falls back to an empty ID when extraction fails | `Sources/YouTubeKit/YouTube.swift:94-95` | reviewer |
| Broken ~2026-03-07 → 03-16 (pinned-player workaround `fc47290` → `4140995`) | git log | reviewer. Whether yt-dlp broke at the same time was not checked. |

## yt-dlp (2026.08.19) and Deno

| Detail | Citation | Check |
|---|---|---|
| `--ignore-config`, `--no-plugin-dirs`, `--no-js-runtimes`, `--js-runtimes` options | `yt_dlp/options.py:424,456,460,477` | reviewer |
| HTTP downloads in 10 MiB chunks | `yt_dlp/extractor/youtube/_video.py:3229` (`CHUNK_SIZE = 10 << 20`) | reviewer |
| Googlevideo URLs expire after ~6 h and are IP-bound | `expire` / `ip` query params seen in sample URLs | first-pass. S1 measures it. |
| Deno 2.9.7, `deno-aarch64-apple-darwin.zip` + `.sha256sum` | GitHub release metadata | first-pass. Re-checked at pin time. |
| A downloaded ad-hoc-signed binary runs from a hardened-runtime parent when not quarantined, and a quarantined one is blocked; App Sandbox is effectively ruled out | Apple platform behaviour | **unverified on macOS**. S1 tests it. |

## PerTh watermark

| Detail | Citation | Check |
|---|---|---|
| PerTh is MIT; the implicit watermark net is `perth_net_250000.pth.tar`, 37,429,684 bytes | `resemble-ai/Perth` LICENSE and `pretrained/implicit/` | reviewer |
