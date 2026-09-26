# Architecture decision records

The maintainer delegated decisions on 2026-09-23 with one overriding goal: **run at best performance on the MacBook Pro M5 Pro (24 GB), with a very good-looking interface.** Every ADR below is judged against that goal first. NVIDIA/CUDA support stays in the design, but it is secondary.

Each ADR records the decision, why it was made, and what would reopen it. Versions are pinned in the lockfiles (`engine/uv.lock`, `app/package-lock.json`, `app/src-tauri/Cargo.lock`) once they are generated on the reference Mac.

---

## ADR-001 — Hybrid desktop app: Tauri 2 shell + Svelte UI + Python engine
- **Decision:** a Tauri 2 shell (Rust) hosts a webview UI written in TypeScript + Svelte 5 + Vite. A local Python inference engine runs as a sidecar.
- **Why:**
  - Tauri uses the system WebKit on macOS, so the shell costs little memory (~30 MB, against ~150 MB+ for Electron). That leaves RAM for models on a 24 GB machine.
  - Svelte compiles to small, fast DOM code and makes a custom, polished design easy without a heavy UI kit.
- **Reopen if:** YouTube embedding fails in WKWebView in a way the loopback origin (ADR-006) can't fix.

## ADR-002 — Engine in Python, Apple Silicon first
- **Decision:** the engine is Python 3.11–3.12. There are two backends selected at runtime:
  - **`apple`:** MLX + PyTorch MPS. This is the primary, tuned backend.
  - **`cuda`:** PyTorch CUDA + CTranslate2 + llama.cpp. Secondary.
- **Why:** chatterbox-telugu, pyannote, Whisper toolchains and yt-dlp are all Python-native. MLX gives the best Apple Silicon throughput for Whisper and LLMs.
- **Reopen if:** first-run install size or startup time misses §10 on the M5 Pro.

## ADR-003 — Per-stage engines on the M5 Pro

| Stage | Apple (primary) | CUDA (secondary) | Why for the M5 Pro |
|---|---|---|---|
| ASR + word timestamps | `mlx-whisper`, large-v3-turbo, `word_timestamps=True` | `faster-whisper` | MLX is the fastest Whisper on Apple GPUs. DTW word timing covers every Whisper language, Hindi included. |
| VAD | Silero (torch CPU) | same | Tiny; the CPU is fine and leaves the GPU free |
| Diarization | pyannote community-1 on MPS | pyannote on CUDA | Best open DER; CC-BY-4.0 |
| Translation | `mlx-lm` + TranslateGemma 4B 4-bit | `llama-cpp-python`, GGUF | ~2.2 GB. The 12B model breaks the 24 GB × 60 % budget next to TTS. |
| Condense / JSON | `mlx-lm` + Qwen3-4B-Instruct 4-bit, loaded on demand | llama.cpp | Only needed when a line overflows |
| TTS | chatterbox-telugu, PyTorch **MPS**, fp16 | PyTorch CUDA | The maintainer's own pipeline, unchanged, PerTh watermark included |

- **Memory plan:** the steady-state resident set is Whisper + diarization + TranslateGemma + Chatterbox, about 8–9 GB by weight sizes. The condenser loads lazily. Measured in Phase 0 (S2–S4).
- **Reopen if:** a measurement shows a stage misses its RTF share of the 1.5× real-time budget.

## ADR-004 — The GPU is serialised, the front-end is pipelined
- **Decision:** one asyncio priority queue owns the GPU (MLX/MPS work: ASR, translation, TTS), with priority set by distance to the playhead. The CPU-side work (fetching, decoding, VAD) runs in parallel threads.
- **Why:** Apple unified memory and a single GPU; running MLX and MPS concurrently causes contention and memory spikes (spec §5).

## ADR-005 — Timing core is pure Python with property tests
- **Decision:** the §6.6 isochrony cascade, DubTimeline, DurationEstimator, AksharaCounter and Telugu TextNormalizer live in `maata_engine.timing` / `maata_engine.text`. They have no ML imports and are tested with `hypothesis`.
- **Why:** the spec calls this the heart of the product. Pure code runs in CI on any OS.

## ADR-006 — YouTube IFrame player served from a loopback HTTP origin
- **Decision:** the engine serves the built UI at `http://127.0.0.1:<port>/`, and the Tauri window loads that URL. The UI embeds `https://www.youtube-nocookie.com/embed/…` through the IFrame API, muted.
- **Why:** YouTube's player refuses embeds without a valid HTTP origin/referrer, and a custom `tauri://` scheme risks error 152/153. The nocookie domain is the privacy-friendlier embed.
- **Reopen if:** S1 shows `tauri://` works everywhere, which would drop one moving part.

## ADR-007 — Engine ⇄ UI protocol
- **Decision:** a single WebSocket at `/ws?token=…`.
  - JSON messages carry control and events.
  - Dub audio is sent as binary frames: a 16-byte header (unit id u32, sample rate u32, sample count u32, reserved) followed by float32 PCM.
  - The token is random per launch, created by the shell and passed to both sides.
- **Why:** one connection and no per-unit HTTP requests. PCM decodes straight into `AudioBuffer`s.

## ADR-008 — Sync: Web Audio scheduled against an extrapolated player clock
- **Decision:**
  - `VideoClock` fits video time against `performance.now()` from `getCurrentTime()` samples, re-anchoring on state, rate and seek events.
  - The `SyncEngine` schedules each unit with `AudioBufferSourceNode.start(when)` at the context time mapped from its `s_u`, corrected by `outputLatency`.
  - Freezes are `pauseVideo()`. Slow-downs use the nearest available IFrame rate at or above 0.85×; if none fits, the cascade falls back to condense.
- **Why:** these are the only clocks available in a webview. S5 measures how accurate they are.

## ADR-009 — Visual design
- **Decision:** a dark, cinematic theme by default, with a light theme too.
  - A warm turmeric-to-vermilion accent gradient, taken from the Telugu cultural palette.
  - Glass panels over a subtly animated backdrop.
  - Inter for Latin text and Noto Sans Telugu for Telugu, both self-hosted. No network fonts, because the app runs offline.
  - The "AI dub" badge is always visible.
  - Motion respects `prefers-reduced-motion`.
- **Why:** it's the maintainer's explicit goal, and the design matches the app's content (video first, chrome recedes).

## ADR-010 — Stream resolution: yt-dlp inside the engine
- **Decision:** yt-dlp runs as a Python library in the engine, pinned, with `remote_components` disabled and no self-update. Deno is its pinned JS runtime.
- **Why:** yt-dlp is the most actively maintained resolver. YouTubeKit was Swift-only and no longer fits.

## ADR-011 — Engine distribution
- **Decision:**
  - The installer carries the shell and UI only.
  - On first run, the Model Manager fetches a pinned python-build-standalone runtime plus the `uv.lock`-pinned wheels for the detected backend, then the models. Everything is verified by sha256.
  - In development, `uv run` from `engine/` is used directly.
- **Why:** it keeps installers small and follows spec §3.3 / §7.

## ADR-012 — Linux target
- **Decision:** Ubuntu 22.04 / 24.04 x64 with NVIDIA driver ≥ 550 is supported. Other distros are best-effort.

## ADR-013 — Chatterbox fork pinned in the engine lock; gated models pinned by LFS pointer
- **Decision:**
  - The maintainer's Chatterbox fork (the only one with `te`) is an `apple`/`cuda` engine dependency, pinned by commit (`shankarpandala/chatterbox@f30b0d1`) as a uv git source and locked in `engine/uv.lock`. `CHATTERBOX_SRC` is gone.
  - The fork hard-pins its training stack (torch 2.6, numpy<2). `[tool.uv] override-dependencies` relaxes torch/torchaudio/numpy/safetensors to what pyannote 4 needs (torch ≥ 2.8), and keeps the fork's tested `transformers==5.2.0`, whose Llama internals T3 uses.
  - `exclude-dependencies` drops what Telugu inference never imports: `gradio` (demo UI), `spacy-pkuseg`/`pykakasi` (zh/ja text front-ends, imported lazily), and dev tooling leaked by `s3tokenizer` (`pre-commit`) and `indic-nlp-library` (Sphinx).
  - `indic-nlp-library` is added: the fork's tokenizer applies its Telugu normalisation, and silently skips it when the library is absent.
  - Hugging Face masks LFS sha256 values in its API for some gated repos (pyannote community-1). Those files are pinned by `lfs_oid`, the git blob id of their LFS pointer at the pinned commit. The fetcher verifies content by rebuilding the canonical pointer from the file's sha256 and size.
  - Without the gated diarization model, the engine dubs with one speaker (spec §12 Phase 1) and `maata-bench fetch` skips it with instructions.
- **Why:** it makes the whole engine reproducible from `uv.lock`, and it keeps the maintainer's TTS pipeline unchanged while meeting pyannote 4's torch floor. Verified on the M5 Pro (24 GB): torch 2.14 MPS runs chatterbox-telugu end to end.
- **Reopen if:** an S4 measurement shows a quality or speed regression versus torch 2.6, or the fork publishes a release with relaxed pins.

## ADR-014 — Faster Chatterbox-Telugu on MPS without touching the fork
- **Decision:** the engine synthesizes through its own copy of the fork's `generate()` path (`backends/torch_common.py`), with three changes:
  - The T3 transformer runs in **bf16**; sampling and guidance stay fp32.
  - Generation is **capped by the line's time budget** (`max_seconds` → speech tokens at 25/s), instead of a fixed 1000 tokens (40 s).
  - The S3Gen flow-matching **step count is configurable** (`cfm_steps`, default 10).
  - Unchanged: CFG, min-p, repetition penalty, the PerTh watermark, and dropping the final token's audio.
- **Why:** measured on the M5 Pro (8 Tenglish lines, about 23 s of speech, built-in voice): stock RTF 0.698, bf16 0.643, bf16 with 6 CFM steps 0.472. A per-token EOS check was measured as no gain and is not used. Whisper-te back-transcription CER was too noisy to rank the variants, so 6 steps waits for a listening check by the maintainer.
- **Reopen if:** the listening check prefers 6 steps (make it the default), or an MLX T3 port is approved (a new dependency).

## ADR-016 — Staged pipeline, speaker pre-pass and pacing (amends ADR-004 and spec §6.2, §6.6–6.8)
- **Decision:**
  - **Stages.** A session runs four tasks: a diarizer, an ASR frontend, a translator and a voicer, all bounded by a lookahead (default 10 min, UI setting 5/10/20).
  - **One GPU lock.** MLX work (ASR, translation) and MPS work (diarization, TTS) take turns. The translator stays about 60 s ahead of the next line to voice, so the two interleave.
    - Measured on the M5 Pro: one short batch of translation beside TTS was 1.14x faster than serial.
    - Under sustained load, TTS slowed from RTF 0.70 to 2.79 beside Qwen3.5-4B and to 3.99 beside Qwen3.5-9B. The pair was about 29 % slower than taking turns.
  - **Speakers.** A pre-pass diarizes the first 10 minutes in 3-minute blocks. A global `SpeakerRegistry` links blocks by pyannote centroid embeddings. Each speaker is cloned from the best 10 s of clean, non-overlapped speech before any line is voiced. Speakers found later are cloned before their lines.
  - **Sentence units.** ASR chunks end at sentence boundaries, and the segmenter splits only at sentence ends or long pauses.
  - **Lossless-first cascade.** A line is first fitted with speed-up (at most 1.25x), IFrame slow-down and a freeze. It is condensed only when those can't absorb it, and then with a prompt that keeps every fact. It is placed with an overdraft freeze if still long. Audio is never trimmed.
  - **Pacing.** The engine reports merged dubbed ranges, measured throughput and a target lead of `R·(1−x)+30 s`. The UI waits for `max(user's prepare-ahead, target lead)` before playing and pauses only when the dub truly runs out.
  - **Logs.** `~/Library/Logs/Maata/engine.log` gets one line per dubbed sentence, and `~/Library/Caches/Maata/<video>/units.jsonl` gets a full per-line trace.
  - **Telemetry.** pyannote.audio's default OpenTelemetry export (to otel.pyannote.ai) and huggingface_hub telemetry are forced off when `maata_engine` is imported.
- **Why:** the maintainer reported one voice for everyone, incomplete sentences, bookish Telugu and stop-and-go playback. An instrumented trace on 3 minutes of their podcast found 0.15x realtime, 4.7x more Telugu speech than slot time, and lines replaced by summaries of earlier lines. The causes were context re-translation, condense-before-absorb and tail trimming.
- **Reopen if:** pyannote's centroid linking splits or merges speakers on real videos, or the lookahead needs more memory than the 60 % budget allows.

## ADR-015 — Translators: Qwen3.5 instruct for spoken Tenglish, TranslateGemma for formal Telugu
- **Decision:**
  - **Default style.** Spoken Telugu with English words ("Tenglish") comes from an instruct LLM through `MLXChatTranslator`. The prompt (`text/tenglish.py`) has a system instruction and six original few-shot pairs. The previous two (source, Telugu) lines go in as earlier chat turns, as context only. The pinned model is `mlx-community/Qwen3.5-9B-MLX-4bit@938d8919` (Apache-2.0); Qwen3.5-4B is the faster alternative.
  - **Formal style.** "More formal Telugu" uses TranslateGemma 4B, loaded on first use. The maintainer accepted the Gemma Terms of Use on 2026-09-24.
  - **Condensing** uses the instruct model with a keep-every-fact prompt, so the Qwen3-4B condenser is retired.
  - **Fixes to the old TranslateGemma path:** it translates only the current line (previous lines were prepended and re-translated), stops at `<end_of_turn>`, and sizes max_tokens from the line.
- **Why:** bake-off on 16 whole sentences from the maintainer's podcast, M5 Pro, one model at a time (Latin-script share of words, seconds per line, and a line-by-line read for faithfulness):

  | Config | English words | s/line |
  |---|---|---|
  | TranslateGemma | 0 % | 0.71 |
  | TranslateGemma → Qwen3.5-4B rewrite | 21 % | 1.55 |
  | Qwen3.5-4B | 22 % | 0.99 |
  | Qwen3.5-9B | 27 % | 1.77 |
  | TranslateGemma → Qwen3.5-9B rewrite | 28 % | 2.26 |

  - Qwen3.5-9B read as the most natural and faithful.
  - TranslateGemma made meaning errors (an invented "she", a flipped subject). They carried through the two-pass rewrite, which sometimes returned the formal draft unchanged.
  - Gemma 4 E4B (Apache-2.0) reached 29 % English words, but leaked a thinking-channel token on a long line and dropped content.
  - Qwen3-4B looped and mistranslated.
- **Reopen if:** end-to-end throughput needs the 4B (see the pipeline benchmark), or a Telugu-tuned open model beats Qwen3.5-9B on the same bake-off.

## ADR-017 — ElevenLabs-style timing, natural Telugu, and better voice references (amends ADR-015, ADR-016)
- **Context:** the maintainer's verdict on the ADR-016 build was that the pace was not in sync with the video, the English mix was forced and unnatural, and the clones were not close to the speakers. An 11-agent research pass on ElevenLabs, YouTube, open TTS, Telugu code-mixing and the local models set the direction. The measurements behind it are in `docs/spikes/results/` when committed.
- **Decision:**
  - **Timing (`timing/planner.py`).** Each line is written to fit its own segment at the voice's measured pace, then placed on the video's own clock by a rolling-horizon planner.
    - A line may start up to 0.3 s early into silence.
    - Speed-up is gentle and steady, at most 1.2x by default, done by resampling the mel before the vocoder (pitch-preserving, CosyVoice's method).
    - Lag is at most 0.6 s, or 1.0 s before a long pause.
    - A line that still doesn't fit is re-synthesized once with a concise wording, capped at 10 % of lines.
    - Freezes are a metered last resort: at most 0.6 s per event and 1 s per minute. The video is never slowed down, and audio is never cut.
    - The ADR-016 cascade (speed-up, video slow-down, freeze, condense) is retired as the normal path.
  - **Translation (`text/tenglish.py`).** The prompt describes spoken Telugu as heard on Telugu YouTube explainers and podcasts, with no English quota.
    - English appears only where people really say it: names, brands, technical and modern terms, established loans. Grammar stays Telugu, English verbs appear only as a stem with చేయు/అవు, and spoken forms are used.
    - It uses 16 original example pairs, versioned by `SHOTS_VERSION` and `PROMPT_HASH`.
    - A pure `lint()` flags forced English, inflected verbs before light verbs, bookish words and echoes; a flagged line gets one targeted retry.
    - When a line is predicted to overrun, three wordings are requested (normal, concise, very concise).
    - The session re-translates a line without context when it echoes the previous line's Telugu while its English doesn't repeat. On the maintainer's podcast this happened on 2 of 15 lines, and those lines would have been lost.
  - **Voices.**
    - Timbre (S3Gen) comes from the speaker's best single clean span of 8–10 s, never stitched; `SpeakerRegistry.best_span`.
    - T3's speaker embedding is averaged over up to 60 s of clean clips; `clean_clips`.
    - T3 is prompted with the speaker's own English, with cfg 0.5.
    - With no clean span of 8 s or more, the stitched reference is used.
    - Each voice gets a calibration take that seeds its speaking rate.
    - Native Telugu prompt clips ("balanced" / "natural" voice match) were measured and did not help, so they are not used.
  - **Segmentation.** Short speaker-label flips inside another speaker's speech (up to 3 words or under 1 s) are absorbed, so one-word units disappear. Split numbers from ASR are rejoined (`text/asr_fix.py`).
- **Evidence (M5 Pro, the maintainer's podcast):**

  | Measure | Before | After |
  |---|---|---|
  | Guest clone: stitched reference → single clip + averaged identity | similarity 0.400, pace 3.43 aksharas/s | similarity 0.466, pace 5.11 aksharas/s |
  | Host, short reference | stitched stays best: 0.645 | — |

  - 90 s smoke run of the new pipeline:
    - Onset lag: p50 0.00 s, p95 +0.02 s, max +0.60 s.
    - Rate: mean 1.008, max 1.07.
    - Freezes 0, overdraft 0.
    - Telugu audio was 0.78x of the target slots.
    - English words 14 %.
    - Throughput 0.97x realtime.
- **Reopen if:** the maintainer's listening disagrees, or a long-video benchmark misses p95 onset lag ≤ 0.5 s or 1 s/min of freezes.
- **Amendment (2026-09-25): calibration and counting in Telugu script** (`docs/research/dubbing-2026-09/ARCHITECTURE.md` §3.7 step 5, §4.5; gap-2 C, gap-4 E3):
  - **Calibration.** Each cloned voice is calibrated from three original Telugu-script sentences of about 15, 28 and 45 aksharas, with their English loans in Telugu script (టాపిక్, ఛానెల్, ఫోన్), instead of one sentence with Latin "friends" and "share".
    - Rate and overhead are fitted together by least squares and set directly. One sentence misjudged the multi-line rate by −13 % to +18 %.
    - Each take is capped at the slowest pace the estimator represents (1.0 s + aksharas / 1.5). A take that runs to its cap is a runaway and is left out; a slow voice is still measured.
    - A new clone is used only once it is calibrated; its speaker keeps their earlier voice until then.
    - Cost: the three takes say about 87 aksharas, against 32 for the old single take, so about 2.7x the speech per voice, most of it before first audio. `maata-bench pipeline` reports the total as `calibration_seconds`.
  - **Keys.** The duration estimator is keyed by (voice, cfg, exaggeration, reference-audio hash). A preset is keyed by its name.
    - The key is taken when the voice is built. So a voice rebuilt from new reference audio or with new settings is a new voice, calibrated afresh.
    - Changing cfg or exaggeration on a voice already built does not re-key or recalibrate it. Nothing does that yet; §3.7 steps 4 and 6 must rebuild the voice when they change a setting.
  - **Counting.** `count_units` counts the Telugu-script line (`spoken`) only; Latin letters count nothing.
    - Every length target, prediction and pace update uses `spoken`, whichever script the TTS reads.
    - The count gives about 9 % more units per line, so `DEFAULT_RATE` goes from 5.5 to 6.1 aksharas/s.
    - The *k* prior counts `full` the same way, per English syllable.
    - A line's target is (speech time − the voice's overhead) × its rate. §4.3 leaves the overhead out, but the band rule includes it, and calibration now fits it directly. Without the change, a `full` written to its target lands above the band on any line shorter than ten times the overhead.
  - **Lint.** `lint()` knows a line's English words from its English map, not from Latin letters.
  - **Not yet measured:** the calibrated rates with the new sentences on the M5 Pro.
- **Amendment (2026-09-25): timing v2** (`docs/research/dubbing-2026-09/ARCHITECTURE.md` §3.10, §7 step 7; gap-3 §3–4; research 08 R3, R4, R7):
  - **A sentence is placed whole over its span** and drifts across the English pauses inside it, within the same lag limits (0.6 s, or 1.0 s before a long pause).
  - **Hard breaks** (`SourceUnit.breaks`, pauses of 1 s or more inside a sentence) are met with **one take per piece**, using Claude's `pieces` (§4.4). This is the method chosen out of the two the design allows.
    - Why not one take with its Telugu pause found afterwards: there is no Telugu aligner yet, and Chatterbox doesn't reliably pause at a comma, so the pause might not be there. Claude writes each piece to be said on its own, so separate takes put the break exactly where Telugu breaks (fluency, not English content).
    - When it applies. All four must hold; otherwise the line is said whole:
      - the wording voiced is `full`;
      - the line has at least two pieces;
      - there are enough hard breaks that no other line starts inside (a break with someone else's line in it isn't one to meet);
      - placed at their predicted lengths, every piece before the last fits without a freeze or overdraft.
    - How the pieces are placed:
      - one at a time, each against its own stretch of English, from its own English onset. The same limits apply: the lag limit, a rate between 1.0 and the ceiling, and an early start of up to 0.3 s into the pause before it, but only into its silence: another speaker's diarized speech, or another line, running into the pause bounds it (`LineSlot.heard`), as speech before a line's onset bounds the line;
      - the silence between two pieces is whatever the placement leaves, so it meets the English pause;
      - with fewer pieces than breaks + 1, each join meets the break whose share of the English speech before it best matches the Telugu said before that join.
    - Fallback: if, after synthesis, a piece before the last would need a freeze or overdraft, the pieces are said as one line drifting over the span, 0.3 s apart (`piece_gap`).
    - Cost: one synthesis call per piece instead of one per line, on lines with hard breaks only.
    - Adopted ahead of §7.1's unit-and-contract bake-off (U; A, the whole sentence drifting; B, split back at breaks with `pieces`), which hasn't run. The risk it exists to catch is naturalness: each piece is its own take, so it may close with a sentence-final fall, and the joined fallback puts 0.3 s between takes. `maata-bench pipeline --timing v2-whole` is arm A from the same build (timing v2 with every sentence said whole).
  - **Soft anchors** (0.3 s breaths at a clause mark) are used only by a line placed at rate 1.0, with no freeze, that ends before the next line's onset − guard.
    - Candidate pauses are the take's own pauses, found on its audio at the natural pace: 10 ms frames 40 dB under the take's 95th-percentile level, in runs of at least 0.15 s. The take's leading and trailing silence counts however short: it isn't voiced.
    - Each candidate is scored as duration match × break plausibility:
      - duration match: 1 − (silence to insert ÷ 0.8 s);
      - plausibility: the pause's length (full credit at 0.25 s), times 1.0 near one of the wording's inner punctuation marks (by akshara share), else 0.5.
    - The best candidate scoring at least 0.5 gets that silence at its middle, so the Telugu resumes when the English does. It never runs past the line's window, and it moves no start and changes no rate.
  - **Asymmetric onset cost.** `w_lag` is split into `w_early` 2 and `w_late` 1; the knock-on to the next line counts as late.
    - The 0.3 s early start goes only into real silence. `prev_end` is now the latest end of the lines before and of any diarized speech before the onset, whoever the speaker.
    - One exception: a turn of the line's own speaker that runs into the onset is the line's own beginning, heard before its first word, so it doesn't count.
  - **Windowed lookahead.** The planner is given the next line (always) and then up to 8 lines starting within 30 s, each with its predicted duration.
    - A line that fits at its onset is placed exactly as before.
    - A line that doesn't fit is compared across these candidates: its one-line optimum, plus 5 rates from its minimum to its ceiling, each at its best start and at its latest start.
    - Each candidate is priced by its own cost plus a greedy placement of the lines in the window after it. The cheapest wins; ties go to the one-line optimum.
    - Evidence so far is pure logic, not a listening result. `test_planner.py`'s six-line chain of long lines is placed with a smaller worst lag and no line needing a shorter wording. A local planner-only run on random conversations showed small gains in onset lag, freezes and overdraft on dense talk and no change on roomy talk; no JSON is committed for it, so it isn't quoted here.
  - **Rate.**
    - `speed_cap` stays 1.2 and the 1.0 floor stays.
    - The design also puts a ceiling of max(1.0, min(`speed_cap`, N aksharas/s ÷ the voice's calibrated rate)) on each line, with N = 7.5 inferred from Kannada read speech. The mechanism is built (`PlannerSettings.akshara_ceiling`) but **off by default** (2026-09-25).
      - The first real run on the M5 Pro calibrated a voice at 8.45 aksharas/s, with its fit held at the 1.0 s overhead limit. That put the ceiling at 1.0: no line could be sped up, and onset lag p95 reached 0.63 s, against the 0.3 s guard.
      - The ceiling depends on how the calibration splits time between overhead and rate, and N was never measured on Telugu. So it stays off until a Telugu measurement sets N; `speed_cap` 1.2 still applies.
  - **Uneven compression.**
    - A sped-up take first shortens its inner pauses, each to no less than 120 ms, and only the rest is taken by speeding its speech. The speech rate is solved so that the take lasts exactly duration ÷ rate.
    - The cuts go at each pause's middle, with 2.5 ms fades.
    - Every take is vocoded once at 1.0 to find its pauses, and that audio is reused when it plays at 1.0. A sped-up take is vocoded again at its speech rate.
    - Cost: a second vocoder pass on sped-up lines. A take that a shorter re-synthesis replaces has already been flowed and vocoded once.
    - Chatterbox has no phoneme durations, so this needs an A/B by ear.
  - **Unchanged:** freezes stay metered (0.6 s per event, 1 s per minute); an interrupted line (`cut_off`) stays unfinished; different speakers overlap only as far as the source did; the seek-back chain rebuild.
  - **Plan and protocol.**
    - `Plan.wall` (the UI's `audioWall`) now includes any silence set between parts: one buffer per line, played from `Plan.start`.
    - `Plan.played` is the take time alone. `speech_fill` is now measured on it.
    - A rephrase that replaces a line said in parts is said whole, from the same start at the same rate.
  - **Metrics** (§3.10), in `planner.stats()`, `units.jsonl` and `maata-bench pipeline`:
    - Speech-level end error. It is the last voiced moment of the dub minus the end of the speaker's last diarized turn inside the line (the span when there is none), which leaves out the take's own trailing silence. The turns are the overlap-aware ones: a speaker whose last words run under an interjection ends where they stop, not where exclusive diarization hands the overlap to the other speaker. The benchmark reports its p10, p50 and p90, the share of lines ending more than 1 s early, and the median for lines longer than 8 s.
    - Speech-level overlap (IoU of speech and voiced time).
    - Seconds per minute of source speech with no dub over it, skipped lines included.
    - Anchor onset error: the voiced onset of each piece or anchored part minus its English onset.
    - `rate_step_p90`, and the regression guards.
    - Per unit in `units.jsonl`: `said`, `parts`, `rate_cap`, `hard_breaks`, `speech`, `voiced`, `anchor_errors`, `end_error_s` and `overlap_speech`. A skipped line records its `speech`.
    - `maata-bench pipeline` adds a `timing` block, computed from `units.jsonl` as the lines were finally voiced, with its `guards`, plus the planner's own `stats()`.
    - `--baseline JSON` (in `verify-mac.sh`, `MAATA_BENCH_BASELINE`) measures the relative targets against a committed run: `rate_step_p90` not worse; the > 1 s-early share and the long-line end error at least halved.
  - **The baseline** is a `maata-bench pipeline --timing v1` run (`verify-mac.sh` with `MAATA_BENCH_TIMING=v1`) on the same fixture, committed, then passed as `--baseline` to the default run.
    - Why not a build from before this change: none of them writes these metrics (no `timing` block, no `speech` or `voiced` in `units.jsonl`), so the "step-0 baseline" of §3.10 can't be made that way.
    - `--timing v1` is this build timed as before: `planner.V1` (early and late onsets cost the same, a one-line lookahead, no akshara ceiling, uniform compression, no soft anchors), no pieces, and an early start bounded by the lines before the onset alone. Every metric above is still written. `test_planner.py` pins each of its settings.
    - So the relative targets measure timing v2 alone, on the same translation, segmentation and voices. The design's figures for older builds (8.2 s/min silent while speaking; a −1.98 s long-line end error, measured at unit level) are context, not the baseline.
    - The mode is a measurement switch: `Session(timing=...)`, not a user setting. The JSON records it as `timing_mode`.
  - **Not implemented:** the intelligibility ratio (CER at the applied rate ÷ CER at 1.0, for lines above 1.1×). It needs step 6's QA recogniser.
  - **Reopen also if**
    - a committed run fails a regression guard: onset lag p95 > 0.3 s, rate p90 > 1.1, `rate_step_p90` worse than the baseline, or freezes > 1 s per 10 min;
    - §7.1's unit-and-contract bake-off doesn't favour pieces: they stay only if B wins on sync and loses no more than 5 naturalness points against A. Otherwise lines are said whole (A).
  - **Not yet measured:** nothing here has run on the M5 Pro. Still owed: the baseline JSON (a `--timing v1` run), the improvement targets (silent while speaking ≤ 3 s/min, end-error median within ±0.3 s, the long-line end error halved), the blind sync rating (≥ 60 % preference) and the unit-and-contract bake-off.

## ADR-018 — Translator: Gemma 3 12B, plain line-by-line prompt (supersedes ADR-015's model choice)
- **Decision:**
  - **Default model.** The spoken-Telugu style uses `mlx-community/gemma-3-12b-it-4bit@86cc6a8d` (Gemma Terms of Use, accepted), pinned in `models.lock.json`. Qwen3.5-9B is removed.
  - **Prompt.** `tenglish.plain_messages`: the natural spoken-Telugu instructions and 16 original examples, the previous two lines as earlier chat turns, then the bare line.
  - **Removed from the line prompt:**
    - The "next line" note, which the model translated as well.
    - The length note on normal lines. Concise wordings (`translate_candidates`) are asked for only when a line is predicted to overrun its slot.
    - Lint-driven rewrites. Lint still runs and is logged.
  - **Formal style.** It stays on TranslateGemma. The 12B was the most faithful model, but it doesn't fit alongside the chat model, so the 4B remains.
  - **Echoes.** The session's echo guard still re-translates a line without context when it echoes the previous one.
- **Why:** blind judging of original podcast-style scenes (37 lines, 5 topics). Each line was scored 0–3 for meaning, grammar, naturalness and English use by three independent reviewers with different emphases. Labels were shuffled per scene.

  **Round 1**

  | System | Overall /10 | Major meaning errors | Naturalness |
  |---|---|---|---|
  | Qwen3.5-9B (the model the maintainer heard) | 2.9 | 51 % of lines | 1.48 |
  | Qwen3.5-9B, windowed | 3.7 | 31 % | 1.79 |
  | Gemma 4 E4B | 5.8 | 11 % | 2.14 |
  | Gemma 3 12B, windowed | 6.4 | 18 % | — |
  | Gemma 3 12B, line by line | 6.7 | 5 % | 2.50 |

  Windowed JSON translation shifted lines when a sentence spanned two of them.

  **Round 2** (Gemma 3 12B line by line as the anchor)

  | System | Overall /10 | Major meaning errors | Naturalness |
  |---|---|---|---|
  | Gemma 3 12B, plain | 6.6 | 5 % | 2.52 |
  | Gemma 3 12B, the app's old prompt (length note + lint retries) | 5.9 | 13.5 % | — |
  | TranslateGemma 12B | 5.3 | 7 % | 1.82 |
  | Gemma 3 12B, the app's old prompt + "next line" note | 3.5 | 50.5 % | — |

  - TranslateGemma 12B had the best meaning (2.58) but was bookish.
  - Sarvam-Translate (4-bit MLX) fell into repetition loops.
  - Gemma 4 12B needs the `gemma4_unified` architecture, which `mlx-lm` 0.31.3 (the latest release) can't load. It would need `mlx-vlm` (MIT), a new dependency.

  **Round 3** (Hy-MT2-7B, Tencent, Apache-2.0, 4-bit MLX, with the model card's prompts and sampling)

  | System | Overall /10 | Mean rank | Meaning | Naturalness | Major meaning errors |
  |---|---|---|---|---|---|
  | Gemma 3 12B, plain | 6.2 | 1.6 | 2.46 | 2.47 | 5 % |
  | Hy-MT2-7B, style prompt (spoken Telugu) | 5.9 | 2.0 | 2.66 | 2.19 | 8 % |
  | Hy-MT2-7B, default prompt | 5.2 | 2.4 | 2.61 | 1.77 | 4 % |

  - Hy-MT2 is more literal: better meaning and grammar, but less natural, and naturalness is the maintainer's main complaint.
  - The first Hy-MT2 run was void: its tokenizer uses about 11 tokens per Telugu word, against about 3 for Gemma, so the harness's token cap cut off most lines. The app's cap is now `max(192, 24 × English words)`, so no model can have a line cut off.

  **Round 4** (Gemma 4 12B, Apache-2.0, `mlx-community/gemma-4-12B-it-4bit@73bcf090`)

  The switch rule was set before scoring. Gemma 4 replaces Gemma 3 only if all three hold against Gemma 3:
  - overall is no more than 0.2 lower;
  - the share of lines with a major meaning error is no more than 2.7 points higher (one line);
  - naturalness is no more than 0.1 lower.

  | System | Overall /10 | First-place votes | Naturalness | Major meaning errors |
  |---|---|---|---|---|
  | Gemma 4 12B, 4-bit | 6.4 | 11 / 15 | 2.31 | 14 % (5 lines) |
  | Gemma 3 12B, plain | 5.7 | 4 / 15 | 2.35 | 5 % (2 lines) |

  - Result: not adopted. Gemma 4 is preferred overall but fails the meaning-error rule.
  - Its errors are dropped or changed numbers and facts (90 became 60, "half" dropped, an age garbled, a count dropped, a wrong ingredient), which is typical of 4-bit loss. The 5-bit build (`@a2fa92fa`, 8.23 GB, the incumbent's footprint) is the next arm under the same rule.
  - Running Gemma 4 on the pinned `mlx-lm` 0.31.3 needs three things:
    - Map `gemma4_unified` onto `gemma4` and drop `vision_embedder` weights, as upstream PR #1349 does.
    - Pass `enable_thinking=False` explicitly, or it thinks.
    - Honour `generation_config.json`'s `suppress_tokens`, or `<image|>` leaks into lines and can end them early.
- **Reopen if:** the maintainer's "latest SOTA that fits" sweep finds a newer model that wins the same blind bake-off, or `mlx-lm` gains `gemma4_unified`.

## ADR-019 — Translation through the Claude CLI; no local LLMs (supersedes ADR-018's model choice)
- **Decision (maintainer, 2026-09-24):** "dont use any local LLM model use claude cli for translating english to telugu which I have subscription for only speech to text and telugu text to speech and voice cloining use local models".
  - All English→Telugu text work (translation, shorter or fuller wordings, coverage review, rephrasing) goes through the `claude` CLI, signed in with the maintainer's own subscription (`claude auth login`).
  - Speech recognition, diarization, TTS and voice cloning stay local. **Only transcript text leaves the machine; audio never does.**
  - The local translators (Gemma 3 12B, TranslateGemma 4B, the CUDA llama.cpp path), their model pins and their extras are retired.
  - CLAUDE.md's "on-device inference only" and "no network except yt-dlp" constraints are amended for this one exception.
- **How the CLI is called** (`engine/src/maata_engine/claude_cli.py`):
  - One sealed `claude -p` process per call, with `--safe-mode` (no hooks, plugins, MCP servers, CLAUDE.md or skills; subscription sign-in still works), `--tools ""`, Maata's own `--system-prompt`, `--no-session-persistence`, `--strict-mcp-config`, `--disable-slash-commands`, `--settings '{"fastMode":false}'`, `--output-format stream-json --verbose` and `--json-schema`.
  - Model ids are full ids, never an alias and never Fable: `claude-sonnet-5` by default, with `--fallback-model claude-opus-5-5` only when the installed CLI can run it.
  - A version gate refuses CLIs older than 2.1.205 (schema validation) and any model the installed version can't run (Opus 5.5 needs 2.1.280).
  - Every call runs in one fixed, empty working directory under the engine's cache dir, so calls share the prompt cache.
  - The prompt goes on stdin. Telemetry, error reporting, `/feedback`, the auto-updater and non-essential traffic are off in its environment.
  - The system prompt goes in argv, so `ps` on this machine shows it. For the scene translator that includes the video's brief: its metadata and, from brief v1 on, the names, terms, idioms and ASR fixes quoted from the transcript. No transcript line goes in argv.
  - Failures are classed as `missing`, `not_signed_in`, `outdated`, `usage_limit` (which limit, and when it resets), `transient`, `stalled`, `timeout`, `bad_output` or `failed`, so the UI can say what to do.
  - A watchdog stops a CLI that shows nothing at start (#91987) or doesn't answer in time: SIGINT, then SIGTERM, then SIGKILL. `transient` and `stalled` calls are retried after an exponential back-off, never at once. An Opus or Sonnet weekly limit moves calls to the other family, including the call that hit it when its retries are used up.
  - A call can be cancelled: its process gets SIGINT (then SIGTERM, SIGKILL), also while it waits to retry. Effort can be set per call (the scene translator uses medium for scene, brief and re-translate calls, low for fit and rephrase calls).
  - A schema-bound reply counts only if it validates against the schema, locally too.
  - Each call reports a usage record for `units.jsonl` through the client's `trace` callback: call type, the model that answered, seconds, and input, cache-read, cache-creation and output tokens. The session wires it into `units.jsonl`.
- **In the pipeline** (`session.py`, 2026-09-25):
  - Scenes of sentence units go to Claude up to the lookahead horizon, up to three calls at once, with no GPU lock: 20–30 s for the first scene after the start or a seek, about 60 s for the second, then up to 150 s and 30 lines, cut at a speaker turn or a pause of 1 s or more.
  - Brief v0 (the video's metadata and the pre-pass talk shares) lets scene 1 start at once; brief v1, made from the pre-pass transcript in the background, takes effect at the next scene boundary.
  - Tiers are asked for by a local length prediction (1.4 aksharas per English syllable, then each speaker's running median) and chosen by the band rule (ARCHITECTURE §4.5).
  - The voicer never waits on Claude. A line whose take still runs long ships as a provisional take, and a rephrase queued in the background replaces it only if it is back in time for the new take to be voiced a minute before the playhead reaches it.
  - A seek cancels the calls outside the new window. A failure the user must fix (not signed in, a usage limit, the CLI missing or too old, a stall) holds new scene calls and reaches the UI as a `claude_error` event, with the reset time for a usage limit; `hello` carries the CLI's state (installed, version, signed in, models).
  - The TTS reads the Telugu-script wording; a `ttsScript: "latin"` setting rebuilds the Latin form from the English-word map for the D6 listening A/B.
- **Local LLMs retired** (2026-09-25): `MLXChatTranslator`, `MLXTranslateGemma`, `StyledTranslator` and `LlamaCppTranslator`, the Gemma entries in `models.lock.json`, the `mlx-lm` and `llama-cpp-python` extras (with `diskcache` and `sentencepiece`, which only they needed), and the local-LLM prompt code in `text/tenglish.py`. ARCHITECTURE D2 and §7 step 1 had gated this on the bake-off and the usage run; the maintainer's decision above retires them now, so those gates decide the model and the thresholds instead, with no local fallback.
- **Why:**
  - Five blind rounds (ADR-018) found no local model that fits 24 GB good enough. The best, Gemma 3 12B, still dropped or changed facts on 5% of lines, and Gemma 4 12B at 4-bit on 14%.
  - Removing the local translator also frees about 8 GB of unified memory and about 1.13 GPU-s per video second, the largest cost in the pipeline (research `gap-1`).
- **Verified on 2026-09-24:**
  - The standalone CLI (2.1.201) signed in with a Max plan.
  - Sonnet 5 and Opus 4.8 translated the 37-line eval set scene by scene with no missing lines, at 1.6 s/line and 1.9 s/line.
  - Haiku 4.5 took 23.5 s/line, and is not used.
  - Opus 5.5 needs CLI ≥ 2.1.280, and Fable 5.1 needs ≥ 2.1.251.
  - The `opus` alias resolved to Opus 4.8 on 2.1.201, so model ids are always pinned explicitly.
- **Policy (personal scope):**
  - Claude Code's legal-and-compliance page (fetched 2026-09-24) allows an end user to sign in to the unmodified CLI with their own subscription. It says products "should use API key authentication", and that developers may not collect or intermediate Claude.ai credentials.
  - Maata spawns the user's own CLI and never touches tokens.
  - Before any distribution, the questions in `docs/research/dubbing-2026-09/ARCHITECTURE.md` §4.10 go to Anthropic.
- **Design:** `docs/research/dubbing-2026-09/ARCHITECTURE.md` §4:
  - scene-batched, sentence-complete, id-keyed units;
  - a Telugu-script `spoken` field plus an English-word map;
  - length tiers chosen locally;
  - a coverage review;
  - a per-line cache;
  - the voicer never waits on Claude.
- **Model bake-off (2026-09-25, ARCHITECTURE §7.1):**
  - 75 original lines in 10 scenes, run through the product `ClaudeTranslator` scene path (Telugu-script `spoken` contract).
  - Each scene was scored blind by 3 raters with different emphases, labels rotated per scene.
  - The rubric is ADR-018's: meaning, grammar, naturalness and English use, each 0–3, plus overall /10.
  - Two judge panels were run, because judges may prefer their own model family: one on Opus, one on Sonnet.

  | Arm | Opus panel overall / naturalness | Sonnet panel overall / naturalness | Lines with a major meaning error (either panel) | Speed | Output tokens (75 lines) |
  |---|---|---|---|---|---|
  | **Opus 5.5, medium** | **8.87 / 2.87** | **8.52 / 2.94** | **0** | 2.9 s/line | 27k |
  | Opus 5.5, low | 8.17 / 2.82 | 8.10 / 2.91 | 0 | 1.6 s/line | 14k |
  | Sonnet 5, high | 7.58 / 2.72 | 8.07 / 2.91 | 2 | 7.1 s/line | 57k |
  | Sonnet 5, medium | 7.17 / 2.67 | 7.39 / 2.84 | 3 | 6.9 s/line | 51k |

  - The pre-set rule: Sonnet 5 is the default and wins ties; Opus 5.5 is chosen only with at least 3 fewer major-error lines or naturalness +0.2, and nothing else worse.
    - The Opus panel alone passes it on naturalness (+0.20).
    - The Sonnet panel alone does not (+0.10), a sign of some self-preference in the Opus panel.
    - Pooled, it passes on errors: 0 against 3 lines.
  - Sonnet's errors were real: an inverted "your dreams are too big for where you come from", and dropped "every time", "around six percent" and "paying for dinner".
  - **Decision:** `claude-opus-5-5` at medium effort is the default, with `claude-sonnet-5` as the fallback. Sonnet 5 covers an Opus weekly limit (the client switches family until the reset) and CLIs older than 2.1.280 (health says to update).
  - **Step-1 quality gate:** passed (0 of 75 major-error lines; naturalness 2.9).
  - **Caveat:** Opus usage comes out of the same weekly Opus allowance as the maintainer's own Claude Code work. The usage run (§4.8, D20) is still owed.
- **Reopen if:**
  - a native listener's review disagrees with the bake-off;
  - the one-hour usage run exceeds the maintainer's plan thresholds;
  - Anthropic's terms change.
- **Amendment (2026-09-25): the coverage review, and the band rule audited** (`docs/research/dubbing-2026-09/ARCHITECTURE.md` §4.2, §4.5, §4.6, §7 step 4; gap-4 E5):
  - **The review.** One call per scene, on its own byte-identical system prompt and schema (`REVIEW_SYSTEM`, `REVIEW_SCHEMA`, `REVIEW_HASH` in `text/scene_prompt.py`). It carries no brief, so every video shares one prompt cache on the review model.
    - In: each line's English, the Telugu of the wording the band rule picks for it, and the scene's context lines.
    - Out, per id: class C (complete), m (minor drop), P (phrase dropped) or E (meaning error); the missing English words; additions; and the kind of an E error (negation, number, name, question, addition, other).
    - Ids a reply lacks are asked for once more. After that the line stays unreviewed; it is never left untranslated.
  - **Model.** `claude-sonnet-5` at high effort, with `claude-opus-5-5` as the fallback; `ClaudeTranslator(review_model=…, review_fallback=…, review_effort=…)` changes it.
    - Why: Opus 5.5 translates the scenes, so a Sonnet reviewer draws on the other family's weekly limit, and its misses are less likely to be the scene model's own (§4.2, inferred).
    - The §7.1 bake-off hasn't chosen it yet; its arms are Sonnet 5 at high effort and Opus 5.5 at low.
  - **Re-translation.** Each P or E line gets one re-translation from its English, with the missing words named. An E line is told what kind of error was found, and is never shown its Telugu.
    - It runs on the scene model at medium effort and passes the scene validators.
    - Claude doesn't review it again. The deterministic checks class it (`qa/coverage.py`). They compare like with like: the re-translation's wording on the tier the review classed, set against the reviewed wording. If the re-translation lacks that tier, its `full` is set against the line's `full`. The classes:
      - E when a negation, a question or a number doesn't match the English. A mismatch the reviewed wording has too doesn't count, unless the review found that very error. Otherwise a heuristic that misses the English (a number said as "a pair") would class every re-translation E.
      - P, for a P line, while the wording is no longer in aksharas than the reviewed one;
      - else C.
    - The better class is kept, and a tie keeps the reviewed wording. `units.jsonl` and `maata-bench` also keep the review's own class, so a C from the checks never counts as a reviewed C.
    - A re-translation goes into the line cache only after the review has judged it, whichever wording is kept. If a seek cancels the review, or a failure the user must fix stops it, before a P or E line's re-translation comes back, the cache keeps the first wording without a class, and the line is reviewed again the next time it is shown. A re-translation that did come back is still judged.
    - A re-translation that is asked for again (its reply lacked it or failed the checks) is still told the review's finding.
  - **In the session.**
    - The review runs in the scene's task, right after the scene call and before the scene's lines count as translated. The voicer never waits on it.
    - A seek that drops the scene drops its review too, and the lines are asked for again.
    - A review that fails leaves its lines unreviewed, but they are still voiced. A failure the user must fix (sign-in, a usage limit) is shown and holds scene calls, as any Claude failure does. A review that goes through clears that banner, as a scene call does; on a re-watch served from the line cache it may be the only call that goes through Claude.
  - **Line cache.** The class is stored with the line in `lines.jsonl`, with who classed it, the tier it classed and the review's first class.
    - A line served from the cache keeps its class and isn't reviewed again.
    - A cached line with no class (made before this change, or its review failed) is reviewed the next time it is shown.
    - A fit that only adds tiers keeps the line's class. A fit that replaces the wording the review classed drops the class: the line counts as unreviewed, and is reviewed the next time it is shown.
  - **Cost, not yet measured.** One more call per scene (about 25 per video-hour on a first showing, §4.8), plus the re-translations. The review also sits between a scene call and its first voiced line, the first scene included, so it adds to time to first audio.
  - **Band rule audit (§4.5).**
    - The "absorbed from above" test now prices a wording as `_plan` places it: with the next line's slot and predicted duration. So the next line's own lag limit applies (1.0 s before a long gap, else 0.6 s); before, it was always 0.6 s.
    - A fit for a line under the band no longer asks for `fuller` when the line's longer wordings overshoot the band. It asks for the tier between the two if the line lacks one, else for the chosen tier again, closer to the slot. When the chosen tier is `full`, which a fit never rewrites, it asks for the longer one instead.
      - Either way, the fit is cut from the nearest longer wording: that wording is the call's `current`, with its overflow, so the tier asked for is shorter than it. A fit copies `current` into `full`, and its tiers are checked against that copy. Given the short pick as `current`, a wording closer to the slot is longer than the copy and would always be dropped.
      - The `fuller` case still starts from the pick. The fit prompt is unchanged: `current` is described as the chosen wording, and the per-line cache (`PROMPT_HASH`) stays valid.
    - A fit never takes a wording that would break the akshara order with a tier the line already had; the line keeps its own.
    - Kept on purpose:
      - a line absorbed from above asks for no fit;
      - fill is measured against speech time;
      - lateness carried from an earlier line's overdraft still counts against a wording from above, so a chain that runs late picks the shorter wording.
  - **Metrics.**
    - A class counts for a line only when it is the class of the wording voiced (`qa.coverage.voiced_class`). The pick can move after the review: a fit adds a tier, the voice's own pace changes the pick, a shorter re-synthesis, or a kept re-translation voiced on another of its tiers. Such lines count as `other_tier`, next to `unreviewed`, and never as C.
    - `units.jsonl`, per voiced line: `coverage` (class, by, tier, first, missing, added, error); `coverage_voiced` (the class as counted, `other_tier` or `unreviewed`); `speech_fill` (seconds played ÷ speech seconds); `required_rate` (natural take seconds ÷ speech seconds). Per scene: its class counts, `other_tier` and `unreviewed`.
    - `maata-bench pipeline`, under `dub`:
      - the coverage shares, and the review's first-pass shares, each with `other_tier` and `unreviewed`;
      - the re-translations kept;
      - the speech fill, summed and as a median;
      - the share of lines whose required rate is in [0.9, 1.2].
  - **Still owed.**
    - Running the gap-4 E5 labels (`gap4/coverage.py`) as a regression fixture. It needs real review calls on the maintainer's transcript, which stays out of the repo.
    - The §7 step 4 acceptance numbers (C + m ≥ 95 %, P ≤ 2 %, E ≤ 1 %, fill 0.90–1.00, required rate in band ≥ 75 %), which need the one-hour run.
