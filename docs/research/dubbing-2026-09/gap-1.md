# Gap 1: can the target pipeline stay ahead of playback on the M5 Pro? (2026-09-24)

## Question

Once every recommended step is in the pipeline, can Maata still dub faster than the video plays on the M5 Pro (24 GB)? What does each stage cost? The answer settles four design choices:
- stream with a lookahead, or render the whole video first (as ElevenLabs does);
- how many takes (N) to synthesize per line;
- whether QA checks every line or only flagged lines;
- whether the per-video brief can wait for a full-video ASR pass.

## Short answer

**Today: no.** The 20:06 session ran at 0.59x real time.
- The trace does support a per-stage split. `units.jsonl` has none, but `engine.log` timestamps every diarization block and ASR chunk.
- All four stages share one GPU lock (`server.py:57`), so the GPU work adds up.
- **The local translator (Gemma 3 12B), not TTS, used about two-thirds of the GPU time.** Translation cost about 1.13 GPU-s per video-s, TTS 0.46, diarization about 0.05 and ASR about 0.04. The sum, about 1.70 s, matches the observed 1.71 s of summed `wall_s` per video second.

**Moving translation to `claude -p` takes it off the GPU.** One take then costs about 0.55 GPU-s per video-s, or about 1.8x real time. That leaves room for roughly 0.25–0.35 s/s of extra GPU work if throughput is to stay at 1.25x or better.

**The target pipeline keeps ahead only if it is shaped for it.** Scenario C in section 4 measures this.
- **Naive version: fails.** Three takes run one after another, plus QA retakes, cost about 1.4 GPU-s per video-s (0.70x). A 60-minute video would need about 25 minutes of wall-clock waiting before playback.
- **Shaped version: works, with a thin margin.** It needs about 0.7–0.9 GPU-s per video-s (1.1–1.4x). Its parts:
  - 2 takes in one batched T3 decode, or 3 with a faster S3Gen;
  - the Telugu ASR check on the CPU;
  - retakes only for lines that fail QA.

**Two things are still unmeasured and could break the shaped version:**
1. **Claude's throughput.** It is a separate pipeline running beside the GPU work, not free time. One serial stream of scene calls could cost anywhere from about 0.1 to 1.0 s of Claude decoding per video second. The spread comes from Claude's unknown Telugu tokenization and from how much each call outputs.
2. **Memory pressure.** Under heavy swap on this machine (from other processes), T3 decoding in this study's bench slowed 1.6–3.2x against ADR-014.

**Recommendation:** stream with a lookahead; don't render the whole video. Use N = 2 batched by default, run QA on every take on the CPU, and build the brief from video metadata plus the first 10 minutes. Details are in section 6.

## Corrections to the premises

1. **"Speech fills 83% of the span (557.8 s of 675 s)."** 557.8 s is the summed *dub* audio (`audio_s`), not source speech.
   - The source units' spans add up to 632.7 s, or 94% of 674.9 s.
   - The dub-to-source ratio is 0.88.
   - Your per-take cost of about 0.39 s of TTS per video-s (0.826 × 0.47) still holds, because it uses dub seconds.
2. **Pre-roll.** (1−r)·T is the *lead*: seconds of dubbed video that must be ready when play starts. The time the user waits is (1−r)·T / r of wall clock.
   - At r = 0.59, a 60-minute video needs 41.8 min of waiting, not 25.
   - The maintainer's test video is 10,506 s (2.9 h, longer than YouTube's own 120-minute auto-dub limit). It would need about 122 min.
3. **The app caps the lead, so it stalls.** It never banks that much.
   - The UI plays once `min(remaining, max(prepareAhead, targetLead), 0.8 × lookahead)` is ready, which is 480 s with the default 600 s lookahead (`app/src/lib/buffer.ts:6`).
   - The engine does no work past playhead + lookahead (`session.py:224`).
   - At r = 0.59, a 480 s lead runs out after 480 / 0.41 ≈ 19.5 min of playback. A stall is certain for any video longer than about 20 minutes.
   - In the 20:06 log, ASR paused at 604 s from 20:11:15 until 20:24:18, which is when the playhead passed about 4 s. So playback began about 17.4 min after `open`, with about 497 s dubbed. This is inferred from the horizon rule, and it is consistent with "playback not smooth".
4. **"GPU time, not memory, is binding."** Half right.
   - Removing the translator frees about 1.1 GPU-s per video-s, the largest single gain.
   - Memory also sets speed through swap (section 3), so the freed 8 GB matters beyond the table in `sota_ranking.md`.
5. **`wall_s` is not a TTS time.** It runs from when the voicer picks up a line to when it is sent, including every wait for the shared lock. Unit 1 took 158.1 s because it queued behind translator, ASR and diarization turns.
6. **#87652's 6.1 s is not the cost of a scene call.** The reporter measured Haiku 4.5 at effort low, with a tiny fixed schema, on Linux arm64.
   - Median process total was 6,056 ms on 2.1.201 and 5,970 ms on 2.1.222.
   - That is a floor per call. A scene call's time is dominated by output length and thinking.
7. **#83859 (about 405 s stall), refining report 05.**
   - The first reporter used WebSearch/WebFetch.
   - A commenter (2026-08-25) saw it with MCP tools instead and concluded it "doesn't seem tool-specific".
   - Both setups had tools and ran under launchd. Maata's calls use no tools. Whether it applies is still unverified.

## Evidence

### 1. Stage costs measured in the 20:06 session (M5 Pro, 2026-09-24)

**Sources:**
- `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`: 98 units, 674.9 s of video, 1,917 English words.
- `~/Library/Logs/Maata/engine.log`, lines 605–738.
- `engine/src/maata_engine/{server.py, session.py}`, read-only.

**How the stages run:** one `asyncio.Lock` is shared by MLX (ASR, translation) and MPS (diarization, TTS), per `server.py:54-57` and ADR-016. So the GPU stages run one at a time, and their costs add.

| Stage | Measured | GPU-s per video-s | How |
|---|---|---|---|
| Diarization, pyannote community-1, 195 s blocks (MPS) | 6.5–8.5 s per block normally; 14.9 s and 24.5 s on two blocks (cause unknown); 68.6 s for 1,080 s covered | 0.036–0.044 typical, 0.064 mean | "diarized … in X s" log lines, timed inside the lock |
| ASR, Whisper large-v3-turbo on MLX with word timestamps, 60 + 5 s chunks | 1.7–3.4 s per chunk, about 2.3 s median | about 0.03–0.05 | gap from the previous lock holder's log line to "heard" |
| Clone plus calibration take | 4.8–7.3 s to clone, 7.7–10.0 s to calibrate, per speaker | one-off, about 15 s per speaker | log timestamps |
| **Translation**, Gemma 3 12B 4-bit, line by line plus candidates (ADR-018) | about 7.6–8.1 s per unit | **about 1.13** | residual, see below |
| **TTS, 1 take**: bf16 T3, S3Gen 6 steps, HiFT, watermark, save; uncontended | wall = 0.85 s + 0.411 × audio_s (r = 0.998, n = 10; units 88–97) | **0.46** | the tail units, after translation had finished |
| Total | | about 1.70, i.e. r = 0.59 | the observed 1,157 s / 674.9 s is 1.71 |

**How the translation residual was derived:**
- **Clean window (20:11:15 to 20:24:15, 780 s).** No ASR or diarization ran; only the translator and the voicer held the GPU.
  - Modelled TTS for units 2–72 was 222.7 s, leaving 557.6 s for about 69 units translated, or about 8.1 s each.
  - On average those units had 21 source words and 92 Telugu characters.
- **Whole session (1,258.5 s from `open` to the last unit).**
  - Deductions: diarization 68.6 s, ASR about 28 s, clones about 30 s, TTS about 319 s.
  - That leaves about 813 s for 107 translated units, or 7.6 s each (about 1.13 s per video-s).
- ADR-016 had already measured that TTS beside an MLX LLM, run concurrently, is slower than taking turns (RTF 0.70 → 2.79 or 3.99), which is why the lock exists.

**Splitting the one-take TTS cost:**
- ADR-014 (8 lines, built-in voice) gives T3 = 0.313 s per audio-s and S3Gen flow plus HiFT = 0.30 s per line + 0.049 s per audio-s (`tts_speed.json`, 2026-09-24 12:44).
- The in-app fit adds about 0.55 s per line + 0.049 s per audio-s for the watermark, render, save and send.
- On this video (0.145 lines and 0.826 dub-s per video-s), that is 0.259 (T3) + 0.084 (flow + HiFT) + 0.120 (per-line overhead) = **0.463 s/s**.
- **T3's autoregressive decode is 56% of TTS.**

**Earlier runs agree:**
- The 13:44 session ran at 0.74x and the 18:31 session at 0.54x.
- `maata-bench` gave 0.71x and 0.69x on 300 s (`bench_e2e.json`, 13:10).

### 2. New scratch bench: N takes in one batched T3 decode (M5 Pro, 2026-09-24)

**Setup:**
- Script: `scratchpad/gap1_batch_takes.py`. Results: `scratchpad/gap1_batch_takes.json` and `.log`.
- The engine's own `_t3_tokens` loop (bf16, CFG, min-p 0.05, repetition 1.2), widened from 2 rows to 2N rows (N conditional + N unconditional). Samples are independent per row, and the loop stops when every take has emitted EOS.
- The 8 Telugu lines from ADR-014, the checkpoint's built-in voice, local weights, `HF_HUB_OFFLINE=1`.
- For each line, the serial run (N separate calls) and the batched run went back to back.

| N | T3 batched ÷ N serial takes (per line; median, range) | Equivalent single takes | ms per step: batched vs serial |
|---|---|---|---|
| 1 | 0.99 (0.64–1.26) | 1.0 | 20.0 vs 20.3 |
| 2 | 0.59 (0.35–0.94); pooled 0.59 | about 1.2 | 22.3 vs 21.1 |
| 3 | 0.46 (0.23–0.66); pooled 0.42 | about 1.3–1.4 | 33.1 vs 27.0 |
| 4 (4 lines only) | 0.35 (0.32–0.84); pooled 0.47 | about 1.4–1.9 | 69.2 vs 41.1 |

**Caveats:**
- The machine swapped heavily during the run: swap went from 9.1 to 16.9 GB used of 17.4 GB, and the 1-minute load average from 6.7 to 9.2. The swapping came from processes this session can't see, since `ps` is sandboxed.
- The run was stopped partway through N = 4 to relieve the machine.
- Absolute speeds are contaminated. T3 ran at 20–41 ms per token, against about 13 ms in ADR-014 (7.7 s for about 592 tokens). Only the back-to-back ratios should be used.
- It is a scratch result: one voice, 8 lines, not committed.

**Findings:**
- **T3 decode is limited by overhead, not by memory bandwidth.**
  - The model is 0.5B, about 1 GB in bf16. At the M5 Pro's 307 GB/s (Apple spec page, fetched 2026-09-24), a step could take about 3 ms; it takes 13–20 ms.
  - The engine loop also forces a host sync every step (`int(nxt) == eos`).
  - This fits the upstream reports:
    - A maintainer-side comment says T3's HF path has "a lot of CPU-GPU sync" (chatterbox #127, 2025-06-07).
    - CUDA graphs gave about 1.6x at batch 1 on a 4090 (#552, 2026-08-21).
    - chatterbox-vllm claims ">10x with batching", after which S3Gen takes 81% of the time (README v0.1.3, undated, fetched 2026-09-24).
- **Answer to "do N takes cost about one?":** yes for N = 2 (about 1.2x); for N = 3 about 1.3–1.4x.
- **S3Gen flow and HiFT are not batched and still cost N times.**
  - Under contention they measured 0.8–1.1 s (flow) and 0.2–0.4 s (HiFT) per take.
  - ADR-014, uncontended, measured about 0.44 s per line for both together.
- **Meanflow (`s3gen_meanflow`, 1–2 steps) was not benched.** It isn't on disk, and fetching it needs approval.
  - The upstream README says Turbo's distilled decoder went "from 10 steps to just one".
  - From ADR-014, cutting 10 → 6 steps saved 0.40 s per line, so a 2-step flow plausibly costs about 0.15–0.25 s per line (INFERRED).
- The engine's own loop never builds `AlignmentStreamAnalyzer`, so it avoids the per-call hook leak that slows repeated multilingual `generate()` calls to 3.46x (chatterbox #352, comment 2026-08-27; fix PR #455 still unmerged).

### 3. Memory pressure is a throughput risk (measured, not explained)

- Same code path: 13 ms per T3 token at 12:44, and 20.3, 27.0 and 41.1 ms per token later as swap climbed. That is 1.6–3.2x slower.
- At 1.6x, even the shaped scenario D (section 4) falls to about 0.8x.
- **Implication:** when `verify-mac.sh` measures real-model runs, it must record `vm.swapusage`, the load average and thermal state. The app should also keep resident models small, which dropping the 8 GB local LLM already helps.

### 4. Target-pipeline GPU budget (inferred from sections 1–2; per video second on the maintainer's podcast)

Every row includes diarization about 0.05 and Whisper ASR about 0.04. Rows C–F also include speaker similarity about 0.01 (estimate). "Retakes" are QA-driven re-synthesis.

| Scenario | GPU-s per video-s | r | Wait before play, 60-min video | Wait, 2.9 h video |
|---|---|---|---|---|
| A: today (Gemma on the lock, 1 take) | 1.70 | 0.59 | 41.8 min | 122 min |
| B: Claude off the GPU, 1 take | 0.55 | 1.81 | start-up only | start-up only |
| C: 3 serial takes, 15% retakes | 1.42 | 0.70 | 25.3 min | 74 min |
| D: 2 takes batched, 15% retakes | 0.78 | 1.28 | start-up only | start-up only |
| E: 3 takes batched, 6-step flow, 15% retakes | 0.92 | 1.09 | start-up only | start-up only |
| E2: E with meanflow at 2 steps (estimated) | 0.79 | 1.27 | start-up only | start-up only |
| F: 1 take; a batched pair only for the 25% of lines that fail QA | 0.71 | 1.41 | start-up only | start-up only |

**Throughput needed to start within the maintainer's 5–10 min:** T / (T + P).
- 60-minute video: r ≥ 0.923 (5 min) or 0.857 (10 min).
- The 2.9 h test video: r ≥ 0.972 (5 min) or 0.946 (10 min).

Aim for r ≥ 1.25, to absorb swap or thermal slowdowns, long lines, seeks and retakes.

**Other stages' GPU costs:**

| Stage | Cost | Basis |
|---|---|---|
| Qwen3-ASR-1.7B, `sota_ranking.md` arm 1 | 0.09–0.14 s/s against about 0.04 for turbo; adds 0.05–0.10 | 7–11x real time, one community benchmark (Casper015, 2026-09-22) |
| Qwen3-ForcedAligner, if used on the GPU | adds a second pass | unmeasured |
| MFA (CPU) | stays off the lock | |
| Nemotron diarization | unmeasured | |
| Telugu QA ASR on the GPU (for example BuzzASR, large-v3 lineage, on MLX) | about 0.1–0.2 s/s at N = 2 | inferred at about 2–4x turbo's cost per second of audio |

- The GPU-based Telugu QA would erase scenario D's margin, so run a GPU checker only on flagged lines.
- **Telugu QA on the CPU (IndicConformer-600M ONNX):** zero GPU time.
  - Its CPU speed is unmeasured: the model card gives no RTF, the weights are gated, and they are not on this disk.
  - A FLOP estimate is 0.1–0.5 s per 5 s take (INFERRED). That is about 0.1–0.4 CPU-s per video-s at N = 2, and the M5 Pro has 15–18 cores.
  - CTC `forced_align`, loudness, EQ and room tone are negligible on the CPU.

### 5. The Claude side is a parallel pipeline with its own throughput (inferred; nothing measured)

**Output volume.**
- The dub text for 674.9 s (98 lines) is 8,641 characters. Tokenized locally:

  | Tokenizer | Telugu dub tokens | English source tokens |
  |---|---|---|
  | cl100k | 11,061 | 2,254 |
  | o200k | 2,645 | 2,166 |
  | Gemma 3 | 2,520 | 2,400 |
  | Qwen3.5 | 4,081 | 2,340 |

- Claude's current tokenizer isn't published for Telugu. For English it uses about 1.8 tokens per word, roughly 1.35x the old tokenizer (Anthropic models overview, fetched 2026-09-24).
- **Per hour of video:**
  - A "lean" design outputs one full wording, short and minimal variants only for the roughly 30% of lines at risk, and a term glossary for the spoken forms: about 29k (o200k-like) to 106k (cl100k-like) output tokens.
  - A "rich" design outputs te, te_short, te_min and a full spoken-form copy for every line, as in reports 05 R1 and 10 R2: about 73k to 253k tokens.
  - Thinking comes on top: it can't be disabled on Opus 5.5 (Anthropic Opus 5.5 page, fetched 2026-09-24).

**Speed.**
- Anthropic publishes only relative latency: Opus 5.5 "Moderate", Sonnet 5 "Fast".
- Artificial Analysis (fetched 2026-09-24; measurement dates not shown; max effort):

  | Model | Output speed | Time to first token |
  |---|---|---|
  | Sonnet 5 | 79.3 tok/s | 150.02 s |
  | Opus 5 | 54.9 tok/s | 46.18 s |
  | Opus 5.5 | no figure | no figure |

- Anthropic says Opus 5.5's output is ">30% faster than Opus 5" (2026-09-22, via report 05). That implies about 71 tok/s (INFERRED).

**Serial decode time per hour of video at 71 tok/s:**

| Design | Minutes per hour | s per video-s |
|---|---|---|
| lean | 7–25 | 0.11–0.41 |
| rich | 17–59 | 0.29–0.99 |

Add about 5–6.5 min per hour for 50–65 calls at the roughly 6 s per-call floor, plus thinking time, which is unknown at low or medium effort. The max-effort time-to-first-token figures show thinking can dominate.

**Per 150 s scene:**
- lean: 15–80 s of decoding;
- rich: 38–192 s. At the rich, cl100k-like end, **one serial stream is slower than real time.**

**Implications:**
- Run 2–3 scene calls at once.
- Keep output lean. A spoken-form *term list* per scene replaces a full Telugu-script copy.
- Make the first scene short (about 60 s) so first audio isn't held up.
- Subscription concurrency and limits are unpublished (support articles updated 2026-09-23), and the transient-throttle message exists. So measure both.

### 6. How ElevenLabs and YouTube schedule dubbing (disclosed)

- **ElevenLabs:**
  - "Realtime or live dubbing is not currently available" (docs, overview/capabilities/dubbing, fetched 2026-09-24).
  - "Dubbing v2 returns a single lossless audio file" (eleven-creative/products/dubbing, fetched 2026-09-24).
  - So the whole file is processed before anything plays.
- **YouTube:**
  - Dubs are generated when a video is uploaded, then published or held for the creator's manual review.
  - Videos over 120 min are ineligible (Help 15569972, fetched 2026-09-24, undated).
- **Both render offline, ahead of any viewer.** Maata can't: the user picks a video and wants it within 5–10 min.
  - At r = 1.3, rendering a 60-minute video up front would take 46 min.
  - So Maata needs streaming with a lookahead. Anything ElevenLabs does over the whole video (brief, glossary, QA) has to become rolling, per scene.

## Answer to the four design choices

1. **Stream or render first?** Stream with a lookahead. The prerequisites, in order:
   1. Translation off the GPU lock. This is already decided.
   2. Batched takes.
   3. QA on the CPU.
   4. A voicer-first lock policy. The lock is FIFO today, and one voicer line can queue behind every other stage up to four times.
   5. The lead formula fixed to wall-clock terms.
   6. Keep working past the lookahead while paused, when r < 1.25.

   Also offer "prepare the whole video" as an explicit mode for long videos; the PCM is already cached per unit.
2. **N:**
   - 2 takes in one batched T3 decode by default (about 1.2x one take).
   - 3 only for short lines, or as the retake after a QA failure.
   - Never 3 run one after another (scenario C, 0.70x).
   - Revisit N = 3 by default only if meanflow benches at about 0.2 s per line or less (scenario E2).
3. **QA gate:**
   - Every take goes through the CPU selector: IndicConformer CER plus CTC alignment plus WeSpeaker similarity. This fits only if IndicConformer runs at about RTF 0.1 or less on the M5 Pro CPU, which must be measured.
   - GPU-based or second-family checks run only on flagged lines: cap hit, akshara/duration mismatch, similarity outlier, CER over threshold.
4. **Brief:**
   - **Don't block on the full video.** Full-video Whisper ASR is cheap, about 0.035 s/s: about 2 min for a 1-hour video and about 6 min for the 2.9 h one. Then the brief call adds about 1 min or more (roughly 18k input tokens per hour of English).
   - Waiting fits a 5–10 min budget only for videos up to about an hour.
   - **Instead:**
     1. Make brief v1 from the title, description and chapters plus the 10-minute pre-pass transcript.
     2. Run full ASR in the background.
     3. Swap in brief v2 at a scene boundary. Each change costs one prompt-cache rewrite.

## Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Today's split: translation about 1.13, TTS 0.46, diarization about 0.05, ASR about 0.04 s/s | High | Log-derived; the parts reconcile with the observed total to within 1%; one session, one video |
| One-take TTS about 0.46 s/s | High | Fit r = 0.998, and consistent with ADR-014 |
| Batched-take ratios (N = 2 about 1.2x, N = 3 about 1.3–1.4x) | Medium-low | Scratch, heavy swap, 8 lines, one voice, not committed |
| The scenario table | Medium | Inferred arithmetic on measured parts. Retake rate (15–25%), similarity cost and meanflow cost are assumptions |
| The Claude side | Low | No calls made: the installed CLI is 2.1.201, and Opus 5.5 needs 2.1.280 or later. Updating is a configuration change this study did not make. Claude's Telugu tokenization and speeds at low or medium effort are unknown |
| IndicConformer CPU cost | Low | FLOP estimate only |
| **Overall: "keeps ahead only if shaped"** | Medium | |

## What Maata should do

1. **Instrument before building.**
   - **Per unit, in `units.jsonl`:**
     - `lock_wait_s`;
     - `t3_s`, `t3_tokens`, `t3_steps`;
     - `flow_s`, `hift_s`, `watermark_s`;
     - `takes`, `qa_cpu_s`, `cer`, `sim`;
     - `translate_ready_at`.
   - **Per block, as their own events:** `{"event":"diar",a,b,secs,lock_wait}` and `{"event":"asr",a,b,secs,words}`.
   - **Per Claude call:** `{"event":"claude",scene,model,effort,secs,ttft,input,cache_read,cache_creation,output_tokens}`, taken from the stream-json `usage`.
   - **Per session:** a summary of GPU-busy seconds by stage, throughput, the lead over time, and stalls.
   - **In `verify-mac.sh` JSON:** `vm.swapusage`, the load average and thermal state.
   - Then commit a `verify-mac.sh` run. Needs no approval.
2. **Commit a clean batched-takes bench.**
   - Engine-integrated N ∈ {1, 2, 3} on the maintainer's cloned voices, on an idle machine.
   - Alternate the order of serial and batched runs.
   - Score each take for CER and similarity, to confirm batched takes are as good as serial ones.
   - Needs no approval; the scratch script is a starting point.
3. **Fix pacing.**
   - The wait before playback should be lead ÷ r in wall-clock time.
   - Let the engine keep working past the lookahead while paused if r < 1.25.
   - Give the voicer priority on the GPU lock whenever the lead is under about 60 s.
4. **Time real `claude -p` scene calls.** Once the maintainer approves updating the CLI to 2.1.280 or later:
   - Opus 5.5 at low and medium effort, and Sonnet 5 at medium.
   - 60 s and 150 s scenes, lean and rich schemas, 1–3 concurrent calls.
   - Log output tokens per Telugu character.
   - This decides the lean-or-rich design and the concurrency.
5. **Bench on the M5 Pro CPU and GPU:**
   - IndicConformer-600M ONNX on the CPU for a 3–5 s line. Needs the HF gate accepted and a pin.
   - `s3gen_meanflow` at 1–2 steps. Needs a fetch approval and a pin.
   - These two decide between scenario D, E2 and F as the default.

## Sources (all fetched or read 2026-09-24 unless dated)

**Local, M5 Pro:**
- `~/Library/Caches/Maata/4Vz6L8B73i4/units.jsonl`: sessions at 13:44, 18:31 and 20:06.
- `~/Library/Logs/Maata/engine.log`, lines 605–738.
- `scratchpad/tts_speed.json` (12:44; ADR-014 data).
- `scratchpad/bench_e2e.json` (13:10).
- `scratchpad/gap1_batch_takes.{py,json,log}` (this study).

**Repo (HEAD 7cfa760, read-only):**
- `engine/src/maata_engine/server.py:54-57`;
- `session.py` (constants at lines 47–60; `_dub`; `_horizon`);
- `backends/torch_common.py:205-282`;
- `pacing.py`;
- `app/src/lib/buffer.ts:6`;
- `docs/DECISIONS.md`: ADR-014, ADR-016, ADR-018.

**GitHub (via `gh`):**
- resemble-ai/chatterbox:
  - #127 (2025-06-06; comment 2025-06-07);
  - #552 (2026-08-21);
  - #352 (2025-11-12; comment 2026-08-27);
  - PR #455 (2026-02-01, open).
- anthropics/claude-code:
  - #87652 (2026-08-18; closed as stale 2026-09-21);
  - #83859 (2026-08-04, open; comment 2026-08-25);
  - #91987 (2026-09-04, open).
- github.com/randombk/chatterbox-vllm README (v0.1.3, undated).
- github.com/resemble-ai/chatterbox README (undated).

**Vendors:**
- apple.com/macbook-pro/specs: M5 Pro, 307 GB/s.
- elevenlabs.io/docs/overview/capabilities/dubbing and /docs/eleven-creative/products/dubbing.
- support.google.com/youtube/answer/15569972.
- huggingface.co/ai4bharat/indic-conformer-600m-multilingual: no speed figures; onnxruntime 1.20.1.
- artificialanalysis.ai/models/claude-sonnet-5 and /claude-opus-5 (measurement dates not shown).
- Anthropic docs saved in `scratchpad/cc_docs` (models overview; Opus 5.5 and Sonnet 5 pages).

**Earlier reports:**
- 01 (F1), 05 (A1, A9, B1, E1), 06 (B8, B9, R4), 10 (R1–R3), `sota_ranking.md` §2.
- Petrov et al., NeurIPS 2023, via report 05.
