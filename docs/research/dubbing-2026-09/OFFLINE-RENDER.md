# Maata: the offline whole-video render

Status: design, 2026-09-25. Replaces the real-time (streaming) pipeline of waves 1–2 (`session.py`, ARCHITECTURE §3.12, §5.2–5.3, ADR-016) with a render that works on the whole video first and plays a finished dub. Everything else in ARCHITECTURE.md stands: the models, the Claude contract (§4), timing v2 (§3.10) and the voice lane (§3.7).

Numbers marked **(measured)** come from the maintainer's run on video `4Vz6L8B73i4` (numeric fields of `units.jsonl` and `lines.jsonl` only) or from the wave reports. Numbers marked **(estimate)** are built from those numbers and have not been run. Nothing here has run on the M5 Pro.

## 0. The request and what went wrong

**The maintainer (2026-09-25):**
> "Okay there are only 2 speakers in the video I gave but identified more speakers. instead of doing this realtime lets do this for the whole video, first extract the audio and transcript and generate suitable translation then dub it with sync to original video"

**His run** (a two-person podcast, 10,506 s = 2 h 55 min; streaming build of wave 2):
- **Speakers.** The 10-minute pre-pass found the right 2 speakers and cloned each from 60 s of clean speech.
  - The 3-minute diarization blocks after it found 2 local speakers in 28 of the 35 blocks, 1 in two and 3 in five **(measured)**.
  - Clusters that failed to link to the registry became new speakers. S5 and S6 were cloned from 11.1 s and 8.6 s of audio **(measured)**.
  - Cause: a 3-minute block gives a local cluster only seconds of speech, so its centroid is noisy. `SpeakerRegistry.add_block` then either links it (threshold 0.5) or founds a new speaker.
- **Throughput.** Only 300 s were dubbed before he stopped.
  - 57 lines took 1,761 s of wall time. The voicer waited 1,141 s for the GPU while diarization and ASR ran ahead to 6,300 s and 6,067 s.
  - Stages sharing the machine doubled the TTS cost: T3 ran at a median of 27.5 ms/token, against 12.7 ms/token across all his runs **(measured)**.

**What the offline shape fixes by construction:**
- Diarization sees the whole file at once. pyannote clusters every 10 s chunk's embedding globally, with a speaker-count control, so there is nothing to link.
- Stages no longer compete for the GPU: each runs alone, at its idle speed.
- The translation brief, the voices and the timing plan all see the whole video before they commit.

## 1. Shape

```
prepare(url, options)
  │
  ├─ fetch ─────── audio.m4a (yt-dlp) → render/audio16k.npy (float32, memory-mapped)
  ├─ speakers ──── pyannote on the WHOLE file (auto 1–6, or the user's count) → settle → render/diarization.json
  ├─ transcript ── Whisper over the whole file, 60 s chunks, punctuation transfer + coverage guard → render/transcript.jsonl
  │     └─ brief ── built part by part as the transcript grows (≈45 min per part); complete when it is
  ├─ units ─────── sentence units over the whole transcript (derived; recomputed, never stored)
  ├─ voices ────── per speaker, from the best clean speech anywhere in the video → render/voices.json
  ├─ translate ─── all scenes of the range, 3 lanes, review + re-translation (coverage gate) → lines.jsonl (existing cache)
  │     └─ voice lines ── N=2 takes per line as soon as its scene is reviewed → render/takes/*.npz + takes.jsonl
  ├─ timing ────── one global plan over the timeline (actual take durations), fix-ups, re-plan → render/plan.json
  ├─ mix ───────── per line: squeeze/parts → speaker gain → peak limit → watermark → render/pcm/<key>.npy
  ├─ manifest ──── render/manifest.json (the playback contract)
  └─ export ────── (on request) dub-only WAV
watch(videoId) → manifest → muted IFrame + SyncEngine; PCM read from disk near the playhead
```

- **One GPU stage at a time.** Order: speakers, then transcript, then voices, then voice lines. Claude runs beside the GPU (it never takes the GPU).
- **Every stage writes to disk as it goes.** A crash, a pause or an app restart resumes at the first stage whose output is missing or stale (§2.1).
- **Nothing is played until the range is rendered.** The original audio is never played; the export holds dub speech only.

## 2. Stages

### 2.1 The job, the cache layout and fingerprints

**One `RenderJob` per video** (`engine/src/maata_engine/render.py`). The engine owns it, not a WebSocket connection, so a render survives a UI reload. The engine runs one job at a time; others wait in a queue.

**Cache layout.** Existing files in `<cache>/<video_id>/` stay where they are. The render's own files go in `render/`:

| File | What | Written |
|---|---|---|
| `audio.m4a` | yt-dlp's audio (existing) | fetch |
| `lines.jsonl`, `briefs.jsonl` | the translator's line and brief caches (existing, unchanged format) | translate, brief |
| `units.jsonl` | trace (existing), plus `stage` events | every stage |
| `render/job.json` | settings, status, per-stage progress and timings | every progress change (atomic) |
| `render/audio16k.npy` | mono float32 16 kHz analysis audio, opened with `mmap_mode="r"` | fetch |
| `render/diarization.json` | whole-file turns, exclusive turns, centroids, merges, talk time | speakers |
| `render/transcript.jsonl` | one row per ASR chunk: words and guard results | transcript (appended) |
| `render/voices.json` | per speaker: spans used, reference hash, calibration takes, pace, overhead | voices |
| `render/takes.jsonl` + `render/takes/<take_key>.npz` | per line, the chosen take of each wording or piece | voice lines (appended) |
| `render/plan.json` | the global plan, per line | timing |
| `render/pcm/<pcm_key>.npy` | final line PCM, 24 kHz float16 (`_save_pcm`: atomic) | mix |
| `render/manifest.json` | the playback contract (§2.12) | manifest (atomic) |

`job.json`:
```json
{"version": 1, "videoId": "…", "title": "…", "channel": "…", "duration": 10505.66,
 "settings": {"speakers": null, "style": "colloquial", "range": [0, 900], "speedCap": 1.2,
              "allowFreeze": true, "ttsScript": "telugu", "gentle": false},
 "status": "running", "stage": "voice_lines",
 "stages": {"fetch": {"state": "done", "done": 162.0, "total": 162.0, "unit": "MB", "seconds": 41.2, "gpu_s": 0.0},
            "speakers": {"state": "done", "done": 1, "total": 1, "seconds": 512.0, "gpu_s": 498.1}, "…": {}},
 "rendered": [[0, 900]], "error": null, "createdAt": 0, "updatedAt": 0}
```
- `status` is one of `queued | running | paused | waiting | failed | done`. `waiting` means Claude is held back by a usage limit or a sign-in problem while GPU work goes on.
- A stage's `state` is one of `todo | running | done | failed`.

**Fingerprints.** Each stage file starts with the inputs it was made from, written as `"inputs": {…}`. A stage re-runs when the inputs it would use now differ; otherwise it is served from disk.

| Stage | Inputs |
|---|---|
| speakers | audio fingerprint (size and mtime of `audio.m4a`, plus the sha256 of its first 1 MB); pyannote commit (`models.lock.json`); count hint or `"auto"`; bounds; settle constants; `DIAR_VERSION` |
| transcript | audio fingerprint; Whisper commit; `CHUNK`, `CHUNK_PAD`; a hash of `STYLE_PROMPT`; `ASR_VERSION`. **Not the speaker count:** a re-diarization changes labels, not where speech is, so ASR is reused |
| voices | the speaker's spans (via its reference hash); TTS settings; `CALIBRATION_TE` |

**Content-addressed caches.** Lines, takes and PCM files are keyed by content:
- lines by the existing `line_key` (English, speaker, style);
- takes by `take_key = sha256([tts_text, voice key, n, cfm_steps, TAKES_VERSION])[:20]`;
- PCM by `pcm_key = sha256([take keys, rounded plan fields, gain_db, MIX_VERSION])[:20]`.

So a re-run touches only what changed.

**Cancel and resume.**
- `RenderJob.cancel()` cancels the stage tasks. Running Claude calls get SIGINT through `ClaudeTranslator.aclose()`. A GPU hold ends after the line in progress.
- Everything completed is on disk, so `resume()` is `run()` again.
- Unit ids are the index of a unit in onset order over the whole transcript. They are deterministic for a given transcript and diarization.

### 2.2 Fetch

- `resolve.py`: `Resolver.resolve(ref, cache_dir, download=True, progress=None)`. With `download=False` it returns metadata only; the prepare view uses this (§5). `progress(done_bytes, total_bytes)` comes from yt-dlp's `progress_hooks`. yt-dlp stays pinned, with no remote components and never `-U`.
- `RenderJob._fetch()`: decode once with `decode_audio`, write `render/audio16k.npy`, then keep `self.audio = np.load(…, mmap_mode="r")`. A 2.9 h video's 672 MB then stops counting as resident memory until it is touched.
- `DemoResolver` keeps its synthetic 180 s video.

### 2.3 Speakers: diarize the whole file once

**Backend API.** `Diarizer.diarize(audio, *, num_speakers=None, min_speakers=None, max_speakers=None, progress=None) -> DiarBlock` replaces `diarize_block` (backends/base.py). It diarizes the whole file at offset 0.
- `PyannoteDiarizer.diarize` calls the pipeline once on the whole waveform with those arguments (pyannote 4.0.7 `SpeakerDiarization.apply` takes `num_speakers` / `min_speakers` / `max_speakers`).
  - Its `hook(step, artifact, completed=, total=)` drives `progress` for the "segmentation" and "embeddings" steps.
  - It returns turns, exclusive turns and `speaker_embeddings` as centroids, exactly as `diarize_block` does now.
- `SingleSpeakerDiarizer.diarize` returns one turn.
- `MockDiarizer.diarize` honours `num_speakers`. In auto mode it returns speakers A and B in 14 s turns, plus a 4 s blip labelled C near 100 s whose embedding is close to A's. The mock thereby reproduces the maintainer's failure, and the settle step (below) must fix it.

**Count control.**
- Auto: `min_speakers=1, max_speakers=6` (the UI's range).
- With the user's count: `num_speakers=k`. pyannote's VBx then re-clusters with KMeans to exactly k.

**Settle** (new pure function `speakers.settle(registry, *, min_talk=30.0, min_share=0.01, same_cos=0.85, hinted=False) -> list[tuple[str, str, str]]`):
- The registry is built with one `SpeakerRegistry.add_block(block)` over [0, duration]. Every clip-selection and turn query of `speakers.py` is reused unchanged.
- In auto mode, `settle` merges speakers using `SpeakerRegistry.merge`:
  - a speaker whose exclusive talk time is under max(`min_talk`, `min_share` × total talk) merges into the speaker nearest by centroid cosine, or, with no centroid, into the speaker it shares the most adjacent speech with;
  - any pair whose centroids have cosine ≥ `same_cos` merges.
- With a user count (`hinted=True`) nothing merges: the user's number wins.
- It returns `(from, into, why)` with why = `talk` or `same`.
- Ids are then renumbered S1..Sk by first speech, so the same two people get the same ids run after run. That keeps line-cache keys stable.

**Report.** `speakers.activity(registry, sid, duration, bins=120) -> list[float]` gives each speaker's share of each of 120 slices of the video, for the UI's activity strip.

**`diarization.json` format:**
```json
{"inputs": {…}, "labels": {"SPEAKER_00": "S1"}, "turns": [["S1", 3.1, 9.8]], "exclusive": [["S1", 3.1, 9.8]],
 "centroids": {"S1": [256 floats]}, "merged": [["S3", "S1", "talk"]],
 "speakers": [{"id": "S1", "talkSeconds": 5120.3, "share": 0.52, "firstAt": 3.1, "turns": 812}]}
```
On resume, the registry is rebuilt from this file with `add_block` and the recorded merges (deterministic).

**Changing the count.** `set_speakers {videoId, speakers}` re-runs this stage with the new hint. Downstream:
- units are recomputed from the cached transcript;
- lines whose speaker changed miss the line cache and are translated again;
- a speaker whose reference spans changed gets a new voice key, so their takes are made again;
- everything else is reused.

The render does **not** stop for the user after diarization. The speakers card appears during the transcript stage, and a correction made before voice lines start costs only the diarization re-run (§7).

### 2.4 Transcript

- `RenderJob._transcript()` reuses the frontend's per-chunk logic unchanged:
  - `CHUNK` 60 s + `CHUNK_PAD` 5 s;
  - `speech` spans for the coverage guard from `registry.turns_in(…, exclusive=False)`;
  - `fix_words`, then `_sentence_cut`, then `_asr_checks`.
  - It runs from 0 to the end of the video whatever the range, because the brief needs the full transcript.
- Each chunk appends one row to `transcript.jsonl`:
```json
{"a": 0.0, "b": 58.4, "next": 58.4, "language": "en",
 "words": [["Hello", 0.6, 0.9, 0.98]], "checks": {"redecoded": [], "recovered": 0, "rejected": 1, "punctuated": true}}
```
- Resume continues at the last row's `next`. A torn last line is ignored, as `_Jsonl` does.
- The `asr` trace event is kept.
- Progress: video seconds transcribed / duration.

### 2.5 Sentence units

- `RenderJob._units()` is pure and derived. It is never stored; it is recomputed on every run from `transcript.jsonl` and the registry.
  - Per chunk: `segment(owned, turns, prev_text=…)`, as `_frontend` does.
  - Then one `merge_fragments` pass over the whole list, so a sentence split across a chunk cut rejoins.
- Unit ids are assigned after that pass. The list is indexed by onset (a sorted list with `bisect`), so the per-line queries (`_slot`, `_ahead`, `_heard_in`, `_context_for`, `_after`) cost O(log n + k), not O(n).
  - The streaming versions scan every unit. At about 1,900 lines, the plan alone would make tens of millions of Python iterations.
- `UnitState` loses `chunk`. It gains `take` (the cached `Said` record) and `pcm` (the final file).

### 2.6 Brief from the full transcript

- The brief is complete before any scene is translated. It is built with the existing `ClaudeTranslator.make_brief(meta, transcript, previous)`:
  - `meta` is the yt-dlp metadata plus `talk_shares` from the whole-file diarization;
  - the transcript is `(speaker, text)` per unit.
- **Parts.** One call if the transcript is ≤ `BRIEF_PART_WORDS` (8,000 words, about 45 min of talk). Otherwise v1 is made from the first part and v2, v3, … are diffs from each later part, each passing the brief so far as `previous`.
  - Why parts: the CLI's per-call timeout is 240 s (`claude_cli.py`), and a 2.9 h transcript is about 28k words.
  - Each part starts as soon as the transcript passes its end and the previous part's brief is back, so only the last part waits after ASR.
  - Parts are cached by message hash (`briefs.jsonl`).
- `use_brief(final)` is called once, before the first scene. There is no v0 → v1 swap.

### 2.7 Translation: all scenes, full context, reviewed

- **Scenes.** The units of the range are cut up front with `scene_cut(run, nth=2, final=True)`, repeatedly: up to 150 s and 30 lines, at a speaker turn or a pause ≥ 1 s. The short latency scenes (`LEAD_SCENE`, `SECOND_SCENE`) are dropped.
- **Three lanes.** The scene list is split into 3 contiguous blocks. Each lane translates its block in order, one scene at a time: `_scene(req)` (scene call, then review, then one re-translation of P/E lines, all existing code).
  - So every scene except the first of each lane gets the previous scene's chosen Telugu as `context_before`, which is what "full context" means here. Only two scenes of a video start with English-only context, where a seek used to cause it.
  - 3 lanes is the same concurrency as today, and within the translator's semaphore of 3.
  - Lane 1 starts at the range start, so voice lines can begin within a few minutes.
- **Fits.** `_fit_spec` / `_fit` are unchanged: low effort, tiers added to the line.
- **Failures.**
  - `_claude_failed` / `_claude_ok` are unchanged: back-off, and waiting for a usage-limit reset.
  - A lane whose scene was released (`_release`) retries that scene after the hold.
  - The job's status becomes `waiting` while GPU work goes on.
- **Coverage gate.** When all lanes are done:
  - scenes with unreviewed lines are submitted again: cached lines cost no scene call, and only the missing reviews run;
  - lines never translated get one line-by-line attempt.
  - What remains is reported: counts of C/m/P/E/unreviewed/skipped in `job.json` and the manifest.
  - Skipped lines are listed in the manifest. They are never padded and never English (§4.9).
  - The render does not block on the gate.
- **Cache reuse.** The maintainer's `lines.jsonl` has 776 distinct lines: 644 made under brief v1 (642 reviewed) and 132 under brief v0 **(measured)**.
  - `_cached` serves the 644 whenever text and speaker match. The 132 v0 lines are asked for again.
  - Decision M1 (§11) is whether to keep the 644 or re-translate them under the full-transcript brief.

### 2.8 Voices from the whole video

`RenderJob._voices()` runs right after units, on the GPU, while the brief is on Claude. Per speaker, most talk first:
- **One build path** (ARCHITECTURE §3.7 step 3). Every speaker goes through `prepare_voice_parts`:
  - timbre: `registry.best_span(sid, 10, min_len=8)` (after the first minute), else the stitched `reference_clips` trimmed to 10 s;
  - identity: `clean_clips(sid, max_total=60)` **spread over the video**, taking the best clip from each of 6 equal time slices in turn, so one stretch's microphone doesn't dominate.
  - A speaker with under `REF_MIN` (4 s) of clean speech gets a preset voice. The settle step makes this rare.
- **Pooled S3Gen x-vector** (new in `torch_common.prepare_voice_parts`): average `s3gen.speaker_encoder.inference` over the same clips and replace `gen["embedding"]`. T3's speaker embedding is already averaged over them.
- **Calibration:** the three `CALIBRATION_TE` sentences, as now. Their `(spoken, seconds)` are stored in `voices.json`.
- **On resume:**
  - the voice is rebuilt from the stored spans (4–8 s per speaker **(measured)**, build_s);
  - the duration estimator is re-fitted from the stored calibration takes with `DurationEstimator.calibrate`, with no synthesis;
  - then the estimator's `observe` is replayed from `takes.jsonl`.

### 2.9 Voice lines: quality-first takes

**Worker loop.** The earliest line (by onset) whose scene is reviewed and that has no take row is voiced next. Per line:
1. `_choose(st, key)`, the band rule on predicted durations.
2. `_say(st, words, voice, key, seconds, cost, n)` with a **fixed** n:
   - `TAKES_N = 2` in one batched decode, 3 for lines with under 3 s of speech;
   - a third seed when every take fails (`qa.take`), with no budget;
   - pieces at hard breaks as timing v2 decides (`_pieces`).
3. Store the chosen take of each wording or piece:
   - `ChatterboxTeluguTTS.pack_take(take) -> dict[str, np.ndarray]` holds the flowed mel as float16 plus `n_tokens`;
   - `MockTTS` stores its samples;
   - both also keep the natural audio as float16 and its pauses.
4. Append a `takes.jsonl` row:
```json
{"unit": 412, "line": "<line_key>", "tier": "full", "voice": ["S2", 0.5, 0.5, "<ref hash>"],
 "takes": [{"key": "<take_key>", "seconds": 4.12, "pauses": [[1.8, 2.1]], "pace": 4.05, "failed": null}],
 "cost": {"t3_s": 1.9, "flow_s": 0.9, "render_s": 0.2, "takes": 2}}
```

**What goes.** `_takes_n`, `_retake_ok`, the throughput governor, the provisional-take path and the `unit` messages. The `unit` trace event is written at mix time instead, when its plan is final.

**Order.** Units are voiced in onset order within what is translated. Lane 1 feeds the start of the range first.

### 2.10 One global timing plan

`RenderJob._timing()` starts once every line of the range has a take (or is skipped):
1. **Plan.** A fresh `TimelinePlanner(PlannerSettings(speed_cap, max_freeze, freeze_budget))` places every line in onset order with the existing `_plan(st, said)`.
   - `_predicted(x)` now returns the line's **actual** take duration when it has a take, and the estimate otherwise.
   - So each placement's windowed lookahead (the next line, then up to 8 lines within 30 s) prices real audio. That is the difference from streaming, where the lines ahead were guesses.
2. **Fix-ups.** For lines with `plan.needs_shorter`, up to `FIXUP_SHARE` = 15 % of lines:
   - (a) a shorter tier the line already has, voiced and kept if it passes and is shorter (the `_dub` logic, moved);
   - (b) otherwise, **one batched `rephrase` request per scene** (the `rephrase` call type, with no deadline), then the most complete wording predicted to fit (`_replace`'s rule), voiced.
3. **Re-plan** from scratch. At most two rounds.
4. Write `plan.json`: per line, the `Plan` fields (`start`, `rate`, `wall`, `lag`, `freeze`, `freeze_at`, `parts`, `said`, `voiced`, `overdraft`, `needs_shorter`).

The planner is pure and deterministic. A re-plan after extending a range, or after a fix-up, gives the same plan for every line it doesn't affect, so their PCM keys don't change.

### 2.11 Mix: per-speaker loudness, then the watermark last

New pure module `engine/src/maata_engine/mix.py` (numpy only; no new dependency):
- `gated_rms_db(x, sr) -> float`: 400 ms blocks, an absolute gate at −70 dBFS, a relative gate 10 dB under the mean of the blocks kept.
  - It is BS.1770's gating without K-weighting, so it reports dBFS, not LUFS. pyloudnorm is locked only as the Chatterbox fork's dependency; using it directly needs approval (M5).
- `speaker_gains(src_db, dub_db, talk, target=-20.0, spread=6.0, limit=12.0) -> dict[str, float]`:
  - each speaker's dub is brought to target + (their source level − the talk-weighted mean source level), with that offset clipped to ±`spread`;
  - so the speakers keep the source's balance, and a badly recorded speaker isn't copied down to a whisper;
  - the gain is clipped to ±`limit`.
  - `src_db` is measured on `clean_clips(sid, max_total=120)` of the analysis audio, and `dub_db` on the speaker's natural takes.
- `limit_peak(x, ceiling_db=-1.0) -> np.ndarray`: scales a line down if its 4× oversampled peak (numpy FFT interpolation) passes the ceiling.
- `mix_timeline(lines, duration, sr) -> np.ndarray` and `write_wav(path, x, sr)` (stdlib `wave`, 16-bit) for the export.

**Per line,** in `RenderJob._mix()`:
1. `audio = _play(said, plan)`: squeeze and cut, parts at their times. A rate ≠ 1 re-vocodes the cached mel.
2. `audio *= 10 ** (gain_db / 20)`.
3. `limit_peak(audio)`.
4. `tts.watermark(audio)`: the PerTh watermark, now the **last** step (ARCHITECTURE §3.8, §3.11).
   - `vocode(take, rate, watermark=True)` keeps its default for the streaming path until step 6 deletes it. The render calls it with `watermark=False`.
   - `MockTTS` has no watermark.
5. `_save_pcm(render/pcm/<pcm_key>.npy)`.

Existing files are skipped. The `unit` trace event (the same fields as today) is written here.

### 2.12 The render manifest

`render/manifest.json` is written atomically after the mix. It is the only thing playback reads.
```json
{"version": 1, "videoId": "…", "title": "…", "channel": "…", "duration": 10505.66,
 "range": [0, 10505.66], "ranges": [[0, 10505.66]], "sampleRate": 24000,
 "settings": {"style": "colloquial", "speedCap": 1.2, "allowFreeze": true, "ttsScript": "telugu", "speakers": "auto"},
 "speakers": [{"id": "S1", "label": "Speaker 1", "talkSeconds": 5120.3, "voice": "cloned", "referenceSeconds": 60.0,
               "gainDb": -1.4, "pace": 6.5}],
 "lines": [{"id": 0, "speaker": "S1", "start": 3.12, "end": 7.85, "srcStart": 3.10, "srcEnd": 7.85,
            "audioRate": 1.04, "audioWall": 4.61, "lag": 0.02, "freeze": 0.0,
            "edits": [], "said": "whole", "tier": "full", "voice": "cloned",
            "source": "…", "telugu": "…", "coverage": "C", "flags": [], "pcm": "pcm/3fa2…c1.npy", "samples": 110640}],
 "skipped": [{"id": 12, "start": 40.1, "end": 41.0, "why": "not translated"}],
 "stats": {"lines": 1900, "coverage": {"C": 0.0, "m": 0.0, "P": 0.0, "E": 0.0, "unreviewed": 0.0},
           "lag_p95": 0.0, "rate_p90": 0.0, "freeze_s_per_10min": 0.0, "fixups": 0}}
```
- Each line carries the `DubUnit` fields the UI already uses (`start`, `end`, `audioRate`, `audioWall`, `lag`, `freeze`, `edits`, `voice`, `source`, `telugu`). Playback reuses `SyncEngine` as it is.
- `stats` reuses `timing_metrics` / `dub_metrics` from `bench.py`.

### 2.13 Export

- `export {videoId}` → `RenderJob.export_wav(dest)`. It places each manifest line's final PCM at its `start` on the video clock (`mix_timeline`: overlapping tails are summed, then `limit_peak`), and writes a 24 kHz mono 16-bit WAV.
- It reads only `render/pcm/*.npy`, never `audio16k.npy` or `audio.m4a`, so the original audio cannot get in (a test asserts this, §9).
- A freeze can't hold a picture in a WAV: that line's tail runs into the gap after it (≤ 0.6 s per hold, ≤ 1 s per minute, under ADR-017's metering).
- Destination: `~/Downloads/Maata/<title> (Telugu dub).wav` by default (M8). The UI shows the path.
- Size: about 504 MB for 2.9 h **(estimate)**.

## 3. Range renders and extension

- `settings.range = [a, b]` (the prepare view's options: whole video, first 15 minutes, or a custom from–to).
  - Fetch, speakers, transcript and brief always cover the **whole** video: the brief needs the full transcript, and the speaker count needs the whole file.
  - Translate, voice lines, timing and mix cover the units that start in [a, b).
- The manifest's `ranges` is [[a, b′]], where b′ is the end of the last line placed. Watching outside it shows the "Not dubbed here yet" card (§5).
- **Extend** is a `prepare` with a larger range. Every cache applies:
  - speakers, transcript and brief: no work;
  - translate: only scenes past b′; lane boundaries are re-cut over the remaining scenes;
  - voice lines: only lines without takes;
  - timing: a full re-plan. It is cheap, and lines near the old boundary now see their real neighbours;
  - mix: only lines whose `pcm_key` changed, which are about the last few lines before b′ and everything new.
- **Speakers or style changed.** Speakers: §2.3. Style: `line_key` includes it, so lines are translated again; takes and PCM follow.

## 4. Progress, cancel, resume, library: the WebSocket protocol

**Stages reported** (`render.STAGES`), each with `done` / `total` in its own unit:

| key | label | unit | progress source |
|---|---|---|---|
| fetch | Download | MB | yt-dlp hook |
| speakers | Speakers | fraction | pyannote hook (segmentation 40 %, embeddings 50 %, clustering 10 %) |
| transcript | Transcript | video s | per chunk |
| brief | Video brief | parts | per part |
| voices | Voices | speakers | per speaker |
| translate | Translation | lines | reviewed lines / lines in range |
| voice_lines | Telugu speech | speech s | speech seconds voiced / in range |
| timing | Timing | lines | placed lines, per round |
| mix | Mix | lines | files written |

**ETA.**
- The running stage: remaining work ÷ its measured rate, once it has 30 s of work behind it.
- Later stages: remaining work × a per-stage rate from `<cache>/render-rates.json`. That file is a per-machine EWMA of seconds per unit, updated when each stage finishes. It starts from the priors in §7.
- The overall ETA is the running stage's ETA plus the later stages' ETAs. Translate overlaps voice lines, so they count as max(translate, voice lines).
- Every stage writes a `{"event": "stage", "key", "seconds", "gpu_s", "done", "total"}` trace row when it finishes.

**Client → engine** (new or changed):

| message | effect |
|---|---|
| `inspect {url}` | metadata only (`resolve(download=False)`) → `video` |
| `prepare {url, speakers: "auto"\|1..6, style, range: [a, b]\|null, speedCap, allowFreeze, ttsScript, gentle}` | create or resume the job; queued if another runs |
| `pause {videoId}` / `resume {videoId}` | cancel at a safe point / run again from disk |
| `set_speakers {videoId, speakers}` | re-run from speakers (§2.3) |
| `set_voice {videoId, speaker, usePreset}` | re-run from voices for that speaker (its takes re-keyed) |
| `renders {}` | → `renders` |
| `watch {videoId}` | → `video`, `manifest` |
| `audio {ids}` | PCM frames of those lines (existing) |
| `seek {time}` | re-send frames of the lines in [t − 5, t + 180) (existing behaviour, new window) |
| `speed {speed}` | re-send the window's frames time-stretched (existing `_send_audio`) |
| `export {videoId}` | → `exported` |

**Engine → client** (new or changed; binary frames unchanged, ADR-007):

| message | payload |
|---|---|
| `video` | existing fields + `thumbnail`, `render` (a `renders` item or null), `estimate: {seconds: [lo, hi], claudeCalls: [lo, hi]}` for the options chosen |
| `render` | `{videoId, status, stage, stages: [{key, label, state, done, total, unit, seconds, eta}], range, eta, elapsed, coverage?, error?}`, at most 2/s and on every state change, **to every connected client** |
| `speakers_found` | `{videoId, mode: "auto"\|"hint", bounds, speakers: [{id, label, talkSeconds, share, firstAt, turns, activity[120]}], merged: [{from, into, why}]}` |
| `renders` | `{items: [{videoId, title, channel, duration, status, range, ranges, stage, eta, updatedAt, bytes}]}`, built by scanning `<cache>/*/render/job.json` |
| `manifest` | the manifest, minus `settings` |
| `exported` | `{videoId, path, bytes}` |
| `claude_error` / `claude_ok` | unchanged, plus `videoId` |

**Server** (`server.py`):
- `Engine.jobs: RenderQueue` holds jobs by video id, one running at a time.
- `Engine.clients` is the set of connected senders; `render` and `speakers_found` go to all of them.
- A connection's `finally` no longer closes the job; it closes only that connection's `Playback` (below).
- **Keep-awake:** while a job runs on macOS, the engine holds `/usr/bin/caffeinate -i -w <engine pid>` and kills it when the job pauses or ends (M7).
- **Playback:** `playback.py` `Playback` holds a manifest, `user_speed` and the send callbacks.
  - `ask(ids)` reads `render/pcm/<key>.npy` with the existing `_send_audio`.
  - `seek(t)` sends the window [t − 5, t + `WATCH_WINDOW` = 180).
  - `set_speed(s)` re-sends that window stretched.

## 5. The UI flow (ADR-009's look)

The same glass cards over the animated backdrop, the turmeric-to-vermilion accent, Inter and Noto Sans Telugu, the "AI dub" badge.

**1. Home** (`Hero.svelte`):
- The headline stays. "Continue watching" becomes **Your dubs** (`Library.svelte`): cards with the thumbnail, title, a status chip and the watch position.
  - Chip examples: "Ready · 2:55:06", "Preview ready · 0:00–15:00", "Rendering · Telugu speech 43 % · about 2 h 10 min left", "Paused at Translation", "Waiting for Claude · resets 18:00".
- A card opens Watch (done) or Progress (otherwise).

**2. Prepare** (`Prepare.svelte`, after a pasted link; `inspect`):
- A video card: thumbnail from `i.ytimg.com` (already allowed by the CSP), title, channel, duration.
- Options:
  - **Speakers:** a segmented control Auto · 1 · 2 · 3 · 4 · 5 · 6;
  - **Style:** Everyday spoken / More formal (`STYLE_OPTIONS`);
  - **Range:** Whole video · First 15 minutes · Custom (from–to);
  - "More options": speed-up cap, allow freeze, TTS script (the existing controls, moved from `SettingsSheet`), and "Keep my Mac responsive (slower)" (`gentle`).
- An estimate line from `video.estimate`: "About 3–6 h on this Mac · about 190 Claude calls".
- The primary button **Prepare dub**. A video with a render shows its status and **Continue** / **Watch** instead.

**3. Progress** (`Progress.svelte`):
- A vertical stepper of the stages, grouped for the eye as Download → Speakers → Transcript & brief → Translation → Telugu speech → Timing & mix. Each row shows state, done/total, time taken and ETA; the running row has a thin gradient bar.
- The overall ETA and elapsed time.
- **Pause** / **Resume**, and the existing `ClaudeBanner` when Claude waits.
- **After speakers** (`SpeakersFound.svelte`):
  - "Found 2 speakers": per speaker an avatar in its `speakerHue`, talk time and share bar (the SidePanel speaker row's look), and an activity strip of 120 bins across the video;
  - a line for merges ("1 short voice merged into Speaker 1");
  - "Wrong count?" with the same Auto/1–6 control and **Re-run speakers** (`set_speakers`), which says what it costs ("about 8 min; nothing translated is lost").
  - Speakers are identified by talk time and timing only. **No audio sample is played**: the original audio is never played.
- When the range is done: **Watch**. A preview also shows **Extend to the whole video**.

**4. Watch** (`Stage.svelte` + a trimmed `SidePanel.svelte`):
- `watch` → `manifest` → `app.units`, then each line goes to `sync.addUnit`, and `missing.add(all spans)`. `reaskMissing` / `MissingAudio.ask` fetches PCM within `ASK_AHEAD` (90 s) and `evict` drops audio outside [t − 60, t + 180].
- `readyRanges` = `manifest.ranges`. `needSeconds` in watch mode = min(remaining − `END_SLACK`, `WATCH_NEED` = 10 s), with no `targetLead`.
- Outside the rendered ranges, `bufferStep` stalls as today. The prep card then reads "Not dubbed here yet", with **Extend** and **Back to the dub**.
- `SyncEngine`, `VideoClock`, freezes and the muted player are unchanged.
- The side panel keeps "Now speaking" and Speakers (voice status; the preset switch now asks "Re-voice Speaker 2 with a stock voice? Re-renders their 930 lines, about 50 min" and sends `set_voice`).
- The side panel gains a **Render** card: coverage shares, lag p95, freezes, and **Export dub track (WAV)**. It loses Pipeline, the lead meter and "Prepare the whole video".

**State** (`state.svelte.ts`):
- `view: "home" | "prepare" | "progress" | "watch"`, `renders`, `job` (the `render` message for the open video), `found` (`speakers_found`), `manifest`.
- Pure helpers go in a new `lib/render.ts`: stage grouping, `fmtEta` reuse, the estimate text, chip text. They are tested with vitest.

## 6. The streaming machinery: what to delete, what to keep

The rule: keep every pure piece and every per-line method, and delete everything whose only job was to beat the playhead.

| Piece | Fate | Why |
|---|---|---|
| 10-min pre-pass (`PREPASS`, `_prepass_heard/_done`, `speaker_scan`, `_clone_all`) | delete | the whole file is diarized first |
| 3-min blocks (`DIAR_BLOCK/OVERLAP/AHEAD`, `_diarizer`), block linking | delete the loop; **keep `SpeakerRegistry`**, fed by one `add_block` | its clip selection, turn queries and `merge` are what voices, units and timing use |
| `_clone_later/_clone_new`, clone on first line (`_voice_for` cloning) | delete | every speaker is known before voicing |
| Frontend loop, `Chunk`, `_pass_over`, `ReadyRanges`, `_mark_ready`, `_chunk_ready` | delete; **keep** `_sentence_cut`, `_asr_checks`, `CHUNK`, `CHUNK_PAD` | ASR runs to the end once |
| Translator loop: `_next_scene`, `_run_next`, `_fresh_at`, `LEAD_SCENE`, `SECOND_SCENE`, `SCENE_URGENT`, `BANK_SHARE`, seek cancellation | delete; **keep** `scene_cut` (nth ≥ 2), `CONTEXT_*`, `_scene`, `_review`, `_release`, `_fit`, `_claude_failed/_ok` | lanes over a fixed scene list |
| Brief v0 → v1 swap at a scene boundary | delete | the brief is final before scene 1 |
| Lookahead horizon (`_horizon`, `_play_horizon`, `_listen_until`, `_frontier`, `_caught_up`, `lookahead`) | delete | no playhead while rendering |
| Throughput governor (`pacing.Throughput`, `target_lead`, `takes_for`, `SLOW_R`, `_takes_n`, `_recovering`, `RETAKE_SHARE`, `_retake_ok`) | delete | takes are fixed, quality-first |
| `gpu.py` `GpuScheduler` | **keep**, used at one priority | one GPU owner still holds if a speaker re-run overlaps a voice-line hold; tested, 85 lines |
| Async rephrase with a deadline, provisional takes, `_replace`, `_shortened`, `REPHRASE_LEAD` | replace with the synchronous fix-up pass (§2.10); keep the `rephrase` call type and `_replace`'s wording rule | the whole timeline is known |
| Engine-side PCM window (`raw`, `_saving`, `_in_window`, `_evict_raw`, `KEEP_BEHIND`, `_resend_near`, `_send_audio_for`) | replace with `Playback` over the manifest; **keep** `_send_audio` | PCM lives on disk from the mix on |
| Protocol `open`, `playhead`, `prepare`, `player` rates, `ready`, `status`, `unit`, `unit_skipped`, `speaker_scan` | delete (the manifest carries units and skips); **keep** `seek`, `audio`, `speed`, `claude_*`, binary frames | |
| UI pacing (`targetLead`, `throughput`, `prepareAll`, `etaSeconds`, `LOOKAHEAD_USE`, the look-ahead and prepare-ahead settings) | delete; **keep** `MissingAudio`, `clipLead`, `contiguousLead`, `startDecision`, `bufferStep`, `SyncEngine`, `unitSpan` | watch-mode windowing is the same machinery |
| `TimelinePlanner` (rolling horizon, `place/evaluate/reset`, the seek-back rebuild) | **keep** | placed in onset order with actual durations, it is the global plan; `reset` and the rebuild are harmless |
| `bench.realtime_metrics` | replace with `render_metrics` (per-stage seconds, GPU-s per video-s, peak RSS, Claude calls and tokens from `claude` events) | |

**Least risky diff.** Build beside the streaming path, then delete it:
1. Move the per-line code from `session.py` into `dubber.py` (`class Dubber`). This is a pure move; `Session(Dubber)` keeps working, and every existing test passes.
2. Build `render.py` (`RenderJob(Dubber)`) and the new protocol.
3. Switch the UI.
4. Delete `Session`'s streaming parts, `pacing.py`'s governor and their tests. Port the line-level tests to `Dubber`.

## 7. Budgets on the M5 Pro for the 2.9 h video

**Measured inputs** (his run, 2026-09-25, streaming, stages interleaved):

| What | Measured | Note |
|---|---|---|
| Diarization | 393 GPU-s over 35 blocks covering 0–6,300 s | 0.058 GPU-s per s of block audio; uncontended first blocks 5.9–6.5 s per 195 s = 0.031 |
| ASR (2 passes + re-decodes) | 870 GPU-s over 114 chunks covering 0–6,067 s | 0.143 GPU-s per video-s; uncontended first chunks 3.0–6.6 s per 65 s = 0.046–0.10 |
| Voice build + calibration | 7.4–7.9 s + 34 s per speaker (contended); 3.7–4.2 + 19–21 s (earlier open) | 3 takes each |
| TTS | 57 lines, 266 s of Telugu for 301 s of video: T3 244 s (7,576 steps, median 27.5 ms/token), flow 150 s, vocoder 39 s | 1.63 GPU-s per s of Telugu audio; T3 median over all his runs 12.7 ms/token |
| Claude scene calls | 37 calls, 776 lines, median 103 s, 484k output tokens (624/line) | Opus 5.5 medium |
| Claude review | 35 calls, 751 lines, median 34 s, 122k output tokens (163/line) | Sonnet 5 high |
| Claude fit / retranslate / brief | 33 (median 8.7 s, 38k out) / 11 (9.4 s) / 2 (21.5 s) | |
| Line density | 776 lines in 0–4,290 s = 0.18 lines/s | whole video ≈ 1,900 lines **(estimate)** |

**Wall time per stage, whole video, one stage at a time (estimate):**

| Stage | Work | Wall | Basis |
|---|---|---|---|
| Fetch | audio already cached; decode 2.9 h | 0.5–1 min | PyAV decode unmeasured |
| Speakers | 10,506 s, whole file | 6–11 min GPU + 1–3 min CPU clustering | 0.031–0.058 GPU-s/s; AHC unmeasured at this size |
| Transcript | 10,506 s | 8–25 min | 0.046–0.143 GPU-s/s |
| Brief | 3–4 parts | 1–2 min after ASR (the rest overlaps it) | 12–67 s per measured brief call, larger input |
| Voices | 2 speakers | 1–1.5 min | measured per speaker |
| Translate | ≈1,260 new lines (644 cached): ≈50 scene, ≈55 review, ≈50 fit, ≈17 re-translate calls on 3 lanes | 40–60 min, **overlapped** with voice lines | measured medians ÷ 3 |
| Voice lines | ≈9,400 s of Telugu (0.89 × 10,506), N=2 batched | **2.2–4.7 h** | idle ≈0.83 GPU-s per audio-s (T3 at 13 ms/step × 1.2 for N=2, flow and vocoder at half their contended cost); contended 1.63, +0.18 for N=2 |
| Timing | 1,900 lines, ≤2 rounds; fix-ups ≈3–5 % of lines (his runs re-synthesized 10 of 433 lines and made 2 provisional) | 5–15 min | planner CPU < 1 min with the onset index |
| Mix | vocode sped-up lines + DSP + watermark, 1,900 files | 10–25 min | render_s 0.69 s/line contended |
| **Total** | | **≈3–6.5 h** | voice lines dominate |

- **First 15 minutes as a preview:** speakers + transcript + brief + voices (about 17–43 min, all reused later), then about 800 s of Telugu (11–24 min), then timing and mix (1–2 min). About **30–70 min** **(estimate)**.
  - Extending to the whole video then costs about 2.5–5.5 h.
  - This is the price of the request: nothing is heard before the whole-file diarization, transcript and brief exist. The streaming build played within minutes, with the wrong speakers.

**Peak memory (engine; the WebView adds 1–2 GB) (estimate):**

| Phase | Resident |
|---|---|
| Speakers | runtime 2–3 GB, Whisper 1.6 GB and Chatterbox 2.2–3.2 GB (all loaded at start), waveform tensor 0.67 GB, AHC distance matrix 1–4 GB (condensed float64 over n ≈ 16–31k chunk embeddings: 10,506 one-second chunks × 1.5–3 active local speakers): **8–13 GB** |
| Transcript | 7–9 GB (Whisper peaks about 2 GB on 65 s chunks) |
| Translate + voice lines | 6–8 GB, plus 3 Claude CLIs at 0.2–0.4 GB each; one line's PCM in memory at a time |
| Mix / export | + ≤ 0.5 GB (the export timeline as float32; written in 10-minute blocks) |

- Target: ≤ 13.5 GB (ARCHITECTURE §5.1). Only the diarization transient can pass it, so step 1 measures it on the 2.9 h file.
- Fallback if it fails: `DIAR_WINDOW` = 3,600 s. pyannote runs per hour, and the hour clusters are merged by centroid AHC under the same count control. Each cluster then has minutes of speech, not seconds, unlike the 3-minute blocks. This is an approved-fallback decision (M9).

**Claude, the rest of his video (estimate from measured per-line rates):**
- About **180–200 calls**: ≈50–60 scene, ≈55–60 review, ≈50 fit, ≈17 re-translate, 3–4 brief, ≈5–10 batched rephrase.
- About **1.1M output tokens**: scene ≈0.79M, review ≈0.21M, fit ≈0.06M, the rest ≈0.03M. Cache reads plus writes are about 1.3M.
- From scratch (≈1,900 lines): about 270–300 calls and 1.6M output tokens.
- Opus 5.5 usage comes out of the maintainer's weekly Opus allowance (ADR-019 caveat). A 5-hour-window limit makes the job `waiting` and GPU work goes on.

**How the machine stays usable for hours:**
- One GPU stage at a time: no MLX and MPS contention, which doubled T3's cost in his run.
- Memory stays at 6–9 GB outside diarization:
  - analysis audio memory-mapped;
  - PCM streamed to disk line by line;
  - watch mode holds 4 minutes of PCM.
- **Gentle** (opt-in): after each voice-line GPU hold, sleep 0.4 × its length, about 30 % slower, leaving GPU time for the user's own work.
- Pause at any time; everything done is kept.
- `caffeinate -i` stops idle sleep while a job runs (setting, M7).
- Claude stays at 3 CLI processes, mostly waiting on the network.

## 8. Wave-3 items folded in

Only those the offline shape makes natural and cheap:
- **Voice from the whole video with a pooled speaker embedding** (§3.7 step 3):
  - one build path for every speaker;
  - identity clips spread over the whole video;
  - the S3Gen x-vector pooled over them.
  - Not folded: span choice by centroid cosine (it needs the embedding model exposed), the +1 st shift, and cfg per voice (listening tests).
- **Per-speaker loudness** (§3.11): speaker gains and a peak ceiling, numpy only.
  - Not folded: the per-line ±3 dB, EQ, re-reverb and room tone (D5).
- **Watermark at the end of the DSP chain** (§3.8): applied once per final line.
  - Detection on the engine output and on the export is verified on the M5 Pro (the perth detector is already installed with the fork). It is recorded as an ADR-014 amendment.
- **Claude usage in check:**
  - the line cache (644 of his lines);
  - full 150 s / 30-line scenes;
  - brief parts cached by message;
  - reviews only for lines without a stored class;
  - rephrases batched per scene;
  - re-runs that re-translate only lines whose text or speaker changed;
  - the call estimate shown before "Prepare dub".

## 9. Tests (mock backend, no models, no Claude)

The mock backend must render the demo video end to end offline. `DemoResolver`'s 180 s video, `MockDiarizer.diarize`, `MockTranscriber`, `MockSceneTranslator` over `MockClaude`, and `MockTTS` with no mel (takes store samples; the rate goes through `wsola`).

**Engine:**
- `tests/test_speakers.py`:
  - `test_settle_merges_short_talk_into_nearest`;
  - `test_settle_merges_near_identical_centroids`;
  - `test_settle_keeps_every_speaker_when_hinted`;
  - `test_settle_renumbers_by_first_speech`;
  - `test_activity_bins_sum_to_talk`.
- `tests/test_mix.py`:
  - gated RMS on a tone at a known level (±0.2 dB);
  - gains keep the source balance and clip at ±6 / ±12;
  - `limit_peak` holds −1 dBFS on an oversampled peak;
  - `mix_timeline` places lines at their starts and sums overlaps;
  - `write_wav` reads back with stdlib `wave`.
- `tests/test_render.py` (new):
  - `test_demo_renders_end_to_end`: every stage done, the manifest validates, each line's `pcm` exists with `samples` frames, the lines cover 0–180 s in order, the auto speaker blip is merged (2 speakers), `MockClaude.calls` > 0, no process spawned;
  - `test_cancel_then_resume_reuses_every_stage`: cancel during voice lines, then resume. Counters on `MockDiarizer`, `MockTranscriber`, `MockTTS` and `MockClaude` show 0 diarize and 0 transcribe calls, and TTS only for lines without takes;
  - `test_range_then_extend_reuses_caches`: [0, 60) then the whole video: diarize and transcribe once; lines well inside [0, 60) keep their `take_key` and `pcm_key`;
  - `test_speaker_hint_reruns_from_speakers_only`: `speakers=1` → diarize again, transcribe not; lines that kept their speaker cost no Claude call;
  - `test_fingerprint_change_invalidates_downstream` (the transcript survives a speaker change; voices don't);
  - `test_global_plan_uses_actual_durations` (`_predicted` returns `take_s`);
  - `test_fixups_use_shorter_tier_then_batched_rephrase` (one rephrase call per scene);
  - `test_watermark_is_last` (a fake TTS records the order: gain, limit, watermark, once per line);
  - `test_export_contains_only_dub_pcm` (the export is the sum of the line PCMs; its correlation with the demo's analysis audio is ≈ 0, and the function never opens `audio16k.npy`);
  - `test_torn_rows_are_ignored_on_resume` (`transcript.jsonl`, `takes.jsonl`).
- `tests/test_engine_ws.py`:
  - `test_prepare_reports_progress_then_watch_serves_manifest`: `render` messages walk the stages to `done`; `watch` → `manifest`; `audio {ids}` → frames with valid headers; `seek` re-sends the window;
  - `test_render_survives_reconnect`;
  - `test_pause_and_resume`;
  - `test_renders_lists_jobs`;
  - `test_set_speakers_reruns`.
- `tests/test_models_bench.py`: `maata-bench pipeline FILE --backend mock` runs the render and prints `render`, `dub` and `timing` blocks.

**UI (vitest + svelte-check):**
- `render.test.ts`: stage grouping, ETA and estimate text, chip text per status;
- `buffer.test.ts`: watch-mode `need`, and a stall outside the rendered ranges;
- `state.test.ts`: `view` transitions on `video`, `render`, `manifest`.

**Demo:** `uv run maata-engine --backend mock --demo --ui ../app/dist --token demo`, then paste any link → Prepare → the progress stages → Watch plays the demo dub.

## 10. Implementation plan

Six sequential steps. Each leaves the suites green. Steps 1–4 build beside the streaming app, step 5 switches the app to offline, and step 6 deletes streaming.

1. **Whole-file speakers and the job skeleton.** §2.1–2.5: `Diarizer.diarize` (pyannote, single, mock), `speakers.settle` / `activity`, `render.py` (`RenderJob`: fetch, speakers, transcript, units, `job.json`, fingerprints, cancel and resume), `resolve(download=, progress=)`. The pyannote memory check (§7) is a live item.
2. **Shared line code and translation.** Move the per-line code into `dubber.py` (a pure move). Add §2.6 (brief parts) and §2.7 (lanes, review, coverage gate, cache reuse) to `RenderJob`.
3. **Voices and voice lines.** §2.8 (one build path, spread clips, pooled x-vector, calibration stored) and §2.9 (fixed-N takes, the take cache, resume).
4. **Global plan, mix, manifest, export, ranges, bench.** §2.10–2.13, §3, `mix.py`, the watermark last, `maata-bench pipeline` on `RenderJob` with `render_metrics`.
5. **Protocol and UI.** §4 and §5: the engine-owned queue, broadcast progress, library, `set_speakers` / `set_voice`, `Playback`; the Library, Prepare, Progress, SpeakersFound and Watch views.
6. **Delete streaming.** §6: the streaming `Session`, the governor, the old protocol and UI pacing; tests ported; ADR-020; `verify-mac.sh`; the demo checked end to end.

## 11. Decisions for the maintainer

- **M1 — Cached translations from the streaming runs.** Reuse the 644 reviewed lines made under the pre-pass brief (saving about a third of the Claude work on this video), or translate everything again under the full-transcript brief. Recommend reuse, with a "Translate again" option later.
- **M2 — Speaker settle defaults.** Auto bounds 1–6. In auto mode, merge speakers under max(30 s, 1 % of talk), and pairs with centroid cosine ≥ 0.85. A chosen count disables merging. A real cameo under 30 s would be voiced by the nearest speaker. Approve, or change the floor.
- **M3 — Speaker check.** The render goes on after diarization, and a correction re-runs it; the alternative is to stop and wait for confirmation. Recommend going on: a multi-hour job shouldn't wait for an absent user.
- **M4 — Quality-first synthesis budget.** N=2 batched takes per line (3 under 3 s of speech), a third seed on failure, and fix-ups on up to 15 % of lines, against today's 10 % `RESYNTH_SHARE` (ARCHITECTURE D8).
- **M5 — Loudness measure.** Numpy gated RMS (no new dependency), or pyloudnorm LUFS: it is locked only as the Chatterbox fork's dependency, so a direct import needs approval.
- **M6 — The watermark moves to the end of the chain** (an ADR-014 amendment), with detection checked on the M5 Pro.
- **M7 — Keep-awake.** Spawn `caffeinate -i` while a job runs (default on, a setting)?
- **M8 — Export destination and format.** `~/Downloads/Maata/`, 24 kHz mono 16-bit WAV?
- **M9 — Diarization memory fallback.** Approve one-hour windows linked by centroid AHC if the 2.9 h whole-file run passes 13.5 GB.
- **M10 — Source audio retention** (gap-6 Y12). Delete `audio.m4a` and `audio16k.npy` when a whole-video render finishes, and fetch again for a later re-run, or keep them until the render is deleted. Recommend deleting.
- **M11 — Delete real-time mode entirely** (no watching while it dubs), per the request. Confirm.
- **M12 — Record the change as ADR-020.** It supersedes ADR-016's staged streaming and pacing, and amends ADR-017 (the rephrase path) and ADR-019 (the in-pipeline section). The CLAUDE.md status line and the demo command are the maintainer's to update.

## 12. Risks

- **Whole-file pyannote on 2.9 h is unmeasured.**
  - The AHC distance matrix is 1–4 GB transient.
  - The segmentation and embedding time is extrapolated from 3-minute blocks.
  - VBx may still over-split a long file. The auto merge and the user's count cover that, and the fallback is M9.
- **Hours before the whole dub plays.** 3–6.5 h for his video, and 30–70 min even for a 15-minute preview. Thermal throttling on a multi-hour run can stretch it further.
- **Voice-line cost is the least certain number.** The idle estimate (0.83 GPU-s per audio-s) rests on T3 at 13 ms/token without contention, and on flow and vocoder costs halved from a contended run. The batched N=2 cost is unmeasured on MPS (wave 2 report).
- **Brief quality over parts.** The diff briefs (v2+) are unproven at this size. The 240 s CLI timeout forces parts on long videos.
- **Claude usage in one sitting.** About 190 calls and 1.1M output tokens may hit a 5-hour window. The job waits and GPU work continues, but the finish time moves.
- **Cache staleness after code changes.** Every stage fingerprint must carry a version constant (`DIAR_VERSION`, `ASR_VERSION`, `TAKES_VERSION`, `MIX_VERSION`; `PROMPT_HASH` already exists), or a resumed render mixes old and new behaviour.
- **Unit ids change after a speaker re-run.** Content keys protect the files, but the UI must reload the manifest, and the trace ids of the two runs differ.
- **Line-cache hit rate on his video is unmeasured.** Whole-transcript segmentation may cut some sentences differently from the 60 s-chunk run.
- **Disk.** About 1.7–2 GB per 2.9 h video (mels 75 MB, natural takes and final PCM about 450 MB each, analysis audio 672 MB). The library shows sizes.
- **Watermark detection after the chain and after export mixing** is unverified (M6).
- **Deleting streaming removes the only fast path to a first listen.** If the preview wait proves too long, the lever is a faster first stage (for example, a transcript-first order for the range), not a return to linking blocks.
