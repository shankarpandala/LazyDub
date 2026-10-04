# Maata: the background dub (a whole video in, a dubbed MP4 out)

Status: design, 2026-09-25, revised the same day after an engineering and a product review (Appendix A), retargeted on 2026-10-03 to the maintainer's new request: a background job that saves a finished, dubbed MP4 instead of playing the dub in the app (Appendix B lists what changed), and revised the same day after two reviews of the retarget (Appendix C). It replaces the real-time (streaming) pipeline of waves 1–2 (`session.py`, ARCHITECTURE §3.12, §5.2–5.3, ADR-016) and the in-app player (ADR-006's IFrame, ADR-008's sync). Everything else in ARCHITECTURE.md stands: the models, the Claude contract (§4), timing v2 (§3.10), the voice lane as built (§3.7 steps 1–2, ADR-017) and the watermark in `vocode` (ADR-014).

Steps 1–5 of the first plan are built in the worktree (`dubber.py`, `render.py`, `playback.py`, the job protocol in `server.py`): §2.1–§2.13 describe them, with what changes marked **Changes (2026-10-03)**. §2.14–§2.18 are new. The plan is §10.

Numbers marked **(measured)** come from the maintainer's run on video `4Vz6L8B73i4` (numeric fields of `units.jsonl` and `lines.jsonl` only), from the wave reports, from the reviewers' CPU-only scipy probe, or from the checks named where they are quoted (published benchmarks, PyAV probes run on this Mac on 2026-10-03). Numbers marked **(estimate)** are built from those and have not been run. Nothing here has run on the M5 Pro.

## 0. The requests

**2026-09-25:**
> "Okay there are only 2 speakers in the video I gave but identified more speakers. instead of doing this realtime lets do this for the whole video, first extract the audio and transcript and generate suitable translation then dub it with sync to original video"

His run (a two-person podcast, 10,506 s = 2 h 55 min; streaming build of wave 2):
- **Speakers.** The 10-minute pre-pass found the right 2 speakers. The 3-minute diarization blocks after it found 2 local speakers in 28 of the 35 blocks, 1 in two and 3 in five (at 1425, 2685, 3765, 3945 and 5565 s); clusters that failed to link became new speakers, cloned from 11.1 s and 8.6 s of audio **(measured)**. A 3-minute block gives a local cluster only seconds of speech, so its centroid is noisy.
- **Throughput.** 57 lines took 1,761 s of wall time; the voicer waited 1,141 s for the GPU while diarization and ASR ran ahead. T3 ran at a median of 27.5 ms/token, against 12.7 ms/token across all his runs **(measured)**.

**2026-10-03:**
> "I want this app to dub the complete video in the background to clone, match, synthesize etc and give the final output video instead of playing in the app"

and "if required, use https://huggingface.co/shankarpandala/chatterbox-telugu for cloning" (it already is the cloning model, pinned in `models.lock.json`). His answers to the follow-up questions:
- **Background sound:** the original music and effects, separated on the device by a source-separation model, with the English speech removed and the Telugu voices on top. "The original audio is never played" is relaxed for music and effects only (CLAUDE.md, amended).
- **Audio tracks:** Telugu only.
- **Subtitles:** two soft subtitle tracks in the MP4: Telugu (the dub's text at the dub's times) and English (the source transcript at its times).
- **Scope:** "ignore the licenses part, I am not intrested in only for personal viewing only and as much as local except for the claude cli llm for translating to general public speaking english". Licences don't constrain any choice; everything runs locally except the Claude CLI; the Telugu is the everyday speech of the general public.

**What the shape fixes by construction:**
- Diarization sees the whole file at once: pyannote embeds each active speaker of every 10 s window (a 1 s step) and clusters all of those embeddings together, with a speaker-count control, so there is nothing to link.
- Stages no longer compete for the GPU: each runs alone, at its idle speed.
- The brief, the voices and every scene's translation see the whole video's speakers and transcript before a line is voiced.
- The output is a file, so nothing has to beat a playhead: the dub keeps the video's own timing, and a job can run overnight with the window closed.

## 1. Shape

```
prepare(url, options) → queued → runs when the GPU is free (one job at a time)
  │
  ├─ fetch ──────── audio.<ext> + thumbnail (yt-dlp) → render/audio16k.f32 (memory-mapped)
  ├─ speakers ───── pyannote on the WHOLE file (auto 1–6, or the user's count) → settle → stable ids → diarization.json
  │                 → speakers_found: the speaker check (non-blocking, notified)
  ├─ transcript ─── Whisper over the whole file, 60 s chunks, punctuation transfer + coverage guard → transcript.jsonl
  ├─ units ──────── sentence units over the whole transcript (derived; recomputed, never stored)
  ├─ voices ─────── per speaker, clips chosen from the whole video → voices.json              ┐ side by side
  ├─ brief ──────── from the full transcript, in parts → briefs.jsonl                          ┘ (GPU | Claude)
  ├─ separate ───── Mel-Band RoFormer (MLX) → music & effects bed, 60 s blocks → render/bed/  ┐ side by side: the
  ├─ translate ──── every scene of the range, 3 lanes, reviewed → lines.jsonl                   │ GPU in this order
  ├─ voice_lines ── the dub loop (waits for separate): N takes per line, fix-ups → takes        │ (separate, then the
  ├─ finish ─────── final plan trailing the loop → line PCM render/pcm/ → manifest.json         │ dub loop), Claude and
  ├─ video ──────── the video-only stream (yt-dlp, H.264 ≤ 1080p first) → video.<ext>          ┘ the network beside it
  └─ export ─────── mix (voices + ducked bed, loudness) → AAC + 2 mov_text + copied video → ~/Movies/Maata/<title> (Telugu).mp4
```

- **Whole video first.** No line is voiced before the whole-file speakers, the full transcript, every speaker's voice and the full-transcript brief exist.
- **One GPU stage at a time, one job at a time.** The GPU order is speakers, transcript, voices, separate, then the dub loop. Claude and the video download run beside the GPU and never take it. Other jobs wait in a FIFO queue (§4).
- **Every stage writes to disk as it goes.** A pause, a quit, a crash or an app restart resumes from disk (§2.1).
- **The output is one MP4.** One video stream (copied from YouTube's stream, frame for frame), one Telugu AAC audio stream (the dub over the original music and effects), two soft subtitle streams (Telugu at dub times, English at source times). The English speech is removed by the on-device separator; a faint residue can remain where nothing masks it (§2.15, §12).
- **The video keeps its timing.** There are no freezes in a file (§2.11): overruns are absorbed by shorter wordings, fix-ups and a small speed-up, and what is left lets the next line start a little later. A speaker's lines never overlap each other; a line may overlap another speaker's as much as the English did (crosstalk).
- **What leaves the Mac:** the transcript text and the video's public metadata (title, channel, up to 2,000 characters of description, chapters, tags) and talk shares go to the Claude CLI (ADR-019); yt-dlp fetches the audio, the thumbnail and the video from YouTube. Nothing else: the separation, mix and export stages send nothing, and the Library shows thumbnails from the cache.

## 2. Stages

### 2.1 The job, the cache layout, fingerprints, pause and resume

**One `RenderJob` per video** (`engine/src/maata_engine/render.py`, a subclass of `Dubber`, §6). The engine owns it, not a WebSocket connection, so a job survives a closed window. One job runs at a time; the others are `queued` (§4).

**Cache layout.** Existing files in `<cache>/<video_id>/` stay where they are. The job's own files go in `render/`:

| File | What | Written |
|---|---|---|
| `audio.<ext>` | yt-dlp's audio (usually m4a, sometimes webm: always read through `video.audio_path`) | fetch |
| `thumb.<ext>` | yt-dlp's thumbnail as YouTube serves it (no conversion), served to the Library from the loopback | fetch |
| `video.<ext>` | yt-dlp's video-only stream (H.264 mp4 first, §2.2); deleted once the whole video's MP4 is written. Only `video.*` in the job's own folder is ever deleted: the demo's `demo.mp4` and a bench's local file are not the job's (§2.2) | video |
| `lines.jsonl`, `briefs.jsonl` | the translator's line and brief caches (existing format). Rephrase answers are not stored here (§2.10) | translate, brief |
| `units.jsonl` | trace (existing), plus `stage` events | every stage |
| `render/job.json` | settings, status, per-stage progress, timings and attempts, the finished prefix, the output | every progress change (atomic) |
| `render/audio16k.f32` | mono float32 16 kHz analysis audio, raw, `np.memmap(mode="r")`; deleted when the job is done | fetch |
| `render/diarization.json` | whole-file turns, exclusive turns, centroids, merges, id map, talk time | speakers |
| `render/transcript.jsonl` | one row per ASR chunk: words and guard results | transcript (appended) |
| `render/voices.json`, `render/voices/<sid>.npy` | per speaker: spans used, reference hash, voice key, calibration pairs; the "Hear voice" sample | voices |
| `render/skipped.jsonl` | lines the translator's ladder skipped (§2.8) | translate (appended) |
| `render/bed.json`, `render/bed/<k>.npy`, `render/bed_vocals.npy` | the separator's inputs and done blocks; the music & effects bed, 60 s blocks, 44.1 kHz stereo float16; the vocals stem's mean square at 10 Hz (§2.14); deleted with the source video | separate |
| `render/takes.jsonl`, `render/takes/<take_key>.npz` | per line, the chosen take of each wording or piece (§2.9) | dub (appended) |
| `render/fixups.jsonl` | rephrase answers and voiced-wording reviews; a `settled` row per scene (§2.10) | dub (appended) |
| `render/pcm/<pcm_key>.npy` | final line PCM, 24 kHz float16 (`_save_pcm`: atomic) | dub |
| `render/manifest.json` | the final lines: the export's contract (§2.13) | once, when finish completes |

The output MP4 goes to the output folder (§2.17), never into the cache, and nothing in the engine deletes it (only the job's own unfinished `.part.mp4`).

`job.json`:
```json
{"version": 1, "videoId": "…", "title": "…", "channel": "…", "duration": 10505.66,
 "settings": {"speakers": null, "style": "colloquial", "stopAt": null, "speedCap": 1.2, "ttsScript": "telugu",
              "presets": []},
 "status": "running", "stage": "voice_lines", "finalUntil": 2530.4, "queuedAt": 0,
 "stages": {"fetch": {"state": "done", "done": 3162.0, "total": 3162.0, "unit": "MB", "seconds": 141.2, "gpu_s": 0.0,
                      "attempts": 1, "interrupted": 0}, "…": {}},
 "source": {"file": "audio.m4a", "audioStart": 0.0, "thumb": "thumb.webp",
            "video": {"file": "video.mp4", "owned": true, "codec": "h264", "width": 1920, "height": 1080, "fps": 30.0,
                      "start": 0.0, "fingerprint": {}},
            "description": "…", "chapters": [], "tags": []},
 "output": null, "slept": null, "coverage": null, "error": null, "createdAt": 0, "updatedAt": 0, "elapsed": 0.0,
 "expired": false}
```
- `status` is one of `queued | running | paused | interrupted | waiting | failed | done`.
  - `queued`: waiting for the job before it (`queuedAt` orders the queue, §4).
  - `waiting`: Claude is held back by a usage limit or a sign-in problem while GPU work goes on.
  - `interrupted`: Maata was quit while the job ran. The engine stopped it gracefully (§4: the stage's start isn't counted as a crash) and it continues at the next launch.
  - A job the engine finds `running` or `waiting` at startup was stopped by a crash or a kill; it is set to `interrupted` and continues too, and those starts count toward the crash-loop guard (§4).
  - `done` with `stopAt` set is a finished preview (§3).
- `output` is `{"path", "bytes", "kind": "whole" | "preview", "at", "inputs", "loudness": {"I", "TP", "LRA"}, "warning"}` once the MP4 is written (§2.17).
- `source.video` is filled by the `video` stage (§2.2). `owned` is false for the demo's and the bench's files, which are never deleted; `fingerprint` and the codec fields stay after the file is deleted, so a re-run can tell whether the output still matches (§2.17).
- `slept`: `{"at", "resumed"}` (wall-clock) after the engine noticed the Mac slept while the job ran (§4).
- **Changes (2026-10-03):** `allowFreeze` leaves the settings (a file has no freezes, §2.11; an old job.json's key is ignored); `watchedAt` goes (nothing is watched in the app); `queued`, `queuedAt`, `output`, `slept`, `source.audioStart`, `source.thumb` and `source.video` are new.
- A stage's `state` is one of `todo | running | done | failed`. `attempts` counts its starts (the crash-loop guard, §4); `interrupted` counts the starts that found an earlier one still `running`.
- `source` keeps the cache file names and the metadata the brief needs. A resume whose fetch inputs still match is served from it and never calls yt-dlp, so it works offline or while YouTube asks for a sign-in check.
- `run()` is single-entry: a resume asked for while a paused run is still finishing its call waits for it, and a pause that comes after that resume wins. A job resumed by its video id with no settings keeps the ones in job.json.

**Fingerprints.** Each stage file starts with the inputs it was made from, written as `"inputs": {…}`. A stage re-runs when the inputs it would use now differ; otherwise it is served from disk.

| Stage | Inputs |
|---|---|
| speakers | audio fingerprint (size and sha256 of the source file; no mtime); pyannote commit (`models.lock.json`); count hint or `"auto"`; bounds; settle constants; `DIAR_VERSION` |
| transcript | audio fingerprint; Whisper commit; `CHUNK`, `CHUNK_PAD`; a hash of `STYLE_PROMPT`; `ASR_VERSION`. Not the diarization |
| voices | the speaker's spans (via its reference hash); the TTS commit and settings; `CALIBRATION_TE`; `VOICE_VERSION` |
| separate | audio fingerprint; separator commit; sample rate, chunk, overlap, block, compute dtype; `SEP_VERSION`. Not the speakers or the text: what depends on them is computed at export from `bed_vocals.npy` (§2.14) |
| video | the video id and the format selector (`VIDEO_FORMAT`); the result is `source.video.fingerprint` |
| export | the manifest's lines (PCM keys, starts, texts, speakers); digests of the English cue list and of the duck spans (diarized turns); the bed's inputs; the video fingerprint; the mix and subtitle constants; `EXPORT_VERSION`; the output path (§2.17) |

**Deleted intermediates are made again only when needed.** After a whole-video export the video, the bed and the analysis audio are deleted (§2.17). A later `run()` of that job (the same settings, or a change that leaves the lines as they were) must not download 2–5 GB and separate for 6–20 min only to find the MP4 unchanged. So `video` and `separate` are served from the output when `output.kind` is "whole", the file at `output.path` exists with `output.bytes`, and `output.inputs`' video and bed parts equal what those stages would use now (`source.video.fingerprint` and the format selector; the separate inputs above). The export then compares its full inputs: equal, the MP4 is served; different (the lines changed), the export first re-makes what it lacks, by the same code: the bed's blocks on the GPU and the video download. The analysis audio is decoded again by fetch (about a minute) whenever it is missing, as today.

**Content-addressed caches.** Everything below the stage files is keyed by content, never by unit id: lines by `line_key` (English, speaker, style; the translator adds `PROMPT_HASH` and the model); voices by `VoiceKey(sid, cfg, exaggeration, reference)` with `reference` = `f"{VOICE_VERSION}:{hash}"`; take rows by (line key, onset, wording, TTS text hash, voice key) and take files by `take_key = sha256([tts_text, voice key, n, cfm_steps, TAKES_VERSION])[:20]`; PCM by `pcm_key = sha256([take keys, rounded plan fields, MIX_VERSION])[:20]`; bed blocks by their index under the separate stage's inputs. A re-run touches only what changed. Unit ids are display-only.

**GPU calls, pause and resume.**
- Every GPU call goes through `Dubber._on_gpu(fn, *args, cost=None, priority=VOICE)`. It takes the `GpuScheduler` hold, runs `fn` in a worker thread, and on cancellation keeps the hold until the thread has returned (a shielded future), so a resume never starts a second pyannote run or a second `synthesize_takes` beside the first.
- Long calls are cooperative. pyannote's hook checks a `threading.Event` and raises to abort `apply` within one batch. ASR chunks, voice builds, separator batches and takes finish their call (seconds). The export (CPU, a worker thread) checks the same event per 10 s block (§2.17).
- **Pause** stops GPU work at the next call boundary. Claude calls already in flight finish into the line cache; no new call starts.
- **Resume** is `run()` again: every stage re-checks its fingerprint and continues from its files.

### 2.2 Fetch: the audio first, the video beside the dub

- `resolve.py`: `Resolver.resolve(ref, cache_dir, download=True, progress=None)` returns `ResolvedVideo` with `audio_path` as today, and a new `Resolver.fetch_video(video, cache_dir, progress=None) -> VideoFile(path, owned)`. yt-dlp stays pinned (2026.8.19 in `uv.lock`), with `remote_components: []` and never `-U`.
- **fetch** (the first stage, unchanged but for the thumbnail): `bestaudio[ext=m4a]/bestaudio` → `audio.<ext>`, plus the thumbnail as YouTube serves it (`writethumbnail`, no conversion, so no ffmpeg; `outtmpl` `{"default": "audio.%(ext)s", "thumbnail": "thumb.%(ext)s"}`, so `_audio_files` never mistakes it for audio). The speaker check comes after a 162 MB download, not a 2–5 GB one.
- **video** (a stage of its own, §1): it runs beside separation, translation and the dub loop, since it needs the network and no GPU; the export waits for it.
  - A **second** `YoutubeDL({…the audio options, "format": VIDEO_FORMAT, "outtmpl": "video.%(ext)s"})` calls `process_ie_result(copy.deepcopy(info), download=True)` on the info that fetch's `extract_info` returned (kept in memory for an hour of wall-clock time, since its signed format URLs expire after about 6 h, and let go of once the video is there; a resume without it, or after that hour, calls `extract_info` once more). When the stage runs again because its selector or the recorded file's fingerprint changed, the job's own `video.*` is deleted first, so the resolver downloads instead of handing back the old file. Both instances set `"fixup": "never"`: with Homebrew's ffmpeg on the `PATH` (a terminal launch) yt-dlp would otherwise remux the DASH files, and without it (a Finder launch) it wouldn't, so the files would differ by how Maata was started. No selector ever uses `+` (a merge needs ffmpeg).
  - `VIDEO_FORMAT` = `bv[vcodec^=avc1][height<=1080][ext=mp4]/bv[vcodec^=av01][height<=1080][ext=mp4]/bv[height<=1080]/b[vcodec^=avc1][ext=mp4][height<=1080]`. H.264 up to 1080p first (QuickTime plays it everywhere; YouTube serves H.264 only up to 1080p); then AV1 in MP4 (copied as is: QuickTime decodes AV1 in hardware on M3 and later, so on the M5 Pro); then any video-only stream up to 1080p (VP9, copied with a warning that QuickTime won't play it; IINA or VLC will); last, a muxed H.264 format (its audio track is ignored). There is no unbounded `bv`, which could pick a 10 GB 4K or 8K AV1 stream.
  - `progress(done, total)` from the hooks; the total is the chosen format's `filesize` or `filesize_approx`. The download stops at the next chunk on pause, like the audio's.
  - `source.video` records the file, `owned`, codec, size, frame rate, the stream's start time and the fingerprint (size and sha256), read with PyAV.
  - **Disk check** before the download, on the cache volume: the video's size plus about 0.9 GB per hour of video (the bed and the export's audio) plus 1 GB of headroom **(estimate, §7)**; the output folder's volume is checked too, for the MP4 (the video's size plus about 0.1 GB per hour). Fetch keeps today's smaller check (`DISK_PER_HOUR` for the analysis audio, takes and PCM).
  - Served from the output after a whole export (§2.1), and from the file when its fingerprint matches.
- **Why not first:** downloading 2–5 GB before the speakers would delay the speaker check and every GPU stage by the download (2–10 min, unmeasured); beside the dub loop it costs no wall time. If YouTube refuses the video formats, the job fails in its first hour, not at the end, and a resume after a yt-dlp update keeps everything done.
- **YouTube's JavaScript challenges.** The pinned yt-dlp solves YouTube's n-challenge only with a JavaScript runtime and the `yt-dlp-ejs` scripts; neither is in `uv.lock` (`remote_components` is empty), and `deno` is on this Mac's terminal `PATH` (`/opt/homebrew/bin`) but not on a Finder-launched app's. His 162 MB audio fetches worked; a 2–5 GB video stream may be missing from the format list or throttled. `scripts/check-youtube-video.sh URL` (no download beyond a 200 MB range) runs the pinned yt-dlp with the engine's exact options and a Finder-like `PATH`, lists the formats `VIDEO_FORMAT` can pick, and times the range. The maintainer runs it once before the first real job; adding `yt-dlp-ejs` (a new dependency) is a question only if it shows a problem (§11 M18).
- `decode_audio_to(src, dest, sr=SR_ANALYSIS)` decodes the audio frame by frame straight into `render/audio16k.f32`. **One origin for every decode:** sample 0 is the first decoded sample of the audio stream (after libavformat applies the edit list and AAC priming), and `source.audioStart` records that sample's time in the stream. The separator's 44.1 kHz decode (§2.14) counts samples the same way, so a dub time t means the same instant in the analysis audio, the bed and the output.
- **Demo, bench and tests.**
  - `DemoResolver` writes a synthetic 180 s `<cache>/<id>/demo.mp4` once with PyAV: `libx264` with its defaults (high profile, B-frames, so packets have dts ≠ pts), 320×180, 10 fps colour bars with one white frame at exactly 10.0 s; AAC 44.1 kHz stereo, a soft chord bed and a 1 kHz beep starting at exactly 10.0 s. It is both the audio and the video (`owned` false).
  - `LocalResolver(path)` (moved from `bench.py`) does the same for a local video file outside the cache: the job records its absolute path, so a resume finds it, and never deletes it. `maata-bench pipeline` runs its job in its own cache (`~/Library/Caches/Maata-bench`, which the engine never scans, so a bench job never shows in the Library, counts toward retention or is continued by the app), saves the MP4 in `<that cache>/out` unless `--out` says otherwise, starts each run from a clean job (every stage measured on this run's file, translation included), and refuses a file with no video stream before any model loads.
  - Tests also use a **YouTube-shaped fake resolver**: a video-only `video.mp4` (libx264 with B-frames) and an audio-only `audio.m4a` (AAC with its priming and a non-zero start), as YouTube's DASH files are, so the A/V origin, the cleanup and resume are tested on two files, not only on one muxed file.

### 2.3 Speakers: diarize the whole file once

As built (step 2), unchanged:
- **Backend API.** `Diarizer.diarize(audio, *, num_speakers=None, min_speakers=None, max_speakers=None, step=None, progress=None, cancel=None) -> DiarBlock`. `PyannoteDiarizer.diarize` calls the pipeline once on the whole waveform with the count arguments; its hook drives progress (segmentation 40 %, embeddings 50 %, clustering indeterminate), checks `cancel` up to the start of clustering and runs the memory guard. `MockDiarizer.diarize` honours `num_speakers` and, in auto mode, adds a 4 s blip "C" near 100 s whose embedding is close to A's, which the settle step must merge.
- **Memory guard.** VBx's scipy centroid linkage peaks at about 8.4·n² bytes for n retained embeddings **(measured by both reviewers)**. At the segmentation hook the guard counts the (window, local speaker) pairs active for at least 20 % of the window; over `DIAR_CLUSTER_BUDGET` (3 GB) it aborts and diarizes again at a 2 s step (half the windows, a quarter of the matrix, still one clustering). The same coarse step is used when a speakers stage was interrupted by a crash since it last finished (an out-of-memory kill leaves no exception; a quit is a graceful stop and doesn't count, §4).
- Resident models stay low here: Chatterbox loads on first use and Whisper's cached model is released when the transcript stage ends.
- **Count control.** Auto: `min_speakers=1, max_speakers=6`. With the user's count: `num_speakers=k`.
- **Settle** (`speakers.settle`): in auto mode a speaker under max(30 s, 1 % of talk), capped at 10 % of talk, merges into the nearest by centroid cosine (else the one it shares the most adjacent speech with), and pairs with cosine ≥ 0.85 merge; with a user count nothing merges. Ids are renumbered S1..Sk by first speech.
- **Stable ids** (`speakers.match_ids`): on a re-run each new speaker takes the previous id it shares the most exclusive speech with, one to one; so the unchanged person keeps their id, line keys, voice and takes.
- **Report.** `speakers.activity` gives each speaker's share of 120 slices of the video for the speaker check's activity strip.
- `diarization.json` keeps `inputs`, `step`, `labels`, `turns`, `exclusive`, `stray`, `centroids`, `merged` and `speakers`; on resume the registry is rebuilt from it deterministically.
- **Changing the count** (`set_speakers`) pauses the job, re-runs this stage with the new hint and the previous ids, and queues the job at the front (§4). Downstream everything comes from the content-keyed caches: units are recomputed; the brief is made again; lines whose English and speaker are unchanged hit the line cache; a speaker keeps their voice while ≥ 90 % of their stored reference speech is still theirs; the final plan is made again from the takes that remain valid. Separation is untouched (it doesn't depend on speakers).

**The speaker check** (M3) does not stop the job. `speakers_found` goes to every window and to a notification ("Maata found 2 speakers in <title>. Check them before the Telugu speech starts, in about 25 min."). A correction costs only the diarization while the transcript runs; until the dub loop starts (after voices and separation, §7: about 16–50 min after the speakers stage for his video) it costs no TTS, only the brief and the changed lines' translation; after that, the changed lines are voiced again. The UI says which, before and after.

### 2.4 Transcript

As built, unchanged: `RenderJob._transcript()` runs Whisper from 0 to the end of the video whatever the range (`CHUNK` 60 s + `CHUNK_PAD` 5 s; diarized speech for the coverage guard; `fix_words`, `_sentence_cut`, `_asr_checks`), one `transcript.jsonl` row per chunk (`a`, `b`, `next`, `language`, `words`, `checks`). Resume reads the rows with a plain reader, ignores a torn last line, continues at the last row's `next` and restores the language. Progress: video seconds transcribed / duration.

### 2.5 Sentence units

As built, unchanged: `RenderJob._units()` recomputes the units from `transcript.jsonl` and the registry on every run, in a worker thread (per chunk `segment`, then one `merge_fragments` pass over the whole list), assigns ids in onset order and indexes them by onset, so per-line queries cost O(log n + k). `UnitState` carries `take` (the take row) and `pcm` (the final file's key). The English subtitles come from these units and their word times (§2.16).

### 2.6 Voices from the whole video

As built, unchanged: per speaker, most talk first, today's build path (`_build_voice` = `_voice_from(*_voice_spans(sid))`, ADR-017) choosing its timbre span and identity clips from the whole video; a preset voice under `REF_MIN` (4 s) of clean speech; the three `CALIBRATION_TE` sentences calibrate the pace, and the first usable take is saved (vocoded, watermarked) as `render/voices/<sid>.npy`, the speaker's **Hear voice** sample: synthetic Telugu of a fixed sentence, never source audio. `voices.json` keeps the inputs and per speaker the spans, hash, voice key fields, calibration pairs, pace and overhead. On resume the voice is rebuilt from the stored spans and the estimator re-fitted with no synthesis; after a speakers re-run a voice is kept while ≥ 90 % (`KEEP_VOICE`) of its build's speech is still the speaker's.

### 2.7 Brief from the full transcript

As built, unchanged: `ClaudeTranslator.make_brief(meta, transcript, previous)` once the transcript is done, beside the voices stage, in parts of ≤ `BRIEF_PART_WORDS` (8,000) words cut at transcript-row boundaries (v1 from the first part, diffs from the later ones; cached by message hash); `BRIEF_TRIES` (3) unusable replies end it with what it has, recorded in `job.json` `brief`. `use_brief(final)` is called once, before the first scene.

### 2.8 Translation: every scene, full context, reviewed

As built, unchanged except the prompt (§2.18):
- Waits for the brief and the voices stage (the band rule uses each speaker's calibrated pace).
- The whole video's units are cut once into scenes of up to 150 s and 30 lines (`scene_cut(…, nth=2, final=True)`). A preview translates the scenes that start before its stop point plus `PAST_STOP` (30 s).
- Scenes the line cache holds whole are served first, in order; the others are split into 3 contiguous lanes, each translating its block in order with `_scene(req)` (scene call, review, one re-translation of P/E lines), so every scene but a lane's first has the previous scene's Telugu as context.
- Fits are waited for before a scene is voiced; failures back off and wait out holds with the job `waiting`; a scene released by a `ClaudeCLIError` is retried after the hold; skipped lines go to `render/skipped.jsonl`.
- The coverage report (C/m/P/E/otherTier/unreviewed/skipped) goes to `job.json`.
- **Changes (2026-10-03):** the lanes now start while the separator runs (§2.14), so they bank 10–30 min of translation before the dub loop starts, which removes the old risk of lane 1 holding the dub loop back. The register retarget (§2.18) changes `PROMPT_HASH` and `BRIEF_HASH`, so the maintainer's 776 cached lines (1,592 rows, all under prompt hash `d548212e604d`) and his brief are not served: every line is translated again (decision M1 is moot).

### 2.9 The dub loop: quality-first takes on a working plan

As built (step 4), unchanged except that it waits for the separator: `voice_lines` starts its loop only once the `separate` stage has finished (the GPU order of §1); until then it reports "waiting for the background sound".

One task owns the GPU and does, in this order of preference: fix-up takes the final plan waits on (§2.10), final PCM for lines the final plan has placed (§2.12), and the next line to voice.
- **Voicing a line**, once its scene is ready (reviewed, or its review failed while Claude is held; its fit back): `_choose` against the working planner W (the reviewed tier unless it left the band and another fits); `TAKES_N` = 2 takes in one batched decode (3 under 3 s of speech), a third seed when every take fails; pieces at hard breaks; placed on W; a shorter tier it already has if it runs long, within the fix-up budget; else queued for its scene's rephrase.
- **Stored**: each chosen take as `render/takes/<take_key>.npz` (mel as float16 and `n_tokens` for Chatterbox via `pack_take`/`unpack_take`; samples for the mock), written atomically before its `takes.jsonl` row (line key, `at` onset, wording, TTS hash, voice key, takes with spoken text, seconds, pauses, pace; `fixups`, `made`, `coverage`, `cost`). Take files are found by key before any synthesis and never written over.
- **On resume** a line whose latest matching row's take files all read whole is restored with no `_choose` and placed on W again; the estimator replays the rows' `made` paces; restored lines ask for no fit.
- A disk that refuses a take file or a row fails the job with a message (`RenderError`); a resume goes on from the rows on disk.

### 2.10 Scene fix-ups and the voiced-wording review

As built (step 4), unchanged: once a scene's last line is voiced, one batched `rephrase` for its long lines and one `review` of the wordings voiced without a class run beside the GPU (`RenderJob._settle`); rephrase answers and reviews go to `render/fixups.jsonl`, never to `lines.jsonl`; fix-up syntheses are capped at `max(3, FIXUP_SHARE` (15 %) `× lines)`; a kept fix-up is placed with `_shortened`; a line still long is flagged `long`; while Claude is held the scene settles as it is, its lines flagged.

### 2.11 The final plan, trailing the dub loop — and no freezes

As built (step 5): a second planner F places the lines in onset order once every line of a line's lookahead window (the next line, then up to 8 lines within 30 s) has a take in a settled scene or is skipped; `_predicted` is the take's actual duration. F asks for no fix-ups; it is a pure function of the takes and the slots, so a resume re-plans from the first line in seconds and finds each PCM file already there. Placement runs in a worker thread.

**Changes (2026-10-03): no freezes in a file.** A freeze held the YouTube player still while a long line finished; an exported video must keep its timing, so both planners run with `max_freeze = 0` and `freeze_budget = 0`. Today `RenderJob` passes `RenderSettings.allow_freeze`, which defaults to True (and the server's `prepare` sends `allowFreeze: true`): the setting goes, `RenderJob` passes `allow_freeze=False` explicitly, and a test asserts both planners' `max_freeze == freeze_budget == 0`. A line that overruns its slot is handled, in this order:
1. **Wording** (§2.9): the reviewed tier unless it left the band, else a shorter tier the line has.
2. **Shorter-tier resynthesis**, then **the scene's batched rephrase** (§2.10), within the 15 % fix-up budget.
3. **The planner's absorbers** (ADR-017, unchanged): a steady speed-up up to `speed_cap` (1.2× by default, and the voice's akshara ceiling), pauses inside the take shortened first; an early start up to 0.3 s into silence; a lag up to 0.6 s (1.0 s before a long pause).
4. **Drift**: what is left is the planner's overdraft. The line plays whole (audio is never cut) and the next line starts that much later, after `min_gap`; the lateness is carried down the chain until a pause absorbs it. The line is flagged `long`, and the stats count `overdraft_s` and `overdraft_carried_s`.
- **Overlap, as the planner already does it** (`TimelinePlanner._floor`): a speaker's lines never overlap each other; a line whose English overlapped the previous speaker's (`overlaps_prev`, an interjection or crosstalk) may overlap that speaker's dub by as much as the English did. So the mix adds lines into the track (§2.15) and the subtitles merge overlapping cues (§2.16).
- At the very end, a last line that overruns the video makes the audio stream that much longer than the video (players hold the last frame); nothing is cut.
- Decision M15: drift (the default) or a higher speed cap for overdraft lines, up to the 1.25× that `Dubber` clamps `speed_cap` to.

### 2.12 Line PCM

As built (step 5), unchanged. Per line placed by F, in the dub loop: `audio = _play(said, plan)` (squeeze and cut, parts at their times; a rate ≠ 1 re-vocodes the take's mel through `vocode(take, rate)`, which applies the **PerTh watermark** as it does today, ADR-014), saved as `render/pcm/<pcm_key>.npy` (float16, atomic) unless the file is there, and the `unit` trace event written. So every voice sample that reaches the mix is already watermarked, before mixing (§2.15).

### 2.13 The final lines (manifest.json)

**Changes (2026-10-03):** the manifest is no longer a playback contract that grows while a window watches. It is written **once**, atomically, when the finish stage completes, and it is the export's input: the final lines, the skips, the speakers and the stats.
```json
{"version": 2, "videoId": "…", "title": "…", "channel": "…", "duration": 10505.66, "stopAt": null,
 "complete": true, "sampleRate": 24000,
 "settings": {"style": "colloquial", "speedCap": 1.2, "ttsScript": "telugu", "speakers": "auto", "presets": []},
 "speakers": [{"id": "S1", "label": "Speaker 1", "talkSeconds": 5120.3, "voice": "cloned", "referenceSeconds": 60.0,
               "pace": 6.5, "sample": true}],
 "lines": [{"id": 0, "speaker": "S1", "start": 3.12, "srcStart": 3.10, "srcEnd": 7.85, "audioRate": 1.04,
            "audioWall": 4.61, "lag": 0.02, "said": "whole", "tier": "full", "voice": "cloned", "source": "…",
            "telugu": "…", "coverage": "C", "flags": [], "pcm": "3fa2…c1", "samples": 110640}],
 "skipped": [{"id": 12, "start": 40.1, "end": 41.0, "why": "not translated"}],
 "stats": {"lines": 1900, "coverage": {}, "lag_p95": 0.0, "rate_p90": 0.0, "overdraft_s": 0.0, "fixups": 0}}
```
- Gone from it: `ranges`, `freeze`, `edits`, `budget`, `units`, `end` (they served the player and its HUD). `version` 2.
- Gone from the code: `MANIFEST_GROW`, `MANIFEST_WALL`, `MANIFEST_GAP`, the progressive `_rewrite` with its `_floor` and `_old_pcm`, and the `manifest` event. `finalUntil` stays in `job.json` for the progress view ("Telugu speech final up to 42:10").
- **Garbage collection** runs once after the manifest is written: take and PCM files that neither the take rows in use nor the manifest name are deleted (`_collect`, `_sweep`, as built).
- When the finish stage ends, the TTS model is released (a new `release()` on the TTS backends, like the transcriber's): the export needs no GPU, and Chatterbox's 2.2–3.2 GB would otherwise stay resident through it (§7).

### 2.14 Background sound: separating music and effects from speech

**The model: Mel-Band RoFormer "Kim Vocal 2", run with MLX.** Compared on published quality (MVSep's Multisong benchmark, mvsep.com/en/algorithms, read 2026-10-03; higher is better) and on cost:

| Model | Vocals SDR | Instrum. SDR | Instrum. bleedless | Weights | On the M5 Pro |
|---|---|---|---|---|---|
| htdemucs_ft (Demucs v4, fine-tuned) | 8.33 | 14.63 | — | public | PyTorch MPS; a bag of 4 per-source models |
| BS-RoFormer (viperx) | 10.87 | 17.17 | — | public (UVR's repo) | PyTorch; mlx-audio's port covers Mel-Band models only |
| **Mel-Band RoFormer (Kim Vocal 2)** | **11.01** | **17.32** | **46.72** | public (HF `KimberleyJSN/melbandroformer`) | **MLX port in mlx-audio, pre-converted bf16 weights** |
| Mel-Band RoFormer (becruily deux) | 11.35 | 17.66 | 42.11 | not checked | needs a conversion |
| BS-RoFormer (MVSep 2025.07) | 11.89 | 18.20 | 49.12 | not downloadable (MVSep's service) | — |
| BandIt Plus (cinematic: speech/music/effects) | DnR test: speech 15.64, music 9.18, effects 9.69 | | | ZFTurbo's releases | PyTorch (MPS, already a dependency); trained on English dialogue mixes, which is what the source is |

- Multisong measures singing over music. The job here is removing English **speech** from dialogue, music and effects, and that is what the bake-off (M10) judges: the speech energy left in the bed over the speech turns, and a listening check.
- Kim's model is about 2.7 dB better than htdemucs_ft on vocals and on the instrumental, and has the best instrumental bleedless score (the least voice left in the bed) of the public models: the closest published proxy for what the viewer hears under the Telugu. It is the default; BandIt Plus (a dialogue/cinematic separator) and Kim with test-time augmentation (channel swap and polarity, twice the GPU time: 12–40 min against a 2.4–5 h dub) are the bake-off's other arms, with becruily deux and viperx's Mel-Band model.
- It has a maintained MLX implementation (`mlx_audio.sts.models.mel_roformer`, Blaizzy/mlx-audio) and a pre-converted checkpoint whose card reports 66.08 dB SDR between the MLX and PyTorch outputs on one 8 s clip. MLX keeps it on the same GPU path as Whisper, serialised by the `GpuScheduler`.
- It is a single-stem vocal model: the bed is `mixture − vocals`, as MVSep's instrumental scores for it are computed. It treats all voice as vocals: laughter, crowd voices and singing leave the bed with the speech. That suits a dub of talk; a video whose crowd or songs matter is an M10 question.

**Vendoring, not a dependency.** mlx-audio 0.5.7 requires `transformers>=5.14`, which conflicts with the `transformers==5.2.0` override T3 needs (ADR-013), plus `huggingface_hub`, `miniaudio`, `sounddevice` and others Maata never imports. Its model file is **not** self-contained: since commit `39c9325990` (2026-04-25) `mel_roformer/model.py` imports `stft`, `istft` and `mel_filters` from `mlx_audio.dsp` (a 952-line module) and `get_model_path` from `mlx_audio.utils`, inside functions; the commit before it used `librosa` for the filterbank. So three pieces are vendored, all from one commit, `a8546e64ac5fdc7ace03e4b75dd2daad444a614e` (2026-04-27, the file's latest; the pinned weights' `config.json` says they were converted by `8380ab8` of 2026-04-25 in bf16 with mlx 0.31.0, and the refactors since changed only the STFT code, not the key layout):
- `separation/mel_roformer.py`: `model.py` and `config.py`, with the `kim_vocal_2` preset and `sanitize`; `from_pretrained` loses `get_model_path` and loads only `mx.load(models_dir / "mel-roformer-kim-vocal-2-mlx" / "model.safetensors")`, with `load_weights(..., strict=True)` (upstream passes `strict=False`, so a key `sanitize` missed would stay random, silently, and the "bed" would carry the English).
- `separation/dsp.py`: `stft`, `istft`, `mel_filters` and the window functions they use, from `mlx_audio/dsp.py` (they import only `mlx` and `numpy`).
- Each file's header records the source path and commit; ADR-020 records them with the weights pin. No new package: `mlx` (0.32.2) and `numpy` are already locked for the apple extra.

**Precision.** The weights are bf16 and the upstream model computes its STFT, window and filterbank in float32; MLX promotes bf16 × float32 to float32, so as copied it would run the whole transformer in float32 with upcast weights. The separator chooses explicitly: STFT, mask application and iSTFT in float32; the band-split input cast to `SEP_DTYPE` (bf16) so the transformer and mask estimator run in bf16. `verify-mac.sh` compares bf16 against float32 on its test mix; if bf16 loses more than 0.1 dB SI-SDR, `SEP_DTYPE` becomes float32 (its inputs fingerprint includes the dtype).

**Pinned weights** (`models.lock.json`, role `separation`, backends `["apple"]`; fetched by `maata-bench fetch` like every model, loaded with `HF_HUB_OFFLINE=1`; checked against Hugging Face on 2026-10-03):
- repo `mlx-community/mel-roformer-kim-vocal-2-mlx`, revision `64cbfcb004e39430e5f584552c05949440ec39ce`;
- `config.json` 833 B, git oid `4e7f9ed2cfcb2b3232166a5c803adf20d6816031`;
- `model.safetensors` 456,483,463 B, sha256 `312c38e5b698f8dfaa4d6064e8f79010744825828917871a9d22673a43eb7fe5`;
- converted from `MelBandRoformer.ckpt` (913,106,900 B, sha256 `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`) at `KimberleyJSN/melbandroformer@ac9b0614`;
- architecture: dim 384, depth 6, 8 heads × 64, 60 mel bands, STFT 2048 / hop 441, 44.1 kHz stereo, chunk 352,800 samples (8 s), 50 % overlap; about 228M parameters.
- Recorded as ADR-020 (the pin and the vendored files; no licence analysis).

**Backend API** (`backends/base.py`): `Separator` with `sample_rate` (44,100), `chunk` (352,800) and `vocals(batch: np.ndarray[b, 2, chunk]) -> np.ndarray[b, 2, chunk]`; `release()`. `Backend.separator: Separator | None`.
- `apple`: `MLXMelRoFormerSeparator(models_dir / "mel-roformer-kim-vocal-2-mlx")`, loaded on first use; `mx.set_cache_limit` bounds MLX's buffer cache while the stage runs, `mx.clear_cache()` after each block, and `release()` drops the model when the stage ends.
- `mock`: `MockSeparator`, whose vocals are exactly half of each chunk, so any chunking or overlap-add error shows in tests.
- `cuda`: None for now; the export then has no bed (Telugu voices only), says so in the job card and in `output.warning`. A PyTorch port of the same checkpoint is a follow-up (M17).

**The stage** (`render.py` `_separate`, chunking in a new pure module `separate.py`):
- **Range:** [0, duration) for the whole video; [0, min(duration, stopAt + 2 × `PAST_STOP`)) for a preview, which covers the preview's cut (§2.17: its last line ends by stopAt + 30 s and the next keyframe a few seconds later); continuing the preview separates only the blocks it lacks.
- **Blocks:** the output is cut into `SEP_BLOCK` = 60 s blocks, `render/bed/<k:05d>.npy`, float16 `[2, 2,646,000]`, written atomically; done blocks are skipped on resume.
- **One global chunk grid:** chunks start every hop (4 s) on a grid over the whole file, after half a chunk of padding at each end (reflect padding as ZFTurbo's `demix` does when the audio is longer than twice the pad, zero padding otherwise, so a 3 s file works); each chunk's output is weighted by a periodic Hann window and overlap-added, divided by the window sum. A block computes exactly the chunks that cover it, so its samples equal a whole-file run's up to float rounding (the mock test checks it), at about one extra chunk per block (6 %).
- **Audio in:** `audio.<ext>` decoded with PyAV at 44.1 kHz stereo (mono duplicated), sample 0 at the same origin as the analysis audio (§2.2), into a sliding buffer of one block plus half a chunk each side (about 68 s, 24 MB). A resume decodes from the start and discards samples up to its first missing block, counting them as `decode_audio_to` does: seeking would bring AAC pre-roll artefacts, approximate sample positions and fresh resampler state (Opus at 48 kHz), so a resumed block would not equal a whole-file run. Decoding is far faster than real time (seconds per hour of audio).
- **GPU:** `SEP_BATCH` (2; a constant, 4 to be measured on the M5 Pro) chunks per `_on_gpu` call, so a pause stops within a few seconds; progress is video seconds separated.
- **The vocals envelope:** per block, the mean square of the vocals stem (`mixture − bed`, both channels) in 0.1 s frames goes into `render/bed_vocals.npy` (float32, 10 values per second: 0.4 MB for 2.9 h). It doesn't depend on the speakers; the export averages it over the current diarized speech turns for the voice/bed balance (§2.15) and `verify-mac.sh` reads the English left in the bed from it. A speakers re-run therefore never separates again and never leaves a stale balance.
- `bed.json` = `{"inputs", "blocks": [k, …]}`; inputs as in §2.1.
- With `backend.separator` None: no bed, and `job.json` says so (`bed`: `{"separator", "until", "note"}`, the note saying the file will have the Telugu voices only).
- **Served from the output** after a whole export (§2.1): the stage does nothing while the MP4 it went into still matches; the export re-makes missing blocks only if it must mix again.
- **As built (step mp4-4):** the 44.1 kHz decode (`resolve.decode_stereo`) stops at the stream's declared duration as `decode_audio_to` does, so both decodes have one length and one origin. The stage doesn't need the file's length up front: a block's samples are the same for any length at or past the end of what its chunks read, so it passes what the decode has reached until the decode ends. A block listed in bed.json whose file doesn't read whole, or whose frames in `bed_vocals.npy` aren't all there, is made again. The model is loaded on first use and released when the stage ends (and after the export's re-make). A models folder filled before the separator was pinned lacks its files: `Separator.missing()` names them, and a job fails at once (at `run()`'s start, unless a whole video's MP4 still matches, which needs the model only to mix again; and before any block, the export's re-make included) saying to run `maata-bench fetch`.

**Where it sits.** After voices, before the dub loop, beside the translation lanes and the video download (§1): `ORDER = (fetch), (speakers), (transcript), (units), (voices, brief), (separate, translate, voice_lines, finish, video), (export)`, with `voice_lines` waiting for `separate` (an `asyncio.Event`; its stage shows "waiting for the background sound").
- Not first after fetch: translation needs the voices, so separating before them would delay everything by its 6–20 min; after them, the lanes bank translation meanwhile.
- Not after the dub loop: the total is the same either way (the GPU is serial), but here a separator failure shows within minutes, not hours, and the speaker check stays free of TTS cost longer.
- **ETA:** in that group the GPU runs `separate` and then the dub loop, so the group's ETA (and the estimate before Dub) is `max(translate, video, separate + max(voice_lines, finish))`, not the longest stage alone. On a backend with no separator the stage is served at once, and the estimate, a queued job's time left and the ETA count nothing for it (as built).

**Memory.** Weights 0.46 GB (bf16). Per 8 s chunk the transformer works on 801 frames × 60 bands = 48,060 tokens of dim 384: about 37 MB per activation in bf16 (twice that in float32), 150 MB for the feed-forward's hidden layer; MLX's fused attention keeps the 801-long time attention from materialising (unfused it would be about 0.6 GB per layer). A batch of 2 is about 1–2.5 GB **(estimate)**; PyTorch MPS still holds Chatterbox (loaded by the voices stage) beside it. A 2.9 h file never sits in memory: the stage holds one block.

### 2.15 The mix

`export.py` and a pure `mix.py`, run by the `export` stage on the CPU. The mix is made at **44.1 kHz stereo** (`MIX_SR`), the separator's rate and YouTube's m4a rate, so the bed is never resampled; only the 24 kHz voice is.

**The Telugu track** from `manifest.json`: each final line's PCM (`render/pcm/<key>.npy`, 24 kHz mono, already watermarked, §2.12) is **added** (`np.add` into the block, never assigned: lines of different speakers may overlap, §2.11) at its `start` (the final plan's start, in dub seconds, §2.2's origin) in a 24 kHz mono track built block by block (10 s blocks, only the lines that touch the block are read), multiplied by its speaker's gain, and resampled to 44.1 kHz by one streaming PyAV `AudioResampler` (libswresample; its delay is measured once and compensated, and the test of §9 checks a click lands on its sample), then copied to both channels.

**Per-speaker loudness.** Each speaker's voice is set to the same loudness: the integrated loudness (BS.1770, gated) of the concatenation of their lines' PCM, measured with FFmpeg's `ebur128` filter through a PyAV filter graph, sets a gain of `VOICE_REF − L_s` (−20 LUFS reference), clamped to ±6 dB so a near-silent clone is never pumped into noise. One quieter clone no longer sits under the other.

**The voice/bed balance: the original's.** The Telugu voices sit over the bed at the level the English speech sat over it: the voices' mean square over their line spans is matched to the original vocals' mean square over the current diarized speech turns, read from `bed_vocals.npy` (§2.14). So a vlog with loud music stays a vlog with loud music, and a quiet podcast bed stays quiet (decision M11).

**Ducking** to mask separation residue. What the separator leaves of the English is where the English was spoken, and the Telugu is close but not identical in time (lag ≤ 0.6 s, the Telugu often shorter). The bed's gain envelope:
- `DUCK_DB` (−6 dB) inside the union of the Telugu line spans and the English speech turns (`registry.turns_in(…, exclusive=False)`), with gaps under 0.5 s bridged (no pumping between words);
- `DUCK_BARE_DB` (−12 dB) inside English speech with no Telugu line within `BARE_GAP` (1.0 s) of it, lasting at least 1.5 s: skipped or untranslated lines, where nothing masks the residue. Short tails after a shorter Telugu line stay at −6 dB, so the music doesn't dip between lines;
- a 0.15 s raised-cosine attack before a span and a 0.4 s release after it; 0 dB elsewhere.
- It is a deterministic envelope from the plan and the diarization, not a compressor, so it is testable exactly and never reacts to the bed's own music. A listening check on the M5 Pro sets both depths (M11).
- **As built (step mp4-4):** the bare spans are the English speech (its gaps under 0.5 s bridged too) less the Telugu line spans widened by `BARE_GAP` each side, in pieces of at least 1.5 s; they lie inside the duck spans, so the envelope in dB is `DUCK_DB` times the duck spans' ramp plus `DUCK_BARE_DB − DUCK_DB` times the bare spans' ramp, and every transition (0 to −6, −6 to −12, 0 to −12) takes exactly the attack or the release. The balance is a gain on the bed (the voices keep their per-speaker level), clamped to ±`BED_CLAMP` (12 dB): a separator that left the speech turns near-silent in its vocals stem would otherwise raise the music over the voices. Its inputs (the bed's inputs, the duck-span digest, the duck and balance constants) are in `output.inputs`; `EXPORT_VERSION` is 2.

**Loudness and peaks** of the whole mix, in two passes over the blocks:
1. Pass 1 builds the mix (voices × gains + bed × envelope) and measures its integrated loudness with `ebur128` (no true-peak oversampling, so it is fast).
2. Pass 2 builds it again, applies one static gain to `MIX_LUFS` = −16 LUFS (Apple's recommended level for spoken programmes), then FFmpeg's `alimiter` (`limit` −2 dBFS sample peak, `attack` 5 ms, `release` 50 ms, `level=0` so it never makes up gain, and **`latency=1`**), and encodes.
- `latency=1` matters: with the default the limiter delays its output by its attack time, 219 samples (5 ms) at 44.1 kHz (measured with PyAV 18.1.0 on this Mac: an impulse at sample 10,000 came out at 10,219; with `latency=1` at 10,000), which would shift every voice and the bed against the picture.
- `output.loudness = {"I", "TP", "LRA"}` is measured with `ebur128` (`peak=true`) on a decode of the **encoded AAC** (seconds per hour), since AAC adds its own overshoot. The −2 dBFS sample ceiling is meant to keep the true peak under `TP_CEIL` −1 dBTP after encoding; a measured TP over it is flagged in the job card.
- Both passes are numpy blocks plus libavfilter: no new dependency (verified on this Mac: PyAV 18.1.0's FFmpeg has `ebur128`, `alimiter`, `loudnorm`, `aresample`; an EBU 1 kHz stereo sine at −23 dBFS reads −23.000 LUFS through the graph).

**The watermark** is on every voice sample before the mix: PerTh is applied inside `vocode` to each take (ADR-014, unchanged). The mix, the gains and AAC come after it. Whether it is still detectable in the final AAC is checked on the M5 Pro (`resemble-perth`'s detector on a decoded export, in `verify-mac.sh`), not assumed.

**Encoding.** AAC-LC, 44.1 kHz stereo, 192 kbps, with `aac_at` (Apple's AudioToolbox encoder) where PyAV has it (macOS), else FFmpeg's `aac`. The encoder's priming is compensated by the MP4 edit list libavformat writes; the A/V sync tests (§9) decode the output and would catch an uncompensated priming (23 ms).

**No bed** (the cuda backend, or a separator failure the user chose to skip): the Telugu voices alone, normalised the same way, with `output.warning` saying so.

### 2.16 Subtitles

A pure module `subtitles.py` makes the cues; the export writes them as two **mov_text** (tx3g) streams.
- **Telugu**, at the dub's times: for each final voiced line, its Telugu-script wording (`telugu`, the wording actually voiced, also when the TTS read the Latin form) over [`start`, `start + samples / 24000`]. Skipped lines have none.
- **English**, at the source times: every transcript unit (voiced, skipped or untranslated alike), its text over its word times.
- **Cue shaping:** at most 2 rows (`SUB_ROWS`) of `SUB_ROW` = 42 characters for English and 42 **aksharas** for Telugu (a conjunct is up to three code points, so counting code points would make Telugu rows half as wide on screen; `text/akshara`); a longer line is split at word boundaries into consecutive cues, each timed by its share of aksharas (Telugu) or by its words' own timestamps (English); rows break at the space nearest the middle; a cue lasts at least 0.7 s where the next allows; a preview's cues are cut at its end.
- **Overlaps are merged, not cut.** tx3g shows one sample at a time, and dub lines (and English units) of different speakers overlap on crosstalk (§2.11). Where cues overlap, the time is cut at every cue start and end, and each piece shows every cue active in it, one block per speaker, each prefixed "– " when there are two or more (up to 2 rows per speaker). An interjection inside a long line therefore shows beside it for its own time; neither text is cut to a flash.
- **Writing them** (verified with PyAV 18.1.0 on this Mac): a stream from `add_stream("mov_text")` opens only with a `codec_context.subtitle_header` (a minimal ASS header; without it `avcodec_open2` fails), time base 1/1000; each cue is muxed as a packet of a 2-byte big-endian length plus the UTF-8 text, `pts` its start and `duration` its length in ms. libavformat fills the gaps with empty samples itself. Telugu script needs no font in the file: players render tx3g with system fonts (macOS ships Telugu fonts; how QuickTime shows it is checked on the M5 Pro, §10 step 6).
- Stream metadata: `language` `tel` / `eng` (ISO 639-2), `title` "Telugu" / "English"; dispositions off, so neither shows until picked in the player's Subtitles menu (M12). libavformat's mp4 muxer still enables the first track of each type when none is default (its `enable_tracks`), so the export clears the `enabled` flag in both subtitle tracks' `tkhd` in place after the mux (both stay in alternate group 3). Cues are on the file's clock and none starts before its 0: where the video starts after the audio, a cue that began earlier shows from 0 and one that ended earlier is left out (a negative sample time puts the whole mov_text track out of time).

### 2.17 The MP4

`export.py`, the `export` stage: one pass that muxes, with PyAV only (no ffmpeg binary), into `<output dir>/.<name>.part.mp4`, then `os.replace` to the final name, so a half-written file never looks finished.

**Streams** (in this order):
1. **Video, copied.** `add_stream_from_template` of `video.<ext>`'s stream; packets in decode order, `pts`/`dts` shifted by the stream's start time so the video starts at 0. Frame for frame what YouTube sent, with its timestamps: the dub's clock is the video's. H.264 and AV1 are copied as they are; VP9 too, with `output.warning` "QuickTime can't play VP9; open it in IINA or VLC" (M16). There is no re-encode path.
2. **Audio:** the mix (§2.15), AAC, `language=tel`, disposition `default`.
3. **Subtitles:** Telugu, then English (§2.16).

- **A/V origin.** Dub time t is placed at `t + audioStart − videoStart` in the output, from the two streams' start times recorded at fetch and at the video stage (usually both 0 for YouTube's DASH files, but a local file, or an m4a with an edit list, may differ). The tests check it on the demo and on the YouTube-shaped two-file fake (§9).
- **Interleaving:** per 10 s block, the video packets with `dts` before the block's end, then the block's AAC packets, then the cues that start in it; libavformat interleaves the rest.
- **Preview cut:** a preview ends at the first video keyframe at or after max(`stopAt`, the end of the last final line's audio), so the last line is never cut mid-word; packets from that keyframe on are left out, so the copied video needs no re-encode; the audio and the cues end there too (the bed covers it, §2.14).
- **`movflags=+faststart`**: the `moov` atom precedes `mdat` (verified in the probe), so QuickTime opens a 3 GB file at once. Writing it at the end rewrites the file once (tens of seconds for 3 GB); the stage shows "Finishing the file…" meanwhile, and it can't be paused.
- **Pause and failure.** The export runs in a worker thread and checks the pause event per 10 s block; on a pause or a failure it closes and deletes its `.part.mp4`, and a resume exports again from the start (a few minutes; the result is byte-identical, which a test checks). `remove` deletes the job's `.part.mp4` too.
- **Disk:** before it starts, the output folder's volume must hold the video's size plus about 0.1 GB per hour of audio, else the job fails saying how much is needed (it was checked at the video stage too, but hours have passed).
- **Metadata:** `title` "<title> (Telugu)", `artist` the channel, `comment` "Telugu dub by Maata (AI voices) of https://youtu.be/<id>", `date`.
- **Output folder and name:** the engine setting `outputDir` (default `~/Movies/Maata`, created on first use, with a clear error if it can't be written; `~/Movies` is not a folder macOS asks permission for). The name is `<title> (Telugu).mp4`, or `<title> (Telugu, first 15 min).mp4` for a preview; the title is sanitised (`/ \ : * ? " < > |` and control characters become spaces, runs of spaces collapse, leading dots go, at most 150 UTF-8 bytes). If the name exists and isn't this job's own earlier output, " (2)", " (3)", … is added. A re-export of the same job replaces its own file: `output.path` while it still holds the recorded bytes, in the same folder, under the name or a numbered form of it (so a job that went to " (2)" stays there); a file at that path with other bytes (moved away, then another job's) is someone else's and never replaced. `remove` never deletes a `.part.mp4` that is the running job's.
- **Fingerprint:** `output.inputs` (§2.1). A resume finds the export done when the file at `output.path` exists with its recorded size and the inputs match; otherwise the export runs again from the start (a partial `.part.mp4` is deleted first). If the user moved or deleted the MP4, the job card says "File moved or deleted" with **Show folder** and **Save again**.
- **After a whole-video export** the job is `done`, the TTS is released (§2.13), and its large sources go (M6): `<cache>/<id>/video.*` (only when `source.video.owned` and the resolved path lies directly in the job's folder), `render/bed/`, `render/bed.json`, `render/bed_vocals.npy` and `render/audio16k.f32`. Never a path a resolver returned outside the cache: the demo's `demo.mp4` and a bench's input file stay byte-identical (a test checks). A later re-run that changes no line serves the MP4 without fetching or separating (§2.1); one that changes lines downloads the video and separates again, and the UI says so before it starts. A preview keeps its sources, since continuing needs them.
- **Open video / Show in Finder:** the engine runs `/usr/bin/open <path>` or `/usr/bin/open -R <path>` for the job's own recorded output path, never a path the UI sends (`xdg-open` / `explorer /select,` on the other platforms).

### 2.18 The translation register: the Telugu ordinary people speak

Retargeted from "a fluent, educated speaker on a Telugu YouTube explainer or podcast" to the everyday spoken Telugu of the general public. Three places decide the words, so all three change: the scene prompt, the brief (its glossary fixes how every recurring term is said, and every scene must follow it), and the lint that reports on the result.

**`text/scene_prompt.py`, the scene prompt:**
- **`ROLE`:** the words ordinary Telugu people say aloud in everyday talk: at home, with friends, at work, in a shop and on the phone; never bookish. Not "a Telugu YouTuber". The TTS sentence stays (every line must be speakable as written).
- **`STYLE` (colloquial, the default):**
  - keep: the central dialect, neutral between Andhra and Telangana; no grandhika words (యొక్క, తద్వారా, మరియు, కావున). Drop the "Sishta vyavaharika" label (the educated standard), which pulls toward written forms;
  - the English rule, from "where educated Telugu speakers really say it in English: names, brands, technical and modern terms, established loans" to: an English word wherever ordinary Telugu people say it in English in everyday talk (not only names and technical terms: everyday loans such as ఫోన్, టైం, బైక్, టికెట్, ఆఫీస్, సారీ, ఓకే; not బస్, since `SCRIPT` counts బస్సు as an absorbed Telugu word), and a Telugu word wherever they say it in Telugu; still no English quota either way, and never English grammar words (the lint's rule);
  - the verb rule, from "Everyday verbs stay Telugu" to: verbs people say in Telugu stay Telugu (తిను, వెళ్ళు, చూడు, చెప్పు); verbs they commonly say in English take the bare English stem with చేయు/అవు (చేంజ్ చేయి, ట్రై చేయి, వెయిట్ చేయి, ఇంక్రీజ్ అయింది), never an -ed or -ing form;
  - add: prefer the simple everyday word to the Sanskrit-heavy or written one where people use the simple one;
  - add: the brief's `register` sets politeness and how speakers address each other, never how bookish the words are: a formal English lecture still comes out in everyday spoken Telugu (the formal style is the user's choice, not the video's);
  - keep unchanged: spoken verb forms; fillers by their meaning; the brief's address forms; idioms as Telugu equivalents; numbers as spoken words; and **every fact, name, number, negation and question kept, nothing added**.
- **`STYLE` (formal)**, `SCRIPT`, `LENGTH` and `CONTRACT` are unchanged: the Telugu-script contract (`qa/validators.py`) holds as before.
- **Examples (`SHOTS`):** each of the 18 is re-read for words ordinary people wouldn't use and rewritten where needed (for instance the brain example's శరీర బరువులో and శాతం, and బడికి/ఉద్యోగాలు in the first), keeping its English map and tiers valid; one example is added where a formal English source (a lecture's sentence) is rendered in everyday Telugu. The block stays marked MAINTAINER REVIEW for a native speaker.
- `SHOTS_VERSION` becomes `"scene-v3"`; `PROMPT_HASH` follows (its pin in `tests/test_scene_prompt.py` is updated).

**The brief (`BRIEF_SYSTEM`):**
- `glossary`: `keep_english` is true wherever ordinary Telugu people say the term in English; `spoken` is a Telugu rendering only when it is the everyday word, never a coined or Sanskrit-heavy one.
- `register`: only the tone and how the speakers address the audience and each other; word choice follows the scene prompt's style.
- One brief serves both styles (its message and cache key carry none), so `BRIEF_SYSTEM` names no style, and the scene prompt's `BRIEF_HEADER` adds: in the formal style, a term with a common Telugu word is said in Telugu even when its glossary entry keeps English.
- `BRIEF_HASH` changes (3–4 brief calls for his video); `REVIEW_HASH` stays (the review is meaning-only).

**The lint (`text/tenglish.py`):** `_BASIC_VERBS` (verbs the lint reports when said in English) loses the everyday loans ordinary speakers do say in English with చేయు/అవు (change, increase, decrease, and others the implementer finds commonly said that way, such as learn) and keeps the core Telugu verbs (think, see, eat, go, come, know, say, tell, give, take, …); `LINT_VERSION` becomes `"lint-v4"`. The lint only reports, in the trace and the metrics: it never blocks or retries a line, and no other check catches a line with too much English (§12).

Cost: the line cache is keyed by `PROMPT_HASH`, so every line is translated again (§7).

## 3. The preview and continuing

- **Preview.** `settings.stopAt` is null (the whole video) or 900 (the first 15 minutes). Fetch, speakers, transcript, units, voices and the brief always cover the whole video. Translation and voicing cover up to `stopAt` + 30 s and separation up to `stopAt` + 60 s; F finalises the lines that start before `stopAt`; the export cuts at the first keyframe at or after max(`stopAt`, the end of the last final line's audio) and writes `<title> (Telugu, first 15 min).mp4`. The job is then `done` with `output.kind` "preview".
- **Continue to the whole video** is a `prepare` with `stopAt` null (queued like any job). Every cache applies: the stages before translation do no work, separation adds only the missing blocks, the lines already voiced keep their takes and PCM keys, and the export writes the whole file under its own name. The preview file stays (the user deletes it).
- **Speakers, voice or style changed:** §2.3, §2.9; `line_key` includes the style.

## 4. The queue, progress and the protocol

**The queue.** One job runs at a time; the GPU is the reason.
- `prepare` creates or updates the job and runs it if nothing runs, else marks it `queued` with `queuedAt` (FIFO). There is no `busy` any more.
- When the running job stops for any reason (done, failed, paused), the engine starts the oldest queued job.
- `pause` of the running job lets the next queued one start; `pause` of a queued job takes it out of the queue (`paused`).
- `resume` (and `set_speakers`, `set_voice`, "Continue to the whole video") queues the job; a re-run of the job that was running goes to the front, anything else to the back.
- **`remove {videoId, forget}`** takes a job out of the library:
  - `videoId` must match YouTube's 11-character id pattern (`[A-Za-z0-9_-]{11}`) and resolve to a direct child of the cache folder; anything else (an empty id, `..`, a path) is refused, so a bad message can never delete the cache or escape it.
  - A queued job leaves the queue first. A running job, or one whose `run()` hasn't returned yet (a pause still finishing its Claude calls), is refused with "Pause it first" (the UI pauses, then removes).
  - It deletes `render/` (job, takes, PCM, bed), the media (`audio.*`, `video.*`, `thumb.*`) and the job's own `.part.mp4`, and keeps `lines.jsonl`, `briefs.jsonl` and `units.jsonl`, so dubbing the video again doesn't pay for the same translations; `forget: true` deletes the whole folder. The MP4 in the output folder always stays.
- **Quitting is graceful; a crash is not.**
  - When the shell quits it closes the engine's stdin and waits up to 5 s (§5). On that EOF the lifeline thread asks the event loop (`loop.call_soon_threadsafe`) to stop: the running job's task is cancelled, and the existing `CancelledError` path in `RenderJob._stage` sets the stage back to `todo` and takes the start out of `attempts`; the job is written `interrupted`, then the engine exits. `os._exit` stays only as the fallback if that takes over 4 s.
  - So a quit never counts toward the crash-loop guard and never marks the speakers stage interrupted (which would force the coarse diarization step). Claude calls in flight are cancelled; a resume asks again.
  - A crash, an out-of-memory kill or a `kill -9` leaves the job `running` or `waiting`, as today.
- **Recovery at startup.** Jobs found `running` or `waiting` (a crash) become `interrupted`; together with the jobs a quit left `interrupted` they are queued first, most recently updated first, ahead of the jobs that were already queued, which keep their order. The crash-loop guard fails a job instead when a stage was started `CRASH_STARTS` = 3 times without finishing (graceful stops don't count), naming the stage. A job recovered from a crash posts a notification ("Maata restarted after a problem and is continuing <title>"). So quitting Maata pauses the queue, and opening Maata continues it.
- **Keep-awake** (M8): while a job runs with a real backend on macOS, `/usr/bin/caffeinate -i -w <engine pid>` holds off idle sleep; it ends with the job. It does not stop the sleep that closing a MacBook's lid causes: the job then stops until the Mac wakes, and continues by itself. The app says so (§5).
- **Sleep, noticed:** while a job runs, the engine compares wall-clock time with `time.monotonic()` at each progress tick; a gap over 60 s means the Mac slept, recorded as `slept: {at, resumed}` and shown as "Paused while the Mac slept · resumed 06:40". Elapsed time and the stage rates use `time.monotonic()`, which on macOS is `mach_absolute_time` and stops during sleep (checked on this Mac), so the ETA stays honest after a wake.

**Stages reported** (`render.STAGES`):

| key | label | unit | progress |
|---|---|---|---|
| fetch | Download | MB | yt-dlp hooks (the audio) |
| speakers | Speakers | fraction | pyannote hook (clustering indeterminate) |
| transcript | Transcript | video s | per chunk |
| units | Sentences | lines | |
| voices | Voices | speakers | per speaker |
| brief | Video brief | parts | per part |
| separate | Background sound | video s | per block |
| translate | Translation | lines | lines of the range whose scene is reviewed |
| voice_lines | Telugu speech | speech s | speech seconds voiced / in range |
| finish | Finishing | lines | lines final / in range |
| video | Video download | MB | yt-dlp hooks (the video stream) |
| export | Saving the video | video s | per block of pass 2 (pass 1 counts as the first third); "Finishing the file…" while faststart rewrites it |

**ETA:** the running stage's own rate once it has 30 s of work behind it, else the static priors of §7 (`PRIORS`, with new `separate`, `video` and `export` entries); a group of stages side by side counts as its longest, except the group where the GPU runs `separate` before the dub loop (§2.14); while `waiting` on a usage limit with a known reset, at least that time. The estimate shown before "Dub" uses the same priors (`render.estimate`), shown as a range, plus, when the button reads "Add to queue", the time left of the jobs ahead (`video.estimate.ahead`). The priors are not learned from finished jobs (rejected in Appendix A: an EWMA file); the running stage's own rate takes over after 30 s.

**Client → engine:**

| message | effect |
|---|---|
| `inspect {url}` | metadata only → `video` (with the estimate) |
| `prepare {url, speakers: "auto"\|1..6, style, stopAt: null\|900, speedCap, ttsScript}` | create or resume the job: runs, or `queued` |
| `pause {videoId}` / `resume {videoId}` | §4 the queue |
| `remove {videoId, forget?}` | §4 the queue: validated id; not while it runs |
| `set_speakers {videoId, speakers}` / `set_voice {videoId, speaker, usePreset}` | re-run from that stage (§2.3, §2.9), queued at the front |
| `renders {}` | → `renders` |
| `voice_sample {videoId, speaker}` | one binary frame (ADR-007) with the speaker's Hear voice sample (id `0xFFFFFF00` + speaker index) |
| `open_output {videoId, reveal}` | the engine opens the job's MP4 in the default player, or shows it in Finder (or its folder, when the file is gone) |
| `settings {outputDir}` | updates the engine's settings file → `settings` to every window |

**Engine → client:**

| message | payload |
|---|---|
| `hello` | backend, device, demo, Claude's state, `renders`, the running job's `render`, `settings` |
| `video` | metadata, thumbnail, its job (a `renders` item or null), `estimate: {seconds: [lo, hi], claudeCalls: [lo, hi], ahead: s}` |
| `render` | `{videoId, status, stage, stages: [{key, label, state, done, total, unit, seconds, eta, extra?}], stopAt, finalUntil, eta, elapsed, position?, output?, slept?, coverage?, error?}`, at most 2/s and on every state change, to every window |
| `speakers_found` | `speakers` with talk, share, activity[120], sample; `merged`; `changed?` (built by `RenderJob` today; the UI for it is new, §5) |
| `renders` | items `{videoId, title, channel, duration, thumb, status, position, stopAt, finalUntil, stage, eta, updatedAt, bytes, output, expired}`; `thumb` is a loopback URL to the cached thumbnail |
| `settings` | `{outputDir}` |
| `claude_error` / `claude_ok` | unchanged, plus `videoId` |
| `error` | `{message, retryable}` |

Removed (2026-10-03): `watch`, `audio`, `speed`, `manifest`, `busy`, and every binary frame but the Hear voice sample; the streaming session's `open`, `playhead`, `player`, `seek`, `prepare` (prepare-ahead), `speaker_preset`, `diag`, `close` with it (§6).

**Settings live in the engine**, in `~/Library/Application Support/Maata/settings.json` (next to `Models`; `--config-dir`, a temporary folder in tests), not in the cache, which users and cleaners treat as disposable, and not in the UI's `localStorage`: the UI is served from `http://127.0.0.1:<port>` with a port chosen at each launch, so its storage origin changes every time. The only setting is `outputDir`; New dub starts from the options of the last job (from `renders`), and the theme follows the system.

**Notifications** (`notify.py`): on macOS the engine posts them with `/usr/bin/osascript` (`display notification`, the texts passed as `argv`, never spliced into the script), so no dependency is added; they appear under Script Editor's name and icon, and on recent macOS only once Script Editor is allowed to notify (System Settings ▸ Notifications), which Settings mentions. Nothing depends on them: the Library and the job view show every event they report. A proper app-named notification needs `tauri-plugin-notification`, a new dependency: a question for the maintainer (M14). Sent for: speakers found (with the free-correction window), done (whole or preview), failed, the first Claude hold of a job (with the reset time), and a job continued after a crash. Never for the demo or the mock; tests stub it.

**Retention** (SPEC §8, M6), at startup and whenever a job ends:
- Never the running job, a queued one, or one that is `paused`, `interrupted`, `waiting` or a `done` preview: those wait to be continued, and losing their takes and PCM would cost hours of TTS (and a preview's bed and video, which continuing needs). They lose their voice data only by the age rule.
- A `done` whole-video job's large sources go when its MP4 is written (§2.17).
- Age: a job not updated for 14 days loses its voice data (voices, takes, PCM, bed) and source media, keeping its translations, transcript and speakers.
- Cap: above the 5 GB cache cap, only `done` whole-video jobs (least recently updated first) and expired leftovers are trimmed; a paused job holding a 6 GB video survives another job ending (a test checks).
- The output folder is never touched.

## 5. The app (ADR-009's look)

The same glass cards over the animated backdrop, the turmeric-to-vermilion accent, Inter and Noto Sans Telugu, the "AI dub" badge.

**1. Library** (home; `Library.svelte`, replacing `Hero`'s "Continue watching"):
- The headline and a paste field ("Paste a YouTube link").
- The jobs, newest activity first: thumbnail (the cached one, from the loopback, so the Library makes no network calls), title, channel, length, a status chip and actions.
  - Chips: "Queued · 2nd in line", "Dubbing · Telugu speech 43 % · about 2 h 10 min left", "Paused at Translation", "Continuing after Maata was closed", "Paused while the Mac slept · resumed 06:40", "Waiting for Claude · resets 18:00", "Done · 2:55:06 · 3.1 GB", "Preview done · first 15 min", "File moved or deleted", "Failed · <message>", "Expired · Dub again to re-voice".
  - Actions: **Open video** and **Show in Finder** (done), Pause / Resume, Remove (with a confirm in the card; a running job is paused first).
- The lines "Maata keeps dubbing while its window is closed. Quitting Maata pauses; it continues when you open it again." and "Keep the lid open and the Mac plugged in; the screen can turn off."

**2. New dub** (`NewDub.svelte`, after a pasted link; `inspect`):
- A video card: thumbnail from `i.ytimg.com` (no job yet, so nothing is cached), title, channel, duration.
- Options, starting from the last job's: **Speakers** Auto · 1 · 2 · 3 · 4 · 5 · 6; **Style** Everyday spoken / More formal; **Length** Whole video · First 15 minutes; "More options": speed-up cap and TTS script; the output folder (from settings, with "Change" opening Settings).
- The estimate line: "About 3–6½ h on this Mac · about 310–360 Claude calls · saves to Movies ▸ Maata", plus "after the 1 job ahead (about 2 h)" when the button reads Add to queue.
- The primary button **Dub**, or **Add to queue** while another job runs. A video with a job shows its status and **Continue** / **Open video** instead.

**3. Job** (`Job.svelte`):
- A vertical stepper of the stages, grouped as Download → Speakers → Transcript → Voices & brief → Background sound → Translation → Telugu speech → Finishing → Video download → Saving the video. Each row: state, done/total, time taken and ETA; the running row has a thin gradient bar, clustering an indeterminate one, the dub loop "waiting for the background sound" while separation runs, the export "Finishing the file…" while faststart rewrites it.
- The overall ETA and elapsed time; Pause / Resume; the existing `ClaudeBanner` when Claude waits; the sleep note.
- **Speaker check** (`SpeakersFound.svelte`, new: the engine sends `speakers_found` today, but the app only has the streaming `speaker_scan` handling): "Found 2 speakers", per speaker an avatar in its `speakerHue`, talk time, a share bar and the 120-bin activity strip; merges ("1 short voice merged into Speaker 1"); **Hear voice** once voices exist (a small Web Audio player for the one frame, decoded by `engine.ts`); "Wrong count?" with Auto/1–6 and **Re-run speakers**, saying what it costs now ("free until the background sound is done, in about 20 min" → "lines whose speaker changes are translated and voiced again"); the stock-voice switch per speaker (`set_voice`) with its cost.
- **Done:** the output's name, size and folder, **Open video**, **Show in Finder**, the coverage shares, lines flagged long or unreviewed, overdraft seconds, the loudness, `output.warning` if any; a preview adds **Continue to the whole video**; a missing file shows "File moved or deleted" with **Show folder** and **Save again**.

**4. Settings** (`Settings.svelte`, replacing `SettingsSheet`): the output folder (a path field the engine checks and creates), and a note that macOS shows Maata's notifications under Script Editor, which must be allowed to notify in System Settings ▸ Notifications.

**State** (`state.svelte.ts`): `view: "library" | "new" | "job"`, `jobs` (`renders`), `job` (the open job's `render`), `found` (`speakers_found`), `video` (`inspect`), `settings`, `claude`, `error`. Pure helpers in `lib/jobs.ts` (stage grouping, chip and ETA text, the estimate line, queue position text), tested with vitest. The `?v=` / `MAATA_OPEN` launch path opens New dub with the link.

**The shell** (`main.rs`), with no new crate and no new capability:
- **Close hides (macOS):** closing the main window hides it (`WindowEvent::CloseRequested` → `api.prevent_close()` + `hide()`), so the engine and the job keep running with the Dock icon; clicking the Dock icon (`RunEvent::Reopen`, Tauri 2.11.6) shows it again. Windows and Linux keep today's close-quits behaviour.
- **Quit is graceful:** on `RunEvent::Exit` the shell drops the engine's stdin (the lifeline, §4), polls `try_wait` for up to 5 s, and only then `kill()`s it. ⌘Q is never blocked; there is no quit confirmation (the UI has no access to the shell, `capabilities/default.json`, and a native dialog would be a new dependency).
- **The engine is restarted if it dies.** When the engine's stdout ends while the shell isn't quitting (an out-of-memory kill, an MPS crash), the shell starts it again with a new token and port and points the window at the new URL, hidden or not; the engine's startup recovery continues the job (§4), and the crash-loop guard stops a job that keeps crashing. At most 3 restarts in 10 minutes, then the window shows the error. Without this, an overnight job would stop silently until Maata was quit and reopened.

**As built (step mp4-5).**
- **What the engine adds for the UI** (§4's tables are otherwise as built): `renders` items also carry `thumb` (`/thumb/<id>?token=…`, served from the cache behind the token, only the thumbnail job.json records in the job's own folder; none for the demo), `settings` and `createdAt` (New dub starts from the newest job's options; Continue keeps a job's own), `slept` and `error` (the chips), and `stages` (no ETA), `elapsed`, `coverage` and `report` (the job view of a job that isn't running, which sends no `render`). `video.estimate.preview` is the first 15 minutes' estimate, so the Length switch asks YouTube nothing. job.json and `render` carry `report` (`lines`, `long`, `unreviewed`, `overdraft`, `carried` from the manifest) for the Done card. `inspect`'s `video` and `error` answers echo the `url` asked, so the UI shows only the answer to the link pasted last. job.json's `output.bed` says whether the MP4 has the background sound (the Done card's "over the original music and sounds" only then). The CSP keeps `'self'` and `i.ytimg.com` images; `script-src 'self'`, `frame-src 'none'`.
- **New dub** starts from the newest job's style, length, speed-up cap and TTS script, but always at Auto speakers: a count belongs to its video, and a wrong one silently changes the voices.
- **Remove** of a running job pauses it, then asks the engine again every 0.7 s until its run has returned (the engine answers "Pause it first." meanwhile), for up to 2 minutes.
- **The speaker check**: a job broadcasts `speakers_found` when its speakers stage runs or is served, and the engine sends each window, right after `hello`, the check of every job whose speakers stage is done for its count, rebuilt from diarization.json (`render.found_on_disk`). A reload, an engine restart or a relaunch keeps it, for a paused, queued or done job too. The job view hides it while that job's speakers stage runs again (a Wrong count? in progress), and a removed job's goes with it.
- **The library is the newer word for a job that isn't running**: on `renders`, the UI drops the last `render` of every job not running or waiting (the engine sends no `render` when the queue moves up or an MP4 goes missing).
- **Quit**: ⌘Q goes `applicationWillTerminate` → `RunEvent::Exit` (tao 0.35.3), never through `CloseRequested`, so hiding on close can't catch it; the exit hook waits at most 5 s. Checked on this Mac with a debug build on the mock engine: a killed engine is restarted (new port, the window follows) three times, the fourth stays down and the window shows the error page (`splash.html?error=`); a shell killed mid-job leaves the job `interrupted` (the stage's start not counted) and it continues at the next launch. ⌘Q itself was not driven (no UI automation grant).

## 6. What goes, what stays

The rule: keep every pure piece and every per-line method; delete everything whose only job was to beat a playhead or play in the app.

| Piece | Fate | Why |
|---|---|---|
| In-app YouTube player (`player.ts`, `Stage.svelte`, `Scrubber.svelte`), the IFrame CSP entries | delete | the output is a file |
| `SyncEngine`, `VideoClock` (`sync.ts`, `clock.ts`), their tests; freezes | delete | nothing plays in the app; files have no freezes |
| Buffer pacing (`buffer.ts`: `MissingAudio`, `bufferStep`, `startDecision`, lead), `SidePanel.svelte`, `DebugHud.svelte`, the watch view of `App.svelte` | delete | |
| `SettingsSheet.svelte` (look-ahead, prepare-ahead, clone strength, playback speed), `voice.ts` and `cloneStrength` | delete; `Settings.svelte` replaces it | dead or streaming-only |
| Streaming `Session` (`session.py`: pre-pass, blocks, frontend loop, translator loop, horizon, governor, provisional takes, PCM window, `Chunk`), `pacing.py`, `diarize_block`, the streaming protocol in `server.py` | delete, after the line-level tests are ported to `RenderJob`/`Dubber` while `Session` still exists | the background job replaces it (M7 answered on 2026-10-03) |
| `Playback` (`playback.py`: `ask`, `set_speed`, `update`), `watch`, `audio`, `speed`, the `manifest` push, `busy`, `watchedAt` | delete; keep `frame()` and the Hear voice sample | nothing is watched |
| Progressive manifest (`MANIFEST_GROW/WALL/GAP`, `_floor`, `_old_pcm`) | delete; the manifest is written once (§2.13) | |
| `bench.realtime_metrics` and the bench's `Session` run | replace, once (§10 step 3), with a `RenderJob` run through `LocalResolver` and `render_metrics` (per-stage seconds and GPU-s, GPU-s per video-s, peak RSS, Claude calls and tokens by type) | |
| `Dubber`, `RenderJob`, `SpeakerRegistry`, `TimelinePlanner` (W and F), `GpuScheduler`, the translator, `scene_cut`, `_scene`, `_review`, `_fit`, the take cache, the fix-ups, `_play`, `_save_pcm`, `vocode`'s watermark, `ClaudeBanner` | **keep** | the job is built from them |
| `SpeakersFound.svelte`, `Library.svelte`, `NewDub.svelte`, `Job.svelte`, `Settings.svelte`, `lib/jobs.ts` | **new** | the job UI (§5) |
| The loopback-served UI (ADR-006) and the binary frame (ADR-007) | **keep** | the UI still comes from the engine; the frame carries the Hear voice sample |

**As built (step 6, 2026-10-04).** Session's per-line tests were ported first (`test_lines_translate.py`, `test_lines_voice.py`, `test_lines_asr.py`, and `test_calibration.py`, `test_trace.py` rewritten on a `RenderJob` with no stage run: `tests/fakes.py`'s `bare_job`, `put`, `voice`), with both suites green, then deleted: `session.py`, `pacing.py`, `server.py`'s streaming branches and `DIAG_FIELDS`, `diarize_block` (the protocol, the mock, both torch diarizers), `bench.realtime_metrics`, and from `Dubber` what only `Session` used: the `_retake_ok`, `_unit_done` and `_speakers_changed` hooks, `_clone`, `_build_voice` and its own `_voice_for` (a clone on a line's path; `RenderJob._voice_for` never makes one), its resynthesis budget, `UnitState.chunk`, `provisional` and `replacement`, the unused `_after`, and the short first scenes: `scene_cut(run)` is the whole-video cut that `scene_cut(run, 2, True)` was, and `_scene`/`_release` lost the `nth` of a run of scenes. `TimelinePlanner.reset` (a seek's) went too, and `playback.frame` lost its playback rate. Kept: `SpeakerRegistry.add_block` (the speakers stage registers its one block), `DemoResolver.load_audio` (the fetch stage's fallback with no file), WSOLA (`_render` for a TTS without mel takes), and timing v1 and v2-whole, which no flag selects any more (a question for the maintainer). The bench reports MLX's peak memory and the separate stage's blocks.

## 7. Budgets on the M5 Pro for the 2.9 h video

**Measured inputs** (his run, 2026-09-25, streaming, stages interleaved): diarization 0.031–0.058 GPU-s per s of audio; ASR 0.046–0.143 GPU-s per video-s; voice build and calibration 3.7–7.9 + 19–34 s per speaker; TTS 1.63 GPU-s per s of Telugu contended (T3 median 27.5 ms/token there, 12.7 over all his runs); Claude: scene calls median 103 s and 624 output tokens/line (Opus 5.5 medium), review 34 s and 163 tokens/line (Sonnet 5 high); 0.18 lines/s; scipy centroid linkage at 2.1× its condensed matrix.

**Wall time per stage, one GPU stage at a time (estimate):**

| Stage | Work | Wall | Basis |
|---|---|---|---|
| Fetch | audio 162 MB (already in his cache) and the thumbnail; decode to 16 kHz | 1–2 min | |
| Video download | 1080p H.264, about 2–5 GB at YouTube's 1.5–4 Mbit/s for talking heads | 2–10 min (more if YouTube throttles), beside the dub loop: no wall time | download speed unmeasured (§2.2) |
| Speakers | 10,506 s | 6–12 min + 1–3 min CPU clustering | |
| Transcript | 10,506 s | 8–25 min | |
| Voices ‖ brief | 2 speakers ‖ 3–4 parts | 2–4 min | |
| Separate ‖ translate | 10,506 s at 44.1 kHz | **6–20 min** (twice that in float32, §2.14) | ≈2.9 TFLOP per 8 s chunk by layer arithmetic (48k tokens × 6 × (time + frequency layer) at dim 384), so ≈0.73 TFLOP per audio-s with 50 % overlap; 7.6 PFLOP at 8–25 TFLOP/s effective in bf16 |
| Translate | ≈1,900 lines from scratch: ≈75–90 scene, ≈80–90 review, ≈75 fit, ≈25 re-translate calls on 3 lanes | 60–90 min, beside separation and the dub | measured medians ÷ 3 |
| Dub: voice lines + final PCM | ≈9,400 s of Telugu | **2.4–5 h** | idle ≈0.83 GPU-s per audio-s, contended 1.81; PCM re-vocoding inside |
| Export | 2 mix passes, AAC, mux 2–5 GB, the faststart rewrite | 3–8 min | numpy blocks; ebur128 and AAC far faster than real time; no re-encode (§2.17) |
| **Total** | | **≈2.9–6.4 h** | the dub loop dominates |

- The dub loop starts after speakers, transcript, voices and separation: about **25–65 min** after the audio download **(estimate)**; the lanes have translated the first 30–90 min of video by then, and the video download has finished.
- A first-15-minutes preview: the whole-file stages (about 17–45 min), separation of 15.5 min (about 1 min), 15 min of voicing (about 12–25 min) and the export (seconds): **about 30–70 min**.

**Peak memory (engine; the WebView adds its own, far less without the YouTube player) (estimate):**

| Phase | Resident |
|---|---|
| Speakers | runtime 2–3 GB, pyannote < 0.5 GB, the waveform tensor 0.67 GB, the clustering transient ≤ 3 GB (the guard): **6–7.5 GB** |
| Transcript | runtime + Whisper ≈ 2 GB: 4–5.5 GB |
| Separate (Chatterbox already loaded by the voices stage) | runtime + Chatterbox 2.2–3.2 GB + separator 0.46 GB + 1–2.5 GB per batch + 3 Claude CLIs 0.6–1.2 GB: **6.5–10 GB** |
| Dub | runtime + Chatterbox + 3 Claude CLIs (separator released): 5–7.5 GB |
| Export (the TTS released at the end of finish, §2.13) | runtime + 10 s blocks and PyAV: < 4 GB |

Target ≤ 13.5 GB (ARCHITECTURE §5.1).

**Claude (estimate):** the register retarget voids the line cache, so the whole video is translated from scratch: about **310–360 calls** and **1.65M output tokens** (scene ≈1.2M, review ≈0.3M, the rest ≈0.15M), from the measured per-line rates. Opus 5.5 usage comes out of the maintainer's weekly Opus allowance (ADR-019); a 5-hour-window limit makes the job `waiting` while GPU work goes on.

**Disk (estimate) for his video:** during the job: video 2–5 GB, audio 0.16 GB, analysis audio 0.67 GB, bed 1.86 GB (44.1 kHz stereo float16) and its vocals envelope 0.4 MB, PCM 0.45 GB, mels 0.08 GB; at export the MP4 adds 2–5 GB (on the output folder's volume, which may be another disk), so the peak is **7.5–13 GB**. After a whole-video export the cache keeps about 0.7 GB (audio, takes, PCM, the JSON files) and the output folder holds the MP4 (the video's size plus about 0.25 GB of audio).

**How the Mac stays usable for hours:** one GPU stage at a time (no MLX/MPS contention); memory under 10 GB outside diarization, which is guarded; Pause at any time; the window can be closed; Quit pauses gracefully; the shell restarts a dead engine; `caffeinate -i` stops idle sleep, not lid-close sleep (the job stops while the lid is shut and continues after a wake; the app says so); Claude stays at 3 CLI processes.

## 8. What this design does not fold in

- **Voice building** (pooled S3Gen x-vector, clips spread over time slices, one build path): ARCHITECTURE step 8, gated by a blind "sounds like him/her" test. The voices are still cloned from the original mixture, before separation runs, so on a video with a music bed the music goes into the clone reference. This work makes the fix cheap: separate only the chosen reference spans (a few chunks, seconds of GPU) before `_build_voice`, as one arm of that blind test; it is not part of this plan.
- **Loudness matched to each original speaker** (ARCHITECTURE §3.11): the mix equalises the Telugu speakers to each other and keeps the original voice/bed balance (§2.15), but doesn't copy each English speaker's own level.
- **The watermark at the end of the chain** (§3.8): PerTh stays per take inside `vocode` (ADR-014); its survival through the mix and AAC is checked, not changed.
- **Running a queued job's download or translation while another job has the GPU**: one job at a time is simpler, and the GPU is the bottleneck.
- **Re-encoding the video** (VP9 to H.264 with VideoToolbox): H.264 and AV1 play on his Mac as copied; a VP9-only video is copied with a warning (M16).
- **Learning the ETA priors from finished jobs**: the running stage's own rate takes over after 30 s (§4).
- **Sidecar `.srt` files** and burnt-in subtitles: the soft tracks are what was asked for.

## 9. Tests (mock backend, no models, no Claude, no network)

The mock backend dubs `DemoResolver`'s synthetic `demo.mp4` (§2.2) end to end offline: `MockDiarizer.diarize`, `MockTranscriber`, `MockSceneTranslator` over `MockClaude`, `MockTTS`, `MockSeparator`. The CPU fake mel TTS of `test_batched_takes.py` covers batched takes and re-vocoding. Media is synthesised in the tests with PyAV and numpy (colour bars, tones, beeps), never downloaded; the A/V, cleanup and resume tests also run on the YouTube-shaped two-file fake (§2.2). The real separator weights are never loaded.

**Engine:**
- `tests/test_resolve.py` (with a monkeypatched `YoutubeDL`): the audio instance and a second video instance with `process_ie_result`, both with `fixup: "never"`, `remote_components: []`, no `+` in any selector, `VIDEO_FORMAT` exactly; the thumbnail's `outtmpl`; partial files ignored. The demo file's beep and flash at exactly 10.0 s, and its B-frames (some packet with dts ≠ pts).
- `tests/test_separate.py`: the global chunk grid's overlap-add reproduces a whole-file run block by block (MockSeparator: bed = mixture/2 exactly, across chunk joins and block edges); a preview separates only its blocks and continuing adds the rest; a pause between batches and a resume skip the done blocks and give identical samples (decoded from the start, not sought); a torn block file is redone; the inputs fingerprint (model, chunk, block, dtype, `SEP_VERSION`) invalidates; `bed_vocals.npy` equals the vocals stem's mean square per 0.1 s; a mono source is duplicated to stereo; a 3 s file (zero padding) and a 48 kHz Opus source work; the 16 kHz analysis decode and the 44.1 kHz decode put the demo's beep at the same time (±1 ms).
- `tests/test_mel_roformer.py` (skipped where `mlx` can't be imported): a tiny-config forward with random weights (shapes only); `istft(stft(x))` returns `x` (the window normalisation); `sanitize` on a tiny synthetic safetensors file with the real checkpoint's key names, loaded with `strict=True` (a key `sanitize` misses fails the load); the real `config.json`'s fields build the full config.
- `tests/test_mix.py`: lines are added at their `start` (a click in a line's PCM appears at the planned sample at 44.1 kHz, after the resampler); two overlapping lines of different speakers are both audible, summed; per-speaker gains equalise two speakers 6 dB apart within 0.5 LU and clamp at ±6 dB; the balance matches the original vocals' level over the current turns; the ducking envelope is −6 dB inside bridged spans, −12 dB in bare English speech, with the specified ramps, and 0 dB elsewhere; the output's integrated loudness is −16 ± 0.5 LUFS and its true peak ≤ −1.0 dBTP after AAC on a synthetic mix with hot peaks; the limiter adds no delay (an impulse keeps its sample); `ebur128` reads −23.0 LUFS on the EBU 1 kHz case; no bed gives voices only with the warning.
- `tests/test_subtitles.py`: cue splitting at 2 rows of 42 characters (English) or 42 aksharas (Telugu), akshara-proportional timing for Telugu and word-timed for English, minimum duration, overlapping cues merged with both texts and "– " prefixes, preview clipping, and the tx3g packet layout.
- `tests/test_export.py`, on the demo and on the two-file fake: exactly one video stream whose packets are byte-identical to the source's (stream copy), one AAC stream at 44.1 kHz stereo with `language=tel` and the default disposition, two `mov_text` streams `tel` and `eng` with their cue texts at their times; durations within one video frame (plus a last overrun); **A/V sync within ±2 ms**: the 1 kHz beep's onset in the decoded output audio (a narrow 1 kHz detector; the demo's mock lines stay clear of 9–11 s) against the flash frame's time, and a mock line containing a click, planned at the flash time, against the same frame; `moov` before `mdat`; the metadata; the file appears only under its final name. Also: a preview with a line straddling `stopAt` ends at the first keyframe after that line's end; an AV1 source (`libsvtav1`) is copied, and a VP9 one copied with the warning; name sanitising and " (2)" on a foreign file; pause mid-export deletes the `.part.mp4` and a resume gives a byte-identical file; a whole export deletes `video.*`, the bed and the analysis audio and a preview keeps them; a bench export leaves its input file byte-identical; a done whole job run again with no change calls neither the resolver's video download nor the separator, and serves the MP4.
- `tests/test_render.py` (kept, adjusted): the existing stage, resume, fix-up and final-plan tests, with freezes gone (both planners at `max_freeze == freeze_budget == 0`; overruns become carried overdraft, never a freeze, the next line after it with no overlap, the line flagged `long`) and the manifest written once (no file before finish completes); `test_the_dub_loop_waits_for_the_separator`; `test_a_speaker_rerun_doesnt_separate_again`; `estimate()` counts `separate` in series with the dub loop.
- `tests/test_engine_ws.py`: `prepare` reports progress through every stage to `done` with `output` set; a second `prepare` is `queued` and runs after the first; `pause` of the running job starts the next; `resume` queues; a re-run goes to the front; startup re-queues interrupted jobs first and the crash-loop guard holds; **graceful quit**: a lifeline EOF during a running stage leaves that stage `todo` with `attempts` unchanged and the job `interrupted`, three graceful quits never trip the guard, three crashes (the job left `running`) do; `remove` refuses an empty id, `..` and a running job, keeps the translations, and `forget` deletes the folder; `open_output` opens only the job's own recorded path (the runner is stubbed); `voice_sample` is dub audio; notifications fire on speakers found, done, failed and a crash recovery (stubbed); `settings` persists `outputDir` across engine restarts; retention never touches the output folder, a queued, paused or preview job (a paused job holding a 6 GB fake video survives another job ending), and ages by `updatedAt`.
- `tests/test_scene_prompt.py`: the `scene-v3` pin and the new `BRIEF_HASH` pin; `REVIEW_HASH` unchanged; every rewritten example passes the validators. `tests/test_tenglish.py`: `lint-v4`, and the everyday loans no longer reported.
- `tests/test_models_bench.py`: `maata-bench pipeline FILE --backend mock` on a synthetic video runs the job to an MP4 and prints `render`, `dub`, `timing` and `export` blocks.

**UI (vitest + svelte-check):** `jobs.test.ts` (stage grouping, chip text per status including the sleep and missing-file chips, ETA text, queue position, the estimate line with the jobs ahead); `state.test.ts` (views on `video`, `render`, `renders`, `speakers_found`, `settings`).

**Demo:** `uv run maata-engine --backend mock --demo --ui ../app/dist --token demo`, open `http://127.0.0.1:8765/?token=demo`, paste any link → New dub → Dub → the stages → Done → the MP4 in the output folder plays with Telugu (mock) audio over the demo's bed and both subtitle tracks.

## 10. Implementation plan

Six sequential steps from the worktree as it stands (steps 1–5 of the first plan built; step 5 unreviewed, with 3 of its render tests failing in the full suite). Each leaves the engine and UI suites green (step 1 excepts those three known failures, which step 2 fixes); the last leaves the app working end to end: paste a link, get a dubbed MP4.

1. **The translation register** (§2.18): the scene prompt, the brief prompt and the lint, with their pins. Small and independent, so the maintainer can judge the Telugu early.
2. **The job, retargeted (engine).** Review step 5 and fix its three failing tests (the loop test passes on its own and fails only under the full suite's load: first look for a real on-loop callback with asyncio's debug mode and `slow_callback_duration`; if there is none, make the test measure the loop's own stalls rather than the largest gap under GIL contention); no freezes (§2.11); the manifest written once (§2.13); `Playback`, `watch`, `audio`, `speed`, `busy` and `watchedAt` removed; the FIFO queue, `remove` and recovery; the graceful stop on the lifeline; the sleep note and the queue-ahead estimate (§4).
3. **The video and the MP4, voices only.** The `video` stage, the thumbnail, the demo video, `LocalResolver` and the two-file fake (§2.2); `mix.py` without a bed, `subtitles.py`, `export.py` and the `export` stage (§2.15–§2.17); the TTS release; settings, `open_output` and notifications; retention and the owned-file cleanup; the bench rewritten once to a `RenderJob` and an MP4; `scripts/check-youtube-video.sh`.
4. **The background sound.** The `Separator` API, the vendored MLX Mel-Band RoFormer (model, config, dsp), its pinned weights; `separate.py` and the `separate` stage with the vocals envelope; the bed in the mix (balance, ducking); the served-from-output rule for `separate`; the ETA in series; ADR-020.
5. **The app.** The Library, New dub, Job, speaker check and Settings views (§5); the in-app player and its pacing removed; the shell's hide-on-close, reopen, graceful quit and engine restart.
6. **The streaming engine removed, and the whole checked.** First the line-level tests ported to `Dubber`/`RenderJob` while `Session` still exists (both suites green, the mapping in the notes), then `Session`, `pacing.py` and the streaming protocol deleted (§6); `verify-mac.sh` on a synthesised speech + music video; the docs (ADR-021, ARCHITECTURE pointers, README); the demo run end to end.

## 11. Decisions for the maintainer

- **M1 — Cached translations from the streaming runs.** Moot: the register retarget (§2.18) changes `PROMPT_HASH` and `BRIEF_HASH`, so every line is translated again under the new register and a new brief (about 90–100 more calls and 0.5M more Opus output tokens on his video than reusing his 776 cached lines).
- **M2 — Speaker settle defaults.** Unchanged: auto bounds 1–6; merge under max(30 s, 1 % of talk) and pairs at cosine ≥ 0.85; a chosen count disables merging.
- **M3 — The speaker check doesn't stop the job.** It is shown, notified and free to correct until the dub loop starts (about 16–50 min after the speakers stage on his video). The alternative is a "Check speakers before dubbing" option that pauses the job until confirmed; recommended against for overnight jobs.
- **M4 — Quality-first synthesis budget.** Unchanged: N=2 takes (3 under 3 s), a third seed on failure, fix-ups on up to 15 % of lines.
- **M5 — Diarization memory fallback.** Unchanged: a 2 s step over a 3 GB clustering estimate or after a speakers stage a crash interrupted (a quit no longer counts, §4).
- **M6 — Retention.** Once the whole video's MP4 is written, its source video, bed and analysis audio are deleted (2.5–7 GB on his video); a re-run that changes no line serves the MP4 without fetching or separating, one that changes lines downloads and separates again. Paused, queued and preview jobs are never trimmed by the 5 GB cap, only by the 14-day age rule. The output folder is never touched. Keep the sources instead (faster re-runs, more disk)?
- **M7 — Real-time mode.** Answered on 2026-10-03: the in-app player and the streaming engine are deleted (§6).
- **M8 — Keep-awake.** `caffeinate -i` while a job runs; it can't stop lid-close sleep, so the app asks to keep the lid open and the Mac plugged in.
- **M9 — ADRs and CLAUDE.md.** ADR-020 records the separator and its pin (step 4). ADR-021 records the background MP4 job: it supersedes ADR-016 (streaming, pacing), ADR-008 (sync) and ADR-006's player half, and amends ADR-007 (one frame left), ADR-017 (no freezes; the register) and ADR-019 (the in-pipeline section, the brief's register). The CLAUDE.md status and demo lines are the maintainer's to update (step 6 proposes the text). Its hard constraint "The original English speech never reaches it" promises more than a separator can: the proposed wording is "The original English speech is removed by the on-device separator (a faint residue can remain where no Telugu covers it)".
- **M10 — The separator.** Mel-Band RoFormer Kim Vocal 2 in MLX (§2.14), the default by the best published proxy (instrumental bleedless on Multisong). A bake-off on speech content, judged by the English left in the bed over speech turns and by listening on his podcast: Kim, Kim with test-time augmentation, BandIt Plus (dialogue/cinematic, PyTorch MPS), becruily deux and viperx's Mel-Band model. The vendored code loads any Mel-Band preset. Kim removes laughter, crowd voices and singing with the speech; say if a video type needs them kept.
- **M11 — Mix targets.** −16 LUFS integrated, −1 dBTP, Telugu speakers equalised to each other, the original voice/bed balance, the bed −6 dB under speech and −12 dB under English speech that no Telugu covers. A listening check on the M5 Pro sets `DUCK_DB` (−4 to −10 dB) and `DUCK_BARE_DB` (−10 to −20 dB).
- **M12 — Subtitles off by default** in the file (picked in the player's menu); or the Telugu track on by default.
- **M13 — Output folder and names.** `~/Movies/Maata`, `<title> (Telugu).mp4`, `<title> (Telugu, first 15 min).mp4`; the preview file is kept when the whole video is dubbed.
- **M14 — Notifications** through `osascript` (no dependency; shown as Script Editor, which must be allowed to notify) now; `tauri-plugin-notification` (a new Rust and npm dependency, so a question) would show them as Maata.
- **M15 — Overruns without freezes:** drift (a line's lateness carried until a pause; default) or a higher speed cap for those lines, up to the 1.25× `Dubber` allows.
- **M16 — Videos without H.264 at ≤ 1080p:** AV1 is copied (it plays in QuickTime on M3 and later Macs, so on the M5 Pro); a VP9-only video is copied with a warning (IINA or VLC play it, QuickTime doesn't). A VideoToolbox re-encode to H.264 (10–20 min for 2.9 h, some picture loss) can be added if VP9-only videos turn up.
- **M17 — The CUDA backend** gets no separator in this plan: its export has Telugu voices only, and says so. A PyTorch port of the same checkpoint is the follow-up.
- **M18 — YouTube's JavaScript challenges.** Run `scripts/check-youtube-video.sh` on the M5 Pro on his video before the first real job (§2.2). If the video formats are missing or throttled, the fix is the `yt-dlp-ejs` package (a new dependency) with `deno` passed to yt-dlp by path: approve?

## 12. Risks

- **Separation residue.** Kim's bed is the best public one by bleedless score, but reverb tails, breaths and faint words of the English can remain; ducking masks them under Telugu speech and lowers them where no Telugu plays, but doesn't remove them. Songs lose their singing in the bed; translated lyrics are spoken. `verify-mac.sh` measures the English left in the bed on its test mix against a threshold.
- **Separation speed, precision and parity on the M5 Pro are unmeasured.** The 6–20 min is FLOP arithmetic; the MLX/PyTorch parity is the converter's report on one 8 s clip; bf16 against float32 is untested. The M5 Pro run measures them (`verify-mac.sh`: time per block, MLX peak memory, SI-SDR of the vocals on a synthesised speech + music mix, bf16 against float32) and fails below an SI-SDR floor before any long job depends on the separator.
- **YouTube's video formats.** yt-dlp's access to the video-only DASH formats can break separately from audio (signature, the n-challenge without a JavaScript runtime and `yt-dlp-ejs`, PO tokens, SABR-only clients); the `video` stage then fails within the job's first hour with the reason, and a resume after a deliberate yt-dlp update keeps all the work done. M18 checks it before the first job.
- **Whole-file pyannote on 2.9 h is unmeasured** (the guard bounds clustering; GIL pauses in pdist may stall the event loop for seconds near the budget).
- **Hours before the file exists.** 2.9–6.4 h for his video, more under thermal throttling; the preview gives the first 15 minutes in 30–70 min.
- **Voice-line cost is the least certain number** (T3 idle speed and the batched N=2 cost on MPS are unmeasured).
- **Drift without freezes.** A long line pushes the next ones late until a pause; on dense talk, lines may run more than 0.6 s behind for a while. The stats report it; M15 is the lever.
- **The watermark after the mix and AAC** is unverified until the M5 Pro check.
- **Claude usage:** from scratch, about 310–360 calls and 1.65M output tokens may hit a 5-hour window; the job waits and the GPU continues, and scenes reached during a hold are voiced unreviewed and flagged.
- **The register change** is judged by the prompt's examples and a native listener, not by a metric. The review is meaning-only and the lint only reports (it never blocks or retries a line), so nothing in the pipeline stops a line with too much English or too bookish a word.
- **Disk:** 7.5–13 GB at the peak for his video; the free-disk checks (at fetch, at the video download and before the export, on the cache's and the output folder's volumes) fail a job early rather than half way through writing.
- **Cache staleness after code changes:** every fingerprint and key carries a version (`DIAR_VERSION`, `ASR_VERSION`, `VOICE_VERSION`, `TAKES_VERSION`, `MIX_VERSION`, `SEP_VERSION`, `EXPORT_VERSION`, `PROMPT_HASH`, `BRIEF_HASH`).
- **Closing is not quitting, and the lid matters.** On macOS the job runs on with the window closed; Quit pauses it; closing the lid stops it until the Mac wakes. The Library says so; notifications report the end, if Script Editor may notify.
- **App Nap and background priority** for a hidden window are unmeasured: the engine is a child process, not a GUI app, so it shouldn't be napped, but the M5 Pro walkthrough compares a preview's GPU-s per video-s with the window hidden against the bench's.
- **QuickTime's Telugu subtitles** depend on system fonts and its tx3g rendering: expected to work, checked by eye on the M5 Pro.

## Appendix A. Review changes (2026-09-25)

Two reviewers (engineering and product) checked the first draft against the code and the spec; every finding was checked against the worktree before it was acted on. Items marked † were later changed by the 2026-10-03 request (Appendix B).

**Accepted**
- *Export removed* † (both, blocker): SPEC §2 and §13 excluded exporting dubbed media then. The maintainer's request of 2026-10-03 makes the export the product.
- *Cancel freed the GPU while the thread ran* (engineering, blocker): `_on_gpu` and the cooperative pyannote abort (§2.1), with a concurrency test.
- *Clustering memory about 2× the estimate* (both): the guard at the segmentation hook, the 2 s step fallback, lazy Chatterbox and Whisper release (§2.3).
- *No recovery after a quit or crash* (both): `interrupted`, auto-resume at startup and the crash-loop guard (§4).
- *Speaker re-runs invalidated everything; take rows keyed by unit id* (both): `match_ids`, voice reuse, content-keyed take rows (§2.3, §2.9).
- *Rephrases overwrote reviewed lines in `lines.jsonl`* (engineering): rephrase answers go to `fixups.jsonl` (§2.10).
- *The voice worker priced lines against an empty chain; the global plan waited for nothing* (engineering) and *nothing to watch for hours* † (product): the working planner W and the trailing final plan F (§2.9, §2.11). The progressive manifest that let the prefix be watched is gone (§2.13).
- *Coverage gate too late; the voiced wording often not the reviewed one; fits too late* (engineering): review retries after a hold, the voiced-wording review, waiting for a scene's fit (§2.8–2.10).
- *Step 2 was not a pure move; the order was wrong* (engineering): the `Dubber` extraction came first, with explicit hooks.
- *SPEC §8 retention* (both): GC, expiry and the cap (§2.13, §4), a free-disk check.
- *Loudness matching without the §6.3 ruling; the watermark move* † (product) and *wave-3 voice items without their A/B* (both): removed then. The 2026-10-03 mix equalises the Telugu speakers and keeps the original balance (§2.15); the watermark stays in `vocode`; the voice items stay out (§8).
- *Watch pull-only; `cloneStrength` dead; ADR names the SPEC text it amends* † (product): watching is gone; `cloneStrength` is deleted (§6).
- *"Hear voice"* (product): kept, from a calibration take (§2.6, §5).
- *M1 framed as free* (product) and *PROMPT_HASH* (engineering): M1 is now moot (§2.18, §11).
- Minor engineering and product items: as built in steps 1–5.

**Adapted**
- *Quit confirmation* (both): "quitting pauses, opening resumes", stated in the UI; now also "closing the window keeps dubbing" (§5).
- *Delete streaming only after an M5 Pro run* † (product): the 2026-10-03 request decides it; the deletion is the last step (§10).
- *Range preview*: a stop point ("First 15 minutes"), now exported as its own MP4 (§3).
- *Retry a failed review before the scene counts* (engineering) against *never stall the GPU on Claude* (product): both, as built.

**Rejected**
- *A second queued job taking the GPU while the first waits on Claude* (engineering): still rejected; the 2026-10-03 queue runs one job at a time (§4).
- *Hour windows linked by centroids*: replaced by the coarse step.
- *A pcm hash in the binary frame header*: moot (no playback).

## Appendix B. The retarget (2026-10-03)

What the request changed, and why:
- **The product is a file.** The export the first design removed is now the deliverable: `separate` and `export` stages, the mix, the subtitles and the MP4 (§2.14–§2.17). SPEC §2 and §13's exclusion of exported media is lifted by the maintainer for personal viewing (CLAUDE.md, amended).
- **Background, queued.** The single job slot with `busy` became a FIFO queue; closing the window keeps the job running on macOS; notifications report it (§4, §5).
- **No in-app playback.** The player, sync, pacing, `Playback` and the progressive manifest go (§6, §2.13); the manifest is the export's input, written once.
- **No freezes.** A file keeps the video's timing: overruns become carried drift (§2.11).
- **The original music and effects** stay under the Telugu, separated locally (Kim's Mel-Band RoFormer in MLX, vendored, pinned), ducked under speech and loudness-normalised; the English speech is removed by the separator, with a faint residue possible (§2.14, §2.15, Appendix C).
- **Licences don't constrain choices** (personal viewing): the separator was chosen on quality and on an MLX path; no licence analysis is recorded.
- **The register** is the everyday Telugu of the general public, with English where ordinary people use it (§2.18); this voids the line cache, which makes M1 moot.
- **Step 5's open items** (its review never ran; three render tests fail in the worktree) are in step 2 of the new plan (§10), after the register (step 1).

## Appendix C. Review changes (2026-10-03)

Two reviewers checked the retarget against the code and the sources; every finding was checked before it was acted on (the checks are named). Section numbers point to where the change landed.

**Accepted**
- *The vendored model doesn't import on its own* (both): checked on GitHub. `mel_roformer/model.py` at `a8546e64ac5f` imports `stft`, `istft`, `mel_filters` from `mlx_audio.dsp` and `get_model_path` from `mlx_audio.utils`; the commit before the refactor (`44481968f5`) used `librosa`. Upstream loads with `strict=False`. Now: model, config and the needed `dsp` functions vendored from one commit, `strict=True`, and tests for `sanitize` with the real key names and for `istft(stft(x))` (§2.14, §9).
- *Quit hard-kills the engine and counts as a crash* (engineering): checked in `main.rs` (`child.kill()` on `RunEvent::Exit`), `server._exit_when_stdin_closes` (`os._exit(0)`) and `RenderJob._stage` (only `CancelledError` takes a start out of `attempts`). Three quits during the dub loop would fail the job; one during speakers would force the coarse step. Now a graceful stop on the lifeline and in the shell, with tests (§4, §5).
- *Re-running a done job downloads and separates again* (engineering): checked in `_fetched` and the stage fingerprints. Now the video moves to its own stage, and `video` and `separate` are served from an output that still matches (§2.1, §2.2).
- *The 5 GB cap expires paused jobs* (engineering): checked: `_run_job` sets `self.job = None` before `retain(running=None)`, and `expire` deletes takes and PCM. Now only done whole-video jobs and leftovers count toward the cap (§4).
- *`alimiter` delays its output* (engineering): measured on this Mac with PyAV 18.1.0: 219 samples (5 ms) by default, none with `latency=1`. Now `latency=1`, the sync tests at ±2 ms, and the loudness measured after AAC (§2.15, §9).
- *Lines of different speakers do overlap* (engineering): checked in `TimelinePlanner._floor` (`overlaps_prev`, the `inside` horizon). Now the true rule, the mix adds, and the subtitles merge overlapping cues (§1, §2.11, §2.15, §2.16).
- *Cleanup could delete the demo's or the user's file* (engineering): `DemoResolver` and `LocalResolver` return one file for both streams. Now only `video.*` in the job's own folder, owned, is deleted; `LocalResolver`'s path is recorded whole (`_describe` stored only the name) (§2.2, §2.17).
- *The demo is one muxed file; YouTube sends two* (engineering): the YouTube-shaped two-file fake, B-frames in the demo, a sync check of the dub itself, and one decode origin for 16 kHz and 44.1 kHz (§2.2, §9).
- *Two format selections in one fetch, `+`, `fixup`, the unbounded `bv`* (engineering): checked: one `YoutubeDL` with a fixed format today, Homebrew `ffmpeg` and `deno` on this Mac's terminal `PATH` only, no `yt-dlp-ejs` in `uv.lock`. Now a second instance with `process_ie_result`, `fixup: "never"`, a bounded selector, and a check script for the maintainer (§2.2, M18).
- *verify-mac.sh can't measure separation* (engineering): checked: it builds a speech-only `say` WAV. Step 6 now synthesises a speech + music video and reports SI-SDR, the English left in the bed, time per block, memory, the A/V offset, loudness and the watermark (§10).
- *`remove` on an unvalidated id* (engineering): the id pattern, a direct child of the cache, not while running, translations kept unless `forget` (§4).
- *Speech energy depends on the speakers; the export's fingerprint misses inputs* (both): a 10 Hz vocals envelope independent of the speakers, and the English cues and duck spans in `output.inputs` (§2.1, §2.14, §2.15).
- *Resume by seeking; reflect padding on short audio* (engineering): decode from the start and discard; zero padding under twice the pad (§2.14).
- *bf16 weights, float32 compute* (engineering): an explicit compute dtype, measured on the M5 Pro (§2.14).
- *The ETA in the separate group* (both): `separate` in series with the dub loop; the queue ahead in the estimate (§2.14, §4).
- *The preview cut can end a line mid-word* (engineering): the cut follows the last line's end; the bed covers it (§2.17, §3).
- *Disk at export; settings in the cache* (engineering, product): checks on the output volume at the video stage and before the export; settings in Application Support (§2.2, §2.17, §4).
- *Export pause and memory* (engineering): per-block pause with the `.part.mp4` deleted, "Finishing the file…", and the TTS released before the export (§2.13, §2.17, §7).
- *Wrong statements in step 1* (engineering): checked: `RenderJob` passes `s.allow_freeze` (default True), and `test_heavy_passes_do_not_block_the_loop` passes alone (1.2 s). The plan says so (§2.11, §10).
- *`SpeakersFound.svelte` doesn't exist* (engineering): checked (`App.svelte` handles only `speaker_scan`). Marked new (§5, §6).
- *The register retarget is only partial* (product, and engineering on the brief's `register`): checked: `BRIEF_SYSTEM`'s glossary and `register`, STYLE's "Sishta vyavaharika" and "Everyday verbs stay Telugu", ROLE's "Telugu YouTuber", and `_BASIC_VERBS` listing change, increase, decrease, learn, remember and forget; the lint is trace-only (`render.py` writes it into the `unit` event). Now the brief and the lint change too, `BRIEF_HASH` changes, and §12 no longer says the lint catches anything (§2.18).
- *A dead engine stays dead while the window is hidden* (product): the shell restarts it (§5).
- *Lid-close sleep* (product): the app says to keep the lid open and the Mac plugged in, and notes a sleep. Elapsed time and stage rates were already sleep-free: `time.monotonic()` is `mach_absolute_time` on this Mac (checked), which stops in sleep (§4, §5).
- *Promising that no English reaches the file* (product): reworded, a deeper duck where English plays with no Telugu over it, a measured threshold in `verify-mac.sh`, and the CLAUDE.md wording put to the maintainer (§1, §2.15, M9, M11).
- *The re-encode path is gold-plating; settings defaults and theme* (product): AV1 copied (it plays on M3 and later), VP9 copied with a warning, no re-encode; Settings keeps only the output folder (§2.2, §2.17, §4, M16).
- *`.part.mp4` left behind, a moved MP4, the output volume* (product): deleted on pause, failure and remove; "File moved or deleted"; the output volume checked (§2.17).
- *What leaves the Mac; thumbnails from i.ytimg.com* (product): stated in §1; the Library uses the thumbnail cached at fetch (§2.2, §5).
- *Figures* (product): checked his `lines.jsonl`: 1,592 rows, 776 distinct lines, all under prompt hash `d548212e604d` (M1 said 644); M15's 1.3× exceeded `Dubber`'s 1.25× clamp. Corrected (§2.8, M1, M15).
- *Steps too large; the bench written twice; register bundled with the repairs* (both): six steps; the register first; the bench once in step 3; Session's tests ported before Session goes; the export before the separator (§10).
- *Voices cloned from the mixture* (product): noted as the cheap follow-up this enables (§8).
- *Notifications may not show at all* (engineering): Settings explains the Script Editor permission, nothing depends on a notification, and step 5 checks it on this Mac (§4, §5).

**Adapted**
- *Separator choice ranked on a singing benchmark* (engineering): Kim stays the default as the best published proxy, but M10's bake-off is now on speech content and includes BandIt Plus (whose English dialogue training is a match, not a drawback, as the table had it) and Kim with test-time augmentation (§2.14, M10).
- *Learn the ETA priors from his finished jobs* (product): not adopted (Appendix A removed the EWMA file); the running stage's own rate takes over after 30 s and the estimate stays a range; the jobs ahead are now added (§4).
- *Lazy video fetch at export time* (engineering): the video gets its own stage beside the dub loop rather than at the export, so a YouTube refusal shows in the first hour, and it is skipped when the output is served (§2.2).
- *A deeper duck wherever English plays without Telugu* (product): only in English speech with no Telugu within 1 s, lasting 1.5 s or more, so short tails don't make the music pump (§2.15).
- *A QuickTime screenshot with each subtitle track in verify-mac.sh* (product): selecting a track can't be scripted reliably; the walkthrough opens the MP4 in QuickTime and the maintainer checks both tracks by eye (§10 step 6).
- *Compare a hidden window's throughput in verify-mac.sh* (product): the bench doesn't run the app, so the comparison is part of the M5 Pro walkthrough (§12).

**Rejected**
- None. Every finding held up when checked, in fact or in substance; where the fix differs from the one proposed, it is listed under Adapted.
