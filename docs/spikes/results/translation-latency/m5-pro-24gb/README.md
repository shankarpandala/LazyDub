# Translation batching check — 2026-10-05

`2026-10-05-pipeline.json` retains five fresh whole-video runs per policy: A is the old 150-second / 30-line cap,
B is the new 30-second / 6-line cap. The complete brief, calibration, models, prompts, tiers and reviews remain.
The fixture is the same original 60-second speech-plus-music video described in
`../../pipeline/m5-pro-24gb/2026-10-05-quality-smoke.json`. This is a short synthetic workload, not a long-video
throughput or native-listening evaluation. Read the report's method and limitations alongside its medians.

| Median across five runs | Previous cap | New cap | Change |
|---|---:|---:|---:|
| Completed export | 104.85 s | 94.82 s | 9.6% less time |
| First retained speech take | 80.54 s | 61.59 s | 23.5% less time |
| Claude calls | 4 | 6 | 50% more |
| Claude output tokens | 7,999 | 10,499 | 31.3% more |

All ten runs voiced 11 units without skips, used the requested models, and ran on AC power with nominal start/end
thermal states. Local calibration was identical. Some runs needed extra correction or fitting calls; none were
discarded for being slower or imperfect. A warning-label repair landed during the series; it changes reporting from
the final plan, without changing synthesis, scheduling or PCM keys. Per-run revision/diff fingerprints are retained.

The fixture repeats two authored lines. The old single scene returned identical wordings for them, allowing cached
speech reuse. Separate concurrent batches returned different valid wordings, requiring new speech takes; median
voicing GPU time increased from 20.94 to 27.07 seconds. Coalescing repeated text safely and measuring representative
long videos are follow-up work. The observed export improvement comes from concurrency and earlier readiness despite
that extra work; it does not establish a general throughput gain.

`2026-10-05-boundary-quality.json` contains a separate text-only comparison on 16 original linked sentences.
Both selected outputs passed the production meaning review and an independent model inspection of nine semantic
expectations. The expectations were kept out of the translator/reviewer prompts. This does not assess speech,
pronunciation or native-speaker naturalness. The source checklist is
`scripts/fixtures/translation-boundaries-original.json` at the repository root.

`2026-10-05-independent-text-inspection.json` inspects the authored content of all ten video runs directly. It records
minor phrasing differences, not just automated grades, and separates two non-authored ASR tail units. One candidate
turns the isolated tail word “you” into “thanks” despite an automated C verdict. That reviewer miss and the ASR artifact
remain limitations; this speed change does not fix recognition quality.

Final validation passed all 1,079 engine tests (180 render/translation/trace, 39 WebSocket, 860 remaining tests).
The final candidate export also passed copied-video, audio/subtitle layout, loudness, A/V synchronization and
watermark checks; the verifier's full output is retained in `2026-10-05-pipeline.json` under `candidate_export_check`.

To repeat the text quality check from the repository root:

```sh
engine/.venv/bin/python scripts/bench_translation_scenes.py \
  --fixture scripts/fixtures/translation-boundaries-original.json \
  --run --order AB --out /tmp/maata-boundary-quality.json
```

For whole-video measurements, start `scripts/verify_mac.py proxy WORK` as in `scripts/verify-mac.sh`, and use that
script's outbound sandbox and localhost proxy environment. For each arm, run the following in a fresh process with
a dedicated benchmark cache and output directory; alternate A/B order and collect at least five samples per arm.
The old cap remains an explicit benchmark override even after the production default changes.

```sh
engine/.venv/bin/python scripts/bench_scene_pipeline.py VIDEO.mp4 \
  --scene-seconds 150 --scene-lines 30 --seed 42 \
  --cache /tmp/maata-cap-A1 --mp4-dir /tmp/maata-cap-A1-out --out /tmp/maata-cap-A1.json
engine/.venv/bin/python scripts/bench_scene_pipeline.py VIDEO.mp4 \
  --scene-seconds 30 --scene-lines 6 --seed 42 \
  --cache /tmp/maata-cap-B1 --mp4-dir /tmp/maata-cap-B1-out --out /tmp/maata-cap-B1.json
```

These commands must run under the sandbox/proxy above for audio measurements. They use existing pinned local weights
and the sealed Claude CLI; do not allow libraries to fetch models. Each report records model-loading time separately,
the first retained speech take, completed-export latency, stage costs, all final line traces, usage, machine, lockfile
hashes, software versions, power and thermals. The first retained take is not yet final playable PCM. Provider output
and prompt caching vary even with the local calibration seed fixed; retain the slower and imperfect runs too.
