# Original Telugu audition, 2026-10-10

This isolated experiment compares three original Telugu phrases through upstream
OmniVoice and Maata's current Chatterbox Telugu backend. It changes no production
runtime, model lock, job, preset, or default. Listening is required before choosing
a voice. ASR agreement does not establish native pronunciation or naturalness.

OmniVoice uses `language="te"`, its auto voice, 32 steps, float16 MPS generation,
and the upstream CPU Higgs codec. There is no reference or voice-design instruction.
The available Chatterbox baseline uses builtin `conds.pt`; native Telugu reference
provenance is unknown. This evaluates those configurations, not every possible
voice or reference of either model. Text includes spoken numbers and Telugu-script
loanwords; neither model receives a Latin substitution in this first screen.
The completed `names_numbers` fixture contains U+200C in `వాట్సాప్‌లో` and is
passed unchanged to both models. Production normalizes/rejects joiners. Preserve
this trial as recorded; that sentence is not an exact production-text comparison.

Source commit: `k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2`.
The source's model loading, optional ASR and audio utilities were inspected before
import. Model files and codec are pinned in this directory's `models.lock.json`.
The upstream model card declares CC-BY-NC; upstream code is Apache-2.0.

From the repository root, install only the separate runtime at an explicit
isolated destination:

```sh
UV_PROJECT_ENVIRONMENT="$HOME/Library/Caches/Maata/model-evaluation-2026-10-10/omnivoice-venv" \
  uv sync --project scripts/experiments/omnivoice --frozen --no-dev
```

Fetch via Maata's existing commit/size/hash-checking downloader. This is the only
runner stage that needs the network:

```sh
engine/.venv/bin/python scripts/experiments/omnivoice_smoke.py fetch
```

Run each inference stage sequentially in a fresh process with network denied:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  "$HOME/Library/Caches/Maata/model-evaluation-2026-10-10/omnivoice-venv/bin/python" \
  scripts/experiments/omnivoice_smoke.py omnivoice
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_smoke.py chatterbox
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_smoke.py watermark
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_smoke.py roundtrip
```

The runner additionally forces HF/Transformers offline, uses only verified local
model directories, rejects automatic OmniVoice model resolution, and disables its
ASR loader. The independent roundtrip uses the installed pinned MLX Whisper turbo
with forced Telugu, after TTS processes exit. No audio leaves the machine.

Artifacts are under
`~/Library/Caches/Maata/model-evaluation-2026-10-10/audio/`. `results.json` is updated
atomically, with model identity, first/warm generation and load times, true sample
rate, peak process RSS/MPS allocations, waveform checks, raw/marked WAV paths,
PerTh detection, and raw local ASR result paths. Chatterbox's existing watermark is
preserved; OmniVoice receives the same installed PerTh implementation. Raw files
are for local diagnostic comparison; audition uses the marked files. The runner
refuses accidental repeated synthesis into the same results directory. To repeat
an explicitly authorized trial, pass `--out /absolute/new/directory` to every stage.

These are three different sentences, one seed each. Times are component timings,
not full-video throughput. Peak RSS is per process on macOS and does not include
all system/GPU allocations. Waveform tail energy is not a word-ending verdict.

## Production adapter validation after Voice B selection

`omnivoice_adapter_smoke.py` uses the provisioned production runtime and verified
model assets via the same automatic-mode constructor configuration as the Apple
backend. The three original sentences are normalized by the engine. The seed is
predeclared as 20261010 for every call, with no retries or reference conditioning;
this does not guarantee consistent speaker identity. The earlier conditioned trial
remains preserved in `audio/adapter-timed/` and its committed evidence.
It separates preparation from generation timing, reuses one worker, checks complete
natural takes against an advisory cap, and replays saved takes without synthesis.

After obtaining the GPU slot, run exactly one synthesis stage, followed by the
CPU export check and local ASR in separate processes:

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_adapter_smoke.py synthesize
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_adapter_smoke.py export
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  engine/.venv/bin/python scripts/experiments/omnivoice_adapter_smoke.py roundtrip
```

Results are in `audio/adapter-auto/` beneath the evaluation cache. The export uses
generated speech over a synthetic test-card video through production mastering,
AAC encoding and final loudness measurement; it checks decoded PerTh per line.
Planner and energy-based coverage against authored windows are diagnostic: no
recorded source speech exists in this fixture, and native listening is still needed.
