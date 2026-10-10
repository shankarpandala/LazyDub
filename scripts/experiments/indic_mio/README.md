# Indic-Mio local screen: blocked before synthesis

The separate frozen environment and verified model files were prepared on
2026-10-10. No Indic-Mio audio was generated and no production default was changed.

The pinned MioCodec constructor automatically calls TorchAudio's WavLM download
before loading its codec checkpoint with `strict=False`. An offline audit built
the identical `wavlm_model(**WAVLM_BASE_PLUS._params)` architecture and required
strict checkpoint loading. It failed: 199 WavLM model keys and four other state
entries are absent. The runner deliberately stops before using any uninitialized
weights. It must not be changed to accept missing weights without resolving and
pinning the actual upstream dependency and understanding the other missing keys.

Evidence: `docs/spikes/results/model-evaluation-2026-10-10/audio/indic-mio-blocker.json`.
Logs, installed environment and downloaded models remain under
`~/Library/Caches/Maata/model-evaluation-2026-10-10/` for a later bounded resolution.
`models.lock.json` pins the text model and 24 kHz codec. The model card's example
that writes these codec samples at 44.1 kHz is not followed.

```sh
UV_PROJECT_ENVIRONMENT="$HOME/Library/Caches/Maata/model-evaluation-2026-10-10/indic-mio-venv" \
  uv sync --project scripts/experiments/indic_mio --frozen --no-dev
engine/.venv/bin/python scripts/experiments/indic_mio_smoke.py fetch
# Currently fails safely on incomplete checkpoint state, before synthesis:
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  "$HOME/Library/Caches/Maata/model-evaluation-2026-10-10/indic-mio-venv/bin/python" \
  scripts/experiments/indic_mio_smoke.py synthesize
```

The intended C configuration uses MPS bfloat16 Qwen3 and the upstream CPU float32
codec/SDPA path, with the actual global embedding of the local FLEURS Telugu
reference. No fabricated or randomly initialized reference is permitted. The
original three phrases are normalized through production `qa.validators.wording`.
The same reference is separately evaluated through working OmniVoice as D.
