#!/usr/bin/env bash
# Separate OmniVoice runtime and pinned local assets; never alters engine/.venv.
# MAATA_SETUP_OFFLINE=1 uses cached Python dependencies. Model fetching is always
# the existing Maata verified downloader; already verified local files are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
maata_repo="$(pwd)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo "OmniVoice setup requires Apple Silicon." >&2; exit 1; }
command -v uv >/dev/null || { echo "Install uv before running setup." >&2; exit 1; }
maata_engine_python="${MAATA_ENGINE_PYTHON:-$maata_repo/engine/.venv/bin/python}"
[[ -x "$maata_engine_python" ]] || { echo "Set up the engine Python environment first." >&2; exit 1; }
maata_support="${MAATA_DATA_DIR:-$HOME/Library/Application Support/Maata}"
maata_runtime="$maata_support/runtimes/omnivoice"
maata_eval="${MAATA_EVALUATION_DIR:-$HOME/Library/Caches/Maata/model-evaluation-2026-10-10}"
maata_uv_args=(--project "$maata_repo/engine/runtimes/omnivoice" --frozen --no-dev)
[[ "${MAATA_SETUP_OFFLINE:-0}" == 1 ]] && maata_uv_args+=(--offline)
UV_PROJECT_ENVIRONMENT="$maata_runtime" uv sync "${maata_uv_args[@]}"

MAATA_PROVISION_ROOT="$maata_support" MAATA_PROVISION_CACHE="$maata_eval" \
  MAATA_PROVISION_REPO="$maata_repo" PYTHONPATH="$maata_repo/engine/src" \
  "$maata_engine_python" - <<'PY'
import json, os, shutil
from pathlib import Path
from maata_engine.models import _spec, fetch_model, verify

repo = Path(os.environ['MAATA_PROVISION_REPO'])
support = Path(os.environ['MAATA_PROVISION_ROOT'])
cache = Path(os.environ['MAATA_PROVISION_CACHE'])
lock = json.loads((repo / 'engine/runtimes/omnivoice/models.lock.json').read_text())
for model in lock['models']:
    target = support / 'Models' / model['id']
    source = cache / 'models' / model['id']
    for entry in model['files']:
        spec = _spec(entry)
        dst, src = target / spec.path, source / spec.path
        if verify(dst, spec):
            continue
        if verify(src, spec):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + '.provision-part')
            shutil.copy2(src, tmp)
            if not verify(tmp, spec):
                raise RuntimeError(f'Copied model file failed verification: {spec.path}')
            tmp.replace(dst)
    fetch_model(model, support / 'Models')
    assert all(verify(target / f['path'], _spec(f)) for f in model['files'])
    print('Verified model:', target)
PY

# Automatic Telugu mode needs no reference audio. Existing audition assets are
# preserved, but neither discovered nor promoted into production conditioning.
echo "OmniVoice runtime ready: $maata_runtime/bin/python"
