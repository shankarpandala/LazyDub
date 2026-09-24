#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (reference: MacBook Pro M5 Pro, 24 GB).
#   ./scripts/setup-mac.sh            # engine deps, models (~9.4 GB), UI build
# Env: HF_TOKEN   – needed once for pyannote/speaker-diarization-community-1 (accept its terms on HF first)
#      CHATTERBOX_SRC – path or git URL of the patched Chatterbox that supports `te` (chatterbox-telugu)
set -euo pipefail
cd "$(dirname "$0")/.."
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
die() { printf "\033[31m✗ %s\033[0m\n" "$*"; exit 1; }

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "Maata's primary backend needs an Apple Silicon Mac."
ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
(( ram_gb >= 16 )) || die "At least 16 GB of memory is required (found ${ram_gb} GB)."
free_gb=$(df -g . | awk 'NR==2{print $4}')
(( free_gb >= 20 )) || die "About 20 GB of free disk is needed for models and the engine (found ${free_gb} GB)."
bold "✓ $(sysctl -n machdep.cpu.brand_string), ${ram_gb} GB"

for tool in uv node npm cargo deno; do
  command -v "$tool" >/dev/null || die "Missing '$tool'. Install: uv (astral.sh/uv), node 22, rustup, deno (yt-dlp's JS runtime: brew install deno)."
done

bold "→ Engine (Python, MLX + MPS)"
( cd engine && uv sync --extra apple --extra resolve )
if [[ -n "${CHATTERBOX_SRC:-}" ]]; then
  ( cd engine && uv pip install "${CHATTERBOX_SRC}" )
else
  echo "  ! CHATTERBOX_SRC not set: install your patched Chatterbox (with 'te') before dubbing:"
  echo "    CHATTERBOX_SRC=/path/to/chatterbox ./scripts/setup-mac.sh"
fi

bold "→ Models (pinned by commit + sha256; resumable)"
( cd engine && uv run maata-bench fetch --backend apple ) || die "Model download failed (see above). Re-run to resume."

bold "→ UI"
( cd app && npm ci && npm run build )

bold "Done. Launch with:  (cd app && npm run tauri dev)"
