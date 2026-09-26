#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (reference: MacBook Pro M5 Pro, 24 GB).
#   ./scripts/setup-mac.sh            # engine deps, models (~4.9 GB: ASR, diarization, TTS), UI build
# Env: HF_TOKEN   – needed once for pyannote/speaker-diarization-community-1 (accept its terms on HF first);
#                   without it, Maata dubs every speaker with one voice until the model is added.
# Translation runs through your own Claude Code (ADR-019): no translation model is downloaded. Install Claude Code and
# sign in with `claude auth login`; only the English transcript, as text, goes to Anthropic.
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
  command -v "$tool" >/dev/null || die "Missing '$tool'. Install: uv (astral.sh/uv), node 22.12+, rustup, deno (yt-dlp's JS runtime: brew install deno)."
done
# Vite 8's native bundler needs Node ^20.19 or >=22.12; older versions make npm skip it silently.
node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)' \
  || die "Node $(node --version) at $(command -v node) is too old: Maata needs 20.19+ or 22.12+ (e.g. put /opt/homebrew/bin first on PATH)."

bold "→ Engine (Python, MLX + MPS; chatterbox-telugu's patched Chatterbox is pinned in engine/uv.lock)"
( cd engine && uv sync --extra apple --extra resolve )

bold "→ Models (pinned by commit + sha256; resumable)"
( cd engine && uv run maata-bench fetch --backend apple ) || die "Model download failed (see above). Re-run to resume."

bold "→ Claude Code (translation)"
if command -v claude >/dev/null || [[ -x "${MAATA_CLAUDE_BIN:-}" ]]; then
  echo "  $("${MAATA_CLAUDE_BIN:-claude}" --version 2>/dev/null || echo "claude found")"
  echo "  Sign in once, if you haven't: claude auth login"
else
  echo "  ! Claude Code isn't installed: Maata translates through it. Install it, then run: claude auth login"
fi

bold "→ UI"
( cd app && npm ci && npm run build )

bold "Done. Launch with:  (cd app && npm run tauri dev)"
