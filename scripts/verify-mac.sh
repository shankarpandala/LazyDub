#!/usr/bin/env bash
# Produce the evidence that Maata runs well on this Mac (reference: MacBook Pro M5 Pro, 24 GB).
#
#   ./scripts/verify-mac.sh [YOUTUBE_URL]
#
# 1. Generates a self-recorded 2-voice English test clip with macOS `say` (no third-party media).
# 2. Runs the real dubbing pipeline on it with the Apple backend and saves the JSON metrics to
#    docs/spikes/results/<machine>/pipeline-<timestamp>.json. Text is translated through your signed-in Claude Code
#    (ADR-019); MAATA_BENCH_TRANSLATOR=mock runs the pipeline offline instead (translation not measured).
#    MAATA_BENCH_TIMING=v1 times the lines as before timing v2: commit that run as the baseline, then
#    MAATA_BENCH_BASELINE=<that JSON, from the repo root> measures the timing targets (ADR-017, ARCHITECTURE §3.10)
#    against it.
# 3. Launches Maata on YOUTUBE_URL (optional), waits, and saves a screenshot next to the JSON.
# Review both files, then commit them: they are the measured numbers (spec: no number without JSON).
set -euo pipefail
cd "$(dirname "$0")/.."
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
die() { printf "\033[31m✗ %s\033[0m\n" "$*"; exit 1; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "Run this on the Apple Silicon Mac."

machine=$(sysctl -n hw.model | tr -c 'A-Za-z0-9,.\n-' '_')
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="docs/spikes/results/${machine}"
mkdir -p "$out"
work=$(mktemp -d)

bold "→ Test clip (macOS say, two voices, ~70 s)"
voices=($(say -v '?' | awk '/en_(US|GB)/{print $1}' | head -2))
(( ${#voices[@]} == 2 )) || voices=(Samantha Daniel)
lines=(
  "Welcome back to the channel. Today we are going to build a small machine learning project from scratch."
  "That sounds great. What do we need before we start writing any code?"
  "Just a laptop and about twenty minutes. First, we load the data and look at a few examples."
  "And how do we know the model is actually learning something useful?"
  "We keep a separate test set, and we measure accuracy on data the model has never seen before."
  "Perfect. Let's get started, and tell us in the comments what you want to see next."
)
i=0
for l in "${lines[@]}"; do
  v=${voices[$((i % 2))]}
  say -v "$v" -o "$work/l$i.aiff" "$l"
  i=$((i + 1))
done
( cd engine && uv run --no-sync python - "$work" <<'PY'
import sys, pathlib, numpy as np
from maata_engine.resolve import decode_audio
import wave
w = pathlib.Path(sys.argv[1])
parts = []
for f in sorted(w.glob("l*.aiff"), key=lambda p: int(p.stem[1:])):
    parts += [decode_audio(f), np.zeros(int(0.7 * 16000), np.float32)]
y = np.concatenate(parts)
y = np.tile(y, max(1, int(np.ceil(70 * 16000 / len(y)))))[: 70 * 16000]
with wave.open(str(w / "clip.wav"), "wb") as o:
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
    o.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
PY
)

bold "→ Pipeline on the Apple backend (real models)"
baseline=""
if [[ -n "${MAATA_BENCH_BASELINE:-}" ]]; then
  [[ -f "$MAATA_BENCH_BASELINE" ]] || die "No baseline JSON at $MAATA_BENCH_BASELINE"
  baseline="$(cd "$(dirname "$MAATA_BENCH_BASELINE")" && pwd)/$(basename "$MAATA_BENCH_BASELINE")"
fi
( cd engine && uv run --no-sync maata-bench pipeline "$work/clip.wav" --backend apple \
    ${MAATA_BENCH_TRANSLATOR:+--translator "$MAATA_BENCH_TRANSLATOR"} \
    ${MAATA_BENCH_TIMING:+--timing "$MAATA_BENCH_TIMING"} \
    ${baseline:+--baseline "$baseline"} ) | tee "$out/pipeline-${stamp}.json"

if [[ -n "${1:-}" ]]; then
  bold "→ Launching Maata on $1"
  ( cd app && MAATA_OPEN="$1" npm run tauri dev >"$work/app.log" 2>&1 & echo $! >"$work/app.pid" )
  sleep 45
  osascript -e 'tell application "System Events" to set frontmost of (first process whose name contains "maata" or name contains "Maata") to true' 2>/dev/null || true
  sleep 2
  screencapture -x "$out/ui-${stamp}.png" && bold "  screenshot: $out/ui-${stamp}.png"
fi

bold "Done. Review, then commit:"
echo "  git add $out && git commit -m 'M5 Pro verification results' && git push"
