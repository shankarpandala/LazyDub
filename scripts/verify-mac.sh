#!/usr/bin/env bash
# Produce the evidence that Maata runs well on this Mac (reference: MacBook Pro M5 Pro, 24 GB), on a video made here
# (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §10 step 6, §12). Its measurements are scripts/verify_mac.py's.
#
#   ./scripts/verify-mac.sh
#
# 1. Fetches the pinned models (`maata-bench fetch --backend apple`; pyannote needs HF_TOKEN and its accepted terms).
# 2. Synthesises a test video with PyAV (MAATA_VERIFY_SECONDS, 60-180, default 120): colour bars with one white frame
#    at 10 s; its audio is the macOS `say` voices Samantha and Daniel (the speech stem) over a music stem (synthetic
#    chords and percussion, or MAATA_VERIFY_MUSIC, a local CC0/CC-BY file checked against MAATA_VERIFY_MUSIC_SHA256)
#    with a 1 kHz beep at the white frame. Both stems are kept.
# 3. Runs the separator on its mix: seconds per block, MLX peak memory, the vocals' SI-SDR against the speech stem
#    (below 8 dB the run fails here, before anything depends on the separator), bf16 against float32 on one block (the
#    SEP_DTYPE it says to use), and the English left in the bed over the speech turns (MAATA_VERIFY_ENGLISH_MAX_DB,
#    default -20 dB).
# 4. Dubs it to an MP4 with `maata-bench pipeline --backend apple` (per-stage seconds, GPU seconds, peak RSS and MLX
#    memory, Claude calls), outbound network blocked by sandbox-exec except the Claude CLI's HTTPS to Anthropic, which
#    goes through a local proxy that tunnels to nothing else. Text is translated through your signed-in Claude Code
#    (ADR-019); MAATA_BENCH_TRANSLATOR=mock runs it offline (translation not measured). MAATA_BENCH_BASELINE=<an earlier
#    committed pipeline JSON, from the repo root> measures the timing targets against that run's.
# 5. Checks the MP4: its streams (one copied H.264 video, one Telugu AAC, two mov_text tracks), faststart, loudness,
#    the A/V offset (the white frame against the beep, which the bed carries) and the PerTh watermark under the Telugu.
# 6. Writes docs/spikes/results/<machine>/verify-<timestamp>.json, opens the MP4 in QuickTime with a checklist for
#    your eyes and ears, launches Maata and saves a screenshot of its Library next to the JSON.
# Review both files, then commit them: they are the measured numbers (spec: no number without JSON). The MP4 and the
# stems stay in the work folder it prints (not committed).
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
die() { printf "\033[31m✗ %s\033[0m\n" "$*"; exit 1; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "Run this on the Apple Silicon Mac."
command -v sandbox-exec >/dev/null || die "sandbox-exec is missing: the bench must run with outbound network blocked."

machine=$(sysctl -n hw.model | tr -c 'A-Za-z0-9,.\n-' '_')
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="docs/spikes/results/${machine}"
mkdir -p "$out"
work=$(mktemp -d -t maata-verify)
helper=(uv run --no-sync python "$root/scripts/verify_mac.py")
proxy_pid=""
trap '[[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null || true' EXIT
bold "  work folder: $work"

bold "→ Models (pinned, checksummed)"
( cd engine && uv run --no-sync maata-bench fetch --backend apple )

bold "→ Test video (macOS say over music, a white frame and a beep at 10 s)"
# Two real speaking voices, by name: the listing's first English names are novelty voices (Albert, Bad News sings).
voices=(Samantha Daniel)
listing=$(say -v '?')
for v in "${voices[@]}"; do
  grep -q "^$v " <<<"$listing" ||
    die "The say voice $v is missing: add it in System Settings > Accessibility > Spoken Content."
done
lines=(
  "Welcome back to the channel. Today we are going to build a small machine learning project from scratch."
  "That sounds great. What do we need before we start writing any code?"
  "Just a laptop and about twenty minutes. First, we load the data and look at a few examples."
  "And how do we know the model is actually learning something useful?"
  "We keep a separate test set, and we measure accuracy on data the model has never seen before."
  "Perfect. Let's get started, and tell us in the comments what you want to see next."
)
aiffs=()
for i in "${!lines[@]}"; do
  say -v "${voices[$((i % 2))]}" -o "$work/l$i.aiff" "${lines[$i]}"
  aiffs+=("$work/l$i.aiff")
done
( cd engine && MAATA_VERIFY_VOICES="${voices[*]}" "${helper[@]}" video "$work" "${aiffs[@]}" )

# Outbound network: only 127.0.0.1 (the proxy below, and the engine's own loopback); everything else is refused.
cat >"$work/offline.sb" <<'SB'
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
SB
offline=(sandbox-exec -f "$work/offline.sb")

bold "→ Separator on the test mix (MLX, bf16 and float32)"
( cd engine && "${offline[@]}" "${helper[@]}" separation "$work" "$HOME/Library/Application Support/Maata/Models" ) ||
  die "The separation check didn't run (see above)."
floor=$(cd engine && uv run --no-sync python -c 'import json, sys; print(json.load(open(sys.argv[1]))["si_sdr_ok"])' \
        "$work/separation.json")
if [[ "$floor" != True ]]; then
  ( cd engine && "${helper[@]}" combine "$work" "$root/$out/verify-${stamp}.json" ) || true
  die "The separator's vocals are under the 8 dB SI-SDR floor: see $out/verify-${stamp}.json. Don't start a long job."
fi

bold "→ Dubbing it to an MP4 (Apple backend, real models; network: the Claude CLI to Anthropic only)"
( cd engine && exec .venv/bin/python "$root/scripts/verify_mac.py" proxy "$work" ) &  # (outside the sandbox)
proxy_pid=$!
for _ in $(seq 50); do [[ -s "$work/proxy.port" ]] && break; sleep 0.1; done
[[ -s "$work/proxy.port" ]] || die "The proxy didn't start."
port=$(cat "$work/proxy.port")
baseline=""
if [[ -n "${MAATA_BENCH_BASELINE:-}" ]]; then
  [[ -f "$MAATA_BENCH_BASELINE" ]] || die "No baseline JSON at $MAATA_BENCH_BASELINE"
  baseline="$(cd "$(dirname "$MAATA_BENCH_BASELINE")" && pwd)/$(basename "$MAATA_BENCH_BASELINE")"
fi
dub_status=0
( cd engine && HTTPS_PROXY="http://127.0.0.1:$port" HTTP_PROXY="http://127.0.0.1:$port" NO_PROXY="" \
    "${offline[@]}" uv run --no-sync maata-bench --cache "$work/bench" pipeline "$work/test.mp4" --backend apple \
    --out "$work/out" ${MAATA_BENCH_TRANSLATOR:+--translator "$MAATA_BENCH_TRANSLATOR"} \
    ${baseline:+--baseline "$baseline"} ) | tee "$work/pipeline.json" || dub_status=$?
kill "$proxy_pid" 2>/dev/null || true
proxy_pid=""
if (( dub_status != 0 )); then
  rm -f "$work/pipeline.json"  # (the bench prints its JSON only when the job is done)
  ( cd engine && "${helper[@]}" combine "$work" "$root/$out/verify-${stamp}.json" ) || true
  die "The dub didn't finish (see above). $out/verify-${stamp}.json keeps the separator's numbers."
fi
mp4=$(cd engine && uv run --no-sync python -c 'import json, sys; print(json.load(open(sys.argv[1]))["export"]["path"])' \
      "$work/pipeline.json")

bold "→ The MP4: streams, loudness, A/V offset, watermark"
( cd engine && "${offline[@]}" "${helper[@]}" export "$work" "$mp4" "$work/bench/localbench0/render/manifest.json" )
status=0
( cd engine && "${helper[@]}" combine "$work" "$root/$out/verify-${stamp}.json" ) || status=$?

bold "→ QuickTime: check these by eye and ear"
open -a "QuickTime Player" "$mp4"
cat <<EOF
  [ ] View > Subtitles lists Telugu and English; each shows its lines (Telugu at the dub's times, English at the
      speakers').
  [ ] The Telugu subtitles render properly: joined letters (conjuncts), no boxes or question marks.
  [ ] The beep sounds with the white frame at 0:10 (A/V in sync by eye).
  [ ] The Telugu voices sound like the two speakers; the music plays on under them; no English is audible.
EOF

bold "→ Maata's Library"
( cd app && npm run tauri dev >"$work/app.log" 2>&1 & echo $! >"$work/app.pid" )
for _ in $(seq 300); do
  osascript -e 'tell application "System Events" to (name of processes) contains "maata"' 2>/dev/null | grep -q true && break
  sleep 1
done
sleep 8
osascript -e 'tell application "System Events" to set frontmost of (first process whose name contains "maata" or name contains "Maata") to true' 2>/dev/null || true
sleep 2
screencapture -x "$out/library-${stamp}.png" && bold "  screenshot: $out/library-${stamp}.png"

bold "Done. Results: $out/verify-${stamp}.json (the MP4 and stems: $work). Review, then commit:"
echo "  git add $out && git commit -m 'M5 Pro verification results' && git push"
(( status == 0 )) || die "A check failed: see \"checks\" in $out/verify-${stamp}.json."
