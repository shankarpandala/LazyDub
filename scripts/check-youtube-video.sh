#!/usr/bin/env bash
# Check, once on this Mac before the first real dub, that YouTube serves the video Maata's video stage asks for
# (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §2.2, M18):
#
#   ./scripts/check-youtube-video.sh YOUTUBE_URL
#
# It runs the pinned yt-dlp through `uv run` with exactly the engine's options (imported from maata_engine.resolve,
# never copied here) and the PATH a Finder-launched Maata has (no Homebrew: no ffmpeg, no deno), so it sees what the
# app will see. It lists what each alternative of VIDEO_FORMAT would pick and which one the engine takes, then downloads
# the first 200 MB of that format, times it (the progress hook stops it there) and deletes it. Nothing else is
# downloaded, and nothing is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -eq 1 ]] || { echo "usage: $0 YOUTUBE_URL" >&2; exit 2; }
uv=$(command -v uv) || { echo "uv isn't on the PATH: run ./scripts/setup-mac.sh first" >&2; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd engine
env -i HOME="$HOME" TMPDIR="$work" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  "$uv" run --no-sync python - "$1" "$work" <<'PY'
import copy
import shutil
import sys
import time
from pathlib import Path

from yt_dlp import YoutubeDL

from maata_engine.resolve import VIDEO_FORMAT, audio_options, video_options, watch_url
from maata_engine.urls import parse_youtube_url

LIMIT = 200e6  # bytes timed, then stopped
url, work = sys.argv[1], Path(sys.argv[2])
ref = parse_youtube_url(url)
print(f"PATH as a Finder launch: {shutil.which('ffmpeg') or 'no ffmpeg'}, {shutil.which('deno') or 'no deno'}")
try:  # as the engine's fetch does: extracted with the audio's options, then the video chosen from that info
    with YoutubeDL(audio_options(work, work)) as ydl:
        info = ydl.extract_info(watch_url(ref.video_id), download=False)
except Exception as e:
    sys.exit(f"YouTube didn't give this video's formats: {str(e).splitlines()[0]}")
print(f"{info.get('title')!r}: {float(info.get('duration') or 0) / 60:.1f} min, {len(info.get('formats') or [])} formats")


def describe(f: dict) -> str:
    size = f.get("filesize") or f.get("filesize_approx")
    return (f"{f.get('format_id')} {f.get('vcodec')} {f.get('width')}x{f.get('height')} {f.get('fps')} fps "
            f".{f.get('ext')} {'%.0f MB' % (size / 1e6) if size else 'size unknown'}")


for alt in VIDEO_FORMAT.split("/"):
    with YoutubeDL({**video_options(work, work), "format": alt}) as y:
        try:
            print(f"  {alt}: {describe(y.process_ie_result(copy.deepcopy(info), download=False))}")
        except Exception as e:  # nothing matches this alternative
            print(f"  {alt}: none ({str(e).splitlines()[0][:120]})")
with YoutubeDL(video_options(work, work)) as y:
    try:
        chosen = y.process_ie_result(copy.deepcopy(info), download=False)
    except Exception as e:
        sys.exit(f"The video stage would fail: {e}")
print(f"the engine takes: {describe(chosen)}")


class Enough(Exception):
    pass


got = [0]


def hook(d: dict) -> None:
    got[0] = int(d.get("downloaded_bytes") or 0)
    if d.get("status") == "downloading" and got[0] >= LIMIT:
        raise Enough


t0 = time.monotonic()
with YoutubeDL(video_options(work, work, hook)) as y:
    try:
        y.process_ie_result(copy.deepcopy(info), download=True)
    except Enough:
        pass
    except Exception as e:
        if not isinstance(e.__cause__, Enough) and "Enough" not in repr(e):
            sys.exit(f"The download failed after {got[0] / 1e6:.0f} MB: {e}")
seconds = time.monotonic() - t0
print(f"downloaded {got[0] / 1e6:.0f} MB in {seconds:.1f} s: {got[0] / 1e6 / seconds:.1f} MB/s"
      if seconds > 0 else "nothing downloaded")
for f in work.glob("video.*"):
    f.unlink()
PY
