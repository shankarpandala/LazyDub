"""YouTube URL parsing (spec §6.1): watch?v=, youtu.be/, /shorts/, /embed/, with t= offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_T = re.compile(r"^(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s?)?$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}


class URLError(ValueError):
    """Raised with a user-facing message when a URL isn't a supported YouTube video link."""


@dataclass(frozen=True, slots=True)
class VideoRef:
    video_id: str
    start: float = 0.0
    is_short: bool = False


def _parse_t(value: str | None) -> float:
    if not value:
        return 0.0
    m = _T.match(value.strip())
    if not m or not any(m.groupdict().values()):
        return 0.0
    h, mi, s = (int(m.group(k) or 0) for k in ("h", "m", "s"))
    return float(h * 3600 + mi * 60 + s)


def parse_youtube_url(raw: str) -> VideoRef:
    text = raw.strip()
    if not text:
        raise URLError("Paste a YouTube link to start.")
    if "://" not in text:
        text = "https://" + text
    u = urlparse(text)
    host = (u.hostname or "").lower()
    qs = parse_qs(u.query)
    t = _parse_t((qs.get("t") or qs.get("start") or [None])[0])
    parts = [p for p in u.path.split("/") if p]

    vid: str | None = None
    short = False
    if host in ("youtu.be", "www.youtu.be"):
        vid = parts[0] if parts else None
    elif host in _HOSTS:
        if parts[:1] == ["watch"]:
            vid = (qs.get("v") or [None])[0]
        elif parts[:1] in (["shorts"], ["embed"], ["v"]) and len(parts) >= 2:
            vid, short = parts[1], parts[0] == "shorts"
        elif parts[:1] == ["live"]:
            raise URLError("Live streams aren't supported yet.")
        elif parts[:1] == ["playlist"]:
            raise URLError("Playlists aren't supported. Paste a link to a single video.")
    else:
        raise URLError("That doesn't look like a YouTube link.")

    if not vid or not _ID.match(vid):
        raise URLError("Couldn't find a video in that link.")
    return VideoRef(video_id=vid, start=t, is_short=short)
