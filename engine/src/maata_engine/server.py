"""Engine entry point: serves the UI over loopback HTTP (ADR-006) and a token-guarded WebSocket (ADR-007).

The engine owns the render jobs (docs/research/dubbing-2026-09/OFFLINE-RENDER.md §4), not a connection: one runs at a
time and the others wait in a FIFO queue, so a job survives a closed window or a UI reload, and its progress and the
speakers it found go to every window. Nothing of a render is played in the app: its dub is a file, saved in the output
folder (the engine's one setting, kept in the config folder), which the engine opens or shows in Finder when asked, and
notifications report what a job did. Quitting Maata (the shell closes the engine's stdin) stops the running job
gracefully: it is written interrupted and continues, first in line, at the next launch, as does a job a crash left
running, unless a stage of it keeps crashing. Retention expires the voice data of renders not updated for a while, and
of done whole-video jobs while the cache is over its cap. The only binary frame is a speaker's Hear voice sample
(ADR-007, ADR-021)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import Awaitable, Callable, Collection, Coroutine
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from . import notify
from . import settings as config
from .backends import load_backend
from .codex_cli import DEFAULT_MODEL, CodexCLI
from .gpu import GpuScheduler
from .playback import voice_sample
from .render import (STAGES, RenderJob, RenderSettings, _read_json, _write_json, estimate, found_on_disk, left,
                     output_view, shown_state)
from .resolve import DemoResolver, ResolveError, YtDlpResolver
from .urls import URLError, parse_youtube_url

log = logging.getLogger("maata.server")

HEALTH_TIMEOUT = 10.0  # s: `hello` waits no longer than this for the Claude CLI's version and sign-in state
CRASH_STARTS = 3       # a stage started this many times without finishing fails its job at startup (§4)
RETAIN_DAYS = 14.0     # a render not updated for this long loses its voice data (SPEC §8, M6)
CACHE_CAP = 5e9        # bytes: above this, the renders updated least recently lose theirs too
QUIT_WAIT = 4.0        # s: a quit that hasn't stopped the engine by then (a worker thread finishing) exits it at once
PREVIEW_AT = 900.0     # s: New dub's "First 15 minutes" (§3, §5)
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")  # a YouTube video id: the name of a job's folder in the cache
REMOVE_KEEP = ("lines.jsonl", "briefs.jsonl", "units.jsonl")  # what `remove` leaves of a job's folder (§4)
CAFFEINATE = "/usr/bin/caffeinate"
# What a render keeps of its own once expired: its translations, transcript, speakers and trace (§4). The voices, takes,
# line PCM, the manifest, the audio and the background sound go.
EXPIRE_RENDER = ("voices", "voices.json", "takes", "takes.jsonl", "pcm", "manifest.json", "audio16k.f32", "bed",
                 "bed.json", "bed_vocals.npy")
EXPIRE_MEDIA = ("audio.*", "video.*")  # the source media the job downloaded (never the output folder's MP4)
EXPIRE_STAGES = ("fetch", "voices", "separate", "voice_lines", "finish", "video", "export")

SendJSON = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

# Nothing plays in the app (§6): no YouTube player, so no YouTube frame or script. New dub shows a video's thumbnail
# from i.ytimg.com before it has a job; the Library shows the cached one from this origin (`/thumb/<id>`).
_CSP = (
    "default-src 'self'; script-src 'self'; frame-src 'none'; img-src 'self' data: https://i.ytimg.com; "
    "style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; media-src 'self' blob:"
)
THUMB_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
               ".gif": "image/gif"}


def _static(ui_dir: Path | None, path: str) -> Response:
    if ui_dir is None:
        return Response(HTTPStatus.NOT_FOUND, "Not Found", Headers(), b"UI not built")
    rel = path.split("?", 1)[0].lstrip("/") or "index.html"
    target = (ui_dir / rel).resolve()
    if not str(target).startswith(str(ui_dir.resolve())) or not target.is_file():
        target = ui_dir / "index.html"  # SPA fallback
    body = target.read_bytes()
    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    headers = Headers({"Content-Type": ctype, "Content-Length": str(len(body)), "Content-Security-Policy": _CSP,
                       "Cache-Control": "no-cache", "Referrer-Policy": "strict-origin-when-cross-origin"})
    return Response(HTTPStatus.OK, "OK", headers, body)


@dataclass(eq=False)
class Client:
    """A connected window: how to reach it."""

    send_json: SendJSON
    send_bytes: SendBytes


class Engine:
    def __init__(self, backend_name: str | None, models_dir: Path, cache_dir: Path, ui_dir: Path | None, token: str, demo: bool,
                 config_dir: Path | None = None) -> None:
        self.backend = load_backend(backend_name, models_dir)
        self.resolver = DemoResolver() if demo else YtDlpResolver()
        self.cache_dir, self.ui_dir, self.token, self.demo = cache_dir, ui_dir, token, demo
        self.config_dir = config_dir if config_dir is not None else config.default_config_dir()
        self.settings = config.load(self.config_dir)  # {"outputDir"}: where the dubbed MP4s go (§4)
        self.opener: Callable[[list[str]], None] = notify.spawn  # runs `open` / `open -R` (tests stub it)
        # Notifications (§4): never for the demo or the mock backend.
        self.notifications = not demo and self.backend.name != "mock"
        self._held: set[str] = set()             # jobs whose first Claude hold of this run was notified
        # One GPU owner at a time for the MLX (ASR, separation) and MPS/CUDA (diarization, TTS) work of the job running
        # (gpu.py): concurrent MLX and MPS work measured slower than taking turns on the M5 Pro (ADR-004). Translation
        # goes through the Claude CLI (ADR-019) and never asks for it.
        self.gpu = GpuScheduler()
        # The job running, until its run() returns (a pause finishing its Claude calls included); the others are on disk.
        self.job: RenderJob | None = None
        self._job_task: asyncio.Task | None = None
        # The jobs waiting to run, first in line first (§4): (video id, settings; None: job.json's). A queued job's
        # job.json says `queued`, from `queuedAt`; but a re-run of the job running waits here only, with its new
        # settings, since that job's run writes job.json until it stops.
        self.queue: list[tuple[str, RenderSettings | None]] = []
        self.clients: set[Client] = set()        # every connected window
        self._control = asyncio.Lock()           # prepare, pause, resume, remove and the re-runs, one at a time
        self._tasks: set[asyncio.Task] = set()
        self._awake: subprocess.Popen | None = None
        self._inspected: dict[str, tuple[str, str, float]] = {}  # video id -> title, channel, duration (`inspect`)
        self._quitting = False                   # Maata is quitting: no job starts any more
        self.stopped = asyncio.Event()           # the graceful stop is done: the server closes, the engine exits

    def claude_health(self) -> dict | None:
        """The Codex CLI's state for the legacy `hello.claude` field, checked afresh on every connect.
        None for the demo engine, which never calls a translation CLI."""
        return None if self.backend.name == "mock" else CodexCLI(self.cache_dir).health()

    async def claude_hello(self) -> dict | None:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.claude_health), HEALTH_TIMEOUT)
        except asyncio.TimeoutError:  # the check runs on in its thread; the UI isn't kept waiting for it
            return {"installed": True, "version": None, "signedIn": None, "models": [], "model": DEFAULT_MODEL,
                    "provider": "codex", "problem": "stalled",
                    "message": f"Codex CLI didn't answer within {HEALTH_TIMEOUT:.0f} s."}

    def process_request(self, conn: ServerConnection, req: Request) -> Response | None:
        u = urlparse(req.path)
        if u.path == "/ws" or u.path.startswith("/thumb/"):
            if parse_qs(u.query).get("token", [""])[0] != self.token:
                return Response(HTTPStatus.FORBIDDEN, "Forbidden", Headers(), b"bad token")
            return None if u.path == "/ws" else self._thumb(u.path.removeprefix("/thumb/"))
        return _static(self.ui_dir, req.path)

    def _thumb_file(self, vid: str, doc: dict | None = None) -> Path | None:
        """A job's cached thumbnail (job.json's `source.thumb`, as yt-dlp wrote it at fetch), if it is there."""
        video_dir = self._video_dir(vid)
        if video_dir is None:
            return None
        if doc is None:
            doc = _read_json(video_dir / "render" / "job.json") or {}
        name = str((doc.get("source") or {}).get("thumb") or "")
        path = video_dir / name
        return path if name and "/" not in name and path.suffix.lower() in THUMB_TYPES and path.is_file() else None

    def _thumb(self, vid: str) -> Response:
        """`/thumb/<video id>?token=`: the Library's thumbnails, from the cache, so it makes no network calls (§5). Only a
        job's own recorded thumbnail, behind the token: which videos were dubbed is nobody else's business."""
        path = self._thumb_file(vid)
        if path is None:
            return Response(HTTPStatus.NOT_FOUND, "Not Found", Headers(), b"no thumbnail")
        body = path.read_bytes()
        return Response(HTTPStatus.OK, "OK", Headers({
            "Content-Type": THUMB_TYPES[path.suffix.lower()], "Content-Length": str(len(body)),
            "Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"}), body)

    # ---- the render jobs and their queue (OFFLINE-RENDER §4) --------------------------------------------------------
    async def start(self) -> None:
        """At startup (§4): a job a crash left running or waiting is `interrupted`; with the jobs a quit left interrupted
        it is queued first, most recently updated first, ahead of the jobs already queued, which keep their order
        (`queuedAt`). A job one stage of which was started CRASH_STARTS times without finishing fails instead (the
        crash-loop guard: graceful stops never count), saying which. A job the user paused stays paused. Then the first
        in line runs, and retention."""
        interrupted, queued = [], []
        for path, doc in await asyncio.to_thread(self._scan):
            vid = path.parent.parent.name
            crashed = doc.get("status") in ("running", "waiting")
            if crashed:
                doc["status"] = "interrupted"
                await asyncio.to_thread(_write_json, path, doc)
            if doc.get("status") == "queued":
                queued.append((float(doc.get("queuedAt") or 0.0), vid))
            if doc.get("status") != "interrupted":
                continue
            stages = doc.get("stages") or {}
            stuck = [(label, stages[key].get("attempts", 0)) for key, label, _ in STAGES
                     if (stages.get(key) or {}).get("state") == "running"
                     and (stages[key].get("attempts") or 0) >= CRASH_STARTS]
            if stuck:
                label, n = stuck[0]
                doc.update(status="failed", error=f"Maata stopped {n} times during {label}, so this dub didn't continue "
                                                  f"on its own. Resume it to try again.")
                await asyncio.to_thread(_write_json, path, doc)
                log.warning("render %s: not continued, %s was started %d times without finishing", vid, label, n)
                continue
            if crashed:
                log.warning("render %s recovered after a crash in %s: it continues", vid, doc.get("stage"))
                self._notify(f"Maata restarted after a problem and is continuing {doc.get('title') or vid}.")
            interrupted.append((float(doc.get("updatedAt") or 0.0), vid))
        self.queue += [(vid, None) for _, vid in sorted(interrupted, reverse=True)]
        self.queue += [(vid, None) for _, vid in sorted(queued)]
        self._next()
        await self._retain()

    def _job(self, ref: str, settings: RenderSettings | None) -> RenderJob:
        return RenderJob(self.backend, self.resolver, self.cache_dir, ref, settings, gpu=self.gpu, on_event=self._job_event,
                         output_dir=Path(self.settings["outputDir"]).expanduser())

    def _launch(self, job: RenderJob) -> None:
        self.job = job
        self._held.discard(job.video_id)
        self._keep_awake(True)
        self._job_task = asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: RenderJob) -> None:
        try:
            status = await job.run()
            self._notify_end(job, status)
        except Exception:
            log.exception("render %s stopped unexpectedly", job.video_id)
        finally:
            if self.job is job:
                self.job = None
                self._keep_awake(False)
                self._next()  # at once: a prepare in between would jump the queue
        await self._retain()  # a job ended

    def _next(self) -> None:
        """Start the first job in line, when none runs and Maata isn't quitting (§4). The jobs behind it move up."""
        if self.job is not None or not self.queue or self._quitting:
            return
        vid, settings = self.queue.pop(0)
        self._launch(self._job(vid, settings))
        if self.queue:
            self._task(self._send_renders())

    def _position(self, vid: str) -> int | None:
        """A job's place in the queue (1: next to run), or None."""
        return next((n for n, (v, _) in enumerate(self.queue, 1) if v == vid), None)

    async def _submit(self, ref: str, settings: RenderSettings) -> None:
        """Run a job with `settings` now if nothing runs, else queue it (§4). The job running as asked is left as it is;
        a re-run of the job running (other settings, or a resume while its pause finishes) pauses it and is first in
        line, with its new settings, from when it has stopped. A job already queued keeps its place with the new
        settings; any other goes to the back. job.json says `queued` (but for that re-run's), and every window gets the
        job's `render` snapshot."""
        vid = _video_id(ref)
        job = self.job
        if job is not None and job.video_id == vid:
            if settings == job.settings and not job.stopping:
                return
            job.pause()
            self.queue = [(vid, settings)] + [(v, s) for v, s in self.queue if v != vid]
            await self._job_event(job.snapshot())
            return
        at = next((n for n, (v, _) in enumerate(self.queue) if v == vid), None)
        if job is None and at is None and not self.queue:
            self._launch(self._job(ref, settings))
            return
        if at is None:
            self.queue.append((vid, settings))
        else:
            self.queue[at] = (vid, settings)
        queued = self._job(ref, settings)
        if not queued.doc["title"] and vid in self._inspected:  # a job that hasn't fetched yet: what `inspect` found
            title, channel, duration = self._inspected[vid]
            queued.doc.update(title=title, channel=channel, duration=duration)
        await queued.hold("queued", time.time() if at is None else None)
        self._next()

    async def _pause(self, vid: str) -> None:
        """Pause a job (§4): the running one stops at its next safe point, and the next in line starts once it has; a
        queued one leaves the queue, `paused`."""
        async with self._control:
            queued = self._position(vid) is not None
            self.queue = [(v, s) for v, s in self.queue if v != vid]
            if self.job is not None and self.job.video_id == vid:
                self.job.pause()
            elif queued:
                await self._job(vid, None).hold("paused")
                await self._send_renders()  # the jobs behind it moved up

    async def shutdown(self) -> None:
        """Quitting is graceful (§4): no job starts any more, and the running job is written as the stop leaves it before
        its task is cancelled (`RenderJob.quit`; a model call in a worker thread may outlast QUIT_WAIT): each of its
        running stages back to `todo` without counting the start (a quit never trips the crash-loop guard, nor sends
        the speakers to the coarse step), and the job `interrupted` (with the settings of a re-run that was waiting
        for it), so it continues, first in line, at the next launch. A job whose pause was still finishing, with no
        re-run waiting, is written `paused` instead: the user paused it, and it stays paused. Queued jobs stay queued.
        Then `stopped` is set: the server closes and the engine exits."""
        self._quitting = True
        job, task = self.job, self._job_task
        if job is not None and task is not None and not task.done():
            rerun = next((s for v, s in self.queue if v == job.video_id and s is not None), None)
            if rerun is not None:
                job.doc["settings"] = rerun.to_json()
            job.quit(paused=job.stopping and rerun is None)
            task.cancel()
            await asyncio.wait([task])
        self._keep_awake(False)
        self.stopped.set()

    def quit(self) -> None:
        """The shell went away (the stdin lifeline calls this on the loop): stop gracefully."""
        self._task(self.shutdown())

    def _notify(self, body: str) -> None:
        """A notification (§4), unless this is the demo or the mock backend."""
        if self.notifications:
            notify.post("Maata", body)

    def _notify_end(self, job: RenderJob, status: str) -> None:
        d = job.doc
        title = d.get("title") or job.video_id
        if status == "done":
            out = d.get("output") or {}
            if out.get("kind") == "preview":
                self._notify(f"The first {round((job.settings.stop_at or 0) / 60)} minutes of {title} are dubbed.")
            else:
                self._notify(f"{title} is dubbed: {Path(str(out.get('path') or '')).name} is saved.")
        elif status == "failed":
            self._notify(f"Dubbing {title} stopped: {d.get('error') or 'something went wrong.'}")

    def _keep_awake(self, on: bool) -> None:
        """While a render runs with a real backend on macOS, `caffeinate -i -w <engine pid>` keeps the Mac from idle
        sleep (M8); it goes when the render pauses or ends, or the engine does. Never for the demo or the mock."""
        if on and self._awake is None and sys.platform == "darwin" and self.backend.name != "mock" and not self.demo \
                and os.path.exists(CAFFEINATE):
            try:
                self._awake = subprocess.Popen([CAFFEINATE, "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                log.warning("couldn't keep the Mac awake", exc_info=True)
        elif not on and self._awake is not None:
            self._awake.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._awake.wait(timeout=1.0)
            self._awake = None

    async def _job_event(self, msg: dict) -> None:
        """What a job says goes to every window: its progress (`render`, with its place in the queue), the speakers it
        found and Claude's state."""
        if msg.get("type") == "render":
            msg = {**msg, "position": self._position(msg["videoId"])}
        elif msg.get("type") == "speakers_found" and msg.get("fresh"):
            title = self.job.doc.get("title") if self.job is not None else ""
            n = len(msg.get("speakers") or [])
            self._notify(f"Maata found {n} speaker{'s' if n != 1 else ''} in {title or msg['videoId']}. Check them "
                         f"before the Telugu speech starts, in about {max(1, round((msg.get('freeFor') or 0) / 60))} min.")
        elif msg.get("type") == "claude_error" and msg.get("videoId") not in self._held:
            self._held.add(msg["videoId"])  # the first hold of the job's run
            title = self.job.doc.get("title") if self.job is not None else ""
            until = (f" It goes on at {time.strftime('%H:%M', time.localtime(msg['resetsAt']))}."
                     if msg.get("resetsAt") else "")
            self._notify(f"Codex is holding back {title or msg['videoId']}: {msg.get('message')}{until}")
        for c in list(self.clients):
            await c.send_json(msg)

    async def _send_renders(self) -> None:
        """The library as it is now, to every window: a job removed, or the queue moved up."""
        msg = {"type": "renders", "items": await asyncio.to_thread(self._renders)}
        for c in list(self.clients):
            await c.send_json(msg)

    async def _retain(self) -> None:
        """Retention (SPEC §8, §13, M6), at startup and whenever a render ends; never the one running, nor one in the
        queue (an interrupted job startup queued included: its job.json still says `interrupted`)."""
        keep = {v for v, _ in self.queue} | ({self.job.video_id} if self.job is not None else set())
        try:
            gone = await asyncio.to_thread(retain, self.cache_dir, keep, time.time())
        except Exception:
            log.exception("retention failed")
            return
        if gone:
            log.info("expired the voice data of %d renders: %s", len(gone), ", ".join(gone))

    # ---- the library ----------------------------------------------------------------------------------------------
    def _scan(self) -> list[tuple[Path, dict]]:
        return [(p, doc) for p in sorted(self.cache_dir.glob("*/render/job.json")) if (doc := _read_json(p)) is not None]

    def _renders(self) -> list[dict]:
        """The `renders` items (§4): every render in the cache, the running one as it is now, newest activity first."""
        items = []
        for path, doc in self._scan():
            vid = path.parent.parent.name
            if self.job is not None and self.job.video_id == vid:
                doc = self.job.doc
            items.append(self._item(doc, path.parent.parent))
        return sorted(items, key=lambda x: -(x["updatedAt"] or 0))

    def _item(self, doc: dict, video_dir: Path) -> dict:
        """A `renders` item. `thumb`: the cached thumbnail's loopback URL (None without one, the demo's included);
        `settings`: the job's options, which New dub starts from for the next job and Continue keeps; `slept` and
        `error` for the Library's chips; and what the job view shows of a job that isn't running, which sends no
        `render` (§5): its stages (no ETA), elapsed time, coverage and report."""
        running = self.job is not None and self.job.video_id == doc.get("videoId")
        settings = doc.get("settings") or {}
        thumb = self._thumb_file(video_dir.name, doc)
        stages = doc.get("stages") or {}
        return {"videoId": doc.get("videoId"), "title": doc.get("title", ""), "channel": doc.get("channel", ""),
                "duration": doc.get("duration", 0.0), "status": doc.get("status"),
                "thumb": f"/thumb/{video_dir.name}?token={self.token}" if thumb is not None else None,
                "position": self._position(video_dir.name), "stopAt": settings.get("stopAt"), "settings": settings,
                "finalUntil": doc.get("finalUntil"), "stage": doc.get("stage"),
                "eta": self.job.snapshot()["eta"] if running else None, "updatedAt": doc.get("updatedAt"),
                "createdAt": doc.get("createdAt"), "bytes": _bytes(video_dir), "output": output_view(doc),
                "slept": doc.get("slept"), "error": doc.get("error"), "expired": bool(doc.get("expired")),
                "stages": [{"key": key, "label": label, **{f: stages[key].get(f) for f in
                                                          ("state", "done", "total", "unit", "seconds")},
                            "state": shown_state(key, stages[key], float(doc.get("duration") or 0.0),
                                                 settings.get("stopAt")), "eta": None}
                           for key, label, _ in STAGES if isinstance(stages.get(key), dict)],
                "elapsed": doc.get("elapsed") or 0.0, "coverage": doc.get("coverage"), "report": doc.get("report")}

    def _found(self) -> list[dict]:
        """The speaker check of every job whose speakers stage is done for its count (`render.found_on_disk`), the
        running job's from its own doc: what a window connecting now has missed (§5)."""
        found = []
        for path, doc in self._scan():
            if self.job is not None and self.job.video_id == path.parent.parent.name:
                doc = self.job.doc
            if (msg := found_on_disk(path.parent.parent, doc)) is not None:
                found.append(msg)
        return found

    async def _send_found(self, c: Client) -> None:
        """After `hello`, the speaker checks a new window needs (the job view's Wrong count?, Hear voice and stock-voice
        switch): a job broadcasts its own only while its speakers stage runs, which a reload, an engine restart or a
        relaunch misses."""
        for msg in await asyncio.to_thread(self._found):
            await c.send_json(msg)

    async def _render_item(self, vid: str) -> dict | None:
        video_dir = self.cache_dir / vid
        if self.job is not None and self.job.video_id == vid:
            doc = self.job.doc
        else:
            doc = await asyncio.to_thread(_read_json, video_dir / "render" / "job.json")
        return None if doc is None else await asyncio.to_thread(self._item, doc, video_dir)

    async def _video(self, vid: str, start: float, title: str, channel: str, duration: float,
                     stop_at: float | None = None) -> dict:
        """The `video` message (§4): the metadata, its thumbnail, its render (or None) and an estimate for the options,
        with the seconds of work ahead of it when a job runs (`ahead`), and the estimate of a first-15-minutes preview
        (`preview`), so New dub's Length switch needs no second `inspect` (a YouTube call)."""
        separates = self.backend.separator is not None
        return {"type": "video", "videoId": vid, "start": start, "title": title, "channel": channel, "duration": duration,
                "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", "render": await self._render_item(vid),
                "estimate": {**estimate(duration, stop_at, separates=separates),
                             "preview": estimate(duration, PREVIEW_AT, separates=separates),
                             "ahead": await asyncio.to_thread(self._ahead, vid)}}

    def _ahead(self, vid: str) -> int | None:
        """Seconds of work before a new job for `vid` would start (§4): what the running job has left and what the queued
        jobs have (`render.left`, by the priors); None when nothing runs."""
        job = self.job
        if job is None:
            return None
        ahead = 0.0 if job.video_id == vid else float(job.snapshot()["eta"] or 0.0)
        for v, _ in list(self.queue):
            if v not in (vid, job.video_id):  # (a re-run of the job running is counted in its time left)
                ahead += left(_read_json(self.cache_dir / v / "render" / "job.json") or {},
                              separates=self.backend.separator is not None)
        return round(ahead)

    # ---- messages ---------------------------------------------------------------------------------------------------
    def _task(self, coro: Coroutine) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda t: t.cancelled() or t.exception() is None
                               or log.error("request failed", exc_info=t.exception()))

    async def _inspect(self, c: Client, msg: dict) -> None:
        try:
            ref = parse_youtube_url(str(msg.get("url", "")))
            video = await asyncio.to_thread(self.resolver.resolve, ref, self.cache_dir, download=False)
        except (URLError, ResolveError) as e:
            await c.send_json({"type": "error", "message": str(e), "retryable": isinstance(e, ResolveError),
                               "url": msg.get("url")})
            return
        self._inspected[ref.video_id] = (video.title, video.channel, video.duration)
        await c.send_json({**await self._video(ref.video_id, ref.start, video.title, video.channel, video.duration,
                                               _stop_at(msg.get("stopAt"))), "url": msg.get("url")})

    def _video_dir(self, vid: str) -> Path | None:
        """A job's folder: `vid` must be a YouTube video id naming a direct child of the cache folder (§4); None for
        anything else (an empty id, `..`, a path, a link out of the cache), so no message reaches outside the cache."""
        if not VIDEO_ID.fullmatch(vid):
            return None
        path = (self.cache_dir / vid).resolve()
        return path if path.parent == self.cache_dir.resolve() else None

    async def _settings(self, vid: str) -> RenderSettings:
        """A job's settings now: those of a re-run waiting for it, the running job's, else job.json's."""
        waiting = next((s for v, s in self.queue if v == vid and s is not None), None)
        if waiting is not None:
            return waiting
        if self.job is not None and self.job.video_id == vid:
            return self.job.settings
        doc = await asyncio.to_thread(_read_json, self.cache_dir / vid / "render" / "job.json") or {}
        return RenderSettings.from_json(doc.get("settings"))

    async def _prepare(self, c: Client, msg: dict) -> None:
        """Create or update a job, and run it, or queue it (§4). An old UI's `allowFreeze` is ignored (§2.11)."""
        async with self._control:
            try:
                ref = parse_youtube_url(str(msg.get("url", "")))
                saved = await self._settings(ref.video_id)
                k = msg.get("speakers", "auto")
                settings = RenderSettings(
                    speakers=None if k in (None, "auto") else int(k), style=str(msg.get("style") or "colloquial"),
                    stop_at=_stop_at(msg.get("stopAt")), speed_cap=float(msg.get("speedCap") or 1.2),
                    tts_script=str(msg.get("ttsScript") or "telugu"), presets=saved.presets)
            except (URLError, ValueError, TypeError) as e:
                await c.send_json({"type": "error", "message": str(e), "retryable": False})
                return
            await self._submit(f"https://youtu.be/{ref.video_id}", settings)

    async def _resume(self, c: Client, vid: str) -> None:
        """Run the job again from disk with its settings: now if nothing runs, else queued (§4)."""
        async with self._control:
            video_dir = self._video_dir(vid)
            if video_dir is None or not (video_dir / "render" / "job.json").is_file():
                await c.send_json({"type": "error", "message": "There is no dub of this video to resume.",
                                   "retryable": False})
                return
            await self._submit(vid, await self._settings(vid))

    async def _rerun(self, c: Client, vid: str, change: Callable[[RenderSettings], RenderSettings]) -> None:
        """`set_speakers`, `set_voice`: change the job's settings and run it again from disk (only what the change
        touches is done again, §2.3, §2.9): the job running is paused and goes first in line, any other is queued."""
        async with self._control:
            video_dir = self._video_dir(vid)
            if video_dir is None or not (video_dir / "render" / "job.json").is_file():
                return
            try:
                settings = change(await self._settings(vid))
            except (ValueError, TypeError) as e:
                await c.send_json({"type": "error", "message": str(e), "retryable": False})
                return
            await self._submit(vid, settings)

    async def _remove(self, c: Client, vid: str, forget: bool) -> None:
        """`remove` (§4): take a job out of the library. Refused for anything but a video id naming a folder of the
        cache, and while the job runs or its run hasn't returned (a pause finishing its Claude calls): "Pause it
        first". It leaves the queue first. Its render/ and media go; its translations and trace stay, unless `forget`,
        which deletes the whole folder. Then every window gets `renders`."""
        async with self._control:
            video_dir = self._video_dir(vid)
            if video_dir is None:
                await c.send_json({"type": "error", "message": "There is no dub of this video to remove.",
                                   "retryable": False})
                return
            self.queue = [(v, s) for v, s in self.queue if v != vid]
            if self.job is not None and self.job.video_id == vid:
                await c.send_json({"type": "error", "message": "Pause it first.", "retryable": False})
                return
            running = self.job.doc.get("part") if self.job is not None else None
            await asyncio.to_thread(_delete, video_dir, forget, Path(running) if running else None)
            log.info("render %s removed%s", vid, " with its translations" if forget else "")
            await self._send_renders()

    async def _set_settings(self, c: Client, msg: dict) -> None:
        """`settings {outputDir}` (§4): the folder is checked (absolute, made if missing, writable), saved in the config
        folder and sent to every window; a job that starts from now on saves there."""
        try:
            folder = await asyncio.to_thread(config.output_dir, str(msg.get("outputDir") or ""))
        except config.SettingsError as e:
            await c.send_json({"type": "error", "message": str(e), "retryable": False})
            return
        self.settings = {**self.settings, "outputDir": str(folder)}
        await asyncio.to_thread(config.save, self.config_dir, self.settings)
        for x in list(self.clients):
            await x.send_json({"type": "settings", **self.settings})

    async def _open_output(self, c: Client, vid: str, reveal: bool) -> None:
        """`open_output {videoId, reveal}` (§2.17): the job's own recorded MP4 opened in the default player, or shown in
        Finder; its folder when the file is gone. Never a path the UI sends."""
        video_dir = self._video_dir(vid)
        doc = None
        if video_dir is not None:
            doc = self.job.doc if self.job is not None and self.job.video_id == vid else \
                await asyncio.to_thread(_read_json, video_dir / "render" / "job.json")
        out = (doc or {}).get("output")
        if not isinstance(out, dict) or not out.get("path"):
            await c.send_json({"type": "error", "message": "This dub has no saved video yet.", "retryable": False})
            return
        path = Path(out["path"])
        if path.is_file():
            cmd = _open_command(path, reveal)
        elif path.parent.is_dir():
            cmd = _open_command(path.parent, False)
        else:
            await c.send_json({"type": "error", "message": "The saved video and its folder were moved or deleted.",
                               "retryable": False})
            return
        try:
            self.opener(cmd)
        except OSError as e:
            await c.send_json({"type": "error", "message": f"Couldn't open it ({e.strerror or e}).", "retryable": False})

    async def _voice_sample(self, c: Client, vid: str, speaker: str) -> None:
        video_dir = self._video_dir(vid)
        if video_dir is None or not await voice_sample(video_dir / "render", speaker, self.backend.tts.sample_rate,
                                                       c.send_bytes):
            log.info("no Hear voice sample for %s of %s", speaker, vid)

    async def handler(self, ws: ServerConnection) -> None:
        # Once the UI has gone, sends are dropped: a task still answering it (an inspect, a voice sample) must not fail
        # on it.
        async def sj(msg: dict) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(json.dumps(msg, ensure_ascii=False))

        async def sb(data: bytes) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(data)

        client = Client(sj, sb)
        await sj({"type": "hello", "backend": self.backend.name, "device": self.backend.device, "demo": self.demo or self.backend.name == "mock",
                  "claude": await self.claude_hello(), "renders": await asyncio.to_thread(self._renders),
                  "render": {**self.job.snapshot(), "position": self._position(self.job.video_id)}
                  if self.job is not None else None, "settings": self.settings})
        self.clients.add(client)  # (before the speaker checks: one sent meanwhile reaches this window too)
        self._task(self._send_found(client))
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                vid = str(msg.get("videoId", ""))
                # The background dub (OFFLINE-RENDER §4).
                if kind == "inspect":
                    self._task(self._inspect(client, msg))
                elif kind == "prepare" and "url" in msg:
                    self._task(self._prepare(client, msg))
                elif kind == "pause":
                    self._task(self._pause(vid))
                elif kind == "resume":
                    self._task(self._resume(client, vid))
                elif kind == "set_speakers":
                    k = msg.get("speakers", "auto")
                    self._task(self._rerun(client, vid, lambda s, k=k: replace(
                        s, speakers=None if k in (None, "auto") else int(k))))
                elif kind == "set_voice":
                    sid, on = str(msg.get("speaker", "")), bool(msg.get("usePreset"))
                    self._task(self._rerun(client, vid, lambda s, sid=sid, on=on: replace(
                        s, presets=tuple(set(s.presets) | {sid}) if on else tuple(set(s.presets) - {sid}))))
                elif kind == "remove":
                    self._task(self._remove(client, vid, bool(msg.get("forget"))))
                elif kind == "renders":
                    await sj({"type": "renders", "items": await asyncio.to_thread(self._renders)})
                elif kind == "voice_sample":
                    self._task(self._voice_sample(client, vid, str(msg.get("speaker", ""))))
                elif kind == "settings":
                    self._task(self._set_settings(client, msg))
                elif kind == "open_output":
                    self._task(self._open_output(client, vid, bool(msg.get("reveal"))))
        finally:
            self.clients.discard(client)  # the jobs run on


def retain(cache_dir: Path, keep: Collection[str], now: float, days: float = RETAIN_DAYS, cap: float = CACHE_CAP
           ) -> list[str]:
    """Retention (SPEC §8, §13, M6; OFFLINE-RENDER §4): renders last updated (`updatedAt`) over `days` ago lose their
    voice data and source media (`expire`); then, while the cache holds more than `cap` bytes, done whole-video jobs (their
    MP4 is written) lose theirs, least recently updated first, and expired jobs any of it they hold again. A job waiting
    to be continued (paused, interrupted, waiting, or a done preview) never counts toward the cap: losing its takes would
    cost hours of speech synthesis; it follows the age rule only. Never those in `keep` (the running one and the
    engine's queue), nor one job.json says is queued: they are about to run (§4). The output folder is never touched.
    Returns the video ids expired."""
    renders = []
    for path in cache_dir.glob("*/render/job.json"):
        doc = _read_json(path)
        vid = path.parent.parent.name
        if doc is None or vid in keep or doc.get("status") == "queued":
            continue
        renders.append((float(doc.get("updatedAt") or 0.0), vid, doc))
    renders.sort(key=lambda x: (x[0], x[1]))
    gone = [vid for at, vid, doc in renders if not doc.get("expired") and now - at > days * 86400.0]
    for vid in gone:
        expire(cache_dir / vid, next(doc for _, v, doc in renders if v == vid))
    size = _bytes(cache_dir)
    for _, vid, doc in renders:
        if size <= cap:
            break
        whole = doc.get("status") == "done" and (doc.get("settings") or {}).get("stopAt") is None
        if vid in gone or not (whole or doc.get("expired")):
            continue
        before = bool(doc.get("expired"))
        freed = expire(cache_dir / vid, doc)
        size -= freed
        if freed or not before:  # (an expired job is listed only when it held something again)
            gone.append(vid)
    return gone


def expire(video_dir: Path, doc: dict) -> int:
    """A render's voice data (voices, takes, line PCM, the manifest), its analysis audio, its background sound (the bed)
    and its source media (audio and video; and the pcm/ the streaming player of before left) are deleted; its
    translations, transcript, speakers, trace and the record of its output stay, and job.json says `expired`: a prepare
    voices it again. Returns the bytes freed."""
    render_dir = video_dir / "render"
    paths = [render_dir / name for name in EXPIRE_RENDER] + [p for g in EXPIRE_MEDIA for p in video_dir.glob(g)] \
        + [video_dir / "pcm"]
    freed = 0
    for p in paths:
        freed += _bytes(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                p.unlink()
    doc.update(expired=True, finalUntil=None)
    for key in EXPIRE_STAGES:
        if isinstance((doc.get("stages") or {}).get(key), dict):
            doc["stages"][key].update(state="todo", done=0)
    _write_json(render_dir / "job.json", doc)
    return freed


def _delete(video_dir: Path, forget: bool, running: Path | None = None) -> None:
    """A removed job's files (§4): everything in its folder, render/ (the job, takes, PCM), its media (audio, video,
    thumbnail) and the pcm/ the streaming player of before left included, but its translations (lines.jsonl,
    briefs.jsonl) and trace (units.jsonl), so dubbing the video again doesn't pay for them again; with `forget` the whole
    folder goes. The job's own unfinished .part.mp4 in the output folder goes too, unless it is the running job's
    (`running`: a job of the same title writes the same name); its finished MP4 always stays."""
    part = Path(str((_read_json(video_dir / "render" / "job.json") or {}).get("part") or ""))
    if part.name.startswith(".") and part.name.endswith(".part.mp4") \
            and (running is None or part.resolve() != running.resolve()):
        with contextlib.suppress(OSError):
            part.unlink()
    if forget:
        shutil.rmtree(video_dir, ignore_errors=True)
        return
    for p in video_dir.iterdir():
        if p.name in REMOVE_KEEP:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                p.unlink()


def _bytes(path: Path) -> int:
    """Bytes of a file, or of the files under a directory."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.stat(os.path.join(root, f)).st_size
    return total


def _open_command(path: Path, reveal: bool) -> list[str]:
    """Open `path` in its default app, or with `reveal` show it in the file manager."""
    if sys.platform == "darwin":
        return ["/usr/bin/open", "-R", str(path)] if reveal else ["/usr/bin/open", str(path)]
    if sys.platform == "win32":
        return ["explorer", f"/select,{path}"] if reveal else ["explorer", str(path)]
    return ["xdg-open", str(path.parent if reveal else path)]


def _stop_at(value: object) -> float | None:
    return None if value in (None, "", 0) else float(value)  # type: ignore[arg-type]


def _video_id(ref: str) -> str:
    return ref if "/" not in ref and "." not in ref else parse_youtube_url(ref).video_id


async def run(args: argparse.Namespace) -> None:
    token = args.token or secrets.token_urlsafe(24)
    ui_dir = Path(args.ui).resolve() if args.ui else None
    engine = Engine(args.backend, Path(args.models).expanduser(), Path(args.cache).expanduser(), ui_dir, token, args.demo,
                    Path(args.config_dir).expanduser() if args.config_dir else None)
    async with serve(engine.handler, "127.0.0.1", args.port, process_request=engine.process_request, max_size=2**24) as server:
        port = server.sockets[0].getsockname()[1]
        # The Tauri shell parses this line to find the engine.
        print(f"MAATA_ENGINE_READY port={port} token={token} backend={engine.backend.name}", flush=True)
        if args.stdin_lifeline:
            threading.Thread(target=_stop_when_stdin_closes, args=(asyncio.get_running_loop(), engine),
                             name="stdin-lifeline", daemon=True).start()
        await engine.start()  # a render Maata was closed during continues
        await engine.stopped.wait()  # (set only by a graceful stop: otherwise it serves until killed)


def _setup_logging(log_dir: Path) -> None:
    """Log to stderr and to a rotating engine.log, so a dub can be inspected after the fact."""
    from logging.handlers import RotatingFileHandler

    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "engine.log", maxBytes=20_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
    except OSError:
        log.warning("can't write logs to %s", log_dir, exc_info=True)
    logging.getLogger("websockets").setLevel(logging.WARNING)  # per-request lines drown the dub log


def _stop_when_stdin_closes(loop: asyncio.AbstractEventLoop, engine: Engine) -> None:
    """The shell holds our stdin open. EOF means it's gone (it quit, or was killed without cleanup): the engine stops
    gracefully (`Engine.shutdown`: the running job is written interrupted, to continue at the next launch) rather than
    linger holding gigabytes of models, and exits at once if that takes more than QUIT_WAIT s (a worker thread still
    finishing a model call keeps a normal exit waiting)."""
    sys.stdin.buffer.read()
    log.info("shell went away; stopping")
    fallback = threading.Timer(QUIT_WAIT, os._exit, (0,))
    fallback.daemon = True
    fallback.start()
    try:
        loop.call_soon_threadsafe(engine.quit)
    except RuntimeError:  # the loop has closed already
        os._exit(0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maata-engine")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--token")
    p.add_argument("--backend", choices=["apple", "cuda", "mock"])
    p.add_argument("--models", default="~/Library/Application Support/Maata/Models" if sys.platform == "darwin" else "~/.local/share/maata/models")
    p.add_argument("--cache", default="~/Library/Caches/Maata" if sys.platform == "darwin" else "~/.cache/maata")
    p.add_argument("--ui", help="directory of the built web UI to serve")
    p.add_argument("--demo", action="store_true", help="no network: synthetic video audio (with --backend mock)")
    p.add_argument("--stdin-lifeline", action="store_true", help="exit when stdin closes (the launching shell holds it)")
    p.add_argument("--log-dir", default="~/Library/Logs/Maata" if sys.platform == "darwin" else "~/.local/state/maata/logs")
    p.add_argument("--config-dir", help="the engine's settings folder (default: next to Models; ~/.config/maata on Linux)")
    args = p.parse_args(argv)
    _setup_logging(Path(args.log_dir).expanduser())
    Path(args.cache).expanduser().mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
