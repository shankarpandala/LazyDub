"""Engine entry point: serves the UI over loopback HTTP (ADR-006) and a token-guarded WebSocket (ADR-007)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import secrets
import sys
import threading
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from .backends import load_backend
from .claude_cli import DEFAULT_MODEL, ClaudeCLI
from .gpu import GpuScheduler
from .resolve import DemoResolver, YtDlpResolver
from .session import Session

log = logging.getLogger("maata.server")
DIAG_FIELDS = ("audio", "rate", "gain", "volume", "buffers", "live", "units", "lead", "waiting", "missing", "time")

HEALTH_TIMEOUT = 10.0  # s: `hello` waits no longer than this for the Claude CLI's version and sign-in state

_CSP = (
    "default-src 'self'; script-src 'self' https://www.youtube.com https://s.ytimg.com; "
    "frame-src https://www.youtube-nocookie.com https://www.youtube.com; img-src 'self' data: https://i.ytimg.com; "
    "style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; media-src 'self' blob:"
)


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


class Engine:
    def __init__(self, backend_name: str | None, models_dir: Path, cache_dir: Path, ui_dir: Path | None, token: str, demo: bool) -> None:
        self.backend = load_backend(backend_name, models_dir)
        self.resolver = DemoResolver() if demo else YtDlpResolver()
        self.cache_dir, self.ui_dir, self.token, self.demo = cache_dir, ui_dir, token, demo
        # One GPU owner at a time for the MLX (ASR) and MPS/CUDA (diarization, TTS) work of all sessions, by priority
        # (gpu.py, ARCHITECTURE §5.3): concurrent MLX and MPS work measured slower than taking turns on the M5 Pro
        # (ADR-004; ADR-016). Translation goes through the Claude CLI (ADR-019) and never asks for it.
        self.gpu = GpuScheduler()

    def claude_health(self) -> dict | None:
        """The Claude CLI's state for `hello`, checked afresh on every connect (the user may have signed in or updated
        since); None for the demo engine, which never calls Claude."""
        return None if self.backend.name == "mock" else ClaudeCLI(self.cache_dir).health()

    async def claude_hello(self) -> dict | None:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.claude_health), HEALTH_TIMEOUT)
        except asyncio.TimeoutError:  # the check runs on in its thread; the UI isn't kept waiting for it
            return {"installed": True, "version": None, "signedIn": None, "models": [], "model": DEFAULT_MODEL,
                    "problem": "stalled", "message": f"Claude Code didn't answer within {HEALTH_TIMEOUT:.0f} s."}

    def process_request(self, conn: ServerConnection, req: Request) -> Response | None:
        u = urlparse(req.path)
        if u.path == "/ws":
            if parse_qs(u.query).get("token", [""])[0] != self.token:
                return Response(HTTPStatus.FORBIDDEN, "Forbidden", Headers(), b"bad token")
            return None
        return _static(self.ui_dir, req.path)

    async def handler(self, ws: ServerConnection) -> None:
        # Once the UI has gone, sends are dropped: the session is closed as the loop below ends, and a stage or a
        # re-send still under way until then must not fail on it.
        async def sj(msg: dict) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(json.dumps(msg, ensure_ascii=False))

        async def sb(data: bytes) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(data)

        session: Session | None = None
        await sj({"type": "hello", "backend": self.backend.name, "device": self.backend.device, "demo": self.demo or self.backend.name == "mock",
                  "claude": await self.claude_hello()})
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "open":
                    if session:
                        await session.close()
                    session = Session(self.backend, self.resolver, self.cache_dir, sj, sb, gpu=self.gpu,
                                      lookahead=float(msg.get("lookahead") or 600), style=str(msg.get("style") or "colloquial"),
                                      speed_cap=float(msg.get("speedCap") or 1.2), allow_freeze=bool(msg.get("allowFreeze", True)),
                                      clone_strength=str(msg.get("cloneStrength") or "closest"),
                                      tts_script=str(msg.get("ttsScript") or "telugu"))
                    asyncio.create_task(session.open(str(msg.get("url", ""))))
                elif session and kind == "player":
                    session.set_player_rates([float(r) for r in msg.get("rates", [])])
                elif session and kind == "seek":
                    session.seek(float(msg["time"]))
                elif session and kind == "playhead":
                    session.set_playhead(float(msg["time"]), bool(msg["playing"]) if "playing" in msg else None)
                elif session and kind == "audio":
                    session.ask_audio([int(i) for i in msg.get("ids", [])])
                elif session and kind == "prepare":
                    session.set_prepare_all(bool(msg.get("whole")))
                elif session and kind == "speed":
                    await session.set_speed(float(msg["speed"]))
                elif session and kind == "speaker_preset":
                    await session.set_speaker_preset(str(msg["speaker"]), bool(msg["usePreset"]))
                elif kind == "diag":  # the UI's playback state, for the log only (App.svelte: audio output, gain, held audio)
                    log.info("ui playback %s", {k: msg[k] for k in DIAG_FIELDS if isinstance(msg.get(k), (int, float, str, bool))})
                elif kind == "close" and session:
                    await session.close()
                    session = None
        finally:
            if session:
                await session.close()


async def run(args: argparse.Namespace) -> None:
    token = args.token or secrets.token_urlsafe(24)
    ui_dir = Path(args.ui).resolve() if args.ui else None
    engine = Engine(args.backend, Path(args.models).expanduser(), Path(args.cache).expanduser(), ui_dir, token, args.demo)
    async with serve(engine.handler, "127.0.0.1", args.port, process_request=engine.process_request, max_size=2**24) as server:
        port = server.sockets[0].getsockname()[1]
        # The Tauri shell parses this line to find the engine.
        print(f"MAATA_ENGINE_READY port={port} token={token} backend={engine.backend.name}", flush=True)
        await server.serve_forever()


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


def _exit_when_stdin_closes() -> None:
    """The shell holds our stdin open. EOF means it's gone, even if it was killed without cleanup,
    so exit rather than linger holding gigabytes of models."""
    sys.stdin.buffer.read()
    log.info("shell went away; exiting")
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
    args = p.parse_args(argv)
    _setup_logging(Path(args.log_dir).expanduser())
    if args.stdin_lifeline:
        threading.Thread(target=_exit_when_stdin_closes, name="stdin-lifeline", daemon=True).start()
    Path(args.cache).expanduser().mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
