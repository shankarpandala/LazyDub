"""Engine entry point: serves the UI over loopback HTTP (ADR-006) and a token-guarded WebSocket (ADR-007)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import secrets
import sys
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from .backends import load_backend
from .resolve import DemoResolver, YtDlpResolver
from .session import Session

log = logging.getLogger("maata.server")

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
        self.gpu = asyncio.Lock()

    def process_request(self, conn: ServerConnection, req: Request) -> Response | None:
        u = urlparse(req.path)
        if u.path == "/ws":
            if parse_qs(u.query).get("token", [""])[0] != self.token:
                return Response(HTTPStatus.FORBIDDEN, "Forbidden", Headers(), b"bad token")
            return None
        return _static(self.ui_dir, req.path)

    async def handler(self, ws: ServerConnection) -> None:
        async def sj(msg: dict) -> None:
            await ws.send(json.dumps(msg, ensure_ascii=False))

        async def sb(data: bytes) -> None:
            await ws.send(data)

        session: Session | None = None
        await sj({"type": "hello", "backend": self.backend.name, "device": self.backend.device, "demo": self.demo or self.backend.name == "mock"})
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "open":
                    if session:
                        await session.close()
                    session = Session(self.backend, self.resolver, self.cache_dir, sj, sb, gpu_lock=self.gpu)
                    asyncio.create_task(session.open(str(msg.get("url", ""))))
                elif session and kind == "seek":
                    session.seek(float(msg["time"]))
                elif session and kind == "playhead":
                    session.set_playhead(float(msg["time"]))
                elif session and kind == "speed":
                    await session.set_speed(float(msg["speed"]))
                elif session and kind == "speaker_preset":
                    await session.set_speaker_preset(str(msg["speaker"]), bool(msg["usePreset"]))
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maata-engine")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--token")
    p.add_argument("--backend", choices=["apple", "cuda", "mock"])
    p.add_argument("--models", default="~/Library/Application Support/Maata/Models" if sys.platform == "darwin" else "~/.local/share/maata/models")
    p.add_argument("--cache", default="~/Library/Caches/Maata" if sys.platform == "darwin" else "~/.cache/maata")
    p.add_argument("--ui", help="directory of the built web UI to serve")
    p.add_argument("--demo", action="store_true", help="no network: synthetic video audio (with --backend mock)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    Path(args.cache).expanduser().mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
