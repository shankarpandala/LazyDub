"""End-to-end over the real WebSocket with the demo engine: no network, no models."""

import asyncio
import json
import struct

import pytest
from websockets.asyncio.client import connect

from maata_engine.server import Engine
from websockets.asyncio.server import serve


@pytest.fixture
async def engine_url(tmp_path):
    eng = Engine("mock", tmp_path / "models", tmp_path / "cache", None, "tok", demo=True)
    async with serve(eng.handler, "127.0.0.1", 0, process_request=eng.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws"


async def test_rejects_bad_token(engine_url):
    with pytest.raises(Exception):
        async with connect(engine_url + "?token=nope") as ws:
            await ws.recv()


async def test_demo_session_streams_placed_units(engine_url):
    async with connect(engine_url + "?token=tok") as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello" and hello["backend"] == "mock"
        await ws.send(json.dumps({"type": "open", "url": "https://youtu.be/dQw4w9WgXcQ"}))
        units, audio = [], {}
        async def pump():
            while len(audio) < 8:
                m = await ws.recv()
                if isinstance(m, bytes):
                    uid, sr, n, _ = struct.unpack("<IIII", m[:16])
                    assert len(m) == 16 + 4 * n and sr == 24000
                    audio[uid] = n
                else:
                    msg = json.loads(m)
                    if msg["type"] == "unit":
                        units.append(msg)
                    assert msg.get("type") != "error", msg
        await asyncio.wait_for(pump(), 20)
    starts = [u["start"] for u in units]
    assert starts == sorted(starts)
    for u in units:
        assert 1.0 <= u["audioRate"] <= 1.2
        assert u["telugu"]
