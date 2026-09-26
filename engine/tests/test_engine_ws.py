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
        assert u["telugu"] and abs(u["lag"]) < 1.0  # each on its own line's time, not pushed along by another
    assert hello["claude"] is None  # the demo engine never calls Claude


async def test_seek_forward_then_back_keeps_every_line_on_its_own_time(engine_url):
    """A seek into video not processed yet, then back to the start: the lines voiced after the seek back (behind the ones
    voiced far ahead) start on their own English, not after the far ones, and what was dubbed near the start is sent
    again. Closing right after a seek, while that re-send runs, is quiet."""
    async with connect(engine_url + "?token=tok", max_size=2**24) as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "open", "url": "https://youtu.be/dQw4w9WgXcQ", "lookahead": 60}))
        units, audio, phase = {}, [], ["start"]

        async def pump(done) -> None:
            while not done():
                m = await ws.recv()
                if isinstance(m, bytes):
                    audio.append((phase[0], struct.unpack("<IIII", m[:16])[0]))
                else:
                    msg = json.loads(m)
                    assert msg["type"] not in ("error", "claude_error"), msg
                    if msg["type"] == "unit":
                        units[msg["id"]] = (phase[0], msg)

        await asyncio.wait_for(pump(lambda: len(audio) >= 3), 20)
        phase[0] = "far"
        await ws.send(json.dumps({"type": "seek", "time": 140.0}))
        await asyncio.wait_for(pump(lambda: any(p == "far" and u["start"] >= 130 for p, u in units.values())), 20)
        phase[0] = "back"
        await ws.send(json.dumps({"type": "seek", "time": 1.0}))
        await ws.send(json.dumps({"type": "playhead", "time": 1.0}))
        await asyncio.wait_for(pump(lambda: sum(p == "back" and u["start"] < 130 for p, u in units.values()) >= 3), 20)
        await ws.send(json.dumps({"type": "seek", "time": 150.0}))
    back = [u for p, u in units.values() if p == "back" and u["start"] < 130]
    assert back and all(abs(u["lag"]) < 1.0 for u in back), [(u["id"], u["start"], u["lag"]) for u in back]
    assert any(p == "back" and uid in {i for i, (q, _) in units.items() if q == "start"} for p, uid in audio)


def test_stdin_lifeline_exits_when_shell_goes_away(tmp_path):
    """The shell holds the engine's stdin; if it dies (even SIGKILL), the engine must not linger."""
    import subprocess
    import sys

    p = subprocess.Popen(
        [sys.executable, "-m", "maata_engine.server", "--backend", "mock", "--demo", "--port", "0",
         "--models", str(tmp_path / "m"), "--cache", str(tmp_path / "c"), "--stdin-lifeline"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        assert p.stdout.readline().startswith("MAATA_ENGINE_READY ")
        p.stdin.close()  # what the kernel does to the pipe when the shell process dies
        assert p.wait(timeout=10) == 0
    finally:
        p.kill()
