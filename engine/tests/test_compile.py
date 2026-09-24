"""Every engine module must at least compile on the oldest supported Python, including backends
whose ML dependencies aren't installed in CI."""

import pathlib
import py_compile

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "maata_engine"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)
