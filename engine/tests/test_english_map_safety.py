"""A failed or incomplete semantic review must never authorize Latin substitutions. Original test sentences only."""

from types import SimpleNamespace

import pytest

from maata_engine.backends.base import Coverage, LineResult, SceneRequest, SceneResult, Wording
from maata_engine.backends.claude_translator import line_json
from maata_engine.dubber import Dubber
from maata_engine.qa.validators import wording


def job_with(reviewer):
    job = object.__new__(Dubber)
    job.units = {1: object()}
    job._pick = lambda state, line: ("full", True)
    job.tts_script = "latin"
    job.tr = SimpleNamespace(review=reviewer)
    return job


@pytest.mark.asyncio
async def test_unexpected_reviewer_exception_preserves_complete_telugu():
    async def broken(*args):
        raise RuntimeError("unexpected parser failure")

    job = job_with(broken)
    original = Wording("ఫోన్ చేసే ఫ్రెండ్.", ((0, "phone"), (2, "friend")))
    result = await job._review(SceneRequest(1, ()), {1: LineResult(1, {"full": original})})
    assert result[1].full.spoken == original.spoken
    assert result[1].full.english == ()
    assert job._tts_text(result[1].full) == original.spoken


@pytest.mark.asyncio
@pytest.mark.parametrize("coverage,may_map", [
    (None, False), (Coverage("C", tier="full", by="validators"), False),
    (Coverage("P", tier="full"), False), (Coverage("C", tier="concise"), False),
    (Coverage("C", tier="full"), True), (Coverage("m", tier="full"), True),
])
async def test_only_actual_semantically_reviewed_tier_may_substitute_latin(coverage, may_map):
    original = Wording("ఫోన్ చేసే ఫ్రెండ్.", ((0, "phone"), (2, "friend")))
    line = LineResult(1, {"full": original}, coverage=coverage)

    async def reviewer(*args):
        return SceneResult(1, "review", lines={1: line})

    job = job_with(reviewer)
    result = await job._review(SceneRequest(1, ()), {1: line})
    assert job._tts_text(result[1].full) == ("phone చేసే friend." if may_map else original.spoken)


def test_validated_map_cache_round_trip_uses_surface_anchors():
    original = Wording("ఫోన్ ఫోన్ అన్నాడు.", ((1, "phone"),))
    raw = line_json(LineResult(1, {"full": original}))["full"]
    assert raw["english"] == [{"word": "ఫోన్", "occurrence": 1, "en": "phone"}]
    restored, errors, flags = wording(raw)
    assert not errors and not flags and restored == original
