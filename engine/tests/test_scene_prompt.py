"""The scene prompt (ARCHITECTURE §4.3, §4.4): a byte-identical system prompt, versioned examples written to the
Telugu-script contract, the per-scene user message and the draft-07 schemas."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from maata_engine.backends.base import Brief, GlossaryEntry, LineSpec, SceneRequest, SpeakerNote, VideoMeta, Wording
from maata_engine.claude_cli import check_schema, validate
from maata_engine.qa.validators import check_line, script_problems
from maata_engine.text import scene_prompt as sp

META = VideoMeta("How kites fly", "Sky Club", "A short talk about kites.", ((0.0, "Intro"), (61.5, "Wind")), ("kites",),
                 (("S1", 0.7), ("S2", 0.3)))
BRIEF_V1 = Brief(1, META, "Kites and wind", "casual; the host says మీరు to viewers",
                 (SpeakerNote("S1", "Ravi", "male", "host", "polite", (("S2", "familiar"),)),),
                 (GlossaryEntry("kite", "కైట్", True, "as in English"),), ("Sky Club",), ("go fly a kite",), "lakhs",
                 (("kyte", "kite"),))


def test_prompt_version_and_hash_are_pinned():
    # Changing the prompt, an example, the schema or the message layout moves the hash. When it does on purpose, bump
    # SHOTS_VERSION and update both pins here: the hash keys the line cache, so old lines are never served.
    assert (sp.SHOTS_VERSION, sp.PROMPT_HASH) == ("scene-v2", "d548212e604d")
    assert sp.prompt_hash() == sp.PROMPT_HASH and sp.brief_hash() == sp.BRIEF_HASH
    assert sp.review_hash() == sp.REVIEW_HASH != sp.PROMPT_HASH  # the review's prompt is its own


def test_prompt_hash_is_the_same_in_every_process():
    code = "from maata_engine.text.scene_prompt import PROMPT_HASH, BRIEF_HASH; print(PROMPT_HASH, BRIEF_HASH)"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip() for seed in ("1", "2")}
    assert outs == {f"{sp.PROMPT_HASH} {sp.BRIEF_HASH}"}


@pytest.mark.parametrize("name", ["SYSTEM_HEAD", "SYSTEM_TAIL", "BRIEF_HEADER", "SHOTS_VERSION"])
def test_prompt_hash_follows_the_fixed_text(monkeypatch, name):
    monkeypatch.setattr(sp, name, getattr(sp, name) + " ")
    assert sp.prompt_hash() != sp.PROMPT_HASH


def test_prompt_hash_follows_the_schema_and_the_message_layout(monkeypatch):
    monkeypatch.setitem(sp.SCENE_SCHEMA, "required", ["lines", "glossary_additions"])
    assert sp.prompt_hash() != sp.PROMPT_HASH
    monkeypatch.undo()
    monkeypatch.setattr(sp, "_json", lambda obj: json.dumps(obj, ensure_ascii=False))  # a layout change
    assert sp.prompt_hash() != sp.PROMPT_HASH


def test_system_prompt_is_byte_identical_for_a_brief_and_the_brief_is_the_only_part_that_changes():
    a, b = sp.system_prompt(BRIEF_V1), sp.system_prompt(Brief(**{f: getattr(BRIEF_V1, f) for f in BRIEF_V1.__slots__}))
    assert a == b
    v0 = sp.system_prompt(Brief(0, META))
    assert v0 != a and v0.startswith(sp.SYSTEM_HEAD) and a.startswith(sp.SYSTEM_HEAD)
    assert v0.endswith(sp.SYSTEM_TAIL) and a.endswith(sp.SYSTEM_TAIL)
    assert "Kites and wind" in a and "Kites and wind" not in v0  # v0 is metadata only
    assert '"talk_share":0.7' in v0 and "A short talk about kites." in v0


def test_the_system_prompt_has_every_section_of_the_design():
    text = sp.system_prompt(Brief(0, META))
    for heading in ("STYLE (colloquial", "STYLE (formal", "SCRIPT CONTRACT", "LENGTH CONTRACT", "EXAMPLES", "VIDEO BRIEF",
                    "JSON CONTRACT"):
        assert heading in text
    order = [text.index(h) for h in ("STYLE (colloquial", "SCRIPT CONTRACT", "LENGTH CONTRACT", "EXAMPLES", "VIDEO BRIEF",
                                     "JSON CONTRACT")]
    assert order == sorted(order)
    for call in ('"scene"', '"fit"', '"retranslate"', '"rephrase"'):
        assert call in sp.SYSTEM_TAIL


def test_a_long_description_is_trimmed():
    long = VideoMeta("t", description="x" * 5000)
    assert len(sp.brief_dict(Brief(0, long))["video"]["description"]) == sp.DESCRIPTION_MAX


def test_about_sixteen_original_examples_cover_the_contract():
    shots = sp.shot_lines()
    assert 15 <= len({line["en"] for line, _, _ in shots}) <= 18
    replies = [reply for _, reply, _ in shots]
    assert any(not r["full"]["english"] for r in replies)                    # lines with no English at all
    assert sum(len(r["full"]["english"]) >= 4 for r in replies) >= 2         # technical lines
    assert any("concise" in r and "very_concise" in r for r in replies)
    assert any("fuller" in r for r in replies)
    assert any("pieces" in r for r in replies) and any(r.get("unfinished") for r in replies)
    assert any(style == "formal" for _, _, style in shots)
    assert any(r["delivery"].get("question") for r in replies)


@pytest.mark.parametrize("k", range(len(sp.SHOTS)))
def test_every_example_keeps_the_contract_it_teaches(k):
    line, reply, _ = sp.shot_lines()[k]
    assert validate({"lines": [reply]}, sp.SCENE_SCHEMA) == []
    spec = LineSpec(line["id"], "S1", line["en"], line.get("start", 0.0), line.get("end", 5.0), 5.0, 20.0,
                    tuple(line["want"]), breaks=tuple(line.get("breaks", ())), cut_off=line.get("cut_off", False))
    got, why = check_line(reply, spec)
    assert why == [] and got is not None
    assert got.flags == ()  # no repairs, and no negation, number, question or glossary suspicion
    for tier in line["want"]:
        assert tier in got.tiers and script_problems(got.tiers[tier].spoken) == []
        assert len(got.tiers[tier].english) == len(reply[tier]["english"])  # every entry points at a real word


def test_short_tier_examples_are_visibly_shorter_than_their_english():
    # Models shorten only with clearly short demos (research 05 C2).
    for line, reply, _ in sp.shot_lines():
        if "very_concise" in reply:
            assert len(reply["very_concise"]["spoken"].split()) < 0.5 * len(line["en"].split())


@pytest.mark.parametrize("schema", [sp.SCENE_SCHEMA, sp.BRIEF_SCHEMA, sp.REVIEW_SCHEMA])
def test_schemas_are_draft07_the_local_validator_fully_checks(schema):
    check_schema(schema)
    assert '"format"' not in json.dumps(schema)
    json.dumps(schema)  # serialisable for --json-schema


def test_the_scene_schema_is_the_designs():
    line = sp.SCENE_SCHEMA["properties"]["lines"]["items"]
    assert set(line["properties"]) == {"id", "full", "fuller", "concise", "very_concise", "pieces", "moved", "unfinished",
                                       "delivery"}
    assert line["required"] == ["id", "full", "delivery"]
    assert sp.SCENE_SCHEMA["definitions"]["wording"]["required"] == ["spoken", "english"]


SPEC = LineSpec(412, "S2", "The wind lifts the kite.", 603.1, 609.8, 5.93, 33.4, ("full", "concise"), "S1", (606.2,),
                False, {"energy_rel": 1.3, "rate_rel": 0.9, "pitch_rel_st": 1.234})


def test_the_user_message_is_the_designs():
    req = SceneRequest(7, (SPEC,), "scene", (("Hello.", "నమస్కారం."), ("Before a seek.", None)), ("Next one.", "And more."))
    msg = json.loads(sp.user_message(req, "colloquial", [GlossaryEntry("kite", "కైట్")]))
    assert list(msg) == ["scene", "call", "style", "glossary_recent", "context_before", "context_after_en", "lines"]
    assert msg["glossary_recent"] == [{"term": "kite", "spoken": "కైట్"}]
    assert msg["context_before"] == [{"en": "Hello.", "te": "నమస్కారం."}, {"en": "Before a seek."}]  # no Telugu yet
    assert msg["lines"] == [{"id": 412, "speaker": "S2", "to": "S1", "start": 603.1, "end": 609.8, "speech_s": 5.93,
                             "target_aksharas": 33, "want": ["full", "concise"], "breaks": [606.2], "cut_off": False,
                             "delivery_hint": {"energy_rel": 1.3, "rate_rel": 0.9, "pitch_rel_st": 1.23},
                             "en": "The wind lifts the kite."}]


def test_cached_lines_of_the_scene_travel_as_context_only():
    req = SceneRequest(3, (SPEC,))
    done = LineSpec(411, "S1", "Look up.", 600.0, 603.0, 2.5, 12.0)
    msg = json.loads(sp.user_message(req, done=[(done, "పైకి చూడండి.")]))
    assert msg["context_done"] == [{"id": 411, "start": 600.0, "en": "Look up.", "te": "పైకి చూడండి."}]
    assert [line["id"] for line in msg["lines"]] == [412]


@pytest.mark.parametrize("call,fields", [("fit", {"current": "ఇది.", "overflow_aksharas": 6}),
                                         ("retranslate", {"missing": ["wind"]}),
                                         ("rephrase", {"failing": "ఇది.", "finding": {"cer": 0.4}})])
def test_each_call_type_carries_its_own_fields(call, fields):
    spec = LineSpec(1, "S1", "x", 0, 1, 1, 5, current="ఇది.", overflow=5.6, missing=("wind",), finding={"cer": 0.4},
                    problems=("missing from the reply",))
    line = json.loads(sp.user_message(SceneRequest(1, (spec,), call)))["lines"][0]
    assert {k: line[k] for k in fields} == fields
    others = {"current", "overflow_aksharas", "missing", "failing", "finding"} - set(fields)
    assert not others & set(line)
    assert line["problems"] == ["missing from the reply"]


def test_the_brief_message_carries_metadata_transcript_and_the_previous_brief():
    msg = json.loads(sp.brief_message(META, [("S1", "Kites need wind."), ("S2", "Right.")]))
    assert msg["call"] == "brief" and msg["video"]["title"] == "How kites fly" and "previous" not in msg
    assert msg["video"]["speakers"] == [{"id": "S1", "talk_share": 0.7}, {"id": "S2", "talk_share": 0.3}]
    assert msg["transcript"] == [{"speaker": "S1", "en": "Kites need wind."}, {"speaker": "S2", "en": "Right."}]
    again = json.loads(sp.brief_message(META, [], BRIEF_V1))
    assert again["previous"]["glossary"] == [{"term": "kite", "spoken": "కైట్", "keep_english": True,
                                              "note": "as in English"}]
    assert again["previous"]["speakers"][0]["address"] == [{"to": "S2", "form": "familiar"}]


def test_brief_v1_renders_what_the_scene_calls_need():
    d = sp.brief_dict(BRIEF_V1)
    assert d["topic"] == "Kites and wind" and d["numbers"] == "lakhs"
    assert d["asr_fixes"] == [{"heard": "kyte", "meant": "kite"}]
    assert d["speakers"][0]["gender"] == "male"
    assert "idioms" in d and "entities" in d
    assert set(sp.brief_dict(Brief(1, META))) == {"video"}  # empty fields are left out


# ---- the coverage review's call (§4.2, §4.6) ----------------------------------------------------------------------------
def test_the_review_message_has_english_and_the_chosen_telugu_with_the_scenes_context():
    req = SceneRequest(7, (SPEC,), "scene", (("Hello.", "నమస్కారం."), ("Before a seek.", None)), ("Next one.",))
    cut = LineSpec(413, "S1", "So what I", 610.0, 611.0, 0.9, 5.0, cut_off=True)
    msg = json.loads(sp.review_message(req, [(SPEC, Wording("గాలి కైట్ని పైకి లేపుతుంది.", ((1, "kite"),))),
                                             (cut, Wording("అంటే నేను…"))]))
    assert list(msg) == ["scene", "call", "context_before", "lines", "context_after_en"] and msg["call"] == "review"
    assert msg["context_before"] == [{"en": "Hello.", "te": "నమస్కారం."}, {"en": "Before a seek."}]
    assert msg["lines"] == [{"id": 412, "en": "The wind lifts the kite.", "te": "గాలి కైట్ని పైకి లేపుతుంది."},
                            {"id": 413, "en": "So what I", "te": "అంటే నేను…", "cut_off": True}]
    assert msg["context_after_en"] == ["Next one."]


def test_the_review_system_prompt_is_fixed_and_names_every_class():
    assert "VIDEO BRIEF" not in sp.REVIEW_SYSTEM  # the same for every video: one prompt cache on the review model
    for cls in ('"C"', '"m"', '"P"', '"E"', '"missing"', '"added"', '"error"', '"cut_off"', '"context_before"'):
        assert cls in sp.REVIEW_SYSTEM
    item = sp.REVIEW_SCHEMA["properties"]["lines"]["items"]
    assert item["required"] == ["id", "class", "missing", "added", "error"]
    assert validate({"lines": [{"id": 1, "class": "P", "missing": ["wind"], "added": [], "error": "none"}]},
                    sp.REVIEW_SCHEMA) == []
    assert validate({"lines": [{"id": 1, "class": "X", "missing": [], "added": [], "error": "none"}]}, sp.REVIEW_SCHEMA)
