import json

from maata_engine.backends.apple import MLXTranslator


def test_translategemma_rope_override(tmp_path):
    cfg = {"model_type": "gemma3", "text_config": {"hidden_size": 2560, "rope_scaling": None,
           "rope_parameters": {"full_attention": {"rope_type": "linear", "factor": 8.0}, "sliding_attention": {"rope_type": "default"}}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    o = MLXTranslator.rope_override(tmp_path)
    assert o["text_config"]["rope_scaling"] == {"type": "linear", "factor": 8.0}
    assert o["text_config"]["hidden_size"] == 2560  # the rest of text_config is kept


def test_no_override_when_already_scaled(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"rope_scaling": {"type": "linear", "factor": 8}}}))
    assert MLXTranslator.rope_override(tmp_path) == {}
