"""N takes of a line in one batched T3 decode (ARCHITECTURE §3.8), checked on the CPU with small stand-in modules in place
of Chatterbox's T3 (no weights): 2N rows give the same takes as N single decodes with the same seeds, the end token is
read back every few steps rather than every step, each take is cut at its end, and a take that runs to its cap has
failed. S3Gen's flow runs only for the take that is vocoded."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.generation.logits_process")

from maata_engine.backends import torch_common  # noqa: E402
from maata_engine.backends.torch_common import ChatterboxTeluguTTS, ChatterboxVoice, MelTake  # noqa: E402

VOCAB, DIM, SOS, EOS = 12, 8, 10, 11
TEXT = torch.tensor([[1, 5, 6, 7, 2], [1, 5, 6, 7, 2]])  # the line's text tokens, twice (conditional, unconditional)


class TinyT3:
    """T3's embeddings and `prepare_input_embeds`, shaped like the fork's: row 1 is the unconditional one (text zeroed)."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(0)
        self.hp = SimpleNamespace(start_speech_token=SOS, stop_speech_token=EOS, start_text_token=1, stop_text_token=2)
        self.speech_emb = torch.nn.Embedding(VOCAB, DIM)
        self.text_emb = torch.nn.Embedding(20, DIM)
        self.pos = torch.nn.Embedding(1100, DIM)
        for m in (self.speech_emb, self.text_emb, self.pos):
            torch.nn.init.normal_(m.weight, generator=g)
        self.speech_pos_emb = SimpleNamespace(get_fixed_embedding=lambda i: self.pos.weight[i].view(1, 1, DIM))

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight):
        text = self.text_emb(text_tokens)
        if cfg_weight > 0.0:
            text = torch.cat([text[:1], torch.zeros_like(text[1:2])])
        return torch.cat([text, self.speech_emb(speech_tokens)], dim=1), 0


class TinyBackend:
    """T3's transformer and speech head: each row's logits depend on everything that row was fed (a running sum, its
    "cache") and on nothing in the other rows; computed elementwise, so a row's numbers don't depend on the batch."""

    def __init__(self, eos_bias: float = 0.0) -> None:
        g = torch.Generator().manual_seed(1)
        self.w = torch.randn(DIM, VOCAB, generator=g)
        self.eos_bias, self.calls, self.rows = eos_bias, 0, []

    def __call__(self, inputs_embeds, past_key_values=None, use_cache=True, output_hidden_states=True, return_dict=True):
        self.calls += 1
        self.rows.append(inputs_embeds.shape[0])
        state = inputs_embeds.sum(dim=1) + (0 if past_key_values is None else past_key_values)
        logits = 3.0 * (torch.tanh(state).unsqueeze(-1) * self.w).sum(dim=1)
        logits[:, EOS] += self.eos_bias
        logits[:, SOS] = -1e4  # never sampled here, so one in a take would be the prompt's left in
        return SimpleNamespace(logits=logits.unsqueeze(1), past_key_values=state)


def tiny_tts(eos_bias: float = 0.0) -> ChatterboxTeluguTTS:
    tts = object.__new__(ChatterboxTeluguTTS)  # skip loading weights
    tts._torch, tts._t3_dtype, tts._backend, tts.cfm_steps, tts.cfg_weight = torch, None, TinyBackend(eos_bias), 6, 0.5
    tts.model = SimpleNamespace(t3=TinyT3(), conds=SimpleNamespace(t3=None), device="cpu")
    return tts


def decode(tts, n=1, seeds=None, cap=200, cfg=0.5):
    """The takes of one decode; the steps it ran go in `tts.steps`."""
    with torch.inference_mode():
        takes, tts.steps = tts._t3_tokens(TEXT, cap, cfg, n, seeds)
    return takes


def test_n_takes_in_one_batched_decode_equal_n_single_decodes_with_the_same_seeds():
    tts = tiny_tts()
    batched = decode(tts, n=3, seeds=[3, 4, 5])
    assert tts._backend.rows[0] == 6  # 2N rows: three conditional, three unconditional
    singles = [decode(tiny_tts(), seeds=[s])[0] for s in (3, 4, 5)]
    for (tokens, capped), (one, one_capped) in zip(batched, singles):
        assert torch.equal(tokens, one) and capped == one_capped is False
        assert EOS not in tokens.tolist() and SOS not in tokens.tolist()  # cut at its end token, no start token
    assert len({tuple(t.tolist()) for t, _ in batched}) == 3  # three different takes, not one copied


def test_the_end_token_is_read_back_every_few_steps_and_each_take_is_cut_at_its_own():
    tts = tiny_tts()
    reads = []
    real = torch.Tensor.__bool__

    def counting(self):
        reads.append(1)
        return real(self)

    torch.Tensor.__bool__ = counting
    try:
        (tokens, capped), = decode(tts, seeds=[7])
    finally:
        torch.Tensor.__bool__ = real
    steps = len(tokens) + 1                                   # its tokens, then its end token
    ran = -(-steps // torch_common.EOS_CHECK) * torch_common.EOS_CHECK  # to the next check
    assert tts._backend.calls == ran                          # the prefill, and one pass per step but the last
    assert tts.steps == ran                                   # what the decode's time went on, past the end too
    assert not capped and len(reads) <= ran // torch_common.EOS_CHECK + 1  # not one read per step


def test_a_take_that_runs_to_its_cap_without_ending_has_failed():
    tts = tiny_tts(eos_bias=-1e4)                             # never samples the end token
    takes = decode(tts, n=2, seeds=[1, 2], cap=20)
    assert [c for _, c in takes] == [True, True] and all(len(t) == 20 for t, _ in takes)
    assert tts._backend.calls == tts.steps == 20               # no pass for a token past the cap

    # One take ending under the cap and one running to it, in the same batch.
    cap = 16
    short = next(s for s in range(100) if not decode(tiny_tts(), seeds=[s], cap=cap)[0][1])
    long = next(s for s in range(100) if decode(tiny_tts(), seeds=[s], cap=cap)[0][1])
    (a, a_capped), (b, b_capped) = decode(both := tiny_tts(), n=2, seeds=[short, long], cap=cap)
    assert (a_capped, b_capped) == (False, True) and len(a) < cap and len(b) == cap and both.steps == cap


def test_unseeded_takes_sample_from_the_global_generator():
    torch.manual_seed(11)
    first = decode(tiny_tts(), n=2)
    torch.manual_seed(11)
    again = decode(tiny_tts(), n=2)
    assert all(torch.equal(x, y) for (x, _), (y, _) in zip(first, again))


def test_synthesize_takes_shares_one_decode_and_flows_only_the_take_vocoded(monkeypatch):
    pytest.importorskip("chatterbox.mtl_tts")
    flows = []

    def flow(speech, ref_dict, n_cfm_timesteps, finalize):
        time.sleep(0.01)
        flows.append(int(speech.shape[-1]))
        return torch.zeros(1, 80, 2 * speech.shape[-1])

    tts = tiny_tts()
    t3 = tts.model.t3
    tts.model = SimpleNamespace(t3=t3, conds=None, device="cpu", s3gen=SimpleNamespace(flow_inference=flow),
                                tokenizer=SimpleNamespace(text_to_tokens=lambda text, language_id: torch.ones(1, 3, dtype=torch.long)))
    t3.hp.start_text_token, t3.hp.stop_text_token = 1, 2
    monkeypatch.setattr(tts, "_t3_tokens", lambda tt, max_new, cfg, n, seeds: (time.sleep(0.02), ([
        (torch.arange(0, 40), False), (torch.arange(0, 50), True)][:n], 50))[1])
    voice = ChatterboxVoice(SimpleNamespace(t3=None, gen={"ref": 1}))
    takes = tts.synthesize_takes("ఒక చిన్న వాక్యం", voice, n=2)
    assert [t.n_tokens for t in takes] == [40, 50] and [t.capped for t in takes] == [False, True]
    assert takes[0].seconds == pytest.approx(39 / 25) and all(t.mel is None and t.gen == {"ref": 1} for t in takes)
    assert takes[0].t3_s == takes[1].t3_s and 0.005 <= takes[0].t3_s < 0.1  # one decode, its time shared out
    assert [t.t3_tokens for t in takes] == [40, 50] and [t.t3_steps for t in takes] == [50, 50]  # its steps on each
    assert flows == []  # nothing flowed yet
    tts._flow(takes[0])
    assert flows == [40] and takes[0].mel is not None and takes[0].speech is None and takes[0].flow_s >= 0.01
    tts._flow(takes[0])
    assert flows == [40]  # flowed once


def test_a_take_with_no_speech_tokens_is_empty():
    take = MelTake(None, 0)
    assert take.seconds == 0.0 and not take.capped
