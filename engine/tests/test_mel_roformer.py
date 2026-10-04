"""The vendored MLX Mel-Band RoFormer (OFFLINE-RENDER §2.14, §9; ADR-020), with no real weights: a tiny model's forward
pass (shapes, and the precision split), the STFT round trip, `sanitize` and a strict load on a tiny checkpoint whose keys
follow the converted checkpoint's layout, and the pinned config.json building the full config. Skipped where mlx can't
be imported."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from maata_engine.separation import mel_roformer as mr  # noqa: E402
from maata_engine.separation.dsp import istft as dsp_istft  # noqa: E402
from maata_engine.separation.dsp import stft as dsp_stft  # noqa: E402

# The pinned config.json (mlx-community/mel-roformer-kim-vocal-2-mlx@64cbfcb0, 833 bytes), its fields.
PINNED_CONFIG = {
    "_class": "MelRoFormerConfig", "model_type": "mel_band_roformer", "checkpoint_family": "kim_vocal_2", "dim": 384,
    "depth": 6, "heads": 8, "dim_head": 64, "num_bands": 60, "num_stems": 1, "ff_mult": 4, "mlp_expansion_factor": 4,
    "mask_estimator_depth": 2, "n_fft": 2048, "hop_length": 441, "win_length": 2048, "sample_rate": 44100,
    "chunk_size": 352800, "num_overlap": 2, "_dtype": "bfloat16", "_mlx_version": "0.31.0", "_tool_version": "8380ab8",
    "_source_input": "MelBandRoformer.ckpt"}
# The pinned checkpoint's band-split input sizes, band by band (read from its safetensors header: the shapes of
# band_split.to_features.<i>.1.weight), and its tensor and value counts.
REAL_BANDS = [28, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 36, 36, 36, 40, 40, 44, 52, 52, 52,
              60, 64, 68, 76, 80, 80, 88, 96, 104, 112, 116, 124, 132, 144, 156, 164, 176, 188, 200, 216, 228, 244, 264,
              284, 304, 320, 344, 372, 396, 420, 452, 488, 520]
REAL_TENSORS, REAL_VALUES = 708, 228_203_172

TINY = dict(dim=16, depth=2, heads=2, dim_head=8, num_bands=6, n_fft=256, hop_length=64, win_length=256,
            chunk_size=4096)


def checkpoint_layout(c: mr.MelRoFormerConfig, band_dims: list[int]) -> dict[str, tuple[int, ...]]:
    """Tensor names and shapes as the converted checkpoint has them (PyTorch's names: RMSNorm scales as `gamma`, the
    output projection inside a Sequential, the mask MLP's linears at Sequential indices 0, 2, 4, the RoPE frequencies
    stored), built from the config alone, not from the MLX model."""
    out: dict[str, tuple[int, ...]] = {}
    for i, bd in enumerate(band_dims):
        out |= {f"band_split.to_features.{i}.0.gamma": (bd,), f"band_split.to_features.{i}.1.weight": (c.dim, bd),
                f"band_split.to_features.{i}.1.bias": (c.dim,)}
    inner, ff = c.heads * c.dim_head, c.dim * c.ff_mult
    for d in range(c.depth):
        for axis in (0, 1):
            p = f"layers.{d}.{axis}"
            out |= {f"{p}.layers.0.0.norm.gamma": (c.dim,), f"{p}.layers.0.0.rotary_embed.freqs": (c.dim_head // 2,),
                    f"{p}.layers.0.0.to_gates.weight": (c.heads, c.dim), f"{p}.layers.0.0.to_gates.bias": (c.heads,),
                    f"{p}.layers.0.0.to_q.weight": (inner, c.dim), f"{p}.layers.0.0.to_k.weight": (inner, c.dim),
                    f"{p}.layers.0.0.to_v.weight": (inner, c.dim), f"{p}.layers.0.0.to_out.0.weight": (c.dim, inner),
                    f"{p}.layers.0.1.net.0.gamma": (c.dim,), f"{p}.layers.0.1.net.1.weight": (ff, c.dim),
                    f"{p}.layers.0.1.net.1.bias": (ff,), f"{p}.layers.0.1.net.4.weight": (c.dim, ff),
                    f"{p}.layers.0.1.net.4.bias": (c.dim,), f"{p}.norm.gamma": (c.dim,)}
    hidden = c.mlp_hidden
    for i, bd in enumerate(band_dims):
        sizes = [(hidden, c.dim)] + [(hidden, hidden)] * (c.mask_estimator_depth - 1) + [(bd * 2, hidden)]
        for n, (o, k) in enumerate(sizes):
            out |= {f"mask_estimators.0.to_freqs.{i}.0.{2 * n}.weight": (o, k),
                    f"mask_estimators.0.to_freqs.{i}.0.{2 * n}.bias": (o,)}
    return out


def write_checkpoint(folder, config: dict, tensors: dict[str, tuple[int, ...]]) -> None:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    weights = {k: mx.array(rng.standard_normal(s).astype(np.float32) * 0.1).astype(mx.bfloat16)
               for k, s in tensors.items()}
    mx.save_safetensors(str(folder / "model.safetensors"), weights)


def test_the_pinned_config_builds_the_full_model_config_and_its_bands_match_the_checkpoint():
    c = mr.MelRoFormerConfig.from_dict(PINNED_CONFIG)
    assert (c.dim, c.depth, c.heads, c.dim_head, c.num_bands) == (384, 6, 8, 64, 60)
    assert (c.n_fft, c.hop_length, c.win_length, c.sample_rate, c.chunk_size, c.num_overlap) == \
        (2048, 441, 2048, 44100, 352800, 2)
    assert c.mask_estimator_depth == 2 and c.checkpoint_family == "kim_vocal_2" and c == mr.MelRoFormerConfig.kim_vocal_2()
    # the vendored filterbank gives the checkpoint's 60 band sizes, and the layout its tensor and value counts
    bands = mr.MelFilterbank(c).band_dims
    assert bands == REAL_BANDS
    layout = checkpoint_layout(c, bands)
    assert len(layout) == REAL_TENSORS and sum(int(np.prod(s)) for s in layout.values()) == REAL_VALUES


def test_a_tiny_model_separates_a_batch_with_its_shapes_in_bf16_and_returns_float32():
    c = mr.MelRoFormerConfig(**TINY)
    model = mr.MelRoFormer(c)
    seen = []
    real = mr.Transformer.__call__

    def spy(self, x):
        seen.append(x.dtype)
        return real(self, x)

    mr.Transformer.__call__ = spy
    try:
        x = mx.array(np.random.default_rng(1).standard_normal((2, 2, c.chunk_size)).astype(np.float32) * 0.1)
        y = model(x)
        mx.eval(y)
    finally:
        mr.Transformer.__call__ = real
    assert y.shape == (2, 2, c.chunk_size) and y.dtype == mx.float32
    assert seen and set(seen) == {mx.bfloat16}  # the band-split input cast to the compute dtype (§2.14)
    assert np.isfinite(np.array(y)).all()


@pytest.mark.parametrize("n_fft,hop", [(2048, 441), (256, 64)])
def test_the_istft_of_the_stft_gives_the_signal_back(n_fft, hop):
    n = 441 * 64  # a whole number of hops, as the model's chunk is (352,800 = 800 x 441)
    x = np.random.default_rng(2).standard_normal((1, 2, n)).astype(np.float32)
    window = mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32))  # periodic Hann, as the model's
    real, imag = mr.stft(mx.array(x), n_fft, hop, window)
    assert real.shape == (1, 2, n_fft // 2 + 1, 1 + n // hop) and real.dtype == mx.float32
    back = mr.istft(real, imag, n_fft, hop, window, n)
    assert np.abs(np.array(back) - x).max() < 1e-4
    # the dsp functions themselves (one signal), with the window given by name
    spec = dsp_stft(mx.array(x[0, 0]), n_fft=n_fft, hop_length=hop, window="hann")
    assert spec.shape == (1 + n // hop, n_fft // 2 + 1)
    again = dsp_istft(dsp_stft(mx.array(x[0, 0]), n_fft=n_fft, hop_length=hop, window=window).transpose(1, 0),
                      hop_length=hop, win_length=n_fft, window=window, center=True, normalized=True)
    assert np.abs(np.array(again)[:n] - x[0, 0]).max() < 1e-4


def test_sanitize_maps_the_checkpoints_names_and_a_strict_load_takes_every_weight(tmp_path):
    c = mr.MelRoFormerConfig(**TINY, checkpoint_family="kim_vocal_2")
    layout = checkpoint_layout(c, mr.MelFilterbank(c).band_dims)
    # one layer's projections packed as an older converter left them: split again by `sanitize`
    p = "layers.1.0.layers.0.0"
    inner = c.heads * c.dim_head
    for k in ("to_q", "to_k", "to_v"):
        del layout[f"{p}.{k}.weight"]
    layout[f"{p}.to_qkv.weight"] = (3 * inner, c.dim)
    config = {**{k: getattr(c, k) for k in TINY}, "checkpoint_family": "kim_vocal_2", "_dtype": "bfloat16"}
    write_checkpoint(tmp_path / "ok", config, layout)
    model = mr.MelRoFormer.from_pretrained(tmp_path / "ok", mx.bfloat16)
    assert model.config.depth == 2 and model.dtype == mx.bfloat16
    loaded = dict(mx.load(str(tmp_path / "ok" / "model.safetensors")))
    q = model.layers[1][0].layers[0][0].to_q.weight
    assert q.dtype == mx.bfloat16 and np.array_equal(np.array(q.astype(mx.float32)),
                                                      np.array(loaded[f"{p}.to_qkv.weight"][:inner].astype(mx.float32)))
    norm = model.band_split.to_features[3][0].weight
    assert np.array_equal(np.array(norm.astype(mx.float32)),
                          np.array(loaded["band_split.to_features.3.0.gamma"].astype(mx.float32)))
    mask = model.mask_estimators[0].to_freqs[2][2][0].bias  # the third linear: Sequential index 4
    assert np.array_equal(np.array(mask.astype(mx.float32)),
                          np.array(loaded["mask_estimators.0.to_freqs.2.0.4.bias"].astype(mx.float32)))
    x = mx.array(np.random.default_rng(3).standard_normal((1, 2, 4096)).astype(np.float32) * 0.1)
    assert model(x).shape == (1, 2, 4096)
    # a weight missing from the file fails the load (upstream's strict=False would leave it random)
    dropped = {k: s for k, s in layout.items() if k != "layers.0.1.layers.0.1.net.4.bias"}
    write_checkpoint(tmp_path / "missing", config, dropped)
    with pytest.raises(ValueError, match="Missing"):
        mr.MelRoFormer.from_pretrained(tmp_path / "missing")
    # and so does a weight `sanitize` doesn't know
    write_checkpoint(tmp_path / "extra", config, {**layout, "band_split.to_features.0.2.weight": (4, 4)})
    with pytest.raises(ValueError, match="not in model"):
        mr.MelRoFormer.from_pretrained(tmp_path / "extra")
    # the folder's own two files, nothing else: no hub lookup
    (tmp_path / "ok" / "model.safetensors").rename(tmp_path / "ok" / "weights.safetensors")
    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        mr.MelRoFormer.from_pretrained(tmp_path / "ok")


def test_the_apple_separator_loads_its_folder_on_first_use_and_lets_go(tmp_path):
    from maata_engine.backends.apple import MLXMelRoFormerSeparator

    c = mr.MelRoFormerConfig(dim=8, depth=1, heads=2, dim_head=4, num_bands=4, mask_estimator_depth=1)
    assert (c.sample_rate, c.chunk_size, c.n_fft, c.hop_length) == (44_100, 352_800, 2048, 441)  # the real grid
    config = {k: getattr(c, k) for k in ("dim", "depth", "heads", "dim_head", "num_bands", "mask_estimator_depth",
                                         "n_fft", "hop_length", "win_length", "sample_rate", "chunk_size")}
    write_checkpoint(tmp_path / "sep", config, checkpoint_layout(c, mr.MelFilterbank(c).band_dims))
    sep = MLXMelRoFormerSeparator(tmp_path / "sep")
    assert sep._model is None and sep.dtype == "bfloat16"  # nothing loaded until it is used
    before = mx.set_cache_limit(1 << 40)
    mx.set_cache_limit(before)
    batch = (0.1 * np.random.default_rng(5).standard_normal((2, 2, sep.chunk))).astype(np.float32)
    out = sep.vocals(batch)
    assert out.shape == batch.shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert sep._model is not None and sep._model.dtype == mx.bfloat16
    assert mx.set_cache_limit(before) == MLXMelRoFormerSeparator.CACHE_LIMIT  # bounded while loaded
    mx.set_cache_limit(MLXMelRoFormerSeparator.CACHE_LIMIT)
    sep.release()
    assert sep._model is None and mx.set_cache_limit(before) == before  # restored
    sep.release()  # twice: nothing to do
    # a model folder for another grid is refused rather than chunked wrongly
    write_checkpoint(tmp_path / "other", {**config, "chunk_size": 4096}, checkpoint_layout(c, mr.MelFilterbank(c).band_dims))
    with pytest.raises(ValueError, match="352800"):
        MLXMelRoFormerSeparator(tmp_path / "other").vocals(batch)


def test_the_apple_separator_says_which_model_files_its_folder_lacks(tmp_path):
    """A models folder filled before the separator was pinned lacks its folder: the job asks for a fetch (render)."""
    from maata_engine.backends.apple import MLXMelRoFormerSeparator

    sep = MLXMelRoFormerSeparator(tmp_path / "mel-roformer-kim-vocal-2-mlx")
    assert sep.missing() == ["config.json", "model.safetensors"]
    (tmp_path / "mel-roformer-kim-vocal-2-mlx").mkdir()
    (tmp_path / "mel-roformer-kim-vocal-2-mlx" / "config.json").write_text("{}")
    assert sep.missing() == ["model.safetensors"] and sep._model is None
    (tmp_path / "mel-roformer-kim-vocal-2-mlx" / "model.safetensors").write_bytes(b"")
    assert sep.missing() == []
