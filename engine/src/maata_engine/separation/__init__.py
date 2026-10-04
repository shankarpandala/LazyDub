"""The separator under the dub's background sound (OFFLINE-RENDER §2.14, ADR-020): Mel-Band RoFormer "Kim Vocal 2" on
MLX, vendored from Blaizzy/mlx-audio at a8546e64ac5fdc7ace03e4b75dd2daad444a614e (`mel_roformer.py`: the model and its
config; `dsp.py`: the STFT, inverse STFT and mel filterbank it uses). Both import mlx: only the apple backend loads them.
The chunking over a whole file is `maata_engine.separate` (numpy only)."""
