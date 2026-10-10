# Separate OmniVoice runtime

`scripts/setup-omnivoice.sh` installs this frozen environment at
`~/Library/Application Support/Maata/runtimes/omnivoice/`. The existing engine
environment retains its dependencies. The package name in this copied lock is
unchanged from the tested evaluation environment so the exact lock remains usable.

Setup verifies the pinned files in `models.lock.json`, reuses verified evaluation
cache files if available, and otherwise uses Maata's existing pinned downloader.
`MAATA_SETUP_OFFLINE=1` also requires cached Python dependencies. The model/codec
source and license are recorded in the manifest; code is Apache-2.0 and model
weights are CC-BY-NC according to the upstream model card.

Production uses the user-selected automatic Telugu configuration: 32 steps, no
reference audio, instructions, duration or speed override, and no text normalization
inside the model. The engine's existing Telugu normalizer still applies. Seed
20261010 is reset before every generation for reproducibility; this is not a promise
of consistent speaker identity across different sentences. Audio never conditions
on English source voices.

A trial that conditioned speech on the selected synthetic B conversation produced
a short output with a possible missing final clause. That trial was rejected as the
production default. Existing reference WAVs/prompt caches are preserved locally but
are not read by automatic mode. The historical `create_seed.py` helper reconstructs
the earlier audition artifact only; setup never calls it.

The worker is persistent, uses Python isolated mode, verifies runtime/model pins,
and runs with networking denied. Generated PCM is cached by model, automatic mode,
seed and processing policy. PerTh is applied after timing changes. All audio remains
outside Git; no weights are bundled with the app.
