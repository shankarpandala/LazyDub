# Original Telugu male/female profile screen

Six single takes through the production adapter, with exact `male` / `female` voice-design instructions. No source reference, pitch shift, rate control, retries or seed search. Model and source revisions are in results.json. All media stays outside Git.

47 adapter/worker CPU tests passed. All six waveform checks pass and PerTh detect scores are 1.0. This does not certify naturalness or perceived voice gender.

**Listening follow-up:** the female names/numbers sample's local ASR omits ఫైళ్లు (files); the male ASR retains it. The user listened to all six samples and replied **“Both match and all words are audible.”** That sample concern was not confirmed by listening. This accepts the bounded samples, not every future utterance or whole-video quality.

The original pitch measurements in results.json used the earlier source helper thresholds. Current CPU-only measurements and exact source hash are separate in current-pitch-diagnostics.json. They are diagnostics, not voice-gender or identity acceptance. Voice design upstream is trained on Chinese and English; Telugu generalization is not guaranteed.

Run the saved harness only on an idle machine from this repository, with outbound network denied. It refuses to overwrite an existing result directory. The repository root is resolved from the script location; no packages or downloads are required. Synthesis and local ASR are separate processes:

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' engine/.venv/bin/python docs/spikes/results/model-evaluation-2026-10-10/voice-mapping/profiles-smoke.py synth
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' engine/.venv/bin/python docs/spikes/results/model-evaluation-2026-10-10/voice-mapping/profiles-smoke.py asr
```

Official pinned source: https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/docs/voice-design.md
