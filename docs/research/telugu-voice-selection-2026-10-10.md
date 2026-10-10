# Telugu voice selection — 10 October 2026

The maintainer cancelled the long-video runs because the voices sounded unnatural and Telugu pronunciation was poor.
This evaluation therefore prioritizes speech fidelity and native delivery over throughput. Cancelled jobs remain
stopped; completed exports and production caches are not evaluation inputs or overwritten outputs.

## Decision and scope

Use **OmniVoice for the Apple app's natural Telugu voice mode**. The maintainer listened to the three blind A/B pairs
and selected **Voice B (OmniVoice)** as clearly better. It runs in an isolated runtime using automatic Telugu
speech, without an English source reference or a voice-clone prompt. Automatic voice identity can vary between lines.
A trial using B as a fixed reference produced a short greeting with a possible omitted final clause; that configuration
was rejected as the default. Original evidence remains available. Long-video quality and the previous timing issues remain unproven. No cloud
audio is used.

Keep the current ASR, separator and production text defaults during this voice comparison. Test meaning acceptance
separately: a longer corrective translation must not become “complete” merely because deterministic checks pass.

## Model shortlist

| Candidate | Fit for this task | Current decision |
| --- | --- | --- |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) | Telugu is explicitly supported. Cloning, reusable voice prompts and duration controls; upstream supports MPS. Cross-language references can transfer accent, and auto voice can vary between calls. | User-selected B; isolated persistent worker, MPS model and CPU codec, default 32 steps. Automatic Telugu; no reference conditioning or promise of consistent speaker identity. |
| [Indic-Mio](https://huggingface.co/SPRINGLab/Indic-Mio) | Small 0.6B model covering Indian languages and code-mixing, with codec-based cloning. Published speed is not an M5 measurement. | Downloaded and verified, but blocked before synthesis: its codec checkpoint omits WavLM weights and four other state entries. Strict loading rejected it; no unverified embedding or silent model download. The example also writes 24k codec output as 44.1k. |
| [Bodhan Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak) | Native Telugu stock voices and code-mixed speech; different tradeoff from cloning. Access is gated. | Not fetched or tested. Access and local runtime must be established before selection. |
| [Current Chatterbox Telugu](https://huggingface.co/shankarpandala/chatterbox-telugu) | Existing local cloning and PerTh integration. Its card acknowledges a quality gap relative to upstream English. | Baseline, not an established quality winner. Builtin conditioning is not a verified native Telugu reference. |

Do not replace the Telugu fine-tune with upstream Chatterbox Turbo/Nano or V3 merely because they are newer:
the [current supported-language list](https://github.com/resemble-ai/chatterbox#supported-languages) does not establish
Telugu support for those replacements.

OmniVoice's package needs a newer Transformers than the Chatterbox runtime. Its exact source, dependency resolution
and model files are in `engine/runtimes/omnivoice/`; the original audition pins remain in
`scripts/experiments/omnivoice/`. `scripts/setup-omnivoice.sh` provisions the separate runtime and verified model
under Application Support/Maata. The existing engine environment is unchanged. Every model file uses the existing
pinned downloader, and inference runs with outbound networking denied. No automatic ASR/model downloads are permitted.

The additional human Telugu reference trial was not uniformly better: its expressive sentence had more ASR errors.
It is retained as evidence, not promoted. The separately tested synthetic reference is labelled honestly in metadata and is not used by the default mode.
The initial names/numbers trial also exposed a formatting joiner; removing it with the existing production normalizer
improved the OmniVoice roundtrip for that one seeded sentence. These are diagnostics, not native accuracy estimates.

## What to borrow from commercial dubbing

[ElevenLabs Dubbing v2](https://elevenlabs.io/blog/introducing-dubbing-v2) describes source-performance conditioning
and timing-aware translation. Its [dubbing controls](https://elevenlabs.io/docs/overview/capabilities/dubbing) expose
the tradeoff between speaker resemblance and natural target-language speech. This supports a native-voice comparison
and segment-level audition workflow. It does not disclose a drop-in local model or prove Telugu quality on this Mac.

The legacy cloning backend uses a source voice reference but does not condition each Telugu line on its matching
source performance. Its reference selector also lacks an acoustic-quality score for music/reverb. The selected
automatic mode prioritizes native delivery; source-performance and speaker-identity preservation remain separate work.

## Text screen

Six original connected lines were generated once each with Luna low, Sol 6.1 medium and Astra medium, using identical
production schemas and prompts. One blind Astra high review judged 5/6 Luna and 6/6 each stronger-model full wordings
complete; the Luna disagreement concerns how explicitly “by six” is expressed. Each actual planned Latin TTS form was
included. This is a small diagnostic, not a representative accuracy or pronunciation benchmark.

[Official model guidance](https://developers.openai.com/api/docs/models) and
[evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices) support testing a
stronger model for difficult review, but do not establish Telugu listening quality. The actual-call results and
limitations are in `docs/spikes/results/model-evaluation-2026-10-10/text/`. Production defaults remain unchanged;
the user's reported speech issue cannot be solved by claiming a text-model win.

## App integration and acceptance before another long dub

The Apple backend selects OmniVoice by default; `MAATA_TTS=chatterbox` remains an explicit developer rollback.
There is no silent fallback to the rejected voice. The worker is loaded once. Each call resets a predeclared seed of 20261010 and uses 32 steps with no duration,
speed, instructions or reference override. This seed policy improves reproducibility, not voice-identity guarantees.
Its natural PCM takes carry model/mode/seed/runtime policy identity, can be restored without synthesis, and receive
PerTh after timing changes. A completed long utterance is reported as over budget, not falsely classified as truncated.
The planner still reports timing pressure; no words are cropped to make a slot fit. Pause, timeout, shutdown and parent
exit stop the isolated worker. Missing assets fail before video preprocessing with setup instructions.

The UI explains that source voices are not cloned and automatic voice can vary between lines. Old completed dubs retain their old voices and
outputs; incompatible old voice samples are not relabelled or played as the new model. The cancelled 175-minute job's
stale running marker was persisted as paused using the existing API; all other cached files and output hashes matched.

1. Initial blind listening completed: user chose B. Listen to the actual automatic app preview for pronunciation,
   complete endings, natural rate, emotion, code-mixing and variation between lines.
2. Source-speaker cloning and a stable narrator remain separate, unvalidated modes for this model.
3. Run a short original timed scene through synthesis, timing and export. Check missing/repeated words and watermark
   after processing, plus measured source-speech coverage. No automatic slowdown or padding is a quality fix.
4. Only then expose a selected model for a short user-requested video preview, before another full queue.

The previous 128/771 early-ending flags remain unresolved. An ASR roundtrip is a useful omission diagnostic, but
neither a low character-error rate nor a valid waveform certifies native Telugu pronunciation or naturalness.

## Actual app validation

The automatic adapter generated exactly three original normalized lines without retries. Its greeting PCM is identical
to the originally selected B, and the local ASR diagnostic recovers the final clause lost by the conditioned trial.
The other two sentences use the same predeclared per-call seed rather than the original audition's different seeds;
listen to their actual app previews before extrapolating to a long video. Saved-take PCM replay was exact and all three
PerTh watermarks survived the real mastered AAC export (-16.00 LUFS, -2.26 dBTP). The median energy-voiced ending is
0.83 seconds early against synthetic authored windows. This does not resolve the older real-source underfill issue.

The native app was rebuilt and opened with OmniVoice. It did not restart the cancelled job. Completed caches and all
39 saved exports retain their hashes. During startup inspection, a separate client removed the paused entry and then
started a new video; neither action was sent by the reviewing agent. See the startup evidence for this distinction.
The live test is user work, separate from the three-utterance smoke; do not treat its unfinished state as acceptance.

Software validation passed 1,414 engine tests, 73 UI tests and the native release build. A final seven-case regression
run verifies worker timeouts, empty output and cancellation cannot silently skip speech and can resume successfully.
The app was gracefully restarted to activate that fix; all 117 captured cached-audio files survived unchanged and
only the already-running new test resumed. Full-suite and final-patch source hashes are recorded separately in
`integration-validation.json`. These tests establish software behavior, not native-speaker acceptance.

The client-started 7:55 test subsequently completed: 65 rendered lines, no skipped lines, H.264 video/AAC audio and
both subtitle streams. Its final labels are 57 C, 1 m and 7 P, with one long-timing flag. These are reported coverage
labels, not independent acoustic acceptance; the completed file must still be heard. Export loudness was -16.02 LUFS
and -1.80 dBTP with the background bed. Whole-export watermark/native listening and underfill remain unverified.
The output hash and sanitized results are in `user-test-completed.json`. It is preserved, not automatically redubbed.

## Male/female matching follow-up

The user reported that automatic voices were assigned inappropriately. ADR-029 adds explicit OmniVoice male/female
instructions and local source-pitch matching, with a per-speaker manual override. Ambiguous source evidence stops
for a choice instead of choosing a random voice. These instructions do not clone the source speaker or guarantee
stable timbre across all lines.

Six original profile samples were generated offline without retries. All waveform and PerTh checks passed. ASR
missed “ఫైళ్లు” in the female names/numbers sentence, but the user listened and confirmed **“Both match and all
words are audible.”** This resolves that sample's listening concern; it does not certify all future speech.
The upstream instruction training is English/Chinese, so this Telugu listening result is especially relevant.
See [the pinned voice-design documentation](https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/docs/voice-design.md)
and `../spikes/results/model-evaluation-2026-10-10/voice-mapping/` for evidence. Earlier underfill and full-video
listening remain separate, unresolved work.
