#!/usr/bin/env python3
"""Offline, original-text policy check with production Apple TTS; run only with the queue stopped.

Example (from the repository root, after the supervisor has stopped):
  sandbox-exec -p '(version 1)(allow default)(deny network-outbound)' \
    engine/.venv/bin/python /tmp/maata-approved-full-apple.py \
    --approval /tmp/maata-full-audio-approval-v2.json --out /tmp/maata-approved-full-apple-run

Uses a previously recorded live review of this exact original pair. No CLI, remote calls, source media, downloads,
or output files from the user's queue are used. This exercises the core RenderJob calibration/take/PCM resume path,
not queue lifecycle, native listening, ASR accuracy, or whole-video performance.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path('/Users/shankarpandala/projects/LazyDub')
sys.path[:0] = [str(ROOT / 'engine/src'), str(ROOT / 'engine/tests')]

EN = 'Restart the router only after both uploads finish.'
FULL = 'రెండు అప్లోడ్లు పూర్తయ్యాకే రౌటర్ని రీస్టార్ట్ చేయండి.'
SHORT = 'అప్లోడ్ పూర్తయ్యాక రౌటర్ని రీస్టార్ట్ చేయండి.'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_approval(path):
    from maata_engine.backends.base import Wording
    from maata_engine.qa.coverage import check_review, check_review_fallbacks
    from maata_engine.text.tenglish import latin_spoken

    doc = json.loads(path.read_text())
    full = Wording(doc['full']['spoken'], tuple(tuple(x) for x in doc['full']['english']))
    short = Wording(doc['concise']['spoken'], tuple(tuple(x) for x in doc['concise']['english']))
    assert (doc['en'], full, short) == (EN, Wording(FULL, ((4, 'restart'),)), Wording(SHORT, ((3, 'restart'),)))
    actual = doc['actual_prompt']
    assert (actual['en'], actual['te'], actual['tts']) == (EN, SHORT, latin_spoken(SHORT, short.english))
    assert actual['fallback'] == {'te': FULL, 'tts': latin_spoken(FULL, full.english)}
    verdict, missing = check_review(doc['review'], [0])
    assert not missing and verdict[0].cls in ('P', 'E') and check_review_fallbacks(doc['review'], [0]) == {0}
    return doc, full, short


class RecordedReview:
    """Replay exactly one previously obtained semantic answer; any other request fails closed."""
    def __init__(self, approval):
        self.approval = approval
        self.model = approval['provenance']['model']
        self.calls = []

    def ask(self, system, prompt, schema=None, call='text', **kw):
        from maata_engine.claude_cli import ClaudeReply, validate

        msg = json.loads(prompt)
        expected = dict(self.approval['actual_prompt'], id=0)
        assert call == 'review' and msg['lines'] == [expected] and not self.calls
        assert not validate(self.approval['review'], schema)
        self.calls.append(call)
        return ClaudeReply(text='', data=self.approval['review'], seconds=0.0, model=self.model)


def waveform(samples, sr):
    import numpy as np
    import perth
    from maata_engine.qa.take import audio_failure

    x = np.asarray(samples, np.float32)
    finite = bool(np.isfinite(x).all())
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size and finite else None
    return {'samples': len(x), 'seconds': len(x) / sr, 'finite': finite,
            'peak': float(np.max(np.abs(x))) if x.size and finite else None, 'rms': rms,
            'rms_dbfs': 20 * np.log10(max(rms, 1e-12)) if rms is not None else None,
            'fraction_abs_ge_1': float(np.mean(np.abs(x) >= 1.0)) if x.size and finite else None,
            'watermark_confidence': float(np.mean(perth.PerthImplicitWatermarker().get_watermark(
                x, sample_rate=sr, round=False))),
            'production_audio_failure': audio_failure(x)}


async def run(args, report):
    # These variables only disable network features; no model fetch is ever invoked.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    from fakes import bare_job
    from maata_engine import render
    from maata_engine.backends import load_backend
    from maata_engine.backends.base import LineResult, SceneRequest, VideoMeta
    from maata_engine.backends.claude_translator import ClaudeTranslator, brief_v0
    from maata_engine.dubber import UnitState, VoiceCost
    from maata_engine.qa.coverage import approved_full
    from maata_engine.speakers import GlobalSpeaker
    from maata_engine.text.scene_prompt import PROMPT_HASH, REVIEW_HASH
    from maata_engine.types import SourceUnit

    approval, full, short = load_approval(args.approval)
    assert approval['provenance']['prompt_hash'] == PROMPT_HASH
    assert approval['provenance']['review_hash'] == REVIEW_HASH
    report.update({'scope': __doc__, 'approval_sha256': sha(args.approval), 'approval_provenance': approval['provenance'],
                   'current_prompt_hash': PROMPT_HASH, 'current_review_hash': REVIEW_HASH,
                   'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   'working_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
                   'script_sha256': sha(Path(__file__)), 'seed': args.seed})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    b = load_backend('apple', args.models)
    report['backend'] = {'name': b.name, 'device': b.device, 'cfm_steps': b.tts.cfm_steps,
                         'cfg_weight': b.tts.cfg_weight, 'exaggeration': b.tts.exaggeration,
                         't3_dtype': str(b.tts._t3_dtype)}
    cache = args.out / 'cache'

    def make_job():
        j = bare_job(cache, seconds=60.0, tts=b.tts, tts_script='latin')
        j.b = b
        j._budget = 3.0
        # Original, single-speaker timing fixture; no source recording, cloning, ASR or diarization needed.
        j.registry.speakers['S1'] = GlobalSpeaker('S1', 'Speaker 1', None, 10.0, 10.0)
        client = RecordedReview(approval)
        j.tr = ClaudeTranslator(cache, j.video_id, brief_v0(VideoMeta('Original router instruction', 'Maata tests')),
                                cli=client)
        return j

    job = make_job()
    cost, calibration = VoiceCost(), []
    voice, _, key = await job._voice_for('S1', cost)
    start = time.perf_counter()
    count = await job._calibrate('S1', key, voice, cost, keep=calibration)
    report['calibration'] = {'seconds': time.perf_counter() - start, 'usable_takes': count,
                             'rate': job.estimator.rate(key), 'overhead': job.estimator.overhead(key)}
    assert count >= 2, 'Insufficient calibration: do not claim measured eligibility from a prior'
    calibration_path = args.out / 'calibration.json'
    calibration_path.write_text(json.dumps([[text, secs] for text, secs, _ in calibration], ensure_ascii=False, indent=2))

    full_pred, short_pred = (job.estimator.estimate(w.spoken, key) for w in (full, short))
    line = LineResult(0, {'full': full, 'concise': short}, model=approval['provenance']['model'], brief_version=0)
    st = None
    for ratio in (1.0, 1.03, 1.06, 1.09):
        speech = short_pred / ratio
        for gap in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            unit = SourceUnit(0, 'S1', 10.0, 10.0 + speech, EN)
            candidate = UnitState(unit, unit.end + gap, speech_s=speech, scene=1, line=line)
            job._index([candidate])
            chosen, fits = job._pick(candidate, line)
            if chosen == 'concise' and fits and job._absorbs(candidate, full_pred):
                st = candidate
                break
        if st is not None:
            break
    assert st is not None, 'No concise-selected, full-absorbable original slot at this calibrated pace'
    report['eligibility'] = {'speech_s': st.speech_s, 'gap_s': st.next_start - st.unit.end,
                             'full_pred_s': full_pred, 'short_pred_s': short_pred,
                             'initial_choice': job._pick(st, line)[0], 'full_absorbed': job._absorbs(st, full_pred)}
    req = SceneRequest(1, (job._line_spec(st, ('full', 'concise')),))
    answer = await job._review(req, {0: line})
    assert answer is not None and approved_full(answer[0])
    st.line = answer[0]
    job._choose(st)
    assert st.tier == 'full' and st.line.full == full
    report['recorded_review_replayed'] = job.tr.cli.calls

    synth = b.tts.synthesize_takes
    syntheses = []
    def watched(text, *pos, **kw):
        assert text == job._tts_text(full), 'A rejected short wording must never be synthesized'
        syntheses.append({'text': text, 'n': pos[3] if len(pos) > 3 else kw.get('n', 1)})
        return synth(text, *pos, **kw)
    b.tts.synthesize_takes = watched
    start = time.perf_counter()
    assert not await job._voice_line(st)
    report['voice_wall_s'] = time.perf_counter() - start
    natural = job._said[0].audio[0]
    sf.write(args.out / 'approved-full-natural.wav', natural, b.tts.sample_rate, subtype='FLOAT')
    await job._settle(render.Scene(1, [st], 0, reviewed=True))
    assert st.tier == st.take['wording'] == 'full' and job._resynths == 0
    assert job.tr.cli.calls == ['review']
    plan = job._plan(st, render._said_of(st.take), render.TimelinePlanner(job.planner.s))
    pcm_key = job._pcm_key(st, plan)
    await job._final_pcm(st, plan, pcm_key)
    pcm_path = job.render_dir / 'pcm' / f'{pcm_key}.npy'
    pcm = np.load(pcm_path).astype(np.float32)
    sf.write(args.out / 'approved-full-timed.wav', pcm, b.tts.sample_rate, subtype='FLOAT')
    report['audio'] = {'natural': waveform(natural, b.tts.sample_rate), 'timed': waveform(pcm, b.tts.sample_rate),
                       'take_failure': st.take['takes'][0]['failed'], 'candidate_failures': st.take['cost']['take_failures'],
                       'natural_wav': str(args.out / 'approved-full-natural.wav'),
                       'timed_wav': str(args.out / 'approved-full-timed.wav')}
    report['placement'] = {'plan': asdict(plan), 'flags': job.flags, 'speed_cap': job.speed_cap,
                           'effective_cap': job._slot(st).rate_cap, 'coverage': render.coverage_json(st.line.coverage),
                           'syntheses': syntheses, 'fixups': st.take['fixups']}
    assert 1.0 <= plan.rate <= job.speed_cap + 1e-9 and not plan.freeze
    assert ('long' in job.flags.get(0, [])) == plan.needs_shorter
    assert report['audio']['natural']['production_audio_failure'] is None
    assert report['audio']['timed']['production_audio_failure'] is None
    assert all(report['audio'][kind]['watermark_confidence'] >= 0.5 for kind in ('natural', 'timed'))
    before = sha(pcm_path), len(syntheses), job.flags.copy()
    job.pause()

    again = make_job()
    again.estimator.calibrate(again._key('S1'), json.loads(calibration_path.read_text()))
    restored = UnitState(st.unit, st.next_start, speech_s=st.speech_s, scene=1)
    again._index([restored])
    restored.line = again.tr.cached(again._line_spec(restored, ('full', 'concise')))
    assert restored.line is not None and approved_full(restored.line)
    again._load_takes(render._read_rows(again.render_dir / 'takes.jsonl'),
                      render._read_rows(again.render_dir / 'fixups.jsonl'))
    assert await again._voice_line(restored)
    replay = again._plan(restored, render._said_of(restored.take), render.TimelinePlanner(again.planner.s))
    assert replay == plan and again._pcm_key(restored, replay) == pcm_key
    await again._final_pcm(restored, replay, pcm_key)
    assert (sha(pcm_path), len(syntheses), again.flags) == before and again.tr.cli.calls == []
    report['resume'] = {'take_restored': True, 'pcm_identical': True, 'pcm_sha256': before[0],
                         'new_syntheses': 0, 'new_text_calls': 0, 'same_plan': True, 'same_flags': True}

    # This is a clearly labelled counterfactual CPU timing check on the REAL measured waveform, never a second
    # eligible translation or another synthesis. It proves actual overrun keeps full with visible long at the cap.
    for factor in (1.5, 2.0, 4.0, 8.0, 16.0):
        tight = max(0.05, st.take_s / (job.speed_cap * factor))
        constrained = replace(restored, unit=replace(restored.unit, end=restored.unit.start + tight),
                              next_start=restored.unit.start + tight, speech_s=tight)
        again._index([constrained])
        again._wording(constrained, again._key('S1'))
        overrun = again._plan(constrained, render._said_of(constrained.take), render.TimelinePlanner(again.planner.s))
        if overrun.needs_shorter:
            break
    again._flag_line(constrained, overrun.needs_shorter)
    assert constrained.tier == 'full' and overrun.needs_shorter and again.flags == {0: ['long']}
    assert 1.0 <= overrun.rate <= job.speed_cap + 1e-9 and not overrun.freeze
    assert again._shorter_tier(constrained, again._key('S1'), tight) is None
    report['counterfactual_tight_window'] = {'scope': 'CPU placement of measured full audio; not original eligibility',
                                            'speech_s': tight, 'tightening_factor': factor,
                                            'plan': asdict(overrun), 'flags': again.flags,
                                            'kept_tier': constrained.tier, 'new_syntheses': 0}
    b.tts.synthesize_takes = synth
    b.tts.release()
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Fresh directory under /tmp; never a user cache')
    parser.add_argument('--models', type=Path, default=Path.home() / 'Library/Application Support/Maata/Models')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert args.out.is_relative_to(Path('/tmp').resolve()), 'Output must remain under /tmp'
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'started_at': time.time()}
    try:
        asyncio.run(run(args, report))
    except BaseException as err:
        report['error'] = f'{type(err).__name__}: {err}'
        raise
    finally:
        report['ended_at'] = time.time()
        (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
