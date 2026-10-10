from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'engine/src'),str(ROOT/'scripts/experiments')]
import omnivoice_smoke as common
OUT=Path.home()/'Library/Caches/Maata/voice-mapping-2026-10-10'
OUT.mkdir(exist_ok=True)
common.offline()
if sys.argv[1:]==['asr']:
    common.roundtrip(OUT)
    raise SystemExit
import numpy as np,soundfile as sf
from maata_engine.backends.omnivoice import OmniVoiceTTS
from maata_engine.qa.validators import wording
from maata_engine.qa.take import audio_failure

def idle():
    busy=[]
    for p in (Path.home()/'Library/Caches/Maata').glob('*/render/job.json'):
        d=json.loads(p.read_text())
        if d.get('status') in ('running','waiting','queued'):
            busy.append((p.parent.parent.name,d.get('status')))
    if busy:
        raise RuntimeError('Production work became active; stopping smoke: '+str(busy))

idle()
if (OUT/'results.json').exists(): raise RuntimeError('Refusing to rerun existing six-take screen')
tts=OmniVoiceTTS(common.PRODUCTION_MODELS/'omnivoice',voice_dir=common.PRODUCTION_MODELS/'omnivoice/voices')
phrases=[]
for x in common.PHRASES:
    w,problems,_=wording({'spoken':x['text'],'english':[]})
    assert w is not None and not problems
    phrases.append({**x,'text':w.spoken})
report={'version':1,'scope':'Six authored Telugu profile samples, one take per phrase/profile, no user media',
 'phrases':phrases,'runs':{'omnivoice-profiles':{'cache_identity':tts.cache_identity}},'samples':[],
 'limitations':['Gender instructions are trained upstream on English/Chinese; Telugu generalization is screened only.',
 'No native-listening, perceived gender, stable identity or long-video acceptance claim.',
 'No reference conditioning, forced duration, pitch shift or speech-rate control; fixed seed 20261010 each call.',
 'Three different sentences per profile are not repeated throughput measurements.',
 'ASR roundtrip is diagnostic and cannot prove pronunciation, gender or meaning completeness.'],
 'sources':['https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/docs/voice-design.md'],
 'network':'sandbox-exec deny network*; pinned local models and offline environment',
 'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 [ROOT/'engine/src/maata_engine/backends/omnivoice.py',ROOT/'engine/src/maata_engine/backends/_omnivoice_worker.py']}}
common.save(OUT,report)
try:
    for profile in ('male','female'):
        idle()
        began=time.perf_counter(); voice=tts.preset_voice(profile); preparation=time.perf_counter()-began
        report['runs'][profile]={'profile_identity':tts.profile_cache_identity(profile),'voice_identity':voice.identity,'prepare_seconds':preparation}
        for phrase in phrases:
            idle()
            began=time.perf_counter(); take=tts.synthesize_mel(phrase['text'],voice,'te'); generation=time.perf_counter()-began
            began=time.perf_counter(); audio=tts.vocode(take,1.0); marking=time.perf_counter()-began
            raw=OUT/f"{profile}-{phrase['id']}-raw.wav"; marked=OUT/f"{profile}-{phrase['id']}-marked.wav"
            sf.write(raw,take.samples,24000,subtype='FLOAT');sf.write(marked,audio,24000,subtype='FLOAT')
            sample={**phrase,'model':profile,'path':str(marked),'raw_path':str(raw),'sample_rate':24000,
             'duration_seconds':take.seconds,'generation_seconds':generation,'watermark_seconds':marking,
             'capped':take.capped,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
             'sha256':hashlib.sha256(marked.read_bytes()).hexdigest(),'stats':common.stats(audio,24000),
             'audio_failure':audio_failure(audio), 'profile':profile,'instruct':profile,'seed':20261010,
             'watermark_detect_score':float(np.asarray(tts._watermarker.get_watermark(audio,sample_rate=24000)).mean())}
            report['samples'].append(sample);common.save(OUT,report)
            print(json.dumps({'profile':profile,'id':phrase['id'],'generation_s':generation,'duration_s':take.seconds}),flush=True)
finally:
    tts.release()
    report['worker_released']=tts._process is None
    common.save(OUT,report)
