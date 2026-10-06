"""Original six-control correction review pilot. Two sealed text-only calls, no production cache/audio."""
from __future__ import annotations
import json,platform,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
from maata_engine.backends.base import Coverage,LineSpec,SceneRequest,Wording
from maata_engine.codex_cli import CodexCLI,DEFAULT_MODEL,DEFAULT_EFFORT
from maata_engine.qa.coverage import check_review,coverage_json,redo_class
from maata_engine.qa.validators import meaning_flags
from maata_engine.text.akshara import count_units
from maata_engine.text.scene_prompt import REVIEW_SYSTEM,REVIEW_SCHEMA,REVIEW_HASH,PROMPT_HASH,review_message

def main():
 dest=Path(__file__).parent
 cases=json.loads((dest/'inputs.json').read_text())['cases']
 assert len(cases)==6 and DEFAULT_MODEL=='gpt-6-luna' and DEFAULT_EFFORT=='low'
 # Recompute the deterministic comparator from the checked-out engine, not the recorded labels.
 for c in cases:
  before,candidate=Wording(c['before']),Wording(c['candidate'])
  c.update(before_units=count_units(before.spoken),candidate_units=count_units(candidate.spoken),
           before_flags=meaning_flags(c['en'],before),candidate_flags=meaning_flags(c['en'],candidate),
           deterministic=coverage_json(redo_class(c['en'],Coverage('P',tuple(c['missing']),tier='full'),before,candidate)))
 requests=[]
 for n,order in enumerate((cases,list(reversed(cases))),1):
  specs=tuple(LineSpec(c['id'],'S1',c['en'],j*8.0,(j+1)*8.0,8.0,40.0) for j,c in enumerate(order))
  req=SceneRequest(n,specs)
  prompt=review_message(req,[(spec,Wording(c['candidate'])) for spec,c in zip(specs,order)])
  requests.append({'round':n,'ids':[c['id'] for c in order],'prompt':prompt})
 report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','scope':'Two independent fresh ephemeral CLI invocations of production reviewer on six original candidate controls. No audio, translation, production caches, or policy changes.','model':DEFAULT_MODEL,'effort':DEFAULT_EFFORT,'fallback':None,'review_hash':REVIEW_HASH,'prompt_hash':PROMPT_HASH,'system':REVIEW_SYSTEM,'schema':REVIEW_SCHEMA,'requests':requests,'rounds':[],'transport_retries':0,'timeout_s':120,'startup_timeout_s':60,'host':platform.platform(),'background_load':'User-authorized main application is transcribing another long video; this is not a latency benchmark.','limitations':['Six deliberately adversarial authored controls, not a representative sample.','Expected labels and before-P labels are author judgments; no native-human or audio evaluation.','Only candidate English/te/actualtts and empty context are sent, never expected labels, deterministic outputs or before wordings.','Two replies do not establish reviewer reliability, independence of model errors, or a production improvement.','English maps are empty so te==actualtts; this does not exercise Latin substitution errors.','No speed comparison or automatic policy adoption.'],'events':[]}
 def save(): (dest/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 save()
 for request in requests:
  record={'round':request['round'],'ids':request['ids']};began=time.monotonic()
  with tempfile.TemporaryDirectory(prefix='maata-original-correction-review-') as cache:
   client=CodexCLI(Path(cache),model=DEFAULT_MODEL,effort=DEFAULT_EFFORT,retries=0,timeout=120,startup_timeout=60,trace=report['events'].append)
   try:
    reply=client.ask(REVIEW_SYSTEM,request['prompt'],REVIEW_SCHEMA,'review',effort=DEFAULT_EFFORT,tags={'experiment':'original-correction-controls','round':request['round'],'lines':6})
    got,why=check_review(reply.data,request['ids'])
    record.update(status='complete' if not why and len(got)==6 else 'incomplete',output=reply.data,raw_text=reply.text,usage=reply.usage,model=reply.model,seconds=reply.seconds,classes={str(i):coverage_json(c) for i,c in got.items()},missing_or_invalid=why)
   except Exception as error:
    record.update(status='failed',error={'kind':getattr(error,'kind',type(error).__name__),'message':str(error)})
  record['elapsed_s']=round(time.monotonic()-began,3);report['rounds'].append(record);save()
  print(json.dumps({'round':request['round'],'status':record['status'],'elapsed_s':record['elapsed_s'],'classes':{i:c['class'] for i,c in record.get('classes',{}).items()},'error':record.get('error')},ensure_ascii=False),flush=True)
 report['status']='complete' if all(r['status']=='complete' for r in report['rounds']) else 'failed_or_incomplete'
 summary={'status':report['status'],'model':report['model'],'effort':report['effort'],'review_hash':report['review_hash'],'review_calls':len(report['rounds']),'cli_event_count':len(report['events']),'failed_calls':sum(r['status']!='complete' for r in report['rounds']),'cases':[],'limits':report['limitations']}
 for c in cases:
  verdicts=[r.get('classes',{}).get(str(c['id']),{}).get('class') for r in report['rounds']]
  summary['cases'].append({'id':c['id'],'control':c['control'],'before_units':c['before_units'],'candidate_units':c['candidate_units'],'expected_classes':c['expected'],'deterministic_class':c['deterministic']['class'],'semantic_classes':verdicts,'both_semantic_match_authored_expectation':all(v in c['expected'] for v in verdicts)})
 summary['deterministic_matches_authored_expectation']=sum(c['deterministic_class'] in c['expected_classes'] for c in summary['cases'])
 summary['semantic_matches_by_round']=[sum(c['semantic_classes'][n] in c['expected_classes'] for c in summary['cases']) for n in range(len(report['rounds']))]
 save();(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'summary':summary,'directory':str(dest)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
