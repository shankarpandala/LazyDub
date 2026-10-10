import asyncio,importlib.util,json,pathlib,time,hashlib
ROOT=pathlib.Path('/Users/shankarpandala/projects/LazyDub')
spec=importlib.util.spec_from_file_location('bench_translation_scenes',ROOT/'scripts/bench_translation_scenes.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=pathlib.Path('/tmp/maata-lane-comparison');OUT.mkdir(exist_ok=True)
texts=[
'Ravi and Meena are preparing a small library event.',
'Ravi keeps the blue box, and Meena keeps the red one.',
'The blue box is called Aster; the red box is called Cedar.',
'Aster contains twelve books, while Cedar contains nine.',
'Neither box contains books that are for sale.',
'Ravi thinks his box may need a stronger handle.',
'He will replace it only if the old one breaks.',
'Meena does not need to replace the handle on hers.',
'Their event starts at ten, not at eleven.',
'Visitors may borrow a book, but they cannot keep it forever.',
'Meena wrote the return date on a yellow card.',
'She put that card inside Cedar before closing the lid.',
'It must stay there until the first visitor arrives.',
'Ravi put a different card inside Aster.',
'His card lists the twelve titles, not their prices.',
'Three of those books have large print.',
'The other nine use ordinary print.',
'He reserved the large print books for visitors who requested them.',
'They are not available to everyone yet.',
'Meena has not reserved any books from her box.',
'One visitor asked whether Cedar contained a dictionary.',
'She said it contained two, but neither was new.',
'Both dictionaries can still be borrowed.',
'The visitor chose the smaller one and thanked Meena.',
'He did not take the larger one as well.',
'Ravi recorded that choice in the shared notebook.',
'The notebook belongs to the library, not to either volunteer.',
'It cost two hundred and fifty rupees, including delivery.',
'A replacement cover might cost another fifty rupees.',
'Meena has not ordered that cover yet.',
'So the final cost is not confirmed.',
'Ravi will check the old cover before buying a new one.',
'If it can be repaired, they will keep it.',
'That decision does not affect the return dates.',
'A parent asked for one extra day to return a book.',
'Meena agreed for that parent, but not for every visitor.',
'The exception applies only to that borrowed book.',
'All the other books are still due on Friday.',
'Ravi wrote the exception beside the title in the notebook.',
'He did not change the date on every card.',
'By noon, Aster had five books left and Cedar had four.',
'Meena checked both counts before announcing them.',
'She said those numbers described books still in the boxes.',
'They did not describe the total number of visitors.',
'Two visitors had only looked around and borrowed nothing.',
'Ravi thanked them too, because attendance did not require borrowing.',
'The volunteers might repeat the event next month.',
'They will decide after the borrowed books come back.',
'Until then, another event is only a possibility.',
'Neither volunteer has promised a date.',
'Meena will keep Cedar at the library until Friday.',
'Ravi will bring Aster back tomorrow morning.',
'He must return the blue box, even if it is empty.',
'She will check both boxes again before closing the library.'
]
assert len(texts)==54
fixture={'version':1,'title':'Original library event lane scheduling fixture','duration':270.0,
'provenance':{'source':'Original authored English, no user transcript or audio','purpose':'Bounded text-only lane scheduling comparison'},
'settings':{'speakers':1,'style':'colloquial','stopAt':None,'speedCap':1.2,'ttsScript':'latin','presets':[]},
'brief':{'version':1,'meta':{'title':'Library event','channel':'Maata original fixture','description':'Original fictional library event.','chapters':[],'tags':[],'talk_shares':[]},'topic':'Ravi and Meena lend library books from two labeled boxes. Ravi has blue Aster with twelve books; Meena has red Cedar with nine.','register':'Neutral everyday spoken Telugu; polite audience address మీరు.','speakers':[{'id':'S1','name':'','gender':'unknown','role':'narrator','audience':'polite','address':[]}],'glossary':[{'term':'Aster','spoken':'ఆస్టర్','keep_english':True,'note':'Name of Ravi’s blue book box'},{'term':'Cedar','spoken':'సీడర్','keep_english':True,'note':'Name of Meena’s red book box'}],'entities':['Ravi','Meena','Aster','Cedar'],'idioms':[],'numbers':'Retain exact numbers and uncertainty; speak them in Telugu.','asr_fixes':[]},
'lines':[{'id':i,'speaker':'S1','start':5.0*i,'end':5.0*i+5,'speech_s':4.5,'text':t} for i,t in enumerate(texts)],
'semantic_expectations':[
{'ids':[1,2,3,5,6,7],'check':'Ravi owns blue Aster/12 books; Meena red Cedar/9. Only his handle may need replacement, conditional on breaking.'},
{'ids':[10,11,12,13,14],'check':'The yellow return-date card stays in Cedar until first visitor. Ravi has a different title-list card, not prices.'},
{'ids':[15,16,17,18,19],'check':'Three large-print books reserved only for requesting visitors; other nine ordinary. Not available to everyone yet. Meena has no reservations.'},
{'ids':[20,21,22,23,24],'check':'Cedar has two non-new dictionaries. Visitor took smaller only, not both.'},
{'ids':[27,28,29,30,31,32],'check':'250 includes delivery; possible extra50 cover un-ordered; finalcost unconfirmed; retain old cover if repairable.'},
{'ids':[34,35,36,37,38,39],'check':'One parent/book gets one extra day; all others due Friday. Not blanket date change.'},
{'ids':[40,41,42,43,44],'check':'Aster five/Cedar four left are remaining books, not visitor totals; two visitors borrowed nothing.'},
{'ids':[46,47,48,49],'check':'Next-month repeat possible only; decision after returns, no promised date.'},
{'ids':[50,51,52,53],'check':'Meena keeps red Cedar until Friday; Ravi returns blue Aster tomorrow even empty. She later checks both.'}]}
b.save(OUT/'fixture.json',fixture)
fixture=b.normalize_fixture(fixture,ROOT/'scripts/fixtures/translation-scenes.json')
fixture['models']={'scene':'gpt-6-luna','review':'gpt-6-luna','effort':'low','review_effort':'low','light_effort':'low'}
b.CAPS={'A':(30.0,6),'B':(30.0,6)}
WEIGHTS={'A':(1.,1.,1.),'B':(1.,1.5,2.)}
original=b.TextJob._translation_lanes
plans={}
for arm,w in WEIGHTS.items():
 import tempfile
 with tempfile.TemporaryDirectory() as tmp:
  j=b.TextJob(pathlib.Path(tmp),fixture,(30.0,6));sc=j._cut_scenes();assert len(sc)==9
  plans[arm]=[[s.no for s in lane] for lane in original(j,sc,weights=w)]
report={'scope':'Bounded original-text lane scheduling pilot; no audio, moving voicer frontier, or video completion claim','models':fixture['models'],'fallback':None,'order':'ABBA','weights':WEIGHTS,'plans':plans,'prompt_hash':b.PROMPT_HASH,'review_hash':b.REVIEW_HASH,'fixture_sha256':b.digest(OUT/'fixture.json'),'limitations':['Two runs per arm maximum, below production measurement minimum five.','Provider prefix cache uncontrolled; fresh local text cache per arm.','No native-speaker/human listening evaluation; automated coverage does not establish meaning correctness.','Original fixture and generated text retained only under /tmp; no user transcript/audio used.'],'runs':[],'status':'running'}
b.save(OUT/'results.json',report)
async def main():
 start=time.monotonic()
 for n,arm in enumerate(report['order'],1):
  if n>2 and time.monotonic()-start>315:
   report['status']='bounded_partial';break
  def lanes(self,scenes,*,weights=None):return original(self,scenes,weights=WEIGHTS[arm])
  b.TextJob._translation_lanes=lanes
  result=await b.run_arm(fixture,arm,False,150.)
  result['sequence']=n;result['lane_weights']=WEIGHTS[arm];result['lane_scenes']=plans[arm]
  report['runs'].append(result);b.save(OUT/'results.json',report)
  print(json.dumps({k:result.get(k) for k in ['sequence','arm','status','first_contiguous_ready_seconds','all_text_ready_seconds','calls','coverage','quality_eligible']},ensure_ascii=False),flush=True)
  if result['status']!='completed' or result['model_mismatch']:report['status']='failed';break
 else:report['status']='completed'
 b.TextJob._translation_lanes=original
 report['elapsed_s']=time.monotonic()-start
 # Preserve original outputs/requests for semantic checks; separate sanitized aggregates exclude all line text.
 aggregated=[]
 for r in report['runs']:
  reviews=[e for e in r['events'] if e.get('event')=='review']
  aggregated.append({k:r[k] for k in ['sequence','arm','lane_weights','lane_scenes','status','first_contiguous_ready_seconds','all_text_ready_seconds','translation_stage_seconds','ready_by_scene','contiguous_ready','coverage','calls','usage','quality_eligible','valid_hygiene','hygiene_start','hygiene_end','model_mismatch']})
  aggregated[-1]['corrections']={'review_events':len(reviews),'retranslated_lines':sum(len(e.get('retranslated',[])) for e in reviews),'replaced_lines':sum(len(e.get('replaced',[])) for e in reviews)}
  aggregated[-1]['call_metrics']=[{k:e.get(k) for k in ['call','scene','model','effort','wall_s','startup_s','input_tokens','cache_read_tokens','output_tokens','error']} for e in r['events'] if e.get('event')=='claude']
 b.save(OUT/'results.json',report)
 b.save(OUT/'aggregate.json',{**{k:v for k,v in report.items() if k!='runs'},'runs':aggregated})
 print(json.dumps({'status':report['status'],'elapsed_s':report['elapsed_s'],'path':str(OUT/'aggregate.json')}),flush=True)
asyncio.run(main())
