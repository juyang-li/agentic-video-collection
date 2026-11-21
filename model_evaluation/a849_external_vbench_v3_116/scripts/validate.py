"""CPU final acceptance: immutable inputs, exact reuse, media, six metrics, four-model static intersection."""
import argparse,csv,time,concurrent.futures
from common import *
from score import MODELS,inventory
from contracts import DIMS,generation_input,digest

def check_media(pair):
 import av
 path,frames,fps,canvas=pair;h=sha(path);key=digest(dict(sha256=h,frames=frames,fps=fps,canvas=canvas,verifier_sha256=sha(__file__)));cache=ROOT/'metadata/media_validation'/f'{key}.json'
 if cache.exists():assert read(cache)['status']=='passed';return str(path)
 with av.open(str(path))as c:
  c.streams.video[0].thread_count=1;assert c.streams.video[0].average_rate==fps;count=0
  for frame in c.decode(video=0):assert[frame.width,frame.height]==canvas;count+=1
 assert count==frames,(path,count,frames)
 atomic(cache,dict(status='passed',output_sha256=h,frames=count,fps=fps,canvas_wh=canvas,verifier_sha256=sha(__file__)))
 return str(path)

def check_h3_pixels(item):
 import av,hashlib
 sid=item['id'];comparison=ROOT/'outputs/h3_pdd8'/f'{sid}.mp4';r=read(comparison.with_suffix('.json'));native=Path(r['native_path']);canvas=r['comparison']['canvas_wh'];key=digest(dict(native_sha256=r['native_sha256'],comparison_sha256=r['output_sha256'],verifier_sha256=sha(__file__)));cache=ROOT/'metadata/media_validation'/f'h3_pixels--{key}.json'
 if cache.exists():assert read(cache)['status']=='passed';return sid
 assert sha(native)==r['native_sha256']and sha(comparison)==r['output_sha256']
 selected=[]
 with av.open(str(native))as c:
  stream=c.streams.video[0];stream.thread_count=1;assert stream.average_rate==24;count=0;audio=len(c.streams.audio)
  for n,frame in enumerate(c.decode(video=0)):
   assert[frame.width,frame.height]==canvas
   if n<=96 and n%2==0:selected.append(hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest())
   count+=1
 assert count==107 and len(selected)==49
 with av.open(str(comparison))as c:
  c.streams.video[0].thread_count=1;actual=[hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()for f in c.decode(video=0)]
 assert actual==selected,'Scoring pixels differ from actual native generated indices0,2,...96'
 atomic(cache,dict(status='passed',id=sid,native_sha256=r['native_sha256'],comparison_sha256=r['output_sha256'],native_decoded_frames=count,native_audio_streams=audio,comparison_generated_indices=list(range(0,97,2)),comparison_pixel_sha256=actual,verifier_sha256=sha(__file__)))
 return sid

def main(allow_pending):
 items=rows(ROOT/'metadata/inputs.jsonl');assert len(items)==116 and len({i['id']for i in items})==116;manifest=sha(ROOT/'metadata/inputs.jsonl');freeze=read(ROOT/'metadata/freeze.json');assert manifest==freeze['input_manifest_sha256']
 assert sum(i['cohort']=='retained88'for i in items)==88 and sum(i['cohort']=='candidate28'for i in items)==28
 assert [i['id']for i in items]==sorted(i['id']for i in items)
 from collections import defaultdict
 numbered=defaultdict(list)
 for i in items:prefix,n=i['id'].rsplit('_',1);numbered[prefix].append(int(n))
 assert all(ns==list(range(1,len(ns)+1))for ns in numbered.values())
 separation=read(ROOT/'metadata/reference_group_separation.json');assert not set(separation['hair_person_ids'])&set(separation['identity_person_ids']) and separation['all_people_unique']
 gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved'and gate['manifest_sha256']==manifest
 origins=read(ROOT/'metadata/input_origins.json')
 for item in items:
  if item['cohort']!='retained88':continue
  parent,sid=origins[item['id']];prior=Path(parent);original=next(r for r in rows(prior/'metadata/inputs.jsonl')if r['id']==sid)
  for m in MODELS[:3]:
   c=read(ROOT/'conditions'/item['id']/'condition.json')if m=='vace'else None;old=read(prior/'conditions'/sid/'condition.json')if m=='vace'else None
   assert generation_input(item,m,c)==generation_input(original,m,old)
 reuse=read(ROOT/'metadata/reuse_outputs.json');assert reuse['complete']==264 and len(reuse['reused'])==264 and not reuse['parent_outputs_pending']
 for r in reuse['reused']:
  receipt=Path(r['original_receipt']);assert sha(receipt)==r['original_receipt_sha256']and read(receipt)['output_sha256']==r['output_sha256'];assert(ROOT/'outputs'/r['model']/f"{r['id']}.mp4").resolve()==receipt.with_suffix('.mp4').resolve()
 preflight=read(ROOT/'metadata/h3_preflight.json');runtime=read(ROOT/'metadata/h3_runtime_contract.json');assert preflight['status']=='passed'and preflight['input_manifest_sha256']==manifest and preflight['runtime_contract_sha256']==sha(ROOT/'metadata/h3_runtime_contract.json')
 assert runtime['num_inference_steps']==8 and runtime['pdd_config']==dict(num_steps=32,block_size=4,fused_steps=8,num_inference_steps=8,video_shift=12.,audio_shift=3.)and runtime['tp_size']==2 and runtime['ulysses_degree']==2
 videos=inventory();assert len(videos)==464 and all(r['generation_status']=='complete'for r in videos);media=[]
 for item in items:
  sid=item['id']
  for m in MODELS[:3]:media.append((ROOT/'outputs'/m/f'{sid}.mp4',49,12,item['canvas_wh']))
  h3=read(ROOT/'outputs/h3_pdd8'/f'{sid}.json');native=read(ROOT/'outputs/h3_pdd8_native'/f'{sid}.json');canvas=[1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344]
  assert h3['native_sha256']==native['output_sha256']==sha(h3['native_path']);assert sha(h3['browser_path'])==h3['browser_sha256'];assert h3['comparison']['generated_frame_indices']==list(range(0,97,2))and h3['comparison']['lossless_pixel_preservation']is True
  assert native['contract']['runtime_contract_sha256']==sha(ROOT/'metadata/h3_runtime_contract.json')and native['contract']['steps']==8
  from contracts import digest
  assert native['contract']['request_sha256']==digest(read(ROOT/'metadata/h3_requests'/f'{sid}.json'))
  if native['contract']['manifest_sha256']!=manifest:
   proof=read(ROOT/'metadata/h3_reuse'/f'{sid}.json');assert proof['new_manifest_sha256']==manifest and proof['output_sha256']==native['output_sha256'] and proof['original_receipt_sha256']==sha(ROOT/'outputs/h3_pdd8_native'/f'{sid}.json')
  media.extend([(ROOT/'outputs/h3_pdd8'/f'{sid}.mp4',49,12,canvas),(Path(h3['browser_path']),49,12,canvas)])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  decoded=list(pool.map(check_media,media));h3_pixel_checks=list(pool.map(check_h3_pixels,items))
 scores=list(csv.DictReader((ROOT/'metadata/per_video_metrics.csv').open()));assert len(scores)==2784 and len({(r['id'],r['model'],r['dimension'])for r in scores})==2784
 missing=[r for r in scores if r['status']!='complete'];assert all(r['score']==''and r['raw_score']==''for r in missing)
 observations=read(ROOT/'metadata/visual_observations.json');reviewed=0
 for r in videos:
  note=observations['samples'][r['id']]['models'].get(r['model'],{})
  if note.get('reviewed'):assert note['output_sha256']==r['output_sha256'];reviewed+=1
 subsets=read(ROOT/'metadata/temporal_flickering_static_paired.json')
 for cohort,s in subsets.items():
  assert s['filter_complete'],'Official static filtering incomplete'
  eligible=[r['id']for r in s['filter_results']if all(r[m]is True for m in('source',*MODELS))];assert eligible==s['static_eligible_task_ids']
  assert s['n']==len(s['task_ids'])and set(s['task_ids'])<=set(eligible)
  if not s['task_ids']:assert s['status']in['not_applicable','incomplete_scores']and all(v['mean']is None for v in s['models'].values())
 assert read(ROOT/'metadata/all_media_validation.json')['complete_generations']==464
 from review import check_links
 check_links(ROOT/'review/all')
 status='passed'if reviewed==464 else'awaiting_visual_review'
 if not allow_pending:assert reviewed==464,'Direct visual review incomplete'
 atomic(ROOT/'metadata/final_validation.json',dict(status=status,manifest_sha256=manifest,outputs=464,full_decoded_scoring_and_browser_videos=len(decoded),native_h3_lossless_pixel_checks=len(h3_pixel_checks),expected_scores=2784,valid_scores=2784-len(missing),missing_scores=[dict(id=r['id'],model=r['model'],dimension=r['dimension'],status=r['status'])for r in missing],direct_reviewed=reviewed,retained88_effective_inputs_unchanged=True,parent_outputs_reused=264,source_plus_four_model_static_intersection=True,validated_unix=time.time()))
 print('Validation',status,'reviewed',reviewed,'/464; missing scores',len(missing),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--allow-pending-visual',action='store_true');a=p.parse_args();main(a.allow_pending_visual)
