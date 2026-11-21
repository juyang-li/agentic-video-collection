"""CPU final integrity, missingness, static-pair and package checks."""
import csv,zipfile,argparse
from collections import Counter
from common import *
from contracts import DIMS,MODELS,generation_input,digest,common_static

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--allow-pending-visual',action='store_true');args=parser.parse_args()
 import av
 if (ROOT/'metadata/identity_protocol_correction.json').exists():assert read(ROOT/'metadata/identity_protocol_correction.json')['status']=='validated','Corrected full-head identity decision pending'
 items=rows(ROOT/'metadata/inputs.jsonl');frozen=read(ROOT/'metadata/freeze.json');assert len(items)==96 and frozen['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 assert dict(Counter(r['group'] for r in items))==frozen['groups']
 gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved'
 reuses={(r['id'],r['model']):r for r in read(ROOT/'metadata/reuse_outputs.json')['reused']};failures=[];checked=[]
 for item in items:
  sid=item['id'];contracts={}
  for model in MODELS:
   path=ROOT/'outputs'/model/f'{sid}.mp4';receipt=path.with_suffix('.json');r=read(receipt) if receipt.exists() else {'status':'missing'}
   if r['status']!='complete':failures.append(dict(id=sid,model=model,status=r['status']));continue
   assert sha(path)==r['output_sha256'] and r['contract']['steps']==40 and r['contract']['seed']==42
   if (sid,model) in reuses:
    proof=reuses[sid,model];assert sha(proof['original_generation_receipt'])==proof['original_generation_receipt_sha256']
    c=read(ROOT/'conditions'/sid/'condition.json') if model=='vace' else None
    assert digest(generation_input(item,model,c))==proof['effective_input_sha256']
   else:assert r['contract']['input_manifest_sha256']==frozen['input_manifest_sha256']
   with av.open(str(path)) as con:
    stream=con.streams.video[0];assert str(stream.average_rate) in ['12','12/1'];frames=list(con.decode(video=0))
   assert len(frames)==49 and all([frame.width,frame.height]==item['canvas_wh'] for frame in frames)
   assert r['protocol_audit']['no_gt'] and r['full_decode_frames']==49
   contracts[model]=r['contract'];checked.append(dict(id=sid,model=model,frames=49,output_sha256=r['output_sha256']))
  if all(m in contracts for m in ['ltx_base','a849']):
   a,b=contracts['ltx_base'],contracts['a849'];assert {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}=={'checkpoint','checkpoint_sha256'}
 records=list(csv.DictReader((ROOT/'metadata/per_video_metrics.csv').open()))
 main_records=[r for r in records if r['cohort']=='main'];assert len(main_records)==1728 and {r['dimension'] for r in records}==set(DIMS)
 assert all(r['score']=='' for r in records if r['status']!='complete')
 static=read(ROOT/'metadata/temporal_flickering_static_paired.json')['main'];flags={(r['id'],m):r[m] for r in static['filter_results'] for m in ('source',*MODELS)}
 eligible=common_static([r['id'] for r in items],flags);assert eligible==static['static_eligible_task_ids']
 assert static['filter_complete'],'Static filter unfinished; not a final zero-subset conclusion'
 assert all(static['models'][m]['n']==static['n'] for m in MODELS)
 visual=read(ROOT/'metadata/visual_observations.json');assert set(visual['samples'])=={r['id'] for r in items}
 visual_complete=visual['inspection_complete']
 if not args.allow_pending_visual:assert visual_complete
 for sid,sample in visual['samples'].items():
  for m in MODELS:
   if m not in sample['models']:
    assert args.allow_pending_visual;continue
   v=sample['models'][m];assert v['reviewed'] is True and v['output_sha256']==read(ROOT/'outputs'/m/f'{sid}.json').get('output_sha256')
 package=read(ROOT/'metadata/package.json')
 with zipfile.ZipFile(package['path']) as z:
  assert z.testzip() is None and package['offline_page'] in z.namelist()
 metrics=read(ROOT/'metadata/group_metrics.json');valid=sum(r['status']=='complete' for r in main_records);assert valid==metrics['valid_main_scores']
 atomic(ROOT/'metadata/final_validation.json',dict(status=('passed' if visual_complete else 'awaiting_visual_review') if valid==1728 and not failures else 'partial',technical_status='passed' if valid==1728 and not failures else 'partial',visual_inspection_complete=visual_complete,valid_main_scores=valid,expected_main_scores=1728,generation_failures=failures,checked_outputs=checked,static_paired_n=static['n'],package_crc='passed',input_manifest_sha256=frozen['input_manifest_sha256'],scope='Integrity and completeness only; low quality valid outputs retained, not a claim that edits succeeded.'))
 print('Validated',len(checked),'outputs;',valid,'main scores; static paired',static['n'],flush=True)
if __name__=='__main__':main()
