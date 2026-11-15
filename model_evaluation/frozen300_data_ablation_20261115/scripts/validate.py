from common import *
from report import ALLMODELS,DIMS,scopes,new_values,historical
import csv
def main():
 assert not os.environ.get('SLURM_JOB_ID')
 completion=read(ROOT/'metadata/completion.json');assert completion['status']=='complete' and completion['valid_new_score_keys']==5400 and completion['errors']==0
 assert read(ROOT/'metadata/smoke.json')['status']=='passed' and read(ROOT/'metadata/reuse_audit.json')['status']=='passed'
 assert read(ROOT/'metadata/inference_protocol_audit.json')['status']=='passed'
 media=load_cases();items={r['id']:r for r in rows(PKG/'inputs.jsonl')};assert len(media)==len({(r['id'],r['model']) for r in media})==600
 assert {(r['id'],r['model']) for r in media}=={(sid,m) for sid in items for m in MODELS}
 assert sum(r['num_frames'] for r in media)==62424
 impl=read(ROOT/'metadata/implementation.json');assert sha(PKG/'inputs.jsonl')==impl['input_manifest_sha256'] and sha(PKG/'ground_truth.jsonl')==impl['gt_manifest_sha256']
 for name,h in impl['vbench_identity']['official_sources_sha256'].items():assert sha(VBENV/'runtime/VBench'/name)==h
 for name,h in impl['vbench_identity']['weights_sha256'].items():assert sha(VBENV/'runtime/vbench_models'/name)==h
 for name,h in impl['gt_environment']['lpips_weights'].items():assert sha(name)==h
 assert sha('/scratch/prj0000000275/jcy/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==impl['gt_environment']['alexnet_sha256']
 actual={}
 for case in media:
  assert sha(case['path'])==case['output_sha256']
  cpu=read(path_score('cpu',case));lp=read(path_score('lpips',case))
  assert compatible(path_score('cpu',case),case) and compatible(path_score('lpips',case),case)
  assert len(cpu['values'])==len(lp['values'])==case['num_frames'] and cpu['gt_pts']==lp['gt_pts'] and cpu['alignment']==lp['alignment']
  assert abs(cpu['mean']['ssim']-math.fsum(v[0] for v in cpu['values'])/case['num_frames'])<1e-12
  inf=any(v[2] for v in cpu['values']);assert cpu['mean']['psnr_infinite']==inf
  if inf:assert cpu['mean']['psnr_db'] is None
  else:assert abs(cpu['mean']['psnr_db']-math.fsum(v[1] for v in cpu['values'])/case['num_frames'])<1e-12
  assert abs(lp['score']-math.fsum(lp['values'])/case['num_frames'])<1e-12
  for r in new_values(case):
   assert r['status']=='complete' and (r['psnr_infinite'] or math.isfinite(r['score']))
   if r['dimension'] in VBDIMS:assert abs(r['score']-(r['raw_score']/100 if r['dimension']=='imaging_quality' else r['raw_score']))<1e-12
   actual[r['id'],r['model'],r['dimension']]=r
 for m in ['ltx_distilled_v7','a849_v7']:
  for r in historical(m):actual[r['id'],r['model'],r['dimension']]=r
 table=list(csv.DictReader((ROOT/'report/per_video_metrics.csv').open()));assert len(table)==len(actual)==10800
 assert {(r['id'],r['model'],r['dimension']) for r in table}==set(actual)
 for r in table:
  v=actual[r['id'],r['model'],r['dimension']]
  assert all(r[k]==str(v[k]) for k in ['uid','dataset','group','output_sha256','gt_sha256','status','origin'])
  assert (r['psnr_infinite']=='True')==v['psnr_infinite']
  if v['score'] is None:assert r['score']==''
  else:assert float(r['score'])==v['score']
 summary=list(csv.DictReader((ROOT/'report/summary.csv').open()));scope_map={(level,label):ids for level,label,ids in scopes()}
 assert len(summary)==len({(r['level'],r['group'],r['model'],r['dimension']) for r in summary})==504
 for r in summary:
  ids=scope_map[r['level'],r['group']];values=[actual[sid,r['model'],r['dimension']] for sid in ids]
  assert int(r['n'])==int(r['paired_n'])==int(r['expected'])==len(ids)
  if any(x['psnr_infinite'] for x in values):assert r['mean']==r['paired_mean']=='' and r['psnr_infinite']=='True'
  else:
   mean=math.fsum(x['score'] for x in values)/len(values);assert abs(float(r['mean'])-mean)<1e-12 and abs(float(r['paired_mean'])-mean)<1e-12
 count=0;keys=set();frames={}
 for case in media:
  c=read(path_score('cpu',case));l=read(path_score('lpips',case))
  for i,(v,lp) in enumerate(zip(c['values'],l['values'])):frames[case['id'],case['model'],i]=(v[0],v[1],v[2],lp)
 for line in (ROOT/'report/frames.jsonl').open():
  r=__import__('json').loads(line);key=(r['id'],r['model'],r['frame']);assert key not in keys;keys.add(key);count+=1
  assert (r['ssim'],r['psnr_db'],r['psnr_infinite'],r['lpips'])==frames[key]
 assert count==62424 and keys==set(frames)
 delta_rows=list(csv.DictReader((ROOT/'report/deltas.csv').open()));assert len(delta_rows)==126
 for r in delta_rows:
  ids=scope_map[r['level'],r['group']];assert int(r['paired_n'])==len(ids)
  for field,a,b in [('real_minus_base','real232','ltx_distilled_v7'),('synthetic_minus_base','synthetic600','ltx_distilled_v7'),('synthetic_minus_real','synthetic600','real232')]:
   av=[actual[sid,a,r['dimension']] for sid in ids];bv=[actual[sid,b,r['dimension']] for sid in ids]
   if any(v['psnr_infinite'] for v in av+bv):assert r[field]==''
   else:assert abs(float(r[field])-(math.fsum(v['score'] for v in av)-math.fsum(v['score'] for v in bv))/len(ids))<1e-12
 assert read(ROOT/'report/failures.json')==[]
 atomic(ROOT/'metadata/final_validation.json',{'status':'passed','new_score_keys':5400,'new_outputs':600,'gt_frame_pairs':62424,'report_rows_including_reuse':10800,'summary_rows_verified':504,'static_filter':'not_requested','human_reviewed':False,'validated_unix':time.time()})
 print('VALIDATION_PASSED_5400',flush=True)
if __name__=='__main__':main()
