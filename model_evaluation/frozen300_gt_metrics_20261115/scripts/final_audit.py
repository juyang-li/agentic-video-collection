"""Independent exact-coverage and aggregation audit, after evaluator finishes."""
from common import *
import math,numpy as np

def main():
 items=rows(PKG/'inputs.jsonl');expected={(m,r['id']) for m in MODELS for r in items};byid={r['id']:r for r in items};records={}
 for p in (ROOT/'scores').glob('*/*.json'):
  r=read(p);key=(r['model'],r['id']);assert key not in records;records[key]=r
 assert set(records)==expected and len(records)==1200
 digest=sha(ROOT/'metadata/protocol.json');total=0
 for (model,sid),r in records.items():
  assert r['status']=='complete' and r['protocol_sha256']==digest
  assert r['num_frames']==byid[sid]['num_frames']==len(r['frames'])==len(r['alignment']['selected'])
  assert [f['frame'] for f in r['frames']]==list(range(r['num_frames']))
  for key in ['ssim','lpips']:
   vals=[f[key] for f in r['frames']];assert all(math.isfinite(v) for v in vals)
   assert abs(float(np.mean(vals))-r['mean'][key])<1e-10
  infinite=any(f['psnr_infinite'] for f in r['frames']);assert infinite==r['mean']['psnr_infinite']
  if not infinite:assert abs(np.mean([f['psnr_db'] for f in r['frames']])-r['mean']['psnr_db'])<1e-10
  total+=r['num_frames']
 assert total==124848
 summary=read(ROOT/'report/summary.json')
 for row in summary:
  selected=[r for (m,sid),r in records.items() if m==row['model'] and (row['level']=='overall' or r['dataset' if row['level']=='dataset' else 'group']==row['group'])]
  assert len(selected)==row['videos']
  for key in ['ssim','lpips']:assert abs(sum(r['mean'][key] for r in selected)/len(selected)-row[key])<1e-10
 assert all((ROOT/'report'/name).is_file() for name in ['REPORT.md','summary.csv','videos.csv','frames.jsonl'])
 atomic(ROOT/'metadata/completion_audit.json',{'status':'passed','videos':1200,'frame_pairs':total,'all_models_300':True,'per_video_means_verified':True,'summary_means_verified':True,'protocol_sha256':digest})
 print('COMPLETION_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
