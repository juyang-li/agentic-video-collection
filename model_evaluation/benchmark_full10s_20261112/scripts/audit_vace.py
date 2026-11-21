"""Independent VACE identity/condition/parameter audit and complete video decode."""
from common import *
import argparse,time

def main(completed_only):
 require_cpu_task()
 import av
 items=rows(ROOT/'metadata/inputs.jsonl');assert len(items)==80
 historical={r['id']:r for r in rows(ROOT/'metadata/historical_inputs.jsonl')}
 ready=read(ROOT/'runtime/vace_ready.json');manifest=sha(ROOT/'metadata/inputs.jsonl')
 worker=sha(ROOT/'scripts/vace_worker.py');checks=[]
 for item in items:
  sid=item['id'];path=ROOT/'outputs/vace'/f'{sid}.json'
  if completed_only and (not path.exists() or read(path)['status']!='complete'):continue
  state=read(path);assert state['status']=='complete' and state['full_decode_frames']==121
  old=historical[sid]
  assert item['uid']==old['uid'] and item['prompt']==old['prompt']
  assert [r['sha256'] for r in item['references']]==[r['sha256'] for r in old['references']]
  assert item['source']['sha256']==old['source']['sha256']==sha(item['source']['absolute_path'])
  assert item['timestamps_seconds']==[i/12 for i in range(121)]
  assert not ({'gt','target','target_caption'} & item.keys())
  for ref in item['references']:assert sha(ref['absolute_path'])==ref['sha256']
  c=state['contract'];condition_path=ROOT/'conditions'/sid/'condition.json';condition=read(condition_path)
  assert c['worker_sha256']==worker and c['input_manifest_sha256']==manifest
  assert c['condition_sha256']==sha(condition_path)
  assert c['revision']==ready['vace']['revision'] and c['pipeline']==ready['pipeline']
  assert sha(c['pipeline']['path'])==c['pipeline']['sha256']
  assert (c['num_frames'],c['fps'],c['steps'],c['seed'],c['cfg'],c['conditioning_scale'],c['flow_shift'])==(121,12,40,42,5,1,5)
  assert (c['scheduler'],c['vae_dtype'],c['transformer_dtype'])==('UniPCMultistepScheduler','float32','bfloat16')
  assert len(condition['mask_frames'])==len(condition['mask_sha256'])==121
  for p,h in zip(condition['mask_frames'],condition['mask_sha256'],strict=True):assert sha(p)==h
  mode='official_pose_control' if item['group']=='action' else 'raw_source_camera_control' if item['group']=='camera' else 'official_repaint_gray128'
  assert c['input_video_mode']==mode and state['protocol_audit']['no_gt']
  if mode=='official_repaint_gray128':assert state['official_repaint_erasure']['masked_rgb']==128
  if item['group']=='action':
   assert state['protocol_audit']['reference_count']==3 and state['protocol_audit']['pose_adapter']
   # Worker indexes 0/60/120; its historical role labels 0/24/48 are stale display text.
  out=ROOT/'outputs/vace'/f'{sid}.mp4';assert sha(out)==state['output_sha256']
  n=0
  with av.open(str(out)) as video:
   stream=video.streams.video[0];assert stream.average_rate==12
   for frame in video.decode(video=0):
    assert [frame.width,frame.height]==item['canvas_wh'];n+=1
  assert n==121
  checks.append({'id':sid,'frames_decoded':n,'fps':12,'canvas_wh':item['canvas_wh'],'sha256':state['output_sha256'],'generation_seconds':state['generation_seconds'],'peak_allocated_bytes':state['peak_allocated_bytes'],'source_condition':mode,'actual_source_appearance_indices':[0,60,120] if item['group']=='action' else None})
 assert checks and (completed_only or len(checks)==80)
 suffix='_partial' if completed_only else ''
 atomic(ROOT/'metadata'/f'technical_audit_vace{suffix}.json',{'status':'passed','scope':'completed subset' if completed_only else 'all 80 samples','samples':len(checks),'checked_unix':time.time(),'checks':checks,'human_quality_passed':False})
 print('VACE_TECHNICAL_AUDIT_PASSED',len(checks),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--completed-only',action='store_true');main(p.parse_args().completed_only)
