"""Independent completed-stage inventory, container and recorded full-decode audit."""
from common import *
import argparse,time

def main(model):
 require_job();import av
 inputs=rows(ROOT/'metadata/inputs.jsonl');assert len(inputs)==80
 original={i['id']:i for i in rows(ROOT/'metadata/historical_inputs.jsonl')}
 assert {i['id'] for i in inputs}=={i['id'] for i in original.values() if i['dataset']=='ue'}
 assert len(read(ROOT/'metadata/short_clip_exceptions.json'))==20
 manifest=sha(ROOT/'metadata/inputs.jsonl');worker=sha(ROOT/'scripts/native_worker.py');ckpt=sha(CHECKPOINT) if model=='a849' else None
 records=[]
 for item in inputs:
  sid=item['id'];old=original[sid]
  assert item['uid']==old['uid'] and item['prompt']==old['prompt']
  assert item['source']['sha256']==old['source']['sha256']
  assert [r['sha256'] for r in item['references']]==[r['sha256'] for r in old['references']]
  assert item['num_frames']==121 and item['fps']==12 and item['timestamps_seconds']==[i/12 for i in range(121)]
  assert not ({'target','target_caption','gt'} & item.keys())
  status=read(ROOT/'outputs'/model/f'{sid}.json');c=status['contract'];out=ROOT/'outputs'/model/f'{sid}.mp4'
  assert status['status']=='complete' and status['full_decode_frames']==121
  assert c['input_manifest_sha256']==manifest and c['worker_sha256']==worker and c['checkpoint_sha256']==ckpt
  assert (c['num_frames'],c['fps'],c['steps'],c['seed'])==(121,12,40,42)
  assert (c['cfg'],c['stg'],c['rescale'],c['ref_cfg'])==(3,1,.7,1)
  assert c['reference_protocol']=='reference_positions_v7_refs_source_target'
  assert status['protocol_audit']['no_gt'] and status['protocol_audit']['target_last']
  assert sha(out)==status['output_sha256']
  with av.open(str(out)) as video:
   stream=video.streams.video[0]
   assert stream.frames==121 and stream.average_rate==12 and [stream.width,stream.height]==item['canvas_wh']
  records.append({'id':sid,'group':item['group'],'file':str(out),'full_decode_frames':status['full_decode_frames'],'denoise_seconds':status['denoise_seconds'],'decode_seconds':status['decode_seconds'],'denoise_peak_allocated_bytes':status['denoise_peak_allocated_bytes'],'decode_peak_allocated_bytes':status['decode_peak_allocated_bytes'],'started_unix':status['started_unix'],'completed_unix':status['completed_unix']})
 assert len(list((ROOT/'outputs'/model).glob('*.mp4')))==80
 result={'status':'passed','model':model,'samples':80,'checked_unix':time.time(),'inference_manifest_sha256':manifest,'records':records,'generation_wall_seconds':max(r['completed_unix'] for r in records)-min(r['started_unix'] for r in records),'sum_denoise_seconds':sum(r['denoise_seconds'] for r in records),'peak_allocated_bytes':max(max(r['denoise_peak_allocated_bytes'],r['decode_peak_allocated_bytes']) for r in records),'full_decode_coverage':'all 80 files decoded during inference before SHA receipt; audit rechecks each file SHA and container specs','human_quality_passed':False}
 atomic(ROOT/'metadata'/f'technical_audit_{model}.json',result);print('TECHNICAL_AUDIT_PASSED',model,len(records),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['a849','ltx_base'],required=True);main(p.parse_args().model)
