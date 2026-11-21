"""Audit actual completed requests and native performance records against the plan."""
from common import *
import argparse,time

def gate(stage):
 if os.environ.get("SLURM_JOB_ID"):require_job()
 items=rows(ROOT/'metadata/inputs.jsonl');byid={r['id']:r for r in items}
 old={r['id']:r for r in rows(OLD/'metadata/inputs.jsonl')}
 assert len(items)==len(old)==80 and set(byid)==set(old)
 logs=(ROOT/'logs/server.log').read_text().splitlines()
 line=next(l for l in logs if 'server_args: {' in l);config=json.loads(line[line.index('{'):line.rindex('}')+1])
 assert config['num_gpus']==4 and config['tp_size']==2 and config['ulysses_degree']==2
 assert config['encoder_parallel']=='auto' and config['quantization'] is None and not config['enable_torch_compile']
 assert config['cache_dit_config'] is None and config['lora_path'] is None and not config['enable_cfg_parallel']
 assert config['attention_backend'] is None and not config['component_attention_backends'] and config['prompt_enhancer_config'] is None
 assert config['batching_max_size']==1
 atomic(ROOT/'metadata/server_config.json',config)
 ids=['hair_color_01','single_piece_01','hair_01','site_01'] if stage=='pilot' else [g+'_01' for g in GROUPS] if stage=='first' else list(byid)
 if stage=='completed':ids=[sid for sid in byid if (ROOT/'outputs'/f'{sid}.json').exists() and read(ROOT/'outputs'/f'{sid}.json')['status']=='complete'];assert ids
 checks=[]
 for sid in ids:
  item=byid[sid];prior=old[sid]
  assert {k:item[k] for k in prior}==prior
  assert prior['prompt']['en'] in item['h3_prompt']
  state=read(ROOT/'outputs'/f'{sid}.json');assert state['status']=='complete'
  assert state['full_decode_frames']==243 and state['fps']==24 and state['canvas_wh']==item['h3_canvas_wh']
  assert sha(ROOT/'outputs'/f'{sid}.mp4')==state['output_sha256']
  assert state['contract']['sglang_build_patch_sha256']==sha(ROOT/'metadata/sglang_build_portability.patch')
  assert state['contract']['serve_script_sha256']==sha(ROOT/'scripts/serve.sh')
  body=state['posted_request'];assert body['prompt']==item['h3_prompt'] and not body.get('enhance_prompt',False)
  assert body['num_inference_steps']==50 and body['seed']==42 and body['flow_shift']==12 and body['audio_flow_shift']==3
  assert body['task']=='ref2va' and body['quality']=='lossless' and body['reference_resize_mode']=='max'
  assert len(body['conditions'])==1+len(prior['references'])
  assert body['conditions'][0]['type']=='video' and body['conditions'][0]['uri']==Path(item['prepared_source']['path']).as_uri()
  motion_iter=iter(item['prepared_motion'])
  for c,r in zip(body['conditions'][1:],prior['references'],strict=True):
   expected={'type':'video','role':'reference','uri':Path(next(motion_iter)['path']).as_uri(),'start_time_seconds':0.0} if r['type']=='motion' else {'type':'image','role':'reference','uri':Path(r['absolute_path']).as_uri()}
   assert c==expected
  audit=read(ROOT/'metadata/input_audits'/f'{sid}_source.json');assert audit['frames_full_decoded']==243 and audit['sha256']==state['contract']['source_sha256']
  perf=read(ROOT/'metadata/perf'/f'{sid}.json');assert len(perf['denoise_steps_ms'])==50
  stages={s['name']:s['duration_ms'] for s in perf['steps']}
  assert stages['MiniMaxH3TextEncodingStage']>0 and stages['MiniMaxH3VisualEncodingStage']>0 and stages['MiniMaxH3DenoisingStage']>0
  assert state['server_response'].get('revised_prompt') is None
  assert abs(float(state['server_response']['seconds'])-243/24)<1e-5
  checks.append({'id':sid,'native_updates':50,'source_and_reference_routes':'Qwen + VisualVAE native stages; frozen material URIs','text_encoding_ms':stages['MiniMaxH3TextEncodingStage'],'visual_encoding_ms':stages['MiniMaxH3VisualEncodingStage'],'denoise_ms':stages['MiniMaxH3DenoisingStage'],'decode_ms':stages['MiniMaxH3DecodingStage'],'frames':243,'fps':24,'no_gt':True,'context_ir':False})
 atomic(ROOT/'metadata'/f'protocol_gate_{stage}.json',{'status':'passed','checked_unix':time.time(),'checks':checks,'scope':'actual requests, source identity, native encoder stages, 50 updates and full output decode; not human quality'})
 print('PROTOCOL_GATE_PASSED',stage,len(ids),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','first','all','completed'],required=True);a=p.parse_args();gate(a.stage)
