"""CPU acceptance of actual receipts against the frozen per-model protocol."""
from common import *
from contracts import digest,generation_input

def main():
 items=rows(ROOT/'metadata/inputs.jsonl');manifest=sha(ROOT/'metadata/inputs.jsonl');runtime=sha(ROOT/'metadata/h3_runtime_contract.json');checks=[]
 config_sha=sha(CONFIG);weights_record_sha=sha(NATIVE/'runtime/models/native_model_revision.json');checkpoint_sha=sha(CHECKPOINT)
 worker_shas={name:sha(ROOT/'scripts'/name)for name in ['native_worker.py','vace_worker.py','h3_worker.py']}
 for item in items:
  sid=item['id'];condition=read(ROOT/'conditions'/sid/'condition.json')
  for model in ['ltx_base','a849','vace']:
   receipt=ROOT/'outputs'/model/f'{sid}.json';r=read(receipt);c=r['contract'];assert r['status']=='complete'and c['steps']==40 and c['seed']==42 and c['canvas_wh']==item['canvas_wh']and c['num_frames']==49 and c['fps']==12
   if item['cohort']=='candidate28':
    assert c['input_manifest_sha256']==manifest and c['worker_sha256']==worker_shas['vace_worker.py'if model=='vace'else'native_worker.py']
   if model=='vace':
    assert all(c[k]==v for k,v in dict(cfg=5,conditioning_scale=1.,flow_shift=5.,scheduler='UniPCMultistepScheduler',vae_dtype='float32',transformer_dtype='bfloat16').items());assert r['adapted_prompt']==condition['adapted_prompt']and r['original_instruction']==item['prompt']['en'];assert r['official_repaint_erasure']['masked_rgb']==128
    if item['cohort']=='candidate28':assert c['condition_sha256']==sha(ROOT/'conditions'/sid/'condition.json')
   else:
    assert all(c[k]==v for k,v in dict(cfg=3,stg=1,stg_blocks=[28],rescale=.7,ref_cfg=1.,reference_protocol='reference_positions_v7_refs_source_target').items());assert c['config_sha256']==config_sha and c['native_weights_record_sha256']==weights_record_sha
    if model=='a849':assert Path(c['checkpoint']).resolve()==CHECKPOINT.resolve()and c['checkpoint_sha256']==checkpoint_sha
    else:assert c['checkpoint']is None
    assert r['protocol_audit']['optional_reference_count']==len(item['references'])and r['protocol_audit']['no_gt']is True
   checks.append(dict(id=sid,model=model,receipt_sha256=sha(receipt),output_sha256=r['output_sha256'],effective_input_sha256=digest(generation_input(item,model,condition if model=='vace'else None))))
  receipt=ROOT/'outputs/h3_pdd8_native'/f'{sid}.json';r=read(receipt);c=r['contract'];assert r['status']=='complete'and c['runtime_contract_sha256']==runtime and c['request_sha256']==digest(read(ROOT/'metadata/h3_requests'/f'{sid}.json'))
  assert all(c[k]==v for k,v in dict(steps=8,seed=42,tp_size=2,ulysses_degree=2,native_frames=107,native_fps=24).items())
  if item['cohort']=='candidate28':assert c['manifest_sha256']==manifest and c['worker_sha256']==worker_shas['h3_worker.py']
  else:
   proof=read(ROOT/'metadata/h3_reuse'/f'{sid}.json');assert proof['status']=='passed'and proof['new_manifest_sha256']==manifest and proof['original_receipt_sha256']==sha(receipt)
  checks.append(dict(id=sid,model='h3_pdd8',receipt_sha256=sha(receipt),output_sha256=r['output_sha256'],request_sha256=c['request_sha256']))
 assert len(checks)==464
 atomic(ROOT/'metadata/protocol_acceptance.json',dict(status='passed',outputs=len(checks),manifest_sha256=manifest,runtime_contract_sha256=runtime,checks=checks))
 print('All464 actual generation protocols accepted',flush=True)
if __name__=='__main__':main()
