"""Refresh only exact-input parent outputs; current original96 can finish asynchronously."""
from common import *
from contracts import generation_input,digest
from reuse import link

def main():
 manifest=sha(ROOT/'metadata/inputs.jsonl');origins=read(ROOT/'metadata/input_origins.json');proofs=[];missing=[]
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  if item['cohort']!='retained88':continue
  root,sid=origins[item['id']];root=Path(root);prior=next(r for r in rows(root/'metadata/inputs.jsonl')if r['id']==sid)
  for model in ['ltx_base','a849','vace']:
   path=root/'outputs'/model/f'{sid}.mp4';receipt=path.with_suffix('.json')
   r=read(receipt)if receipt.exists()else {'status':'missing'}
   if r['status']!='complete':missing.append([item['id'],model]);continue
   old_c=read(root/'conditions'/sid/'condition.json')if model=='vace'else None;new_c=read(ROOT/'conditions'/item['id']/'condition.json')if model=='vace'else None
   effective=generation_input(item,model,new_c);assert effective==generation_input(prior,model,old_c)
   assert sha(path)==r['output_sha256'] and r['contract']['steps']==40 and r['contract']['seed']==42
   contract=r['contract'];assert contract['canvas_wh']==item['canvas_wh']and contract['num_frames']==49 and contract['fps']==12
   if model=='vace':
    assert all(contract[k]==v for k,v in dict(cfg=5,flow_shift=5.,conditioning_scale=1.,scheduler='UniPCMultistepScheduler',vae_dtype='float32',transformer_dtype='bfloat16').items())
    assert r['adapted_prompt']==new_c['adapted_prompt']and r['original_instruction']==item['prompt']['en']
    assert r['official_repaint_erasure']['masked_rgb']==128
   else:
    assert all(contract[k]==v for k,v in dict(cfg=3,stg=1,stg_blocks=[28],rescale=.7,ref_cfg=1.).items())
    assert contract['reference_protocol']=='reference_positions_v7_refs_source_target'and contract['config_sha256']==sha(CONFIG)
    assert contract['native_weights_record_sha256']==sha(NATIVE/'runtime/models/native_model_revision.json')
    assert contract['checkpoint']is None if model=='ltx_base'else Path(contract['checkpoint']).resolve()==CHECKPOINT.resolve()
   link(path,ROOT/'outputs'/model/f"{item['id']}.mp4");link(receipt,ROOT/'outputs'/model/f"{item['id']}.json")
   proofs.append(dict(id=item['id'],model=model,output_sha256=r['output_sha256'],effective_input_sha256=digest(effective),inference_protocol_verified=True,actual_vace_prompt_matches_frozen_condition=True if model=='vace'else None,original_manifest=str(root/'metadata/inputs.jsonl'),original_manifest_sha256=sha(root/'metadata/inputs.jsonl'),original_receipt=str(receipt),original_receipt_sha256=sha(receipt),reason='Same effective source, ordered references, full instruction, temporal/canvas settings and actual VACE masks/adapted prompt; unchanged40-step inference protocol'))
 atomic(ROOT/'metadata/reuse_outputs.json',dict(input_manifest_sha256=manifest,complete=len(proofs),expected=264,reused=proofs,parent_outputs_pending=missing))
 print('Exact outputs reused',len(proofs),'/264; pending parents',len(missing),flush=True)
if __name__=='__main__':main()
