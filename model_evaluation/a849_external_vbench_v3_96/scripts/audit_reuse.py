"""CPU proof of input equivalence; preserve original generation receipts verbatim."""
from common import *
from contracts import generation_input,MODELS,digest
from reuse import link
OLD=ROOT.parent/'a849_external_vbench_v2_96';DIAG=Path(read(ROOT/'metadata/identity_diagnostic_location.json')['root']) if (ROOT/'metadata/identity_diagnostic_location.json').exists() else ROOT.parent/'a849_external_vbench_v3_identity'

def main():
 items=rows(ROOT/'metadata/inputs.jsonl');proofs=[]
 for item in items:
  if item['group']=='identity':root=DIAG;sid=item['id']
  elif item.get('old_id'):root=OLD;sid=item['old_id']
  else:continue
  old=next(r for r in rows(root/'metadata/inputs.jsonl') if r['id']==sid)
  if item['mask_spec']!=old['mask_spec']:continue
  old_px=read(root/'inputs/prepared'/sid/'pixels.json');new_px=read(ROOT/'inputs/prepared'/item['id']/'pixels.json')
  assert old_px['frame_sha256']==new_px['frame_sha256'] and old_px['pixel_audit']==new_px['pixel_audit']
  for model in MODELS:
   src=root/'outputs'/model/f'{sid}.mp4';receipt=src.with_suffix('.json')
   if not receipt.exists():continue
   r=read(receipt)
   if r.get('status')!='complete':continue
   new_c=read(ROOT/'conditions'/item['id']/'condition.json') if model=='vace' else None
   old_c=read(root/'conditions'/sid/'condition.json') if model=='vace' else None
   new_input=generation_input(item,model,new_c);old_input=generation_input(old,model,old_c)
   if new_input!=old_input:continue
   assert sha(src)==r['output_sha256'] and r['contract']['steps']==40 and r['contract']['seed']==42
   contract=r['contract']
   assert contract['canvas_wh']==item['canvas_wh'] and contract['fps']==12 and contract['num_frames']==49
   if model=='vace':
    assert all(contract[k]==v for k,v in dict(cfg=5,flow_shift=5.,conditioning_scale=1.,scheduler='UniPCMultistepScheduler',vae_dtype='float32',transformer_dtype='bfloat16').items())
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json');assert contract['revision']==ready['vace']['revision']
    assert r['official_repaint_erasure']['masked_rgb']==128
    for path,h in zip(old_c['mask_frames'],old_c['mask_sha256'],strict=True):assert sha(path)==h
   else:
    assert all(contract[k]==v for k,v in dict(cfg=3,stg=1,stg_blocks=[28],rescale=.7,ref_cfg=1.).items())
    assert contract['reference_protocol']=='reference_positions_v7_refs_source_target' and contract['config_sha256']==sha(CONFIG)
    assert contract['native_weights_record_sha256']==sha(NATIVE/'runtime/models/native_model_revision.json')
    assert (contract['checkpoint'] is None) if model=='ltx_base' else contract['checkpoint']==str(CHECKPOINT)
   dst=ROOT/'outputs'/model/f"{item['id']}.mp4";link(src,dst);link(receipt,dst.with_suffix('.json'))
   proofs.append(dict(id=item['id'],old_id=sid,model=model,old_manifest=str(root/'metadata/inputs.jsonl'),old_manifest_sha256=sha(root/'metadata/inputs.jsonl'),original_generation_manifest_sha256=r['contract']['input_manifest_sha256'],original_generation_receipt=str(receipt),original_generation_receipt_sha256=sha(receipt),source_frame_sha256=new_px['frame_sha256'],effective_input=new_input,effective_input_sha256=digest(new_input),output_sha256=r['output_sha256'],input_and_protocol_equivalent=True))
 atomic(ROOT/'metadata/reuse_outputs.json',dict(new_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),reused=proofs,policy='Original output and receipt bytes unchanged. Exact effective source, prompt, ordered reference, temporal/canvas settings, task mask spec and VACE actual masks/adapted prompt verified.'))
 print('Verified reused outputs',len(proofs),flush=True)
if __name__=='__main__':main()
