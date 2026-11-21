"""CPU-only input-cache/contract checks before a new, isolated A849 submission."""
import sys
from native_cache import *

def main():
 from compact_text_cache import unpack_text_features
 import torch,yaml
 require=read(ROOT/'package/validation.json');assert require['status']=='passed'
 digest=sha(ROOT/'package/inputs.jsonl');index=read(ROOT/'metadata/native_cache_index.json')
 cfg=yaml.safe_load(CONFIG.read_text());assert cfg['reference_protocol']=='reference_positions_v7_refs_source_target'
 lock=read(ROOT/'package/model_interfaces.json');assert sha(CHECKPOINT)==lock['a849']['sha256']
 for m in lock['v7']['modules']:assert sha(m['file'])==m['sha256']
 derived=0;existing=0
 for item in rows(ROOT/'package/inputs.jsonl'):
  e=index[item['id']];assert e['input_sha256']==digest
  text=unpack_text_features(checked_load(e['text']),expected_contract=e['text']['receipt']['contract'])
  assert set(text)=={'video_prompt_embeds','prompt_attention_mask'}
  assert len(e['references'])==len(item['references'])
  for ref,m in [(e['source'],item['source']),*zip(e['references'],item['references'])]:
   if ref.get('derived'):derived+=1;continue
   payload=checked_load(ref);existing+=1
   assert payload['latents'].shape[0]==128 and payload['fps']==12
   if 'sample_frame_indices' in m:
    w,h=item['canvas_wh'];assert list(payload['latents'].shape)==[128,(item['num_frames']-1)//8+1,h//32,w//32]
   else:assert payload['latents'].shape[1]==1
 write(ROOT/'metadata/a849_preflight.json',{'status':'passed','samples':len(index),'input_manifest_sha256':digest,'checkpoint_sha256':lock['a849']['sha256'],'existing_input_media_cache_payloads_verified':existing,'input_video_payloads_to_encode_on_gpu':derived,'text_payloads_verified':300,'gt_or_target_caption_loaded':False,'inference_status':'not_run','old_outputs_reused':False})
 print('A849_PREFLIGHT_PASSED',existing,derived,flush=True)
if __name__=='__main__':main()
