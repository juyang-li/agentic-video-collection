"""New source/reference/text encodings only; never loads a target video."""
import argparse
import sys
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths
add_script_paths()

def save_payload(path,payload,identity,details):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.pt');torch.save(payload,tmp);tmp.replace(path)
    atomic(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path),details=details))

def checked_load(path):
    import torch
    receipt=read(path.with_suffix('.json'))
    assert receipt['identity']['manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    assert receipt['sha256']==sha(path)
    return torch.load(path,map_location='cpu',weights_only=True)

def load_sample(item):
    from compact_text_cache import unpack_text_features
    prefix=ROOT/'cache'/item['id']
    source=checked_load(prefix/'source.pt')
    text=prefix/'text.pt';receipt=read(text.with_suffix('.json'))
    conditions=unpack_text_features(checked_load(text),expected_contract=receipt['details']['contract'])
    refs=[dict(type=r['type'],view=r.get('view',''),index=i,latent=checked_load(prefix/f'ref_{i}.pt')) for i,r in enumerate(item['references'])]
    return dict(source=source,references=refs,conditions=conditions)

def run(stage,shard):
    require_job()
    import numpy as np
    import torch
    from PIL import Image
    from ltx_trainer.model_loader import load_video_vae_encoder,load_text_encoder,load_embeddings_processor,embedding_weight_paths
    from encoding_helpers import weight_paths
    from native_image_cache_contract import prepare_image
    from encode_native_dataset import encode_text
    record,weights=weight_paths(NATIVE/'runtime/models/native_model_revision.json',NATIVE/'runtime/verification/native_weights_verified.json')
    model=None;items=rows(ROOT/'metadata/inputs.jsonl')[shard::4]
    for item in items:
        prefix=ROOT/'cache'/item['id']
        identity=dict(manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),weights=record['revision'],script_sha256=sha(__file__))
        jobs=[('text',None)] if stage=='text' else [('source',item['source']),*[(f'ref_{i}',r) for i,r in enumerate(item['references'])]]
        for role,r in jobs:
            dst=prefix/(role+'.pt')
            if dst.exists():
                old=read(dst.with_suffix('.json'))['identity']
                assert old['manifest_sha256']==identity['manifest_sha256'] and old['weights']==identity['weights']
                checked_load(dst);continue
            if model is None:
                if stage=='text':
                    encoder=load_text_encoder(weights['text_encoders'],device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
                    processor=load_embeddings_processor(embedding_weight_paths(weights['diffusion_models'],weights['text_encoders']),gemma_model_path=weights['text_encoders'],device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
                    processor.audio_connector=None
                    model=(encoder,processor)
                else:model=load_video_vae_encoder(weights['vae'],device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
            with torch.inference_mode():
                if stage=='text':
                    payload,details=encode_text(dict(prompt_en=item['prompt']['en'],caption_key=item['id']),{'weights':record},*model,verify_connector=True)
                else:
                    assert sha(r['absolute_path'])==r['sha256']
                    if role=='source':
                        prepared=ROOT/'inputs/prepared'/item['id']
                        receipt=read(prepared/'pixels.json')
                        assert receipt['input_manifest_sha256']==identity['manifest_sha256']
                        paths=[prepared/f'{i:03d}.png' for i in range(49)]
                        assert [sha(p) for p in paths]==receipt['frame_sha256']
                        frames=[Image.open(p).convert('RGB') for p in paths];details=receipt
                    else:
                        frame,details=prepare_image(Image.open(r['absolute_path']).convert('RGB'));frames=[frame]
                    px=torch.from_numpy(np.stack([np.asarray(f) for f in frames])).float().div_(255).sub_(.5).div_(.5).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)
                    latent=model(px)
                    assert torch.isfinite(latent).all()
                    assert latent.shape[2]==(7 if role=='source' else 1)
                    payload=dict(latents=latent[0].cpu().contiguous(),num_frames=latent.shape[2],height=latent.shape[3],width=latent.shape[4],fps=12)
                    del px,latent
                save_payload(dst,payload,identity,details)
            print(item['id'],role,'complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['media','text'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run(a.stage,a.shard)
