"""Resident native worker for base or A849; per-sample failures are persisted."""
import argparse
import sys
import time
import traceback
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths
add_script_paths()

SMOKE=['identity_01','whole_outfit_01','action_01','camera_01']
SECOND=[g+'_01' for g in GROUPS if g+'_01' not in SMOKE]


def work(model_name, stage, shard, control=False):
    require_job(exact=True)
    import torch
    import yaml
    from accelerate.utils import send_to_device
    from native_cached_dataset import collate_single_sample
    from native_reference_inference import load_inference_models,generate_latent,encode_negative_context,build_generation_state
    from native_reference_strategy import reference_segments
    from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
    from ltx_pipelines.utils.media_io import encode_video
    from encoding_helpers import weight_paths
    from native_cache import load_sample
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    selected=SMOKE if stage=='smoke' else SECOND if stage=='second' else [r['id'] for r in inputs]
    tasks=[r for r in inputs if r['id'] in selected][shard::4]
    if control:
        tasks=[r for r in inputs if r['id'] in ['identity_01','whole_outfit_01']][shard::4]
    if not tasks: return
    cfg=yaml.safe_load(CONFIG.read_text())
    assert cfg['reference_protocol']=='reference_positions_v7_refs_source_target'
    weights_record,weights=weight_paths(NATIVE/'runtime/models/native_model_revision.json',NATIVE/'runtime/verification/native_weights_verified.json')
    ckpt=CHECKPOINT if model_name=='a849' else None
    if ckpt:
        receipt=read(ckpt.with_suffix('.contract.json'))
        expected=receipt['files'][ckpt.name]
        assert ckpt.stat().st_size==expected['bytes'] and sha(ckpt)==expected['sha256']
        assert receipt['identity']['config']['lora']==cfg['lora']
        assert receipt['identity']['config']['reference_protocol']==cfg['reference_protocol']
    settings=read(NATIVE/'configs/inference/stronger_guidance/inference.json')['settings']
    start=time.monotonic()
    device=torch.device('cuda:0')
    negative=encode_negative_context(weights,settings['negative_prompt'],device=device)
    model,processor,scale=load_inference_models(weights,cfg['lora'],checkpoint=ckpt,protocol=cfg['reference_protocol'],device=device,ref_cfg=1.)
    decoder=VideoDecoder(str(weights['vae']),dtype=torch.bfloat16,device=device)
    negative=negative.to(device)
    load_seconds=time.monotonic()-start
    for item in tasks:
        sid=item['id']+('_swapref' if control else '')
        out=ROOT/'outputs'/model_name/f'{sid}.mp4'
        status_path=out.with_suffix('.json')
        contract={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'checkpoint':str(ckpt) if ckpt else None,'checkpoint_sha256':sha(ckpt) if ckpt else None,'config_sha256':sha(CONFIG),'worker_sha256':sha(__file__),'native_inference_sha256':sha(NATIVE/'scripts/common/native_reference_inference.py'),'reference_protocol':cfg['reference_protocol'],'steps':40,'seed':42,'cfg':3,'stg':1,'stg_blocks':[28],'rescale':.7,'ref_cfg':1,'fps':12,'num_frames':49,'canvas_wh':item['canvas_wh']}
        if status_path.exists():
            old=read(status_path)
            if old.get('status')=='complete':
                assert old['contract']==contract and out.is_file() and sha(out)==old['output_sha256']
                continue
        state={'id':sid,'model':model_name,'status':'running','contract':contract,'model_load_seconds':load_seconds,'started_unix':time.time()}
        atomic(status_path,state)
        try:
            sample=load_sample(item)
            if control:
                other=next(r for r in inputs if r['group']==item['group'] and r['id']!=item['id'] and r['references']!=item['references'])
                sample['references']=load_sample(other)['references']
                state['control_reference_from']=other['id']
                state['control_change']='Only optional reference set; source, text, seed and all generation settings fixed'
            batch=send_to_device(collate_single_sample([sample]),device)
            assert 'video_latents' not in batch
            segments=[{'role':role,'slot':slot,'shape':list(raw['latents'].shape)} for role,slot,raw in reference_segments(batch,protocol_version=cfg['reference_protocol'])]
            # Build exactly the native state once for a recorded layout audit.
            tools,audit_state=build_generation_state(batch['source'],batch['references'],scale_factors=scale,protocol=cfg['reference_protocol'],seed=42)
            n=tools.patchifier.get_token_count(tools.target_shape)
            assert audit_state.latent.shape[1]==n+sum(torch.tensor(s['shape'][2:]).prod().item() for s in segments)
            state['protocol_audit']={'segments':segments,'optional_reference_count':len(batch['references']),'reference_order':[(r['type'],r['view']) for r in sample['references']], 'packed_tokens':audit_state.latent.shape[1],'target_tokens':n,'target_last':True,'no_gt':True,'time_position_min':float(audit_state.positions[:,0].min()),'time_position_max':float(audit_state.positions[:,0].max())}
            del tools,audit_state
            atomic(status_path,state)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            latent=generate_latent(model,processor,source=batch['source'],references=batch['references'],conditions=batch['conditions'],scale_factors=scale,protocol=cfg['reference_protocol'],seed=42,steps=40,negative_context=negative,cfg_scale=3,stg_scale=1,guidance_rescale=.7,stg_blocks=(28,),guidance_batch_size=1,ref_cfg=1.)
            torch.cuda.synchronize();state['denoise_seconds']=time.monotonic()-t;state['denoise_peak_allocated_bytes']=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
            t=time.monotonic();tmp=out.with_suffix('.part.mp4')
            with torch.inference_mode():
                frames=decoder(latent,generator=torch.Generator(device=device).manual_seed(42))
                encode_video(frames,fps=12,audio=None,output_path=str(tmp),video_chunks_number=1,thread_count=2)
            import av
            with av.open(str(tmp)) as c:
                decoded=list(c.decode(video=0));w,h=item['canvas_wh']
                assert len(decoded)==49 and all((f.width,f.height)==(w,h) for f in decoded)
            tmp.replace(out)
            torch.cuda.synchronize()
            state.update(status='complete',decode_seconds=time.monotonic()-t,decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),output_sha256=sha(out),output=str(out),full_decode_frames=49,completed_unix=time.time())
            del batch,sample,latent,frames,decoded
        except Exception:
            state.update(status='failed',error=traceback.format_exc(),completed_unix=time.time())
            traceback.print_exc()
            torch.cuda.empty_cache()
        atomic(status_path,state)
        print(json.dumps({'id':sid,'model':model_name,'status':state['status'],'denoise_seconds':state.get('denoise_seconds')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ltx_base','a849'],required=True);p.add_argument('--stage',choices=['smoke','second','all'],required=True);p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--control',action='store_true');a=p.parse_args();work(a.model,a.stage,a.shard,a.control)
