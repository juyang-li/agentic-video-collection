"""Pinned Diffusers WanVACE worker: 720p, 49 frames, 40 UniPC steps, seed42."""
import argparse
import inspect
import sys
import time
import traceback
from common import *

def work(stage,shard,control=False):
    require_job(exact=True)
    import av
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan,WanVACEPipeline,UniPCMultistepScheduler
    from media import video_frames,write_preview
    from native_worker import SMOKE,SECOND
    ready=read(ROOT/'runtime/vace_ready.json');modelpath=ready['vace']['path']
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    ids=SMOKE if stage=='smoke' else SECOND if stage=='second' else [i['id'] for i in inputs]
    selected=[r for r in inputs if r['id'] in ids][shard::4]
    if control:selected=[r for r in inputs if r['id'] in ['identity_01','whole_outfit_01']][shard::4]
    if not selected:return
    assert torch.cuda.device_count()==1
    t=time.monotonic()
    vae=AutoencoderKLWan.from_pretrained(modelpath,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=WanVACEPipeline.from_pretrained(modelpath,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=5.)
    pipe.to('cuda')
    assert pipe.vae.dtype==torch.float32 and pipe.transformer.dtype==torch.bfloat16
    load_seconds=time.monotonic()-t
    for item in selected:
        sid=item['id']+('_swapref' if control else '')
        output=ROOT/'outputs/vace'/f'{sid}.mp4';status_path=output.with_suffix('.json')
        condition_path=ROOT/'conditions'/item['id']/'condition.json'
        record={'id':sid,'model':'vace','status':'running','started_unix':time.time(),'model_load_seconds':load_seconds}
        contract={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'worker_sha256':sha(__file__),'model_repo':ready['vace']['repo_id'],'revision':ready['vace']['revision'],'pipeline':ready['pipeline'],'condition_sha256':sha(condition_path) if condition_path.exists() else None,'steps':40,'seed':42,'cfg':5,'conditioning_scale':1.,'flow_shift':5.,'scheduler':'UniPCMultistepScheduler','vae_dtype':'float32','transformer_dtype':'bfloat16','canvas_wh':item['canvas_wh'],'num_frames':49,'fps':12,'input_video_mode':('official_pose_control' if item['group']=='action' else 'raw_source_camera_control' if item['group']=='camera' else 'official_repaint_gray128')}
        if status_path.exists() and read(status_path).get('status')=='complete':
            old=read(status_path)
            compatible=old['contract']==contract
            if not compatible and item['group'] in ('action','camera'):
                compatibility=read(ROOT/'metadata/vace_control_legacy_compat.json')
                expected_legacy=dict(contract)
                expected_legacy.pop('input_video_mode')
                expected_legacy['worker_sha256']=compatibility['legacy_worker_sha256']
                compatible=(item['group'] in compatibility['permitted_groups'] and old['contract']==expected_legacy and sha(compatibility['snapshot'])==compatibility['legacy_worker_sha256'])
            assert compatible and sha(output)==old['output_sha256']
            continue
        record['contract']=contract;atomic(status_path,record)
        try:
            c=read(condition_path);assert c['status']=='complete'
            video,_=video_frames(item,item['source'])
            mask=[Image.open(p).convert('RGB') for p in c['mask_frames']]
            assert len(mask)==49 and all(sha(p)==h for p,h in zip(c['mask_frames'],c['mask_sha256'],strict=True))
            refs=[];reference_roles=[]
            if c.get('pose_video'):
                refs=[video[0],video[24],video[48]]
                reference_roles=['source_frame_0','source_frame_24','source_frame_48']
                video=[Image.open(ROOT/'conditions'/item['id']/f'pose_{i:03d}.png').convert('RGB') for i in range(49)]
            else:
                ref_item=item
                if control:
                    ref_item=next(r for r in inputs if r['group']==item['group'] and r['id']!=item['id'] and r['references']!=item['references'])
                    record['control_reference_from']=ref_item['id']
                for r in ref_item['references']:
                    assert r['type']!='motion'
                    refs.append(Image.open(r['absolute_path']).convert('RGB'))
                    reference_roles.append(r['type']+':'+r.get('view',''))
            # Official InpaintingVideoAnnotator erases editable RGB content before VACE encoding.
            # Pose and the explicitly specified raw-source camera control remain un-erased.
            if contract['input_video_mode']=='official_repaint_gray128':
                erased=[]
                for frame,mask_frame in zip(video,mask,strict=True):
                    original_pixels=np.asarray(frame)
                    pixels=original_pixels.copy()
                    editable=np.asarray(mask_frame)[:,:,0]>127
                    pixels[editable]=128
                    assert np.array_equal(pixels[~editable],original_pixels[~editable])
                    assert np.all(pixels[editable]==128)
                    erased.append(Image.fromarray(pixels))
                video=erased
                preview=ROOT/'conditions'/item['id']/f'conditioning_input_{sid}.mp4'
                write_preview(video,preview,12)
                record['conditioning_input_preview']=str(preview)
                record['official_repaint_erasure']={'masked_rgb':128,'outside_mask':'bitwise_original','source':'official InpaintingVideoAnnotator.apply_seg_mask; Diffusers PIL convention128'}
            w,h=item['canvas_wh']
            # Call official preprocessor once for a recorded mask and reference audit.
            pv,pm,pr=pipe.preprocess_conditions(video=video,mask=mask,reference_images=refs or None,batch_size=1,height=h,width=w,num_frames=49,dtype=torch.float32,device='cpu')
            assert tuple(pv.shape)==(1,3,49,h,w) and tuple(pm.shape)==(1,3,49,h,w)
            assert len(pr)==1 and len(pr[0])==len(refs)
            assert set(torch.unique(pm).tolist())<={0.,1.}
            original_mask=np.asarray(mask[0])[:,:,0]>127
            assert np.array_equal(pm[0,0,0].numpy()>0.5,original_mask),'mask direction or pixel mapping changed'
            record['protocol_audit']={'video_shape':list(pv.shape),'mask_shape':list(pm.shape),'white_is_editable':True,'mask_range':[float(pm.min()),float(pm.max())],'reference_argument_order':reference_roles,'official_prepend_latent_order':reference_roles[::-1],'reference_count':len(refs),'reference_canvases':[list(r.shape) for r in pr[0]],'pose_adapter':bool(c.get('pose_video')),'no_gt':True,'source_condition':contract['input_video_mode']}
            record['adapted_prompt']=c['adapted_prompt'];record['original_instruction']=item['prompt']['en']
            del pv,pm,pr;atomic(status_path,record)
            # Preserve every original clause; reject silent tokenizer truncation.
            token_count=len(pipe.tokenizer(record['adapted_prompt'])['input_ids'])
            assert token_count<=512,f'Prompt truncated: {token_count}'
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            with torch.inference_mode():
                generated=pipe(video=video,mask=mask,reference_images=refs or None,prompt=record['adapted_prompt'],negative_prompt='blurry, excessive noise, compression artifacts, flickering, distorted proportions, deformed facial features, extra limbs, disfigured hands, unnatural transitions',height=h,width=w,num_frames=49,num_inference_steps=40,guidance_scale=5.,conditioning_scale=1.,generator=torch.Generator(device='cuda').manual_seed(42),output_type='pil',max_sequence_length=512).frames[0]
            torch.cuda.synchronize();record['generation_seconds']=time.monotonic()-t;record['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            assert len(generated)==49 and all(f.size==(w,h) for f in generated)
            tmp=output.with_suffix('.part.mp4');write_preview(generated,tmp,12)
            with av.open(str(tmp)) as con:assert sum(1 for _ in con.decode(video=0))==49
            tmp.replace(output)
            record.update(status='complete',output=str(output),output_sha256=sha(output),full_decode_frames=49,completed_unix=time.time())
            del generated,video,mask,refs
        except Exception:
            record.update(status='failed',error=traceback.format_exc(),completed_unix=time.time());traceback.print_exc();torch.cuda.empty_cache()
        atomic(status_path,record)
        print(json.dumps({'id':sid,'model':'vace','status':record['status'],'seconds':record.get('generation_seconds')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','second','all'],required=True);p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--control',action='store_true');a=p.parse_args();work(a.stage,a.shard,a.control)
