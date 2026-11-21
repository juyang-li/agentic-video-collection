"""Single resident BF16 distilled transformer, no LoRA; fail closed on technical errors."""
from common import *
import time, traceback, importlib.metadata, socket

def run():
    assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ.get('SLURM_JOB_ID')
    import torch, av, gc
    from accelerate.utils import send_to_device
    from native_cached_dataset import collate_single_sample
    from compact_text_cache import unpack_text_features, validate_connector
    from ltx_trainer.model_loader import load_transformer,load_embeddings_processor,embedding_weight_paths,read_video_scale_factors
    from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
    from ltx_pipelines.utils.media_io import encode_video
    from sampling import generate
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
    free,total=torch.cuda.mem_get_info();assert total-free<2*1024**3,'GPU is busy'
    assert read(ROOT/'metadata/weights.json')['status']=='passed'
    assert read(ROOT/'metadata/cpu_tests.json')['status']=='passed'
    assert read(ROOT/'metadata/cpu_tests.json')['sampling_sha256']==sha(ROOT/'scripts/sampling.py')
    assert read(ROOT/'metadata/timeline_audit.json')['status']=='passed'
    assert read(ROOT/'metadata/pairing.json')['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    index=read(ROOT/'metadata/cache_index.json');inputs={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')}
    _,paths=weights();paths['diffusion_models']=DISTILLED
    device=torch.device('cuda:0');started=time.monotonic()
    atomic(ROOT/'metadata/run_status.json',{'status':'loading','job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'order':ORDER})
    model=load_transformer(paths['diffusion_models'],device='cpu',dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
    assert not any('lora_' in name for name,_ in model.named_parameters())
    processor=load_embeddings_processor(checkpoint_path=embedding_weight_paths(paths['diffusion_models'],paths['text_encoders']),gemma_model_path=paths['text_encoders'],device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    processor.feature_extractor=None;processor.audio_connector=None
    decoder=VideoDecoder(str(paths['vae']),dtype=torch.bfloat16,device=device)
    scale=read_video_scale_factors(paths['vae']);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    load_seconds=time.monotonic()-started
    software={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','peft','safetensors','av','numpy']}
    atomic(ROOT/'metadata/runtime.json',{'model_load_seconds':load_seconds,'gpu_model':torch.cuda.get_device_name(0),'software':software,'cuda':torch.version.cuda,'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'worker_sha256':sha(__file__),'sampling_sha256':sha(ROOT/'scripts/sampling.py'),'native_inference_sha256':sha(NATIVE/'scripts/common/native_reference_inference.py'),'sampler_sha256':sha(NATIVE/'runtime/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py')})
    def checked(ref):
        p=resolve_path(ref['path']);receipt=read(p.with_suffix('.json')) if ref.get('derived') else ref['receipt']['artifact']
        assert sha(p)==receipt['sha256'];return torch.load(p,map_location='cpu',weights_only=True)
    for sid in ORDER:
        item=inputs[sid];entry=index[sid];out=ROOT/f'outputs/{sid}.mp4';status_path=out.with_suffix('.json')
        assert not status_path.exists(),'Explicit new-run decision needed; no automatic retries'
        state={'id':sid,'status':'running','started_unix':time.time(),'model_load_seconds':load_seconds,'contract':{'experiment':'LTX2.5 distilled＋DataMake V7 单阶段实验','model_revision':'49b093a39c19169a8a81721dcc541e07e945232f','dit_sha256':DISTILLED_SHA,'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'protocol':PROTOCOL,'sigmas':SIGMAS,'steps':8,'cfg':1,'stg':0,'rescale':0,'ref_cfg':1,'seed':42,'noise_seed':10042,'eta':1,'s_noise':1,'sampler':'official euler_ancestral_denoising_loop','lora':None,'dtype':'BF16','fps':12,'num_frames':121,'canvas_wh':item['canvas_wh'],'stages':1},'gpu_model':torch.cuda.get_device_name(0),'software':software}
        atomic(status_path,state);atomic(ROOT/'metadata/run_status.json',{'status':'running','current_id':sid,'job_id':os.environ['SLURM_JOB_ID']})
        try:
            start=time.monotonic();validate_connector(processor.video_connector,entry['text']['receipt']['contract'])
            sample={'source':checked(entry['source']),'references':[{k:r[k] for k in ('type','view','index')}|{'latent':checked(r)} for r in entry['references']],'conditions':unpack_text_features(checked(entry['text']),expected_contract=entry['text']['receipt']['contract'])}
            w,h=item['canvas_wh'];assert list(sample['source']['latents'].shape)==[128,16,h//32,w//32]
            batch=send_to_device(collate_single_sample([sample]),device);assert 'video_latents' not in batch
            state['cache_load_seconds']=time.monotonic()-start
            def progress(steps):
                state['step_indices']=steps;atomic(status_path,state)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            latent,audit=generate(model,processor,batch,scale,progress)
            torch.cuda.synchronize();state['denoise_seconds']=time.monotonic()-start;state['protocol_audit']=audit
            state['denoise_peak_allocated_bytes']=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            # Materialize official decoder chunks so pure VAE time and encode/write time are separate.
            with torch.inference_mode(): frames=list(decoder(latent,generator=torch.Generator(device=device).manual_seed(42)))
            torch.cuda.synchronize();state['vae_decode_seconds']=time.monotonic()-start;start=time.monotonic()
            tmp=out.with_suffix('.part.mp4');encode_video(iter(frames),fps=12,audio=None,output_path=str(tmp),video_chunks_number=len(frames),thread_count=2)
            torch.cuda.synchronize();state['write_seconds']=time.monotonic()-start
            state['decode_seconds']=state['vae_decode_seconds']+state['write_seconds'];start=time.monotonic()
            with av.open(str(tmp)) as c:
                n=0
                for f in c.decode(video=0): assert [f.width,f.height]==item['canvas_wh'];n+=1
                assert n==121
            state['validation_seconds']=time.monotonic()-start;tmp.replace(out)
            state['vae_write_seconds']=state['decode_seconds']
            # Match historical decode_seconds, which includes full video verification.
            state['decode_seconds']+=state['validation_seconds']
            state.update(status='complete',full_decode_frames=n,output=str(out),output_sha256=sha(out),completed_unix=time.time(),decode_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            state['elapsed_seconds']=time.time()-state['started_unix'];atomic(status_path,state)
            del sample,batch,latent,frames;gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'id':sid,'status':'complete','denoise_seconds':state['denoise_seconds'],'decode_seconds':state['decode_seconds']}),flush=True)
        except Exception:
            state.update(status='failed',failure_reason=traceback.format_exc(),completed_unix=time.time());atomic(status_path,state);raise
    atomic(ROOT/'metadata/run_status.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'completed_unix':time.time(),'completed_ids':ORDER})
if __name__=='__main__':
    try: run()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        for sid in ORDER:
            p=ROOT/f'outputs/{sid}.json'
            if not p.exists(): atomic(p,{'id':sid,'status':'not_run','failure_reason':'Stopped after technical failure; no automatic retry','upstream_error':error})
        atomic(ROOT/'metadata/run_status.json',{'status':'failed','error':error,'completed_unix':time.time()})
        raise
