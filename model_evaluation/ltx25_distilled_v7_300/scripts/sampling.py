"""V7 state/extraction with the installed official LTX2.5 ancestral sampler."""
from common import *
import torch
from native_reference_inference import build_generation_state, extract_generated_latent, make_denoiser
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop

def sample_state(state, transformer, denoiser, *, progress=None):
    assert list(DISTILLED_SIGMA_VALUES)==SIGMAS and len(SIGMAS)-1==8
    mask=state.denoise_mask.tensor==0
    clean=state.clean_latent
    steps=[]
    def audited(model, video, audio, sigmas, step):
        assert audio is None and step==len(steps)
        assert torch.equal(video.latent.masked_select(mask),clean.masked_select(mask)), 'Condition altered before step'
        assert bool(torch.isfinite(video.latent).all())
        result=denoiser(model,video,audio,sigmas,step)
        assert bool(torch.isfinite(result.video.denoised).all())
        steps.append(step)
        if progress: progress(steps.copy())
        return result
    result=euler_ancestral_denoising_loop(torch.tensor(SIGMAS,device=state.latent.device,dtype=torch.float32),state,None,transformer,audited,noise_seed=10042,model_dtype=torch.bfloat16,eta=1.,s_noise=1.)
    assert steps==list(range(8))
    assert torch.equal(result.video.latent.masked_select(mask),clean.masked_select(mask))
    return result.video,{'actual_updates':len(steps),'step_indices':steps,'condition_tokens_unchanged':True,'finite_each_step':True,'packed_shape':list(state.latent.shape),'condition_tokens':int((mask[...,0] if mask.shape[-1]>1 else mask.squeeze(-1)).sum())}

@torch.inference_mode()
def generate(model,processor,batch,scale,progress):
    tools,state=build_generation_state(batch['source'],batch['references'],scale_factors=scale,protocol=PROTOCOL,seed=42)
    n=tools.patchifier.get_token_count(tools.target_shape)
    assert bool((state.denoise_mask.tensor[:,:-n]==0).all()) and bool((state.denoise_mask.tensor[:,-n:]==1).all())
    features=batch['conditions']['video_prompt_embeds']
    context,_,mask=processor.create_embeddings(features,None,convert_to_additive_mask(batch['conditions']['prompt_attention_mask'],features.dtype))
    assert mask is not None and bool((mask==1).all()) and bool(torch.isfinite(context).all())
    transformer,denoiser=make_denoiser(model,context,cfg_scale=1.,stg_scale=0.,guidance_rescale=0.)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        final,audit=sample_state(state,transformer,denoiser,progress=progress)
    result=extract_generated_latent(tools,final,protocol=PROTOCOL)
    assert bool(torch.isfinite(result).all()) and result.shape==batch['source']['latents'].shape
    audit.update(target_last=True,target_tokens=n,latent_shape=list(result.shape),no_gt=True,reference_order=[(r['type'][0],r['view'][0]) for r in batch['references']])
    return result,audit
