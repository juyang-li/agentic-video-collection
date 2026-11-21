"""Portable snapshot of the verified VACE LightX2V adapter. See README.md.
Use a fresh base pipeline once; applying twice would add the vector deltas twice.
"""
from pathlib import Path
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler

class FourStepEuler(FlowMatchEulerDiscreteScheduler):
    def set_timesteps(self,num_inference_steps=None,device=None,**kwargs):
        assert num_inference_steps==4
        # LightX2V's [1000,750,500,250], shifted by5, with terminal0.
        return super().set_timesteps(4,device=device,sigmas=[1.,.75,.5,.25])

def apply_lightx2v(pipe,lora_path):
    if getattr(pipe,'_datamake_lightx2v_applied',False):
        raise ValueError('Already applied to this pipeline; create a fresh base pipeline')
    lora_path=Path(lora_path)
    state=load_file(str(lora_path))
    # The installed official converter omits norm deltas and two standalone biases.
    # Apply those additive vectors explicitly; let official Diffusers/PEFT handle all LoRA matrices.
    extras={}
    for key,value in state.items():
        k=key.removeprefix('diffusion_model.')
        if 'norm' in k and k.endswith(('.diff','.diff_b')):
            dest=k.replace('.self_attn.','.attn1.').replace('.cross_attn.','.attn2.').replace('.norm3.','.norm2.')
            dest=dest.removesuffix('.diff_b')+'.bias' if dest.endswith('.diff_b') else dest.removesuffix('.diff')+'.weight'
        elif k=='time_projection.1.diff_b':dest='condition_embedder.time_proj.bias'
        elif k=='patch_embedding.diff_b':dest='patch_embedding.bias'
        else:continue
        param=pipe.transformer.get_parameter(dest)
        assert param.shape==value.shape,(key,dest,param.shape,value.shape)
        with torch.no_grad():param.add_(value.to(param))
        extras[key]={'target':dest,'max_abs_delta':float(value.abs().max())}
    converted=pipe.lora_state_dict(str(lora_path.parent),weight_name=lora_path.name,local_files_only=True)
    assert len(extras)+len(converted)==len(state),(len(extras),len(converted),len(state))
    pipe.load_lora_weights(str(lora_path.parent),weight_name=lora_path.name,local_files_only=True,adapter_name='lightx2v')
    pipe.set_adapters('lightx2v',adapter_weights=1.)
    from peft import get_peft_model_state_dict
    actual=get_peft_model_state_dict(pipe.transformer,adapter_name='lightx2v')
    expected={k.removeprefix('transformer.'):v for k,v in converted.items()}
    assert set(actual)==set(expected),(set(actual)-set(expected),set(expected)-set(actual))
    for k,v in expected.items():assert torch.equal(actual[k].cpu(),v.to(actual[k].dtype)),k
    receipt={'original_tensors':len(state),'official_lora_tensors':len(converted),'additional_vectors':extras,'all_loaded_values_checked':True}
    del state,converted,actual,expected
    pipe.scheduler=FourStepEuler(shift=5.)
    pipe._datamake_lightx2v_applied=True
    return receipt
