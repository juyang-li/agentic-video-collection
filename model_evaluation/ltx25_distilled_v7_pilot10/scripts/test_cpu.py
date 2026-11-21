"""Exercise real V7 packing/extraction and official stochastic sampler on tiny CPU tensors."""
from common import *
import torch
from sampling import sample_state,build_generation_state,extract_generated_latent
from ltx_trainer.model_loader import read_video_scale_factors
from ltx_pipelines.utils.types import VideoAudio,DenoisedLatentResult
from native_reference_strategy import reference_segments

def main():
    _,paths=weights();scale=read_video_scale_factors(paths['vae']);tests=[]
    def raw(f,value):
        return {'latents':torch.full((1,128,f,2,3),value,dtype=torch.bfloat16),'fps':12,'num_frames':f,'height':2,'width':3}
    for kinds in [[('identity','front')],[('outfit','front'),('outfit','back')],[('motion','')],[]]:
        source=raw(2,.25)
        refs=[{'type':kind,'view':view,'index':i,'latent':raw(2 if kind=='motion' else 1,float(i+1))} for i,(kind,view) in enumerate(kinds)]
        original=source['latents'].clone(); originals=[r['latent']['latents'].clone() for r in refs]
        tools,state=build_generation_state(source,refs,scale_factors=scale,protocol=PROTOCOL,seed=42)
        n=tools.patchifier.get_token_count(tools.target_shape)
        segments=reference_segments({'source':source,'references':refs},protocol_version=PROTOCOL)
        assert [s[0] for s in segments]==[('motion' if k=='motion' else 'image') for k,v in kinds]+['source']
        assert bool((state.denoise_mask.tensor[:,:-n]==0).all()) and bool((state.denoise_mask.tensor[:,-n:]==1).all())
        expected=torch.cat([x[2]['latents'].permute(0,2,3,4,1).reshape(1,-1,128) for x in segments],dim=1)
        assert torch.equal(state.clean_latent[:,:-n],expected)
        def fake(model,video,audio,sigmas,step):
            # Deliberately predict values that conflict with conditions; the sampler must protect them.
            return VideoAudio(DenoisedLatentResult(torch.full_like(video.latent,-2)),None)
        final,audit=sample_state(state,None,fake)
        out=extract_generated_latent(tools,final,protocol=PROTOCOL)
        assert out.shape==source['latents'].shape and bool((out==-2).all())
        assert torch.equal(source['latents'],original)
        assert all(torch.equal(r['latent']['latents'],v) for r,v in zip(refs,originals))
        tests.append({'references':kinds,**audit,'target_extraction_exact':True,'input_tensors_unchanged':True,'ordered_prefix_exact':True})
    atomic(ROOT/'metadata/cpu_tests.json',{'status':'passed','tests':tests,'sampling_sha256':sha(ROOT/'scripts/sampling.py'),'sigmas':SIGMAS})
    print('CPU V7/sampler tests passed')
if __name__=='__main__':main()
