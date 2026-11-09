"""No-adapter LTX-2.5 distilled probe; full RGB guide retained for controlled comparison."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SIGMAS='1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'

def graph(row,output,reference_only=False):
    g={}
    def add(i,kind,**inputs):g[str(i)]={'class_type':kind,'inputs':inputs};return str(i)
    add(1,'UNETLoader',unet_name='ltx-2.5-22b-distilled-transformer-bf16.safetensors',weight_dtype='default')
    add(2,'CLIPLoader',clip_name='gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',type='ltxv',device='default')
    add(3,'VAELoader',vae_name='ltx-2.5-video-vae-conv-bf16.safetensors')
    add(4,'VAELoader',vae_name='ltx-2.5-audio-vae-bf16.safetensors')
    add(5,'LTXBenchSource',path=str(PROJECT/row['source_video']),max_frames=97,fps=24.0)
    add(6,'CLIPTextEncode',text=row['instruction'],clip=['2',0])
    add(7,'CLIPTextEncode',text='',clip=['2',0])
    add(8,'LTXVConditioning',positive=['6',0],negative=['7',0],frame_rate=24.0)
    add(9,'EmptyLTXVLatentVideo',width=['5',1],height=['5',2],length=['5',3],batch_size=1)
    guide_image=['5',0]
    if reference_only:
        add(30,'LTXBenchRef',path=str(PROJECT/row['reference_image']),width=['5',1],height=['5',2],repeat=['5',3])
        guide_image=['30',0]
    guide_inputs=dict(positive=['8',0],negative=['8',1],vae=['3',0],latent=['9',0],image=guide_image,
        frame_idx=0,strength=1.0,latent_downscale_factor=1.0,crop='disabled',use_tiled_encode=False,tile_size=256,tile_overlap=64)
    add(11,'LTXAddVideoICLoRAGuide',**guide_inputs)
    cond='11'
    if row.get('reference_image') and not reference_only:
        add(12,'LTXBenchRef',path=str(PROJECT/row['reference_image']),width=['5',1],height=['5',2],repeat=1)
        add(13,'LTXAddVideoICLoRAGuide',**{**guide_inputs,'positive':['11',0],'negative':['11',1],
            'latent':['11',2],'image':['12',0],'frame_idx':-1})
        cond='13'
    add(14,'LTXVEmptyLatentAudio',frames_number=['5',3],frame_rate=24.0,batch_size=1,audio_vae=['4',0])
    add(15,'LTXVConcatAVLatent',video_latent=[cond,2],audio_latent=['14',0])
    add(16,'CFGGuider',model=['1',0],positive=[cond,0],negative=[cond,1],cfg=1.0)
    add(17,'RandomNoise',noise_seed=row['seed'])
    add(18,'ManualSigmas',sigmas=SIGMAS)
    add(19,'KSamplerSelect',sampler_name='euler_ancestral')
    add(20,'SamplerCustomAdvanced',noise=['17',0],guider=['16',0],sampler=['19',0],sigmas=['18',0],latent_image=['15',0])
    add(21,'LTXVSeparateAVLatent',av_latent=['20',0])
    add(22,'LTXVCropGuides',positive=[cond,0],negative=[cond,1],latent=['21',0])
    add(23,'VAEDecodeTiled',samples=['22',2],vae=['3',0],tile_size=768,overlap=128,temporal_size=64,temporal_overlap=8)
    add(24,'LTXBenchSave',images=['23',0],path=str(Path(output).resolve()),fps=24.0)
    return g
