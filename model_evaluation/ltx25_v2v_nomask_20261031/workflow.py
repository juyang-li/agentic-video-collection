from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROMPT='A cinematic close-up of the same short-haired person wearing large opaque black sunglasses with black frames. The black sunglasses clearly cover both eyes. The person wears the same light-colored coat and looks around in the same dim outdoor setting, with natural expression and movement.'
CASES=[('v2v_no_adapter',None),('v2v_inpaint_fullrgb','ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors'),('v2v_deblur','ltx-2.3-22b-ic-lora-deblur-0.9.safetensors')]
def graph(sid,adapter):
    g={}
    def add(i,kind,**kw):g[str(i)]={'class_type':kind,'inputs':kw}
    add(1,'UNETLoader',unet_name='ltx-2.5-22b-distilled-transformer-bf16.safetensors',weight_dtype='default')
    add(2,'CLIPLoader',clip_name='gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',type='ltxv',device='default')
    add(3,'VAELoader',vae_name='ltx-2.5-video-vae-conv-bf16.safetensors')
    add(4,'VAELoader',vae_name='ltx-2.5-audio-vae-bf16.safetensors')
    add(5,'LTXBenchSource',path=str(ROOT/'assets/source49.mp4'),max_frames=49,fps=24.0)
    add(6,'CLIPTextEncode',text=PROMPT,clip=['2',0])
    add(7,'CLIPTextEncode',text='pc game, console game, video game, cartoon, childish, ugly',clip=['2',0])
    add(8,'LTXVConditioning',positive=['6',0],negative=['7',0],frame_rate=24.0)
    add(9,'EmptyLTXVLatentVideo',width=768,height=448,length=49,batch_size=1)
    if adapter:add(10,'LTXICLoRALoaderModelOnly',model=['1',0],lora_name=adapter,strength_model=1.0)
    add(11,'LTXAddVideoICLoRAGuide',positive=['8',0],negative=['8',1],vae=['3',0],latent=['9',0],image=['5',0],frame_idx=0,strength=1.0,latent_downscale_factor=1.0,crop='disabled',use_tiled_encode=False,tile_size=256,tile_overlap=64)
    add(12,'LTXRouteSilentAudio',frames=49,fps=24.0)
    add(13,'VAEEncodeAudio',audio=['12',0],vae=['4',0])
    add(14,'LTXVSetAudioRefTokens',positive=['11',0],negative=['11',1],audio_latent=['13',0])
    add(15,'LTXVConcatAVLatent',video_latent=['11',2],audio_latent=['14',2])
    add(16,'CFGGuider',model=['10',0] if adapter else ['1',0],positive=['14',0],negative=['14',1],cfg=1.0)
    add(17,'RandomNoise',noise_seed=43)
    add(18,'ManualSigmas',sigmas='1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0')
    add(19,'KSamplerSelect',sampler_name='euler_ancestral')
    add(20,'SamplerCustomAdvanced',noise=['17',0],guider=['16',0],sampler=['19',0],sigmas=['18',0],latent_image=['15',0])
    add(21,'LTXVSeparateAVLatent',av_latent=['20',0])
    add(22,'LTXVCropGuides',positive=['14',0],negative=['14',1],latent=['21',0])
    add(23,'VAEDecodeTiled',samples=['22',2],vae=['3',0],tile_size=768,overlap=192,temporal_size=128,temporal_overlap=32)
    add(24,'LTXBenchSave',images=['23',0],path=str(ROOT/f'outputs/generated/{sid}.mp4'),fps=24.0)
    assert not any('mask' in n['class_type'].lower() or 'blend' in n['class_type'].lower() or 'preprocess' in n['class_type'].lower() for n in g.values())
    assert not any('mask' in k.lower() for n in g.values() for k in n['inputs'])
    return g
