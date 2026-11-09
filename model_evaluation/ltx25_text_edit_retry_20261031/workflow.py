"""Four explicit no-adapter probes on LTX-2.5 dev; no automatic broad sweep."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def graph(row, output, reference_only=False):
    assert not reference_only and not row.get('reference_image')
    g = {}

    def add(i, kind, **inputs):
        g[str(i)] = {'class_type': kind, 'inputs': inputs}

    add(1, 'UNETLoader', unet_name='ltx-2.5-22b-dev-transformer-bf16.safetensors', weight_dtype='default')
    add(2, 'CLIPLoader', clip_name='gemma4-12b-with-proj-ltx-2.5-bf16.safetensors', type='ltxv', device='default')
    add(3, 'VAELoader', vae_name='ltx-2.5-video-vae-conv-bf16.safetensors')
    add(4, 'VAELoader', vae_name='ltx-2.5-audio-vae-bf16.safetensors')
    add(5, 'LTXBenchSource', path=str(PROJECT / row['source_video']), max_frames=97, fps=24.0)
    add(6, 'CLIPTextEncode', text=row['instruction'], clip=['2', 0])
    add(7, 'CLIPTextEncode', text='low quality, blurry, distorted face, deformed, flickering, text, watermark', clip=['2', 0])
    add(8, 'LTXVConditioning', positive=['6', 0], negative=['7', 0], frame_rate=24.0)
    add(9, 'EmptyLTXVLatentVideo', width=['5', 1], height=['5', 2], length=['5', 3], batch_size=1)

    if row['route'] == 'rgb_guide':
        add(11, 'LTXAddVideoICLoRAGuideAdvanced', positive=['8', 0], negative=['8', 1],
            vae=['3', 0], latent=['9', 0], image=['5', 0], frame_idx=0, strength=1.0,
            attention_strength=row['attention_strength'], latent_downscale_factor=1.0,
            crop='disabled', use_tiled_encode=False, tile_size=256, tile_overlap=64)
        video = ['11', 2]
        positive, negative = ['11', 0], ['11', 1]
        schedule_latent = ['9', 0]
    else:
        assert row['route'] == 'source_latent_init'
        add(11, 'VAEEncode', pixels=['5', 0], vae=['3', 0])
        video = ['11', 0]
        positive, negative = ['8', 0], ['8', 1]
        schedule_latent = video

    add(14, 'LTXVEmptyLatentAudio', frames_number=['5', 3], frame_rate=24.0, batch_size=1, audio_vae=['4', 0])
    add(15, 'LTXVConcatAVLatent', video_latent=video, audio_latent=['14', 0])
    add(16, 'CFGGuider', model=['1', 0], positive=positive, negative=negative, cfg=row['cfg'])
    add(17, 'RandomNoise', noise_seed=row['seed'])
    add(18, 'LTXVScheduler', steps=row['steps'], max_shift=2.05, base_shift=0.95,
        stretch=True, terminal=0.1, latent=schedule_latent)
    sigmas = ['18', 0]
    if row['route'] == 'source_latent_init':
        add(26, 'SplitSigmasDenoise', sigmas=sigmas, denoise=row['denoise'])
        sigmas = ['26', 1]
    add(19, 'KSamplerSelect', sampler_name='euler')
    add(20, 'SamplerCustomAdvanced', noise=['17', 0], guider=['16', 0], sampler=['19', 0],
        sigmas=sigmas, latent_image=['15', 0])
    add(21, 'LTXVSeparateAVLatent', av_latent=['20', 0])
    decoded_latent = ['21', 0]
    if row['route'] == 'rgb_guide':
        add(22, 'LTXVCropGuides', positive=positive, negative=negative, latent=decoded_latent)
        decoded_latent = ['22', 2]
    add(23, 'VAEDecodeTiled', samples=decoded_latent, vae=['3', 0], tile_size=768,
        overlap=192, temporal_size=128, temporal_overlap=32)
    add(24, 'LTXBenchSave', images=['23', 0], path=str(Path(output).resolve()), fps=24.0)
    return g
