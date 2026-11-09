"""Official single-stage distilled T2V core, with optional image path disabled."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SIGMAS = '1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'


def graph(row, output, reference_only=False):
    assert not reference_only and not row.get('source_video') and not row.get('reference_image')
    g = {}

    def add(i, kind, **inputs):
        g[str(i)] = {'class_type': kind, 'inputs': inputs}

    add(1, 'UNETLoader', unet_name='ltx-2.5-22b-distilled-transformer-bf16.safetensors', weight_dtype='default')
    add(2, 'CLIPLoader', clip_name='gemma4-12b-with-proj-ltx-2.5-bf16.safetensors', type='ltxv', device='default')
    add(3, 'VAELoader', vae_name='ltx-2.5-video-vae-conv-bf16.safetensors')
    add(4, 'VAELoader', vae_name='ltx-2.5-audio-vae-bf16.safetensors')
    add(6, 'CLIPTextEncode', text=row['instruction'], clip=['2', 0])
    add(7, 'CLIPTextEncode', text='', clip=['2', 0])
    add(8, 'LTXVConditioning', positive=['6', 0], negative=['7', 0], frame_rate=row['fps'])
    add(9, 'EmptyLTXVLatentVideo', width=row['width'], height=row['height'], length=row['frames'], batch_size=1)
    add(14, 'LTXVEmptyLatentAudio', frames_number=row['frames'], frame_rate=row['fps'], batch_size=1, audio_vae=['4', 0])
    add(15, 'LTXVConcatAVLatent', video_latent=['9', 0], audio_latent=['14', 0])
    add(16, 'CFGGuider', model=['1', 0], positive=['8', 0], negative=['8', 1], cfg=1.0)
    add(17, 'RandomNoise', noise_seed=row['seed'])
    add(18, 'ManualSigmas', sigmas=SIGMAS)
    add(19, 'KSamplerSelect', sampler_name='euler_ancestral')
    add(20, 'SamplerCustomAdvanced', noise=['17', 0], guider=['16', 0], sampler=['19', 0], sigmas=['18', 0], latent_image=['15', 0])
    add(21, 'LTXVSeparateAVLatent', av_latent=['20', 0])
    add(23, 'VAEDecodeTiled', samples=['21', 0], vae=['3', 0], tile_size=768, overlap=192, temporal_size=128, temporal_overlap=32)
    add(24, 'LTXBenchSave', images=['23', 0], path=str(Path(output).resolve()), fps=row['fps'])
    return g
