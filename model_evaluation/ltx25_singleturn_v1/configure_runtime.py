from pathlib import Path
import os,json

ROOT=Path(__file__).resolve().parent
COMFY=ROOT/'runtime/ComfyUI'
(ROOT/'runtime/comfy_user_cpu').mkdir(parents=True,exist_ok=True)
(ROOT/'metadata').mkdir(exist_ok=True)
custom=COMFY/'custom_nodes/ltx25_bench';custom.mkdir(exist_ok=True)
target=custom/'__init__.py'
if not target.exists():target.symlink_to(ROOT/'bench_nodes.py')
(ROOT/'runtime/model_paths.yaml').write_text('ltx25_bench:\n  base_path: '+str(ROOT/'models')+'\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n  loras: loras\n')
print('Runtime paths configured')
