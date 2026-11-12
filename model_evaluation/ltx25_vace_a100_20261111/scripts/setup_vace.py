import importlib.metadata as metadata
import inspect
import os
from common import *
require_job(exact=True)
os.environ['HF_HOME']=str(ROOT/'runtime/hf_cache')
from huggingface_hub import HfApi,snapshot_download
from diffusers import WanVACEPipeline
from diffusers.pipelines.wan import pipeline_wan_vace
api=HfApi()
records={}
for key,repo,patterns in [('vace','Wan-AI/Wan2.1-VACE-14B-diffusers',None),('dino','IDEA-Research/grounding-dino-base',None),('pose','ali-vilab/VACE-Annotators',['pose/*']),('human_parsing','mattmdjaga/segformer_b2_clothes',['*.json','*.safetensors'])]:
    previous=read(ROOT/'runtime/download_revisions.json') if (ROOT/'runtime/download_revisions.json').exists() else {}
    revision=previous.get(key,{}).get('revision') or api.model_info(repo).sha
    previous[key]={'repo_id':repo,'revision':revision}
    atomic(ROOT/'runtime/download_revisions.json',previous)
    path=snapshot_download(repo,revision=revision,allow_patterns=patterns,cache_dir=str(ROOT/'runtime/hf_cache'))
    records[key]={'repo_id':repo,'revision':revision,'path':path}
    print(key,revision,path,flush=True)
source=Path(inspect.getfile(pipeline_wan_vace))
records['human_parsing']['id2label']=read(Path(records['human_parsing']['path'])/'config.json')['id2label']
records['pipeline']={'path':str(source),'sha256':sha(source),'signature':str(inspect.signature(WanVACEPipeline.__call__))}
records['packages']={p:metadata.version(p) for p in ['torch','torchvision','diffusers','transformers','numpy','Pillow','onnxruntime-gpu','huggingface_hub','matplotlib']}
assert 'reference_images' in inspect.signature(WanVACEPipeline.__call__).parameters
records['sam2']={'path':'/scratch/prj0000000275/jcy/sam2','checkpoint':'/scratch/prj0000000275/jcy/sam2_ckpt/sam2.1_hiera_base_plus.pt'}
records['pose_code']={'path':'/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/tools/VACE/vace/annotators/pose.py'}
records['pose_code']['sha256']=sha(records['pose_code']['path'])
records['status']='ready'
atomic(ROOT/'runtime/vace_ready.json',records)
