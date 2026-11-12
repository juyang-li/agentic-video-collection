from common import *
require_job(exact=True)
from huggingface_hub import HfApi,snapshot_download
repo='mattmdjaga/segformer_b2_clothes'
revision=HfApi().model_info(repo).sha
path=snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.safetensors'],cache_dir=str(ROOT/'runtime/hf_cache'))
config=read(Path(path)/'config.json')
assert config['id2label']['11']=='Face' and config['id2label']['2']=='Hair'
ready=read(ROOT/'runtime/vace_ready.json')
ready['human_parsing']={'repo_id':repo,'revision':revision,'path':path,'id2label':config['id2label']}
atomic(ROOT/'runtime/vace_ready.json',ready)
print('Skin mask parser ready:',revision,flush=True)
