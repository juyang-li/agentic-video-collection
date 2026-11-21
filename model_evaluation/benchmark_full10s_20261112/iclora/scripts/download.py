from pathlib import Path
import os,json,hashlib,time
assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue'
from huggingface_hub import HfApi,hf_hub_download
root=Path(__file__).resolve().parents[1]
repo='Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients';revision='5f83f6f7e5e6cd6eadfade02a57f03b9212d6434';filename='ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors'
tokenfile=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token');token=tokenfile.read_text().strip() if tokenfile.exists() else None
info=HfApi(token=token).model_info(repo,revision=revision,files_metadata=True)
entry=next(s for s in info.siblings if s.rfilename==filename)
expected=entry.lfs.sha256
(root/'metadata/model_identity.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'filename':filename,'bytes':entry.size,'sha256':expected},indent=2))
start=time.time();path=Path(hf_hub_download(repo,filename,revision=revision,token=token,local_dir=root/'runtime/models'))
with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected
card=hf_hub_download(repo,'README.md',revision=revision,token=token,local_dir=root/'metadata/model_card')
(root/'metadata/download_complete.json').write_text(json.dumps({'status':'passed','path':str(path),'sha256':actual,'download_seconds':time.time()-start,'completed_unix':time.time()},indent=2))
print('IC-LoRA download and official SHA256 verification passed',flush=True)
