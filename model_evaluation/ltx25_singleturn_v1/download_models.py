"""Download exact, approved official LTX weights; no automatic license acceptance."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parent
BASE='Lightricks/LTX-2.5'
BASE_REV='49b093a39c19169a8a81721dcc541e07e945232f'
ADAPTER='Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients'
ADAPTER_REV='5f83f6f7e5e6cd6eadfade02a57f03b9212d6434'
JOBS=[
 (BASE,BASE_REV,'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'),
 (BASE,BASE_REV,'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'),
 (BASE,BASE_REV,'vae/ltx-2.5-video-vae-conv-bf16.safetensors'),
 (BASE,BASE_REV,'vae/ltx-2.5-audio-vae-bf16.safetensors'),
 (ADAPTER,ADAPTER_REV,'ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors'),
]

def main():
    token=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token').read_text().strip()
    def one(job):
        repo,rev,name=job;dest=ROOT/'models'/('loras/'+name if repo==ADAPTER else name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            print('downloading',name,flush=True)
            path=hf_hub_download(repo_id=repo,filename=name,revision=rev,token=token,cache_dir=str(ROOT/'runtime/hf-cache'))
            if dest.is_symlink() and not dest.exists():dest.unlink()
            try:os.link(Path(path).resolve(),dest)
            except OSError:shutil.copy2(path,dest)
        print('ready',name,dest.stat().st_size,flush=True)
        with dest.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
        return dict(repo=repo,revision=rev,file=name,local=str(dest),bytes=dest.stat().st_size,sha256=checksum)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,JOBS))
    (ROOT/'metadata').mkdir(exist_ok=True)
    (ROOT/'metadata/model_lock.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('All required model files downloaded',flush=True)

if __name__=='__main__':main()
