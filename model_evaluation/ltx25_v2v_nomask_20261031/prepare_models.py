from pathlib import Path
import requests,json,hashlib,shutil
R=Path(__file__).resolve().parent
token=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token').read_text().strip()
h={'Authorization':'Bearer '+token};repo='Lightricks/LTX-2.3-22b-IC-LoRA-Deblur'
res=requests.get('https://huggingface.co/api/models/'+repo,headers=h,timeout=60);res.raise_for_status();info=res.json()
(R/'metadata/deblur_repo.json').write_text(json.dumps(info,indent=2)+'\n')
rev=info['sha'];name='ltx-2.3-22b-ic-lora-deblur-0.9.safetensors';out=R/'models/loras'/name
readme=requests.get(f'https://huggingface.co/{repo}/raw/{rev}/README.md',headers=h,timeout=60)
if readme.ok:(R/'metadata/deblur_README.txt').write_text(readme.text)
if not out.exists():
    response=requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',headers=h,stream=True,timeout=(30,180));print('weight_access_status',response.status_code,flush=True);response.raise_for_status()
    tmp=out.with_suffix('.part')
    with tmp.open('wb') as f:
        for block in response.iter_content(8*1024*1024):f.write(block)
    tmp.replace(out)
old=json.loads((R.parent/'ltx25_official_edit_routes_20261031/metadata/model_lock.json').read_text())
rows=[r for r in old if '-dev-' not in r['file']]
with out.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
rows.append(dict(repo=repo,revision=rev,file='loras/'+name,local=str(out),bytes=out.stat().st_size,sha256=sha))
(R/'metadata/model_lock.json').write_text(json.dumps(rows,indent=2)+'\n')
print('download_complete',out.name,out.stat().st_size,sha,flush=True)
