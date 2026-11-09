from pathlib import Path
import json,os,hashlib
import requests
R=Path(__file__).resolve().parent
p=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token');token=p.read_text().strip() if p.exists() else None
jobs=[('LTX-2.5-22b-IC-LoRA-Day-To-Night',['examples/Examples_6__step_003000_02.mp4','examples/p_ind_35318605_night.mp4','ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors']),('LTX-2.5-22b-IC-LoRA-Ingredients',['examples/ingredients_lora_1.mp4','examples/ingredients_lora_3.mp4']),('LTX-2.3-22b-IC-LoRA-Instant-Shave',['examples/v2v_sbs_output_instantshave_01.mp4'])]
rows=[]
for name,files in jobs:
 if name!='LTX-2.5-22b-IC-LoRA-Day-To-Night':continue
 info=json.loads((R/'metadata'/f'{name}_repo.json').read_text())
 for f in files:
  print('downloading',name,f,flush=True)
  out=R/('models/loras' if f.endswith('.safetensors') else 'assets')/Path(f).name;out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists():
   url='https://huggingface.co/Lightricks/'+name+'/resolve/'+info['sha']+'/'+f
   response=requests.get(url,headers={'Authorization':'Bearer '+token} if token else {},stream=True,timeout=(30,180));response.raise_for_status()
   tmp=out.with_suffix(out.suffix+'.part')
   with tmp.open('wb') as stream:
    for block in response.iter_content(8*1024*1024):stream.write(block)
   tmp.replace(out)
  with out.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  rows.append(dict(repo='Lightricks/'+name,revision=info['sha'],file=f,local=str(out),bytes=out.stat().st_size,sha256=sha))
  (R/'metadata/downloads.json').write_text(json.dumps(rows,indent=2)+'\n');print('saved',out.name,out.stat().st_size,flush=True)
