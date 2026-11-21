"""CPU fetch public dataset annotations; never infer identities from face recognition."""
from pathlib import Path
import requests,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
s=requests.Session();s.trust_env=False
url='https://drive.google.com/uc?export=download&id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS'
r=s.get(url,timeout=120);r.raise_for_status();raw=r.content
lines=raw.decode().splitlines();assert len(lines)==202599 and all(len(x.split())==2 for x in lines[:10]),'Expected official CelebA identity annotation text'
(ROOT/'metadata/identity_CelebA.txt').write_bytes(raw);(ROOT/'metadata/identity_annotation_source.json').write_text(json.dumps(dict(url=url,source_code='https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/celeb_a/celeb_a_dataset_builder.py',sha256=hashlib.sha256(raw).hexdigest(),rows=len(lines)),indent=2)+'\n');print('CelebA identity metadata',len(lines),flush=True)
r=s.get('https://huggingface.co/api/datasets',params={'search':'celebamask','limit':12},timeout=60);r.raise_for_status();repos=[x['id']for x in r.json()];print('Mapping source candidates',repos,flush=True)
for repo in repos:
 try:
  r=s.get('https://huggingface.co/api/datasets/'+repo,timeout=60);r.raise_for_status();info=r.json();matches=[x['rfilename']for x in info.get('siblings',[])if 'mapping' in x['rfilename'].lower()or 'image_list' in x['rfilename'].lower()]
  print(repo,matches,flush=True)
  for name in matches:
   url=f'https://huggingface.co/datasets/{repo}/resolve/{info["sha"]}/{name}';r=s.get(url,timeout=120);r.raise_for_status();data=r.content
   if len(data)>100000 and ('orig_file'in data[:200].decode(errors='ignore')):
    (ROOT/'metadata/CelebA-HQ-to-CelebA-mapping.txt').write_bytes(data);(ROOT/'metadata/hq_mapping_source.json').write_text(json.dumps(dict(repo=repo,revision=info['sha'],path=name,url=url,sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n');print('Mapping acquired',repo,name,flush=True);return_value=True;break
  else:continue
  break
 except Exception as e:print('Metadata source unavailable',repo,type(e).__name__,flush=True)
