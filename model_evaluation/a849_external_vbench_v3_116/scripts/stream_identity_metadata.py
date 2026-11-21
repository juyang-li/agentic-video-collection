"""Read pinned Arrow shards on CPU; retain annotation columns only, not image assets."""
from pathlib import Path
import sys,requests,concurrent.futures,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/cpu_deps'))
import pyarrow as pa
REV='6a19f6d699728f745f38b94adffe82ccb37ba224';REPO='negedng/CelebA-attrs-identity'
FILES=[f'data/train-{n:05d}-of-00003.arrow'for n in range(3)]+['data/valid-00000-of-00001.arrow','data/test-00000-of-00001.arrow']
def fetch(name):
 s=requests.Session();s.trust_env=False;url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}';result={}
 with s.get(url,stream=True,timeout=(30,180))as response:
  response.raise_for_status();response.raw.decode_content=True;reader=pa.ipc.open_stream(response.raw)
  for batch in reader:
   image=batch.column(batch.schema.get_field_index('image'));paths=image.field('path').to_pylist();fields=['identity','Male','Attractive','Young','Blurry','Wearing_Hat'];columns={k:batch.column(batch.schema.get_field_index(k)).to_pylist()for k in fields}
   for n,path in enumerate(paths):
    key=Path(path).name;assert key not in result;result[key]={k:int(v[n])for k,v in columns.items()}
 print('Annotation shard complete',name,len(result),flush=True);return result
all_rows={}
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
 for data in pool.map(fetch,FILES):
  assert not(set(data)&set(all_rows));all_rows.update(data)
assert len(all_rows)==202599 and set(all_rows)=={f'{n:06d}.jpg'for n in range(1,202600)}
raw=''.join(f"{key} {all_rows[key]['identity']}\n"for key in sorted(all_rows)).encode();md5=hashlib.md5(raw).hexdigest();(ROOT/'metadata/identity_CelebA_reconstructed.txt').write_bytes(raw)
(ROOT/'metadata/celeba_identity_attributes.json').write_text(json.dumps(all_rows,separators=(',',':'))+'\n')
(ROOT/'metadata/identity_annotation_provenance.json').write_text(json.dumps(dict(status='complete',repo=REPO,revision=REV,files=FILES,rows=len(all_rows),reconstructed_md5=md5,official_torchvision_identity_md5='32bd1bd63d3c78cd57e08160ec5ed1e2',matches_official_md5=md5=='32bd1bd63d3c78cd57e08160ec5ed1e2',method='Public dataset path+identity annotation; no face recognition; no stored image data',completed_unix=time.time()),indent=2)+'\n');print('All identity annotations complete',len(all_rows),'MD5',md5,flush=True)
