"""CPU acquisition of pinned official OmniEdit object tasks."""
from common import *
import requests,concurrent.futures
from urllib.parse import quote
OLD=ROOT.parent/'a849_external_vbench_v2_96'
info=read(OLD/'metadata/omni_revision.json');atomic(ROOT/'metadata/omni_revision.json',info)
ids={f'Sp_Object_{kind}_{n:02d}' for kind,nums in [('adding',[2,4,6,10]),('removal',[2,6,8,10])] for n in nums}
items=[r for r in read(OLD/'inputs/omni/spatial_benchmark.json') if r['video_id'] in ids]
def fetch(r):
 url=f"https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/resolve/{info['sha']}/"+quote(r['source_video'])
 dst=ROOT/'inputs/omni'/r['source_video'];dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():
  s=requests.Session();s.trust_env=False
  res=s.get(url,timeout=180);res.raise_for_status();dst.write_bytes(res.content)
 return dict(id=r['video_id'],official=r,path=str(dst),source_sha256=sha(dst),instruction=r['editing_instruction'],url=url)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(fetch,items))
atomic(ROOT/'metadata/object_candidates.json',records)
