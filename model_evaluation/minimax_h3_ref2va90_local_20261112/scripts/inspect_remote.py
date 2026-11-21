import json,urllib.request,shutil,time,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ.get('SLURM_JOB_ID')=='199810'
if (root/'metadata/model_download_manifest.json').exists():
 m=json.loads((root/'metadata/model_download_manifest.json').read_text());print('Using frozen model revision',m['revision']);raise SystemExit(0)
url='https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3?blobs=true'
with urllib.request.urlopen(url,timeout=60) as r:m=json.load(r)
(root/'metadata/hub_model.json').write_text(json.dumps(m,indent=2)+'\n')
files=[f for f in m['siblings'] if f['rfilename'].startswith('Ref2VA/') or f['rfilename']=='model_index.json']
size=sum(f.get('size',0) for f in files)
print(json.dumps({'revision':m['sha'],'files':len(files),'bytes':size,'GiB':size/1024**3,'free_GiB':shutil.disk_usage(root).free/1024**3},indent=2),flush=True)
for f in files:print(f['rfilename'],f.get('size'))
assert shutil.disk_usage(root).free>=size+60*1024**3,'Need model + runtime + at least 40GiB reserve'
(root/'metadata/model_download_manifest.json').write_text(json.dumps({'repo':'MiniMaxAI/MiniMax-H3','revision':m['sha'],'files':files,'bytes':size,'checked_unix':time.time()},indent=2)+'\n')
