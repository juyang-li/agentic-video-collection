import os,json,time,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ.get('SLURM_JOB_ID')=='199810'
from huggingface_hub import snapshot_download
m=json.loads((root/'metadata/model_download_manifest.json').read_text())
if (root/'metadata/download_complete.json').exists() and all((root/'runtime/models/H3'/f['rfilename']).is_file() and (root/'runtime/models/H3'/f['rfilename']).stat().st_size==f['size'] for f in m['files']):
 print('MODEL_ALREADY_DOWNLOADED',m['revision'],flush=True);raise SystemExit(0)
remaining=sum(f['size'] for f in m['files'] if not (root/'runtime/models/H3'/f['rfilename']).is_file())
assert shutil.disk_usage(root).free>=remaining+60*1024**3
started=time.time()
p=snapshot_download(m['repo'],revision=m['revision'],allow_patterns=[f['rfilename'] for f in m['files']],local_dir=str(root/'runtime/models/H3'),max_workers=8)
for f in m['files']:
 q=Path(p)/f['rfilename'];assert q.is_file() and q.stat().st_size==f['size'],str(q)
(root/'metadata/download_complete.json').write_text(json.dumps({'status':'complete','revision':m['revision'],'path':p,'seconds':time.time()-started,'bytes':m['bytes']},indent=2)+'\n')
print('MODEL_DOWNLOAD_COMPLETE',flush=True)
