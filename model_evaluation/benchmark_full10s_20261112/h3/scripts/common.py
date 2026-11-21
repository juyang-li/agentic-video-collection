import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent
NATIVE=Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
BUNDLE=NATIVE/'data/mixed/outfit_camera_720p/sources.json'
GROUPS=['identity','single_piece','whole_outfit','action','site','hair','hair_color','camera']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def freeze_jsonl(p,data):
 p=Path(p);s=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data)
 if p.exists():assert p.read_text()==s,f'Frozen data differs: {p}'
 else:p.write_text(s)
def require_job():
 assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue','Require existing testqueue allocation 199810'

def same_inference_contract(previous,current):
 # Read-only telemetry may improve without changing a generated sample.
 ignored={'telemetry_script_sha256'}
 return {k:v for k,v in previous.items() if k not in ignored}=={k:v for k,v in current.items() if k not in ignored}
