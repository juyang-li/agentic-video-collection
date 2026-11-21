import json,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT.parent
PKG=EVAL/'frozen300_20261113/package'
REV='0f999ecc4cbdebac8165318b1274686b7912466f'
REPO='X-GenGroup/frozen300-h3-vace-accel-20261114'
MODELS=['ltx_distilled_v7','a849_v7','h3_pdd8','vace_lightx2v4']
SMOKE=['identity_01','whole_outfit_01','action_01','coinve_06','coinve_11','tripvvt_01']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
