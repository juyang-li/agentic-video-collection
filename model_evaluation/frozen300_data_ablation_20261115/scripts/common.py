import hashlib,json,os,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT.parent
PKG=EVAL/'frozen300_20261113/package'
GTOLD=EVAL/'frozen300_gt_metrics_20261115'
VBOLD=EVAL/'frozen300_vbench_20261115'
VBENV=EVAL/'a849_external_vbench_v2_96'
GTPY=GTOLD/'runtime/bin/python'
VBPY=VBENV/'runtime/vbench/bin/python'
JOB='205167'
MODELS=['real232','synthetic600']
SOURCES={m:Path('/scratch/prj0000000275/jcy/xks/fastref')/f'ltx_distilled_{"real232" if m=="real232" else "synthetic600"}_frozen300_20261115' for m in MODELS}
SMOKE=['identity_01','whole_outfit_01','action_01','coinve_06','coinve_11','tripvvt_01']
VBDIMS=['subject_consistency','background_consistency','aesthetic_quality','imaging_quality','temporal_flickering','motion_smoothness']
GTDIMS=['ssim','psnr_db','lpips']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def load_cases():return rows(ROOT/'metadata/media.jsonl')
def identity():return sha(ROOT/'metadata/implementation.json')
def path_score(kind,case,dim=None):return ROOT/'scores'/kind/case['model']/(f'{case["id"]}--{dim}.json' if dim else f'{case["id"]}.json')
def compatible(p,case):
 if not p.exists():return False
 r=read(p)
 return r.get('status')=='complete' and r.get('implementation_sha256')==identity() and r.get('output_sha256')==case['output_sha256'] and r.get('gt_sha256')==case['gt_sha256']
def receipt(case,**extra):
 return dict(id=case['id'],uid=case['uid'],model=case['model'],dataset=case['dataset'],group=case['group'],output_sha256=case['output_sha256'],gt_sha256=case['gt_sha256'],implementation_sha256=identity(),created_unix=time.time(),**extra)
def require_node():
 assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ.get('SLURM_JOB_ID')==JOB
def cases_for(phase):return [r for r in load_cases() if phase=='full' or r['id'] in SMOKE]
def failures(kind,case,error,dim=None):atomic(path_score(kind,case,dim),receipt(case,status='failed',error=error,dimension=dim))
