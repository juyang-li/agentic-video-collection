import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/'benchmark_full10s_20261112'
FROZEN = ROOT.parent/'frozen300_20261113/package'
NATIVE = Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
DISTILLED = ROOT.parent/'ltx25_singleturn_v1/models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'
DISTILLED_SHA = '31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4'
PILOT = ROOT.parent/'ltx25_distilled_v7_pilot10'
FROZEN_ROOT = FROZEN.parent
SMOKE = ['identity_01','whole_outfit_01','action_01','coinve_01','tripvvt_01','coinve_11']
REMOTE = '/apps/bypy/lance_review/ltx25_distilled_v7_300_review.zip'
PROTOCOL = 'reference_positions_v7_refs_source_target'
SIGMAS = [1., .99375, .9875, .98125, .975, .909375, .725, .421875, 0.]
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths, resolve_path
add_script_paths()
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def weights():
    from encoding_helpers import weight_paths
    record,paths=weight_paths(NATIVE/'runtime/models/native_model_revision.json',NATIVE/'runtime/verification/native_weights_verified.json')
    return record,paths

def contract(item):
    return {'experiment':'LTX2.5 Base（distilled DiT）＋DataMake V7',
        'model_revision':'49b093a39c19169a8a81721dcc541e07e945232f',
        'dit_sha256':DISTILLED_SHA,'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),
        'protocol':PROTOCOL,'sigmas':SIGMAS,'steps':8,'cfg':1,'stg':0,'rescale':0,
        'ref_cfg':1,'seed':42,'noise_seed':10042,'decode_seed':42,'eta':1,'s_noise':1,
        'sampler':'official euler_ancestral_denoising_loop','lora':None,'dtype':'BF16',
        'fps':12,'num_frames':item['num_frames'],'canvas_wh':item['canvas_wh'],'stages':1,
        'implementation_sha256':sha(ROOT/'metadata/implementation.json')}

def verify_implementation():
    receipt=read(ROOT/'metadata/implementation.json')
    for name,digest in receipt['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in receipt['shared_source_sha256'].items(): assert sha(name)==digest,name
