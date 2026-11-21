"""CPU-only adapter staging; full preflight still requires the reviewed mask/input gate."""
import concurrent.futures,importlib.util
from common import *
from contracts import digest
from h3_prepare import prepare_source,TIME_CODE

def main():
    assert not os.environ.get('SLURM_JOB_ID')
    spec=importlib.util.spec_from_file_location('h3_native_time',TIME_CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    nframes=mod.minimax_h3_align_frame_count(4*24);assert nframes==107
    jobs={}
    for item in rows(ROOT/'metadata/inputs.jsonl'):
        p=read(ROOT/'inputs/prepared'/item['id']/'pixels.json');assert p['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
        canvas=[1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344]
        key=digest(dict(frames=p['frame_sha256'],canvas=canvas,native_frames=nframes));jobs[key]=(key,item,canvas,nframes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(prepare_source,jobs.values()))
    atomic(ROOT/'metadata/h3_source_staging.json',dict(status='complete',sources=len(jobs),no_generation=True,full_preflight_and_direct_input_gate_still_required=True))

if __name__=='__main__':main()
