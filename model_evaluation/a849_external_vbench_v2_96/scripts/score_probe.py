"""One-video compatibility check before smoke scoring; not a benchmark report."""
import traceback
from common import *

if __name__=='__main__':
    require_job()
    import importlib
    import subprocess
    import torch
    from vbench import VBench
    from score import DIMS
    results={};out=ROOT/'metadata/vbench/probe'
    for dim in DIMS:
        try:
            bench=VBench(torch.device('cuda'),str(ROOT/'runtime/VBench/VBench_full_info.json'),str(out))
            bench.evaluate(str(ROOT/'outputs/ltx_base/identity_01.mp4'),dim,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
            results[dim]={'status':'complete'}
        except Exception:
            results[dim]={'status':'failed','error':traceback.format_exc()};traceback.print_exc()
        atomic(ROOT/'metadata/vbench_probe.json',results)
        print(dim,results[dim]['status'],flush=True)
    atomic(ROOT/'metadata/vbench_install.json',dict(status='passed' if all(r['status']=='complete' for r in results.values()) else 'failed',revision=subprocess.check_output(['git','-C',str(ROOT/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip(),dino_revision=subprocess.check_output(['git','-C',str(ROOT/'runtime/vbench_models/dino_model/facebookresearch_dino_main'),'rev-parse','HEAD'],text=True).strip(),pip_freeze=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True),setup_note='Unmodified official source imported by .pth; setup.py rejects CUDA12.6. Missing expired-certificate weights fetched from pinned TLS mirror and verified using Git LFS SHA256.'))
