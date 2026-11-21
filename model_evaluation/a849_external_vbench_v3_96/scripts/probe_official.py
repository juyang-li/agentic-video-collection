"""A real API/decode probe of pinned official temporal and static functions."""
import sys,argparse,importlib.util
from common import *
def main():
 require_job()
 vendor=ROOT.parent/'a849_external_vbench_v2_96/runtime/VBench'
 from vbench.temporal_flickering import cal_score
 path=ROOT/'inputs/prepared/object_removal_02/source.mp4'
 spec=importlib.util.spec_from_file_location('official_static_probe',vendor/'static_filter.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 weights=ROOT.parent/'a849_external_vbench_v2_96/runtime/vbench_models/raft_model/models/raft-things.pth'
 filt=mod.StaticFilter(argparse.Namespace(model=str(weights),small=False,mixed_precision=False,alternate_corr=False),device='cuda')
 raw=float(cal_score(str(path)));result=bool(filt.infer(str(path)))
 atomic(ROOT/'metadata/official_metric_probe.json',dict(status='passed',video=str(path),media_sha256=sha(path),official_flicker_score=raw,official_static=result,static_params=filt.params,static_source_sha256=sha(vendor/'static_filter.py'),note='One actual source decode/API probe, not the final paired-subset result.'))
 print(raw,result,flush=True)
if __name__=='__main__':main()
