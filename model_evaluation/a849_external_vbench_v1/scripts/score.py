"""Official VBench custom_input, six dimensions; preserve per-video failures."""
import argparse
import math
import subprocess
import sys
import traceback
from common import *

DIMS=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']

def run(stage,shard):
    require_job()
    import torch
    from vbench import VBench
    frozen=read(ROOT/'metadata/freeze.json');items=rows(ROOT/'metadata/inputs.jsonl')
    ids=frozen['smoke'] if stage=='smoke' else [r['id'] for r in items]
    dest=ROOT/'runtime/score_inputs'/stage;dest.mkdir(parents=True,exist_ok=True)
    mapping={};missing=[]
    for model in ['ltx_base','a849','vace']:
        for sid in ids:
            out=ROOT/'outputs'/model/f'{sid}.mp4';status=out.with_suffix('.json')
            key=f'{model}--{sid}'
            if status.is_file() and read(status).get('status')=='complete':
                assert out.is_file() and sha(out)==read(status)['output_sha256']
                dst=dest/f'{key}.mp4'
                try:dst.symlink_to(out)
                except FileExistsError:
                    assert dst.is_symlink() and dst.resolve()==out.resolve(),f'Unexpected score input: {dst}'
                mapping[key]=dict(id=sid,model=model,output_sha256=sha(out))
            else:missing.append(dict(id=sid,model=model,status='missing_generation'))
    result_root=ROOT/'metadata/vbench'/stage;result_root.mkdir(parents=True,exist_ok=True)
    atomic(result_root/f'input_mapping_{shard}.json',dict(mapping=mapping,missing=missing,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl')))
    for dim in DIMS[shard::4]:
        bench=VBench(torch.device('cuda'),str(ROOT/'runtime/VBench/VBench_full_info.json'),str(result_root))
        contract=dict(vbench_revision=subprocess.check_output(['git','-C',str(ROOT/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip(),mode='custom_input',dimension=dim,local=True,read_frame=False,imaging_quality_preprocessing_mode='longer',manifest_sha256=frozen['input_manifest_sha256'],mapping=mapping,script_sha256=sha(__file__))
        receipt=result_root/f'{dim}_receipt.json'
        if receipt.exists() and read(receipt).get('status')=='complete':
            assert read(receipt)['contract']==contract;continue
        try:
            bench.evaluate(str(dest),dim,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
            raw=result_root/f'{dim}_eval_results.json'
            assert raw.is_file()
            atomic(receipt,dict(status='complete',contract=contract,result_sha256=sha(raw)))
        except Exception:
            # Retry by video so a single bad decode/score cannot erase others.
            errors=[dict(scope='batch',error=traceback.format_exc())];per_video={}
            for key in mapping:
                try:
                    name=dim+'--'+key
                    bench.evaluate(str(dest/f'{key}.mp4'),name,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
                    per_video[key]=read(result_root/f'{name}_eval_results.json')
                except Exception:errors.append(dict(video=key,error=traceback.format_exc()))
            atomic(receipt,dict(status='partial' if per_video else 'failed',contract=contract,per_video_results=per_video,errors=errors))
        print(stage,dim,read(receipt)['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','all'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run(a.stage,a.shard)
