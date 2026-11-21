"""Balance remaining19 tasks over4 GPUs without changing frozen task order.

Only the worker's iteration order is adapted. Model code, per-item seed, inputs,
generation settings and existing worker contracts remain identical to smoke.
"""
import argparse
from common import *

if __name__=='__main__':
    require_job()
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);p.add_argument('--stage',choices=['all'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    import importlib
    worker=importlib.import_module('vace_worker' if a.model=='vace' else 'native_worker')
    original_rows=worker.rows
    manifest=ROOT/'metadata/inputs.jsonl';records=original_rows(manifest)
    reuse_file=ROOT/'metadata/reuse_outputs.json';retained=[]
    if reuse_file.exists():
        reuse=read(reuse_file);assert reuse['new_manifest_sha256']==sha(manifest)
        old={r['id']:r for r in original_rows(ROOT/'metadata/prior24/inputs.jsonl')}
        for r in reuse['reused']:
            if r['model']!=a.model:continue
            item=next(i for i in records if i['id']==r['id']);assert item==old[item['id']]
            status=read(ROOT/'outputs'/a.model/f'{item["id"]}.json')
            assert status['status']=='complete' and status['output_sha256']==r['output_sha256']
            assert sha(ROOT/'outputs'/a.model/f'{item["id"]}.mp4')==r['output_sha256']
            retained.append(item['id'])
    ordered=[r for r in records if r['id'] not in retained]
    assert len(records)==96 and len(ordered)+len(retained)==96
    atomic(ROOT/'metadata'/f'dispatch_all_{a.model}_shard{a.shard}.json',dict(manifest_sha256=sha(manifest),wrapper_sha256=sha(__file__),order=[r['id'] for r in ordered],assigned=[r['id'] for r in ordered[a.shard::4]],retained_unchanged_task_ids=retained,reason='Skip only input-identical reused outputs; all4 revised identity references are generated again. Pending tasks balanced by fixed stride across4 GPUs.'))
    def dispatch_rows(path):
        return ordered if Path(path).resolve()==manifest.resolve() else original_rows(path)
    worker.rows=dispatch_rows
    if a.model=='vace':worker.work(a.stage,a.shard)
    else:worker.work(a.model,a.stage,a.shard)
