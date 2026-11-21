"""Run only frozen targets with matching reviewed source/mask receipts."""
import argparse,importlib
from common import *
from contracts import digest

def main():
    require_job();p=argparse.ArgumentParser();p.add_argument('--model',choices=['a849','ltx_base','vace'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    manifest=ROOT/'metadata/inputs.jsonl';gate=read(ROOT/'metadata/generation_gate.json')
    assert gate['status']=='approved' and gate['manifest_sha256']==sha(manifest)
    all_items=rows(manifest);assets={r['id']:r for r in gate['assets']}
    targets=read(ROOT/'metadata/generation_targets.json')[a.model]
    assert len(targets)==len(set(targets)) and set(targets)<={r['id']for r in all_items}
    for item in all_items:
        sid=item['id'];assert digest(item)==assets[sid]['input_digest']
        assert sha(ROOT/'inputs/prepared'/sid/'pixels.json')==assets[sid]['source_pixels_receipt_sha256']
        assert sha(ROOT/'conditions'/sid/'condition.json')==assets[sid]['condition_receipt_sha256']
    pending=[r for r in all_items if r['id'] in targets]
    worker=importlib.import_module('vace_distilled_worker'if a.model=='vace'else'native_worker');original=worker.rows
    worker.rows=lambda path:pending if Path(path).resolve()==manifest.resolve()else original(path)
    if a.model=='vace':worker.work('all',a.shard)
    else:worker.work(a.model,'all',a.shard)

if __name__=='__main__':main()
