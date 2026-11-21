"""Only frozen generation targets; unchanged parent outputs are reused."""
import argparse,importlib
from common import *
from contracts import digest

def main():
 require_job();parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);parser.add_argument('--shard',type=int,required=True);a=parser.parse_args()
 manifest=ROOT/'metadata/inputs.jsonl';gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved'and gate['manifest_sha256']==sha(manifest)
 items=rows(manifest);assets={r['id']:r for r in gate['assets']}
 for item in items:
  sid=item['id'];assert digest(item)==assets[sid]['input_digest'];assert sha(ROOT/'conditions'/sid/'condition.json')==assets[sid]['condition_receipt_sha256'];assert sha(ROOT/'inputs/prepared'/sid/'pixels.json')==assets[sid]['source_pixels_receipt_sha256']
 target_ids=set(read(ROOT/'metadata/generation_targets.json')[a.model]);pending=[r for r in items if r['id']in target_ids];assert len(pending)==len(target_ids)==28
 worker=importlib.import_module('vace_worker'if a.model=='vace'else'native_worker');original_rows=worker.rows
 worker.rows=lambda path:pending if Path(path).resolve()==manifest.resolve()else original_rows(path)
 if a.model=='vace':worker.work('all',a.shard)
 else:worker.work(a.model,'all',a.shard)
if __name__=='__main__':main()
