"""Skip only audited reuses, balance remaining work while preserving each task seed."""
import argparse,importlib
from common import *
from contracts import digest,generation_input

def main():
 require_job();p=argparse.ArgumentParser();p.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 frozen=read(ROOT/'metadata/freeze.json');assert frozen['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved' and gate['manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 worker=importlib.import_module('vace_worker' if a.model=='vace' else 'native_worker');original_rows=worker.rows;manifest=ROOT/'metadata/inputs.jsonl';items=rows(manifest)
 assets={r['id']:r for r in gate['assets']}
 for item in items:
  sid=item['id'];assert digest(item)==assets[sid]['input_digest']
  assert sha(ROOT/'conditions'/sid/'condition.json')==assets[sid]['condition_receipt_sha256']
  assert sha(ROOT/'inputs/prepared'/sid/'pixels.json')==assets[sid]['source_pixels_receipt_sha256']
 reuse=read(ROOT/'metadata/reuse_outputs.json');assert reuse['new_manifest_sha256']==sha(manifest)
 skipped=set()
 for proof in reuse['reused']:
  if proof['model']!=a.model:continue
  item=next(r for r in items if r['id']==proof['id']);c=read(ROOT/'conditions'/item['id']/'condition.json') if a.model=='vace' else None
  assert digest(generation_input(item,a.model,c))==proof['effective_input_sha256']
  assert sha(ROOT/'outputs'/a.model/f"{item['id']}.mp4")==proof['output_sha256'];skipped.add(item['id'])
 pending=[r for r in items if r['id'] not in skipped]
 worker.rows=lambda path:pending if Path(path).resolve()==manifest.resolve() else original_rows(path)
 if a.model=='vace':worker.work('all',a.shard)
 else:worker.work(a.model,'all',a.shard)
if __name__=='__main__':main()
