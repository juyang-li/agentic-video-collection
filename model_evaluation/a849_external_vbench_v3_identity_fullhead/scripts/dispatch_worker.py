import argparse,importlib
from common import *
if __name__=='__main__':
 require_job();p=argparse.ArgumentParser();p.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 assert read(ROOT/'metadata/input_review.json')['status']=='approved'
 worker=importlib.import_module('vace_worker' if a.model=='vace' else 'native_worker');original=worker.rows;manifest=ROOT/'metadata/inputs.jsonl';items=[r for r in rows(manifest) if r['id']!='identity_01']
 worker.rows=lambda path:items if Path(path).resolve()==manifest.resolve() else original(path)
 if a.model=='vace':worker.work('all',a.shard)
 else:worker.work(a.model,'all',a.shard)
