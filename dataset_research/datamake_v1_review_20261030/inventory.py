from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json, gzip, hashlib

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/00_engine/outputs/dataset_package_v1')
DATA=SOURCE/'dataset_v1'

def main():
    (ROOT/'metadata').mkdir(parents=True,exist_ok=True)
    for kind,key in [('single_turn','samples'),('multi_turn','chains')]:
        cache=ROOT/'metadata'/f'{kind}.jsonl.gz'
        index=DATA/f'chains/{kind}/index.json'
        records=json.loads(index.read_text())[key]
        if not cache.exists():
            def read(r):
                p=DATA/r['path'];b=p.read_bytes();d=json.loads(b)
                d['_path']=r['path'];d['_sha256']=hashlib.sha256(b).hexdigest()
                return d
            with ThreadPoolExecutor(max_workers=24) as pool,gzip.open(cache,'wt',encoding='utf-8') as f:
                for i,d in enumerate(pool.map(read,records)):
                    f.write(json.dumps(d,ensure_ascii=False)+'\n')
                    if (i+1)%4000==0: print(kind,i+1,flush=True)
        with gzip.open(cache,'rt') as f: rows=[json.loads(s) for s in f]
        turns=rows if kind=='single_turn' else [t for c in rows for t in c['turns']]
        summary={'count':len(rows),'turns':len(turns),'axes':dict(Counter(t['axis'] for t in turns)),
          'axis_mode':dict(Counter(t['axis']+'/'+t['mode'] for t in turns)),
          'splits':dict(Counter(r['split'] for r in rows)),
          'ref_types':dict(Counter(r['type'] for t in turns for r in t['references'])),
          'kinds':dict(Counter(r.get('kind','single') for r in rows)),
          'lengths':dict(Counter(r.get('length',1) for r in rows)),
          'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()}
        (ROOT/'metadata'/f'{kind}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(kind,json.dumps(summary,ensure_ascii=False),flush=True)
        if kind=='single_turn':
            seen=set()
            for t in turns:
                k=t['axis']+'/'+t['mode']
                if k not in seen:
                    seen.add(k);print('EXAMPLE',json.dumps(t,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
