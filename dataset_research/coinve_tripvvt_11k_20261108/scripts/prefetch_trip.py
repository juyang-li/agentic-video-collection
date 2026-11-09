"""Prepare the frozen 3K TripVVT pool in isolation while CoinVE downloads.

Publish complete media and CPU records with atomic hard links, never overwrite
another worker's files. Main preparation still owns exclusion, sampling and splits.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import copy
import os
from pathlib import Path
import prepare as p

MAIN=p.ROOT
STAGE=MAIN/'trip_prefetch'


def publish_json(path,value,label):
    temporary=STAGE/'published'/(path.stem+'.'+label+'.json')
    p.save(temporary,value);path.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(temporary,path)
    except FileExistsError:pass


def one(row):
    previous=MAIN/'metadata/technical'/f"{row['sample_id']}.json"
    if previous.exists():
        cached=p.read(previous)
        fingerprints={role:dict(path=str(p.asset_path(row,role).resolve()),bytes=p.asset_path(row,role).stat().st_size,
                    mtime_ns=p.asset_path(row,role).stat().st_mtime_ns) for role in p.roles(row) if p.asset_path(row,role).is_file()}
        scores=MAIN/'metadata/scores'/f"{row['sample_id']}.json"
        if cached.get('fingerprints')==fingerprints and len(fingerprints)==len(p.roles(row)):
            if cached['status']!='ok' or (scores.exists() and p.read(scores).get('script_sha256')==p.sha(p.TRIP/'score_pair_similarity.py')):
                return dict(sample_id=row['sample_id'],status=cached['status'],reused_main_record=True)
    result=p.process(row)
    if result.get('reason')=='official_asset_missing':
        publish_json(MAIN/'metadata/technical'/f"{row['sample_id']}.json",result,'technical')
        return dict(sample_id=row['sample_id'],status=result['status'],reason=result['reason'])
    paths={}
    for role in p.roles(row):
        source=p.asset_path(row,role)
        existing=p.HISTORY['assets'].get('tripvvt:'+row[role+'_member'])
        destination=Path(existing) if existing and Path(existing).is_file() else MAIN/'media/tripvvt'/source.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source!=destination:
            try:os.link(source,destination)
            except FileExistsError:
                assert os.path.samefile(source,destination) or p.sha(source)==p.sha(destination),'different media under the same frozen ID'
        paths[role]=destination.resolve()
    record=copy.deepcopy(result)
    record['fingerprints']={role:dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for role,path in paths.items()}
    if record['status']=='ok':
        for role,path in paths.items():record['media'][role]['path']=str(path)
        score=p.read(STAGE/'metadata/scores'/f"{row['sample_id']}.json")
        for role in ('source','target'):score[role]=str(paths[role])
        publish_json(MAIN/'metadata/scores'/f"{row['sample_id']}.json",score,'score')
    publish_json(MAIN/'metadata/technical'/f"{row['sample_id']}.json",record,'technical')
    return dict(sample_id=row['sample_id'],status=record['status'])


def main():
    STAGE.mkdir(exist_ok=True)
    p.HISTORY=p.read(MAIN/'metadata/history.json')
    calibration=p.read(p.HISTORY['score_cache'])
    assert calibration['script_sha256']==p.sha(p.TRIP/'score_pair_similarity.py')
    p.SCORE_CACHE=calibration['records']
    rows=[r for r in p.jsonl(MAIN/'metadata/candidates.jsonl') if r['dataset']=='tripvvt' and r['candidate_rank']<=1000]
    assert len(rows)==3000 and len({r['target_member'] for r in rows})==3000
    for row in rows:
        for role in p.roles(row):
            path=MAIN/'media/tripvvt'/(row['sample_id']+'_'+role+('.png' if role=='reference' else '.mp4'))
            if path.is_file():p.HISTORY['assets'].setdefault('tripvvt:'+row[role+'_member'],str(path))
    p.ROOT=STAGE
    completed=[];failed=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,r):r for r in rows}
        for future in as_completed(futures):
            row=futures[future]
            try:completed.append(future.result())
            except Exception as error:
                failed.append(dict(sample_id=row['sample_id'],error_type=type(error).__name__))
            if (len(completed)+len(failed))%25==0:
                print('TripVVT prefetch',len(completed),'/',3000,'retry_required',len(failed),flush=True)
                p.save(STAGE/'status.json',dict(completed=len(completed),expected=3000,failed=failed,updated_utc=datetime.now(timezone.utc).isoformat()))
    p.save(STAGE/'status.json',dict(status='complete' if not failed else 'retry_required',completed=len(completed),expected=3000,failed=failed,
        records=completed,updated_utc=datetime.now(timezone.utc).isoformat()))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
