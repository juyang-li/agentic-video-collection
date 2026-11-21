"""Source-only human candidate expansion; no model-output-driven selection."""
import concurrent.futures
import re
from urllib.parse import quote
from common import *

def main():
    require_job()
    import requests
    from candidates import sheet
    pinned=read(ROOT/'metadata/omni_revision.json')
    candidates=[]
    for name in ['spatial','temporal']:
        for rec in read(ROOT/'inputs/omni'/f'{name}_benchmark.json'):
            if re.search(r'\b(man|woman|girl|boy|person|people|men|women|children|child|baby|surfer|surfers)\b',rec['editing_instruction'],re.I) or '/Clothes_change/' in rec['source_video']:
                candidates.append(rec)
    atomic(ROOT/'metadata/expansion_candidate_scope.json',dict(seed=20261112,revision=pinned['sha'],selection='All human-keyword spatial/temporal records plus clothes category; source clips only, original edit is provenance and not the custom task',candidate_ids=[r['video_id'] for r in candidates]))
    def download(rec):
        path=ROOT/'inputs/omni'/rec['source_video']
        if not path.exists():
            session=requests.Session();session.trust_env=False
            url='https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/resolve/'+pinned['sha']+'/'+quote(rec['source_video'])
            res=session.get(url,timeout=180);res.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.download');tmp.write_bytes(res.content);tmp.replace(path)
        return dict(id=rec['video_id'],path=str(path),instruction=rec['editing_instruction'],official=rec,source_sha256=sha(path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records=list(pool.map(download,candidates))
    jsonl(ROOT/'metadata/omni_candidates_unreviewed.jsonl',records)
    pages=sheet(records,'expanded_omni_candidates')
    jsonl(ROOT/'metadata/omni_candidates.jsonl',records)
    atomic(ROOT/'metadata/expanded_candidate_sheets.json',pages)
    print('Human candidate sources',len(records),'sheets',len(pages),flush=True)

if __name__=='__main__':main()
