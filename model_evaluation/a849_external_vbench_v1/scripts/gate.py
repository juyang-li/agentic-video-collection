"""Prevent generation before freeze and prevent expansion before smoke review."""
import argparse
from common import *

def main(before=None,after=None):
    require_job()
    frozen=read(ROOT/'metadata/freeze.json')
    assert frozen['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    if before:
        overlap=read(ROOT/'metadata/training_overlap.json')
        assert not overlap['source_hits'] and not overlap['hits']
        assert overlap['status']=='complete'
        review=read(ROOT/'metadata/input_visual_review.json')
        assert review['input_manifest_sha256']==frozen['input_manifest_sha256']
        required=[r for r in inputs if before=='all' or r['id'] in frozen['smoke']]
        assert {r['id'] for r in required}<=set(review['approved_ids'])
        for r in required:
            c=read(ROOT/'conditions'/r['id']/'condition.json')
            assert c['status']=='complete' and c['input_manifest_sha256']==frozen['input_manifest_sha256']
        if before=='all':
            if frozen.get('smoke_gate')=='skipped_by_explicit_user_request':
                authorization=read(ROOT/'metadata/full_launch_authorization.json')
                assert authorization['input_manifest_sha256']==frozen['input_manifest_sha256']
                assert authorization['explicit_user_skip_smoke'] is True and authorization['status']=='references_verified'
            else:
                smoke=read(ROOT/'metadata/smoke_review.json')
                assert smoke['input_manifest_sha256']==frozen['input_manifest_sha256'] and smoke['expand'] is True
                assert set(smoke['reviewed_ids'])==set(frozen['smoke'])
                assert read(ROOT/'metadata/smoke_media_validation.json')['status']=='passed'
                scores=read(ROOT/'metadata/smoke_group_metrics.json')
                assert scores['valid_scores']==scores['expected_scores']==90
    if after:
        ids=frozen['smoke'] if after=='smoke' else [r['id'] for r in inputs]
        failures=[]
        for m in ['ltx_base','a849','vace']:
            for sid in ids:
                p=ROOT/'outputs'/m/(sid+'.json')
                if not p.exists() or read(p).get('status')!='complete':failures.append((m,sid))
        atomic(ROOT/'metadata'/f'{after}_generation_gate.json',dict(status='complete' if not failures else 'partial',failures=failures))
        # Failed samples are recorded; expansion is a separate visual decision.
        print(after,'failures',failures,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before');p.add_argument('--after');a=p.parse_args();main(a.before,a.after)
