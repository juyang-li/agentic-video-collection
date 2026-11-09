#!/usr/bin/env python3
"""Check frozen selection and complete Stage 1 evidence before publishing a reserve."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from run_stage1_metadata import evaluate, read_csv
from run_stage1_quality import required_roles, resolve_local_path
from calibrate_stage1_thresholds import METRICS, decision
from build_stage1_expansion_manifest import candidate_map

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--selection-only',action='store_true')
    args=ap.parse_args()
    out=args.run_dir.resolve()
    selection=json.loads((out/'selection_summary.json').read_text())
    rows=read_csv(out/'manifest.csv')
    assert len(rows)==20000
    assert sha(out/'manifest.csv')==selection['manifest_sha256']
    assert dict(Counter(row['branch'] for row in rows))==selection['quotas']
    ids={row['sample_id'] for row in rows}
    identities={(row['dataset'],row['pair_id']) for row in rows}
    assert len(ids)==len(identities)==len(rows)
    assert not any(row['dataset'] in {'coinve','insave','openve'} for row in rows)
    adjacency=defaultdict(set)
    def link(row):
        if not row.get('source_locator') or not row.get('target_locator'):
            return
        a=(row['dataset'],row['source_locator'])
        b=(row['dataset'],row['target_locator'])
        adjacency[a].add(b)
        adjacency[b].add(a)
    for path in selection['history_manifest_paths']:
        previous=read_csv(ROOT/path)
        used={(row['dataset'],row['pair_id']) for row in previous}
        assert not identities & used, path
        for row in previous:
            link(row)
    final=read_csv(ROOT.parent/'3_lance_lora_finetuning/artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv')
    assert not identities & {(r['dataset'],r['pair_id']) for r in final}
    heldout={(r['dataset'],r[k+'_locator']) for r in final if r['split'] in {'test','validation'} for k in ('source','target')}
    candidate_media={(r['dataset'],r[k+'_locator']) for r in rows for k in ('source','target')}
    assert not heldout & candidate_media
    for row in final:
        link(row)
    populations=candidate_map(ROOT/'filter_config.json',selection['config'],set(selection['quotas']))
    for population in populations.values():
        for row in population:
            link(row)
    reachable=set(heldout)
    frontier=deque(heldout)
    while frontier:
        for neighbor in adjacency.get(frontier.popleft(),()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    assert not candidate_media & reachable
    assert sha(out/'thresholds.json')==sha(ROOT/'artifacts/filter_run_v1/stage1_1_thresholds.json')
    assert sha(out/'policy.json')==sha(ROOT/'stage1_1_future_policy.json')
    for row in rows:
        result=evaluate({**row,'source_local':'','target_local':''})
        assert result['metadata_decision']=='accepted',row['sample_id']
        assert row['stage2_status']=='not_requested'
        assert row['revision']==selection['config']['sources'][row['branch']]['revision']
        assert not row.get('source_video_id') and not row.get('target_video_id')
        if row['dataset']=='reco':
            assert row['source_provenance']==row['target_provenance']=='generation_side_unresolved'
            assert row['quality_checked_roles']=='both'
        if row['branch']=='goku_add':
            assert row['source_provenance']=='edited' and row['target_provenance']=='original'
            assert row['quality_checked_roles']=='source'
        if row['branch']=='goku_remove':
            assert row['source_provenance']=='original' and row['target_provenance']=='edited'
            assert row['quality_checked_roles']=='target'
        if row['dataset']=='egoedit':
            assert set(required_roles(row))=={k for k in ('source','target') if row[k+'_provenance']=='edited'}
    report={'selection_status':'passed','input_pairs':len(rows),
        'manifest_sha256':sha(out/'manifest.csv'),'history_files_checked':len(selection['history_manifest_paths']),
        'metadata_evaluated_pairs':len(rows),'frozen_policy_unchanged':True,
        'heldout_direct_media_overlap':0,'heldout_connected_media_overlap':0,
        'heldout_reachable_media':len(reachable),'pair_id_context_preserved':True}
    if args.selection_only:
        (out/'selection_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
        return

    completion=json.loads((out/'completion_summary.json').read_text())
    acquisition=json.loads((out/'acquisition_state.json').read_text())
    assert acquisition['status'].startswith('complete')
    scores=read_csv(out/'scores.csv')
    score_map={row['video_id']:row for row in scores}
    assert len(score_map)==len(scores)
    decisions=read_csv(out/'pair_decisions.csv')
    survivors=read_csv(out/'stage1_survivors.csv')
    quarantined=read_csv(out/'quarantine.csv')
    decided_ids={row['sample_id'] for row in decisions}
    quarantine_ids={row['sample_id'] for row in quarantined}
    assert len(decided_ids)==len(decisions) and len(quarantine_ids)==len(quarantined)
    assert not decided_ids & quarantine_ids
    assert decided_ids | quarantine_ids == ids
    survivor_ids={row['sample_id'] for row in survivors}
    assert len(survivor_ids)==len(survivors)
    assert survivor_ids=={r['sample_id'] for r in decisions if r['stage1_1_decision'] in {'accepted','uncertain'}}
    thresholds=json.loads((out/'thresholds.json').read_text())['metrics']
    policy=json.loads((out/'policy.json').read_text())
    evaluated_roles=0
    # Recalculate each decision from raw metrics, with independent aggregation.
    for pair in decisions:
        votes=[]
        for role in ('source','target'):
            score=score_map[pair['pair_id']+':'+role]
            assert score['local_path']==pair[role+'_local']
            assert score['technical_status'] in {'pass','fail_short'}
            local=resolve_local_path(score['local_path'])
            assert local.is_file() and local.stat().st_size>0
            duration=float(score['duration_seconds'])
            assert math.isfinite(duration) and duration>0
            if score['technical_status']=='fail_short':
                assert duration<3.000001
                votes.append('rejected')
                continue
            assert duration>=3
            assert int(score['sampled_frames'])==min(12,int(score['frame_count']))
            if role not in required_roles(pair):
                continue
            values={k:float(score[k]) for k in (*METRICS,'frame_diff_mean','frame_diff_p95')}
            assert all(math.isfinite(v) for v in values.values())
            vote,_=decision(values,thresholds)
            nf=policy['near_freeze']
            if values['frame_diff_mean']<=nf['frame_diff_mean_max'] and values['frame_diff_p95']<=nf['frame_diff_p95_max']:
                vote='rejected'
            strict=pair['operation']=='remove' or (pair['operation']=='add' and pair['source_provenance']=='edited' and pair['target_provenance']=='original')
            clarity=policy['remove_edited_side_clarity_hard_gate']['thresholds']
            if strict and (values['musiq_p10']<clarity['musiq_p10_min'] or values['sharpness_p10']<clarity['sharpness_p10_min']):
                vote='rejected'
            votes.append(vote)
            evaluated_roles+=1
        expected='rejected' if 'rejected' in votes else ('uncertain' if 'uncertain' in votes else 'accepted')
        assert pair['stage1_1_decision']==expected,pair['sample_id']
    # All aliases for one actual media path must carry identical raw metrics.
    raw={}
    for score in scores:
        values=tuple(score.get(k) for k in (*METRICS,'frame_diff_mean','frame_diff_p95','duration_seconds','technical_status'))
        previous=raw.setdefault(score['local_path'],values)
        assert previous==values,score['video_id']
    assert completion['input_pairs']==len(rows)
    assert completion['decision_counts']==dict(Counter(r['stage1_1_decision'] for r in decisions))
    assert completion['quarantined_pairs']==len(quarantined)
    assert completion['stage2_status']=='not_requested' and completion['api_judge_calls']==0
    report.update(status='passed',decided_pairs=len(decisions),quarantined_pairs=len(quarantined),
        survivor_pairs=len(survivors),decision_counts=completion['decision_counts'],
        quality_roles_independently_recalculated=evaluated_roles,
        score_rows=len(scores),unique_scored_media=len(raw),
        output_sha256={name:sha(out/name) for name in ['scores.csv','pair_decisions.csv','stage1_survivors.csv','quarantine.csv']})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
