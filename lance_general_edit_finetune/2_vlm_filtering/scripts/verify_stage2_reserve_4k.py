#!/usr/bin/env python3
"""Verify Stage 2 coverage, quota selection, frozen inputs and billing records."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import run_stage2_reserve_4k as run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=run.DEFAULT_RUN)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    out = args.run_dir.resolve()
    config = json.loads((out / 'config.json').read_text())
    source = Path(config['source']) / 'stage1_survivors.csv'
    assert run.sha(source) == config['source_sha256']
    assert run.sha(Path(config['old_pool'])) == config['old_pool_sha256']
    assert run.sha(Path(run.judge.__file__)) == config['judge_code_sha256']
    assert run.sha(Path(run.judge.rect.__file__)) == config['visual_code_sha256']
    assert hashlib.sha256((out / 'judge_prompt.txt').read_text().strip().encode()).hexdigest() == run.PROMPT_HASH
    input_rows = run.judge.read_csv(source)
    source_map = {r['sample_id']: r for r in input_rows}
    candidate = run.judge.read_csv(out / 'candidate_manifest.csv')
    assert {r['sample_id']: r for r in candidate} == source_map
    snapshot = json.loads((out / 'source_verification.json').read_text())
    assert snapshot['status'] == 'passed' and snapshot['heldout_connected_media_overlap'] == 0
    assert snapshot['output_sha256']['stage1_survivors.csv'] == config['source_sha256']

    attempts = run.read_jsonl(out / 'score_attempts.jsonl')
    latest = {a['result']['sample_id']: a['result'] for a in attempts}
    scores = run.judge.read_csv(out / 'gemini_scores.csv')
    score_map = {r['sample_id']: r for r in scores}
    assert len(score_map) == len(scores) and score_map == latest
    assert set(latest) <= set(source_map)
    events = run.read_jsonl(out / 'request_journal.jsonl')
    requests = {}
    for event in events:
        rid = event['request_id']
        if event['event'] == 'started':
            assert rid not in requests
        else:
            assert requests[rid]['event'] == 'started'
            assert requests[rid]['sample_id'] == event['sample_id']
        requests[rid] = event
    assert not any(e['event'] == 'started' for e in requests.values()), 'unresolved in-flight request'
    generations = {e['generation_id']: e for e in requests.values() if e.get('generation_id')}
    assert len(generations) == sum(bool(e.get('generation_id')) for e in requests.values())
    assert max(Counter(e['sample_id'] for e in requests.values()).values(), default=0) <= 3
    replayed = 0
    for attempt in attempts:
        r = attempt['result']
        row = source_map[r['sample_id']]
        assert r['prompt_sha256'] == run.PROMPT_HASH
        assert r['model_id'] == run.MODEL
        view = run.judge.build_audit_view(row)
        assert r['training_instruction'] == row['instruction']
        assert r['audit_operation'] == view.audit_operation
        assert r['audit_left_training_role'] == view.left_training_role
        assert r['audit_right_training_role'] == view.right_training_role
        if r['parse_status'] == 'pass':
            e = generations[r['generation_id']]
            assert e['sample_id'] == r['sample_id'] and e['provider'] == 'Google AI Studio'
            assert r['sampled_full_pairs'] == '5'
            assert r['reasoning_effort'] == 'minimal'
            data = e['response']
            raw = run.judge.extract_message(data)
            assert raw == r['raw_response']
            value = run.judge.rect.extract_json_object(raw)
            normalized = run.judge.normalize_judgment(value, removal_expected=view.audit_operation == 'remove')
            assert normalized['decision'] == r['model_decision']
            for field in run.judge.CHECK_FIELDS:
                assert normalized[field] == r[field]
            if r['model_decision'] == 'accepted':
                assert all(r[f] == 'pass' for f in run.judge.CHECK_FIELDS)
            if e['usage'].get('cost') is not None:
                assert abs(float(r['cost_usd']) - float(e['usage']['cost'])) < 1e-8
            replayed += 1

    pools = {s: run.judge.read_csv(out / (s + '.csv')) for s in ('accepted', 'rejected', 'quarantine', 'not_requested')}
    seen = set()
    for state, rows in pools.items():
        ids = {r['sample_id'] for r in rows}
        assert len(ids) == len(rows) and not seen & ids
        seen.update(ids)
        for row in rows:
            assert row['stage2_status'] == state
            sid = row['sample_id']
            if state == 'not_requested':
                assert sid not in latest
            else:
                assert run.status(latest[sid]) == state
            # Only status and next-stage columns may change from frozen Stage 1.
            for key, value in source_map[sid].items():
                if key not in {'stage2_status', 'next_stage'}:
                    assert row[key] == value, (sid, key)
    assert seen == set(source_map)
    accepted = pools['accepted']
    for row in accepted:
        for role in ('source', 'target'):
            path = run.judge.resolve_media(row[role + '_local'])
            assert path.is_file() and path.stat().st_size > 0
    old = run.judge.read_csv(Path(config['old_pool']))
    old_pair_ids = {(r['dataset'], r['pair_id']) for r in old}
    assert not old_pair_ids & {(r['dataset'], r['pair_id']) for r in accepted}
    heldout_media = {(r['dataset'], r[role + '_locator']) for r in old if r['split'] in {'validation', 'test'} for role in ('source', 'target')}
    new_media = {(r['dataset'], r[role + '_locator']) for r in accepted for role in ('source', 'target')}
    assert not heldout_media & new_media
    assert not any(r['dataset'] == 'coinve' for r in input_rows)
    if not args.partial:
        assert dict(Counter(r['branch'] for r in accepted)) == run.TARGETS
        final = run.judge.read_csv(out / 'final_edit_pool_4k.csv')
        assert len(final) == 4000 and {r['sample_id'] for r in final} == {r['sample_id'] for r in accepted}
        assert {r['sample_id']: r for r in final} == {r['sample_id']: r for r in accepted}
    recorded_cost = sum(float(e.get('usage', {}).get('cost') or 0) for e in requests.values())
    unknown_cost = sum(e.get('usage', {}).get('cost') is None for e in requests.values())
    progress = json.loads((out / 'progress.json').read_text())
    assert abs(progress['recorded_cost_usd'] - recorded_cost) < 1e-8
    assert progress['requests_without_reported_cost'] == unknown_cost
    reconciled = {}
    if (out / 'request_cost_reconciliation.json').exists():
        reconciliation = json.loads((out / 'request_cost_reconciliation.json').read_text())
        diagnostic = json.loads((out / 'access_diagnostic.json').read_text())
        assert run.sha(out / 'access_diagnostic.json') == reconciliation['diagnostic_sha256']
        assert reconciliation['new_key_usage_before_run_usd'] == reconciliation['new_key_usage_after_run_usd'] == diagnostic['new_key_usage_usd'] == 0
        reconciled = reconciliation['request_cost_usd']
        for rid, cost in reconciled.items():
            assert cost == 0 and requests[rid]['http_status'] == 403
            assert requests[rid]['event'] == 'transport_error'
        assert len(reconciled) == diagnostic['new_run_requests']
    expected_accounted = sum(float(e.get('usage', {}).get('cost')) if e.get('usage', {}).get('cost') is not None
                             else reconciled.get(rid, run.REQUEST_RESERVE) for rid, e in requests.items())
    assert abs(progress['conservative_accounted_cost_usd'] - expected_accounted) < 1e-8
    assert progress['conservative_accounted_cost_usd'] <= config['budget_usd']
    report = dict(status='passed', complete=not args.partial, input_pairs=len(input_rows),
                  judged_pairs=len(scores), accepted_pairs=len(accepted),
                  decision_counts={s: len(rs) for s, rs in pools.items()},
                  accepted_by_branch=dict(Counter(r['branch'] for r in accepted)),
                  api_requests=len(requests), parsed_responses_replayed=replayed,
                  recorded_cost_usd=recorded_cost, requests_without_reported_cost=unknown_cost,
                  requests_reconciled_no_charge=len(reconciled),
                  valid_decided_pairs=len(pools['accepted'])+len(pools['rejected']),
                  historical_pair_overlap=0, heldout_direct_media_overlap=0,
                  heldout_connected_overlap_source_verification=0,
                  historical_pool_unchanged=True, coinve_included=False,
                  output_sha256={p.name: run.sha(p) for p in out.glob('*.csv')})
    run.atomic_json(out / ('partial_verification.json' if args.partial else 'verification.json'), report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
