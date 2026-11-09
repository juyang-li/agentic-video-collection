#!/usr/bin/env python3
"""Fill six frozen quotas from the verified reserve, using the production judge.

Every HTTP attempt and every parsed result is append-only. Resume never reruns
valid decisions. Media, visual preparation, prompt and decision rules are reused
unchanged from the first 4K. This script does not assign new train/test splits.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.error
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import run_openrouter_gemini_flash_judge as frozen

judge = frozen.judge
ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT.parent / '1_specialist_filtering'
DEFAULT_SOURCE = S1 / 'artifacts/stage1_reserve_20k_20260906'
DEFAULT_RUN = ROOT / 'artifacts/stage2_next4k_20260906'
MODEL = 'google/gemini-3.8-flash'
PROMPT_HASH = '32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c'
TARGETS = dict(goku_remove=600, goku_add=400, egoedit_replace=1000,
               reco_add=600, reco_remove=600, reco_replace=800)
SEED = 20260906
REQUEST_RESERVE = 0.03


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(row):
    return hashlib.sha256(f"{SEED}:{row['sample_id']}".encode()).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    tmp = path.with_suffix('.part.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields):
    tmp = path.with_suffix('.part.csv')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def status(result):
    if result.get('parse_status') == 'pass' and result.get('model_decision') in {'accepted', 'rejected'}:
        return result['model_decision']
    return 'quarantine'


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    verification = json.loads((source / 'verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['heldout_connected_media_overlap'] == 0
    input_path = source / 'stage1_survivors.csv'
    assert sha(input_path) == verification['output_sha256']['stage1_survivors.csv']
    rows = judge.read_csv(input_path)
    assert len(rows) == 18662
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert len({(r['dataset'], r['pair_id']) for r in rows}) == len(rows)
    for row in rows:
        assert row['branch'] in TARGETS and row['dataset'] != 'coinve'
        assert row['stage1_1_decision'] in {'accepted', 'uncertain'}
        assert row['stage1_technical_status'] == 'pass'
        judge.build_audit_view(row)
        for role in ('source', 'target'):
            p = judge.resolve_media(row[role + '_local'])
            assert p.is_file() and p.stat().st_size > 0, str(p)
    prompt = judge.DEFAULT_PROMPT.read_text().strip()
    assert judge.prompt_hash(prompt) == PROMPT_HASH
    old_pool = ROOT.parent / '3_lance_lora_finetuning/artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv'
    config = dict(version='next4k_gemini_only_v1', source=str(source),
                  source_sha256=sha(input_path), source_rows=len(rows),
                  old_pool=str(old_pool), old_pool_sha256=sha(old_pool),
                  seed=SEED, target_by_branch=TARGETS, model=MODEL,
                  prompt_sha256=PROMPT_HASH, reasoning_effort='minimal', max_tokens=2048,
                  provider_order=['google-ai-studio'], data_collection='deny', allow_fallbacks=False,
                  frame_samples=9, full_pair_count=5, roi_pair_count=5,
                  full_max_long_edge=640, roi_max_long_edge=640, analysis_long_edge=512,
                  minimum_roi_reliability=0.12, jpeg_quality=90, temperature=0.0,
                  max_api_attempts_per_pair=3, request_reserve_usd=REQUEST_RESERVE,
                  budget_usd=55.0, coinve_included=False,
                  selection_policy='branch seeded SHA256 order; stop each branch at its accepted quota',
                  split_policy='unassigned; existing first-batch splits remain frozen',
                  judge_code_sha256=sha(Path(judge.__file__)),
                  visual_code_sha256=sha(Path(judge.rect.__file__)))
    if (out / 'config.json').exists():
        assert json.loads((out / 'config.json').read_text()) == config, 'run configuration changed'
    else:
        atomic_json(out / 'config.json', config)
        (out / 'judge_prompt.txt').write_text(prompt + '\n')
        (out / 'source_verification.json').write_bytes((source / 'verification.json').read_bytes())
        write_csv(out / 'candidate_manifest.csv', sorted(rows, key=lambda r: (r['branch'], stable_key(r))), list(rows[0]))
    return rows, config, prompt


class RecordedTransport:
    def __init__(self, out, budget):
        self.out, self.budget = out, budget
        self.lock = threading.Lock()
        self.context = threading.local()
        self.blocked = False
        self.halt_reason = ''
        self.original = judge.post_json
        self.events = read_jsonl(out / 'request_journal.jsonl')
        self.latest = {e['request_id']: e for e in self.events}
        reconciliation = out / 'request_cost_reconciliation.json'
        self.reconciled = json.loads(reconciliation.read_text())['request_cost_usd'] if reconciliation.exists() else {}
        self.charge = sum(self.accounted(e) for e in self.latest.values())

    def accounted(self, event):
        cost = event.get('usage', {}).get('cost')
        if cost is not None:
            return float(cost)
        return float(self.reconciled.get(event['request_id'], REQUEST_RESERVE))

    def append(self, name, value):
        with (self.out / name).open('a', encoding='utf-8') as f:
            f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
            f.flush()

    def __call__(self, api_url, api_key, payload, timeout, max_attempts):
        assert max_attempts == 1  # Retries are recorded individually by the outer loop.
        with self.lock:
            if self.charge + REQUEST_RESERVE > self.budget:
                self.blocked = True
                self.halt_reason = 'budget_limit'
                raise RuntimeError('run_budget_limit_reached')
            event = dict(request_id=uuid.uuid4().hex, sample_id=self.context.sample_id,
                         attempt=self.context.attempt, event='started', timestamp=time.time())
            self.charge += REQUEST_RESERVE
            self.latest[event['request_id']] = event
            self.append('request_journal.jsonl', event)
        try:
            data, attempts, elapsed = self.original(api_url, api_key, payload, timeout, 1)
        except Exception as exc:
            # Do not persist HTTP bodies that might echo request headers or content.
            cause = exc.__cause__
            code = cause.code if isinstance(cause, urllib.error.HTTPError) else None
            finished = dict(event, event='transport_error', timestamp=time.time(),
                            error_type=type(exc).__name__, http_status=code)
            with self.lock:
                self.latest[event['request_id']] = finished
                self.append('request_journal.jsonl', finished)
                if code in {401, 402, 403}:
                    self.blocked = True
                    self.halt_reason = 'account_unavailable'
            self.context.http_status = code
            raise RuntimeError(f'api_transport_error_http_{code or "unknown"}') from None
        finished = dict(event, event='response', timestamp=time.time(),
                        generation_id=data.get('id'), provider=data.get('provider'),
                        model=data.get('model'), usage=data.get('usage') or {}, response=data)
        with self.lock:
            self.charge += self.accounted(finished) - REQUEST_RESERVE
            self.latest[event['request_id']] = finished
            self.append('request_journal.jsonl', finished)
        return data, attempts, elapsed

    def summary(self):
        with self.lock:
            events = list(self.latest.values())
        costs = [float(e['usage']['cost']) for e in events if e.get('usage', {}).get('cost') is not None]
        return dict(api_requests=len(events), recorded_cost_usd=sum(costs),
                    requests_without_reported_cost=len(events)-len(costs),
                    requests_reconciled_no_charge=sum(e['request_id'] in self.reconciled for e in events),
                    conservative_accounted_cost_usd=sum(self.accounted(e) for e in events),
                    budget_usd=self.budget)


def score_pair(row, common, transport, attempts_before):
    result = None
    for attempt in range(attempts_before + 1, 4):
        transport.context.sample_id = row['sample_id']
        transport.context.attempt = attempt
        transport.context.http_status = None
        result = judge.score_row(row, **common)
        # Provider/model drift must not produce canonical accepted records.
        if result.get('parse_status') == 'pass':
            if result.get('provider') != 'Google AI Studio' or result.get('model_id') != MODEL:
                result.update(parse_status='error', model_decision='uncertain', error='unexpected_model_or_provider')
        with transport.lock:
            transport.append('score_attempts.jsonl', dict(attempt=attempt, result=result))
        if result['parse_status'] == 'pass' or transport.blocked:
            break
        if transport.context.http_status in {400, 401, 402, 403, 404}:
            break
        if attempt < 3:
            time.sleep(2 ** attempt)
    assert result is not None
    return result


def save_progress(out, rows, latest, transport, config, state):
    ordered = [latest[r['sample_id']] for r in rows if r['sample_id'] in latest]
    counts = Counter(status(r) for r in ordered)
    accepted = Counter(r['branch'] for r in ordered if status(r) == 'accepted')
    by_branch = {b: dict(Counter(status(r) for r in ordered if r['branch'] == b)) for b in TARGETS}
    summary = dict(status=state, updated_at=time.time(), candidate_pairs=len(rows),
                   judged_pairs=len(ordered), decision_counts=dict(counts), by_branch=by_branch,
                   attempted_pairs=len(ordered), valid_decided_pairs=counts['accepted']+counts['rejected'],
                   deficits={b: n-accepted[b] for b, n in TARGETS.items()},
                   **transport.summary(), coinve_included=False)
    write_csv(out / 'gemini_scores.csv', ordered, judge.FIELDS)
    atomic_json(out / 'progress.json', summary)
    return summary


def finalize(out, rows, latest, config):
    outputs = {k: [] for k in ('accepted', 'rejected', 'quarantine', 'not_requested')}
    for row in rows:
        result = latest.get(row['sample_id'])
        state = status(result) if result else 'not_requested'
        merged = dict(row, stage2_status=state, stage2_model=MODEL if result else '',
                      stage2_prompt_sha256=PROMPT_HASH if result else '',
                      stage2_generation_id=result.get('generation_id', '') if result else '',
                      stage2_reason_codes_json=result.get('reason_codes_json', '') if result else '',
                      stage2_evidence_json=result.get('evidence_json', '') if result else '',
                      next_stage='accepted_pool' if state == 'accepted' else state)
        outputs[state].append(merged)
    fields = list(next(iter(outputs.values()))[0]) if outputs['accepted'] else list(rows[0])
    for state, records in outputs.items():
        write_csv(out / f'{state}.csv', records, fields)
    accepted = outputs['accepted']
    counts = Counter(r['branch'] for r in accepted)
    if dict(counts) == TARGETS:
        selected = sorted(accepted, key=lambda r: (list(TARGETS).index(r['branch']), stable_key(r)))
        write_csv(out / 'final_edit_pool_4k.csv', selected, fields)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    ap.add_argument('--run', action='store_true', help='Perform authorized paid Stage 2 calls; otherwise only prepare.')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--max-new-pairs', type=int)
    ap.add_argument('--retry-account-errors', action='store_true',
                    help='After account access is restored, retry prior HTTP 401/402/403 failures within the three-attempt limit.')
    args = ap.parse_args()
    assert 1 <= args.concurrency <= 16
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out / '.run.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows, config, prompt = prepare(args.source.resolve(), out)
    if not args.run:
        print(json.dumps(dict(status='prepared', candidate_pairs=len(rows), quotas=TARGETS, budget_usd=55)))
        return
    cv2.setNumThreads(1)
    api_key = judge.load_api_key('LIJUYANG_ICLR', None) or judge.load_api_key('OPEN_ROUTER_QWEN', None)
    assert api_key, 'authorized API key is unavailable'
    transport = RecordedTransport(out, config['budget_usd'])
    judge.post_json = transport
    common = dict(model=MODEL, rubric=prompt, api_key=api_key, api_url=judge.DEFAULT_API_URL,
                  frame_samples=9, full_pair_count=5, roi_pair_count=5,
                  full_max_long_edge=640, roi_max_long_edge=640, analysis_long_edge=512,
                  expand_factor=2.0, minimum_roi_reliability=0.12, jpeg_quality=90,
                  max_tokens=2048, timeout=60.0, max_attempts=1, data_collection='deny',
                  reasoning_effort='minimal', exclude_reasoning=False, temperature=0.0,
                  provider_order=['google-ai-studio'], mock=False)
    attempts = read_jsonl(out / 'score_attempts.jsonl')
    latest = {a['result']['sample_id']: a['result'] for a in attempts}
    attempt_counts = {a['result']['sample_id']: a['attempt'] for a in attempts}
    if args.retry_account_errors:
        latest = {sid: result for sid, result in latest.items()
                  if not (result.get('parse_status') == 'error'
                          and attempt_counts[sid] < 3
                          and any(f'api_transport_error_http_{code}' in result.get('error', '')
                                  for code in (401, 402, 403)))}
    # Completed quarantines remain quarantined on resume; a separate, explicit
    # retry policy would be required to touch them again.
    accepted = Counter(r['branch'] for r in latest.values() if status(r) == 'accepted')
    queues = {b: deque(sorted([r for r in rows if r['branch'] == b and r['sample_id'] not in latest], key=stable_key)) for b in TARGETS}
    pending_by_branch = Counter()
    pending = {}
    submitted = completed = 0
    rotation = deque(TARGETS)
    atomic_json(out / 'runtime.json', dict(pid=os.getpid(), started_at=time.time(), concurrency=args.concurrency))
    save_progress(out, rows, latest, transport, config, 'running')
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        while True:
            while len(pending) < args.concurrency and not transport.blocked:
                if args.max_new_pairs is not None and submitted >= args.max_new_pairs:
                    break
                branch = None
                for _ in TARGETS:
                    b = rotation[0]
                    rotation.rotate(-1)
                    if queues[b] and accepted[b] + pending_by_branch[b] < TARGETS[b]:
                        branch = b
                        break
                if branch is None:
                    break
                row = queues[branch].popleft()
                future = executor.submit(score_pair, row, common, transport, attempt_counts.get(row['sample_id'], 0))
                pending[future] = row
                pending_by_branch[branch] += 1
                submitted += 1
            if not pending:
                break
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                result = future.result()
                latest[row['sample_id']] = result
                pending_by_branch[row['branch']] -= 1
                accepted[row['branch']] += status(result) == 'accepted'
                completed += 1
            if completed % 25 == 0 or not done:
                summary = save_progress(out, rows, latest, transport, config, 'running')
                print(json.dumps(summary), flush=True)
    state = 'complete' if all(accepted[b] == n for b, n in TARGETS.items()) else (transport.halt_reason if transport.blocked else 'partial')
    finalize(out, rows, latest, config)
    summary = save_progress(out, rows, latest, transport, config, state)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
