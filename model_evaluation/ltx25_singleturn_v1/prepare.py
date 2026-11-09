"""Freeze a small non-action benchmark; never pass evaluation GT to inference."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REAL = ROOT / 'model_evaluation/real_person_multiturn_benchmark_v2'
CLOTH = ROOT / 'dataset_research/ref_edit_review_20261030/review/ref_edit_clothing25_motion25_review_20261030'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def relative(p):
    return p.relative_to(ROOT).as_posix()


def main():
    chains = {c['chain_id']: c for c in map(json.loads, (REAL / 'benchmark.jsonl').read_text().splitlines())}
    cards = json.loads((CLOTH / 'manifest.json').read_text())['cards']
    inputs, evaluation = [], []
    for c in cards:
        if c['kind'] != 'garment':
            continue
        source, ref, target = [CLOTH / p['path'] for p in c['panels']]
        inputs.append(dict(sample_id=c['id'], task=c['task'], protocol='source_video+reference_image+instruction',
                           source_video=relative(source), reference_image=relative(ref), instruction=c['instruction_en'],
                           source_sha256=digest(source), reference_sha256=digest(ref), seed=42,
                           group_id=c['provenance']['group_id'], shard=len(inputs) % 4))
        evaluation.append(dict(sample_id=c['id'], target_video=relative(target), target_sha256=digest(target),
                               provenance=c['provenance'], notes=c['notice'],
                               instruction_origin='existing_review_manifest'))

    # Explicitly rewrite only the edits that mention prior states. All use original videos.
    appearance = [1, 2, 4, 6, 7, 8, 12, 14, 17, 19, 21, 22, 26, 28, 29]
    outfit = {
        1: 'Change the woman’s top to a soft cream blouse with a subtle sheen.',
        2: 'Change the woman’s jacket to a dark green coat.',
        4: 'Change the person’s jacket to a bright red color.',
        5: 'Change the man’s suit jacket to a dark navy blazer while keeping the white shirt.',
        6: 'Change the performer’s costume to a deep red velvet robe with gold embroidery along the sleeves and hem.',
        9: 'Change the woman’s black textured outfit to a dark green leather jacket.',
        10: 'Change the gray turtleneck sweater to a dark navy crewneck sweater.',
        12: 'Change the woman’s black top to a dark navy hoodie.',
        16: 'Replace the soldier’s uniform with a dark navy jacket featuring visible shoulder insignia.',
        20: 'Change the seated man’s plain light shirt to a dark zip-up hoodie.',
    }
    cameras = [2, 3, 5, 6, 9, 10, 13, 14, 15, 25]
    selection = [('appearance', n, 2 if n == 19 else 1) for n in appearance]
    selection += [('outfit', n, None) for n in outfit]
    selection += [('camera', n, None) for n in cameras]
    for axis, n, round_number in selection:
        c = chains[f'rpmte_{n:03}']
        turn = next(t for t in c['turns'] if t['axis'] == axis and
                    (round_number is None or t['round'] == round_number))
        instruction = outfit[n] if axis == 'outfit' else turn['instruction']
        instruction += (' Preserve the source subject’s identity and original motion, other people, '
                        'and all content unrelated to this edit. ' +
                        ('Preserve the original scene, clothing, and appearance.' if axis == 'camera' else
                         'Preserve the original camera and background.'))
        source = REAL / c['source']['media']
        sid = f'{axis}_{n:03}'
        task = ('camera_angle_and_distance' if n == 15 else 'camera_framing') if axis == 'camera' else (
            'text_outfit' if axis == 'outfit' else f"appearance_{turn['appearance_family']}")
        inputs.append(dict(sample_id=sid, task=task, protocol='source_video+instruction',
                           source_video=relative(source), reference_image=None, instruction=instruction,
                           source_sha256=digest(source), reference_sha256=None, seed=42,
                           group_id=c['source']['official_sha256'], shard=len(inputs) % 4))
        evaluation.append(dict(sample_id=sid, target_video=None, target_sha256=None,
                               provenance=dict(chain_id=c['chain_id'], original_round=turn['round'],
                                               original_instruction=turn['instruction'], source=c['source']),
                               notes='Independent edit from original source; no history input or generated previous round.',
                               instruction_origin='singleturn_rewrite' if axis == 'outfit' else 'existing_atomic_edit_with_source_preservation'))
    assert len(inputs) == 60
    assert len({r['sample_id'] for r in inputs}) == 60
    assert all('action' not in r['task'] for r in inputs)
    assert all(Path(ROOT / r['source_video']).is_file() for r in inputs)
    for name, records in [('inputs.jsonl', inputs), ('evaluation_only.jsonl', evaluation)]:
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    # Four matched image ablations distinguish image following from text-conditioned edits.
    ablations = []
    for sid in ['cloth_004', 'cloth_010', 'cloth_019', 'cloth_023']:
        base = next(r for r in inputs if r['sample_id'] == sid)
        ablations.append(dict(base_sample_id=sid, sample_id=sid + '_wrong_ref',
                              intervention='replace_reference_only',
                              replacement_reference_from={'cloth_004':'cloth_008', 'cloth_010':'cloth_013',
                                                          'cloth_019':'cloth_020', 'cloth_023':'cloth_024'}[sid],
                              seed=42, status='planned_after_protocol_smoke'))
    (OUT / 'reference_controls.json').write_text(json.dumps(ablations, ensure_ascii=False, indent=2) + '\n')
    summary = dict(status='inputs_prepared_not_run', count=len(inputs),
                   task_counts=dict(collections.Counter(r['task'] for r in inputs)),
                   protocol_counts=dict(collections.Counter(r['protocol'] for r in inputs)),
                   shard_counts=dict(collections.Counter(r['shard'] for r in inputs)),
                   unique_sources=len({r['source_sha256'] for r in inputs}),
                   input_manifest_sha256=digest(OUT / 'inputs.jsonl'),
                   evaluation_manifest_sha256=digest(OUT / 'evaluation_only.jsonl'),
                   optional_reference_controls=4,
                   exclusions=['all action edits', 'all motion references', 'history inputs', 'rollback tasks'],
                   model_status='official composition must pass source+ref-image protocol smoke before formal inference')
    (OUT / 'inventory_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
