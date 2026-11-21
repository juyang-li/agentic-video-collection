"""Wait for existing evaluator to release its lock, then extend frozen scope."""
import fcntl
from pathlib import Path
from engine import ROOT, PKG, read, rows, save, sha

def main():
    lock=(ROOT/'metadata/run.lock').open('a')
    print('Waiting for current evaluator lock before migration',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    tasks=read(ROOT/'metadata/tasks.json');protocol=read(ROOT/'metadata/protocol.json')
    if protocol.get('scope_version')=='six_models_v2':print('Already extended',flush=True);return
    save(ROOT/'metadata/tasks_before_ablations.json',tasks);save(ROOT/'metadata/protocol_before_ablations.json',protocol)
    original={r['id']:r for r in rows(PKG/'inputs.jsonl')};manifest=sha(PKG/'inputs.jsonl');models=[]
    for scheme,step in [('real',232),('synthetic',600)]:
        base=Path('/scratch/prj0000000275/jcy/xks/fastref')/f'ltx_distilled_{scheme}{step}_frozen300_20261115'
        records=rows(base/'results.jsonl');byid={r['id']:r for r in records};assert len(records)==len(byid)==300 and set(byid)==set(original)
        model=f'ltx_{scheme}{step}';adapter=read(base/'metadata/adapter.json')
        for t in tasks:
            if t['dataset']!='frozen300':continue
            r=byid[t['id']];o=original[t['id']]
            assert r['status']=='complete' and r['contract']['input_manifest_sha256']==manifest
            assert r['uid']==o['uid'] and r['source_sha256']==o['source']['sha256'] and r['original_prompt']==o['prompt']['en']
            assert r['ordered_reference_sha256s']==[a['sha256'] for a in o['references']]
            assert r['fps']==o['fps'] and r['num_frames']==o['num_frames'] and r['canvas_wh']==o['canvas_wh']
            assert Path(r['output']).is_file() and r['full_decode_frames']==o['num_frames']
            t['outputs'].append({'model':model,'path':r['output'],'sha256':r['output_sha256'],'checkpoint_step':step,'checkpoint_sha256':adapter['sha256']})
        models.append({'model_id':model,'result_manifest':str(base/'results.jsonl'),'result_manifest_sha256':sha(base/'results.jsonl'),'adapter':adapter})
    assert sum(len(t['outputs']) for t in tasks)==2264
    save(ROOT/'metadata/ablation_models.json',models)
    protocol.update(scope_version='six_models_v2',budget=None,minimum_balance=None)
    save(ROOT/'metadata/tasks.json',tasks);save(ROOT/'metadata/protocol.json',protocol)
    sels=read(ROOT/'metadata/selections.json')
    counts={'tasks':416,'pairs':2264,'base_scores':4528,'repeat_scores':sum(len(t['outputs'])*2 for t in tasks if t['key'] in sels['repeat']),'checkpoint_scores':sum(len(t['outputs'])*2 for t in tasks if t['key'] in sels['checkpoint100']),'spending_cap':None}
    save(ROOT/'metadata/extended_scope.json',counts);print('EXTENDED',counts,flush=True)

if __name__=='__main__':main()
