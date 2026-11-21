"""Required score coverage/provenance checks; no browser compatibility suite."""
import re
from engine import ROOT, METRICS, read, save, sha, attempt_files

def main():
    tasks=read(ROOT/'metadata/tasks.json');sels=read(ROOT/'metadata/selections.json');protocol=read(ROOT/'metadata/protocol.json');expected=set();cost=0.;providers=set();models=set()
    assert protocol['prompt_hashes']=={f.name:sha(f) for f in (ROOT/'prompts').glob('*.md')}
    for t in tasks:
        cases=read(ROOT/'metadata/prepared'/f'{t["key"]}.json');assert len(cases)==len(t['outputs'])
        bymodel={c['model']:c for c in cases}
        for out in t['outputs']:
            c=bymodel[out['model']]
            assert c['source_sha256']==t['source_sha256'] and c['output_sha256']==out['sha256']
            assert len(c['cards'])==len(t['indices']) and c['instruction']==t['prompt']['en']
            assert not {'gt','caption','ground_truth','vace_description'} & set(c)
            for repeat in ([0,1] if t['key'] in sels['repeat'] else [0]):
                for metric in METRICS:
                    name=f'{c["key"]}__{metric}__r{repeat}.json';expected.add(name);r=read(ROOT/'results'/name)
                    assert r['status']=='ok' and isinstance(r['score'],int) and 1<=r['score']<=5
                    assert r['parameters']==protocol['parameters']
                    assert r['cards']==c['cards'] and r['references']==(c['references'] if metric=='edit_achievement' else [])
                    assert r['prompt_sha256']==protocol['prompt_hashes'][f'{c["attribute"]}_{metric}.md']
                    providers.add(r['response'].get('provider'));models.add(r['response'].get('model'))
    actual={f.name for f in (ROOT/'results').glob('*.json') if re.search(r'__r[01]\.json$',f.name)}
    assert expected==actual,(len(expected),len(actual),list(expected-actual)[:5])
    assert len(expected)==4748
    for p in attempt_files():cost+=read(p).get('response',{}).get('usage',{}).get('cost',0) or 0
    assert read(ROOT/'report/full/coverage.json')['successful_scores']==4528
    assert read(ROOT/'report/full/stability_summary.json')['completed']==220
    save(ROOT/'metadata/completion_audit.json',{'status':'passed','tasks':416,'pairs':2264,'base_scores':4528,'repeat_scores':220,'total_scores':len(expected),'cost_all_attempts_usd':cost,'judge_providers':sorted(providers),'judge_models':sorted(models),'cp_version':protocol['cp_version'],'browser_compatibility':'skipped_user_requested'})
    print('AUDIT passed',len(expected),'scores; total cost',cost)

if __name__=='__main__':main()
