"""Completion gate: all expected scores, input fingerprints, and offline links."""
import hashlib,html,json,re
from pilot import ROOT,JUDGES,load,save,messages,sha

def main():
    cs=load(ROOT/'metadata/candidates.json');selected=load(ROOT/'metadata/stability_selection.json')['candidate_ids']
    expected={(j,c['id'],m,r) for j in JUDGES for c in cs for m in ['edit_achievement','content_preservation'] for r in ([0,1,2] if c['id'] in selected else [0])}
    parsed=[load(p) for p in (ROOT/'metadata/parsed').glob('*_r*.json')]
    actual={(p['judge'],p['candidate_id'],p['metric'],p['repeat']) for p in parsed if p['status']=='ok'}
    missing=sorted(expected-actual);unexpected=sorted(actual-expected);request_checks=0
    for c in cs:
        for metric in ['edit_achievement','content_preservation']:
            msgs,prompt_sha=messages(c,metric)
            for j in JUDGES:
                for r in ([0,1,2] if c['id'] in selected else [0]):
                    path=ROOT/'metadata/parsed'/f'{j}_{c["id"]}_{metric}_r{r}.json'
                    if not path.exists():continue
                    p=load(path)
                    if p['status']!='ok':continue
                    source=load(ROOT/p['parse_source'])
                    raw=source['response']['choices'][0]['message']['content'].strip()
                    fence=re.fullmatch(r'```(?:text)?\s*\n(.*?)\n```',raw,re.S)
                    if fence:raw=fence[1].strip()
                    match=re.fullmatch(r'score:\s*([1-5])[ \t]*\nreason:[ \t]*([^\n]+)',raw)
                    assert match and int(match[1])==p['score'] and match[2]==p['reason']
                    assert p['prompt_sha256']==prompt_sha
                    request=dict(source['parameters'],messages=msgs)
                    digest=hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                    assert digest==p['request_sha256'],path
                    assert len(p['reference_inputs'])==(len(c['references']) if metric=='edit_achievement' else 0)
                    assert 1<=p['score']<=5 and p['reason'].strip()
                    request_checks+=1
    page=(ROOT/'review/index.html').read_text()
    links=re.findall(r'(?:href|src)="(media/[^"]+)"',page)
    assert all((ROOT/'review'/p).is_file() for p in links)
    assert len(re.findall('<section id=',page))==10 and len(re.findall('<article>',page))==40
    assert all(x not in page for x in ['deepseek/','qwen/','gpt-6.1','ltx_base','h3_pdd8'])
    if not missing:
        assert len(re.findall(r'<td>[1-5]/5<br>',page))==240
        for p in parsed:
            if p['repeat']==0 and p['status']=='ok':assert html.escape(p['reason']) in page
    inv=load(ROOT/'review/inventory.json');assert len(inv)==40
    tasks=load(ROOT/'review/tasks.json');assert len(tasks)==10
    for t in tasks:
        assert sha(ROOT/'review'/t['full_source'])==t['full_source_sha256']
        if 'gt' in t:
            assert t['official_caption'].strip() and t['gt_input_to_judges'] is False
            for key in ['gt','full_gt']:assert sha(ROOT/'review'/t[key])==t[key+'_sha256']
    for row in inv:
        for field in ['source','output']:assert sha(ROOT/'review'/row[field])==row[field+'_sha256']
        for f in row['references']+row['timecards']:assert sha(ROOT/'review'/f['path'])==f['sha256']
    report={'expected_scoring_items':len(expected),'verified_scoring_items':len(actual),'verified_request_fingerprints':request_checks,'missing':missing,'unexpected':unexpected,'media_links':len(links),'scripts_sha256':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},'requirements_sha256':sha(ROOT/'requirements.txt'),'status':'complete' if not missing and not unexpected else 'incomplete'}
    save(ROOT/'metadata/completion_audit.json',report);print(report)
    if missing or unexpected:raise SystemExit(1)

if __name__=='__main__':main()
