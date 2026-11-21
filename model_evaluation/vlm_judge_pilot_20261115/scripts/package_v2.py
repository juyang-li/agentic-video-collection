"""Lightweight packaging checks; browser testing explicitly skipped by user."""
import json,re,zipfile
from pilot import ROOT,load,save

def main():
    d=ROOT/'review_v2';page=(d/'index.html').read_text();scores=load(d/'scores.json')
    assert len(scores)==240 and len(load(d/'display_manifest.json'))==51
    assert len(re.findall('<article ',page))==40 and len(re.findall('<section ',page))==10
    assert 'Judge A' not in page and '候选 C' not in page and 'src="evidence/' not in page
    for r in scores:
        old=load(ROOT/'metadata/parsed'/f'{r["judge"]}_{r["candidate_id"]}_{r["metric"]}_r0.json')
        assert (r['score'],r['reason'])==(old['score'],old['reason'])
    for url in re.findall(r'(?:src|href|poster)="([^"]+)"',page):
        path=url.split('?',1)[0].split('#',1)[0]
        if path:assert (d/path).exists(),path
    save(ROOT/'metadata/review_v2_validation.json',{'status':'passed_basic_checks','browser_playback':'skipped_by_explicit_user_request','proxies':51,'tasks':10,'outputs':40,'unchanged_base_scores':240,'api_calls_added':0})
    archive=ROOT/'review_v2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(d.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('ZIP ready',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':main()
