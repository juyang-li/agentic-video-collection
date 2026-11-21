"""Deterministic offline parsing; never calls a model or changes raw responses."""
import collections, re
from pilot import ROOT,load,save

def main():
    counts=collections.Counter()
    for selected in (ROOT/'results').glob('*_r*.json'):
        if '_attempt' in selected.stem:continue
        paths=sorted(selected.parent.glob(selected.stem+'_attempt*.json'),key=lambda p:int(re.search(r'_attempt(\d+)$',p.stem)[1]))
        parsed=None
        for path in paths:
            r=load(path)
            if r.get('http_status')!=200:continue
            choice=r['response']['choices'][0]
            if choice.get('finish_reason')!='stop':continue
            raw=choice['message'].get('content') or '';text=raw.strip();fenced=False
            fence=re.fullmatch(r'```(?:text)?\s*\n(.*?)\n```',text,re.S)
            if fence:text=fence[1].strip();fenced=True
            m=re.fullmatch(r'score:\s*([1-5])[ \t]*\nreason:[ \t]*([^\n]+)',text)
            if not m:continue
            parsed=dict(r,status='ok',score=int(m[1]),reason=m[2],parse_source=str(path.relative_to(ROOT)),format_compliant=not fenced,normalization='outer_code_fence_only' if fenced else 'none',attempts_available=len(paths))
            break
        if parsed is None:parsed=dict(load(selected),parse_source=None,format_compliant=False)
        save(ROOT/'metadata/parsed'/selected.name,parsed)
        counts[parsed['judge']+':'+parsed['status']+':'+str(parsed['format_compliant'])]+=1
    save(ROOT/'metadata/normalization_summary.json',dict(counts));print(dict(counts))

if __name__=='__main__':main()
