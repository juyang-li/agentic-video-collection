#!/usr/bin/env python3
"""Collect recent, actually released motion-reference benchmarks and provenance."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,random,re,time,tarfile
import requests

ROOT=Path(__file__).resolve().parent
HF=['MCG-NJU/X-Dance','theFoxofSky/RealisDance-Val','Ruihang/MoveBench','RefVideo6M/RefVideo6M']
GH=['MCG-NJU/SteadyDancer','damo-cv/RealisDance','ali-vilab/Wan-Move','Wan-Video/Wan2.2','Wan-Video/Wan-Animate-2','zai-org/SCAIL-2','Orange-3DV-Team/MoCha','Phantom-video/HuMo']


def get(url,stream=False):
    for attempt in range(3):
        try:
            r=requests.get(url,stream=stream,timeout=(20,60))
            if r.status_code==200:return r
            status=r.status_code;r.close()
        except requests.RequestException as e:status=type(e).__name__
        if attempt<2:time.sleep(2)
    raise RuntimeError('download failed: '+str(status)+' '+url.split('?')[0])


def jsave(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def download(item):
    url,path=item
    if path.exists() and path.stat().st_size:return str(path.relative_to(ROOT))
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part')
    with get(url,True) as r,tmp.open('wb') as f:
        for block in r.iter_content(1024*1024):f.write(block)
    tmp.replace(path);return str(path.relative_to(ROOT))


def parallel(jobs):
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,f in enumerate(as_completed([pool.submit(download,j) for j in jobs]),1):
            print(f'{i}/{len(jobs)} {f.result()}',flush=True)


def inventory():
    facts={'observed_at':'2026-10-30','window':['2025-04-30','2026-10-30'],'datasets':{},'projects':{}}
    for repo in HF:
        d=get('https://huggingface.co/api/datasets/'+repo).json()
        facts['datasets'][repo]={k:d.get(k) for k in ['sha','createdAt','downloads','likes','gated']}
        tree=get('https://huggingface.co/api/datasets/'+repo+'/tree/main?recursive=true&limit=1000').json()
        jsave(ROOT/'metadata'/f'{repo.replace("/","_")}_tree.json',tree)
        print('dataset',repo,facts['datasets'][repo],flush=True)
    for repo in GH:
        d=get('https://api.github.com/repos/'+repo).json()
        facts['projects'][repo]={k:d.get(k) for k in ['stargazers_count','forks_count','created_at','default_branch','pushed_at']}
        branch=d['default_branch']
        text=get('https://raw.githubusercontent.com/'+repo+'/'+branch+'/README.md').text
        (ROOT/'metadata'/f'{repo.replace("/","_")}_README.md').write_text(text)
        print('project',repo,facts['projects'][repo]['stargazers_count'],flush=True)
    jsave(ROOT/'metadata/research_facts.json',facts)


def assets():
    facts=json.loads((ROOT/'metadata/research_facts.json').read_text());jobs=[]
    for repo,folder in [('MCG-NJU/X-Dance','xdance'),('theFoxofSky/RealisDance-Val','realis')]:
        tree=json.loads((ROOT/'metadata'/f'{repo.replace("/","_")}_tree.json').read_text())
        rev=facts['datasets'][repo]['sha']
        for item in tree:
            member=item['path']
            if item['type']!='file':continue
            wanted=(folder=='xdance' and (member.endswith('/video.mp4') or member.endswith('/pose.mp4') or member.endswith('/prompt.txt') or (member.startswith('Ref-') and member.endswith('.png'))))
            wanted|=(folder=='realis' and (member.startswith('ref/') and member.endswith('.png')))
            if wanted:jobs.append((f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{member}',ROOT/'assets'/folder/member))
    parallel(jobs)


def realis(ids):
    facts=json.loads((ROOT/'metadata/research_facts.json').read_text());repo='theFoxofSky/RealisDance-Val';rev=facts['datasets'][repo]['sha']
    jobs=[]
    for n in ids:
        for member in [f'gt/{n:04d}.mp4',f'prompt/{n:04d}.txt']:
            jobs.append((f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{member}',ROOT/'assets/realis'/member))
    parallel(jobs)


def wan():
    facts=json.loads((ROOT/'metadata/research_facts.json').read_text());jobs=[];examples=[]
    for repo in ['Wan-Video/Wan2.2','Wan-Video/Wan-Animate-2']:
        tree=get(f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1').json()
        revision=tree['sha'];jsave(ROOT/'metadata'/f'{repo.replace("/","_")}_tree.json',tree)
        for row in tree['tree']:
            member=row['path']
            if repo.endswith('Wan2.2'):wanted=member.startswith('examples/wan_animate/') and member.endswith(('.mp4','.jpeg','.png','.jpg'))
            else:wanted=member.startswith('examples/') and member.endswith(('.mp4','.jpeg','.png','.jpg'))
            if wanted:
                dest=ROOT/'assets'/('wan1' if repo.endswith('Wan2.2') else 'wan2')/member
                jobs.append((f'https://raw.githubusercontent.com/{repo}/{revision}/{member}',dest))
                examples.append({'repo':repo,'revision':revision,'member':member,'local':str(dest.relative_to(ROOT))})
    jsave(ROOT/'metadata/wan_official_examples.json',examples);parallel(jobs)


def move():
    facts=json.loads((ROOT/'metadata/research_facts.json').read_text());repo='Ruihang/MoveBench';rev=facts['datasets'][repo]['sha']
    archive=ROOT/'assets/movebench/en.tar.gz'
    download((f'https://huggingface.co/datasets/{repo}/resolve/{rev}/en.tar.gz',archive))
    print('MoveBench archive downloaded',archive.stat().st_size,flush=True)
    captions=[]
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if m.name.endswith('/single_track.txt') or m.name=='single_track.txt':
                text=tar.extractfile(m).read().decode()
                (ROOT/'metadata/movebench_single_track.txt').write_text(text)
                for line in text.splitlines():
                    if not line.strip():continue
                    ident,caption=line.split(maxsplit=1)
                    if re.search(r'\b(man|woman|person|people|dancer|boy|girl|athlete)\b',caption,re.I) and re.search(r'danc|walk|jump|stretch|exercis|yoga|runn|sport|gym|boxing|climb|swim',caption,re.I):
                        captions.append({'id':Path(ident).stem,'caption':caption})
                break
    rng=random.Random(20261030);rng.shuffle(captions);selected=captions[:18]
    assert len(selected)>=8
    jsave(ROOT/'metadata/movebench_selection.json',{'repo':repo,'revision':rev,'human_active_candidates':len(captions),'samples':selected})
    wanted={x['id'] for x in selected};count=0
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if not m.isfile():continue
            rel=Path(m.name)
            if rel.stem not in wanted or not any(k in rel.parts for k in ['video','first_frame']):continue
            dest=ROOT/'assets/movebench'/('video' if rel.suffix=='.mp4' else 'first_frame')/rel.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(tar.extractfile(m).read())
            count+=1;print('MoveBench extract',count,rel.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inventory','assets','realis','wan','move']);p.add_argument('--ids',default='');a=p.parse_args()
    if a.phase=='inventory':inventory()
    elif a.phase=='assets':assets()
    elif a.phase=='realis':realis([int(i) for i in a.ids.split(',') if i])
    elif a.phase=='move':move()
    else:wan()
