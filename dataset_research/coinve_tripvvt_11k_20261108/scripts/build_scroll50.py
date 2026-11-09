#!/usr/bin/env python3
"""Frozen 40+10 scroll-only survey from already determined eligible prefixes."""
import argparse
import base64
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import html
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile
import prepare as p

ROOT=Path(__file__).resolve().parents[1]
NAME='coinve40_tripvvt10_scroll50_20261108'
OUT=ROOT/'reviews'/NAME
BUNDLE=OUT/NAME


def select():
    OUT.mkdir(exist_ok=True)
    if (OUT/'selection.jsonl').exists():
        assert p.sha(OUT/'selection.jsonl')==p.read(OUT/'selection_receipt.json')['selection_sha256'];return
    p.HISTORY=p.read(ROOT/'metadata/history.json')
    hash_locks=defaultdict(set,{k:set(v) for k,v in p.HISTORY['evaluation_hash_locks'].items()})
    cache=p.read(ROOT/'metadata/historical_media_hashes.json')
    for row in p.csv_rows(p.ART/'coinve_ablation_v1/coinve_4k_pool.csv'):
        for role in ('source','target'):
            if row[role+'_local'] in cache:hash_locks[cache[row[role+'_local']]['sha256']].add('train' if row['split']=='train' else 'heldout')
    rows=p.jsonl(ROOT/'metadata/candidates.jsonl');eligible=[];seen_hashes={};completed={}
    # Replay the deterministic selection prefix; do not count an out-of-order completed worker.
    for split in p.SPLITS:
        for key,n in p.quotas('coinve',split).items():
            accepted=[]
            for row in (r for r in rows if r['dataset']=='coinve' and r['split']==split and p.stratum(r)==key and p.group_allowed(r,split)):
                path=ROOT/'metadata/technical'/f"{row['sample_id']}.json"
                if not path.exists():break
                t=p.read(path)
                if t['status']!='ok':continue
                hashes=[t['media'][role]['sha256'] for role in ('source','target')]
                if any(('heldout' in hash_locks[v] and split=='train') or ('train' in hash_locks[v] and split!='train') or (v in seen_hashes and seen_hashes[v]!=split) for v in hashes):continue
                for v in hashes:seen_hashes[v]=split
                accepted.append(row)
                if len(accepted)==n:break
            eligible+=accepted;completed[split+':'+key]=len(accepted)
    rng=random.Random(p.SEED);chosen=[];used_groups=set()
    chosen=rng.sample(eligible,40)
    excluded=set(p.read(ROOT/'metadata/similarity_exclusions.json')['excluded_ids'])
    trip=[r for r in rows if r['dataset']=='tripvvt' and r['candidate_rank']<=1000 and p.good(r) and r['sample_id'] not in excluded]
    chosen += [dict(r,split='pending') for r in rng.sample(trip,10)]
    chosen.sort(key=lambda r:(r['dataset'],r.get('category',''),int(r['sample_id'].split('-')[1])))
    result=[]
    for row in chosen:
        t=p.read(ROOT/'metadata/technical'/f"{row['sample_id']}.json");assert t['status']=='ok'
        r=dict(row,common_start_seconds=0,common_end_seconds=t['common_end_seconds'],selection_status='determined CoinVE quota prefix' if row['dataset']=='coinve' else 'eligible after complete 3K technical and similarity filtering; final split pending')
        for role in p.roles(r):
            assert p.sha(t['media'][role]['path'])==t['media'][role]['sha256']
            r[role+'_original']=t['media'][role]['path'];r[role+'_original_sha256']=t['media'][role]['sha256']
        if r['dataset']=='tripvvt':assert r['caption']==r['original_metadata']['text'] and 'reference image' in r['prompt']
        result.append(r)
    assert len(result)==50 and len({r['sample_id'] for r in result})==50
    p.write_rows(OUT/'selection.jsonl',result)
    p.save(OUT/'selection_receipt.json',dict(seed=p.SEED,coinve=40,tripvvt=10,coinve_tasks=dict(Counter(p.stratum(r) for r in result if r['dataset']=='coinve')),
        trip_categories=dict(Counter(r['category'] for r in result if r['dataset']=='tripvvt')),available_coinve_prefix_counts=completed,
        candidate_manifest_sha256=p.sha(ROOT/'metadata/candidates.jsonl'),selection_sha256=p.sha(OUT/'selection.jsonl'),
        scope='interim survey of determined/eligible pool, not the final 11K acceptance',sampling='uniform random within each dataset, 40 CoinVE + 10 TripVVT, no exclusion of previously reviewed samples',rating_controls=0))
    print('frozen 50 pairs',flush=True)


def build():
    rows=p.jsonl(OUT/'selection.jsonl');BUNDLE.mkdir(exist_ok=True)
    for folder in ('media','originals','prompts'):(BUNDLE/folder).mkdir(exist_ok=True)
    def one(job):
        r,role=job;src=Path(r[role+'_original']);original=BUNDLE/'originals'/f"{r['sample_id']}_{role}{src.suffix}"
        if not original.exists():os.link(src,original)
        assert p.sha(original)==r[role+'_original_sha256']
        dest=BUNDLE/'media'/f"{r['sample_id']}_{role}{'.png' if role=='reference' else '.mp4'}"
        receipt=OUT/'conversion_cache'/(dest.name+'.json');receipt.parent.mkdir(exist_ok=True)
        if role=='reference':
            if not dest.exists():os.link(src,dest)
            return (r['sample_id'],role,dict(path='media/'+dest.name,sha256=p.sha(dest),original='originals/'+original.name))
        if not (dest.exists() and receipt.exists() and p.read(receipt)['sha256']==p.sha(dest)):
            info=p.read(ROOT/'metadata/technical'/f"{r['sample_id']}.json")['media'][role]
            temp=dest.with_suffix('.part.mp4')
            command=[p.FFMPEG,'-nostdin','-v','error','-y','-threads','1','-i',str(src),'-map','0:v:0','-an']
            if info['codec']=='h264':command+=['-c:v','copy']
            else:command+=['-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','1','-fps_mode','passthrough']
            subprocess.run(command+['-movflags','+faststart',str(temp)],check=True)
            actual=p.video_probe(temp)
            assert all(actual[k]==info[k] for k in ('width','height','frames','fps')),(r['sample_id'],role)
            temp.replace(dest);p.save(receipt,dict(sha256=p.sha(dest),full_decode=True,original_sha256=r[role+'_original_sha256'],spec=actual))
        return (r['sample_id'],role,dict(path='media/'+dest.name,sha256=p.sha(dest),original='originals/'+original.name))
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,[(r,role) for r in rows for role in p.roles(r)]))
    lookup={(sid,role):v for sid,role,v in results};cards=[]
    for row in rows:
        r=dict(row)
        for role in p.roles(r):
            value=lookup[(r['sample_id'],role)];r[role]=value['path'];r[role+'_sha256']=value['sha256'];r[role+'_original']=value['original']
        prompt=r['prompt']
        if r['dataset']=='tripvvt':prompt+='\n\nOfficial caption (scene/target description, not the edit instruction):\n'+r['caption']
        (BUNDLE/'prompts'/f"{r['sample_id']}.txt").write_text(prompt+'\n');cards.append(r)
    p.write_rows(BUNDLE/'manifest.jsonl',cards);p.csv_write(BUNDLE/'inventory.csv',cards)
    p.save(BUNDLE/'selection_receipt.json',p.read(OUT/'selection_receipt.json'))
    sections=[];esc=html.escape
    for i,r in enumerate(cards,1):
        label='CoinVE' if r['dataset']=='coinve' else 'TripVVT · '+r['category']
        panels=[]
        for role in (('source','reference','target') if r['dataset']=='tripvvt' else ('source','target')):
            title={'source':'SOURCE · 输入视频','target':'TARGET · 配对目标视频','reference':'REFERENCE · 服装参考图'}[role]
            tag=f'<img loading="lazy" src="{esc(r[role])}" alt="服装参考图">' if role=='reference' else f'<video controls playsinline preload="none" src="{esc(r[role])}"></video>'
            panels.append(f'<figure><figcaption>{title}</figcaption>{tag}<a href="{esc(r[role+"_original"])}" target="_blank">打开原始素材</a></figure>')
        caption=f'<div class="caption"><h3>完整官方 caption · 场景/目标描述</h3><p>此段保留官方原文；编辑指令为上面的 prompt。</p><pre>{esc(r["caption"])}</pre></div>' if r['dataset']=='tripvvt' else ''
        sections.append(f'<section id="{r["sample_id"]}" data-end="{r["common_end_seconds"]}"><h2>{i:02d} · {r["sample_id"]} <small>{label}</small></h2><h3>完整编辑 prompt</h3><pre class="prompt">{esc(r["prompt"])}</pre>{caption}<p class="interval">共同播放区间：0 — {r["common_end_seconds"]:.6f} 秒 · 原速，不拉伸、不补长尾</p><div class="panels">{"".join(panels)}</div><div class="playback"><button class="play">从头同步播放</button><button class="pause">暂停</button></div></section>')
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoinVE＋TripVVT · 50 对滚动审核</title><style>
body{margin:0;background:#edf1f5;color:#17212d;font:16px ReviewCJK,Arial,sans-serif}main{max-width:1500px;margin:auto;padding:26px}header,section{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:24px;margin-bottom:24px}h1{margin:0 0 12px}h2{margin:0 0 20px}h3{font-size:16px;margin:18px 0 8px}small{font-size:15px;font-weight:400;color:#526172}.prompt,pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 ReviewCJK,Arial,sans-serif;margin:0}.prompt{background:#eef5ff;border-left:4px solid #4278c4;padding:14px}.caption{margin-top:16px;border:1px solid #dbe1e8;padding:0 16px 16px;border-radius:8px}.caption p,.interval,header p{color:#526172;font-size:14px}.panels{display:flex;gap:16px;align-items:start}figure{margin:0;flex:1;min-width:0}figcaption{font-weight:600;font-size:14px;margin-bottom:8px}video,img{display:block;width:100%;height:350px;object-fit:contain;background:#101820;border-radius:6px}figure a{display:inline-block;font-size:13px;margin-top:8px;color:#315d9d}button{font:14px ReviewCJK,Arial,sans-serif;border:1px solid #bfcddd;background:#f5f8fc;border-radius:6px;padding:9px 16px;margin:12px 10px 0 0;cursor:pointer}@media(max-width:720px){main{padding:10px}header,section{padding:14px}.panels{flex-direction:column}figure{width:100%}h2{font-size:19px}video,img{height:300px}}
</style></head><body><main><header><h1>CoinVE＋TripVVT · 50 对滚动审核</h1><p>CoinVE 40 对 · TripVVT 10 对。全部样本在此页连续展开，直接向下滚动查看；无分页、无打分栏。</p><p>本包从当前已确定的 CoinVE 配额前缀和已完成筛选的 TripVVT 候选池固定抽样。完整原始素材在 originals/，浏览器视频在 media/。本包不表示人工审核通过，也不是最终 11K 验收。</p></header>__SECTIONS__</main><script>
let active=null;for(const card of document.querySelectorAll('section')){const vs=[...card.querySelectorAll('video')],end=Number(card.dataset.end);let guard=false;const pause=()=>vs.forEach(v=>v.pause());card.querySelector('.play').onclick=async()=>{if(active&&active!==pause)active();active=pause;for(const v of vs)v.currentTime=0;await Promise.all(vs.map(v=>v.play().catch(()=>pause())))};card.querySelector('.pause').onclick=pause;for(const v of vs){v.addEventListener('timeupdate',()=>{if(v.currentTime>=end)pause()});v.addEventListener('ended',pause);v.addEventListener('seeking',()=>{if(guard)return;guard=true;for(const o of vs)if(o!==v&&Math.abs(o.currentTime-v.currentTime)>.08)o.currentTime=Math.min(v.currentTime,end);setTimeout(()=>guard=false,80)})}}
</script></body></html>'''.replace('__SECTIONS__',''.join(sections))
    from fontTools import subset
    from fontTools.ttLib import TTFont
    fontdir=p.RESEARCH/'ref_edit_review_20261030/assets/font';font=TTFont(fontdir/'NotoSansCJKsc-Regular.otf')
    sub=subset.Subsetter();sub.populate(text=document);sub.subset(font);font.flavor='woff';font.save(BUNDLE/'review_font.woff')
    face="@font-face{font-family:ReviewCJK;src:url(data:font/woff;base64,"+base64.b64encode((BUNDLE/'review_font.woff').read_bytes()).decode()+") format('woff');font-display:block}"
    (BUNDLE/'index.html').write_text(document.replace('<style>','<style>'+face));shutil.copyfile(fontdir/'LICENSE.txt',BUNDLE/'FONT_LICENSE.txt')
    (BUNDLE/'README.txt').write_text('完整解压后用 Chrome / Edge 打开 index.html，全部 50 对在同一页面向下滚动查看，无打分栏、无下一页。\nCoinVE 40 对、TripVVT 10 对，TripVVT 完整编辑 prompt 与官方 caption 均在视频上方直接展示。服装参考图不可省略。\n原素材 originals/；H.264 浏览副本 media/；prompts/ 保存完整文本；manifest.jsonl 和 inventory.csv 保存来源、哈希与共同区间。\n固定 seed 20261108；这是当前确定候选的小规模审核，不代表最终 11K 验收或人工通过。\n')
    print('built 50 scroll cards',flush=True)


def package(skip_validation=False):
    if skip_validation:
        p.save(BUNDLE/'validation_status.json',dict(status='skipped',reason='user explicitly requested direct upload without verification',browser_validation_performed=False))
    else:
        validation=p.read(BUNDLE/'browser_validation.json');assert validation['status']=='passed' and validation['index_sha256']==p.sha(BUNDLE/'index.html')
    archive=OUT/(NAME+'.zip')
    if archive.exists():return
    temp=archive.with_suffix('.zip.part')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for f in sorted(BUNDLE.rglob('*')):
            if f.is_file() and f.name!='SHA256SUMS':z.write(f,str(f.relative_to(OUT)))
    if not skip_validation:
        with zipfile.ZipFile(temp) as z:assert z.testzip() is None
    temp.replace(archive);p.save(OUT/'package_info.json',dict(file=archive.name,bytes=archive.stat().st_size,remote_path='/apps/bypy/lance_review/'+archive.name,pairs=50,coinve=40,tripvvt=10,rating_controls=0))


def upload():
    path=p.RESEARCH/'recent_motion_review_20261030/deliver.py';spec=importlib.util.spec_from_file_location('scroll_delivery',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    archive=OUT/(NAME+'.zip')
    remote_path='/apps/bypy/lance_review/'+archive.name
    uploaded=m.upload(archive,remote_path)
    if uploaded.get('status')!='complete':raise RuntimeError('upload tool did not report completion')
    p.save(OUT/'baidu_delivery_receipt.json',dict(file=archive.name,remote_path=remote_path,
        status='complete',uploaded_utc=datetime.now(timezone.utc).isoformat(),pairs=50,coinve=40,tripvvt=10,
        rating_controls=0,uploaded_files=[archive.name],completion_basis='upload tool reported success',
        extra_verification_performed=False,separate_manifest_uploaded=False,separate_hash_uploaded=False))
    print('uploaded',remote_path,flush=True)



def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','build','verify','package','upload','all']);parser.add_argument('--skip-validation',action='store_true');a=parser.parse_args()
    for phase in (['select','build','verify','package'] if a.phase=='all' else [a.phase]):
        if phase=='verify' and a.skip_validation:
            print('browser validation skipped at user request',flush=True);continue
        print('phase',phase,flush=True)
        if phase=='package':package(skip_validation=a.skip_validation)
        elif phase=='verify':
            interpreter=p.PROJECT/'model_evaluation/ivebench_lance/runtime/review_browser/bin/python'
            subprocess.run([str(interpreter),str(ROOT/'scripts/verify_scroll50.py')],check=True)
        else:globals()[phase]()

if __name__=='__main__':main()
