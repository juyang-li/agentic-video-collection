from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse,base64,csv,hashlib,html,io,json,shutil,subprocess,zipfile
import cv2,imageio_ffmpeg
from fontTools import subset
from fontTools.ttLib import TTFont
from inventory import ROOT,DATA,SOURCE

NAME='datamake_v1_balanced80_scroll_20261030'
PACKAGE=ROOT/'review'/NAME
FF=imageio_ffmpeg.get_ffmpeg_exe()

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def local(p):return 'media/'+p.replace('/','__')
def info(p):
    cap=cv2.VideoCapture(str(p));d={k:int(cap.get(v)) for k,v in [('width',3),('height',4),('frames',7)]};d['fps']=cap.get(5);cap.release()
    assert d['frames']>0 and d['fps']>0,p
    d['duration']=d['frames']/d['fps'];return d

def convert(path,expected,policy):
    raw=DATA/path;out=PACKAGE/local(path);raw_hash=sha(raw)
    assert raw_hash==expected['sha256'],('source changed or integrity mismatch',path)
    if out.suffix.lower()=='.mp4':
        before=info(raw)
        if not out.exists():
            tmp=out.with_suffix('.part.mp4')
            codec=['-c:v','copy'] if policy=='lossless_remux' else ['-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p']
            run=subprocess.run([FF,'-nostdin','-v','error','-y','-i',str(raw),'-map','0:v:0','-an',
                *codec,'-movflags','+faststart',str(tmp)],capture_output=True,text=True)
            assert run.returncode==0,(path,run.stderr[-500:]);tmp.replace(out)
        after=info(out)
        assert before==after,(path,before,after)
        run=subprocess.run([FF,'-nostdin','-v','error','-threads','2','-i',str(out),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert run.returncode==0,(path,run.stderr[-500:])
        poster=out.with_suffix('.jpg')
        cap=cv2.VideoCapture(str(out));cap.set(cv2.CAP_PROP_POS_FRAMES,30);ok,img=cap.read();cap.release();assert ok
        scale=min(1,600/max(img.shape[:2]));cv2.imwrite(str(poster),cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale))),[cv2.IMWRITE_JPEG_QUALITY,92])
        return dict(source_path=path,path=local(path),source_sha256=raw_hash,sha256=sha(out),bytes=out.stat().st_size,
                    poster=poster.relative_to(PACKAGE).as_posix(),full_decode_ok=True,video_policy=policy,**after)
    if not out.exists():shutil.copy2(raw,out)
    assert sha(out)==raw_hash
    img=cv2.imread(str(out));assert img is not None
    return dict(source_path=path,path=local(path),source_sha256=raw_hash,sha256=raw_hash,bytes=out.stat().st_size,
                width=img.shape[1],height=img.shape[0])

def main():
    PACKAGE.mkdir(parents=True,exist_ok=True);(PACKAGE/'media').mkdir(exist_ok=True);(PACKAGE/'provenance').mkdir(exist_ok=True)
    selected=json.loads((ROOT/'selection.json').read_text())
    nodes={x['id']:x for x in json.loads((DATA/'nodes/index.json').read_text())['nodes']}
    integrity={r['path'].removeprefix('dataset_v1/'):r for r in json.loads((SOURCE/'integrity.json').read_text())['files']}
    media=set();cards=[]
    for s in selected['samples']:
        t=s['sample'];c=s['chain'];ids=[t['source_node_id'],t['target_node_id']]
        refs=list(t['references'])
        if c:
            ids+=c['history_node_ids'];refs += [r for turn in c['turns'] for r in turn['references']]
        media.update(nodes[n]['path'] for n in ids);media.update(r['path'] for r in refs)
        rawpath=c['_path'] if c else t['_path'];raw=DATA/rawpath
        assert sha(raw)==integrity[rawpath]['sha256']==(c or t)['_sha256']
        shutil.copy2(raw,PACKAGE/'provenance'/f'{s["review_id"]}.json')
        cards.append(dict(review_id=s['review_id'],group=s['group'],group_zh=s['group_zh'],kind=s['kind'],split=s['split'],
            edge_id=s['edge_id'],axis=t['axis'],mode=t['mode'],slot=t.get('slot'),prompt=t['prompt'],
            source_node_id=t['source_node_id'],target_node_id=t['target_node_id'],source=local(nodes[t['source_node_id']]['path']),
            target=local(nodes[t['target_node_id']]['path']),references=[{**r,'path':local(r['path']),'original_path':r['path']} for r in t['references']],
            source_attributes={**nodes[t['source_node_id']]['attributes'],'expression_id':nodes[t['source_node_id']]['expression_id']},
            target_attributes={**nodes[t['target_node_id']]['attributes'],'expression_id':nodes[t['target_node_id']]['expression_id']},
            chain_id=c['chain_id'] if c else None,chain_kind=c['kind'] if c else None,
            edit_round=t.get('edit_round'),source_history_index=t.get('source_history_index'),
            history=[dict(index=i,node_id=n,path=local(nodes[n]['path'])) for i,n in enumerate(c['history_node_ids'])] if c else [],
            chain_turns=[{**turn,'references':[{**r,'path':local(r['path'])} for r in turn['references']]} for turn in c['turns']] if c else [],
            original_json=f'provenance/{s["review_id"]}.json',original_json_sha256=sha(raw)))
    print('unique media',len(media),'source bytes',sum((DATA/p).stat().st_size for p in media),flush=True)
    policyfile=ROOT/'metadata/video_policies.json'
    if policyfile.exists():policies=json.loads(policyfile.read_text())
    else:
        policies={p:('crf18' if (PACKAGE/local(p)).exists() else 'lossless_remux') for p in media if p.endswith('.mp4')}
        save(policyfile,policies)
    specs=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(convert,p,integrity[p],policies.get(p,'original_image')):p for p in sorted(media)}
        for fut in as_completed(jobs):
            specs.append(fut.result())
            if len(specs)%10==0:print('media processed',len(specs),'/',len(media),flush=True)
    manifest=dict(name=NAME,source_package=str(SOURCE),selection_seed=selected['seed'],selection_method=selected['method'],
                  cards=cards,media=sorted(specs,key=lambda r:r['path']))
    save(PACKAGE/'manifest.json',manifest)
    save(PACKAGE/'technical_validation.json',dict(status='passed',media=len(specs),videos=sum(x['path'].endswith('.mp4') for x in specs),
         images=sum(not x['path'].endswith('.mp4') for x in specs),source_integrity_sha256_verified=True,
         video_policy='Completed CRF18 H264 derivatives retained; remaining originals losslessly remuxed; native resolution/frame count/fps/duration, no audio',
         video_policy_counts=dict(Counter(x['video_policy'] for x in specs if 'video_policy' in x)),
         image_policy='original PNG byte copy',source_integrity_file_sha256=sha(SOURCE/'integrity.json')))
    for name in ['README.md','SCHEMA.md']:shutil.copy2(SOURCE/name,PACKAGE/'provenance'/('SOURCE_'+name))
    for name in ['prompt_statistics.json','single_turn_summary.json','multi_turn_summary.json']:shutil.copy2(ROOT/'metadata'/name,PACKAGE/name)
    shutil.copytree(DATA/'catalogs',PACKAGE/'provenance/catalogs',dirs_exist_ok=True)
    for name in ['REPORT.md','PACKAGE_README.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,PACKAGE/('README.md' if name=='PACKAGE_README.md' else name))
    with (PACKAGE/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['review_id','group_zh','kind','split','edge_id','source_node_id','target_node_id','chain_id','edit_round','source_history_index','reference_count','prompt_en','prompt_zh']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cards:w.writerow({**{k:c[k] for k in fields if k in c},'reference_count':len(c['references']),'prompt_en':c['prompt']['en'],'prompt_zh':c['prompt']['zh']})
    page(manifest)
    print('page ready',PACKAGE,flush=True)

def page(m):
    esc=html.escape;specs={r['path']:r for r in m['media']}
    catalogs={}
    for p in (DATA/'catalogs').glob('*.json'):
        d=json.loads(p.read_text());catalogs[p.stem]={r['id']:r.get('name',r.get('code')) for r in d.get('items',[])}
    labels={'identity':'身份参考图','hair':'发型参考图','garment':'服装参考图','scene':'空场景参考图','motion':'动作参考视频'}
    viewnames={'front':'正面','back':'背面','target_camera_initial_frame':'目标取景首帧'}
    def panel(path,label,primary=False):
        s=specs[path];isvideo=path.endswith('.mp4')
        content=(f'<video class="{"primary" if primary else "context"}" controls muted playsinline preload="none" data-src="{esc(path)}" poster="{esc(s["poster"])}"></video>' if isvideo else f'<a href="{esc(path)}" target="_blank"><img loading="lazy" src="{esc(path)}" alt="{esc(label)}"></a>')
        detail=f'{s["width"]}×{s["height"]}'+(f' · {s["fps"]:g} FPS · {s["duration"]:g}s' if isvideo else '')
        return f'<div class="panel"><h4>{esc(label)}</h4>{content}<p class="spec">{detail} · <a href="{esc(path)}" target="_blank">打开文件</a></p></div>'
    def reference_panels(refs,primary=False):
        return '<div class="refs">'+''.join(panel(r['path'],f'REF {i+1} · '+labels[r['type']]+' '+viewnames.get(r.get('view'),r.get('view','')),primary) for i,r in enumerate(refs))+'</div>'
    def prompts(p):return '<div class="prompts"><h4>Prompt · 原始中文</h4><p>'+esc(p['zh'])+'</p><h4>Prompt · Original English</h4><p lang="en">'+esc(p['en'])+'</p></div>'
    def value(k,v):
        if k=='clothing':return json.dumps(v,ensure_ascii=False)
        return str(v)+' · '+str(catalogs.get(k.removesuffix('_id'),{}).get(v,''))
    parts=[];groups=[]
    for c in m['cards']:
        if c['group'] not in groups:groups.append(c['group'])
    for gi,g in enumerate(groups):
        cs=[c for c in m['cards'] if c['group']==g]
        parts.append(f'<section id="group{gi}"><h2>{gi+1}. {esc(cs[0]["group_zh"])} · 10 条</h2>')
        for c in cs:
            ctx=f'多轮链 {c["chain_id"]} / {c["chain_kind"]} / 第 {c["edit_round"]} 轮 / 从 history[{c["source_history_index"]}] 编辑' if c['chain_id'] else '独立单轮'
            parts.append(f'<article id="{c["review_id"]}" data-group="{gi}"><h3>{c["review_id"]} · 边 {c["edge_id"]} · {esc(c["group_zh"])}</h3><p class="meaning">{esc(ctx)} · {c["split"]} · source node {c["source_node_id"]} → target node {c["target_node_id"]}</p>')
            parts.append(prompts(c['prompt']))
            parts.append('<div class="pair">'+panel(c['source'],f'SOURCE · 输入视频 · node {c["source_node_id"]}',True)+panel(c['target'],f'TARGET · 合成配对目标 · node {c["target_node_id"]}',True)+'</div>')
            parts.append(reference_panels(c['references'],True) if c['references'] else '<p class="note">本条没有 ref image / ref video；原始 references=[]，按纯文字条件展示。</p>')
            parts.append('<div class="tools"><button class="play">本条视频从头同步播放</button><button class="pause">暂停本条视频</button></div>')
            a,b=c['source_attributes'],c['target_attributes'];diff=[k for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)]
            rows=''.join(f'<tr><td>{esc(k)}</td><td>{esc(value(k,a.get(k)))}</td><td>{esc(value(k,b.get(k)))}</td></tr>' for k in diff)
            parts.append(f'<details><summary>节点属性差异与原始记录</summary><table><thead><tr><th>属性</th><th>source</th><th>target</th></tr></thead><tbody>{rows}</tbody></table><p><a href="{c["original_json"]}" target="_blank">原始样本/整条链 JSON</a> · axis={c["axis"]} · mode={c["mode"]}</p></details>')
            if c['history']:
                parts.append('<details class="history"><summary>展开完整多轮上下文：历史视频、各轮 prompt 和全部参考素材</summary><p>history 编号从 0 开始；分叉输入按 source_history_index，不能默认用上一轮。下面也包含当前审核轮之后的链条，供核对完整关系。</p><div class="refs">')
                parts.extend(panel(h['path'],f'history[{h["index"]}] · node {h["node_id"]}'+(' · 本轮输入' if h['index']==c['source_history_index'] else '')+(' · 本轮输出' if h['index']==c['edit_round'] else '')) for h in c['history'])
                parts.append('</div>')
                for t in c['chain_turns']:
                    parts.append(f'<div class="turn"><h4>第 {t["edit_round"]} 轮 · {esc(t["axis"])} / {esc(t["mode"])} · history[{t["source_history_index"]}] → history[{t["edit_round"]}] · node {t["source_node_id"]} → {t["target_node_id"]}</h4>'+prompts(t['prompt'])+reference_panels(t['references'])+'</div>')
                parts.append('</details>')
            parts.append('</article>')
        parts.append('</section>')
    nav=''.join(f'<a href="#group{i}">{esc(next(c["group_zh"] for c in m["cards"] if c["group"]==g))}</a>' for i,g in enumerate(groups))
    template=(ROOT/'template.html').read_text();page=template.replace('__CONTENT__',''.join(parts)).replace('__NAV__',nav)
    fontpath=ROOT.parent/'ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    report=(ROOT/'REPORT.md').read_text()
    font=TTFont(fontpath);sub=subset.Subsetter();sub.populate(text=page+report);sub.subset(font);buf=io.BytesIO();font.save(buf)
    face="@font-face{font-family:CJK;src:url(data:font/otf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('opentype');font-display:block}"
    (PACKAGE/'index.html').write_text(page.replace('__FONT__',face));shutil.copy2(fontpath.parent/'LICENSE.txt',PACKAGE/'FONT_LICENSE.txt')
    # An offline HTML reader keeps the report accessible when the browser downloads .md files.
    (PACKAGE/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Prompt 风格报告</title><style>'+face+'body{background:#10151d;color:#e9edf4;margin:30px auto;padding:20px;max-width:1200px;font:16px/1.8 CJK,system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#85ceff}</style><a href="index.html">返回 80 条审核页</a><pre>'+esc(report)+'</pre></html>')

def finalize():
    check=json.loads((PACKAGE/'browser_validation.json').read_text());assert check['status']=='passed' and check['cards']==80
    assert check['html_sha256']==sha(PACKAGE/'index.html')
    files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (PACKAGE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACKAGE)}\n' for p in files))
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    d=dict(file=str(archive.resolve()),bytes=archive.stat().st_size,sha256=sha(archive),cards=80,
           remote_path='/apps/bypy/lance_review/'+archive.name,status='ready_for_upload',remote_full_download_verification=False)
    save(archive.with_suffix('.sha256.json'),d);print(json.dumps(d,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');p.add_argument('--page',action='store_true');a=p.parse_args()
    if a.finalize:finalize()
    elif a.page:page(json.loads((PACKAGE/'manifest.json').read_text()))
    else:main()
