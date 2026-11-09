#!/usr/bin/env python3
"""Build a continuous, score-free, offline motion-data browsing package."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,base64,csv,hashlib,html,io,json,shutil,subprocess,zipfile
import cv2,imageio_ffmpeg
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT=Path(__file__).resolve().parent
NAME='recent_motion_datasets_scroll50_20261030'
PACKAGE=ROOT/'review'/NAME
FF=imageio_ffmpeg.get_ffmpeg_exe()


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def info(p):
    c=cv2.VideoCapture(str(p));d={k:int(c.get(v)) for k,v in [('width',3),('height',4),('frames',7)]};d['fps']=c.get(5)
    d['duration']=round(d['frames']/d['fps'],3) if d['fps'] else 0;fourcc=int(c.get(cv2.CAP_PROP_FOURCC));d['fourcc']=''.join(chr(fourcc>>8*i&255) for i in range(4));c.release()
    assert d['frames']>0 and d['fps']>0,p
    return d


def video(job):
    raw,out=job;before=info(raw)
    if not out.exists():
        codec=['-c:v','copy'] if before['fourcc']=='avc1' else ['-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p']
        tmp=out.with_suffix('.part.mp4')
        run=subprocess.run([FF,'-nostdin','-v','error','-y','-i',str(raw),'-map','0:v:0','-an',*codec,'-movflags','+faststart',str(tmp)],capture_output=True,text=True)
        if run.returncode:raise RuntimeError(out.name+': '+run.stderr[-500:])
        tmp.replace(out)
    after=info(out)
    assert (before['width'],before['height'])==(after['width'],after['height'])
    assert abs(before['duration']-after['duration'])<.15
    check=subprocess.run([FF,'-nostdin','-v','error','-threads','2','-i',str(out),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
    assert check.returncode==0,(out,check.stderr[-500:])
    poster=out.with_suffix('.jpg')
    cap=cv2.VideoCapture(str(out));cap.set(cv2.CAP_PROP_POS_FRAMES,int(after['frames']*.15));ok,frame=cap.read();cap.release();assert ok
    scale=min(1,600/max(frame.shape[:2]));thumb=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
    cv2.imwrite(str(poster),thumb,[cv2.IMWRITE_JPEG_QUALITY,90])
    return {'path':out.relative_to(PACKAGE).as_posix(),'poster':poster.relative_to(PACKAGE).as_posix(),
            'original_path':raw.relative_to(ROOT).as_posix(),'original_sha256':sha(raw),'sha256':sha(out),
            'bytes':out.stat().st_size,'full_decode_ok':True,'original_resolution_preserved':True,'original_speed_preserved':True,**after}


def make():
    PACKAGE.mkdir(parents=True,exist_ok=True);(PACKAGE/'media').mkdir(exist_ok=True)
    facts=json.loads((ROOT/'metadata/research_facts.json').read_text());sel=json.loads((ROOT/'selection.json').read_text());cards=[];jobs=[]
    def image(raw,name,label):
        out=PACKAGE/'media'/(name+raw.suffix.lower())
        if not out.exists():shutil.copy2(raw,out)
        return {'type':'image','path':out.relative_to(PACKAGE).as_posix(),'label':label,'original_path':raw.relative_to(ROOT).as_posix()}
    def vid(raw,name,label):
        out=PACKAGE/'media'/(name+'.mp4');jobs.append((raw,out))
        return {'type':'video','path':out.relative_to(PACKAGE).as_posix(),'label':label}
    refs=[*sorted((ROOT/'assets/xdance/Ref-Female-Whole-Body').glob('*.png')),*sorted((ROOT/'assets/xdance/Ref-Male-Whole-Body').glob('*.png'))]
    upper=[ROOT/'assets/xdance/Ref-Male-Upper-Body/00001.png',ROOT/'assets/xdance/Ref-Male-Upper-Body/00002.png',ROOT/'assets/xdance/Ref-Female-Upper-Body/00001.png',refs[0]]
    repo='MCG-NJU/X-Dance'
    for mode,count in [('Complex',8),('Simple',4)]:
        for i in range(1,count+1):
            ident=f'xdance_{mode.lower()}_{i:02d}';d=ROOT/'assets/xdance'/('Video-'+mode)/f'{i:05d}'
            ref=refs[(i-1)%len(refs)] if mode=='Complex' else upper[i-1]
            cards.append({'id':ident,'group':'xdance','title':sel['xdance_'+mode.lower()+'_labels'][i-1],
                'semantics':'跨来源人物图片＋动作视频。没有跨人物编辑后 GT；此处图片配对用于浏览，来自官方参考图库，不冒称官方固定配对清单。',
                'note':'复杂动作、转身、遮挡等是这个 benchmark 的难例。原始 RGB 视频和官方 pose 分开显示；pose 不是结果视频。',
                'repo':repo,'revision':facts['datasets'][repo]['sha'],'native_member':f'Video-{mode}/{i:05d}/video.mp4',
                'caption':(d/'prompt.txt').read_text().strip(),
                'panels':[image(ref,ident+'_ref','人物外观参考图'),vid(d/'video.mp4',ident+'_drive','原始动作视频 RGB'),vid(d/'pose.mp4',ident+'_pose','官方动作表示 Pose（非结果）')]})
    repo='theFoxofSky/RealisDance-Val'
    for n,title in zip(sel['realis_ids'],sel['realis_labels']):
        ident=f'realis_{n:04d}';d=ROOT/'assets/realis'
        cards.append({'id':ident,'group':'realis','title':title,
            'semantics':'官方同来源参考图＋原视频，用于 self-reenactment（自重演）。原视频既提供动作也可作重建 GT，不是“旧动作→新动作”的跨动作编辑 GT。',
            'note':'优先看动作是否清楚、人物是否完整、遮挡和物体交互是否符合目标任务。保留原始分辨率和完整时长。',
            'repo':repo,'revision':facts['datasets'][repo]['sha'],'native_member':f'gt/{n:04d}.mp4',
            'caption':(d/'prompt'/f'{n:04d}.txt').read_text().strip(),
            'panels':[image(d/'ref'/f'{n:04d}.png',ident+'_ref','官方参考图（同一原视频）'),vid(d/'gt'/f'{n:04d}.mp4',ident+'_drive','原视频／动作素材（自重演 GT）')]})
    repo='Ruihang/MoveBench';ms=json.loads((ROOT/'metadata/movebench_selection.json').read_text());captions={r['id']:r['caption'] for r in ms['samples']}
    for n,title in zip(sel['move_ids'],sel['move_labels']):
        ident='move_'+n.removeprefix('Pexels_');d=ROOT/'assets/movebench'
        cards.append({'id':ident,'group':'movebench','title':title,
            'semantics':'Wan-Move 的点轨迹控制 benchmark 人物子集：同视频首帧＋原视频，可作为独立动作参考素材；没有跨人物动作替换 GT。',
            'note':'补充区：原始发布为 832×480，低于前两组的多数片段。这里保留原分辨率供直接比较，不上采样伪装高清。',
            'repo':repo,'revision':facts['datasets'][repo]['sha'],'native_member':f'en/video/{n}.mp4','caption':captions[n],
            'panels':[image(d/'first_frame'/f'{n}.jpg',ident+'_ref','官方首帧'),vid(d/'video'/f'{n}.mp4',ident+'_drive','原视频／人物动作素材')]})
    examples=json.loads((ROOT/'metadata/wan_official_examples.json').read_text())
    for folder,mode,title in [('wan1','animate','Wan-Animate：人物动画'),('wan1','replace','Wan-Animate：角色替换'),('wan2','demo1','Wan-Animate-2：直接视频驱动')]:
        if folder=='wan1':d=ROOT/'assets/wan1/examples/wan_animate'/mode;rawim=d/'image.jpeg';rawvideo=d/'video.mp4'
        else:d=ROOT/'assets/wan2/examples/demo1';rawim=d/'reference.png';rawvideo=d/'template.mp4'
        provenance=next(x for x in examples if x['local']==rawvideo.relative_to(ROOT).as_posix());ident=folder+'_'+mode
        cards.append({'id':ident,'group':'wan','title':title,'repo':provenance['repo'],'revision':provenance['revision'],
            'native_member':provenance['member'],'caption':'',
            'semantics':'官方仓库随代码发布的输入示例，单列展示；不是已公开的完整训练数据集，也不是随机抽查样本。',
            'note':'Animate 的 replacement 主要替换角色外观并沿用驱动动作；不要与保留 source 外观、只替换其动作混为同一评测协议。',
            'panels':[image(rawim,ident+'_ref','官方角色参考图'),vid(rawvideo,ident+'_drive','官方 Driving Video')]})
    assert len(cards)==50 and Counter(c['group'] for c in cards)=={'xdance':12,'realis':24,'movebench':11,'wan':3}
    reports=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(video,j) for j in jobs]),1):
            reports.append(f.result());print(f'media prepared {i}/{len(jobs)}',flush=True)
    specs={r['path']:r for r in reports}
    for card in cards:
        for p in card['panels']:
            if p['type']=='video':p.update(specs[p['path']])
    data={'name':NAME,'cards':cards,'research_facts':facts,'selection_method':sel['selection_method'],
          'no_rating_fields':True,'no_model_inference':True,'original_video_speed_and_resolution':True}
    save(PACKAGE/'manifest.json',data);save(PACKAGE/'research_facts.json',facts)
    with (PACKAGE/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','dataset','title','semantics','native_member','revision','video','width','height','fps','duration']);w.writeheader()
        for c in cards:
            p=next(p for p in c['panels'] if p['type']=='video')
            w.writerow({'id':c['id'],'dataset':c['repo'],'title':c['title'],'semantics':c['semantics'],'native_member':c['native_member'],'revision':c['revision'],'video':p['path'],**{k:p[k] for k in ['width','height','fps','duration']}})
    save(PACKAGE/'technical_validation.json',reports)
    write_page(data)
    shutil.copy2(ROOT/'RESEARCH.md',PACKAGE/'RESEARCH.md')
    (PACKAGE/'README.md').write_text('# 近期动作数据浏览包\n\n解压后打开 index.html，整页连续向下滚动。无评分栏、无前后翻页按钮。点击播放或全屏查看，画面离开可见区后自动暂停。\n\n共50条：X-Dance 12、RealisDance-Val 24、MoveBench人物子集11、Wan官方示例3。前三组是公开数据集，最后一组仅为官方输入示例。\n\n原视频保留分辨率、帧率、完整时长；H.264视频只做无损重封装和移除音轨，其他编码按原分辨率转H.264。没有超分、补帧、变速或模型推理。\n\n这些素材的公开协议主要是“人物图片＋驱动视频”或“首帧＋轨迹”，不自动构成“source视频＋独立动作视频→编辑后GT”。逐条已标明协议与GT含义。\n\n日期范围按数据实际公开时间2025-04-30至2026-10-30。RealisDance-DiT论文为2025-04，数据公开库创建于2025-05-21，两者日期明确区分。热度数字为2026-10-30快照；star不是数据集用户数，HF下载次数也不是独立人数。\n\n研究结论、正式来源、授权与待申请数据见RESEARCH.md；逐文件来源和规格见manifest.json。\n')
    print(PACKAGE,flush=True)


def write_page(data):
    esc=html.escape
    sections=[('xdance','① X-Dance · 独立动作参考','2025-11 公开；SteadyDancer 652 stars，HF 月下载191、likes20。12条驱动原片全部展示，包含8段复杂动作和4段日常动作。'),
      ('realis','② RealisDance-Val · 高清动作素材','公开库创建于2025-05-21；关联项目616 stars，HF月下载903。自100条原视频中目视选24条。论文发表于2025-04，数据公开时间单独列明。'),
      ('movebench','③ MoveBench · Wan-Move 人物子集','2025-12公开；关联项目662 stars，HF月下载321。选11条人物运动素材；原始480p，作为补充比较。'),
      ('wan','④ Wan-Animate 官方输入示例','Wan-Animate于2025-09、Wan-Animate-2于2026-08发布。示例仅解释实际输入形式，不冒充完整训练数据。')]
    parts=[]
    for group,title,intro in sections:
        parts.append(f'<section id="{group}"><h2>{esc(title)}</h2><p class="intro">{esc(intro)}</p>')
        for c in [x for x in data['cards'] if x['group']==group]:
            panels=[]
            for p in c['panels']:
                if p['type']=='image':content=f'<img loading="lazy" src="{esc(p["path"])}" alt="人物参考图">';detail='原始参考图片'
                else:
                    content=f'<video controls muted playsinline preload="none" data-src="{esc(p["path"])}" poster="{esc(p["poster"])}"></video>'
                    detail=f'{p["width"]}×{p["height"]} · {p["fps"]:.2f} FPS · {p["duration"]:.2f}s'
                panels.append(f'<div class="panel"><h4>{esc(p["label"])}</h4>{content}<p class="spec">{esc(detail)} · <a href="{esc(p["path"])}" target="_blank">单独打开原分辨率文件</a></p></div>')
            url=('https://github.com/' if c['group']=='wan' else 'https://huggingface.co/datasets/')+c['repo']
            parts.append(f'<article id="{c["id"]}" data-group="{group}"><h3>{esc(c["id"])} · {esc(c["title"])}</h3><p class="meaning">{esc(c["semantics"])}</p><div class="grid cols{len(panels)}">'+''.join(panels)+f'</div><div class="tools"><button class="play">本组从头播放</button><button class="pause">本组暂停</button></div><p class="note">{esc(c["note"])}</p><details><summary>官方描述与来源</summary><p>{esc(c["caption"])}</p><a href="{url}">{esc(c["repo"])}</a><p class="spec">版本 {esc(c["revision"])} · 文件 {esc(c["native_member"])}</p></details></article>')
        parts.append('</section>')
    template=(ROOT/'scroll_template.html').read_text();page=template.replace('__CONTENT__',''.join(parts))
    fullfont=ROOT.parent/'ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    font=TTFont(fullfont);sub=subset.Subsetter();sub.populate(text=page);sub.subset(font);buf=io.BytesIO();font.save(buf)
    face="@font-face{font-family:CJK;src:url(data:font/otf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('opentype');font-display:block}"
    (PACKAGE/'index.html').write_text(page.replace('__FONT__',face))
    shutil.copy2(fullfont.parent/'LICENSE.txt',PACKAGE/'FONT_LICENSE.txt')


def finalize():
    check=json.loads((PACKAGE/'browser_validation.json').read_text());assert check['status']=='passed' and check['cards']==50 and check['rating_controls']==0 and check['external_requests']==0
    assert check['html_sha256']==sha(PACKAGE/'index.html')
    files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (PACKAGE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACKAGE).as_posix()}\n' for p in files))
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'file':str(archive.resolve()),'bytes':archive.stat().st_size,'sha256':sha(archive),'cards':50,
             'remote_path':'/apps/bypy/lance_review/'+archive.name,'status':'ready_for_upload','remote_full_download_verification':False}
    save(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt,ensure_ascii=False))


def refresh_titles():
    data=json.loads((PACKAGE/'manifest.json').read_text());sel=json.loads((ROOT/'selection.json').read_text())
    labels={f'realis_{n:04d}':s for n,s in zip(sel['realis_ids'],sel['realis_labels'])}
    for c in data['cards']:
        if c['id'] in labels:c['title']=labels[c['id']]
    save(PACKAGE/'manifest.json',data);write_page(data)
    with (PACKAGE/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        if row['id'] in labels:row['title']=labels[row['id']]
    with (PACKAGE/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');p.add_argument('--refresh-titles',action='store_true');a=p.parse_args()
    if a.finalize:finalize()
    elif a.refresh_titles:refresh_titles()
    else:make()
