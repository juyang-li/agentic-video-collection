#!/usr/bin/env python3
"""Freeze, fetch and package a fresh, target-disjoint TripVVT garment survey."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time
import zipfile

import requests
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PREVIOUS=ROOT.parent/'ref_edit_review_20261030'
NAME='tripvvt_reference50_20261108'
OUTPUT=ROOT/NAME
BUNDLE=OUTPUT/NAME
ORIGINALS=OUTPUT/'originals'
REPO='TripVVT/TripVVT-10K'
REVISION='596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3'
SEED=20261108
QUOTAS={'upper_body':17,'lower_body':17,'dresses':16}
LABELS={'upper_body':'上衣','lower_body':'下装','dresses':'官方 dresses 类'}
METADATA_HASHES={'train':'1f1d1368f182836d697c0a0daa99169b45b61c2bf1a4ce7cef57b629d8538b83',
                 'test':'590d5b1ed925641f616238cfab9ad69318c1032f29d0c26af34916a2921db62b'}
FIELDS={'person_relevance':'人物编辑相关','instruction_correctness':'目标是否完成换衣指令',
        'logical_correctness':'编辑与动作逻辑','content_preservation':'身份、动作、背景等非编辑内容保持',
        'temporal_consistency':'跨帧稳定性','visual_quality':'画面质量','technical_quality':'文件播放与技术质量',
        'reference_match':'目标服装是否匹配参考图','edit_visible':'源视频到目标是否有实际换衣变化'}
sys.path.insert(0,str(ROOT.parent/'coinve_research'))
import package_random_review as media_tools


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def select():
    OUTPUT.mkdir(exist_ok=True)
    excluded={r['target_member'] for r in read_csv(ROOT/'random_pairs.csv')}
    excluded.update(r['output'] for r in json.loads((PREVIOUS/'metadata/garment_selection.json').read_text())['samples'])
    population=[]
    for split in ('train','test'):
        path=PREVIOUS/f'metadata/tripvvt_{split}.jsonl'
        assert sha(path)==METADATA_HASHES[split]
        for index,line in enumerate(path.read_text().splitlines()):
            if line.strip():population.append({**json.loads(line),'official_split':split,'official_row':index,'metadata_index':len(population)})
    assert len(population)==10031 and len(excluded)==40
    used=set(excluded);selected=[];rng=random.Random(SEED)
    for category,quota in QUOTAS.items():
        candidates=[r for r in population if r['cloth_type']==category and r['output'] not in excluded]
        rng.shuffle(candidates);count=0
        for r in candidates:
            if r['output'] in used:continue
            used.add(r['output']);row=dict(r)
            row.update(sample_id=f"tripvvt-{r['metadata_index']:05d}",sample_rank=len(selected)+1,
                       repo_id=REPO,revision=REVISION,selection_seed=SEED,
                       instruction_origin='local category template; not official caption')
            region={'upper_body':('上衣','upper-body garment'),'lower_body':('下装','lower-body garment'),'dresses':('服装','garment/outfit')}[category]
            row['instruction_zh']=f'将源视频中目标人物的{region[0]}替换为服装参考图所示衣物，保持人物身份、动作、其他衣物、其他人物、背景和镜头不变。'
            row['instruction_en']=f"Replace the target person's {region[1]} with the garment shown in the reference image. Preserve identity, motion, other clothing, other people, background, and camera."
            row['source_original']=f"originals/{row['sample_id']}_source.mp4"
            row['target_original']=f"originals/{row['sample_id']}_target.mp4"
            row['reference_original']=f"originals/{row['sample_id']}_garment.png"
            selected.append(row);count+=1
            if count==quota:break
        assert count==quota
    path=OUTPUT/'sample_manifest.csv'
    if path.exists():assert [r['sample_id'] for r in read_csv(path)]==[r['sample_id'] for r in selected], 'frozen sample differs'
    else:write_csv(path,selected)
    summary=dict(name=NAME,repo_id=REPO,revision=REVISION,metadata_sha256=METADATA_HASHES,
        seed=SEED,population=10031,quotas=QUOTAS,excluded_prior_target_count=len(excluded),
        excluded_prior_targets=sorted(excluded),unique_target_count=50,sample_size=50,
        sampling='category-stratified random order, without replacement by target across all categories; retain original official split only as provenance',
        sample_manifest_sha256=sha(path),official_split_counts=dict(Counter(r['official_split'] for r in selected)),
        model_judge_calls=0,model_inference_performed=False,visual_quality_replacements=0,
        split_warning='official train/test share target videos; this survey is not a clean held-out training split')
    save(OUTPUT/'sampling_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def download():
    rows=read_csv(OUTPUT/'sample_manifest.csv');ORIGINALS.mkdir(exist_ok=True)
    jobs=[(r,key,role) for r in rows for key,role in [('input','source'),('output','target'),('cloth_image','garment')]]
    token_file=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token')
    token=token_file.read_text().strip() if token_file.exists() else None
    def one(job):
        row,key,role=job
        path=ORIGINALS/f"{row['sample_id']}_{role}{'.png' if role=='garment' else '.mp4'}"
        if path.exists() and path.stat().st_size:return dict(file=path.name,status='preserved',bytes=path.stat().st_size)
        session=requests.Session();session.trust_env=False
        if token:session.headers['Authorization']='Bearer '+token
        status='unknown'
        try:
            for attempt in range(4):
                try:
                    with session.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row[key]}',stream=True,timeout=(20,90)) as response:
                        if response.status_code!=200:
                            status=str(response.status_code);raise RuntimeError('HTTP '+status)
                        expected=response.headers.get('Content-Length')
                        temporary=path.with_suffix(path.suffix+'.part')
                        with temporary.open('wb') as f:
                            for block in response.iter_content(1024*1024):f.write(block)
                        if expected and temporary.stat().st_size!=int(expected):raise RuntimeError('length mismatch')
                        assert temporary.stat().st_size>0
                        temporary.replace(path)
                        return dict(file=path.name,status='downloaded',bytes=path.stat().st_size)
                except (requests.RequestException,RuntimeError) as error:
                    status=type(error).__name__+' '+status
                    if attempt<3:time.sleep(2**attempt)
            raise RuntimeError('asset download failed: '+status)
        finally:session.close()
    results=[];failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:results.append(future.result())
            except Exception as error:failures.append(dict(sample_id=job[0]['sample_id'],role=job[2],error=type(error).__name__))
            if (len(results)+len(failures))%10==0:print('assets',len(results),'/',150,'failures',len(failures),flush=True)
            save(OUTPUT/'download_status.json',dict(complete=len(results),expected=150,failures=failures,assets=results))
    if failures:raise SystemExit('download incomplete; retry frozen sample, do not replace on transient errors')


def build():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    rows=sorted(read_csv(OUTPUT/'sample_manifest.csv'),key=lambda r:(list(QUOTAS).index(r['cloth_type']),int(r['metadata_index'])))
    BUNDLE.mkdir(exist_ok=True);(BUNDLE/'media').mkdir(exist_ok=True)
    media_tools.BUNDLE=BUNDLE
    with ThreadPoolExecutor(max_workers=6) as pool:
        videos=list(pool.map(media_tools.prepare_video,[(r,role,ORIGINALS) for r in rows for role in ('source','target')]))
    by_role={(r['sample_id'],r['role']):r for r in videos}
    cards=[];inventory=[];images=[]
    for row in rows:
        sid=row['sample_id'];source=by_role[(sid,'source')];target=by_role[(sid,'target')]
        s,t=source['original_info'],target['original_info']
        assert all(s[key]==t[key] for key in ('width','height','frames','fps_rational')), ('pair spec mismatch',sid,s,t)
        original=ORIGINALS/f'{sid}_garment.png'
        with Image.open(original) as im:im.verify()
        with Image.open(original) as im:image_info=dict(width=im.width,height=im.height,format=im.format)
        ref=f'media/{sid}_garment.png';shutil.copyfile(original,BUNDLE/ref)
        ref_hash=sha(original);images.append(dict(sample_id=sid,path=ref,sha256=ref_hash,bytes=original.stat().st_size,**image_info))
        panels=[dict(type='video',path=source['review_media'],label='SOURCE · 合成换装源视频',description=f"{s['width']}×{s['height']} · {s['fps']:.3f} FPS · {s['duration']:.3f} 秒"),
                dict(type='image',path=ref,label='GARMENT · 单张服装参考图',description='官方服装参考；请核对款式、颜色、图案与真实 target 是否对应'),
                dict(type='video',path=target['review_media'],label='TARGET · 原始真实视频',description=f"{t['width']}×{t['height']} · {t['fps']:.3f} FPS · {t['duration']:.3f} 秒；不是本次模型输出")]
        provenance=dict(repo_id=REPO,revision=REVISION,official_split=row['official_split'],official_row=int(row['official_row']),
            official_category=row['cloth_type'],source_member=row['input'],reference_member=row['cloth_image'],target_member=row['output'],
            source_sha256=source['original_sha256'],target_sha256=target['original_sha256'],reference_sha256=ref_hash,
            group_id=REPO+':'+row['output'],original_source_spec=s,original_target_spec=t,
            official_caption=row['text'],caption_role='global scene/target description, not edit instruction',
            instruction_origin=row['instruction_origin'],mask_member=row['mask_condition'],pose_member=row['pose_dw'],
            mask_pose_downloaded=False,pair_semantics='synthetic source + garment reference -> original real target',
            seed=SEED,split_warning='Official test targets overlap train. Do not use official split to claim video-disjoint evaluation.')
        cards.append(dict(id=sid,hash=source['original_sha256'][:5],task='try-on-'+row['cloth_type'],kind='garment',category=row['cloth_type'],
            title='参考图换衣 · '+LABELS[row['cloth_type']],instruction_zh=row['instruction_zh'],instruction_en=row['instruction_en'],
            notice='source 是合成换装视频，target 是原始真实视频。参考图是目标条件，不能用文字描述代替。\n重点检查：参考服装与 target 是否匹配、是否近似没换衣，以及人物身份、动作和背景是否保持。dresses 为官方类别，实际可能包含外套或套装。',
            panels=panels,not_applicable=[],provenance=provenance))
        inventory.append({**row, 'hash':source['original_sha256'][:5],'source_media':source['review_media'],'reference_media':ref,'target_media':target['review_media'],
            'source_original_sha256':source['original_sha256'],'reference_sha256':ref_hash,'target_original_sha256':target['original_sha256'],
            'source_review_sha256':source['review_sha256'],'target_review_sha256':target['review_sha256']})
    assert len(cards)==len({c['provenance']['target_member'] for c in cards})==50
    manifest=dict(version=NAME,cards=cards,fields=FIELDS,model_inference_performed=False,default_labels_are_not_results=True)
    save(BUNDLE/'manifest.json',manifest);write_csv(BUNDLE/'inventory.csv',inventory)
    if not (BUNDLE/'audit.csv').exists():write_csv(BUNDLE/'audit.csv',[dict(id=c['id'],hash=c['hash'],task=c['task'],**{f:'1' for f in FIELDS},reviewed=False,notes='') for c in cards])
    save(BUNDLE/'media_inventory.json',dict(videos=videos,images=images))
    shutil.copyfile(OUTPUT/'sample_manifest.csv',BUNDLE/'sample_manifest.csv')
    summary=json.loads((OUTPUT/'sampling_summary.json').read_text())
    summary.update(full_decode_videos=100,reference_images_verified=50,source_target_specs_match=50,
        source_specs=dict(Counter(f"{r['original_info']['width']}x{r['original_info']['height']} {r['original_info']['fps_rational']} {r['original_info']['frames']}frames" for r in videos if r['role']=='source')),
        duration_range_seconds=[min(r['original_info']['duration'] for r in videos),max(r['original_info']['duration'] for r in videos)],
        review_video_bytes=sum(r['review_bytes'] for r in videos),builder_sha256=sha(Path(__file__)),browser_validation='pending')
    save(BUNDLE/'summary.json',summary)
    template=(PREVIOUS/'review_template.html').read_text()
    template=template.replace('换衣与动作替换 · 数据审核','TripVVT · 50 组参考图换衣审核').replace('换衣 25 ＋ 动作替换 25','TripVVT · 全新 50 组参考图换衣')
    template=re.sub(r'<select id="filter">.*?</select>','<select id="filter"><option value="all">全部 50 组</option><option value="upper_body">上衣 17 组</option><option value="lower_body">下装 17 组</option><option value="dresses">dresses 16 组</option></select>',template)
    template=re.sub(r'const LABELS=\{.*?\};','const LABELS='+json.dumps(FIELDS,ensure_ascii=False)+';',template)
    template=template.replace("c.kind===$('filter').value","c.category===$('filter').value")
    template=template.replace('<p class="muted">${esc(p.description)}</p>','<p class="muted">${esc(p.description)}</p><a target="_blank" rel="noopener" href="${esc(p.path)}">独立查看完整素材 ↗</a>')
    template=template.replace('原速播放、不补帧、不循环。时长不同时各片段独立结束；可用播放器控件单独查看或全屏。','原速同步播放，不补帧、不变速。两侧暂停和跳转同步；可用独立链接查看完整素材。')
    # Pause and seek either side together; keep the full-duration clips intact.
    template=template.replace("$('instructionZh').textContent=c.instruction_zh;", "wireSync();$('instructionZh').textContent=c.instruction_zh;")
    sync="""let syncGuard=false;function wireSync(){const vs=videos();for(const v of vs){v.onplay=()=>{for(const o of vs)if(o!==v&&o.paused)o.play().catch(()=>pause())};v.onpause=()=>{for(const o of vs)if(!o.paused)o.pause()};v.onseeking=()=>{if(syncGuard)return;syncGuard=true;for(const o of vs)if(o!==v&&Math.abs(o.currentTime-v.currentTime)>.08)o.currentTime=v.currentTime;setTimeout(()=>syncGuard=false,60)};v.ontimeupdate=()=>{if(syncGuard||v.paused)return;for(const o of vs)if(o!==v&&!o.paused&&Math.abs(o.currentTime-v.currentTime)>.2){syncGuard=true;o.currentTime=v.currentTime;setTimeout(()=>syncGuard=false,60)}};v.onended=pause;}}\n"""
    template=template.replace("function render(){",sync+"function render(){",1)
    payload=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    font_path=PREVIOUS/'assets/font/NotoSansCJKsc-Regular.otf';font=TTFont(font_path)
    text=template+payload;sub=subset.Subsetter();sub.populate(text=text);sub.subset(font);font.flavor='woff'
    font.save(OUTPUT/'review_font.woff')
    assert {ord(c) for c in text if '\u4e00'<=c<='\u9fff'}<=set(font.getBestCmap())
    face="@font-face{font-family:ReviewCJK;src:url(data:font/woff;base64,"+base64.b64encode((OUTPUT/'review_font.woff').read_bytes()).decode()+") format('woff');font-display:block;}"
    (BUNDLE/'index.html').write_text(template.replace('__REVIEW_DATA__',payload).replace('__FONT_FACE__',face))
    shutil.copyfile(font_path.parent/'LICENSE.txt',BUNDLE/'FONT_LICENSE.txt')
    (BUNDLE/'browser_validation.json').unlink(missing_ok=True)
    report=['# TripVVT 50 组参考图换衣抽查','', '| 官方类别 | 组数 |','|---|---:|']
    report += [f'| {LABELS[k]} | {n} |' for k,n in QUOTAS.items()]
    report += ['', '50 个真实 target 均不重复，且不与此前 40 个审核 target 重叠。类别配额用于展示，不代表全库自然占比。',
        '', '## 技术规格','',json.dumps(summary['source_specs'],ensure_ascii=False,indent=2),
        '',f"视频时长范围：{summary['duration_range_seconds'][0]:.3f}–{summary['duration_range_seconds'][1]:.3f} 秒；100 个视频完整解码，50 张参考图验证通过。",'',
        '## 审核重点','', '参考与 target 的款式、颜色、图案是否一致；source→target 是否真的换衣；人物、动作、背景和非目标衣物是否保持；有无闪烁、形变或融合错误。',
        '', '官方 text 是全局场景/目标描述，原文见 manifest.json；页面中引用参考图的指令由本地类别模板生成。未调用质量 judge，未生成模型结果。']
    (BUNDLE/'REPORT.md').write_text('\n'.join(report)+'\n')
    (BUNDLE/'README.md').write_text('''# TripVVT 全新 50 组参考图换衣审核

完整解压 ZIP，用 Chrome / Edge 打开 index.html，无需服务器或联网。

- 上衣 17、下装 17、官方 dresses 类 16；固定 seed 20261108，排除此前 40 个真实 target，全包按 target 去重。不是全库自然分布抽样。
- SOURCE 是合成换装视频，GARMENT 是官方单张服装图，TARGET 是原始真实视频；本次未进行编辑模型推理。
- 同步从头播放、同步暂停/跳转，保留原速、完整分辨率、帧率和时长。浏览器视频 H.264/yuv420p/faststart，无音轨；原始媒体保存在包外 originals/。
- 可按服装类别筛选；所有审核项用 1/0/2，默认 1 仅为填写起点，reviewed=false 不计作人工通过。支持备注、CSV/JSON 导入导出和浏览器保存。
- 参考服装匹配和是否实际换衣为独立审核项，特别注意近似 no-op、款式不符和 dresses 类别不准确。多人物场景还需确认编辑对象。
- 页面指令来自本地类别模板，明确引用服装参考；官方 text 仅是场景/目标描述，完整原文保留在 manifest.json。
- manifest.json 为完整来源与语义，inventory.csv 为媒体/哈希/指令总目录，audit.csv 为未审核默认表，REPORT.md 与 summary.json 为类别及实际规格，media_inventory.json 为转码与完整解码记录。
- 官方 train/test 的真实 target 存在重叠，本包只保留其来源标签，不作为干净的泛化评测 split。mask/pose 路径仅保留在 metadata，未下载。本轮不转换 LTX2.5 训练格式或确定混合比例。
- 来源：TripVVT/TripVVT-10K，revision 596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3。许可记录见本地调研：CC BY-NC 4.0。
''')
    print('built',BUNDLE,flush=True)


def package():
    report=json.loads((BUNDLE/'browser_validation.json').read_text())
    assert report['status']=='passed' and report['cards_checked']==50 and report['index_sha256']==sha(BUNDLE/'index.html')
    inventory=json.loads((BUNDLE/'media_inventory.json').read_text())
    for r in inventory['videos']:assert sha(BUNDLE/r['review_media'])==r['review_sha256']
    for r in inventory['images']:assert sha(BUNDLE/r['path'])==r['sha256']
    summary=json.loads((BUNDLE/'summary.json').read_text());summary['browser_validation']='passed';save(BUNDLE/'summary.json',summary)
    files=[p for p in BUNDLE.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and not (p.parent.name=='media' and p.suffix=='.json')]
    (BUNDLE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(BUNDLE)}\n' for p in sorted(files)))
    files.append(BUNDLE/'SHA256SUMS');archive=OUTPUT/(NAME+'.zip')
    assert not archive.with_suffix('.baidu_upload.json').exists(),'do not replace uploaded release'
    temporary=archive.with_suffix('.part.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(files):z.write(p,Path(NAME)/p.relative_to(BUNDLE))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert sum(n.endswith('.mp4') for n in z.namelist())==100
        assert sum(n.endswith('_garment.png') for n in z.namelist())==50
    temporary.replace(archive)
    receipt=dict(path=str(archive.resolve()),bytes=archive.stat().st_size,sha256=sha(archive),triplet_count=50,
        video_count=100,reference_count=50,zip_crc='passed',browser_validation='passed',remote_path='/apps/bypy/lance_review/'+archive.name)
    save(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt),flush=True)


def upload():
    import build_balanced_review as delivery
    delivery.OUTPUT=OUTPUT;delivery.BUNDLE=BUNDLE;delivery.VERSION=NAME
    delivery.upload()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['select','download','build','package','upload']);args=parser.parse_args()
    globals()[args.phase]()
