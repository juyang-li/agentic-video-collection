#!/usr/bin/env python3
"""Package the 10 lowest-change pairs for human threshold calibration."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image
import build_review50 as shared

ROOT=Path(__file__).resolve().parent
NAME='tripvvt_lowchange10_20261108'
OUTPUT=ROOT/NAME
BUNDLE=OUTPUT/NAME
CALIBRATION=ROOT/'tripvvt_similarity_calibration_20261108'
BASE=ROOT/'tripvvt_reference50_20261108/tripvvt_reference50_20261108'
PREVIOUS=ROOT.parent/'ref_edit_review_20261030/review/ref_edit_clothing25_motion25_review_20261030'
SCORE_KEYS=('ssim_median','phash_median','rgb_mae_median','roi_mae_median','roi_changed_fraction_median')


def choose():
    report=json.loads((CALIBRATION/'pair_similarity.json').read_text())
    initial=json.loads((CALIBRATION/'initial_benchmark.json').read_text())
    assert report['successful']==report['pair_count']==75 and report['errors']==0
    old={r['id']:r for r in initial['records']}
    for row in report['records']:
        for key in SCORE_KEYS:
            assert abs(row[key]-old[row['sample_id']][key])<1e-9,(row['sample_id'],key)
    selected=sorted(report['records'],key=lambda r:(r['roi_mae_median'],r['sample_id']))[:10]
    old_ids=[r['id'] for r in sorted(initial['records'],key=lambda r:(r['roi_mae_median'],r['id']))[:10]]
    assert [r['sample_id'] for r in selected]==old_ids
    return selected,report,initial


def build():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    selected,report,initial=choose()
    OUTPUT.mkdir(exist_ok=True);BUNDLE.mkdir(exist_ok=True);(BUNDLE/'media').mkdir(exist_ok=True)
    (BUNDLE/'browser_validation.json').unlink(missing_ok=True)
    lookup={}
    for folder in (BASE,PREVIOUS):
        for card in json.loads((folder/'manifest.json').read_text())['cards']:
            if card['kind']=='garment':lookup[card['id']]=card
    shared.media_tools.BUNDLE=BUNDLE
    jobs=[(dict(sample_id=r['sample_id']),role,r) for r in selected for role in ('source','target')]
    def prepare(job):
        row,role,record=job
        # Reuse the original path without creating or modifying original files.
        original=Path(record[role])
        dest=BUNDLE/'media'/f"{row['sample_id']}_{role}.mp4"
        source_hash=shared.sha(original)
        info=shared.media_tools.probe(original)
        mode='lossless_video_stream_remux' if (info['codec'],info['pix_fmt'])==('h264','yuv420p') else 'h264_crf18_full_resolution'
        import subprocess
        receipt=dest.with_suffix('.json')
        previous=json.loads(receipt.read_text()) if receipt.exists() and dest.exists() else None
        if previous and previous['original_sha256']==source_hash and previous['review_sha256']==shared.sha(dest):return previous
        temp=dest.with_suffix('.part.mp4')
        cmd=[shared.media_tools.FFMPEG,'-nostdin','-v','error','-y','-threads','2','-i',str(original),'-map','0:v:0','-an']
        if mode=='lossless_video_stream_remux':cmd+=['-c:v','copy']
        else:cmd+=['-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','2','-fps_mode','passthrough']
        subprocess.run(cmd+['-movflags','+faststart',str(temp)],check=True)
        out=shared.media_tools.probe(temp)
        assert all(info[k]==out[k] for k in ('width','height','frames','fps_rational'))
        assert shared.media_tools.faststart(temp) and (out['codec'],out['pix_fmt'])==('h264','yuv420p')
        decoded=subprocess.run([shared.media_tools.FFMPEG,'-nostdin','-v','error','-xerror','-threads','2','-i',str(temp),'-map','0:v:0','-f','null','-'],capture_output=True,text=True,check=True)
        assert not decoded.stderr.strip()
        temp.replace(dest)
        result=dict(sample_id=row['sample_id'],role=role,original_sha256=source_hash,original_info=info,
            review_info=out,review_media='media/'+dest.name,review_sha256=shared.sha(dest),
            review_bytes=dest.stat().st_size,conversion=mode,full_decode='passed',faststart=True)
        shared.save(receipt,result);return result
    with ThreadPoolExecutor(max_workers=4) as pool:videos=list(pool.map(prepare,jobs))
    video_map={(r['sample_id'],r['role']):r for r in videos}
    cards=[];inventory=[];images=[];local_input=[]
    for rank,row in enumerate(selected,1):
        sid=row['sample_id'];card=copy.deepcopy(lookup[sid]);card['low_change_rank']=rank
        card['category']=row['category']
        original_provenance=card['provenance']
        for role in ('source','target'):
            m=video_map[(sid,role)]
            assert m['original_sha256']==original_provenance[role+'_sha256'],(sid,role)
        reference=Path(row['reference']);ref='media/'+sid+'_garment.png'
        with Image.open(reference) as im:im.verify()
        assert shared.sha(reference)==original_provenance['reference_sha256']
        shutil.copyfile(reference,BUNDLE/ref)
        images.append(dict(sample_id=sid,path=ref,sha256=shared.sha(reference),bytes=reference.stat().st_size))
        card['panels'][0]['path']=video_map[(sid,'source')]['review_media']
        card['panels'][1]['path']=ref
        card['panels'][2]['path']=video_map[(sid,'target')]['review_media']
        card['title']=f"低变化 #{rank:02d} · "+shared.LABELS[row['category']]
        card['notice']='这是 75 对实测样本中变化指标最低的 10 对，按变化从小到大排列；不是已确认的 no-op，也未用于自动筛除。\n请判断换衣变化是否足够、参考图是否匹配。source 为合成换装视频，target 为原始真实视频。'
        card['similarity']=dict(rank=rank,**{k:row[k] for k in SCORE_KEYS},
            sample_times_seconds=row['sample_times_seconds'],
            region_x=report['parameters']['region_x'],region_y=report['parameters']['region_y'][row['category']],
            note='固定类别区域，不是服装分割。ROI_MAE 越低代表变化越小；SSIM 越高代表结构越相似；这些指标不能证明语义 no-op。')
        cards.append(card)
        inventory.append(dict(rank=rank,sample_id=sid,category=row['category'],cohort=row['cohort'],
            source=card['panels'][0]['path'],reference=ref,target=card['panels'][2]['path'],
            instruction_zh=card['instruction_zh'],instruction_en=card['instruction_en'],
            official_caption=original_provenance['official_caption'],**{k:row[k] for k in SCORE_KEYS},
            source_original_sha256=original_provenance['source_sha256'],
            reference_sha256=original_provenance['reference_sha256'],target_original_sha256=original_provenance['target_sha256']))
        local_input.append(dict(sample_id=sid,category=row['category'],source=card['panels'][0]['path'],
            target=card['panels'][2]['path'],reference=ref,source_frames=121,target_frames=121,source_fps=24,target_fps=24))
    assert len({c['provenance']['target_member'] for c in cards})==10
    manifest=dict(version=NAME,cards=cards,fields=shared.FIELDS,
        selection_policy='roi_mae_ascending',population=75,default_labels_are_not_results=True,
        production_filtering_performed=False)
    shared.save(BUNDLE/'manifest.json',manifest)
    # UI follows explicitly requested ranking; CSV records also retain rank.
    shared.write_csv(BUNDLE/'inventory.csv',inventory)
    if not (BUNDLE/'audit.csv').exists():
        shared.write_csv(BUNDLE/'audit.csv',[dict(id=c['id'],hash=c['hash'],task=c['task'],rank=c['low_change_rank'],
            **{f:'1' for f in shared.FIELDS},reviewed=False,notes='') for c in cards])
    shared.write_csv(BUNDLE/'scoring_input10.csv',local_input)
    shared.save(BUNDLE/'media_inventory.json',dict(videos=videos,images=images))
    shared.save(OUTPUT/'selection_receipt.json',dict(population=75,count=10,
        categories=dict(Counter(r['category'] for r in selected)),
        ordered_ids=[r['sample_id'] for r in selected],
        source_metrics_sha256=shared.sha(CALIBRATION/'pair_similarity.json'),
        all_75_scores_reproduced=True,original_top10_reproduced=True))
    shutil.copyfile(OUTPUT/'selection_receipt.json',BUNDLE/'selection_receipt.json')
    shutil.copyfile(CALIBRATION/'pair_similarity.csv',BUNDLE/'similarity_scores75.csv')
    shutil.copyfile(ROOT/'score_pair_similarity.py',BUNDLE/'score_pair_similarity.py')
    benchmark=dict(initial={k:initial[k] for k in ['pair_count','workers','serial10_seconds','parallel_seconds','serial1000_seconds','parallel1000_seconds']},
        replay={k:report[k] for k in ['pair_count','workers','serial10_seconds','parallel_seconds','serial1000_seconds','parallel1000_seconds','environment','parameters','script_sha256']},
        caveat='本地已下载素材；不含下载，可能命中操作系统缓存，负载和 I/O 会影响速度。')
    shared.save(BUNDLE/'benchmark.json',benchmark)
    page=(BASE/'index.html').read_text()
    marker="const DATA=";end_marker=",KEY=DATA.version+':audit';"
    start=page.index(marker)+len(marker);end=page.index(end_marker,start)
    payload=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    page=page[:start]+payload+page[end:]
    page=page.replace('TripVVT · 50 组参考图换衣审核','TripVVT · 低变化 10 对审核').replace('TripVVT · 全新 50 组参考图换衣','TripVVT · 低变化 10 对审核')
    page=page.replace(' / 50',' / 10')
    page=re.sub(r'<select id="filter">.*?</select>','<select id="filter"><option value="all">全部 10 对 · 按变化排名</option><option value="upper_body">上衣 3 对</option><option value="lower_body">下装 5 对</option><option value="dresses">dresses 2 对</option></select>',page)
    old_prompt='<div class="instruction"><div id="instructionZh"></div><div class="muted" id="instructionEn"></div></div>'
    assert page.count(old_prompt)==1
    page=page.replace(old_prompt,'')
    prompt='<section class="instruction" style="margin-bottom:12px"><strong>换衣 prompt（本地参考图条件指令）</strong><div id="instructionZh"></div><div class="muted" id="instructionEn"></div><details id="captionDetails"><summary>官方 text（目标服装和场景描述，不是换衣命令）</summary><p id="officialCaption"></p></details></section>'
    page=page.replace('<div class="media" id="media">',prompt+'<div class="media" id="media">')
    page=page.replace('<details><summary>查看来源、配对方式与技术规格</summary>',
        '<details id="similarityDetails"><summary>展开相似度数值与计算区域</summary><pre class="source" id="similarityValues"></pre></details><details><summary>查看来源、配对方式与技术规格</summary>')
    page=page.replace("$('instructionZh').textContent=c.instruction_zh;",
        "$('similarityDetails').open=false;$('similarityValues').textContent=JSON.stringify(c.similarity,null,2);$('officialCaption').textContent=c.provenance.official_caption;$('instructionZh').textContent=c.instruction_zh;")
    page=re.sub(r'@font-face\{font-family:ReviewCJK;.*?\}', '__FONT_FACE__',page)
    font_path=shared.PREVIOUS/'assets/font/NotoSansCJKsc-Regular.otf';font=TTFont(font_path)
    sub=subset.Subsetter();sub.populate(text=page);sub.subset(font);font.flavor='woff'
    font.save(OUTPUT/'review_font.woff')
    assert {ord(c) for c in page if '\u4e00'<=c<='\u9fff'}<=set(font.getBestCmap())
    face='@font-face{font-family:ReviewCJK;src:url(data:font/woff;base64,'+base64.b64encode((OUTPUT/'review_font.woff').read_bytes()).decode()+") format('woff');font-display:block;}"
    (BUNDLE/'index.html').write_text(page.replace('__FONT_FACE__',face))
    shutil.copyfile(font_path.parent/'LICENSE.txt',BUNDLE/'FONT_LICENSE.txt')
    text=['# TripVVT 低变化 10 对：阈值校准审核','',
        '完整解压，用 Chrome/Edge 打开 index.html。按服装区域 RGB 差异中位数从低到高取 10 对；同分按 ID。来自 75 对本地样本：上衣 3、下装 5、dresses 2。',
        '', '## 如何审核','',
        '三栏为完整 source、单张服装参考、真实 target。prompt 位于素材上方，官方 text 单独标注。相似度数值默认折叠。请重点填写“是否有实际换衣变化”和“是否匹配参考图”，可备注你希望保留或剔除的情况。',
        '', '1=通过、0=不通过、2=存疑；默认 1、reviewed=false 不是人工通过。支持同步播放/暂停/跳转、类别筛选和 CSV/JSON 导入导出。阈值未冻结，正式数据池未筛除。',
        '', '## 排名','', '| 排名 | ID | 类别 | ROI RGB 差异 | 分块 SSIM | 哈希距离 |','|---|---|---|---:|---:|---:|']
    text += [f"| {i} | {r['sample_id']} | {r['category']} | {r['roi_mae_median']:.6f} | {r['ssim_median']:.6f} | {r['phash_median']:g} |" for i,r in enumerate(selected,1)]
    text += ['', '## 计算方法与复跑','',
        '纯 CPU，无视觉模型。每侧取共同最后有效帧时刻的 10/30/50/70/90% 时间点，缩放至 144×256，计算 63 位感知哈希、8×8 分块 SSIM 和 RGB 差异。RGB 差异除以 255 后归一化，区域横向均为 15–85%；上衣纵向 15–60%、下装 45–95%、dresses 15–95%。这是固定区域，不是服装分割，指标不等同于语义 no-op。',
        '', '脚本依赖 Python、numpy、Pillow、FFmpeg/ffprobe。在本目录运行：', '',
        '```bash','python score_pair_similarity.py --manifest scoring_input10.csv --out local_rescore --workers 4',
        '```','', 'FFmpeg 未在 PATH 时可用 --ffmpeg /path/to/ffmpeg 指定。完整 75 对输入清单、原始实测和复跑结果另保存在项目的 tripvvt_similarity_calibration_20261108/；本 ZIP 只包含排名前 10 对素材。',
        '', '## 速度','',
        f"初次实测：串行 10 对 {initial['serial10_seconds']:.2f} 秒，4 并发 75 对 {initial['parallel_seconds']:.2f} 秒；1K 对线性估计 {initial['parallel1000_seconds']/60:.2f} 分钟。",
        f"脚本复跑：串行 10 对 {report['serial10_seconds']:.2f} 秒，4 并发 75 对 {report['parallel_seconds']:.2f} 秒；1K 对线性估计 {report['parallel1000_seconds']/60:.2f} 分钟。",
        '估计不含下载；本地 I/O、缓存和 CPU 负载会影响结果。详细环境及参数见 benchmark.json。',
        '', 'manifest.json 保存逐条来源和语义；inventory.csv 保存排名、指令和哈希；similarity_scores75.csv 为完整分布；audit.csv 为未审核初始表。原始 50 对和 25 对审核包保持原版本，标注通过 ID 对应，不自动覆盖已有结果。',
        '', '后续池规模已确定为训练 10,000＋验证 500＋测试 500，共 11,000 条；本次仅校准 TripVVT 筛除规则。']
    (BUNDLE/'README.md').write_text('\n'.join(text)+'\n')
    shared.save(BUNDLE/'summary.json',dict(name=NAME,triplet_count=10,video_count=20,reference_count=10,
        source_population=75,categories=dict(Counter(r['category'] for r in selected)),
        score_reproduction_tolerance=1e-9,ranking_reproduced=True,full_decode_videos=20,
        metric_threshold_frozen=False,production_filtering_performed=False,browser_validation='pending'))
    print('built',BUNDLE,flush=True)


def package():
    check=json.loads((BUNDLE/'browser_validation.json').read_text())
    assert check['status']=='passed' and check['cards_checked']==10 and check['index_sha256']==shared.sha(BUNDLE/'index.html')
    reproduction=json.loads((BUNDLE/'reproduction_validation.json').read_text())
    assert reproduction['status']=='passed' and reproduction['pair_count']==10 and reproduction['review_media_scores_match_originals']
    inv=json.loads((BUNDLE/'media_inventory.json').read_text())
    for r in inv['videos']:assert shared.sha(BUNDLE/r['review_media'])==r['review_sha256']
    for r in inv['images']:assert shared.sha(BUNDLE/r['path'])==r['sha256']
    summary=json.loads((BUNDLE/'summary.json').read_text());summary['browser_validation']='passed';shared.save(BUNDLE/'summary.json',summary)
    files=[p for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and not(p.parent.name=='media' and p.suffix=='.json')]
    (BUNDLE/'SHA256SUMS').write_text(''.join(f'{shared.sha(p)}  {p.relative_to(BUNDLE)}\n' for p in files));files.append(BUNDLE/'SHA256SUMS')
    archive=OUTPUT/(NAME+'.zip');assert not archive.with_suffix('.baidu_upload.json').exists()
    temp=archive.with_suffix('.part.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as z:
        for p in files:z.write(p,Path(NAME)/p.relative_to(BUNDLE))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert sum(n.endswith('.mp4') for n in z.namelist())==20
        assert sum(n.endswith('_garment.png') for n in z.namelist())==10
    temp.replace(archive)
    receipt=dict(path=str(archive.resolve()),bytes=archive.stat().st_size,sha256=shared.sha(archive),
        triplet_count=10,video_count=20,reference_count=10,zip_crc='passed',browser_validation='passed',
        remote_path='/apps/bypy/lance_review/'+archive.name)
    shared.save(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt),flush=True)


def upload():
    import build_balanced_review as delivery
    delivery.OUTPUT=OUTPUT;delivery.BUNDLE=BUNDLE;delivery.VERSION=NAME;delivery.upload()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['build','package','upload'])
    args=parser.parse_args();globals()[args.phase]()
