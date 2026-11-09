#!/usr/bin/env python3
"""Measure CoinVE length ratios and freeze a 30-pair threshold-boundary review."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import build_random_review as shared
import package_random_review as media

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
POOL=PROJECT/'lance_general_edit_finetune/3_lance_lora_finetuning/artifacts/coinve_ablation_v1'
NAME='coinve_duration_ratio30_20261108'
OUTPUT=ROOT/NAME
BUNDLE=OUTPUT/NAME
THRESHOLDS=(1.0,1.1,1.25,1.5,2.0)
SEED=20261108


def csv_write(path,rows):
    import csv
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def measure():
    metadata={r['sample_id']:r for r in shared.read_csv(POOL/'candidate_manifest.csv')}
    probes={r['sample_id']:r for r in map(json.loads,(POOL/'technical_probe.jsonl').open())}
    cohort=[]
    for sid,probe in probes.items():
        row=metadata[sid]
        ops=json.loads(row['operations_json']);instructions=json.loads(row['instructions_json'])
        if any('styl' in str(o).lower() for o in ops) or len(instructions) not in (2,3):continue
        assert len(instructions)==len(ops)==int(row['instruction_count'])
        s,t=probe.get('source'),probe.get('target')
        if not s or not t:continue
        if any(not math.isfinite(v[k]) or v[k]<=0 for v in (s,t) for k in ('frames','fps','duration')):continue
        if min(s['duration'],t['duration'])<4:continue
        if (s['width'],s['height'])!=(t['width'],t['height']):continue
        if abs(s['fps']-t['fps'])>max(.1,.01*s['fps']):continue
        frame_ratio=max(s['frames'],t['frames'])/min(s['frames'],t['frames'])
        duration_ratio=max(s['duration'],t['duration'])/min(s['duration'],t['duration'])
        cohort.append(dict(sample_id=sid,instruction_count=len(instructions),anchor_operation=row['anchor_operation'],
            source_frames=s['frames'],target_frames=t['frames'],source_fps=s['fps'],target_fps=t['fps'],
            source_duration=s['duration'],target_duration=t['duration'],
            frame_ratio=frame_ratio,duration_ratio=duration_ratio,ratio=max(frame_ratio,duration_ratio),
            frame_difference=abs(s['frames']-t['frames']),duration_difference=abs(s['duration']-t['duration'])))
    assert len(cohort)==2693, 're-audit the cohort if frozen inputs change'
    stats=[]
    for threshold in THRESHOLDS:
        kept=sum(r['ratio']<=threshold+1e-9 for r in cohort)
        stats.append(dict(threshold=threshold,definition='max(frame_ratio,duration_ratio) <= threshold + 1e-9',
            total=len(cohort),retained=kept,rejected=len(cohort)-kept,rejected_percent=100*(len(cohort)-kept)/len(cohort)))
    equal_tolerant=sum(r['source_frames']==r['target_frames'] and r['duration_difference']<=max(1/r['source_fps'],1/r['target_fps'])+1e-6 for r in cohort)
    summary=dict(name=NAME,repo_id=shared.REPO_ID,revision=shared.REVISION,seed=SEED,
        probed_pairs=len(probes),population=len(cohort),thresholds=stats,
        equal_frames_one_frame_time_tolerance=dict(retained=equal_tolerant,rejected=len(cohort)-equal_tolerant,
            rejected_percent=100*(len(cohort)-equal_tolerant)/len(cohort)),
        base_conditions='no Stylize, exactly 2 or 3 subinstructions, both >=4s, equal dimensions, FPS difference <=max(.1,1% source FPS)',
        input_manifest_sha256=shared.digest(POOL/'candidate_manifest.csv'),probe_log_sha256=shared.digest(POOL/'technical_probe.jsonl'),
        ratio_definition='max(max(source_frames,target_frames)/min(source_frames,target_frames), max(source_duration,target_duration)/min(source_duration,target_duration))',
        limitation='Local probed cohort, not all 200,916 rows. Historical source-filename span>=151 prefilter can bias against short clips; these are cohort rates, not exact whole-dataset rates.',
        four_to_five_seconds=sum(4<=min(r['source_duration'],r['target_duration'])<5 for r in cohort),
        task_counts=dict(Counter(r['instruction_count'] for r in cohort)),model_judge_calls=0)
    return metadata,cohort,summary


def select():
    OUTPUT.mkdir(exist_ok=True)
    metadata,cohort,summary=measure();selected=[];used=set()
    def take(target,side,n,count):
        candidates=[r for r in cohort if r['instruction_count']==n and r['sample_id'] not in used
            and (abs(r['ratio']-1)<1e-9 if side=='equal' else r['ratio']<=target+1e-9 if side=='below' else r['ratio']>target+1e-9)]
        candidates.sort(key=lambda r:(abs(r['ratio']-target),hashlib.sha256(f"{SEED}\0{target}\0{side}\0{r['sample_id']}".encode()).hexdigest()))
        result=[]
        for row in candidates:
            m=metadata[row['sample_id']]
            if not all(Path(m[role+'_local']).is_file() for role in ('source','target')):continue
            result.append(dict(**m,**{k:v for k,v in row.items() if k not in m},
                comparison_group=f'{target:g}x',boundary_side=side,boundary_threshold=target))
            used.add(row['sample_id'])
            if len(result)==count:break
        assert len(result)==count,(target,side,n)
        selected.extend(result)
    for n in (2,3):take(1.0,'equal',n,3)
    for target in (1.1,1.25,1.5,2.0):
        for side in ('below','above'):
            for n in (2,3):take(target,side,n,2 if (side=='below' and n==2) or (side=='above' and n==3) else 1)
    selected.sort(key=lambda r:(r['boundary_threshold'],r['ratio'],r['sample_id']))
    for rank,row in enumerate(selected,1):
        row['sample_rank']=rank
        row['branch']=f"{row['boundary_threshold']:g} 倍组 · "+{'equal':'等长','below':'阈值内','above':'阈值外'}[row['boundary_side']]
    manifest=OUTPUT/'sample_manifest.csv'
    if manifest.exists():assert [r['sample_id'] for r in shared.read_csv(manifest)]==[r['sample_id'] for r in selected]
    else:csv_write(manifest,selected)
    csv_write(OUTPUT/'cohort_measurements.csv',cohort)
    csv_write(OUTPUT/'threshold_rates.csv',summary['thresholds'])
    summary.update(sample_size=30,group_counts=dict(Counter(r['comparison_group'] for r in selected)),
        sample_task_counts=dict(Counter(r['instruction_count'] for r in selected)),
        sampling='6 pairs each at 1/1.1/1.25/1.5/2; each non-equal group has 3 below and 3 above cutoff, and 3 two-task/3 three-task pairs; no visual-quality replacement',
        sample_manifest_sha256=shared.digest(manifest),selection_ids=[r['sample_id'] for r in selected])
    shared.write_json(OUTPUT/'sampling_summary.json',summary)
    print(json.dumps(dict(population=summary['population'],thresholds=summary['thresholds'],groups=summary['group_counts'],equal_tolerant=summary['equal_frames_one_frame_time_tolerance']),ensure_ascii=False),flush=True)


def report_markdown(summary):
    lines=['# CoinVE 时长/帧数倍数阈值：30 对边界审核','',
        '201 帧的容器时长≈帧数/FPS：24fps 为 8.375 秒、25fps 为 8.040 秒、29.97fps 为 6.707 秒、30fps 为 6.700 秒。最后一帧时间戳约为200/FPS，与容器时长相差一帧。',
        '', '## 比例口径','',
        '帧数比=max(两侧帧数)/min(两侧帧数)，时长比=max(两侧时长)/min(两侧时长)，综合倍数 R=max(帧数比,时长比)。严格规则保留 R≤阈值（浮点比较容差1e-9）。统计来自既有完整解码探测记录，并非按审核包30条估计。时长使用该记录的帧数/平均FPS；本次容器读取的细小差异另存 container_duration/container_fps，不覆盖统计输入值。视频面板展示本次容器规格，极小读数差不等于动作错位。',
        '', '## 各阈值的筛除比例','',
        f"分母为本地 {summary['probed_pairs']:,} 对已探测数据中，满足无 Stylize、2–3 个子任务、source/target 均至少4秒和基本技术条件的 {summary['population']:,} 对。",'',
        '| 最大倍数 | 保留 | 筛掉 | 筛除比例 |','|---|---:|---:|---:|']
    for row in summary['thresholds']:
        lines.append(f"| {row['threshold']:g}× | {row['retained']:,} | {row['rejected']:,} | {row['rejected_percent']:.2f}% |")
    q=summary['equal_frames_one_frame_time_tolerance']
    lines+=['',f"严格1×对23.976/23.98之类的帧率记录差异敏感。若按“帧数相同，时长差不超过两侧较慢FPS的一帧时长”定义等长，可保留 {q['retained']:,} 对、筛掉 {q['rejected']:,} 对，筛除率 {q['rejected_percent']:.2f}%。此行只是额外对照，不改变上面的严格规则。",'',
        '这些是本地候选样本的统计，不是20万全库的精确比例。旧候选曾做 source 文件名帧跨度≥151 的预筛，可能偏向较长片段；正式扩量到4秒时不能继续沿用该预筛。',
        '', '## 30对如何选择','',
        '1×、1.1×、1.25×、1.5×、2×五档各6对；非等长四组均取阈值内3对和阈值外3对。每档含3个两任务、3个三任务样本；内侧为2个两任务+1个三任务，外侧相反。按到阈值的距离选最近样本，距离相同用固定seed哈希排序，30个ID不重复。不按人物、画质或编辑效果换样。',
        '', '## 如何看','',
        'index.html 逐对显示完整 prompt、帧数、时长、差值、两个比值以及各阈值的保留/排除判断。联动播放只比较从0开始的共同区间；完整 source 和 target 均保留，无变速、补帧或裁短。长source无对应target的尾段请用“独立查看完整视频”链接查看。',
        '', '长度比不是逐帧动作对齐或编辑质量的证明。包中“阈值内/外”只是技术长度判断，人工审核默认1、reviewed=0，未确认不算通过。没有冻结正式筛选门槛，也没有运行质量judge。']
    return '\n'.join(lines)+'\n'


def build():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    rows=shared.read_csv(OUTPUT/'sample_manifest.csv');summary=json.loads((OUTPUT/'sampling_summary.json').read_text())
    BUNDLE.mkdir(exist_ok=True);(BUNDLE/'media').mkdir(exist_ok=True);(BUNDLE/'browser_validation.json').unlink(missing_ok=True)
    media.BUNDLE=BUNDLE
    jobs=[(r,role,Path(r[role+'_local']).parent) for r in rows for role in ('source','target')]
    with ThreadPoolExecutor(max_workers=6) as pool:inventory=list(pool.map(media.prepare_video,jobs))
    selected_ids={r['sample_id'] for r in rows}
    for cached in (BUNDLE/'media').iterdir():
        if cached.name.split('_')[0] not in selected_ids and cached.suffix in ('.mp4','.json'):
            cached.unlink()  # Superseded, rebuildable reviewer copies only.
    by_role={(r['sample_id'],r['role']):r for r in inventory}
    items=[];review=[]
    for row in rows:
        entry=dict(row)
        ratio_info={k:float(row[k]) for k in ('ratio','frame_ratio','duration_ratio','duration_difference','frame_difference','boundary_threshold')}
        item=dict(sample_id=row['sample_id'],branch=row['branch'],comparison_group=row['comparison_group'],
            boundary_side=row['boundary_side'],ratio_info=ratio_info,instructions=json.loads(row['instructions_json']),
            operations=json.loads(row['operations_json']),objects=json.loads(row['objects_json']))
        for role in ('source','target'):
            m=by_role[(row['sample_id'],role)]
            assert m['original_info']['frames']==int(row[role+'_frames'])
            assert abs(m['original_info']['duration']-float(row[role+'_duration']))<1/m['original_info']['fps']
            item[role+'_media']=m['review_media'];item[role+'_info']=m['review_info']
            entry[role+'_media']=m['review_media'];entry[role+'_original_sha256']=m['original_sha256'];entry[role+'_review_sha256']=m['review_sha256']
            entry[role+'_container_duration']=m['original_info']['duration']
            entry[role+'_container_fps']=m['original_info']['fps']
        items.append(item);review.append(entry)
    shared.write_json(BUNDLE/'media_inventory.json',inventory);csv_write(BUNDLE/'review_manifest.csv',review)
    if not (BUNDLE/'audit.csv').exists():
        csv_write(BUNDLE/'audit.csv',[dict(sample_id=r['sample_id'],branch=r['branch'],human_decision=1,reviewed=0,notes='') for r in rows])
    for name in ('sample_manifest.csv','cohort_measurements.csv','threshold_rates.csv'):
        shutil.copyfile(OUTPUT/name,BUNDLE/name)
    payload=dict(survey_version=NAME,repo_id=summary['repo_id'],revision=summary['revision'],items=items)
    template=(ROOT/'random_review_template.html').read_text().replace('50 对随机审核','30 对长度倍数审核').replace('max="50"','max="30"')
    start='从不含 Stylize 的完整 metadata 中，排除此前已看过的 10 条范围内样本，以 seed 20260906 无放回均匀随机抽取 50 对。未按人物、画质、编辑效果或视频时长筛选，未根据内容换样。'
    template=template.replace(start,'从本地无 Stylize、2–3任务、两侧至少4秒的2693对中，取1/1.1/1.25/1.5/2倍五档各6对；非等长组含阈值内外各3对。倍数取帧数比、时长比的较大值；这里只抽边界案例，不按30条计算筛除率。完整统计见 REPORT.md。')
    template=template.replace('<span class="tag" id="taskCount"></span>','<span class="tag" id="taskCount"></span><span class="tag" id="ratioGroup"></span>')
    table='<section class="panel"><strong>201帧：24fps≈8.38秒，25fps≈8.04秒，30fps≈6.70秒</strong><details><summary>展开各阈值筛除比例（本地2693对统计）</summary><table style="border-spacing:16px 4px"><tr><th>上限</th><th>保留</th><th>筛掉</th><th>筛除率</th></tr>'
    for stat in summary['thresholds']:
        table+=f"<tr><td>{stat['threshold']:g}×</td><td>{stat['retained']}</td><td>{stat['rejected']}</td><td>{stat['rejected_percent']:.2f}%</td></tr>"
    table+='</table><p>严格1×会计入细小帧率记录差异；等帧数并容许一帧时差时，筛除率为57.00%。这些是旧预筛候选样本统计，不是全库精确比例。<a href="REPORT.md" target="_blank">查看完整报告</a></p></details><label>跳到对比组：<select id="groupJump"><option value="0">等长（6对）</option><option value="6">1.1倍附近（6对）</option><option value="12">1.25倍附近（6对）</option><option value="18">1.5倍附近（6对）</option><option value="24">2倍附近（6对）</option></select></label></section>'
    template=template.replace('<main>','<main>'+table)
    template=template.replace('<ol id="instructions">','<p id="ratioDescription"></p><p class="muted" id="thresholdDecisions"></p><strong>完整编辑 prompt</strong><ol id="instructions">')
    template=template.replace("$('taskCount').textContent=x.instructions.length+' 个编辑任务';", "$('taskCount').textContent=x.instructions.length+' 个编辑任务';$('ratioGroup').textContent=x.branch;const r=x.ratio_info;$('ratioDescription').textContent=`帧数比 ${r.frame_ratio.toFixed(6)}× · 时长比 ${r.duration_ratio.toFixed(6)}× · 综合 ${r.ratio.toFixed(6)}×；相差 ${r.frame_difference} 帧、${r.duration_difference.toFixed(3)} 秒`;$('thresholdDecisions').textContent='按长度规则：'+[1,1.1,1.25,1.5,2].map(t=>`${t}× ${r.ratio<=t+1e-9?'保留':'排除'}`).join(' · ');")
    template=template.replace("$('timeline').addEventListener", "$('groupJump').onchange=()=>{index=Number($('groupJump').value);save();render();};$('timeline').addEventListener")
    text=template+json.dumps(payload,ensure_ascii=False)
    font_path=PROJECT/'dataset_research/ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    font=TTFont(font_path);sub=subset.Subsetter();sub.populate(text=text);sub.subset(font);font.flavor='woff';font.save(OUTPUT/'review_font.woff')
    assert {ord(c) for c in text if '\u4e00'<=c<='\u9fff'}<=set(font.getBestCmap())
    document=template.replace('__SURVEY_DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')).replace('__CJK_FONT__',base64.b64encode((OUTPUT/'review_font.woff').read_bytes()).decode())
    (BUNDLE/'index.html').write_text(document);shutil.copyfile(font_path.parent/'LICENSE.txt',BUNDLE/'FONT_LICENSE.txt')
    report=report_markdown(summary);(BUNDLE/'REPORT.md').write_text(report)
    (BUNDLE/'README.md').write_text('# CoinVE 30对长度倍数审核\n\n完整解压后用 Chrome/Edge 打开 index.html，无需联网。完整prompt显示在视频上方。按1/1.1/1.25/1.5/2倍五档各6对浏览，非等长组含阈值内外各3对。可用顶部菜单跳到分组。\n\n视频保持完整分辨率、帧率、时长，浏览器副本H.264/yuv420p/faststart、无音轨。同步播放共同区间，较长一侧完整尾段可从“独立查看完整视频”查看，未变速或padding。\n\n1通过、0不通过、2存疑，默认1且reviewed=0不是人工通过。支持CSV/JSON导入导出。倍数只描述技术长度差，不自动判定编辑质量。\n\n'+report+'\n来源、原始哈希及转码记录见sample_manifest.csv、review_manifest.csv、media_inventory.json；完整2693对长度测量见cohort_measurements.csv，筛除率见threshold_rates.csv。原始媒体沿用旧池，只新增审核副本。\n')
    summary.update(full_decode_videos=60,review_video_bytes=sum(r['review_bytes'] for r in inventory),browser_validation='pending',builder_sha256=shared.digest(Path(__file__)))
    shared.write_json(BUNDLE/'summary.json',summary)
    print('built',BUNDLE,flush=True)


def package():
    check=json.loads((BUNDLE/'browser_validation.json').read_text())
    assert check['status']=='passed' and check['sample_count']==30 and check['index_sha256']==shared.digest(BUNDLE/'index.html')
    records=json.loads((BUNDLE/'media_inventory.json').read_text())
    assert len(records)==60
    for r in records:assert shared.digest(BUNDLE/r['review_media'])==r['review_sha256']
    summary=json.loads((BUNDLE/'summary.json').read_text());summary['browser_validation']='passed';shared.write_json(BUNDLE/'summary.json',summary)
    files=[p for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and not(p.parent.name=='media' and p.suffix=='.json')]
    (BUNDLE/'SHA256SUMS').write_text(''.join(f'{shared.digest(p)}  {p.relative_to(BUNDLE)}\n' for p in files));files.append(BUNDLE/'SHA256SUMS')
    archive=OUTPUT/(NAME+'.zip');assert not archive.with_suffix('.baidu_upload.json').exists()
    temp=archive.with_suffix('.part.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in files:z.write(p,Path(NAME)/p.relative_to(BUNDLE))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None and sum(n.endswith('.mp4') for n in z.namelist())==60
    temp.replace(archive)
    receipt=dict(path=str(archive.resolve()),bytes=archive.stat().st_size,sha256=shared.digest(archive),pair_count=30,media_count=60,
        zip_crc='passed',browser_validation='passed',remote_path='/apps/bypy/lance_review/'+archive.name)
    shared.write_json(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt),flush=True)


def upload():
    import build_balanced_review as delivery
    delivery.OUTPUT=OUTPUT;delivery.BUNDLE=BUNDLE;delivery.VERSION=NAME;delivery.upload()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['select','build','package','upload'])
    args=parser.parse_args();globals()[args.phase]()
