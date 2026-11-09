#!/usr/bin/env python3
"""Build the task-balanced CoinVE expansion survey, preserving official pairs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import build_random_review as fetch
import package_random_review as package
from sample_random_pairs import EXPECTED_METADATA_SHA256, shard_and_member

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
VERSION = "coinve_balanced50_20261108"
OUTPUT = ROOT / VERSION
BUNDLE = OUTPUT / VERSION
ORIGINALS = OUTPUT / "originals"
OLD_POOL = PROJECT / "lance_general_edit_finetune/3_lance_lora_finetuning/artifacts/coinve_ablation_v1/coinve_4k_pool.csv"
SEED = 20261108
GROUPS = {
    "person_attribute": "人物局部属性",
    "whole_subject": "主体整体编辑",
    "object": "物体编辑",
    "background": "背景编辑",
    "stylize": "风格化",
}
ATTRIBUTE = re.compile(r"\b(hair\w*|beard\w*|mustache|moustache|bald|shirt\w*|t-shirt\w*|blouse|dress\w*|jacket\w*|coat\w*|sweater\w*|hoodie\w*|vest\w*|suit\w*|outfit\w*|cloth\w*|pants|trousers|jeans|skirt\w*|shorts|uniform\w*|apron\w*|shoe\w*|boot\w*|sneaker\w*|hat\w*|cap|beanie|helmet\w*|glasses|sunglasses|spectacles|headphone\w*|earring\w*|necklace\w*|bracelet\w*|ring|watch|scarf|tie|mask|lipstick|makeup|face|facial)\b", re.I)
WHOLE = re.compile(r"^(?:add|remove|replace|insert|delete|change)\s+(?:(?:a|an|the|all|two|three|small|large|young|old|adult|little|standing|sitting|walking|running|existing|main)\s+)*(?:man|woman|boy|girl|person|people|child|children|baby|male|female|dog|cat|bird|horse|cow|sheep|lion|tiger|bear|animal|poodle|puppy|kitten)\b", re.I)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify(text, operation, obj):
    """Describe the requested target, not the owner mentioned later in a prompt."""
    if "styl" in operation.lower():
        return "stylize"
    target = re.split(r"\b(?:with|wearing|worn by|held by|holding|behind|next to|standing|sitting|located|to|into|at|on|from|in)\b", text, maxsplit=1, flags=re.I)[0]
    if re.search(r'\b(paper mache|taxidermy|puppet|stuffed|statue|figurine|toy)\b',target,re.I):
        return "object"
    if re.search(r"\b(painting|photograph|clock|sign|panels|ukulele|microfiber cloth)\b",target,re.I):
        return "object"
    # Background comes before attribute words such as 'background with clothes'.
    if re.search(r"\b(background|backdrop|wall|floor|ceiling|sky|ground|road|landscape|scenery)\b", target, re.I):
        return "background"
    if ATTRIBUTE.search(target) or re.search(r"\b(top|cardigan|goatee|headscarf|tattoo\w*|boutonniere|blazer|puffer|fedora)\b",target,re.I):
        return "person_attribute"
    if WHOLE.search(text.strip()) or re.search(r"^(?:add|remove|replace)\b.*\b(cat|dog|retriever|poodle|seagull|donkey|horse|bird|puppy|kitten)\b",target,re.I):
        return "whole_subject"
    if re.search(r"\b(cabinet\w*|refrigerator|curtain|drapery|house|building\w*|forest|mountain\w*|lawn|fence|interior|countertop|backsplash|wallpaper|skyline|bookshelf|door|window|structure|station)\b",target,re.I):
        return "background"
    return "object"


def candidates():
    metadata = ROOT / "metadata/metadata_coinve200k.jsonl"
    assert fetch.digest(metadata) == EXPECTED_METADATA_SHA256
    old = fetch.read_csv(OLD_POOL)
    assert len(old) == 4000
    excluded = {r[k] for r in old for k in ("source_dataset_path", "target_dataset_path")}
    by_group = defaultdict(list)
    with metadata.open() as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            if any(item[k] in excluded for k in ("source_video_path", "edited_video_path")):
                continue
            frame = re.search(r"_(\d+)to(\d+)(?:_[0-9a-f]+)?\.mp4$", item["source_video_path"])
            if not frame or int(frame[2])-int(frame[1])+1 < 151:
                continue
            ops = item["instruction_operation"]
            anchor = next((o.lower() for o in ops if "styl" not in o.lower()), None)
            if anchor not in ("add", "remove", "replace"):
                continue
            ins, objs = item["instruction"], item["instruction_object"]
            assert len(ins) == len(ops) == len(objs)
            tags = [classify(t, o, ob) for t, o, ob in zip(ins, ops, objs)]
            row = dict(survey_version=VERSION, sample_rank=0, sample_id=f"coinve-{index:06d}",
                       metadata_index=index, dataset="CoinVE-200K", branch="", display_group="",
                       anchor_operation=anchor, repo_id=fetch.REPO_ID, revision=fetch.REVISION,
                       instruction_count=len(ins), has_stylize=int("stylize" in tags),
                       instruction="\n".join(f"{i}. {s}" for i,s in enumerate(ins,1)),
                       instructions_json=json.dumps(ins,ensure_ascii=False),
                       operations_json=json.dumps(ops), objects_json=json.dumps(objs),
                       task_groups_json=json.dumps(tags),
                       instruction_masks_json=json.dumps(item["instruction_mask_video_paths"]),
                       combined_mask_path=item["combined_mask_video_path"])
            for role,key in (("source","source_video_path"),("target","edited_video_path")):
                row[role+"_path"] = item[key]
                row[role+"_shard"],row[role+"_member"] = shard_and_member(item[key])
                row[role+"_original"] = f"originals/{row['sample_id']}_{role}.mp4"
            for group in set(tags):
                key = hashlib.sha256(f"{SEED}\0{group}\0{index}".encode()).hexdigest()
                by_group[group].append((key,row))
    for group in by_group:
        by_group[group].sort(key=lambda v:v[0])
    return by_group,excluded


def select(preview=False):
    groups,used = candidates()
    rejected_path = OUTPUT / "technical_replacements.json"
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else {}
    selected = {}
    # A pair may have several task types. One display bucket only; all tags survive.
    for group in ("whole_subject","person_attribute","object","background","stylize"):
        values=[]
        for _,row in groups[group]:
            if row["sample_id"] in rejected or row["source_path"] in used or row["target_path"] in used:
                continue
            used.update((row["source_path"],row["target_path"]))
            values.append({**row,"branch":GROUPS[group],"display_group":group})
            if len(values)==10:
                break
        assert len(values)==10,group
        selected[group]=values
    rows=[r for group in GROUPS for r in selected[group]]
    for rank,row in enumerate(rows,1):
        row["sample_rank"]=rank
    if preview:
        for r in rows:
            print(r["sample_id"],r["display_group"],r["task_groups_json"],r["instruction"].replace("\n"," | "))
        return
    OUTPUT.mkdir(exist_ok=True)
    path=OUTPUT/"sample_manifest.csv"
    if path.exists():
        previous=fetch.read_csv(path)
        assert [r['sample_id'] for r in previous]==[r['sample_id'] for r in rows], "frozen selection differs; archive technical-failure manifest first"
    else:
        save_csv(path,rows)
    summary=dict(survey_version=VERSION, repo_id=fetch.REPO_ID, revision=fetch.REVISION,
        metadata_sha256=EXPECTED_METADATA_SHA256, old_pool_sha256=fetch.digest(OLD_POOL), seed=SEED,
        sampling="five task display groups x 10; stable SHA256 order within group, unique source/target; retain every ordered subinstruction",
        task_label_policy="instruction target text rules, manually inspected; labels describe intent, not visual success",
        candidate_memberships={g:len(v) for g,v in groups.items()}, display_groups={g:10 for g in GROUPS},
        excluded_old_pool_pairs=4000, source_filename_prefilter="frame span >=151", common_duration_min_seconds=5,
        quality_filter="technical only; no visual quality replacement", model_judge_calls=0,
        stylize_pairs=sum(r['has_stylize'] for r in rows), sample_size=50,
        instruction_count=dict(Counter(str(r['instruction_count']) for r in rows)),
        atomic_operations=dict(Counter(o for r in rows for o in json.loads(r['operations_json']))),
        atomic_task_groups=dict(Counter(t for r in rows for t in json.loads(r['task_groups_json']))),
        pairs_by_task_group=dict(Counter(t for r in rows for t in set(json.loads(r['task_groups_json'])))),
        sample_manifest_sha256=fetch.digest(path),selection_ids=[r['sample_id'] for r in rows],
        distribution_warning="Balanced demonstration, not an estimate of natural population proportions")
    fetch.write_json(OUTPUT/"sampling_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def configure():
    fetch.OUTPUT=OUTPUT
    package.OUTPUT=OUTPUT
    package.BUNDLE=BUNDLE
    package.VERSION=VERSION
    package.ROOT=OUTPUT


def probe():
    rows=fetch.read_csv(OUTPUT/'sample_manifest.csv')
    def one(row):
        try:
            infos={role:package.probe(ORIGINALS/f"{row['sample_id']}_{role}.mp4") for role in ('source','target')}
        except Exception as error:
            return dict(sample_id=row['sample_id'],status='reject',reasons=['probe_error:'+type(error).__name__])
        reasons=[]
        s,t=infos['source'],infos['target']
        if min(s['duration'],t['duration'])+1e-6<5: reasons.append('common_duration_under_5s')
        if (s['width'],s['height'])!=(t['width'],t['height']): reasons.append('resolution_mismatch')
        if abs(s['fps']-t['fps'])>max(.1,.01*s['fps']): reasons.append('fps_mismatch')
        return dict(sample_id=row['sample_id'],status='reject' if reasons else 'pass',reasons=reasons,**infos)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(one,rows))
    fetch.write_json(OUTPUT/'technical_probe.json',results)
    failed={r['sample_id']:r['reasons'] for r in results if r['status']!='pass'}
    if failed:
        fetch.write_json(OUTPUT/'technical_failures.json',failed)
        raise SystemExit(f"Technical failures require recorded same-group replacements: {failed}")
    print('50 pair technical checks passed',flush=True)


def repair():
    """Record only objective technical failures, then advance stable group order."""
    failures=json.loads((OUTPUT/'technical_failures.json').read_text())
    assert failures
    path=OUTPUT/'technical_replacements.json'
    rejected=json.loads(path.read_text()) if path.exists() else {}
    for sample_id,reasons in failures.items():
        assert reasons and all(reason in ('common_duration_under_5s','resolution_mismatch','fps_mismatch') for reason in reasons), 'investigate transfer/probe errors before replacing'
        rejected[sample_id]=reasons
    fetch.write_json(path,rejected)
    history=OUTPUT/f'sample_manifest_before_repair{len(rejected):02d}.csv'
    (OUTPUT/'sample_manifest.csv').rename(history)
    select()


def annotate():
    path=OUTPUT/'sample_manifest.csv'
    rows=fetch.read_csv(path)
    for row in rows:
        labels=[]
        for text,group in zip(json.loads(row['instructions_json']),json.loads(row['task_groups_json'])):
            target=re.split(r'\b(?:with|wearing|worn by|behind|to|at|on|from|in)\b',text,maxsplit=1,flags=re.I)[0]
            if group!='person_attribute':label=GROUPS[group]
            elif re.search(r'\b(hair\w*|beard|goatee|mustache|moustache|bald)\b',target,re.I):label='头发与胡须'
            elif re.search(r'\b(glasses|sunglasses|headphone\w*|necklace|earring\w*|bracelet|ring|watch|hat|cap|fedora|scarf|headscarf|tie|boutonniere)\b',target,re.I):label='配饰'
            elif re.search(r'\b(shirt|t-shirt|blouse|dress|jacket|coat|sweater|hoodie|vest|suit|outfit|clothing|pants|trousers|jeans|skirt|shorts|uniform|apron|top|cardigan|blazer)\b',target,re.I):label='服装'
            else:label='其他人物局部属性'
            labels.append(label)
        row['task_types_json']=json.dumps(labels,ensure_ascii=False)
    save_csv(path,rows)
    summary=json.loads((OUTPUT/'sampling_summary.json').read_text())
    summary['atomic_task_types']=dict(Counter(t for r in rows for t in json.loads(r['task_types_json'])))
    summary['pairs_by_task_type']=dict(Counter(t for r in rows for t in set(json.loads(r['task_types_json']))))
    summary['sample_manifest_sha256']=fetch.digest(path)
    summary['builder_sha256']=fetch.digest(Path(__file__))
    fetch.write_json(OUTPUT/'sampling_summary.json',summary)


def build():
    configure()
    annotate()
    (BUNDLE/'browser_validation.json').unlink(missing_ok=True)
    template=(ROOT/'random_review_template.html').read_text()
    template=template.replace('50 对随机审核','50 对任务均衡审核')
    start='从不含 Stylize 的完整 metadata 中，排除此前已看过的 10 条范围内样本，以 seed 20260906 无放回均匀随机抽取 50 对。未按人物、画质、编辑效果或视频时长筛选，未根据内容换样。'
    template=template.replace(start,'五类任务各展示 10 对，保留 Stylize 和全部复合指令；排除旧 CoinVE-4K 的共享媒体，seed 20261108。共同可用时长至少 5 秒。不按编辑效果换样。这是均衡展示，不能据此估计全库占比。')
    template=template.replace('<span class="tag" id="taskCount"></span>','<span class="tag" id="taskCount"></span><span class="tag" id="taskGroup"></span>')
    template=template.replace("$('sampleId').textContent=x.sample_id;","$('sampleId').textContent=x.sample_id;$('taskGroup').textContent='展示组：'+x.branch;")
    (OUTPUT/'random_review_template.html').write_text(template)
    # The historical font was subset for a different page and lacks new labels.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font_path=PROJECT/'dataset_research/ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    font=TTFont(font_path)
    subsetter=subset.Subsetter()
    text=template+json.dumps(fetch.read_csv(OUTPUT/'sample_manifest.csv'),ensure_ascii=False)
    subsetter.populate(text=text)
    subsetter.subset(font)
    font.flavor='woff'
    font.save(OUTPUT/'review_font.woff')
    assert {ord(c) for c in text if '\u4e00'<=c<='\u9fff'} <= set(font.getBestCmap())
    shutil.copyfile(font_path.parent/'LICENSE.txt',OUTPUT/'font_LICENSE.txt')
    package.build(ORIGINALS,4)
    # Include requested-target labels for each subinstruction in the page and CSV.
    path=BUNDLE/'index.html'
    document=path.read_text()
    document=document.replace("x.operations[i]+' · '+x.objects[i]", "x.operations[i]+' · '+x.objects[i]+' · '+x.task_labels[i]")
    payload_match=re.search(r'(<script id="surveyData" type="application/json">)(.*?)(</script>)',document,re.S)
    payload=json.loads(payload_match[2])
    rows=fetch.read_csv(OUTPUT/'sample_manifest.csv')
    selected_ids={r['sample_id'] for r in rows}
    for cached in (BUNDLE/'media').iterdir():
        match=re.match(r'(coinve-\d+)_',cached.name)
        if match and match[1] not in selected_ids and cached.suffix in ('.mp4','.json'):
            cached.unlink()  # Disposable reviewer copy; original and rejection record stay.
    for item,row in zip(payload['items'],rows):
        item['task_labels']=json.loads(row['task_types_json'])
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    document=document[:payload_match.start(2)]+encoded+document[payload_match.end(2):]
    path.write_text(document)
    summary=json.loads((OUTPUT/'sampling_summary.json').read_text())
    report=['# CoinVE 50 对任务均衡摸底','',
        '五类各 10 对；一对可涉及多个任务类型。以下仅描述本展示集，不代表全库自然分布。', '',
        '## 每对的子任务条数','', '| 子任务数 | pair 数 |','|---|---:|']
    for n,count in sorted(summary['instruction_count'].items()): report.append(f'| {n} | {count} |')
    total=sum(int(k)*v for k,v in summary['instruction_count'].items())
    report += ['',f'合计 {total} 条子任务，平均每对 {total/50:.2f} 条；含风格化 {summary["stylize_pairs"]} 对。','',
        '## 子任务操作','', '| 操作 | 子任务数 |','|---|---:|']
    for op,count in summary['atomic_operations'].items():report.append(f'| {op} | {count} |')
    report += ['','## 细任务分组','', '| 类型 | 涉及 pair 数 | 子任务数 |','|---|---:|---:|']
    for g,label in GROUPS.items():report.append(f'| {label} | {summary["pairs_by_task_group"].get(g,0)} | {summary["atomic_task_groups"].get(g,0)} |')
    report += ['','## 人物属性进一步细分及其他任务','', '| 类型 | 涉及 pair 数 | 子任务数 |','|---|---:|---:|']
    for label,count in summary['atomic_task_types'].items():
        report.append(f'| {label} | {summary["pairs_by_task_type"][label]} | {count} |')
    report += ['','## 逐条任务','']
    for row in rows:
        report += [f'### {row["sample_rank"]}. {row["sample_id"]} · {row["branch"]}', '',row['instruction'],'']
    (BUNDLE/'TASK_REPORT.md').write_text('\n'.join(report)+'\n')
    shutil.copyfile(OUTPUT/'technical_probe.json',BUNDLE/'technical_probe.json')
    if (OUTPUT/'technical_replacements.json').exists():
        shutil.copyfile(OUTPUT/'technical_replacements.json',BUNDLE/'technical_replacements.json')
    for name in ('technical_replacement_details.json','classification_corrections.json'):
        if (OUTPUT/name).exists():shutil.copyfile(OUTPUT/name,BUNDLE/name)
    (BUNDLE/'README.md').write_text(f'''# CoinVE 50 对任务均衡审核

完整解压 ZIP，用 Chrome / Edge 打开 index.html，无需服务器或联网。

- 五个展示组各 10 对，保留所有复合指令和 Stylize；同一对可包含多类任务。
- 这不是全库均匀随机样本，类别占比不能外推全库。任务统计及完整指令见 TASK_REPORT.md。
- 排除旧 CoinVE-4K 的 source/target 共享媒体，固定官方 revision、metadata SHA256 和 seed {SEED}。
- source 文件名跨度至少 151 帧，技术检查要求共同可用时长至少 5 秒、尺寸/帧率匹配。两侧完整片段均保留；同步只播放共同时间段。
- 完整分辨率和帧率保留，浏览器副本 H.264/yuv420p/faststart；无音轨。原始媒体保存在包外 originals/，清单保留映射和原始 SHA256。
- 空格同步播放，左右键翻页；1 接受、0 拒绝、2 存疑。支持备注及 CSV/JSON 导入导出。
- 默认 human_decision=1、reviewed=0；默认值不是人工通过。没有进行模型质量判定，也未因编辑效果差而换样。
- sample_manifest.csv 为来源和任务清单；review_manifest.csv 为浏览器媒体与原始 SHA256 映射；media_inventory.json 保存转码和完整解码记录。
- 本批用于决定后续新增 6k 的任务范围和质量门槛，不绑定 Lance 或 LTX2.5 的训练输入格式。
''')
    print('task report and balanced reviewer ready',flush=True)


def upload():
    script=PROJECT/'model_evaluation/ltx25_singleturn_v1/package_bench.py'
    spec=importlib.util.spec_from_file_location('coinve_delivery',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    side=OUTPUT/f'{VERSION}.sha256.json'
    receipt=json.loads(side.read_text())
    receipt['remote_path']='/apps/bypy/lance_review/'+VERSION+'.zip'
    fetch.write_json(side,receipt)
    module.ROOT=OUTPUT;module.PACKAGE=BUNDLE
    (OUTPUT/'review').mkdir(exist_ok=True)
    module.upload(receipt_name='baidu_delivery_receipt.json')
    helper_path=PROJECT/'dataset_research/recent_motion_review_20261030/deliver.py'
    helper_spec=importlib.util.spec_from_file_location('coinve_listing',helper_path)
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    receipt=json.loads((OUTPUT/'review/baidu_delivery_receipt.json').read_text())
    remote_side=str(Path(receipt['remote_path']).with_suffix('.sha256.json'))
    listing=helper.listing(helper.load_token(helper.CONFIG))
    side_rows=[r for r in listing if r['path']==remote_side]
    assert len(side_rows)==1 and side_rows[0]['size']==side.stat().st_size
    receipt.update(sidecar_remote_path=remote_side,sidecar_remote_fs_id=side_rows[0]['fs_id'],sidecar_remote_size_verified=True)
    fetch.write_json(OUTPUT/'baidu_delivery_receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['preview','select','download','probe','repair','build','package','upload'])
    args=p.parse_args()
    configure()
    if args.phase in ('preview','select'):select(args.phase=='preview')
    elif args.phase=='download':fetch.download(ORIGINALS,6)
    elif args.phase=='probe':probe()
    elif args.phase=='repair':repair()
    elif args.phase=='build':build()
    elif args.phase=='package':package.package()
    else:upload()
