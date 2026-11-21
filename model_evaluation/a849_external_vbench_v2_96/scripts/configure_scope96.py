from pathlib import Path
import ast,os
assert os.environ['SLURM_JOB_PARTITION']=='testqueue'
p=Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96/scripts')
f=p/'aggregate.py';s=f.read_text().replace('# A849 外部视频编辑评测 v1','# A849 外部视频编辑评测：96条扩展版').replace('24条单种子','96条任务单种子').replace('OmniEdit官方6条保留原指令；其他12条为自建参考图任务','OmniEdit原生24条保留原指令，含外观/服装/颜色及5条表情任务；48条为自建参考图任务')
s=s.replace("    visual=ROOT/'metadata/visual_observations.json'", "    lines+=['身份与发型参考均去除背景至RGB128；保持完整原始画幅和分辨率。身份图仅头发填RGB255，其他人物前景像素保持原值；发型图保留真实发色。原24条历史输出只复用16个输入完整相同的任务，共48条，身份/发型组全部重新生成。','','96条任务对应83段不同source；同一视频可用于不同编辑任务，详见source_clusters.json，不当作96段独立视频。','','本轮官方Ingredients IC-LoRA只在后续队列，未纳入这三模型288条结果。','']\n    visual=ROOT/'metadata/visual_observations.json'")
f.write_text(s)
f=p/'review.py';s=f.read_text().replace('background:#26323e;padding:12px;display:flex','background:#26323e;padding:12px;max-height:125px;overflow-y:auto;display:flex').replace('metadata/before_whitehair/inputs.jsonl','metadata/prior24/inputs.jsonl')
s=s.replace("    if stage=='smoke':", "    if frozen.get('revision')=='background_removed_fullframe_96_v3':\n        page=page.replace('<h1>', '<p>96条任务、83段不同source；身份和发型参考均去背景，完整原图不裁剪，身份头发填白。仅复用16条输入相同的任务；Ingredients仍在待办队列。</p><h1>',1)\n    if stage=='smoke':",1)
s=s.replace("'visual_observations.json']", "'visual_observations.json','source_clusters.json']")
f.write_text(s)
for f in p.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
print('96-task verification/report/review scope updated; syntax passed')
