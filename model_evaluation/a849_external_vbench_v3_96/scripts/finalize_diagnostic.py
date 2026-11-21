"""CPU: keep identity diagnostic metrics and delivery separate from the main cohort."""
import csv,subprocess,sys,zipfile
from common import *
from contracts import DIMS,LABELS,MODELS,identity_decision
from reuse import link
from score import DIAG

def main():
 for name in ['per_video_metrics.csv','per_task_report.csv']:
  with (ROOT/'metadata'/name).open() as f:
   reader=csv.DictReader(f);fieldnames=reader.fieldnames;records=[r for r in reader if r['cohort']=='identity_diagnostic']
  with (DIAG/'metadata'/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(records)
 groups=read(ROOT/'metadata/group_metrics.json')['groups']['identity_diagnostic'];paired=read(ROOT/'metadata/temporal_flickering_static_paired.json')['identity_diagnostic'];visual=read(DIAG/'metadata/identity_review.json')
 valid=sum(r['score']!='' and r['status']=='complete' for r in rows_csv(DIAG/'metadata/per_video_metrics.csv'))
 atomic(DIAG/'metadata/group_metrics.json',dict(groups=groups,valid_scores=valid,expected_scores=72))
 atomic(DIAG/'metadata/temporal_flickering_static_paired.json',paired)
 decision=identity_decision(visual['tasks'])
 failed_ids=[r['id'] for r in visual['tasks'] if all(r['models'][m]['clear_identity_transfer'] is False for m in MODELS)]
 assert visual['inspection_complete'] and len(visual['tasks'])==4
 for task in visual['tasks']:
  for model in MODELS:
   observed=task['models'][model];receipt=read(DIAG/'outputs'/model/f"{task['id']}.json")
   assert observed['reviewed'] and receipt['status']=='complete' and observed['output_sha256']==receipt['output_sha256']
 conclusion='按预设规则移出主测，四个名额补为Hair color。' if decision['drop_identity'] else '未达到预设移出条件，主测保留四条Identity和十二条Hair color。'
 lines=['# Identity v3 独立诊断','',f"十二条输出已直接审核；有效质量指标{valid}/72。三个模型均未呈现清晰身份迁移的任务为：{', '.join(failed_ids) or '无'}。{conclusion}全部结果保留。",'', '本轮采用完整头部、保留自然头发的1024×1024 CelebA-HQ灰底参考。初轮02–04参考裁顶，九条旧输出仅保留为历史构图诊断，不用于正式身份去留。01参考合规且有效输入完全相同，其三条输出按SHA原样复用。', '', 'Temporal Flickering为官方全量原始值，受正常运动影响；不直接解释为编辑质量改善。','', '|指标 ↑|基座|A849|VACE|','|---|---:|---:|---:|']
 def fmt(v):return '缺失' if v is None else f'{v:.5f}'
 for dim,label in zip(DIMS,LABELS):lines.append('|'+label+'|'+ '|'.join(fmt(groups['all'][dim]['models'][m]['mean']) for m in MODELS)+'|')
 lines+=['',f"静态配对子集状态：{paired['status']}；n={paired['n']}；ID：{', '.join(paired['task_ids']) or '无/待定'}。",'', '逐任务编辑观察见identity_review.json，六项联表见per_task_report.csv。自建任务无GT；参考为保留自然头发的1024×1024 CelebA-HQ灰底照片。']
 (DIAG/'metadata/report.md').write_text('\n'.join(lines)+'\n')
 for sid in ['identity_01','identity_02']:
  c=read(DIAG/'conditions'/sid/'condition.json');link(c['overlay_video'],DIAG/'conditions'/sid/'overlay.mp4')
 subprocess.run([sys.executable,str(DIAG/'scripts/review.py'),'--stage','all','--pack'],check=True)
 package=read(DIAG/'metadata/package.json')
 with zipfile.ZipFile(package['path']) as archive:assert archive.testzip() is None
 atomic(DIAG/'metadata/final_validation.json',dict(status='passed' if valid==72 and paired['filter_complete'] else 'partial',valid_scores=valid,expected_scores=72,complete_generations=12,visual_inspection_complete=visual['inspection_complete'],identity_decision=decision,all_three_failed_task_ids=failed_ids,package_crc='passed'))
 print('Identity diagnostic quality scores',valid,'/72',flush=True)

def rows_csv(path):
 with path.open() as f:return list(csv.DictReader(f))
if __name__=='__main__':main()
