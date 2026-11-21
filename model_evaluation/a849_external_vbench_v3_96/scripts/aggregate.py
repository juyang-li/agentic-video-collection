"""CPU six-metric tables with missingness and a source+three-model static intersection."""
import csv,statistics
from collections import Counter
from common import *
from contracts import DIMS,LABELS,MODELS,common_static
from score import inventory,DIAG

def mean(values):return statistics.mean(values) if values else None

def main():
 frozen=read(ROOT/'metadata/freeze.json');assert frozen['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 videos=inventory();records=[];flags={};errors=[]
 for r in videos:
  h=r['output_sha256']
  for dim in DIMS:
   p=ROOT/'metadata/vbench'/('raw' if dim=='temporal_flickering' else 'quality')/(f'{h}.json' if dim=='temporal_flickering' else f'{dim}--{h}.json')
   score=read(p) if h and p.exists() else {'status':'missing_score'}
   if score['status']=='complete':assert score['output_sha256']==h
   record={k:r[k] for k in ['cohort','id','group','subgroup','model','output_sha256']}
   record.update(dimension=dim,status='complete' if score['status']=='complete' else 'missing_generation' if not h else score['status'],raw_score=score.get('raw_score'),score=score.get('score'))
   records.append(record)
   if score['status'] not in ['complete','missing_score']:errors.append(score)
 def static_flag(path):
  if not path.exists():return None
  h=sha(path);p=ROOT/'metadata/vbench/static'/f'{h}.json'
  if not p.exists():return None
  r=read(p);assert r['media_sha256']==h
  return r.get('is_static') if r['status']=='complete' else None
 ids={};subsets={}
 for cohort,root in [('main',ROOT),('identity_diagnostic',DIAG)]:
  items=rows(root/'metadata/inputs.jsonl');ids[cohort]=[r['id'] for r in items];cohort_flags={}
  for item in items:
   sid=item['id'];cohort_flags[sid,'source']=static_flag(root/'inputs/prepared'/sid/'source.mp4')
   for model in MODELS:
    generation=next(r for r in videos if r['cohort']==cohort and r['id']==sid and r['model']==model)
    cohort_flags[sid,model]=static_flag(Path(generation['path'])) if generation['output_sha256'] else None
  eligible=common_static(ids[cohort],cohort_flags)
  common=[sid for sid in eligible if all(any(r['cohort']==cohort and r['id']==sid and r['model']==m and r['dimension']=='temporal_flickering' and r['score'] is not None for r in records) for m in MODELS)]
  filter_complete=all(v is not None for v in cohort_flags.values())
  subsets[cohort]=dict(task_ids=common,static_eligible_task_ids=eligible,excluded_missing_flicker_ids=[sid for sid in eligible if sid not in common],n=len(common) if filter_complete else None,status=('applicable' if common else 'not_applicable') if filter_complete else 'pending',filter_complete=filter_complete,filter_results=[{'id':sid,**{m:cohort_flags[sid,m] for m in ('source',*MODELS)}} for sid in ids[cohort]],models={m:dict(n=sum(r['cohort']==cohort and r['id'] in common and r['model']==m and r['dimension']=='temporal_flickering' and r['score'] is not None for r in records),mean=mean([r['score'] for r in records if r['cohort']==cohort and r['id'] in common and r['model']==m and r['dimension']=='temporal_flickering' and r['score'] is not None])) for m in MODELS})
 with (ROOT/'metadata/per_video_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 groups={};pairs=[]
 for cohort in ids:
  selected=[r for r in records if r['cohort']==cohort]
  names=['all']+sorted({r['group'] for r in selected})+['native/'+s for s in sorted({r['subgroup'] for r in selected if r['group']=='native' and r['subgroup']})]
  groups[cohort]={}
  for group in names:
   rs=[r for r in selected if group=='all' or r['group']==group or (group.startswith('native/') and r['group']=='native' and r['subgroup']==group.split('/',1)[1])];groups[cohort][group]={}
   for dim in DIMS:
    ds=[r for r in rs if r['dimension']==dim];models={m:{r['id']:r['score'] for r in ds if r['model']==m and r['score'] is not None} for m in MODELS}
    matched=sorted(set(models['ltx_base'])&set(models['a849']));delta=[models['a849'][sid]-models['ltx_base'][sid] for sid in matched]
    groups[cohort][group][dim]=dict(models={m:dict(n=len(v),mean=mean(list(v.values()))) for m,v in models.items()},paired_n=len(matched),a849_minus_base_mean=mean(delta))
  for sid in ids[cohort]:
   for dim in DIMS:
    v={r['model']:r['score'] for r in selected if r['id']==sid and r['dimension']==dim}
    pairs.append(dict(cohort=cohort,id=sid,dimension=dim,**v,delta=v['a849']-v['ltx_base'] if v['a849'] is not None and v['ltx_base'] is not None else None))
 diagnostic_visual=read(DIAG/'metadata/visual_observations.json') if (DIAG/'metadata/visual_observations.json').exists() else {'samples':{}}
 observations=read(ROOT/'metadata/visual_observations.json') if (ROOT/'metadata/visual_observations.json').exists() else {'samples':{}}
 wide=[]
 for video in videos:
  values={r['dimension']:r['score'] for r in records if r['cohort']==video['cohort'] and r['id']==video['id'] and r['model']==video['model']}
  visual=observations['samples'].get(video['id'],{}).get('models',{}).get(video['model'],{}) if video['cohort']=='main' else diagnostic_visual['samples'].get(video['id'],{}).get('models',{}).get(video['model'],{})
  wide.append(dict(cohort=video['cohort'],id=video['id'],group=video['group'],subgroup=video['subgroup'],model=video['model'],generation_status=video['generation_status'],visual_reviewed=visual.get('reviewed',False),edit_effect=visual.get('edit'),preservation=visual.get('preservation'),notes=visual.get('notes'),**{d:values.get(d) for d in DIMS},output_sha256=video['output_sha256']))
 with (ROOT/'metadata/per_task_report.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(wide[0]));writer.writeheader();writer.writerows(wide)
 atomic(ROOT/'metadata/group_metrics.json',dict(groups=groups,expected_main_scores=len(ids['main'])*18,valid_main_scores=sum(r['cohort']=='main' and r['score'] is not None for r in records),expected_diagnostic_scores=len(ids['identity_diagnostic'])*18,valid_diagnostic_scores=sum(r['cohort']=='identity_diagnostic' and r['score'] is not None for r in records),missing_values='blank/null; never zero-filled',errors=errors,manifest_sha256=frozen['input_manifest_sha256']))
 atomic(ROOT/'metadata/temporal_flickering_static_paired.json',subsets)
 (ROOT/'metadata/paired_differences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
 def fmt(value):return '不适用/缺失' if value is None else f'{value:.5f}'
 valid_main=sum(r['cohort']=='main' and r['score'] is not None for r in records)
 valid_diagnostic=sum(r['cohort']=='identity_diagnostic' and r['score'] is not None for r in records)
 progress=f'有效主测分数 {valid_main}/1728；身份诊断分数 {valid_diagnostic}/72。'
 if valid_main<1728 or valid_diagnostic<72:progress+=' 当前为运行中快照，评分及最终验收尚未完成。'
 visual_n=sum(bool(m.get('reviewed')) for sample in observations['samples'].values() for m in sample.get('models',{}).values())
 if visual_n<288:progress+=f' 主测直接视觉审核{visual_n}/288，最终验收待完成。'
 correction=read(ROOT/'metadata/identity_protocol_correction.json') if (ROOT/'metadata/identity_protocol_correction.json').exists() else {'status':'validated'}
 if correction['status']!='validated':progress+=' 初轮身份参考有裁顶问题，严格全头复测进行中；身份去留与最终主测构成尚未验收。'
 lines=['# A849 测评集 v3','',progress,'',f"主测 {len(ids['main'])} 条；Identity 诊断 4 条。身份规则结论：{frozen['identity_decision']}。",'', 'v3 是调整后的测试集；不以 v2/v3 均值差声称模型进步。有效低质量输出保留；缺失评分留空。', '', '统一共同 0–4 秒、49 帧、12fps，横竖画布保持原协议。LTX dev V7 对照与 A849 仅 LoRA 不同；40 steps、seed42、CFG3、STG1 block28、rescale0.7、ref_cfg1。VACE 40 steps、CFG5、flow_shift5，source-only mask 内 RGB128。ltx_base 是 DataMake V7 条件协议下的对照，不代表官方 Ingredients 最佳能力。', '', '六项采用固定 revision 官方 VBench；Imaging Quality 原值除100。Subject/Background Consistency 衡量输出内部一致性，不能直接证明编辑成功。Temporal Flickering 全量原始分数受正常运动影响，上升不能直接解释为质量改善。未重复帧或冻结视频。', '']
 for cohort in ids:
  lines += [f'## {cohort}', '', '|指标 ↑|基座 n / mean|A849 n / mean|A849−基座配对差|配对n|VACE n / mean|','|---|---:|---:|---:|---:|---:|']
  for dim,label in zip(DIMS,LABELS):
   g=groups[cohort]['all'][dim];m=g['models'];cells={model:f"{m[model]['n']} / {fmt(m[model]['mean'])}" for model in MODELS};lines.append(f"|{label}|{cells['ltx_base']}|{cells['a849']}|{fmt(g['a849_minus_base_mean'])}|{g['paired_n']}|{cells['vace']}|")
  s=subsets[cohort]
  if s['filter_complete']:
   lines += ['',f"Temporal Flickering 官方静态共同子集：n={s['n']}；ID：{', '.join(s['task_ids']) or '不适用'}。",'']
   for model,m in s['models'].items():lines.append(f"- {model}: n={m['n']}, mean={fmt(m['mean'])}")
  else:lines += ['', 'Temporal Flickering 静态筛选尚未完成；配对子集 n 和均值待定，不作零共同样本结论。','']
  lines+=['','|分组|指标 ↑|基座 n / mean|A849 n / mean|VACE n / mean|','|---|---|---:|---:|---:|']
  for group,metrics in groups[cohort].items():
   if group=='all':continue
   for dim,label in zip(DIMS,LABELS):
    m=metrics[dim]['models'];cells=[f"{m[model]['n']} / {fmt(m[model]['mean'])}" for model in MODELS];lines.append('|'+group+'|'+label+'|'+'|'.join(cells)+'|')
  lines+=['','直接视觉编辑效果计数（定性观察；success / partial / failed / uncertain；仅计已审核输出，不是身份识别或自动评测真值）。','', '|分组|模型|已审核/总数|编辑效果计数|保持 good / partial / poor|','|---|---|---:|---|---|']
  for group in groups[cohort]:
   selected=[r for r in wide if r['cohort']==cohort and (group=='all' or r['group']==group or (group.startswith('native/') and r['group']=='native' and r['subgroup']==group.split('/',1)[1]))]
   for model in MODELS:
    expected=[r for r in selected if r['model']==model];reviewed=[r for r in expected if r['visual_reviewed']]
    edits=Counter(r['edit_effect'] for r in reviewed);preservation=Counter(r['preservation'] for r in reviewed)
    lines.append('|'+group+'|'+model+f'|{len(reviewed)}/{len(expected)}|'+ ' / '.join(str(edits[k]) for k in ['success','partial','failed','uncertain'])+'|'+ ' / '.join(str(preservation[k]) for k in ['good','partial','poor'])+'|')
 lines+=['','原始 v2 清单、报告、Dynamic Degree 和输出均保留。正式主表不再使用 Dynamic Degree。静态子集由 source 与三个模型输出全部通过官方 StaticFilter.infer 的共同任务构成；没有共同任务时记为不适用。静态筛选未完成不能作为零样本结论。', '', '逐任务编辑效果与六项分数联表见 per_task_report.csv，直接视觉记录见 visual_observations.json；来源、旧ID与修改原因见 inputs.jsonl 和 revision_log.json。配饰自建任务明确标记，无GT；官方任务保留原文。复用按有效输入、mask、辅助prompt和输出SHA核验，不改写原生成契约。', '', 'VBench 依赖与编码差异沿用 v2，不能将跨模型画质分差完全归因于模型；同源多编辑任务不作为独立视频做显著性推断。']
 (ROOT/'metadata/report.md').write_text('\n'.join(lines)+'\n')
 print('Main metric records',len(ids['main'])*18,'; static paired',subsets['main']['n'],flush=True)
if __name__=='__main__':main()
