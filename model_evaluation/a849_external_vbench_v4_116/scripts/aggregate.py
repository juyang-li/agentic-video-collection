"""CPU: six official metrics, all three cohorts, and source + FOUR model static pairing."""
import csv,statistics
from collections import Counter
from common import *
from contracts import DIMS,LABELS,finite_score
from score import MODELS,inventory

def mean(values):return statistics.mean(values) if values else None

def write_csv(path,records):
 with path.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
 items=rows(ROOT/'metadata/inputs.jsonl');manifest=sha(ROOT/'metadata/inputs.jsonl');assert read(ROOT/'metadata/freeze.json')['input_manifest_sha256']==manifest
 videos=inventory();records=[];flags={};wide=[];errors=[]
 visual=read(ROOT/'metadata/visual_observations.json').get('samples',{})if(ROOT/'metadata/visual_observations.json').exists()else{}
 def static_flag(path):
  if not path.exists():return None
  h=sha(path);p=ROOT/'metadata/vbench/static'/f'{h}.json'
  if not p.exists():return None
  r=read(p);assert r['media_sha256']==h
  return r.get('is_static')if r['status']=='complete'else None
 for item in items:flags[item['id'],'source']=static_flag(ROOT/'inputs/prepared'/item['id']/'source.mp4')
 for video in videos:
  h=video['output_sha256'];values={};sid=video['id'];model=video['model'];flags[sid,model]=static_flag(Path(video['path']))if h else None
  for dim in DIMS:
   p=ROOT/'metadata/vbench'/('raw'if dim=='temporal_flickering'else'quality')/(f'{h}.json'if dim=='temporal_flickering'else f'{dim}--{h}.json')
   result=read(p)if h and p.exists()else{'status':'missing_score'}
   good=result['status']=='complete'
   if good:assert result['output_sha256']==h and finite_score(result['score'])
   row={k:video[k]for k in ['cohort','id','group','subgroup','model','output_sha256']};row.update(dimension=dim,status='complete'if good else'missing_generation'if not h else result['status'],raw_score=result.get('raw_score')if good else None,score=result.get('score')if good else None);records.append(row);values[dim]=row['score']
   if result['status']not in['complete','missing_score']:errors.append(dict(id=sid,model=model,dimension=dim,error=result))
  observation=visual.get(sid,{}).get('models',{}).get(model,{})
  reviewed=observation.get('reviewed')is True and observation.get('output_sha256')==h and h is not None
  wide.append(dict(cohort=video['cohort'],id=sid,group=video['group'],subgroup=video['subgroup'],model=model,generation_status=video['generation_status'],visual_reviewed=reviewed,edit_effect=observation.get('edit')if reviewed else None,preservation=observation.get('preservation')if reviewed else None,notes=observation.get('notes')if reviewed else None,**values,output_sha256=h))
 write_csv(ROOT/'metadata/per_video_metrics.csv',records);write_csv(ROOT/'metadata/per_task_report.csv',wide)
 scopes={c:[i['id']for i in items if c=='all116'or i['cohort']==c]for c in['retained92','new24','all116']};groups={};subsets={}
 for cohort,ids in scopes.items():
  selected=[r for r in records if r['id']in ids];group_names=['all']+sorted({r['group']for r in selected})+['native/'+s for s in sorted({r['subgroup']for r in selected if r['group']=='native'and r['subgroup']})];groups[cohort]={}
  for group in group_names:
   rs=[r for r in selected if group=='all'or r['group']==group or(group.startswith('native/')and r['group']=='native'and r['subgroup']==group.split('/',1)[1])];groups[cohort][group]={}
   for dim in DIMS:
    models={m:{r['id']:r['score']for r in rs if r['model']==m and r['dimension']==dim and r['score']is not None}for m in MODELS};paired=sorted(set.intersection(*(set(models[m])for m in MODELS)));ab=sorted(set(models['ltx_base'])&set(models['a849']));ah=sorted(set(models['a849'])&set(models['h3_pdd8']))
    groups[cohort][group][dim]=dict(models={m:dict(n=len(v),expected=sum(r['model']==m and r['dimension']==dim for r in rs),mean=mean(list(v.values())))for m,v in models.items()},four_model_paired_n=len(paired),four_model_paired_means={m:mean([models[m][sid]for sid in paired])for m in MODELS},a849_base_paired_n=len(ab),a849_minus_base_mean=mean([models['a849'][sid]-models['ltx_base'][sid]for sid in ab]),a849_h3_paired_n=len(ah),a849_minus_h3_mean=mean([models['a849'][sid]-models['h3_pdd8'][sid]for sid in ah]))
  eligible=[sid for sid in ids if all(flags.get((sid,m))is True for m in('source',*MODELS))];complete=all(flags.get((sid,m))is not None for sid in ids for m in('source',*MODELS));flickers={(r['id'],r['model']):r['score']for r in selected if r['dimension']=='temporal_flickering'and r['score']is not None};common=[sid for sid in eligible if all((sid,m)in flickers for m in MODELS)]
  subsets[cohort]=dict(status=('not_applicable'if not eligible else'incomplete_scores'if len(common)<len(eligible)else'applicable')if complete else'pending',filter_complete=complete,n=len(common)if complete else None,task_ids=common,static_eligible_task_ids=eligible,excluded_missing_flicker_ids=[sid for sid in eligible if sid not in common],filter_results=[dict(id=sid,**{m:flags.get((sid,m))for m in('source',*MODELS)})for sid in ids],models={m:dict(n=len(common),mean=mean([flickers[sid,m]for sid in common]))for m in MODELS})
 atomic(ROOT/'metadata/group_metrics.json',dict(groups=groups,expected_scores=2784,valid_scores=sum(r['score']is not None for r in records),missing_values='blank/null; never zero-filled',errors=errors,manifest_sha256=manifest))
 atomic(ROOT/'metadata/temporal_flickering_static_paired.json',subsets)
 fmt=lambda x:'不适用/缺失'if x is None else f'{x:.5f}'
 valid=sum(r['score']is not None for r in records);reviewed=sum(r['visual_reviewed']for r in wide)
 lines=['# A849 v4：116 条任务、四模型','',f'有效分数 {valid}/2784；直接视觉审核 {reviewed}/464。'+(' 最终验收待完成。'if valid<2784 or reviewed<464 else''),'','保留92条未变任务，新增发色8、背景8、TripVVT8。Identity12与object_removal4移出新版主测，历史结果保留。新增TripVVT8从32条冻结候选的输出中筛选，要求A849编辑成功且保持良好；H3允许失败。全部32条结果、入选及排除理由独立交付。','', '共同内容为原视频 0–4 秒。LTX 基座与 A849 保持40步；三者使用49帧、12fps及原横竖画布。H3 使用 README 指定的 Ref2VA PDD 八步、TP2/Ulysses2、seed42、video/audio shift12/3、lossless/max reference。原生为1344×768或768×1344、24fps、107帧；评分直接抽取实际生成帧0,2,…,96，49帧12fps。完整原生视频含音频单独保留，浏览副本不用于评分。输入末端按原生对齐规则补齐，输出评分不插帧、不复制帧、不冻结视频。','', 'LTX 基座使用 DataMake dev V7 条件协议，与 A849 仅 LoRA 不同；40 steps、seed42、CFG3、STG1 block28、rescale0.7、ref_cfg1。它不代表官方 Ingredients 路径的最佳能力。VACE 按逐视频生成记录采样，使用BF16 transformer及FP32 VAE，source-only mask内填RGB128。', '', '各模型原生分辨率与编码存在差异，画质分差不能完全归因于模型。Subject/Background Consistency 衡量输出内部一致性，不能直接证明编辑成功。Temporal Flickering 全量原始分数受正常运动影响，上升不能直接解释为质量改善。使用官方静态筛选，取 source 与四模型全部通过的共同任务；没有共同任务时为不适用。','', 'v4 是按输出表现筛选新增TripVVT的调整测试集，不以 v3/v4 均值差声称模型进步。原v3完整报告保留为对照。低质量有效输出保留，缺失评分不填零。同源多编辑任务不作为独立原视频做显著性推断。','']
 native_origins=Counter(i['task_origin']for i in items if i['group']=='native')
 lines += [f"通用文字编辑20条的原视频均来自固定版本OmniEdit-Bench；其中{native_origins['official']}条使用官方原指令，{native_origins['self_created']}条配饰指令为自建并明确标记。面部4条保留旧native_03、20、23、24，对应新native_01、08、09、10。逐条来源与官方指令核对见native_provenance_audit.json。",'']
 for cohort,ids in scopes.items():
  lines += [f'## {cohort}（{len(ids)} 条）','','|指标 ↑|LTX base n / mean|A849 n / mean|VACE n / mean|H3 PDD8 n / mean|','|---|---:|---:|---:|---:|']
  for dim,label in zip(DIMS,LABELS):
   m=groups[cohort]['all'][dim]['models'];lines.append('|'+label+'|'+ '|'.join(f"{m[k]['n']} / {fmt(m[k]['mean'])}"for k in MODELS)+'|')
  s=subsets[cohort];lines += ['',f"官方静态共同子集：状态 {s['status']}，n={s['n'] if s['filter_complete'] else '待定'}；ID：{', '.join(s['task_ids'])or('不适用'if s['filter_complete']else'待筛选完成')}。",'']
  for m in MODELS:lines.append(f"- {m}: n={s['models'][m]['n']}, mean={fmt(s['models'][m]['mean'])}")
  lines+=['','|分组|指标 ↑|LTX base|A849|VACE|H3 PDD8|','|---|---|---:|---:|---:|---:|']
  for group,dims in groups[cohort].items():
   if group=='all':continue
   for dim,label in zip(DIMS,LABELS):lines.append('|'+group+'|'+label+'|'+'|'.join(f"{dims[dim]['models'][m]['n']} / {fmt(dims[dim]['models'][m]['mean'])}"for m in MODELS)+'|')
  lines+=['','|模型|直接审核/总数|编辑 success / partial / failed / uncertain|保持 good / partial / poor|','|---|---:|---|---|']
  for m in MODELS:
   rs=[r for r in wide if r['id']in ids and r['model']==m and r['visual_reviewed']];e=Counter(r['edit_effect']for r in rs);p=Counter(r['preservation']for r in rs)
   lines.append('|'+m+f'|{len(rs)}/{len(ids)}|'+' / '.join(str(e[k])for k in['success','partial','failed','uncertain'])+'|'+' / '.join(str(p[k])for k in['good','partial','poor'])+'|')
 lines+=['','逐视频六项分数见 per_video_metrics.csv；编辑效果与指标联表见 per_task_report.csv；分组及四模型配对均值见 group_metrics.json。静态子集逐任务条件见 temporal_flickering_static_paired.json。历史 Dynamic Degree 及全部旧版输出和诊断保留。']
 (ROOT/'metadata/report.md').write_text('\n'.join(lines)+'\n');print('Metrics',valid,'/2784; direct reviews',reviewed,'/464',flush=True)
if __name__=='__main__':main()
