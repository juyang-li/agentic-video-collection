"""Login: exact-key audit, full six metrics and source+four-model static intersection."""
from common import *
import csv,math,statistics

def mean(v):return statistics.mean(v) if v else None
def csvwrite(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def static_summary(ids,flags,scores):
 complete=all(flags.get((sid,m)) is not None for sid in ids for m in ['source']+MODELS)
 eligible=[sid for sid in ids if all(flags.get((sid,m)) is True for m in ['source']+MODELS)]
 scored=[sid for sid in eligible if all(scores.get((sid,m)) is not None for m in MODELS)]
 status='pending' if not complete else 'not_applicable' if not eligible else 'incomplete' if scored!=eligible else 'complete'
 return {'status':status,'filter_complete':complete,'n':len(scored) if complete else None,'task_ids':scored if complete else [],'eligible_ids':eligible,'means':{m:mean([scores[sid,m] for sid in scored]) if complete else None for m in MODELS}}

def main():
 assert not os.environ.get('SLURM_JOB_ID')
 use_static=static_required()
 items=rows(ROOT/'metadata/inputs.jsonl');media=rows(ROOT/'metadata/scoring_media.jsonl');sources=rows(ROOT/'metadata/source_media.jsonl')
 assert len(items)==300 and len(media)==1200
 assert {(r['id'],r['model']) for r in media}=={(r['id'],m) for r in items for m in MODELS}
 results=[];flags={};flickers={};errors=[];impl=sha(ROOT/'metadata/implementation.json')
 identity=read(ROOT/'metadata/implementation.json');manifest=sha(ROOT/'metadata/inputs.jsonl')
 assert manifest==identity['input_manifest_sha256']==sha(PKG/'inputs.jsonl')
 for r in media:
  h=r['output_sha256'];assert sha(r['path'])==h
  for dim in DIMS:
   p=ROOT/'metadata/vbench'/('raw' if dim=='temporal_flickering' else 'quality')/(f'{h}.json' if dim=='temporal_flickering' else f'{dim}--{h}.json')
   v=read(p) if p.exists() else {'status':'missing'}
   good=v['status']=='complete'
   if good:
    assert v['output_sha256']==h and v['implementation_sha256']==impl
    assert v['vbench_revision']==identity['vbench_revision'] and v['manifest_sha256']==manifest
    if dim=='temporal_flickering':assert v['official_source_sha256']==identity['official_sources_sha256']['vbench/temporal_flickering.py']
    assert all(math.isfinite(v[k]) for k in ['raw_score','score'])
    assert abs(v['score']-(v['raw_score']/100 if dim=='imaging_quality' else v['raw_score']))<1e-10
   row={k:r[k] for k in ['id','uid','dataset','group','model','output_sha256']};row.update(dimension=dim,status=v['status'],raw_score=v.get('raw_score') if good else None,score=v.get('score') if good else None);results.append(row)
   if not good:errors.append({'id':r['id'],'model':r['model'],'dimension':dim,'status':v['status'],'error':v.get('error')})
   elif dim=='temporal_flickering':flickers[r['id'],r['model']]=v['score']
 for r in (media+sources if use_static else []):
  h=r['output_sha256'];p=ROOT/f'metadata/vbench/static/{h}.json';v=read(p) if p.exists() else {'status':'missing'}
  if v['status']=='complete':
   assert v['media_sha256']==h and v['implementation_sha256']==impl and type(v['is_static']) is bool
   assert v['vbench_revision']==identity['vbench_revision'] and v['manifest_sha256']==manifest
   assert v['official_source_sha256']==identity['official_sources_sha256']['static_filter.py']
   assert v['raft_weights_sha256']==identity['weights_sha256']['raft_model/models/raft-things.pth']
  flags[r['id'],r['model']]=v['is_static'] if v['status']=='complete' else None
  if v['status']!='complete':errors.append({'id':r['id'],'model':r['model'],'dimension':'static_filter','status':v['status'],'error':v.get('error')})
 out=ROOT/'report';out.mkdir(exist_ok=True);csvwrite(out/'per_video_metrics.csv',results)
 summaries=[]
 scopes=[('overall','all',[i['id'] for i in items])]+[('dataset',d,[i['id'] for i in items if i['dataset']==d]) for d in ['ue','coinve','tripvvt']]+[('task',g,[i['id'] for i in items if i['group']==g]) for g in dict.fromkeys(i['group'] for i in items)]
 for level,label,ids in scopes:
  for dim in DIMS:
   maps={m:{r['id']:r['score'] for r in results if r['model']==m and r['dimension']==dim and r['id'] in ids and r['score'] is not None} for m in MODELS}
   common=sorted(set.intersection(*(set(v) for v in maps.values())))
   for m in MODELS:summaries.append({'level':level,'group':label,'model':m,'dimension':dim,'n':len(maps[m]),'expected':len(ids),'mean':mean(list(maps[m].values())),'paired_n':len(common),'paired_mean':mean([maps[m][sid] for sid in common])})
 csvwrite(out/'summary.csv',summaries);atomic(out/'summary.json',summaries)
 if use_static:
  static={f'{level}/{label}':static_summary(ids,flags,flickers) for level,label,ids in scopes}
  atomic(out/'temporal_flickering_static_paired.json',static)
  csvwrite(out/'static_flags.csv',[{'id':i['id'],'model':m,'is_static':flags[i['id'],m]} for i in items for m in ['source']+MODELS])
  s=static['overall/all']
 else:
  s={'status':'not_requested','n':None}
  atomic(out/'static_filter_status.json',{'status':'not_requested','reason':'Explicit user instruction: 不用静态筛选','partial_cache_used_in_report':False,'static_paired_means_reported':False})
 atomic(out/'failures.json',errors)
 status='complete' if not errors and len(results)==7200 else 'incomplete'
 lines=['# frozen300：VBench custom_input 六项测评','',f'状态：{status}；有效分数 {sum(r["score"] is not None for r in results)}/7200。','', '四模型：LTX2.5 distilled＋V7（8步）、dev＋A849＋V7（40步）、H3-PDD8、VACE-LightX2V4。各300条，冻结UE200/CoinVE75/TripVVT25，不按输出表现筛选。','', '按冻结时间轴评价全部区间，12fps；H3抽取真实生成帧2i并RGB无损编码，逐帧像素哈希验证。保留各自原生画布，未使用GT指标测评的缩放去边框视频。各模型尺寸与编码仍有差异，成像分差不能完全归因于模型。', '', '六项是输出质量及内部一致性，不代表编辑成功、身份参考匹配或SOURCE保持。custom_input不等于官方prompt suite总榜，不提供六项平均Total Score。全部数值列统一为score×100；Imaging Quality的score仅对官方raw值除100一次。', '', '|指标 ↑（×100）|'+'|'.join(MODELS)+'|','|---|'+'---:|'*4]
 for dim in DIMS:
  vals=[next(r for r in summaries if r['level']=='overall' and r['dimension']==dim and r['model']==m) for m in MODELS]
  lines.append('|'+dim+'|'+'|'.join(f'{r["mean"]*100:.5f} (n={r["n"]})' if r['mean'] is not None else f'N/A (n={r["n"]})' for r in vals)+'|')
 lines+=['', 'Temporal Flickering为全量原始分数，受正常运动影响，不能视为纯闪烁排名。']
 if use_static:
  ids_text=', '.join(s['task_ids']) if s['task_ids'] else '无（交集为空）' if s['filter_complete'] else '待筛选完成'
  lines += ['官方RAFT逐视频筛选后，额外取SOURCE与四模型均静态的共同任务集合；该交集规则属于本项目比较设计。',f'静态共同子集：{s["status"]}；n={s["n"]}；ID：{ids_text}。']
  for m in MODELS:lines.append(f'- {m}: '+(f'{s["means"][m]*100:.5f}' if s['means'][m] is not None else 'N/A'))
  lines+=['空交集只在全部筛选完成时记N/A；缺分数/筛选失败不填零、不当作非静态。','静态标记：[static_flags.csv](static_flags.csv)；共同子集：[temporal_flickering_static_paired.json](temporal_flickering_static_paired.json)。']
 else:
  lines+=['按用户最新要求取消静态筛选，仅报告全量原始闪烁分数；不计算静态共同子集，不使用已产生的部分筛选缓存推断静态结果。范围变更见[user_override.json](../metadata/user_override.json)，状态见[static_filter_status.json](static_filter_status.json)。']
 lines+=['', '完整分组、配对交集、逐视频原始/统一尺度分数见CSV/JSON。',
 f'固定VBench commit：`{identity["vbench_revision"]}`；`local=True`、`mode=custom_input`、`imaging_quality_preprocessing_mode=longer`。官方源码及评分权重SHA见[implementation.json](../metadata/implementation.json)，实际依赖见[environment.json](../metadata/environment.json)和[pip_freeze.txt](../metadata/pip_freeze.txt)。',
 '清单与媒体核验见[frozen_media_audit.json](../metadata/frozen_media_audit.json)和[pairing_audit.json](../metadata/pairing_audit.json)。GPU评分按已准备媒体的固定清单分批执行，各批次输入SHA及退出记录保存在[metadata/batches](../metadata/batches/)，最终按完整300任务汇总。',
 '逐视频分数：[per_video_metrics.csv](per_video_metrics.csv)；总体及分组：[summary.csv](summary.csv)；失败项：[failures.json](failures.json)。',
 '本轮只交付自动指标及技术核验，不声称1200条已经人工编辑质量审核；不重新生成模型结果，不自动上传审核包。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 atomic(ROOT/'metadata/completion.json',{'status':status,'expected_scores':7200,'valid_scores':sum(r['score'] is not None for r in results),'errors':len(errors),'static_filter_required':use_static,'static_status':s['status'],'static_paired_n':s['n'],'static_unique_expected':len({r['output_sha256'] for r in media+sources}) if use_static else 0,'human_reviewed':False})
 print('AGGREGATE',status,'errors',len(errors),flush=True)
if __name__=='__main__':main()
