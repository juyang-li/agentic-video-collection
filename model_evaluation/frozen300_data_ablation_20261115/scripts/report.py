from common import *
import csv,statistics

ALLMODELS=['ltx_distilled_v7']+MODELS+['a849_v7']
DIMS=GTDIMS+VBDIMS
def csvwrite(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def scalar(case,dim,score=None,raw=None,status='complete',infinite=False,origin='new',error=None):
 return {k:case[k] for k in ['id','uid','model','dataset','group','output_sha256','gt_sha256']}|dict(dimension=dim,score=score,raw_score=raw,psnr_infinite=infinite,status=status,origin=origin,error=error)

def invalid_status(record):
 return 'invalid_cache' if record['status']=='complete' else record['status']

def new_values(case):
 result=[]
 p=path_score('cpu',case);cpu=read(p) if p.exists() else {'status':'missing'}
 for dim in GTDIMS[:2]:
  good=compatible(p,case);v=cpu['mean'][dim] if good else None;inf=bool(cpu.get('mean',{}).get('psnr_infinite',False)) if dim=='psnr_db' else False
  result.append(scalar(case,dim,v,v,'complete' if good else invalid_status(cpu),inf,error=cpu.get('error')))
 p=path_score('lpips',case);l=read(p) if p.exists() else {'status':'missing'};good=compatible(p,case);v=l.get('score') if good else None
 result.append(scalar(case,'lpips',v,v,'complete' if good else invalid_status(l),error=l.get('error')))
 for dim in VBDIMS:
  p=path_score('vbench',case,dim);r=read(p) if p.exists() else {'status':'missing'};good=compatible(p,case)
  result.append(scalar(case,dim,r.get('score') if good else None,r.get('raw_score') if good else None,'complete' if good else invalid_status(r),error=r.get('error')))
 return result

def historical(model):
 result=[]
 for item in rows(PKG/'inputs.jsonl'):
  r=read(GTOLD/f'scores/{model}/{item["id"]}.json');case={k:r[k] for k in ['id','uid','dataset','group','output_sha256','gt_sha256']};case['model']=model
  for dim in GTDIMS:
   v=r['mean'][dim];result.append(scalar(case,dim,v,v,infinite=r['mean']['psnr_infinite'] if dim=='psnr_db' else False,origin='historical_reuse'))
  for dim in VBDIMS:
   h=r['output_sha256'];p=VBOLD/'metadata/vbench'/('raw' if dim=='temporal_flickering' else 'quality')/(f'{h}.json' if dim=='temporal_flickering' else f'{dim}--{h}.json');v=read(p)
   result.append(scalar(case,dim,v['score'],v['raw_score'],origin='historical_reuse'))
 return result

def scopes():
 items=rows(PKG/'inputs.jsonl')
 return [('overall','all',[r['id'] for r in items])]+[('dataset',g,[r['id'] for r in items if r['dataset']==g]) for g in ['ue','coinve','tripvvt']]+[('task',g,[r['id'] for r in items if r['group']==g]) for g in dict.fromkeys(r['group'] for r in items)]

def main():
 assert not os.environ.get('SLURM_JOB_ID')
 media=load_cases();data=[v for case in media for v in new_values(case)]+historical('ltx_distilled_v7')+historical('a849_v7')
 assert len(data)==10800 and len({(r['id'],r['model'],r['dimension']) for r in data})==10800
 errors=[r for r in data if r['origin']=='new' and r['status']!='complete']
 out=ROOT/'report';out.mkdir(exist_ok=True);csvwrite(out/'per_video_metrics.csv',data);atomic(out/'failures.json',errors)
 summaries=[];deltas=[]
 for level,label,ids in scopes():
  for dim in DIMS:
   maps={m:{r['id']:r for r in data if r['model']==m and r['dimension']==dim and r['id'] in ids and r['status']=='complete'} for m in ALLMODELS}
   paired=sorted(set.intersection(*(set(maps[m]) for m in ['ltx_distilled_v7']+MODELS)))
   summary={}
   for model in ALLMODELS:
    selected=list(maps[model].values());inf=any(r['psnr_infinite'] for r in selected)
    mean=None if not selected or inf else statistics.mean(r['score'] for r in selected)
    pv=[maps[model][sid] for sid in paired if sid in maps[model]];pinf=any(r['psnr_infinite'] for r in pv)
    paired_mean=None if not pv or pinf else statistics.mean(r['score'] for r in pv)
    summaries.append(dict(level=level,group=label,model=model,dimension=dim,n=len(selected),expected=len(ids),mean=mean,psnr_infinite=inf,paired_n=len(pv),paired_mean=paired_mean,role='different_protocol_reference' if model=='a849_v7' else 'main'))
    summary[model]=paired_mean
   def delta(a,b):return None if summary[a] is None or summary[b] is None else summary[a]-summary[b]
   deltas.append(dict(level=level,group=label,dimension=dim,paired_n=len(paired),real_minus_base=delta('real232','ltx_distilled_v7'),synthetic_minus_base=delta('synthetic600','ltx_distilled_v7'),synthetic_minus_real=delta('synthetic600','real232')))
 csvwrite(out/'summary.csv',summaries);atomic(out/'summary.json',summaries);csvwrite(out/'deltas.csv',deltas)
 nframes=0
 with (out/'frames.jsonl').open('w') as f:
  for case in media:
   pc=path_score('cpu',case);pl=path_score('lpips',case)
   if not (compatible(pc,case) and compatible(pl,case)):continue
   c=read(pc);l=read(pl);assert len(c['values'])==len(l['values'])==case['num_frames']
   assert c['gt_pts']==l['gt_pts'] and c['alignment']==l['alignment']
   for index,((s,p,inf),lp) in enumerate(zip(c['values'],l['values'])):
    f.write(json.dumps(dict(id=case['id'],model=case['model'],frame=index,ssim=s,psnr_db=p,psnr_infinite=inf,lpips=lp),allow_nan=False)+'\n');nframes+=1
 status='complete' if not errors else 'incomplete';valid=5400-len(errors)
 labels={'ltx_distilled_v7':'蒸馏基座8步','real232':'纯真实LoRA step232','synthetic600':'纯合成LoRA step600','a849_v7':'混合A849 dev40步（参考）'}
 lines=['# 冻结300条：训练数据消融测评','',f'状态：{status}；新增有效指标 {valid}/5400；失败/缺失 {len(errors)}。','', '表中前三个模型均为LTX2.5 distilled＋V7、8步、seed42、相同输入/GT与时间轴。新两组使用dev训练LoRA迁移到蒸馏基座，强度1、rank64/alpha64；232/600为checkpoint步数，不是训练数据条数。', '末列混合A849为dev基座、40步及原CFG/STG配置的历史结果；同表展示便于对照，仍单独标为不同推理协议的参考，不纳入前三者的同协议配对差值。', '训练数据来源和checkpoint步数均不同，以下是现有checkpoint的观察性对照，不单独归因为数据类型的严格因果实验。','', '|指标|'+'|'.join(labels[m] for m in ALLMODELS)+'|','|---|'+'---:|'*len(ALLMODELS)]
 def display(model,dim,level='overall',group='all'):
  r=next(r for r in summaries if (r['model'],r['dimension'],r['level'],r['group'])==(model,dim,level,group))
  if r['psnr_infinite']:v='∞'
  elif r['mean'] is None:v='N/A'
  else:v=f'{r["mean"]*(100 if dim in VBDIMS else 1):.5f}'
  return f'{v} (n={r["n"]})'
 for dim in DIMS:lines.append('|'+dim+(' ↓' if dim=='lpips' else ' ↑')+(' ×100' if dim in VBDIMS else ' dB' if dim=='psnr_db' else '')+'|'+'|'.join(display(m,dim) for m in ALLMODELS)+'|')
 lines+=['','## 分数据集的GT保真度','', '总体是300条视频等权，其中UE占200条；以下分组用于观察合成与真实域，不以总体掩盖域间差异。A849末列沿用同一GT评分协议，但推理协议不同。','', '|数据集／指标|'+'|'.join(labels[m] for m in ALLMODELS)+'|','|---|'+'---:|'*len(ALLMODELS)]
 for group in ['ue','coinve','tripvvt']:
  for dim in GTDIMS:lines.append('|'+group+' / '+dim+(' ↓' if dim=='lpips' else ' ↑')+'|'+'|'.join(display(m,dim,'dataset',group) for m in ALLMODELS)+'|')
 if not errors:
  lines+=['','### 观察','']
  for level,group,label in [('overall','all','总体'),('dataset','ue','UE'),('dataset','coinve','CoinVE'),('dataset','tripvvt','TripVVT')]:
   wins={m:[] for m in MODELS}
   for dim in GTDIMS:
    means={m:next(r['mean'] for r in summaries if (r['level'],r['group'],r['model'],r['dimension'])==(level,group,m,dim)) for m in MODELS}
    if all(v is not None for v in means.values()) and means[MODELS[0]]!=means[MODELS[1]]:
     winner=min(means,key=means.get) if dim=='lpips' else max(means,key=means.get);wins[winner].append(dim)
   for model,dimensions in wins.items():
    if dimensions:lines.append(f'- {label}：{labels[model]}在'+ '、'.join(dimensions)+'上优于另一个消融checkpoint。')
  lines+=['','这些是与GT的数值比较。总体含三分之二UE，需结合真实域分组；两组checkpoint步数不同，也不能由这些分数替代编辑指令或身份的人工审核。']
 lines+=['','## 混合训练A849参考','', 'A849总体九指标及分数据集GT三指标已并列展示在上方表格，全部每项n=300，分数据集为UE200、CoinVE75、TripVVT25。A849使用dev基座、40步及原CFG/STG配置；仅复用历史已验收分数，不能将与蒸馏8步模型的差异单独归因于训练数据。']
 lines+=['','## 协议与结果入口','', 'GT三指标：全部冻结12fps采样点，按原GT索引/输出PTS对齐，固定contain几何去padding，LANCZOS有效720p；SSIM高斯11/sigma1.5/population covariance；PSNR逐帧MSE后求dB；LPIPS AlexNet v0.1、FP32、关闭AMP/TF32、batch4（OOM仅降batch）。先帧平均、再视频等权。', 'VBench：原始1280×736或对应竖屏直接输入官方custom_input，保持固定commit/权重与longer成像预处理。全部六列统一score×100，成像质量只除100一次。Temporal Flickering为全量原始分数，受正常运动影响，不做静态筛选，不输出Total Score。', '六项质量/一致性与GT保真度不能自动判定编辑指令通过。没有新增人工通过标记，也没有上传审核包。','', '- [总体与分组统计](summary.csv)：总体、UE/CoinVE/TripVVT及10任务组，主表配对交集和各自n。','- [逐视频九指标](per_video_metrics.csv)、[逐帧GT指标](frames.jsonl)、[差值](deltas.csv)。差值为原尺度直接相减，LPIPS下降才表示改善；VBench差值显示时可乘100。','- [失败项](failures.json)、[输入及权重身份](../metadata/implementation.json)、[历史复用证据](../metadata/reuse_audit.json)。','- [验收](../metadata/final_validation.json)、[执行用时](../metadata/timing.json)、[资源释放](../metadata/release.json)。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 atomic(ROOT/'metadata/completion.json',{'status':status,'new_outputs':600,'expected_new_score_keys':5400,'valid_new_score_keys':valid,'errors':len(errors),'all_report_rows':len(data),'summary_rows':len(summaries),'new_gt_frame_pairs':nframes,'static_filter':'not_requested','human_reviewed':False})
 print('REPORT',status,valid,'/5400',flush=True)
if __name__=='__main__':main()
