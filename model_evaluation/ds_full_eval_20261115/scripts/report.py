"""Reproducible tables and paired, source-clustered diagnostic comparisons."""
import argparse, collections, csv, hashlib, json, random, statistics
from engine import ROOT, METRICS, read, save, attempt_files

def csv_save(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        if not records:return
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def interval(values,seed):
    # Each value is (source SHA, paired task difference). Resample entire sources.
    clusters=collections.defaultdict(list)
    for key,value in values:clusters[key].append(value)
    keys=sorted(clusters);rng=random.Random(seed);boot=[]
    for _ in range(1000):
        selected=[v for key in rng.choices(keys,k=len(keys)) for v in clusters[key]]
        boot.append(statistics.mean(selected))
    boot.sort();return boot[24],boot[974]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',action='store_true');args=parser.parse_args()
    tasks=read(ROOT/'metadata/tasks.json');sels=read(ROOT/'metadata/selections.json');protocol=read(ROOT/'metadata/protocol.json')
    if args.checkpoint:tasks=[t for t in tasks if t['key'] in sels['checkpoint100']]
    destination=ROOT/'report'/('checkpoint100' if args.checkpoint else 'full');destination.mkdir(parents=True,exist_ok=True)
    scores=[];missing=[];pairs=[];records={};cost=0.;attempts=0
    for p in attempt_files():
        r=read(p);cost+=r.get('response',{}).get('usage',{}).get('cost',0) or 0;attempts+=1
    for t in tasks:
        for out in t['outputs']:
            key=t['key']+'__'+out['model'];pair={'dataset':t['dataset'],'task_key':t['key'],'group':t['group'],'subgroup':t['subgroup'],'source_sha256':t['source_sha256'],'model':out['model'],'instruction':t['prompt']['en'],'source':t['source'],'output':out['path']}
            for metric in METRICS:
                p=ROOT/'results'/f'{key}__{metric}__r0.json'
                if not p.exists() or read(p).get('status')!='ok':missing.append(str(p.relative_to(ROOT)));continue
                r=read(p)
                allowed={v for k,v in protocol['prompt_hashes'].items() if k.endswith('_'+metric+'.md')}
                assert r['prompt_sha256'] in allowed and r['parameters']==protocol['parameters'], 'mixed protocol versions'
                records[(t['key'],out['model'],metric)]=r
                row={k:pair[k] for k in ['dataset','task_key','group','subgroup','source_sha256','model']}
                row.update(metric=metric,score=r['score'],reason=r['reason'],request_signature=r['signature'],prompt_sha256=r['prompt_sha256'],judge_model=r['response'].get('model'),provider=r['response'].get('provider'),seconds=r['seconds'],cost_usd=r['response'].get('usage',{}).get('cost'),cp_version=protocol.get('cp_version'))
                scores.append(row);pair[metric]=r['score'];pair[metric+'_reason']=r['reason']
            if all(m in pair for m in METRICS):pairs.append(pair)
    expected_scores=sum(len(t['outputs'])*2 for t in tasks)
    save(destination/'coverage.json',{'tasks':len(tasks),'expected_scores':expected_scores,'successful_scores':len(scores),'missing':missing,'all_attempts':attempts,'all_attempt_cost_usd':cost,'cp_version':protocol.get('cp_version')})
    if missing:
        print(json.dumps({'status':'incomplete','scores':len(scores),'expected':expected_scores,'cost':cost}));return
    csv_save(destination/'scores.csv',scores);csv_save(destination/'pairs.csv',pairs)
    for partition,in_calibration in [('calibration',True),('not_used_for_prompt_tuning',False)]:
        csv_save(destination/(partition+'_scores.csv'),[r for r in scores if (r['task_key'] in sels['checkpoint100'])==in_calibration])
    buckets=collections.defaultdict(list)
    for r in scores:
        for group in ['ALL',r['group']]:buckets[(r['dataset'],group,r['model'],r['metric'])].append(r['score'])
    summary=[]
    for (dataset,group,model,metric),v in sorted(buckets.items()):
        summary.append({'dataset':dataset,'group':group,'model':model,'metric':metric,'n':len(v),'mean':statistics.mean(v),'sd':statistics.stdev(v) if len(v)>1 else 0,**{'score_'+str(i):v.count(i) for i in range(1,6)}})
    csv_save(destination/'summary.csv',summary)
    for partition,in_calibration in [('calibration',True),('not_used_for_prompt_tuning',False)]:
        part=collections.defaultdict(list)
        for r in scores:
            if (r['task_key'] in sels['checkpoint100'])==in_calibration:
                for g in ['ALL',r['group']]:part[(r['dataset'],g,r['model'],r['metric'])].append(r['score'])
        csv_save(destination/(partition+'_summary.csv'),[{'dataset':ds,'group':g,'model':model,'metric':metric,'n':len(v),'mean':statistics.mean(v),'sd':statistics.stdev(v) if len(v)>1 else 0} for (ds,g,model,metric),v in sorted(part.items())])
    comparisons=[];investigate=[];diagnostics=[]
    for ds in ['frozen300','v4_116']:
        a='a849_v7' if ds=='frozen300' else 'a849'
        competitors=['h3_pdd8','vace_lightx2v4' if ds=='frozen300' else 'vace']
        if ds=='frozen300':competitors+=['ltx_distilled_v7','ltx_real232','ltx_synthetic600']
        for competitor in competitors:
            for metric in METRICS:
                for group in ['ALL']+sorted({t['group'] for t in tasks if t['dataset']==ds}):
                    values=[]
                    for t in tasks:
                        if t['dataset']!=ds or (group!='ALL' and t['group']!=group):continue
                        ra=records[(t['key'],a,metric)];rb=records[(t['key'],competitor,metric)];delta=ra['score']-rb['score'];values.append((t['source_sha256'],delta))
                        if group=='ALL' and delta<=-1:diagnostics.append({'dataset':ds,'task_key':t['key'],'group':t['group'],'metric':metric,'competitor':competitor,'a849_score':ra['score'],'competitor_score':rb['score'],'delta':delta,'a849_reason':ra['reason'],'competitor_reason':rb['reason']})
                    seed=20261115+int(hashlib.sha256((ds+competitor+metric+group).encode()).hexdigest()[:8],16)
                    low,high=interval(values,seed);mean=statistics.mean(v for _,v in values)
                    flag=((mean<=-.3 and high<0) if group=='ALL' else (mean<=-1 and len(values)>=4)) and competitor in ['h3_pdd8','vace','vace_lightx2v4']
                    row={'dataset':ds,'group':group,'a849_model':a,'competitor':competitor,'metric':metric,'n':len(values),'unique_sources':len({s for s,_ in values}),'a849_minus_competitor':mean,'ci95_low':low,'ci95_high':high,'investigate':flag};comparisons.append(row)
                    if flag:investigate.append(row)
    csv_save(destination/'comparisons.csv',comparisons);csv_save(destination/'diagnostic_cases.csv',sorted(diagnostics,key=lambda r:(r['delta'],r['dataset'],r['metric'],r['task_key'])))
    csv_save(destination/'ablation_comparisons.csv',[r for r in comparisons if r['dataset']=='frozen300' and r['competitor'] in ['ltx_distilled_v7','ltx_real232','ltx_synthetic600']])
    save(destination/'diagnostic_cases.json',sorted(diagnostics,key=lambda r:(r['delta'],r['dataset'],r['metric'],r['task_key'])))
    save(destination/'investigation.json',{'triggered':bool(investigate),'comparisons':investigate,'policy':'Inspect videos and specific non-target losses; change prompts only for evidenced, model-independent rubric errors; retain genuine performance differences.'})
    if not args.checkpoint:
        stability=[];expected_repeat=sum(len(t['outputs'])*2 for t in tasks if t['key'] in sels['repeat'])
        for t in tasks:
            if t['key'] not in sels['repeat']:continue
            for out in t['outputs']:
                for metric in METRICS:
                    original=records[(t['key'],out['model'],metric)];p=ROOT/'results'/f'{t["key"]}__{out["model"]}__{metric}__r1.json'
                    if not p.exists() or read(p).get('status')!='ok':continue
                    repeat=read(p);assert repeat['signature']==original['signature']
                    stability.append({'dataset':t['dataset'],'task_key':t['key'],'model':out['model'],'metric':metric,'score':original['score'],'repeat_score':repeat['score'],'absolute_difference':abs(original['score']-repeat['score']),'signature':original['signature']})
        csv_save(destination/'stability.csv',stability);save(destination/'stability_summary.json',{'expected':expected_repeat,'completed':len(stability),'exact_agreement':statistics.mean(r['absolute_difference']==0 for r in stability) if stability else None,'within_one':statistics.mean(r['absolute_difference']<=1 for r in stability) if stability else None,'mean_absolute_difference':statistics.mean(r['absolute_difference'] for r in stability) if stability else None})
    print(json.dumps({'status':'complete','tasks':len(tasks),'scores':len(scores),'cost_usd':cost,'investigation_triggered':bool(investigate),'dataset_overall':[r for r in summary if r['group']=='ALL'],'overall_comparisons':[r for r in comparisons if r['group']=='ALL']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
