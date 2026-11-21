"""Build an offline anonymous continuous-scroll review and runtime report."""
import collections, html, json, random, shutil, statistics
from pilot import ROOT, BENCH, load, save, frozen, sha, messages

def main():
    cs=load(ROOT/'metadata/candidates.json');review=ROOT/'review';review.mkdir(exist_ok=True)
    rng=random.Random(20261115);reveal={};parts=[];rows=[];inventory=[];task_inventory=[]
    task_inputs={i['id']:i for i in load(ROOT/'metadata/selection.json')['tasks']}
    provenance={i['id']:i['record'] for i in [json.loads(l) for l in (BENCH/'metadata/selection.jsonl').read_text().splitlines()]}
    def esc(s):return html.escape(str(s))
    def copy(src,name):
        dest=review/'media'/name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(src,dest)
        return 'media/'+name
    def video(src,label,name):
        url=copy(src,name)
        return f'<figure><figcaption>{esc(label)}</figcaption><video controls preload="none" src="{url}"></video></figure>'
    tasks=list(dict.fromkeys(c['task_id'] for c in cs))
    for n,task in enumerate(tasks):
        group=[c for c in cs if c['task_id']==task];rng.shuffle(group)
        section=f'<section id="t{n}"><h2>任务{n+1}</h2><p>{esc(group[0]["instruction"])}</p><button onclick="playGroup(this)">同步播放 / 重置</button><div class="media">'
        section+=video(group[0]['source'],'SOURCE',f't{n}_source.mp4')
        original=copy(task_inputs[task]['source']['absolute_path'],f't{n}_source_full.mp4')
        section+=f'<p><a href="{original}" target="_blank">独立查看完整原始SOURCE</a></p>'
        task_record={'task':f't{n}','instruction':group[0]['instruction'],'full_source':original,'full_source_sha256':sha(task_inputs[task]['source']['absolute_path'])}
        for j,r in enumerate(group[0]['references']):
            url=copy(ROOT/r['path'],f't{n}_r{j}.png');section+=f'<figure><figcaption>R{j+1} {esc(r["type"])}</figcaption><img src="{url}" loading="lazy"></figure>'
        if task_inputs[task]['group']=='tripvvt':
            selection=provenance[task]
            section+=video(BENCH/'inputs/prepared'/task/'gt_review_only.mp4','GT target（仅供人工审核，未输入VLM）',f't{n}_gt.mp4')
            fullgt=copy(selection['target']['absolute_path'],f't{n}_gt_full.mp4')
            section+=f'<p>官方caption（仅审核）：{esc(selection["caption"])}</p><p><a href="{fullgt}" target="_blank">独立查看完整GT</a></p>'
            task_record.update(official_caption=selection['caption'],gt=f'media/t{n}_gt.mp4',gt_sha256=sha(BENCH/'inputs/prepared'/task/'gt_review_only.mp4'),full_gt=fullgt,full_gt_sha256=sha(selection['target']['absolute_path']),gt_input_to_judges=False)
        task_inventory.append(task_record)
        section+='</div>'
        for k,c in enumerate(group):
            judges=['deepseek','qwen','gpt'];rng.shuffle(judges)
            public=f't{n}_candidate{k}';reveal[public]={'candidate_id':c['id'],'model':c['model'],'judges':dict(zip('ABC',judges))}
            inventory.append({'task':f't{n}','candidate':public,'instruction':c['instruction'],'source':f'media/t{n}_source.mp4','source_sha256':c['source_sha256'],'output':f'media/{public}.mp4','output_sha256':c['output_sha256'],'references':[{'path':f'media/t{n}_r{j}.png','type':r['type'],'view':r['view'],'sha256':r['sha256']} for j,r in enumerate(c['references'])],'timecards':[{'path':f'media/{public}_p{f["frame_index"]}.jpg','sha256':f['sha256'],'frame_index':f['frame_index'],'time_seconds':f['time_seconds']} for f in c['cards']]})
            section+=f'<article><h3>候选 C{k+1}</h3>'+video(c['output'],f'C{k+1}',public+'.mp4')
            section+='<table><tr><th>裁判</th><th>编辑达成 EA</th><th>内容保持 CP</th></tr>'
            for label,j in zip('ABC',judges):
                section+=f'<tr><td>Judge {label}</td>'
                for metric in ['edit_achievement','content_preservation']:
                    p=ROOT/'metadata/parsed'/f'{j}_{c["id"]}_{metric}_r0.json'
                    r=load(p) if p.exists() else {'status':'pending'}
                    rows.append(dict(candidate_id=c['id'],judge=j,metric=metric,**{key:r.get(key) for key in ['status','score','seconds','usage','attempt_count']}))
                    section+='<td>'+ (f'{r["score"]}/5<br>{esc(r["reason"])}' if r['status']=='ok' else esc(r['status']))+'</td>'
                section+='</tr>'
            section+='</table><p>API实际输入 timecards：</p><div class="cards">'
            for f in c['cards']:
                url=copy(ROOT/f['path'],public+f'_p{f["frame_index"]}.jpg')
                section+=f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a>'
            section+='</div></article>'
        section+='</section>';parts.append(section)
    frozen(ROOT/'metadata/reveal.json',reveal)
    save(review/'inventory.json',inventory)
    save(review/'tasks.json',task_inventory)
    for p in (ROOT/'prompts').glob('*.md'):
        dest=review/'prompts'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    (review/'README.txt').write_text('解压后打开index.html。10任务×4输出；Judge A/B/C和候选C1–C4的映射在每个候选内独立随机，身份映射保存在包外。\n分数为VLM的1–5分，不是人工审核。EA仅评价目标修改达成；CP仅比较其余内容保持。CP不看reference像素。\ntimecard为API实际输入：16个时间点、SOURCE左/GENERATED右；点击缩略图查看原尺寸。参考图独立输入EA。评分模板在prompts，匿名媒体关联及SHA在inventory.json。\n请先逐项阅读prompt、观看source/reference/output，再比较三位裁判的分数和理由。页面不收集人工通过结论。\n')
    nav='<nav><a href="#top">顶部</a> '+ ' '.join(f'<a href="#t{n}">任务{n+1}</a>' for n in range(len(tasks)))+'</nav>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VLM匿名试评</title><style>body{font:16px sans-serif;margin:70px 24px;background:#fafafa;color:#222}nav{position:fixed;top:0;left:0;right:0;background:#ddd;padding:16px;z-index:9}nav a{margin:8px}section{scroll-margin-top:70px}article{border-top:2px solid #ccc;margin:24px 0;padding-top:8px}.media{display:flex;flex-wrap:wrap}figure{margin:8px}video{width:480px;max-height:350px}figure img{max-width:300px;max-height:350px}table{width:100%;border-collapse:collapse}td,th{border:1px solid #bbb;padding:12px;width:44%}td:first-child,th:first-child{width:12%}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}.cards img{width:100%}p{white-space:pre-wrap}</style><body id="top">'''+nav+'<h1>10任务×4输出：匿名评价器试评</h1><p>分数为VLM的1–5评分；pending/error不是低分。裁判及生成模型匿名映射在包外保存。当前页面无人工评分表，不代表人工通过。视频共同区间0–4秒。</p>'+''.join(parts)+'''<script>function playGroup(b){let vs=b.closest('section').querySelectorAll('video');vs.forEach(v=>{v.currentTime=0;v.play()})}</script></body></html>'''
    (review/'index.html').write_text(page)
    summary={}
    attempts=[load(p) for p in (ROOT/'results').glob('*_attempt*.json')]
    all_results=[load(p) for p in (ROOT/'metadata/parsed').glob('*_r*.json')]
    for j in ['deepseek','qwen','gpt']:
        done=[r for r in rows if r['judge']==j and r['status']=='ok'];lat=[r['seconds'] for r in done]
        summary[j]={'successful':len(done),'expected_base_calls':80,'status_counts':dict(collections.Counter(r['status'] for r in rows if r['judge']==j)),'latency_mean_seconds':statistics.mean(lat) if lat else None,'latency_p95_seconds':sorted(lat)[min(len(lat)-1,int(len(lat)*.95))] if lat else None,'estimated_1500_candidates_hours_at32':len(lat) and statistics.mean(lat)*3000/32/3600,'reported_cost_usd':sum((r.get('usage') or {}).get('cost',0) or 0 for r in done)}
        a=[r for r in attempts if r['judge']==j]
        summary[j].update(api_attempts=len(a),failed_attempts=sum(r['status']!='ok' for r in a),reported_cost_all_attempts_usd=sum((r.get('response',{}).get('usage') or {}).get('cost',0) or 0 for r in a),latency_p50_seconds=statistics.median(lat) if lat else None)
        base_cost=sum((r.get('response',{}).get('usage') or {}).get('cost',0) or 0 for r in a if r['repeat']==0)
        summary[j].update(base_cost_all_attempts_usd=base_cost,estimated_1500_candidates_usd=base_cost/len(done)*3000 if done else None)
        for metric in ['edit_achievement','content_preservation']:
            groups=collections.defaultdict(list)
            for r in all_results:
                if r['judge']==j and r['metric']==metric and r['status']=='ok':groups[r['candidate_id']].append(r)
            complete=[rs for rs in groups.values() if {r['repeat'] for r in rs}=={0,1,2}]
            pairs=[abs(x['score']-y['score']) for rs in complete for n,x in enumerate(rs) for y in rs[n+1:]]
            summary[j][metric+'_stability']={'three_run_cases':len(complete),'pairwise_exact_agreement':sum(p==0 for p in pairs)/len(pairs) if pairs else None,'pairwise_mae':statistics.mean(pairs) if pairs else None}
    save(ROOT/'metadata/runtime_summary.json',summary)
    save(ROOT/'metadata/review_rows.json',rows)
    lines=['# API试评执行结果','', '三评价器使用相同10任务×4既有输出、独立EA/CP评分，另在8候选上各增加两次独立重复。所有费用均为OpenRouter响应usage.cost之和，包含失败重试中已计费的响应；不称为官方ChatGPT credit。','', '|评价器|基础完成/80|三次完整案例EA/CP（各8）|全部attempt费用USD|基础平均/p95延迟秒|1500候选基础费用外推USD|','|---|---:|---:|---:|---:|---:|']
    for j,s in summary.items():
        mean=s['latency_mean_seconds'];p95=s['latency_p95_seconds'];projection=s['estimated_1500_candidates_usd']
        lines.append(f'|{j}|{s["successful"]}|{s["edit_achievement_stability"]["three_run_cases"]}/{s["content_preservation_stability"]["three_run_cases"]}|{s["reported_cost_all_attempts_usd"]:.4f}|{mean:.2f}/{p95:.2f}|{projection:.2f}|' if mean is not None else f'|{j}|0|0/0|0|未测|未测|')
    lines+=['', '1500候选=3000次基础请求。费用与32并发延迟均为当前样本的线性外推，不是报价或完成时限保证；不包含大规模重跑。缓存命中、图像尺寸、reasoning长度和账户并发预算会影响结果。当前OpenRouter账户曾触发402/in_flight_budget_exhausted，GPT恢复时按provider的120秒结算窗口串行补跑，实际墙钟时间不能只按模型延迟计算。', '', 'Qwen的部分回答包含模板示例的外层Markdown代码框。原始响应保留；报告使用首个完整、可确定解析的响应，只允许去完整外框，内部仍须严格两行且finish_reason=stop。具体format_compliant和parse_source见metadata/parsed。不得将格式通过等同于视觉判断正确。', '', '匿名离线审核页review/index.html包含完整prompt、source/reference、40输出、三裁判分数理由和实际16帧timecard。inventory.json绑定匿名媒体和SHA；揭盲映射仅保存在包外metadata/reveal.json。本轮尚无独立人工评分，不能认定某裁判更准确。', '', '当前完成状态以metadata/completion_audit.json为准；完整输入协议、12份中文prompt、原始API响应、请求参数及哈希均保留。未成功项不计为零分。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(summary)

if __name__=='__main__':main()
