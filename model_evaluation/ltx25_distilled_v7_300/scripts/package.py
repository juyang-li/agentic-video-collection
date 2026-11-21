"""CPU final audit, standalone review and actual timing report. GT is used only here."""
from common import *
import av,html,re,shutil,zipfile,time,collections

def main():
    started=time.time();items=rows(ROOT/'metadata/inputs.jsonl');dest=ROOT/'review';dest.mkdir(exist_ok=True);(dest/'media').mkdir(exist_ok=True)
    gt={r['id']:r for r in rows(FROZEN/'ground_truth.jsonl')};records=[];inventory={};sections=[];audits=[]
    def asset(path,expected):
        path=Path(path);name=expected+path.suffix.lower();rel='media/'+name;target=dest/rel
        if rel not in inventory:
            assert sha(path)==expected,str(path)
            if not target.exists():shutil.copy2(path,target)
            assert sha(target)==expected
            inventory[rel]={'sha256':expected,'bytes':target.stat().st_size}
        return rel
    def panel(label,url):
        title=html.escape(label)
        if Path(url).suffix in ['.mp4','.mov','.webm','.mkv']:
            return f'<figure><figcaption>{title}</figcaption><video controls muted playsinline preload="none" src="{url}"></video><a href="{url}">完整视频</a></figure>'
        return f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{url}"><a href="{url}">完整图片</a></figure>'
    for item in items:
        sid=item['id'];p=ROOT/f'outputs/{sid}.json';record=read(p) if p.exists() else {'id':sid,'status':'not_run','failure_reason':'No output record; GPU job ended before sample'}
        done=record['status']=='complete'
        if done:
            assert record['contract']==contract(item) and record['new_run'] is True
            assert record['protocol_audit']['actual_updates']==8 and record['protocol_audit']['condition_tokens_unchanged']
            assert record['protocol_audit']['reference_order']==[[r['type'],r.get('view','')] for r in item['references']]
            path=ROOT/f'outputs/{sid}.mp4';assert sha(path)==record['output_sha256']
            with av.open(str(path)) as c:
                assert float(c.streams.video[0].average_rate)==12
                n=0
                for frame in c.decode(video=0):
                    assert [frame.width,frame.height]==item['canvas_wh'];n+=1
                assert n==item['num_frames']==record['full_decode_frames']
            audits.append({'id':sid,'frames':n,'sha256':record['output_sha256'],'actual_updates':8,'condition_tokens_unchanged':True})
        records.append(record)
        body=panel('SOURCE',asset(FROZEN/item['source']['path'],item['source']['sha256']))
        for i,r in enumerate(item['references']):body+=panel(f'参考{i+1} / {r["type"]} / {r.get("view","")}',asset(FROZEN/r['path'],r['sha256']))
        g=gt[sid];assert g['uid']==item['uid'];body+=panel('GT（仅审核）',asset(FROZEN/g['target']['path'],g['target']['sha256']))
        body+=panel('LTX2.5 Base / distilled DiT＋V7 / 本批新运行',asset(ROOT/f'outputs/{sid}.mp4',record['output_sha256'])) if done else '<p>输出状态：'+html.escape(record['status']+' '+record.get('failure_reason',''))+'</p>'
        caption='<details><summary>官方审核caption（未输入推理）</summary><p>'+html.escape(g['caption'])+'</p></details>' if g.get('caption') else ''
        instructions='<details><summary>有序子指令</summary><pre>'+html.escape(json.dumps(item['ordered_instructions'],ensure_ascii=False,indent=2))+'</pre></details>' if item.get('ordered_instructions') else ''
        end=item['temporal_window']['end_seconds']-item['temporal_window']['start_seconds']
        assert abs(end-(item['num_frames']-1)/12)<1e-6
        sections.append(f'<section id="{sid}" data-search="{html.escape(sid+" "+item["uid"]+" "+item["prompt"]["en"],quote=True)}" data-end="{end}"><h2>{html.escape(sid+" / "+item["uid"]+" / "+item["dataset"])}</h2><p>{html.escape(item["prompt"]["en"])}</p><p>{html.escape(item["prompt"].get("zh",""))}</p>{instructions}{caption}<p>冻结比较区间：0–{end:g}秒；{item["num_frames"]}帧 / 12fps。视频编码时长可多一帧间隔。</p><button class="play">同步播放</button> <button class="pause">暂停</button> <input class="seek" type="range" min="0" max="{end}" step="0.083333333333" value="0"><div class="grid">{body}</div><p>人工观察项目：编辑完成、内容保持、时序、画质；待人工审核，无自动评分。</p></section>')
    good=[r for r in records if r['status']=='complete'];failed=[{'id':r['id'],'status':r['status'],'reason':r.get('failure_reason')} for r in records if r['status']!='complete']
    def summary(group):
        return {'count':len(group),'denoise_sum_seconds':sum(r['denoise_seconds'] for r in group),'decode_sum_seconds':sum(r['decode_seconds'] for r in group),'processing_sum_seconds':sum(r['elapsed_seconds'] for r in group),'mean_processing_seconds':sum(r['elapsed_seconds'] for r in group)/len(group) if group else None}
    runtime=[read(p) for p in sorted((ROOT/'metadata').glob('runtime_*.json'))]
    first={next(r['id'] for r in sorted(good,key=lambda x:x['started_unix']) if r['shard']==shard) for shard in range(4) if any(r['shard']==shard for r in good)}
    historical=[]
    for r in good:
        path=BASE/f'outputs/ltx_base/{r["id"]}.json'
        if not path.exists():continue
        old=read(path);contract_old=old.get('contract',{})
        if old.get('status')!='complete' or contract_old.get('num_frames')!=r['num_frames'] or contract_old.get('canvas_wh')!=r['canvas_wh']:continue
        # Match the exact historical frozen input identity before comparing timings.
        old_input=next((x for x in rows(BASE/'metadata/inputs.jsonl') if x['id']==r['id']),None)
        current=next(x for x in items if x['id']==r['id'])
        if not old_input or old_input['uid']!=current['uid'] or old_input.get('timestamps_seconds')!=current['timestamps_seconds'] or old_input.get('fps')!=current['fps'] or old_input['prompt']!=current['prompt'] or old_input['source']['sha256']!=current['source']['sha256'] or [x['sha256'] for x in old_input['references']]!=[x['sha256'] for x in current['references']]:continue
        a=old['denoise_seconds']+old['decode_seconds'];b=r['denoise_seconds']+r['decode_seconds']
        historical.append({'id':r['id'],'historical_stage_seconds':a,'distilled_stage_seconds':b,'speedup':a/b,'historical_status_sha256':sha(path)})
    wall=max((r['completed_unix'] for r in good),default=0)-min((r['started_unix'] for r in good),default=0)
    report={'experiment':'LTX2.5 Base（distilled DiT）＋DataMake V7','completed':len(good),'total':300,'all_new_runs':True,'overall':summary(good),'excluding_first_per_worker':summary([r for r in good if r['id'] not in first]),'first_per_worker_ids':sorted(first),'per_dataset':{d:summary([r for r in good if r['dataset']==d]) for d in ['ue','coinve','tripvvt']},'per_group':{d:summary([r for r in good if r['group']==d]) for d in sorted({r['group'] for r in good})},'four_worker_observed_processing_wall_seconds':wall,'observed_samples_per_minute':len(good)*60/wall if wall else None,'runtime':runtime,'historical_pairs':historical,'historical_pair_count':len(historical),'historical_paired_speedup':sum(x['historical_stage_seconds'] for x in historical)/sum(x['distilled_stage_seconds'] for x in historical) if historical else None,'dev_300_speedup':None,'failures':failed,'limitations':'Historical pairs only, not a same-session controlled benchmark; software, cache and contention differ. No fabricated 300-sample dev baseline. Four-worker throughput includes smoke gating and per-sample processing, excludes queue/model load/package/upload. Quality requires human review.'}
    atomic(ROOT/'metadata/speed_report.json',report);atomic(dest/'speed_report.json',report)
    (ROOT/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));shutil.copy2(ROOT/'results.jsonl',dest/'results.jsonl')
    table=['# LTX2.5 Base（distilled DiT）＋V7：300条实测','',f'完成 {len(good)}/300，全为本批新运行；先导10条未复用。',f'四worker实测处理墙钟 {wall:.2f}秒（含门槛串行阶段，排队、模型加载、打包与上传另列）。',f'有严格输入配对的历史dev仅 {len(historical)}条；该子集加速比为 {report["historical_paired_speedup"]}，不是300条dev加速比。','历史软件版本、缓存和资源竞争有差异，不是严格同场基准；编辑效果待人工审核。','模型加载与软件版本见speed_report.json/runtime；逐条总耗时、显存及阶段计时见results.jsonl。','','| ID | 状态 | 去噪(s) | VAE(s) | 写盘(s) | 验证(s) | 总处理(s) |','|---|---|---:|---:|---:|---:|---:|']
    for r in records:table.append('| '+' | '.join([r['id'],r['status'],*[f'{r[k]:.3f}' if k in r else '—' for k in ['denoise_seconds','vae_decode_seconds','write_seconds','validation_seconds','elapsed_seconds']]])+' |')
    (ROOT/'SPEED_REPORT.md').write_text('\n'.join(table)+'\n');shutil.copy2(ROOT/'SPEED_REPORT.md',dest/'SPEED_REPORT.md')
    nav=' '.join(f'<a href="#{next(i["id"] for i in items if i["group"]==g)}">{g}</a>' for g in dict.fromkeys(i['group'] for i in items))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LTX2.5 Base distilled＋V7 · 300条</title><style>body{font:16px system-ui;margin:0;background:#12171f;color:#eef}header,section{padding:24px}nav{position:sticky;top:0;background:#273344;padding:12px;z-index:4}a{color:#9cf;margin:5px}section{border-bottom:2px solid #667;scroll-margin-top:110px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}figure{margin:0}video,img{width:100%;max-height:420px;object-fit:contain;background:#000}button{padding:8px}pre{white-space:pre-wrap}.seek{width:45%}</style><header><h1>LTX2.5 Base（distilled DiT）＋DataMake V7 · 300条</h1><p>官方完整蒸馏DiT；无A849/蒸馏LoRA/其他LoRA。BF16、原画幅单阶段，12fps、按冻结清单取帧。</p><p>权重SHA256：31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4；revision：49b093a39c19169a8a81721dcc541e07e945232f。</p><p>ancestral Euler，8次更新，CFG1/STG0/rescale0/ref_cfg1，eta1/s_noise1，seed42/noise seed10042/decode seed42。sigma=[1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0]。参考→SOURCE→生成区，GT与caption仅审核。</p><p>完整蒸馏训练算法尚未确认；ancestral Euler是推理采样器。无自动评分，技术通过不等于人工效果通过。<a href="SPEED_REPORT.md">速度报告</a><a href="results.jsonl">逐条记录</a></p></header>'''+f'<nav>{nav}<p>筛选 ID / UID / 英文指令：<input id="filter" type="search"></p></nav>'+''.join(sections)+'''<script>document.querySelector('#filter').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!s.dataset.search.toLowerCase().includes(q))};document.querySelectorAll('section').forEach(s=>{const vs=[...s.querySelectorAll('video')],seek=s.querySelector('.seek'),end=+s.dataset.end;s.querySelector('.play').onclick=()=>vs.forEach(v=>{v.currentTime=+seek.value;v.play().catch(()=>{})});s.querySelector('.pause').onclick=()=>vs.forEach(v=>v.pause());seek.oninput=()=>vs.forEach(v=>v.currentTime=+seek.value);if(vs[0])vs[0].ontimeupdate=()=>{if(vs[0].currentTime>=end){vs.forEach(v=>v.pause());seek.value=end}else{seek.value=vs[0].currentTime;if(!vs[0].paused)vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-vs[0].currentTime)>.15)v.currentTime=vs[0].currentTime})}}});</script></html>'''
    (dest/'index.html').write_text(page)
    atomic(dest/'media_inventory.json',inventory);atomic(ROOT/'metadata/output_audit.json',{'status':'passed' if len(good)==300 else 'partial','outputs':audits})
    for name in ['preflight.json','weights.json','cpu_tests.json','implementation.json','run_status.json']:
        if (ROOT/'metadata'/name).exists():shutil.copy2(ROOT/'metadata'/name,dest/name)
    shutil.copy2(ROOT/'README.md',dest/'README.md')
    urls=re.findall(r'(?:href|src)="([^"]+)"',page)
    assert len(sections)==300 and all(u.startswith('#') or (dest/u).is_file() for u in urls)
    zipname='ltx25_distilled_v7_300_review.zip' if len(good)==300 else 'ltx25_distilled_v7_300_partial_review.zip'
    archive=ROOT/zipname;tmp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,Path('review')/p.relative_to(dest))
    tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    atomic(ROOT/'metadata/package_status.json',{'status':'complete' if len(good)==300 else 'partial','successful_outputs':len(good),'total':300,'zip_path':str(archive),'zip_sha256':sha(archive),'zip_crc_verified':True,'offline_links_verified':True,'media_count':len(inventory),'seconds':time.time()-started})
    print('PACKAGE_READY',len(good),'/300',flush=True)
if __name__=='__main__':main()
