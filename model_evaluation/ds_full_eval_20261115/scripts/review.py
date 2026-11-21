"""40-task named offline review; display encodes never become judge inputs."""
import argparse, concurrent.futures, csv, html, json, re, shutil, zipfile
from fractions import Fraction
from pathlib import Path
import av
from engine import ROOT, PKG, V4, METRICS, read, rows, save

DEST=ROOT/'review'
LABELS={'ltx_distilled_v7':'LTX 2.5 Distilled v7','a849_v7':'A849 v7','h3_pdd8':'H3 PDD8','vace_lightx2v4':'VACE LightX2V 4-step','ltx_base':'LTX 2.5 Dev Base','a849':'A849','vace':'VACE','ltx_real232':'Real-only LoRA step232 (Distilled)','ltx_synthetic600':'Synthetic-only LoRA step600 (Distilled)'}
CSS='''body{font:16px/1.5 system-ui,sans-serif;background:#f4f6f8;color:#17212b;margin:130px 24px 30px}nav{position:fixed;top:0;left:0;right:0;background:#fff;padding:12px;z-index:9;box-shadow:0 2px 5px #bbb;max-height:95px;overflow:auto}nav a{display:inline-block;margin:3px 8px}section{scroll-margin-top:125px;margin-bottom:50px}.inputs,.outputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article,.panel{background:white;border:1px solid #ccd3dc;border-radius:8px;padding:14px}video{width:100%;height:340px;background:#111;object-fit:contain}figure{margin:0}figure img{max-width:100%;height:320px;object-fit:contain}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccd3dc;padding:9px;vertical-align:top}.prompt{white-space:pre-wrap;padding:16px;background:#e8eef4}.toolbar{position:sticky;top:120px;background:#eef3f8;padding:10px;z-index:5}.toolbar input{width:220px}button{padding:8px;margin:3px;cursor:pointer}.error{color:#b00020;white-space:pre-wrap}.cards{max-width:1500px;margin:auto}.cards img{width:100%;height:auto}.note{color:#526371}@media(max-width:850px){.inputs,.outputs{grid-template-columns:1fr}body{margin:145px 10px}video{height:280px}}'''
JS=r'''
let active=null,busy=false;
const videos=s=>[...s.querySelectorAll('video[data-sync]')];
function pause(s){videos(s).forEach(v=>v.pause());s.dataset.playing='0';}
async function ready(v){if(v.readyState>=2)return;await new Promise((ok,bad)=>{v.addEventListener('loadeddata',ok,{once:true});v.addEventListener('error',()=>bad(Error('视频无法播放：'+v.getAttribute('src'))),{once:true});v.load();});}
async function control(b,op){let s=b.closest('section');try{if(active&&active!==s)pause(active);active=s;await Promise.all(videos(s).map(ready));if(op==='pause'){pause(s);return;}if(op==='reset'){pause(s);videos(s).forEach(v=>v.currentTime=0);return;}await Promise.all(videos(s).map(v=>v.play()));s.dataset.playing='1';}catch(e){pause(s);s.querySelector('.status').textContent=e.message;}}
async function seekSlider(x){let s=x.closest('section');await Promise.all(videos(s).map(ready));videos(s).forEach(v=>v.currentTime=Number(x.value));}
document.querySelectorAll('video').forEach(v=>{v.addEventListener('error',()=>{v.parentElement.querySelector('.error').textContent='播放失败，请完整解压或独立打开视频。';});v.addEventListener('play',()=>{let s=v.closest('section');if(active&&active!==s)pause(active);active=s;});v.addEventListener('seeked',()=>{let s=v.closest('section');if(v.hasAttribute('data-master')&&!busy){busy=true;videos(s).forEach(x=>{if(x!==v&&Math.abs(x.currentTime-v.currentTime)>1/12)x.currentTime=v.currentTime;});busy=false;}});});
function tick(){if(active&&active.dataset.playing==='1'){let m=active.querySelector('[data-master]'),end=Number(active.dataset.end);active.querySelector('input[type=range]').value=Math.min(end,m.currentTime);if(m.currentTime>=end){pause(active);videos(active).forEach(v=>v.currentTime=end);}else videos(active).forEach(v=>{if(v!==m&&Math.abs(v.currentTime-m.currentTime)>1/12)v.currentTime=m.currentTime;});}requestAnimationFrame(tick);}tick();
'''
def esc(x):return html.escape(str(x))
def copy(src,name):
    dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():shutil.copyfile(src,dst)
    return name
def encode(job):
    name,src,n,indices=job;dst=DEST/'media'/f'{name}.mp4';receipt=ROOT/'metadata/review_proxies'/f'{name}.json'
    if dst.exists() and receipt.exists():return read(receipt)
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.part.mp4');wanted=set(range(n) if indices is None else indices);count=0
    with av.open(str(src)) as inp,av.open(str(tmp),'w',options={'movflags':'+faststart'}) as dest:
        s=inp.streams.video[0];s.thread_count=1;stream=dest.add_stream('libx264',rate=12);stream.width=s.codec_context.width+(s.codec_context.width%2);stream.height=s.codec_context.height+(s.codec_context.height%2);stream.pix_fmt='yuv420p';stream.thread_count=1;stream.options={'crf':'16','preset':'fast','profile':'main'}
        for i,frame in enumerate(inp.decode(s)):
            if i in wanted:
                if count==0:frame.to_image().save(DEST/'media'/f'{name}.jpg',quality=95)
                f=frame.reformat(width=stream.width,height=stream.height,format='yuv420p');f.pts=count;f.time_base=Fraction(1,12)
                for packet in stream.encode(f):dest.mux(packet)
                count+=1
            if i>=max(wanted):break
        for packet in stream.encode():dest.mux(packet)
    assert count==n,(name,count,n);tmp.replace(dst)
    r={'name':name,'source':str(src),'path':f'media/{name}.mp4','frames':n,'fps':12,'input_frame_indices':indices,'codec':'h264','profile':'Main','pixel_format':'yuv420p','crf':16,'faststart':True,'browser_compatibility_testing':'skipped_user_requested'};save(receipt,r);print('REVIEW',name,flush=True);return r
def video(name,label,master=False):
    return f'<figure><figcaption>{esc(label)} · <a target="_blank" href="media/{name}.mp4">独立打开</a></figcaption><video controls muted playsinline preload="none" data-sync {"data-master" if master else ""} poster="media/{name}.jpg" src="media/{name}.mp4"></video><p class="error"></p></figure>'
def main(hair24=False):
    global DEST
    if hair24:DEST=ROOT/'review_hair24'
    assert read(ROOT/'report/full/coverage.json')['successful_scores']==4528
    assert read(ROOT/'report/full/stability_summary.json')['completed']==220
    DEST.mkdir(exist_ok=True);sels=read(ROOT/'metadata/selections.json');all_tasks=read(ROOT/'metadata/tasks.json')
    tasks=[t for t in all_tasks if (t['dataset']=='v4_116' and t['group']=='hair') ] if hair24 else [t for t in all_tasks if t['key'] in sels['review']]
    tasks.sort(key=lambda t:(t['dataset'],t['group'],t['key']))
    assert len(tasks)==(24 if hair24 else 40)
    if hair24:
        assert all(len(t['outputs'])==4 for t in tasks)
        selection={'dataset':'v4_116','group':'hair','selection':'all24, no score-based exclusion','tasks':[t['key'] for t in tasks],'models':['ltx_base','a849','vace','h3_pdd8'],'scores_reused':True,'new_api_calls':0}
        save(ROOT/'metadata/hair24_selection.json',selection)
    gt={r['id']:r for r in rows(PKG/'ground_truth.jsonl')};vprovenance={r['id']:r['record'] for r in rows(V4/'metadata/selection.jsonl')};original={r['id']:r for r in rows(PKG/'inputs.jsonl')};jobs=[];gts={};captions={}
    for t in tasks:
        sid=t['key'];n=t['num_frames'];jobs.append((sid+'_source',t['source'],n,None))
        for out in t['outputs']:jobs.append((sid+'_'+out['model'],out['path'],n,None))
        for j,r in enumerate(t['references']):
            if r['type']=='motion':jobs.append((sid+f'_ref{j}',r['path'],n,r['sample_frame_indices']))
        if t['group']=='tripvvt':
            if t['dataset']=='frozen300':
                g=gt[t['id']];jobs.append((sid+'_gt',PKG/g['target']['path'],n,g['sample_frame_indices']));captions[sid]=g['caption']
            else:
                prepared=V4/'inputs/prepared'/t['id']/'gt_review_only.mp4'
                assert prepared.exists(),prepared
                jobs.append((sid+'_gt',prepared,n,None));captions[sid]=vprovenance[t['id']]['caption']
            gts[sid]=True
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:proxies=list(pool.map(encode,jobs))
    save(DEST/'display_manifest.json',proxies)
    sections=[];evidence={};score_rows=[];groups=collections_groups(tasks)
    for t in tasks:
        sid=t['key'];n=t['num_frames'];end=(n-1)/12;cases=read(ROOT/'metadata/prepared'/f'{sid}.json');case_by_model={c['model']:c for c in cases}
        subgroup=f' · {esc(t.get("subgroup") or "其他发型参考")}' if hair24 else ''
        p=f'<section id="{sid}" data-end="{end}"><h2>{esc(sid)} · {esc(t["group"])}{subgroup}</h2><div class="prompt">{esc(t["prompt"]["en"])}</div><div class="toolbar"><button onclick="control(this,\'play\')">同步播放</button><button onclick="control(this,\'pause\')">暂停</button><button onclick="control(this,\'reset\')">重置</button><input type="range" min="0" max="{end}" step="0.0833333333333" value="0" oninput="seekSlider(this)"><span>0–{end:.3f}秒，{n}帧/12fps</span><span class="status error"></span></div><div class="inputs"><div class="panel">'+video(sid+'_source','SOURCE',True)
        raw=original[t['id']]['source'] if t['dataset']=='frozen300' else None
        fullsrc=PKG/raw['path'] if raw else next(r for r in rows(V4/'metadata/inputs.jsonl') if r['id']==t['id'])['source']['absolute_path']
        fullurl=copy(fullsrc,f'media/{sid}_source_full.mp4');p+=f'<a href="{fullurl}" target="_blank">完整原始SOURCE</a></div><div class="panel">'
        if not t['references']:p+='<p>REFERENCE：无，按文字编辑。</p>'
        for j,r in enumerate(t['references']):
            if r['type']=='motion':p+=video(sid+f'_ref{j}',f'REFERENCE {j+1} · 动作')
            else:
                asset=next(a for a in cases[0]['references'] if a['label'].startswith(f'R{j+1}:'));url=copy(ROOT/asset['path'],f'media/{sid}_ref{j}.png');p+=f'<figure><figcaption>REFERENCE {j+1} · {esc(r["type"])} · {esc(r["view"])}</figcaption><img loading="lazy" src="{url}"></figure>'
        p+='</div></div>'
        if sid in gts:p+='<div class="panel">'+video(sid+'_gt','GT（仅供人工审核，未输入Judge）')+f'<p class="prompt">官方caption（仅审核）：{esc(captions[sid])}</p></div>'
        p+='<div class="outputs">'
        order=['ltx_distilled_v7','ltx_base','a849_v7','a849','vace_lightx2v4','vace','h3_pdd8','ltx_real232','ltx_synthetic600']
        ordered=sorted(t['outputs'],key=lambda o:order.index(o['model']))
        for out in ordered:
            c=case_by_model[out['model']];name=sid+'_'+out['model'];k=len(c['cards']);label=LABELS[out['model']]
            p+=f'<article><h3>{label}</h3>'+video(name,label)+f'<p><a href="evidence.html?case={name}" target="_blank">Judge送评帧附录（{k}个时间点）</a></p><table><tr><th>DeepSeek V4.1 Flash · Wafer</th><th>分数</th><th>理由</th></tr>'
            for metric,title in [('edit_achievement','IF'),('content_preservation','CP')]:
                r=read(ROOT/'results'/f'{c["key"]}__{metric}__r0.json');assert r['status']=='ok';score_rows.append({'task':sid,'model':out['model'],'metric':metric,'score':r['score'],'reason':r['reason']});reason=re.sub(r'P(\d{2})',lambda m:f'<a target="_blank" href="evidence.html?case={name}#P{m[1]}">P{m[1]}</a>',esc(r['reason']));p+=f'<tr><td>{title}</td><td>{r["score"]}/5</td><td>{reason}</td></tr>'
            p+='</table></article>';evidence[name]={'title':sid+' · '+label,'cards':[],'references':[]}
            for i,a in enumerate(c['cards']):
                url=copy(ROOT/a['path'],f'evidence/{name}_P{i:02d}.jpg');evidence[name]['cards'].append({'src':url,'label':a['label'],'sha256':a['sha256']})
            for i,a in enumerate(c['references']):
                url=copy(ROOT/a['path'],f'evidence/{sid}_reference{i}'+Path(a['path']).suffix);evidence[name]['references'].append({'src':url,'label':a['label'],'sha256':a['sha256']})
        sections.append(p+'</div><a href="#top">返回顶部</a></section>')
    nav='<nav><a href="#top">顶部</a>'+''.join(f'<a href="#group_{g}">{esc(g)}</a>' for g in groups)+''.join(f'<a href="#{t["key"]}">{esc(t["key"])}</a>' for t in tasks)+'</nav>'
    body='';previous=None
    for t,p in zip(tasks,sections):
        g=t['dataset']+'_'+t['group']
        if g!=previous:body+=f'<h2 id="group_{g}">{esc(g)}</h2>';previous=g
        body+=p
    (DEST/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek完整评测 · 40任务抽查</title><style>'+CSS+'</style><body id="top">'+nav+'<h1>DeepSeek评测：40任务实名抽查</h1><p>IF为目标编辑达成，CP为非目标内容保持，均为1–5分。主页面连续展开，无人工评分控件。每张Judge配对图左SOURCE、右OUTPUT，实际送评图仅在附录显示。CP采用统一新版规则；未编辑但保留来源内容可以得到CP=5。</p><p class="note">浏览视频为H.264 Main/yuv420p/CRF16/faststart副本，评分使用冻结原始输出帧。冻结300与V4分别汇总，模型版本不同；V4部分TripVVT经过查看输出后选样。详细浏览器兼容测试按用户要求跳过。</p><p><a href="report/full/summary.csv">全量汇总</a> · <a href="report/full/pairs.csv">全量逐对结果</a> · <a href="report/full/comparisons.csv">A849配对比较</a></p>'+body+'<script>'+JS+'</script></body></html>')
    if hair24:
        page=(DEST/'index.html').read_text().replace('DeepSeek完整评测 · 40任务抽查','自建116 · 全部24个发型任务').replace('DeepSeek评测：40任务实名抽查','自建V4 116集：全部24个发型任务').replace('全量汇总','发型24任务汇总').replace('全量逐对结果','发型96对结果')
        page=page.replace('<p>IF为目标编辑达成','<p>包含全部24个发型任务，16个celeb_reference与8个其他参考；每任务四模型，共96个编辑结果。复用正式评分，不重新调用API。A849发型CP均分3.625，其中13条不超过3分。</p><p>IF为目标编辑达成',1)
        (DEST/'index.html').write_text(page)
    data=json.dumps(evidence,ensure_ascii=False).replace('</','<\\/')
    (DEST/'evidence.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Judge实际送评图</title><style>'+CSS+'</style><body><a href="index.html">返回主审核页</a><h1 id="title"></h1><p>每张P配对图对应一个时间点（左SOURCE、右OUTPUT）；不是多次评分。REFERENCE仅输入IF，未输入CP。</p><div class="cards" id="cards"></div><script>const cases='+data+''';let c=cases[new URLSearchParams(location.search).get('case')];if(c){document.querySelector('#title').textContent=c.title;for(let [kind,items] of [['REFERENCE',c.references],['P',c.cards]])items.forEach((x,i)=>{let f=document.createElement('figure');f.id=kind+String(i).padStart(2,'0');let label=document.createElement('figcaption');label.textContent=x.label;let img=document.createElement('img');img.loading='lazy';img.src=x.src;f.append(label,img);document.querySelector('#cards').append(f);});if(location.hash)document.querySelector(location.hash)?.scrollIntoView();}else document.querySelector('#title').textContent='请从主审核页选择结果';</script></body></html>''')
    save(DEST/'review_scores.json',score_rows)
    for f in (ROOT/'prompts').glob('*.md'):copy(f,'prompts/'+f.name)
    if hair24:
        target=DEST/'report/full';target.mkdir(parents=True,exist_ok=True)
        for name in ['scores.csv','pairs.csv','summary.csv','comparisons.csv']:
            with (ROOT/'report/full'/name).open() as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;selected=[r for r in reader if r['dataset']=='v4_116' and r['group']=='hair']
            with (target/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(selected)
        for c in [c for t in tasks for c in read(ROOT/'metadata/prepared'/f'{t["key"]}.json')]:
            for metric in METRICS:
                name=f'{c["key"]}__{metric}__r0.json';copy(ROOT/'results'/name,'judge_results/'+name)
        copy(ROOT/'metadata/hair24_selection.json','protocol/selection.json')
    else:
        shutil.copytree(ROOT/'report',DEST/'report',dirs_exist_ok=True)
    for name in ['protocol.json','selections.json','cp_revision.json','checkpoint100_protocol.json','checkpoint100_review.json','camera_revision.json','ablation_models.json','extended_scope.json','completion_audit.json']:
        p=ROOT/'metadata'/name
        if p.exists():copy(p,'protocol/'+name)
    (DEST/'README.txt').write_text('完整解压后打开index.html。30冻结任务×六模型＋10自建任务×四模型，共220对视频。IF与CP均为DeepSeek评分，无人工评分表。原始API结果、费用与抽帧记录保留在项目中。报告见report/full；Judge实际图片见独立附录。\n')
    if hair24:(DEST/'README.txt').write_text('完整解压后打开index.html。自建V4 116集全部24个发型任务，每个任务包含SOURCE、发型参考、完整指令及LTX基座/A849/VACE/H3四模型视频、IF与CP评分理由。共96个结果、192项正式评分，复用原分数，未增加API调用。分组参考为16个celeb_reference和8个其他参考。主页面不铺抽帧，实际送评图在evidence.html独立附录。report/full仅含发型子集，judge_results含原始评分响应。视频为H.264 Main/yuv420p浏览副本；按用户要求跳过细致浏览器兼容测试。\n')
    archive=ROOT/('ds_full_eval_20261115_v4_hair24_review.zip' if hair24 else 'ds_full_eval_20261115_review40.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted(DEST.rglob('*')):
            if f.is_file():z.write(f,Path(DEST.name)/f.relative_to(DEST))
    save(ROOT/'metadata'/('hair24_review_delivery.json' if hair24 else 'review_delivery.json'),{'tasks':len(tasks),'pairs':sum(len(t['outputs']) for t in tasks),'scores':len(score_rows),'display_clips':len(proxies),'zip':str(archive),'browser_checks':'skipped_user_requested'})
    print('REVIEW ZIP',archive,flush=True)
def collections_groups(tasks):return list(dict.fromkeys(t['dataset']+'_'+t['group'] for t in tasks))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hair24',action='store_true');main(parser.parse_args().hair24)
