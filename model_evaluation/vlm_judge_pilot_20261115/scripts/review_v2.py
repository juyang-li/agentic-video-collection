"""Named, browser-compatible review; consumes frozen results without rescoring."""
import concurrent.futures,html,json,re,shutil
from fractions import Fraction
import av
from PIL import Image
from pilot import ROOT,BENCH,load,save,sha,MODELS

DEST=ROOT/'review_v2'
LABELS={'ltx_base':'LTX Base','a849':'A849','vace':'VACE','h3_pdd8':'H3 PDD8'}
JUDGES={'gpt':'GPT-6.1 Sol (high)','deepseek':'DeepSeek V4.1 Flash','qwen':'Qwen3-VL-235B Instruct'}
CSS='''body{font:16px system-ui,sans-serif;background:#f4f6f8;color:#17212b;margin:76px 24px}nav{position:fixed;top:0;left:0;right:0;background:#fff;padding:15px;z-index:9;box-shadow:0 2px 5px #bbb;overflow:auto;white-space:nowrap}nav a{margin:0 10px}section{scroll-margin-top:75px;margin-bottom:50px}.inputs,.outputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article,.panel{background:white;border:1px solid #ccd3dc;border-radius:8px;padding:14px}video{width:100%;height:320px;background:#111;object-fit:contain}figure{margin:0}figure img{max-width:100%;height:320px;object-fit:contain}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccd3dc;padding:9px;vertical-align:top}td:first-child{width:22%}.prompt{white-space:pre-wrap;padding:16px;background:#e8eef4}.toolbar{position:sticky;top:55px;background:#eef3f8;padding:10px;z-index:5}.toolbar input{width:220px}button{padding:8px;margin:3px;cursor:pointer}.error{color:#b00020;white-space:pre-wrap}.cards img{width:100%;height:auto}.cards{max-width:1400px;margin:auto}.note{color:#526371}@media(max-width:850px){.inputs,.outputs{grid-template-columns:1fr}body{margin:90px 10px}video{height:280px}}'''

def esc(x):return html.escape(str(x))
def copy(src,name):
    p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():shutil.copyfile(src,p)
    return name

def encode(job):
    name,src=job;out=DEST/'media'/f'{name}.mp4';receipt=ROOT/'metadata/v2_proxies'/f'{name}.json'
    original=sha(src)
    if out.exists() and receipt.exists():
        r=load(receipt)
        if r['original_sha256']==original and r['proxy_sha256']==sha(out):return r
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.part.mp4')
    with av.open(str(src)) as inp,av.open(str(tmp),'w',options={'movflags':'+faststart'}) as dest:
        s=inp.streams.video[0];s.thread_count=1
        stream=dest.add_stream('libx264',rate=12);stream.width=s.codec_context.width;stream.height=s.codec_context.height;stream.pix_fmt='yuv420p';stream.thread_count=1
        stream.options={'crf':'16','preset':'fast','profile':'main','level':'4.1'}
        count=0
        for n,f in enumerate(inp.decode(s)):
            if n==0:f.to_image().save(DEST/'media'/f'{name}.jpg',quality=95)
            f=f.reformat(format='yuv420p');f.pts=n;f.time_base=Fraction(1,12)
            for packet in stream.encode(f):dest.mux(packet)
            count+=1
        for packet in stream.encode():dest.mux(packet)
    assert count==49,(name,count)
    tmp.replace(out)
    with av.open(str(out)) as c:
        s=c.streams.video[0];s.thread_count=1
        assert s.codec_context.format.name=='yuv420p' and s.codec_context.profile=='Main'
        decoded=list(c.decode(s));assert len(decoded)==49
        assert all(abs(float(f.pts*f.time_base)-i/12)<1e-5 for i,f in enumerate(decoded))
    r={'name':name,'original_path':str(src),'original_sha256':original,'proxy':f'media/{name}.mp4','proxy_sha256':sha(out),'frames':49,'fps':12,'codec':'h264','profile':'Main','pixel_format':'yuv420p','crf':16,'level':'4.1','faststart':True}
    save(receipt,r);print('encoded',name,flush=True);return r

def video(name,label,master=False):
    return f'<figure><figcaption>{esc(label)}</figcaption><video controls muted playsinline preload="none" poster="media/{name}.jpg" src="media/{name}.mp4" data-src="media/{name}.mp4" {"data-master" if master else ""}></video><p class="error"></p></figure>'

JS=r'''
let active=null,busy=false;
function videos(s){return [...s.querySelectorAll('video')];}
function pause(s){videos(s).forEach(v=>v.pause());s.dataset.playing='0';}
async function ready(v){if(!v.src)v.src=v.dataset.src;if(v.readyState>=2)return;await new Promise((ok,bad)=>{v.addEventListener('loadeddata',ok,{once:true});v.addEventListener('error',()=>bad(Error(v.dataset.src+' 播放失败，媒体错误码 '+v.error?.code)),{once:true});v.load();});}
async function control(b,op){let s=b.closest('section');try{if(active&&active!==s)pause(active);active=s;await Promise.all(videos(s).map(ready));if(op==='pause'){pause(s);return;}if(op==='reset'){pause(s);videos(s).forEach(v=>v.currentTime=0);return;}await Promise.all(videos(s).map(v=>v.play()));s.dataset.playing='1';}catch(e){pause(s);s.querySelector('.status').textContent=e.message;}}
async function seekSlider(input){let s=input.closest('section');await Promise.all(videos(s).map(ready));videos(s).forEach(v=>v.currentTime=Number(input.value));}
document.querySelectorAll('video').forEach(v=>{v.addEventListener('error',()=>{v.parentElement.querySelector('.error').textContent='无法播放 '+v.dataset.src+'；媒体错误码 '+v.error?.code+'。请完整解压后重试，或点独立打开。';});v.addEventListener('pointerdown',()=>ready(v).catch(()=>{}),{once:true});v.addEventListener('play',()=>{let s=v.closest('section');if(active&&active!==s)pause(active);active=s;});v.addEventListener('seeked',()=>{let s=v.closest('section');if(s.dataset.playing==='1'&&v.hasAttribute('data-master')&&!busy){busy=true;videos(s).forEach(x=>{if(x!==v&&Math.abs(x.currentTime-v.currentTime)>1/12)x.currentTime=v.currentTime;});busy=false;}});});
function tick(){if(active&&active.dataset.playing==='1'){let vs=videos(active),m=active.querySelector('[data-master]');active.querySelector('input').value=Math.min(4,m.currentTime);if(m.currentTime>=4){pause(active);vs.forEach(v=>v.currentTime=4);}else vs.forEach(v=>{if(v!==m&&Math.abs(v.currentTime-m.currentTime)>1/12)v.currentTime=m.currentTime;});}requestAnimationFrame(tick);}tick();
'''

def main():
    DEST.mkdir(exist_ok=True);cs=load(ROOT/'metadata/candidates.json');tasks=load(ROOT/'metadata/selection.json')['tasks']
    jobs=[]
    for t in tasks:
        jobs.append((t['id']+'_source',BENCH/'inputs/prepared'/t['id']/'source.mp4'))
        if t['group']=='tripvvt':jobs.append((t['id']+'_gt',BENCH/'inputs/prepared'/t['id']/'gt_review_only.mp4'))
    jobs += [(c['task_id']+'_'+c['model'],c['output']) for c in cs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:proxies=list(pool.map(encode,jobs))
    save(DEST/'display_manifest.json',proxies)
    provenance={r['id']:r['record'] for r in [json.loads(l) for l in (BENCH/'metadata/selection.jsonl').read_text().splitlines()]}
    sections=[];evidence={};score_rows=[]
    for t in tasks:
        sid=t['id'];items=sorted([c for c in cs if c['task_id']==sid],key=lambda c:MODELS.index(c['model']))
        p=f'<section id="{sid}"><h2>{sid} · {esc(t["group"])}</h2><div class="prompt">{esc(t["prompt"]["en"])}</div><div class="toolbar"><button onclick="control(this,\'play\')">同步播放</button><button onclick="control(this,\'pause\')">暂停</button><button onclick="control(this,\'reset\')">重置</button><input type="range" min="0" max="4" step="0.0833333333333" value="0" oninput="seekSlider(this)"><span>共同区间0–4秒</span><span class="status error"></span></div><div class="inputs"><div class="panel">'+video(sid+'_source','SOURCE',True)
        full=copy(t['source']['absolute_path'],f'media/{sid}_source_full.mp4');p+=f'<a href="{full}" target="_blank">完整原始SOURCE（独立查看）</a></div><div class="panel">'
        if not items[0]['references']:p+='<p>REFERENCE：无，按文字编辑。</p>'
        for k,ref in enumerate(items[0]['references']):
            url=copy(ROOT/ref['path'],f'media/{sid}_ref{k}.png');p+=f'<figure><figcaption>REFERENCE {esc(ref["type"])}</figcaption><img src="{url}"></figure>'
        p+='</div></div>'
        if t['group']=='tripvvt':
            p+='<div class="panel">'+video(sid+'_gt','GT（仅供人工审核，未输入VLM）')+f'<p class="prompt">官方caption：{esc(provenance[sid]["caption"])}</p></div>'
        p+='<div class="outputs">'
        for c in items:
            name=sid+'_'+c['model'];p+=f'<article id="{name}"><h3>{LABELS[c["model"]]}</h3>'+video(name,LABELS[c['model']])+f'<p><a target="_blank" href="media/{name}.mp4">独立打开视频</a> · <a target="_blank" href="evidence.html?case={name}">查看VLM送评帧（16个时间点）</a></p><table><tr><th>裁判</th><th>IF / EA</th><th>CP</th></tr>'
            for j,label in JUDGES.items():
                p+=f'<tr><td>{label}</td>'
                for metric in ['edit_achievement','content_preservation']:
                    r=load(ROOT/'metadata/parsed'/f'{j}_{c["id"]}_{metric}_r0.json');assert r['status']=='ok'
                    score_rows.append({'candidate_id':c['id'],'task_id':sid,'model':c['model'],'judge':j,'metric':metric,'score':r['score'],'reason':r['reason']})
                    reason=esc(r['reason']);reason=re.sub(r'P(\d{2})',lambda m:f'<a target="_blank" href="evidence.html?case={name}#P{m[1]}">P{m[1]}</a>',reason)
                    p+=f'<td><b>{r["score"]}/5</b><p>{reason}</p></td>'
                p+='</tr>'
            p+='</table></article>'
            evidence[name]={'title':sid+' · '+LABELS[c['model']],'cards':[]}
            for i,f in enumerate(c['cards']):
                url=copy(ROOT/f['path'],f'evidence/{name}_P{i:02d}.jpg');evidence[name]['cards'].append({'src':url,'frame_index':f['frame_index'],'time_seconds':f['time_seconds'],'sha256':f['sha256']})
        p+='</div><a href="#top">返回顶部</a></section>';sections.append(p)
    nav='<nav><a href="#top">顶部</a>'+''.join(f'<a href="#{t["id"]}">{t["id"]}</a>' for t in tasks)+'</nav>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>VLM试评 V2 · 实名</title><style>'+CSS+'</style><body id="top">'+nav+'<h1>VLM试评 V2：实名视频对比</h1><p class="note">播放的是H.264 Main / yuv420p兼容副本；评分使用原始输出帧。16个时间点的左右对照图仅在独立附录中显示。IF/EA为目标编辑达成，CP为其他内容保持；均为VLM的1–5分。未收集人工评分。</p>'+''.join(sections)+'<script>'+JS+'</script></body></html>'
    (DEST/'index.html').write_text(page)
    appendix='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VLM实际送评帧</title><style>'''+CSS+'''</style><body><a href="index.html">返回主审核页</a><h1 id="title"></h1><p>这是VLM实际接收的16张timecard，每张SOURCE在左、OUTPUT在右，共32幅画面；不是32次评分。</p><div class="cards" id="cards"></div><script>const cases='''+json.dumps(evidence,ensure_ascii=False).replace('</','<\/')+''';let key=new URLSearchParams(location.search).get('case');let c=cases[key];if(c){document.querySelector('#title').textContent=c.title;c.cards.forEach((x,i)=>{let f=document.createElement('figure');f.id='P'+String(i).padStart(2,'0');let label=document.createElement('figcaption');label.textContent=f.id+' · frame '+x.frame_index+' · '+x.time_seconds.toFixed(3)+' s';let img=document.createElement('img');img.loading='lazy';img.src=x.src;f.append(label,img);document.querySelector('#cards').append(f);});if(location.hash)document.querySelector(location.hash)?.scrollIntoView();}else document.querySelector('#title').textContent='请从主审核页选择一个结果';</script></body></html>'''
    (DEST/'evidence.html').write_text(appendix);save(DEST/'scores.json',score_rows)
    for p in (ROOT/'prompts').glob('*.md'):copy(p,f'prompts/{p.name}')
    (DEST/'README.txt').write_text('完整解压后打开index.html。模型及裁判全部实名；送评帧在独立evidence.html附录。主播放器均为兼容浏览副本，评分基于原始帧，原始文件和评分记录保留在项目中。\n')
    assert len(score_rows)==240 and len(proxies)==51
    print('Built V2:51 proxies,240 scores,40 evidence cases',flush=True)

if __name__=='__main__':main()
