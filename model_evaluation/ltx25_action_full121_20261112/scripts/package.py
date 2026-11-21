"""Build and fully decode an offline, unscored, full-duration review."""
from common import *
import html,shutil,time,zipfile
from html.parser import HTMLParser

def build():
    require_job(exact=True)
    from media import video_frames,write_preview
    import av
    started=time.time();name='action_full121_base_vs_a849_scroll_20261112'
    out=ROOT/'review'/name;(out/'media').mkdir(parents=True,exist_ok=True)
    selected={r['id']:r for r in rows(ROOT/'metadata/selection.jsonl')}
    historical={(r['model'],r['id']):r['historical_contract'] for r in read(ROOT/'metadata/historical_contracts.json')}
    inventory=[];cards=[];checks=[];generations=[]
    for item in rows(ROOT/'metadata/inputs.jsonl'):
        sid=item['id'];media=[]
        roles=[('SOURCE',item['source']),*[(f'REFERENCE {i+1} · motion',r) for i,r in enumerate(item['references'])],('GT（仅审核）',selected[sid]['record']['target'])]
        for i,(role,record) in enumerate(roles):
            assert sha(record['absolute_path'])==record['sha256']
            dst=out/'media'/f'{sid}_{i:02d}.mp4'
            if not dst.exists():
                frames,audit=video_frames(item,record)
                assert len(frames)==121
                tmp=dst.with_suffix('.part.mp4');write_preview(frames,tmp,12);tmp.replace(dst)
                atomic(ROOT/'metadata/previews'/f'{sid}_{i:02d}.json',audit)
            media.append({'role':role,'path':str(dst.relative_to(out))})
        for model in ('ltx_base','a849'):
            state=read(ROOT/'outputs'/model/f'{sid}.json')
            assert state['status']=='complete' and state['full_decode_frames']==121
            assert state['contract']['num_frames']==121 and state['contract']['fps']==12
            prior=historical[model,sid]
            fixed=('checkpoint','checkpoint_sha256','config_sha256','native_inference_sha256','reference_protocol','steps','seed','cfg','stg','stg_blocks','rescale','ref_cfg','fps','canvas_wh')
            assert all(state['contract'][k]==prior[k] for k in fixed),(model,sid,'changed guidance/model settings')
            assert state['contract']['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
            src=ROOT/'outputs'/model/f'{sid}.mp4';assert sha(src)==state['output_sha256']
            dst=out/'media'/f'{sid}_{model}.mp4';shutil.copyfile(src,dst)
            media.append({'role':model,'path':str(dst.relative_to(out))});generations.append(state)
        for m in media:
            p=out/m['path'];w,h=item['canvas_wh']
            with av.open(str(p)) as c:
                assert c.streams.video[0].average_rate==12
                n=0
                for frame in c.decode(video=0):
                    assert (frame.width,frame.height)==(w,h);n+=1
                assert n==121,(str(p),n)
            checks.append({'path':m['path'],'frames':n,'fps':12,'canvas_wh':[w,h]})
        inventory.append({'id':sid,'uid':item['uid'],'prompt':item['prompt'],'media':media})
        panels=''.join('<figure><figcaption>'+html.escape(m['role'])+'</figcaption><video controls preload="none" playsinline src="'+m['path']+'"></video></figure>' for m in media)
        prompts=''.join('<p class="prompt">'+html.escape(v)+'</p>' for v in item['prompt'].values())
        cards.append('<article id="'+sid+'"><h2>'+sid+' · '+item['uid']+'</h2>'+prompts+'<button onclick="playCard(this)">同步从头播放</button><div class="media">'+panels+'</div><a href="#top">返回顶部</a></article>')
    assert len(generations)==20
    nav='<nav><a href="#top">顶部</a> <a href="#task-action">动作复测</a> '+ ' '.join('<a href="#'+x['id']+'">'+x['id']+'</a>' for x in inventory)+'</nav>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Action 完整时长复测</title><style>body{font:16px/1.55 system-ui;background:#15191e;color:#edf1f4;margin:24px;padding-top:100px}a{color:#8dccff}nav{position:fixed;top:0;left:0;right:0;padding:12px;background:#202934;display:flex;gap:12px;flex-wrap:wrap;z-index:10}[id]{scroll-margin-top:110px}article{border:1px solid #667;padding:16px;margin:24px 0;scroll-margin-top:110px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}.media{display:flex;flex-wrap:wrap;gap:12px}figure{width:360px;margin:0}video{width:100%;max-height:560px;background:black}button{padding:8px}</style><div id="top"></div>'''+nav+'''<h1 id="task-action">Action 完整时长复测 · 基座与 A849</h1><p>沿用历史 action_01–action_10；SOURCE、动作参考、GT 与输出均为 121 帧、12fps，取样时刻 0–10 秒，末端按训练规则保持原视频最后一帧，编码播放长度为 121/12 秒。保持原完整提示词、参考身份、横竖幅、分辨率、seed42、40步和指导参数。GT 仅供审核，不进入推理。</p><p>旧4秒结果见先前审核包 a100_results_scroll_20261111.zip（网盘 /apps/bypy/lance_review/）。本包没有并列旧4秒结果，不能据此宣称已证实改善或时间截取就是原因。无评分表；技术检查通过不代表人工质量通过。</p>'''+''.join(cards)+'''<script>
async function playCard(b){const vs=[...b.closest('article').querySelectorAll('video')];document.querySelectorAll('video').forEach(v=>v.pause());await Promise.all(vs.map(v=>new Promise(resolve=>{if(v.readyState>=1)resolve();else{v.addEventListener('loadedmetadata',resolve,{once:true});v.load();}})));vs.forEach(v=>{v.currentTime=0;v.playbackRate=1});await Promise.all(vs.map(v=>v.play().catch(()=>{})));}
</script></html>'''
    (out/'index.html').write_text(page)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key=='src':assert (out/value).is_file()
    Links().feed(page)
    shutil.copyfile(ROOT/'metadata/preflight.json',out/'preflight.json')
    shutil.copyfile(ROOT/'metadata/historical_contracts.json',out/'historical_contracts.json')
    atomic(out/'manifest.json',inventory);atomic(out/'validation.json',{'status':'passed','outputs':20,'videos_full_decoded':len(checks),'checks':checks,'human_quality':'not assessed'})
    times={m:{'denoise_seconds':sum(g['denoise_seconds'] for g in generations if g['model']==m),'decode_seconds':sum(g['decode_seconds'] for g in generations if g['model']==m),'peak_allocated_bytes':max(max(g['denoise_peak_allocated_bytes'],g['decode_peak_allocated_bytes']) for g in generations if g['model']==m)} for m in ('ltx_base','a849')}
    timing={'models':times,'generation_wall_seconds':max(g['completed_unix'] for g in generations)-min(g['started_unix'] for g in generations),'package_seconds':time.time()-started,'elapsed_before_upload_seconds':time.time()-read(ROOT/'metadata/execution.json')['driver_submitted_unix']}
    atomic(ROOT/'metadata/final_summary.json',timing)
    text='# Action 完整时长复测\n\n20条输出、10条冻结历史样本。121帧、12fps、40步、seed42，原横竖幅与1280×736/736×1280分辨率。source及motion复用身份、SHA256、时间轴和tensor核对后的训练缓存（16 latent帧）。GT仅供审核。\n\n与旧版协议差异：49→121帧、4→10秒、7→16 latent帧；训练端点规则不变。未并列旧结果，不宣称效果改善或因果。技术检查50段视频完整解码及页面媒体链接通过，不代表人工质量通过。\n\n实际计时与显存（bytes，PyTorch峰值分配）：\n\n```json\n'+json.dumps(timing,ensure_ascii=False,indent=2)+'\n```\n\n原始日志、逐条时间及显存见logs/和outputs/。运行仅接入testqueue现有分配199810；首测两个模型action_01后扩展。结果与模型环境保留，无清理、提交或推送。\n\n网盘路径：/apps/bypy/lance_review/'+name+'.zip。上传状态以 review/ 中上传器回执为准；仅上传ZIP，无远端复核。\n'
    (ROOT/'README.md').write_text(text);(out/'README.md').write_text(text)
    with zipfile.ZipFile(out.with_suffix('.part.zip'),'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out.parent)))
    out.with_suffix('.part.zip').replace(out.with_suffix('.zip'))
    atomic(ROOT/'metadata/package.json',{'status':'complete','zip':str(out.with_suffix('.zip')),'completed_unix':time.time()})
    print(out.with_suffix('.zip'),flush=True)
if __name__=='__main__':build()
