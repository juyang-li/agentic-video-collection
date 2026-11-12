"""Offline scroll review; complete prompts, references and evaluation-only GT."""
import argparse
import html
import shutil
import zipfile
from common import *
from review_layout import show_condition_preview, task_navigation, NAV_CSS

def build_diagnostics(out,inventory):
    """Keep the six reference-swap probes separate from the formal100."""
    by_id={c['id']:c for c in inventory}
    cards=[];records=[]
    esc=lambda x:html.escape(str(x))
    for model in ('ltx_base','a849','vace'):
        for sid in ('identity_01','whole_outfit_01'):
            state=read(ROOT/'outputs'/model/f'{sid}_swapref.json')
            assert state['status']=='complete'
            alt_id=state['control_reference_from']
            original=by_id[sid];alternative=by_id[alt_id]
            media=[original['media'][0]]
            for label,card in [('ORIGINAL REFERENCE',original),('SWAPPED REFERENCE',alternative)]:
                for m in card['media']:
                    if m['role'].startswith('REFERENCE'):
                        media.append({**m,'role':label+' '+m['role']})
            media.append({'role':model+' original reference output','video':True,'path':f'media/{sid}_{model}.mp4'})
            dst=out/'media'/f'{sid}_{model}_swapref.mp4'
            shutil.copyfile(ROOT/'outputs'/model/f'{sid}_swapref.mp4',dst)
            media.append({'role':model+' swapped reference output','video':True,'path':str(dst.relative_to(out))})
            panels=''.join('<figure><figcaption>'+esc(m['role'])+'</figcaption>'+('<video controls preload="none" src="'+esc(m['path'])+'"></video>' if m['video'] else '<img loading="lazy" src="'+esc(m['path'])+'">')+'</figure>' for m in media)
            cards.append('<article><h2>'+model+' · '+sid+'</h2><p class="prompt">'+esc(original['prompt']['en'])+'</p><p>seed42；只替换参考：'+sid+' → '+alt_id+'。SOURCE、文字、mask、采样参数均固定。替换参考没有对应GT，不用原GT评价新目标。</p><div class="media">'+panels+'</div></article>')
            records.append({'model':model,'sample_id':sid,'reference_from':alt_id,'generation':state,'media':media})
    atomic(out/'reference_controls.json',records)
    (out/'reference_controls.html').write_text('<!doctype html><meta charset="utf-8"><title>固定seed参考对照</title><style>body{font:16px/1.5 system-ui;background:#14171b;color:#edf1f4;padding:24px}.prompt{white-space:pre-wrap}.media{display:flex;flex-wrap:wrap;gap:12px}figure{width:360px;margin:0}video,img{width:100%;max-height:540px;object-fit:contain;background:black}article{margin:30px 0;border-top:1px solid #667;padding-top:16px}</style><h1>六个固定seed换参考诊断</h1><p>不计入100条正式样本。对照只检查参考条件与模型响应，不以像素差异代替编辑完成判断。</p>'+''.join(cards))


def build(kind, allow_login=False):
    if not allow_login:
        require_job()
    from media import video_frames,write_preview
    selected=rows(ROOT/'metadata/selection.jsonl')
    inputs={x['id']:x for x in rows(ROOT/'metadata/inputs.jsonl')}
    name='a100_'+kind+'_scroll_20261111'
    out=ROOT/'review'/name
    out.mkdir(parents=True,exist_ok=True)
    inventory=[]
    sample_indices={x["id"]:i for i,x in enumerate(selected)}
    def build_card(selected_item):
        item=inputs[selected_item['id']]
        card={k:item[k] for k in ('id','uid','group','subgroup','dataset','prompt')}
        card['caption']=selected_item['record'].get('caption')
        card['media']=[]
        roles=[('SOURCE',item['source']),*[(f'REFERENCE {i+1} · {r["type"]} · {r.get("view", "")}',r) for i,r in enumerate(item['references'])],('GT',selected_item['record']['target'])]
        for i,(role,r) in enumerate(roles):
            video=role in ('SOURCE','GT') or r.get('type')=='motion'
            dst=out/'media'/f'{item["id"]}_{i:02d}{".mp4" if video else Path(r["absolute_path"]).suffix}'
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                verified_input_preview=ROOT/'review/a100_inputs_scroll_20261111/media'/dst.name
                if kind!='inputs' and verified_input_preview.is_file():
                    shutil.copyfile(verified_input_preview,dst)
                elif video:
                    frames,details=video_frames(item,r)
                    tmp=dst.with_suffix('.part.mp4')
                    write_preview(frames,tmp,12)
                    tmp.replace(dst)
                    atomic(ROOT/'metadata/previews'/f'{item["id"]}_{i:02d}.json',details)
                else: shutil.copyfile(r['absolute_path'],dst)
            card['media'].append({'role':role,'video':video,'path':str(dst.relative_to(out)), 'original':r['absolute_path']})
        if kind in ('conditions','results'):
            cpath=ROOT/'conditions'/item['id']/'condition.json'
            if cpath.exists():
                condition=read(cpath)
                card['vace_condition']=condition
                prepared=ROOT/'conditions'/item['id']/f'conditioning_input_{item["id"]}.mp4'
                if prepared.exists() and show_condition_preview("prepared_input", sample_indices[item["id"]]):
                    dst=out/'media'/f'{item["id"]}_vace_prepared_input.mp4';shutil.copyfile(prepared,dst)
                    card['media'].append({'role':'VACE prepared input (editable RGB=128)','video':True,'path':str(dst.relative_to(out))})
                    card['vace_condition']['repainting_input']='official masked RGB fill128; preview only; inference uses original pixels outside mask'
                for key in ('overlay_video','pose_video'):
                    if condition.get(key) and show_condition_preview(key, sample_indices[item['id']]):
                        p=Path(condition[key]);dst=out/'media'/f'{item["id"]}_vace_{key}.mp4';shutil.copyfile(p,dst)
                        card['media'].append({'role':'VACE '+key,'video':True,'path':str(dst.relative_to(out))})
        if kind=='results':
            card['generations']={m:read(ROOT/'outputs'/m/f'{item["id"]}.json') for m in ('ltx_base','a849','vace')}
            assert all(v['status']=='complete' for v in card['generations'].values())
            for model in ('ltx_base','a849','vace'):
                p=ROOT/'outputs'/model/f'{item["id"]}.mp4'
                if not p.is_file(): raise RuntimeError(f'Missing formal output {p}')
                dst=out/'media'/f'{item["id"]}_{model}.mp4';shutil.copyfile(p,dst)
                card['media'].append({'role':model,'video':True,'path':str(dst.relative_to(out))})
        # Rebuilding into an existing folder must not retain old optional previews in the ZIP.
        for role in ('prepared_input', 'overlay_video'):
            if not show_condition_preview(role, sample_indices[item['id']]):
                (out/'media'/f'{item["id"]}_vace_{role}.mp4').unlink(missing_ok=True)
        return card
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        for card in pool.map(build_card, selected):
            inventory.append(card)
            atomic(ROOT/'metadata/package_progress.json',{'kind':kind,'completed':len(inventory),'total':100})
    atomic(out/'manifest.json',inventory)
    shutil.copyfile(ROOT/'metadata/sampling.json',out/'sampling.json')
    if kind=='results':
        build_diagnostics(out,inventory)
        shutil.copyfile(ROOT/'metadata/input_observability.json',out/'input_observability.json')
        shutil.copyfile(ROOT/'metadata/mask_limitations.json',out/'mask_limitations.json')
    mask_notes=read(ROOT/'metadata/mask_limitations.json')['notes'] if kind=='results' else {}
    esc=lambda x:html.escape(str(x))
    sections=[]
    for g in GROUPS:
        cards=[]
        for x in inventory:
            if x['group']!=g:continue
            m=''.join('<figure><figcaption>'+esc(v['role'])+'</figcaption>'+(
                '<video controls preload="none" playsinline src="'+esc(v['path'])+'"></video>' if v['video'] else '<img loading="lazy" src="'+esc(v['path'])+'">')+'</figure>' for v in x['media'])
            prompts=''.join('<p class="prompt">'+esc(v)+'</p>' for v in x['prompt'].values() if v)
            caption='<p>完整官方 caption（仅评测侧）：'+esc(x['caption'])+'</p>' if x.get('caption') else ''
            condition=('<p class="prompt">VACE 适配文字：'+esc(x['vace_condition']['adapted_prompt'])+'</p><details><summary>输入条件与处理记录</summary><pre>'+esc(json.dumps(x['vace_condition'],ensure_ascii=False,indent=2))+'</pre></details>') if x.get('vace_condition') else ''
            if x['id'] in mask_notes:
                condition='<p>条件局限：'+esc(mask_notes[x['id']])+'</p>'+condition
            cards.append('<article><h3>'+esc(x['id'])+' · '+esc(x['uid'])+'</h3>'+prompts+caption+'<p>来源：'+esc(x['dataset'])+'；子类：'+esc(x['subgroup'] or '—')+'</p><button onclick="playCard(this)">同步从头播放本条视频</button><div class="media">'+m+'</div>'+condition+'</article>')
        sections.append('<section id="task-'+esc(g)+'"><h2>'+LABELS[g]+'</h2>'+''.join(cards)+'</section>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A100 单轮审核</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#14171b;color:#edf1f4;margin:24px}h1,h2,h3{line-height:1.3}.prompt,pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #54606c;border-radius:8px;margin:24px 0;padding:16px}.media{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:360px}video,img{width:100%;max-height:560px;object-fit:contain;background:#000}button{margin:8px 0;padding:8px}figcaption{font-weight:bold}
'''+NAV_CSS+'''</style><div id="top"></div>'''+task_navigation(GROUPS,LABELS)+'''<h1>100 条单轮审核 · '''+kind+'''</h1><p>seed=20261111；仅正式 test_manifest；UE 80＋CoinVE 10＋TripVVT 10。源/GT/动作参考统一取 0–4 秒的 49 个时刻，12fps；横屏 1280×736，竖屏 736×1280，保持比例补边。包含 t=4 秒的端点帧，编码播放长度为 49/12 秒。图片展示完整原图，LTX 编码按原生协议最小补边。</p><p>TripVVT source 含合成换装，真实数据集来源不等于全部原生实拍。镜头组只评价前 4 秒运动，不要求完成 10 秒完整环绕。GT 只供评测；镜头组清单已有的目标首帧是合法参考。</p><p>无评分控件，样本未被标记为人工通过。评估时分别判断编辑完成、内容保持、时序和画质。动作组 VACE 使用参考 pose＋源首/中/末帧外观参考，与 LTX 的原生动作参考输入不同。</p>'''+''.join(sections)+'''<script>
function playCard(b){document.querySelectorAll('video').forEach(v=>v.pause());const vs=[...b.closest('article').querySelectorAll('video')];vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});if(vs.length){vs[0].onpause=()=>vs.forEach(v=>v.pause());vs[0].onseeked=()=>vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-vs[0].currentTime)>.15)v.currentTime=vs[0].currentTime});}}
</script></html>'''
    if kind=='results':
        page=page.replace('<h1>100 条单轮审核', '<p><a href="reference_controls.html">固定seed换参考诊断（六条，独立于正式100条）</a></p><h1>100 条单轮审核')
    (out/'index.html').write_text(page)
    # Validate every referenced file and decode every generated preview before ZIP.
    import av
    checked=set()
    validation_cards=inventory+(read(out/'reference_controls.json') if kind=='results' else [])
    for c in validation_cards:
        for m in c['media']:
            p=out/m['path']
            assert p.is_file() and p.stat().st_size>0
            if m['video'] and p not in checked:
                with av.open(str(p)) as con:
                    frames=list(con.decode(video=0))
                    assert len(frames)==49,(p,len(frames))
                checked.add(p)
    atomic(out/'validation.json',{'status':'passed','cards':len(inventory),'diagnostics':6 if kind=='results' else 0,'videos_full_decoded':len(checked),'scope':'offline media linkage and full decode; no human quality acceptance'})
    zip_path=out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path.with_suffix('.part.zip'),'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out.parent)))
    zip_path.with_suffix('.part.zip').replace(zip_path)
    atomic(ROOT/'metadata'/f'package_{kind}.json',{'status':'complete','zip':str(zip_path),'cards':100})
    print(zip_path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['inputs','conditions','results'],default='inputs');p.add_argument('--allow-login',action='store_true',help='Explicit user authorization on 2026-11-11 for input packaging/upload without waiting for GPUs');a=p.parse_args();assert not a.allow_login or a.kind=='inputs';build(a.kind,a.allow_login)
