"""Portable continuous-scroll input/output review; no blank scoring controls."""
import argparse,html,re,shutil,zipfile
from collections import defaultdict
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',action='store_true');a=p.parse_args()
    items=sorted(rows(ROOT/'metadata/inputs.jsonl'),key=lambda r:r['id']);models=read(ROOT/'metadata/freeze.json')['models']
    bundle=ROOT/'review/scroll';assets=bundle/'assets';assets.mkdir(parents=True,exist_ok=True)
    inventory=[];selections={r['id']:r['record']for r in rows(ROOT/'metadata/selection.jsonl')}if(ROOT/'metadata/selection.jsonl').exists()else{}
    decision_path=ROOT/'metadata/selection_decision.json'
    if not decision_path.exists():decision_path=ROOT/'metadata/tripvvt_selection_decision.json'
    decisions={r['id']:r for r in read(decision_path)['decisions']}if decision_path.exists()else{}
    observations=read(ROOT/'metadata/visual_observations.json').get('samples',{})if(ROOT/'metadata/visual_observations.json').exists()else{}
    limitations=read(ROOT/'metadata/task_limitations.json').get('cases',{})if(ROOT/'metadata/task_limitations.json').exists()else{}
    esc=lambda s:html.escape(str(s),quote=True)
    def put(source,name):
        source=Path(source);dst=assets/name
        if dst.exists():assert sha(dst)==sha(source)
        else:shutil.copy2(source,dst)
        return 'assets/'+name
    def video(path,label):return f'<figure><figcaption>{esc(label)}</figcaption><video controls muted playsinline preload="metadata" src="{esc(path)}"></video></figure>'
    groups=defaultdict(list)
    for item in items:groups[item.get('review_group',item['group'])].append(item)
    nav=''.join(f'<a href="#group-{esc(g)}">{esc(g)} ({len(rs)})</a> 'for g,rs in sorted(groups.items()))
    options=''.join(f'<option value="#{esc(r["id"])}">{esc(r["id"])}</option>'for r in items)
    warning='新增TripVVT为看过输出后筛选：要求A849编辑成功且保持良好，H3允许失败。全部32候选及排除原因独立保留；不可用v3/v4均值差声称模型进步。'if len(items)==116 else'32条固定候选全部保留；待完成逐条审核后选择8条。A849必须编辑成功且保持良好，H3允许失败。'
    if (ROOT/'metadata/selection_decision.json').exists():
        decision=read(ROOT/'metadata/selection_decision.json');warning+=f" 已完成审核，状态：{decision['status']}；入选：{', '.join(decision['selected'])}；不足：{decision['shortfall']}。"
    header=f'<header id="top"><strong>{esc(ROOT.name)} · {len(items)} tasks</strong><nav>{nav}</nav><label>跳转样本 <select onchange="location.hash=this.value">{options}</select></label> <a href="#top">顶部</a> <a href="input_audit.html">输入审核</a></header>'
    links=' '.join(f'<a href="{name}">{esc(label)}</a>'for name,label in [('report.md','六项质量报告'),('selection_report.md','32条选样说明'),('per_task_report.csv','逐任务指标与审核'),('per_video_metrics.csv','逐视频六项指标')]if(ROOT/'metadata'/name).exists())
    header=header.replace('</header>','<p>'+links+'</p></header>')
    sections=[];input_sections=[];completed=0
    names=dict(a849='A849 · 40 steps',h3_pdd8='H3 Ref2VA · PDD8 · TP2/Ulysses2',ltx_base='LTX base · devV7 · 40 steps',vace='VACE')
    first_group={rs[0]['id']:g for g,rs in groups.items()}
    for index,item in enumerate(items):
        sid=item['id'];media={};parts=[]
        if sid in first_group:parts.append(f'<h2 id="group-{esc(first_group[sid])}">{esc(first_group[sid])}</h2>')
        title=f'<h3 id="{esc(sid)}">{esc(sid)} <a href="#top">↑</a></h3>'
        provenance=f'UID: {item["uid"]} | old: {item.get("legacy_task_id")} | pilot: {item.get("pilot_id")} | {item.get("task_origin","")}'
        text=title+f'<p class="provenance">{esc(provenance)}</p><div class="prompt">{esc(item["prompt"]["en"])}</div>'
        if sid in limitations:text+='<p><b>输入限制：</b>'+esc(limitations[sid]['limitation'])+'</p>'
        candidate_id=item.get('pilot_id',sid)
        if candidate_id in decisions and (len(items)==32 or item.get('pilot_id')):
            d=decisions[candidate_id];text+=f'<p><b>32条候选选样：'+('入选'if d['selected']else'未入选')+'</b> · '+esc('; '.join(d['reasons'])or'通过A849成功＋良好保持门槛，按比较优先级及冻结ID选择')+'</p>'
        source=put(ROOT/'inputs/prepared'/sid/'source.mp4',f'{sid}_source.mp4');media['source']=source
        raw_source=put(item['source']['absolute_path'],f'original_source_{item["source"]["sha256"][:20]}.mp4')
        text+=f'<p><a href="{esc(raw_source)}">完整原始 SOURCE（独立播放）</a></p>'
        inputs=[video(source,'SOURCE · canonical 0–4s')]
        refs=[]
        for n,ref in enumerate(item['references']):
            path=put(ref['absolute_path'],f'{sid}_reference_{n}{Path(ref["absolute_path"]).suffix}');refs.append(path)
            inputs.append(f'<figure><figcaption>REFERENCE {n+1} · {esc(ref["type"])}</figcaption><img src="{esc(path)}" alt="reference"></figure>')
        media['references']=refs;caption=''
        if item['group']=='tripvvt':
            record=selections[sid];gt=put(record['target']['absolute_path'],f'{sid}_gt.mp4');media['gt']=gt;media['official_caption']=record['caption']
            inputs.append(video(gt,'TARGET / GT · audit only; never inference input'))
            text+=f'<p><a href="{esc(gt)}">完整原始 GT（独立播放，仅审阅）</a></p>'
            caption=f'<p><b>Official caption（完整，审阅用）</b></p><div class="prompt">{esc(record["caption"])}</div>'
        outputs=[];media['outputs']={}
        for model in sorted(models,key=lambda m:names[m]):
            receipt=ROOT/'outputs'/model/f'{sid}.json'
            if not receipt.exists()or read(receipt)['status']!='complete':outputs.append(f'<p>{esc(names[model])}: pending</p>');continue
            r=read(receipt);path=Path(r['browser_path'])if model=='h3_pdd8'else ROOT/'outputs'/model/f'{sid}.mp4'
            relative=put(path,f'{sid}_{model}.mp4');media['outputs'][model]=relative;completed+=1
            block=video(relative,names[model]);note=observations.get(sid,{}).get('models',{}).get(model,{})
            if note.get('reviewed')and note.get('output_sha256')==r['output_sha256']:
                annotation=f'<p>{esc(note["edit"])} / {esc(note["preservation"])}</p><p>{esc(note["notes"])}</p>'
                block=block.replace('</figure>',annotation+'</figure>')
            outputs.append(block)
        examples=''
        if index<2:
            c=read(ROOT/'conditions'/sid/'condition.json');ex=[]
            for key,label in [('overlay_video','VACE source mask overlay (example)'),('prepared_input','VACE RGB128 prepared input (example)')]:
                path=c.get(key)
                if path and Path(path).exists():ex.append(video(put(path,f'{sid}_{key}.mp4'),label))
            # Native VACE receipts record the actual gray128 conditioning file.
            vpath=ROOT/'conditions'/sid/f'conditioning_input_{sid}.mp4'
            if vpath.exists()and not c.get('prepared_input'):ex.append(video(put(vpath,f'{sid}_prepared.mp4'),'VACE RGB128 prepared input (example)'))
            examples='<div class="media">'+''.join(ex)+'</div>'if ex else''
        controls='<button onclick="playGroup(this)">同步播放此样本 0–4s</button>'
        input_block=text+caption+controls+'<div class="media">'+''.join(inputs)+'</div>'+examples
        input_sections.append('<section>'+''.join(parts)+input_block+'</section>')
        sections.append('<section>'+''.join(parts)+input_block+'<div class="media outputs">'+''.join(outputs)+'</div></section>')
        inventory.append(dict(id=sid,group=item['group'],uid=item['uid'],prompt=item['prompt'],provenance=item.get('provenance'),legacy_task_id=item.get('legacy_task_id'),pilot_id=item.get('pilot_id'),media=media))
    style='body{margin:0;background:#181c22;color:#eee;font:15px system-ui}header{position:sticky;top:0;background:#252c36;padding:10px;z-index:5}nav a{display:inline-block;margin:5px}a{color:#9ecbff}section{padding:20px;border-bottom:2px solid #495564;scroll-margin-top:150px}h2,h3{scroll-margin-top:150px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere;background:#262e38;padding:12px}.media{display:flex;flex-wrap:wrap;gap:10px}figure{flex:1 1 240px;margin:8px 0;min-width:180px;max-width:600px}video,img{display:block;width:100%;max-height:430px;object-fit:contain;background:black}figcaption{padding:6px;font-weight:bold}.provenance{overflow-wrap:anywhere;color:#bdc8d5}button{padding:8px;margin-top:10px}'
    js='function playGroup(b){let vs=[...b.closest("section").querySelectorAll("video")];vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})}document.querySelectorAll("video").forEach(v=>v.addEventListener("timeupdate",()=>{if(v.currentTime>4){v.pause();v.currentTime=4}}));'
    def page(parts):return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+esc(ROOT.name)+'</title><style>'+style+'</style>'+header+'<p>'+esc(warning)+'</p>'+''.join(parts)+'<script>'+js+'</script></html>'
    (bundle/'index.html').write_text(page(sections));(bundle/'input_audit.html').write_text(page(input_sections));atomic(bundle/'inventory.json',inventory)
    shutil.copy2(ROOT/'metadata/inputs.jsonl',bundle/'frozen_inputs.jsonl')
    for name in ['report.md','selection_report.md','per_video_metrics.csv','per_task_report.csv','group_metrics.json','temporal_flickering_static_paired.json','selection_decision.json','tripvvt_selection_decision.json','pair_reviews.json','renumbering.json','visual_observations.json','scope.json','task_limitations.json','freeze.json','training_overlap.json','native_provenance_audit.json','reference_processing.json','reuse_outputs.json','final_validation.json','generation_gate.json','input_review.json','new16_input_review.json','new16_mask_refinement.json','inference_protocols.json']:
        source=ROOT/'metadata'/name
        if source.exists():shutil.copy2(source,bundle/name)
    # Verify every local link from both offline pages before producing an archive.
    for pagepath in [bundle/'index.html',bundle/'input_audit.html']:
        for linkpath in re.findall(r'(?:src|href)="([^"]+)"',pagepath.read_text()):
            if linkpath.startswith('#'):continue
            assert (bundle/html.unescape(linkpath)).is_file(),linkpath
    atomic(ROOT/'metadata/review_build.json',dict(status='passed',tasks=len(items),completed_outputs=completed,expected_outputs=len(items)*len(models),offline_links_checked=True,ordered_ids=[r['id']for r in items]))
    if a.package:
        assert completed==len(items)*len(models),'Do not label a partial package final'
        if len(items)==116:assert read(ROOT/'metadata/final_validation.json')['status']=='passed'
        else:assert read(ROOT/'metadata/selection_decision.json')['status']in ['complete','shortfall']
        path=ROOT/'review'/f'{ROOT.name}_scroll.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as archive:
            for source in sorted(bundle.rglob('*')):
                if source.is_file():archive.write(source,source.relative_to(bundle))
        with zipfile.ZipFile(path)as archive:assert archive.testzip()is None
        atomic(ROOT/'metadata/package.json',dict(status='complete',path=str(path),bytes=path.stat().st_size,tasks=len(items),outputs=completed,zip_crc='passed'))
        print(path,flush=True)
    print('Review built',completed,'/',len(items)*len(models),flush=True)

if __name__=='__main__':main()
