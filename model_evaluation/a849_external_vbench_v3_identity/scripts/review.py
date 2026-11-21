"""Offline scroll review, exact output validation, and temporal contact sheets."""
import argparse
import csv
import html
import shutil
import zipfile
from common import *
from contracts import DIMS,LABELS

def run(stage,pack=False):
    import av
    from PIL import Image,ImageDraw,ImageOps
    frozen=read(ROOT/'metadata/freeze.json');inputs=rows(ROOT/'metadata/inputs.jsonl')
    selected=[r for r in inputs if stage=='all' or r['id'] in frozen['smoke']]
    dest=ROOT/'review'/stage;dest.mkdir(parents=True,exist_ok=True)
    media=dest/'media';media.mkdir(exist_ok=True)
    sections=[];validation=[];seen_groups=set()
    score_rows=list(csv.DictReader((ROOT/'metadata/per_video_metrics.csv').open())) if (ROOT/'metadata/per_video_metrics.csv').exists() else []
    cohort='identity_diagnostic' if ROOT.name.endswith('_identity') else 'main'
    metric_map={(r['id'],r['model'],r['dimension']):r['score'] for r in score_rows if r.get('cohort','main')==cohort}
    visual=read(ROOT/'metadata/visual_observations.json').get('samples',{}) if (ROOT/'metadata/visual_observations.json').exists() else {}
    def copy(src,name):
        dst=media/name
        if not dst.exists():shutil.copy2(src,dst)
        return 'media/'+name
    for index,item in enumerate(selected):
        sid=item['id'];figures=[];temporal=[]
        if item['group'] not in seen_groups:
            sections.append(f'<h2 id="group-{item["group"]}">{item["group"]}</h2>');seen_groups.add(item['group'])
        src=ROOT/'inputs/prepared'/sid/'source.mp4'
        videos=[('SOURCE',src)]
        ref_figures=[]
        for i,r in enumerate(item['references']):
            path=Path(r['absolute_path']);url=copy(path,f'{sid}_ref_{i}{path.suffix}')
            ref_figures.append(f'<figure><figcaption>REFERENCE {html.escape(r["type"])}</figcaption><img src="{url}"></figure>')
        # Real TripVVT GT is review-only; custom tasks never synthesize a GT.
        if item['group']=='tripvvt':
            sel=next(r for r in rows(ROOT/'metadata/selection.jsonl') if r['id']==sid)['record']
            from media import video_frames,write_preview
            frames,_=video_frames(item,sel['target']);gt=ROOT/'inputs/prepared'/sid/'gt_review_only.mp4'
            if not gt.exists():write_preview(frames,gt)
            videos.append(('GT (review only)',gt))
        for model in ['ltx_base','a849','vace']:
            p=ROOT/'outputs'/model/f'{sid}.mp4';status=p.with_suffix('.json')
            if status.exists() and read(status).get('status')=='complete':
                assert sha(p)==read(status)['output_sha256'];videos.append((model,p))
            else:figures.append(f'<p>{model}: generation failed/missing; see generation records.</p>')
        for role,path in videos:
            with av.open(str(path)) as c:
                rate=str(c.streams.video[0].average_rate);frames=list(c.decode(video=0))
            assert len(frames)==49 and all([f.width,f.height]==item['canvas_wh'] for f in frames)
            assert rate in ['12','12/1']
            validation.append(dict(id=sid,role=role,path=str(path),frames=len(frames),fps=rate,canvas_wh=item['canvas_wh'],sha256=sha(path)))
            temporal.append((role,[frames[i].to_image() for i in [0,6,12,18,24,30,36,42,48]]))
            url=copy(path,f'{sid}_{role.split()[0].replace("/","_")}.mp4')
            figure=f'<figure><figcaption>{html.escape(role)}</figcaption><video data-sync controls preload="metadata" muted loop src="{url}"></video>'
            if role=='SOURCE':
                original=Path(item['source']['absolute_path'])
                original_url=copy(original,f'{sid}_original_source{original.suffix}')
                figure+=f'<p><a href="{original_url}" target="_blank">独立查看完整原始SOURCE</a></p>'
            elif role.startswith('GT'):
                original=Path(sel['target']['absolute_path'])
                original_url=copy(original,f'{sid}_original_gt{original.suffix}')
                figure+=f'<p><a href="{original_url}" target="_blank">独立查看完整原始GT（仅审核）</a></p>'
            figure+='</figure>'
            if role=='SOURCE':figures.append(figure);figures.extend(ref_figures)
            else:figures.append(figure)
        if index<2:
            for label,path in [('VACE mask overlay',ROOT/'conditions'/sid/'overlay.mp4'),('VACE prepared input',ROOT/'conditions'/sid/f'conditioning_input_{sid}.mp4')]:
                if path.is_file():
                    url=copy(path,f'{sid}_{path.name}');figures.append(f'<figure><figcaption>{label}</figcaption><video controls muted src="{url}"></video></figure>')
        canvas=Image.new('RGB',(1620,len(temporal)*205),'#20242a');draw=ImageDraw.Draw(canvas)
        for row,(role,frames) in enumerate(temporal):
            draw.text((4,row*205+2),sid+' '+role,fill='white')
            for col,f in enumerate(frames):
                canvas.paste(ImageOps.contain(f,(178,168)),(col*180,row*205+22));draw.text((col*180,row*205+190),f'{col*.5:.1f}s',fill='white')
        canvas.save(dest/f'{sid}_temporal.jpg',quality=93)
        caption=''
        if item['group']=='tripvvt':caption='<p>官方 caption（仅审核，不进入模型）：'+html.escape(sel['caption'])+'</p><p>TripVVT指令沿用已有参考图换衣模板；本地独立测试划分，非官方排行榜。</p>'
        actual_vace_prompt=read(ROOT/'conditions'/sid/'condition.json')['adapted_prompt']
        caption+='<p>任务来源：'+html.escape(str(item.get('task_origin','self_created_reference_edit')))+'；旧ID：'+html.escape(str(item.get('old_id')))+'；修改原因：'+html.escape(item.get('revision_reason',''))+'</p>'
        quality='<h3>逐视频质量指标 ↑（只读；Temporal Flickering为全量原始分数，受正常运动影响）</h3><div style="overflow:auto"><table><tr><th>模型</th>'+''.join('<th>'+label+'</th>' for label in LABELS)+'</tr>'
        for model in ['ltx_base','a849','vace']:
            cells=[]
            for dim in DIMS:
                value=metric_map.get((sid,model,dim));cells.append('待生成/评分' if value in [None,''] else f'{float(value):.5f}')
            quality+='<tr><th>'+model+'</th>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>'
        quality+='</table></div><h3>直接视觉编辑观察</h3>'
        for model in ['ltx_base','a849','vace']:
            observation=visual.get(sid,{}).get('models',{}).get(model)
            if observation:
                quality+='<p>'+model+' · '+html.escape(observation['edit'])+' / '+html.escape(observation['preservation'])+'：'+html.escape(observation['notes'])+'</p>'
            else:quality+='<p>'+model+'：待生成或待直接审核。</p>'
        sections.append(f'<article id="{sid}"><h2>{sid} · {item["group"]}</h2><p class="prompt">{html.escape(item["prompt"]["en"])}</p><p class="prompt">{html.escape(item["prompt"].get("zh",""))}</p><p class="prompt">VACE actual adapted prompt: {html.escape(actual_vace_prompt)}</p>{caption}<p><button onclick="playTogether(this)">从0秒同步播放</button> <button onclick="pauseTogether(this)">暂停本条</button></p><div class="media">'+''.join(figures)+'</div>'+quality+'<p>Source: '+html.escape(item['uid'])+'</p><a href="#top">返回顶部</a></article>')
    nav='<nav><a href="#top">顶部</a>'+''.join(f'<a href="#group-{g}">{g}</a>' for g in dict.fromkeys(r['group'] for r in selected))+''.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in selected)+'</nav>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A849 外部编辑评测</title><style>body{font:16px/1.6 system-ui;background:#171d23;color:#eee;margin:22px;padding-top:150px}nav{position:fixed;top:0;left:0;right:0;background:#26323e;padding:12px;max-height:125px;overflow-y:auto;display:flex;gap:10px;flex-wrap:wrap;z-index:10}a{color:#9dd6ff}article{border:1px solid #65707a;margin:24px 0;padding:16px;scroll-margin-top:165px}.media{display:flex;gap:12px;flex-wrap:wrap}figure{width:310px;margin:0}video,img{width:100%;max-height:550px;object-fit:contain;background:black}table{border-collapse:collapse}td,th{border:1px solid #65707a;padding:6px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}</style><div id="top"></div>'''+nav+'''<h1>A849 / LTX2.5 dev / Wan2.1-VACE-14B 外部编辑评测</h1><p>同一冻结 source、编辑语义和参考图。49帧、12fps，取样时刻0–4秒；编码长度49/12秒。保持比例补边1280×736或736×1280。自建任务无GT；TripVVT的真实GT仅审核。无评分控件，无Gemini。技术检查不代表人工质量通过；VBench仅六项custom_input，不是官方榜单或总分。</p><p>视频均静音。各播放器按原始时序播放，不变速。缺失结果如实列出；低质量有效结果保留。</p>'''+''.join(sections)+'</html>'
    visual_count=sum(bool(m.get('reviewed')) for sample in visual.values() for m in sample.get('models',{}).values())
    if visual_count<3*len(selected):page=page.replace('<h1>',f'<p><strong>直接视觉审核{visual_count}/{3*len(selected)}；最终验收尚未完成。</strong></p><h1>',1)
    generation_count=sum(r['role'] in ['ltx_base','a849','vace'] for r in validation)
    if generation_count<3*len(selected):page=page.replace('<h1>',f'<p><strong>运行中快照：有效输出{generation_count}/{3*len(selected)}；生成和评分尚未完成，不计作最终验收。</strong></p><h1>',1)
    if frozen.get('revision')=='background_removed_fullframe_96_v3':
        page=page.replace('<h1>', '<p>96条任务、83段不同source；身份和发型参考均去背景，完整原图不裁剪，身份头发填白。仅复用16条输入相同的任务；Ingredients仍在待办队列。</p><h1>',1)
    if stage=='smoke':
        notice=f'<p><strong>首测预览：当前包含{generation_count}/15条已完成输出（整轮计划72条）。缺失结果逐条标为待生成。此包是运行中的快照，完整评分和最终结论尚未完成。</strong></p>'
        page=page.replace('<h1>A849 / LTX2.5 dev / Wan2.1-VACE-14B 外部编辑评测</h1>','<h1>A849 / LTX2.5 dev / Wan2.1-VACE-14B 外部编辑评测</h1>'+notice)
    elif frozen.get('revision')=='whitehair_fullframe_v2':
        notice='<p>身份参考遵从用户更新：保留完整原图及分辨率，仅分割出的头发填白，不裁剪。用户查看预览后授权直接全量，未要求完成smoke评分关卡。输入完全相同的有效旧结果按记录复用，身份组全部重新生成。</p>'
        page=page.replace('<h1>A849 / LTX2.5 dev / Wan2.1-VACE-14B 外部编辑评测</h1>','<h1>A849 / LTX2.5 dev / Wan2.1-VACE-14B 外部编辑评测</h1>'+notice)
    page=page.replace('</html>','''<script>
let activeVideos=[],syncTimer=null;
function pauseTogether(button){button.closest('article').querySelectorAll('video').forEach(v=>v.pause());}
function playTogether(button){
  if(syncTimer)clearInterval(syncTimer);
  activeVideos.forEach(v=>v.pause());
  activeVideos=[...button.closest('article').querySelectorAll('video[data-sync]')];
  activeVideos.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});
  syncTimer=setInterval(()=>{
    const leader=activeVideos[0];
    if(!leader||leader.paused)return;
    for(const v of activeVideos.slice(1))if(!v.paused&&Math.abs(v.currentTime-leader.currentTime)>.15)v.currentTime=leader.currentTime;
  },100);
}
</script></html>''')
    detail_files=sorted((ROOT/'review').glob('*_face_detail.jpg'))+sorted((ROOT/'review').glob('*_all49.jpg'))
    if stage=='all' and detail_files:
        details=dest/'details';details.mkdir(exist_ok=True)
        for path in detail_files:shutil.copy2(path,details/path.name)
        links=''.join(f'<li><a href="details/{html.escape(p.name)}" target="_blank">{html.escape(p.name)}</a></li>' for p in detail_files)
        extra='<section id="details"><h2>放大及加密抽帧 · 仅审核</h2><p>这些诊断视图用于检查细节，没有改变模型实际输入、完整身份参考图或生成输出。</p><ul>'+links+'</ul></section>'
        page=page.replace('</html>',extra+'</html>')
    page=page.replace('article{border:','[id]{scroll-margin-top:165px}article{border:')
    page=page.replace('<h1>A849 /', '<h1>v3 · A849 /')
    (dest/'index.html').write_text(page)
    atomic(ROOT/'metadata'/f'{stage}_media_validation.json',dict(status='passed' if generation_count==3*len(selected) else 'partial',complete_generations=generation_count,expected_generations=3*len(selected),videos=validation,page=str(dest/'index.html')))
    for name in ['inputs.jsonl','selection.jsonl','freeze.json','training_overlap.json','reference_processing.json','input_visual_review.json','full_launch_authorization.json','reuse_outputs.json','vace_prompt_corrections.json','task_limitations.json','smoke_review.json','visual_smoke_partial.json','per_video_metrics.csv','per_task_report.csv','group_metrics.json','report.md','visual_observations.json','source_clusters.json','revision_log.json','temporal_flickering_static_paired.json','generation_gate.json','vace_prompt_corrections.json','identity_diagnostic.json']:
        src=ROOT/'metadata'/name
        if src.is_file():shutil.copy2(src,dest/name)
    prior=ROOT/'metadata/prior24/inputs.jsonl'
    if prior.is_file():shutil.copy2(prior,dest/'prior_input_manifest.jsonl')
    receipts=dest/'generation_records';receipts.mkdir(exist_ok=True)
    for item in selected:
        for model in ['ltx_base','a849','vace']:
            src=ROOT/'outputs'/model/f'{item["id"]}.json'
            if src.exists():shutil.copy2(src,receipts/f'{model}--{item["id"]}.json')
    raw_scores=ROOT/'metadata/vbench'/stage
    if raw_scores.is_dir():
        scores=dest/'scores';scores.mkdir(exist_ok=True)
        for p in raw_scores.glob('*.json'):shutil.copy2(p,scores/p.name)
    input_page=ROOT/'review/inputs/index.html'
    if input_page.exists():
        input_masks=dest/'input_masks';input_masks.mkdir(exist_ok=True)
        input_html=input_page.read_text()
        for item in selected:
            sid=item['id'];strip=ROOT/'review/inputs'/f'{sid}_mask.jpg'
            shutil.copy2(strip,input_masks/f'{sid}.jpg')
            input_html=input_html.replace(f'{sid}_source.mp4',f'{sid}_SOURCE.mp4').replace(f'{sid}_gt.mp4',f'{sid}_GT.mp4').replace(f'{sid}_mask.jpg',f'input_masks/{sid}.jpg')
            for i in range(len(item['references'])):input_html=input_html.replace(f'{sid}_ref{i}',f'{sid}_ref_{i}')
        (dest/'input_review.html').write_text(input_html)
        page=page.replace('<h1>', '<p><a href="input_review.html">冻结输入及mask审核页</a> · <a href="report.md">六项指标与分组报告</a></p><h1>',1)
        (dest/'index.html').write_text(page)
    # Every browser media link must resolve locally, with no external resources.
    from html.parser import HTMLParser
    class LinkCheck(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key=='src':
                    assert not value.startswith(('http:','https:','/')) and (dest/value).is_file(),value
                elif key=='href' and not value.startswith('#'):
                    assert not value.startswith(('http:','https:','/')) and (dest/value).is_file(),value
    LinkCheck().feed(page)
    if input_page.exists():LinkCheck().feed(input_html)
    if pack:
        output=ROOT/'review'/('a849_external_vbench_v3_identity_smoke_scroll.zip' if stage=='smoke' else 'a849_external_vbench_v3_identity_scroll.zip')
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in dest.rglob('*'):
                if p.is_file():z.write(p,Path('a849_external_vbench_v3_identity')/p.relative_to(dest))
        atomic(ROOT/'metadata'/('smoke_package.json' if stage=='smoke' else 'package.json'),dict(path=str(output),bytes=output.stat().st_size,offline_page='a849_external_vbench_v3_identity/index.html',stage=stage,complete_generations=generation_count))
    print('review',stage,len(validation),'videos',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','all'],required=True);p.add_argument('--pack',action='store_true');a=p.parse_args();run(a.stage,a.pack)
