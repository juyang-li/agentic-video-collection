"""All24 original inputs, current references and real TripVVT GT; no outputs."""
import html
import shutil
import zipfile
from html.parser import HTMLParser
from common import *

def main():
    require_job()
    import av
    inputs=rows(ROOT/'metadata/inputs.jsonl');frozen=read(ROOT/'metadata/freeze.json')
    assert len(inputs)==96 and sha(ROOT/'metadata/inputs.jsonl')==frozen['input_manifest_sha256']
    selected={r['id']:r for r in rows(ROOT/'metadata/selection.jsonl')}
    processing=read(ROOT/'metadata/reference_processing.json');original_refs={}
    for rec in processing:
        if rec.get('role')=='background':continue
        for key in ['processed_path']:
            if key in rec:original_refs[str(Path(rec[key]).resolve())]=rec
    folder=ROOT/'review/original96';folder.mkdir(parents=True,exist_ok=True)
    media=folder/'media';media.mkdir(exist_ok=True)
    inventory=[];sections=[];groups=set()
    def copy(path,name,expected=None):
        src=Path(path);dst=media/name
        if expected:assert sha(src)==expected
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(dst)==sha(src)
        return 'media/'+name
    def video(path,name,label,expected=None):
        url=copy(path,name,expected)
        with av.open(str(path)) as c:
            stream=c.streams.video[0]
            assert stream.codec_context.name=='h264',f'Browser copy needed: {path}'
            probe=dict(codec=stream.codec_context.name,width=stream.width,height=stream.height,fps=str(stream.average_rate),frames=stream.frames)
        inventory.append(dict(file=url,role=label,original=str(path),sha256=sha(path),**probe))
        return f'<figure><figcaption>{html.escape(label)}</figcaption><video data-sync controls muted preload="metadata" src="{url}"></video><p><a href="{url}" target="_blank">独立查看完整视频</a></p></figure>'
    for index,item in enumerate(inputs):
        sid=item['id'];group=item['group'];figs=[]
        if group not in groups:sections.append(f'<h2 id="group-{group}">{group}</h2>');groups.add(group)
        src=item['source'];figs.append(video(src['absolute_path'],sid+'_original_source.mp4','SOURCE · 原始完整视频',src['sha256']))
        prepared=ROOT/'inputs/prepared'/sid/'source.mp4';prepared_url=copy(prepared,sid+'_model_input49.mp4')
        for i,ref in enumerate(item['references']):
            path=Path(ref['absolute_path']);url=copy(path,f'{sid}_reference{i}{path.suffix}',ref['sha256'])
            inventory.append(dict(file=url,role='actual_reference',id=sid,sha256=ref['sha256'],type=ref['type']))
            figs.append(f'<figure><figcaption>REFERENCE · 实际推理参考 {html.escape(ref["type"])}</figcaption><img src="{url}"></figure>')
            original=original_refs.get(str(path.resolve()))
            if original:
                source=Path(original['path']);url=copy(source,f'{original["name"]}_original_portrait{source.suffix}',original['sha256'])
                figs.append(f'<figure><figcaption>原始参考照片 · 未处理</figcaption><img src="{url}"></figure>')
        caption=''
        if group=='tripvvt':
            r=selected[sid]['record'];target=r['target']
            figs.append(video(target['absolute_path'],sid+'_original_gt.mp4','TARGET · 官方原始GT，仅审核',target['sha256']))
            caption='<p class="prompt"><strong>完整官方 caption（仅审核，不进入推理）：</strong> '+html.escape(r['caption'])+'</p>'
        extra=''
        if index<2:
            for label,p in [('VACE source mask overlay',ROOT/'conditions'/sid/'overlay.mp4'),('VACE prepared RGB input',ROOT/'conditions'/sid/f'conditioning_input_{sid}.mp4')]:
                if p.exists():
                    url=copy(p,sid+'_'+p.name);extra+=f'<figure><figcaption>{label}</figcaption><video controls muted preload="metadata" src="{url}"></video></figure>'
        official=''
        if item.get('original_omni_instruction'):
            official='<p class="prompt"><strong>源视频原有OmniEdit指令（自建任务未用于本次推理）：</strong> '+html.escape(item['original_omni_instruction'])+'</p>'
        actual_vace=read(ROOT/'conditions'/sid/'condition.json')['adapted_prompt']
        sections.append(f'<article id="{sid}"><h2>{sid} · {group}</h2><p class="prompt"><strong>完整实际编辑指令：</strong> {html.escape(item["prompt"]["en"])}</p><p class="prompt">{html.escape(item["prompt"].get("zh",""))}</p>{official}{caption}<p class="prompt"><strong>VACE实际补充描述：</strong> {html.escape(actual_vace)}</p><p><button onclick="playCommon(this)">按绝对时间同步播放0–4秒</button> <button onclick="pauseCommon(this)">暂停本条</button> <a href="{prepared_url}" target="_blank">查看实际49帧模型输入</a></p><div class="media">'+''.join(figs)+extra+'</div><p>来源ID：'+html.escape(item['uid'])+'</p><a href="#top">顶部</a></article>')
    nav='<nav><a href="#top">顶部</a>'+''.join(f'<a href="#group-{g}">{g}</a>' for g in dict.fromkeys(r['group'] for r in inputs))+''.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in inputs)+'</nav>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A849 original24原始素材审核包</title><style>body{font:16px/1.6 system-ui;background:#172029;color:#eee;margin:22px;padding-top:145px}nav{position:fixed;top:0;left:0;right:0;background:#263440;padding:12px;display:flex;gap:10px;flex-wrap:wrap;z-index:10}a{color:#9ed7ff}[id]{scroll-margin-top:160px}article{border:1px solid #64727c;padding:16px;margin:24px 0}.media{display:flex;flex-wrap:wrap;gap:12px}figure{width:340px;margin:0}video,img{width:100%;max-height:580px;object-fit:contain;background:black}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:7px}</style><div id="top"></div>'''+nav+'''<h1>A849 · 全量24条 original原始素材审核包</h1><p>完整24条冻结任务：身份4、发型4、背景4、OmniEdit原生人物编辑6、TripVVT换衣6。仅原始输入及参考资料，无生成结果、无评分控件。原始SOURCE和TripVVT真实GT保留完整长度，并提供实际49帧/12fps模型输入入口。</p><p>身份参考遵从最新要求：完整原图尺寸与构图，仅分割出的头发填白，无裁剪、无缩放，其他像素按位保持。同时展示未处理原照片。发型使用单张正面参考。自建任务没有GT；TripVVT GT及官方caption仅审核，不进入模型。</p><p>同步按钮只播放共同绝对时间0–4秒，不变速、不补尾；各播放器和独立入口可查看完整素材。音频不参与比较，默认静音。VACE诊断实例仅前两条展示。</p>'''+''.join(sections)+'''<script>
let active=[],timer=null;
function pauseCommon(b){b.closest('article').querySelectorAll('video').forEach(v=>v.pause());}
function playCommon(b){if(timer)clearInterval(timer);active.forEach(v=>v.pause());active=[...b.closest('article').querySelectorAll('video[data-sync]')];active.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});timer=setInterval(()=>{const lead=active[0];if(!lead)return;if(lead.currentTime>=4){active.forEach(v=>v.pause());clearInterval(timer);return;}if(!lead.paused)for(const v of active.slice(1))if(Math.abs(v.currentTime-lead.currentTime)>.15)v.currentTime=lead.currentTime;},50);}
</script></html>'''
    page=page.replace('original24','original96').replace('全量24条','全量96条').replace('完整24条冻结任务：身份4、发型4、背景4、OmniEdit原生人物编辑6、TripVVT换衣6','完整96条冻结任务：身份16、发型16、背景16、OmniEdit原生人物编辑24、TripVVT换衣24；对应83段不同source').replace('其他像素按位保持','其他人物前景像素按位保持，背景填中性灰RGB128').replace('发型使用单张正面参考','发型使用去背景的单张正面参考，保持原发色与完整画幅').replace('background:#263440;padding:12px;display:flex','background:#263440;padding:12px;max-height:125px;overflow-y:auto;display:flex')
    (folder/'index.html').write_text(page)
    for name in ['inputs.jsonl','selection.jsonl','freeze.json','reference_processing.json','training_overlap.json','vace_prompt_corrections.json','task_limitations.json']:
        path=ROOT/'metadata'/name
        if path.exists():shutil.copy2(path,folder/name)
    (folder/'README.txt').write_text('解压后打开index.html，素材相对路径可离线查看。\n96条任务、83段不同SOURCE、72份逐任务参考、4张未处理原照片、24条TripVVT真实GT、96份49帧模型输入。\n身份和发型均去背景填灰，保持完整原画幅；身份头发填白，发型保留原发色。无生成结果或评分控件。同步共同0–4秒，完整原素材可独立查看。\n')
    class Check(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k=='src' or k=='href' and not v.startswith('#'):
                    assert not v.startswith(('http:','https:','/')) and (folder/v).is_file(),v
    Check().feed(page)
    assert len([x for x in inventory if x['role'].startswith('SOURCE')])==96
    assert len([x for x in inventory if x['role'].startswith('TARGET')])==24
    assert len([x for x in inventory if x['role']=='actual_reference'])==72
    atomic(folder/'inventory.json',dict(manifest_sha256=frozen['input_manifest_sha256'],items=inventory))
    output=ROOT/'review/a849_external_vbench_v2_96_original96_scroll.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,Path('a849_external_original96')/p.relative_to(folder))
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    receipt=dict(path=str(output),bytes=output.stat().st_size,samples=96,source_originals=96,unique_sources=83,actual_references=72,unedited_portraits=4,tripvvt_gt_originals=24,model_input_copies=96,generated_outputs=0,local_crc='passed',offline_links='passed',manifest_sha256=frozen['input_manifest_sha256'])
    atomic(ROOT/'metadata/original96_package.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
