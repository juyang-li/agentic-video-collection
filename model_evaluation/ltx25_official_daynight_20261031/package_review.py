from pathlib import Path
import json,html,hashlib,shutil,zipfile
import av,numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;P=R/'review/ltx25_official_daynight2_scroll_20261031';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for d in ['media','generated','provenance']:(P/d).mkdir(parents=True,exist_ok=True)
 rows=[json.loads(s) for s in (R/'inputs.jsonl').read_text().splitlines()];cards=[];man=[];tiles=[]
 for sid in ['mountains','cafe']:
  row=next(d for d in rows if d['sample_id']==sid+'_daynight')
  sources=[(f'media/{sid}_source.mp4',R/f'assets/{sid}_source.mp4','官方示例输入视频'),(f'generated/{sid}_daynight.mp4',R/f'outputs/generated/{sid}_daynight.mp4','本次：官方日转夜adapter'),(f'generated/{sid}_bare.mp4',R/f'outputs/generated/{sid}_bare.mp4','本次：去掉adapter的对照'),(f'media/{sid}_official_target.mp4',R/f'assets/{sid}_official_target.mp4','官方发布结果（非本次生成）')]
  for rel,src,label in sources:
   shutil.copy2(src,P/rel);assert sha(src)==sha(P/rel)
  for suffix in ['daynight','bare']:
   sample=sid+'_'+suffix;record=json.loads((R/f'outputs/records/{sample}.json').read_text());g=json.loads((R/f'outputs/workflows/{sample}.json').read_text());assert record['status'] in ['generated','recovered_tmp'] and record['source_video_used']
   assert record['instruction']==row['instruction'] and g['11']['inputs']['image']==['5',0]
   if suffix=='daynight':assert g['16']['inputs']['model']==['10',0] and g['10']['inputs']['lora_name']=='ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors'
   else:assert g['16']['inputs']['model']==['1',0] and '10' not in g
   p=R/f'outputs/generated/{sample}.mp4'
   with av.open(str(p)) as c:
    st=c.streams.video[0];n=sum(1 for _ in c.decode(st));assert n==97
   man.append({**record,'output':f'generated/{sample}.mp4','source_video':f'media/{sid}_source.mp4','official_target':f'media/{sid}_official_target.mp4','output_sha256':sha(p)})
  panels=''.join(f'<div><h4>{label}</h4><video controls muted playsinline preload="none" src="{rel}"></video><a href="{rel}">打开视频</a></div>' for rel,src,label in sources)
  cards.append(f'<article id="{sid}"><h2>{sid}</h2><p class="prompt">{html.escape(row["instruction"])}</p><div class="panels">{panels}</div><p>97帧 · 24 FPS · seed42 · 官方单阶段distilled：8步、CFG1</p></article>')
  tile=Image.new('RGB',(1200,500),'#202020');draw=ImageDraw.Draw(tile);draw.text((8,5),sid+' | SOURCE / OUR ADAPTER / BARE / OFFICIAL TARGET',fill='white')
  for j,(rel,src,label) in enumerate(sources):
   with av.open(str(src)) as c:
    for i,f in enumerate(c.decode(video=0)):
     if i==48:
      im=f.to_image();im.thumbnail((298,460));tile.paste(im,(j*300+(300-im.width)//2,28+(460-im.height)//2));break
  tiles.append(tile)
 intro='''<h1>LTX-2.5 官方视频编辑：日景→夜景</h1><p>两个官方视频输入与原始prompt。每例展示本次官方Day-to-Night adapter输出、去掉adapter的对照，以及官方发布结果。官方结果单独标注，不能冒充本次生成。</p><p>使用完整source视频条件，没有独立参考图。LTX-2.5 BF16 distilled＋官方日转夜adapter（strength1），不是裸基模通用编辑。原视频从官方展示视频的输入面板提取，具体差异见报告。连续下拉，无评分栏。</p><p><a href="REPORT.md">运行报告与观察</a></p>'''
 css='body{background:#101720;color:#eee;max-width:1800px;margin:30px auto;padding:20px;font:16px/1.7 system-ui}a{color:#8bd1ff}article{border:1px solid #465464;margin:20px 0;padding:20px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}video{width:100%;max-height:650px;background:#000}.prompt{white-space:pre-wrap}@media(max-width:900px){.panels{grid-template-columns:1fr}}'
 (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LTX官方视频编辑</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
 (P/'manifest.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in man))
 report='''# 官方日转夜视频编辑复现

两条官方案例：山景、咖啡馆。各自原始prompt从官方模型卡widget提取；原source及官方结果从同一官方发布展示视频的上下/左右面板提取，分别CRF16编码。97帧24FPS，768×448/448×768，与训练桶相符。官方发布视频保留在项目官方素材目录；本包保留提取面板。没有把官方夜景视频喂给模型，也没有把它作为本次输出。

LTX-2.5 22B BF16 distilled＋Day-to-Night 0.9 adapter（rank32，strength1）。完整source多帧RGB guide frame_idx0，strength1、downscale1；没有参考图或首帧替代原视频。文字使用官方原始prompt，seed42，无提示增强、无超分，sampler Euler ancestral。使用官方链接的LTX-2.5 generic V2V single-stage distilled流程，固定8步CFG1；模型卡另有30步CFG3–4/STG建议，本次没有混用那套参数，不能声称完全复现该30步结果。官方seed和原始source master未提供，不要求与官方目标逐像素一致。

每条额外跑了去掉adapter的对照，source、prompt、seed、sigma、预处理保持相同。输出4条，source2条，官方发布target2条。生成同时包含音频latent，但只导出无声视频。官方结果与本次推理结果清楚区分。

来源：https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night
工作流：https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json
'''
 findings=R/'metadata/visual_findings.md'
 if findings.exists():report+='\n'+findings.read_text()
 (P/'REPORT.md').write_text(report)
 for p in (R/'metadata').glob('*'):
  if p.is_file() and p.suffix in ['.json','.txt','.md'] and 'model_card' not in p.name:shutil.copy2(p,P/'provenance'/p.name)
 shutil.copytree(R/'outputs/records',P/'provenance/records',dirs_exist_ok=True);shutil.copytree(R/'outputs/workflows',P/'provenance/workflows',dirs_exist_ok=True)
 sheet=Image.new('RGB',(1200,1000))
 for i,t in enumerate(tiles):sheet.paste(t,(0,i*500))
 sheet.save(R/'metadata/comparison_contact.jpg')
 (P/'package_validation.json').write_text(json.dumps(dict(status='passed',cases=2,our_outputs=4,fully_decoded_outputs=4,official_targets=2),indent=2)+'\n')
 archive=P.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
  for p in sorted(P.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(P.parent))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 receipt=dict(status='ready_local',file=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),cases=2,our_outputs=4)
 (R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
