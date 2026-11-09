from pathlib import Path
import json,hashlib,shutil,html,zipfile,re
import av,numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;PROJECT=R.parents[1];OLD=R.parent/'ltx25_singleturn_v1';P=R/'review/ltx25_base_text10_scroll_20261031'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def frame(p,k):
 with av.open(str(p)) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i==k:return f.to_image()

def main():
 rows=[json.loads(s) for s in (R/'inputs.jsonl').read_text().splitlines()];assert len(rows)==10
 for name in ['media','generated','previous','provenance']:(P/name).mkdir(parents=True,exist_ok=True)
 cards=[];manifest=[];metrics=[];tiles=[]
 for d in rows:
  sid=d['sample_id'];rec=json.loads((R/f'outputs/records/{sid}.json').read_text());assert rec['status'] in ['generated','recovered_tmp']
  g=json.loads((R/f'outputs/workflows/{sid}.json').read_text());assert g['16']['inputs']['model']==['1',0] and not any('lora' in n['class_type'].lower() and 'loader' in n['class_type'].lower() for n in g.values())
  src=PROJECT/d['source_video'];out=Path(rec['output']);prev=OLD/f'outputs/generated/{sid}.mp4'
  sources=[('media/'+sid+'.mp4',src),('generated/'+sid+'.mp4',out),('previous/'+sid+'.mp4',prev)]
  for rel,p in sources:
   q=P/rel
   if not q.exists():shutil.copy2(p,q)
   assert sha(q)==sha(p)
  assert sha(out)!=sha(src)
  # Inspect all frames, not just a successful first-frame decode.
  with av.open(str(out)) as c:
   stream=c.streams.video[0];n=sum(1 for _ in c.decode(stream));assert n==rec['media']['frames']
  k=(n-1)//2;a=np.asarray(frame(out,k),dtype=np.float32);b=np.asarray(frame(prev,k),dtype=np.float32)
  item=dict(sample_id=sid,no_adapter_vs_ingredients_middle_rgb_mae_0_255=float(np.abs(a-b).mean()),frames=n,output_sha256=sha(out))
  metrics.append(item)
  panel=''.join(f'<div><h4>{label}</h4><video controls muted playsinline preload="none" src="{rel}"></video><a href="{rel}">打开视频</a></div>' for (rel,_),label in zip(sources,['原视频（完整素材）','LTX-2.5，无adapter（本次）','LTX-2.5＋Ingredients（上一轮）']))
  cards.append(f'<article id="{sid}"><h2>{sid} · {html.escape(d["task"])}</h2><p class="prompt">{html.escape(d["instruction"])}</p><div class="panels">{panel}</div><p>{rec["media"]["width"]}×{rec["media"]["height"]} · {n}帧 · 24 FPS · seed42 · 8步 · CFG1</p></article>')
  manifest.append({**d,'source_video':sources[0][0],'output':sources[1][0],'previous_output':sources[2][0],**item})
  tile=Image.new('RGB',(768,290),'#202020');draw=ImageDraw.Draw(tile);draw.text((8,6),sid+' | NO ADAPTER / PREVIOUS INGREDIENTS',fill='white')
  for j,arr in enumerate([a,b]):
   im=Image.fromarray(arr.astype('uint8'));im.thumbnail((380,260));tile.paste(im,(j*384+(384-im.width)//2,28+(260-im.height)//2))
  tiles.append(tile)
 css='body{background:#101720;color:#eee;max-width:1600px;margin:30px auto;padding:20px;font:16px/1.6 system-ui}a{color:#8bd1ff}article{padding:18px;margin:20px 0;border:1px solid #465464;border-radius:12px}.prompt{white-space:pre-wrap}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}video{width:100%;max-height:500px;background:#000}@media(max-width:800px){.panels{grid-template-columns:1fr}}'
 intro='''<h1>LTX-2.5 无adapter · 10例文字视频编辑</h1><p>4条外观、4条文字换衣、2条镜头任务。原视频＋文字，没有参考图或动作任务。与上一轮对比只移除了Ingredients；仍用同样的完整RGB视频guide、原prompt、seed与采样参数。模型是BF16 distilled，不是非蒸馏训练权重。该条件接法是实验探测，尚不代表官方已验证的通用指令编辑支持。</p><p>连续下拉，无评分栏。原视频保留完整文件；两轮模型输出使用相同的约4秒内原速片段。</p><p><a href="REPORT.md">运行报告</a> · <a href="manifest.jsonl">对应清单</a></p>'''
 (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LTX 无adapter文字编辑10例</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
 (P/'manifest.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in manifest))
 report='''# LTX-2.5 无adapter文字视频编辑10例

模型：已固定revision的LTX-2.5 22B BF16 distilled；无Ingredients，无其它LoRA。四个权重文件（transformer、Gemma4 encoder、两个VAE）见provenance/model_lock.json。seed42，8步Euler ancestral，CFG1，原始英文prompt，无增强，无超分。

协议：完整原视频按原速24FPS采样，最多97帧，等比例padding至768×448或448×768；RGB source guide frame_idx=0、strength1、downscale1。去掉上一轮IC-LoRA loader，CFGGuider直接连接原始UNETLoader，保留其它设置。它用于隔离adapter作用；不声称无adapter的通用RGB-guide文字编辑能力已有官方任务验证。输出均来自采样后裁去guide、VAE解码。

调度：先申请4卡，预测约2小时50分等待后按用户要求换为2卡。2卡独立allocation183476立即获配，先srun一条烟测再在同一allocation上并行完成十条（烟测样本复用，不重复）。没有使用其它既有GPU作业。

所有10条都保留，不按编辑效果筛选。页面显示原视频、本次无adapter结果、上一轮Ingredients结果。完整解码和文件对应检查见package_validation.json；定量像素差仅是诊断，不是任务成功率。编辑语义观察见下文，不以生成成功代替编辑成功。
'''
 findings=R/'metadata/visual_findings.md'
 if findings.exists():report+='\n'+findings.read_text()
 (P/'REPORT.md').write_text(report)
 for name in ['model_lock.json','protocol.json','run_config.json','comfyui_revision.txt','ltx_nodes_revision.txt','environment_freeze.txt','smoke_status.json','monitor_status.json','browser_validation.json','slurm_accounting.txt','text_edit_recheck.json','comparison_metrics.json']:
  if (R/'metadata'/name).exists():shutil.copy2(R/'metadata'/name,P/'provenance'/name)
 shutil.copytree(R/'outputs/records',P/'provenance/records',dirs_exist_ok=True);shutil.copytree(R/'outputs/workflows',P/'provenance/workflows',dirs_exist_ok=True)
 validation=dict(status='passed',cases=10,fully_decoded_outputs=10,no_adapter_graphs=10,source_output_distinct=True,local_media_paths=True)
 (P/'package_validation.json').write_text(json.dumps(validation,indent=2)+'\n');(R/'metadata/comparison_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 for start in [0,5]:
  sheet=Image.new('RGB',(768,290*5))
  for i,t in enumerate(tiles[start:start+5]):sheet.paste(t,(0,i*290))
  sheet.save(R/f'metadata/base_comparison_{start//5+1}.jpg')
 zpath=P.with_suffix('.zip')
 with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_STORED) as z:
  for p in sorted(P.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(P.parent))
 with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
 receipt=dict(status='ready_local',package=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),cases=10)
 (R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
