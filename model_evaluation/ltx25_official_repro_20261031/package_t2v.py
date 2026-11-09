from pathlib import Path
import json,html,hashlib,shutil,zipfile
import av,numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;P=R/'review/ltx25_official_t2v2_scroll_20261031'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for d in ['generated','provenance']:(P/d).mkdir(parents=True,exist_ok=True)
 rows=[json.loads(s) for s in (R/'inputs.jsonl').read_text().splitlines()];assert len(rows)==2
 cards=[];manifest=[];contact=[]
 for d in rows:
  sid=d['sample_id'];rec=json.loads((R/f'outputs/records/{sid}.json').read_text());g=json.loads((R/f'outputs/workflows/{sid}.json').read_text())
  assert rec['status'] in ['generated','recovered_tmp'] and not rec['source_video_used']
  assert g['16']['inputs']['model']==['1',0] and not any('lora' in v['class_type'].lower() or v['class_type'] in ['LTXBenchSource','LTXBenchRef','LTXVAddGuide'] for v in g.values())
  src=Path(rec['output']);out=P/f'generated/{sid}.mp4';shutil.copy2(src,out);assert sha(src)==sha(out)
  selected={0:None,60:None,120:None};n=0
  with av.open(str(out)) as c:
   stream=c.streams.video[0]
   for i,f in enumerate(c.decode(stream)):
    n+=1
    if i in selected:selected[i]=f.to_image()
   assert n==121 and stream.width==768 and stream.height==512
  tile=Image.new('RGB',(960,245),'#202020');draw=ImageDraw.Draw(tile);draw.text((8,5),sid+' | frames 0 / 60 / 120',fill='white')
  for j,im in enumerate(selected.values()):
   assert im is not None;im.thumbnail((320,214));tile.paste(im,(j*320,28))
  contact.append(tile)
  cards.append('<article><h2>'+sid+'</h2><p>'+html.escape(d['instruction'])+'</p><video controls muted playsinline preload="none" src="generated/'+sid+'.mp4"></video><p>768×512 · 121帧 · 24 FPS · seed42 · 8步 · CFG1 · 无adapter／无图像或视频输入</p></article>')
  manifest.append({**d,'output':f'generated/{sid}.mp4','output_sha256':sha(out),'media':rec['media'],'config_sha256':rec['config_sha256']})
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LTX-2.5 官方纯文本案例</title><style>body{background:#101720;color:#eee;max-width:1100px;margin:30px auto;padding:20px;font:16px/1.7 system-ui}article{border:1px solid #465464;padding:20px;margin:20px 0}video{width:100%;max-height:650px;background:black}a{color:#8bd1ff}</style><h1>LTX-2.5 官方纯文生视频案例</h1><p>使用官方模型卡prompt与官方单阶段distilled生成流程，检验基础文字生成链路。这里没有原视频，不是视频编辑测评。没有adapter、参考图或RGB guide。</p><p>模型与代码revision见provenance；关闭提示增强与可选图像条件。联合采样音视频，本包仅导出无声视频。连续下拉，无评分栏。</p><a href="REPORT.md">配置与观察</a>'+''.join(cards)+'</html>'
 (P/'index.html').write_text(page);(P/'manifest.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in manifest))
 report='''# 官方纯文本案例复现

当前运行的是文字生成新视频，不是原视频编辑。官方模型卡中的金毛犬草地prompt及该页面库用法示例中的海上日落prompt作为基础正对照。使用已经固定revision的LTX-2.5 BF16 distilled、Gemma4编码器、官方允许的conv VAE和音频VAE。无任何adapter，model直接从UNETLoader接入CFGGuider。没有source视频、图像、IC guide或任何输入latent拷贝。

采用官方LTX-2.5_T2V_I2V_Single_Stage_Distilled.json的生成节点与固定8步sigma，Euler ancestral、CFG1、seed42；768×512、121帧、24FPS。只禁用该流程可选的image输入和prompt enhancer，无超分。工作流的GUI封装被展开为API节点，保存使用项目I/O节点，解码使用官方VAEDecodeTiled，输出无声。这是官方单阶段配置的基础复现，不声称与官方双阶段高分辨率成片逐像素一致，也没有已知官方seed对应golden视频。

官方文档区分distilled推理权重与dev训练权重，并明确dev transformer可训练。因此，此前视频编辑接法失败不能推导基模无法训练。

来源：
- https://huggingface.co/Lightricks/LTX-2.5
- https://docs.ltx.io/open-source-model/usage-guides/text-to-video
- https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json
'''
 findings=R/'metadata/t2v_visual_findings.md'
 if findings.exists():report+='\n'+findings.read_text()
 (P/'REPORT.md').write_text(report)
 for name in ['model_lock.json','protocol.json','run_config.json','smoke_status.json','shard0_status.json','monitor_status.json','comfyui_revision.txt','ltx_nodes_revision.txt','slurm_accounting.txt','browser_validation.json']:
  if (R/'metadata'/name).exists():shutil.copy2(R/'metadata'/name,P/'provenance'/name)
 shutil.copytree(R/'outputs/workflows',P/'provenance/workflows',dirs_exist_ok=True);shutil.copytree(R/'outputs/records',P/'provenance/records',dirs_exist_ok=True)
 shutil.copy2(R/'metadata/official_workflows/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json',P/'provenance/official_workflow.json')
 (P/'package_validation.json').write_text(json.dumps(dict(status='passed',cases=2,fully_decoded=2,frames_each=121,no_adapter=True,no_visual_conditioning=True),indent=2)+'\n')
 sheet=Image.new('RGB',(960,490))
 for i,im in enumerate(contact):sheet.paste(im,(0,i*245))
 sheet.save(R/'metadata/t2v_contact.jpg')
 archive=P.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
  for p in sorted(P.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(P.parent))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 receipt=dict(status='ready_local',file=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),cases=2)
 (R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
