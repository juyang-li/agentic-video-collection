from common import *
import csv

def main():
 records=[read(p) for p in sorted((ROOT/'scores').glob('*/*.json'))];assert len(records)==1200
 summaries=[]
 for model in MODELS:
  selected=[r for r in records if r['model']==model];assert len(selected)==300
  groups=[('overall','all',selected)]+[('dataset',d,[r for r in selected if r['dataset']==d]) for d in ['ue','coinve','tripvvt']]+[('task',g,[r for r in selected if r['group']==g]) for g in sorted({r['group'] for r in selected})]
  for level,label,rs in groups:
   summaries.append({'model':model,'level':level,'group':label,'videos':len(rs),'frames':sum(r['num_frames'] for r in rs),'ssim':sum(r['mean']['ssim'] for r in rs)/len(rs),'psnr_db':None if any(r['mean']['psnr_infinite'] for r in rs) else sum(r['mean']['psnr_db'] for r in rs)/len(rs),'psnr_infinite':any(r['mean']['psnr_infinite'] for r in rs),'lpips':sum(r['mean']['lpips'] for r in rs)/len(rs)})
 out=ROOT/'report';out.mkdir(exist_ok=True);atomic(out/'summary.json',summaries)
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 with (out/'videos.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['model','id','dataset','group','num_frames','ssim','psnr_db','psnr_infinite','lpips']);w.writeheader()
  for r in records:w.writerow({k:r[k] for k in ['model','id','dataset','group','num_frames']}|r['mean'])
 with (out/'frames.jsonl').open('w') as f:
  for r in records:
   for row in r['frames']:f.write(json.dumps({'model':r['model'],'id':r['id'],**row})+'\n')
 text=['# 四模型对GT：冻结300条','', '全冻结帧；逐帧→视频内平均→视频等权平均。只对GT，无SOURCE比较。有效画面1280×720或720×1280，去除协议padding，无时间/光流最优配准。','', '| 模型 | 视频 | 帧对 | SSIM↑ | PSNR(dB)↑ | LPIPS↓ |','|---|---:|---:|---:|---:|---:|']
 for s in summaries:
  if s['level']=='overall':text.append(f'| {s["model"]} | {s["videos"]} | {s["frames"]} | {s["ssim"]:.6f} | '+('∞' if s['psnr_infinite'] else f'{s["psnr_db"]:.6f}')+f' | {s["lpips"]:.6f} |')
 text+=['','H3为PDD8加速版，VACE为LightX2V4加速版；不是普通50/40步基线。A849为dev＋step849；LTX Base为完整distilled DiT＋V7。','三指标表示与单一GT的重建保真度，不直接代表指令成功或时序质量。动作/镜头可存在合理多解，全图指标也可能被大面积未变化区域主导。','分数据集与任务统计见summary.csv，逐帧见frames.jsonl，逐视频见videos.csv。']
 (out/'REPORT.md').write_text('\n'.join(text)+'\n');atomic(ROOT/'metadata/completion.json',{'status':'complete','videos':1200,'frame_pairs':sum(r['num_frames'] for r in records),'failures':0})
if __name__=='__main__':main()
