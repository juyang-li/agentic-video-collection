from common import *
from evaluate import location,validate,gt_frames,output_frames
from PIL import Image,ImageDraw
import time

def main():
 items=rows(PKG/'inputs.jsonl');gts={r['id']:r for r in rows(PKG/'ground_truth.jsonl')};audit=[]
 for item in items:
  gt=gts[item['id']];assert gt['uid']==item['uid'] and gt['temporal_window']==item['temporal_window']
  assert len(gt['sample_frame_indices'])==item['num_frames']
  for model in MODELS:
   path,metadata=location(model,item['id']);validate(item,model,path,metadata)
   interval=metadata.get('comparison_interval_seconds')
   if interval:assert interval==[item['temporal_window']['start_seconds'],item['temporal_window']['end_seconds']]
   if model=='h3_pdd8':assert metadata['h3_covers_frozen_end'] is True
   audit.append({'id':item['id'],'model':model,'sha256':metadata['output_sha256'],'gt_sha256':gt['target']['sha256']})
 atomic(ROOT/'metadata/pair_audit.json',{'status':'passed','combinations':len(audit),'records':audit})
 for item in items:
  if item['id'] not in SMOKE:continue
  size=(320,180) if item['canvas_wh'][0]>item['canvas_wh'][1] else (180,320)
  truth,_=gt_frames(gts[item['id']],size);ids=[0,len(truth)//2,len(truth)-1]
  images=[[Image.fromarray(truth[i]) for i in ids]]
  for model in MODELS:
   path,metadata=location(model,item['id']);frames,alignment=output_frames(path,item,metadata,model,size)
   images.append([Image.fromarray(frames[i]) for i in ids])
  w,h=size;sheet=Image.new('RGB',(w*3,5*(h+24)),(35,35,35));draw=ImageDraw.Draw(sheet)
  for row,name in enumerate(['GT']+MODELS):
   draw.text((4,row*(h+24)+4),name,fill='white')
   for col,img in enumerate(images[row]):sheet.paste(img,(col*w,row*(h+24)+24))
  p=ROOT/f'preview/audit/{item["id"]}.jpg';p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p)
 print('PAIR_AUDIT_PASSED',len(audit),flush=True)
if __name__=='__main__':main()
