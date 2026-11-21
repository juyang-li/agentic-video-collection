"""Class-agnostic CPU SAM2 hair boundary refinement from SOURCE-only prompts."""
from common import *
require_job()
import argparse,os,sys,time
p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[len(cpus)//2:]);os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,cv2,numpy as np
from PIL import Image,ImageDraw
from media import video_frames,write_preview
torch.set_num_threads(min(8,len(os.sched_getaffinity(0))))
ready=read(ROOT/'runtime/vace_ready.json');sys.path.insert(0,ready['sam2']['path'])
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',ready['sam2']['checkpoint'],device='cpu'))
item=next(i for i in rows(ROOT/'metadata/inputs.jsonl') if i['id']=='hair_color_10');assert item['source']['attributes']['hair_color_id']==7
assert not (ROOT/'outputs/vace/hair_color_10.json').exists()
base=ROOT/'conditions/hair_color_10';old=read(base/'condition.json');rois={x['frame']:x['head_roi'] for x in old['source_crop_repair']['frames']}
frames,pixel=video_frames(item,item['source']);indices=[59,60,61,87,120] if args.probe else list(range(121));masks=[];details=[]
for idx in indices:
 image=np.asarray(frames[idx]);h,w=image.shape[:2];x1,y1,x2,y2=rois[idx];crop=image[y1:y2,x1:x2];ch,cw=crop.shape[:2]
 hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV);white=(hsv[:,:,1]<110)&(hsv[:,:,2]>90)
 yy,xx=np.mgrid[:ch,:cw];ellipse=((xx-cw*.5)/(cw*.35))**2+((yy-ch*.5)/(ch*.44))**2<1
 ys,xs=np.where(white&ellipse);assert len(xs)>5
 points=[];labels=[]
 for tx in [cw*.42,cw*.58]:
  target=np.array([tx,ch*.31]);j=int(np.argmin((xs-target[0])**2+(ys-target[1])**2));points.append([int(xs[j]),int(ys[j])]);labels.append(1)
 for point in [[1,ch-2],[cw-2,ch-2]]:points.append(point);labels.append(0)
 ny,nx=np.where(white&~ellipse&(xx<cw*.12))
 if len(nx):j=int(np.argmin((ny-ch*.4)**2));points.append([int(nx[j]),int(ny[j])]);labels.append(0)
 predictor.set_image(crop)
 with torch.inference_mode():candidates,scores,_=predictor.predict(point_coords=np.array(points,np.float32),point_labels=np.array(labels,np.int32),multimask_output=True)
 selected=candidates[int(np.argmax(scores))].astype(np.uint8)*255
 selected=cv2.bitwise_and(selected,white.astype(np.uint8)*255);selected=cv2.dilate(selected,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)));selected=cv2.bitwise_and(selected,white.astype(np.uint8)*255)
 mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=selected;assert mask.any() and (mask>0).mean()<.1
 masks.append(mask);details.append({'frame':idx,'head_roi':rois[idx],'points':points,'labels':labels,'sam_score':float(max(scores))});print('CPU SAM hair',idx,flush=True)
if args.probe:
 canvas=Image.new('RGB',(1500,640),'white');draw=ImageDraw.Draw(canvas)
 for col,(idx,mask) in enumerate(zip(indices,masks)):
  rgb=np.asarray(frames[idx]);red=np.zeros_like(rgb);red[:,:,0]=255;a=(mask>0)[:,:,None]*.45;overlay=Image.fromarray((rgb*(1-a)+red*a).astype(np.uint8))
  for row,im in enumerate([frames[idx],overlay]):
   w,h=im.size;im=im.crop((int(.35*w),0,int(.65*w),int(.45*h)));im.thumbnail((300,290));canvas.paste(im,(col*300,row*320+25))
  draw.text((col*300,0),str(idx),fill='black')
 canvas.save(ROOT/'metadata/white_hair_sam_probe.jpg')
else:
 out=base/'sam_white_cpu';out.mkdir(exist_ok=True);overlays=[]
 for idx,(frame,mask) in enumerate(zip(frames,masks,strict=True)):
  Image.fromarray(mask).save(out/f'mask_{idx:03d}.png');rgb=np.asarray(frame);red=np.zeros_like(rgb);red[:,:,0]=255;a=(mask>0)[:,:,None]*.45;overlays.append(Image.fromarray((rgb*(1-a)+red*a).astype(np.uint8)))
 write_preview(overlays,out/'overlay.mp4',12);files=[out/f'mask_{i:03d}.png' for i in range(121)];areas=[float((m>0).mean()) for m in masks]
 assert not (ROOT/'outputs/vace/hair_color_10.json').exists();atomic(ROOT/'metadata/protocol_attempts/white_semantic_only_condition.json',old)
 old.update(identity={**old['identity'],'script_sha256':sha(__file__)},mask_frames=[str(f) for f in files],mask_sha256=[sha(f) for f in files],overlay_video=str(out/'overlay.mp4'),mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/121,source_crop_repair={'method':'SOURCE-only per-frame CPU SAM2 head crop, white-hair positive prompts and background negatives; visible source-color constraint excludes brown skin','frames':details},completed_unix=time.time());atomic(base/'condition.json',old)
 print('CPU_SAM_WHITE_MASK_COMPLETE',flush=True)
