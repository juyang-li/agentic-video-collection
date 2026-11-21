"""CPU source-only repair: reject a large Haar false positive over hair/window."""
from common import *
import cv2,numpy as np
from PIL import Image
from media import write_preview

def main():
 sid='accessory_01';item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']==sid)
 out=ROOT/'conditions'/sid;c=read(out/'condition.json');atomic(out/'before_haar_false_positive_repair.json',c)
 old=ROOT.parent/'a849_external_vbench_v2_96';face_c=read(old/'conditions/identity_01/condition.json')
 detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml');overlays=[];paths=[];boxes=[];previous=None
 for i in range(49):
  frame=Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB');rgb=np.asarray(frame);face=np.asarray(Image.open(face_c['mask_frames'][i]))>127
  candidates=detector.detectMultiScale(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),scaleFactor=1.05,minNeighbors=4)
  def iou(b):
   x,y,w,h=map(int,b);intersection=int(face[y:y+h,x:x+w].sum());return intersection/(w*h+int(face.sum())-intersection)
  if len(candidates):box=max(candidates,key=iou);assert iou(box)>.40;previous=box
  else:assert previous is not None;box=previous
  x,y,w,h=map(int,box);boxes.append([x,y,w,h]);a,b,d,e=item['mask_spec']['relative_roi'];m=np.zeros(face.shape,np.uint8)
  m[max(0,round(y+b*h)):min(frame.height,round(y+e*h)),max(0,round(x+a*w)):min(frame.width,round(x+d*w))]=255
  dilation=item['mask_spec']['dilation_pixels'];m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
  path=out/f'mask_{i:03d}.png';Image.fromarray(m).save(path);paths.append(path)
  pixels=rgb.copy();sel=m>127;pixels[sel]=(pixels[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(pixels))
 write_preview(overlays,out/'overlay.mp4')
 c.update(mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],repair=dict(method='Choose Haar face box with maximum IoU against source face parsing; reject larger hair/window false positive',source_face_condition=str(old/'conditions/identity_01/condition.json'),source_face_condition_sha256=sha(old/'conditions/identity_01/condition.json'),face_boxes=boxes,script_sha256=sha(__file__)),human_protocol_review='pending')
 atomic(out/'condition.json',c);print(sid,'source face IoU repair complete',flush=True)
if __name__=='__main__':main()
