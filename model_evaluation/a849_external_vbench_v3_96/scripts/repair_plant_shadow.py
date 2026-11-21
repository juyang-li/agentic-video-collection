"""CPU source-planned reconstruction of the plant's cast shadow; preserve glass vase."""
from common import *
import cv2,numpy as np
from PIL import Image
from media import write_preview

def main():
 sid='object_removal_06';out=ROOT/'conditions'/sid;c=read(out/'condition.json');atomic(out/'before_shadow_reconstruction.json',c)
 px=read(ROOT/'inputs/prepared'/sid/'pixels.json');frame0=Image.open(ROOT/'inputs/prepared'/sid/'000.png');w,h=frame0.size;l,t,r,b=px['pixel_audit']['geometry']['padding_ltrb'];aw=w-l-r;ah=h-t-b
 polygon=[(.43,.27),(.94,.32),(.99,.66),(.91,.88),(.44,.88),(.43,.64)]
 shadow=np.zeros((h,w),np.uint8);cv2.fillPoly(shadow,[np.array([[round(l+x*aw),round(t+y*ah)] for x,y in polygon],dtype=np.int32)],255)
 overlays=[];paths=[];areas=[]
 for i,path in enumerate(c['mask_frames']):
  m=np.maximum(np.asarray(Image.open(path)),shadow);dst=out/f'mask_{i:03d}.png';Image.fromarray(m).save(dst);paths.append(dst);areas.append(float((m>127).mean()))
  rgb=np.asarray(Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB')).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
 write_preview(overlays,out/'overlay.mp4')
 c.update(mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_area_min=min(areas),mask_area_max=max(areas))
 c['repair'].update(shadow_polygon_source_normalized=polygon,shadow_method='Source-planned cast-shadow restoration region on wall, to avoid forcing a plant shadow to remain after deletion',shadow_script_sha256=sha(__file__),note='Vase exterior silhouette remains protected; stems through transparent glass and the cast plant shadow require local reconstruction.')
 atomic(out/'condition.json',c)
 # Keep repaired accessory summary metadata consistent with its final mask pixels.
 p=ROOT/'conditions/accessory_01/condition.json';r=read(p);areas=[float((np.asarray(Image.open(path))>127).mean()) for path in r['mask_frames']];r.update(mask_area_min=min(areas),mask_area_max=max(areas));atomic(p,r)
 print(sid,'cast-shadow region prepared',flush=True)
if __name__=='__main__':main()
