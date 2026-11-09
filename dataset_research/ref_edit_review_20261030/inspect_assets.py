#!/usr/bin/env python3
"""Contact sheets for a direct visual check of the selected inputs."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def frame(path, fraction=0.4):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((n-1)*fraction)))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"unreadable {path.name}")
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), n/fps


def fit(im, size):
    im = im.copy()
    im.thumbnail(size)
    out = Image.new('RGB',size,'#111111')
    out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
    return out


def main():
    dest=ROOT/'metadata/previews';dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'metadata/garment_selection.json').read_text())['samples']
    for start in range(0,25,5):
        sheet=Image.new('RGB',(720,5*270),'#111111');d=ImageDraw.Draw(sheet)
        for i,r in enumerate(rows[start:start+5]):
            y=i*270
            d.text((5,y+3),f"{r['sample_id']} {r['cloth_type']}",fill='white')
            for j,k in enumerate(['input','cloth_image','output']):
                path=ROOT/'assets/tripvvt'/r[k]
                im=Image.open(path) if k=='cloth_image' else frame(path)[0]
                sheet.paste(fit(im,(240,240)),(j*240,y+25))
        sheet.save(dest/f'garment_{start//5+1}.jpg')
    meta=ROOT/'metadata/motion_reference_selection.json'
    if not meta.exists():return
    rows=json.loads(meta.read_text())['samples']
    for cls in ['wave','clap','turn','walk']:
        group=[r for r in rows if r['class']==cls]
        sheet=Image.new('RGB',(960,len(group)*205),'#111111');d=ImageDraw.Draw(sheet)
        for i,r in enumerate(group):
            path=ROOT/'assets/hmdb'/cls/Path(r['member']).name
            if not path.exists():continue
            try:
                im,duration=frame(path)
                d.text((5,i*205+2),f"{i+1}: {path.name[:100]} [{duration:.2f}s]",fill='white')
                for j,t in enumerate([0.1,0.5,0.9]):sheet.paste(fit(frame(path,t)[0],(320,180)),(j*320,i*205+23))
            except RuntimeError:
                d.text((5,i*205+25),'UNREADABLE',fill='red')
        sheet.save(dest/f'motion_{cls}.jpg')
    print(dest)


if __name__=='__main__':main()
