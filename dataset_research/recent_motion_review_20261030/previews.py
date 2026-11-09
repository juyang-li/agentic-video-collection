from pathlib import Path
from PIL import Image,ImageDraw
import argparse,cv2,json

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'metadata/previews'


def fit(im,size):
    im=im.convert('RGB');im.thumbnail(size);out=Image.new('RGB',size,'#111');out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out


def image_grids():
    DEST.mkdir(parents=True,exist_ok=True);paths=sorted((ROOT/'assets/realis/ref').glob('*.png'))
    for start in range(0,len(paths),20):
        im=Image.new('RGB',(1000,960),'#111');draw=ImageDraw.Draw(im)
        for j,p in enumerate(paths[start:start+20]):
            x=j%5*200;y=j//5*240;im.paste(fit(Image.open(p),(200,215)),(x,y+25));draw.text((x+7,y+5),p.stem,fill='white')
        im.save(DEST/f'realis_refs_{start//20+1}.jpg')


def video_grids():
    DEST.mkdir(parents=True,exist_ok=True)
    groups={'xdance':sorted((ROOT/'assets/xdance').glob('Video-*/*/video.mp4')),
            'realis':sorted((ROOT/'assets/realis/gt').glob('*.mp4')),
            'move':sorted((ROOT/'assets/movebench/video').glob('*.mp4'))}
    stats={}
    for group,paths in groups.items():
        for start in range(0,len(paths),8):
            im=Image.new('RGB',(960,8*210),'#111');draw=ImageDraw.Draw(im)
            for row,p in enumerate(paths[start:start+8]):
                cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                label=str(p.relative_to(ROOT/'assets'))
                stats[label]={'frames':n,'fps':fps,'width':width,'height':height,'duration':n/fps if fps else 0}
                draw.text((5,row*210+2),f'{label} {width}x{height} {n/fps if fps else 0:.2f}s',fill='white')
                for col,t in enumerate([.15,.5,.85]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int((n-1)*t));ok,f=cap.read()
                    if ok:im.paste(fit(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(320,180)),(col*320,row*210+25))
                cap.release()
            im.save(DEST/f'{group}_videos_{start//8+1}.jpg')
    (ROOT/'metadata/video_specs.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['images','videos']);a=p.parse_args()
    image_grids() if a.kind=='images' else video_grids()
