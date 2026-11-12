"""Contact sheets and source difference diagnostics, not model quality scores."""
import argparse
from common import *

def build(only=None):
    require_job(exact=True)
    import av
    import numpy as np
    from PIL import Image,ImageDraw
    from media import video_frames
    out=ROOT/'metadata/visual_checks';out.mkdir(exist_ok=True)
    items=rows(ROOT/'metadata/inputs.jsonl');report=[]
    if only:items=[i for i in items if i['id'] in only]
    for item in items:
        files={m:ROOT/'outputs'/m/f'{item["id"]}.mp4' for m in ['ltx_base','a849','vace']}
        files={m:p for m,p in files.items() if p.exists()}
        if not files:continue
        with av.open(str(ROOT/'review/a100_inputs_scroll_20261111/media'/f'{item["id"]}_00.mp4')) as c:
            source=[f.to_image() for f in c.decode(video=0)]
        assert len(source)==49
        videos={'SOURCE':source}
        measurements={}
        for m,p in files.items():
            try:
                with av.open(str(p)) as c:frames=[f.to_image() for f in c.decode(video=0)]
                if len(frames)!=49:continue
                videos[m]=frames
                mae=[float(np.abs(np.asarray(a).astype(np.float32)-np.asarray(b).astype(np.float32)).mean()) for a,b in zip(source,frames,strict=True)]
                measurements[m]={'source_rgb_mae_mean_0_255':sum(mae)/49,'source_rgb_mae_max_0_255':max(mae),'note':'Source is the verified browser preview; compression affects pixel differences. Difference alone does not prove instruction completion.'}
            except Exception as e:measurements[m]={'error':str(e)}
        for model_name,frames in videos.items():
            film=Image.new('RGB',(1792,7*172),(24,24,24));fd=ImageDraw.Draw(film)
            for idx,f in enumerate(frames):
                im=f.copy();im.thumbnail((256,147));col,row=idx%7,idx//7
                film.paste(im,(col*256+(256-im.width)//2,row*172+25))
                fd.text((col*256+4,row*172+4),f'{idx:02d} {idx/12:.2f}s',fill='white')
            film.save(out/f'{item["id"]}_{model_name}_full49.jpg',quality=92)
        cellw,cellh=320,220
        sheet=Image.new('RGB',(cellw*3,cellh*len(videos)),(24,24,24));draw=ImageDraw.Draw(sheet)
        for row,(m,frames) in enumerate(videos.items()):
            for col,idx in enumerate([0,24,48]):
                im=frames[idx].copy();im.thumbnail((cellw,cellh-25))
                sheet.paste(im,(col*cellw+(cellw-im.width)//2,row*cellh+25))
                draw.text((col*cellw+4,row*cellh+4),f'{m} {idx/12:.1f}s',fill='white')
        sheet.save(out/f'{item["id"]}.jpg',quality=92)
        report.append({'id':item['id'],'measurements':measurements})
    atomic(out/'measurements.json',report)
    print(len(report),'contact sheets')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',action='append');build(p.parse_args().only)
