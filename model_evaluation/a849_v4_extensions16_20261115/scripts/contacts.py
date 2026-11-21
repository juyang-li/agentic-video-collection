"""CPU timestamp contacts for direct inspection; GT is audit-only."""
import argparse, concurrent.futures
from common import *
from PIL import Image, ImageOps, ImageDraw

def frames(path,times):
    import av
    chosen=[None for _ in times];distances=[float('inf') for _ in times];last=-1
    with av.open(str(path)) as c:
        stream=c.streams.video[0]; stream.thread_count=1
        for f in c.decode(video=0):
            last=float(f.time)
            indices=[i for i,t in enumerate(times) if abs(last-t)<distances[i]]
            if indices:
                image=f.to_image()
                for i in indices:chosen[i]=image;distances[i]=abs(last-times[i])
            if last>max(times)+.15:break
    assert last >= max(times)-.05, (path,last)
    return chosen

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['trip','omni','masks','outputs'],default='trip');a=p.parse_args()
    dest=ROOT/'review';dest.mkdir(exist_ok=True)
    if a.mode=='trip':
        from select_pool import POOL
        records=rows(ROOT/'metadata/provisional32.jsonl')
        def render(r):
            images=frames(POOL/r['source'],[0,1,2,3,4])
            images += [Image.open(POOL/r['reference']).convert('RGB'),frames(POOL/r['target'],[2])[0]]
            return r['pilot_id']+' '+r['sample_id']+' '+r['category'], images
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: rendered=list(executor.map(render,records))
    elif a.mode=='omni':
        parent=ROOT.parent/'a849_external_vbench_v3_116'
        used={r['source']['sha256'] for r in rows(parent/'metadata/inputs.jsonl')}
        records=[r for r in rows(ROOT.parent/'a849_external_vbench_v2_96/metadata/omni_candidates.jsonl') if r['source_sha256'] not in used and r['probe']['duration_last_pts']>=4]
        atomic(ROOT/'metadata/omni_unused.json',records)
        def render(r):return r['id'],frames(r['path'],[0,1,2,3,4])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:rendered=list(executor.map(render,records))
    else:
        import numpy as np
        rendered=[]
        for r in rows(ROOT/'metadata/inputs.jsonl'):
            sid=r['id'];ims=[]
            c=read(ROOT/'conditions'/sid/'condition.json')
            for n in [0,12,24,36,48]:
                im=Image.open(ROOT/'inputs/prepared'/sid/f'{n:03}.png').convert('RGB');rgb=np.array(im)
                m=np.asarray(Image.open(c['mask_frames'][n]))>127
                rgb[m]=(rgb[m]*.5+np.array([255,0,0])*.5).astype('uint8');ims.append(Image.fromarray(rgb))
            rendered.append((sid,ims))
    cols=7 if a.mode=='trip' else 5;cw=220 if cols==7 else 300;rh=265 if a.mode=='trip' else 220
    for start in range(0,len(rendered),4):
        chunk=rendered[start:start+4];sheet=Image.new('RGB',(cw*cols,rh*len(chunk)),'#20242a');d=ImageDraw.Draw(sheet)
        for row,(label,images) in enumerate(chunk):
            for col,im in enumerate(images):
                sheet.paste(ImageOps.contain(im,(cw-6,rh-34)),(col*cw,row*rh+30))
            d.text((4,row*rh+6),label+(' | source 0,1,2,3,4s | REF | GT2s' if a.mode=='trip' else ' | 0,1,2,3,4s'),fill='white')
        path=dest/f'{a.mode}_contact_{start//4:02}.jpg';sheet.save(path,quality=92);print(path,flush=True)

if __name__=='__main__':main()
