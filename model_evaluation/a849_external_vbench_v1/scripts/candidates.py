"""Source-only full-duration contact sheets, before task freeze or generation."""
import random
from common import *

def sheet(records, name):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    pages = []
    for start in range(0, len(records), 5):
        canvas = Image.new('RGB', (1440, 5*205), '#20242a')
        draw = ImageDraw.Draw(canvas)
        for row, rec in enumerate(records[start:start+5]):
            path = rec['path']
            with av.open(path) as c:
                c.streams.video[0].thread_count = 2
                frames = list(c.decode(video=0))
            times = [float(f.pts*f.time_base) for f in frames]
            times = np.array(times)-times[0]
            targets = np.linspace(0, times[-1], 8)
            draw.text((4,row*205+2), rec['id']+' | '+rec.get('instruction','')[:155], fill='white')
            for col,t in enumerate(targets):
                i = int(np.argmin(abs(times-t)))
                img = ImageOps.contain(frames[i].to_image(), (178,170))
                canvas.paste(img,(col*180,row*205+24))
                draw.text((col*180,row*205+190),f'{times[i]:.2f}s',fill='white')
            rec['probe'] = {'duration_last_pts':float(times[-1]), 'frames':len(frames), 'dimensions':[frames[0].width, frames[0].height], 'review_timestamps':targets.tolist()}
        dst = ROOT/'review'/f'{name}_{start//5:02d}.jpg'
        canvas.save(dst,quality=90)
        pages.append(str(dst))
    return pages

if __name__ == '__main__':
    require_job()
    omni = read(ROOT/'inputs/omni/spatial_benchmark.json')
    records = [{'id':r['video_id'],'path':str(ROOT/'inputs/omni'/r['source_video']),'instruction':r['editing_instruction'],'official':r} for r in omni if (ROOT/'inputs/omni'/r['source_video']).is_file()]
    pages = sheet(records,'omni_candidates')
    src = next(s for s in read(BUNDLE)['sources'] if s['name']=='tripvvt')
    pool = rows(src['test_manifest']); rng = random.Random(20261112)
    chosen = []
    for cat in ['upper_body','lower_body','dresses']:
        group = sorted([r for r in pool if r['references'][0]['attributes']['category']==cat],key=lambda r:r['sample_id'])
        rng.shuffle(group)
        chosen.extend(group[:2])
    trip = [{'id':f'tripvvt_{i+1:02d}', 'path':r['source']['absolute_path'],'instruction':r['prompt']['en'],'record':r} for i,r in enumerate(chosen)]
    pages += sheet(trip,'trip_candidates')
    jsonl(ROOT/'metadata/candidates.jsonl',records+trip)
    atomic(ROOT/'metadata/candidate_sheets.json',pages)
    print(json.dumps(pages),flush=True)
