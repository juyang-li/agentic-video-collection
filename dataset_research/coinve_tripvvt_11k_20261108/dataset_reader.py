"""Dependency-free interface for the shared manifest or an unpacked delivery ZIP.

Video decoding is left to the training loader. All paths resolve against the
manifest directory, never the process working directory.
"""
import argparse
import json
import math
from pathlib import Path


def iter_samples(manifest, *, split=None, dataset=None):
    """Yield complete records with absolute source/target/reference paths.

    split: train, validation, test, or None. dataset: coinve, tripvvt, or None.
    Returns all original fields, with media paths resolved to pathlib.Path.
    """
    manifest=Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig') as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            if split is not None and row['split']!=split:continue
            if dataset is not None and row['dataset']!=dataset:continue
            for role in ('source','target','reference'):
                if row.get(role):row[role]=(manifest.parent/row[role]).resolve()
            yield row


def sample_times(row, *, fps, frames=None, start_seconds=0):
    """Return shared absolute times at the requested sampling rate.

    A fixed frame count must fit the common interval; reject rather than stretch
    the timeline or pad the longer tail. These are time coordinates, not source
    frame indices: each video must be decoded at these same coordinates.
    """
    fps=float(fps);start=float(start_seconds);end=float(row['common_end_seconds'])
    if not math.isfinite(fps) or fps<=0:raise ValueError('fps must be positive and finite')
    if not math.isfinite(start) or start<float(row['common_start_seconds']) or start>end:raise ValueError('start is outside the shared interval')
    if frames is None:frames=math.floor((end-start)*fps+1e-8)+1
    if isinstance(frames,bool) or not isinstance(frames,int) or frames<1:raise ValueError('frames must be a positive integer')
    if start+(frames-1)/fps>end+1e-8:raise ValueError('requested frames exceed common interval; choose fewer frames or a valid temporal crop')
    return [start+i/fps for i in range(frames)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path)
    parser.add_argument('--split',choices=('train','validation','test'));parser.add_argument('--dataset',choices=('coinve','tripvvt'))
    parser.add_argument('--fps',type=float,default=12);parser.add_argument('--frames',type=int,default=49);parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    for index,row in enumerate(iter_samples(args.manifest,split=args.split,dataset=args.dataset)):
        if index>=args.limit:break
        times=sample_times(row,fps=args.fps,frames=args.frames)
        print(json.dumps(dict(sample_id=row['sample_id'],dataset=row['dataset'],split=row['split'],
            source=str(row['source']),target=str(row['target']),reference=str(row['reference']) if row.get('reference') else None,
            prompt=row['prompt'],caption=row.get('caption'),sample_times_seconds=times),ensure_ascii=False))

if __name__=='__main__':main()
