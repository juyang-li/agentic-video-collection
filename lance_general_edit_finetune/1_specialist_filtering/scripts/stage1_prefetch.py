"""Bounded CPU decode prefetch; GPU metrics retain the production protocol."""
from collections import deque
import time

import numpy as np
import run_stage1_quality as quality


def prepare(job):
    started = time.monotonic()
    try:
        path = quality.resolve_local_path(job['local_path'])
        _, frames, count, fps = quality.decode_frames(path, 12)
        # Retain only the twelve 224px frames, not full-resolution RGB batches.
        return {'frames': frames, 'count': count, 'fps': fps,
                'decode_seconds': time.monotonic()-started}
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}',
                'decode_seconds': time.monotonic()-started}


def prepared_jobs(pool, items, window=8):
    iterator = iter(items)
    pending = deque()
    for _ in range(window):
        item = next(iterator, None)
        if item is not None:
            pending.append((item, pool.submit(prepare, item[1]['job'])))
    while pending:
        item, future = pending.popleft()
        yield item, future.result()
        following = next(iterator, None)
        if following is not None:
            pending.append((following, pool.submit(prepare, following[1]['job'])))


def score_prepared(job, prepared, models):
    started = time.monotonic()
    result = quality.blank_result(job)
    try:
        if 'error' in prepared:
            result.update(technical_status='error', error=prepared['error'])
        else:
            frames, count, fps = prepared['frames'], prepared['count'], prepared['fps']
            duration = count/fps
            result.update(technical_status='pass' if duration>=3 else 'fail_short',
                frame_count=str(count),fps=f'{fps:.6f}',duration_seconds=f'{duration:.6f}',
                sampled_frames=str(len(frames)))
            if duration>=3:
                result.update({key:f'{value:.8f}' for key,value in quality.cheap_metrics(frames).items()})
                if models is not None:
                    musiq = models.musiq_scores(frames,8)
                    result['musiq_p10']=f'{quality.percentile(musiq,10):.8f}'
                    result['musiq_mean']=f'{float(np.mean(musiq)):.8f}'
                    result.update({key:f'{value:.8f}' for key,value in models.dino_metrics(frames,8).items()})
    except Exception as error:
        result.update(technical_status='error',error=f'{type(error).__name__}: {error}')
    result['elapsed_seconds']=f"{prepared['decode_seconds']+time.monotonic()-started:.6f}"
    return result
