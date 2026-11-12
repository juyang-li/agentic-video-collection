"""Validate agent-written protocol observations before expanding generation."""
import argparse
from common import *
from native_worker import SMOKE,SECOND

def require(stage):
    ids=SMOKE if stage=='smoke' else SECOND
    gate=read(ROOT/'metadata'/f'gate_{stage}.json')
    assert gate['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    assert gate['status']=='protocol_verified'
    assert gate['full_video_review'] and gate['reference_order_verified'] and gate['time_window_verified'] and gate['vace_mask_direction_tracking_verified'] and gate['pose_adapter_verified']
    assert gate['model_quality_not_used_to_resample']
    if stage=='smoke':assert gate['reference_controls_reviewed']
    for model in ('ltx_base','a849','vace'):
        for sid in ids:
            status=read(ROOT/'outputs'/model/f'{sid}.json')
            assert status['status']=='complete' and status['full_decode_frames']==49
            assert sha(ROOT/'outputs'/model/f'{sid}.mp4')==status['output_sha256']
            assert status['protocol_audit']['no_gt']
    print(f'{stage}: protocol gate verified',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require',choices=['smoke','second'],required=True);require(p.parse_args().require)
