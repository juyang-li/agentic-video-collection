from build import ROOT, NATIVE, BUNDLE, GROUPS, read, rows, sha, write
import os
from pathlib import Path
CONFIG=NATIVE/'configs/training/mixed720p_sequential_visual_dropout/train.yaml'
CHECKPOINT=NATIVE/'experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.safetensors'
atomic=write
def require_job(exact=False):
 if os.environ.get('SLURM_JOB_PARTITION')!='testqueue' or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU execution requires testqueue')
