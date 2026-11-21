#!/usr/bin/env bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000275
#SBATCH --job-name=a849_v3_104_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104/logs/slurm_%j.log
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
if "$PY" "$TASK/scripts/check_generation.py" --phase all && [[ -f "$TASK/metadata/gpu_scores_complete.json" ]];then exit 0;fi
bash "$TASK/scripts/gpu_pipeline.sh"
# Keep the allocation available at the user's request after generation/evaluation.
while [[ -e /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96/metadata/retain_allocation.request ]];do sleep 30;done
