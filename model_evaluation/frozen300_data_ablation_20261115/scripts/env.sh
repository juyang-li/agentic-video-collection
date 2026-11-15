export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
BENCH=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
export HF_HOME="$BENCH/runtime/hf" TORCH_HOME="$BENCH/runtime/torch" VBENCH_CACHE_DIR="$BENCH/runtime/vbench_models"
export LD_LIBRARY_PATH=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
