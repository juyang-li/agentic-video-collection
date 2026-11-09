#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: launch_video_edit_seed42.sh RUNTIME_ROOT MODEL_PATH CONFIG_PATH OUTPUT_DIR" >&2
    exit 2
fi

RUNTIME_ROOT=$(realpath "$1")
MODEL_PATH=$(realpath "$2")
CONFIG_PATH=$(realpath "$3")
OUTPUT_DIR=$(realpath "$4")
ENV_PREFIX=${ENV_PREFIX:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-}
LANCE_MAX_DURATION=${LANCE_MAX_DURATION:-6.0}

# Keep the pinned upstream checkout unchanged while making all three relevant
# seeds explicit: framework/global, validation ordering, and diffusion noise.
source "$RUNTIME_ROOT/benchmarks/sample_env.sh"
lance_setup_common_env
lance_setup_distributed_env 1
lance_setup_shard_env 1

cd "$RUNTIME_ROOT"
accelerate launch \
    --num_machines "$NUM_MACHINES" \
    --num_processes "$TOTAL_RANK" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MAIN_PROCESS_IP" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    inference_lance.py \
    --model_path "$MODEL_PATH" \
    --vit_type qwen_2_5_vl_original \
    --llm_qk_norm true \
    --llm_qk_norm_und true \
    --llm_qk_norm_gen true \
    --tie_word_embeddings false \
    --validation_num_timesteps 30 \
    --validation_timestep_shift 3.5 \
    --copy_init_moe true \
    --max_num_frames 121 \
    --max_latent_size 64 \
    --latent_patch_size 1 1 1 \
    --visual_und true \
    --visual_gen true \
    --vae_model_type wan \
    --apply_qwen_2_5_vl_pos_emb true \
    --apply_chat_template false \
    --cfg_type 0 \
    --global_seed 42 \
    --validation_data_seed 42 \
    --video_height 480 \
    --video_width 480 \
    --num_frames 50 \
    --max_duration "$LANCE_MAX_DURATION" \
    --task video_edit \
    --save_path_gen "$OUTPUT_DIR" \
    --resolution video_480p \
    --text_template true \
    --cfg_text_scale 4.0 \
    --use_KVcache true \
    --enhance_prompt false \
    --val_dataset_config_file "$CONFIG_PATH"
