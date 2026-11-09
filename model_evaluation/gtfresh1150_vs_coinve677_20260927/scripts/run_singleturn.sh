#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "Usage: run_singleturn.sh EVAL_ROOT LANCE_ROOT MODEL_PATH CHECKPOINT CONFIG RUN_NAME" >&2
    exit 2
fi

EVAL_ROOT=$(realpath "$1")
LANCE_ROOT=$(realpath "$2")
MODEL_PATH=$(realpath "$3")
CHECKPOINT=$(realpath "$4")
CONFIG=$(realpath "$5")
RUN_NAME=$6
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(realpath "$SCRIPT_DIR/../../../../")
TOOLS_ROOT="$WORKSPACE/video_collection/model_evaluation/lance_tools"
ENV_PREFIX=${ENV_PREFIX:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance}
OUTPUT_DIR="$EVAL_ROOT/outputs_raw/$RUN_NAME"
LOG_PATH="$EVAL_ROOT/logs/$RUN_NAME.log"
PENDING_CONFIG="$EVAL_ROOT/configs/${RUN_NAME}_pending.json"
TECHNICAL_CSV="$EVAL_ROOT/technical/${RUN_NAME}.csv"
VIT_PATH="$LANCE_ROOT/models/Lance/Qwen2.5-VL-ViT"

for path in \
    "$SCRIPT_DIR/adapter_inference_lance.py" \
    "$TOOLS_ROOT/prepare_pending_config.py" \
    "$TOOLS_ROOT/verify_outputs.py" \
    "$LANCE_ROOT/lance/inference_lance.py" \
    "$MODEL_PATH/model.safetensors" \
    "$VIT_PATH/vit.safetensors" \
    "$CHECKPOINT/checkpoint_complete.json" \
    "$CHECKPOINT/datamake_lora_config.json" \
    "$CHECKPOINT/adapter.safetensors" \
    "$CONFIG"; do
    if [[ ! -s "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR" "$EVAL_ROOT/logs" "$EVAL_ROOT/configs" "$EVAL_ROOT/technical"
export PATH="$ENV_PREFIX/bin:$PATH"
export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$ENV_PREFIX/bin/python" "$TOOLS_ROOT/prepare_pending_config.py" \
    "$CONFIG" "$OUTPUT_DIR" "$PENDING_CONFIG" --requeue-invalid | tee -a "$LOG_PATH"

PENDING_COUNT=$("$ENV_PREFIX/bin/python" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$PENDING_CONFIG")
if [[ "$PENDING_COUNT" -gt 0 ]]; then
    source "$LANCE_ROOT/lance/benchmarks/sample_env.sh"
    lance_setup_common_env
    lance_setup_distributed_env 1
    lance_setup_shard_env 1
    cd "$LANCE_ROOT/lance"
    accelerate launch \
        --num_machines "$NUM_MACHINES" \
        --num_processes "$TOTAL_RANK" \
        --machine_rank "$MACHINE_RANK" \
        --main_process_ip "$MAIN_PROCESS_IP" \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --mixed_precision bf16 \
        "$SCRIPT_DIR/adapter_inference_lance.py" \
        --datamake_lance_root "$LANCE_ROOT/lance" \
        --datamake_lora_checkpoint "$CHECKPOINT" \
        --model_path "$MODEL_PATH" \
        --vit_path "$VIT_PATH" \
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
        --max_duration 6.0 \
        --task video_edit \
        --save_path_gen "$OUTPUT_DIR" \
        --resolution video_480p \
        --text_template true \
        --cfg_text_scale 4.0 \
        --use_KVcache true \
        --enhance_prompt false \
        --val_dataset_config_file "$PENDING_CONFIG" 2>&1 | tee -a "$LOG_PATH"
fi

"$ENV_PREFIX/bin/python" "$TOOLS_ROOT/verify_outputs.py" \
    "$CONFIG" "$OUTPUT_DIR" --inventory "$TECHNICAL_CSV" | tee -a "$LOG_PATH"
