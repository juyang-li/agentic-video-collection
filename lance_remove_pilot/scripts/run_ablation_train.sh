#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PILOT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT="$PILOT_ROOT/runtime/upstream"
PYTHON_BIN=${PYTHON_BIN:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/torchrun}

: "${RUN_NAME:?RUN_NAME is required}"
: "${DATASET_CONFIG:?DATASET_CONFIG is required}"

TOTAL_STEPS=${TOTAL_STEPS:-500}
AUTO_RESUME=${AUTO_RESUME:-False}
SAVE_EVERY=${SAVE_EVERY:-499}
CKPT_DEBUG_STEPS=${CKPT_DEBUG_STEPS:-249}
NUM_WORKERS=${NUM_WORKERS:-2}
RUN_DIR="$PILOT_ROOT/runs/$RUN_NAME"

if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
    echo "Lance environment is missing Python or torchrun" >&2
    exit 2
fi
if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Dataset config is missing: $DATASET_CONFIG" >&2
    exit 2
fi
if [[ "$AUTO_RESUME" != "True" && -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run: $RUN_DIR" >&2
    exit 2
fi
if [[ "$AUTO_RESUME" == "True" && ! -d "$RUN_DIR/checkpoints" ]]; then
    echo "AUTO_RESUME=True but no checkpoint directory exists: $RUN_DIR/checkpoints" >&2
    exit 2
fi
for value in "$TOTAL_STEPS" "$SAVE_EVERY"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "TOTAL_STEPS and SAVE_EVERY must be positive integers" >&2
        exit 2
    fi
done

mkdir -p "$RUN_DIR"
cd "$RUNTIME_ROOT"

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::FutureWarning:deepspeed.runtime.zero.linear"
export PYTHONPATH="$RUNTIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$TORCHRUN_BIN" \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 1 \
    train/unified_train.py \
    --model_path ./downloads/Lance_3B_Video \
    --vit_path ./downloads/Qwen2.5-VL-ViT \
    --vit_type qwen_2_5_vl_original \
    --llm_qk_norm True \
    --llm_qk_norm_und True \
    --llm_qk_norm_gen True \
    --tie_word_embeddings False \
    --max_num_frames 121 \
    --max_latent_size 64 \
    --latent_patch_size 1 1 1 \
    --load_from_lance_checkpoint True \
    --init_from_vlm_checkpoint False \
    --copy_init_moe False \
    --visual_und True \
    --visual_gen True \
    --vae_model_type wan \
    --freeze_llm False \
    --freeze_llm_embed_tokens True \
    --freeze_und_params True \
    --freeze_und False \
    --freeze_vit True \
    --dataset_config_file "$DATASET_CONFIG" \
    --val_dataset_config_file "$DATASET_CONFIG" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor 2 \
    --expected_num_tokens 65536 \
    --max_num_tokens 65536 \
    --max_num_tokens_per_sample 65536 \
    --prefer_buffer_before 65536 \
    --max_buffer_size 50 \
    --require_und_gen False \
    --data_seed 20260824 \
    --global_seed 20260824 \
    --text_cond_dropout_prob 0 \
    --vae_cond_dropout_prob 0 \
    --vit_cond_dropout_prob 0 \
    --warmup_steps 0 \
    --lr_scheduler constant \
    --lr 0.00001 \
    --min_lr 0.0000001 \
    --ce_weight 0.25 \
    --mse_weight 1.0 \
    --timestep_shift 4.0 \
    --use_flex True \
    --num_replicate 1 \
    --num_shard 1 \
    --use_ema False \
    --auto_resume "$AUTO_RESUME" \
    --resume_model_only False \
    --load_data_status False \
    --outputs_dir "$PILOT_ROOT/runs" \
    --total_steps "$TOTAL_STEPS" \
    --save_every "$SAVE_EVERY" \
    --log_every 1 \
    --ckpt_debug_steps "$CKPT_DEBUG_STEPS" \
    --wandb_project lance_remove_ablation \
    --wandb_name "$RUN_NAME" \
    --wandb_offline True \
    --validation_step -1 \
    --apply_qwen_2_5_vl_pos_emb True \
    --apply_chat_template False \
    2>&1 | tee "$RUN_DIR/train.log"

"$PYTHON_BIN" "$PILOT_ROOT/scripts/plot_metrics.py" \
    "$RUN_DIR/metrics.csv" \
    "$RUN_DIR/loss_curve.svg" \
    --title "Lance removal ablation: $RUN_NAME"
