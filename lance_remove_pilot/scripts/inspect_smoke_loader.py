#!/usr/bin/env python3
"""Build real Lance packed batches and audit the v2v conditioning semantics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


PILOT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PILOT_ROOT / "runtime" / "upstream"
sys.path.insert(0, str(RUNTIME_ROOT))

from data.data_utils import add_special_tokens  # noqa: E402
from data.dataset_base_train import DataConfig, PackedDataset  # noqa: E402
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402


EXPECTED_DATASETS = {
    "x2v_interleave_local:insave_smoke",
    "x2v_interleave_local:goku_smoke",
}


def audit_batch(batch: dict, batch_number: int) -> dict:
    data_indexes = batch["batch_data_indexes"]
    counts = Counter(item["dataset_name"] for item in data_indexes)
    if set(counts) != EXPECTED_DATASETS:
        raise ValueError(
            f"batch {batch_number} datasets={sorted(counts)}, "
            f"expected={sorted(EXPECTED_DATASETS)}"
        )
    if any(count < 1 for count in counts.values()):
        raise ValueError(f"batch {batch_number} has an empty data source")

    vae_indexes = set(batch["packed_vae_token_indexes"].tolist())
    mse_indexes = set(batch["mse_loss_indexes"].tolist())
    condition_indexes = vae_indexes - mse_indexes
    if not mse_indexes or not condition_indexes:
        raise ValueError(
            f"batch {batch_number} must have both target MSE and clean VAE condition tokens"
        )
    if not mse_indexes.issubset(vae_indexes):
        raise ValueError(f"batch {batch_number} MSE indexes escape VAE token indexes")
    if "ce_loss_indexes" in batch and len(batch["ce_loss_indexes"]) > 0:
        raise ValueError(f"batch {batch_number} unexpectedly contains CE targets")
    if len(batch.get("padded_videos_vit", [])) < len(EXPECTED_DATASETS):
        raise ValueError(f"batch {batch_number} is missing source ViT conditions")
    if len(batch.get("padded_videos", [])) < 2 * len(EXPECTED_DATASETS):
        raise ValueError(f"batch {batch_number} is missing source/target VAE videos")

    return {
        "batch": batch_number,
        "sequence_length": int(batch["sequence_length"]),
        "datasets": dict(sorted(counts.items())),
        "samples": len(data_indexes),
        "vit_condition_videos": len(batch["padded_videos_vit"]),
        "vae_source_and_target_videos": len(batch["padded_videos"]),
        "vae_tokens": len(vae_indexes),
        "clean_vae_condition_tokens": len(condition_indexes),
        "target_mse_tokens": len(mse_indexes),
        "ce_target_tokens": int(len(batch.get("ce_loss_indexes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=PILOT_ROOT / "exports" / "loader_audit.json"
    )
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive")

    config_path = PILOT_ROOT / "configs" / "smoke_mixed.yaml"
    model_path = RUNTIME_ROOT / "downloads" / "Lance_3B_Video"
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
    llm_config = json.loads((model_path / "llm_config.json").read_text())

    data_config = DataConfig.from_yaml(str(config_path))
    data_config.vae_downsample = (4, 16, 16)
    data_config.latent_patch_size = (1, 1, 1)
    data_config.max_latent_size = 64
    data_config.max_num_frames = 121
    data_config.text_cond_dropout_prob = 0
    data_config.vae_cond_dropout_prob = 0
    data_config.vit_cond_dropout_prob = 0

    # Parquet paths in the YAML are intentionally relative to the Lance runtime.
    os.chdir(RUNTIME_ROOT)
    dataset = PackedDataset(
        data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=0,
        expected_num_tokens=65536,
        max_num_tokens=65536,
        max_num_tokens_per_sample=65536,
        prefer_buffer_before=65536,
        max_buffer_size=16,
        interpolate_pos=False,
        use_flex=True,
        data_status=None,
        apply_chat_template=False,
        image_token_id=llm_config["video_token_id"],
        cfg_type=0,
        cfg_uncond_token_id=151643,
        require_und_gen=False,
        N_key_frame=-1,
        incre_time_pro=0,
    )
    dataset.set_epoch(20260824)
    iterator = iter(dataset)
    audits = [audit_batch(next(iterator), index) for index in range(args.batches)]
    report = {
        "status": "ok",
        "config": str(config_path),
        "batches": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
