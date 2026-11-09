#!/usr/bin/env python3
"""Freeze OmniEdit-48 inputs and prepare smoke/formal four-GPU jobs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
WORKSPACE = ROOT.parents[2]
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE_ROOT = DATAMAKE / "10_research/model_experiments/lance"
ALIGNED = WORKSPACE / "video_collection/model_evaluation/omni_baselines_20260926/aligned_subset.csv"
BASE = LANCE_ROOT / "models/Lance/Lance_3B_Video"
MODELS = {
    "coinve_real_step677": WORKSPACE / (
        "video_collection/lance_general_edit_finetune/3_lance_lora_finetuning/"
        "runs/lance_coinve_4k_r64_seed42_e2_v1/checkpoints/0000677"
    ),
    "gtfresh_step1150": LANCE_ROOT / (
        "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/"
        "checkpoints/0001150"
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_frozen(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to replace differing frozen file: {path}")
    path.write_text(text, encoding="utf-8")


def config_for(rows: list[dict[str, str]]) -> dict[str, dict]:
    return {
        f"{int(row['aligned_index']):04d}": {
            "interleave_array": [row["editing_instruction"], row["official_source_video"]],
            "element_dtype_array": ["text", "video"],
            "istarget_in_interleave": [0, 0],
        }
        for row in rows
    }


def make_job(model_id: str, checkpoint: Path, label: str, shard: int, rows: list[dict[str, str]]) -> dict:
    run_name = f"{label}_{model_id}_shard{shard:02d}"
    config = ROOT / "configs" / f"{run_name}.json"
    write_frozen(config, config_for(rows))
    return {
        "model_id": model_id,
        "gpu_slot": shard,
        "sample_count": len(rows),
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "command": [
            "bash", str(ROOT / "scripts/run_singleturn.sh"), str(ROOT), str(LANCE_ROOT),
            str(BASE), str(checkpoint), str(config), run_name,
        ],
    }


def main() -> None:
    if sha256(ALIGNED) != "654e39544aaf445f320bdc9cde3b214c347a71df4572d9385b66b305cac353dd":
        raise ValueError("aligned OmniEdit-48 manifest changed")
    with ALIGNED.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 48 or len({row["category"] for row in rows}) != 24:
        raise ValueError("expected 48 rows across 24 categories")
    for row in rows:
        source = Path(row["official_source_video"])
        if not source.is_file() or sha256(source) != row["official_source_sha256"]:
            raise ValueError(f"source mismatch: {source}")

    model_receipts = {}
    for model_id, checkpoint in MODELS.items():
        complete = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        if complete.get("status") != "complete":
            raise ValueError(f"incomplete checkpoint: {checkpoint}")
        model_receipts[model_id] = {
            "checkpoint": str(checkpoint),
            "completed_steps": complete["completed_steps"],
            "adapter_sha256": sha256(checkpoint / "adapter.safetensors"),
            "lora_schema": json.loads((checkpoint / "datamake_lora_config.json").read_text())["schema_version"],
        }

    smoke_rows = [
        next(row for row in rows if "1280x720" in row["official_source_repo_path"]),
        next(row for row in rows if "720x1280" in row["official_source_repo_path"]),
    ]
    smoke_groups = []
    omni_groups = []
    for model_id, checkpoint in MODELS.items():
        smoke_groups.append({"model_id": model_id, "jobs": [make_job(model_id, checkpoint, "smoke", 0, smoke_rows)]})
        jobs = []
        for shard in range(4):
            shard_rows = [row for index, row in enumerate(rows) if index % 4 == shard]
            jobs.append(make_job(model_id, checkpoint, "omni48", shard, shard_rows))
        omni_groups.append({"model_id": model_id, "jobs": jobs})

    common = {
        "schema_version": "gtfresh_coinve_omni_jobs_v1",
        "seed": 42,
        "inference": {"steps": 30, "shift": 3.5, "cfg_text": 4.0, "resolution": "480p", "fps": 12},
        "aligned_manifest": str(ALIGNED),
        "aligned_manifest_sha256": sha256(ALIGNED),
        "base_model": str(BASE),
        "base_model_sha256": sha256(BASE / "model.safetensors"),
        "models": model_receipts,
    }
    write_frozen(METADATA / "smoke_jobs.json", {**common, "kind": "smoke", "groups": smoke_groups})
    write_frozen(METADATA / "omni_jobs.json", {**common, "kind": "omni48", "groups": omni_groups})
    write_frozen(METADATA / "evaluation_receipt.json", {**common, "omni_samples_per_model": 48, "categories": 24})
    print(json.dumps({"status": "prepared", "smoke_outputs": 4, "omni_outputs": 96,
                      "models": model_receipts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
