#!/usr/bin/env python3
"""Prepare compatible inference-only checkpoints and paired multi-turn jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
WORKSPACE = ROOT.parents[2]
DATAMAKE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake")
LANCE = DATAMAKE / "10_research/model_experiments/lance"
RUNNER = LANCE / "scripts/run_datamake_lora_visual_sanity.py"
SUITE = LANCE / "configs/datamake_mixed_complete_chain_review_20260911.json"
ENV_PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
COINVE = WORKSPACE / (
    "video_collection/lance_general_edit_finetune/3_lance_lora_finetuning/"
    "runs/lance_coinve_4k_r64_seed42_e2_v1/checkpoints/0000677"
)
GTFRESH = LANCE / (
    "outputs/training_runs/datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/"
    "checkpoints/0001150"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace differing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f"staged file mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_legacy() -> Path:
    stage = ROOT / "inference_checkpoints/coinve_real_step677"
    stage.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("adapter.safetensors", "datamake_lora_config.json"):
        source = COINVE / name
        destination = stage / name
        link_or_copy(source, destination)
        artifacts[name] = {"bytes": destination.stat().st_size, "sha256": sha256(destination)}

    source_contract_path = COINVE / "datamake_online_training_contract.json"
    contract = json.loads(source_contract_path.read_text())
    source_yaml = Path(contract["dataset_config"])
    dataset = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if set(dataset) != {"D_general_edit"}:
        raise ValueError(f"unexpected legacy dataset keys: {sorted(dataset)}")
    compatible_yaml = stage / "lance_dataset.yaml"
    compatible_yaml.write_text(
        yaml.safe_dump({"D1": dataset["D_general_edit"]}, sort_keys=False), encoding="utf-8"
    )
    contract["dataset_config"] = str(compatible_yaml)
    contract["dataset_config_sha256"] = sha256(compatible_yaml)
    staged_contract = stage / "datamake_online_training_contract.json"
    if staged_contract.exists():
        staged_contract.unlink()
    write_json(staged_contract, contract)
    artifacts[staged_contract.name] = {
        "bytes": staged_contract.stat().st_size, "sha256": sha256(staged_contract)
    }

    marker = stage / "checkpoint_complete.json"
    marker_payload = {
        "status": "complete", "completed_steps": 677, "adapter_only": True,
        "checkpoint_format_version": "inference_compat_from_v5",
        "source_checkpoint": str(COINVE), "artifacts": artifacts,
    }
    if marker.exists() and json.loads(marker.read_text()) != marker_payload:
        marker.unlink()
    write_json(marker, marker_payload)
    return stage


def main() -> None:
    coinve_stage = stage_legacy()
    jobs = []
    for slot, (model_id, checkpoint) in enumerate((
        ("coinve_real_step677", coinve_stage),
        ("gtfresh_step1150", GTFRESH),
    )):
        output = ROOT / "multiturn" / model_id
        jobs.append({
            "model_id": model_id, "gpu_slot": slot, "checkpoint": str(checkpoint),
            "output_dir": str(output), "log": str(ROOT / "logs" / f"multiturn_{model_id}.log"),
            "command": [str(ENV_PYTHON), str(RUNNER), "--checkpoint", str(checkpoint),
                        "--output-dir", str(output), "--suite", str(SUITE)],
        })
    write_json(METADATA / "multiturn_jobs.json", {
        "schema_version": "gtfresh_coinve_multiturn_jobs_v1",
        "suite": str(SUITE), "suite_sha256": sha256(SUITE),
        "expected_generated_videos_per_model": 28,
        "headline_autoregressive_videos_per_model": 12,
        "jobs": jobs,
    })
    print(json.dumps({"status": "prepared", "jobs": jobs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
