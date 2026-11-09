#!/usr/bin/env python3
"""Derive the two CoinVE ablation profiles from the frozen Single-4K profile."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE = STAGE_ROOT / "configs/lance_4k_r64_seed42_e2_v1.json"
SPECS = {
    "coinve": {
        "profile_id": "lance_coinve_4k_r64_seed42_e2_v1",
        "run_name": "lance_coinve_4k_r64_seed42_e2_v1",
        "data": STAGE_ROOT / "artifacts/lance_coinve_4k_r64_seed42_e2_v1/data",
        "config": STAGE_ROOT / "configs/lance_coinve_4k_r64_seed42_e2_v1.json",
        "wandb_id": "lance-coinve-4k-r64-seed42-e2-v1",
        "tags": ["general-edit", "coinve", "4k", "multitask", "r64", "seed42", "2epoch", "4xh200"],
    },
    "mixed": {
        "profile_id": "lance_mixed_2k2k_r64_seed42_e2_v1",
        "run_name": "lance_mixed_2k2k_r64_seed42_e2_v1",
        "data": STAGE_ROOT / "artifacts/lance_mixed_2k2k_r64_seed42_e2_v1/data",
        "config": STAGE_ROOT / "configs/lance_mixed_2k2k_r64_seed42_e2_v1.json",
        "wandb_id": "lance-mixed-2k2k-r64-seed42-e2-v1",
        "tags": ["general-edit", "single2k", "coinve2k", "multitask", "r64", "seed42", "2epoch", "4xh200"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base = json.loads(BASE_PROFILE.read_text())
    for label, spec in SPECS.items():
        data_root = spec["data"]
        manifest = data_root / "manifest.json"
        dataset_yaml = data_root / "lance_dataset.yaml"
        schedule = data_root / "training_schedule.csv"
        for path in (manifest, dataset_yaml, schedule):
            if not path.is_file():
                raise FileNotFoundError(path)
        value = deepcopy(base)
        value["profile_id"] = spec["profile_id"]
        value["created_at"] = date.today().isoformat()
        value["data"].update({
            "train_chain_manifest": str(manifest.resolve()),
            "train_chain_manifest_sha256": sha256_file(manifest),
            "train_config": str(dataset_yaml.resolve()),
            "train_config_sha256": sha256_file(dataset_yaml),
            "training_schedule": str(schedule.resolve()),
            "training_schedule_sha256": sha256_file(schedule),
        })
        value["run"]["run_name"] = spec["run_name"]
        value["wandb"].update({
            "run_id": spec["wandb_id"],
            "group": "coinve-ablation-r64-seed42",
            "tags": spec["tags"],
        })
        value["evaluation"].update({
            "benchmark": "IVEBench",
            "comparison_checkpoint_step": 677,
            "comparison_models": ["single4k_step677", "coinve4k_step677", "mixed2k2k_step677"],
            "protocol": "all three unordered model pairs, both A/B orders, blinded labels",
        })
        destination = spec["config"]
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if destination.exists() and destination.read_text() != payload:
            raise ValueError(f"refuse to overwrite changed profile: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
        print(f"{label}: {destination} sha256={sha256_file(destination)}")


if __name__ == "__main__":
    main()
