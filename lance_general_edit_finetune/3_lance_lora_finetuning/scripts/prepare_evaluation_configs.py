#!/usr/bin/env python3
"""Prepare four single-GPU Lance jobs per candidate; later stages require the winner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
BASE_ROOT = ROOT.parents[1] / "model_evaluation/lance_base"
CANDIDATES = (451, 677, 902)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def freeze_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing frozen configuration differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "remaining", "test", "omni"), required=True)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    output = args.evaluation_root.resolve()
    profile = json.loads((ROOT / "configs/lance_4k_r64_seed42_e2_v1.json").read_text())
    checkpoints = Path(profile["run"]["output_root"]) / profile["run"]["run_name"] / "checkpoints"
    decision_hash = None
    if args.stage == "screen":
        steps = list(CANDIDATES)
    else:
        decision_path = args.decision or output / "promotion_decision.json"
        decision = json.loads(decision_path.read_text())
        winner = decision["provisional_winner_step"]
        if winner not in CANDIDATES or decision["promotion_steps"] != [winner]:
            raise ValueError("later stages require exactly the selected numeric winner")
        if decision.get("promotion_action") != "winner_only_by_numeric_rank":
            raise ValueError("unexpected promotion policy")
        steps = [winner]
        decision_hash = hashlib.sha256(decision_path.read_bytes()).hexdigest()

    paths = {
        "screen": output / "validation_screen48.csv",
        "remaining": output / "validation_remaining147.csv",
        "test": output / "internal_test204.csv",
        "omni": BASE_ROOT / "omni_edit_spatial_240.csv",
    }
    manifest_path = paths[args.stage]
    rows = read_rows(manifest_path)
    expected = {"screen": 48, "remaining": 147, "test": 204, "omni": 240}[args.stage]
    id_field = "video_id" if args.stage == "omni" else "sample_id"
    if len(rows) != expected or len({row[id_field] for row in rows}) != expected:
        raise ValueError(f"{args.stage}: expected {expected} unique samples")
    jobs, tasks = [], []
    for step in steps:
        for shard in range(4):
            run_name = f"{args.stage}_step{step:07d}_shard{shard:02d}"
            config_path = output / "configs" / f"{run_name}.json"
            config = {}
            for index, row in enumerate(rows, 1):
                if (index - 1) % 4 != shard:
                    continue
                source = Path(row["official_source_video"] if args.stage == "omni" else row["source_path"])
                if not source.is_file() or int(row["generation_seed"]) != 42:
                    raise ValueError(f"invalid seed or missing source: {row[id_field]}")
                instruction = row["editing_instruction"] if args.stage == "omni" else row["instruction"]
                config[f"{index:04d}"] = {
                    "interleave_array": [instruction, str(source)],
                    "element_dtype_array": ["text", "video"],
                    "istarget_in_interleave": [0, 0],
                }
                tasks.append({
                    "stage": args.stage, "checkpoint_step": step,
                    "sample_id": row[id_field], "sample_index": index,
                    "branch": row.get("branch", row.get("category", "")),
                    "instruction": instruction, "source_path": str(source),
                    "dataset_target_path": row.get("dataset_target_path", ""),
                    "generated_video": str(output / "outputs_raw" / run_name / f"{index:06d}.mp4"),
                    "generation_seed": 42, "gpu_slot": shard, "run_name": run_name,
                })
            freeze_json(config_path, config)
            jobs.append({
                "checkpoint_step": step, "gpu_slot": shard, "run_name": run_name,
                "sample_count": len(config), "config_path": str(config_path),
                "adapter_checkpoint": str(checkpoints / f"{step:07d}"),
                "command": ["bash", str(ROOT / "scripts/run_lora_video_edit.sh"),
                            str(output), profile["source"]["runtime"], profile["model"]["model_path"],
                            str(checkpoints / f"{step:07d}"), str(config_path), run_name],
            })
    freeze_json(output / f"{args.stage}_jobs.json", {
        "stage": args.stage, "generation_seed": 42, "shards_per_checkpoint": 4,
        "checkpoint_steps": steps, "generation_count": len(tasks),
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "promotion_decision_sha256": decision_hash, "jobs": jobs,
    })
    freeze_json(output / f"{args.stage}_generation_tasks.json", tasks)
    print(json.dumps({"stage": args.stage, "jobs": len(jobs), "generations": len(tasks)}))


if __name__ == "__main__":
    main()
