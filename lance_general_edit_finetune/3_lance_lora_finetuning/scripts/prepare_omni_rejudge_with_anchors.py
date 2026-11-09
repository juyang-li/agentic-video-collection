#!/usr/bin/env python3
"""Build an interleaved LoRA/Base rejudge manifest with calibration anchors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
BASE = ROOT.parents[1] / "model_evaluation/lance_base"
METRICS = ("accuracy", "preservation", "realism", "consistency", "overall_score")


def read_scores(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full-base", action="store_true",
                        help="also score all 240 Base outputs under the new judge protocol")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    fine_tasks = json.loads((EVALUATION / "omni_generation_tasks.json").read_text())
    if len(fine_tasks) != 240 or {int(task["sample_index"]) for task in fine_tasks} != set(range(1, 241)):
        raise ValueError("expected the complete OmniEdit Spatial 240 task set")
    fine_tasks.sort(key=lambda task: int(task["sample_index"]))
    vertex = read_scores(BASE / "gemini38_low_vertex_v2/scores.csv")
    studio = read_scores(BASE / "gemini38_low_openrouter_v1/scores.csv")

    combined, anchor_plan = [], []
    for task in fine_tasks:
        fine = {**task, "stage": "omni_rejudge_v3", "variant": "finetune_step677"}
        combined.append(fine)
        sample_id = task["sample_id"]
        if sample_id not in vertex:
            raise ValueError(f"missing historical Base output: {sample_id}")
        reference = vertex[sample_id]
        if reference["source_path"] != task["source_path"]:
            raise ValueError(f"Base source mismatch: {sample_id}")
        if args.full_base:
            combined.append({
                **task,
                "stage": "omni_rejudge_base_full_temp0_v1",
                "checkpoint_step": "base",
                "variant": "base_full",
                "generated_video": reference["generated_video"],
                "run_name": "base_full_interleaved_temp0",
            })
        if int(task["sample_index"]) % 10:
            continue
        if sample_id not in vertex or sample_id not in studio:
            raise ValueError(f"anchor lacks two historical Base scores: {sample_id}")
        anchor = {
            **task,
            "stage": "omni_rejudge_base_anchor_v3",
            "checkpoint_step": "base_anchor",
            "variant": "base_anchor",
            "generated_video": reference["generated_video"],
            "run_name": "base_anchor_interleaved",
        }
        combined.append(anchor)
        anchor_plan.append({
            "sample_id": sample_id,
            "sample_index": int(task["sample_index"]),
            "category": task["branch"],
            "historical_vertex": {key: float(reference[key]) for key in METRICS},
            "historical_studio": {key: float(studio[sample_id][key]) for key in METRICS},
        })

    expected = 504 if args.full_base else 264
    if len(combined) != expected or len(anchor_plan) != 24 or len({x["category"] for x in anchor_plan}) != 24:
        raise ValueError(f"expected {expected} interleaved tasks and one Base anchor for each of 24 categories")
    write_json(output / "tasks_with_base_anchors.json", combined)
    write_json(output / "calibration_plan.json", {
        "purpose": "same-protocol interleaved LoRA/Base rejudge with repeated Base severity anchors",
        "judge_result_scope": {
            "finetune": 240,
            "base_full": 240 if args.full_base else 0,
            "base_anchors_excluded_from_model_means": 24,
        },
        "interleaving": ("LoRA then Base for each sample, plus one repeated category-covering Base anchor after every ten samples"
                         if args.full_base else "one category-covering Base anchor after every ten LoRA tasks"),
        "historical_base_runs": ["gemini38_low_openrouter_v1", "gemini38_low_vertex_v2"],
        "anchors": anchor_plan,
    })
    print(json.dumps({"output": str(output), "tasks": expected, "finetune": 240,
                      "base_full": 240 if args.full_base else 0, "anchors": 24}))


if __name__ == "__main__":
    main()
