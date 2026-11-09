#!/usr/bin/env python3
"""Merge Short Subject local/API scores and compute paired Base-vs-LoRA results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
LOCAL = ARTIFACTS / "local_metrics"
ABSOLUTE = ARTIFACTS / "gemini_absolute_temp0_v1"
PAIRWISE = ARTIFACTS / "gemini_pairwise_temp0_v1"
STABILITY = ARTIFACTS / "gemini_stability_temp0_v1"
EXPECTED_SAMPLES = 91
MANIFEST_NAME = "short_subject_manifest.csv"
BENCHMARK_LABEL = "IVEBench Short / Subject Editing"
STAGES = {
    "short_subject": (91, "short_subject_manifest.csv", "IVEBench Short / Subject Editing"),
    "long_subject": (43, "manifest.csv", "IVEBench Long / Subject Editing"),
    "attribute_all": (94, "manifest.csv", "IVEBench Short+Long / Attribute Editing"),
}

LOCAL_METRICS = {
    "subject_consistency": "quality",
    "background_consistency": "quality",
    "temporal_flickering": "quality",
    "motion_smoothness": "quality",
    "vtss": "quality",
    "overall_semantic_consistency": "compliance",
    "phrase_semantic_consistency": "compliance",
    "semantic_fidelity": "fidelity",
    "motion_fidelity": "fidelity",
}
ALL_METRICS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "vtss", "overall_semantic_consistency",
    "phrase_semantic_consistency", "instruction_satisfaction",
    "semantic_fidelity", "motion_fidelity", "content_fidelity",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_summary(path: Path, tasks: int) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete" or summary.get("tasks_succeeded") != tasks or summary.get("tasks_failed") != 0:
        raise ValueError(f"incomplete API score set: {path}: {summary}")
    return summary


def local_scores(model: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    paths = list((LOCAL / model).glob("*/*.json"))
    metric_paths = {path.stem: path for path in paths if path.stem in LOCAL_METRICS}
    if set(metric_paths) != set(LOCAL_METRICS):
        raise ValueError(f"missing local metrics for {model}: {set(LOCAL_METRICS) - set(metric_paths)}")
    for metric, path in metric_paths.items():
        data = json.loads(path.read_text())
        if (data.get("status") != "complete"
                or data.get("records") != EXPECTED_SAMPLES
                or data.get("error_records") != 0):
            raise ValueError(f"invalid local result: {path}")
        for item in data["details"]:
            sample_id = Path(item["video_name"]).stem
            score = float(item["video_results"])
            if score == -1 or not math.isfinite(score):
                raise ValueError(f"invalid {metric} result for {sample_id}: {score}")
            result.setdefault(sample_id, {})[metric] = score
    if len(result) != EXPECTED_SAMPLES or any(set(values) != set(LOCAL_METRICS) for values in result.values()):
        raise ValueError(f"local metric coverage mismatch for {model}")
    return result


def normalized(metric: str, value: float) -> float:
    if metric in ("instruction_satisfaction", "content_fidelity"):
        return (value - 1.0) / 4.0
    if metric == "vtss":
        return (value + 0.05) / 0.15
    return value


def aggregate_values(model_rows: list[dict[str, float]]) -> dict:
    metric_means = {metric: mean(values[metric] for values in model_rows) for metric in ALL_METRICS}
    quality_values = {
        metric: normalized(metric, metric_means[metric])
        for metric in (
            "subject_consistency", "background_consistency", "temporal_flickering",
            "motion_smoothness", "vtss",
        )
    }
    quality = (
        quality_values["subject_consistency"]
        + quality_values["background_consistency"]
        + quality_values["temporal_flickering"]
        + quality_values["motion_smoothness"]
        + 4 * quality_values["vtss"]
    ) / 8
    compliance_values = [
        normalized("overall_semantic_consistency", metric_means["overall_semantic_consistency"]),
        normalized("phrase_semantic_consistency", metric_means["phrase_semantic_consistency"]),
        *([normalized("instruction_satisfaction", metric_means["instruction_satisfaction"])] * 3),
    ]
    fidelity_values = [
        normalized("semantic_fidelity", metric_means["semantic_fidelity"]),
        normalized("motion_fidelity", metric_means["motion_fidelity"]),
        *([normalized("content_fidelity", metric_means["content_fidelity"])] * 3),
    ]
    dimensions = {"quality": quality, "compliance": mean(compliance_values), "fidelity": mean(fidelity_values)}
    return {
        "total_score": mean(dimensions.values()), "dimensions": dimensions,
        "metric_means": metric_means,
        "quantity_accuracy": None,
        "quantity_accuracy_note": "structurally N/A because this stage contains no quantity_modification tasks; excluded from the compliance denominator",
    }


def aggregate(model_rows: dict[str, dict[str, float]]) -> dict:
    return aggregate_values(list(model_rows.values()))


def aggregate_from_ids(rows: dict[str, dict[str, float]], ids: list[str]) -> dict:
    return aggregate_values([rows[sample_id] for sample_id in ids])


def main() -> None:
    global ARTIFACTS, LOCAL, ABSOLUTE, PAIRWISE, STABILITY, EXPECTED_SAMPLES, MANIFEST_NAME, BENCHMARK_LABEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="short_subject")
    args = parser.parse_args()
    EXPECTED_SAMPLES, MANIFEST_NAME, BENCHMARK_LABEL = STAGES[args.stage]
    ARTIFACTS = ROOT / "artifacts" / args.stage
    LOCAL = ARTIFACTS / "local_metrics"
    ABSOLUTE = ARTIFACTS / "gemini_absolute_temp0_v1"
    PAIRWISE = ARTIFACTS / "gemini_pairwise_temp0_v1"
    STABILITY = ARTIFACTS / "gemini_stability_temp0_v1"
    full_tasks = EXPECTED_SAMPLES * 2
    stability_tasks = min(9, EXPECTED_SAMPLES) * 2
    absolute_summary = validate_summary(ABSOLUTE / "summary.json", full_tasks)
    pairwise_summary = validate_summary(PAIRWISE / "summary.json", full_tasks)
    stability_summary = validate_summary(STABILITY / "summary.json", stability_tasks)
    manifest = read_csv(ARTIFACTS / MANIFEST_NAME)
    if len(manifest) != EXPECTED_SAMPLES:
        raise ValueError(f"manifest must contain {EXPECTED_SAMPLES} samples")
    by_id = {row["sample_id"]: row for row in manifest}

    scores = {model: local_scores(model) for model in ("base", "lora_step677")}
    absolute = read_csv(ABSOLUTE / "scores.csv")
    if len(absolute) != full_tasks:
        raise ValueError(f"absolute score CSV must contain {full_tasks} rows")
    absolute_by_key = {(row["sample_id"], row["model_kind"]): row for row in absolute}
    if len(absolute_by_key) != full_tasks:
        raise ValueError("duplicate absolute API scores")
    for (sample_id, model), item in absolute_by_key.items():
        scores[model][sample_id]["instruction_satisfaction"] = float(item["instruction_satisfaction"])
        scores[model][sample_id]["content_fidelity"] = float(item["content_fidelity"])

    pairwise_rows = read_csv(PAIRWISE / "scores.csv")
    if len(pairwise_rows) != full_tasks:
        raise ValueError(f"pairwise score CSV must contain {full_tasks} rows")
    pair_by_sample: dict[str, list[dict[str, str]]] = {}
    for item in pairwise_rows:
        pair_by_sample.setdefault(item["sample_id"], []).append(item)
    if set(pair_by_sample) != set(by_id) or any(len(values) != 2 for values in pair_by_sample.values()):
        raise ValueError("pairwise A/B and B/A coverage mismatch")

    stability_rows = read_csv(STABILITY / "scores.csv")
    if len(stability_rows) != stability_tasks:
        raise ValueError(f"stability score CSV must contain {stability_tasks} rows")
    stability_diffs = []
    for item in stability_rows:
        original = absolute_by_key[(item["sample_id"], item["model_kind"])]
        stability_diffs.append({
            "sample_id": item["sample_id"], "model_kind": item["model_kind"],
            "instruction_abs_diff": abs(float(item["instruction_satisfaction"]) - float(original["instruction_satisfaction"])),
            "fidelity_abs_diff": abs(float(item["content_fidelity"]) - float(original["content_fidelity"])),
        })

    sample_rows = []
    ordered_ids = sorted(by_id, key=lambda value: int(by_id[value].get("generation_id") or by_id[value]["official_id"]))
    for sequence_index, sample_id in enumerate(ordered_ids, 1):
        source = by_id[sample_id]
        base_api = absolute_by_key[(sample_id, "base")]
        lora_api = absolute_by_key[(sample_id, "lora_step677")]
        pair = sorted(pair_by_sample[sample_id], key=lambda item: int(item["order"]))
        row: dict[str, object] = {
            "sample_id": sample_id, "sequence_index": sequence_index,
            "generation_id": int(source.get("generation_id") or source["official_id"]),
            "official_id": int(source["official_id"]), "subset": source.get("subset", "short"),
            "category": source["category"], "subcategory": source["subcategory"],
            "instruction": source["edit_prompt"], "source_prompt": source["source_prompt"],
            "target_prompt": source["target_prompt"], "generation_seed": 42,
            "base_instruction_reason": base_api["instruction_reason"],
            "base_fidelity_reason": base_api["fidelity_reason"],
            "lora_instruction_reason": lora_api["instruction_reason"],
            "lora_fidelity_reason": lora_api["fidelity_reason"],
        }
        for metric in ALL_METRICS:
            base_value = scores["base"][sample_id][metric]
            lora_value = scores["lora_step677"][sample_id][metric]
            row[f"base_{metric}"] = base_value
            row[f"lora_{metric}"] = lora_value
            row[f"delta_{metric}"] = lora_value - base_value
        for dimension in ("instruction", "fidelity", "overall"):
            values = [item[f"normalized_{dimension}_winner"] for item in pair]
            row[f"pairwise_{dimension}_order1"] = values[0]
            row[f"pairwise_{dimension}_order2"] = values[1]
            row[f"pairwise_{dimension}_consensus"] = values[0] if values[0] == values[1] else "order_inconsistent"
        row["pairwise_order1_reason"] = pair[0]["reason"]
        row["pairwise_order2_reason"] = pair[1]["reason"]
        sample_rows.append(row)

    reports = {model: aggregate(values) for model, values in scores.items()}
    rng = random.Random(42)
    identifiers = sorted(by_id)
    bootstrap = []
    for _ in range(10_000):
        sampled = [identifiers[rng.randrange(len(identifiers))] for _ in identifiers]
        base = aggregate_from_ids(scores["base"], sampled)
        lora = aggregate_from_ids(scores["lora_step677"], sampled)
        bootstrap.append(lora["total_score"] - base["total_score"])
    bootstrap.sort()
    lower = bootstrap[math.floor(0.025 * len(bootstrap))]
    upper = bootstrap[math.ceil(0.975 * len(bootstrap)) - 1]

    pairwise_counts = {}
    for dimension in ("instruction", "fidelity", "overall"):
        counts = {value: 0 for value in ("base", "lora_step677", "tie", "order_inconsistent")}
        for row in sample_rows:
            counts[str(row[f"pairwise_{dimension}_consensus"])] += 1
        pairwise_counts[dimension] = counts
    total_cost = sum(float(summary["reported_cost_usd"]) for summary in (
        absolute_summary, pairwise_summary, stability_summary))
    report = {
        "status": "complete", "benchmark": BENCHMARK_LABEL,
        "stage": args.stage, "samples": EXPECTED_SAMPLES, "generation_seed": 42,
        "comparison": "Lance Base vs LoRA step 677",
        "base": reports["base"], "lora_step677": reports["lora_step677"],
        "lora_minus_base_total": reports["lora_step677"]["total_score"] - reports["base"]["total_score"],
        "paired_bootstrap_total_delta_95_ci": [lower, upper], "bootstrap_seed": 42,
        "bootstrap_replicates": 10_000, "pairwise_consensus_counts": pairwise_counts,
        "judge_stability": {
            "tasks": len(stability_diffs),
            "instruction_exact_match_rate": mean(item["instruction_abs_diff"] == 0 for item in stability_diffs),
            "fidelity_exact_match_rate": mean(item["fidelity_abs_diff"] == 0 for item in stability_diffs),
            "instruction_mean_absolute_difference": mean(item["instruction_abs_diff"] for item in stability_diffs),
            "fidelity_mean_absolute_difference": mean(item["fidelity_abs_diff"] for item in stability_diffs),
            "details": stability_diffs,
        },
        "api_cost_usd": total_cost,
        "scoring_note": "Official IVEBench aggregation and weights, with Gemini API replacing local Qwen IS/CF; quantity accuracy is N/A on this category-restricted subset.",
        "comparability_note": "This category-restricted score is suitable for paired Base-vs-LoRA comparison, not a replacement for the full 600-video IVEBench headline score.",
    }
    write_csv(ARTIFACTS / "sample_results.csv", sample_rows)
    write_json(ARTIFACTS / "evaluation_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "judge_stability"}, indent=2))


if __name__ == "__main__":
    main()
