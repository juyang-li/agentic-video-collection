#!/usr/bin/env python3
"""Separate LoRA results from Base severity anchors and compare all prior runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
BASE = ROOT.parents[1] / "model_evaluation/lance_base"
METRICS = ("accuracy", "preservation", "realism", "consistency", "overall_score")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in rows(path)}


def score(row: dict[str, str], metric: str) -> float:
    return float(row[metric])


def means(group: list[dict[str, str]]) -> dict[str, float]:
    return {metric: statistics.mean(score(row, metric) for row in group) for metric in METRICS}


def comparison(left: dict[str, dict[str, str]], right: dict[str, dict[str, str]]) -> dict:
    shared = sorted(left.keys() & right.keys())
    output = {"paired_samples": len(shared), "metrics": {}}
    for metric in METRICS:
        differences = [score(left[key], metric) - score(right[key], metric) for key in shared]
        output["metrics"][metric] = {
            "mean_left_minus_right": statistics.mean(differences),
            "mean_absolute_difference": statistics.mean(abs(value) for value in differences),
            "exact_agreement_rate": sum(value == 0 for value in differences) / len(differences),
            "within_one_rate": sum(abs(value) <= 1 for value in differences) / len(differences),
        }
    differences = [score(left[key], "overall_score") - score(right[key], "overall_score") for key in shared]
    output["overall_wins_ties_losses"] = {
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }
    return output


def spatial_bootstrap(left: dict[str, dict[str, str]], right: dict[str, dict[str, str]], resamples: int = 10000) -> dict:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for sample_id in sorted(left.keys() & right.keys()):
        category = left[sample_id]["branch"]
        if category != right[sample_id]["category"]:
            raise ValueError(f"category mismatch: {sample_id}")
        grouped[category].append([score(left[sample_id], metric) - score(right[sample_id], metric)
                                  for metric in METRICS])
    if len(grouped) != 24 or any(len(values) != 10 for values in grouped.values()):
        raise ValueError("stratified bootstrap requires 24 categories of ten paired samples")
    rng = np.random.default_rng(42)
    bootstrap = np.zeros((resamples, len(METRICS)))
    category_means = {}
    for category, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        category_means[category] = dict(zip(METRICS, array.mean(axis=0).tolist()))
        bootstrap += array[rng.integers(0, 10, size=(resamples, 10))].mean(axis=1) / 24
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        "paired_stratified_bootstrap_95_ci": {
            metric: intervals[:, position].tolist() for position, metric in enumerate(METRICS)
        },
        "bootstrap_seed": 42,
        "bootstrap_resamples": resamples,
        "bootstrap_scope": "resample pairs within each category; conditional on one fixed judge response and generation seed",
        "category_mean_differences": category_means,
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    combined = rows(root / "scores_with_anchors/scores.csv")
    fine = [row for row in combined if row["variant"] == "finetune_step677"]
    anchors = [row for row in combined if row["variant"] == "base_anchor"]
    if len(fine) != 240 or len(anchors) != 24:
        raise ValueError(f"incomplete results: finetune={len(fine)} anchors={len(anchors)}")
    if len({row["sample_id"] for row in fine}) != 240 or len({row["branch"] for row in anchors}) != 24:
        raise ValueError("duplicate LoRA result or incomplete category anchor coverage")

    new_fine = {row["sample_id"]: row for row in fine}
    new_anchor = {row["sample_id"]: row for row in anchors}
    old_fine = index(EVALUATION / "omni_scores_vertex_v2/scores.csv")
    base_vertex = index(BASE / "gemini38_low_vertex_v2/scores.csv")
    base_studio = index(BASE / "gemini38_low_openrouter_v1/scores.csv")
    summary = json.loads((root / "scores_with_anchors/summary.json").read_text())
    report = {
        "status": "complete",
        "protocol": json.loads((root / "scores_with_anchors/protocol.json").read_text()),
        "cost": {
            "known_usd": summary["reported_cost_usd_known_requests"],
            "requests_without_reported_cost": summary["requests_without_reported_cost"],
            "requests_including_retries": summary["all_recorded_requests_including_retries"],
        },
        "new_finetune": {
            "samples": 240,
            "mean_scores": means(fine),
            "accuracy_distribution": dict(sorted(Counter(int(row["accuracy"]) for row in fine).items())),
            "accuracy_zero_count": sum(int(row["accuracy"]) == 0 for row in fine),
        },
        "base_anchors": {
            "samples": 24,
            "categories": 24,
            "new_mean_scores": means(anchors),
            "new_vs_historical_vertex": comparison(new_anchor, base_vertex),
            "new_vs_historical_studio": comparison(new_anchor, base_studio),
            "new_accuracy_zero_count": sum(int(row["accuracy"]) == 0 for row in anchors),
        },
        "paired_comparisons": {
            "new_finetune_vs_historical_base_vertex": comparison(new_fine, base_vertex),
            "new_finetune_vs_old_finetune_vertex": comparison(new_fine, old_fine),
        },
    }
    report["paired_comparisons"]["new_finetune_vs_historical_base_vertex"].update(
        spatial_bootstrap(new_fine, base_vertex)
    )
    write_json(root / "REJUDGE_REPORT.json", report)
    with (root / "finetune_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fine[0]))
        writer.writeheader(); writer.writerows(fine)
    anchor_accuracy_shift = report["base_anchors"]["new_vs_historical_vertex"]["metrics"]["accuracy"]["mean_left_minus_right"]
    overall_shift = report["base_anchors"]["new_vs_historical_vertex"]["metrics"]["overall_score"]["mean_left_minus_right"]
    new = report["new_finetune"]["mean_scores"]
    paired = report["paired_comparisons"]["new_finetune_vs_historical_base_vertex"]
    old = report["paired_comparisons"]["new_finetune_vs_old_finetune_vertex"]
    markdown = f"""# OmniEdit Spatial LoRA step 677 · Vertex v3 独立重评

- LoRA：240/240；Base 宽严度哨兵：24/24（24 类各 1 条，穿插评分）。
- 新 LoRA 均分：Accuracy `{new['accuracy']:.4f}`，Preservation `{new['preservation']:.4f}`，Realism `{new['realism']:.4f}`，Consistency `{new['consistency']:.4f}`，Overall `{new['overall_score']:.4f}`。
- 新 LoRA Accuracy=0：`{report['new_finetune']['accuracy_zero_count']}` 条。
- Base 哨兵相对历史 Vertex：Accuracy 均值偏移 `{anchor_accuracy_shift:+.4f}`，Overall 均值偏移 `{overall_shift:+.4f}`。
- 新 LoRA 相对历史 Base Vertex 的 Overall 均值差：`{paired['metrics']['overall_score']['mean_left_minus_right']:+.4f}`，分层配对 bootstrap 95% CI `{paired['paired_stratified_bootstrap_95_ci']['overall_score'][0]:+.4f}` 至 `{paired['paired_stratified_bootstrap_95_ci']['overall_score'][1]:+.4f}`；胜/平/负 `{paired['overall_wins_ties_losses']['wins']}/{paired['overall_wins_ties_losses']['ties']}/{paired['overall_wins_ties_losses']['losses']}`。
- 新 LoRA 相对旧 LoRA 评分的 Overall 均值差：`{old['metrics']['overall_score']['mean_left_minus_right']:+.4f}`。
- 已知 API 费用：`${report['cost']['known_usd']:.6f}`；无费用回执请求：`{report['cost']['requests_without_reported_cost']}`。

完整统计见 `REJUDGE_REPORT.json`；仅 LoRA 结果见 `finetune_scores.csv`；所有原始请求回执保存在 `scores_with_anchors/records/`。
"""
    (root / "RESULTS.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
