#!/usr/bin/env python3
"""Calibrate conservative Stage 1.1 thresholds against the existing human audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "artifacts/filter_run_v1/quality_calibration_scores.csv"
DEFAULT_REVIEWS = ROOT / "artifacts/review_results/specialist_error_survey_460_reco_v3_review.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/filter_run_v1/stage1_1_thresholds.json"

METRICS = {
    "musiq_p10": "low_bad",
    "sharpness_p10": "low_bad",
    "frame_diff_p05": "low_bad",
    "freeze_ratio": "high_bad",
    "black_frame_ratio": "high_bad",
    "white_frame_ratio": "high_bad",
    "dino_max_feature_jump": "high_bad",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def badness_auc(pass_values: np.ndarray, fail_values: np.ndarray, direction: str) -> float:
    if direction == "low_bad":
        comparisons = fail_values[:, None] < pass_values[None, :]
    else:
        comparisons = fail_values[:, None] > pass_values[None, :]
    return float(np.mean(comparisons))


def triggered(value: float, threshold: float, direction: str) -> bool:
    return value < threshold if direction == "low_bad" else value > threshold


def decision(values: dict[str, float], thresholds: dict[str, dict[str, float | str]]) -> tuple[str, list[str]]:
    hard_reasons: list[str] = []
    uncertain_reasons: list[str] = []
    low_musiq = triggered(
        values["musiq_p10"], float(thresholds["musiq_p10"]["hard_threshold"]), "low_bad"
    )
    low_sharpness = triggered(
        values["sharpness_p10"],
        float(thresholds["sharpness_p10"]["hard_threshold"]),
        "low_bad",
    )
    if low_musiq and low_sharpness:
        hard_reasons.append("joint_extreme_low_quality")
    if values["frame_diff_p05"] < float(thresholds["frame_diff_p05"]["zero_pass_threshold"]):
        hard_reasons.append("extreme_freeze_repeat")
    if values["black_frame_ratio"] > float(thresholds["black_frame_ratio"]["zero_pass_threshold"]):
        hard_reasons.append("black_frames")
    if values["white_frame_ratio"] > float(thresholds["white_frame_ratio"]["zero_pass_threshold"]):
        hard_reasons.append("white_frames")
    if hard_reasons:
        return "rejected", hard_reasons

    for metric, direction in METRICS.items():
        if metric in {"black_frame_ratio", "white_frame_ratio"}:
            continue
        if not bool(thresholds[metric]["enabled_for_uncertain"]):
            continue
        threshold = float(thresholds[metric]["uncertain_threshold"])
        if triggered(values[metric], threshold, direction):
            uncertain_reasons.append(f"boundary_{metric}")
    if uncertain_reasons:
        return "uncertain", uncertain_reasons
    return "accepted", []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scores = read_csv(args.scores)
    reviews = {row["sample_id"]: row for row in read_csv(args.reviews)}
    labeled: list[tuple[str, dict[str, float], dict[str, str]]] = []
    for score in scores:
        status = reviews.get(score["sample_id"], {}).get(f"{score['role']}_status", "")
        if status not in {"pass", "hard_fail", "uncertain"}:
            continue
        if score["technical_status"] != "pass" or any(not score[name] for name in METRICS):
            continue
        labeled.append((status, {name: float(score[name]) for name in METRICS}, score))

    pass_rows = [values for status, values, _ in labeled if status == "pass"]
    fail_rows = [values for status, values, _ in labeled if status == "hard_fail"]
    if len(pass_rows) < 50 or len(fail_rows) < 10:
        raise ValueError("not enough pass/hard_fail edited-side labels to calibrate thresholds")

    thresholds: dict[str, dict[str, float | str]] = {}
    for metric, direction in METRICS.items():
        pass_values = np.asarray([row[metric] for row in pass_rows], dtype=np.float64)
        fail_values = np.asarray([row[metric] for row in fail_rows], dtype=np.float64)
        low_quantile = 2 if direction == "low_bad" else 98
        uncertain_quantile = 5 if direction == "low_bad" else 95
        auc = badness_auc(pass_values, fail_values, direction)
        thresholds[metric] = {
            "direction": direction,
            "hard_threshold": float(np.percentile(pass_values, low_quantile)),
            "uncertain_threshold": float(np.percentile(pass_values, uncertain_quantile)),
            "zero_pass_threshold": float(pass_values.min() if direction == "low_bad" else pass_values.max()),
            "badness_auc": auc,
            "enabled_for_uncertain": auc >= 0.6,
            "pass_count": len(pass_values),
            "hard_fail_count": len(fail_values),
        }

    outcomes = Counter()
    reasons = Counter()
    outcomes_by_branch: dict[str, Counter[str]] = {}
    for human_status, values, score in labeled:
        automatic, automatic_reasons = decision(values, thresholds)
        outcomes[f"{human_status}->{automatic}"] += 1
        outcomes_by_branch.setdefault(score["branch"], Counter())[automatic] += 1
        reasons.update(automatic_reasons)

    output = {
        "filter_version": "general_edit_stage1_pilot_v1",
        "calibration_source": "specialist_error_survey_460_reco_v3 edited-side labels",
        "labeled_videos": len(labeled),
        "human_status_counts": dict(Counter(status for status, _, _ in labeled)),
        "threshold_method": {
            "hard_low_quality": "MUSIQ p10 and sharpness p10 both beyond the 2% pass tail",
            "hard_freeze_or_blank": "strictly beyond the observed pass range",
            "uncertain_boundary": "any metric with calibration AUC >= 0.6 beyond the 5% pass tail",
            "note": "DINOv2 and single quality scores only create uncertain samples; they do not reject alone",
        },
        "metrics": thresholds,
        "calibration_outcomes": dict(outcomes),
        "automatic_reason_counts": dict(reasons),
        "automatic_decisions_by_branch": {
            branch: dict(counts) for branch, counts in sorted(outcomes_by_branch.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
