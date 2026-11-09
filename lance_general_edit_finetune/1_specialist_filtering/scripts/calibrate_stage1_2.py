#!/usr/bin/env python3
"""Calibrate Stage 1.2 robot decisions from the compact overall review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v2"
METRICS = ("crop_musiq_p10", "siglip_quality_margin", "nonedit_dino_cosine_p10")
DECISION_FIELDS = [
    "filter_version",
    "sample_id",
    "stage1_2_decision",
    "stage1_2_reason",
    "stage1_2_badness",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def robust_parameters(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in METRICS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        result[field] = {"median": float(median), "scale": max(float(q75 - q25), 1e-8)}
    return result


def badness(row: dict[str, str], parameters: dict[str, dict[str, float]]) -> float:
    return max(
        (parameters[field]["median"] - float(row[field])) / parameters[field]["scale"]
        for field in METRICS
    )


def human_labels(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, str] = {}
    for row in read_csv(path):
        value = row.get("overall_status", "")
        if value in {"usable", "unusable", "uncertain"}:
            result[row["sample_id"]] = value
    return result


def threshold_for(
    labeled: list[tuple[float, str]],
    desired: str,
    minimum_precision: float,
    minimum_examples: int,
    upper_tail: bool,
) -> tuple[float | None, dict[str, float | int]]:
    values = sorted({score for score, _ in labeled})
    candidates: list[tuple[int, float, float]] = []
    for threshold in values:
        selected = [
            label
            for score, label in labeled
            if (score >= threshold if upper_tail else score <= threshold)
            and label != "uncertain"
        ]
        if len(selected) < minimum_examples:
            continue
        precision = selected.count(desired) / len(selected)
        if precision >= minimum_precision:
            candidates.append((len(selected), precision, threshold))
    if not candidates:
        return None, {"examples": 0, "precision": 0.0}
    examples, precision, threshold = max(candidates, key=lambda item: (item[0], item[1]))
    return threshold, {"examples": examples, "precision": precision}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_RUN / "stage1_2_robot_scores.csv")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_RUN / "stage1_2_thresholds.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_2_decisions.csv")
    parser.add_argument("--accept-precision", type=float, default=0.95)
    parser.add_argument("--reject-precision", type=float, default=0.90)
    parser.add_argument("--minimum-examples", type=int, default=10)
    args = parser.parse_args()

    scores = [
        row
        for row in read_csv(args.scores)
        if row.get("technical_status") == "pass" and row.get("model_backend") != "mock"
        and all(row.get(field) for field in METRICS)
    ]
    if not scores:
        raise ValueError("no complete real Stage 1.2 scores")
    parameters = robust_parameters(scores)
    labels = human_labels(args.review)
    labeled = [(badness(row, parameters), labels[row["sample_id"]]) for row in scores if row["sample_id"] in labels]
    accept_threshold, accept_stats = threshold_for(
        labeled, "usable", args.accept_precision, args.minimum_examples, False
    )
    reject_threshold, reject_stats = threshold_for(
        labeled, "unusable", args.reject_precision, args.minimum_examples, True
    )
    enabled = accept_threshold is not None and reject_threshold is not None
    thresholds = {
        "filter_version": "stage1_2_robot_filter_v1",
        "calibration_status": "enabled" if enabled else "pending_or_insufficient",
        "metrics": list(METRICS),
        "robust_parameters": parameters,
        "labeled_examples": len(labeled),
        "accept": {"threshold_max": accept_threshold, **accept_stats},
        "reject": {"threshold_min": reject_threshold, **reject_stats},
        "minimum_accept_precision": args.accept_precision,
        "minimum_reject_precision": args.reject_precision,
        "minimum_examples": args.minimum_examples,
    }
    args.thresholds.parent.mkdir(parents=True, exist_ok=True)
    args.thresholds.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False) + "\n")

    decisions: list[dict[str, str]] = []
    by_id = {row["sample_id"]: row for row in scores}
    for sample_id, row in by_id.items():
        value = badness(row, parameters)
        if not enabled:
            decision, reason = "uncertain", "stage1_2_not_calibrated"
        elif reject_threshold is not None and value >= reject_threshold:
            decision, reason = "rejected", "robot_visual_quality_hard_fail"
        elif accept_threshold is not None and value <= accept_threshold:
            decision, reason = "accepted", "robot_visual_quality_pass"
        else:
            decision, reason = "uncertain", "robot_visual_quality_boundary"
        decisions.append(
            {
                "filter_version": "stage1_2_robot_filter_v1",
                "sample_id": sample_id,
                "stage1_2_decision": decision,
                "stage1_2_reason": reason,
                "stage1_2_badness": f"{value:.8f}",
            }
        )
    write_csv(args.output, decisions)
    print(
        json.dumps(
            {
                "scores": len(scores),
                "labeled": len(labeled),
                "calibration_status": thresholds["calibration_status"],
                "decisions": dict(Counter(row["stage1_2_decision"] for row in decisions)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

