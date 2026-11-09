#!/usr/bin/env python3
"""Explore how well continuous VEFX scores separate human usable/unusable pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def auc(labels: list[int], values: list[float]) -> float | None:
    """Tie-aware Mann-Whitney AUC; larger value means more likely bad."""
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ordered = sorted(zip(values, labels), key=lambda item: item[0])
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[position] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, ordered) if label)
    return (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def point(labels: list[int], quality: list[float], threshold: float) -> dict[str, Any]:
    # Low VEFX quality is predicted unusable.
    predictions = [value <= threshold for value in quality]
    tp = sum(label == 1 and prediction for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and not prediction for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and not prediction for label, prediction in zip(labels, predictions))
    return {
        "threshold_reject_at_or_below": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unusable_recall": ratio(tp, tp + fn),
        "usable_false_reject_rate": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(labels)),
        "rejected_precision": ratio(tp, tp + fp),
    }


def evaluate_metric(rows: list[dict[str, str]], metric: str) -> dict[str, Any]:
    labels = [int(row["human_status"] == "unusable") for row in rows]
    quality = [float(row[metric]) for row in rows]
    thresholds = sorted(set(quality))
    points = [point(labels, quality, threshold) for threshold in thresholds]
    best = max(
        points,
        key=lambda item: (
            (item["unusable_recall"] or 0) - (item["usable_false_reject_rate"] or 0),
            item["accuracy"] or 0,
        ),
    )
    recall80 = [item for item in points if (item["unusable_recall"] or 0) >= 0.8]
    min_fpr80 = min(
        recall80,
        key=lambda item: (item["usable_false_reject_rate"], -item["accuracy"]),
    ) if recall80 else None
    badness = [-value for value in quality]
    return {
        "metric": metric,
        "badness_roc_auc": auc(labels, badness),
        "best_youden_on_same_data": best,
        "minimum_fpr_at_recall80_on_same_data": min_fpr80,
        "usable_mean": sum(
            value for value, label in zip(quality, labels) if not label
        ) / max(1, sum(not label for label in labels)),
        "unusable_mean": sum(
            value for value, label in zip(quality, labels) if label
        ) / max(1, sum(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read_csv(args.scores) if row.get("status") == "pass"]
    decisive = [row for row in rows if row["human_status"] in {"usable", "unusable"}]
    one_sided = [row for row in decisive if row["generated_side"] != "BOTH"]
    for row in rows:
        row["Min"] = str(min(float(row[k]) for k in ("IF", "RQ", "EE")))
        row["GeoAgg"] = str(
            (float(row["IF"]) ** 2 * float(row["RQ"]) * float(row["EE"])) ** 0.25
        )
    payload = {
        "score_file": str(args.scores),
        "rows": len(rows),
        "human_counts": dict(Counter(row["human_status"] for row in rows)),
        "interpretation": (
            "Thresholds are selected and evaluated on the same development set; "
            "they are diagnostic only and require a fresh frozen validation set."
        ),
        "all_decisive": {
            metric: evaluate_metric(decisive, metric)
            for metric in ("IF", "RQ", "EE", "Overall", "Min", "GeoAgg")
        },
        "one_sided_generated_only": {
            metric: evaluate_metric(one_sided, metric)
            for metric in ("IF", "RQ", "EE", "Overall", "Min", "GeoAgg")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
