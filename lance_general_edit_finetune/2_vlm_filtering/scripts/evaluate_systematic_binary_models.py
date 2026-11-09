#!/usr/bin/env python3
"""Compare binary Stage-2 model decisions with the frozen human test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DECISION_FIELDS = ("model_decision", "decision", "final_decision")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def decision(row: dict[str, str]) -> str:
    return next((row.get(field, "").strip().lower() for field in DECISION_FIELDS if row.get(field)), "")


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    tp = sum(r["human_decision"] == "rejected" and r["model_decision"] == "rejected" for r in rows)
    fn = sum(r["human_decision"] == "rejected" and r["model_decision"] == "accepted" for r in rows)
    fp = sum(r["human_decision"] == "accepted" and r["model_decision"] == "rejected" for r in rows)
    tn = sum(r["human_decision"] == "accepted" and r["model_decision"] == "accepted" for r in rows)
    reject_recall = ratio(tp, tp + fn)
    accept_recall = ratio(tn, tn + fp)
    balanced_accuracy = (
        (reject_recall + accept_recall) / 2
        if reject_recall is not None and accept_recall is not None
        else None
    )
    return {
        "rows": len(rows),
        "human_accepted": tn + fp,
        "human_rejected": tp + fn,
        "true_reject": tp,
        "missed_reject": fn,
        "false_reject": fp,
        "true_accept": tn,
        "reject_recall": reject_recall,
        "accept_recall": accept_recall,
        "accept_false_reject_rate": ratio(fp, fp + tn),
        "reject_precision": ratio(tp, tp + fp),
        "accuracy": ratio(tp + tn, len(rows)),
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate(path: Path, labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    predictions = read_csv(path)
    by_id: dict[str, dict[str, str]] = {}
    for row in predictions:
        sample_id = row.get("sample_id", "")
        if sample_id in by_id:
            raise ValueError(f"duplicate prediction in {path}: {sample_id}")
        if sample_id in labels:
            value = decision(row)
            if value not in {"accepted", "rejected"}:
                raise ValueError(
                    f"model prediction must be binary for {sample_id}: {value!r}"
                )
            by_id[sample_id] = row
    missing = set(labels) - set(by_id)
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} scored sample IDs")

    joined: list[dict[str, str]] = []
    for sample_id, label in labels.items():
        row = dict(label)
        row["model_decision"] = decision(by_id[sample_id])
        joined.append(row)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined:
        for field in ("audit_operation", "audit_relation", "dataset", "branch"):
            groups[f"{field}={row.get(field, '')}"].append(row)
    return {
        "score_file": str(path),
        "model_id": next(
            (row.get("model_id", "") for row in predictions if row.get("model_id")),
            path.stem,
        ),
        "overall": metrics(joined),
        "subgroups": {name: metrics(rows) for name, rows in sorted(groups.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    label_rows = read_csv(args.labels)
    labels = {row["sample_id"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("duplicate label sample IDs")
    invalid = [
        row["sample_id"]
        for row in label_rows
        if row.get("human_decision") not in {"accepted", "rejected"}
    ]
    if invalid:
        raise ValueError("label file must contain accepted/rejected only")

    results = [evaluate(path, labels) for path in args.scores]
    payload = {
        "labels": str(args.labels),
        "scored_rows": len(labels),
        "positive_class": "rejected",
        "primary_metrics": [
            "reject_recall",
            "balanced_accuracy",
            "accuracy",
            "accept_false_reject_rate",
        ],
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        value = result["overall"]
        print(
            f"{result['model_id']}: n={value['rows']} "
            f"reject_recall={value['reject_recall']} "
            f"false_reject={value['accept_false_reject_rate']}"
        )


if __name__ == "__main__":
    main()
