#!/usr/bin/env python3
"""Evaluate one or more pair-judge score CSVs against frozen human labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("parse_status") == "pass"]
    decisive_human = [
        row for row in parsed if row.get("human_status") in {"usable", "unusable"}
    ]
    model_decisive = [
        row for row in decisive_human
        if row.get("model_decision") in {"accepted", "rejected"}
    ]
    good = [row for row in decisive_human if row["human_status"] == "usable"]
    bad = [row for row in decisive_human if row["human_status"] == "unusable"]
    caught = sum(row["model_decision"] == "rejected" for row in bad)
    false_reject = sum(row["model_decision"] == "rejected" for row in good)
    correct = sum(
        (row["human_status"] == "usable" and row["model_decision"] == "accepted")
        or (row["human_status"] == "unusable" and row["model_decision"] == "rejected")
        for row in decisive_human
    )
    selective_correct = sum(
        (row["human_status"] == "usable" and row["model_decision"] == "accepted")
        or (row["human_status"] == "unusable" and row["model_decision"] == "rejected")
        for row in model_decisive
    )
    return {
        "rows": len(rows),
        "parsed": len(parsed),
        "parse_errors": len(rows) - len(parsed),
        "human_counts": dict(Counter(row.get("human_status", "") for row in parsed)),
        "decision_counts": dict(Counter(row.get("model_decision", "") for row in parsed)),
        "confusion": dict(Counter(
            f"human={row.get('human_status','')}|model={row.get('model_decision','')}"
            for row in parsed
        )),
        "decisive_human_rows": len(decisive_human),
        "model_decision_coverage": ratio(len(model_decisive), len(decisive_human)),
        "strict_accuracy_uncertain_is_wrong": ratio(correct, len(decisive_human)),
        "selective_accuracy": ratio(selective_correct, len(model_decisive)),
        "unusable_recall": ratio(caught, len(bad)),
        "usable_false_reject_rate": ratio(false_reject, len(good)),
        "rejected_precision": ratio(
            caught,
            sum(row["model_decision"] == "rejected" for row in decisive_human),
        ),
        "accepted_precision": ratio(
            sum(
                row["human_status"] == "usable" and row["model_decision"] == "accepted"
                for row in decisive_human
            ),
            sum(row["model_decision"] == "accepted" for row in decisive_human),
        ),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0) for row in parsed),
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in parsed),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in parsed),
    }


def evaluate(path: Path, labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    scores = read_csv(path)
    joined: list[dict[str, str]] = []
    for score in scores:
        label = labels.get(score.get("sample_id", ""))
        if label is None:
            continue
        row = dict(score)
        row.update({
            "cohort": label.get("cohort", ""),
            "human_status": label.get("human_status", ""),
            "human_notes": label.get("human_notes", ""),
        })
        joined.append(row)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined:
        groups[f"dataset={row.get('dataset','')}"] .append(row)
        groups[f"operation={row.get('operation','')}"] .append(row)
        groups[f"cohort={row.get('cohort','')}"] .append(row)
    result = {
        "score_file": str(path),
        "model_id": next((row.get("model_id", "") for row in joined), ""),
        "overall": metrics(joined),
        "subgroups": {name: metrics(rows) for name, rows in sorted(groups.items())},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {row["sample_id"]: row for row in read_csv(args.manifest)}
    results = [evaluate(path, labels) for path in args.scores]
    payload = {"manifest": str(args.manifest), "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        overall = result["overall"]
        print(
            f"{result['model_id']}: rows={overall['rows']} "
            f"recall={overall['unusable_recall']} "
            f"false_reject={overall['usable_false_reject_rate']} "
            f"accuracy={overall['strict_accuracy_uncertain_is_wrong']} "
            f"cost=${overall['total_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
