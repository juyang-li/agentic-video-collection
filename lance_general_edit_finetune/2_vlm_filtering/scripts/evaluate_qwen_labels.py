#!/usr/bin/env python3
"""Report raw Qwen judgment behavior on a human-labeled manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/vlm_run_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=RUN / "historical_qwen_scores.csv")
    parser.add_argument("--labels", type=Path, default=RUN / "historical_pair_labels.csv")
    parser.add_argument("--output", type=Path, default=RUN / "historical_qwen_evaluation.json")
    parser.add_argument("--current-scope-only", action="store_true")
    args = parser.parse_args()

    labels = {
        row["sample_id"]: row
        for row in read_csv(args.labels)
        if not args.current_scope_only or row.get("current_scope") != "0"
    }
    matched = [row for row in read_csv(args.scores) if row["sample_id"] in labels]
    confusion: Counter[str] = Counter()
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in matched:
        human = labels[row["sample_id"]]["human_label"]
        model = row.get("model_decision", "parse_error")
        confusion[f"human={human}|model={model}"] += 1
        by_operation[row.get("operation", "")][f"human={human}|model={model}"] += 1
    decided_accepted = [
        labels[row["sample_id"]]["human_label"]
        for row in matched
        if row.get("model_decision") == "accepted"
        and labels[row["sample_id"]]["human_label"] != "uncertain"
    ]
    decided_rejected = [
        labels[row["sample_id"]]["human_label"]
        for row in matched
        if row.get("model_decision") == "rejected"
        and labels[row["sample_id"]]["human_label"] != "uncertain"
    ]
    report = {
        "matched": len(matched),
        "parse_errors": sum(row.get("parse_status") != "pass" for row in matched),
        "raw_decisions": dict(Counter(row.get("model_decision", "") for row in matched)),
        "confidence_values": dict(Counter(row.get("model_confidence", "") for row in matched)),
        "confusion": dict(confusion),
        "accepted_precision_excluding_human_uncertain": (
            decided_accepted.count("usable") / len(decided_accepted) if decided_accepted else 0.0
        ),
        "rejected_precision_excluding_human_uncertain": (
            decided_rejected.count("unusable") / len(decided_rejected) if decided_rejected else 0.0
        ),
        "by_operation": {key: dict(value) for key, value in by_operation.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
