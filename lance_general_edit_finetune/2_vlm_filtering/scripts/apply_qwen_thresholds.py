#!/usr/bin/env python3
"""Apply the frozen Qwen calibration gates without trusting raw zero-shot labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/vlm_run_v1"
FIELDS = [
    "filter_version",
    "sample_id",
    "qwen_decision",
    "qwen_reason",
    "model_decision",
    "model_confidence",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def decide(row: dict[str, str], thresholds: dict) -> tuple[str, str]:
    if row.get("model_backend") == "mock":
        raise ValueError("mock Qwen scores cannot produce formal decisions")
    if thresholds.get("calibration_status") != "enabled":
        return "uncertain", "qwen_not_calibrated"
    if row.get("parse_status") != "pass":
        return "uncertain", "qwen_parse_error"
    confidence = float(row.get("model_confidence") or 0)
    raw = row.get("model_decision")
    reject_min = thresholds["reject"].get("confidence_min")
    if raw == "rejected" and reject_min is not None and confidence >= reject_min:
        return "rejected", "qwen_edit_failure"
    accept_min = thresholds["accept"].get("confidence_min")
    operation_enabled = thresholds.get("operation_acceptance", {}).get(
        row.get("operation", ""), {}
    ).get("enabled", False)
    if raw == "accepted" and operation_enabled and accept_min is not None and confidence >= accept_min:
        return "accepted", "qwen_edit_success"
    return "uncertain", "qwen_boundary_or_unsupported"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=RUN / "qwen_pair_scores.csv")
    parser.add_argument("--thresholds", type=Path, default=RUN / "qwen_thresholds.json")
    parser.add_argument("--output", type=Path, default=RUN / "qwen_decisions.csv")
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    output = []
    for row in read_csv(args.scores):
        decision, reason = decide(row, thresholds)
        output.append(
            {
                "filter_version": thresholds["filter_version"],
                "sample_id": row["sample_id"],
                "qwen_decision": decision,
                "qwen_reason": reason,
                "model_decision": row.get("model_decision", ""),
                "model_confidence": row.get("model_confidence", ""),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)
    print(json.dumps(dict(Counter(row["qwen_decision"] for row in output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
