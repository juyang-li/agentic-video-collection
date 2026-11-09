#!/usr/bin/env python3
"""Turn calibrated Qwen3 scores into formal Stage 2 decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen3_openrouter_v1"
FIELDS = [
    "filter_version",
    "sample_id",
    "qwen_decision",
    "qwen_reason",
    "model_decision",
    "operation",
    "reason_codes_json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def decide(row: dict[str, str], gates: dict) -> tuple[str, str]:
    if row.get("model_backend") == "mock":
        raise ValueError("mock Qwen3 scores cannot produce formal decisions")
    if gates.get("calibration_status") != "enabled":
        return "uncertain", "qwen3_not_calibrated"
    if row.get("parse_status") != "pass":
        return "uncertain", "qwen3_parse_error"
    raw = row.get("model_decision")
    if raw == "rejected" and gates.get("reject", {}).get("enabled"):
        return "rejected", "qwen3_edit_failure"
    operation_enabled = gates.get("operation_acceptance", {}).get(
        row.get("operation", ""), {}
    ).get("enabled", False)
    if raw == "accepted" and gates.get("accept", {}).get("enabled") and operation_enabled:
        return "accepted", "qwen3_edit_success"
    return "uncertain", "qwen3_boundary_or_unsupported_operation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=RUN / "qwen3_pair_scores.csv")
    parser.add_argument("--gates", type=Path, default=RUN / "qwen3_calibration_gates.json")
    parser.add_argument("--output", type=Path, default=RUN / "qwen3_decisions.csv")
    args = parser.parse_args()

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    output = []
    for row in read_csv(args.scores):
        decision, reason = decide(row, gates)
        output.append(
            {
                "filter_version": gates["filter_version"],
                "sample_id": row["sample_id"],
                "qwen_decision": decision,
                "qwen_reason": reason,
                "model_decision": row.get("model_decision", ""),
                "operation": row.get("operation", ""),
                "reason_codes_json": row.get("reason_codes_json", ""),
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
