#!/usr/bin/env python3
"""Evaluate Qwen3 pair decisions against human labels and freeze hard gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen3_openrouter_v1"
DEFAULT_SCORES = RUN / "calibration_holdout_scores.csv"
DEFAULT_LABELS = RUN / "calibration_holdout.csv"
DEFAULT_REPORT = RUN / "calibration_holdout_evaluation.json"
DEFAULT_GATES = RUN / "qwen3_calibration_gates.json"
OPERATIONS = ("add", "remove", "replace")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(
    scores: list[dict[str, str]],
    labels: list[dict[str, str]],
    *,
    accept_precision_min: float,
    reject_precision_min: float,
    operation_accept_precision_min: float,
    minimum_operation_accepts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_by_id = {row["sample_id"]: row for row in labels}
    score_ids = [row.get("sample_id", "") for row in scores]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(score_ids).items() if sample_id and count > 1
    )
    missing_ids = sorted(set(labels_by_id) - set(score_ids))
    unexpected_ids = sorted(set(score_ids) - set(labels_by_id))
    matched = [row for row in scores if row.get("sample_id") in labels_by_id]
    parse_errors = [row for row in matched if row.get("parse_status") != "pass"]
    judge_versions = sorted({row.get("judge_version", "") for row in matched if row.get("judge_version")})
    prompt_hashes = sorted({row.get("prompt_sha256", "") for row in matched if row.get("prompt_sha256")})
    model_ids = sorted({row.get("model_id", "") for row in matched if row.get("model_id")})
    backends = sorted({row.get("model_backend", "") for row in matched if row.get("model_backend")})

    confusion: Counter[str] = Counter()
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in matched:
        human = labels_by_id[row["sample_id"]].get("human_label", "")
        model = row.get("model_decision", "parse_error")
        key = f"human={human}|model={model}"
        confusion[key] += 1
        by_operation[row.get("operation", "")][key] += 1

    decided = [
        row
        for row in matched
        if labels_by_id[row["sample_id"]].get("human_label") in {"usable", "unusable"}
        and row.get("parse_status") == "pass"
    ]
    accepted = [row for row in decided if row.get("model_decision") == "accepted"]
    rejected = [row for row in decided if row.get("model_decision") == "rejected"]
    accepted_correct = sum(
        labels_by_id[row["sample_id"]].get("human_label") == "usable" for row in accepted
    )
    rejected_correct = sum(
        labels_by_id[row["sample_id"]].get("human_label") == "unusable" for row in rejected
    )
    accepted_precision = ratio(accepted_correct, len(accepted))
    rejected_precision = ratio(rejected_correct, len(rejected))

    operation_metrics: dict[str, Any] = {}
    operation_gates: dict[str, Any] = {}
    for operation in OPERATIONS:
        operation_accepted = [
            row
            for row in accepted
            if row.get("operation") == operation
        ]
        operation_correct = sum(
            labels_by_id[row["sample_id"]].get("human_label") == "usable"
            for row in operation_accepted
        )
        precision = ratio(operation_correct, len(operation_accepted))
        enabled = (
            accepted_precision is not None
            and accepted_precision >= accept_precision_min
            and len(operation_accepted) >= minimum_operation_accepts
            and precision is not None
            and precision >= operation_accept_precision_min
        )
        operation_metrics[operation] = {
            "accepted_decisions": len(operation_accepted),
            "correct_accepted_decisions": operation_correct,
            "accepted_precision": precision,
        }
        operation_gates[operation] = {
            "enabled": enabled,
            "precision_min": operation_accept_precision_min,
            "minimum_accepted_decisions": minimum_operation_accepts,
            **operation_metrics[operation],
        }

    complete = (
        not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and not parse_errors
        and len(judge_versions) == 1
        and len(prompt_hashes) == 1
        and len(model_ids) == 1
        and backends == ["openrouter"]
    )
    accept_gate = (
        complete
        and accepted_precision is not None
        and accepted_precision >= accept_precision_min
        and any(value["enabled"] for value in operation_gates.values())
    )
    reject_gate = (
        complete
        and rejected_precision is not None
        and rejected_precision >= reject_precision_min
    )
    enabled_operations = [
        operation for operation, value in operation_gates.items() if value["enabled"]
    ]
    # Full expansion is intentionally stricter than enabling an individual route:
    # the run must be complete and both global hard decisions must be calibrated.
    calibration_status = "enabled" if accept_gate and reject_gate else "pending_or_failed"

    report = {
        "calibration_set_rows": len(labels),
        "score_rows": len(scores),
        "matched_rows": len(matched),
        "parse_errors": len(parse_errors),
        "duplicate_sample_ids": duplicate_ids,
        "missing_sample_ids": missing_ids,
        "unexpected_sample_ids": unexpected_ids,
        "human_counts": dict(
            Counter(labels_by_id[row["sample_id"]].get("human_label", "") for row in matched)
        ),
        "model_decisions": dict(Counter(row.get("model_decision", "") for row in matched)),
        "confusion": dict(confusion),
        "accepted_precision": accepted_precision,
        "accepted_precision_denominator": len(accepted),
        "rejected_precision": rejected_precision,
        "rejected_precision_denominator": len(rejected),
        "usable_false_reject_rate": ratio(
            sum(
                row.get("model_decision") == "rejected"
                for row in decided
                if labels_by_id[row["sample_id"]].get("human_label") == "usable"
            ),
            sum(
                labels_by_id[row["sample_id"]].get("human_label") == "usable"
                for row in decided
            ),
        ),
        "unusable_false_accept_rate": ratio(
            sum(
                row.get("model_decision") == "accepted"
                for row in decided
                if labels_by_id[row["sample_id"]].get("human_label") == "unusable"
            ),
            sum(
                labels_by_id[row["sample_id"]].get("human_label") == "unusable"
                for row in decided
            ),
        ),
        "by_operation_confusion": {
            key: dict(value) for key, value in sorted(by_operation.items())
        },
        "by_operation_acceptance": operation_metrics,
        "calibration_status": calibration_status,
        "judge_versions": judge_versions,
        "prompt_hashes": prompt_hashes,
        "model_ids": model_ids,
        "model_backends": backends,
    }
    gates = {
        "filter_version": "general_edit_stage2_qwen3_openrouter_v1",
        "calibration_status": calibration_status,
        "calibration_complete": complete,
        "judge_version": judge_versions[0] if len(judge_versions) == 1 else "",
        "prompt_sha256": prompt_hashes[0] if len(prompt_hashes) == 1 else "",
        "model_id": model_ids[0] if len(model_ids) == 1 else "",
        "model_backend": backends[0] if len(backends) == 1 else "",
        "accept": {
            "enabled": accept_gate,
            "precision_min": accept_precision_min,
            "observed_precision": accepted_precision,
            "decisions": len(accepted),
        },
        "reject": {
            "enabled": reject_gate,
            "precision_min": reject_precision_min,
            "observed_precision": rejected_precision,
            "decisions": len(rejected),
        },
        "operation_acceptance": operation_gates,
        "enabled_operations": enabled_operations,
        "source_report": "",
    }
    return report, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--accept-precision-min", type=float, default=0.95)
    parser.add_argument("--reject-precision-min", type=float, default=0.90)
    parser.add_argument("--operation-accept-precision-min", type=float, default=0.90)
    parser.add_argument("--minimum-operation-accepts", type=int, default=3)
    args = parser.parse_args()

    report, gates = evaluate(
        read_csv(args.scores),
        read_csv(args.labels),
        accept_precision_min=args.accept_precision_min,
        reject_precision_min=args.reject_precision_min,
        operation_accept_precision_min=args.operation_accept_precision_min,
        minimum_operation_accepts=args.minimum_operation_accepts,
    )
    gates["source_report"] = str(args.output)
    for path, value in ((args.output, report), (args.gates, gates)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
