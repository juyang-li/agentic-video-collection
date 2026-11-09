#!/usr/bin/env python3
"""Calibrate Qwen confidence gates against historical and prospective reviews."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/vlm_run_v1"
VALID_LABELS = {"usable", "unusable", "uncertain"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def collect_labels(paths: list[Path]) -> tuple[dict[str, str], Counter[str]]:
    labels: dict[str, str] = {}
    sources: Counter[str] = Counter()
    for path in paths:
        for row in read_csv(path):
            # Historical InsAVE labels remain useful diagnostics but the entire
            # branch is structurally excluded from the current 4,018-pair pool.
            if row.get("current_scope") == "0":
                continue
            label = row.get("human_label") or row.get("overall_status", "")
            if label not in VALID_LABELS:
                continue
            existing = labels.get(row["sample_id"])
            if existing and existing != label:
                raise ValueError(f"conflicting human labels for {row['sample_id']}")
            labels[row["sample_id"]] = label
            sources[path.name] += 1
    return labels, sources


def candidate_threshold(
    rows: list[dict[str, str]],
    labels: dict[str, str],
    model_decision: str,
    desired_label: str,
    required_precision: float,
    minimum_examples: int,
) -> tuple[float | None, dict[str, float | int]]:
    values = sorted(
        {float(row["model_confidence"]) for row in rows if row["sample_id"] in labels},
        reverse=True,
    )
    candidates: list[tuple[int, float, float]] = []
    for threshold in values:
        selected = [
            labels[row["sample_id"]]
            for row in rows
            if row["sample_id"] in labels
            and row.get("model_decision") == model_decision
            and float(row["model_confidence"]) >= threshold
            and labels[row["sample_id"]] != "uncertain"
        ]
        if len(selected) < minimum_examples:
            continue
        precision = selected.count(desired_label) / len(selected)
        if precision >= required_precision:
            candidates.append((len(selected), precision, threshold))
    if not candidates:
        return None, {"examples": 0, "precision": 0.0}
    examples, precision, threshold = max(candidates, key=lambda item: (item[0], item[1]))
    return threshold, {"examples": examples, "precision": precision}


def operation_accept_precision(
    rows: list[dict[str, str]], labels: dict[str, str], threshold: float | None
) -> dict[str, dict[str, float | int | bool]]:
    result: dict[str, dict[str, float | int | bool]] = {}
    for operation in sorted({row.get("operation", "") for row in rows}):
        chosen = [
            labels[row["sample_id"]]
            for row in rows
            if row.get("operation") == operation
            and row["sample_id"] in labels
            and labels[row["sample_id"]] != "uncertain"
            and row.get("model_decision") == "accepted"
            and threshold is not None
            and float(row["model_confidence"]) >= threshold
        ]
        precision = chosen.count("usable") / len(chosen) if chosen else 0.0
        result[operation] = {
            "examples": len(chosen),
            "precision": precision,
            "enabled": len(chosen) >= 5 and precision >= 0.90,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--prospective-review", type=Path)
    parser.add_argument("--output", type=Path, default=RUN / "qwen_thresholds.json")
    parser.add_argument("--accept-precision", type=float, default=0.95)
    parser.add_argument("--reject-precision", type=float, default=0.90)
    parser.add_argument("--minimum-examples", type=int, default=10)
    parser.add_argument("--required-prospective", type=int, default=100)
    args = parser.parse_args()

    label_paths = list(args.labels)
    prospective_count = 0
    if args.prospective_review is not None and args.prospective_review.is_file():
        label_paths.append(args.prospective_review)
        prospective_count = sum(
            (row.get("overall_status") in VALID_LABELS) for row in read_csv(args.prospective_review)
        )
    labels, label_sources = collect_labels(label_paths)
    scores: list[dict[str, str]] = []
    for path in args.scores:
        scores.extend(
            row
            for row in read_csv(path)
            if row.get("parse_status") == "pass" and row.get("model_backend") != "mock"
        )
    unique_scores = {row["sample_id"]: row for row in scores}
    scores = list(unique_scores.values())
    accept_threshold, accept_stats = candidate_threshold(
        scores, labels, "accepted", "usable", args.accept_precision, args.minimum_examples
    )
    reject_threshold, reject_stats = candidate_threshold(
        scores, labels, "rejected", "unusable", args.reject_precision, args.minimum_examples
    )
    operations = operation_accept_precision(scores, labels, accept_threshold)
    prospective_ready = prospective_count >= args.required_prospective
    enabled = (
        prospective_ready
        and accept_threshold is not None
        and reject_threshold is not None
        and any(bool(item["enabled"]) for item in operations.values())
    )
    output = {
        "filter_version": "qwen2_5_vl_pair_filter_v1",
        "calibration_status": "enabled" if enabled else "pending_or_insufficient",
        "human_labels": len(labels),
        "matched_scores": sum(row["sample_id"] in labels for row in scores),
        "label_sources": dict(label_sources),
        "prospective_review_count": prospective_count,
        "required_prospective_review_count": args.required_prospective,
        "accept": {"confidence_min": accept_threshold, **accept_stats},
        "reject": {"confidence_min": reject_threshold, **reject_stats},
        "operation_acceptance": operations,
        "minimum_accept_precision": args.accept_precision,
        "minimum_reject_precision": args.reject_precision,
        "minimum_examples": args.minimum_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
