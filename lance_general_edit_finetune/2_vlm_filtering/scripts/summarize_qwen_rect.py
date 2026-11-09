#!/usr/bin/env python3
"""Summarize rectangular Qwen scores and, when available, human-label errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/vlm_run_v4"
CHECK_FIELDS = ["instruction_following", "visible_artifact", "content_preservation"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def score_counts(rows: list[dict[str, str]]) -> dict[str, object]:
    by_branch: dict[str, Counter[str]] = defaultdict(Counter)
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_branch[row.get("branch", "")][row.get("model_decision", "")] += 1
        by_operation[row.get("operation", "")][row.get("model_decision", "")] += 1
    completed = [row for row in rows if row.get("parse_status") == "pass"]
    return {
        "rows": len(rows),
        "parse_errors": len(rows) - len(completed),
        "decisions": dict(Counter(row.get("model_decision", "") for row in rows)),
        "checks": {
            field: dict(Counter(row.get(field, "") for row in completed)) for field in CHECK_FIELDS
        },
        "by_branch": {key: dict(value) for key, value in sorted(by_branch.items())},
        "by_operation": {key: dict(value) for key, value in sorted(by_operation.items())},
        "mean_input_tokens": mean(int(row.get("input_tokens") or 0) for row in rows) if rows else 0,
        "mean_elapsed_seconds": mean(float(row.get("elapsed_seconds") or 0) for row in rows)
        if rows
        else 0,
        "peak_gpu_memory_mb": max(
            (float(row.get("peak_gpu_memory_mb") or 0) for row in rows), default=0
        ),
        "mean_roi_reliability": mean(
            float(row.get("roi_reliability_mean") or 0) for row in rows
        )
        if rows
        else 0,
        "pairs_with_any_roi_fallback": sum(
            int(row.get("roi_fallback_count") or 0) > 0 for row in rows
        ),
    }


def label_metrics(
    scores: list[dict[str, str]], labels: list[dict[str, str]], current_scope_only: bool
) -> dict[str, object]:
    label_map = {
        row["sample_id"]: row
        for row in labels
        if not current_scope_only or row.get("current_scope") != "0"
    }
    matched = [row for row in scores if row.get("sample_id") in label_map]
    confusion: Counter[str] = Counter()
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in matched:
        human = label_map[row["sample_id"]].get("human_label", "")
        model = row.get("model_decision", "parse_error")
        key = f"human={human}|model={model}"
        confusion[key] += 1
        by_operation[row.get("operation", "")][key] += 1

    usable = [row for row in matched if label_map[row["sample_id"]].get("human_label") == "usable"]
    unusable = [
        row for row in matched if label_map[row["sample_id"]].get("human_label") == "unusable"
    ]
    accepted_decided = [
        row
        for row in matched
        if row.get("model_decision") == "accepted"
        and label_map[row["sample_id"]].get("human_label") != "uncertain"
    ]
    rejected_decided = [
        row
        for row in matched
        if row.get("model_decision") == "rejected"
        and label_map[row["sample_id"]].get("human_label") != "uncertain"
    ]
    return {
        "matched": len(matched),
        "human_counts": dict(
            Counter(label_map[row["sample_id"]].get("human_label", "") for row in matched)
        ),
        "confusion": dict(confusion),
        "accepted_precision": (
            sum(label_map[row["sample_id"]].get("human_label") == "usable" for row in accepted_decided)
            / len(accepted_decided)
            if accepted_decided
            else 0
        ),
        "rejected_precision": (
            sum(
                label_map[row["sample_id"]].get("human_label") == "unusable"
                for row in rejected_decided
            )
            / len(rejected_decided)
            if rejected_decided
            else 0
        ),
        "usable_false_reject_rate": (
            sum(row.get("model_decision") == "rejected" for row in usable) / len(usable)
            if usable
            else 0
        ),
        "unusable_false_accept_rate": (
            sum(row.get("model_decision") == "accepted" for row in unusable) / len(unusable)
            if unusable
            else 0
        ),
        "usable_accept_rate": (
            sum(row.get("model_decision") == "accepted" for row in usable) / len(usable)
            if usable
            else 0
        ),
        "unusable_reject_rate": (
            sum(row.get("model_decision") == "rejected" for row in unusable) / len(unusable)
            if unusable
            else 0
        ),
        "by_operation": {key: dict(value) for key, value in sorted(by_operation.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=RUN / "qwen_rect_pair_scores.csv")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--current-scope-only", action="store_true")
    parser.add_argument("--output", type=Path, default=RUN / "qwen_rect_report.json")
    args = parser.parse_args()

    scores = read_csv(args.scores)
    report: dict[str, object] = {"scores": score_counts(scores)}
    if args.labels:
        labels = read_csv(args.labels)
        report["human_evaluation"] = label_metrics(scores, labels, args.current_scope_only)
        if args.baseline_scores:
            report["baseline_human_evaluation"] = label_metrics(
                read_csv(args.baseline_scores), labels, args.current_scope_only
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
