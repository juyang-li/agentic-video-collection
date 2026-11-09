#!/usr/bin/env python3
"""Merge prior explicit labels with completed explicit systematic reviews."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import build_systematic_binary_benchmark as systematic
import finalize_systematic_binary_benchmark as finalize


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
DEFAULT_DEVELOPMENT = RUN / "development_labels.csv"
DEFAULT_BENCHMARK = RUN / "benchmark_manifest_300.csv"
DEFAULT_REVIEW = RUN / "review_results/stage2_systematic_binary_benchmark_v1_review.csv"
DEFAULT_BACKUP = RUN / "review_results/stage2_systematic_binary_benchmark_v1_backup.json"
DEFAULT_STAGE1 = PROJECT / "1_specialist_filtering/artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_OUTPUT = RUN / "labels/explicit_binary_dataset.csv"
DEFAULT_MODEL_INPUT = RUN / "labels/explicit_binary_v12_input.csv"
DEFAULT_SUMMARY = RUN / "labels/explicit_binary_dataset_summary.json"
ALLOWED = {"accepted", "rejected"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def nested_counts(
    rows: list[dict[str, str]], group: str, label: str = "human_decision"
) -> dict[str, dict[str, int]]:
    return {
        key: dict(Counter(row[label] for row in rows if row[group] == key))
        for key in sorted({row[group] for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-input", type=Path, default=DEFAULT_MODEL_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    development, _ = read_csv(args.development)
    benchmark, benchmark_fields = read_csv(args.benchmark)
    reviews, _ = read_csv(args.review)
    stage1_rows, stage1_fields = read_csv(args.stage1)
    if len(benchmark) != 300 or len(reviews) != 300:
        raise ValueError("systematic benchmark/review must each contain 300 rows")
    benchmark_by_id = {row["sample_id"]: row for row in benchmark}
    review_by_id = {row["sample_id"]: row for row in reviews}
    if len(benchmark_by_id) != 300 or set(review_by_id) != set(benchmark_by_id):
        raise ValueError("systematic benchmark/review IDs are not identical and unique")

    backup = json.loads(args.backup.read_text(encoding="utf-8"))
    if backup.get("version") != systematic.VERSION:
        raise ValueError("review backup survey version does not match benchmark")
    for sample_id, value in backup.get("reviews", {}).items():
        if sample_id not in review_by_id:
            raise ValueError(f"backup contains unknown sample ID: {sample_id}")
        csv_decision = review_by_id[sample_id].get("human_decision", "")
        if value.get("decision", "") != csv_decision:
            raise ValueError(f"CSV/JSON decision mismatch: {sample_id}")

    current: list[dict[str, str]] = []
    for row in benchmark:
        review = review_by_id[row["sample_id"]]
        decision = review.get("human_decision", "").strip().lower()
        if decision not in ALLOWED:
            continue
        normalized = finalize.normalized_test_row(
            row, decision, review.get("notes", "")
        )
        normalized.update({
            "annotation_partition": "test_v1_partial_completed",
            "benchmark_usage": "blind_test_partial_completed",
        })
        current.append(normalized)

    for row in development:
        if row.get("human_decision") not in ALLOWED:
            raise ValueError("development contains a non-binary decision")
    development_ids = {row["sample_id"] for row in development}
    current_ids = {row["sample_id"] for row in current}
    if development_ids & current_ids:
        raise ValueError("development and systematic explicit labels overlap")
    combined = [*development, *current]
    if len({row["sample_id"] for row in combined}) != len(combined):
        raise ValueError("combined labels contain duplicate sample IDs")
    write_csv(args.output, combined, finalize.REGISTRY_FIELDS)

    stage1 = {row["sample_id"]: row for row in stage1_rows}
    model_rows: list[dict[str, str]] = []
    for label in combined:
        metadata = stage1.get(label["sample_id"])
        if metadata is None:
            metadata = {
                "sample_id": label["sample_id"],
                "dataset": label["dataset"],
                "branch": label["branch"],
                "operation": label["training_operation"],
                "instruction": label["training_instruction"],
                "source_local": label["training_source_local"],
                "target_local": label["training_target_local"],
                "source_provenance": label["source_provenance"],
                "target_provenance": label["target_provenance"],
                "provenance_basis": label.get("provenance_basis", ""),
            }
        row = systematic.construction_aware_row(metadata)
        row.update({
            "human_decision": label["human_decision"],
            "human_notes": label.get("human_notes", ""),
            "annotation_source": label.get("annotation_source", ""),
            "annotation_partition": label.get("annotation_partition", ""),
            "benchmark_usage": label.get("benchmark_usage", ""),
            "audit_operation_expected": label["audit_operation"],
            "audit_relation_expected": label["audit_relation"],
        })
        model_rows.append(row)
    model_fields = list(dict.fromkeys([
        *stage1_fields,
        *benchmark_fields,
        "human_decision",
        "human_notes",
        "annotation_source",
        "annotation_partition",
        "benchmark_usage",
        "audit_operation_expected",
        "audit_relation_expected",
    ]))
    write_csv(args.model_input, model_rows, model_fields)

    review_decisions = Counter(
        row.get("human_decision", "").strip().lower() for row in reviews
    )
    summary = {
        "dataset_version": "explicit_binary_dataset_20260904_partial_v1",
        "rows": len(combined),
        "development_rows": len(development),
        "new_systematic_explicit_rows": len(current),
        "new_systematic_review_status": {
            "accepted": review_decisions["accepted"],
            "rejected": review_decisions["rejected"],
            "uncertain_excluded": review_decisions["uncertain"],
            "unfinished_excluded": review_decisions[""],
        },
        "labels": dict(Counter(row["human_decision"] for row in combined)),
        "partitions": dict(Counter(row["annotation_partition"] for row in combined)),
        "datasets": dict(Counter(row["dataset"] for row in combined)),
        "branches": dict(Counter(row["branch"] for row in combined)),
        "audit_operations": dict(Counter(row["audit_operation"] for row in combined)),
        "labels_by_dataset": nested_counts(combined, "dataset"),
        "labels_by_branch": nested_counts(combined, "branch"),
        "labels_by_audit_operation": nested_counts(combined, "audit_operation"),
        "audit_relations": dict(Counter(row["audit_relation"] for row in combined)),
        "provenance": dict(Counter(
            f"{row['source_provenance']}->{row['target_provenance']}"
            for row in combined
        )),
        "label_policy": "explicit accepted/rejected only; uncertain and blank excluded",
        "evaluation_warning": (
            "development was used during prior filter development; only the completed "
            "systematic subset is blind-test evidence, and that subset is incomplete"
        ),
        "sample_ids": [row["sample_id"] for row in combined],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
