#!/usr/bin/env python3
"""Validate the 300-pair review and freeze binary test/combined labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
DEFAULT_BENCHMARK = RUN / "benchmark_manifest_300.csv"
DEFAULT_DEVELOPMENT = RUN / "development_labels.csv"
DEFAULT_OUTPUT = RUN / "labels"
ALLOWED = {"accepted", "rejected", "uncertain"}


REGISTRY_FIELDS = [
    "sample_id",
    "dataset",
    "branch",
    "training_operation",
    "audit_operation",
    "training_instruction",
    "audit_instruction",
    "source_provenance",
    "target_provenance",
    "provenance_basis",
    "generated_role",
    "audit_relation",
    "reference_training_role",
    "generated_training_role",
    "audit_left_training_role",
    "audit_right_training_role",
    "artifact_checked_roles",
    "instruction_following_direction",
    "content_preservation_direction",
    "training_source_local",
    "training_target_local",
    "reference_local",
    "generated_local",
    "audit_left_local",
    "audit_right_local",
    "human_decision",
    "raw_human_status",
    "human_notes",
    "annotation_source",
    "annotation_partition",
    "benchmark_usage",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized_test_row(
    row: dict[str, str], decision: str, notes: str
) -> dict[str, str]:
    reference_role = row["reference_training_role"]
    generated_role = row["generated_training_role"]
    left_role = row["audit_left_training_role"]
    right_role = row["audit_right_training_role"]
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "branch": row["branch"],
        "training_operation": row["operation"],
        "audit_operation": row["audit_operation_expected"],
        "training_instruction": row["instruction"],
        "audit_instruction": row["audit_instruction"],
        "source_provenance": row["source_provenance"],
        "target_provenance": row["target_provenance"],
        "provenance_basis": row.get("provenance_basis", ""),
        "generated_role": row["generated_role"],
        "audit_relation": row["audit_relation"],
        "reference_training_role": reference_role,
        "generated_training_role": generated_role,
        "audit_left_training_role": left_role,
        "audit_right_training_role": right_role,
        "artifact_checked_roles": row["artifact_checked_roles"],
        "instruction_following_direction": row["instruction_following_direction"],
        "content_preservation_direction": row["content_preservation_direction"],
        "training_source_local": row["source_local"],
        "training_target_local": row["target_local"],
        "reference_local": row[f"{reference_role}_local"] if reference_role else "",
        "generated_local": row[f"{generated_role}_local"] if generated_role else "",
        "audit_left_local": row[f"{left_role}_local"],
        "audit_right_local": row[f"{right_role}_local"],
        "human_decision": decision,
        "raw_human_status": decision,
        "human_notes": notes,
        "annotation_source": "systematic_review_300",
        "annotation_partition": "test_v1",
        "benchmark_usage": "blind_test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = {
        "test": args.output_dir / "test_v1_binary_labels.csv",
        "uncertain": args.output_dir / "test_v1_uncertain.csv",
        "registry": args.output_dir / "binary_label_registry.csv",
        "summary": args.output_dir / "label_summary.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"outputs exist; pass --overwrite: {existing}")

    benchmark = read_csv(args.benchmark)
    if len(benchmark) != 300 or len({row["sample_id"] for row in benchmark}) != 300:
        raise ValueError("frozen benchmark must contain exactly 300 unique rows")
    reviews = read_csv(args.human_review)
    if len(reviews) != 300:
        raise ValueError(f"review must contain exactly 300 rows, got {len(reviews)}")
    review_by_id: dict[str, dict[str, str]] = {}
    for row in reviews:
        sample_id = row.get("sample_id", "")
        decision = row.get("human_decision", row.get("decision", "")).strip().lower()
        if not sample_id or sample_id in review_by_id:
            raise ValueError(f"blank or duplicate review sample_id: {sample_id!r}")
        if decision not in ALLOWED:
            raise ValueError(
                f"invalid or unfinished decision for {sample_id}: {decision!r}"
            )
        row = dict(row)
        row["_decision"] = decision
        review_by_id[sample_id] = row
    benchmark_ids = {row["sample_id"] for row in benchmark}
    if set(review_by_id) != benchmark_ids:
        raise ValueError(
            f"review ID mismatch: missing={len(benchmark_ids-set(review_by_id))}, "
            f"extra={len(set(review_by_id)-benchmark_ids)}"
        )

    test_rows: list[dict[str, str]] = []
    uncertain_rows: list[dict[str, str]] = []
    for row in benchmark:
        review = review_by_id[row["sample_id"]]
        normalized = normalized_test_row(
            row, review["_decision"], review.get("notes", "")
        )
        if normalized["human_decision"] == "uncertain":
            normalized["benchmark_usage"] = "excluded_uncertain"
            uncertain_rows.append(normalized)
        else:
            test_rows.append(normalized)

    development = read_csv(args.development)
    invalid_development = [
        row["sample_id"]
        for row in development
        if row.get("human_decision") not in {"accepted", "rejected"}
    ]
    if invalid_development:
        raise ValueError("development labels contain non-binary decisions")
    overlap = {row["sample_id"] for row in development} & {
        row["sample_id"] for row in test_rows + uncertain_rows
    }
    if overlap:
        raise ValueError(f"development/test overlap: {sorted(overlap)[:5]}")

    write_csv(outputs["test"], test_rows)
    write_csv(outputs["uncertain"], uncertain_rows)
    write_csv(outputs["registry"], [*development, *test_rows])
    summary = {
        "raw_review_rows": len(reviews),
        "raw_review_decisions": dict(Counter(
            row["_decision"] for row in review_by_id.values()
        )),
        "test_v1_scored_rows": len(test_rows),
        "test_v1_excluded_uncertain": len(uncertain_rows),
        "test_v1_labels": dict(Counter(row["human_decision"] for row in test_rows)),
        "test_v1_operations": dict(Counter(row["audit_operation"] for row in test_rows)),
        "development_rows": len(development),
        "combined_registry_rows": len(development) + len(test_rows),
        "scoring_policy": "score accepted/rejected only; exclude human uncertain",
    }
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
