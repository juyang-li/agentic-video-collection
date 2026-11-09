#!/usr/bin/env python3
"""Collect prior explicit human decisions into the Stage-2 development split.

Only labels that the reviewer explicitly saved as usable/unusable are mapped to
accepted/rejected.  Historical uncertain rows are excluded even when their notes
sound positive or negative.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import run_vefx_reward as routed
import build_systematic_binary_benchmark as systematic


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE1 = PROJECT / "1_specialist_filtering"
DEFAULT_STAGE1 = STAGE1 / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_HISTORICAL = ROOT / "artifacts/vlm_run_v1/historical_pair_labels.csv"
DEFAULT_ARTIFACT = (
    ROOT
    / "artifacts/qwen72b_artifact_risk_v1/"
    "artifact_risk_30_qwen72b_audit_v3_review.csv"
)
DEFAULT_HOLDOUT = (
    ROOT
    / "artifacts/v12_holdout40_v1/review_results/"
    "qwen72b_v12_unseen_holdout40_v1_review.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/systematic_binary_benchmark_v1/development_labels.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "artifacts/systematic_binary_benchmark_v1/development_labels_summary.json"
)


FIELDS = [
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
    "annotation_source",
    "annotation_partition",
    "benchmark_usage",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def canonical_row(
    metadata: dict[str, str], raw_status: str, annotation_source: str
) -> dict[str, str]:
    decision = {"usable": "accepted", "unusable": "rejected"}[raw_status]
    metadata = systematic.construction_aware_row(metadata)
    audit = routed.route(metadata)
    reference_role = audit["reference_training_role"]
    generated_role = audit["generated_training_role"]
    left_role = audit["audit_left_training_role"]
    right_role = audit["audit_right_training_role"]
    audit_operation = (
        {"add": "remove", "remove": "add"}.get(metadata["operation"], metadata["operation"])
        if audit["audit_relation"] == "inverse"
        else metadata["operation"]
    )
    return {
        "sample_id": metadata["sample_id"],
        "dataset": metadata["dataset"],
        "branch": metadata["branch"],
        "training_operation": metadata["operation"],
        "audit_operation": audit_operation,
        "training_instruction": metadata["instruction"],
        "audit_instruction": audit["audit_instruction"],
        "source_provenance": metadata["source_provenance"],
        "target_provenance": metadata["target_provenance"],
        "provenance_basis": metadata.get("provenance_basis", ""),
        "generated_role": audit["generated_side"].lower(),
        "audit_relation": audit["audit_relation"],
        "reference_training_role": reference_role,
        "generated_training_role": generated_role,
        "audit_left_training_role": left_role,
        "audit_right_training_role": right_role,
        "artifact_checked_roles": audit["artifact_checked_roles"],
        "instruction_following_direction": audit["instruction_following_direction"],
        "content_preservation_direction": audit["content_preservation_direction"],
        "training_source_local": metadata["source_local"],
        "training_target_local": metadata["target_local"],
        "reference_local": metadata[f"{reference_role}_local"] if reference_role else "",
        "generated_local": metadata[f"{generated_role}_local"] if generated_role else "",
        "audit_left_local": metadata[f"{left_role}_local"],
        "audit_right_local": metadata[f"{right_role}_local"],
        "human_decision": decision,
        "raw_human_status": raw_status,
        "annotation_source": annotation_source,
        "annotation_partition": "development",
        "benchmark_usage": "development_only_not_blind_test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--artifact-review", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--holdout-review", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.overwrite:
        for path in (args.output, args.summary):
            if path.exists():
                raise FileExistsError(f"output exists; pass --overwrite: {path}")

    stage1 = {row["sample_id"]: row for row in read_csv(args.stage1)}
    records: list[dict[str, str]] = []

    # This file derives usable/unusable only from explicit pass/hard-fail fields;
    # its uncertain rows have never been recoded from free-text notes.
    for row in read_csv(args.historical):
        if row.get("current_scope") != "1" or row.get("human_label") not in {
            "usable",
            "unusable",
        }:
            continue
        records.append(canonical_row(row, row["human_label"], "historical_pair_review"))

    # The risk set contains upstream-excluded garment rows.  Keeping only IDs in
    # the current Stage-1 manifest makes this a valid Stage-2 evaluation source.
    for review in read_csv(args.artifact_review):
        raw = review.get("human_status", "")
        metadata = stage1.get(review["sample_id"])
        if metadata is not None and raw in {"usable", "unusable"}:
            records.append(canonical_row(metadata, raw, "artifact_risk30_raw_review"))

    # Deliberately read the raw holdout review, not postreview_policy_labels.csv:
    # the latter converted some uncertain notes to keep/delete.
    for review in read_csv(args.holdout_review):
        raw = review.get("overall_status", "")
        metadata = stage1.get(review["sample_id"])
        if metadata is not None and raw in {"usable", "unusable"}:
            records.append(canonical_row(metadata, raw, "holdout40_raw_review"))

    by_id: dict[str, dict[str, str]] = {}
    for row in records:
        previous = by_id.get(row["sample_id"])
        if previous and previous["human_decision"] != row["human_decision"]:
            raise ValueError(f"conflicting explicit labels: {row['sample_id']}")
        if previous is None:
            by_id[row["sample_id"]] = row
    records = sorted(by_id.values(), key=lambda row: (row["dataset"], row["branch"], row["sample_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(args.output)

    summary = {
        "rows": len(records),
        "labels": dict(Counter(row["human_decision"] for row in records)),
        "audit_operations": dict(Counter(row["audit_operation"] for row in records)),
        "labels_by_operation": {
            operation: dict(Counter(
                row["human_decision"]
                for row in records
                if row["audit_operation"] == operation
            ))
            for operation in ("remove", "add", "replace")
        },
        "datasets": dict(Counter(row["dataset"] for row in records)),
        "annotation_sources": dict(Counter(row["annotation_source"] for row in records)),
        "uncertain_policy": "excluded regardless of notes",
        "benchmark_usage": "development_only_not_blind_test",
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
