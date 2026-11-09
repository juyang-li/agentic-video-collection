#!/usr/bin/env python3
"""Run Stage 1.0 metadata gates and emit auditable reason codes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from filter_common import (
    classify_target_scope,
    is_handheld_object_remove,
    metadata_rejection_reasons,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/filter_run_v1/pilot_manifest_phase_a.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/filter_run_v1/stage1_0_results.csv"
DEFAULT_SURVIVORS = ROOT / "artifacts/filter_run_v1/stage1_0_survivors.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/filter_run_v1/stage1_0_summary.json"

ALLOWED_SCOPES = {
    "insave_remove": {"person"},
    "goku_remove": {"person", "object", "accessory"},
    "goku_add": {"person", "object", "accessory"},
    "egoedit_replace": {"object"},
    "reco_add": {"person", "object", "clothes", "accessory"},
    "reco_remove": {"person", "object", "clothes", "accessory"},
    "reco_replace": {"person", "object", "clothes", "accessory", "appearance", "background"},
}
PROVENANCE_VALUES = {"original", "edited", "generation_side_unresolved", "unknown"}
QUALITY_ROLES = {"source", "target", "both"}
RESULT_FIELDS = [
    "metadata_decision",
    "metadata_reasons",
    "technical_status",
    "next_stage",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def evaluate(row: dict[str, str]) -> dict[str, str]:
    reasons = metadata_rejection_reasons(
        row.get("dataset", ""), row.get("operation", ""), row.get("instruction", "")
    )
    branch = row.get("branch", "")
    scope = classify_target_scope(row.get("instruction", ""))
    if branch == "insave_remove":
        reasons.append("excluded_insave_person_remove_branch")
    if is_handheld_object_remove(branch, scope, row.get("instruction", "")):
        reasons.append("excluded_handheld_object_remove")
    if branch not in ALLOWED_SCOPES:
        reasons.append("unknown_branch")
    elif scope not in ALLOWED_SCOPES[branch]:
        reasons.append("excluded_task_scope")
    if scope != row.get("target_scope"):
        reasons.append("stale_target_scope")
    if not row.get("source_locator") or not row.get("target_locator"):
        reasons.append("missing_media_locator")
    if row.get("source_provenance") not in PROVENANCE_VALUES:
        reasons.append("invalid_source_provenance")
    if row.get("target_provenance") not in PROVENANCE_VALUES:
        reasons.append("invalid_target_provenance")
    if row.get("quality_checked_roles") not in QUALITY_ROLES:
        reasons.append("invalid_quality_checked_roles")

    reasons = sorted(set(reasons))
    rejected = bool(reasons)
    local_paths = [row.get("source_local", ""), row.get("target_local", "")]
    media_ready = all(
        value and ((Path(value) if Path(value).is_absolute() else ROOT / value).is_file())
        for value in local_paths
    )
    return {
        "metadata_decision": "rejected" if rejected else "accepted",
        "metadata_reasons": "|".join(reasons),
        "technical_status": "ready" if media_ready else "pending_media",
        "next_stage": "stop" if rejected else "stage1_technical",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    input_rows = read_csv(args.input)
    if not input_rows:
        raise ValueError("input manifest is empty")
    results = [{**row, **evaluate(row)} for row in input_rows]
    fieldnames = list(input_rows[0]) + RESULT_FIELDS
    survivors = [row for row in results if row["metadata_decision"] == "accepted"]
    write_csv(args.output, results, fieldnames)
    write_csv(args.survivors, survivors, fieldnames)

    reason_counts = Counter(
        reason
        for row in results
        for reason in row["metadata_reasons"].split("|")
        if reason
    )
    summary = {
        "filter_version": input_rows[0]["filter_version"],
        "input_rows": len(results),
        "accepted": len(survivors),
        "rejected": len(results) - len(survivors),
        "pending_media": sum(row["technical_status"] == "pending_media" for row in survivors),
        "media_ready": sum(row["technical_status"] == "ready" for row in survivors),
        "decision_by_branch": {
            branch: dict(Counter(row["metadata_decision"] for row in results if row["branch"] == branch))
            for branch in sorted({row["branch"] for row in results})
        },
        "reason_counts": dict(reason_counts),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
