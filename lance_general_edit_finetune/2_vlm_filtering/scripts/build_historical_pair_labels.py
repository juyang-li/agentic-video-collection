#!/usr/bin/env python3
"""Export completed historical pair judgments for prompt validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
SPECIALIST_SCRIPTS = SPECIALIST_ROOT / "scripts"
if str(SPECIALIST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SPECIALIST_SCRIPTS))

from build_handheld_remove_review_sample import is_handheld_object_remove
from filter_common import classify_target_scope, metadata_rejection_reasons
DEFAULT_MANIFEST = SPECIALIST_ROOT / "artifacts/survey_manifest_reco_v3.csv"
DEFAULT_REVIEW = SPECIALIST_ROOT / "artifacts/review_results/specialist_error_survey_460_reco_v3_review.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/vlm_run_v1/historical_pair_labels.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reviews = {row["sample_id"]: row for row in read_csv(args.review)}
    output: list[dict[str, str]] = []
    for row in read_csv(args.manifest):
        review = reviews.get(row["sample_id"], {})
        status = review.get("pair_status", "")
        if status not in {"pass", "hard_fail", "uncertain"}:
            continue
        roles = {
            "source": ("source",),
            "target": ("target",),
            "both": ("source", "target"),
        }[row.get("quality_checked_roles", "both")]
        quality_statuses = [review.get(f"{role}_status", "uncertain") for role in roles]
        if status == "hard_fail" or "hard_fail" in quality_statuses:
            human_label = "unusable"
        elif status == "pass" and all(value == "pass" for value in quality_statuses):
            human_label = "usable"
        else:
            human_label = "uncertain"
        output.append(
            {
                **row,
                "human_pair_status": status,
                "human_quality_status": "|".join(quality_statuses),
                "human_label": human_label,
                "pair_errors": review.get("pair_errors", ""),
                "pair_notes": review.get("pair_notes", ""),
                "current_scope": (
                    "0"
                    if row.get("branch") == "insave_remove"
                    or metadata_rejection_reasons(
                        row.get("dataset", ""), row.get("operation", ""), row.get("instruction", "")
                    )
                    or is_handheld_object_remove(
                        {**row, "target_scope": classify_target_scope(row.get("instruction", ""))}
                    )
                    else "1"
                ),
            }
        )
    if not output:
        raise ValueError("no completed historical pair labels")
    fields = list(output[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "rows": len(output),
                "current_scope_rows": sum(row["current_scope"] == "1" for row in output),
                "human_labels": dict(Counter(row["human_label"] for row in output)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
