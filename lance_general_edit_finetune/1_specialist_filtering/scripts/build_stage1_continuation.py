#!/usr/bin/env python3
"""Build the immutable Stage 1 continuation input after structural exclusions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_handheld_remove_review_sample import is_handheld_object_remove
from build_specialist_queues import is_robot_candidate
from filter_common import metadata_rejection_reasons


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/filter_run_v1/stage1_1_pair_decisions.csv"
DEFAULT_RUN = ROOT / "artifacts/filter_run_v2"
VERSION = "general_edit_stage1_continuation_v3"
SURVIVOR_DECISIONS = {"accepted", "uncertain"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def structural_reasons(row: dict[str, str]) -> list[str]:
    reasons = metadata_rejection_reasons(
        row.get("dataset", ""), row.get("operation", ""), row.get("instruction", "")
    )
    if row.get("branch") == "insave_remove":
        reasons.append("excluded_insave_person_remove_branch")
    if is_handheld_object_remove(row):
        reasons.append("excluded_handheld_object_remove")
    return reasons


def enrich(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    result.update(
        {
            "continuation_version": VERSION,
            "source_video_id": row.get("source_video_id") or f"{row['sample_id']}:source",
            "target_video_id": row.get("target_video_id") or f"{row['sample_id']}:target",
            "route_stage1_2": "1" if is_robot_candidate(row) else "0",
            "route_stage2": "1",
        }
    )
    return result


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    survivors = [row for row in rows if row.get("stage1_1_decision") in SURVIVOR_DECISIONS]
    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in survivors:
        reasons = structural_reasons(row)
        item = enrich(row)
        item["structural_decision"] = "rejected" if reasons else "accepted"
        item["structural_reasons"] = "|".join(reasons)
        (excluded if reasons else included).append(item)
    return included, excluded


def validate(included: list[dict[str, str]], excluded: list[dict[str, str]]) -> None:
    ids = [row["sample_id"] for row in included]
    if len(ids) != len(set(ids)):
        raise ValueError("continuation input contains duplicate sample_id")
    overlap = set(ids) & {row["sample_id"] for row in excluded}
    if overlap:
        raise ValueError(f"included and excluded sets overlap: {sorted(overlap)[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--expect-input",
        type=int,
        default=4003,
        help="Set negative to disable the frozen-pilot count assertion.",
    )
    args = parser.parse_args()

    source_rows = read_csv(args.input)
    included, excluded = build(source_rows)
    validate(included, excluded)
    if args.expect_input >= 0 and len(included) != args.expect_input:
        raise ValueError(f"expected {args.expect_input} continuation pairs, got {len(included)}")

    fields = list(
        dict.fromkeys(
            list(source_rows[0])
            + [
                "continuation_version",
                "source_video_id",
                "target_video_id",
                "structural_decision",
                "structural_reasons",
                "route_stage1_2",
                "route_stage2",
            ]
        )
    )
    write_csv(args.run_dir / "stage1_next_input.csv", included, fields)
    write_csv(args.run_dir / "stage1_structural_exclusions.csv", excluded, fields)
    robot = [row for row in included if row["route_stage1_2"] == "1"]
    write_csv(args.run_dir / "stage1_2_robot_queue.csv", robot, fields)

    summary = {
        "continuation_version": VERSION,
        "source": str(args.input),
        "source_rows": len(source_rows),
        "source_stage1_1_survivors": sum(
            row.get("stage1_1_decision") in SURVIVOR_DECISIONS for row in source_rows
        ),
        "continuation_input": len(included),
        "structural_exclusions": len(excluded),
        "stage1_2_robot": len(robot),
        "by_original_stage1_1_decision": dict(
            Counter(row["stage1_1_decision"] for row in included)
        ),
        "by_branch": dict(Counter(row["branch"] for row in included)),
        "exclusion_reasons": dict(
            Counter(
                reason
                for row in excluded
                for reason in row["structural_reasons"].split("|")
                if reason
            )
        ),
    }
    summary_path = args.run_dir / "stage1_continuation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
