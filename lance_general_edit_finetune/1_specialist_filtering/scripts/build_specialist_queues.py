#!/usr/bin/env python3
"""Route Stage 1.1 survivors to active specialist and pair stages."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from filter_common import is_robot_cg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"
HUMAN_SCOPES = {"person", "clothes", "accessory", "appearance"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty specialist queue: {path}")
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def is_robot_candidate(row: dict[str, str]) -> bool:
    return row.get("dataset") == "reco" and (
        "robot_cg" in row.get("pilot_group", "")
        or is_robot_cg(row.get("instruction", ""))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument("--robot-output", type=Path, default=DEFAULT_RUN / "stage1_2_robot_queue.csv")
    parser.add_argument("--human-output", type=Path, default=DEFAULT_RUN / "stage1_5_human_queue.csv")
    parser.add_argument(
        "--legacy-human-route",
        action="store_true",
        help="Reproduce the retired Stage 1.5 queue; disabled for current/future runs.",
    )
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_RUN / "stage1_6_pair_candidate_queue.csv")
    parser.add_argument(
        "--legacy-pair-route",
        action="store_true",
        help="Reproduce the retired Stage 1.6 queue; disabled after moving pair checks to Stage 2.",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "specialist_queue_summary.json")
    args = parser.parse_args()

    rows = read_csv(args.input)
    survivors = [
        row for row in rows if row["stage1_1_decision"] in {"accepted", "uncertain"}
    ]
    robot = [
        {**row, "route_stage": "stage1_2", "route_reason": "reco_robot_or_cg"}
        for row in survivors
        if is_robot_candidate(row)
    ]
    robot_ids = {row["sample_id"] for row in robot}
    human = (
        [
            {**row, "route_stage": "stage1_5", "route_reason": f"scope_{row['target_scope']}"}
            for row in survivors
            if row["sample_id"] not in robot_ids and row.get("target_scope") in HUMAN_SCOPES
        ]
        if args.legacy_human_route
        else []
    )
    pair = (
        [
            {**row, "route_stage": "stage1_6", "route_reason": "stage1_1_survivor"}
            for row in survivors
        ]
        if args.legacy_pair_route
        else []
    )

    write_csv(args.robot_output, robot)
    if args.legacy_human_route:
        write_csv(args.human_output, human)
    if args.legacy_pair_route:
        write_csv(args.pair_output, pair)
    summary = {
        "input_pairs": len(rows),
        "stage1_1_survivors": len(survivors),
        "stage1_2_robot": len(robot),
        "stage1_5_human": len(human),
        "stage1_5_status": "legacy_reproduction" if args.legacy_human_route else "disabled",
        "stage1_6_pair_candidates": len(pair),
        "stage1_6_status": "legacy_reproduction" if args.legacy_pair_route else "disabled_moved_to_stage2",
        "robot_by_branch": dict(Counter(row["branch"] for row in robot)),
        "human_by_scope": dict(Counter(row["target_scope"] for row in human)),
        "stage1_1_survivors_by_decision": dict(
            Counter(row["stage1_1_decision"] for row in survivors)
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
