#!/usr/bin/env python3
"""Build a deterministic, previously unseen pool for Stage 1.2 and VLM calibration."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
DEFAULT_INPUT = SPECIALIST_ROOT / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/vlm_run_v1/calibration_pool_240.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/vlm_run_v1/calibration_pool_240_summary.json"
VERSION = "general_edit_stage2_calibration_pool_240_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def reviewed_ids() -> set[str]:
    paths = [
        SPECIALIST_ROOT / "artifacts/survey_manifest_reco_v3.csv",
        SPECIALIST_ROOT / "artifacts/filter_run_v1/stage1_audit_300.csv",
        SPECIALIST_ROOT / "artifacts/filter_run_v1/stage1_uncertain_review_15.csv",
        SPECIALIST_ROOT / "artifacts/filter_run_v1/stage1_handheld_remove_review_15.csv",
    ]
    result: set[str] = set()
    for path in paths:
        if path.is_file():
            result.update(row["sample_id"] for row in read_csv(path))
    return result


def round_robin_sample(
    rows: list[dict[str, str]], count: int, seed: int, keys: tuple[str, ...]
) -> list[dict[str, str]]:
    randomizer = random.Random(seed)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    strata = sorted(groups)
    randomizer.shuffle(strata)
    for values in groups.values():
        randomizer.shuffle(values)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if groups[stratum]:
                selected.append(groups[stratum].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"cannot sample {count} rows from {len(rows)} candidates")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-output", type=Path, default=ROOT / "artifacts/vlm_run_v1/calibration_robot_pool_120.csv")
    parser.add_argument("--general-output", type=Path, default=ROOT / "artifacts/vlm_run_v1/calibration_general_pool_120.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--robot-count", type=int, default=120)
    parser.add_argument("--general-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    excluded = reviewed_ids()
    candidates = [row for row in read_csv(args.input) if row["sample_id"] not in excluded]
    robot = [row for row in candidates if row.get("route_stage1_2") == "1"]
    general = [row for row in candidates if row.get("route_stage1_2") != "1"]
    selected_robot = round_robin_sample(
        robot, args.robot_count, args.seed, ("operation", "stage1_1_decision", "target_scope")
    )
    selected_general = round_robin_sample(
        general,
        args.general_count,
        args.seed + 1,
        ("branch", "operation", "stage1_1_decision", "target_scope"),
    )
    selected = [
        {**row, "calibration_pool_version": VERSION, "calibration_route": route}
        for route, rows in (("robot", selected_robot), ("general", selected_general))
        for row in rows
    ]
    fields = list(dict.fromkeys(list(selected[0]) + ["calibration_pool_version", "calibration_route"]))
    for path, values in (
        (args.output, selected),
        (args.robot_output, [row for row in selected if row["calibration_route"] == "robot"]),
        (args.general_output, [row for row in selected if row["calibration_route"] == "general"]),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part.csv")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(path)
    summary = {
        "version": VERSION,
        "rows": len(selected),
        "previously_reviewed_ids": len(excluded),
        "route_counts": dict(Counter(row["calibration_route"] for row in selected)),
        "branch_counts": dict(Counter(row["branch"] for row in selected)),
        "operation_counts": dict(Counter(row["operation"] for row in selected)),
        "original_decision_counts": dict(Counter(row["stage1_1_decision"] for row in selected)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
