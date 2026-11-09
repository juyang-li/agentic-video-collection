#!/usr/bin/env python3
"""Select the 60 robot + 40 general compact calibration review."""

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
RUN = ROOT / "artifacts/vlm_run_v1"
VERSION = "general_edit_stage1_2_stage2_calibration_100_v1"
METRICS = (
    ("crop_musiq_p10", True),
    ("siglip_quality_margin", True),
    ("nonedit_dino_cosine_p10", True),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def percentile_ranks(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[field]))
    denominator = max(1, len(ordered) - 1)
    return {row["sample_id"]: index / denominator for index, row in enumerate(ordered)}


def robot_badness(rows: list[dict[str, str]]) -> dict[str, float]:
    ranks = {field: percentile_ranks(rows, field) for field, _ in METRICS}
    return {
        row["sample_id"]: max(1.0 - ranks[field][row["sample_id"]] for field, _ in METRICS)
        for row in rows
    }


def diverse_take(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    randomizer = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("operation", ""), row.get("branch", ""))].append(row)
    keys = sorted(groups)
    randomizer.shuffle(keys)
    for values in groups.values():
        randomizer.shuffle(values)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"cannot select {count} rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=RUN / "calibration_pool_240.csv")
    parser.add_argument("--robot-scores", type=Path, default=RUN / "calibration_robot_scores.csv")
    parser.add_argument("--qwen-scores", type=Path, default=RUN / "calibration_qwen_scores.csv")
    parser.add_argument("--output", type=Path, default=RUN / "calibration_review_100.csv")
    parser.add_argument("--summary", type=Path, default=RUN / "calibration_review_100_summary.json")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    pool = {row["sample_id"]: row for row in read_csv(args.pool)}
    robot_scores = [
        row for row in read_csv(args.robot_scores)
        if row.get("technical_status") == "pass" and row.get("model_backend") != "mock"
    ]
    qwen_scores = [
        row for row in read_csv(args.qwen_scores)
        if row.get("parse_status") == "pass" and row.get("model_backend") != "mock"
    ]
    if len(robot_scores) < 60:
        raise ValueError(f"need at least 60 real robot scores, got {len(robot_scores)}")
    badness = robot_badness(robot_scores)
    ranked = sorted(robot_scores, key=lambda row: badness[row["sample_id"]])
    third = len(ranked) // 3
    robot_selected: list[dict[str, str]] = []
    for tier, values in (
        ("stage1_2_high", ranked[:third]),
        ("stage1_2_boundary", ranked[third : 2 * third]),
        ("stage1_2_low", ranked[2 * third :]),
    ):
        for row in diverse_take(values, 20, args.seed + len(robot_selected)):
            robot_selected.append({**row, "review_stratum": tier})

    selected_ids = {row["sample_id"] for row in robot_selected}
    general_scores = [
        row
        for row in qwen_scores
        if row["sample_id"] not in selected_ids
        and pool.get(row["sample_id"], {}).get("calibration_route") == "general"
    ]
    quotas = {"accepted": 14, "rejected": 13, "uncertain": 13}
    general_selected: list[dict[str, str]] = []
    for decision, quota in quotas.items():
        values = [row for row in general_scores if row["model_decision"] == decision]
        if len(values) < quota:
            remaining = [row for row in general_scores if row["sample_id"] not in {x["sample_id"] for x in general_selected}]
            values = values + [row for row in remaining if row not in values]
        for row in diverse_take(values, quota, args.seed + 100 + len(general_selected)):
            actual = row.get("model_decision", "unknown")
            suffix = "" if actual == decision else f"_fallback_for_{decision}"
            general_selected.append({**row, "review_stratum": f"qwen_{actual}{suffix}"})

    selected_scores = robot_selected + general_selected
    rows: list[dict[str, str]] = []
    for rank, score in enumerate(selected_scores, 1):
        source = pool[score["sample_id"]]
        rows.append(
            {
                **source,
                "survey_version": VERSION,
                "sample_rank": str(rank),
                "review_stratum": score["review_stratum"],
                "source_video_id": source.get("source_video_id") or f"{source['sample_id']}:source",
                "target_video_id": source.get("target_video_id") or f"{source['sample_id']}:target",
            }
        )
    fields = list(dict.fromkeys(["survey_version", "sample_rank", "review_stratum"] + list(rows[0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    summary = {
        "survey_version": VERSION,
        "rows": len(rows),
        "strata": dict(Counter(row["review_stratum"] for row in rows)),
        "branches": dict(Counter(row["branch"] for row in rows)),
        "operations": dict(Counter(row["operation"] for row in rows)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
