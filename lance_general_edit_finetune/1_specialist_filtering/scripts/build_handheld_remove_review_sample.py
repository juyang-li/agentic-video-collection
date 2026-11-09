#!/usr/bin/env python3
"""Build a deterministic 15-pair review of hand-held object removal."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from filter_common import is_handheld_object_remove as common_is_handheld_object_remove


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"
SURVEY_VERSION = "general_edit_stage1_handheld_remove_review_15_v1"
QUOTAS = {"goku_remove": 8, "reco_remove": 7}
CATEGORY_PATTERNS = [
    ("phone_tablet", re.compile(r"\b(phone|smartphone|tablet|camera)\b", re.IGNORECASE)),
    ("book_paper", re.compile(r"\b(book|album|paper|pencil)\b", re.IGNORECASE)),
    ("container", re.compile(r"\b(cup|glass|bottle|bowl|bucket|vase)\b", re.IGNORECASE)),
    ("tool", re.compile(r"\b(microphone|paintbrush|vacuum|sword|rod|sparkler)\b", re.IGNORECASE)),
    ("instrument", re.compile(r"\b(guitar|instrument)\b", re.IGNORECASE)),
    ("bag_box", re.compile(r"\b(bag|backpack|box|gift|basket)\b", re.IGNORECASE)),
    ("animal", re.compile(r"\b(dog|rabbit|fish)\b", re.IGNORECASE)),
    ("footwear", re.compile(r"\b(shoe|shoes|sandal|sandals)\b", re.IGNORECASE)),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def auditable(row: dict[str, str]) -> bool:
    return all(
        row.get(f"{role}_local") and (ROOT / row[f"{role}_local"]).is_file()
        for role in ("source", "target")
    )


def category(instruction: str) -> str:
    for name, pattern in CATEGORY_PATTERNS:
        if pattern.search(instruction):
            return name
    return "other"


def is_handheld_object_remove(row: dict[str, str]) -> bool:
    return common_is_handheld_object_remove(
        row.get("branch", ""), row.get("target_scope", ""), row.get("instruction", "")
    )


def diverse_sample(
    rows: list[dict[str, str]], quota: int, randomizer: random.Random
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[category(row["instruction"])].append(row)
    keys = sorted(groups)
    randomizer.shuffle(keys)
    for values in groups.values():
        randomizer.shuffle(values)
    selected: list[dict[str, str]] = []
    while len(selected) < quota:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise ValueError(f"cannot fill quota={quota}; available={len(rows)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[DEFAULT_RUN / "stage1_audit_300.csv", DEFAULT_RUN / "stage1_uncertain_review_15.csv"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_handheld_remove_review_15.csv")
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_RUN / "stage1_handheld_remove_review_15_summary.json"
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    excluded: set[str] = set()
    for path in args.exclude:
        if path.is_file():
            excluded.update(row["sample_id"] for row in read_csv(path))
    candidates = [
        row
        for row in read_csv(args.input)
        if row.get("stage1_1_decision") in {"accepted", "uncertain"}
        and row.get("sample_id") not in excluded
        and is_handheld_object_remove(row)
        and auditable(row)
    ]

    randomizer = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for branch, quota in QUOTAS.items():
        branch_rows = [row for row in candidates if row["branch"] == branch]
        if len(branch_rows) < quota:
            raise ValueError(f"branch={branch} quota={quota} available={len(branch_rows)}")
        selected.extend(diverse_sample(branch_rows, quota, randomizer))

    output_rows: list[dict[str, str]] = []
    for rank, row in enumerate(selected, 1):
        output_rows.append(
            {
                **row,
                "survey_version": SURVEY_VERSION,
                "sample_rank": str(rank),
                "handheld_category": category(row["instruction"]),
                "source_video_id": row.get("source_video_id") or f"{row['pair_id']}:source",
                "target_video_id": row.get("target_video_id") or f"{row['pair_id']}:target",
                "direction": (
                    f"{row.get('source_provenance', 'unknown')}_to_"
                    f"{row.get('target_provenance', 'unknown')}"
                ),
            }
        )

    source_fields = list(read_csv(args.input)[0])
    fields = list(
        dict.fromkeys(
            [
                "survey_version",
                "sample_rank",
                "handheld_category",
                "source_video_id",
                "target_video_id",
                "direction",
            ]
            + source_fields
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.output)

    summary = {
        "survey_version": SURVEY_VERSION,
        "rows": len(output_rows),
        "seed": args.seed,
        "excluded_previous_samples": len(excluded),
        "branch_counts": dict(Counter(row["branch"] for row in output_rows)),
        "category_counts": dict(Counter(row["handheld_category"] for row in output_rows)),
        "decision_counts": dict(Counter(row["stage1_1_decision"] for row in output_rows)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
