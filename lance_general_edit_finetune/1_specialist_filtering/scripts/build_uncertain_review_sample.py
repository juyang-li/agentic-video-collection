#!/usr/bin/env python3
"""Build a deterministic, previously unseen review sample from Stage 1.1 uncertain pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"
SURVEY_VERSION = "general_edit_stage1_uncertain_review_15_v1"
QUOTAS = {
    "egoedit_replace": 10,
    "goku_add": 1,
    "goku_remove": 1,
    "reco_add": 1,
    "reco_remove": 1,
    "reco_replace": 1,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def auditable(row: dict[str, str]) -> bool:
    return all(
        row.get(f"{role}_local")
        and (ROOT / row[f"{role}_local"]).resolve().is_file()
        for role in ("source", "target")
    )


def round_robin_sample(
    rows: list[dict[str, str]], quota: int, randomizer: random.Random
) -> list[dict[str, str]]:
    """Cover EgoEdit construction directions and boundary reasons before repeating a group."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("source_provenance", "unknown"),
            row.get("target_provenance", "unknown"),
            row.get("stage1_1_reasons", "unknown"),
        )
        groups[key].append(row)
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
    parser.add_argument("--exclude", type=Path, default=DEFAULT_RUN / "stage1_audit_300.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_uncertain_review_15.csv")
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_RUN / "stage1_uncertain_review_15_summary.json"
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    rows = read_csv(args.input)
    excluded = (
        {row["sample_id"] for row in read_csv(args.exclude)} if args.exclude.is_file() else set()
    )
    all_candidates = [
        row
        for row in rows
        if row.get("stage1_1_decision") == "uncertain"
        and row.get("branch") != "insave_remove"
        and auditable(row)
    ]
    fresh_candidates = [row for row in all_candidates if row.get("sample_id") not in excluded]

    randomizer = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for branch, quota in QUOTAS.items():
        fresh_rows = [row for row in fresh_candidates if row.get("branch") == branch]
        branch_rows = fresh_rows
        if len(branch_rows) < quota:
            branch_rows = [row for row in all_candidates if row.get("branch") == branch]
        if len(branch_rows) < quota:
            raise ValueError(f"branch={branch} quota={quota} available={len(branch_rows)}")
        if branch == "egoedit_replace":
            chosen = round_robin_sample(branch_rows, quota, randomizer)
        else:
            chosen = randomizer.sample(branch_rows, quota)
        selected.extend(chosen)

    output_rows: list[dict[str, str]] = []
    for rank, row in enumerate(selected, 1):
        output_rows.append(
            {
                **row,
                "survey_version": SURVEY_VERSION,
                "sample_rank": str(rank),
                "source_video_id": row.get("source_video_id") or f"{row['pair_id']}:source",
                "target_video_id": row.get("target_video_id") or f"{row['pair_id']}:target",
                "direction": (
                    f"{row.get('source_provenance', 'unknown')}_to_"
                    f"{row.get('target_provenance', 'unknown')}"
                ),
            }
        )

    fields = list(
        dict.fromkeys(
            [
                "survey_version",
                "sample_rank",
                "source_video_id",
                "target_video_id",
                "direction",
            ]
            + list(rows[0])
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
        "excluded_existing_audit_rows": len(excluded),
        "reused_existing_audit_rows": sum(row["sample_id"] in excluded for row in output_rows),
        "branch_counts": dict(Counter(row["branch"] for row in output_rows)),
        "reason_counts": dict(Counter(row["stage1_1_reasons"] for row in output_rows)),
        "direction_counts": dict(Counter(row["direction"] for row in output_rows)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
