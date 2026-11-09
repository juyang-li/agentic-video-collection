#!/usr/bin/env python3
"""Build a deterministic 300-pair audit across accept/boundary/reject decisions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def stratified_sample(
    rows: list[dict[str, str]], quota: int, seed: int
) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["branch"], row["pilot_group"], row["target_scope"])].append(row)
    randomizer = random.Random(seed)
    for key in groups:
        randomizer.shuffle(groups[key])
    keys = sorted(groups)
    randomizer.shuffle(keys)
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
            raise ValueError(f"cannot fill audit quota={quota}; available={len(rows)}")
    return selected


def auditable(row: dict[str, str]) -> bool:
    return (
        row.get("stage1_technical_status") == "pass"
        and all(
            row.get(f"{role}_local")
            and (ROOT / row[f"{role}_local"]).resolve().is_file()
            for role in ("source", "target")
        )
    )


def direction(row: dict[str, str]) -> str:
    return f"{row.get('source_provenance', 'unknown')}_to_{row.get('target_provenance', 'unknown')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_audit_300.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "stage1_audit_300_summary.json")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--quota-per-decision", type=int, default=100)
    args = parser.parse_args()

    rows = read_csv(args.input)
    selected: list[dict[str, str]] = []
    seed_offsets = {"accepted": 1, "uncertain": 2, "rejected": 3}
    for decision in ("accepted", "uncertain", "rejected"):
        candidates = [
            row
            for row in rows
            if row["stage1_1_decision"] == decision and auditable(row)
        ]
        chosen = stratified_sample(
            candidates, args.quota_per_decision, args.seed + seed_offsets[decision]
        )
        selected.extend(
            {
                **row,
                "audit_bucket": decision,
                "survey_version": "general_edit_stage1_audit_300_v1",
                "source_video_id": row.get("source_video_id") or f"{row['pair_id']}:source",
                "target_video_id": row.get("target_video_id") or f"{row['pair_id']}:target",
                "direction": direction(row),
            }
            for row in chosen
        )
    for rank, row in enumerate(selected, 1):
        row["audit_rank"] = str(rank)
        row["sample_rank"] = str(rank)

    compatibility = [
        "survey_version",
        "sample_rank",
        "source_video_id",
        "target_video_id",
        "direction",
    ]
    fields = list(dict.fromkeys(["audit_rank", "audit_bucket"] + compatibility + list(rows[0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)
    summary = {
        "rows": len(selected),
        "seed": args.seed,
        "decision_counts": dict(Counter(row["audit_bucket"] for row in selected)),
        "branch_counts": dict(Counter(row["branch"] for row in selected)),
        "pilot_group_counts": dict(Counter(row["pilot_group"] for row in selected)),
        "scope_counts": dict(Counter(row["target_scope"] for row in selected)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
