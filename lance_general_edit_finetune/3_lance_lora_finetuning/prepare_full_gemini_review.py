#!/usr/bin/env python3
"""Prepare every low-risk Qwen direct accept for supplemental Gemini review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def interleave_by_branch(
    rows: list[dict[str, str]], seed: int
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("branch", "")].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_key(seed, row["sample_id"]))
    ordered: list[dict[str, str]] = []
    branches = sorted(groups)
    while any(groups.values()):
        for branch in branches:
            if groups[branch]:
                ordered.append(groups[branch].pop())
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    seen: set[str] = set()
    for path in args.cascade:
        part, part_fields = read_csv(path)
        fields = list(dict.fromkeys([*fields, *part_fields]))
        for row in part:
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate sample ID across cascades: {sample_id}")
            seen.add(sample_id)
            rows.append(row)

    selected = [
        dict(row, supplemental_gemini_reason="qwen_accept_low_risk_not_previously_routed")
        for row in rows
        if row.get("cascade_route") == "qwen_accept_low_risk"
        and row.get("cascade_status") == "accepted"
        and row.get("gemini_decision") in {"", "not_routed"}
    ]
    selected = interleave_by_branch(selected, args.seed)
    if len(selected) != 1882:
        raise ValueError(f"expected 1882 supplemental rows, found {len(selected)}")

    fields = list(dict.fromkeys([*fields, "supplemental_gemini_reason"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)

    summary = {
        "selection_seed": args.seed,
        "selection_policy": "branch_interleaved_then_seeded_sample_hash",
        "cascade_rows": len(rows),
        "supplemental_rows": len(selected),
        "by_dataset": dict(Counter(row.get("dataset", "") for row in selected)),
        "by_branch": dict(Counter(row.get("branch", "") for row in selected)),
        "reason": "Gemini full review of prior qwen_accept_low_risk rows",
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
