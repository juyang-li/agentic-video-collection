#!/usr/bin/env python3
"""Deterministically cap Stage-1 survivors before paid Stage-2 judging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def parse_quota(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        branch, separator, count = value.partition("=")
        if not separator or not branch or int(count) < 0:
            raise ValueError(f"invalid --quota value: {value}")
        result[branch] = int(count)
    return result


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--quota", action="append", default=[], metavar="BRANCH=N")
    parser.add_argument("--exclude-cascade", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    rows, fields = read_csv(args.input)
    quotas = parse_quota(args.quota)
    excluded: set[str] = set()
    for path in args.exclude_cascade:
        prior, _ = read_csv(path)
        excluded.update(row["sample_id"] for row in prior)
    candidates = [row for row in rows if row["sample_id"] not in excluded]

    selected: list[dict[str, str]] = []
    branches = sorted({row.get("branch", "") for row in candidates})
    for branch in branches:
        branch_rows = [row for row in candidates if row.get("branch", "") == branch]
        branch_rows.sort(key=lambda row: stable_key(args.seed, row["sample_id"]))
        selected.extend(branch_rows[: quotas.get(branch, len(branch_rows))])
    selected.sort(key=lambda row: (row.get("branch", ""), stable_key(args.seed, row["sample_id"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)

    available = Counter(row.get("branch", "") for row in candidates)
    chosen = Counter(row.get("branch", "") for row in selected)
    summary = {
        "selection_seed": args.seed,
        "input_rows": len(rows),
        "excluded_prior_ids": len(excluded),
        "eligible_rows": len(candidates),
        "available_by_branch": dict(available),
        "requested_quotas": quotas,
        "selected_rows": len(selected),
        "selected_by_branch": dict(chosen),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
