#!/usr/bin/env python3
"""Merge production cascade rounds and freeze a leakage-safe 4K editing pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGETS = {
    "goku_remove": 600,
    "goku_add": 400,
    "egoedit_replace": 1000,
    "reco_add": 600,
    "reco_remove": 600,
    "reco_replace": 800,
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def leakage_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    dsu = DisjointSet([row["sample_id"] for row in rows])
    media_owner: dict[tuple[str, str], str] = {}
    for row in rows:
        for role in ("source", "target"):
            locator = row.get(f"{role}_locator") or row.get(f"{role}_local", "")
            if not locator:
                continue
            key = (row.get("dataset", ""), locator)
            owner = media_owner.setdefault(key, row["sample_id"])
            dsu.union(owner, row["sample_id"])
    members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        members[dsu.find(row["sample_id"])].append(row["sample_id"])
    result: dict[str, str] = {}
    for values in members.values():
        group_id = hashlib.sha256("|".join(sorted(values)).encode()).hexdigest()[:16]
        result.update({value: group_id for value in values})
    return result


def split_for_group(seed: int, group_id: str) -> str:
    bucket = int(stable_key(seed, group_id)[:8], 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    seen: set[str] = set()
    for path in args.cascade:
        part, part_fields = read_csv(path)
        fields = list(dict.fromkeys([*fields, *part_fields]))
        for row in part:
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample across cascade rounds: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
    accepted = [row for row in rows if row.get("cascade_status") == "accepted"]
    rejected = [row for row in rows if row.get("cascade_status") != "accepted"]
    available = Counter(row["branch"] for row in accepted)
    deficits = {branch: target - available[branch] for branch, target in TARGETS.items() if available[branch] < target}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_accepted_available.csv", accepted, fields)
    write_csv(args.output_dir / "all_not_accepted.csv", rejected, fields)
    selected: list[dict[str, str]] = []
    if not deficits:
        for branch, target in TARGETS.items():
            candidates = [row for row in accepted if row["branch"] == branch]
            candidates.sort(key=lambda row: stable_key(args.seed, row["sample_id"]))
            selected.extend(candidates[:target])
        groups = leakage_groups(selected)
        for row in selected:
            row["split_group_id"] = groups[row["sample_id"]]
            row["split"] = split_for_group(args.seed, row["split_group_id"])
        fields = list(dict.fromkeys([*fields, "split_group_id", "split"]))
        write_csv(args.output_dir / "final_edit_pool_4k.csv", selected, fields)

    summary = {
        "selection_seed": args.seed,
        "cascade_rows": len(rows),
        "accepted_available": len(accepted),
        "not_accepted": len(rejected),
        "target_by_branch": TARGETS,
        "accepted_available_by_branch": dict(available),
        "deficits": deficits,
        "final_pool_status": "complete" if not deficits else "needs_more_candidates",
        "final_pool_rows": len(selected),
        "final_pool_by_branch": dict(Counter(row["branch"] for row in selected)),
        "final_pool_by_split": dict(Counter(row.get("split", "") for row in selected)),
        "split_policy": "shared-media connected component hash; train/validation/test=90/5/5 in expectation",
    }
    (args.output_dir / "finalize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    if deficits:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

