#!/usr/bin/env python3
"""Upgrade an existing candidate manifest to the frozen Stage-1 v2 policy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/filter_run_v1/pilot_manifest_5k.csv"
DEFAULT_RUN = ROOT / "artifacts/filter_run_v3"
VERSION = "general_edit_stage1_quality_v2"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_input.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "stage1_input_summary.json")
    args = parser.parse_args()

    source = read_csv(args.input)
    if not source:
        raise ValueError("candidate manifest is empty")
    output: list[dict[str, str]] = []
    for row in source:
        item = dict(row)
        item["filter_version"] = VERSION
        if item.get("dataset") == "reco":
            item.update(
                {
                    "source_provenance": "generation_side_unresolved",
                    "target_provenance": "generation_side_unresolved",
                    "provenance_basis": "official_training_roles_without_pair_lineage",
                    "quality_checked_roles": "both",
                }
            )
        output.append(item)

    ids = [row["sample_id"] for row in output]
    pairs = [(row["dataset"], row["pair_id"]) for row in output]
    if len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
        raise ValueError("candidate manifest contains duplicate IDs")
    fields = list(source[0])
    write_csv(args.output, output, fields)
    summary = {
        "filter_version": VERSION,
        "source": str(args.input),
        "rows": len(output),
        "by_branch": dict(Counter(row["branch"] for row in output)),
        "quality_checked_roles": dict(Counter(row["quality_checked_roles"] for row in output)),
        "provenance_pairs": dict(
            Counter(
                f"{row['source_provenance']}->{row['target_provenance']}"
                for row in output
            )
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

