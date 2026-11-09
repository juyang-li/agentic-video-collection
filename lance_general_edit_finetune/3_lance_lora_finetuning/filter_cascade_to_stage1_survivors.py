#!/usr/bin/env python3
"""Restrict an existing Stage-2 audit table to a corrected Stage-1 survivor set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--survivors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    cascade, fields = read_csv(args.cascade)
    survivors, _ = read_csv(args.survivors)
    survivor_ids = {row["sample_id"] for row in survivors}
    selected = [row for row in cascade if row["sample_id"] in survivor_ids]
    removed = [row for row in cascade if row["sample_id"] not in survivor_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)

    summary = {
        "input_rows": len(cascade),
        "survivor_ids": len(survivor_ids),
        "output_rows": len(selected),
        "removed_rows": len(removed),
        "removed_by_branch": dict(Counter(row.get("branch", "") for row in removed)),
        "removed_by_final_decision": dict(
            Counter(row.get("final_decision", "") for row in removed)
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
