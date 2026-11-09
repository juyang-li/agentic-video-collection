#!/usr/bin/env python3
"""Merge disjoint called and explicitly skipped Stage-2 score rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    seen: set[str] = set()
    for path in args.input:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = list(dict.fromkeys([*fields, *(reader.fieldnames or [])]))
            for row in reader:
                sample_id = row.get("sample_id", "")
                if not sample_id or sample_id in seen:
                    raise ValueError(f"missing or duplicate sample_id: {sample_id!r}")
                seen.add(sample_id)
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print({"rows": len(rows), "inputs": [str(path) for path in args.input]})


if __name__ == "__main__":
    main()
