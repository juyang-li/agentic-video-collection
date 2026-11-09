#!/usr/bin/env python3
"""Merge independently scored Stage 1 CSV shards by stable video_id."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from run_stage1_quality import FIELDS, read_csv, write_results


def completeness(row: dict[str, str]) -> tuple[int, int]:
    """Prefer full model scores, then completed technical rows."""
    full = bool(row.get("musiq_p10") and row.get("dino_max_feature_jump"))
    technical = row.get("technical_status") in {"pass", "fail_short", "error"}
    return int(full), int(technical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, dict[str, str]] = {}
    duplicates = 0
    replacements = 0
    for path in args.input:
        for row in read_csv(path):
            video_id = row["video_id"]
            if video_id in merged:
                duplicates += 1
                if completeness(row) > completeness(merged[video_id]):
                    merged[video_id] = row
                    replacements += 1
            else:
                merged[video_id] = row

    rows = sorted(
        merged.values(),
        key=lambda row: (row["dataset"], row["branch"], row["pair_id"], row["role"]),
    )
    write_results(args.output, rows)
    summary = {
        "inputs": [str(path) for path in args.input],
        "input_rows": sum(sum(1 for _ in read_csv(path)) for path in args.input),
        "unique_videos": len(rows),
        "duplicate_rows": duplicates,
        "completeness_replacements": replacements,
        "status_counts": dict(Counter(row["technical_status"] for row in rows)),
        "full_scores": sum(bool(row.get("musiq_p10")) for row in rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
