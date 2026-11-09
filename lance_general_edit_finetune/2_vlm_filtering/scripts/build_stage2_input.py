#!/usr/bin/env python3
"""Exclude calibrated Stage 1.2 hard failures before full Qwen inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE1_RUN = PROJECT_ROOT / "1_specialist_filtering/artifacts/filter_run_v2"
RUN = ROOT / "artifacts/vlm_run_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=STAGE1_RUN / "stage1_next_input.csv")
    parser.add_argument("--stage1-2", type=Path, default=STAGE1_RUN / "stage1_2_decisions.csv")
    parser.add_argument("--output", type=Path, default=RUN / "stage2_input.csv")
    parser.add_argument("--summary", type=Path, default=RUN / "stage2_input_summary.json")
    args = parser.parse_args()

    source = read_csv(args.input)
    decisions = {row["sample_id"]: row for row in read_csv(args.stage1_2)}
    output: list[dict[str, str]] = []
    excluded: list[str] = []
    for row in source:
        if row.get("route_stage1_2") != "1":
            output.append(row)
            continue
        stage1 = decisions.get(row["sample_id"])
        if stage1 is None:
            raise ValueError(f"missing Stage 1.2 decision for {row['sample_id']}")
        if stage1.get("stage1_2_decision") == "rejected":
            excluded.append(row["sample_id"])
        else:
            output.append(row)
    fields = list(source[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)
    summary = {
        "source_rows": len(source),
        "stage2_input": len(output),
        "stage1_2_hard_rejected_skipped": len(excluded),
        "by_branch": dict(Counter(row["branch"] for row in output)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
