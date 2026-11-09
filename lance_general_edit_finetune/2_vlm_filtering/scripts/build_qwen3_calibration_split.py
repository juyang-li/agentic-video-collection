#!/usr/bin/env python3
"""Split current-policy historical labels into prompt-dev and untouched holdout sets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen3_openrouter_v1"
DEFAULT_LABELS = ROOT / "artifacts/vlm_run_v1/historical_pair_labels.csv"
DEFAULT_DEV_SOURCE = ROOT / "artifacts/vlm_run_v4/current_scope_smoke_36.csv"


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
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dev-source", type=Path, default=DEFAULT_DEV_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    args = parser.parse_args()

    labels = [row for row in read_csv(args.labels) if row.get("current_scope") == "1"]
    dev_source_ids = {row["sample_id"] for row in read_csv(args.dev_source)}
    dev = [row for row in labels if row["sample_id"] in dev_source_ids]
    holdout = [row for row in labels if row["sample_id"] not in dev_source_ids]
    if not dev or not holdout:
        raise ValueError("calibration split must contain both dev and holdout rows")
    all_ids = [row["sample_id"] for row in labels]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("current-scope labels contain duplicate sample_id")
    if {row["sample_id"] for row in dev} & {row["sample_id"] for row in holdout}:
        raise ValueError("dev and holdout sets overlap")

    fields = list(labels[0])
    write_csv(args.output_dir / "calibration_dev.csv", dev, fields)
    write_csv(args.output_dir / "calibration_holdout.csv", holdout, fields)
    summary = {
        "source_current_scope": len(labels),
        "dev": len(dev),
        "holdout": len(holdout),
        "dev_human_labels": dict(Counter(row["human_label"] for row in dev)),
        "holdout_human_labels": dict(Counter(row["human_label"] for row in holdout)),
        "dev_by_operation": dict(Counter(row["operation"] for row in dev)),
        "holdout_by_operation": dict(Counter(row["operation"] for row in holdout)),
    }
    (args.output_dir / "calibration_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
