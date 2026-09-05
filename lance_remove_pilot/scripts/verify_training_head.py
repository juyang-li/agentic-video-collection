#!/usr/bin/env python3
"""Verify finite early metrics and dataset sampling without waiting for completion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--expected-dataset", action="append", required=True)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.csv"
    samples_path = args.run_dir / "dataset_samples.csv"
    log_path = args.run_dir / "train.log"
    if not metrics_path.is_file() or not samples_path.is_file() or not log_path.is_file():
        raise SystemExit("early verification failed: metrics, samples, or train log is missing")
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) < args.min_steps:
        raise SystemExit(f"early verification failed: only {len(metrics)} steps")
    head = metrics[: args.min_steps]
    if [int(row["step"]) for row in head] != list(range(args.min_steps)):
        raise SystemExit("early verification failed: optimizer steps are not continuous")
    for row in head:
        for field in ("total_loss", "mse_loss", "grad_norm", "total_mse_tokens"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f"early verification failed: {field}={value} at step {row['step']}")
        if float(row["total_ce_tokens"]) != 0:
            raise SystemExit(f"early verification failed: CE targets at step {row['step']}")

    with samples_path.open(encoding="utf-8", newline="") as handle:
        sample_rows = [row for row in csv.DictReader(handle) if int(row["step"]) < args.min_steps]
    counts = Counter()
    for row in sample_rows:
        counts[row["dataset_name"]] += int(row["samples"])
    expected = set(args.expected_dataset)
    if set(counts) != expected:
        raise SystemExit(
            f"early verification failed: datasets={sorted(counts)}, expected={sorted(expected)}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in ("[TRAINING EXCEPTION]", "Non-finite loss", "CUDA out of memory"):
        if marker in log_text:
            raise SystemExit(f"early verification failed: log contains {marker!r}")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(args.run_dir),
                "verified_steps": args.min_steps,
                "losses": [float(row["total_loss"]) for row in head],
                "grad_norms": [float(row["grad_norm"]) for row in head],
                "dataset_samples": dict(sorted(counts.items())),
                "max_reserved_mb": max(float(row["max_memory_reserved_mb"]) for row in head),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
