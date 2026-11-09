#!/usr/bin/env python3
"""Assemble the human-binary overlap covered by the historical Qwen72B v12 run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
DEFAULT_LABELS = RUN / "labels/explicit_binary_dataset.csv"
DEFAULT_OUTPUT = RUN / "qwen72b_v12_existing_overlap"
DEFAULT_SCORES = (
    ROOT
    / "artifacts/qwen72b_artifact_risk_v1/"
    "artifact_risk_30_best_routed_v12_scores.csv",
    ROOT
    / "artifacts/v12_holdout40_v1/v12_holdout40_best_v12_scores.csv",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


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
    parser.add_argument("--score", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    labels, label_fields = read_csv(args.labels)
    score_paths = tuple(args.score or DEFAULT_SCORES)
    score_rows: list[dict[str, str]] = []
    score_fields: list[str] = []
    for path in score_paths:
        rows, fields = read_csv(path)
        score_rows.extend(rows)
        score_fields.extend(fields)
    scores = {row["sample_id"]: row for row in score_rows}
    if len(scores) != len(score_rows):
        raise ValueError("historical v12 score sources contain duplicate IDs")

    overlap_labels = [row for row in labels if row["sample_id"] in scores]
    overlap_ids = {row["sample_id"] for row in overlap_labels}
    overlap_scores = [scores[row["sample_id"]] for row in overlap_labels]
    if not overlap_labels:
        raise ValueError("no overlap between explicit labels and v12 scores")
    if {row["sample_id"] for row in overlap_scores} != overlap_ids:
        raise ValueError("failed to assemble exact v12 overlap")

    write_csv(args.output_dir / "labels.csv", overlap_labels, label_fields)
    write_csv(
        args.output_dir / "scores.csv",
        overlap_scores,
        list(dict.fromkeys(score_fields)),
    )
    print(f"v12_overlap_rows={len(overlap_labels)}")


if __name__ == "__main__":
    main()
