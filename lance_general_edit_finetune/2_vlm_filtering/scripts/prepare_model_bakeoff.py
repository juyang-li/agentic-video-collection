#!/usr/bin/env python3
"""Join the frozen human labels to the current Stage-1 media manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_LABELS = ROOT / "artifacts/pal4vst_v1/experiment_manifest.csv"
DEFAULT_MEDIA = (
    PROJECT_ROOT
    / "1_specialist_filtering/artifacts/filter_run_v2/stage1_next_input.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/model_bakeoff_v1/current_scope_63.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/model_bakeoff_v1/manifest_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    fields = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    labels = read_csv(args.labels)
    media = {row["sample_id"]: row for row in read_csv(args.media)}
    output: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for label in labels:
        sample_id = label["sample_id"]
        source = media.get(sample_id)
        if source is None:
            excluded.append(label)
            continue
        row = dict(source)
        for field in (
            "cohort",
            "human_status",
            "human_notes",
            "human_label_basis",
        ):
            row[field] = label.get(field, "")
        # Provenance is audit-critical.  The current manifest and frozen label
        # manifest must agree instead of silently taking one side.
        for field in ("dataset", "branch", "operation", "instruction",
                      "source_provenance", "target_provenance"):
            if label.get(field, "") != source.get(field, ""):
                raise ValueError(
                    f"manifest mismatch for {sample_id} field={field}: "
                    f"labels={label.get(field)!r}, media={source.get(field)!r}"
                )
        output.append(row)

    if len(labels) != 70 or len(output) != 63 or len(excluded) != 7:
        raise ValueError(
            f"expected frozen70 -> current63 + excluded7, got "
            f"{len(labels)} -> {len(output)} + {len(excluded)}"
        )
    write_csv(args.output, output)
    summary = {
        "source_human_manifest": str(args.labels),
        "source_current_media_manifest": str(args.media),
        "frozen_label_rows": len(labels),
        "current_scope_rows": len(output),
        "excluded_rows": len(excluded),
        "current_human_counts": dict(Counter(row["human_status"] for row in output)),
        "current_dataset_operation_counts": {
            f"{dataset}/{operation}": count
            for (dataset, operation), count in sorted(
                Counter((row["dataset"], row["operation"]) for row in output).items()
            )
        },
        "excluded_sample_ids": [row["sample_id"] for row in excluded],
        "excluded_reason": (
            "These seven Goku clothing samples were removed from the current "
            "Stage-1 candidate manifest by the later branch policy."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
