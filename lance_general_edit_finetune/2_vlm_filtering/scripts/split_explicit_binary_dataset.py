#!/usr/bin/env python3
"""Freeze prompt-development train/validation and untouched systematic test splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
LABELS = RUN / "labels"
DEFAULT_LABELS = LABELS / "explicit_binary_dataset.csv"
DEFAULT_INPUT = LABELS / "explicit_binary_v12_input.csv"
DEFAULT_OUTPUT = LABELS / "splits"


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


def id_hash(rows: list[dict[str, str]]) -> str:
    value = "\n".join(row["sample_id"] for row in rows) + "\n"
    return hashlib.sha256(value.encode()).hexdigest()


def composition(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row["human_decision"] for row in rows)),
        "datasets": dict(Counter(row["dataset"] for row in rows)),
        "branches": dict(Counter(row["branch"] for row in rows)),
        "audit_operations": dict(Counter(row["audit_operation_expected"] for row in rows)),
        "audit_relations": dict(Counter(row["audit_relation_expected"] for row in rows)),
        "sample_id_sha256": id_hash(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    args = parser.parse_args()

    labels, label_fields = read_csv(args.labels)
    inputs, input_fields = read_csv(args.model_input)
    label_by_id = {row["sample_id"]: row for row in labels}
    if len(label_by_id) != len(labels) or {row["sample_id"] for row in inputs} != set(label_by_id):
        raise ValueError("label and model-input IDs must be identical and unique")
    merged: list[dict[str, str]] = []
    for row in inputs:
        value = dict(row)
        label = label_by_id[row["sample_id"]]
        value.update({
            "human_decision": label["human_decision"],
            "human_notes": label.get("human_notes", ""),
            "annotation_source": label["annotation_source"],
            "annotation_partition": label["annotation_partition"],
            "benchmark_usage": label["benchmark_usage"],
            "audit_operation": label["audit_operation"],
            "audit_relation": label["audit_relation"],
            "audit_operation_expected": label["audit_operation"],
            "audit_relation_expected": label["audit_relation"],
        })
        merged.append(value)

    development = [row for row in merged if row["annotation_partition"] == "development"]
    test = [
        row for row in merged
        if row["annotation_partition"] == "test_v1_partial_completed"
    ]
    if len(development) != 139 or len(test) != 91:
        raise ValueError("expected 139 development and 91 completed systematic rows")

    rng = random.Random(args.seed)
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in development:
        strata[(row["branch"], row["human_decision"])].append(row)
    validation_ids: set[str] = set()
    for key in sorted(strata):
        group = list(strata[key])
        rng.shuffle(group)
        count = round(len(group) * args.validation_fraction)
        if len(group) >= 2:
            count = min(len(group) - 1, max(1, count))
        else:
            count = 0
        validation_ids.update(row["sample_id"] for row in group[:count])
    train = [row for row in development if row["sample_id"] not in validation_ids]
    validation = [row for row in development if row["sample_id"] in validation_ids]

    output_fields = list(dict.fromkeys([*input_fields, *label_fields]))
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_csv(args.output_dir / f"{name}.csv", rows, output_fields)
    summary = {
        "split_version": "explicit_binary_prompt_eval_split_v1",
        "seed": args.seed,
        "policy": (
            "stratify development by branch+label into train/validation; keep all "
            "completed systematic test_v1 rows untouched as final test"
        ),
        "test_leakage_policy": "do not call or inspect model test predictions before prompt freeze",
        "train": composition(train),
        "validation": composition(validation),
        "test": composition(test),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
