#!/usr/bin/env python3
"""Uniformly sample Ditto pairs from the archive-front local/0130 subpopulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "metadata" / "local.json"
OUTPUT_PATH = ROOT / "random_pairs.csv"
EXPECTED_METADATA_SHA256 = (
    "206004a6d14eb1f4ad820d1798c395f396c3b031d9613f77499a945ab472cdd7"
)
SUBPOPULATION_PREFIX = "local/0130/"
DEFAULT_SEED = 20260821
DEFAULT_SAMPLE_SIZE = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_task(instruction: str) -> str:
    first_word = instruction.strip().split(maxsplit=1)[0].lower()
    return {
        "add": "add",
        "change": "change",
        "remove": "remove",
        "replace": "replace",
        "transform": "transform",
        "introduce": "add",
        "alter": "change",
    }.get(first_word, "other")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    metadata_sha256 = sha256_file(METADATA_PATH)
    if metadata_sha256 != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata SHA256 changed; inspect before resampling")
    with METADATA_PATH.open(encoding="utf-8") as file:
        full_population = json.load(file)
    population = [
        (index, item)
        for index, item in enumerate(full_population)
        if item["source_path"].startswith(SUBPOPULATION_PREFIX)
        and item["edited_path"].startswith(SUBPOPULATION_PREFIX)
    ]
    if not 0 < args.sample_size <= len(population):
        raise ValueError("sample-size must fit the local/0130 subpopulation")
    subpopulation_indices = random.Random(args.seed).sample(
        range(len(population)), args.sample_size
    )

    fields = [
        "sample_rank",
        "metadata_index",
        "subpopulation_index",
        "sample_id",
        "official_category",
        "task",
        "seed",
        "subpopulation",
        "subpopulation_size",
        "instruction",
        "source_path",
        "edited_path",
        "source_file",
        "target_file",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, subpopulation_index in enumerate(subpopulation_indices, start=1):
            metadata_index, item = population[subpopulation_index]
            sample_id = f"ditto-{metadata_index:06d}"
            writer.writerow(
                {
                    "sample_rank": rank,
                    "metadata_index": metadata_index,
                    "subpopulation_index": subpopulation_index,
                    "sample_id": sample_id,
                    "official_category": "local",
                    "task": infer_task(item["instruction"]),
                    "seed": args.seed,
                    "subpopulation": SUBPOPULATION_PREFIX,
                    "subpopulation_size": len(population),
                    "instruction": item["instruction"],
                    "source_path": item["source_path"],
                    "edited_path": item["edited_path"],
                    "source_file": f"random_pair_examples/{sample_id}_source.mp4",
                    "target_file": f"random_pair_examples/{sample_id}_target.mp4",
                }
            )
    print(
        f"subpopulation={len(population)} sample_size={args.sample_size} "
        f"seed={args.seed} indices={subpopulation_indices}"
    )


if __name__ == "__main__":
    main()
