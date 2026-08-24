#!/usr/bin/env python3
"""Uniformly sample 15 TripVVT triplets from the complete official metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = Path("/tmp/tripvvt_probe")
OUTPUT_PATH = ROOT / "random_pairs.csv"
REPO_ID = "TripVVT/TripVVT-10K"
REVISION = "596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3"
EXPECTED_HASHES = {
    "train": "1f1d1368f182836d697c0a0daa99169b45b61c2bf1a4ce7cef57b629d8538b83",
    "test": "590d5b1ed925641f616238cfab9ad69318c1032f29d0c26af34916a2921db62b",
}
EXPECTED_COUNTS = {"train": 9_931, "test": 100}
SEED = 20260822
SAMPLE_SIZE = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(metadata_dir: Path, split: str) -> list[dict[str, object]]:
    path = metadata_dir / f"{split}.jsonl"
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_HASHES[split]:
        raise ValueError(f"{split} metadata SHA256 changed: {actual_hash}")
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"unexpected {split} row count: {len(rows)}")
    return rows


def normalized_instruction(cloth_type: str) -> str:
    regions = {
        "upper_body": "upper-body clothing",
        "lower_body": "lower-body clothing",
        "dresses": "outfit",
    }
    return f"Replace the person's {regions[cloth_type]} with the referenced garment."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    args = parser.parse_args()

    split_rows = {
        split: read_split(args.metadata_dir, split) for split in ("train", "test")
    }
    population: list[tuple[str, int, dict[str, object]]] = []
    for split in ("train", "test"):
        for split_index, row in enumerate(split_rows[split]):
            population.append((split, split_index, row))

    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(population)), SAMPLE_SIZE)
    output_rows: list[dict[str, object]] = []
    for rank, population_index in enumerate(selected_indices, start=1):
        split, split_index, row = population[population_index]
        sample_id = f"tripvvt-{rank:04d}"
        cloth_type = str(row["cloth_type"])
        output_rows.append(
            {
                "sample_rank": rank,
                "population_index": population_index,
                "split": split,
                "split_index": split_index,
                "sample_id": sample_id,
                "task": f"try-on-{cloth_type}",
                "seed": SEED,
                "population_size": len(population),
                "repo_id": REPO_ID,
                "revision": REVISION,
                "instruction": normalized_instruction(cloth_type),
                "official_text": str(row["text"]).strip(),
                "cloth_type": cloth_type,
                "source_member": row["input"],
                "target_member": row["output"],
                "cloth_member": row["cloth_image"],
                "line_member": row["line_image"],
                "mask_member": row["mask_condition"],
                "pose_member": row["pose_dw"],
                "source_file": f"examples/{sample_id}_source.mp4",
                "target_file": f"examples/{sample_id}_target.mp4",
                "cloth_file": f"examples/{sample_id}_cloth.png",
                "declared_width": row["width"],
                "declared_height": row["height"],
            }
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"population={len(population)} sample_size={SAMPLE_SIZE} seed={SEED}")
    print(f"selected_indices={selected_indices}")
    print(f"unique_targets={len({row['target_member'] for row in output_rows})}")
    train_targets = {str(row["output"]) for row in split_rows["train"]}
    test_targets = {str(row["output"]) for row in split_rows["test"]}
    print(
        f"population_unique_targets={len(train_targets | test_targets)} "
        f"train_test_target_overlap={len(train_targets & test_targets)}"
    )


if __name__ == "__main__":
    main()
