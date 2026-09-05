#!/usr/bin/env python3
"""Select a deterministic 8+8 InsAVE/GOKU removal smoke manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


PILOT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PILOT_ROOT.parent
INSAVE_POPULATION = PROJECT_ROOT / "dataset_research/insave_research/shard_population.csv"
GOKU_POPULATION = PROJECT_ROOT / "dataset_research/goku_research/remove/shard_population.csv"
DEFAULT_OUTPUT = PILOT_ROOT / "smoke_manifest.csv"

DATASETS = {
    "insave": {
        "repo_id": "suimu/InsAVE-80K",
        "revision": "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289",
        "shard": "train_data/add_and_remove/part_01.tar",
    },
    "goku": {
        "repo_id": "bigfacing/GOKU-2M",
        "revision": "f289f4b245648dc57db04afaaeb68659440c0fa6",
        "shard": "remove/remove.videos.part32.tar",
    },
}

FIELDS = [
    "pair_id",
    "dataset",
    "repo_id",
    "revision",
    "official_id",
    "instruction",
    "source_member",
    "target_member",
    "shard",
    "selection_seed",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty population: {path}")
    return rows


def select_rows(path: Path, dataset: str, count: int, seed: int) -> list[dict[str, str]]:
    population = read_csv(path)
    if len(population) < count:
        raise ValueError(f"{dataset} population has {len(population)} rows, need {count}")
    selected = random.Random(seed).sample(population, count)
    metadata = DATASETS[dataset]
    output: list[dict[str, str]] = []
    for row in selected:
        if dataset == "insave":
            official_id = row["metadata_index"]
            pair_id = f"insave-{int(official_id):05d}"
        else:
            official_id = row["case_id"]
            pair_id = f"goku-{official_id}"
        instruction = row["instruction"].strip()
        if not instruction:
            raise ValueError(f"empty instruction for {pair_id}")
        output.append(
            {
                "pair_id": pair_id,
                "dataset": dataset,
                "repo_id": metadata["repo_id"],
                "revision": metadata["revision"],
                "official_id": official_id,
                "instruction": instruction,
                "source_member": row["source_member"],
                "target_member": row["target_member"],
                "shard": metadata["shard"],
                "selection_seed": str(seed),
            }
        )
    return sorted(output, key=lambda item: item["pair_id"])


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-dataset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.count_per_dataset < 1:
        raise ValueError("count-per-dataset must be positive")

    rows = select_rows(
        INSAVE_POPULATION,
        "insave",
        args.count_per_dataset,
        args.seed,
    )
    rows.extend(
        select_rows(
            GOKU_POPULATION,
            "goku",
            args.count_per_dataset,
            args.seed + 1,
        )
    )
    write_manifest(args.output, rows)
    print(
        f"wrote={args.output} rows={len(rows)} "
        f"insave={sum(r['dataset'] == 'insave' for r in rows)} "
        f"goku={sum(r['dataset'] == 'goku' for r in rows)}"
    )


if __name__ == "__main__":
    main()
