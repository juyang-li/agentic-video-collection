#!/usr/bin/env python3
"""Freeze the 500-pair mixed/InsAVE ablation manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


PILOT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PILOT_ROOT.parent
INSAVE_POPULATION = PROJECT_ROOT / "dataset_research/insave_research/shard_population.csv"
GOKU_POPULATION = PROJECT_ROOT / "dataset_research/goku_research/remove/shard_population.csv"

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
    "selection_rank",
]


def read_population(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty population: {path}")
    return rows


def select(path: Path, dataset: str, count: int, seed: int) -> list[dict[str, str]]:
    population = read_population(path)
    eligible = []
    for row in population:
        instruction = row["instruction"].strip()
        if not instruction or not instruction.lower().startswith("remove"):
            continue
        if dataset == "goku" and row.get("task") != "remove":
            continue
        eligible.append(row)
    if len(eligible) < count:
        raise ValueError(f"{dataset} has {len(eligible)} eligible rows, need {count}")

    metadata = DATASETS[dataset]
    selected = random.Random(seed).sample(eligible, count)
    result = []
    for rank, row in enumerate(selected):
        if dataset == "insave":
            official_id = row["metadata_index"]
            pair_id = f"insave-{int(official_id):05d}"
        else:
            official_id = row["case_id"]
            pair_id = f"goku-{official_id}"
        result.append(
            {
                "pair_id": pair_id,
                "dataset": dataset,
                "repo_id": metadata["repo_id"],
                "revision": metadata["revision"],
                "official_id": official_id,
                "instruction": row["instruction"].strip(),
                "source_member": row["source_member"],
                "target_member": row["target_member"],
                "shard": metadata["shard"],
                "selection_seed": str(seed),
                "selection_rank": str(rank),
            }
        )
    return result


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["pair_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    all_path = PILOT_ROOT / "experiment_manifest_700.csv"
    mixed_path = PILOT_ROOT / "mixed_300_200_manifest.csv"
    control_path = PILOT_ROOT / "insave_500_manifest.csv"
    metadata_path = PILOT_ROOT / "experiment_data.json"
    outputs = [all_path, mixed_path, control_path, metadata_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.rebuild:
        raise FileExistsError(f"refusing to overwrite: {existing}; pass --rebuild")

    insave = select(INSAVE_POPULATION, "insave", 500, args.seed)
    goku = select(GOKU_POPULATION, "goku", 200, args.seed + 1)
    mixed = insave[:300] + goku
    all_rows = insave + goku

    if len({row["pair_id"] for row in all_rows}) != 700:
        raise ValueError("pair IDs are not unique")
    if {row["pair_id"] for row in mixed if row["dataset"] == "insave"} - {
        row["pair_id"] for row in insave
    }:
        raise ValueError("mixed InsAVE rows are not a subset of the control")

    write_manifest(all_path, all_rows)
    write_manifest(mixed_path, mixed)
    write_manifest(control_path, insave)
    metadata_path.write_text(
        json.dumps(
            {
                "selection_seed": args.seed,
                "insave_seed": args.seed,
                "goku_seed": args.seed + 1,
                "insave_control_pairs": 500,
                "mixed_insave_pairs": 300,
                "mixed_goku_pairs": 200,
                "mixed_insave_is_control_subset": True,
                "selection_basis": "official remove metadata plus full technical decode",
                "human_review_status": "not_reviewed",
                "datasets": DATASETS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"all={all_path} rows=700 mixed={mixed_path} rows=500 "
        f"control={control_path} rows=500"
    )


if __name__ == "__main__":
    main()
