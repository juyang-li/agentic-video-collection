#!/usr/bin/env python3
"""Sample 15 visual-only reverse-add pairs from the scanned InsAVE shard."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
POPULATION_PATH = ROOT / "shard_population.csv"
OUTPUT_PATH = ROOT / "random_add_pairs.csv"
SEED = 20260827
SAMPLE_SIZE = 15
AUDIO_WORDS = re.compile(
    r"\b(audio|music|saying|sound|speak(?:s|ing)?|speech|spoken|talking|voice)\b",
    re.IGNORECASE,
)


def eligible(instruction: str) -> bool:
    normalized = instruction.strip()
    return (
        normalized.lower().startswith("add ")
        and "<S>" not in normalized
        and "<E>" not in normalized
        and AUDIO_WORDS.search(normalized) is None
    )


def main() -> None:
    with POPULATION_PATH.open(encoding="utf-8", newline="") as file:
        population = list(csv.DictReader(file))
    candidates = [row for row in population if eligible(row["instruction_reverse"])]
    selected = random.Random(SEED).sample(range(len(candidates)), SAMPLE_SIZE)
    rows: list[dict[str, object]] = []
    for rank, candidate_index in enumerate(selected, 1):
        item = candidates[candidate_index]
        metadata_index = int(item["metadata_index"])
        sample_id = f"insave-add-{metadata_index:05d}"
        rows.append(
            {
                "sample_rank": rank,
                "metadata_index": metadata_index,
                "subpopulation_index": candidate_index,
                "sample_id": sample_id,
                "official_category": "add_and_remove_reverse",
                "task": "add",
                "seed": SEED,
                "repo_id": "suimu/InsAVE-80K",
                "revision": "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289",
                "shard": "part_01.tar",
                "subpopulation": "part01_reverse_add_without_audio_terms",
                "subpopulation_size": len(candidates),
                "instruction": item["instruction_reverse"],
                "source_member": item["target_member"],
                "target_member": item["source_member"],
                "source_file": f"examples_add/{sample_id}_source.mp4",
                "target_file": f"examples_add/{sample_id}_target.mp4",
            }
        )
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"population={len(population)} eligible={len(candidates)} "
        f"sample_size={SAMPLE_SIZE} seed={SEED} indices={selected}"
    )


if __name__ == "__main__":
    main()
