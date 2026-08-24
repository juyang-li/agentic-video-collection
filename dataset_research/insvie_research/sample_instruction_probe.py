#!/usr/bin/env python3
"""Reproduce the task-prefix metadata probe for InsViE-1M."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "train_insvie_align.csv"
OUTPUT = ROOT / "random_instruction_probe.csv"
TASK_SEEDS = {"add": 20260828, "remove": 20260829, "replace": 20260830}
SAMPLE_SIZE = 15


def family(video: str) -> str:
    lowered = video.lower()
    for prefix in (
        "pexel-static",
        "pexel",
        "instructp2p",
        "magicbrush",
        "openvid-static",
        "openvid",
    ):
        if lowered.startswith(prefix):
            return prefix
    return lowered.split("-", 1)[0]


def task_for(instruction: str) -> str | None:
    first = (instruction.strip().lower().split(maxsplit=1) or [""])[0].rstrip(":_")
    if first in {"add", "remove"}:
        return first
    if first in {"replace", "swap"}:
        return "replace"
    return None


def main() -> None:
    groups: dict[str, list[dict[str, object]]] = {
        task: [] for task in TASK_SEEDS
    }
    family_counts: dict[str, Counter[str]] = {
        task: Counter() for task in TASK_SEEDS
    }
    with METADATA.open(encoding="utf-8-sig", newline="") as file:
        for metadata_index, row in enumerate(csv.DictReader(file)):
            task = task_for(row["instruct"])
            if task is None:
                continue
            source_family = family(row["video"])
            family_counts[task][source_family] += 1
            groups[task].append(
                {
                    "metadata_index": metadata_index,
                    "video": row["video"],
                    "instruction": row["instruct"],
                    "source_family": source_family,
                }
            )
    samples: list[dict[str, object]] = []
    for task, seed in TASK_SEEDS.items():
        selected = random.Random(seed).sample(range(len(groups[task])), SAMPLE_SIZE)
        for rank, population_index in enumerate(selected, 1):
            item = groups[task][population_index]
            samples.append(
                {
                    "task": task,
                    "sample_rank": rank,
                    "seed": seed,
                    "population_index": population_index,
                    "population_size": len(groups[task]),
                    **item,
                }
            )
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    for task in TASK_SEEDS:
        print(
            f"{task} population={len(groups[task])} "
            f"families={dict(family_counts[task])}"
        )


if __name__ == "__main__":
    main()
