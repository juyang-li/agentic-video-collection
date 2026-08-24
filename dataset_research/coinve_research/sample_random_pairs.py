#!/usr/bin/env python3
"""Draw a reproducible uniform sample from CoinVE-200K metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA = ROOT / "metadata" / "metadata_coinve200k.jsonl"
DEFAULT_OUTPUT = ROOT / "random_pairs.csv"
DEFAULT_SEED = 20260821
DEFAULT_SAMPLE_SIZE = 15
EXPECTED_METADATA_SHA256 = (
    "d1f0b643d2c5cc1cfe855351d2642fb1b6c0cb403f5301002111ac8c2246a5a6"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_and_member(dataset_path: str) -> tuple[str, str]:
    parts = Path(dataset_path).parts
    if len(parts) < 3:
        raise ValueError(f"unexpected CoinVE path: {dataset_path}")
    shard_path = str(Path(parts[0]) / f"{parts[1]}.tar")
    # The official README shows a shard directory, but the released tar files
    # store MP4 members at the archive root (preceded by a PAX header).
    member_path = parts[-1]
    return shard_path, member_path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()

    metadata_sha256 = sha256_file(args.metadata)
    if args.metadata == DEFAULT_METADATA and metadata_sha256 != EXPECTED_METADATA_SHA256:
        raise ValueError(
            "metadata SHA256 changed; inspect the official revision before resampling"
        )

    population = load_jsonl(args.metadata)
    if not 0 < args.sample_size <= len(population):
        raise ValueError("sample-size must be between 1 and the population size")
    indices = random.Random(args.seed).sample(range(len(population)), args.sample_size)

    fields = [
        "sample_rank",
        "metadata_index",
        "sample_id",
        "task",
        "seed",
        "instruction_count",
        "instructions",
        "operations",
        "objects",
        "source_path",
        "target_path",
        "source_shard",
        "target_shard",
        "source_member",
        "target_member",
        "source_file",
        "target_file",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(indices, start=1):
            item = population[index]
            source_path = str(item["source_video_path"])
            target_path = str(item["edited_video_path"])
            source_shard, source_member = shard_and_member(source_path)
            target_shard, target_member = shard_and_member(target_path)
            instructions = list(item["instruction"])
            sample_id = f"coinve-{index:06d}"
            writer.writerow(
                {
                    "sample_rank": rank,
                    "metadata_index": index,
                    "sample_id": sample_id,
                    "task": f"compose-{len(instructions)}",
                    "seed": args.seed,
                    "instruction_count": len(instructions),
                    "instructions": " || ".join(instructions),
                    "operations": " || ".join(item["instruction_operation"]),
                    "objects": " || ".join(item["instruction_object"]),
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_shard": source_shard,
                    "target_shard": target_shard,
                    "source_member": source_member,
                    "target_member": target_member,
                    "source_file": f"examples/{sample_id}_source.mp4",
                    "target_file": f"examples/{sample_id}_target.mp4",
                }
            )

    print(
        f"population={len(population)} sample_size={len(indices)} seed={args.seed} "
        f"metadata_sha256={metadata_sha256}"
    )
    print(f"indices={indices}")


if __name__ == "__main__":
    main()
