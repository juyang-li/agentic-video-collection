#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


REPO_ID = "alibaba-yuanjing-aigclab/ViViD"
REVISION = "a185550ba9bd48461a75bd6d66381fcc15bc837a"
ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "random_pairs.csv")
    parser.add_argument(
        "--extract-output", type=Path, default=ROOT / "extract_members.txt"
    )
    args = parser.parse_args()

    metadata_root = ROOT / "metadata" / "ViViD"
    population = []
    for category in ("dresses", "lower_body", "upper_body"):
        for split in ("train", "test"):
            path = metadata_root / category / f"{split}_pairs.txt"
            for split_index, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                video, garment = line.split()
                prefix = f"ViViD/{category}"
                population.append(
                    {
                        "split": split,
                        "split_index": split_index,
                        "category": category,
                        "video": video,
                        "garment": garment,
                        "agnostic_member": f"{prefix}/agnostic/{video}",
                        "target_member": f"{prefix}/videos/{video}",
                        "garment_member": f"{prefix}/images/{garment}",
                    }
                )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(population)), args.count)
    selected = [population[index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_rank", "population_index", "split", "split_index", "sample_id",
        "task", "seed", "population_size", "repo_id", "revision", "instruction",
        "category", "video", "garment", "agnostic_member", "target_member",
        "garment_member",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (index, row) in enumerate(zip(indices, selected), 1):
            writer.writerow(
                {
                    "sample_rank": rank,
                    "population_index": index,
                    **row,
                    "sample_id": f"vivid-{rank:04d}",
                    "task": f"try-on-{row['category']}",
                    "seed": args.seed,
                    "population_size": len(population),
                    "repo_id": REPO_ID,
                    "revision": REVISION,
                    "instruction": "Reconstruct the person's clothing from the referenced garment.",
                }
            )

    members = []
    for row in selected:
        members.extend((row["agnostic_member"], row["garment_member"], row["target_member"]))
    args.extract_output.write_text("\n".join(members) + "\n")


if __name__ == "__main__":
    main()
