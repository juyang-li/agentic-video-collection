#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


REPO_ID = "JianhaoZeng/Eevee"
REVISION = "6088d199f5046d87d193ceedf39e2931eff4bd95"
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

    population = []
    specs = (
        ("dresses", "dresses"),
        ("lower_body", "lower"),
        ("upper_body", "upper"),
    )
    for category, filename_prefix in specs:
        for split in ("train", "test"):
            path = ROOT / "metadata" / f"{filename_prefix}_{split}.csv"
            for split_index, row in enumerate(csv.DictReader(path.open())):
                sample_dir = row["Person"]
                prefix = f"{category}/{sample_dir}"
                population.append(
                    {
                        "split": split,
                        "split_index": split_index,
                        "category": category,
                        "sample_dir": sample_dir,
                        "official_garment_id": row["Garment"],
                        "garment_member": f"{prefix}/garment.png",
                        "person_member": f"{prefix}/person.png",
                        "agnostic_member": f"{prefix}/video_0_agnostic_sam.mp4",
                        "target_member": f"{prefix}/video_0.mp4",
                        "closeup_member": f"{prefix}/video_1.mp4",
                        "caption_member": f"{prefix}/garment_caption.txt",
                    }
                )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(population)), args.count)
    selected = [population[index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_rank", "population_index", "split", "split_index", "sample_id",
        "task", "seed", "population_size", "repo_id", "revision", "instruction",
        "category", "sample_dir", "official_garment_id", "garment_member",
        "person_member", "agnostic_member", "target_member", "closeup_member",
        "caption_member",
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
                    "sample_id": f"eevee-{rank:04d}",
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
        members.extend(
            (
                row["agnostic_member"], row["garment_member"], row["person_member"],
                row["target_member"], row["closeup_member"], row["caption_member"],
            )
        )
    args.extract_output.write_text("\n".join(members) + "\n")


if __name__ == "__main__":
    main()
