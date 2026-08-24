#!/usr/bin/env python3
"""Select one case-weighted Señorita object-swap shard, then sample 15 pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA = Path("/tmp/senorita_probe/updated_data_obj_swap_videos.json")
OUTPUT_PATH = ROOT / "random_pairs.csv"
REPO_ID = "SENORITADATASET/Senorita"
REVISION = "6e2eac59caf6321353d2712e31815fda2ff573cb"
EXPECTED_METADATA_SHA256 = (
    "c4d46c8fd44ddd7b04f16f1eedc1fd83720a55ee9732b7239e17622b860527d9"
)
SEED = 20260822
SAMPLE_SIZE = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_id(path: str) -> int:
    parts = Path(path.removeprefix("./")).parts
    if len(parts) != 3 or parts[0] != "obj_swap_videos":
        raise ValueError(f"unexpected video path: {path}")
    return int(parts[1])


def archive_member(path: str) -> str:
    parts = Path(path.removeprefix("./")).parts
    return f"{parts[1]}/{parts[2]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    metadata_hash = sha256_file(args.metadata)
    if metadata_hash != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata SHA256 changed; inspect before sampling")
    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(rows) != 177_336:
        raise ValueError(f"unexpected population size: {len(rows)}")

    rng = random.Random(SEED)
    anchor_index = rng.randrange(len(rows))
    selected_shard = shard_id(rows[anchor_index]["source_video_path"])
    shard_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if shard_id(row["source_video_path"]) == selected_shard
    ]
    local_indices = rng.sample(range(len(shard_rows)), SAMPLE_SIZE)

    output_rows: list[dict[str, object]] = []
    for rank, local_index in enumerate(local_indices, start=1):
        metadata_index, row = shard_rows[local_index]
        source_member = archive_member(row["source_video_path"])
        target_member = archive_member(row["target_video_path"])
        stem = Path(source_member).stem
        sample_id = f"senorita-{rank:04d}"
        output_rows.append(
            {
                "sample_rank": rank,
                "metadata_index": metadata_index,
                "shard_index": local_index,
                "sample_id": sample_id,
                "official_category": "object_swap",
                "task": "replace",
                "seed": SEED,
                "anchor_index": anchor_index,
                "repo_id": REPO_ID,
                "revision": REVISION,
                "shard": f"obj_swap_upload/{selected_shard}.tar.gz",
                "subpopulation": f"object_swap_shard_{selected_shard}",
                "subpopulation_size": len(shard_rows),
                "instruction": row["enhanced_instruction"].strip(),
                "original_instruction": row["instruction"].strip(),
                "source_member": source_member,
                "target_member": target_member,
                "source_file": f"examples/{sample_id}_source.mp4",
                "target_file": f"examples/{sample_id}_target.mp4",
                "official_stem": stem,
            }
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"metadata_sha256={metadata_hash}")
    print(
        f"population={len(rows)} anchor_index={anchor_index} "
        f"selected_shard={selected_shard} shard_population={len(shard_rows)} "
        f"sample_size={SAMPLE_SIZE} seed={SEED} local_indices={local_indices}"
    )


if __name__ == "__main__":
    main()
