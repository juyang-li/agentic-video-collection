#!/usr/bin/env python3
"""Join EgoEditData annotations and uniformly sample 15 replacement edges."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = ROOT / "annotation"
DATA_TABLE = ANNOTATION_DIR / "egoeditdata_data_table.csv"
EDITS_TABLE = ANNOTATION_DIR / "egoeditdata_edits_table.csv"
OUTPUT = ROOT / "random_replace_pairs.csv"
REPO_ID = "liguang0115/EgoEdit"
REVISION = "a9bd3faafe51a581b359acd7c6ad5c432508ff60"
DATA_SHA256 = "81d0f5bb6fe03d49f8a34a922f306c33b356419f3283945380400df8c5f08efe"
EDITS_SHA256 = "9a21c8f12f519377365a4bcc8385bc3139e9554630458cccd76417dd33a89a8c"
SEED = 20260901
SAMPLE_SIZE = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def basename(url: str) -> str:
    return Path(urlparse(url).path).name


def main() -> None:
    if sha256_file(DATA_TABLE) != DATA_SHA256:
        raise ValueError("EgoEditData data-table SHA256 changed")
    if sha256_file(EDITS_TABLE) != EDITS_SHA256:
        raise ValueError("EgoEditData edits-table SHA256 changed")

    with DATA_TABLE.open(encoding="utf-8-sig", newline="") as file:
        data_rows = list(csv.DictReader(file))
    urls = {row["data_id"]: row["data_url"] for row in data_rows}
    if len(urls) != len(data_rows):
        raise ValueError("duplicate data_id in data table")

    population: list[dict[str, str]] = []
    with EDITS_TABLE.open(encoding="utf-8-sig", newline="") as file:
        for metadata_index, row in enumerate(csv.DictReader(file)):
            source_url = urls.get(row["source_data_id"])
            target_url = urls.get(row["target_data_id"])
            if not source_url or not target_url:
                continue
            instruction = row["source_to_target_edit_prompt"].strip()
            if not instruction:
                continue
            population.append(
                {
                    "metadata_index": str(metadata_index),
                    "edit_id": row["data_id"],
                    "source_data_id": row["source_data_id"],
                    "target_data_id": row["target_data_id"],
                    "instruction": instruction,
                    "source_url": source_url,
                    "target_url": target_url,
                    "source_member_basename": basename(source_url),
                    "target_member_basename": basename(target_url),
                }
            )
    if len(population) < SAMPLE_SIZE:
        raise RuntimeError(f"population too small: {len(population)}")

    selected = random.Random(SEED).sample(range(len(population)), SAMPLE_SIZE)
    rows: list[dict[str, object]] = []
    for rank, population_index in enumerate(selected, 1):
        item = population[population_index]
        sample_id = f"egoedit-{item['edit_id'][:10]}"
        rows.append(
            {
                "sample_rank": rank,
                "metadata_index": item["metadata_index"],
                "population_index": population_index,
                "sample_id": sample_id,
                "task": "replace",
                "seed": SEED,
                "repo_id": REPO_ID,
                "revision": REVISION,
                "subpopulation": "all_joined_egoeditdata_edges",
                "subpopulation_size": len(population),
                "edit_id": item["edit_id"],
                "source_data_id": item["source_data_id"],
                "target_data_id": item["target_data_id"],
                "instruction": item["instruction"],
                "source_url": item["source_url"],
                "target_url": item["target_url"],
                "source_member_basename": item["source_member_basename"],
                "target_member_basename": item["target_member_basename"],
                "source_file": f"examples/{sample_id}_source.mp4",
                "target_file": f"examples/{sample_id}_target.mp4",
            }
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scene_keys = {
        "_".join(str(row["source_data_id"]).split("_")[:2]) for row in rows
    }
    unique_members = {
        str(row[field])
        for row in rows
        for field in ("source_member_basename", "target_member_basename")
    }
    print(
        f"data_rows={len(data_rows)} joined_population={len(population)} "
        f"sample_size={SAMPLE_SIZE} seed={SEED} indices={selected} "
        f"scene_keys={len(scene_keys)} unique_members={len(unique_members)}"
    )


if __name__ == "__main__":
    main()
