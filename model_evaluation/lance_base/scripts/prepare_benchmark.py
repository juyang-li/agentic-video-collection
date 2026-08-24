#!/usr/bin/env python3
"""Freeze and optionally download the 15-case OmniEdit Spatial slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "official_metadata" / "spatial_benchmark.json"
MANIFEST = ROOT / "benchmark_15.csv"
SOURCE_ROOT = ROOT / "data" / "official_source"
CONFIG_ROOT = ROOT / "configs"
REPO_ID = "OmniEdit-Bench/OmniEdit-Bench"
REPO_REVISION = "7447b1a57b52585c3c2c5ac7f00feefdf89f9d61"
SEED = 42
CATEGORIES = {
    "object_adding": "Sp_Object_adding_",
    "object_removal": "Sp_Object_removal_",
    "object_replacement": "Sp_Object_replacement_",
    "appearance_change": "Sp_Appearance_change_",
    "clothes_change": "Sp_Clothes_change_",
}
FIELDS = [
    "sample_index",
    "video_id",
    "category",
    "source_repo_path",
    "instruction",
    "dataset_revision",
    "local_source",
    "source_sha256",
    "source_bytes",
    "download_status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_rows() -> list[dict[str, str]]:
    rows = json.loads(METADATA.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    selected: list[dict[str, str]] = []
    for category, prefix in CATEGORIES.items():
        candidates = sorted(
            (row for row in rows if row["video_id"].startswith(prefix)),
            key=lambda row: row["video_id"],
        )
        if len(candidates) != 10:
            raise RuntimeError(f"Expected 10 candidates for {category}, found {len(candidates)}")
        for row in sorted(rng.sample(candidates, 3), key=lambda item: item["video_id"]):
            selected.append(
                {
                    "sample_index": str(len(selected) + 1),
                    "video_id": row["video_id"],
                    "category": category,
                    "source_repo_path": row["source_video"],
                    "instruction": row["editing_instruction"],
                    "dataset_revision": REPO_REVISION,
                    "local_source": "",
                    "source_sha256": "",
                    "source_bytes": "",
                    "download_status": "pending",
                }
            )
    return selected


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["category"] for row in rows)
    expected = Counter({category: 3 for category in CATEGORIES})
    if len(rows) != 15 or counts != expected:
        raise RuntimeError(f"Frozen manifest is invalid: rows={len(rows)}, counts={dict(counts)}")
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def download(rows: list[dict[str, str]]) -> None:
    from huggingface_hub import hf_hub_download

    for row in rows:
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REPO_REVISION,
                filename=row["source_repo_path"],
                local_dir=SOURCE_ROOT,
            )
        ).resolve()
        row["local_source"] = str(path)
        row["source_sha256"] = sha256(path)
        row["source_bytes"] = str(path.stat().st_size)
        row["download_status"] = "downloaded"


def write_configs(rows: list[dict[str, str]]) -> None:
    if any(row["download_status"] != "downloaded" for row in rows):
        return

    def payload(items: list[dict[str, str]]) -> dict[str, dict[str, list[object]]]:
        return {
            f"{int(row['sample_index']):04d}": {
                "interleave_array": [row["instruction"], row["local_source"]],
                "element_dtype_array": ["text", "video"],
                "istarget_in_interleave": [0, 0],
            }
            for row in items
        }

    for name, items in (
        ("benchmark_gate.json", rows[:2]),
        ("benchmark_remaining.json", rows[2:]),
        ("benchmark_all.json", rows),
    ):
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        (CONFIG_ROOT / name).write_text(
            json.dumps(payload(items), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if MANIFEST.exists():
        rows = read_manifest()
        print(f"Using frozen manifest: {MANIFEST}")
    else:
        rows = freeze_rows()
        write_manifest(rows)
        print(f"Created frozen manifest: {MANIFEST}")

    if args.download:
        download(rows)
        write_manifest(rows)
        write_configs(rows)
        print(f"Downloaded {len(rows)} source videos")

    for row in rows:
        print(row["sample_index"], row["category"], row["video_id"])


if __name__ == "__main__":
    main()
