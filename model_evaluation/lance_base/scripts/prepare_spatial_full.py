#!/usr/bin/env python3
"""Freeze, download, and shard the complete OmniEdit Spatial benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "official_metadata" / "spatial_benchmark.json"
SOURCE_ROOT = ROOT / "data" / "official_source"
EVALUATOR_ROOT = ROOT / "runtime" / "omni_edit_evaluator"
CONFIG_ROOT = ROOT / "configs" / "spatial_full_seed42"
MANIFEST = ROOT / "omni_edit_spatial_240.csv"
OUTPUT_PREFIX = "omni_edit_spatial_base_seed42_shard"
REPO_ID = "OmniEdit-Bench/OmniEdit-Bench"
REPO_REVISION = "7447b1a57b52585c3c2c5ac7f00feefdf89f9d61"
EXPECTED_SAMPLES = 240
EXPECTED_PER_CATEGORY = 10
FIELDS = [
    "sample_index",
    "video_id",
    "category",
    "editing_instruction",
    "dataset_revision",
    "official_source_repo_path",
    "official_source_video",
    "official_source_sha256",
    "official_source_bytes",
    "dataset_target_video",
    "dataset_target_status",
    "generated_video",
    "generation_seed",
    "shard_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_official_rows() -> list[dict[str, str]]:
    rows = json.loads(METADATA.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES} Spatial rows, found {len(rows)}")
    video_ids = [row["video_id"] for row in rows]
    source_paths = [row["source_video"] for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("Spatial metadata contains duplicate video_id values")
    if len(set(source_paths)) != len(source_paths):
        raise RuntimeError("Spatial metadata contains duplicate source_video paths")
    categories = Counter(video_id.rsplit("_", 1)[0] for video_id in video_ids)
    if len(categories) != 24 or set(categories.values()) != {EXPECTED_PER_CATEGORY}:
        raise RuntimeError(f"Unexpected Spatial category distribution: {dict(categories)}")
    return rows


def download_sources(rows: list[dict[str, str]]) -> None:
    from huggingface_hub import hf_hub_download

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(rows, 1):
        destination = SOURCE_ROOT / row["source_video"]
        if not destination.is_file():
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REPO_REVISION,
                filename=row["source_video"],
                local_dir=SOURCE_ROOT,
            )
        if not destination.is_file():
            raise RuntimeError(f"Download did not create {destination}")
        print(f"source {position:03d}/{len(rows)} {destination}")

    for filename in ("gemini_eval.py", "eval_configs/prompts.json"):
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REPO_REVISION,
                filename=filename,
                local_dir=EVALUATOR_ROOT,
            )
        )
        print(f"evaluator {path}")


def build_manifest(rows: list[dict[str, str]], shards: int) -> list[dict[str, str]]:
    manifest_rows = []
    for sample_index, row in enumerate(rows, 1):
        source = (SOURCE_ROOT / row["source_video"]).resolve()
        if not source.is_file():
            raise RuntimeError(f"Missing official source video: {source}")
        shard_id = (sample_index - 1) % shards
        generated = (
            ROOT
            / "outputs_raw"
            / f"{OUTPUT_PREFIX}{shard_id:02d}"
            / f"{sample_index:06d}.mp4"
        ).resolve()
        manifest_rows.append(
            {
                "sample_index": str(sample_index),
                "video_id": row["video_id"],
                "category": row["video_id"].rsplit("_", 1)[0].removeprefix("Sp_"),
                "editing_instruction": row["editing_instruction"],
                "dataset_revision": REPO_REVISION,
                "official_source_repo_path": row["source_video"],
                "official_source_video": str(source),
                "official_source_sha256": sha256(source),
                "official_source_bytes": str(source.stat().st_size),
                # OmniEdit-Bench Spatial distributes source videos and edit instructions,
                # but no reference edited/target videos.
                "dataset_target_video": "",
                "dataset_target_status": "not_provided_by_omni_edit",
                "generated_video": str(generated),
                "generation_seed": "42",
                "shard_id": f"{shard_id:02d}",
            }
        )
    return manifest_rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    temporary = MANIFEST.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(MANIFEST)


def write_configs(rows: list[dict[str, str]], shards: int) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for shard_id in range(shards):
        items = [row for row in rows if int(row["shard_id"]) == shard_id]
        payload = {
            f"{int(row['sample_index']):04d}": {
                "interleave_array": [row["editing_instruction"], row["official_source_video"]],
                "element_dtype_array": ["text", "video"],
                "istarget_in_interleave": [0, 0],
            }
            for row in items
        }
        path = CONFIG_ROOT / f"shard_{shard_id:02d}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"config {path} samples={len(payload)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")

    rows = read_official_rows()
    if args.download:
        download_sources(rows)
    manifest_rows = build_manifest(rows, args.shards)
    write_manifest(manifest_rows)
    write_configs(manifest_rows, args.shards)
    print(f"manifest {MANIFEST} samples={len(manifest_rows)} shards={args.shards}")


if __name__ == "__main__":
    main()
