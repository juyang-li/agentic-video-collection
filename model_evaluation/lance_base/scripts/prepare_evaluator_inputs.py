#!/usr/bin/env python3
"""Link generated videos into the original OmniEdit evaluator directory layout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "omni_edit_spatial_240.csv"
METADATA = ROOT / "data" / "official_metadata" / "spatial_benchmark.json"
SOURCE_SPATIAL = ROOT / "data" / "official_source" / "Spatial"
BENCHMARK_ROOT = ROOT / "evaluator_workspace"
MODEL_NAME = "lance_base_seed42"


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Refusing to replace non-matching path: {link}")
    link.symlink_to(target, target_is_directory=True)


def link_output(source: Path, destination: Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise RuntimeError(f"Refusing to replace non-matching output: {destination}")
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if len(rows) != 240:
        raise RuntimeError(f"Expected 240 manifest rows, found {len(rows)}")

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    replace_symlink(BENCHMARK_ROOT / "Spatial", SOURCE_SPATIAL)
    shutil.copy2(METADATA, BENCHMARK_ROOT / "spatial_benchmark.json")
    model_dir = BENCHMARK_ROOT / "eval_720p_10s" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing = []
    mapping = []
    for row in rows:
        source = Path(row["generated_video"])
        destination = model_dir / f"{row['video_id']}.mp4"
        if source.is_file():
            link_output(source, destination)
            linked += 1
            status = "linked"
        else:
            missing.append(row["sample_index"])
            status = "missing"
        mapping.append(
            {
                "sample_index": int(row["sample_index"]),
                "video_id": row["video_id"],
                "official_source_video": row["official_source_video"],
                "dataset_target_video": None,
                "generated_video": str(source),
                "evaluator_input": str(destination),
                "status": status,
            }
        )

    mapping_path = BENCHMARK_ROOT / f"{MODEL_NAME}_mapping.json"
    temporary = mapping_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(mapping_path)
    print(f"linked={linked} missing={len(missing)} model_dir={model_dir}")
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing generated sample indexes: {','.join(missing)}")


if __name__ == "__main__":
    main()
