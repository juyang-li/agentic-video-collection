#!/usr/bin/env python3
"""Map all 240 official Spatial rows to source/generated-video judge tasks."""

from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lance_tools"))
from judge_spatial_openrouter import sha256_file, write_json


def prepare() -> Path:
    manifest = ROOT / "omni_edit_spatial_240.csv"
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 240 and len({r["video_id"] for r in rows}) == 240
    assert set(Counter(r["category"] for r in rows).values()) == {10}
    official = {r["video_id"]: r for r in json.loads((ROOT / "data/official_metadata/spatial_benchmark.json").read_text())}
    if set(official) != {r["video_id"] for r in rows}:
        raise ValueError("manifest IDs differ from official Spatial metadata")
    tasks = []
    for row in rows:
        assert row["generation_seed"] == "42" and not row["dataset_target_video"]
        reference = official[row["video_id"]]
        if (row["editing_instruction"] != reference["editing_instruction"]
                or row["official_source_repo_path"] != reference["source_video"]
                or row["dataset_revision"] != "7447b1a57b52585c3c2c5ac7f00feefdf89f9d61"):
            raise ValueError(f"official instruction/source/revision mismatch: {row['video_id']}")
        source = Path(row["official_source_video"])
        if source.resolve() != (ROOT / "data/official_source" / reference["source_video"]).resolve():
            raise ValueError("source is outside the expected official benchmark path")
        expected_generated = ROOT / "outputs_raw/omni_edit_spatial_base_seed42_shard00" / f"{int(row['sample_index']):06d}.mp4"
        if Path(row["generated_video"]).resolve() != expected_generated.resolve():
            raise ValueError("unexpected generated-video path")
        if sha256_file(source) != row["official_source_sha256"]:
            raise ValueError(f"source checksum mismatch: {row['video_id']}")
        tasks.append({
            "stage": "omni_edit_spatial", "checkpoint_step": "base",
            "sample_id": row["video_id"], "sample_index": int(row["sample_index"]),
            "category": row["category"], "instruction": row["editing_instruction"],
            "source_path": str(source), "source_sha256": row["official_source_sha256"],
            "generated_video": row["generated_video"], "generation_seed": 42,
            "dataset_revision": row["dataset_revision"],
        })
    path = ROOT / "evaluator_workspace/judge_tasks.json"
    write_json(path, tasks)
    print(f"prepared {len(tasks)} tasks: {path}")
    return path


if __name__ == "__main__":
    prepare()
