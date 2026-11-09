#!/usr/bin/env python3
"""Freeze generation manifests/jobs for an agreed post-Short evaluation stage."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from decord import VideoReader, cpu
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
TREE = DATA / "dataset_tree.json"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"
UPSTREAM_REVISION = "2dcf8900ee65197f43854c76f61c66a32bb8dbfc"
STAGES = {
    "long_subject": {
        "scope": "long_subject_all_43", "selectors": [("long", "subject_editing")], "count": 43,
        "subcategories": {"add new subject": 13, "remove existing subject": 17, "replace existing subject": 13},
    },
    "attribute_all": {
        "scope": "attribute_all_94", "selectors": [("short", "attribute_editing"), ("long", "attribute_editing")], "count": 94,
        "subcategories": {"color adjustment": 45, "subject scaling": 27, "position change": 22},
    },
}
PROJECT = ROOT.parents[1]
FINETUNE = PROJECT / "lance_general_edit_finetune" / "3_lance_lora_finetuning"
RUNTIME = FINETUNE / "runtime" / "lance"
MODEL = RUNTIME / "downloads" / "Lance_3B_Video"
ADAPTER = FINETUNE / "runs" / "lance_4k_r64_seed42_e2_v1" / "checkpoints" / "0000677"
TOOLS = PROJECT / "model_evaluation" / "lance_tools"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_rows(stage: dict) -> list[dict]:
    rows = []
    for subset, category in stage["selectors"]:
        prompts = json.loads((DATA / f"ivebenchdb_prompt_{subset}.json").read_text())
        rows.extend({**row, "subset": subset} for row in prompts if row["category"] == category)
    rows.sort(key=lambda row: (0 if row["subset"] == "short" else 1, int(row["id"])))
    if len(rows) != stage["count"] or dict(Counter(row["subcategory"] for row in rows)) != stage["subcategories"]:
        raise ValueError("official scope counts changed")
    return rows


def long_model_input(
    official: Path, official_sha256: str, temporary_suffix: str = ".part.mp4"
) -> tuple[Path, list[int]]:
    """Normalize a Long source to the agreed exact 121-frame Lance input."""
    destination = DATA / "model_inputs" / "long_121" / official.name
    provenance_path = destination.with_suffix(".provenance.json")
    original = VideoReader(str(official), ctx=cpu(0), num_threads=2)
    indices = np.linspace(0, len(original) - 1, 121, dtype=int).tolist()
    expected = {
        "policy": "121 uniform full-clip frames encoded at 12 fps for exact Lance source/target alignment",
        "official_path": str(official.resolve()), "official_sha256": official_sha256,
        "official_frames": len(original), "official_indices": indices,
    }
    reusable = False
    if destination.exists() and provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text())
            reader = VideoReader(str(destination), ctx=cpu(0), num_threads=2)
            reusable = (all(provenance.get(key) == value for key, value in expected.items())
                        and provenance.get("model_input_sha256") == digest(destination)
                        and len(reader) == 121 and abs(float(reader.get_avg_fps()) - 12.0) < 1e-4)
        except Exception:
            reusable = False
    if not reusable:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not temporary_suffix.startswith(".") or not temporary_suffix.endswith(".mp4"):
            raise ValueError(f"invalid temporary suffix: {temporary_suffix}")
        temporary = destination.with_suffix(temporary_suffix)
        if temporary.exists():
            temporary.unlink()
        first = original[0]
        size = (int(first.shape[1]), int(first.shape[0]))
        if size[0] % 2 or size[1] % 2:
            raise ValueError(f"Long official source has odd dimensions: {official} {size}")
        writer = imageio_ffmpeg.write_frames(
            str(temporary), size, fps=12, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-crf", "1", "-preset", "fast", "-movflags", "+faststart", "-threads", "4"],
        )
        writer.send(None)
        try:
            for index in indices:
                writer.send(original[index].asnumpy())
        finally:
            writer.close()
        reader = VideoReader(str(temporary), ctx=cpu(0), num_threads=2)
        if len(reader) != 121 or abs(float(reader.get_avg_fps()) - 12.0) >= 1e-4:
            raise ValueError(f"failed to construct exact 121-frame input: {official}")
        for offset in range(0, 121, 16):
            reader.get_batch(list(range(offset, min(offset + 16, 121))))
        temporary.replace(destination)
        write_json(provenance_path, {**expected, "model_input_path": str(destination.resolve()),
                                     "model_input_sha256": digest(destination)})
    return destination, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    args = parser.parse_args()
    stage = STAGES[args.stage]
    artifact = ROOT / "artifacts" / args.stage
    tree = {entry["path"]: entry for entry in json.loads(TREE.read_text()) if entry.get("type") == "file"}
    manifest = []
    for generation_id, row in enumerate(load_rows(stage), 1):
        relative = f"ivebenchdb_mp4_{row['subset']}/{row['src_video_name']}"
        entry = tree[relative]
        source = DATA / row["subset"] / row["src_video_name"]
        if not source.is_file() or source.stat().st_size != int(entry["size"]) or digest(source) != entry["lfs"]["oid"]:
            raise ValueError(f"run prefetch_next_inputs.py first or repair invalid input: {source}")
        official_reader = VideoReader(str(source), ctx=cpu(0), num_threads=2)
        official_fps = float(official_reader.get_avg_fps())
        official_indices = None
        model_input = source
        if row["subset"] == "long":
            model_input, official_indices = long_model_input(source, entry["lfs"]["oid"])
        reader = VideoReader(str(model_input), ctx=cpu(0), num_threads=2)
        fps = float(reader.get_avg_fps())
        rounded_fps = int(round(fps)) if fps > 0 else 24
        duration = min(len(reader) / rounded_fps, 10.1)
        frames = int(round(duration * 12))
        frames = frames // 4 * 4 + 1 if frames % 4 else frames // 4 * 4 + 1 - 4
        if not 1 <= frames <= 121:
            raise ValueError(f"invalid Lance frame count for {source}: {frames}")
        indices = np.linspace(0, len(reader) - 1, frames, dtype=int).tolist()
        first = reader[0]
        manifest.append({
            "sample_id": Path(row["src_video_name"]).stem, "generation_id": generation_id,
            "official_id": int(row["id"]), "subset": row["subset"], "category": row["category"],
            "subcategory": row["subcategory"], "source_prompt": row["source_prompt"],
            "edit_prompt": row["edit_prompt"], "target_prompt": row["target_prompt"],
            "target_phrase": row["target_phrase"], "target_span": row["target_span"],
            "source_path": str(model_input.resolve()), "source_bytes": model_input.stat().st_size,
            "source_sha256": digest(model_input), "source_frames": len(reader),
            "source_fps": round(fps, 6), "source_height": int(first.shape[0]), "source_width": int(first.shape[1]),
            "lance_sampled_frames": frames, "lance_source_indices": json.dumps(indices, separators=(",", ":")),
            "official_source_path": str(source.resolve()), "official_source_sha256": entry["lfs"]["oid"],
            "official_source_frames": len(official_reader), "official_source_fps": round(official_fps, 6),
            "official_to_model_input_indices": json.dumps(official_indices, separators=(",", ":")) if official_indices else "",
        })
    manifest_path = artifact / "manifest.csv"
    artifact.mkdir(parents=True, exist_ok=True)
    with manifest_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    manifest_path.with_suffix(".csv.tmp").replace(manifest_path)

    def config_item(row: dict) -> dict:
        return {"element_dtype_array": ["text", "video"], "interleave_array": [row["edit_prompt"], row["source_path"]],
                "istarget_in_interleave": [0, 0]}

    def make_job(kind: str, slot: int, config: Path, name: str, count: int) -> dict:
        if kind == "base":
            command = ["env", "LANCE_MAX_DURATION=10.1", "bash", str((TOOLS / "run_video_edit.sh").resolve()),
                       str(artifact.resolve()), str(RUNTIME.resolve()), str(MODEL.resolve()), str(config.resolve()), name]
        else:
            command = ["env", "LANCE_MAX_DURATION=10.1", "bash", str((FINETUNE / "scripts/run_lora_video_edit.sh").resolve()),
                       str(artifact.resolve()), str(RUNTIME.resolve()), str(MODEL.resolve()), str(ADAPTER.resolve()),
                       str(config.resolve()), name]
        return {"model_kind": kind, "gpu_slot": slot, "config_path": str(config.resolve()),
                "run_name": name, "sample_count": count, "command": command}

    configs = artifact / "configs"
    groups = []
    smoke = []
    for slot, kind in enumerate(("base", "lora_step677")):
        name = f"smoke_{kind}"; path = configs / f"{name}.json"
        write_json(path, {f"{manifest[0]['generation_id']:06d}": config_item(manifest[0])})
        smoke.append(make_job(kind, slot, path, name, 1))
    groups.append({"name": "matched_smoke", "jobs": smoke})
    for kind in ("base", "lora_step677"):
        jobs = []
        for slot in range(4):
            shard = manifest[slot::4]; name = f"{args.stage}_{kind}_shard{slot:02d}"; path = configs / f"{name}.json"
            write_json(path, {f"{row['generation_id']:06d}": config_item(row) for row in shard})
            jobs.append(make_job(kind, slot, path, name, len(shard)))
        groups.append({"name": f"full_{kind}", "jobs": jobs})
    jobs_path = artifact / "generation_jobs.json"
    write_json(jobs_path, {
        "benchmark": "IVEBench", "dataset_id": "Coraxor/IVEBench", "dataset_revision": DATASET_REVISION,
        "upstream_code_revision": UPSTREAM_REVISION, "scope": stage["scope"], "generation_seed": 42,
        "inference": {"resolution": "video_480p", "save_fps": 12, "steps": 30, "timestep_shift": 3.5,
                      "cfg_text_scale": 4.0, "max_num_frames": 121, "max_duration_seconds": 10.1,
                      "long_input_policy": "exact 121 uniform full-clip frames at 12 fps",
                      "prompt_enhancement": False, "dtype": "bf16"},
        "base_model_path": str(MODEL.resolve()),
        "base_model_sha256": "7f0550e1d1511b29a4740a67c1e18e176302a4ecb3177c8a5850ff5fe6447c25",
        "adapter_path": str(ADAPTER.resolve()), "adapter_sha256": digest(ADAPTER / "adapter.safetensors"),
        "manifest_sha256": digest(manifest_path), "groups": groups,
    })
    summary = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
               "stage": args.stage, "scope": stage["scope"], "samples": len(manifest),
               "subcategories": dict(Counter(row["subcategory"] for row in manifest)),
               "frame_counts": dict(Counter(row["lance_sampled_frames"] for row in manifest)),
               "manifest_sha256": digest(manifest_path), "generation_jobs_sha256": digest(jobs_path)}
    write_json(artifact / "preparation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
