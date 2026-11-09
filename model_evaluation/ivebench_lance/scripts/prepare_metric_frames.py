#!/usr/bin/env python3
"""Build exactly aligned frame folders for the official IVEBench local metrics."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from decord import VideoReader, cpu
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
MANIFEST = ARTIFACTS / "short_subject_manifest.csv"
OUTPUTS = ARTIFACTS / "outputs_raw"
FRAMES = ARTIFACTS / "metric_frames"
OUTPUT_PREFIX = "short_subject"
ID_COLUMN = "official_id"
EXPECTED_SAMPLES = 91
STAGES = {
    "short_subject": ("short_subject_manifest.csv", "short_subject", "official_id", 91),
    "long_subject": ("manifest.csv", "long_subject", "generation_id", 43),
    "attribute_all": ("manifest.csv", "attribute_all", "generation_id", 94),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def generated_path(model_kind: str, output_id: int) -> Path:
    matches = list(OUTPUTS.glob(f"{OUTPUT_PREFIX}_{model_kind}_shard*/{output_id:06d}.mp4"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {model_kind} output for {output_id}, got {matches}")
    return matches[0].resolve()


def crop_to_aspect(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    source_ratio = width / height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        crop_width = max(1, round(height * target_ratio))
        left = (width - crop_width) // 2
        return frame[:, left:left + crop_width]
    if source_ratio < target_ratio:
        crop_height = max(1, round(width / target_ratio))
        top = (height - crop_height) // 2
        return frame[top:top + crop_height, :]
    return frame


def frame_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame_*.png"))


def validate_directory(directory: Path, expected: dict) -> bool:
    marker = directory / "_complete.json"
    if not marker.exists():
        return False
    try:
        if json.loads(marker.read_text()) != expected:
            return False
        files = frame_files(directory)
        if len(files) != expected["frames"]:
            return False
        for path in (files[0], files[-1]):
            with Image.open(path) as image:
                if image.size != (expected["width"], expected["height"]):
                    return False
                image.verify()
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def publish_frames(destination: Path, images: list[np.ndarray], expected: dict) -> None:
    if validate_directory(destination, expected):
        return
    staging = FRAMES / ".staging" / f"{destination.parent.name}_{destination.name}_{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for index, image in enumerate(images):
            Image.fromarray(image).save(staging / f"frame_{index:05d}.png", compress_level=1)
        write_json(staging / "_complete.json", expected)
        if not validate_directory(staging, expected):
            raise ValueError(f"failed to validate staged frames for {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            quarantine = FRAMES / ".replaced" / f"{destination.parent.name}_{destination.name}_{uuid.uuid4().hex}"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(quarantine)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_row(row: dict[str, str]) -> dict:
    official_id = int(row["official_id"])
    output_id = int(row[ID_COLUMN])
    sample_id = row["sample_id"]
    expected_frames = int(row["lance_sampled_frames"])
    source_indices = json.loads(row["lance_source_indices"])
    if len(source_indices) != expected_frames:
        raise ValueError(f"source index mismatch for {sample_id}")

    paths = {
        "source": Path(row["source_path"]).resolve(),
        "base": generated_path("base", output_id),
        "lora_step677": generated_path("lora_step677", output_id),
    }
    readers = {kind: VideoReader(str(path), ctx=cpu(0), num_threads=2) for kind, path in paths.items()}
    for kind in ("base", "lora_step677"):
        if len(readers[kind]) != expected_frames:
            raise ValueError(f"{kind} frame mismatch for {sample_id}: {len(readers[kind])} != {expected_frames}")
        if abs(float(readers[kind].get_avg_fps()) - 12.0) > 1e-4:
            raise ValueError(f"{kind} FPS mismatch for {sample_id}")
    target_height, target_width = map(int, readers["base"][0].shape[:2])
    if tuple(readers["lora_step677"][0].shape[:2]) != (target_height, target_width):
        raise ValueError(f"Base/LoRA resolution mismatch for {sample_id}")

    hashes = {kind: sha256_file(path) for kind, path in paths.items()}
    common = {
        "sample_id": sample_id, "frames": expected_frames, "fps": 12.0,
        "width": target_width, "height": target_height,
        "sampling": "Lance source indices frozen in the stage manifest",
    }
    expectations = {
        kind: {**common, "kind": kind, "input_path": str(paths[kind]), "input_sha256": hashes[kind]}
        for kind in paths
    }
    destinations = {kind: FRAMES / kind / sample_id for kind in paths}
    missing = [kind for kind in paths if not validate_directory(destinations[kind], expectations[kind])]
    if missing:
        source_batch = readers["source"].get_batch(source_indices).asnumpy() if "source" in missing else None
        for kind in missing:
            if kind == "source":
                images = []
                assert source_batch is not None
                for frame in source_batch:
                    cropped = crop_to_aspect(frame, target_width, target_height)
                    images.append(np.asarray(Image.fromarray(cropped).resize(
                        (target_width, target_height), Image.Resampling.LANCZOS)))
            else:
                images = readers[kind].get_batch(list(range(expected_frames))).asnumpy()
            publish_frames(destinations[kind], list(images), expectations[kind])

    return {
        "sample_id": sample_id, "official_id": official_id, "output_id": output_id,
        "frames": expected_frames,
        "width": target_width, "height": target_height, "input_sha256": hashes,
    }


def main() -> None:
    global ARTIFACTS, MANIFEST, OUTPUTS, FRAMES, OUTPUT_PREFIX, ID_COLUMN, EXPECTED_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="short_subject")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    manifest_name, OUTPUT_PREFIX, ID_COLUMN, EXPECTED_SAMPLES = STAGES[args.stage]
    ARTIFACTS = ROOT / "artifacts" / args.stage
    MANIFEST = ARTIFACTS / manifest_name
    OUTPUTS = ARTIFACTS / "outputs_raw"
    FRAMES = ARTIFACTS / "metric_frames"
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} {args.stage} rows, got {len(rows)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare_row, rows))
    results.sort(key=lambda item: item["output_id"])

    info = [{
        "id": int(row[ID_COLUMN]), "src_video_name": row["sample_id"] + ".mp4",
        "category": row["category"], "subcategory": row["subcategory"],
        "source_prompt": row["source_prompt"], "edit_prompt": row["edit_prompt"],
        "target_prompt": row["target_prompt"], "target_span": row["target_span"],
        "target_phrase": row["target_phrase"],
    } for row in rows]
    info.sort(key=lambda item: item["id"])
    info_path = ARTIFACTS / ("short_subject_metric_info.json" if args.stage == "short_subject" else "metric_info.json")
    write_json(info_path, info)
    summary = {
        "status": "complete", "stage": args.stage, "samples": len(results),
        "total_frames_per_kind": sum(item["frames"] for item in results),
        "kinds": ["source", "base", "lora_step677"],
        "policy": "all generated frames; source uses exact frozen Lance indices; source center-cropped and resized to generated dimensions; PNG",
        "info_json": str(info_path.resolve()), "info_sha256": sha256_file(info_path),
        "results": results,
    }
    write_json(ARTIFACTS / "metric_frame_preparation.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
