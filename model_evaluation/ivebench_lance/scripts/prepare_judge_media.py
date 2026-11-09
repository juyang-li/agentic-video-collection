#!/usr/bin/env python3
"""Create temporally paired IVEBench judge media and frozen Gemini task manifests."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from decord import VideoReader, cpu
import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
MANIFEST = ARTIFACTS / "short_subject_manifest.csv"
OUTPUTS = ARTIFACTS / "outputs_raw"
MEDIA = ARTIFACTS / "judge_media"
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
    pattern = f"{OUTPUT_PREFIX}_{model_kind}_shard*/{output_id:06d}.mp4"
    matches = list(OUTPUTS.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one output for {model_kind}/{output_id}, got {matches}")
    return matches[0].resolve()


def crop_to_aspect(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    source_ratio = width / height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        crop_width = max(1, round(height * target_ratio))
        left = (width - crop_width) // 2
        frame = frame[:, left:left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = max(1, round(width / target_ratio))
        top = (height - crop_height) // 2
        frame = frame[top:top + crop_height, :]
    return frame


def encode_frames(destination: Path, frames: list[np.ndarray], size: tuple[int, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = imageio_ffmpeg.write_frames(
        str(temporary), size, fps=5, codec="libx264", pix_fmt_out="yuv420p",
        macro_block_size=2,
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart", "-threads", "2"],
    )
    writer.send(None)
    try:
        for frame in frames:
            image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
            writer.send(np.asarray(image))
    finally:
        writer.close()
    temporary.replace(destination)


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if not len(reader):
        raise ValueError(f"empty video: {path}")
    for offset in range(0, len(reader), 16):
        reader.get_batch(list(range(offset, min(offset + 16, len(reader)))))
    first = reader[0]
    return {
        "path": str(path.resolve()), "frames": len(reader), "fps": round(float(reader.get_avg_fps()), 6),
        "width": int(first.shape[1]), "height": int(first.shape[0]), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_row(row: dict) -> dict:
    official_id = int(row["official_id"])
    output_id = int(row[ID_COLUMN])
    sample_id = row["sample_id"]
    source_path = Path(row["source_path"])
    source_reader = VideoReader(str(source_path), ctx=cpu(0), num_threads=2)
    base_path = generated_path("base", output_id)
    lora_path = generated_path("lora_step677", output_id)
    base_reader = VideoReader(str(base_path), ctx=cpu(0), num_threads=2)
    lora_reader = VideoReader(str(lora_path), ctx=cpu(0), num_threads=2)
    expected_frames = int(row["lance_sampled_frames"])
    if len(base_reader) != expected_frames or len(lora_reader) != expected_frames:
        raise ValueError(f"output frame mismatch for {sample_id}: {len(base_reader)}/{len(lora_reader)}/{expected_frames}")
    if round(float(base_reader.get_avg_fps()), 6) != 12.0 or round(float(lora_reader.get_avg_fps()), 6) != 12.0:
        raise ValueError(f"output FPS mismatch for {sample_id}")
    if base_reader[0].shape[:2] != lora_reader[0].shape[:2]:
        raise ValueError(f"Base and LoRA resolution mismatch for {sample_id}")

    source_indices = json.loads(row["lance_source_indices"])
    if len(source_indices) != expected_frames:
        raise ValueError(f"source index count mismatch for {sample_id}")
    positions = np.linspace(0, expected_frames - 1, min(20, expected_frames), dtype=int).tolist()
    target_height, target_width = base_reader[0].shape[:2]
    scale = min(1.0, 426 / max(target_width, target_height))
    judge_size = (
        max(2, int(target_width * scale) // 2 * 2),
        max(2, int(target_height * scale) // 2 * 2),
    )

    destinations = {
        "source": MEDIA / f"{sample_id}_source.mp4",
        "base": MEDIA / f"{sample_id}_base.mp4",
        "lora_step677": MEDIA / f"{sample_id}_lora_step677.mp4",
    }
    expected_count = len(positions)
    reusable = all(path.exists() for path in destinations.values())
    if reusable:
        infos = {kind: probe(path) for kind, path in destinations.items()}
        reusable = all(info["frames"] == expected_count and info["fps"] == 5.0 for info in infos.values())
    if not reusable:
        source_frames = [
            crop_to_aspect(source_reader[source_indices[position]].asnumpy(), target_width, target_height)
            for position in positions
        ]
        base_frames = [base_reader[position].asnumpy() for position in positions]
        lora_frames = [lora_reader[position].asnumpy() for position in positions]
        encode_frames(destinations["source"], source_frames, judge_size)
        encode_frames(destinations["base"], base_frames, judge_size)
        encode_frames(destinations["lora_step677"], lora_frames, judge_size)
    infos = {kind: probe(path) for kind, path in destinations.items()}
    if len({(info["frames"], info["fps"], info["width"], info["height"]) for info in infos.values()}) != 1:
        raise ValueError(f"judge media not aligned for {sample_id}")
    return {
        **row,
        "official_id": official_id,
        "base_generated_path": str(base_path),
        "base_generated_sha256": sha256_file(base_path),
        "lora_generated_path": str(lora_path),
        "lora_generated_sha256": sha256_file(lora_path),
        "judge_positions": positions,
        "judge_media": infos,
    }


def task_base(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"], "official_id": row["official_id"],
        "subcategory": row["subcategory"], "instruction": row["edit_prompt"],
    }


def main() -> None:
    global ARTIFACTS, MANIFEST, OUTPUTS, MEDIA, OUTPUT_PREFIX, ID_COLUMN, EXPECTED_SAMPLES
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
    MEDIA = ARTIFACTS / "judge_media"
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"expected all {EXPECTED_SAMPLES} {args.stage} samples")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = list(pool.map(prepare_row, rows))
    prepared.sort(key=lambda row: int(row[ID_COLUMN]))

    absolute = []
    pairwise = []
    for row in prepared:
        common = task_base(row)
        media = row["judge_media"]
        for kind in ("base", "lora_step677"):
            absolute.append({
                **common, "kind": "absolute", "model_kind": kind,
                "source_video": media["source"]["path"], "edited_video": media[kind]["path"],
                "source_sha256": media["source"]["sha256"], "edited_sha256": media[kind]["sha256"],
            })
        for order, (left, right) in enumerate((("base", "lora_step677"), ("lora_step677", "base")), 1):
            pairwise.append({
                **common, "kind": "pairwise", "order": order,
                "candidate_a_model": left, "candidate_b_model": right,
                "source_video": media["source"]["path"],
                "candidate_a_video": media[left]["path"], "candidate_b_video": media[right]["path"],
                "source_sha256": media["source"]["sha256"],
                "candidate_a_sha256": media[left]["sha256"], "candidate_b_sha256": media[right]["sha256"],
            })

    stability_ids = {
        row["sample_id"] for row in sorted(
            prepared,
            key=lambda row: hashlib.sha256(f"42\0ivebench-stability\0{row['sample_id']}".encode()).hexdigest(),
        )[:min(9, EXPECTED_SAMPLES)]
    }
    stability = [{**task, "kind": "stability", "repeat_index": 2}
                 for task in absolute if task["sample_id"] in stability_ids]
    write_json(ARTIFACTS / "prepared_media.json", prepared)
    write_json(ARTIFACTS / "judge_tasks_absolute.json", absolute)
    write_json(ARTIFACTS / "judge_tasks_pairwise.json", pairwise)
    write_json(ARTIFACTS / "judge_tasks_stability.json", stability)
    summary = {
        "status": "complete", "stage": args.stage, "samples": len(prepared), "absolute_tasks": len(absolute),
        "pairwise_tasks": len(pairwise), "stability_tasks": len(stability),
        "judge_media_files": len(list(MEDIA.glob("*.mp4"))),
        "policy": "up to 20 paired frames, max side 426, 5 fps, H.264/yuv420p/faststart",
        "absolute_tasks_sha256": sha256_file(ARTIFACTS / "judge_tasks_absolute.json"),
        "pairwise_tasks_sha256": sha256_file(ARTIFACTS / "judge_tasks_pairwise.json"),
        "stability_tasks_sha256": sha256_file(ARTIFACTS / "judge_tasks_stability.json"),
    }
    write_json(ARTIFACTS / "judge_preparation.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
