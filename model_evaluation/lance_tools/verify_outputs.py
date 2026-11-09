#!/usr/bin/env python3
"""Strictly full-decode Lance outputs and write a reusable technical inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from decord import VideoReader, cpu


FIELDS = [
    "sample_index",
    "file",
    "source_frames",
    "source_fps",
    "sampled_source_frames",
    "sampled_source_fps",
    "frames",
    "fps",
    "original_source_duration_seconds",
    "source_duration_seconds",
    "output_duration_seconds",
    "duration_delta_seconds",
    "height",
    "width",
    "bytes",
    "sha256",
    "full_decode_ok",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, int | float | str]:
    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        raise RuntimeError("zero frames")
    for start in range(0, len(reader), 32):
        reader.get_batch(list(range(start, min(start + 32, len(reader)))))
    first = reader[0]
    return {
        "frames": len(reader),
        "fps": round(float(reader.get_avg_fps()), 6),
        "height": int(first.shape[0]),
        "width": int(first.shape[1]),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_video_path(sample: dict) -> Path:
    """Return the conditioning video used by a Lance edit config row."""
    dtypes = sample.get("element_dtype_array", [])
    values = sample.get("interleave_array", [])
    targets = sample.get("istarget_in_interleave", [0] * len(dtypes))
    matches = [
        Path(value)
        for dtype, value, is_target in zip(dtypes, values, targets)
        if dtype == "video" and int(is_target) == 0
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one conditioning video, got {len(matches)}")
    return matches[0]


def lance_sampled_frame_count(
    source_frames: int,
    source_fps: float,
    max_duration: float,
    sample_fps: int = 12,
    temporal: int = 4,
) -> int:
    """Reproduce ValidationDataset's deterministic MultiClipsFrameSampler count."""
    if source_frames < 1 or source_fps <= 0 or max_duration <= 0:
        raise ValueError("invalid source timing or max duration")
    # ValidationDataset quantizes Decord's average FPS before passing it to
    # MultiClipsFrameSampler.  Preserve that behavior at temporal bucket
    # boundaries (for example, 73 frames at 23.976 FPS is sampled using 24).
    sampler_source_fps = int(round(source_fps))
    if sampler_source_fps < 1:
        raise ValueError(f"Lance sampler produced invalid rounded FPS: {sampler_source_fps}")
    duration = min(source_frames / sampler_source_fps, max_duration)
    frames = int(round(duration * sample_fps))
    if frames % temporal:
        frames = frames // temporal * temporal + 1
    else:
        frames = frames // temporal * temporal + 1 - temporal
    if frames < 1:
        raise ValueError(f"Lance sampler produced invalid frame count: {frames}")
    return frames


def probe_aligned_output(path: Path, sample: dict) -> dict[str, int | float | str]:
    """Full-decode output and require exact alignment to Lance's sampled source."""
    info = probe(path)
    source_path = source_video_path(sample)
    source_reader = VideoReader(str(source_path), ctx=cpu(0))
    source_frames = len(source_reader)
    source_fps = float(source_reader.get_avg_fps())
    max_duration = float(os.environ.get("LANCE_MAX_DURATION", "6.0"))
    sampled_fps = 12.0
    sampled_frames = lance_sampled_frame_count(source_frames, source_fps, max_duration)
    output_frames = int(info["frames"])
    output_fps = float(info["fps"])
    if output_frames != sampled_frames:
        raise RuntimeError(
            f"sampled frame-count mismatch: expected={sampled_frames}, output={output_frames}, "
            f"source={source_frames}@{source_fps:.6f}, max_duration={max_duration}, path={source_path}"
        )
    if abs(output_fps - sampled_fps) > 0.01:
        raise RuntimeError(
            f"sampled fps mismatch: expected={sampled_fps:.6f}, output={output_fps:.6f}, source={source_path}"
        )
    original_source_duration = source_frames / source_fps
    source_duration = sampled_frames / sampled_fps
    output_duration = output_frames / output_fps
    return {
        "source_frames": source_frames,
        "source_fps": round(source_fps, 6),
        "sampled_source_frames": sampled_frames,
        "sampled_source_fps": sampled_fps,
        **info,
        "original_source_duration_seconds": round(original_source_duration, 6),
        "source_duration_seconds": round(source_duration, 6),
        "output_duration_seconds": round(output_duration, 6),
        "duration_delta_seconds": round(output_duration - source_duration, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    failures = []
    for key, sample in config.items():
        path = args.output_dir / f"{int(key):06d}.mp4"
        try:
            info = probe_aligned_output(path, sample)
            row = {
                "sample_index": int(key),
                "file": path.name,
                **info,
                "full_decode_ok": 1,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))

    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.inventory.with_suffix(args.inventory.suffix + ".part")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(args.inventory)
        print(f"inventory={args.inventory}")


if __name__ == "__main__":
    main()
