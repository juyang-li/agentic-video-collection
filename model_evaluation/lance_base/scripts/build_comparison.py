#!/usr/bin/env python3
"""Build 15 side-by-side H.264 videos for manual Lance review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import zipfile
from pathlib import Path

import imageio_ffmpeg
from decord import VideoReader, cpu


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmark_15.csv"
TECHNICAL_RESULTS = ROOT / "technical_results.csv"
SCORES = ROOT / "scores.csv"
COMPARISON_ROOT = ROOT / "comparison"
INVENTORY = COMPARISON_ROOT / "inventory.csv"
ARCHIVE = ROOT / "lance_base_comparison_15_h264.zip"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 12
PANEL_WIDTH = 960
PANEL_HEIGHT = 540


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, int | float | str]:
    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        raise RuntimeError(f"Zero-frame video: {path}")
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


def raw_output_path(sample_index: int) -> Path:
    run = "benchmark_gate" if sample_index <= 2 else "benchmark_remaining"
    return ROOT / "outputs_raw" / run / f"{sample_index:06d}.mp4"


def build_video(
    source: Path,
    output: Path,
    target: Path,
    source_duration: float,
    output_duration: float,
    output_frames: int,
    rebuild: bool,
) -> None:
    if target.exists() and not rebuild:
        info = probe(target)
        if info["frames"] != output_frames or info["width"] != 1920 or info["height"] != 540:
            raise RuntimeError(f"Existing comparison has unexpected specs: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.stem}.part.mp4")
    if temp.exists():
        temp.unlink()
    source_pts_scale = output_duration / source_duration
    panel = (
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={PANEL_WIDTH}:{PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    filter_graph = (
        f"[0:v]setpts={source_pts_scale:.12f}*PTS,fps={FPS},{panel},"
        "drawtext=text=SOURCE:x=20:y=20:fontsize=30:fontcolor=white:"
        "borderw=3:bordercolor=black[left];"
        f"[1:v]fps={FPS},{panel},"
        "drawtext=text='LANCE OUTPUT':x=20:y=20:fontsize=30:fontcolor=white:"
        "borderw=3:bordercolor=black[right];"
        "[left][right]hstack=inputs=2:shortest=1[out]"
    )
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(output),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-frames:v",
            str(output_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp),
        ],
        check=True,
    )
    temp.replace(target)
    info = probe(target)
    expected = {"frames": output_frames, "fps": float(FPS), "width": 1920, "height": 540}
    actual = {key: info[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected comparison specs for {target}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with TECHNICAL_RESULTS.open(newline="", encoding="utf-8") as handle:
        technical_by_index = {row["sample_index"]: row for row in csv.DictReader(handle)}
    if len(manifest_rows) != 15 or len(technical_by_index) != 15:
        raise RuntimeError("Expected exactly 15 manifest and technical rows")

    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        index = int(row["sample_index"])
        technical = technical_by_index[str(index)]
        source = Path(row["local_source"])
        output = raw_output_path(index)
        source_duration = int(technical["source_frames"]) / float(technical["source_fps"])
        output_frames = int(technical["output_frames"])
        output_duration = output_frames / float(technical["output_fps"])
        stem = f"{index:02d}-{row['video_id']}"
        comparison = COMPARISON_ROOT / f"{stem}.mp4"
        build_video(
            source,
            output,
            comparison,
            source_duration,
            output_duration,
            output_frames,
            args.rebuild,
        )
        instruction_path = COMPARISON_ROOT / f"{stem}.txt"
        instruction_path.write_text(
            f"video_id: {row['video_id']}\n"
            f"category: {row['category']}\n"
            f"instruction: {row['instruction']}\n",
            encoding="utf-8",
        )
        info = probe(comparison)
        inventory_rows.append(
            {
                "sample_index": index,
                "video_id": row["video_id"],
                "category": row["category"],
                "instruction": row["instruction"],
                "comparison_file": comparison.name,
                "comparison_frames": info["frames"],
                "comparison_fps": info["fps"],
                "comparison_width": info["width"],
                "comparison_height": info["height"],
                "comparison_bytes": info["bytes"],
                "comparison_sha256": info["sha256"],
                "source_time_scale": round(output_duration / source_duration, 6),
            }
        )

    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory_rows)

    if ARCHIVE.exists():
        if not args.rebuild:
            raise RuntimeError(f"Refusing to overwrite existing archive: {ARCHIVE}")
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(COMPARISON_ROOT.iterdir()):
            archive.write(path, arcname=f"comparison/{path.name}")
        for path in (MANIFEST, TECHNICAL_RESULTS, SCORES):
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_file}")

    print(f"comparison_videos={len(inventory_rows)}")
    print(f"comparison_dir={COMPARISON_ROOT}")
    print(f"archive={ARCHIVE}")


if __name__ == "__main__":
    main()
