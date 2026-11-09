#!/usr/bin/env python3
"""Build configurable multi-panel Lance comparison videos and a review archive."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import zipfile
from pathlib import Path

import imageio_ffmpeg

from verify_outputs import probe


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def panel_path(panel: dict[str, object], row: dict[str, str], base: Path) -> Path:
    if "manifest_column" in panel:
        return Path(row[str(panel["manifest_column"])]).resolve()
    index = int(row["sample_index"])
    candidates = [resolve(base, value) / f"{index:06d}.mp4" for value in panel["output_dirs"]]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(f"Expected one output for sample {index}: {candidates}, got {existing}")
    return existing[0]


def build_video(
    paths: list[Path],
    labels: list[str],
    target: Path,
    fps: int,
    panel_width: int,
    panel_height: int,
    rebuild: bool,
) -> dict[str, int | float | str]:
    infos = [probe(path) for path in paths]
    model_infos = infos[1:]
    frames = {int(info["frames"]) for info in model_infos}
    model_fps = {float(info["fps"]) for info in model_infos}
    if len(frames) != 1 or len(model_fps) != 1:
        raise RuntimeError(f"Model outputs are not aligned: frames={frames}, fps={model_fps}")
    output_frames = next(iter(frames))
    output_duration = output_frames / next(iter(model_fps))
    source_duration = int(infos[0]["frames"]) / float(infos[0]["fps"])
    expected_width = panel_width * len(paths)

    if target.exists() and not rebuild:
        info = probe(target)
        expected = {
            "frames": output_frames,
            "fps": float(fps),
            "width": expected_width,
            "height": panel_height,
        }
        if {key: info[key] for key in expected} != expected:
            raise RuntimeError(f"Existing comparison has unexpected specs: {target}")
        return info

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.part.mp4")
    if temporary.exists():
        temporary.unlink()
    panel_filter = (
        f"scale={panel_width}:{panel_height}:force_original_aspect_ratio=decrease,"
        f"pad={panel_width}:{panel_height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )
    filters = []
    stack_inputs = []
    for index, label in enumerate(labels):
        if not re.fullmatch(r"[A-Za-z0-9 +_-]+", label):
            raise ValueError(f"Unsafe panel label: {label!r}")
        timing = f"setpts={output_duration / source_duration:.12f}*PTS," if index == 0 else ""
        filters.append(
            f"[{index}:v]{timing}fps={fps},{panel_filter},"
            f"drawtext=text='{label}':x=16:y=16:fontsize=24:fontcolor=white:"
            f"borderw=3:bordercolor=black[p{index}]"
        )
        stack_inputs.append(f"[p{index}]")
    filters.append(f"{''.join(stack_inputs)}hstack=inputs={len(paths)}:shortest=1[out]")

    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
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
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    temporary.replace(target)
    info = probe(target)
    expected = {
        "frames": output_frames,
        "fps": float(fps),
        "width": expected_width,
        "height": panel_height,
    }
    if {key: info[key] for key in expected} != expected:
        raise RuntimeError(f"Unexpected comparison specs for {target}: {info}")
    return info


def initialize_scores(path: Path, rows: list[dict[str, str]], panels: list[dict[str, object]]) -> None:
    if path.exists():
        return
    model_labels = [str(panel["label"]).lower().replace(" ", "_") for panel in panels[1:]]
    fields = ["sample_index", "video_id", "category", "instruction"]
    for label in model_labels:
        fields.extend(
            [
                f"{label}_accuracy",
                f"{label}_preservation",
                f"{label}_temporal_consistency",
                f"{label}_visual_quality",
                f"{label}_notes",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = resolve(base, config["benchmark_manifest"])
    comparison_dir = resolve(base, config["comparison_dir"])
    inventory_path = resolve(base, config["inventory"])
    scores_path = resolve(base, config["scores"])
    archive_path = resolve(base, config["archive"])
    panels = config["panels"]
    if len(panels) < 2 or "manifest_column" not in panels[0]:
        raise ValueError("First panel must be the source manifest column")

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty benchmark manifest")
    labels = [str(panel["label"]) for panel in panels]
    inventory_rows = []
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, row in enumerate(rows, start=1):
        paths = [panel_path(panel, row, base) for panel in panels]
        index = int(row["sample_index"])
        stem = f"{index:02d}-{row['video_id']}"
        target = comparison_dir / f"{stem}.mp4"
        info = build_video(
            paths,
            labels,
            target,
            int(config.get("fps", 12)),
            int(config.get("panel_width", 480)),
            int(config.get("panel_height", 540)),
            args.rebuild,
        )
        (comparison_dir / f"{stem}.txt").write_text(
            f"video_id: {row['video_id']}\n"
            f"category: {row['category']}\n"
            f"instruction: {row['instruction']}\n"
            f"panels: {' | '.join(labels)}\n",
            encoding="utf-8",
        )
        inventory_row = {
            "sample_index": index,
            "video_id": row["video_id"],
            "category": row["category"],
            "instruction": row["instruction"],
            "comparison_file": target.name,
            "comparison_frames": info["frames"],
            "comparison_fps": info["fps"],
            "comparison_width": info["width"],
            "comparison_height": info["height"],
            "comparison_bytes": info["bytes"],
            "comparison_sha256": info["sha256"],
        }
        for label, path in zip(labels, paths):
            inventory_row[f"{label.lower().replace(' ', '_')}_file"] = str(path)
        inventory_rows.append(inventory_row)
        print(f"comparison={ordinal}/{len(rows)} file={target.name}")

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory_rows)
    initialize_scores(scores_path, rows, panels)

    temporary_archive = archive_path.with_name(f"{archive_path.name}.part")
    if temporary_archive.exists():
        temporary_archive.unlink()
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(comparison_dir.iterdir()):
            archive.write(path, arcname=f"comparison/{path.name}")
        archive.write(config_path, arcname=config_path.name)
        archive.write(manifest, arcname=manifest.name)
        archive.write(scores_path, arcname=scores_path.name)
        for value in config.get("archive_files", []):
            path = resolve(base, value)
            if not path.is_file():
                raise FileNotFoundError(f"Archive file does not exist: {path}")
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(temporary_archive) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    temporary_archive.replace(archive_path)
    print(f"inventory={inventory_path}")
    print(f"archive={archive_path}")


if __name__ == "__main__":
    main()
