#!/usr/bin/env python3
"""Prepare garment-aware TripVVT review files, inventory, and audit."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from openve_research import prepare_random_pairs as common  # noqa: E402


MANIFEST_PATH = ROOT / "random_pairs.csv"
EXAMPLES_DIR = ROOT / "examples"
COMPARISON_DIR = EXAMPLES_DIR / "comparison"
CONTACT_DIR = EXAMPLES_DIR / "contact_sheets"
INVENTORY_PATH = ROOT / "tripvvt_inventory.csv"
AUDIT_PATH = ROOT / "tripvvt_audit.csv"
TASKS = ("try-on-dresses", "try-on-lower_body", "try-on-upper_body")
REVIEW_WIDTH = 480
REVIEW_HEIGHT = 854
CONTACT_WIDTH = 320
CONTACT_HEIGHT = 569
CONTACT_FRAME_COUNT = 6


def fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write_comparison(
    source: Path, target: Path, cloth: Path, output: Path, rebuild: bool
) -> None:
    if output.exists() and not rebuild:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".part.mp4")
    scale_pad = (
        f"scale={REVIEW_WIDTH}:{REVIEW_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={REVIEW_WIDTH}:{REVIEW_HEIGHT}:(ow-iw)/2:(oh-ih)/2:white"
    )
    graph = (
        f"[0:v]{scale_pad},drawtext=text=SOURCE:x=20:y=20:fontsize=30:"
        "fontcolor=white:borderw=3:bordercolor=black[source];"
        f"[2:v]{scale_pad},drawtext=text=GARMENT:x=20:y=20:fontsize=30:"
        "fontcolor=white:borderw=3:bordercolor=black[garment];"
        f"[1:v]{scale_pad},drawtext=text=TARGET:x=20:y=20:fontsize=30:"
        "fontcolor=white:borderw=3:bordercolor=black[target];"
        "[source][garment][target]hstack=inputs=3:shortest=1[out]"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(target),
            "-loop",
            "1",
            "-framerate",
            "24",
            "-i",
            str(cloth),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
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
            "-shortest",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def write_contact_sheet(
    source: Path,
    target: Path,
    cloth: Path,
    paired_frames: int,
    output: Path,
    rebuild: bool,
) -> None:
    if output.exists() and not rebuild:
        return
    cloth_frame = cv2.imread(str(cloth), cv2.IMREAD_COLOR)
    if cloth_frame is None:
        raise RuntimeError(f"cannot read garment image: {cloth}")
    indices = np.linspace(0, max(0, paired_frames - 1), CONTACT_FRAME_COUNT, dtype=int)
    rows: list[np.ndarray] = []
    for index in indices.tolist():
        source_frame = fit_panel(common.read_frame(source, index), CONTACT_WIDTH, CONTACT_HEIGHT)
        garment_frame = fit_panel(cloth_frame, CONTACT_WIDTH, CONTACT_HEIGHT)
        target_frame = fit_panel(common.read_frame(target, index), CONTACT_WIDTH, CONTACT_HEIGHT)
        common.label(source_frame, f"SOURCE f{index}")
        common.label(garment_frame, "GARMENT")
        common.label(target_frame, f"TARGET f{index}")
        rows.append(cv2.hconcat([source_frame, garment_frame, target_frame]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.vconcat(rows)):
        raise RuntimeError(f"cannot write contact sheet: {output}")


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 15 or len({row["sample_id"] for row in rows}) != 15:
        raise ValueError("manifest must contain 15 unique samples")
    unknown = {row["task"] for row in rows} - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    return sorted(
        rows,
        key=lambda row: (TASKS.index(row["task"]), common.vscode_default_sort_key(row["sample_id"])),
    )


def collect_rows(rebuild: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in load_manifest():
        source_path = ROOT / item["source_file"]
        target_path = ROOT / item["target_file"]
        cloth_path = ROOT / item["cloth_file"]
        for path in (source_path, target_path, cloth_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_sha256 = common.sha256_file(source_path)
        target_sha256 = common.sha256_file(target_path)
        cloth_sha256 = common.sha256_file(cloth_path)
        short_hash = target_sha256[: common.HASH_PREFIX_LENGTH]
        source = common.inspect_video(source_path)
        target = common.inspect_video(target_path)
        paired_frames = min(int(source["frame_count"]), int(target["frame_count"]))
        comparison = COMPARISON_DIR / f"{short_hash}-{item['sample_id']}.mp4"
        contact = CONTACT_DIR / f"{short_hash}-{item['sample_id']}.jpg"
        instruction_file = comparison.with_suffix(".txt")
        write_comparison(source_path, target_path, cloth_path, comparison, rebuild)
        write_contact_sheet(
            source_path, target_path, cloth_path, paired_frames, contact, rebuild
        )
        instruction_file.write_text(
            f"task: {item['task']}\n"
            f"instruction: {item['instruction']}\n"
            f"official_text: {item['official_text']}\n",
            encoding="utf-8",
        )
        fps_delta = abs(float(source["fps"]) - float(target["fps"]))
        rows.append(
            {
                "hash": short_hash,
                "task": item["task"],
                "sample_id": item["sample_id"],
                "population_index": item["population_index"],
                "split": item["split"],
                "instruction": item["instruction"],
                "official_text": item["official_text"],
                "source_file": item["source_file"],
                "target_file": item["target_file"],
                "cloth_file": item["cloth_file"],
                "comparison_file": str(comparison.relative_to(ROOT)),
                "contact_sheet": str(contact.relative_to(ROOT)),
                "source_sha256": source_sha256,
                "target_sha256": target_sha256,
                "cloth_sha256": cloth_sha256,
                "source_decode_ok": source["decode_ok"],
                "target_decode_ok": target["decode_ok"],
                "source_width": source["width"],
                "source_height": source["height"],
                "source_frame_count": source["frame_count"],
                "source_fps": source["fps"],
                "source_duration_seconds": source["duration_seconds"],
                "target_width": target["width"],
                "target_height": target["height"],
                "target_frame_count": target["frame_count"],
                "target_fps": target["fps"],
                "target_duration_seconds": target["duration_seconds"],
                "paired_frame_count": paired_frames,
                "frame_count_delta": int(source["frame_count"])
                - int(target["frame_count"]),
                "fps_delta": round(fps_delta, 3),
                "exact_specs_match": source["width"] == target["width"]
                and source["height"] == target["height"]
                and source["frame_count"] == target["frame_count"]
                and fps_delta <= 0.01,
            }
        )
    return rows


def write_inventory(rows: list[dict[str, object]]) -> None:
    with INVENTORY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_audit(rows: list[dict[str, object]], reset: bool) -> bool:
    if AUDIT_PATH.exists() and not reset:
        return False
    fields = [
        "hash",
        "task",
        "person_relevance",
        "instruction_correctness",
        "logical_correctness",
        "content_preservation",
        "temporal_consistency",
        "visual_quality",
        "technical_quality",
        "notes",
    ]
    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hash": row["hash"],
                    "task": row["task"],
                    "person_relevance": "1",
                    "instruction_correctness": "1",
                    "logical_correctness": "1",
                    "content_preservation": "1",
                    "temporal_consistency": "1",
                    "visual_quality": "1",
                    "technical_quality": "1",
                    "notes": "",
                }
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reset-audit", action="store_true")
    args = parser.parse_args()
    rows = collect_rows(args.rebuild)
    write_inventory(rows)
    written = write_audit(rows, args.reset_audit)
    print(f"inventory_rows={len(rows)}")
    print(f"audit_status={'regenerated' if written else 'preserved'}")


if __name__ == "__main__":
    main()
