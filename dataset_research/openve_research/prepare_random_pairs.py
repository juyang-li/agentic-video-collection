#!/usr/bin/env python3
"""Prepare H.264 comparisons, contact sheets, inventory, and audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
EXAMPLES_DIR = ROOT / "examples"
COMPARISON_DIR = EXAMPLES_DIR / "comparison"
CONTACT_DIR = EXAMPLES_DIR / "contact_sheets"
INVENTORY_PATH = ROOT / "openve_inventory.csv"
AUDIT_PATH = ROOT / "openve_audit.csv"
TASKS = ("add", "change", "remove", "replace", "transform", "other")
HASH_PREFIX_LENGTH = 5
REVIEW_HEIGHT = 540
CONTACT_HALF_WIDTH = 480
CONTACT_FRAME_COUNT = 6
PASS = "1"


def vscode_default_sort_key(name: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
        if part
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    capture.release()
    return {
        "decode_ok": opened,
        "width": width,
        "height": height,
        "frame_count": frames,
        "fps": round(fps, 3),
        "duration_seconds": round(frames / fps, 3) if fps else 0.0,
    }


def write_comparison(source: Path, target: Path, output: Path, rebuild: bool) -> None:
    if output.exists() and not rebuild:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".part.mp4")
    graph = (
        f"[0:v]scale=-2:{REVIEW_HEIGHT},"
        "drawtext=text=SOURCE:x=20:y=20:fontsize=30:fontcolor=white:"
        "borderw=3:bordercolor=black[left];"
        f"[1:v]scale=-2:{REVIEW_HEIGHT},"
        "drawtext=text=TARGET:x=20:y=20:fontsize=30:fontcolor=white:"
        "borderw=3:bordercolor=black[right];"
        "[left][right]hstack=inputs=2[out]"
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


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {path}")
    return frame


def fit_half(frame: np.ndarray) -> np.ndarray:
    height = round(frame.shape[0] * CONTACT_HALF_WIDTH / frame.shape[1])
    return cv2.resize(frame, (CONTACT_HALF_WIDTH, height), interpolation=cv2.INTER_AREA)


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Center a resized frame on black without distorting its aspect ratio."""
    if frame.shape[0] == height:
        return frame
    top = (height - frame.shape[0]) // 2
    bottom = height - frame.shape[0] - top
    return cv2.copyMakeBorder(
        frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def label(frame: np.ndarray, text: str) -> None:
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def write_contact_sheet(
    source: Path, target: Path, paired_frames: int, output: Path, rebuild: bool
) -> None:
    if output.exists() and not rebuild:
        return
    indices = np.linspace(0, max(0, paired_frames - 1), CONTACT_FRAME_COUNT, dtype=int)
    rows: list[np.ndarray] = []
    for index in indices.tolist():
        left = fit_half(read_frame(source, index))
        right = fit_half(read_frame(target, index))
        row_height = max(left.shape[0], right.shape[0])
        left = pad_to_height(left, row_height)
        right = pad_to_height(right, row_height)
        label(left, f"SOURCE f{index}")
        label(right, f"TARGET f{index}")
        rows.append(cv2.hconcat([left, right]))
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
        key=lambda row: (TASKS.index(row["task"]), vscode_default_sort_key(row["sample_id"])),
    )


def collect_rows(rebuild: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in load_manifest():
        source_path = ROOT / item["source_file"]
        target_path = ROOT / item["target_file"]
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing pair: {source_path}, {target_path}")
        source_sha256 = sha256_file(source_path)
        target_sha256 = sha256_file(target_path)
        short_hash = target_sha256[:HASH_PREFIX_LENGTH]
        source = inspect_video(source_path)
        target = inspect_video(target_path)
        paired_frames = min(int(source["frame_count"]), int(target["frame_count"]))
        comparison = COMPARISON_DIR / f"{short_hash}-{item['sample_id']}.mp4"
        contact = CONTACT_DIR / f"{short_hash}-{item['sample_id']}.jpg"
        instruction = comparison.with_suffix(".txt")
        write_comparison(source_path, target_path, comparison, rebuild)
        write_contact_sheet(source_path, target_path, paired_frames, contact, rebuild)
        instruction.write_text(item["instruction"] + "\n", encoding="utf-8")
        fps_delta = abs(float(source["fps"]) - float(target["fps"]))
        rows.append(
            {
                "hash": short_hash,
                "task": item["task"],
                "sample_id": item["sample_id"],
                "metadata_index": item["metadata_index"],
                "subpopulation": item["subpopulation"],
                "subpopulation_size": item["subpopulation_size"],
                "instruction": item["instruction"],
                "source_file": item["source_file"],
                "target_file": item["target_file"],
                "comparison_file": str(comparison.relative_to(ROOT)),
                "contact_sheet": str(contact.relative_to(ROOT)),
                "source_sha256": source_sha256,
                "target_sha256": target_sha256,
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
                "frame_count_delta": int(source["frame_count"]) - int(target["frame_count"]),
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
        "hash", "task", "person_relevance", "instruction_correctness",
        "logical_correctness", "content_preservation", "temporal_consistency",
        "visual_quality", "technical_quality", "notes",
    ]
    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hash": row["hash"], "task": row["task"],
                    "person_relevance": PASS, "instruction_correctness": PASS,
                    "logical_correctness": PASS, "content_preservation": PASS,
                    "temporal_consistency": PASS, "visual_quality": PASS,
                    "technical_quality": PASS, "notes": "",
                }
            )
    return True


def main() -> None:
    global ROOT, MANIFEST_PATH, EXAMPLES_DIR, COMPARISON_DIR, CONTACT_DIR
    global INVENTORY_PATH, AUDIT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reset-audit", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    MANIFEST_PATH = args.manifest.resolve()
    ROOT = MANIFEST_PATH.parent
    EXAMPLES_DIR = ROOT / "examples"
    COMPARISON_DIR = EXAMPLES_DIR / "comparison"
    CONTACT_DIR = EXAMPLES_DIR / "contact_sheets"
    INVENTORY_PATH = ROOT / "openve_inventory.csv"
    AUDIT_PATH = ROOT / "openve_audit.csv"
    rows = collect_rows(args.rebuild)
    write_inventory(rows)
    written = write_audit(rows, args.reset_audit)
    print(f"inventory_rows={len(rows)}")
    print(f"audit_status={'regenerated' if written else 'preserved'}")


if __name__ == "__main__":
    main()
