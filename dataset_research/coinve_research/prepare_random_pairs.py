#!/usr/bin/env python3
"""Build CoinVE H.264 comparisons, contact sheets, inventory, and audit."""

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
COMPARISON_DIR = ROOT / "examples" / "comparison"
CONTACT_DIR = ROOT / "examples" / "contact_sheets"
TASKS = ("compose-2", "compose-3", "compose-4", "compose-5")
HASH_PREFIX_LENGTH = 5
PASS = "1"
REVIEW_HEIGHT = 540
CONTACT_HALF_WIDTH = 480
CONTACT_FRAME_COUNT = 6


def vscode_default_sort_key(name: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
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
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    capture.release()
    return {
        "decode_ok": opened,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "duration_seconds": round(frame_count / fps, 3) if fps else 0.0,
    }


def overlay_label(frame: np.ndarray, label: str) -> None:
    cv2.putText(
        frame,
        label,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    width = round(frame.shape[1] * height / frame.shape[0])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_comparison(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    rebuild: bool,
) -> None:
    if output_path.exists() and not rebuild:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".part.mp4")
    filter_graph = (
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
            str(source_path),
            "-i",
            str(target_path),
            "-filter_complex",
            filter_graph,
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
            str(temp_path),
        ],
        check=True,
    )
    temp_path.replace(output_path)


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} from {path}")
    return frame


def fit_contact_half(frame: np.ndarray) -> np.ndarray:
    height = round(frame.shape[0] * CONTACT_HALF_WIDTH / frame.shape[1])
    return cv2.resize(frame, (CONTACT_HALF_WIDTH, height), interpolation=cv2.INTER_AREA)


def write_contact_sheet(
    source_path: Path,
    target_path: Path,
    paired_frames: int,
    output_path: Path,
    rebuild: bool,
) -> None:
    if output_path.exists() and not rebuild:
        return
    indices = np.linspace(
        0, max(0, paired_frames - 1), CONTACT_FRAME_COUNT, dtype=int
    ).tolist()
    rows: list[np.ndarray] = []
    for index in indices:
        source = fit_contact_half(read_frame(source_path, index))
        target = fit_contact_half(read_frame(target_path, index))
        overlay_label(source, f"SOURCE f{index}")
        overlay_label(target, f"TARGET f{index}")
        rows.append(cv2.hconcat([source, target]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.vconcat(rows)):
        raise RuntimeError(f"cannot write contact sheet: {output_path}")


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample_id in manifest")
    unknown = {row["task"] for row in rows} - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    return sorted(
        rows,
        key=lambda row: (
            TASKS.index(row["task"]),
            vscode_default_sort_key(row["sample_id"]),
        ),
    )


def collect_rows(rebuild: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[Path] = []
    for manifest_row in load_manifest():
        for field in ("source_file", "target_file"):
            path = ROOT / manifest_row[field]
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "missing random pair files; run download_random_pairs.py:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    for manifest_row in load_manifest():
        source_path = ROOT / manifest_row["source_file"]
        target_path = ROOT / manifest_row["target_file"]
        source_sha256 = sha256_file(source_path)
        target_sha256 = sha256_file(target_path)
        short_hash = target_sha256[:HASH_PREFIX_LENGTH]
        source = inspect_video(source_path)
        target = inspect_video(target_path)
        paired_frames = min(source["frame_count"], target["frame_count"])
        comparison_path = COMPARISON_DIR / f"{short_hash}-{manifest_row['sample_id']}.mp4"
        contact_path = CONTACT_DIR / f"{short_hash}-{manifest_row['sample_id']}.jpg"
        instruction_path = comparison_path.with_suffix(".txt")
        write_comparison(source_path, target_path, comparison_path, rebuild)
        write_contact_sheet(
            source_path, target_path, paired_frames, contact_path, rebuild
        )
        instruction_lines = manifest_row["instructions"].split(" || ")
        instruction_path.write_text(
            "\n".join(
                f"{index}. {instruction}"
                for index, instruction in enumerate(instruction_lines, start=1)
            )
            + "\n",
            encoding="utf-8",
        )
        fps_delta = abs(float(source["fps"]) - float(target["fps"]))
        rows.append(
            {
                "hash": short_hash,
                "task": manifest_row["task"],
                "sample_id": manifest_row["sample_id"],
                "instruction_count": manifest_row["instruction_count"],
                "instructions": manifest_row["instructions"],
                "operations": manifest_row["operations"],
                "objects": manifest_row["objects"],
                "source_file": manifest_row["source_file"],
                "target_file": manifest_row["target_file"],
                "comparison_file": str(comparison_path.relative_to(ROOT)),
                "contact_sheet": str(contact_path.relative_to(ROOT)),
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
                "frame_count_delta": int(source["frame_count"])
                - int(target["frame_count"]),
                "fps_delta": round(fps_delta, 3),
                "resolution_match": source["width"] == target["width"]
                and source["height"] == target["height"],
                "exact_specs_match": source["frame_count"]
                == target["frame_count"]
                and fps_delta <= 0.01
                and source["width"] == target["width"]
                and source["height"] == target["height"],
            }
        )
    return rows


def write_inventory(rows: list[dict[str, object]]) -> None:
    with (ROOT / "coinve_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_audit(rows: list[dict[str, object]], reset: bool) -> bool:
    audit_path = ROOT / "coinve_audit.csv"
    if audit_path.exists() and not reset:
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
    with audit_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hash": row["hash"],
                    "task": row["task"],
                    "person_relevance": PASS,
                    "instruction_correctness": PASS,
                    "logical_correctness": PASS,
                    "content_preservation": PASS,
                    "temporal_consistency": PASS,
                    "visual_quality": PASS,
                    "technical_quality": PASS,
                    "notes": "",
                }
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reset-audit", action="store_true")
    args = parser.parse_args()
    rows = collect_rows(rebuild=args.rebuild)
    write_inventory(rows)
    audit_written = write_audit(rows, reset=args.reset_audit)
    print(f"inventory_rows={len(rows)}")
    print(f"audit_status={'regenerated' if audit_written else 'preserved'}")


if __name__ == "__main__":
    main()
