#!/usr/bin/env python3
"""Build an inventory and a small default-pass audit sheet for ReCo examples."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
TASKS = ("add", "remove", "replace", "style")
AUDIT_SAMPLES_PER_TASK = 10
HASH_PREFIX_LENGTH = 5
PASS = "1"


def vscode_default_sort_key(name: str) -> tuple[tuple[int, object], ...]:
    """Approximate VS Code's default numeric filename collation for ASCII names."""
    parts = re.split(r"(\d+)", name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


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


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASKS:
        video_paths = sorted(
            (EXAMPLES_DIR / task).glob("*.mp4"),
            key=lambda path: vscode_default_sort_key(path.name),
        )
        for video_path in video_paths:
            instruction_path = video_path.with_suffix(".txt")
            instruction = (
                instruction_path.read_text(encoding="utf-8").strip()
                if instruction_path.is_file()
                else ""
            )
            video = inspect_video(video_path)
            rows.append(
                {
                    "hash": video_path.name[:HASH_PREFIX_LENGTH],
                    "task": task,
                    "instruction": instruction,
                    "files_complete": instruction_path.is_file(),
                    **video,
                }
            )
    return rows


def write_inventory(rows: list[dict[str, object]]) -> None:
    fields = [
        "hash",
        "task",
        "instruction",
        "files_complete",
        "decode_ok",
        "width",
        "height",
        "frame_count",
        "fps",
        "duration_seconds",
    ]
    with (ROOT / "reco_inventory.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(rows: list[dict[str, object]], reset: bool) -> bool:
    audit_path = ROOT / "reco_audit.csv"
    if audit_path.exists() and not reset:
        return False

    selected: list[dict[str, object]] = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        selected.extend(task_rows[:AUDIT_SAMPLES_PER_TASK])

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
        for row in selected:
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
    parser.add_argument(
        "--reset-audit",
        action="store_true",
        help="replace the current audit sheet and discard manual review changes",
    )
    args = parser.parse_args()
    rows = collect_rows()
    write_inventory(rows)
    audit_written = write_audit(rows, reset=args.reset_audit)
    print(f"inventory_rows={len(rows)}")
    print(
        f"audit_status={'regenerated' if audit_written else 'preserved'} "
        f"audit_rows={len(TASKS) * AUDIT_SAMPLES_PER_TASK}"
    )


if __name__ == "__main__":
    main()
