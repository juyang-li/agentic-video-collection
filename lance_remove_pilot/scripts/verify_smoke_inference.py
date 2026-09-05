#!/usr/bin/env python3
"""Verify that the fine-tuned checkpoint produced one complete smoke MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def verify_video(path: Path) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise ValueError(f"empty decoded frame in {path}")
        decoded_frames += 1
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or decoded_frames <= 0:
        raise ValueError(f"invalid video properties: {path}")
    if reported_frames > 0 and reported_frames != decoded_frames:
        raise ValueError(
            f"incomplete decode: reported={reported_frames}, decoded={decoded_frames}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": decoded_frames,
        "duration_seconds": decoded_frames / fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    videos = sorted(args.output_dir.glob("*.mp4"))
    if len(videos) != 1:
        raise SystemExit(f"expected exactly one MP4, found {len(videos)}")
    prompt_path = args.output_dir / "prompt.json"
    if not prompt_path.is_file():
        raise SystemExit(f"missing inference prompt record: {prompt_path}")
    result = verify_video(videos[0])
    report_path = args.output_dir / "decode_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **result}, indent=2))


if __name__ == "__main__":
    main()
