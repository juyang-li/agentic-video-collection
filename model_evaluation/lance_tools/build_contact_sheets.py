#!/usr/bin/env python3
"""Create sparse temporal contact sheets from comparison MP4 files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    with args.inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        video = args.comparison_dir / row["comparison_file"]
        target = args.output_dir / f"{video.stem}.jpg"
        reader = VideoReader(str(video), ctx=cpu(0))
        indices = [round(i * (len(reader) - 1) / (args.frames - 1)) for i in range(args.frames)]
        batch = reader.get_batch(indices).asnumpy()
        images = [Image.fromarray(frame).convert("RGB") for frame in batch]
        width = images[0].width
        height = sum(image.height for image in images)
        if target.exists() and not args.rebuild:
            with Image.open(target) as existing:
                if existing.size != (width, height):
                    raise RuntimeError(f"Unexpected existing contact sheet: {target}")
            continue
        canvas = Image.new("RGB", (width, height), "black")
        y = 0
        for image in images:
            canvas.paste(image, (0, y))
            y += image.height
        canvas.save(target, quality=90, optimize=True)
        print(f"contact_sheet={target}")


if __name__ == "__main__":
    main()
