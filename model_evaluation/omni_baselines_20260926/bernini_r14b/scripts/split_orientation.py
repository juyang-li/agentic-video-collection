#!/usr/bin/env python3
"""Split Bernini task JSON by the orientation encoded in staged source videos."""

import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, required=True)
    args = ap.parse_args()
    tasks = json.loads(args.inputs.read_text(encoding="utf-8"))
    groups = {"landscape": [], "portrait": []}
    for task in tasks:
        frame = VideoReader(task["video"], ctx=cpu(0))[0]
        height, width = frame.shape[:2]
        groups["portrait" if height > width else "landscape"].append(task)
    for name, rows in groups.items():
        path = args.output_prefix.with_name(args.output_prefix.name + f"_{name}.json")
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path}: {len(rows)}")


if __name__ == "__main__":
    main()
