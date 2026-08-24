#!/usr/bin/env python3
"""Require every numeric Lance config key to have a decodable output MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures: list[str] = []
    for key in config:
        path = args.output_dir / f"{int(key):06d}.mp4"
        try:
            reader = VideoReader(str(path), ctx=cpu(0))
            if len(reader) == 0:
                raise RuntimeError("zero frames")
            for start in range(0, len(reader), 32):
                reader.get_batch(list(range(start, min(start + 32, len(reader)))))
            print(
                json.dumps(
                    {
                        "file": path.name,
                        "frames": len(reader),
                        "fps": float(reader.get_avg_fps()),
                        "height": int(reader[0].shape[0]),
                        "width": int(reader[0].shape[1]),
                        "bytes": path.stat().st_size,
                        "full_decode_ok": True,
                    }
                )
            )
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
