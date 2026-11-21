"""Create synthetic 10-second videos for a reproducible example, not a speed benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import lossless_video


def make_demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((72, 128))
    base = np.stack([(xx * 2) % 255, (yy * 3) % 255, ((xx + yy) * 2) % 255], axis=-1).astype(np.uint8)
    frames = [np.roll(base, i // 3, axis=1) for i in range(121)]
    lossless_video(root / "gt.mp4", frames, 12)
    lossless_video(root / "prediction24.mp4", [frames[min(i // 2, 120)] for i in range(243)], 24)
    padded = []
    for frame in frames:
        canvas = np.zeros((80, 128, 3), dtype=np.uint8)
        canvas[4:76] = frame
        padded.append(canvas)
    lossless_video(root / "prediction_padded.mp4", padded, 12)
    rows = [{"id": "demo_001", "model": "identity24", "prediction": "prediction24.mp4", "gt": "gt.mp4"}]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "padded.jsonl").write_text(json.dumps({"id": "demo_001", "prediction": "prediction_padded.mp4", "gt": "gt.mp4"}) + "\n")
    print(root / "manifest.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    make_demo(parser.parse_args().output_dir)
