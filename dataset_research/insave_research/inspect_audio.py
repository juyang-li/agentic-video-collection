#!/usr/bin/env python3
"""Record objective source/target audio properties for the sampled pairs."""

from __future__ import annotations

from array import array
import argparse
import csv
import hashlib
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
OUTPUT_PATH = ROOT / "insave_audio_inventory.csv"
SAMPLE_RATE = 16_000


def decode_audio(path: Path) -> bytes:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"audio decode failed for {path}: {message}")
    return result.stdout


def rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    rows: list[dict[str, object]] = []
    for item in manifest:
        source_pcm = decode_audio(ROOT / item["source_file"])
        target_pcm = decode_audio(ROOT / item["target_file"])
        rows.append(
            {
                "sample_rank": item["sample_rank"],
                "sample_id": item["sample_id"],
                "source_has_audio": bool(source_pcm),
                "target_has_audio": bool(target_pcm),
                "source_pcm_bytes": len(source_pcm),
                "target_pcm_bytes": len(target_pcm),
                "source_duration_seconds": round(
                    len(source_pcm) / (2 * SAMPLE_RATE), 4
                ),
                "target_duration_seconds": round(
                    len(target_pcm) / (2 * SAMPLE_RATE), 4
                ),
                "source_rms": round(rms(source_pcm), 2),
                "target_rms": round(rms(target_pcm), 2),
                "source_pcm_sha256": hashlib.sha256(source_pcm).hexdigest(),
                "target_pcm_sha256": hashlib.sha256(target_pcm).hexdigest(),
                "pcm_exactly_equal": source_pcm == target_pcm,
            }
        )
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    equal = sum(row["pcm_exactly_equal"] is True for row in rows)
    print(f"rows={len(rows)} pcm_exactly_equal={equal}")


if __name__ == "__main__":
    main()
