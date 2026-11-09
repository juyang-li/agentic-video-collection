#!/usr/bin/env python3
"""Transcode an OpenVid offline review bundle to browser-compatible H.264."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

import prepare_openvid_review as review


TRANSCODE = "libx264_crf17_yuv420p_faststart_noaudio"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest_path = bundle / "review_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError("review manifest must contain exactly 50 rows")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for index, row in enumerate(rows, 1):
        path = bundle / str(row["video_path"])
        if row.get("review_transcode") == TRANSCODE:
            if sha256(path) != row.get("video_sha256"):
                raise RuntimeError(f"existing H.264 hash mismatch: {path}")
            continue
        original_hash = sha256(path)
        temporary = path.with_name(path.stem + ".h264.part.mp4")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-map", "0:v:0", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temporary),
        ]
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"transcode did not produce a video: {path}")
        temporary.replace(path)
        row["source_video_sha256"] = original_hash
        row["video_sha256"] = sha256(path)
        row["source_codec"] = "mpeg4"
        row["review_codec"] = "h264"
        row["review_transcode"] = TRANSCODE
        print(f"transcoded={index}/{len(rows)}", flush=True)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary_csv = (bundle / "review_manifest.csv").with_suffix(".part.csv")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary_csv.replace(bundle / "review_manifest.csv")
    temporary_json = manifest_path.with_suffix(".part.json")
    temporary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(manifest_path)
    review.write_html(bundle, rows, int(payload.get("selection_seed", 20260904)))
    review.finalize_existing(bundle, review.DEFAULT_URL, int(payload.get("selection_seed", 20260904)))


if __name__ == "__main__":
    main()
