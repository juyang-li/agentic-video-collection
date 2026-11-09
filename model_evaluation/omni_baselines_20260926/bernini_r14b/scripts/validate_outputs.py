#!/usr/bin/env python3
"""Validate Bernini outputs and finalize the generation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path, ffprobe: str) -> dict:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,pix_fmt:format=duration",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    good = 0
    for record in records:
        output = Path(record["output_path"])
        record["model_revision"] = args.model_revision
        record["code_commit"] = args.code_commit
        try:
            metadata = probe(output, args.ffprobe)
            video = metadata["streams"][0]
            frames = int(video["nb_read_frames"])
            dimensions = (int(video["width"]), int(video["height"]))
            valid = (
                video["codec_name"] in {"h264", "hevc"}
                and dimensions in {(848, 480), (480, 848)}
                and frames == 81
            )
            record["output_sha256"] = sha256(output)
            record["probe"] = metadata
            record["status"] = "complete" if valid else "invalid_technical"
            good += int(valid)
        except Exception as exc:
            record["status"] = "missing" if not output.exists() else "probe_failed"
            record["error"] = repr(exc)
    with args.manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(records), "complete": good, "failed": len(records) - good}))


if __name__ == "__main__":
    main()
