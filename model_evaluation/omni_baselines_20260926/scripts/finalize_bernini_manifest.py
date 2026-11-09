#!/usr/bin/env python3
"""Validate Bernini Tier-A files with Decord and update provenance fields."""

import hashlib
import json
from pathlib import Path

from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "bernini_r14b/manifests/generation_manifest.jsonl"
records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


complete_a = 0
for record in records:
    record["model_revision"] = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c"
    record["code_commit"] = "e6c2cf11843470b08491745947abcba9c3c6b079"
    output = Path(record["output_path"])
    if not output.is_file():
        record["status"] = "missing"
        continue
    video = VideoReader(str(output), ctx=cpu(0))
    height, width = video[0].shape[:2]
    valid = len(video) == 81 and (width, height) in {(848, 480), (480, 848)}
    record["probe"] = {"frames": len(video), "width": width, "height": height}
    record["output_sha256"] = sha256(output)
    record["status"] = "complete" if valid else "invalid_technical"
    if record["tier"] == "A" and valid:
        complete_a += 1

if complete_a != 24:
    raise RuntimeError(f"expected 24 valid Tier-A outputs, found {complete_a}")
with manifest.open("w", encoding="utf-8") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps({"status": "complete", "tier_a": complete_a, "manifest": str(manifest)}))
