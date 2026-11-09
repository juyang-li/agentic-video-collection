#!/usr/bin/env python3
"""Create the package-builder manifest from completed VACE Tier-A outputs."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


with (ROOT / "aligned_subset.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["tier"] == "A"]

records = []
for row in rows:
    eval_id = row["eval_id"]
    mask_meta_path = ROOT / "vace/masks_dino" / eval_id / "mask_manifest.json"
    mask_meta = json.loads(mask_meta_path.read_text(encoding="utf-8"))
    output = ROOT / "vace/outputs_tier_a/outputs" / f"{int(row['aligned_index']):03d}_{eval_id}.mp4"
    mask = Path(mask_meta["mask_video"])
    overlay = Path(mask_meta["overlay_video"])
    for path in (output, mask, overlay):
        if not path.is_file():
            raise FileNotFoundError(path)
    records.append({
        "eval_id": eval_id,
        "tier": "A",
        "aligned_index": int(row["aligned_index"]),
        "model_id": "Wan-AI/Wan2.1-VACE-14B",
        "model_revision": "huggingface_snapshot_2026-09-27",
        "code_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "prompt": row["editing_instruction"],
        "seed": 42,
        "output_path": str(output),
        "output_sha256": sha256(output),
        "mask_path": str(mask),
        "mask_sha256": sha256(mask),
        "overlay_path": str(overlay),
        "overlay_sha256": sha256(overlay),
        "mask_method": mask_meta["mask_method"],
        "mask_qc": mask_meta["mask_qc"],
        "status": "complete",
    })

manifest = ROOT / "vace/generation_manifest.jsonl"
with manifest.open("w", encoding="utf-8") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps({"status": "complete", "rows": len(records), "manifest": str(manifest)}))
