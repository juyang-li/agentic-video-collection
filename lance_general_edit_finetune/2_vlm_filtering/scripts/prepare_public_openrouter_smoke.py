#!/usr/bin/env python3
"""Create a one-row API smoke manifest from VEFX's public demo pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MEDIA = ROOT / "vendor/VEFX-Bench/examples/sample_videos"
DEFAULT_OUTPUT = ROOT / "artifacts/model_bakeoff_v1/public_vefx_smoke.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    row = {
        "sample_id": "public-vefx-object-removal-demo",
        "dataset": "vefx_public_demo",
        "branch": "object_removal",
        "operation": "remove",
        "target_scope": "person",
        "instruction": (
            "Remove the woman with the grey backpack walking on the right side "
            "of the frame."
        ),
        "source_local": str((PUBLIC_MEDIA / "object_removal_original.mp4").resolve()),
        "target_local": str((PUBLIC_MEDIA / "object_removal_edited.mp4").resolve()),
        "source_provenance": "original",
        "target_provenance": "edited",
        "stage1_1_decision": "not_applicable_public_demo",
        "human_status": "unlabeled",
        "cohort": "public_api_smoke",
    }
    for field in ("source_local", "target_local"):
        if not Path(row[field]).is_file():
            raise FileNotFoundError(row[field])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
