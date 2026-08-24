#!/usr/bin/env python3
"""Download only the TripVVT assets referenced by random_pairs.csv."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
CACHE_ROOT = Path("/tmp/tripvvt_probe/assets")
REPO_ID = "TripVVT/TripVVT-10K"
REVISION = "596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3"


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 15:
        raise ValueError("manifest must contain 15 rows")

    fields = (
        ("source_member", "source_file"),
        ("target_member", "target_file"),
        ("cloth_member", "cloth_file"),
    )
    wanted = sorted({row[member_field] for row in rows for member_field, _ in fields})
    snapshot_root = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=wanted,
            local_dir=CACHE_ROOT,
            max_workers=8,
        )
    )

    copied = 0
    for row in rows:
        for member_field, file_field in fields:
            source = snapshot_root / row[member_field]
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = (ROOT / row[file_field]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_name(destination.name + ".part")
            shutil.copyfile(source, part)
            part.replace(destination)
            copied += 1
    print(f"requested_members={len(wanted)} copied_assets={copied}")


if __name__ == "__main__":
    main()
