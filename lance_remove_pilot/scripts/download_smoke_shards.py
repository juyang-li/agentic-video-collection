#!/usr/bin/env python3
"""Download only the two official shards required by the fixed smoke manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from huggingface_hub import hf_hub_download


PILOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PILOT_ROOT / "smoke_manifest.csv"
DEFAULT_OUTPUT = PILOT_ROOT / "media" / "_shards"


def load_downloads(manifest: Path) -> dict[str, dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "repo_id", "revision", "shard"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"unexpected manifest schema: {manifest}")
    downloads: dict[str, dict[str, str]] = {}
    for row in rows:
        record = {
            "repo_id": row["repo_id"],
            "revision": row["revision"],
            "filename": row["shard"],
        }
        previous = downloads.setdefault(row["dataset"], record)
        if previous != record:
            raise ValueError(f"multiple shard definitions for {row['dataset']}")
    return downloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset", choices=["all", "insave", "goku"], default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    downloads = load_downloads(args.manifest)
    selected = sorted(downloads) if args.dataset == "all" else [args.dataset]
    for dataset in selected:
        if dataset not in downloads:
            raise ValueError(f"dataset not present in manifest: {dataset}")
        record = downloads[dataset]
        expected = args.output / record["filename"]
        print(
            f"dataset={dataset} repo={record['repo_id']} "
            f"revision={record['revision']} filename={record['filename']} "
            f"destination={expected}"
        )
        if args.dry_run:
            continue
        resolved = hf_hub_download(
            repo_id=record["repo_id"],
            filename=record["filename"],
            repo_type="dataset",
            revision=record["revision"],
            local_dir=args.output,
        )
        print(f"downloaded={Path(resolved).resolve()}")


if __name__ == "__main__":
    main()
