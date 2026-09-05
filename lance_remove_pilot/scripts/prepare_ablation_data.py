#!/usr/bin/env python3
"""Extract, decode-check, index, and export the two ablation datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import prepare_smoke_data as common


PILOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALL_MANIFEST = PILOT_ROOT / "experiment_manifest_700.csv"
DEFAULT_MIXED_MANIFEST = PILOT_ROOT / "mixed_300_200_manifest.csv"
DEFAULT_CONTROL_MANIFEST = PILOT_ROOT / "insave_500_manifest.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def media_path(row: dict[str, str], role: str) -> Path:
    return PILOT_ROOT / "media" / row["dataset"] / f"{row['pair_id']}_{role}.mp4"


def write_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    records = []
    for row in sorted(rows, key=lambda item: item["pair_id"]):
        records.append(
            {
                "caption": row["instruction"],
                "input_video_bytes": media_path(row, "source").read_bytes(),
                "output_video_bytes": media_path(row, "target").read_bytes(),
            }
        )
    table = pa.Table.from_pylist(records, schema=common.PARQUET_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def verify_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    table = pq.read_table(path)
    if table.schema != common.PARQUET_SCHEMA:
        raise ValueError(f"unexpected schema: {path}")
    expected = sorted(rows, key=lambda item: item["pair_id"])
    actual = table.to_pylist()
    if len(actual) != len(expected):
        raise ValueError(f"row count mismatch: {path}")
    for row, record in zip(expected, actual):
        if record["caption"] != row["instruction"]:
            raise ValueError(f"caption mismatch: {row['pair_id']}")
        for role, field in (("source", "input_video_bytes"), ("target", "output_video_bytes")):
            expected_hash = common.sha256_file(media_path(row, role))
            actual_hash = hashlib.sha256(record[field]).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"{role} bytes mismatch: {row['pair_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-manifest", type=Path, default=DEFAULT_ALL_MANIFEST)
    parser.add_argument("--mixed-manifest", type=Path, default=DEFAULT_MIXED_MANIFEST)
    parser.add_argument("--control-manifest", type=Path, default=DEFAULT_CONTROL_MANIFEST)
    parser.add_argument("--insave-tar", type=Path, required=True)
    parser.add_argument("--goku-tar", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    all_rows = read_rows(args.all_manifest)
    mixed_rows = read_rows(args.mixed_manifest)
    control_rows = read_rows(args.control_manifest)
    if len(all_rows) != 700 or len(mixed_rows) != 500 or len(control_rows) != 500:
        raise ValueError("expected 700 all, 500 mixed, and 500 control rows")
    mixed_insave = [row for row in mixed_rows if row["dataset"] == "insave"]
    mixed_goku = [row for row in mixed_rows if row["dataset"] == "goku"]
    if len(mixed_insave) != 300 or len(mixed_goku) != 200:
        raise ValueError("mixed manifest is not 300 InsAVE + 200 GOKU")

    outputs = {
        "insave_500": PILOT_ROOT / "exports" / "insave_500.parquet",
        "insave_300": PILOT_ROOT / "exports" / "insave_300.parquet",
        "goku_200": PILOT_ROOT / "exports" / "goku_200.parquet",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.rebuild:
        raise FileExistsError(f"refusing to overwrite: {existing}; pass --rebuild")

    common.extract_media(
        all_rows,
        {"insave": args.insave_tar, "goku": args.goku_tar},
        PILOT_ROOT,
    )
    inventory, lookup = common.build_inventory(all_rows, PILOT_ROOT)
    common.write_inventory(PILOT_ROOT / "exports" / "technical_inventory_700.csv", inventory)

    next_db = PILOT_ROOT / "removal.sqlite.next"
    common.initialize_database(
        next_db,
        PILOT_ROOT / "schema.sql",
        all_rows,
        lookup,
        rebuild=args.rebuild,
    )
    next_db.replace(PILOT_ROOT / "removal.sqlite")

    write_parquet(outputs["insave_500"], control_rows)
    write_parquet(outputs["insave_300"], mixed_insave)
    write_parquet(outputs["goku_200"], mixed_goku)
    verify_parquet(outputs["insave_500"], control_rows)
    verify_parquet(outputs["insave_300"], mixed_insave)
    verify_parquet(outputs["goku_200"], mixed_goku)
    print(
        "prepared pairs=700 decoded_videos=1400 "
        + " ".join(f"{name}={path}" for name, path in outputs.items())
    )


if __name__ == "__main__":
    main()
