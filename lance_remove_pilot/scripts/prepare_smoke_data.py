#!/usr/bin/env python3
"""Extract smoke media, validate pairs, build SQLite, and export Lance parquet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sqlite3
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import pyarrow as pa
import pyarrow.parquet as pq


PILOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PILOT_ROOT / "smoke_manifest.csv"
DEFAULT_SCHEMA = PILOT_ROOT / "schema.sql"
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("caption", pa.string(), nullable=False),
        pa.field("input_video_bytes", pa.binary(), nullable=False),
        pa.field("output_video_bytes", pa.binary(), nullable=False),
    ]
)
INVENTORY_FIELDS = [
    "pair_id",
    "dataset",
    "role",
    "path",
    "sha256",
    "bytes",
    "width",
    "height",
    "fps",
    "reported_frames",
    "decoded_frames",
    "duration_seconds",
    "decode_ok",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_id",
        "dataset",
        "repo_id",
        "revision",
        "official_id",
        "instruction",
        "source_member",
        "target_member",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"unexpected manifest schema: {path}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pair_id in manifest")
    return rows


def normalized_member_map(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if member.isfile():
            result[member.name.removeprefix("./")] = member
    return result


def extract_member(
    archive: tarfile.TarFile,
    member_map: dict[str, tarfile.TarInfo],
    member_name: str,
    destination: Path,
) -> None:
    normalized = member_name.removeprefix("./")
    member = member_map.get(normalized)
    if member is None:
        raise FileNotFoundError(f"tar member not found: {normalized}")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"cannot read tar member: {normalized}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    temporary.replace(destination)


def extract_media(
    rows: list[dict[str, str]],
    tar_paths: dict[str, Path],
    pilot_root: Path,
) -> None:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    missing_tar_keys = sorted(set(by_dataset) - set(tar_paths))
    if missing_tar_keys:
        raise ValueError(f"missing tar arguments for: {missing_tar_keys}")

    for dataset, dataset_rows in sorted(by_dataset.items()):
        tar_path = tar_paths[dataset].resolve()
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path, "r:*") as archive:
            member_map = normalized_member_map(archive)
            for row in dataset_rows:
                for role, member_key in (
                    ("source", "source_member"),
                    ("target", "target_member"),
                ):
                    destination = (
                        pilot_root
                        / "media"
                        / dataset
                        / f"{row['pair_id']}_{role}.mp4"
                    )
                    extract_member(archive, member_map, row[member_key], destination)


def probe_video(path: Path) -> dict[str, str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "width": "0",
            "height": "0",
            "fps": "0",
            "reported_frames": "0",
            "decoded_frames": "0",
            "duration_seconds": "0",
            "decode_ok": "0",
        }
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            break
        decoded_frames += 1
    capture.release()
    duration = decoded_frames / fps if fps > 0 else 0.0
    decode_ok = int(
        width > 0
        and height > 0
        and fps > 0
        and decoded_frames > 0
        and (reported_frames <= 0 or decoded_frames == reported_frames)
    )
    return {
        "width": str(width),
        "height": str(height),
        "fps": f"{fps:.6f}",
        "reported_frames": str(reported_frames),
        "decoded_frames": str(decoded_frames),
        "duration_seconds": f"{duration:.6f}",
        "decode_ok": str(decode_ok),
    }


def build_inventory(
    rows: list[dict[str, str]], pilot_root: Path
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    inventory: list[dict[str, str]] = []
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        for role in ("source", "target"):
            path = (
                pilot_root
                / "media"
                / row["dataset"]
                / f"{row['pair_id']}_{role}.mp4"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            probe = probe_video(path)
            item = {
                "pair_id": row["pair_id"],
                "dataset": row["dataset"],
                "role": role,
                "path": path.relative_to(pilot_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": str(path.stat().st_size),
                **probe,
            }
            inventory.append(item)
            lookup[(row["pair_id"], role)] = item

        source = lookup[(row["pair_id"], "source")]
        target = lookup[(row["pair_id"], "target")]
        if source["decode_ok"] != "1" or target["decode_ok"] != "1":
            raise ValueError(f"decode failed for {row['pair_id']}")
        if source["sha256"] == target["sha256"]:
            raise ValueError(f"source and target are identical for {row['pair_id']}")
        for field in ("width", "height", "fps", "decoded_frames"):
            if source[field] != target[field]:
                raise ValueError(
                    f"unaligned pair {row['pair_id']}: {field} "
                    f"source={source[field]} target={target[field]}"
                )
    return inventory, lookup


def write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def initialize_database(
    db_path: Path,
    schema_path: Path,
    rows: list[dict[str, str]],
    media_lookup: dict[tuple[str, str], dict[str, str]],
    rebuild: bool,
) -> None:
    if db_path.exists():
        if not rebuild:
            raise FileExistsError(f"database exists; pass --rebuild: {db_path}")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        dataset_rows: dict[str, tuple[str, str]] = {}
        for row in rows:
            identity = (row["repo_id"], row["revision"])
            previous = dataset_rows.setdefault(row["dataset"], identity)
            if previous != identity:
                raise ValueError(f"inconsistent metadata for {row['dataset']}")
        dataset_ids: dict[str, int] = {}
        for dataset_id, dataset in enumerate(sorted(dataset_rows), start=1):
            repo_id, revision = dataset_rows[dataset]
            connection.execute(
                "INSERT INTO datasets(dataset_id, name, repo_id, revision) VALUES (?, ?, ?, ?)",
                (dataset_id, dataset, repo_id, revision),
            )
            dataset_ids[dataset] = dataset_id
        for row in rows:
            connection.execute(
                """
                INSERT INTO pairs(
                    pair_id, dataset_id, official_id, instruction, source_path, target_path
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["pair_id"],
                    dataset_ids[row["dataset"]],
                    row["official_id"],
                    row["instruction"],
                    media_lookup[(row["pair_id"], "source")]["path"],
                    media_lookup[(row["pair_id"], "target")]["path"],
                ),
            )
        connection.commit()


def export_parquets(db_path: Path, pilot_root: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        datasets = connection.execute(
            "SELECT dataset_id, name FROM datasets ORDER BY name"
        ).fetchall()
        for dataset in datasets:
            pairs = connection.execute(
                """
                SELECT pair_id, instruction, source_path, target_path
                FROM pairs WHERE dataset_id = ? ORDER BY pair_id
                """,
                (dataset["dataset_id"],),
            ).fetchall()
            records = []
            for pair in pairs:
                source_path = pilot_root / pair["source_path"]
                target_path = pilot_root / pair["target_path"]
                records.append(
                    {
                        "caption": pair["instruction"],
                        "input_video_bytes": source_path.read_bytes(),
                        "output_video_bytes": target_path.read_bytes(),
                    }
                )
            table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
            output = pilot_root / "exports" / f"{dataset['name']}_smoke.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output, compression="zstd")
            outputs[dataset["name"]] = output
    return outputs


def verify_outputs(
    db_path: Path,
    outputs: dict[str, Path],
    pilot_root: Path,
    expected_total: int,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        pair_count = connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        if pair_count != expected_total:
            raise ValueError(f"database rows={pair_count}, expected={expected_total}")
        for dataset, parquet_path in sorted(outputs.items()):
            table = pq.read_table(parquet_path)
            if table.schema != PARQUET_SCHEMA:
                raise ValueError(f"unexpected parquet schema: {parquet_path}")
            db_rows = connection.execute(
                """
                SELECT p.pair_id, p.instruction, p.source_path, p.target_path
                FROM pairs p JOIN datasets d ON p.dataset_id = d.dataset_id
                WHERE d.name = ? ORDER BY p.pair_id
                """,
                (dataset,),
            ).fetchall()
            py_rows = table.to_pylist()
            if len(py_rows) != len(db_rows):
                raise ValueError(f"parquet row mismatch for {dataset}")
            for db_row, parquet_row in zip(db_rows, py_rows):
                if parquet_row["caption"] != db_row["instruction"]:
                    raise ValueError(f"caption mismatch: {db_row['pair_id']}")
                if hashlib.sha256(parquet_row["input_video_bytes"]).hexdigest() != sha256_file(
                    pilot_root / db_row["source_path"]
                ):
                    raise ValueError(f"input bytes mismatch: {db_row['pair_id']}")
                if hashlib.sha256(parquet_row["output_video_bytes"]).hexdigest() != sha256_file(
                    pilot_root / db_row["target_path"]
                ):
                    raise ValueError(f"output bytes mismatch: {db_row['pair_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--insave-tar", type=Path, required=True)
    parser.add_argument("--goku-tar", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    pilot_root = args.pilot_root.resolve()
    rows = load_manifest(args.manifest)
    extract_media(
        rows,
        {"insave": args.insave_tar, "goku": args.goku_tar},
        pilot_root,
    )
    inventory, lookup = build_inventory(rows, pilot_root)
    inventory_path = pilot_root / "exports" / "technical_inventory.csv"
    write_inventory(inventory_path, inventory)
    db_path = pilot_root / "removal.sqlite"
    initialize_database(db_path, args.schema, rows, lookup, args.rebuild)
    outputs = export_parquets(db_path, pilot_root)
    verify_outputs(db_path, outputs, pilot_root, len(rows))
    print(
        f"prepared pairs={len(rows)} db={db_path} inventory={inventory_path} "
        + " ".join(f"{name}={path}" for name, path in sorted(outputs.items()))
    )


if __name__ == "__main__":
    main()
