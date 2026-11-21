#!/usr/bin/env python3
"""Internalize every canonical 11K media reference without copying payloads.

External files are SHA-256 checked, hard-linked into media/{dataset}/, checked
again through the new path, and only then are the three canonical manifests
rewritten atomically.  Re-running the script is safe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
MANIFEST_JSONL = BASE / "manifest.jsonl"
MANIFEST_CSV = BASE / "manifest.csv"
CROPS_CSV = BASE / "crops.csv"
ROLES = ("source", "target", "reference")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.internalize.tmp")
    temp.write_text(text, encoding="utf-8", newline="")
    os.replace(temp, path)


def load_rows() -> list[dict]:
    with MANIFEST_JSONL.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def intended_path(row: dict, role: str) -> str:
    suffix = Path(row[role]).suffix.lower()
    if not suffix:
        raise ValueError(f"missing extension: {row['sample_id']} {role}")
    return f"media/{row['dataset']}/{row['sample_id']}_{role}{suffix}"


def internalize(rows: list[dict]) -> list[dict]:
    migrated = []
    for row in rows:
        for role in ROLES:
            old_value = row.get(role)
            if not old_value or old_value.startswith("media/"):
                continue
            expected = row.get(f"{role}_sha256")
            if not expected:
                raise ValueError(f"missing hash: {row['sample_id']} {role}")
            source = (BASE / old_value).resolve()
            new_value = intended_path(row, role)
            target = BASE / new_value
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hash = sha256(source)
            if source_hash != expected:
                raise ValueError(
                    f"source hash mismatch: {row['sample_id']} {role}: "
                    f"{source_hash} != {expected}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target_hash = sha256(target)
                if target_hash != expected:
                    raise ValueError(
                        f"target collision: {target}: {target_hash} != {expected}"
                    )
            else:
                os.link(source, target)
            if sha256(target) != expected:
                raise ValueError(f"post-link hash mismatch: {target}")
            migrated.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "role": role,
                    "old_path": old_value,
                    "new_path": new_value,
                    "sha256": expected,
                    "bytes": target.stat().st_size,
                }
            )
            row[role] = new_value
    return migrated


def write_jsonl(rows: list[dict]) -> None:
    atomic_write(
        MANIFEST_JSONL,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def rewrite_csv(path: Path, path_by_sample: dict[str, dict[str, str]]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
    for row in rows:
        canonical = path_by_sample[row["sample_id"]]
        for role in ROLES:
            if role in row and row.get(role):
                row[role] = canonical.get(role, row[role])
    temp = path.with_name(f".{path.name}.internalize.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def validate(rows: list[dict]) -> dict:
    if len(rows) != 11_000:
        raise ValueError(f"expected 11000 rows, got {len(rows)}")
    splits = Counter(row["split"] for row in rows)
    expected_splits = Counter(train=10_000, validation=500, test=500)
    if splits != expected_splits:
        raise ValueError(f"split mismatch: {splits}")
    media_fields = 0
    for row in rows:
        for role in ROLES:
            value = row.get(role)
            if not value:
                continue
            media_fields += 1
            if not value.startswith("media/"):
                raise ValueError(f"external path remains: {row['sample_id']} {role} {value}")
            path = BASE / value
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = row.get(f"{role}_sha256")
            if expected and sha256(path) != expected:
                raise ValueError(f"final hash mismatch: {row['sample_id']} {role}")
    if media_fields != 24_200:
        raise ValueError(f"expected 24200 media fields, got {media_fields}")
    return {"rows": len(rows), "splits": dict(splits), "media_fields": media_fields}


def main() -> None:
    rows = load_rows()
    migrated = internalize(rows)
    paths = {
        row["sample_id"]: {role: row.get(role) for role in ROLES}
        for row in rows
    }
    write_jsonl(rows)
    rewrite_csv(MANIFEST_CSV, paths)
    rewrite_csv(CROPS_CSV, paths)
    validation = validate(load_rows())
    receipt = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "method": "same-filesystem hard links",
        "migrated_count": len(migrated),
        "migrated_bytes_logical": sum(item["bytes"] for item in migrated),
        "validation": validation,
        "files": migrated,
    }
    receipt_path = BASE / "metadata" / "media_internalization_20261112.json"
    atomic_write(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
