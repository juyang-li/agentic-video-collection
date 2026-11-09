#!/usr/bin/env python3
"""Validate, checksum, and ZIP the immutable human-review package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


FIELDS = ("instruction_correctness", "logical_correctness", "content_preservation",
          "temporal_consistency", "visual_quality", "technical_quality", "overall_decision")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_review.py PACKAGE")
    package = Path(sys.argv[1]).resolve()
    archive = package.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 60 or len({row["review_id"] for row in rows}) != 60:
        raise RuntimeError("expected 60 unique review units")
    if any(row[field] != "1" for row in rows for field in FIELDS):
        raise RuntimeError("review defaults must be 1")
    report = json.loads((package / "browser_validation.json").read_text())
    if report.get("status") != "passed" or report.get("cards") != 30:
        raise RuntimeError("browser validation missing")
    inventory = json.loads((package / "media_inventory.json").read_text())
    if len(inventory) != 254 or any(not item["full_decode_ok"] for item in inventory):
        raise RuntimeError("invalid media inventory")
    summary = json.loads((package / "summary.json").read_text())
    if summary.get("gemini_included") is not False or summary.get("generated_outputs") != 224:
        raise RuntimeError("summary contract mismatch")
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(package).as_posix()}"
        for path in sorted(package.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
    ]
    (package / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(package.parent).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        if {name.split("/")[0] for name in zipped.namelist()} != {package.name}:
            raise RuntimeError("ZIP must have one top-level directory")
        videos = sum(name.endswith(".mp4") for name in zipped.namelist())
    if videos != 254:
        raise RuntimeError(f"expected 254 videos, found {videos}")
    receipt = {
        "status": "complete", "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": sha256(archive), "cards": 30, "review_units": 60, "videos": videos,
        "generated_outputs": 224, "zip_crc_verified": True,
        "single_top_level_directory": True, "browser_validation": "passed",
        "gemini_included": False, "remote_upload_status": "not_uploaded",
    }
    archive.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
