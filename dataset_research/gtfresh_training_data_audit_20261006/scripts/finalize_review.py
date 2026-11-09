#!/usr/bin/env python3
"""Validate, checksum, and ZIP the immutable dataset audit package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


FIELDS = ("person_relevance", "instruction_correctness", "logical_correctness",
          "content_preservation", "temporal_consistency", "visual_quality", "technical_quality")


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
    with (package / "audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        audits = list(csv.DictReader(stream))
    with (package / "inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        inventory_rows = list(csv.DictReader(stream))
    if len(audits) != 26 or len(inventory_rows) != 26:
        raise RuntimeError("expected 26 audit and inventory rows")
    if len({row["sample_id"] for row in audits}) != 26:
        raise RuntimeError("duplicate audit sample ID")
    if any(row[field] != "1" for row in audits for field in FIELDS):
        raise RuntimeError("audit defaults must be 1")
    report = json.loads((package / "browser_validation.json").read_text())
    if report.get("status") != "passed" or report.get("cards") != 26:
        raise RuntimeError("browser validation missing")
    media = json.loads((package / "media_inventory.json").read_text())
    if len(media) != 113 or any(not item["full_decode_ok"] for item in media):
        raise RuntimeError("expected 113 fully decoded media files")
    selection = json.loads((package / "selection_manifest.json").read_text())
    coverage = selection["coverage"]
    if coverage["ue_chains"] != 20 or coverage["ue_turns"] != 81 or coverage["coinve_samples"] != 6:
        raise RuntimeError("selection coverage mismatch")
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
            raise RuntimeError("ZIP must contain one top-level directory")
        videos = sum(name.endswith(".mp4") for name in zipped.namelist())
    if videos != 113:
        raise RuntimeError(f"expected 113 videos, found {videos}")
    receipt = {
        "status": "complete", "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": sha256(archive), "cards": 26, "audit_rows": 26, "videos": videos,
        "synthetic_chains": 20, "synthetic_turns": 81, "coinve_pairs": 6,
        "zip_crc_verified": True, "single_top_level_directory": True,
        "browser_validation": "passed", "automatic_scoring": False,
        "remote_upload_status": "not_uploaded",
    }
    archive.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
