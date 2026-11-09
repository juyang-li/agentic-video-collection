#!/usr/bin/env python3
"""Validate, checksum, and archive the drift review package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


FIELDS = ("edit_applied", "history_state_restored", "no_residual",
          "intervening_edits_preserved", "identity_background_stable",
          "temporal_consistency", "visual_quality", "overall_drift")


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
    if len(rows) != 72 or len({row["review_id"] for row in rows}) != 72:
        raise RuntimeError("expected 72 unique review units")
    if any(row[field] != "1" for row in rows for field in FIELDS):
        raise RuntimeError("review defaults must all be 1")
    if len({row["eval_id"] for row in rows}) != 12:
        raise RuntimeError("expected 12 review cards")
    report = json.loads((package / "browser_validation.json").read_text())
    if report.get("status") != "passed" or report.get("cards") != 12 or report.get("review_units") != 72:
        raise RuntimeError("browser validation missing")
    inventory = json.loads((package / "media_inventory.json").read_text())
    if len(inventory) != 156 or any(not item["full_decode_ok"] for item in inventory):
        raise RuntimeError("media inventory contract failed")
    summary = json.loads((package / "summary.json").read_text())
    if summary.get("gemini_included") is not False or summary.get("generated_outputs") != 144:
        raise RuntimeError("summary contract failed")
    checksum_lines = [f"{sha256(path)}  {path.relative_to(package).as_posix()}"
                      for path in sorted(package.rglob("*"))
                      if path.is_file() and path.name != "SHA256SUMS"]
    (package / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(package.parent).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        if {name.split("/")[0] for name in zipped.namelist()} != {package.name}:
            raise RuntimeError("ZIP must contain exactly one top-level directory")
        videos = sum(name.endswith(".mp4") for name in zipped.namelist())
    if videos != 156:
        raise RuntimeError(f"expected 156 videos, found {videos}")
    receipt = {"status": "complete", "archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "cards": 12, "review_units": 72, "videos": 156,
               "generated_outputs": 144, "zip_crc_verified": True,
               "single_top_level_directory": True, "browser_validation": "passed",
               "gemini_included": False, "remote_upload_status": "not_uploaded"}
    archive.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
