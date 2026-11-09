#!/usr/bin/env python3
"""Bind a passed browser report, validate package contents, and create its immutable ZIP."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


FIELDS = ("instruction_correctness", "logical_correctness", "content_preservation", "temporal_consistency",
          "visual_quality", "technical_quality", "overall_decision", "notes")


def sha256(path: Path) -> str:
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    if len(sys.argv) != 2: raise SystemExit("usage: finalize_review.py PACKAGE_DIRECTORY")
    package = Path(sys.argv[1]).resolve(); archive = package.with_suffix(".zip")
    if archive.exists(): raise FileExistsError(f"refusing to overwrite {archive}")
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    count = len(rows)
    if count not in (24, 48) or len({r["eval_id"] for r in rows}) != count: raise RuntimeError("invalid aligned rows")
    if len({r["category"] for r in rows}) != 24: raise RuntimeError("expected all 24 categories")
    if any(r[f] for r in rows for f in FIELDS): raise RuntimeError("review labels must initially be blank")
    report = json.loads((package / "browser_validation.json").read_text())
    if report.get("status") != "passed" or report.get("sample_count") != count: raise RuntimeError("browser validation not passed")
    checksum_lines = [f"{sha256(p)}  {p.relative_to(package).as_posix()}" for p in sorted(package.rglob("*"))
                      if p.is_file() and p.name != "SHA256SUMS"]
    (package / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for line in (package / "SHA256SUMS").read_text().splitlines():
        expected, rel = line.split("  ", 1)
        if sha256(package / rel) != expected: raise RuntimeError(f"checksum mismatch: {rel}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted(package.rglob("*")):
            if path.is_file(): z.write(path, path.relative_to(package.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError("ZIP CRC failure")
        if {n.split("/")[0] for n in z.namelist()} != {package.name}: raise RuntimeError("ZIP must have one top-level directory")
        videos = sum(n.endswith(".mp4") for n in z.namelist())
    receipt = {"status": "complete", "archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "samples": count, "videos": videos, "zip_crc_verified": True,
               "single_top_level_directory": True, "browser_validation": "passed", "remote_upload_status": "not_uploaded"}
    archive.with_suffix(".sha256.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
