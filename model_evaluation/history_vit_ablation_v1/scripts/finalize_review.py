#!/usr/bin/env python3
"""Validate, checksum, and archive the history-ViT review package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


PACKAGE_NAME = "real_person_history_vit_ablation_base_gtfresh_human_review_20261016"
MODELS = ("lance_base", "gtfresh_step1150")
CONDITIONS = (
    "vit_zero_text_keep",
    "vit_gaussian_text_keep",
    "vit_zero_text_zero",
    "vit_gaussian_text_zero",
)
FIELDS = (
    "edit_applied",
    "history_state_correct",
    "normal_equivalent",
    "identity_background_stable",
    "temporal_consistency",
    "visual_quality",
    "overall_pass",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_review.py PACKAGE")
    package = Path(sys.argv[1]).resolve()
    if package.name != PACKAGE_NAME:
        raise RuntimeError(f"unexpected package name: {package.name}")
    archive = package.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)

    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 192 or len({row["review_id"] for row in rows}) != 192:
        raise RuntimeError("expected 192 unique review units")
    if any(row[field] != "1" for row in rows for field in FIELDS):
        raise RuntimeError("review defaults must all be 1")
    cards = {row["card_id"] for row in rows}
    if len(cards) != 24:
        raise RuntimeError("expected 24 checkpoint cards")
    for card_id in cards:
        coverage = {(row["model_id"], row["condition"]) for row in rows if row["card_id"] == card_id}
        if coverage != {(model, condition) for model in MODELS for condition in CONDITIONS}:
            raise RuntimeError(f"review coverage mismatch: {card_id}")

    report = json.loads((package / "browser_validation.json").read_text(encoding="utf-8"))
    if (report.get("status"), report.get("cards"), report.get("review_units")) != (
            "passed", 24, 192):
        raise RuntimeError("browser validation missing or mismatched")
    if report.get("external_requests") or report.get("javascript_errors"):
        raise RuntimeError("browser validation was not cleanly offline")

    inventory = json.loads((package / "media_inventory.json").read_text(encoding="utf-8"))
    expected_roles = {"source": 12, "direct_source": 48, "normal_replay": 48, "ablation": 192}
    actual_roles = {role: sum(item.get("role") == role for item in inventory) for role in expected_roles}
    if len(inventory) != 300 or actual_roles != expected_roles:
        raise RuntimeError(f"media inventory shape mismatch: {actual_roles}")
    media_paths = [item.get("path") for item in inventory]
    if len(set(media_paths)) != 300:
        raise RuntimeError("expected 300 unique inventory paths")
    for item in inventory:
        path = package / item["path"]
        if (not path.is_file() or sha256(path) != item.get("sha256")
                or item.get("full_decode_ok") is not True
                or item.get("codec_name") != "h264"
                or item.get("pix_fmt") != "yuv420p"
                or item.get("faststart") is not True):
            raise RuntimeError(f"media contract failed: {path}")

    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    summary_contract = (
        summary.get("human_only") is True,
        summary.get("gemini_included") is False,
        summary.get("cards") == 24,
        summary.get("review_units") == 192,
        summary.get("media_files") == 300,
        summary.get("formal_ablation_outputs") == 192,
        summary.get("media_role_counts") == expected_roles,
    )
    if not all(summary_contract):
        raise RuntimeError("summary contract failed")

    if (package / "difference_metrics.csv").is_file():
        with (package / "difference_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            metric_rows = list(csv.DictReader(stream))
        if len(metric_rows) != 192:
            raise RuntimeError("difference_metrics.csv must contain 192 rows")
    if (package / "inference_audit.json").is_file():
        audit = json.loads((package / "inference_audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "passed" or audit.get("metric_rows") != 192:
            raise RuntimeError("inference_audit.json contract failed")

    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(package).as_posix()}"
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
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
            raise RuntimeError("ZIP must contain exactly one top-level directory")
        videos = sum(name.endswith(".mp4") for name in zipped.namelist())
    if videos != 300:
        raise RuntimeError(f"expected 300 videos, found {videos}")

    receipt = {
        "status": "complete",
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "cards": 24,
        "review_units": 192,
        "videos": 300,
        "formal_ablation_outputs": 192,
        "zip_crc_verified": True,
        "single_top_level_directory": True,
        "browser_validation": "passed",
        "gemini_included": False,
        "remote_upload_status": "not_uploaded",
    }
    archive.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
