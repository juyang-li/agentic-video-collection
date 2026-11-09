#!/usr/bin/env python3
"""Validate, checksum, and archive the IVEBench History-ViT review package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


PACKAGE_NAME = "ivebench24_nonrollback_nocamera_history_vit_ablation_base_gtfresh_human_review_20261017"
MODELS = ("lance_base", "gtfresh_step1150")
CONDITIONS = ("vit_zero_text_keep", "vit_gaussian_text_keep")
FIELDS = (
    "edit_applied", "history_preserved", "normal_equivalent",
    "identity_background_stable", "temporal_consistency", "visual_quality", "overall_pass",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if len(rows) != 164 or len({row["review_id"] for row in rows}) != 164:
        raise RuntimeError("expected 164 unique review units")
    if any(row[field] != "1" for row in rows for field in FIELDS):
        raise RuntimeError("review defaults must all be 1")
    cards = {row["card_id"] for row in rows}
    if len(cards) != 41:
        raise RuntimeError("expected 41 checkpoint cards")
    expected_coverage = {(model, condition) for model in MODELS for condition in CONDITIONS}
    for card_id in cards:
        coverage = {(row["model_id"], row["condition"]) for row in rows if row["card_id"] == card_id}
        if coverage != expected_coverage:
            raise RuntimeError(f"review coverage mismatch: {card_id}: {coverage}")

    browser = load_json(package / "browser_validation.json")
    if (
        browser.get("status"), browser.get("cards"), browser.get("review_units"),
        browser.get("videos_per_card"),
    ) != ("passed", 41, 164, 15):
        raise RuntimeError("browser validation missing or mismatched")
    if browser.get("external_requests") or browser.get("javascript_errors"):
        raise RuntimeError("browser validation was not cleanly offline")

    inventory = load_json(package / "media_inventory.json")
    if len(inventory) != 318 or len({item.get("path") for item in inventory}) != 318:
        raise RuntimeError("expected 318 unique physical media files")
    role_counts = {
        role: sum(any(usage.get("role") == role for usage in item.get("usages", [])) for item in inventory)
        for role in ("source", "normal", "ablation")
    }
    if role_counts != {"source": 24, "normal": 130, "ablation": 164}:
        raise RuntimeError(f"media inventory role mismatch: {role_counts}")
    if len({item.get("source_sha256") for item in inventory}) != 318:
        raise RuntimeError("media inventory is not content-deduplicated")
    for item in inventory:
        if any(usage.get("role") == "gt" for usage in item.get("usages", [])):
            raise RuntimeError("GT media must not be included")
        path = package / item["path"]
        contract = (
            path.is_file(), sha256(path) == item.get("sha256"), item.get("full_decode_ok") is True,
            item.get("codec_name") == "h264", item.get("pix_fmt") == "yuv420p",
            item.get("faststart") is True,
        )
        if not all(contract):
            raise RuntimeError(f"media contract failed: {path}: {contract}")

    summary = load_json(package / "summary.json")
    summary_contract = (
        summary.get("human_only") is True, summary.get("gemini_included") is False,
        summary.get("gt_videos") == 0, summary.get("chains") == 24,
        summary.get("turns") == 65, summary.get("cards") == 41,
        summary.get("review_units") == 164, summary.get("media_files") == 318,
        summary.get("normal_outputs") == 130, summary.get("formal_ablation_outputs") == 164,
        summary.get("difference_metric_rows") == 164, summary.get("media_role_counts") == role_counts,
    )
    if not all(summary_contract):
        raise RuntimeError(f"summary contract failed: {summary_contract}")

    with (package / "generation_manifest.jsonl").open(encoding="utf-8") as stream:
        generation = [json.loads(line) for line in stream if line.strip()]
    generation_keys = {
        (row["model_id"], row["condition"], row["chain_id"], row["round"]) for row in generation
    }
    if len(generation) != 294 or len(generation_keys) != 294:
        raise RuntimeError("generation manifest must contain 294 unique outputs")
    if sum(row["condition"] == "normal" for row in generation) != 130:
        raise RuntimeError("generation manifest normal coverage mismatch")

    with (package / "difference_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    metric_keys = {(row["model_id"], row["condition"], row["chain_id"], row["round"]) for row in metrics}
    if len(metrics) != 164 or len(metric_keys) != 164:
        raise RuntimeError("difference_metrics.csv must contain 164 unique rows")
    audit = load_json(package / "inference_audit.json")
    if (audit.get("status"), audit.get("metric_rows"), audit.get("formal_ablation_videos")) != (
        "passed", 164, 164
    ):
        raise RuntimeError("inference_audit.json contract failed")

    required = {
        "index.html", "README.md", "benchmark.json", "review_manifest.csv",
        "generation_manifest.jsonl", "media_inventory.json", "difference_metrics.csv",
        "inference_audit.json", "browser_validation.json", "summary.json",
    }
    if not required.issubset({path.name for path in package.iterdir() if path.is_file()}):
        raise RuntimeError("package is missing required top-level files")
    if any("gemini" in path.name.lower() or "gt_target" in path.name.lower() for path in package.rglob("*")):
        raise RuntimeError("forbidden GT/Gemini artifact included")

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
    if videos != 318:
        raise RuntimeError(f"expected 318 videos, found {videos}")

    receipt = {
        "status": "complete", "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": sha256(archive), "chains": 24, "turns": 65, "cards": 41,
        "review_units": 164, "videos": 318, "gt_videos": 0,
        "normal_outputs": 130, "formal_ablation_outputs": 164,
        "difference_metric_rows": 164, "zip_crc_verified": True,
        "single_top_level_directory": True, "browser_validation": "passed",
        "gemini_included": False, "remote_upload_status": "not_uploaded",
    }
    archive.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
