#!/usr/bin/env python3
"""Verify that an extended survey can accept the prior 400-pair results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def has_review(review: dict[str, object]) -> bool:
    for phase in ("target", "source", "pair"):
        audit = review.get(phase)
        if not isinstance(audit, dict):
            continue
        if (
            audit.get("status")
            or audit.get("notes")
            or float(audit.get("seconds") or 0)
            or audit.get("flags")
            or audit.get("source_flags")
            or any((audit.get("dimensions") or {}).values())
        ):
            return True
    return False


def verify(
    base_manifest: Path,
    extended_manifest: Path,
    prior_json: Path,
    prior_csv: Path,
) -> dict[str, int]:
    base = read_csv(base_manifest)
    extended = read_csv(extended_manifest)
    if len(base) != 400 or len(extended) <= len(base):
        raise ValueError("expected a 400-row base and a larger extended manifest")
    comparable_fields = [field for field in base[0] if field != "survey_version"]
    for index, old in enumerate(base):
        new = extended[index]
        if any(old[field] != new[field] for field in comparable_fields):
            raise ValueError(f"base manifest changed at rank {index + 1}")

    extended_ids = {row["sample_id"] for row in extended}
    extended_videos = {
        row[key]
        for row in extended
        for key in ("source_video_id", "target_video_id")
    }
    backup = json.loads(prior_json.read_text(encoding="utf-8"))
    reviewed_pairs = {
        sample_id
        for sample_id, review in (backup.get("pairs") or {}).items()
        if isinstance(review, dict) and has_review(review)
    }
    decided_pairs = {
        sample_id
        for sample_id, review in (backup.get("pairs") or {}).items()
        if isinstance(review, dict)
        and any((review.get(phase) or {}).get("status") for phase in ("target", "source", "pair"))
    }
    complete_pairs = {
        sample_id
        for sample_id, review in (backup.get("pairs") or {}).items()
        if isinstance(review, dict)
        and all((review.get(phase) or {}).get("status") for phase in ("target", "source", "pair"))
    }
    prior_videos = set((backup.get("video_audits") or {}).keys())
    if not reviewed_pairs <= extended_ids:
        raise ValueError("prior JSON contains reviewed pair IDs missing from the extension")
    if not prior_videos <= extended_videos:
        raise ValueError("prior JSON contains video IDs missing from the extension")

    csv_rows = read_csv(prior_csv)
    csv_reviewed = {
        row["sample_id"]
        for row in csv_rows
        if row.get("target_status") or row.get("source_status") or row.get("pair_status")
    }
    if not csv_reviewed <= extended_ids:
        raise ValueError("prior CSV contains reviewed pair IDs missing from the extension")
    return {
        "base_pairs_preserved": len(base),
        "extended_pairs": len(extended),
        "json_reviewed_pairs_importable": len(reviewed_pairs),
        "json_decided_pairs_importable": len(decided_pairs),
        "json_complete_pairs_importable": len(complete_pairs),
        "json_video_audits_importable": len(prior_videos),
        "csv_reviewed_pairs_importable": len(csv_reviewed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "artifacts/survey_manifest.csv")
    parser.add_argument(
        "--extended-manifest",
        type=Path,
        default=ROOT / "artifacts/survey_manifest_reco_v3.csv",
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=ROOT / "artifacts/review_results/specialist_error_survey_400_v1_backup.json",
    )
    parser.add_argument(
        "--prior-csv",
        type=Path,
        default=ROOT / "artifacts/review_results/specialist_error_survey_400_v1_review.csv",
    )
    args = parser.parse_args()
    result = verify(
        args.base_manifest, args.extended_manifest, args.prior_json, args.prior_csv
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
