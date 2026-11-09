#!/usr/bin/env python3
"""Import the frozen manifest and optional review CSV into simple SQLite tables."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUS = {"pass", "hard_fail", "uncertain"}
DIMENSIONS = (
    "same_scene", "temporal_alignment", "target_present", "instruction_complete",
    "content_preserved", "logical_result", "single_step",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_flags(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def upsert_manifest(connection: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    for row in rows:
        connection.execute(
            "INSERT INTO datasets(dataset_id, repo_id, revision) VALUES (?, ?, ?) "
            "ON CONFLICT(dataset_id) DO UPDATE SET repo_id=excluded.repo_id, revision=excluded.revision",
            (row["dataset"], row["repo_id"], row["revision"]),
        )
        for role in ("source", "target"):
            connection.execute(
                "INSERT INTO videos(video_id, dataset_id, locator, local_path) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET locator=excluded.locator, local_path=excluded.local_path",
                (
                    row[f"{role}_video_id"], row["dataset"], row[f"{role}_locator"],
                    row[f"{role}_local"],
                ),
            )
        connection.execute(
            "INSERT INTO pairs(pair_id, dataset_id, source_video_id, target_video_id, instruction, operation, direction) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(pair_id) DO UPDATE SET "
            "source_video_id=excluded.source_video_id, target_video_id=excluded.target_video_id, "
            "instruction=excluded.instruction, operation=excluded.operation, direction=excluded.direction",
            (
                row["pair_id"], row["dataset"], row["source_video_id"], row["target_video_id"],
                row["instruction"], row["operation"], row["direction"],
            ),
        )


def upsert_reviews(
    connection: sqlite3.Connection,
    manifest: list[dict[str, str]],
    reviews: list[dict[str, str]],
) -> int:
    review_by_id = {row["sample_id"]: row for row in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("review CSV contains duplicate sample_id")
    video_values: dict[str, tuple[str, str, str, float]] = {}
    imported = 0
    for item in manifest:
        review = review_by_id.get(item["sample_id"])
        if review is None:
            continue
        for prefix in ("target", "source", "pair"):
            if review.get(f"{prefix}_status") not in ALLOWED_STATUS:
                raise ValueError(f"invalid or missing {prefix}_status for {item['sample_id']}")
        for role in ("source", "target"):
            value = (
                review[f"{role}_status"],
                json.dumps(split_flags(review.get(f"{role}_defects", "")), ensure_ascii=False),
                review.get(f"{role}_notes", ""),
                float(review.get(f"{role}_seconds", 0) or 0),
            )
            video_id = item[f"{role}_video_id"]
            if video_id in video_values and video_values[video_id][:3] != value[:3]:
                raise ValueError(f"conflicting reused video audit for {video_id}")
            video_values[video_id] = value
        dimensions = {key: review.get(f"pair_{key}", "") for key in DIMENSIONS}
        connection.execute(
            "INSERT INTO pair_audits(survey_version, pair_id, sample_id, sample_rank, status, "
            "errors_json, source_flags_json, dimensions_json, notes, review_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(survey_version, pair_id) DO UPDATE SET status=excluded.status, "
            "errors_json=excluded.errors_json, source_flags_json=excluded.source_flags_json, "
            "dimensions_json=excluded.dimensions_json, "
            "notes=excluded.notes, review_seconds=excluded.review_seconds",
            (
                item["survey_version"], item["pair_id"], item["sample_id"], int(item["sample_rank"]),
                review["pair_status"], json.dumps(split_flags(review.get("pair_errors", "")), ensure_ascii=False),
                json.dumps(split_flags(review.get("source_flags", "")), ensure_ascii=False),
                json.dumps(dimensions, ensure_ascii=False), review.get("pair_notes", ""),
                float(review.get("pair_seconds", 0) or 0),
            ),
        )
        imported += 1
    version = manifest[0]["survey_version"] if manifest else ""
    for video_id, value in video_values.items():
        connection.execute(
            "INSERT INTO video_audits(survey_version, video_id, status, defects_json, notes, review_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(survey_version, video_id) DO UPDATE SET "
            "status=excluded.status, defects_json=excluded.defects_json, notes=excluded.notes, "
            "review_seconds=excluded.review_seconds",
            (version, video_id, *value),
        )
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/survey_manifest.csv")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--database", type=Path, default=ROOT / "artifacts/survey.sqlite")
    args = parser.parse_args()
    manifest = read_csv(args.manifest)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.database)
    try:
        connection.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
        with connection:
            upsert_manifest(connection, manifest)
            imported = upsert_reviews(connection, manifest, read_csv(args.review)) if args.review else 0
    finally:
        connection.close()
    print(f"pairs={len(manifest)} reviews={imported} database={args.database}")


if __name__ == "__main__":
    main()
