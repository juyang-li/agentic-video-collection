#!/usr/bin/env python3
"""Freeze and fetch an unfiltered CoinVE sample for human pair review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_random_pairs import REPO_ID, REVISION, WantedMember, extract_shard
from sample_random_pairs import EXPECTED_METADATA_SHA256, shard_and_member

ROOT = Path(__file__).resolve().parent
VERSION = "coinve_nostylize_random50_20260906"
OUTPUT = ROOT / VERSION


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select():
    metadata = ROOT / "metadata/metadata_coinve200k.jsonl"
    sha = digest(metadata)
    if sha != EXPECTED_METADATA_SHA256:
        raise ValueError("CoinVE metadata hash differs from frozen revision")
    old = {int(r["metadata_index"]) for r in read_csv(ROOT / "random_pairs.csv")}
    eligible, nonstyle_count, total = [], 0, 0
    with metadata.open() as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            total += 1
            if any("styl" in op.lower() for op in item["instruction_operation"]):
                continue
            nonstyle_count += 1
            if index not in old:
                eligible.append((index, item))
    chosen = random.Random(20260906).sample(eligible, 50)
    rows = []
    for rank, (index, item) in enumerate(chosen, 1):
        count = len(item["instruction"])
        assert count == len(item["instruction_operation"]) == len(item["instruction_object"])
        row = dict(survey_version=VERSION, sample_rank=rank,
                   sample_id=f"coinve-{index:06d}", metadata_index=index,
                   dataset="CoinVE-200K", branch=f"compose-{count}",
                   repo_id=REPO_ID, revision=REVISION, instruction_count=count,
                   instruction="\n".join(f"{i}. {s}" for i, s in enumerate(item["instruction"], 1)),
                   instructions_json=json.dumps(item["instruction"], ensure_ascii=False),
                   operations_json=json.dumps(item["instruction_operation"]),
                   objects_json=json.dumps(item["instruction_object"]),
                   instruction_masks_json=json.dumps(item["instruction_mask_video_paths"]),
                   combined_mask_path=item["combined_mask_video_path"])
        for role, key in (("source", "source_video_path"), ("target", "edited_video_path")):
            row[f"{role}_path"] = item[key]
            row[f"{role}_shard"], row[f"{role}_member"] = shard_and_member(item[key])
        rows.append(row)
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT / "sample_manifest.csv"
    if manifest.exists():
        previous = read_csv(manifest)
        assert [r["sample_id"] for r in previous] == [r["sample_id"] for r in rows], "Refuse to change frozen sample"
    else:
        with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(survey_version=VERSION, repo_id=REPO_ID, revision=REVISION,
                   metadata_sha256=sha, metadata_rows=total,
                   nonstylize_population=nonstyle_count, eligible_population=len(eligible),
                   previously_reviewed_excluded=nonstyle_count-len(eligible),
                   sample_size=len(rows), seed=20260906,
                   sampling="uniform without replacement from non-Stylize metadata excluding previous 15 review IDs; random order",
                   quality_filter="none", person_filter="none", resampling="none",
                   instruction_count=dict(Counter(r["instruction_count"] for r in rows)),
                   sample_manifest_sha256=digest(manifest),
                   selection_ids=[r["sample_id"] for r in rows])
    write_json(OUTPUT / "sampling_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def download(cache: Path, workers: int):
    rows = read_csv(OUTPUT / "sample_manifest.csv")
    cache.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        for role in ("source", "target"):
            grouped[row[f"{role}_shard"]].append(WantedMember(
                row[f"{role}_member"], cache / f"{row['sample_id']}_{role}.mp4",
                row["sample_id"], role))
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_shard, shard, wanted): shard for shard, wanted in grouped.items()}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                results[shard] = future.result()
            except Exception as error:
                # Do not log signed CDN URLs from requests exception strings.
                errors[shard] = type(error).__name__
                print(f"download_failed shard={shard} error={type(error).__name__}", flush=True)
            write_json(OUTPUT / "download_status.json", dict(results=results, errors=errors,
                       complete_role_files=sum((cache / f"{r['sample_id']}_{role}.mp4").is_file()
                                               for r in rows for role in ("source", "target"))))
    if errors:
        raise SystemExit(f"{len(errors)} shards require retry; frozen sample remains unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "download"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/coinve_random50_20260906"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "select":
        select()
    else:
        download(args.cache, args.workers)


if __name__ == "__main__":
    main()
