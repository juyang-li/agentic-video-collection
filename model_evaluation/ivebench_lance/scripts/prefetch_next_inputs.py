#!/usr/bin/env python3
"""Download and checksum the already agreed Long Subject and Attribute inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
TREE = DATA / "dataset_tree.json"
DATASET_ID = "Coraxor/IVEBench"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"
SCOPES = {
    "long_subject": [("long", "subject_editing")],
    "attribute_all": [("short", "attribute_editing"), ("long", "attribute_editing")],
}
EXPECTED = {"long_subject": 43, "attribute_all": 94}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selected() -> dict[str, list[dict]]:
    result = {}
    for name, definitions in SCOPES.items():
        rows = []
        for subset, category in definitions:
            prompts = json.loads((DATA / f"ivebenchdb_prompt_{subset}.json").read_text())
            rows.extend({**row, "subset": subset} for row in prompts if row["category"] == category)
        rows.sort(key=lambda row: (row["subset"], int(row["id"])))
        if len(rows) != EXPECTED[name]:
            raise ValueError(f"scope count changed for {name}: {len(rows)}")
        result[name] = rows
    return result


def validate(path: Path, entry: dict) -> None:
    if not path.is_file() or path.stat().st_size != int(entry["size"]) or digest(path) != entry["lfs"]["oid"]:
        raise ValueError(f"official input validation failed: {path}")


def download(row: dict, entry: dict) -> str:
    destination = DATA / row["subset"] / row["src_video_name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate(destination, entry); return "reused"
    partial = destination.with_suffix(".mp4.part")
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{entry['path']}"
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                       stream=True, timeout=(30, 300))
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if block:
                            handle.write(block)
            validate(partial, entry)
            partial.replace(destination)
            return "downloaded"
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    scopes = selected()
    tree = {entry["path"]: entry for entry in json.loads(TREE.read_text()) if entry.get("type") == "file"}
    unique = {}
    for rows in scopes.values():
        for row in rows:
            path = f"ivebenchdb_mp4_{row['subset']}/{row['src_video_name']}"
            if path not in tree:
                raise ValueError(f"official tree entry missing: {path}")
            unique[path] = row
    counts = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, row, tree[path]): path for path, row in unique.items()}
        for future in as_completed(futures):
            counts[future.result()] += 1
            if sum(counts.values()) % 10 == 0:
                print(f"validated {sum(counts.values())}/{len(unique)}", flush=True)
    summary = {
        "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "unique_files": len(unique), "download_actions": dict(counts),
        "scopes": {name: {
            "samples": len(rows), "subcategories": dict(Counter(row["subcategory"] for row in rows)),
            "official_bytes": sum(int(tree[f"ivebenchdb_mp4_{row['subset']}/{row['src_video_name']}"]["size"]) for row in rows),
        } for name, rows in scopes.items()},
    }
    output = DATA / "next_inputs_prefetch.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
