#!/usr/bin/env python3
"""Parallel prefill of all agreed Long inputs using collision-safe temp files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

from prepare_next_stage import DATA, TREE, STAGES, digest, load_rows, long_model_input, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-id", type=int)
    parser.add_argument("--max-id", type=int)
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--report", default="long_121_prefill.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    tree = {entry["path"]: entry for entry in json.loads(TREE.read_text()) if entry.get("type") == "file"}
    selected: dict[str, dict] = {}
    for stage_name in ("long_subject", "attribute_all"):
        for row in load_rows(STAGES[stage_name]):
            if row["subset"] == "long":
                selected[row["src_video_name"]] = row
    # The two existing sequential preparers advance from the front.  Starting
    # from the back minimizes duplicate work while remaining race-safe.
    rows = [row for row in selected.values()
            if (args.min_id is None or int(row["id"]) >= args.min_id)
            and (args.max_id is None or int(row["id"]) <= args.max_id)]
    rows.sort(key=lambda row: int(row["id"]), reverse=not args.ascending)
    if not rows:
        raise ValueError("selected Long prefill range is empty")

    def convert(row: dict) -> dict:
        relative = f"ivebenchdb_mp4_long/{row['src_video_name']}"
        entry = tree[relative]
        source = DATA / "long" / row["src_video_name"]
        if not source.is_file() or source.stat().st_size != int(entry["size"]) or digest(source) != entry["lfs"]["oid"]:
            raise ValueError(f"invalid official Long input: {source}")
        suffix = f".prefill-{os.getpid()}-{threading.get_ident()}.part.mp4"
        destination, _ = long_model_input(source, entry["lfs"]["oid"], suffix)
        return {"source": row["src_video_name"], "destination": destination.name,
                "sha256": digest(destination)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, row): row["src_video_name"] for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
            print(f"prefilled {len(results)}/{len(rows)} {futures[future]}", flush=True)

    results.sort(key=lambda item: item["source"])
    report = Path(args.report)
    if report.name != args.report or report.suffix != ".json":
        raise ValueError("report must be a JSON basename")
    write_json(DATA / "model_inputs" / report, {
        "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers, "min_id": args.min_id, "max_id": args.max_id,
        "ascending": args.ascending, "files": len(results), "results": results,
    })
    print(json.dumps({"status": "complete", "files": len(results)}, indent=2))


if __name__ == "__main__":
    main()
