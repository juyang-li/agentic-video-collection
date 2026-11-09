#!/usr/bin/env python3
"""Download the two configured InsAVE survey archives to temporary storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "survey_config.json")
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("huggingface_hub is required in the acquisition environment") from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = config["branches"]["insave_remove"]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for shard in branch["shards"]:
        path = hf_hub_download(
            repo_id=branch["repo_id"],
            repo_type="dataset",
            revision=branch["revision"],
            filename=shard["archive"],
            local_dir=args.work_dir,
        )
        print(f"downloaded {shard['archive']} -> {path}")


if __name__ == "__main__":
    main()
