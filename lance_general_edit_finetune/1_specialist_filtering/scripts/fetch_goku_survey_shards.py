#!/usr/bin/env python3
"""Download configured GOKU metadata/shards and build their populations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("goku_add", "goku_remove"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "survey_config.json")
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required; install it in the acquisition environment"
        ) from error

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    branch = config["branches"][args.branch]
    task = "add" if args.branch == "goku_add" else "remove"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    metadata = Path(
        hf_hub_download(
            repo_id=branch["repo_id"],
            repo_type="dataset",
            revision=branch["revision"],
            filename=f"{task}/{task}.jsons.tar",
            local_dir=args.work_dir,
        )
    )
    preparer = ROOT / "scripts" / "prepare_goku_population.py"
    for shard in branch["shards"]:
        video_tar = Path(
            hf_hub_download(
                repo_id=branch["repo_id"],
                repo_type="dataset",
                revision=branch["revision"],
                filename=shard["archive"],
                local_dir=args.work_dir,
            )
        )
        population = (config_path.parent / shard["population"]).resolve()
        subprocess.run(
            [
                sys.executable,
                str(preparer),
                "--task",
                task,
                "--metadata-tar",
                str(metadata),
                "--video-tar",
                str(video_tar),
                "--output",
                str(population),
            ],
            check=True,
        )
    print(f"prepared branch={args.branch} shards={len(branch['shards'])}")


if __name__ == "__main__":
    main()
