#!/usr/bin/env python3
"""Atomically persist a complete Bernini Tier-A run from node-local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if len(tasks) != 24:
        raise ValueError(f"expected 24 Tier-A tasks, got {len(tasks)}")

    output_dir = args.output_dir.absolute()
    source_dir = output_dir.resolve()
    expected_names = [Path(task["output"]).name for task in tasks]
    if len(set(expected_names)) != 24:
        raise ValueError("Tier-A task outputs are not unique")

    missing = [name for name in expected_names if not (source_dir / name).is_file()]
    empty = [name for name in expected_names if (source_dir / name).is_file() and (source_dir / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"outputs incomplete: missing={missing}, empty={empty}")

    # A real directory is already persistent; only verify it and mark success.
    if not output_dir.is_symlink():
        if source_dir != output_dir.resolve():
            raise RuntimeError(f"unexpected output path state: {output_dir}")
        args.marker.touch()
        print(json.dumps({"persisted": 24, "output_dir": str(output_dir), "already_shared": True}))
        return

    staging = output_dir.with_name(output_dir.name + ".persisting")
    if staging.exists():
        suffix = time.strftime("%Y%m%dT%H%M%S")
        staging = output_dir.with_name(output_dir.name + f".persisting.{suffix}")
    staging.mkdir(parents=True)

    hashes: dict[str, str] = {}
    for name in expected_names:
        src = source_dir / name
        dst = staging / name
        shutil.copy2(src, dst)
        if dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"size mismatch while persisting {name}")
        src_hash = sha256(src)
        if sha256(dst) != src_hash:
            raise RuntimeError(f"SHA256 mismatch while persisting {name}")
        hashes[name] = src_hash

    backup = output_dir.with_name(output_dir.name + ".node_local_link")
    if backup.exists() or backup.is_symlink():
        backup = output_dir.with_name(output_dir.name + f".node_local_link.{int(time.time())}")
    os.rename(output_dir, backup)
    os.rename(staging, output_dir)

    for name, expected_hash in hashes.items():
        if sha256(output_dir / name) != expected_hash:
            raise RuntimeError(f"post-swap SHA256 mismatch for {name}")
    args.marker.touch()
    print(json.dumps({
        "persisted": 24,
        "output_dir": str(output_dir),
        "node_local_source": str(source_dir),
        "link_backup": str(backup),
    }))


if __name__ == "__main__":
    main()
