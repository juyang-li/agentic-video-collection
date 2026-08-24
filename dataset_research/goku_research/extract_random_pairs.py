#!/usr/bin/env python3
"""Extract selected GOKU source/target MP4s from an uncompressed tar shard."""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
DEFAULT_VIDEO_TAR = Path(
    "/tmp/goku_acquire/swap_alter/swap_alter.videos.part05.tar"
)


@dataclass(frozen=True)
class Destination:
    path: Path
    sample_id: str
    role: str


def load_wanted() -> dict[str, list[Destination]]:
    wanted: dict[str, list[Destination]] = defaultdict(list)
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 15:
        raise ValueError("manifest must contain 15 rows")
    for row in rows:
        for role, member_field, file_field in (
            ("source", "source_member", "source_file"),
            ("target", "target_member", "target_file"),
        ):
            destination = (ROOT / row[file_field]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            wanted[row[member_field]].append(
                Destination(destination, row["sample_id"], role)
            )
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-tar", type=Path, default=DEFAULT_VIDEO_TAR)
    args = parser.parse_args()
    wanted = load_wanted()
    extracted = 0
    with tarfile.open(args.video_tar, "r:") as archive:
        available = {member.name.removeprefix("./"): member for member in archive}
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise FileNotFoundError("missing members: " + ", ".join(missing))
        for member_name, destinations in wanted.items():
            source = archive.extractfile(available[member_name])
            if source is None:
                raise RuntimeError(f"cannot read member: {member_name}")
            first = destinations[0]
            first.path.parent.mkdir(parents=True, exist_ok=True)
            part = first.path.with_name(first.path.name + ".part")
            with part.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            part.replace(first.path)
            for duplicate in destinations[1:]:
                duplicate.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first.path, duplicate.path)
            extracted += len(destinations)
    print(f"extracted_roles={extracted} unique_members={len(wanted)}")


if __name__ == "__main__":
    main()
