#!/usr/bin/env python3
"""Extract selected general_editing source/target videos from part_04."""

from __future__ import annotations

import csv
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
VIDEO_TAR = Path(
    "/tmp/insave_general_replacement_part04/train_data/general_editing/part_04.tar"
)


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 10:
        raise ValueError("manifest must contain 10 rows")
    wanted: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        for member_field, file_field in (
            ("source_member", "source_file"),
            ("target_member", "target_file"),
        ):
            destination = (ROOT / row[file_field]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes root: {destination}")
            wanted[row[member_field]].append(destination)
    extracted = 0
    with tarfile.open(VIDEO_TAR, "r:") as archive:
        available = {member.name.removeprefix("./"): member for member in archive}
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise FileNotFoundError(f"missing members: {missing[:5]}")
        for member_name, destinations in wanted.items():
            source = archive.extractfile(available[member_name])
            if source is None:
                raise RuntimeError(f"cannot read {member_name}")
            first = destinations[0]
            first.parent.mkdir(parents=True, exist_ok=True)
            part = first.with_name(first.name + ".part")
            with part.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            part.replace(first)
            for duplicate in destinations[1:]:
                duplicate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first, duplicate)
            extracted += len(destinations)
    print(f"extracted_roles={extracted} unique_members={len(wanted)}")


if __name__ == "__main__":
    main()
