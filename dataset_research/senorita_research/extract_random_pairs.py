#!/usr/bin/env python3
"""Extract the selected Señorita pairs from one compressed object-swap shard."""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
DEFAULT_ARCHIVE = Path("/tmp/senorita_probe/obj_swap_upload/144.tar.gz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 15:
        raise ValueError("manifest must contain 15 rows")

    wanted: dict[str, Path] = {}
    for row in rows:
        for member_field, file_field in (
            ("source_member", "source_file"),
            ("target_member", "target_file"),
        ):
            destination = (ROOT / row[file_field]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            wanted[row[member_field]] = destination

    extracted = 0
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive
            if member.isfile()
        }
        missing = sorted(set(wanted) - set(members))
        if missing:
            raise FileNotFoundError("missing members: " + ", ".join(missing))
        for name, destination in wanted.items():
            source = archive.extractfile(members[name])
            if source is None:
                raise RuntimeError(f"cannot read member: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_name(destination.name + ".part")
            with part.open("wb") as output:
                shutil.copyfileobj(source, output)
            part.replace(destination)
            extracted += 1
    print(f"extracted_roles={extracted}")


if __name__ == "__main__":
    main()
