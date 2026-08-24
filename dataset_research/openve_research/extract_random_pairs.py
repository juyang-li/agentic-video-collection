#!/usr/bin/env python3
"""Extract selected OpenVE source/target MP4s from a local gzip prefix."""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sample_random_pairs import ConcatenatedReader


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
DEFAULT_PREFIX_PATH = Path("/tmp/openve_local_change_prefix_2g.tar.gz")


@dataclass(frozen=True)
class Destination:
    path: Path
    sample_id: str
    role: str


def load_wanted(manifest_path: Path) -> dict[str, list[Destination]]:
    wanted: dict[str, list[Destination]] = defaultdict(list)
    research_root = manifest_path.resolve().parent
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for role, member_field, file_field in (
            ("source", "source_member", "source_file"),
            ("target", "target_member", "target_file"),
        ):
            destination = (research_root / row[file_field]).resolve()
            if research_root not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            wanted[row[member_field]].append(
                Destination(destination, row["sample_id"], role)
            )
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX_PATH)
    parser.add_argument("--prefix-extra", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    wanted = load_wanted(args.manifest)
    pending = {
        member: [item for item in destinations if not item.path.is_file()]
        for member, destinations in wanted.items()
    }
    pending = {member: items for member, items in pending.items() if items}
    preserved = sum(
        item.path.is_file() for destinations in wanted.values() for item in destinations
    )
    if not pending:
        print(f"done extracted=0 preserved={preserved} scan=skipped")
        return

    extracted = 0
    scanned = 0
    try:
        stream = ConcatenatedReader([args.prefix, *args.prefix_extra])
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                normalized = member.name.removeprefix("./")
                if not member.isfile() or not normalized.lower().endswith(".mp4"):
                    continue
                scanned += 1
                destinations = pending.get(normalized)
                if not destinations:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read member: {member.name}")
                first = destinations[0]
                first.path.parent.mkdir(parents=True, exist_ok=True)
                part_path = first.path.with_name(first.path.name + ".part")
                with part_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                part_path.replace(first.path)
                for duplicate in destinations[1:]:
                    duplicate.path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(first.path, duplicate.path)
                extracted += len(destinations)
                pending.pop(normalized)
                print(
                    f"matched member={normalized} roles="
                    f"{','.join(x.sample_id + ':' + x.role for x in destinations)} "
                    f"pending_members={len(pending)}",
                    flush=True,
                )
                if not pending:
                    break
    except (EOFError, OSError, tarfile.ReadError) as error:
        if pending:
            raise RuntimeError(f"prefix ended with pending members: {sorted(pending)}") from error
    if pending:
        raise FileNotFoundError("selected members not found: " + ", ".join(sorted(pending)))
    print(f"done extracted={extracted} preserved={preserved} scanned_mp4={scanned}")


if __name__ == "__main__":
    main()
