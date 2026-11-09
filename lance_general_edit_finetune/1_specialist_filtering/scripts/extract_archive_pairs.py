#!/usr/bin/env python3
"""Extract configured InsAVE/GOKU survey members from local tar archives."""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Destination:
    path: Path
    sample_id: str
    role: str


def safe_output(relative: str) -> Path:
    output = (ROOT / relative).resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError(f"output escapes specialist root: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/survey_manifest.csv")
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="MANIFEST_NAME=LOCAL_TAR",
        help="repeat for each shard named in the manifest",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    archives: dict[str, Path] = {}
    for value in args.archive:
        name, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --archive mapping: {value}")
        archives[name] = Path(raw_path).resolve()

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_shard: dict[str, dict[str, list[Destination]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["dataset"] not in {"insave", "goku"}:
            continue
        for role in ("source", "target"):
            by_shard[row["shard"]][row[f"{role}_locator"]].append(
                Destination(safe_output(row[f"{role}_local"]), row["sample_id"], role)
            )

    missing_archives = sorted(set(by_shard) - set(archives))
    if missing_archives:
        raise FileNotFoundError(
            "missing --archive mappings for: " + ", ".join(missing_archives)
        )

    extracted = 0
    for shard, wanted in by_shard.items():
        with tarfile.open(archives[shard], "r:") as archive:
            available = {member.name.removeprefix("./"): member for member in archive}
            missing = sorted(set(wanted) - set(available))
            if missing:
                raise FileNotFoundError(f"{shard} is missing members: {missing[:5]}")
            for member_name, destinations in wanted.items():
                pending = [item for item in destinations if args.overwrite or not item.path.exists()]
                if not pending:
                    continue
                source = archive.extractfile(available[member_name])
                if source is None:
                    raise RuntimeError(f"cannot read {member_name} from {shard}")
                first = pending[0].path
                first.parent.mkdir(parents=True, exist_ok=True)
                temporary = first.with_suffix(first.suffix + ".part")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                temporary.replace(first)
                for duplicate in pending[1:]:
                    duplicate.path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(first, duplicate.path)
                extracted += len(pending)
        print(f"shard={shard} extracted_roles={extracted}", flush=True)
    print(f"done extracted_roles={extracted}")


if __name__ == "__main__":
    main()
