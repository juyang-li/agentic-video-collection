#!/usr/bin/env python3
"""Build an OpenVE archive-prefix population and sample 15 pairs.

Defaults reproduce the original ``local_change`` audit. Category wrappers may
override the constants below before calling :func:`main`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import tarfile
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "local_change.csv"
MEMBERS_PATH = ROOT / "prefix_members.txt"
POPULATION_PATH = ROOT / "downloadable_population.csv"
OUTPUT_PATH = ROOT / "random_pairs.csv"
EXPECTED_METADATA_SHA256 = (
    "eaf9b18ab58ef2f91812a01ca5d5656d2b018b72670ed02d4a36c6366e6765bb"
)
REVISION = "d2c134f01c49d22efc2e0d9aefd798772f039a79"
DEFAULT_PREFIX_PATH = Path("/tmp/openve_local_change_prefix_2g.tar.gz")
DEFAULT_PREFIX_BYTES = 2 * 1024**3
DEFAULT_SEED = 20260821
DEFAULT_SAMPLE_SIZE = 15
CATEGORY = "local_change"
ASSET_PATH_PREFIX = "local_change"
ARCHIVE_DIRECTORY = "local_change_new"
SUBPOPULATION = "local_change_same_archive_fixed_prefix"
SAMPLE_ID_PREFIX = "openve"
TASK_OVERRIDE: str | None = None


class ConcatenatedReader:
    """Read several byte-range files as one non-seekable stream."""

    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("at least one prefix part is required")
        self.paths = iter(paths)
        self.current: BinaryIO | None = None

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            chunk = self.current.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
                continue
            self.current.close()
            self.current = None
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_parts(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def infer_task(instruction: str) -> str:
    first_word = instruction.strip().split(maxsplit=1)[0].lower()
    return {
        "add": "add",
        "apply": "change",
        "change": "change",
        "dress": "change",
        "give": "change",
        "make": "change",
        "remove": "remove",
        "replace": "replace",
        "transform": "transform",
        "turn": "transform",
    }.get(first_word, "other")


def scan_complete_mp4_members(prefix_paths: list[Path]) -> list[str]:
    """Return members proven complete; discard the final yielded truncated member."""
    complete: list[str] = []
    pending: str | None = None
    try:
        stream = ConcatenatedReader(prefix_paths)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if pending is not None:
                    complete.append(pending)
                    pending = None
                normalized = member.name.removeprefix("./")
                if member.isfile() and normalized.lower().endswith(".mp4"):
                    pending = normalized
                if len(complete) and len(complete) % 1000 == 0:
                    print(f"scan complete_mp4={len(complete)}", flush=True)
    except (EOFError, OSError, tarfile.ReadError) as error:
        print(f"expected truncated-prefix stop: {type(error).__name__}: {error}")
    if not complete:
        raise RuntimeError("no complete MP4 members found in prefix")
    return complete


def load_population(member_names: set[str]) -> list[dict[str, str]]:
    population: list[dict[str, str]] = []
    with METADATA_PATH.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if reader.fieldnames != ["video", "prompt", "original_video"]:
            raise ValueError(f"unexpected metadata fields: {reader.fieldnames}")
        for metadata_index, row in enumerate(reader):
            source_path = row["original_video"]
            target_path = row["video"]
            if not source_path.startswith(ASSET_PATH_PREFIX + "/"):
                continue
            if not target_path.startswith(ASSET_PATH_PREFIX + "/"):
                continue
            source_member = ARCHIVE_DIRECTORY + "/" + Path(source_path).name
            target_member = ARCHIVE_DIRECTORY + "/" + Path(target_path).name
            if source_member not in member_names or target_member not in member_names:
                continue
            population.append(
                {
                    "metadata_index": str(metadata_index),
                    "instruction": row["prompt"],
                    "source_path": source_path,
                    "edited_path": target_path,
                    "source_member": source_member,
                    "target_member": target_member,
                }
            )
    return population


def write_population(population: list[dict[str, str]]) -> None:
    with POPULATION_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(population[0]))
        writer.writeheader()
        writer.writerows(population)


def write_sample(
    population: list[dict[str, str]],
    eligible_indices: list[int],
    seed: int,
    sample_size: int,
    prefix_bytes: int,
    output_path: Path,
) -> list[int]:
    if len(eligible_indices) < sample_size:
        raise RuntimeError(
            f"eligible population {len(eligible_indices)} is smaller than {sample_size}; "
            "increase the archive prefix"
        )
    indices = random.Random(seed).sample(eligible_indices, sample_size)
    fields = [
        "sample_rank",
        "metadata_index",
        "subpopulation_index",
        "sample_id",
        "official_category",
        "task",
        "seed",
        "revision",
        "prefix_bytes",
        "subpopulation",
        "subpopulation_size",
        "instruction",
        "source_path",
        "edited_path",
        "source_member",
        "target_member",
        "source_file",
        "target_file",
    ]
    subpopulation = SUBPOPULATION
    if len(eligible_indices) != len(population):
        subpopulation += "_remaining"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, subpopulation_index in enumerate(indices, start=1):
            row = population[subpopulation_index]
            metadata_index = int(row["metadata_index"])
            sample_id = f"{SAMPLE_ID_PREFIX}-{metadata_index:06d}"
            writer.writerow(
                {
                    "sample_rank": rank,
                    "metadata_index": metadata_index,
                    "subpopulation_index": subpopulation_index,
                    "sample_id": sample_id,
                    "official_category": CATEGORY,
                    "task": TASK_OVERRIDE or infer_task(row["instruction"]),
                    "seed": seed,
                    "revision": REVISION,
                    "prefix_bytes": prefix_bytes,
                    "subpopulation": subpopulation,
                    "subpopulation_size": len(eligible_indices),
                    "instruction": row["instruction"],
                    "source_path": row["source_path"],
                    "edited_path": row["edited_path"],
                    "source_member": row["source_member"],
                    "target_member": row["target_member"],
                    "source_file": f"examples/{sample_id}_source.mp4",
                    "target_file": f"examples/{sample_id}_target.mp4",
                }
            )
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX_PATH)
    parser.add_argument("--prefix-extra", type=Path, action="append", default=[])
    parser.add_argument("--prefix-bytes", type=int, default=DEFAULT_PREFIX_BYTES)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()

    if sha256_file(METADATA_PATH) != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata SHA256 changed; inspect before sampling")
    prefix_paths = [args.prefix, *args.prefix_extra]
    actual_prefix_bytes = sum(path.stat().st_size for path in prefix_paths)
    if actual_prefix_bytes != args.prefix_bytes:
        raise ValueError(
            f"prefix size mismatch: {actual_prefix_bytes} != {args.prefix_bytes}"
        )
    members = scan_complete_mp4_members(prefix_paths)
    MEMBERS_PATH.write_text("\n".join(members) + "\n", encoding="utf-8")
    population = load_population(set(members))
    if not population:
        raise RuntimeError("no complete pairs found in the fixed prefix")
    write_population(population)
    excluded_metadata: set[str] = set()
    if args.exclude_manifest:
        with args.exclude_manifest.open(encoding="utf-8", newline="") as file:
            excluded_metadata = {
                row["metadata_index"] for row in csv.DictReader(file)
            }
    eligible_indices = [
        index
        for index, row in enumerate(population)
        if row["metadata_index"] not in excluded_metadata
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_PATH.name
    indices = write_sample(
        population,
        eligible_indices,
        args.seed,
        args.sample_size,
        actual_prefix_bytes,
        output_path,
    )
    print(f"prefix_sha256={sha256_parts(prefix_paths)}")
    print(
        f"complete_members={len(members)} population={len(population)} "
        f"eligible={len(eligible_indices)} excluded={len(excluded_metadata)} "
        f"sample_size={args.sample_size} seed={args.seed} indices={indices} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
