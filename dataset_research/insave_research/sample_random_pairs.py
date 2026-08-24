#!/usr/bin/env python3
"""Build the InsAVE add_and_remove part_01 population and sample 15 pairs."""

from __future__ import annotations

import csv
import hashlib
import random
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "add_and_remove.csv"
VIDEO_TAR = Path("/tmp/insave_acquire/train_data/add_and_remove/part_01.tar")
MEMBERS_PATH = ROOT / "shard_members.txt"
POPULATION_PATH = ROOT / "shard_population.csv"
MANIFEST_PATH = ROOT / "random_pairs.csv"
REPO_ID = "suimu/InsAVE-80K"
REVISION = "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289"
EXPECTED_METADATA_SHA256 = (
    "2c0ec8fc5d31f4aa99e611e4739885dd8fed423b0053d82f249444e79de0421a"
)
EXPECTED_VIDEO_SHA256 = (
    "f7f75ef2eb0f8137af2d630c5024ba3b672f1f93600589e33e0f89cb6b6c4c7d"
)
SHARD_SEED = 20260826
PAIR_SEED = 20260826
SAMPLE_SIZE = 15
MEMBER_RE = re.compile(r"(?P<index>\d{5})_(?P<role>original|traget)\.mp4$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_shard() -> str:
    return random.Random(SHARD_SEED).choice(["part_00.tar", "part_01.tar"])


def scan_tar(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    members: list[str] = []
    cases: dict[int, dict[str, str]] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not member.isfile() or not name.lower().endswith(".mp4"):
                continue
            members.append(name)
            match = MEMBER_RE.search(Path(name).name)
            if match:
                cases.setdefault(int(match["index"]), {})[match["role"]] = name
    return members, cases


def load_metadata() -> list[dict[str, str]]:
    with METADATA_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    expected = ["original_video", "target_video", "instruction", "instruction_reverse"]
    if not rows or list(rows[0]) != expected:
        raise ValueError("unexpected metadata schema")
    return rows


def build_population(
    cases: dict[int, dict[str, str]], metadata: list[dict[str, str]]
) -> list[dict[str, str]]:
    population: list[dict[str, str]] = []
    for index in sorted(cases):
        roles = cases[index]
        if set(roles) != {"original", "traget"} or index >= len(metadata):
            continue
        row = metadata[index]
        if Path(row["original_video"]).name != f"{index:05d}_original.mp4":
            raise ValueError(f"source metadata mismatch at {index}")
        if Path(row["target_video"]).name != f"{index:05d}_traget.mp4":
            raise ValueError(f"target metadata mismatch at {index}")
        if not row["instruction"].strip():
            continue
        population.append(
            {
                "metadata_index": str(index),
                "instruction": row["instruction"].strip(),
                "instruction_reverse": row["instruction_reverse"].strip(),
                "source_member": roles["original"],
                "target_member": roles["traget"],
            }
        )
    return population


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(population: list[dict[str, str]]) -> list[int]:
    selected = random.Random(PAIR_SEED).sample(range(len(population)), SAMPLE_SIZE)
    rows: list[dict[str, object]] = []
    for rank, population_index in enumerate(selected, 1):
        item = population[population_index]
        metadata_index = int(item["metadata_index"])
        sample_id = f"insave-{metadata_index:05d}"
        rows.append(
            {
                "sample_rank": rank,
                "metadata_index": metadata_index,
                "subpopulation_index": population_index,
                "sample_id": sample_id,
                "official_category": "add_and_remove",
                "task": "remove",
                "seed": PAIR_SEED,
                "shard_seed": SHARD_SEED,
                "repo_id": REPO_ID,
                "revision": REVISION,
                "shard": expected_shard(),
                "subpopulation": "add_and_remove_part01_complete_cases",
                "subpopulation_size": len(population),
                "instruction": item["instruction"],
                "instruction_reverse": item["instruction_reverse"],
                "source_member": item["source_member"],
                "target_member": item["target_member"],
                "source_file": f"examples/{sample_id}_source.mp4",
                "target_file": f"examples/{sample_id}_target.mp4",
            }
        )
    write_csv(MANIFEST_PATH, rows)  # type: ignore[arg-type]
    return selected


def main() -> None:
    if VIDEO_TAR.name != expected_shard():
        raise ValueError(f"unexpected shard: {VIDEO_TAR.name}")
    metadata_sha = sha256_file(METADATA_PATH)
    video_sha = sha256_file(VIDEO_TAR)
    if metadata_sha != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata SHA256 changed")
    if video_sha != EXPECTED_VIDEO_SHA256:
        raise ValueError("video SHA256 changed")
    members, cases = scan_tar(VIDEO_TAR)
    population = build_population(cases, load_metadata())
    if len(population) < SAMPLE_SIZE:
        raise RuntimeError(f"population too small: {len(population)}")
    MEMBERS_PATH.write_text("\n".join(members) + "\n", encoding="utf-8")
    write_csv(POPULATION_PATH, population)
    selected = write_manifest(population)
    print(f"metadata_sha256={metadata_sha}")
    print(f"video_sha256={video_sha}")
    print(
        f"members={len(members)} cases={len(cases)} population={len(population)} "
        f"sample_size={SAMPLE_SIZE} seed={PAIR_SEED} indices={selected}"
    )


if __name__ == "__main__":
    main()
