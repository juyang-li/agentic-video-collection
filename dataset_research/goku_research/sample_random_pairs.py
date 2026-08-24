#!/usr/bin/env python3
"""Build one GOKU task-shard population and sample 15 cases.

The module defaults reproduce the original ``swap_alter`` audit.  Small task
wrappers may override the constants below before calling :func:`main`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_TAR = Path(
    "/tmp/goku_acquire/swap_alter/swap_alter.jsons.tar"
)
DEFAULT_VIDEO_TAR = Path(
    "/tmp/goku_acquire/swap_alter/swap_alter.videos.part05.tar"
)
MEMBERS_PATH = ROOT / "shard_members.txt"
POPULATION_PATH = ROOT / "shard_population.csv"
OUTPUT_PATH = ROOT / "random_pairs.csv"
REPO_ID = "bigfacing/GOKU-2M"
REVISION = "f289f4b245648dc57db04afaaeb68659440c0fa6"
EXPECTED_METADATA_SHA256 = (
    "0e2acb13ce472578435851ed854a859c21906472a8a6752a7eabd11b582fdb70"
)
EXPECTED_VIDEO_SHA256 = (
    "42360d430b499ef69d378da6c94af07d6922a8f137be3be50c2096b9c6b40dae"
)
SHARD_SEED = 20260821
PAIR_SEED = 20260821
SAMPLE_SIZE = 15
TASK_CATEGORY = "swap_alter"
SHARD_COUNT = 32
SHARD_NUMBER_WIDTH = 2
SUBPOPULATION = "swap_alter_part05_complete_cases"
SAMPLE_ID_PREFIX = "goku"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_shard_name() -> str:
    parts = [
        f"{TASK_CATEGORY}.videos.part{index:0{SHARD_NUMBER_WIDTH}d}.tar"
        for index in range(1, SHARD_COUNT + 1)
    ]
    return random.Random(SHARD_SEED).choice(parts)


def scan_video_tar(video_tar: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    members: list[str] = []
    cases: dict[str, dict[str, str]] = {}
    with tarfile.open(video_tar, "r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            path = Path(name)
            if not member.isfile() or path.suffix.lower() != ".mp4":
                continue
            members.append(name)
            if len(path.parts) != 3 or path.parts[0] != "videos":
                continue
            role = path.name
            if role not in {"source.mp4", "edited.mp4"}:
                continue
            cases.setdefault(path.parts[1], {})[role] = name
    return members, cases


def load_annotations(
    metadata_tar: Path, case_ids: set[str]
) -> dict[str, tuple[str, dict[str, object]]]:
    annotations: dict[str, tuple[str, dict[str, object]]] = {}
    with tarfile.open(metadata_tar, "r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not member.isfile() or not name.endswith("_all.json"):
                continue
            case_id = Path(name).name.removesuffix("_all.json")
            if case_id not in case_ids:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read metadata member: {name}")
            try:
                annotation = json.load(source)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                print(
                    f"skip invalid_metadata member={name} size={member.size} "
                    f"error={type(error).__name__}"
                )
                continue
            if annotation.get("case_id") != case_id:
                raise ValueError(f"case_id mismatch in {name}")
            annotations[case_id] = (name, annotation)
    return annotations


def build_population(
    cases: dict[str, dict[str, str]],
    annotations: dict[str, tuple[str, dict[str, object]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_id in sorted(cases):
        videos = cases[case_id]
        if set(videos) != {"source.mp4", "edited.mp4"} or case_id not in annotations:
            continue
        metadata_member, annotation = annotations[case_id]
        source_member = videos["source.mp4"]
        target_member = videos["edited.mp4"]
        if annotation.get("source_video") != source_member:
            raise ValueError(f"source path mismatch for {case_id}")
        if annotation.get("edited_video") != target_member:
            raise ValueError(f"target path mismatch for {case_id}")
        instruction = str(annotation.get("instruction_en", "")).strip()
        if not instruction:
            continue
        resolution = annotation.get("resolution") or {}
        rows.append(
            {
                "case_id": case_id,
                "task": str(annotation.get("step", "other")),
                "instruction": instruction,
                "source_member": source_member,
                "target_member": target_member,
                "metadata_member": metadata_member,
                "declared_width": str(resolution.get("width", "")),
                "declared_height": str(resolution.get("height", "")),
                "declared_fps": str(annotation.get("fps", "")),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sample(population: list[dict[str, str]]) -> list[int]:
    if len(population) < SAMPLE_SIZE:
        raise RuntimeError(f"population {len(population)} is smaller than {SAMPLE_SIZE}")
    indices = random.Random(PAIR_SEED).sample(range(len(population)), SAMPLE_SIZE)
    rows: list[dict[str, object]] = []
    for rank, index in enumerate(indices, start=1):
        item = population[index]
        sample_id = f"{SAMPLE_ID_PREFIX}-{rank:04d}"
        rows.append(
            {
                "sample_rank": rank,
                "metadata_index": index,
                "subpopulation_index": index,
                "sample_id": sample_id,
                "case_id": item["case_id"],
                "official_category": TASK_CATEGORY,
                "task": item["task"],
                "seed": PAIR_SEED,
                "shard_seed": SHARD_SEED,
                "repo_id": REPO_ID,
                "revision": REVISION,
                "shard": expected_shard_name(),
                "subpopulation": SUBPOPULATION,
                "subpopulation_size": len(population),
                "instruction": item["instruction"],
                "source_member": item["source_member"],
                "target_member": item["target_member"],
                "metadata_member": item["metadata_member"],
                "source_file": f"examples/{sample_id}_source.mp4",
                "target_file": f"examples/{sample_id}_target.mp4",
            }
        )
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-tar", type=Path, default=DEFAULT_METADATA_TAR)
    parser.add_argument("--video-tar", type=Path, default=DEFAULT_VIDEO_TAR)
    args = parser.parse_args()
    if args.video_tar.name != expected_shard_name():
        raise ValueError(f"unexpected shard: {args.video_tar.name}")
    metadata_sha256 = sha256_file(args.metadata_tar)
    if metadata_sha256 != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata SHA256 changed; inspect before sampling")
    video_sha256 = sha256_file(args.video_tar)
    if video_sha256 != EXPECTED_VIDEO_SHA256:
        raise ValueError("video shard SHA256 changed; inspect before sampling")
    members, cases = scan_video_tar(args.video_tar)
    annotations = load_annotations(args.metadata_tar, set(cases))
    population = build_population(cases, annotations)
    if not population:
        raise RuntimeError("no complete annotated cases found")
    MEMBERS_PATH.write_text("\n".join(members) + "\n", encoding="utf-8")
    write_csv(POPULATION_PATH, population)
    indices = write_sample(population)
    print(f"metadata_sha256={metadata_sha256}")
    print(f"video_sha256={video_sha256}")
    print(
        f"members={len(members)} video_cases={len(cases)} "
        f"annotations={len(annotations)} population={len(population)} "
        f"sample_size={SAMPLE_SIZE} seed={PAIR_SEED} indices={indices}"
    )


if __name__ == "__main__":
    main()
