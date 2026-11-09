#!/usr/bin/env python3
"""Build a complete-case population CSV for one GOKU video shard."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path


def scan_video_tar(path: Path) -> dict[str, dict[str, str]]:
    cases: dict[str, dict[str, str]] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            parts = Path(name).parts
            if not member.isfile() or len(parts) != 3 or parts[0] != "videos":
                continue
            if parts[2] not in {"source.mp4", "edited.mp4"}:
                continue
            cases.setdefault(parts[1], {})[parts[2]] = name
    return cases


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
                continue
            try:
                annotation = json.load(source)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            annotations[case_id] = (name, annotation)
    return annotations


def build_rows(
    task: str,
    shard: str,
    cases: dict[str, dict[str, str]],
    annotations: dict[str, tuple[str, dict[str, object]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_id in sorted(cases):
        roles = cases[case_id]
        if set(roles) != {"source.mp4", "edited.mp4"} or case_id not in annotations:
            continue
        metadata_member, annotation = annotations[case_id]
        instruction = str(annotation.get("instruction_en", "")).strip()
        if not instruction:
            continue
        source_member = roles["source.mp4"]
        target_member = roles["edited.mp4"]
        if annotation.get("source_video") != source_member:
            raise ValueError(f"source member mismatch for {case_id}")
        if annotation.get("edited_video") != target_member:
            raise ValueError(f"target member mismatch for {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "task": task,
                "instruction": instruction,
                "shard": shard,
                "source_member": source_member,
                "target_member": target_member,
                "metadata_member": metadata_member,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("add", "remove"), required=True)
    parser.add_argument("--metadata-tar", type=Path, required=True)
    parser.add_argument("--video-tar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = scan_video_tar(args.video_tar)
    annotations = load_annotations(args.metadata_tar, set(cases))
    rows = build_rows(args.task, args.video_tar.name, cases, annotations)
    if not rows:
        raise RuntimeError("no complete annotated GOKU cases found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(
        f"task={args.task} shard={args.video_tar.name} "
        f"video_cases={len(cases)} annotations={len(annotations)} population={len(rows)}"
    )


if __name__ == "__main__":
    main()
