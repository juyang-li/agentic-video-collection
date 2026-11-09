#!/usr/bin/env python3
"""Build the deterministic 4.5k phase-A manifest for the formal Stage 1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from filter_common import classify_target_scope, is_robot_cg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "filter_config.json"
DEFAULT_OUTPUT = ROOT / "artifacts/filter_run_v1/pilot_manifest_phase_a.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/filter_run_v1/pilot_manifest_phase_a_summary.json"

FIELDS = [
    "filter_version",
    "selection_rank",
    "pilot_group",
    "sample_id",
    "dataset",
    "branch",
    "pair_id",
    "operation",
    "target_scope",
    "instruction",
    "repo_id",
    "revision",
    "source_archive",
    "target_archive",
    "source_locator",
    "target_locator",
    "source_local",
    "target_local",
    "source_provenance",
    "target_provenance",
    "provenance_basis",
    "quality_checked_roles",
    "selection_seed",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def excluded_pairs(config_path: Path, config: dict[str, object]) -> set[tuple[str, str]]:
    path = resolve(config_path, str(config["exclusion_manifest"]))
    return {(row["dataset"], row["pair_id"]) for row in read_csv(path)}


def base_row(
    *,
    config: dict[str, object],
    source: dict[str, object],
    dataset: str,
    branch: str,
    pair_id: str,
    operation: str,
    instruction: str,
    source_archive: str,
    target_archive: str,
    source_locator: str,
    target_locator: str,
    source_provenance: str,
    target_provenance: str,
    provenance_basis: str,
    quality_checked_roles: str,
) -> dict[str, str]:
    media_stem = stable_id(f"{dataset}:{pair_id}")
    return {
        "filter_version": str(config["filter_version"]),
        "selection_rank": "",
        "pilot_group": "branch_random",
        "sample_id": f"pilot-{branch}-{stable_id(pair_id)}",
        "dataset": dataset,
        "branch": branch,
        "pair_id": pair_id,
        "operation": operation,
        "target_scope": classify_target_scope(instruction),
        "instruction": instruction.strip(),
        "repo_id": str(source["repo_id"]),
        "revision": str(source["revision"]),
        "source_archive": source_archive,
        "target_archive": target_archive,
        "source_locator": source_locator,
        "target_locator": target_locator,
        "source_local": f"artifacts/filter_run_v1/media/{dataset}/{media_stem}_source.mp4",
        "target_local": f"artifacts/filter_run_v1/media/{dataset}/{media_stem}_target.mp4",
        "source_provenance": source_provenance,
        "target_provenance": target_provenance,
        "provenance_basis": provenance_basis,
        "quality_checked_roles": quality_checked_roles,
        "selection_seed": str(config["seed"]),
    }


def insave_candidates(
    config_path: Path, config: dict[str, object], excluded: set[tuple[str, str]]
) -> list[dict[str, str]]:
    source = config["sources"]["insave_remove"]  # type: ignore[index]
    rows = read_csv(resolve(config_path, str(source["metadata"])))
    output: list[dict[str, str]] = []
    for index, item in enumerate(rows):
        instruction = item["instruction"].strip()
        pair_id = f"insave-{index:05d}"
        if ("insave", pair_id) in excluded or classify_target_scope(instruction) != "person":
            continue
        archive = (
            "train_data/add_and_remove/part_00.tar"
            if index <= 11382
            else "train_data/add_and_remove/part_01.tar"
        )
        output.append(
            base_row(
                config=config,
                source=source,
                dataset="insave",
                branch="insave_remove",
                pair_id=pair_id,
                operation="remove",
                instruction=instruction,
                source_archive=archive,
                target_archive=archive,
                source_locator=Path(item["original_video"]).name,
                target_locator=Path(item["target_video"]).name,
                source_provenance="original",
                target_provenance="edited",
                provenance_basis="official_dataset_rule",
                quality_checked_roles="target",
            )
        )
    return output


def goku_candidates(
    config_path: Path,
    config: dict[str, object],
    branch: str,
    excluded: set[tuple[str, str]],
) -> list[dict[str, str]]:
    source = config["sources"][branch]  # type: ignore[index]
    operation = branch.removeprefix("goku_")
    output: list[dict[str, str]] = []
    for population in source["populations"]:
        for item in read_csv(resolve(config_path, str(population))):
            pair_id = f"goku-{item['case_id']}"
            if ("goku", pair_id) in excluded:
                continue
            reverse_add = operation == "add"
            archive = f"{operation}/{item['shard']}"
            output.append(
                base_row(
                    config=config,
                    source=source,
                    dataset="goku",
                    branch=branch,
                    pair_id=pair_id,
                    operation=operation,
                    instruction=item["instruction"],
                    source_archive=archive,
                    target_archive=archive,
                    source_locator=item["source_member"],
                    target_locator=item["target_member"],
                    source_provenance="edited" if reverse_add else "original",
                    target_provenance="original" if reverse_add else "edited",
                    provenance_basis="official_dataset_rule",
                    quality_checked_roles="source" if reverse_add else "target",
                )
            )
    return output


def ego_direction(source_id: str, target_id: str) -> tuple[str, str, str]:
    source = "original" if source_id.endswith("_no_change") else "edited"
    target = "original" if target_id.endswith("_no_change") else "edited"
    if source == "original" and target == "edited":
        return "original_to_edited", source, target
    if source == "edited" and target == "original":
        return "edited_to_original", source, target
    return "edited_to_edited", source, target


def egoedit_candidates(
    config_path: Path, config: dict[str, object], excluded: set[tuple[str, str]]
) -> dict[str, list[dict[str, str]]]:
    source = config["sources"]["egoedit_replace"]  # type: ignore[index]
    urls = {
        row["data_id"]: row["data_url"]
        for row in read_csv(resolve(config_path, str(source["data_table"])))
    }
    output: dict[str, list[dict[str, str]]] = {
        "original_to_edited": [],
        "edited_to_original": [],
        "edited_to_edited": [],
    }
    for item in read_csv(resolve(config_path, str(source["edits_table"]))):
        source_id = item["source_data_id"]
        target_id = item["target_data_id"]
        instruction = item["source_to_target_edit_prompt"].strip()
        pair_id = f"egoedit-{item['data_id']}"
        if (
            ("egoedit", pair_id) in excluded
            or not instruction
            or source_id not in urls
            or target_id not in urls
            or classify_target_scope(instruction) != "object"
        ):
            continue
        direction, source_provenance, target_provenance = ego_direction(source_id, target_id)
        quality_roles = "both" if direction == "edited_to_edited" else (
            "target" if direction == "original_to_edited" else "source"
        )
        output[direction].append(
            base_row(
                config=config,
                source=source,
                dataset="egoedit",
                branch="egoedit_replace",
                pair_id=pair_id,
                operation="replace",
                instruction=instruction,
                source_archive="egoedit_remote_zip",
                target_archive="egoedit_remote_zip",
                source_locator=Path(urlparse(urls[source_id]).path).name,
                target_locator=Path(urlparse(urls[target_id]).path).name,
                source_provenance=source_provenance,
                target_provenance=target_provenance,
                provenance_basis="pair_metadata",
                quality_checked_roles=quality_roles,
            )
        )
    return output


def reco_candidates(
    config_path: Path,
    config: dict[str, object],
    branch: str,
    excluded: set[tuple[str, str]],
) -> list[dict[str, str]]:
    source = config["sources"][branch]  # type: ignore[index]
    operation = branch.removeprefix("reco_")
    path = resolve(config_path, str(source["metadata"]))
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, item in enumerate(rows):
        item = dict(item)
        item["_metadata_index"] = index
        grouped[Path(item["src_video"]).stem].append(item)
    output: list[dict[str, str]] = []
    for stem, matches in grouped.items():
        if len(matches) != 1:
            continue
        item = matches[0]
        shard = int(item["_metadata_index"]) // 2000
        if shard not in {int(value) for value in source["candidate_shards"]}:
            continue
        pair_id = f"reco-{operation}-{stem}"
        if ("reco", stem) in excluded or ("reco", pair_id) in excluded:
            continue
        output.append(
            base_row(
                config=config,
                source=source,
                dataset="reco",
                branch=branch,
                pair_id=pair_id,
                operation=operation,
                instruction=item["instruction_final_refine"],
                source_archive=f"{operation}/src_videos/{operation}_src_video_{shard:02d}.tar",
                target_archive=f"{operation}/tar_videos/{operation}_tar_video_{shard:02d}.tar",
                source_locator=item["src_video"],
                target_locator=item["tar_video"],
                source_provenance="original",
                target_provenance="edited",
                provenance_basis="official_dataset_rule",
                quality_checked_roles="target",
            )
        )
    return output


def sample_rows(rows: list[dict[str, str]], quota: int, seed: str) -> list[dict[str, str]]:
    if len(rows) < quota:
        raise ValueError(f"cannot fill quota={quota}; candidates={len(rows)} seed={seed}")
    return random.Random(seed).sample(rows, quota)


def balanced_reco_sample(
    by_branch: dict[str, list[dict[str, str]]],
    predicate,
    branch_quotas: dict[str, int],
    seed: int,
    used: set[tuple[str, str]],
) -> list[dict[str, str]]:
    branches = ("reco_add", "reco_remove", "reco_replace")
    output: list[dict[str, str]] = []
    for branch in branches:
        candidates = [
            row for row in by_branch[branch]
            if (row["dataset"], row["pair_id"]) not in used and predicate(row)
        ]
        chosen = sample_rows(candidates, int(branch_quotas[branch]), f"{seed}:{branch}")
        output.extend(chosen)
        used.update((row["dataset"], row["pair_id"]) for row in chosen)
    return output


def build(config_path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    base_quota = int(config["base_quota_per_branch"])
    excluded = excluded_pairs(config_path, config)

    by_branch: dict[str, list[dict[str, str]]] = {
        "insave_remove": insave_candidates(config_path, config, excluded),
        "goku_remove": goku_candidates(config_path, config, "goku_remove", excluded),
        "goku_add": goku_candidates(config_path, config, "goku_add", excluded),
    }
    ego = egoedit_candidates(config_path, config, excluded)
    ego_selected: list[dict[str, str]] = []
    ego_source = config["sources"]["egoedit_replace"]  # type: ignore[index]
    for direction, quota in ego_source["direction_quota"].items():
        ego_selected.extend(sample_rows(ego[direction], int(quota), f"{seed}:ego:{direction}"))
    by_branch["egoedit_replace"] = ego_selected
    for branch in ("reco_add", "reco_remove", "reco_replace"):
        by_branch[branch] = reco_candidates(config_path, config, branch, excluded)

    targeted = config["targeted_quota"]
    used: set[tuple[str, str]] = set()
    robot_rows = balanced_reco_sample(
        by_branch,
        lambda row: is_robot_cg(row["instruction"]),
        targeted["reco_robot_cg"],
        seed + 101,
        used,
    )
    for row in robot_rows:
        row["pilot_group"] = "reco_robot_cg"
    clothes_rows = balanced_reco_sample(
        by_branch,
        lambda row: row["target_scope"] == "clothes",
        targeted["reco_clothes"],
        seed + 102,
        used,
    )
    for row in clothes_rows:
        row["pilot_group"] = "reco_clothes"

    selected: list[dict[str, str]] = []
    for branch in (
        "insave_remove",
        "goku_remove",
        "goku_add",
        "egoedit_replace",
        "reco_add",
        "reco_remove",
        "reco_replace",
    ):
        rows = by_branch[branch]
        if branch == "egoedit_replace":
            chosen = rows
        else:
            candidates = [
                row for row in rows if (row["dataset"], row["pair_id"]) not in used
            ]
            if branch.startswith("reco_"):
                base_shard = int(config["sources"][branch]["base_shard"])  # type: ignore[index]
                archive_suffix = f"_{base_shard:02d}.tar"
                candidates = [
                    row for row in candidates if row["source_archive"].endswith(archive_suffix)
                ]
            chosen = sample_rows(candidates, base_quota, f"{seed}:base:{branch}")
        selected.extend(chosen)
    selected.extend(robot_rows)
    selected.extend(clothes_rows)

    for rank, row in enumerate(selected, 1):
        row["selection_rank"] = str(rank)
    identities = [(row["dataset"], row["pair_id"]) for row in selected]
    if len(selected) != 4500 or len(identities) != len(set(identities)):
        raise ValueError(
            "pilot must contain exactly 4,500 unique phase-A pairs; "
            f"rows={len(selected)} unique={len(set(identities))}"
        )

    summary = {
        "filter_version": config["filter_version"],
        "seed": seed,
        "phase": "A",
        "rows": len(selected),
        "remaining_boundary_quota_after_stage1_scoring": 500,
        "group_counts": dict(Counter(row["pilot_group"] for row in selected)),
        "branch_counts": dict(Counter(row["branch"] for row in selected)),
        "scope_counts": dict(Counter(row["target_scope"] for row in selected)),
        "quality_role_counts": dict(Counter(row["quality_checked_roles"] for row in selected)),
        "source_archive_counts": dict(Counter(row["source_archive"] for row in selected)),
        "target_archive_counts": dict(Counter(row["target_archive"] for row in selected)),
        "excluded_review_pairs": len(excluded),
    }
    return selected, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    rows, summary = build(args.config.resolve())
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
