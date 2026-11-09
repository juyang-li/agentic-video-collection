#!/usr/bin/env python3
"""Build the frozen, branch-balanced 400-pair error-survey manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "survey_config.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "survey_manifest.csv"

FIELDS = [
    "survey_version", "sample_rank", "sample_id", "dataset", "branch",
    "pair_id", "operation", "direction", "instruction", "repo_id", "revision",
    "shard", "source_video_id", "target_video_id", "source_locator",
    "target_locator", "source_local", "target_local", "source_is_original",
    "target_is_original", "source_provenance", "target_provenance",
    "provenance_basis", "quality_checked_roles", "selection_seed",
]

OFFICIAL_PROVENANCE_RULES = {
    (
        "suimu/InsAVE-80K",
        "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289",
        "insave_remove",
    ): ("original", "edited"),
    (
        "bigfacing/GOKU-2M",
        "f289f4b245648dc57db04afaaeb68659440c0fa6",
        "goku_remove",
    ): ("original", "edited"),
    (
        "bigfacing/GOKU-2M",
        "f289f4b245648dc57db04afaaeb68659440c0fa6",
        "goku_add",
    ): ("edited", "original"),
}


def provenance_fields(
    source: str, target: str, basis: str
) -> dict[str, str]:
    """Build compatibility flags and specialist routing from pair roles."""
    if source == "original" and target == "edited":
        direction = "original_to_edited"
    elif source == "edited" and target == "original":
        direction = "edited_to_original"
    elif source == "edited" and target == "edited":
        direction = "edited_to_edited"
    elif source == "original" and target == "original":
        direction = "original_to_original"
    else:
        direction = "unknown"

    edited_roles = [
        role for role, value in (("source", source), ("target", target))
        if value == "edited"
    ]
    quality_roles = edited_roles[0] if len(edited_roles) == 1 else "both"
    return {
        "direction": direction,
        "source_is_original": "1" if source == "original" else "0" if source == "edited" else "",
        "target_is_original": "1" if target == "original" else "0" if target == "edited" else "",
        "source_provenance": source,
        "target_provenance": target,
        "provenance_basis": basis,
        "quality_checked_roles": quality_roles,
    }


def dataset_provenance(repo_id: object, revision: object, branch: str) -> dict[str, str]:
    roles = OFFICIAL_PROVENANCE_RULES.get((str(repo_id), str(revision), branch))
    if roles is None:
        return provenance_fields("unknown", "unknown", "unknown")
    return provenance_fields(*roles, "official_dataset_rule")


def resolve_config_path(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def stable_short_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def basename(url: str) -> str:
    return Path(urlparse(url).path).name


def excluded_values(path: Path | None, field: str) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {row.get(field, "") for row in read_csv(path) if row.get(field)}


def build_insave(
    config_path: Path, survey_version: str, seed: int, config: dict[str, object]
) -> list[dict[str, object]]:
    metadata_path = resolve_config_path(config_path, str(config["metadata"]))
    metadata = read_csv(metadata_path)
    exclude_path = resolve_config_path(config_path, str(config["exclude_manifest"]))
    excluded = {int(value) for value in excluded_values(exclude_path, "metadata_index")}
    output: list[dict[str, object]] = []
    provenance = dataset_provenance(config["repo_id"], config["revision"], "insave_remove")
    for shard_index, shard in enumerate(config["shards"]):  # type: ignore[index]
        first = int(shard["first_index"])
        last = int(shard["last_index"])
        candidates = [
            index
            for index in range(first, min(last, len(metadata) - 1) + 1)
            if index not in excluded
            and metadata[index]["instruction"].strip().lower().startswith("remove")
        ]
        quota = int(shard["quota"])
        if len(candidates) < quota:
            raise RuntimeError(f"InsAVE {shard['name']} has only {len(candidates)} candidates")
        chosen = random.Random(seed + 101 + shard_index).sample(candidates, quota)
        for index in chosen:
            item = metadata[index]
            pair_id = f"insave-{index:05d}"
            output.append(
                {
                    "survey_version": survey_version,
                    "sample_id": pair_id,
                    "dataset": "insave",
                    "branch": "insave_remove",
                    "pair_id": pair_id,
                    "operation": "remove",
                    "direction": provenance["direction"],
                    "instruction": item["instruction"].strip(),
                    "repo_id": config["repo_id"],
                    "revision": config["revision"],
                    "shard": shard["archive"],
                    "source_video_id": f"insave:{index}:original",
                    "target_video_id": f"insave:{index}:target",
                    "source_locator": f"{index:05d}_original.mp4",
                    "target_locator": f"{index:05d}_traget.mp4",
                    "source_local": f"artifacts/originals/insave/{pair_id}_source.mp4",
                    "target_local": f"artifacts/originals/insave/{pair_id}_target.mp4",
                    **provenance,
                    "selection_seed": seed + 101 + shard_index,
                }
            )
    return output


def build_goku(
    config_path: Path,
    survey_version: str,
    seed: int,
    branch: str,
    operation: str,
    config: dict[str, object],
) -> list[dict[str, object]]:
    exclude = None
    if config.get("exclude_manifest"):
        exclude = resolve_config_path(config_path, str(config["exclude_manifest"]))
    excluded = excluded_values(exclude, "case_id")
    output: list[dict[str, object]] = []
    provenance = dataset_provenance(config["repo_id"], config["revision"], branch)
    for shard_index, shard in enumerate(config["shards"]):  # type: ignore[index]
        population_path = resolve_config_path(config_path, str(shard["population"]))
        if not population_path.exists():
            raise FileNotFoundError(
                f"missing {population_path}; run prepare_goku_population.py first"
            )
        candidates = [
            row for row in read_csv(population_path) if row["case_id"] not in excluded
        ]
        quota = int(shard["quota"])
        if len(candidates) < quota:
            raise RuntimeError(f"{branch} {shard['name']} has only {len(candidates)} candidates")
        chosen = random.Random(seed + 201 + shard_index).sample(candidates, quota)
        for item in chosen:
            case_id = item["case_id"]
            pair_id = f"goku-{case_id}"
            short = stable_short_id(pair_id)
            output.append(
                {
                    "survey_version": survey_version,
                    "sample_id": f"{branch}-{short}",
                    "dataset": "goku",
                    "branch": branch,
                    "pair_id": pair_id,
                    "operation": operation,
                    "direction": provenance["direction"],
                    "instruction": item["instruction"].strip(),
                    "repo_id": config["repo_id"],
                    "revision": config["revision"],
                    "shard": shard["archive"],
                    "source_video_id": f"goku:{case_id}:source",
                    "target_video_id": f"goku:{case_id}:edited",
                    "source_locator": item["source_member"],
                    "target_locator": item["target_member"],
                    "source_local": f"artifacts/originals/goku/{short}_source.mp4",
                    "target_local": f"artifacts/originals/goku/{short}_target.mp4",
                    **provenance,
                    "selection_seed": seed + 201 + shard_index,
                }
            )
    return output


def ego_direction(source_id: str, target_id: str) -> str:
    source_original = source_id.endswith("_no_change")
    target_original = target_id.endswith("_no_change")
    if source_original and not target_original:
        return "original_to_edited"
    if not source_original and target_original:
        return "edited_to_original"
    if not source_original and not target_original:
        return "edited_to_edited"
    return "original_to_original"


def scene_key(data_id: str) -> str:
    return "_".join(data_id.split("_")[:2])


def build_egoedit(
    config_path: Path, survey_version: str, seed: int, config: dict[str, object]
) -> list[dict[str, object]]:
    data_path = resolve_config_path(config_path, str(config["data_table"]))
    edits_path = resolve_config_path(config_path, str(config["edits_table"]))
    exclude_path = resolve_config_path(config_path, str(config["exclude_manifest"]))
    urls = {row["data_id"]: row["data_url"] for row in read_csv(data_path)}
    excluded = excluded_values(exclude_path, "edit_id")
    by_direction: dict[str, list[dict[str, str]]] = {}
    for item in read_csv(edits_path):
        if item["data_id"] in excluded or not item["source_to_target_edit_prompt"].strip():
            continue
        if item["source_data_id"] not in urls or item["target_data_id"] not in urls:
            continue
        direction = ego_direction(item["source_data_id"], item["target_data_id"])
        by_direction.setdefault(direction, []).append(item)

    selected: list[dict[str, str]] = []
    used_scenes: set[str] = set()
    quotas = config["direction_quota"]  # type: ignore[index]
    for direction_index, (direction, raw_quota) in enumerate(quotas.items()):
        candidates = list(by_direction.get(direction, []))
        random.Random(seed + 301 + direction_index).shuffle(candidates)
        quota = int(raw_quota)
        for item in candidates:
            key = scene_key(item["source_data_id"])
            if key in used_scenes:
                continue
            selected.append(item)
            used_scenes.add(key)
            if sum(ego_direction(row["source_data_id"], row["target_data_id"]) == direction for row in selected) == quota:
                break
        else:
            raise RuntimeError(f"EgoEdit cannot fill {direction} quota={quota} with unique scenes")

    output: list[dict[str, object]] = []
    for item in selected:
        edit_id = item["data_id"]
        source_id = item["source_data_id"]
        target_id = item["target_data_id"]
        direction = ego_direction(source_id, target_id)
        source_provenance = "original" if source_id.endswith("_no_change") else "edited"
        target_provenance = "original" if target_id.endswith("_no_change") else "edited"
        provenance = provenance_fields(source_provenance, target_provenance, "pair_metadata")
        source_url = urls[source_id]
        target_url = urls[target_id]
        pair_id = f"egoedit-{edit_id}"
        short = stable_short_id(pair_id)
        output.append(
            {
                "survey_version": survey_version,
                "sample_id": f"egoedit-replace-{short}",
                "dataset": "egoedit",
                "branch": "egoedit_replace",
                "pair_id": pair_id,
                "operation": "replace",
                "direction": provenance["direction"],
                "instruction": item["source_to_target_edit_prompt"].strip(),
                "repo_id": config["repo_id"],
                "revision": config["revision"],
                "shard": "egoedit_remote_zip",
                "source_video_id": f"egoedit:{source_id}",
                "target_video_id": f"egoedit:{target_id}",
                "source_locator": basename(source_url),
                "target_locator": basename(target_url),
                "source_local": f"artifacts/originals/egoedit/{basename(source_url)}",
                "target_local": f"artifacts/originals/egoedit/{basename(target_url)}",
                **provenance,
                "selection_seed": seed + 301,
            }
        )
    return output


def validate(rows: list[dict[str, object]], config: dict[str, object]) -> None:
    if len(rows) != 400:
        raise ValueError(f"expected 400 rows, got {len(rows)}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate sample_id")
    expected = {name: int(item["quota"]) for name, item in config["branches"].items()}  # type: ignore[index]
    actual = {name: sum(row["branch"] == name for row in rows) for name in expected}
    if actual != expected:
        raise ValueError(f"branch counts differ: actual={actual}, expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    version = str(config["survey_version"])
    seed = int(config["seed"])
    branches = config["branches"]
    rows: list[dict[str, object]] = []
    rows.extend(build_insave(config_path, version, seed, branches["insave_remove"]))
    rows.extend(build_goku(config_path, version, seed + 10, "goku_remove", "remove", branches["goku_remove"]))
    rows.extend(build_goku(config_path, version, seed + 20, "goku_add", "add", branches["goku_add"]))
    rows.extend(build_egoedit(config_path, version, seed + 30, branches["egoedit_replace"]))
    validate(rows, config)
    for rank, row in enumerate(rows, 1):
        row["sample_rank"] = rank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"survey_version={version} rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
