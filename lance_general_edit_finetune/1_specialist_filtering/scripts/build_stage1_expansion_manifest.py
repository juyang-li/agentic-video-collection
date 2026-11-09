#!/usr/bin/env python3
"""Build deterministic, metadata-qualified expansion rounds for the 4K edit pool."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_filter_pilot as pilot
from run_stage1_metadata import evaluate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "filter_config.json"
DEFAULT_RUN = ROOT / "artifacts/filter_run_v3"
VERSION = "general_edit_stage1_quality_v2"
DEFAULT_QUOTAS = {
    "goku_remove": 1100,
    "goku_add": 600,
    "egoedit_replace": 700,
    "reco_remove": 300,
}
SCOPE_WEIGHTS = {
    "goku_remove": {"object": 0.60, "accessory": 0.25, "person": 0.15},
    "goku_add": {"object": 0.60, "accessory": 0.20, "person": 0.20},
    "egoedit_replace": {"object": 1.0},
    "reco_add": {"object": 0.50, "person": 0.45, "accessory": 0.03, "clothes": 0.02},
    "reco_remove": {"object": 0.50, "person": 0.45, "accessory": 0.03, "clothes": 0.02},
    "reco_replace": {"object": 0.55, "person": 0.35, "accessory": 0.03, "clothes": 0.02, "appearance": 0.03, "background": 0.02},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_quotas(values: list[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_QUOTAS)
    result: dict[str, int] = {}
    for value in values:
        branch, separator, raw_count = value.partition("=")
        if not separator or branch not in SCOPE_WEIGHTS or int(raw_count) < 0:
            raise ValueError(f"invalid --quota: {value}")
        result[branch] = int(raw_count)
    return result


def parse_archives(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        branch, separator, archive = value.partition("=")
        if not separator or branch not in SCOPE_WEIGHTS or not archive:
            raise ValueError(f"invalid --archive: {value}")
        result[branch] = archive
    return result


def candidate_map(
    config_path: Path, config: dict[str, object], branches: set[str]
) -> dict[str, list[dict[str, str]]]:
    excluded = pilot.excluded_pairs(config_path, config)
    result: dict[str, list[dict[str, str]]] = {}
    for branch in branches & {"goku_remove", "goku_add"}:
        result[branch] = pilot.goku_candidates(config_path, config, branch, excluded)
    if "egoedit_replace" in branches:
        ego = pilot.egoedit_candidates(config_path, config, excluded)
        result["egoedit_replace"] = [row for values in ego.values() for row in values]
    for branch in branches & {"reco_add", "reco_remove", "reco_replace"}:
        result[branch] = pilot.reco_candidates(config_path, config, branch, excluded)
    return result


def stratified_sample(
    rows: list[dict[str, str]], branch: str, quota: int, seed: int
) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["target_scope"]].append(row)
    rng = random.Random(f"{seed}:{branch}")
    for values in buckets.values():
        rng.shuffle(values)
    weights = SCOPE_WEIGHTS[branch]
    target = {scope: int(quota * weight) for scope, weight in weights.items()}
    for scope in sorted(weights, key=lambda name: (-weights[name], name)):
        if sum(target.values()) >= quota:
            break
        target[scope] += 1
    selected: list[dict[str, str]] = []
    for scope, count in target.items():
        take = min(count, len(buckets.get(scope, [])))
        selected.extend(buckets[scope][:take])
        buckets[scope] = buckets[scope][take:]
    remainder = [row for scope in sorted(buckets) for row in buckets[scope]]
    rng.shuffle(remainder)
    selected.extend(remainder[: quota - len(selected)])
    if len(selected) != quota:
        raise ValueError(f"branch={branch} quota={quota} available={len(rows)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--quota", action="append")
    parser.add_argument(
        "--archive",
        action="append",
        help="Optionally restrict one branch to an exact source archive: BRANCH=PATH.",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "expansion_round1_input.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "expansion_round1_input_summary.json")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quotas = parse_quotas(args.quota)
    archives = parse_archives(args.archive)
    used: set[tuple[str, str]] = set()
    default_exclude = ROOT / "artifacts/filter_run_v1/pilot_manifest_5k.csv"
    for path in [default_exclude, *args.exclude]:
        if path.is_file():
            used.update((row["dataset"], row["pair_id"]) for row in read_csv(path))

    all_candidates = candidate_map(config_path, config, set(quotas))
    selected: list[dict[str, str]] = []
    rejected_before_media: Counter[str] = Counter()
    for branch, quota in quotas.items():
        eligible: list[dict[str, str]] = []
        for row in all_candidates[branch]:
            if (row["dataset"], row["pair_id"]) in used:
                continue
            if branch in archives and row["source_archive"] != archives[branch]:
                continue
            if branch.startswith("reco_"):
                base_shard = int(config["sources"][branch]["base_shard"])
                if not row["source_archive"].endswith(f"_{base_shard:02d}.tar"):
                    continue
            item = dict(row)
            item["filter_version"] = VERSION
            item["sample_id"] = item["sample_id"].replace("pilot-", f"scale-r{args.round}-", 1)
            item["source_local"] = item["source_local"].replace(
                "artifacts/filter_run_v1/media", f"artifacts/filter_run_v3/media/round{args.round}"
            )
            item["target_local"] = item["target_local"].replace(
                "artifacts/filter_run_v1/media", f"artifacts/filter_run_v3/media/round{args.round}"
            )
            if item["dataset"] == "reco":
                item.update(
                    {
                        "source_provenance": "generation_side_unresolved",
                        "target_provenance": "generation_side_unresolved",
                        "provenance_basis": "official_training_roles_without_pair_lineage",
                        "quality_checked_roles": "both",
                    }
                )
            metadata = evaluate(item)
            if metadata["metadata_decision"] != "accepted":
                rejected_before_media.update(metadata["metadata_reasons"].split("|"))
                continue
            eligible.append(item)
        chosen = stratified_sample(eligible, branch, quota, args.seed + args.round)
        selected.extend(chosen)
        used.update((row["dataset"], row["pair_id"]) for row in chosen)

    for rank, row in enumerate(selected, 1):
        row["selection_rank"] = str(rank)
        row["pilot_group"] = f"scale_expansion_round{args.round}"
    identities = [(row["dataset"], row["pair_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("expansion manifest contains duplicate pairs")
    fields = list(selected[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)
    summary = {
        "filter_version": VERSION,
        "selection_seed": args.seed,
        "round": args.round,
        "rows": len(selected),
        "requested_quotas": quotas,
        "archive_restrictions": archives,
        "by_branch": dict(Counter(row["branch"] for row in selected)),
        "by_branch_scope": dict(Counter(f"{row['branch']}:{row['target_scope']}" for row in selected)),
        "source_archives": dict(Counter(row["source_archive"] for row in selected)),
        "metadata_rejected_before_selection": dict(rejected_before_media),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
