#!/usr/bin/env python3
"""Build 500 new archive-constrained follow-up pairs from Phase-A outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import build_filter_pilot as pilot
from filter_common import is_robot_cg, metadata_rejection_reasons
from run_stage1_metadata import ALLOWED_SCOPES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"
BRANCHES = (
    "insave_remove",
    "goku_remove",
    "goku_add",
    "egoedit_replace",
    "reco_add",
    "reco_remove",
    "reco_replace",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def adaptive_quotas(
    decisions: list[dict[str, str]], total: int = 500, minimum: int = 50
) -> dict[str, int]:
    if minimum * len(BRANCHES) > total:
        raise ValueError("minimum branch coverage exceeds total Phase-B quota")
    rates: dict[str, float] = {}
    for branch in BRANCHES:
        rows = [row for row in decisions if row["branch"] == branch]
        scored = [row for row in rows if row["stage1_1_decision"] != "pending"]
        rates[branch] = (
            sum(row["stage1_1_decision"] == "uncertain" for row in scored) / len(scored)
            if scored
            else 0.0
        )
    remaining = total - minimum * len(BRANCHES)
    weight_sum = sum(rates.values())
    weights = rates if weight_sum else {branch: 1.0 for branch in BRANCHES}
    weight_sum = sum(weights.values())
    raw = {branch: remaining * weights[branch] / weight_sum for branch in BRANCHES}
    quotas = {branch: minimum + math.floor(raw[branch]) for branch in BRANCHES}
    leftovers = total - sum(quotas.values())
    order = sorted(BRANCHES, key=lambda branch: (raw[branch] % 1, branch), reverse=True)
    for branch in order[:leftovers]:
        quotas[branch] += 1
    return quotas


def all_candidates(config_path: Path, config: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    excluded = pilot.excluded_pairs(config_path, config)
    ego = pilot.egoedit_candidates(config_path, config, excluded)
    return {
        "insave_remove": pilot.insave_candidates(config_path, config, excluded),
        "goku_remove": pilot.goku_candidates(config_path, config, "goku_remove", excluded),
        "goku_add": pilot.goku_candidates(config_path, config, "goku_add", excluded),
        "egoedit_replace": [row for direction in ego.values() for row in direction],
        "reco_add": pilot.reco_candidates(config_path, config, "reco_add", excluded),
        "reco_remove": pilot.reco_candidates(config_path, config, "reco_remove", excluded),
        "reco_replace": pilot.reco_candidates(config_path, config, "reco_replace", excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "filter_config.json")
    parser.add_argument("--phase-a", type=Path, default=DEFAULT_RUN / "pilot_manifest_phase_a.csv")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "pilot_manifest_phase_b.csv")
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_RUN / "pilot_manifest_5k.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "pilot_manifest_phase_b_summary.json")
    parser.add_argument("--quota", type=int, default=500)
    parser.add_argument("--minimum-per-branch", type=int, default=50)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase_a = read_csv(args.phase_a)
    decisions = read_csv(args.decisions)
    if any(row["stage1_1_decision"] == "pending" for row in decisions):
        raise ValueError("Phase A still has pending decisions; finish scoring before Phase B")
    quotas = adaptive_quotas(decisions, args.quota, args.minimum_per_branch)
    used = {(row["dataset"], row["pair_id"]) for row in phase_a}
    candidates = all_candidates(config_path, config)
    selected: list[dict[str, str]] = []
    seed = int(config["seed"]) + 200
    for branch in BRANCHES:
        available = [
            row
            for row in candidates[branch]
            if (row["dataset"], row["pair_id"]) not in used
            and not metadata_rejection_reasons(
                row["dataset"], row["operation"], row["instruction"]
            )
            and row["target_scope"] in ALLOWED_SCOPES[branch]
        ]
        quota = quotas[branch]
        if len(available) < quota:
            raise ValueError(f"branch={branch} quota={quota} available={len(available)}")
        chosen = random.Random(f"{seed}:{branch}").sample(available, quota)
        for row in chosen:
            row["selection_seed"] = str(seed)
            if row["dataset"] == "reco" and is_robot_cg(row["instruction"]):
                row["pilot_group"] = "phase_b_robot_cg"
            elif row["dataset"] == "reco" and row["target_scope"] == "clothes":
                row["pilot_group"] = "phase_b_clothes"
            else:
                row["pilot_group"] = "phase_b_adaptive"
        selected.extend(chosen)
        used.update((row["dataset"], row["pair_id"]) for row in chosen)

    for rank, row in enumerate(selected, len(phase_a) + 1):
        row["selection_rank"] = str(rank)
    identities = [(row["dataset"], row["pair_id"]) for row in selected]
    if len(selected) != args.quota or len(identities) != len(set(identities)):
        raise ValueError("Phase B must contain the requested number of unique pairs")
    pilot.write_csv(args.output, selected)
    pilot.write_csv(args.combined_output, phase_a + selected)
    summary = {
        "filter_version": config["filter_version"],
        "phase": "B",
        "seed": seed,
        "rows": len(selected),
        "combined_rows": len(phase_a) + len(selected),
        "selection_rule": "50-per-branch floor; remainder proportional to Phase-A uncertain rate",
        "branch_quotas": quotas,
        "branch_counts": dict(Counter(row["branch"] for row in selected)),
        "group_counts": dict(Counter(row["pilot_group"] for row in selected)),
        "source_archive_counts": dict(Counter(row["source_archive"] for row in selected)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
