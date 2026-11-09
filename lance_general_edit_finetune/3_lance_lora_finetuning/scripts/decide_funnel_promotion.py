#!/usr/bin/env python3
"""Promote the numeric val48 winner; report paired bootstrap as diagnostics only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random


BRANCHES = (
    "goku_remove",
    "goku_add",
    "egoedit_replace",
    "reco_add",
    "reco_remove",
    "reco_replace",
)
CANDIDATES = (451, 677, 902)
EPOCHS = {451: 1.0, 677: 1.5, 902: 2.0}
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
MINIMUM_BRANCH_WINS = 5


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    evaluation = root / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-column", default="overall_score")
    parser.add_argument(
        "--screen-manifest", type=Path, default=evaluation / "validation_screen48.csv"
    )
    parser.add_argument(
        "--remaining-manifest",
        type=Path,
        default=evaluation / "validation_remaining147.csv",
    )
    parser.add_argument(
        "--decision-output", type=Path, default=evaluation / "promotion_decision.json"
    )
    parser.add_argument(
        "--tasks-output", type=Path, default=evaluation / "promotion_tasks.csv"
    )
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_seed(winner: int, competitor: int) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_SEED}\0{winner}\0{competitor}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def paired_comparison(
    scores: dict[int, dict[str, float]],
    sample_branches: dict[str, str],
    winner: int,
    competitor: int,
) -> dict[str, object]:
    differences: dict[str, list[float]] = defaultdict(list)
    for sample_id, branch in sample_branches.items():
        differences[branch].append(scores[winner][sample_id] - scores[competitor][sample_id])
    branch_means = {
        branch: sum(differences[branch]) / len(differences[branch]) for branch in BRANCHES
    }
    rng = random.Random(bootstrap_seed(winner, competitor))
    bootstrap = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_branch_means = []
        for branch in BRANCHES:
            values = differences[branch]
            sampled_branch_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        bootstrap.append(sum(sampled_branch_means) / len(sampled_branch_means))
    lower = quantile(bootstrap, 0.025)
    upper = quantile(bootstrap, 0.975)
    branch_wins = sum(value > 0.0 for value in branch_means.values())
    return {
        "competitor_step": competitor,
        "paired_macro_difference": sum(branch_means.values()) / len(BRANCHES),
        "bootstrap_95_ci": [lower, upper],
        "branch_mean_differences": branch_means,
        "branch_mean_wins": branch_wins,
        "clear": lower > 0.0 and branch_wins >= MINIMUM_BRANCH_WINS,
    }


def main() -> None:
    args = parse_args()
    with args.screen_manifest.open(encoding="utf-8", newline="") as stream:
        screen = list(csv.DictReader(stream))
    sample_branches = {row["sample_id"]: row["branch"] for row in screen}
    if len(screen) != 48 or len(sample_branches) != 48:
        raise ValueError("screen manifest must contain 48 unique samples")
    if {branch: list(sample_branches.values()).count(branch) for branch in BRANCHES} != {
        branch: 8 for branch in BRANCHES
    }:
        raise ValueError("screen manifest must contain exactly eight samples per branch")

    scores: dict[int, dict[str, float]] = {step: {} for step in CANDIDATES}
    with args.scores.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["checkpoint_step"])
            if step not in scores:
                raise ValueError(f"unexpected checkpoint step in scores: {step}")
            sample_id = row["sample_id"]
            if sample_id not in sample_branches:
                raise ValueError(f"unexpected sample in scores: {sample_id}")
            if sample_id in scores[step]:
                raise ValueError(f"duplicate score for step={step}, sample={sample_id}")
            value = float(row[args.score_column])
            if not math.isfinite(value):
                raise ValueError(f"non-finite score for step={step}, sample={sample_id}")
            scores[step][sample_id] = value
    for step in CANDIDATES:
        missing = set(sample_branches) - set(scores[step])
        if missing:
            raise ValueError(f"checkpoint {step} is missing {len(missing)} screen scores")

    macro_scores = {}
    for step in CANDIDATES:
        means = []
        for branch in BRANCHES:
            values = [
                scores[step][sample_id]
                for sample_id, sample_branch in sample_branches.items()
                if sample_branch == branch
            ]
            means.append(sum(values) / len(values))
        macro_scores[step] = sum(means) / len(means)
    ranking = sorted(CANDIDATES, key=lambda step: (-macro_scores[step], step))
    winner = ranking[0]
    comparisons = [
        paired_comparison(scores, sample_branches, winner, competitor)
        for competitor in ranking[1:]
    ]
    clear = all(bool(comparison["clear"]) for comparison in comparisons)
    # Statistical confidence remains useful diagnostic evidence, but the
    # frozen compute policy always expands only the numeric winner.
    promotion_steps = [winner]

    with args.remaining_manifest.open(encoding="utf-8", newline="") as stream:
        remaining = list(csv.DictReader(stream))
    if len(remaining) != 147:
        raise ValueError("remaining validation manifest must contain 147 samples")
    tasks = []
    for step in promotion_steps:
        for row in remaining:
            tasks.append(
                {
                    "task_index": len(tasks) + 1,
                    "checkpoint_step": step,
                    "checkpoint_epoch": EPOCHS[step],
                    **row,
                }
            )
    args.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    with args.tasks_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)

    decision = {
        "schema_version": "lance_general_edit_funnel_promotion_decision_v1",
        "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "screen_manifest_sha256": hashlib.sha256(args.screen_manifest.read_bytes()).hexdigest(),
        "remaining_manifest_sha256": hashlib.sha256(args.remaining_manifest.read_bytes()).hexdigest(),
        "score_column": args.score_column,
        "screen_samples_per_checkpoint": 48,
        "six_branch_macro_scores": {str(step): macro_scores[step] for step in CANDIDATES},
        "ranking": ranking,
        "provisional_winner_step": winner,
        "comparisons": comparisons,
        "clear_winner": clear,
        "promotion_action": "winner_only_by_numeric_rank",
        "promotion_steps": promotion_steps,
        "promotion_task_count": len(tasks),
        "rule": {
            "role": "diagnostic_only_not_a_promotion_gate",
            "paired_bootstrap_confidence": 0.95,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "lower_bound_must_exceed": 0.0,
            "minimum_branch_mean_wins": MINIMUM_BRANCH_WINS,
            "must_beat_every_other_candidate": True,
        },
    }
    args.decision_output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
