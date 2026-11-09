#!/usr/bin/env python3
"""Freeze the adaptive validation funnel and final internal-test manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


BRANCHES = (
    "goku_remove",
    "goku_add",
    "egoedit_replace",
    "reco_add",
    "reco_remove",
    "reco_replace",
)
SEED = 42
CANDIDATES = ((451, 1.0), (677, 1.5), (902, 2.0))
SCREEN_PER_BRANCH = 8


def stable_score(sample_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{SEED}\0{purpose}\0{sample_id}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=root
        / "artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation",
    )
    return parser.parse_args()


def common_row(row: dict[str, str], specialist_root: Path) -> dict[str, object]:
    source_path = (specialist_root / row["source_local"]).resolve()
    target_path = (specialist_root / row["target_local"]).resolve()
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(
            f"missing evaluation media for {row['sample_id']}: "
            f"source={source_path.is_file()} target={target_path.is_file()}"
        )
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "branch": row["branch"],
        "instruction": row["instruction"],
        "source_path": str(source_path),
        "dataset_target_path": str(target_path),
        "generation_seed": SEED,
        "inference_steps": 30,
        "timestep_shift": 3.5,
        "cfg_text_scale": 4.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def round_robin(grouped: dict[str, list[dict[str, str]]]) -> Iterable[dict[str, str]]:
    longest = max(len(rows) for rows in grouped.values())
    for offset in range(longest):
        for branch in BRANCHES:
            rows = grouped[branch]
            if offset < len(rows):
                yield rows[offset]


def main() -> None:
    args = parse_args()
    with args.pool.open(encoding="utf-8-sig", newline="") as stream:
        pool = list(csv.DictReader(stream))
    if len(pool) != 4000 or len({row["sample_id"] for row in pool}) != 4000:
        raise ValueError("expected 4,000 unique frozen pool rows")

    validation = [row for row in pool if row["split"] == "validation"]
    test = [row for row in pool if row["split"] == "test"]
    if len(validation) != 195 or len(test) != 204:
        raise ValueError(f"frozen split counts changed: val={len(validation)} test={len(test)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation:
        grouped[row["branch"]].append(row)
    if set(grouped) != set(BRANCHES):
        raise ValueError(f"validation branches changed: {sorted(grouped)}")

    selected_by_branch: dict[str, list[dict[str, str]]] = {}
    for branch in BRANCHES:
        ranked = sorted(
            grouped[branch],
            key=lambda row: stable_score(row["sample_id"], "evaluation_funnel_v1"),
        )
        if len(ranked) < SCREEN_PER_BRANCH:
            raise ValueError(f"branch {branch} has fewer than {SCREEN_PER_BRANCH} rows")
        selected_by_branch[branch] = ranked[:SCREEN_PER_BRANCH]

    screen = list(round_robin(selected_by_branch))
    screen_ids = {row["sample_id"] for row in screen}
    remaining = sorted(
        (row for row in validation if row["sample_id"] not in screen_ids),
        key=lambda row: stable_score(row["sample_id"], "validation_remaining_v1"),
    )
    ordered_test = sorted(
        test, key=lambda row: stable_score(row["sample_id"], "internal_test_v1")
    )
    if len(screen) != 48 or len(remaining) != 147:
        raise RuntimeError(f"unexpected funnel sizes: {len(screen)} + {len(remaining)}")

    specialist_root = Path(__file__).resolve().parents[2] / "1_specialist_filtering"
    screen_samples = []
    for index, row in enumerate(screen, start=1):
        screen_samples.append({"eval_index": index, **common_row(row, specialist_root)})

    screen_tasks = []
    for checkpoint_step, checkpoint_epoch in CANDIDATES:
        for row in screen_samples:
            screen_tasks.append(
                {
                    "task_index": len(screen_tasks) + 1,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_epoch": checkpoint_epoch,
                    **row,
                }
            )

    remaining_rows = [
        {"remaining_index": index, **common_row(row, specialist_root)}
        for index, row in enumerate(remaining, start=1)
    ]
    test_rows = [
        {"test_index": index, **common_row(row, specialist_root)}
        for index, row in enumerate(ordered_test, start=1)
    ]

    manifests = {
        "screen_samples": args.output_dir / "validation_screen48.csv",
        "screen_generation_tasks": args.output_dir / "validation_screen48_tasks.csv",
        "validation_remaining": args.output_dir / "validation_remaining147.csv",
        "internal_test": args.output_dir / "internal_test204.csv",
    }
    write_csv(manifests["screen_samples"], screen_samples)
    write_csv(manifests["screen_generation_tasks"], screen_tasks)
    write_csv(manifests["validation_remaining"], remaining_rows)
    write_csv(manifests["internal_test"], test_rows)

    summary = {
        "schema_version": "lance_general_edit_adaptive_evaluation_funnel_v1",
        "selection_seed": SEED,
        "base_in_screen": False,
        "half_epoch_in_screen": False,
        "candidate_checkpoints": [
            {"step": step, "epoch": epoch} for step, epoch in CANDIDATES
        ],
        "screen_samples": len(screen_samples),
        "screen_branch_counts": dict(Counter(row["branch"] for row in screen_samples)),
        "screen_generation_tasks": len(screen_tasks),
        "validation_remaining_samples": len(remaining_rows),
        "internal_test_samples": len(test_rows),
        "clear_winner_rule": {
            "metric": "six_branch_macro_average",
            "paired_bootstrap_confidence": 0.95,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": SEED,
            "lower_bound_must_exceed": 0.0,
            "must_beat_every_other_candidate": True,
            "minimum_branch_mean_wins": 5,
        },
        "promotion": "always generate remaining147 for the numeric winner only",
        "final_comparison": {
            "internal_test204": ["winner"],
            "omni_edit_spatial240_generate_here": ["winner"],
            "base_omni_edit": "reuse external lance_base session",
        },
        "generation_counts_excluding_external_base_omni": {
            "selection": 291,
            "end_to_end": 735,
        },
        "manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in manifests.items()
        },
    }
    summary_path = args.output_dir / "funnel_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
