#!/usr/bin/env python3
"""Build a balanced 30-pair audit of Qwen calibration rejects.

The first five rows are low-confidence rejects nearest the model's decision
boundary.  The rest balance the six dataset/task branches as far as the
available 47 rejected calibration examples permit.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/vlm_run_v1"
VERSION = "general_edit_qwen_rejected_30_v1"
QUOTAS = {
    "egoedit_replace": 6,
    "goku_add": 4,
    "goku_remove": 6,
    "reco_add": 1,
    "reco_remove": 7,
    "reco_replace": 6,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=RUN / "calibration_pool_240.csv")
    parser.add_argument("--scores", type=Path, default=RUN / "calibration_qwen_scores.csv")
    parser.add_argument("--prior-review", type=Path, default=RUN / "calibration_review_100.csv")
    parser.add_argument("--output", type=Path, default=RUN / "qwen_rejected_review_30.csv")
    parser.add_argument("--summary", type=Path, default=RUN / "qwen_rejected_review_30_summary.json")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    pool = {row["sample_id"]: row for row in read_csv(args.pool)}
    scores = [
        row
        for row in read_csv(args.scores)
        if row.get("model_decision") == "rejected"
        and row.get("parse_status") == "pass"
        and row.get("model_backend") != "mock"
    ]
    if len(scores) != 47:
        raise ValueError(f"expected 47 real Qwen rejects, got {len(scores)}")
    prior_ids = (
        {row["sample_id"] for row in read_csv(args.prior_review)}
        if args.prior_review.is_file()
        else set()
    )
    randomizer = random.Random(args.seed)
    by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        by_branch[row["branch"]].append(row)
    if set(by_branch) != set(QUOTAS):
        raise ValueError(f"unexpected rejected branches: {sorted(by_branch)}")

    # The minimum rejected confidence is 0.5.  It occurs only in GOKU remove
    # and ReCo remove, so use two plus three to keep the boundary prefix varied.
    minimum_confidence = min(float(row["model_confidence"]) for row in scores)
    boundary_by_branch = {
        branch: [
            row
            for row in by_branch[branch]
            if float(row["model_confidence"]) == minimum_confidence
        ]
        for branch in ("goku_remove", "reco_remove")
    }
    for values in boundary_by_branch.values():
        randomizer.shuffle(values)
        values.sort(key=lambda row: row["sample_id"] in prior_ids)
    boundary = boundary_by_branch["goku_remove"][:2] + boundary_by_branch["reco_remove"][:3]
    if len(boundary) != 5:
        raise ValueError("could not select the five boundary rejects")

    selected = list(boundary)
    selected_ids = {row["sample_id"] for row in selected}
    for branch, quota in QUOTAS.items():
        need = quota - sum(row["branch"] == branch for row in selected)
        candidates = [row for row in by_branch[branch] if row["sample_id"] not in selected_ids]
        randomizer.shuffle(candidates)
        # Prefer examples absent from the earlier 100-pair package where the
        # branch quota allows it, without sacrificing requested branch balance.
        candidates.sort(key=lambda row: row["sample_id"] in prior_ids)
        if len(candidates) < need:
            raise ValueError(f"branch {branch} needs {need}, only {len(candidates)} remain")
        chosen = candidates[:need]
        selected.extend(chosen)
        selected_ids.update(row["sample_id"] for row in chosen)

    if len(selected) != 30 or len(selected_ids) != 30:
        raise ValueError("selection must contain 30 unique samples")

    output_rows: list[dict[str, str]] = []
    for rank, score in enumerate(selected, 1):
        source = pool[score["sample_id"]]
        output_rows.append(
            {
                "survey_version": VERSION,
                "sample_rank": str(rank),
                "review_stratum": "qwen_rejected_boundary" if rank <= 5 else "qwen_rejected_balanced",
                "qwen_confidence": score["model_confidence"],
                "qwen_evidence": score["short_evidence"],
                "was_in_calibration_review_100": "1" if score["sample_id"] in prior_ids else "0",
                **source,
            }
        )
    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.output)

    summary = {
        "survey_version": VERSION,
        "qwen_rejected_population": len(scores),
        "rows": len(output_rows),
        "boundary_prefix_rows": 5,
        "boundary_confidence": minimum_confidence,
        "branches": dict(Counter(row["branch"] for row in output_rows)),
        "operations": dict(Counter(row["operation"] for row in output_rows)),
        "datasets": dict(Counter(row["dataset"] for row in output_rows)),
        "overlap_with_calibration_review_100": sum(
            row["sample_id"] in prior_ids for row in output_rows
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
