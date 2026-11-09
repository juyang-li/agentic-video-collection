#!/usr/bin/env python3
"""Freeze a new, deterministic 40-pair human holdout for v12 review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import compose_best_routed_filter_v12 as routed


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE1 = PROJECT / "1_specialist_filtering"
DEFAULT_INPUT = STAGE1 / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_RUN = ROOT / "artifacts/v12_holdout40_v1"
DEFAULT_OUTPUT = DEFAULT_RUN / "v12_holdout40.csv"
DEFAULT_SUMMARY = DEFAULT_RUN / "v12_holdout40_sampling_summary.json"
BRANCHES = (
    "reco_add",
    "reco_remove",
    "reco_replace",
    "goku_add",
    "goku_remove",
    "egoedit_replace",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def generated_role(row: dict[str, str]) -> str:
    roles = (row.get("source_provenance"), row.get("target_provenance"))
    if roles == ("edited", "original"):
        return "source"
    if roles == ("original", "edited"):
        return "target"
    if roles == ("edited", "edited"):
        return "both"
    return "unknown"


def audit_operation(row: dict[str, str]) -> str:
    operation = row.get("operation", "").lower()
    if generated_role(row) == "source":
        return {"add": "remove", "remove": "add", "replace": "replace"}.get(
            operation, operation
        )
    return operation


def prior_review_ids() -> tuple[set[str], list[str]]:
    paths: set[Path] = set(ROOT.glob("artifacts/**/*.csv"))
    stage1_artifacts = STAGE1 / "artifacts"
    patterns = (
        "review_results/*.csv",
        "*review*.csv",
        "*audit*.csv",
        "survey_manifest*.csv",
        "**/previous_results/*.csv",
    )
    for pattern in patterns:
        paths.update(stage1_artifacts.glob(pattern))
    sample_ids: set[str] = set()
    used_paths: list[str] = []
    for path in sorted(paths):
        try:
            rows, fields = read_csv(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        if "sample_id" not in fields:
            continue
        values = {row.get("sample_id", "") for row in rows if row.get("sample_id")}
        if values:
            sample_ids.update(values)
            used_paths.append(str(path.relative_to(PROJECT)))
    return sample_ids, used_paths


def media_ready(row: dict[str, str]) -> bool:
    return all((STAGE1 / row[f"{role}_local"]).is_file() for role in ("source", "target"))


def is_eyewear_remove(row: dict[str, str]) -> bool:
    return routed.is_eyewear_remove_target(
        row, {"audit_operation": audit_operation(row)}
    )


def diverse_eyewear_sample(
    rows: list[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", ""), row.get("branch", ""))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    output: list[dict[str, str]] = []
    while len(output) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(output) < count:
                output.append(groups[key].pop())
    if len(output) != count:
        raise ValueError(f"only {len(output)} unseen eyewear-remove rows are available")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--general-per-branch", type=int, default=6)
    parser.add_argument("--eyewear-count", type=int, default=4)
    args = parser.parse_args()

    rows, fields = read_csv(args.input)
    reviewed, review_sources = prior_review_ids()
    candidates = [
        row for row in rows if row["sample_id"] not in reviewed and media_ready(row)
    ]
    rng = random.Random(args.seed)

    eyewear_pool = [row for row in candidates if is_eyewear_remove(row)]
    eyewear = diverse_eyewear_sample(eyewear_pool, args.eyewear_count, rng)
    eyewear_ids = {row["sample_id"] for row in eyewear}

    general: list[dict[str, str]] = []
    general_available: dict[str, int] = {}
    for branch in BRANCHES:
        pool = [
            row
            for row in candidates
            if row.get("branch") == branch
            and row["sample_id"] not in eyewear_ids
            and not is_eyewear_remove(row)
        ]
        general_available[branch] = len(pool)
        if len(pool) < args.general_per_branch:
            raise ValueError(f"not enough unseen rows for {branch}: {len(pool)}")
        general.extend(rng.sample(pool, args.general_per_branch))

    selected = [*general, *eyewear]
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("holdout selection contains duplicate sample IDs")
    rng.shuffle(selected)
    output: list[dict[str, str]] = []
    for index, source in enumerate(selected, 1):
        row = dict(source)
        row.update(
            {
                "survey_version": "qwen72b_v12_unseen_holdout40_v1",
                "sample_rank": str(index),
                "generated_role": generated_role(row),
                "audit_operation_expected": audit_operation(row),
                "holdout_stratum": (
                    "eyewear_remove" if row["sample_id"] in eyewear_ids else "general"
                ),
                "risk_group": (
                    "unseen_eyewear_remove"
                    if row["sample_id"] in eyewear_ids
                    else f"unseen_{row.get('branch', 'general')}"
                ),
                "artifact_risk_score": "",
                "artifact_risk_reasons": "",
            }
        )
        output.append(row)

    extra = [
        "survey_version",
        "sample_rank",
        "generated_role",
        "audit_operation_expected",
        "holdout_stratum",
        "risk_group",
        "artifact_risk_score",
        "artifact_risk_reasons",
    ]
    output_fields = list(dict.fromkeys([*extra, *fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    summary = {
        "survey_version": "qwen72b_v12_unseen_holdout40_v1",
        "seed": args.seed,
        "input_rows": len(rows),
        "prior_review_ids": len(reviewed),
        "review_source_files": len(review_sources),
        "unseen_media_ready_candidates": len(candidates),
        "general_per_branch": args.general_per_branch,
        "general_available": general_available,
        "eyewear_available": len(eyewear_pool),
        "eyewear_count": len(eyewear),
        "selected_rows": len(output),
        "selected_branches": dict(Counter(row["branch"] for row in output)),
        "selected_datasets": dict(Counter(row["dataset"] for row in output)),
        "selected_operations": dict(Counter(row["operation"] for row in output)),
        "selected_stage1_1": dict(
            Counter(row["stage1_1_decision"] for row in output)
        ),
        "eyewear_ids": [row["sample_id"] for row in eyewear],
        "sample_ids": [row["sample_id"] for row in output],
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
