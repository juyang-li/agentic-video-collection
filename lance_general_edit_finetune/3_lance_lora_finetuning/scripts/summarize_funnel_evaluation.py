#!/usr/bin/env python3
"""Offline, receipt-backed results; never generate videos, call APIs, or reselect a checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from judge_spatial import canonical_hash, sha256_file, write_json
from run_funnel_evaluation import BASE, COUNTS, EVALUATION, verify_contract

DIMENSIONS = ("accuracy", "preservation", "realism", "consistency")
METRICS = (*DIMENSIONS, "overall_score")


def identity(task: dict) -> tuple[str, str, str]:
    return str(task["stage"]), str(task["checkpoint_step"]), task["sample_id"]


def check_score(score: dict) -> None:
    for metric in DIMENSIONS:
        value = score[metric]
        if type(value) is not int or not 0 <= value <= 5:
            raise ValueError(f"invalid score dimension: {metric}")
    expected = round(0.5 * score["accuracy"] + 0.2 * score["accuracy"] * (
        0.2 * score["preservation"] + 0.15 * score["realism"] + 0.15 * score["consistency"]), 4)
    actual = float(score["overall_score"])
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-9):
        raise ValueError("overall score differs from the frozen official formula")


@lru_cache(maxsize=None)
def media_sha(path: str) -> str:
    return sha256_file(Path(path))


def read_results(root: Path, tasks: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    expected = {identity(task): task for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("duplicate task identity in frozen manifest")
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("judge protocol mismatch")
    protocol_sha = canonical_hash(protocol)
    rows, seen = [], set()
    for path in sorted((root / "records").glob("*/result.json")):
        record = json.loads(path.read_text())
        item = record["identity"]
        task = item["task"]
        key = identity(task)
        if (key in seen or task != expected.get(key) or record.get("status") != "success"
                or item["protocol_sha256"] != protocol_sha or not protocol_path.exists()):
            raise ValueError("unexpected, duplicate, or invalid success receipt")
        if (record.get("returned_provider") != protocol["expected_response_provider"]
                or record.get("returned_model") not in (
                    protocol["settings"]["model"], "google/gemini-3.8-flash-20260902")):
            raise ValueError("returned scoring model/provider mismatch")
        check_score(record["score"])
        if (media_sha(task["source_path"]) != item["source_sha256"]
                or media_sha(task["generated_video"]) != item["generated_sha256"]):
            raise ValueError("current media differs from the media that was scored")
        seen.add(key)
        rows.append({**task, **record["score"], "_source_sha256": item["source_sha256"]})
    costs, unknown = [], 0
    for path in (root / "records").glob("*/attempt_*.json"):
        attempt = json.loads(path.read_text())
        value = attempt.get("usage", {}).get("cost")
        if type(value) in (int, float) and math.isfinite(value):
            costs.append(value)
        else:
            unknown += 1
    return rows, {"known_cost_usd": sum(costs), "requests_without_reported_cost": unknown,
                  "scope": "all recorded attempts including retries; missing cost is not zero"}


def split_summary(tasks: list[dict], rows: list[dict], group_field: str = "branch") -> dict:
    expected = {task["sample_id"]: task for task in tasks}
    present = {row["sample_id"]: row for row in rows}
    if len(expected) != len(tasks) or len(present) != len(rows) or not present.keys() <= expected.keys():
        raise ValueError("split must contain unique, in-scope sample IDs")
    counts = Counter(task[group_field] for task in tasks)
    grouped = defaultdict(list)
    for row in rows:
        if row[group_field] != expected[row["sample_id"]][group_field]:
            raise ValueError("sample group mismatch")
        grouped[row[group_field]].append(row)
    def means(items: list[dict]) -> dict:
        return {metric: mean(item[metric] for item in items) for metric in METRICS} if items else {}
    groups = {group: {"expected": count, "scored": len(grouped[group]),
                      "mean_scores_successes_only": means(grouped[group])}
              for group, count in sorted(counts.items())}
    macro = {metric: mean(groups[group]["mean_scores_successes_only"][metric] for group in groups)
             for metric in METRICS} if groups and all(grouped[g] for g in groups) else {}
    return {"status": "complete" if len(rows) == len(tasks) else "incomplete",
            "expected": len(tasks), "scored": len(rows),
            "missing_sample_ids": sorted(expected.keys() - present.keys()),
            "micro_mean_scores_successes_only": means(rows),
            "group_macro_mean_scores_successes_only": macro, "groups": groups}


def paired_omni(base: list[dict], winner: list[dict], resamples: int = 10000) -> dict:
    before = {row["sample_id"]: row for row in base}
    after = {row["sample_id"]: row for row in winner}
    if len(base) != len(before) or len(winner) != len(after):
        raise ValueError("duplicate Omni scores")
    if len(before) != 240 or before.keys() != after.keys():
        raise ValueError("formal paired comparison requires all 240 identical sample IDs")
    grouped = defaultdict(list)
    all_differences = []
    for sid in sorted(before):
        left, right = before[sid], after[sid]
        if (left["category"] != right["branch"] or left["instruction"] != right["instruction"]
                or left["_source_sha256"] != right["_source_sha256"]
                or str(left["generation_seed"]) != "42" or str(right["generation_seed"]) != "42"
                or str(left["checkpoint_step"]) != "base" or str(right["checkpoint_step"]) != "677"):
            raise ValueError("Base/winner input, category, seed, or checkpoint mismatch")
        differences = [right[metric] - left[metric] for metric in METRICS]
        grouped[right["branch"]].append(differences)
        all_differences.append(differences)
    if len(grouped) != 24 or any(len(values) != 10 for values in grouped.values()):
        raise ValueError("Omni Spatial requires 24 categories of 10 pairs")
    rng = np.random.default_rng(42)
    bootstrap = np.zeros((resamples, len(METRICS)))
    categories = {}
    for category, values in sorted(grouped.items()):
        array = np.array(values, dtype=float)
        categories[category] = dict(zip(METRICS, array.mean(axis=0).tolist()))
        bootstrap += array[rng.integers(0, 10, size=(resamples, 10))].mean(axis=1) / 24
    differences = np.array(all_differences, dtype=float)
    ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {"paired_samples": 240, "difference_direction": "winner_677_minus_base",
            "mean_differences": dict(zip(METRICS, differences.mean(axis=0).tolist())),
            "paired_stratified_bootstrap_95_ci": {metric: ci[:, i].tolist() for i, metric in enumerate(METRICS)},
            "bootstrap_seed": 42, "bootstrap_resamples": resamples,
            "bootstrap_scope": "resample pairs within each category; conditional on one fixed judge response and generation seed, not judge or training-run variability",
            "category_mean_differences": categories,
            "overall_sample_wins_ties_losses": {
                "wins": int((differences[:, -1] > 1e-9).sum()),
                "ties": int((np.abs(differences[:, -1]) <= 1e-9).sum()),
                "losses": int((differences[:, -1] < -1e-9).sum())}}


def build_report(evaluation: Path) -> dict:
    contract = verify_contract(evaluation)
    protocol = json.loads((BASE / "gemini38_low_vertex_v2/protocol.json").read_text())
    decision = json.loads((evaluation / "promotion_decision.json").read_text())
    if decision["promotion_steps"] != [677]:
        raise ValueError("this report is restricted to the frozen winner 677")
    score_file = evaluation / "screen_scores_vertex_v2/scores.csv"
    if sha256_file(score_file) != decision["scores_sha256"]:
        raise ValueError("screen scores used for frozen selection have changed")
    tasks, rows, costs = {}, {}, {}
    for stage, count in COUNTS.items():
        tasks[stage] = json.loads((evaluation / f"{stage}_generation_tasks.json").read_text())
        if len(tasks[stage]) != count:
            raise ValueError("unexpected frozen stage size")
        allowed_steps = {451, 677, 902} if stage == "screen" else {677}
        if any(task["stage"] != stage or task["checkpoint_step"] not in allowed_steps
               or task["generation_seed"] != 42 for task in tasks[stage]):
            raise ValueError("stage, checkpoint, or inference seed changed")
        rows[stage], costs[stage] = read_results(evaluation / f"{stage}_scores_vertex_v2", tasks[stage], protocol)
    screen = {str(step): split_summary([t for t in tasks["screen"] if t["checkpoint_step"] == step],
                                      [r for r in rows["screen"] if r["checkpoint_step"] == step])
              for step in (451, 677, 902)}
    if any(item["expected"] != 48 or len(item["groups"]) != 6
           or any(group["expected"] != 8 for group in item["groups"].values()) for item in screen.values()):
        raise ValueError("screen must have three checkpoints, each with six branches of eight samples")
    val_tasks = [t for t in tasks["screen"] if t["checkpoint_step"] == 677] + tasks["remaining"]
    val_rows = [r for r in rows["screen"] if r["checkpoint_step"] == 677] + rows["remaining"]
    validation = split_summary(val_tasks, val_rows)
    test = split_summary(tasks["test"], rows["test"])
    omni = split_summary(tasks["omni"], rows["omni"])
    base_tasks = json.loads((BASE / "evaluator_workspace/judge_tasks.json").read_text())
    base_rows, base_cost = read_results(BASE / "gemini38_low_vertex_v2", base_tasks, protocol)
    base = split_summary(base_tasks, base_rows, "category")
    comparison = paired_omni(base_rows, rows["omni"]) if len(base_rows) == len(rows["omni"]) == 240 else None
    complete = all(item["status"] == "complete" for item in [*screen.values(), validation, test, omni, base])
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "incomplete",
            "winner_checkpoint_step": 677, "selection_changed": False,
            "selection_uncertainty": "screen numeric winner only; original bootstrap intervals crossed zero",
            "protocol_sha256": contract["canonical_protocol_sha256"],
            "evidence_scope": "frozen task manifests, successful score receipts, current source/output SHA256; full-decode and generation-provenance audits remain separate",
            "screen": screen, "winner_validation195": validation, "winner_test204": test,
            "winner_omni240": omni, "base_omni240": base, "omni_paired_comparison": comparison,
            "internal_test_comparison_scope": "absolute scores only; no Base internal-test comparison was run",
            "finetune_evaluation_costs_by_stage": costs,
            "finetune_evaluation_known_cost_usd": sum(item["known_cost_usd"] for item in costs.values()),
            "finetune_requests_without_reported_cost": sum(item["requests_without_reported_cost"] for item in costs.values()),
            "base_vertex_v2_cost_separate": base_cost}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.evaluation_root)
    if args.output:
        write_json(args.output, report)
        print(json.dumps({"report": str(args.output), "status": report["status"],
                          "validation_scored": report["winner_validation195"]["scored"],
                          "test_scored": report["winner_test204"]["scored"],
                          "omni_scored": report["winner_omni240"]["scored"],
                          "known_cost_usd": report["finetune_evaluation_known_cost_usd"]}))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
