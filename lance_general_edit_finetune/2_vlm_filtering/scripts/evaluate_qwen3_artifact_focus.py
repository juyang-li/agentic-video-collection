#!/usr/bin/env python3
"""Compare focused Qwen3 artifact decisions with frozen human annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen3_openrouter_v1"
DEFAULT_LABELS = RUN / "artifact_focus_eval_38.csv"
DEFAULT_SCORES = RUN / "artifact_focus_eval_v2_5_scores.csv"
DEFAULT_OUTPUT = RUN / "artifact_focus_eval_v2_5_evaluation.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric(rows: list[dict[str, str]], predicate: Any) -> dict[str, Any]:
    numerator = sum(bool(predicate(row)) for row in rows)
    return {
        "numerator": numerator,
        "denominator": len(rows),
        "rate": ratio(numerator, len(rows)),
    }


def compact(row: dict[str, str], label: dict[str, str]) -> dict[str, Any]:
    try:
        reasons = json.loads(row.get("reason_codes_json") or "[]")
    except json.JSONDecodeError:
        reasons = []
    try:
        evidence = json.loads(row.get("evidence_json") or "{}")
    except json.JSONDecodeError:
        evidence = {}
    return {
        "sample_id": row.get("sample_id", ""),
        "dataset": row.get("dataset", ""),
        "operation": row.get("operation", ""),
        "human_label": label.get("human_label", ""),
        "focus_group": label.get("focus_group", ""),
        "focus_defect_tags": label.get("focus_defect_tags", ""),
        "human_evidence": label.get("focus_human_evidence", ""),
        "model_decision": row.get("model_decision", ""),
        "visible_artifact": row.get("visible_artifact", ""),
        "instruction_following": row.get("instruction_following", ""),
        "content_preservation": row.get("content_preservation", ""),
        "reason_codes": reasons,
        "model_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focus-recall-min", type=float, default=0.75)
    parser.add_argument("--clean-artifact-fpr-max", type=float, default=0.20)
    parser.add_argument("--clean-reject-fpr-max", type=float, default=0.25)
    parser.add_argument("--non-ego-hard-recall-min", type=float, default=0.70)
    args = parser.parse_args()

    scores = read_csv(args.scores)
    labels = read_csv(args.labels)
    labels_by_id = {row["sample_id"]: row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("focused labels contain duplicate sample_id")
    score_counts = Counter(row.get("sample_id", "") for row in scores)
    duplicate_ids = sorted(key for key, count in score_counts.items() if key and count > 1)
    missing_ids = sorted(set(labels_by_id) - set(score_counts))
    unexpected_ids = sorted(set(score_counts) - set(labels_by_id))
    matched = [row for row in scores if row.get("sample_id") in labels_by_id]
    parse_errors = [row for row in matched if row.get("parse_status") != "pass"]
    completed = [row for row in matched if row.get("parse_status") == "pass"]

    focus = [
        row
        for row in completed
        if labels_by_id[row["sample_id"]].get("focus_defect_ground_truth") == "1"
    ]
    clean = [
        row
        for row in completed
        if labels_by_id[row["sample_id"]].get("focus_group") == "clean_control"
    ]
    tolerated = [
        row
        for row in completed
        if labels_by_id[row["sample_id"]].get("focus_group") == "tolerated_artifact"
    ]
    hard = [
        row
        for row in completed
        if labels_by_id[row["sample_id"]].get("human_label") == "unusable"
    ]
    non_ego_hard = [row for row in hard if row.get("dataset") != "egoedit"]

    focus_recall = metric(focus, lambda row: row.get("visible_artifact") == "fail")
    focus_rejection_recall = metric(
        focus, lambda row: row.get("model_decision") == "rejected"
    )
    clean_artifact_fpr = metric(
        clean, lambda row: row.get("visible_artifact") == "fail"
    )
    clean_reject_fpr = metric(clean, lambda row: row.get("model_decision") == "rejected")
    hard_recall = metric(hard, lambda row: row.get("model_decision") == "rejected")
    non_ego_hard_recall = metric(
        non_ego_hard, lambda row: row.get("model_decision") == "rejected"
    )
    tolerated_reject_rate = metric(
        tolerated, lambda row: row.get("model_decision") == "rejected"
    )

    per_tag: dict[str, dict[str, Any]] = {}
    tags_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in focus:
        for tag in labels_by_id[row["sample_id"]].get("focus_defect_tags", "").split("|"):
            if tag:
                tags_to_rows[tag].append(row)
    for tag, rows in sorted(tags_to_rows.items()):
        per_tag[tag] = metric(rows, lambda row: row.get("visible_artifact") == "fail")

    identity = {
        "judge_versions": sorted({row.get("judge_version", "") for row in completed}),
        "prompt_hashes": sorted({row.get("prompt_sha256", "") for row in completed}),
        "model_ids": sorted({row.get("model_id", "") for row in completed}),
        "model_backends": sorted({row.get("model_backend", "") for row in completed}),
    }
    complete = (
        not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and not parse_errors
        and len(identity["judge_versions"]) == 1
        and len(identity["prompt_hashes"]) == 1
        and len(identity["model_ids"]) == 1
        and identity["model_backends"] == ["openrouter"]
        and len(focus) >= 7
        and len(clean) >= 20
    )
    gate_checks = {
        "complete": complete,
        "focus_visible_artifact_recall": (
            focus_recall["rate"] is not None
            and focus_recall["rate"] >= args.focus_recall_min
        ),
        "clean_visible_artifact_false_positive_rate": (
            clean_artifact_fpr["rate"] is not None
            and clean_artifact_fpr["rate"] <= args.clean_artifact_fpr_max
        ),
        "clean_any_rejection_rate": (
            clean_reject_fpr["rate"] is not None
            and clean_reject_fpr["rate"] <= args.clean_reject_fpr_max
        ),
        "non_ego_hard_failure_recall": (
            non_ego_hard_recall["rate"] is not None
            and non_ego_hard_recall["rate"] >= args.non_ego_hard_recall_min
        ),
    }
    expansion_status = "enabled" if all(gate_checks.values()) else "not_enabled"

    report = {
        "experiment_version": "qwen3_artifact_focus_holdout_v1",
        "expansion_status": expansion_status,
        "gate_checks": gate_checks,
        "thresholds": {
            "focus_visible_artifact_recall_min": args.focus_recall_min,
            "clean_visible_artifact_false_positive_rate_max": args.clean_artifact_fpr_max,
            "clean_any_rejection_rate_max": args.clean_reject_fpr_max,
            "non_ego_hard_failure_recall_min": args.non_ego_hard_recall_min,
        },
        "rows": {
            "labels": len(labels),
            "scores": len(scores),
            "matched": len(matched),
            "parse_errors": len(parse_errors),
            "missing_sample_ids": missing_ids,
            "unexpected_sample_ids": unexpected_ids,
            "duplicate_sample_ids": duplicate_ids,
        },
        "model_decisions": dict(Counter(row.get("model_decision", "") for row in completed)),
        "visible_artifact_decisions": dict(
            Counter(row.get("visible_artifact", "") for row in completed)
        ),
        "metrics": {
            "focus_visible_artifact_recall": focus_recall,
            "focus_visible_artifact_miss_rate": {
                "numerator": focus_recall["denominator"] - focus_recall["numerator"],
                "denominator": focus_recall["denominator"],
                "rate": (
                    None
                    if focus_recall["rate"] is None
                    else 1.0 - focus_recall["rate"]
                ),
            },
            "focus_final_rejection_recall": focus_rejection_recall,
            "focus_final_rejection_miss_rate": {
                "numerator": (
                    focus_rejection_recall["denominator"]
                    - focus_rejection_recall["numerator"]
                ),
                "denominator": focus_rejection_recall["denominator"],
                "rate": (
                    None
                    if focus_rejection_recall["rate"] is None
                    else 1.0 - focus_rejection_recall["rate"]
                ),
            },
            "clean_visible_artifact_false_positive_rate": clean_artifact_fpr,
            "clean_any_rejection_rate": clean_reject_fpr,
            "hard_failure_rejection_recall": hard_recall,
            "non_ego_hard_failure_rejection_recall": non_ego_hard_recall,
            "tolerated_artifact_rejection_rate": tolerated_reject_rate,
            "focus_visible_artifact_recall_by_tag": per_tag,
        },
        "error_cases": {
            "focus_misses": [
                compact(row, labels_by_id[row["sample_id"]])
                for row in focus
                if row.get("visible_artifact") != "fail"
            ],
            "clean_artifact_false_positives": [
                compact(row, labels_by_id[row["sample_id"]])
                for row in clean
                if row.get("visible_artifact") == "fail"
            ],
            "clean_any_rejections": [
                compact(row, labels_by_id[row["sample_id"]])
                for row in clean
                if row.get("model_decision") == "rejected"
            ],
            "non_ego_hard_misses": [
                compact(row, labels_by_id[row["sample_id"]])
                for row in non_ego_hard
                if row.get("model_decision") != "rejected"
            ],
            "tolerated_artifact_cases": [
                compact(row, labels_by_id[row["sample_id"]]) for row in tolerated
            ],
        },
        "identity": identity,
        "interpretation_limits": [
            "The untouched current-scope holdout has seven explicit focus-defect examples.",
            "It contains no explicit texture_break tag; local_blur, smear_ghost, repair_hole, mask-like repair failure, and local_flicker are the available proxies.",
            "Tolerated artifact examples are not counted as clean false positives.",
            "Passing this focused gate supports expanding the artifact screen; it does not replace the broader representative holdout gate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
