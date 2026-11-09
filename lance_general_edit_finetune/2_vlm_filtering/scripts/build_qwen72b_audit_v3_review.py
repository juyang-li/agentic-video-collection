#!/usr/bin/env python3
"""Build the old/new Qwen72B comparison manifest with prior human labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_OLD = RUN / "artifact_risk_30_scores.csv"
DEFAULT_NEW = RUN / "artifact_risk_30_qwen72b_audit_v3_scores.csv"
DEFAULT_HUMAN = RUN / "qwen32b_vs_72b_artifact_risk_30_v1_review.csv"
DEFAULT_OUTPUT = RUN / "artifact_risk_30_qwen72b_audit_v3_review.csv"
DEFAULT_SUMMARY = RUN / "artifact_risk_30_qwen72b_audit_v3_evaluation.json"
VERSION = "qwen72b_audit_normalized_v3_review_v1"
MODEL = "qwen/qwen2.5-vl-72b-instruct"
OLD_JUDGE = "qwen2_5_vl_72b_openrouter_artifact_pair_judge_v1"
NEW_JUDGE = "qwen2_5_vl_72b_openrouter_audit_normalized_pair_judge_v3"
CHECKS = ("instruction_following", "visible_artifact", "content_preservation")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def by_id(rows: list[dict[str, str]], name: str, expected: set[str]) -> dict[str, dict[str, str]]:
    mapped = {row["sample_id"]: row for row in rows}
    if len(mapped) != len(rows) or set(mapped) != expected:
        raise ValueError(f"{name} IDs do not exactly match the frozen manifest")
    return mapped


def parse_json(value: str, fallback: object) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def validate_scores(rows: list[dict[str, str]], judge: str, name: str) -> None:
    for row in rows:
        if row.get("model_id") != MODEL or row.get("judge_version") != judge:
            raise ValueError(f"unexpected {name} identity for {row.get('sample_id')}")
        if row.get("parse_status") != "pass":
            raise ValueError(f"unparseable {name} score for {row.get('sample_id')}")


def add_score(output: dict[str, str], score: dict[str, str], prefix: str) -> None:
    reasons = parse_json(score.get("reason_codes_json", ""), [])
    evidence = parse_json(score.get("evidence_json", ""), {})
    if not isinstance(reasons, list):
        reasons = []
    if not isinstance(evidence, dict):
        evidence = {}
    output[f"{prefix}_decision"] = score.get("model_decision", "")
    for check in CHECKS:
        output[f"{prefix}_{check}"] = score.get(check, "")
        output[f"{prefix}_{check}_evidence"] = str(evidence.get(check, ""))
    output[f"{prefix}_reason_codes"] = " | ".join(str(value) for value in reasons)


def generated_role(row: dict[str, str]) -> str:
    source = row.get("source_provenance")
    target = row.get("target_provenance")
    if (source, target) == ("edited", "original"):
        return "source"
    if (source, target) == ("original", "edited"):
        return "target"
    if (source, target) == ("edited", "edited"):
        return "both"
    return "unknown"


def capture_status(human: str, model: str) -> str:
    if human == "unusable":
        return "caught_unusable" if model == "rejected" else "missed_unusable"
    if human == "usable":
        return "false_reject" if model == "rejected" else "correct_accept"
    return "human_uncertain"


def metrics(rows: list[dict[str, str]], prefix: str) -> dict[str, object]:
    decided = [row for row in rows if row["human_status"] in {"usable", "unusable"}]
    matrix = Counter(
        f"human={row['human_status']}|model={row[f'{prefix}_decision']}" for row in decided
    )
    bad = sum(row["human_status"] == "unusable" for row in decided)
    good = sum(row["human_status"] == "usable" for row in decided)
    caught = matrix["human=unusable|model=rejected"]
    false_reject = matrix["human=usable|model=rejected"]
    correct = caught + matrix["human=usable|model=accepted"]
    return {
        "decided_rows": len(decided),
        "confusion": dict(matrix),
        "accuracy": correct / len(decided) if decided else None,
        "unusable_recall": caught / bad if bad else None,
        "usable_false_reject_rate": false_reject / good if good else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--old-scores", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new-scores", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--old-judge", default=OLD_JUDGE)
    parser.add_argument("--new-judge", default=NEW_JUDGE)
    parser.add_argument("--survey-version", default=VERSION)
    parser.add_argument("--old-label", default="旧 72B")
    parser.add_argument("--new-label", default="新路由 72B")
    parser.add_argument("--expected-size", type=int, default=30)
    args = parser.parse_args()

    manifest, manifest_fields = read_csv(args.manifest)
    old_rows, _ = read_csv(args.old_scores)
    new_rows, _ = read_csv(args.new_scores)
    human_rows, _ = read_csv(args.human)
    expected = {row["sample_id"] for row in manifest}
    if len(manifest) != args.expected_size or len(expected) != args.expected_size:
        raise ValueError(
            f"expected a frozen manifest of {args.expected_size} unique rows"
        )
    validate_scores(old_rows, args.old_judge, "old")
    validate_scores(new_rows, args.new_judge, "new")
    old = by_id(old_rows, "old scores", expected)
    new = by_id(new_rows, "new scores", expected)
    human = by_id(human_rows, "human review", expected)

    output: list[dict[str, str]] = []
    for source in manifest:
        sample_id = source["sample_id"]
        row = dict(source)
        row["survey_version"] = args.survey_version
        row["old_model_label"] = args.old_label
        row["new_model_label"] = args.new_label
        row["generated_role"] = generated_role(source)
        row["human_status"] = human[sample_id].get("overall_status", "")
        row["human_notes"] = human[sample_id].get("notes", "")
        add_score(row, old[sample_id], "old72b")
        add_score(row, new[sample_id], "new72b")
        row["audit_operation"] = new[sample_id].get("audit_operation", "")
        row["audit_relation"] = new[sample_id].get("audit_relation", "")
        row["audit_left_training_role"] = new[sample_id].get(
            "audit_left_training_role", ""
        )
        row["audit_right_training_role"] = new[sample_id].get(
            "audit_right_training_role", ""
        )
        row["capture_status"] = capture_status(
            row["human_status"], row["new72b_decision"]
        )
        row["decision_changed"] = (
            "1" if row["old72b_decision"] != row["new72b_decision"] else "0"
        )
        output.append(row)

    order = {
        "missed_unusable": 0,
        "false_reject": 1,
        "caught_unusable": 2,
        "correct_accept": 3,
        "human_uncertain": 4,
    }
    output.sort(
        key=lambda row: (
            order[row["capture_status"]],
            0 if row["decision_changed"] == "1" else 1,
            0 if row["generated_role"] == "source" else 1,
            int(row.get("sample_rank") or 0),
        )
    )
    for rank, row in enumerate(output, 1):
        row["sample_rank"] = str(rank)

    score_fields: list[str] = []
    for prefix in ("old72b", "new72b"):
        score_fields.extend(
            [
                f"{prefix}_decision",
                *[f"{prefix}_{check}" for check in CHECKS],
                f"{prefix}_reason_codes",
                *[f"{prefix}_{check}_evidence" for check in CHECKS],
            ]
        )
    extra = [
        "generated_role",
        "old_model_label",
        "new_model_label",
        "human_status",
        "human_notes",
        "audit_operation",
        "audit_relation",
        "audit_left_training_role",
        "audit_right_training_role",
        "capture_status",
        "decision_changed",
        *score_fields,
    ]
    fields = list(dict.fromkeys([*manifest_fields, *extra]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    summary = {
        "survey_version": args.survey_version,
        "old_model_label": args.old_label,
        "new_model_label": args.new_label,
        "rows": len(output),
        "human_counts": dict(Counter(row["human_status"] for row in output)),
        "capture_counts": dict(Counter(row["capture_status"] for row in output)),
        "old72b": metrics(output, "old72b"),
        "new72b": metrics(output, "new72b"),
        "new_decisions": dict(Counter(row["new72b_decision"] for row in output)),
        "new_visible_artifact": dict(
            Counter(row["new72b_visible_artifact"] for row in output)
        ),
        "decision_changed": sum(row["decision_changed"] == "1" for row in output),
        "new_total_cost_usd": sum(float(row.get("cost_usd") or 0) for row in new_rows),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
