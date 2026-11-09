#!/usr/bin/env python3
"""Compose the Stage-2 Qwen/Gemini cascade and benchmark it on human labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROBOT_RE = re.compile(
    r"\b(robot|robotic|android|cyborg|mecha|droid|humanoid)\b|机器人|机械人|机甲|仿生人",
    re.IGNORECASE,
)
BINARY = {"accepted", "rejected"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def unique_map(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in mapped:
            raise ValueError(f"{name} has a missing or duplicate sample_id: {sample_id!r}")
        mapped[sample_id] = row
    return mapped


def normalized_decision(row: dict[str, str]) -> str:
    if row.get("parse_status") != "pass":
        return "error"
    decision = row.get("model_decision", "").strip().lower()
    return decision if decision in {*BINARY, "uncertain"} else "error"


def risk_reasons(source: dict[str, str], qwen: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    audit_operation = (
        qwen.get("audit_operation")
        or source.get("audit_operation_expected")
        or source.get("audit_operation")
        or source.get("operation")
    ).strip().lower()
    if audit_operation == "remove":
        reasons.append("high_risk_remove")
    if source.get("stage1_1_decision", "").strip().lower() == "uncertain":
        reasons.append("stage1_boundary")
    instruction = source.get("instruction") or source.get("training_instruction", "")
    if ROBOT_RE.search(instruction):
        reasons.append("robot_or_synthetic_subject")
    return reasons


def metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = Counter()
    for row in rows:
        truth = row["human_decision"]
        predicted = row["benchmark_binary_decision"]
        if truth == "rejected" and predicted == "rejected":
            counts["true_reject"] += 1
        elif truth == "rejected" and predicted == "accepted":
            counts["missed_bad"] += 1
        elif truth == "accepted" and predicted == "rejected":
            counts["false_reject"] += 1
        elif truth == "accepted" and predicted == "accepted":
            counts["true_accept"] += 1
        else:
            raise ValueError(f"non-binary benchmark row: {row.get('sample_id')}")
    bad = counts["true_reject"] + counts["missed_bad"]
    good = counts["true_accept"] + counts["false_reject"]
    rejected_predictions = counts["true_reject"] + counts["false_reject"]
    total = bad + good
    recall = counts["true_reject"] / bad if bad else None
    good_recall = counts["true_accept"] / good if good else None
    return {
        "rows": total,
        "human_accepted": good,
        "human_rejected": bad,
        "true_accept": counts["true_accept"],
        "true_reject": counts["true_reject"],
        "false_reject": counts["false_reject"],
        "missed_bad": counts["missed_bad"],
        "rejected_recall": recall,
        "accuracy": (counts["true_accept"] + counts["true_reject"]) / total if total else None,
        "accepted_false_reject_rate": counts["false_reject"] / good if good else None,
        "reject_precision": counts["true_reject"] / rejected_predictions if rejected_predictions else None,
        "balanced_accuracy": (recall + good_recall) / 2 if recall is not None and good_recall is not None else None,
    }


def grouped_metrics(rows: list[dict[str, str]], field: str) -> dict[str, object]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(field, "") or "<missing>"].append(row)
    return {name: metrics(group) for name, group in sorted(groups.items())}


def metrics_for_decision_field(
    rows: list[dict[str, str]], field: str
) -> dict[str, object]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        # A non-binary or failed answer is withheld from the training pool, so
        # count it as rejected in end-to-end selection metrics.
        copied["benchmark_binary_decision"] = (
            "accepted" if row.get(field) == "accepted" else "rejected"
        )
        normalized.append(copied)
    return metrics(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--qwen-scores", type=Path, required=True)
    parser.add_argument("--gemini-scores", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows, manifest_fields = read_csv(args.manifest)
    label_rows, _ = read_csv(args.labels)
    qwen_rows, _ = read_csv(args.qwen_scores)
    gemini_rows: list[dict[str, str]] = []
    evaluation_split_by_id: dict[str, str] = {}
    for path in args.gemini_scores:
        part, _ = read_csv(path)
        filename = path.name.lower()
        if "validation" in filename:
            evaluation_split = "validation"
        elif "test" in filename:
            evaluation_split = "test"
        elif "train" in filename:
            evaluation_split = "train"
        else:
            raise ValueError(f"cannot infer evaluation split from Gemini score file: {path}")
        for row in part:
            sample_id = row.get("sample_id", "")
            if sample_id in evaluation_split_by_id:
                raise ValueError(f"sample appears in multiple evaluation splits: {sample_id}")
            evaluation_split_by_id[sample_id] = evaluation_split
        gemini_rows.extend(part)
    manifest = unique_map(manifest_rows, "manifest")
    labels = unique_map(label_rows, "labels")
    qwen = unique_map(qwen_rows, "Qwen scores")
    gemini = unique_map(gemini_rows, "Gemini scores")
    expected = set(labels)
    for name, mapped in (("manifest", manifest), ("Qwen", qwen), ("Gemini", gemini)):
        if set(mapped) != expected:
            missing = sorted(expected - set(mapped))[:5]
            extra = sorted(set(mapped) - expected)[:5]
            raise ValueError(f"{name} IDs differ from labels; missing={missing}, extra={extra}")

    output: list[dict[str, str]] = []
    qwen_cost = 0.0
    routed_gemini_cost = 0.0
    counterfactual_rescues = 0
    counterfactual_conflicts = 0
    for sample_id in labels:
        source = manifest[sample_id]
        label = labels[sample_id]
        qrow = qwen[sample_id]
        grow = gemini[sample_id]
        qdecision = normalized_decision(qrow)
        gdecision = normalized_decision(grow)
        risks = risk_reasons(source, qrow)
        route = ""
        final_status = "quarantine"
        if qdecision == "rejected":
            route = "qwen_hard_reject"
            final_status = "rejected"
        elif qdecision in {"uncertain", "error"}:
            route = "gemini_resolve_qwen_uncertain_or_error"
            final_status = gdecision if gdecision in BINARY else "quarantine"
        elif risks:
            route = "gemini_confirm_qwen_accept_high_risk"
            final_status = gdecision if gdecision in BINARY else "quarantine"
        else:
            route = "qwen_accept_low_risk"
            final_status = "accepted"

        qwen_cost += float(qrow.get("cost_usd") or 0.0)
        if route.startswith("gemini_"):
            routed_gemini_cost += float(grow.get("cost_usd") or 0.0)
        if qdecision == "rejected" and gdecision == "accepted":
            if label["human_decision"] == "accepted":
                counterfactual_rescues += 1
            else:
                counterfactual_conflicts += 1

        # Alternative safety valve: Qwen can fast-accept ordinary low-risk rows,
        # but every Qwen reject/uncertain and every high-risk accept is decided by
        # Gemini. This tells us whether Qwen hard rejects are actually beneficial.
        if qdecision == "accepted" and not risks:
            rescue_status = "accepted"
            rescue_route = "qwen_accept_low_risk"
        else:
            rescue_status = gdecision if gdecision in BINARY else "quarantine"
            rescue_route = "gemini_decides_all_non_low_risk"

        combined = dict(source)
        combined.update(
            {
                "human_decision": label["human_decision"],
                "human_notes": label.get("human_notes", ""),
                "annotation_partition": label.get("annotation_partition", ""),
                "evaluation_split": evaluation_split_by_id[sample_id],
                "qwen_decision": qdecision,
                "qwen_instruction_following": qrow.get("instruction_following", ""),
                "qwen_visible_artifact": qrow.get("visible_artifact", ""),
                "qwen_content_preservation": qrow.get("content_preservation", ""),
                "qwen_reason_codes_json": qrow.get("reason_codes_json", ""),
                "qwen_evidence_json": qrow.get("evidence_json", ""),
                "qwen_cost_usd": qrow.get("cost_usd", ""),
                "gemini_decision": gdecision,
                "gemini_instruction_following": grow.get("instruction_following", ""),
                "gemini_visible_artifact": grow.get("visible_artifact", ""),
                "gemini_content_preservation": grow.get("content_preservation", ""),
                "gemini_reason_codes_json": grow.get("reason_codes_json", ""),
                "gemini_evidence_json": grow.get("evidence_json", ""),
                "gemini_cost_usd": grow.get("cost_usd", ""),
                "cascade_route": route,
                "cascade_risk_reasons": "|".join(risks),
                "cascade_status": final_status,
                # Quarantine is not admitted to training and is therefore a rejection
                # for end-to-end selection metrics.
                "benchmark_binary_decision": (
                    "accepted" if final_status == "accepted" else "rejected"
                ),
                "rescue_variant_route": rescue_route,
                "rescue_variant_status": rescue_status,
                "rescue_variant_binary_decision": (
                    "accepted" if rescue_status == "accepted" else "rejected"
                ),
            }
        )
        output.append(combined)

    extra_fields = [
        "human_decision", "human_notes", "annotation_partition", "evaluation_split",
        "qwen_decision", "qwen_instruction_following", "qwen_visible_artifact",
        "qwen_content_preservation", "qwen_reason_codes_json", "qwen_evidence_json",
        "qwen_cost_usd", "gemini_decision", "gemini_instruction_following",
        "gemini_visible_artifact", "gemini_content_preservation",
        "gemini_reason_codes_json", "gemini_evidence_json", "gemini_cost_usd",
        "cascade_route", "cascade_risk_reasons", "cascade_status",
        "benchmark_binary_decision",
        "rescue_variant_route", "rescue_variant_status",
        "rescue_variant_binary_decision",
    ]
    fields = list(dict.fromkeys([*manifest_fields, *extra_fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    split_rows = {
        split: [row for row in output if row.get("evaluation_split") == split]
        for split in ("train", "validation", "test")
    }
    test_rows = split_rows["test"]
    route_fields = {
        "qwen_only": "qwen_decision",
        "gemini_only": "gemini_decision",
        "qwen_hard_reject_cascade": "benchmark_binary_decision",
        "gemini_rescues_qwen_rejects": "rescue_variant_binary_decision",
    }
    comparison_by_split = {
        split: {
            route_name: metrics_for_decision_field(rows, decision_field)
            for route_name, decision_field in route_fields.items()
        }
        for split, rows in split_rows.items()
    }
    report = {
        "benchmark_version": "qwen72b_prompt_v12_gemini38_cascade_v1",
        "rows": len(output),
        "decision_counts": dict(Counter(row["cascade_status"] for row in output)),
        "route_counts": dict(Counter(row["cascade_route"] for row in output)),
        "risk_counts": dict(
            Counter(
                reason
                for row in output
                for reason in row["cascade_risk_reasons"].split("|")
                if reason
            )
        ),
        "all": metrics(output),
        "test": metrics(test_rows),
        "model_and_route_comparison": {
            "qwen_only_all": metrics_for_decision_field(output, "qwen_decision"),
            "qwen_only_test": metrics_for_decision_field(test_rows, "qwen_decision"),
            "gemini_only_all": metrics_for_decision_field(output, "gemini_decision"),
            "gemini_only_test": metrics_for_decision_field(test_rows, "gemini_decision"),
            "qwen_hard_reject_cascade_all": metrics(output),
            "qwen_hard_reject_cascade_test": metrics(test_rows),
            "gemini_rescues_qwen_rejects_all": metrics_for_decision_field(
                output, "rescue_variant_binary_decision"
            ),
            "gemini_rescues_qwen_rejects_test": metrics_for_decision_field(
                test_rows, "rescue_variant_binary_decision"
            ),
        },
        "model_and_route_comparison_by_split": comparison_by_split,
        "by_split": grouped_metrics(output, "evaluation_split"),
        "source_annotation_partition": grouped_metrics(output, "annotation_partition"),
        "by_dataset": grouped_metrics(output, "dataset"),
        "by_branch": grouped_metrics(output, "branch"),
        "by_audit_operation": grouped_metrics(output, "audit_operation_expected"),
        "by_provenance_relation": grouped_metrics(output, "audit_relation"),
        "cost": {
            "qwen_all_usd": qwen_cost,
            "gemini_routed_usd": routed_gemini_cost,
            "cascade_effective_usd": qwen_cost + routed_gemini_cost,
            "gemini_existing_all_rows_usd": sum(
                float(row.get("cost_usd") or 0.0) for row in gemini.values()
            ),
        },
        "counterfactual_qwen_reject_gemini_accept": {
            "human_accepted_potential_rescues": counterfactual_rescues,
            "human_rejected_conflicts": counterfactual_conflicts,
        },
        "quarantine_is_counted_as_rejected_for_selection_metrics": True,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
