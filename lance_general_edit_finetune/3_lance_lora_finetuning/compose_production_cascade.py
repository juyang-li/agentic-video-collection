#!/usr/bin/env python3
"""Compose the frozen rescue cascade after routed Gemini scoring."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from compose_and_evaluate_cascade import BINARY, normalized_decision, risk_reasons, unique_map


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def grouped_counts(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[row.get(field, "") or "<missing>"][row["cascade_status"]] += 1
    return {name: dict(counts) for name, counts in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-scores", type=Path, required=True)
    parser.add_argument("--gemini-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, fields = read_csv(args.manifest)
    qwen_rows, _ = read_csv(args.qwen_scores)
    gemini_rows, _ = read_csv(args.gemini_scores)
    qwen = unique_map(qwen_rows, "Qwen scores")
    gemini = unique_map(gemini_rows, "Gemini scores")
    expected = {row["sample_id"] for row in rows}
    if set(qwen) != expected:
        raise ValueError("manifest and Qwen score IDs differ")

    output: list[dict[str, str]] = []
    required_gemini: set[str] = set()
    used_gemini: set[str] = set()
    for source in rows:
        sample_id = source["sample_id"]
        qrow = qwen[sample_id]
        skipped = qrow.get("model_backend") == "skipped_preclassified_risk"
        qdecision = "skipped_preclassified_risk" if skipped else normalized_decision(qrow)
        risks = risk_reasons(source, qrow)
        grow: dict[str, str] = {}
        gdecision = "not_routed"
        if skipped:
            if not risks:
                raise ValueError(f"Qwen skip without a preclassified risk: {sample_id}")
            required_gemini.add(sample_id)
            if sample_id not in gemini:
                raise ValueError(f"pre-routed Gemini score missing: {sample_id}")
            grow = gemini[sample_id]
            used_gemini.add(sample_id)
            gdecision = normalized_decision(grow)
            route = "gemini_direct_preclassified_risk"
            final = gdecision if gdecision in BINARY else "quarantine"
        elif not (qdecision == "accepted" and not risks):
            required_gemini.add(sample_id)
            if sample_id not in gemini:
                raise ValueError(f"routed Gemini score missing: {sample_id}")
            grow = gemini[sample_id]
            used_gemini.add(sample_id)
            gdecision = normalized_decision(grow)
            if qdecision == "rejected":
                route = "gemini_decides_qwen_rejected"
            elif qdecision in {"uncertain", "error"}:
                route = "gemini_resolve_qwen_uncertain_or_error"
            else:
                route = "gemini_confirm_qwen_accept_high_risk"
            final = gdecision if gdecision in BINARY else "quarantine"
        elif qdecision == "accepted":
            route = "qwen_accept_low_risk"
            final = "accepted"
        else:
            route = "invalid_qwen_status"
            final = "quarantine"

        combined = dict(source)
        combined.update(
            {
                "qwen_decision": qdecision,
                "qwen_instruction_following": qrow.get("instruction_following", ""),
                "qwen_visible_artifact": qrow.get("visible_artifact", ""),
                "qwen_content_preservation": qrow.get("content_preservation", ""),
                "qwen_generated_side": qrow.get("generated_side", ""),
                "audit_operation": qrow.get("audit_operation", ""),
                "audit_relation": qrow.get("audit_relation", ""),
                "audit_routing_transform": qrow.get("audit_routing_transform", ""),
                "audit_left_training_role": qrow.get("audit_left_training_role", ""),
                "audit_right_training_role": qrow.get("audit_right_training_role", ""),
                "audit_left_label": qrow.get("audit_left_label", ""),
                "audit_right_label": qrow.get("audit_right_label", ""),
                "audit_generated_position": qrow.get("audit_generated_position", ""),
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
                "cascade_status": final,
            }
        )
        output.append(combined)
    if set(gemini) != required_gemini:
        extra = sorted(set(gemini) - required_gemini)[:5]
        missing = sorted(required_gemini - set(gemini))[:5]
        raise ValueError(f"Gemini routed ID mismatch; missing={missing}, extra={extra}")

    extra_fields = [
        "qwen_decision", "qwen_instruction_following", "qwen_visible_artifact",
        "qwen_content_preservation", "qwen_reason_codes_json", "qwen_evidence_json",
        "qwen_generated_side", "audit_operation", "audit_relation",
        "audit_routing_transform", "audit_left_training_role",
        "audit_right_training_role", "audit_left_label", "audit_right_label",
        "audit_generated_position", "qwen_cost_usd", "gemini_decision",
        "gemini_instruction_following",
        "gemini_visible_artifact", "gemini_content_preservation",
        "gemini_reason_codes_json", "gemini_evidence_json", "gemini_cost_usd",
        "cascade_route", "cascade_risk_reasons", "cascade_status",
    ]
    output_fields = list(dict.fromkeys([*fields, *extra_fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    temporary.replace(args.output)
    summary = {
        "cascade_version": "qwen72b_prompt_v12_gemini38_rescue_v2",
        "rows": len(output),
        "decision_counts": dict(Counter(row["cascade_status"] for row in output)),
        "route_counts": dict(Counter(row["cascade_route"] for row in output)),
        "risk_counts": dict(Counter(reason for row in output for reason in row["cascade_risk_reasons"].split("|") if reason)),
        "by_dataset": grouped_counts(output, "dataset"),
        "by_branch": grouped_counts(output, "branch"),
        "qwen_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in qwen.values()),
        "gemini_routed_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in gemini.values()),
    }
    summary["cascade_cost_usd"] = summary["qwen_cost_usd"] + summary["gemini_routed_cost_usd"]
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
