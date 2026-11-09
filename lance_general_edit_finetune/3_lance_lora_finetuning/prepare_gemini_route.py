#!/usr/bin/env python3
"""Select all non-low-risk Qwen rows for Gemini's final rescue decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from compose_and_evaluate_cascade import normalized_decision, risk_reasons, unique_map


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, fields = read_csv(args.manifest)
    scores, _ = read_csv(args.qwen_scores)
    by_id = unique_map(scores, "Qwen scores")
    if set(by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("manifest and Qwen score IDs differ")

    selected: list[dict[str, str]] = []
    route_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for row in rows:
        score = by_id[row["sample_id"]]
        skipped = score.get("model_backend") == "skipped_preclassified_risk"
        decision = "skipped_preclassified_risk" if skipped else normalized_decision(score)
        risks = risk_reasons(row, score)
        reason = ""
        if skipped:
            if not risks:
                raise ValueError(f"Qwen skip without a preclassified risk: {row['sample_id']}")
            reason = "qwen_skipped_preclassified_risk"
        elif decision == "rejected":
            reason = "qwen_rejected_requires_gemini_rescue_decision"
        elif decision in {"uncertain", "error"}:
            reason = "qwen_uncertain_or_error"
        elif decision == "accepted" and risks:
            reason = "qwen_accept_high_risk"
        if not reason:
            continue
        routed = dict(row)
        routed.update(
            {
                "cascade_qwen_decision": decision,
                "cascade_gemini_route": reason,
                "cascade_risk_reasons": "|".join(risks),
            }
        )
        selected.append(routed)
        route_counts[reason] += 1
        risk_counts.update(risks)

    extra = ["cascade_qwen_decision", "cascade_gemini_route", "cascade_risk_reasons"]
    output_fields = list(dict.fromkeys([*fields, *extra]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)
    temporary.replace(args.output)
    summary = {
        "route_version": "qwen72b_prompt_v12_to_gemini38_rescue_v2",
        "input_rows": len(rows),
        "routed_rows": len(selected),
        "route_rate": len(selected) / len(rows) if rows else 0.0,
        "route_counts": dict(route_counts),
        "risk_counts": dict(risk_counts),
        "by_dataset": dict(Counter(row.get("dataset", "") for row in selected)),
        "by_branch": dict(Counter(row.get("branch", "") for row in selected)),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
