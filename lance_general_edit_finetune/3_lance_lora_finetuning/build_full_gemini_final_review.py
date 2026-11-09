#!/usr/bin/env python3
"""Select a deterministic 100-pair audit across the final Gemini filtering paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_cascade_review import coverage_pick


QUOTAS = {
    "final_4k_selected": 20,
    "accepted_reserve": 10,
    "full_review_overturn_reject": 15,
    "gemini_existing_reject": 15,
    "refill_accept": 10,
    "refill_reject": 10,
    "stage1_round4_reject": 10,
    "quarantine": 10,
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--final-pool", type=Path, required=True)
    parser.add_argument("--stage1-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    cascade, cascade_fields = read_csv(args.cascade)
    final_pool, _ = read_csv(args.final_pool)
    stage1, stage1_fields = read_csv(args.stage1_decisions)
    final_ids = {row["sample_id"] for row in final_pool}
    for row in cascade:
        row["selected_4k"] = "yes" if row["sample_id"] in final_ids else "no"

    stage1_rejected: list[dict[str, str]] = []
    for source in stage1:
        if source.get("stage1_1_decision") != "rejected":
            continue
        row = dict(source)
        row.update(
            {
                "selected_4k": "no",
                "gemini_decision": "not_routed",
                "cascade_route": "stage1_specialist_reject",
                "cascade_risk_reasons": row.get("stage1_1_reasons", ""),
                "cascade_status": "rejected",
            }
        )
        stage1_rejected.append(row)

    candidates = {
        "final_4k_selected": [
            row for row in cascade if row["sample_id"] in final_ids
        ],
        "accepted_reserve": [
            row
            for row in cascade
            if row.get("cascade_status") == "accepted" and row["sample_id"] not in final_ids
        ],
        "full_review_overturn_reject": [
            row
            for row in cascade
            if row.get("cascade_route") == "gemini_full_review_supplement"
            and row.get("previous_cascade_status") == "accepted"
            and row.get("cascade_status") == "rejected"
        ],
        "gemini_existing_reject": [
            row
            for row in cascade
            if row.get("cascade_route") == "gemini_full_review_existing"
            and row.get("cascade_status") == "rejected"
        ],
        "refill_accept": [
            row
            for row in cascade
            if row.get("cascade_route") == "gemini_direct_refill_round4"
            and row.get("cascade_status") == "accepted"
        ],
        "refill_reject": [
            row
            for row in cascade
            if row.get("cascade_route") == "gemini_direct_refill_round4"
            and row.get("cascade_status") == "rejected"
        ],
        "stage1_round4_reject": stage1_rejected,
        "quarantine": [
            row for row in cascade if row.get("cascade_status") == "quarantine"
        ],
    }

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    available: dict[str, int] = {}
    for category, quota in QUOTAS.items():
        pool = candidates[category]
        available[category] = len(pool)
        chosen = coverage_pick(pool, quota, used)
        used.update(row["sample_id"] for row in chosen)
        for source in chosen:
            row = dict(source)
            row["audit_category"] = category
            selected.append(row)

    fields = list(
        dict.fromkeys(
            [*cascade_fields, *stage1_fields, "selected_4k", "audit_category"]
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)

    summary = {
        "selection_seed": 20260904,
        "selection_policy": "fixed_category_quota_then_deterministic_maximum_stratum_coverage",
        "rows": len(selected),
        "requested_by_category": QUOTAS,
        "available_by_category": available,
        "selected_by_category": dict(Counter(row["audit_category"] for row in selected)),
        "selected_by_branch": dict(Counter(row.get("branch", "") for row in selected)),
        "selected_by_status": dict(Counter(row.get("cascade_status", "") for row in selected)),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
