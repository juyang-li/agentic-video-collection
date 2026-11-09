#!/usr/bin/env python3
"""Replace every production cascade decision with Gemini's decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BINARY = {"accepted", "rejected"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def grouped(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[row.get(field, "")][row["cascade_status"]] += 1
    return {key: dict(value) for key, value in sorted(result.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, action="append", required=True)
    parser.add_argument("--supplement-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    seen: set[str] = set()
    for path in args.cascade:
        part, part_fields = read_csv(path)
        fields = list(dict.fromkeys([*fields, *part_fields]))
        for row in part:
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample ID: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
    score_rows, _ = read_csv(args.supplement_scores)
    scores = {row["sample_id"]: row for row in score_rows}
    if len(scores) != len(score_rows):
        raise ValueError("supplement scores contain duplicate sample IDs")
    required = {
        row["sample_id"]
        for row in rows
        if row.get("cascade_route") == "qwen_accept_low_risk"
    }
    if set(scores) != required:
        raise ValueError(
            f"supplement coverage mismatch: expected={len(required)} got={len(scores)}"
        )

    output: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        row["previous_cascade_route"] = row.get("cascade_route", "")
        row["previous_cascade_status"] = row.get("cascade_status", "")
        score = scores.get(row["sample_id"])
        if score is not None:
            decision = score.get("model_decision", "")
            if score.get("parse_status") != "pass":
                decision = "uncertain"
            row.update(
                {
                    "gemini_decision": decision,
                    "gemini_instruction_following": score.get("instruction_following", ""),
                    "gemini_visible_artifact": score.get("visible_artifact", ""),
                    "gemini_content_preservation": score.get("content_preservation", ""),
                    "gemini_reason_codes_json": score.get("reason_codes_json", ""),
                    "gemini_evidence_json": score.get("evidence_json", ""),
                    "gemini_cost_usd": score.get("cost_usd", ""),
                    "cascade_route": "gemini_full_review_supplement",
                }
            )
        else:
            decision = row.get("gemini_decision", "")
            row["cascade_route"] = "gemini_full_review_existing"
        row["cascade_status"] = decision if decision in BINARY else "quarantine"
        output.append(row)

    extra = ["previous_cascade_route", "previous_cascade_status"]
    fields = list(dict.fromkeys([*fields, *extra]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    changed = [
        row
        for row in output
        if row["cascade_status"] != row["previous_cascade_status"]
    ]
    summary = {
        "cascade_version": "gemini38_full_review_v1",
        "rows": len(output),
        "decision_counts": dict(Counter(row["cascade_status"] for row in output)),
        "changed_rows": len(changed),
        "change_types": dict(
            Counter(
                f"{row['previous_cascade_status']}->{row['cascade_status']}"
                for row in changed
            )
        ),
        "by_dataset": grouped(output, "dataset"),
        "by_branch": grouped(output, "branch"),
        "supplemental_cost_usd": sum(
            float(row.get("cost_usd") or 0) for row in scores.values()
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
