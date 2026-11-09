#!/usr/bin/env python3
"""Append a Stage-1 refill round judged directly by Gemini to a frozen cascade."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cascade", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gemini-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    base, base_fields = read_csv(args.base_cascade)
    manifest, manifest_fields = read_csv(args.manifest)
    score_rows, _ = read_csv(args.gemini_scores)
    scores = {row["sample_id"]: row for row in score_rows}
    if len(scores) != len(score_rows):
        raise ValueError("Gemini scores contain duplicate sample IDs")
    manifest_ids = {row["sample_id"] for row in manifest}
    if set(scores) != manifest_ids:
        raise ValueError(
            f"Gemini coverage mismatch: manifest={len(manifest_ids)} scores={len(scores)}"
        )
    base_ids = {row["sample_id"] for row in base}
    overlap = base_ids & manifest_ids
    if overlap:
        raise ValueError(f"refill overlaps base cascade: {sorted(overlap)[:5]}")

    appended: list[dict[str, str]] = []
    for source in manifest:
        score = scores[source["sample_id"]]
        decision = score.get("model_decision", "")
        if score.get("parse_status") != "pass" or decision not in BINARY:
            final = "quarantine"
        else:
            final = decision
        row = {
            **source,
            "gemini_decision": decision,
            "gemini_instruction_following": score.get("instruction_following", ""),
            "gemini_visible_artifact": score.get("visible_artifact", ""),
            "gemini_content_preservation": score.get("content_preservation", ""),
            "gemini_reason_codes_json": score.get("reason_codes_json", ""),
            "gemini_evidence_json": score.get("evidence_json", ""),
            "gemini_cost_usd": score.get("cost_usd", ""),
            "cascade_route": "gemini_direct_refill_round4",
            "cascade_risk_reasons": "strict_quota_refill",
            "cascade_status": final,
            "previous_cascade_route": "",
            "previous_cascade_status": "",
        }
        appended.append(row)

    fields = list(
        dict.fromkeys(
            [
                *base_fields,
                *manifest_fields,
                "gemini_decision",
                "gemini_instruction_following",
                "gemini_visible_artifact",
                "gemini_content_preservation",
                "gemini_reason_codes_json",
                "gemini_evidence_json",
                "gemini_cost_usd",
                "cascade_route",
                "cascade_risk_reasons",
                "cascade_status",
                "previous_cascade_route",
                "previous_cascade_status",
            ]
        )
    )
    output_rows = [*base, *appended]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.output)

    branch_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in appended:
        branch_counts[row["branch"]][row["cascade_status"]] += 1
    summary = {
        "cascade_version": "gemini38_full_review_plus_direct_refill_v1",
        "base_rows": len(base),
        "refill_rows": len(appended),
        "output_rows": len(output_rows),
        "refill_decision_counts": dict(Counter(row["cascade_status"] for row in appended)),
        "refill_by_branch": {
            branch: dict(counts) for branch, counts in sorted(branch_counts.items())
        },
        "refill_cost_usd_recorded": sum(
            float(row.get("gemini_cost_usd") or 0) for row in appended
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
