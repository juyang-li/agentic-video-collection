#!/usr/bin/env python3
"""Resolve human uncertain labels into keep/delete/neutral using review notes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


POSITIVE_RE = re.compile(r"基本可用|偏正面|可以用|可保留")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    output: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        raw_status = row.get("overall_status", "").strip()
        raw_notes = row.get("notes", "").strip()
        if raw_status == "usable":
            policy_status, policy_basis = "usable", "explicit_usable_keep"
        elif raw_status == "unusable":
            policy_status, policy_basis = "unusable", "explicit_unusable_delete"
        elif raw_status == "uncertain" and POSITIVE_RE.search(raw_notes):
            policy_status, policy_basis = "usable", "positive_uncertain_keep"
        elif raw_status == "uncertain" and raw_notes:
            policy_status, policy_basis = "unusable", "negative_uncertain_delete"
        elif raw_status == "uncertain":
            policy_status, policy_basis = "uncertain", "neutral_uncertain_flexible"
        else:
            raise ValueError(
                f"unsupported human status for {row.get('sample_id')}: {raw_status!r}"
            )
        row["human_raw_status"] = raw_status
        row["human_raw_notes"] = raw_notes
        row["policy_basis"] = policy_basis
        row["overall_status"] = policy_status
        row["notes"] = (
            f"原始={raw_status}；策略={policy_basis}；备注={raw_notes or '无'}"
        )
        output.append(row)

    extra = ["human_raw_status", "human_raw_notes", "policy_basis"]
    output_fields = list(dict.fromkeys([*fields, *extra]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    summary = {
        "rows": len(output),
        "raw_status": dict(Counter(row["human_raw_status"] for row in output)),
        "policy_status": dict(Counter(row["overall_status"] for row in output)),
        "policy_basis": dict(Counter(row["policy_basis"] for row in output)),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
