#!/usr/bin/env python3
"""Create a minimal manifest containing only videos missing full Stage-1 scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from run_stage1_quality import required_roles


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def video_id(row: dict[str, str], role: str) -> str:
    return row.get(f"{role}_video_id") or f"{row['pair_id']}:{role}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metric-set", choices=("technical", "full"), default="full")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    scores = {row["video_id"]: row for row in read_csv(args.scores)}
    selected: list[dict[str, str]] = []
    missing_roles: Counter[str] = Counter()
    for row in rows:
        missing: list[str] = []
        roles = ("source", "target") if args.metric_set == "technical" else required_roles(row)
        for role in roles:
            score = scores.get(video_id(row, role))
            if args.metric_set == "technical":
                complete = bool(
                    score and score.get("technical_status") in {"pass", "fail_short"}
                )
            else:
                complete = bool(
                    score
                    and (
                        score.get("technical_status") != "pass"
                        or (score.get("musiq_p10") and score.get("dino_max_feature_jump"))
                    )
                )
            if not complete:
                missing.append(role)
        if not missing:
            continue
        item = dict(row)
        item["quality_checked_roles"] = "both" if len(missing) == 2 else missing[0]
        selected.append(item)
        missing_roles.update(f"{row['branch']}:{role}" for role in missing)

    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(args.output)
    summary = {
        "input_pairs": len(rows),
        "metric_set": args.metric_set,
        "pairs_with_missing_scores": len(selected),
        "pairs_with_missing_full_scores": len(selected),
        "missing_video_roles": sum(missing_roles.values()),
        "by_branch_role": dict(missing_roles),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
