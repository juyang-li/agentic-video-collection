#!/usr/bin/env python3
"""Merge Stage 1.1, Stage 1.2 and calibrated Qwen decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_RUN = PROJECT_ROOT / "1_specialist_filtering/artifacts/filter_run_v2"
RUN = ROOT / "artifacts/vlm_run_v1"
FINAL_FIELDS = [
    "final_filter_version",
    "final_decision",
    "final_reasons",
    "stage1_2_decision",
    "stage1_2_reason",
    "qwen_decision",
    "qwen_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_decision(
    row: dict[str, str], stage1_2: dict[str, str] | None, qwen: dict[str, str] | None
) -> tuple[str, list[str], str, str, str, str]:
    robot = row.get("route_stage1_2") == "1"
    if robot and stage1_2 is None:
        stage1_decision, stage1_reason = "uncertain", "stage1_2_missing"
    elif robot:
        stage1_decision = stage1_2.get("stage1_2_decision", "uncertain")
        stage1_reason = stage1_2.get("stage1_2_reason", "stage1_2_missing_reason")
    else:
        stage1_decision, stage1_reason = "not_applicable", ""
    if qwen is None:
        qwen_decision, qwen_reason = "uncertain", "qwen_missing"
    else:
        qwen_decision = qwen.get("qwen_decision", "uncertain")
        qwen_reason = qwen.get("qwen_reason", "qwen_missing_reason")

    if stage1_decision == "rejected" or qwen_decision == "rejected":
        decision = "rejected"
    elif stage1_decision in {"accepted", "not_applicable"} and qwen_decision == "accepted":
        decision = "accepted"
    else:
        decision = "uncertain"
    reasons = [reason for reason in (stage1_reason, qwen_reason) if reason]
    if row.get("stage1_1_decision") == "uncertain" and decision == "accepted":
        reasons.append("promoted_from_stage1_1_uncertain")
    return decision, reasons, stage1_decision, stage1_reason, qwen_decision, qwen_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SPECIALIST_RUN / "stage1_next_input.csv")
    parser.add_argument("--stage1-2", type=Path, default=SPECIALIST_RUN / "stage1_2_decisions.csv")
    parser.add_argument("--qwen", type=Path, default=RUN / "qwen_decisions.csv")
    parser.add_argument("--output-dir", type=Path, default=RUN / "final")
    args = parser.parse_args()

    source = read_csv(args.input)
    stage1 = {row["sample_id"]: row for row in read_csv(args.stage1_2)} if args.stage1_2.is_file() else {}
    qwen = {row["sample_id"]: row for row in read_csv(args.qwen)} if args.qwen.is_file() else {}
    final: list[dict[str, str]] = []
    for row in source:
        decision, reasons, s_decision, s_reason, q_decision, q_reason = merge_decision(
            row, stage1.get(row["sample_id"]), qwen.get(row["sample_id"])
        )
        final.append(
            {
                **row,
                "final_filter_version": "general_edit_stage2_vlm_filter_v1",
                "final_decision": decision,
                "final_reasons": "|".join(reasons),
                "stage1_2_decision": s_decision,
                "stage1_2_reason": s_reason,
                "qwen_decision": q_decision,
                "qwen_reason": q_reason,
            }
        )
    if len({row["sample_id"] for row in final}) != len(final):
        raise ValueError("final output contains duplicate sample_id")
    fields = list(dict.fromkeys(list(source[0]) + FINAL_FIELDS))
    write_csv(args.output_dir / "all_decisions.csv", final, fields)
    for decision in ("accepted", "uncertain", "rejected"):
        write_csv(
            args.output_dir / f"{decision}.csv",
            [row for row in final if row["final_decision"] == decision],
            fields,
        )
    accepted = [row for row in final if row["final_decision"] == "accepted"]
    write_csv(
        args.output_dir / "training_main_candidates.csv",
        [row for row in accepted if row.get("source_provenance") == "original"],
        fields,
    )
    write_csv(
        args.output_dir / "training_auxiliary_candidates.csv",
        [row for row in accepted if row.get("source_provenance") == "edited"],
        fields,
    )
    summary = {
        "filter_version": "general_edit_stage2_vlm_filter_v1",
        "input": len(final),
        "final_decisions": dict(Counter(row["final_decision"] for row in final)),
        "by_branch": {
            branch: dict(Counter(row["final_decision"] for row in final if row["branch"] == branch))
            for branch in sorted({row["branch"] for row in final})
        },
        "promoted_stage1_1_uncertain": sum(
            row["stage1_1_decision"] == "uncertain" and row["final_decision"] == "accepted"
            for row in final
        ),
        "training_main": sum(row.get("source_provenance") == "original" for row in accepted),
        "training_auxiliary": sum(row.get("source_provenance") == "edited" for row in accepted),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
