#!/usr/bin/env python3
"""Export Phase-A decisions, training pools, and a compact reproducible report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_RUN / "stage1_0_results.csv")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_RUN / "stage1_5k_report.md")
    args = parser.parse_args()

    metadata = read_csv(args.metadata)
    decisions = read_csv(args.decisions)
    fields = list(decisions[0])
    by_decision = {
        decision: [row for row in decisions if row["stage1_1_decision"] == decision]
        for decision in ("accepted", "uncertain", "rejected", "pending")
    }
    for decision, rows in by_decision.items():
        write_csv(args.output_dir / f"stage1_1_{decision}.csv", rows, fields)

    accepted = by_decision["accepted"]
    main_pool = [row for row in accepted if row["source_provenance"] == "original"]
    auxiliary_pool = [row for row in accepted if row["source_provenance"] == "edited"]
    provenance_hold = [
        row for row in accepted if row["source_provenance"] not in {"original", "edited"}
    ]
    write_csv(args.output_dir / "training_main_candidates.csv", main_pool, fields)
    write_csv(args.output_dir / "training_auxiliary_candidates.csv", auxiliary_pool, fields)
    write_csv(args.output_dir / "training_provenance_hold.csv", provenance_hold, fields)

    branches = sorted({row["branch"] for row in decisions})
    branch_rows = []
    for branch in branches:
        subset = [row for row in decisions if row["branch"] == branch]
        counts = Counter(row["stage1_1_decision"] for row in subset)
        branch_rows.append(
            [branch, len(subset), counts["accepted"], counts["uncertain"], counts["rejected"], counts["pending"]]
        )
    reasons = Counter(
        reason
        for row in decisions
        for reason in row["stage1_1_reasons"].split("|")
        if reason
    )
    metadata_counts = Counter(row["metadata_decision"] for row in metadata)
    report = [
        "# Stage 1 5k 筛选报告",
        "",
        "过滤版本：`general_edit_stage1_pilot_v1`。本报告由冻结 manifest 和 CSV 决策自动生成。",
        "",
        "## 总览",
        "",
        f"- 5k pilot 输入：{len(metadata)} 对。",
        f"- Metadata：{metadata_counts['accepted']} accepted，{metadata_counts['rejected']} rejected。",
        f"- Stage 1.1：{len(accepted)} accepted，{len(by_decision['uncertain'])} uncertain，{len(by_decision['rejected'])} rejected，{len(by_decision['pending'])} pending。",
        f"- 可直接进入 single-step 主池：{len(main_pool)}；edited-source 辅助池：{len(auxiliary_pool)}；provenance 待确认：{len(provenance_hold)}。",
        "",
        "## 分支结果",
        "",
        markdown_table(
            ["分支", "输入", "accepted", "uncertain", "rejected", "pending"],
            branch_rows,
        ),
        "",
        "## 高频原因",
        "",
    ]
    report.extend(f"- `{reason}`：{count}" for reason, count in reasons.most_common(20))
    report.extend(
        [
            "",
            "## 使用边界",
            "",
            "`training_main_candidates.csv` 只按 source provenance 和 Stage 1.1 结果分池，尚未替代 Stage 1.2/1.5 specialist 与 Stage 1.6 pair/instruction 检查。`uncertain` 不进入当前训练候选。",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(
        f"accepted={len(accepted)} uncertain={len(by_decision['uncertain'])} "
        f"rejected={len(by_decision['rejected'])} pending={len(by_decision['pending'])} "
        f"main={len(main_pool)} auxiliary={len(auxiliary_pool)}"
    )


if __name__ == "__main__":
    main()
