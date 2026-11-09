#!/usr/bin/env python3
"""Render the completed three-model pairwise blind evaluation as reviewable tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/coinve_ablation_v1"
INPUT = ARTIFACT / "gemini_pairwise_blind_v1/summary.json"
OUTPUT_JSON = ARTIFACT / "pairwise_report.json"
OUTPUT_MARKDOWN = ARTIFACT / "pairwise_report.md"
LABELS = {
    "base": "Base",
    "single4k_step677": "Single-4K",
    "coinve4k_step677": "CoinVE-4K",
    "mixed2k2k_step677": "Mixed-4K",
}
DIMENSIONS = {
    "instruction": "指令完成度",
    "fidelity": "内容保真度",
    "overall": "综合偏好",
}
STAGES = {
    "short_subject": "Short",
    "long_subject": "Long",
    "attribute_all": "Attribute",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def decisive_rate(counts: dict, left: str, right: str) -> str:
    decisive = counts[left] + counts[right]
    return "—" if not decisive else f"{counts[left] / decisive:.1%} / {counts[right] / decisive:.1%}"


def build_markdown(summary: dict, title: str = "CoinVE 数据消融：IVEBench 两两盲评") -> str:
    consensus = summary["bidirectional_consensus"]
    lines = [
        f"# {title}",
        "",
        f"- 裁判任务：{summary['tasks_succeeded']:,}/{summary['tasks_requested']:,} 成功",
        f"- 双顺序样本-模型对：{consensus['groups']:,}",
        f"- API 报告费用：${summary.get('reported_cost_usd', 0.0):.4f}",
        "- 计票规则：A/B 与 B/A 归一化后结论一致才计入模型胜出或平局；其余记为顺序不一致。",
        "",
        "## 顺序一致率",
        "",
        "| 维度 | 一致率 |",
        "|---|---:|",
    ]
    for dimension, label in DIMENSIONS.items():
        lines.append(f"| {label} | {summary['order_consistency'][dimension]:.1%} |")
    for pair, values in consensus["by_pair"].items():
        left, right = pair.split("__vs__")
        left_label, right_label = LABELS[left], LABELS[right]
        lines.extend([
            "",
            f"## {left_label} vs {right_label}",
            "",
            f"| 维度 | {left_label} 胜 | {right_label} 胜 | 平 | 顺序不一致 | 有效胜负比例（左/右） |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for dimension, label in DIMENSIONS.items():
            counts = values["counts"][dimension]
            lines.append(
                f"| {label} | {counts[left]} | {counts[right]} | {counts['tie']} | "
                f"{counts['order_inconsistent']} | {decisive_rate(counts, left, right)} |"
            )
        lines.extend([
            "",
            "| 子集 | 左胜 | 右胜 | 平 | 顺序不一致 | 有效胜负比例（左/右） |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for stage in STAGES:
            counts = consensus["by_pair_stage"][pair][stage]["counts"]["overall"]
            lines.append(
                f"| {STAGES[stage]} | {counts[left]} | {counts[right]} | {counts['tie']} | "
                f"{counts['order_inconsistent']} | {decisive_rate(counts, left, right)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--expected-tasks", type=int, default=1368)
    parser.add_argument("--expected-groups", type=int, default=684)
    parser.add_argument("--title", default="CoinVE 数据消融：IVEBench 两两盲评")
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    if not (
        summary.get("status") == "complete"
        and summary.get("tasks_requested") == args.expected_tasks
        and summary.get("tasks_succeeded") == args.expected_tasks
        and summary.get("tasks_failed") == 0
        and summary.get("bidirectional_consensus", {}).get("groups") == args.expected_groups
    ):
        raise ValueError("pairwise judge summary is incomplete")
    report = {
        "status": "complete",
        "source_summary": str(args.input.resolve()),
        "source_summary_sha256": sha256_file(args.input),
        "tasks": args.expected_tasks,
        "bidirectional_groups": args.expected_groups,
        "order_consistency": summary["order_consistency"],
        "bidirectional_consensus": summary["bidirectional_consensus"],
        "reported_cost_usd": summary.get("reported_cost_usd", 0.0),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, report)
    write_text(args.output_markdown, build_markdown(summary, args.title))
    print(json.dumps({
        "status": "complete", "json": str(args.output_json), "markdown": str(args.output_markdown),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
