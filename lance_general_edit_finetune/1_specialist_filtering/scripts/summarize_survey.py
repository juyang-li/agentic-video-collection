#!/usr/bin/env python3
"""Validate the exported review CSV and write a compact error-survey report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUS = {"pass", "hard_fail", "uncertain"}
DEFECT_LABELS = {
    "black_white": "黑屏/白屏",
    "exposure": "严重过曝/欠曝",
    "freeze_repeat": "冻结/重复帧",
    "global_blur": "全局模糊/压缩",
    "local_blur": "局部失焦/边缘软化",
    "smear_ghost": "涂抹/残影",
    "repair_hole": "修补洞/漂浮碎片",
    "mask_seam": "遮罩边缘/纹理断裂",
    "local_flicker": "局部闪烁/时间不稳",
    "geometry_human": "几何/人脸/人体异常",
    "other": "其他视频质量问题",
}
PAIR_LABELS = {
    "unrelated_pair": "source-target 完全错配",
    "temporal_mismatch": "时间/动作/相机轨迹错位",
    "direction_wrong": "编辑方向错误",
    "target_absent": "instruction 目标不在 source",
    "no_op": "no-op 或编辑不足",
    "wrong_object": "删错/加错/换错对象",
    "non_edit_changed": "非编辑区域被改动",
    "physical_logic": "悬空物/姿势/交互不合理",
    "multi_step": "不是 single-step",
    "other": "其他 pair 问题",
}
SOURCE_FLAG_LABELS = {
    "suspected_edited": "疑似编辑生成的视频",
    "edit_residual": "残留上一次编辑痕迹",
    "unsuitable_input": "不适合作为当前任务输入",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_flags(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})" if total else "0/0"


def status_table(rows: list[dict[str, str]], prefix: str) -> list[str]:
    lines = ["| 分支 | 通过 | 明显坏 | 存疑 |", "| --- | ---: | ---: | ---: |"]
    for branch in sorted({row["branch"] for row in rows}):
        subset = [row for row in rows if row["branch"] == branch]
        counts = Counter(row[f"{prefix}_status"] for row in subset)
        lines.append(
            f"| {branch} | {rate(counts['pass'], len(subset))} | "
            f"{rate(counts['hard_fail'], len(subset))} | {rate(counts['uncertain'], len(subset))} |"
        )
    return lines


def trigger_decision(total: int, by_branch: Counter[str]) -> str:
    if total == 0:
        return "本轮未观察到，不新增专门模型"
    if total <= 2 and max(by_branch.values(), default=0) <= 2:
        return "先在对应分支追加 50 条定向确认"
    if total >= 5 or max(by_branch.values(), default=0) >= 3:
        return "达到 specialist 评估触发条件"
    return "证据处于边界，先检查代表样本再决定"


def validate(
    manifest: list[dict[str, str]], review: list[dict[str, str]], allow_incomplete: bool
) -> list[dict[str, str]]:
    manifest_ids = {row["sample_id"] for row in manifest}
    review_by_id = {row["sample_id"]: row for row in review}
    if len(review_by_id) != len(review):
        raise ValueError("review CSV contains duplicate sample_id")
    unknown = set(review_by_id) - manifest_ids
    if unknown:
        raise ValueError(f"review CSV contains unknown sample IDs: {sorted(unknown)[:5]}")
    ordered: list[dict[str, str]] = []
    for item in manifest:
        review_row = review_by_id.get(item["sample_id"])
        if review_row is None:
            if allow_incomplete:
                continue
            raise ValueError(f"missing review row: {item['sample_id']}")
        for prefix in ("target", "source", "pair"):
            status = review_row.get(f"{prefix}_status", "")
            if not status:
                if allow_incomplete:
                    break
                raise ValueError(f"missing {prefix}_status for {item['sample_id']}")
            if status not in ALLOWED_STATUS:
                raise ValueError(f"invalid {prefix}_status={status!r}")
        else:
            merged = dict(item)
            merged.update(review_row)
            ordered.append(merged)
    return ordered


def build_report(rows: list[dict[str, str]], manifest_size: int) -> str:
    lines = [
        "# Specialist error survey 报告",
        "",
        f"完成审阅：**{len(rows)}/{manifest_size}**。本报告区分单视频质量与 pair/instruction 错误。",
        "",
        "## 单视频结论",
        "",
        "### Target",
        "",
        *status_table(rows, "target"),
        "",
        "### Source",
        "",
        *status_table(rows, "source"),
        "",
        "## Pair 结论",
        "",
        *status_table(rows, "pair"),
        "",
        "## 错误模式与自动化触发判断",
        "",
        "| 错误模式 | Target | Source | 判断 |",
        "| --- | ---: | ---: | --- |",
    ]
    target_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    target_branches: dict[str, Counter[str]] = defaultdict(Counter)
    source_branches: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for flag in split_flags(row.get("target_defects", "")):
            target_counts[flag] += 1
            target_branches[flag][row["branch"]] += 1
        for flag in split_flags(row.get("source_defects", "")):
            source_counts[flag] += 1
            source_branches[flag][row["branch"]] += 1
    for flag, label in DEFECT_LABELS.items():
        total = target_counts[flag] + source_counts[flag]
        combined = target_branches[flag] + source_branches[flag]
        lines.append(
            f"| {label} | {target_counts[flag]} | {source_counts[flag]} | "
            f"{trigger_decision(total, combined)} |"
        )

    pair_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for flag in split_flags(row.get("pair_errors", "")):
            pair_counts[flag] += 1
            if len(examples[flag]) < 5:
                examples[flag].append(row["sample_id"])
    lines.extend(
        [
            "",
            "## Pair / instruction 错误",
            "",
            "| 错误模式 | 数量 | 代表 sample ID |",
            "| --- | ---: | --- |",
        ]
    )
    for flag, label in PAIR_LABELS.items():
        lines.append(
            f"| {label} | {pair_counts[flag]} | {', '.join(examples[flag]) or '-'} |"
        )

    source_flag_counts: Counter[str] = Counter()
    source_flag_branches: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for flag in split_flags(row.get("source_flags", "")):
            source_flag_counts[flag] += 1
            source_flag_branches[flag][row["branch"]] += 1
    lines.extend(
        [
            "",
            "## Source 编辑来源风险",
            "",
            "| 标记 | 总数 | 分支分布 |",
            "| --- | ---: | --- |",
        ]
    )
    for flag, label in SOURCE_FLAG_LABELS.items():
        distribution = ", ".join(
            f"{branch}={count}" for branch, count in sorted(source_flag_branches[flag].items())
        )
        lines.append(f"| {label} | {source_flag_counts[flag]} | {distribution or '-'} |")

    ego = [row for row in rows if row["branch"] == "egoedit_replace"]
    lines.extend(
        [
            "",
            "## EgoEdit 方向",
            "",
            "| 方向 | 数量 | Source 明显坏 | Target 明显坏 | Pair 明显坏 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for direction in ("original_to_edited", "edited_to_original", "edited_to_edited"):
        subset = [row for row in ego if row["direction"] == direction]
        lines.append(
            f"| {direction} | {len(subset)} | "
            f"{sum(row['source_status'] == 'hard_fail' for row in subset)} | "
            f"{sum(row['target_status'] == 'hard_fail' for row in subset)} | "
            f"{sum(row['pair_status'] == 'hard_fail' for row in subset)} |"
        )

    seconds = [
        sum(float(row.get(f"{prefix}_seconds", 0) or 0) for prefix in ("target", "source", "pair"))
        for row in rows
    ]
    total_seconds = sum(seconds)
    lines.extend(
        [
            "",
            "## 审阅成本",
            "",
            f"- 浏览器记录的累计审阅时间：{total_seconds / 3600:.2f} 小时。",
            f"- 每条中位时间：{median(seconds):.1f} 秒。" if seconds else "- 尚无时间记录。",
            f"- 按当前中位数估算 1,000 条：{median(seconds) * 1000 / 3600:.2f} 小时。" if seconds else "",
            "",
            "## 下一步",
            "",
            "先人工检查高频错误的代表样本，再按上表触发规则决定是否增加 specialist。"
            "Pair/instruction 错误不塞入 target-only 质量模型。",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/survey_manifest.csv")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/error_survey_report.md")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = read_csv(args.manifest)
    review = read_csv(args.review)
    rows = validate(manifest, review, args.allow_incomplete)
    if not rows:
        raise ValueError("no complete review rows to summarize")
    report = build_report(rows, len(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(f"complete_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
