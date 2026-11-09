#!/usr/bin/env python3
"""Export the frozen 300-pair benchmark composition as a readable workbook."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
DEFAULT_INPUT = RUN / "benchmark_manifest_300.csv"
DEFAULT_OUTPUT = RUN / "stage2_systematic_binary_benchmark_v1_composition.xlsx"


DATASET_NAMES = {"goku": "GOKU", "reco": "ReCo", "egoedit": "EgoEdit"}
OPERATION_NAMES = {"add": "add", "remove": "remove", "replace": "replace"}
RELATION_NAMES = {
    "forward": "正向（target 为生成侧）",
    "inverse": "反向（source 为生成侧，审核时交换）",
    "training_direction_both_generated": "双侧生成（保持训练方向）",
    "training_direction_generation_side_unresolved": (
        "生成侧未解析（指令/内容保持按 source→target，伪影查两侧）"
    ),
}
GENERATED_NAMES = {
    "source": "source",
    "target": "target",
    "both": "source + target",
    "unresolved": "generation side unresolved；检查两侧",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def style_header(sheet, row: int) -> None:
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[row]:
        if cell.value is not None:
            cell.fill = fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")


def set_widths(sheet, widths: dict[int, float]) -> None:
    for index, width in widths.items():
        sheet.column_dimensions[get_column_letter(index)].width = width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_csv(args.input)
    if len(rows) != 300 or len({row["sample_id"] for row in rows}) != 300:
        raise ValueError("input must contain the frozen 300 unique benchmark rows")

    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    summary.sheet_view.showGridLines = False
    summary["A1"] = "Stage 2 systematic 人工审核集构成"
    summary["A1"].font = Font(size=16, bold=True, color="1F4E78")
    summary.merge_cells("A1:F1")
    summary["A3"] = "范围"
    summary["B3"] = "新冻结的 test_v1：300 条；不包含已有 139 条 development 人工标签"
    summary.merge_cells("B3:F3")
    summary["A4"] = "人工入口"
    summary["B4"] = "accepted / rejected / uncertain；最终模型计分排除 uncertain"
    summary.merge_cells("B4:F4")
    summary["A5"] = "任务口径"
    summary["B5"] = (
        "已知生成侧时按 reference→generated；ReCo 逐对生成沿袭未发布，按训练 source→target "
        "检查指令/内容保持，并检查两侧伪影；GOKU add 交换后按 remove 审核"
    )
    summary.merge_cells("B5:F5")
    summary["A6"] = "总量"
    summary["B6"] = len(rows)
    for cell in (summary["A3"], summary["A4"], summary["A5"], summary["A6"]):
        cell.font = Font(bold=True)

    summary.append([])
    summary.append(["数据集", "原始分支", "训练任务", "实际审核任务", "数量", "说明"])
    detail_start = summary.max_row
    strata_order = [
        "goku_add_inverse_remove",
        "goku_remove",
        "reco_remove",
        "reco_add",
        "reco_replace",
        "egoedit_replace",
    ]
    by_stratum = {key: [row for row in rows if row["benchmark_stratum"] == key] for key in strata_order}
    for key in strata_order:
        values = by_stratum[key]
        first = values[0]
        notes = {
            "goku_add_inverse_remove": "source 是生成侧；交换后作为 remove 检查",
            "goku_remove": "正向 remove",
            "reco_remove": "生成侧未解析；训练方向判断，伪影检查两侧",
            "reco_add": "生成侧未解析；训练方向判断，伪影检查两侧",
            "reco_replace": "生成侧未解析；训练方向判断，伪影检查两侧",
            "egoedit_replace": "包含单侧生成与双侧生成，按 provenance 路由",
        }[key]
        summary.append([
            DATASET_NAMES[first["dataset"]],
            first["branch"],
            OPERATION_NAMES[first["operation"]],
            OPERATION_NAMES[first["audit_operation_expected"]],
            len(values),
            notes,
        ])
    summary.append(["合计", "", "", "", len(rows), ""])
    style_header(summary, detail_start)

    summary.append([])
    summary.append(["实际审核任务", "数量", "占 300 条比例"])
    operation_start = summary.max_row
    counts = Counter(row["audit_operation_expected"] for row in rows)
    for operation in ("remove", "replace", "add"):
        summary.append([operation, counts[operation], counts[operation] / len(rows)])
        summary.cell(summary.max_row, 3).number_format = "0.0%"
    summary.append(["合计", len(rows), 1.0])
    summary.cell(summary.max_row, 3).number_format = "0.0%"
    style_header(summary, operation_start)
    set_widths(summary, {1: 18, 2: 30, 3: 16, 4: 18, 5: 12, 6: 54})
    for row in summary.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    details = workbook.create_sheet("300条明细")
    headers = [
        "序号",
        "sample_id",
        "数据集",
        "原始分支",
        "训练任务",
        "实际审核任务",
        "是否 remove 类",
        "审核方向",
        "生成侧/检查范围",
        "source provenance",
        "target provenance",
        "provenance 依据",
        "数据版本来源",
        "审核指令",
    ]
    details.append(headers)
    for row in sorted(rows, key=lambda value: int(value["benchmark_rank"])):
        operation = row["audit_operation_expected"]
        details.append([
            int(row["benchmark_rank"]),
            row["sample_id"],
            DATASET_NAMES[row["dataset"]],
            row["branch"],
            OPERATION_NAMES[row["operation"]],
            OPERATION_NAMES[operation],
            "是" if operation == "remove" else "否",
            RELATION_NAMES[row["audit_relation"]],
            GENERATED_NAMES[row["generated_role"]],
            row["source_provenance"],
            row["target_provenance"],
            row["provenance_basis"],
            " · ".join(
                value
                for value in (
                    row["dataset"], row.get("repo_id", ""), row.get("revision", "")
                )
                if value
            ),
            row["audit_instruction"],
        ])
    style_header(details, 1)
    details.freeze_panes = "A2"
    details.auto_filter.ref = f"A1:N{details.max_row}"
    details.sheet_view.showGridLines = False
    set_widths(details, {
        1: 9, 2: 38, 3: 12, 4: 24, 5: 12, 6: 16, 7: 16,
        8: 48, 9: 34, 10: 28, 11: 28, 12: 68, 13: 48, 14: 90,
    })
    for row in details.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    workbook.properties.title = "Stage 2 systematic binary benchmark v1 composition"
    workbook.properties.subject = "Dataset and task composition for the frozen 300-pair test_v1"
    workbook.properties.creator = "lance_general_edit_finetune"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
