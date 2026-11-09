#!/usr/bin/env python3
"""Skip Qwen only when the frozen cascade would always defer to Gemini."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERAL_ROOT = HERE.parent
STAGE2_SCRIPTS = GENERAL_ROOT / "2_vlm_filtering" / "scripts"
sys.path.insert(0, str(STAGE2_SCRIPTS))

import run_openrouter_qwen3_judge as judge  # noqa: E402
from compose_and_evaluate_cascade import risk_reasons  # noqa: E402

QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"
QWEN_JUDGE_VERSION = "qwen2_5_vl_72b_provenance_aware_pair_judge_prompt_v12"
QWEN_PROMPT = GENERAL_ROOT / "2_vlm_filtering" / "judge_prompt_timecard_v12_provenance_aware.txt"
QWEN_PROMPT_HASH = judge.prompt_hash(QWEN_PROMPT.read_text(encoding="utf-8").strip())


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def skipped_score(row: dict[str, str], risks: list[str]) -> dict[str, str]:
    result = {field: "" for field in judge.FIELDS}
    audit = judge.build_audit_view(row)
    result.update(
        {
            "judge_version": QWEN_JUDGE_VERSION,
            "roi_method_version": judge.ROI_METHOD_VERSION,
            "visual_layout_version": judge.VISUAL_LAYOUT_VERSION,
            "model_id": QWEN_MODEL,
            "model_backend": "skipped_preclassified_risk",
            "prompt_sha256": QWEN_PROMPT_HASH,
            "sample_id": row["sample_id"],
            "dataset": row.get("dataset", ""),
            "branch": row.get("branch", ""),
            "operation": row.get("operation", ""),
            "target_scope": row.get("target_scope", ""),
            "training_operation": row.get("operation", ""),
            "training_instruction": row.get("instruction", ""),
            "audit_operation": audit.audit_operation,
            "audit_relation": audit.audit_relation,
            "audit_routing_transform": audit.routing_transform,
            "audit_left_training_role": audit.left_training_role,
            "audit_right_training_role": audit.right_training_role,
            "audit_left_label": audit.left_label,
            "audit_right_label": audit.right_label,
            "audit_generated_position": audit.generated_position,
            "original_stage1_1_decision": row.get("stage1_1_decision", ""),
            "model_reported_decision": "skipped_preclassified_risk",
            "model_decision": "skipped_preclassified_risk",
            "reason_codes_json": json.dumps(risks, ensure_ascii=False),
            "evidence_json": json.dumps(
                {
                    "routing": (
                        "Qwen omitted because frozen cascade always sends this "
                        "preclassified risk to Gemini for the final decision."
                    )
                },
                ensure_ascii=False,
            ),
            "parse_status": "skipped",
            "cost_usd": "0",
            "cost_source": "not_called",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-input", type=Path, required=True)
    parser.add_argument("--skipped-scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows, fields = read_csv(args.manifest)
    qwen_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    risks_count: Counter[str] = Counter()
    for row in rows:
        audit = judge.build_audit_view(row)
        probe = {"audit_operation": audit.audit_operation}
        risks = risk_reasons(row, probe)
        if risks:
            skipped.append(skipped_score(row, risks))
            risks_count.update(risks)
        else:
            qwen_rows.append(row)

    write_csv(args.qwen_input, qwen_rows, fields)
    write_csv(args.skipped_scores, skipped, judge.FIELDS)
    summary = {
        "selection_version": "selective_qwen_equivalent_cascade_v1",
        "input_rows": len(rows),
        "qwen_called_rows": len(qwen_rows),
        "qwen_skipped_rows": len(skipped),
        "preclassified_risks": dict(risks_count),
        "equivalence": (
            "Qwen is skipped only when the frozen rescue cascade would route the "
            "row to Gemini regardless of Qwen's decision."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
