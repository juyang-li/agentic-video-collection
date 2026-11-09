#!/usr/bin/env python3
"""Compose the best calibrated filter: v8 default, v9 eyewear, Stage-1 policy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_BASE = RUN / "artifact_risk_30_qwen72b_hybrid_v8_scores.csv"
DEFAULT_STRICT = RUN / "artifact_risk_v9_regression8_scores.csv"
DEFAULT_OUTPUT = RUN / "artifact_risk_30_best_routed_v12_scores.csv"
DEFAULT_SUMMARY = RUN / "artifact_risk_30_best_routed_v12_summary.json"
BASE_JUDGE = "qwen2_5_vl_72b_plus_cpu_remove_reconstruction_filter_v8"
STRICT_JUDGE = "qwen2_5_vl_72b_openrouter_unambiguous_remove_pair_judge_v9"
ROUTED_JUDGE = "general_edit_structural_v8_plus_eyewear_v9_filter_v12"
PAIRED_EYEWEAR_RE = re.compile(
    r"^\s*(?:add|remove|delete|erase|insert|place|put)\s+"
    r"(?:(?:a|the)\s+)?(?:pair|set)\s+of\b.{0,90}"
    r"\b(?:eye\s*glasses|glasses|goggles|sunglasses|spectacles|eyewear)\b",
    re.IGNORECASE,
)
DIRECT_EYEWEAR_RE = re.compile(
    r"^\s*(?:add|remove|delete|erase|insert|place|put)\s+"
    r"(?:(?:a|an|the)\s+)?(?:dark|black|clear|transparent|tinted|protective|safety)?\s*"
    r"(?:eye\s*glasses|glasses|goggles|sunglasses|spectacles|eyewear)\b|"
    r"^\s*(?:添加|加入|移除|删除|去掉)\s*(?:一副|一对|这个|该)?\s*"
    r"(?:眼镜|护目镜|太阳镜|墨镜)",
    re.IGNORECASE,
)
RESULT_FIELDS = (
    "model_reported_decision",
    "model_decision",
    "instruction_following",
    "visible_artifact",
    "content_preservation",
    "reason_codes_json",
    "removal_audit_json",
    "frame_audit_json",
    "temporal_audit_json",
    "evidence_json",
    "raw_response",
)


def is_eyewear_remove_target(source: dict[str, str], score: dict[str, str]) -> bool:
    """Route only eyewear that is itself the remove target.

    Incidental mentions such as "a shelf containing glasses" or "a man with
    glasses" must not enter the face-reconstruction branch. Add/replace audits
    are also excluded even when eyewear appears in the description.
    """
    if score.get("audit_operation") != "remove":
        return False
    instruction = source.get("instruction", "")
    return bool(
        PAIRED_EYEWEAR_RE.search(instruction)
        or DIRECT_EYEWEAR_RE.search(instruction)
    )


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def mapped(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{name} contains duplicate sample IDs")
    return result


def json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-scores", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--strict-scores", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--skip-structural-policy",
        action="store_true",
        help="Report Stage-2 quality routing only, without upstream exclusions.",
    )
    args = parser.parse_args()

    manifest, _ = read_csv(args.manifest)
    base_rows, base_fields = read_csv(args.base_scores)
    strict_rows, _ = read_csv(args.strict_scores)
    expected = {row["sample_id"] for row in manifest}
    base = mapped(base_rows, "base scores")
    strict = mapped(strict_rows, "strict scores")
    if len(expected) != len(manifest) or set(base) != expected:
        raise ValueError("manifest and base scores must contain identical unique IDs")
    if any(row.get("judge_version") != BASE_JUDGE for row in base_rows):
        raise ValueError("unexpected base judge identity")
    if any(row.get("judge_version") != STRICT_JUDGE for row in strict_rows):
        raise ValueError("unexpected strict judge identity")

    output: list[dict[str, str]] = []
    routed_eyewear: list[str] = []
    structural_exclusions: list[str] = []
    for source in manifest:
        sample_id = source["sample_id"]
        score = dict(base[sample_id])
        base_cost = float(score.get("cost_usd") or 0.0)
        route = "qwen72b_v8_default"
        strict_cost = 0.0
        strict_decision = ""
        eyewear = is_eyewear_remove_target(source, score)
        if eyewear:
            if sample_id not in strict:
                raise ValueError(f"strict eyewear score missing: {sample_id}")
            strict_score = strict[sample_id]
            if strict_score.get("parse_status") != "pass":
                raise ValueError(f"strict eyewear score is invalid: {sample_id}")
            for field in RESULT_FIELDS:
                score[field] = strict_score.get(field, "")
            strict_cost = float(strict_score.get("cost_usd") or 0.0)
            strict_decision = strict_score.get("model_decision", "")
            route = "qwen72b_v9_eyewear_reconstruction"
            routed_eyewear.append(sample_id)

        quality_decision = score.get("model_decision", "")
        structural_reasons: list[str] = []
        if (
            not args.skip_structural_policy
            and source.get("structural_decision") == "rejected"
        ):
            structural_reasons = [
                reason.strip()
                for reason in source.get("structural_reasons", "").split("|")
                if reason.strip()
            ] or ["structural_policy_rejection"]
            reasons = json_list(score.get("reason_codes_json", ""))
            for reason in structural_reasons:
                policy_reason = f"upstream_{reason}"
                if policy_reason not in reasons:
                    reasons.append(policy_reason)
            score["reason_codes_json"] = json.dumps(
                reasons[:8], ensure_ascii=False, separators=(",", ":")
            )
            score["model_decision"] = "rejected"
            route = "upstream_structural_policy"
            structural_exclusions.append(sample_id)

        score.update(
            {
                "judge_version": ROUTED_JUDGE,
                "model_backend": "openrouter_plus_cpu_plus_structural_routing",
                "cost_usd": f"{base_cost + strict_cost:.9f}",
                "cost_source": "composed_base_plus_routed_calls",
                "base_v8_decision": base[sample_id].get("model_decision", ""),
                "strict_v9_decision": strict_decision,
                "stage2_quality_decision": quality_decision,
                "final_pipeline_decision": score.get("model_decision", ""),
                "filter_route": route,
                "eyewear_strict_route": "1" if eyewear else "0",
                "upstream_structural_decision": source.get(
                    "structural_decision", ""
                ),
                "upstream_structural_reasons": " | ".join(structural_reasons),
            }
        )
        output.append(score)

    extra_fields = [
        "base_v8_decision",
        "strict_v9_decision",
        "stage2_quality_decision",
        "final_pipeline_decision",
        "filter_route",
        "eyewear_strict_route",
        "upstream_structural_decision",
        "upstream_structural_reasons",
    ]
    fields = list(dict.fromkeys([*base_fields, *extra_fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    summary = {
        "judge_version": ROUTED_JUDGE,
        "rows": len(output),
        "base_judge": BASE_JUDGE,
        "strict_judge": STRICT_JUDGE,
        "quality_decisions": dict(
            Counter(row["stage2_quality_decision"] for row in output)
        ),
        "final_decisions": dict(
            Counter(row["final_pipeline_decision"] for row in output)
        ),
        "routes": dict(Counter(row["filter_route"] for row in output)),
        "routed_eyewear_ids": routed_eyewear,
        "structural_exclusion_ids": structural_exclusions,
        "total_composed_cost_usd": sum(
            float(row.get("cost_usd") or 0.0) for row in output
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
