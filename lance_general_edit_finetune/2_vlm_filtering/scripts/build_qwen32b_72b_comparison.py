#!/usr/bin/env python3
"""Merge frozen candidates with Qwen32B and Qwen72B outputs for blind review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_32B = RUN / "artifact_risk_30_qwen32b_scores.csv"
DEFAULT_72B = RUN / "artifact_risk_30_scores.csv"
DEFAULT_OUTPUT = RUN / "artifact_risk_30_qwen32b_vs_72b.csv"
DEFAULT_SUMMARY = RUN / "artifact_risk_30_qwen32b_vs_72b_summary.json"
VERSION = "qwen32b_vs_72b_artifact_risk_30_v1"
MODEL_IDENTITIES = {
    "qwen32b": ("qwen/qwen3-vl-32b-instruct", "qwen3_vl_32b_openrouter_pair_judge_v2_5"),
    "qwen72b": (
        "qwen/qwen2.5-vl-72b-instruct",
        "qwen2_5_vl_72b_openrouter_artifact_pair_judge_v1",
    ),
}
CHECKS = ("instruction_following", "visible_artifact", "content_preservation")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def parse_json(value: str, fallback: object) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def validate_scores(
    rows: list[dict[str, str]], prefix: str, expected_ids: set[str]
) -> dict[str, dict[str, str]]:
    model, judge = MODEL_IDENTITIES[prefix]
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != 30 or len(by_id) != 30 or set(by_id) != expected_ids:
        raise ValueError(f"{prefix} scores are incomplete or do not match the manifest")
    for sample_id, row in by_id.items():
        if row.get("parse_status") != "pass":
            raise ValueError(f"{prefix} parse failure: {sample_id}")
        if row.get("model_id") != model or row.get("judge_version") != judge:
            raise ValueError(f"{prefix} identity mismatch: {sample_id}")
    return by_id


def add_model_fields(
    output: dict[str, str], score: dict[str, str], prefix: str
) -> None:
    reasons = parse_json(score.get("reason_codes_json", ""), [])
    evidence = parse_json(score.get("evidence_json", ""), {})
    if not isinstance(reasons, list):
        reasons = []
    if not isinstance(evidence, dict):
        evidence = {}
    output[f"{prefix}_decision"] = score.get("model_decision", "")
    for check in CHECKS:
        output[f"{prefix}_{check}"] = score.get(check, "")
        output[f"{prefix}_{check}_evidence"] = str(evidence.get(check, ""))
    output[f"{prefix}_reason_codes"] = " | ".join(str(value) for value in reasons)
    output[f"{prefix}_provider"] = score.get("provider", "")
    output[f"{prefix}_cost_usd"] = score.get("cost_usd", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores-32b", type=Path, default=DEFAULT_32B)
    parser.add_argument("--scores-72b", type=Path, default=DEFAULT_72B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    manifest, fields = read_csv(args.manifest)
    rows32, _ = read_csv(args.scores_32b)
    rows72, _ = read_csv(args.scores_72b)
    if len(manifest) != 30 or len({row["sample_id"] for row in manifest}) != 30:
        raise ValueError("manifest must contain 30 unique rows")
    expected_ids = {row["sample_id"] for row in manifest}
    by32 = validate_scores(rows32, "qwen32b", expected_ids)
    by72 = validate_scores(rows72, "qwen72b", expected_ids)

    output = []
    for source in manifest:
        row = dict(source)
        row["risk_rank"] = source.get("sample_rank", "")
        row["survey_version"] = VERSION
        add_model_fields(row, by32[source["sample_id"]], "qwen32b")
        add_model_fields(row, by72[source["sample_id"]], "qwen72b")
        decision_same = row["qwen32b_decision"] == row["qwen72b_decision"]
        artifact_same = (
            row["qwen32b_visible_artifact"] == row["qwen72b_visible_artifact"]
        )
        row["decision_agreement"] = "1" if decision_same else "0"
        row["artifact_agreement"] = "1" if artifact_same else "0"
        if not decision_same:
            row["comparison_stratum"] = "decision_disagreement"
        elif not artifact_same:
            row["comparison_stratum"] = "artifact_disagreement"
        else:
            row["comparison_stratum"] = "full_decision_artifact_agreement"
        output.append(row)

    stratum_order = {
        "decision_disagreement": 0,
        "artifact_disagreement": 1,
        "full_decision_artifact_agreement": 2,
    }
    output.sort(
        key=lambda row: (
            stratum_order[row["comparison_stratum"]],
            int(row["risk_rank"]),
        )
    )
    for rank, row in enumerate(output, 1):
        row["sample_rank"] = str(rank)

    model_fields = []
    for prefix in MODEL_IDENTITIES:
        model_fields.extend(
            [
                f"{prefix}_decision",
                *[f"{prefix}_{check}" for check in CHECKS],
                f"{prefix}_reason_codes",
                *[f"{prefix}_{check}_evidence" for check in CHECKS],
                f"{prefix}_provider",
                f"{prefix}_cost_usd",
            ]
        )
    comparison_fields = [
        "risk_rank",
        "decision_agreement",
        "artifact_agreement",
        "comparison_stratum",
    ]
    output_fields = list(dict.fromkeys([*fields, *comparison_fields, *model_fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    matrix = Counter(
        f"32b={row['qwen32b_decision']}|72b={row['qwen72b_decision']}" for row in output
    )
    artifact_matrix = Counter(
        f"32b={row['qwen32b_visible_artifact']}|72b={row['qwen72b_visible_artifact']}"
        for row in output
    )
    summary = {
        "survey_version": VERSION,
        "rows": len(output),
        "same_input_prompt_layout": True,
        "qwen32b_decisions": dict(Counter(row["qwen32b_decision"] for row in output)),
        "qwen72b_decisions": dict(Counter(row["qwen72b_decision"] for row in output)),
        "decision_matrix": dict(matrix),
        "decision_agreement": {
            "count": sum(row["decision_agreement"] == "1" for row in output),
            "rate": sum(row["decision_agreement"] == "1" for row in output) / len(output),
        },
        "qwen32b_visible_artifact": dict(
            Counter(row["qwen32b_visible_artifact"] for row in output)
        ),
        "qwen72b_visible_artifact": dict(
            Counter(row["qwen72b_visible_artifact"] for row in output)
        ),
        "visible_artifact_matrix": dict(artifact_matrix),
        "visible_artifact_agreement": {
            "count": sum(row["artifact_agreement"] == "1" for row in output),
            "rate": sum(row["artifact_agreement"] == "1" for row in output) / len(output),
        },
        "comparison_strata": dict(Counter(row["comparison_stratum"] for row in output)),
        "qwen32b_cost_usd": sum(float(row["qwen32b_cost_usd"] or 0) for row in output),
        "qwen72b_cost_usd": sum(float(row["qwen72b_cost_usd"] or 0) for row in output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
