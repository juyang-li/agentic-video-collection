#!/usr/bin/env python3
"""Merge the 30-pair risk set with Qwen72B reasons for the offline reviewer."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_SCORES = RUN / "artifact_risk_30_scores.csv"
DEFAULT_OUTPUT = RUN / "artifact_risk_30_with_qwen72b.csv"
DEFAULT_SUMMARY = RUN / "artifact_risk_30_with_qwen72b_summary.json"
EXPECTED_JUDGE = "qwen2_5_vl_72b_openrouter_artifact_pair_judge_v1"
EXPECTED_MODEL = "qwen/qwen2.5-vl-72b-instruct"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def safe_json(value: str, fallback: object) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-size", type=int, default=30)
    parser.add_argument("--expected-judge", default=EXPECTED_JUDGE)
    args = parser.parse_args()

    manifest, fields = read_csv(args.manifest)
    scores, _ = read_csv(args.scores)
    if len(manifest) != args.expected_size or len(scores) != args.expected_size:
        raise ValueError(
            f"expected {args.expected_size} manifest and score rows, "
            f"got {len(manifest)}, {len(scores)}"
        )
    scores_by_id = {row["sample_id"]: row for row in scores}
    if len(scores_by_id) != len(scores):
        raise ValueError("scores contain duplicate sample_id")
    if set(scores_by_id) != {row["sample_id"] for row in manifest}:
        raise ValueError("manifest and score sample IDs do not match")

    output: list[dict[str, str]] = []
    for row in manifest:
        score = scores_by_id[row["sample_id"]]
        if score.get("parse_status") != "pass":
            raise ValueError(f"Qwen72B score is not parseable: {row['sample_id']}")
        if score.get("judge_version") != args.expected_judge:
            raise ValueError(f"unexpected judge version: {score.get('judge_version')}")
        if score.get("model_id") != EXPECTED_MODEL:
            raise ValueError(f"unexpected model ID: {score.get('model_id')}")
        reasons = safe_json(score.get("reason_codes_json", ""), [])
        evidence = safe_json(score.get("evidence_json", ""), {})
        if not isinstance(reasons, list):
            reasons = []
        if not isinstance(evidence, dict):
            evidence = {}
        merged = dict(row)
        merged.update(
            {
                "qwen72b_decision": score.get("model_decision", ""),
                "qwen72b_instruction_following": score.get(
                    "instruction_following", ""
                ),
                "qwen72b_visible_artifact": score.get("visible_artifact", ""),
                "qwen72b_content_preservation": score.get(
                    "content_preservation", ""
                ),
                "qwen72b_reason_codes": " | ".join(str(value) for value in reasons),
                "qwen72b_instruction_evidence": str(
                    evidence.get("instruction_following", "")
                ),
                "qwen72b_artifact_evidence": str(
                    evidence.get("visible_artifact", "")
                ),
                "qwen72b_preservation_evidence": str(
                    evidence.get("content_preservation", "")
                ),
                "qwen72b_provider": score.get("provider", ""),
                "qwen72b_cost_usd": score.get("cost_usd", ""),
            }
        )
        output.append(merged)

    extra = [
        "qwen72b_decision",
        "qwen72b_instruction_following",
        "qwen72b_visible_artifact",
        "qwen72b_content_preservation",
        "qwen72b_reason_codes",
        "qwen72b_instruction_evidence",
        "qwen72b_artifact_evidence",
        "qwen72b_preservation_evidence",
        "qwen72b_provider",
        "qwen72b_cost_usd",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fields, *extra], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)

    summary = {
        "survey_version": manifest[0].get("survey_version", ""),
        "rows": len(output),
        "model": EXPECTED_MODEL,
        "judge_version": args.expected_judge,
        "decisions": dict(Counter(row["qwen72b_decision"] for row in output)),
        "instruction_following": dict(
            Counter(row["qwen72b_instruction_following"] for row in output)
        ),
        "visible_artifact": dict(
            Counter(row["qwen72b_visible_artifact"] for row in output)
        ),
        "content_preservation": dict(
            Counter(row["qwen72b_content_preservation"] for row in output)
        ),
        "providers": dict(Counter(row["qwen72b_provider"] for row in output)),
        "total_cost_usd": sum(float(row["qwen72b_cost_usd"] or 0) for row in output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
