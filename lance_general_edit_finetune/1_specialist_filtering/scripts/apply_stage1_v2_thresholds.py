#!/usr/bin/env python3
"""Apply frozen Stage-1 v2 quality gates and aggregate video decisions to pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from calibrate_stage1_thresholds import METRICS, decision as base_decision
from run_stage1_quality import required_roles, resolve_local_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v3"
DEFAULT_THRESHOLDS = ROOT / "artifacts/filter_run_v1/stage1_1_thresholds.json"
DEFAULT_POLICY = ROOT / "stage1_1_future_policy.json"
VERSION = "general_edit_stage1_quality_v2"


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


def video_id(pair: dict[str, str], role: str) -> str:
    return pair.get(f"{role}_video_id") or f"{pair['pair_id']}:{role}"


def hard_v2_reasons(
    pair: dict[str, str], role: str, score: dict[str, str], policy: dict[str, object]
) -> list[str]:
    reasons: list[str] = []
    near_freeze = policy["near_freeze"]
    if (
        float(score["frame_diff_mean"]) <= float(near_freeze["frame_diff_mean_max"])
        and float(score["frame_diff_p95"]) <= float(near_freeze["frame_diff_p95_max"])
    ):
        reasons.append("sustained_near_freeze")

    # Some official datasets publish an add training pair by reversing a
    # removal-generated clip (notably GOKU add).  The generated source still
    # needs the same strict exposed-background/body clarity gate as a normal
    # removal target.
    removal_generated_side = pair.get("operation") == "remove" or (
        pair.get("operation") == "add"
        and pair.get("source_provenance") == "edited"
        and pair.get("target_provenance") == "original"
    )
    if removal_generated_side:
        clarity = policy["remove_edited_side_clarity_hard_gate"]["thresholds"]
        if float(score["musiq_p10"]) < float(clarity["musiq_p10_min"]):
            reasons.append("global_low_visual_quality")
        if float(score["sharpness_p10"]) < float(clarity["sharpness_p10_min"]):
            reasons.append("global_blur_low_detail")
    return [f"{role}:{reason}" for reason in reasons]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RUN / "stage1_0_survivors.csv")
    parser.add_argument("--scores", type=Path, default=DEFAULT_RUN / "stage1_quality_scores.csv")
    parser.add_argument("--technical-scores", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--video-output", type=Path, default=DEFAULT_RUN / "stage1_1_video_decisions.csv")
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_RUN / "stage1_1_pair_decisions.csv")
    parser.add_argument("--survivors", type=Path, default=DEFAULT_RUN / "stage1_survivors.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "stage1_1_summary.json")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optionally restrict the run to one or more datasets.",
    )
    args = parser.parse_args()

    pairs = read_csv(args.manifest)
    if args.dataset:
        selected_datasets = set(args.dataset)
        pairs = [pair for pair in pairs if pair.get("dataset") in selected_datasets]
    if not pairs:
        raise ValueError("no pairs remain after dataset restriction")
    scores = {row["video_id"]: row for row in read_csv(args.scores)}
    technical = {row["video_id"]: row for row in read_csv(args.technical_scores)}
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["metrics"]
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    video_rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    cache: dict[str, dict[str, str]] = {}
    for pair in pairs:
        technical_reasons: list[str] = []
        for role in ("source", "target"):
            identifier = video_id(pair, role)
            row = technical.get(identifier)
            local_value = pair.get(f"{role}_local", "")
            if not local_value or not resolve_local_path(local_value).is_file():
                technical_reasons.append(f"{role}:media_missing")
            elif row is None:
                technical_reasons.append(f"{role}:technical_score_missing")
            elif row.get("technical_status") != "pass":
                technical_reasons.append(f"{role}:technical_{row.get('technical_status', 'missing')}")

        role_results: list[dict[str, str]] = []
        for role in required_roles(pair):
            identifier = video_id(pair, role)
            if identifier not in cache:
                score = scores.get(identifier)
                reasons: list[str] = []
                automatic = "rejected"
                if score is None:
                    reasons.append(f"{role}:full_score_missing")
                elif score.get("technical_status") != "pass":
                    reasons.append(f"{role}:technical_{score.get('technical_status', 'missing')}")
                elif not all(score.get(name) for name in METRICS):
                    reasons.append(f"{role}:full_metric_missing")
                else:
                    values = {name: float(score[name]) for name in METRICS}
                    automatic, base_reasons = base_decision(values, thresholds)
                    reasons.extend(f"{role}:{reason}" for reason in base_reasons)
                    v2 = hard_v2_reasons(pair, role, score, policy)
                    if v2:
                        automatic = "rejected"
                        reasons.extend(v2)
                result = {
                    "filter_version": VERSION,
                    "video_id": identifier,
                    "sample_id": pair["sample_id"],
                    "dataset": pair["dataset"],
                    "branch": pair["branch"],
                    "pair_id": pair["pair_id"],
                    "role": role,
                    "stage1_1_decision": automatic,
                    "stage1_1_reasons": "|".join(dict.fromkeys(reasons)),
                }
                cache[identifier] = result
                video_rows.append(result)
            role_results.append(cache[identifier])

        decisions = {row["stage1_1_decision"] for row in role_results}
        if technical_reasons or "rejected" in decisions:
            pair_decision = "rejected"
        elif "uncertain" in decisions:
            pair_decision = "uncertain"
        else:
            pair_decision = "accepted"
        reasons = technical_reasons + [
            reason
            for result in role_results
            for reason in result["stage1_1_reasons"].split("|")
            if reason
        ]
        pair_rows.append(
            {
                **pair,
                "stage1_1_decision": pair_decision,
                "stage1_1_reasons": "|".join(dict.fromkeys(reasons)),
                "stage1_1_scored_roles": ",".join(required_roles(pair)),
                "stage1_technical_status": "rejected" if technical_reasons else "pass",
            }
        )

    video_fields = [
        "filter_version", "video_id", "sample_id", "dataset", "branch", "pair_id",
        "role", "stage1_1_decision", "stage1_1_reasons",
    ]
    pair_fields = list(pairs[0]) + [
        "stage1_1_decision", "stage1_1_reasons", "stage1_1_scored_roles",
        "stage1_technical_status",
    ]
    survivors = [row for row in pair_rows if row["stage1_1_decision"] in {"accepted", "uncertain"}]
    write_csv(args.video_output, video_rows, video_fields)
    write_csv(args.pair_output, pair_rows, pair_fields)
    write_csv(args.survivors, survivors, pair_fields)
    summary = {
        "filter_version": VERSION,
        "policy_revision": "v2_reverse_removal_generated_side_20260905",
        "input_pairs": len(pairs),
        "survivors": len(survivors),
        "pair_decisions": dict(Counter(row["stage1_1_decision"] for row in pair_rows)),
        "pair_decisions_by_branch": {
            branch: dict(Counter(row["stage1_1_decision"] for row in pair_rows if row["branch"] == branch))
            for branch in sorted({row["branch"] for row in pair_rows})
        },
        "rejection_reasons": dict(
            Counter(
                reason.split(":", 1)[-1]
                for row in pair_rows
                if row["stage1_1_decision"] == "rejected"
                for reason in row["stage1_1_reasons"].split("|")
                if reason
            )
        ),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
