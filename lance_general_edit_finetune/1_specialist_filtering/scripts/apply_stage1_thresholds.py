#!/usr/bin/env python3
"""Apply frozen Stage 1.1 thresholds to edited-side video scores and aggregate pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from calibrate_stage1_thresholds import METRICS, decision
from run_stage1_quality import required_roles, resolve_local_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/survey_manifest_reco_v3.csv"
DEFAULT_SCORES = ROOT / "artifacts/filter_run_v1/quality_calibration_scores.csv"
DEFAULT_TECHNICAL_SCORES: Path | None = None
DEFAULT_THRESHOLDS = ROOT / "artifacts/filter_run_v1/stage1_1_thresholds.json"
DEFAULT_VIDEOS = ROOT / "artifacts/filter_run_v1/stage1_1_calibration_video_decisions.csv"
DEFAULT_PAIRS = ROOT / "artifacts/filter_run_v1/stage1_1_calibration_pair_decisions.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/filter_run_v1/stage1_1_calibration_decision_summary.json"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--technical-scores", type=Path, default=DEFAULT_TECHNICAL_SCORES)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--finalize-missing",
        action="store_true",
        help="Reject absent local media while leaving existing-but-unscored videos pending.",
    )
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    scores = read_csv(args.scores)
    technical_scores = read_csv(args.technical_scores) if args.technical_scores else []
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["metrics"]
    score_by_video = {row["video_id"]: row for row in scores}
    technical_by_video = {row["video_id"]: row for row in technical_scores}

    video_rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    video_cache: dict[str, dict[str, str]] = {}
    for pair in manifest:
        technical_reasons: list[str] = []
        technical_pending = False
        if technical_by_video:
            for role in ("source", "target"):
                video_id = pair.get(f"{role}_video_id", f"{pair['pair_id']}:{role}")
                technical = technical_by_video.get(video_id)
                if technical is None:
                    local_path = pair.get(f"{role}_local", "")
                    if (
                        args.finalize_missing
                        and (not local_path or not resolve_local_path(local_path).is_file())
                    ):
                        technical_reasons.append(f"{role}:media_missing")
                    else:
                        technical_pending = True
                elif technical["technical_status"] != "pass":
                    technical_reasons.append(
                        f"{role}:technical_{technical['technical_status']}"
                    )
        role_results: list[dict[str, str]] = []
        for role in required_roles(pair):
            video_id = pair.get(f"{role}_video_id", f"{pair['pair_id']}:{role}")
            if video_id not in video_cache:
                score = score_by_video.get(video_id)
                if score is None:
                    local_path = pair.get(f"{role}_local", "")
                    missing_media = args.finalize_missing and (
                        not local_path or not resolve_local_path(local_path).is_file()
                    )
                    result = {
                        "video_id": video_id,
                        "sample_id": pair["sample_id"],
                        "dataset": pair["dataset"],
                        "branch": pair["branch"],
                        "pair_id": pair["pair_id"],
                        "role": role,
                        "stage1_1_decision": "rejected" if missing_media else "pending",
                        "stage1_1_reasons": "media_missing" if missing_media else "missing_scores",
                    }
                elif score["technical_status"] != "pass":
                    result = {
                        "video_id": video_id,
                        "sample_id": pair["sample_id"],
                        "dataset": pair["dataset"],
                        "branch": pair["branch"],
                        "pair_id": pair["pair_id"],
                        "role": role,
                        "stage1_1_decision": "rejected",
                        "stage1_1_reasons": f"technical_{score['technical_status']}",
                    }
                else:
                    values = {name: float(score[name]) for name in METRICS}
                    automatic, reasons = decision(values, thresholds)
                    result = {
                        "video_id": video_id,
                        "sample_id": pair["sample_id"],
                        "dataset": pair["dataset"],
                        "branch": pair["branch"],
                        "pair_id": pair["pair_id"],
                        "role": role,
                        "stage1_1_decision": automatic,
                        "stage1_1_reasons": "|".join(reasons),
                    }
                video_cache[video_id] = result
                video_rows.append(result)
            role_results.append(video_cache[video_id])

        decisions = {row["stage1_1_decision"] for row in role_results}
        if technical_reasons or "rejected" in decisions:
            pair_decision = "rejected"
        elif technical_pending or "pending" in decisions:
            pair_decision = "pending"
        elif "uncertain" in decisions:
            pair_decision = "uncertain"
        else:
            pair_decision = "accepted"
        pair_reasons = technical_reasons + [
            f"{row['role']}:{reason}"
            for row in role_results
            for reason in row["stage1_1_reasons"].split("|")
            if reason
        ]
        pair_reasons = list(dict.fromkeys(pair_reasons))
        pair_rows.append(
            {
                **pair,
                "stage1_1_decision": pair_decision,
                "stage1_1_reasons": "|".join(pair_reasons),
                "stage1_1_scored_roles": ",".join(required_roles(pair)),
                "stage1_technical_status": (
                    "rejected" if technical_reasons else "pending" if technical_pending else "pass"
                ),
            }
        )

    video_fields = [
        "video_id",
        "sample_id",
        "dataset",
        "branch",
        "pair_id",
        "role",
        "stage1_1_decision",
        "stage1_1_reasons",
    ]
    pair_fields = list(manifest[0]) + [
        "stage1_1_decision",
        "stage1_1_reasons",
        "stage1_1_scored_roles",
        "stage1_technical_status",
    ]
    write_csv(args.video_output, video_rows, video_fields)
    write_csv(args.pair_output, pair_rows, pair_fields)
    summary = {
        "filter_version": "general_edit_stage1_pilot_v1",
        "videos": len(video_rows),
        "pairs": len(pair_rows),
        "video_decisions": dict(Counter(row["stage1_1_decision"] for row in video_rows)),
        "pair_decisions": dict(Counter(row["stage1_1_decision"] for row in pair_rows)),
        "pair_decisions_by_branch": {
            branch: dict(
                Counter(
                    row["stage1_1_decision"] for row in pair_rows if row["branch"] == branch
                )
            )
            for branch in sorted({row["branch"] for row in pair_rows})
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
