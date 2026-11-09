#!/usr/bin/env python3
"""Freeze the 300-pair, remove-heavy binary Stage-2 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import build_v12_holdout40 as prior
import run_vefx_reward as routed


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE1 = PROJECT / "1_specialist_filtering"
DEFAULT_INPUT = STAGE1 / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_RUN = ROOT / "artifacts/systematic_binary_benchmark_v1"
DEFAULT_OUTPUT = DEFAULT_RUN / "benchmark_manifest_300.csv"
DEFAULT_REVIEW = DEFAULT_RUN / "review_manifest_300.csv"
DEFAULT_SUMMARY = DEFAULT_RUN / "sampling_summary.json"
VERSION = "stage2_systematic_binary_benchmark_v1"
QUOTAS = {
    "goku_add_inverse_remove": 50,
    "goku_remove": 50,
    "reco_remove": 100,
    "reco_add": 40,
    "reco_replace": 45,
    "egoedit_replace": 15,
}
UNRESOLVED = "generation_side_unresolved"
RECO_PROVENANCE_BASIS = (
    "official_reco_augmentation_rules; released_pair_config_has_training_roles_"
    "but_no_per_pair_generation_lineage"
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stratum(row: dict[str, str]) -> str:
    branch = row.get("branch", "")
    if branch == "goku_add":
        return "goku_add_inverse_remove"
    if branch in {"goku_remove", "reco_remove", "reco_add", "reco_replace", "egoedit_replace"}:
        return branch
    raise ValueError(f"unsupported branch: {branch!r}")


def construction_aware_row(source: dict[str, str]) -> dict[str, str]:
    """Correct training-role metadata that does not establish generation lineage."""
    row = dict(source)
    if row.get("dataset") == "reco":
        row.setdefault("declared_source_provenance", row.get("source_provenance", ""))
        row.setdefault("declared_target_provenance", row.get("target_provenance", ""))
        row.setdefault("declared_provenance_basis", row.get("provenance_basis", ""))
        row.update({
            "source_provenance": UNRESOLVED,
            "target_provenance": UNRESOLVED,
            "provenance_basis": RECO_PROVENANCE_BASIS,
            "quality_checked_roles": "both",
        })
    return row


def normalized_row(source: dict[str, str], rank: int) -> dict[str, str]:
    row = construction_aware_row(source)
    audit = routed.route(row)
    operation = row.get("operation", "").lower()
    audit_operation = (
        {"add": "remove", "remove": "add", "replace": "replace"}.get(
            operation, operation
        )
        if audit["audit_relation"] == "inverse"
        else operation
    )
    row.update({
        "benchmark_version": VERSION,
        "benchmark_rank": str(rank),
        "benchmark_stratum": stratum(row),
        "audit_operation_expected": audit_operation,
        "generated_role": audit["generated_side"].lower(),
        "audit_relation": audit["audit_relation"],
        "reference_training_role": audit["reference_training_role"],
        "generated_training_role": audit["generated_training_role"],
        "audit_left_training_role": audit["audit_left_training_role"],
        "audit_right_training_role": audit["audit_right_training_role"],
        "artifact_checked_roles": audit["artifact_checked_roles"],
        "instruction_following_direction": audit["instruction_following_direction"],
        "content_preservation_direction": audit["content_preservation_direction"],
        "audit_instruction": audit["audit_instruction"],
        "instruction_transform": audit["instruction_transform"],
    })
    return row


def review_row(source: dict[str, str]) -> dict[str, str]:
    left_role = source["audit_left_training_role"]
    right_role = source["audit_right_training_role"]
    unresolved = source["generated_role"] == "unresolved"
    both_generated = source["generated_role"] == "both"
    provenance_display = (
        "生成侧未解析；instruction following 与 content preservation 按 SRC→TARGET；"
        "visible artifact 检查两侧"
        if unresolved
        else (
            "两侧均为生成视频；按训练 SRC→TARGET；visible artifact 检查两侧"
            if both_generated
            else "已按 provenance 重排为 REFERENCE→GENERATED；visible artifact 检查生成侧"
        )
    )
    repository = source.get("repo_id", "")
    revision = source.get("revision", "")
    source_attribution = f"{source['dataset']} · {source['branch']}"
    if repository:
        source_attribution += f" · {repository}"
    if revision:
        source_attribution += f" @ {revision}"
    row = dict(source)
    row.update({
        "survey_version": VERSION,
        "sample_rank": source["benchmark_rank"],
        "review_mode": "binary_accept_reject",
        "training_operation": source["operation"],
        "training_instruction": source["instruction"],
        "training_source_local": source["source_local"],
        "training_target_local": source["target_local"],
        "operation": source["audit_operation_expected"],
        "instruction": source["audit_instruction"],
        "source_local": source[f"{left_role}_local"],
        "target_local": source[f"{right_role}_local"],
        "source_video_id": source.get(f"{left_role}_video_id")
        or f"{source['sample_id']}:{left_role}",
        "target_video_id": source.get(f"{right_role}_video_id")
        or f"{source['sample_id']}:{right_role}",
        "display_source_label": (
            "SRC_VIDEO / GENERATION SIDE UNRESOLVED"
            if unresolved
            else (
                "SRC / TRAIN_SOURCE_GENERATED"
                if both_generated
                else "SRC / REFERENCE_VIDEO"
            )
        ),
        "display_target_label": (
            "TARGET_VIDEO / GENERATION SIDE UNRESOLVED"
            if unresolved
            else (
                "TARGET / TRAIN_TARGET_GENERATED"
                if both_generated
                else "TARGET / GENERATED_VIDEO"
            )
        ),
        "source_attribution": source_attribution,
        "provenance_display": provenance_display,
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--reroute-frozen",
        action="store_true",
        help="Keep the frozen sample IDs/order and only correct routing metadata.",
    )
    args = parser.parse_args()

    if args.reroute_frozen:
        if not args.output.is_file() or not args.summary.is_file():
            raise FileNotFoundError("frozen benchmark/summary is required for rerouting")
        rows, input_fields = read_csv(args.output)
        if len(rows) != 300 or len({row["sample_id"] for row in rows}) != 300:
            raise ValueError("frozen benchmark must contain 300 unique rows")
        expected_ranks = [str(index) for index in range(1, 301)]
        if [row["benchmark_rank"] for row in rows] != expected_ranks:
            raise ValueError("frozen benchmark rank/order changed")
        benchmark = [normalized_row(row, int(row["benchmark_rank"])) for row in rows]
        review = [review_row(row) for row in benchmark]
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    else:
        for path in (args.output, args.review_output, args.summary):
            if path.exists():
                raise FileExistsError(
                    f"frozen v1 artifact already exists and must not be regenerated: {path}"
                )
        rows, input_fields = read_csv(args.input)
        summary = {}
    if args.reroute_frozen:
        prior_ids: set[str] = set()
        prior_sources: list[str] = []
        candidates: list[dict[str, str]] = []
        pools: dict[str, list[dict[str, str]]] = {key: [] for key in QUOTAS}
    else:
        prior_ids, prior_sources = prior.prior_review_ids()
        candidates = [
            row
            for row in rows
            if row["sample_id"] not in prior_ids and prior.media_ready(row)
        ]
        pools = {key: [] for key in QUOTAS}
        for row in candidates:
            pools[stratum(row)].append(row)
        rng = random.Random(args.seed)
        selected: list[dict[str, str]] = []
        for key, quota in QUOTAS.items():
            if len(pools[key]) < quota:
                raise ValueError(f"stratum {key} has only {len(pools[key])}, needs {quota}")
            selected.extend(rng.sample(pools[key], quota))
        if len(selected) != 300 or len({row["sample_id"] for row in selected}) != 300:
            raise ValueError("selection must contain exactly 300 unique sample IDs")
        rng.shuffle(selected)
        benchmark = [normalized_row(row, rank) for rank, row in enumerate(selected, 1)]
        review = [review_row(row) for row in benchmark]

    prefix = [
        "benchmark_version", "benchmark_rank", "benchmark_stratum",
        "audit_operation_expected", "generated_role", "audit_relation",
        "reference_training_role", "generated_training_role", "audit_instruction",
        "audit_left_training_role", "audit_right_training_role",
        "artifact_checked_roles", "instruction_following_direction",
        "content_preservation_direction", "instruction_transform",
        "declared_source_provenance", "declared_target_provenance",
        "declared_provenance_basis",
    ]
    benchmark_fields = list(dict.fromkeys([*prefix, *input_fields]))
    review_prefix = [
        "survey_version", "sample_rank", "review_mode", "benchmark_stratum",
        "sample_id", "dataset", "branch", "operation", "instruction",
        "source_attribution", "provenance_display", "provenance_basis",
        "display_source_label", "display_target_label", "source_video_id",
        "target_video_id", "source_local", "target_local", "generated_role",
        "audit_relation", "training_operation", "training_instruction",
        "training_source_local", "training_target_local",
    ]
    review_fields = list(dict.fromkeys([*review_prefix, *benchmark_fields]))
    write_csv(args.output, benchmark, benchmark_fields)
    write_csv(args.review_output, review, review_fields)

    audit_counts = Counter(row["audit_operation_expected"] for row in benchmark)
    summary.update({
        "benchmark_version": VERSION,
        "seed": summary.get("seed", args.seed),
        "input_rows": summary.get("input_rows", len(rows)),
        "prior_touched_ids_excluded": summary.get("prior_touched_ids_excluded", len(prior_ids)),
        "prior_source_files": summary.get("prior_source_files", len(prior_sources)),
        "unseen_media_ready_candidates": summary.get("unseen_media_ready_candidates", len(candidates)),
        "available_by_stratum": summary.get(
            "available_by_stratum", {key: len(pools[key]) for key in QUOTAS}
        ),
        "quota_by_stratum": QUOTAS,
        "selected_rows": len(benchmark),
        "selected_audit_operations": dict(audit_counts),
        "remove_fraction": audit_counts["remove"] / len(benchmark),
        "selected_datasets": dict(Counter(row["dataset"] for row in benchmark)),
        "selected_branches": dict(Counter(row["branch"] for row in benchmark)),
        "selected_provenance": dict(Counter(
            f"{row['source_provenance']}->{row['target_provenance']}"
            for row in benchmark
        )),
        "selected_generated_roles": dict(Counter(row["generated_role"] for row in benchmark)),
        "selected_audit_relations": dict(Counter(row["audit_relation"] for row in benchmark)),
        "routing_revision": "generation_side_unresolved_v1",
        "generation_side_unresolved_policy": {
            "instruction_following": "source_to_target",
            "visible_artifact": "check_source_and_target",
            "content_preservation": "source_to_target",
        },
        "selected_stage1_1_decisions": dict(Counter(
            row["stage1_1_decision"] for row in benchmark
        )),
        "overlap_with_prior_touched": len(
            {row["sample_id"] for row in benchmark} & prior_ids
        ),
        "review_label_space": ["accepted", "rejected", "uncertain"],
        "scored_label_space": ["accepted", "rejected"],
        "uncertain_policy": "preserve in raw review; exclude from scored benchmark",
        "sample_ids": [row["sample_id"] for row in benchmark],
    })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
