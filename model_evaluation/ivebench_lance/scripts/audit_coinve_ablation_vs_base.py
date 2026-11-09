#!/usr/bin/env python3
"""Audit the supplemental Base-vs-CoinVE/Mixed bidirectional blind comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from audit_coinve_ablation_completion import digest, read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/coinve_ablation_v1"
OUTPUT = ARTIFACT / "gemini_pairwise_vs_base_blind_v1"
PAIRS = ("base__vs__coinve4k_step677", "base__vs__mixed2k2k_step677")
STAGE_SAMPLES = {"short_subject": 91, "long_subject": 43, "attribute_all": 94}
EXPECTED_TASKS = 912
EXPECTED_GROUPS = 456


def main() -> int:
    errors: list[str] = []
    checks: dict[str, object] = {}
    tasks_path = ARTIFACT / "judge_tasks_pairwise_vs_base.json"
    tasks = read_json(tasks_path) if tasks_path.is_file() else []
    checks["tasks"] = len(tasks)
    if len(tasks) != EXPECTED_TASKS or len({digest(task) for task in tasks}) != EXPECTED_TASKS:
        errors.append("expected 912 unique supplemental pairwise tasks")
    stage_counts = Counter(task.get("stage") for task in tasks)
    pair_counts = Counter(task.get("model_pair") for task in tasks)
    checks["tasks_by_stage"] = dict(stage_counts)
    checks["tasks_by_pair"] = dict(pair_counts)
    if stage_counts != Counter({stage: count * 4 for stage, count in STAGE_SAMPLES.items()}):
        errors.append("supplemental task stage counts changed")
    if pair_counts != Counter({pair: 456 for pair in PAIRS}):
        errors.append("supplemental model-pair counts changed")

    media_hash_cache: dict[str, str] = {}
    media_errors: list[str] = []
    for task in tasks:
        if task.get("kind") != "pairwise":
            media_errors.append(f"non-pairwise task: {task.get('sample_id')}")
            continue
        if {task.get("candidate_a_model"), task.get("candidate_b_model")} != set(task.get("model_pair", "").split("__vs__")):
            media_errors.append(f"candidate models do not match pair: {task.get('sample_id')}")
        for field in ("source", "candidate_a", "candidate_b"):
            path_value = task.get(f"{field}_video")
            expected_hash = task.get(f"{field}_sha256")
            if not isinstance(path_value, str) or not isinstance(expected_hash, str):
                media_errors.append(f"missing {field} identity: {task.get('sample_id')}")
                continue
            media_path = Path(path_value)
            if not media_path.is_file():
                media_errors.append(f"missing {field} media: {media_path}")
                continue
            if path_value not in media_hash_cache:
                media_hash_cache[path_value] = sha256_file(media_path)
            if media_hash_cache[path_value] != expected_hash:
                media_errors.append(f"{field} media hash mismatch: {task.get('sample_id')}")
    checks["unique_task_media"] = len(media_hash_cache)
    checks["task_media_errors"] = media_errors
    if len(media_hash_cache) != 912:
        errors.append(f"expected 912 unique supplemental task media files, found {len(media_hash_cache)}")
    errors.extend(media_errors)

    orders: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task in tasks:
        orders[(task.get("sample_id"), task.get("model_pair"))].append(task)
    malformed_orders = 0
    for values in orders.values():
        values.sort(key=lambda task: task.get("order"))
        if len(values) != 2 or [task.get("order") for task in values] != [1, 2]:
            malformed_orders += 1
            continue
        first, second = values
        if not (
            first.get("candidate_a_model") == second.get("candidate_b_model")
            and first.get("candidate_b_model") == second.get("candidate_a_model")
            and first.get("candidate_a_sha256") == second.get("candidate_b_sha256")
            and first.get("candidate_b_sha256") == second.get("candidate_a_sha256")
        ):
            malformed_orders += 1
    checks["sample_pair_groups"] = len(orders)
    checks["malformed_bidirectional_groups"] = malformed_orders
    if len(orders) != EXPECTED_GROUPS or malformed_orders:
        errors.append("supplemental bidirectional A/B symmetry failed")

    summary_path = OUTPUT / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    checks["judge_summary"] = {
        key: summary.get(key) for key in ("status", "tasks_requested", "tasks_succeeded", "tasks_failed")
    }
    if not (
        summary.get("status") == "complete"
        and summary.get("tasks_requested") == EXPECTED_TASKS
        and summary.get("tasks_succeeded") == EXPECTED_TASKS
        and summary.get("tasks_failed") == 0
    ):
        errors.append("supplemental judge summary is not complete")

    result_paths = list((OUTPUT / "records").glob("*/result.json"))
    results = [read_json(path) for path in result_paths]
    successful_tasks = [result.get("identity", {}).get("task") for result in results if result.get("status") == "success"]
    checks["successful_result_records"] = len(successful_tasks)
    if len(successful_tasks) != EXPECTED_TASKS or {digest(task) for task in successful_tasks} != {digest(task) for task in tasks}:
        errors.append("supplemental result identities do not match the task manifest")
    route_errors = sum(
        result.get("returned_model") not in ("google/gemini-3.8-flash", "google/gemini-3.8-flash-20260902")
        or result.get("returned_provider") != "Google AI Studio"
        for result in results if result.get("status") == "success"
    )
    checks["result_route_errors"] = route_errors
    if route_errors:
        errors.append("one or more supplemental results used an unexpected route")

    score_rows = []
    scores_path = OUTPUT / "scores.csv"
    if scores_path.is_file():
        with scores_path.open(newline="", encoding="utf-8") as handle:
            score_rows = list(csv.DictReader(handle))
    checks["score_rows"] = len(score_rows)
    if len(score_rows) != EXPECTED_TASKS:
        errors.append("supplemental scores.csv does not contain 912 rows")

    report_path = ARTIFACT / "pairwise_vs_base_report.json"
    markdown_path = ARTIFACT / "pairwise_vs_base_report.md"
    report = read_json(report_path) if report_path.is_file() else {}
    checks["report"] = {
        "status": report.get("status"), "tasks": report.get("tasks"),
        "bidirectional_groups": report.get("bidirectional_groups"),
        "markdown_exists": markdown_path.is_file(),
    }
    if not (
        report.get("status") == "complete"
        and report.get("tasks") == EXPECTED_TASKS
        and report.get("bidirectional_groups") == EXPECTED_GROUPS
        and summary_path.is_file()
        and report.get("source_summary_sha256") == sha256_file(summary_path)
        and markdown_path.is_file() and markdown_path.stat().st_size > 0
    ):
        errors.append("supplemental result report is missing or stale")

    audit = {"status": "complete" if not errors else "incomplete", "errors": errors, "checks": checks}
    write_json(ARTIFACT / "completion_audit_vs_base.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
