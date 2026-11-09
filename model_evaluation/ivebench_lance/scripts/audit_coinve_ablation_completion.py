#!/usr/bin/env python3
"""Audit the completed three-model, bidirectional IVEBench blind comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_COLLECTION = ROOT.parents[1]
FINETUNE = VIDEO_COLLECTION / "lance_general_edit_finetune/3_lance_lora_finetuning"
ARTIFACT = ROOT / "artifacts/coinve_ablation_v1"
OUTPUT = ARTIFACT / "gemini_pairwise_blind_v1"
MODELS = ("single4k_step677", "coinve4k_step677", "mixed2k2k_step677")
PAIRS = (
    "single4k_step677__vs__coinve4k_step677",
    "single4k_step677__vs__mixed2k2k_step677",
    "coinve4k_step677__vs__mixed2k2k_step677",
)
STAGE_SAMPLES = {"short_subject": 91, "long_subject": 43, "attribute_all": 94}
RUNS = {
    "coinve4k_step677": FINETUNE / "runs/lance_coinve_4k_r64_seed42_e2_v1",
    "mixed2k2k_step677": FINETUNE / "runs/lance_mixed_2k2k_r64_seed42_e2_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_complete(run: Path, step: int) -> bool:
    checkpoint = run / f"checkpoints/{step:07d}"
    marker_path = checkpoint / "checkpoint_complete.json"
    adapter_path = checkpoint / "adapter.safetensors"
    if not marker_path.is_file() or not adapter_path.is_file():
        return False
    marker = read_json(marker_path)
    return bool(
        marker.get("status") == "complete"
        and marker.get("completed_steps") == step
        and marker.get("world_size") == 4
        and marker.get("required_artifact_sizes", {}).get("adapter.safetensors") == adapter_path.stat().st_size
        and adapter_path.stat().st_size > 0
    )


def main() -> int:
    errors: list[str] = []
    checks: dict[str, object] = {}

    checkpoints = {
        model: {str(step): checkpoint_complete(run, step) for step in (451, 677, 902)}
        for model, run in RUNS.items()
    }
    checks["checkpoints"] = checkpoints
    if not all(value for model in checkpoints.values() for value in model.values()):
        errors.append("one or more required checkpoints are incomplete")

    plan_path = ARTIFACT / "generation_plan.json"
    if not plan_path.is_file():
        errors.append("generation plan is missing")
        plan = {}
    else:
        plan = read_json(plan_path)
        if plan.get("schema_version") != "ivebench_coinve_ablation_generation_v1":
            errors.append("generation plan schema changed")
        if plan.get("generation_seed") != 42 or set(plan.get("models", {})) != set(RUNS):
            errors.append("generation plan seed or model set changed")
    jobs = [job for group in plan.get("groups", []) for job in group.get("jobs", [])]
    job_counts = Counter(job.get("model_kind") for job in jobs)
    checks["generation_jobs"] = {"groups": len(plan.get("groups", [])), "jobs": len(jobs), "by_model": dict(job_counts)}
    if len(plan.get("groups", [])) != 3 or len(jobs) != 24 or job_counts != Counter({
        "coinve4k_step677": 12, "mixed2k2k_step677": 12,
    }):
        errors.append("generation group or job counts changed")
    missing_outputs = []
    for job in jobs:
        config_path = Path(job["config_path"])
        output = ROOT / "artifacts" / job["stage"] / "outputs_raw" / job["run_name"]
        if not config_path.is_file():
            missing_outputs.append(f"missing config: {config_path}")
            continue
        if sha256_file(config_path) != job.get("config_sha256"):
            missing_outputs.append(f"config hash changed: {job['run_name']}")
        expected = set(read_json(config_path))
        actual = {path.stem for path in output.glob("*.mp4")}
        if actual != expected:
            missing_outputs.append(f"{job['run_name']}: expected {len(expected)} outputs, found {len(actual)}")
        provenance_path = output / "generation_provenance.json"
        if not provenance_path.is_file():
            missing_outputs.append(f"missing provenance: {job['run_name']}")
        else:
            expected_provenance = {
                "benchmark": "IVEBench",
                "scope": job["stage"],
                "phase": job["phase"],
                "model_kind": job["model_kind"],
                "seed": 42,
                "base_model_sha256": plan.get("base_model_sha256"),
                "adapter_sha256": job["adapter_sha256"],
                "config_sha256": job["config_sha256"],
                "run_name": job["run_name"],
            }
            if read_json(provenance_path) != expected_provenance:
                missing_outputs.append(f"provenance mismatch: {job['run_name']}")
    checks["generation_output_errors"] = missing_outputs
    errors.extend(missing_outputs)

    tasks_path = ARTIFACT / "judge_tasks_pairwise.json"
    tasks = read_json(tasks_path) if tasks_path.is_file() else []
    checks["tasks"] = len(tasks)
    if len(tasks) != 1368 or len({digest(task) for task in tasks}) != 1368:
        errors.append("expected 1,368 unique pairwise tasks")
    stage_counts = Counter(task.get("stage") for task in tasks)
    pair_counts = Counter(task.get("model_pair") for task in tasks)
    checks["tasks_by_stage"] = dict(stage_counts)
    checks["tasks_by_pair"] = dict(pair_counts)
    if stage_counts != Counter({stage: count * 6 for stage, count in STAGE_SAMPLES.items()}):
        errors.append("task stage counts changed")
    if pair_counts != Counter({pair: 456 for pair in PAIRS}):
        errors.append("model-pair counts changed")

    media_hash_cache: dict[str, str] = {}
    task_media_errors = []
    for task in tasks:
        if task.get("kind") != "pairwise":
            task_media_errors.append(f"non-pairwise task: {task.get('sample_id')}")
            continue
        if {task.get("candidate_a_model"), task.get("candidate_b_model")} != set(task.get("model_pair", "").split("__vs__")):
            task_media_errors.append(f"candidate models do not match pair: {task.get('sample_id')}")
        for field in ("source", "candidate_a", "candidate_b"):
            path_value = task.get(f"{field}_video")
            expected_hash = task.get(f"{field}_sha256")
            if not isinstance(path_value, str) or not isinstance(expected_hash, str):
                task_media_errors.append(f"missing {field} media identity: {task.get('sample_id')}")
                continue
            media_path = Path(path_value)
            if not media_path.is_file():
                task_media_errors.append(f"missing {field} media: {media_path}")
                continue
            if path_value not in media_hash_cache:
                media_hash_cache[path_value] = sha256_file(media_path)
            if media_hash_cache[path_value] != expected_hash:
                task_media_errors.append(f"{field} media hash mismatch: {task.get('sample_id')}")
    checks["unique_task_media"] = len(media_hash_cache)
    checks["task_media_errors"] = task_media_errors
    if len(media_hash_cache) != 912:
        errors.append(f"expected 912 unique task media files, found {len(media_hash_cache)}")
    errors.extend(task_media_errors)

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
    if len(orders) != 684 or malformed_orders:
        errors.append("bidirectional A/B task symmetry failed")

    summary_path = OUTPUT / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    checks["judge_summary"] = {
        key: summary.get(key) for key in ("status", "tasks_requested", "tasks_succeeded", "tasks_failed")
    }
    if not (
        summary.get("status") == "complete"
        and summary.get("tasks_requested") == 1368
        and summary.get("tasks_succeeded") == 1368
        and summary.get("tasks_failed") == 0
    ):
        errors.append("judge summary is not complete")

    result_paths = list((OUTPUT / "records").glob("*/result.json"))
    results = [read_json(path) for path in result_paths]
    successful_tasks = [result.get("identity", {}).get("task") for result in results if result.get("status") == "success"]
    checks["successful_result_records"] = len(successful_tasks)
    if len(successful_tasks) != 1368 or {digest(task) for task in successful_tasks} != {digest(task) for task in tasks}:
        errors.append("successful result identities do not match the task manifest")
    result_route_errors = sum(
        result.get("returned_model") not in ("google/gemini-3.8-flash", "google/gemini-3.8-flash-20260902")
        or result.get("returned_provider") != "Google AI Studio"
        for result in results
        if result.get("status") == "success"
    )
    checks["result_route_errors"] = result_route_errors
    if result_route_errors:
        errors.append("one or more results came from an unexpected model or provider")

    scores_path = OUTPUT / "scores.csv"
    score_rows = []
    if scores_path.is_file():
        with scores_path.open(newline="", encoding="utf-8") as handle:
            score_rows = list(csv.DictReader(handle))
    checks["score_rows"] = len(score_rows)
    if len(score_rows) != 1368:
        errors.append("scores.csv does not contain 1,368 rows")

    report_json_path = ARTIFACT / "pairwise_report.json"
    report_markdown_path = ARTIFACT / "pairwise_report.md"
    report = read_json(report_json_path) if report_json_path.is_file() else {}
    checks["report"] = {
        "status": report.get("status"),
        "tasks": report.get("tasks"),
        "bidirectional_groups": report.get("bidirectional_groups"),
        "markdown_exists": report_markdown_path.is_file(),
    }
    if not (
        report.get("status") == "complete"
        and report.get("tasks") == 1368
        and report.get("bidirectional_groups") == 684
        and report.get("source_summary_sha256") == sha256_file(summary_path)
        and report_markdown_path.is_file()
        and report_markdown_path.stat().st_size > 0
    ):
        errors.append("pairwise result report is missing or stale")

    audit = {"status": "complete" if not errors else "incomplete", "errors": errors, "checks": checks}
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACT / "completion_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
