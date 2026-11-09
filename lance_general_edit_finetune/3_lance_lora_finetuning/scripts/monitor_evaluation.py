#!/usr/bin/env python3
"""Read-only evaluation health checks; never restart workers or issue API requests."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from judge_spatial import write_json

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
COUNTS = {"screen": 144, "remaining": 147, "test": 204, "omni": 240}


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def identity(task: dict) -> tuple:
    return task.get("stage"), task.get("checkpoint_step"), task["sample_id"]


def stage_snapshot(evaluation: Path, stage: str, protocol_sha: str) -> dict:
    tasks = read_json(evaluation / f"{stage}_generation_tasks.json", [])
    expected = {identity(task) for task in tasks}
    score_root = evaluation / f"{stage}_scores_vertex_v2"
    summary = read_json(score_root / "summary.json", {})
    successful = set()
    invalid_receipts = 0
    for path in (score_root / "records").glob("*/result.json"):
        record = read_json(path)
        key = identity(record["identity"]["task"])
        if (record.get("status") == "success" and key in expected
                and record["identity"].get("protocol_sha256") == protocol_sha):
            successful.add(key)
        else:
            invalid_receipts += 1
    technical_count = 0
    for job in read_json(evaluation / f"{stage}_jobs.json", {}).get("jobs", []):
        inventory = evaluation / "technical" / f"{job['run_name']}.csv"
        if inventory.is_file():
            with inventory.open(newline="") as stream:
                technical_count += sum(row.get("full_decode_ok") == "1" for row in csv.DictReader(stream))
    generated = sum(Path(task["generated_video"]).is_file() for task in tasks)
    complete = (len(tasks) == len(expected) == generated == technical_count
                == len(successful) == COUNTS[stage] and invalid_receipts == 0
                and summary.get("scope") == "full_manifest"
                and summary.get("status") == "complete"
                and summary.get("tasks_succeeded") == COUNTS[stage]
                and summary.get("tasks_failed") == 0)
    return {"expected": COUNTS[stage], "manifest_tasks": len(tasks),
            "generated_files": generated, "full_decode_reported": technical_count,
            "successful_receipts": len(successful), "invalid_receipts": invalid_receipts,
            "summary_status": summary.get("status", "not_started"),
            "failed_task_ids": summary.get("failed_task_ids", []),
            "known_cost_usd": summary.get("reported_cost_usd_known_requests", 0),
            "requests_without_reported_cost": summary.get("requests_without_reported_cost", 0),
            "complete_reported": complete}


def owned_processes() -> tuple[list[dict], int]:
    processes, uncertain = [], 0
    names = {"datamake_lora_inference_lance.py": "inference",
             "run_evaluation_jobs.py": "generation_runner",
             "judge_spatial_openrouter.py": "judge",
             "run_funnel_evaluation.py": "funnel"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            role = next((role for name, role in names.items()
                         if any(Path(arg).name == name for arg in argv if arg)), None)
            if role is None:
                continue
            cwd = (entry / "cwd").resolve(strict=True)
            if cwd != ROOT.resolve() and str(EVALUATION) not in " ".join(argv):
                continue
            # PID plus start ticks distinguishes a live process from PID reuse.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z":
                processes.append({"pid": int(entry.name), "role": role,
                                  "start_ticks": int(fields[19])})
        except FileNotFoundError:
            continue
        except PermissionError:
            uncertain += 1
    return sorted(processes, key=lambda item: item["pid"]), uncertain


def snapshot(evaluation: Path, previous: dict | None = None) -> dict:
    contract = read_json(evaluation / "judge_contract.json")
    stages = {stage: stage_snapshot(evaluation, stage, contract["canonical_protocol_sha256"])
              for stage in COUNTS}
    processes, uncertain = owned_processes()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=15, check=True)
        gpu = [{"index": int(fields[0]), "utilization_percent": int(fields[1]),
                "memory_mib": int(fields[2])}
               for line in result.stdout.splitlines() if (fields := line.split(","))]
        gpu_error = None
    except (OSError, subprocess.SubprocessError) as error:
        gpu, gpu_error = [], type(error).__name__
    warnings = []
    generation_pending = sum(item["expected"] - item["generated_files"] for item in stages.values())
    generation_live = any(p["role"] in ("inference", "generation_runner") for p in processes)
    if generation_pending and not generation_live:
        warnings.append("generation_pending_without_observed_worker_check_before_restart")
    if any(item["failed_task_ids"] for item in stages.values()):
        warnings.append("scoring_failures_remain_not_counted_as_zero")
    if previous and generation_live:
        before = sum(item["generated_files"] for item in previous.get("stages", {}).values())
        now = sum(item["generated_files"] for item in stages.values())
        if before == now:
            warnings.append("no_new_video_since_previous_check_investigate_live_worker")
    if gpu_error or uncertain:
        warnings.append("partial_observation_failure_is_not_process_termination")
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "stages": stages,
            "processes": processes, "process_observation_uncertain": uncertain,
            "gpu": gpu, "gpu_observation_error": gpu_error, "warnings": warnings,
            "all_scores_complete_reported": all(item["complete_reported"] for item in stages.values()),
            "note": "Counts are health evidence, not a substitute for final SHA256/full-decode audit."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != args.expected_job or args.interval < 60:
        parser.error("allocation mismatch or unsafe check interval")
    evaluation = args.evaluation_root.resolve()
    previous = None
    with (evaluation / ".health_monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            report = snapshot(evaluation, previous)
            report["check_interval_seconds"] = args.interval
            write_json(evaluation / "health_status.json", report)
            with (evaluation / "health_history.jsonl").open("a") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if args.once or report["all_scores_complete_reported"]:
                return 0
            previous = report
            deadline = time.monotonic() + args.interval
            while (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(30, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
