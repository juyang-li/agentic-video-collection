#!/usr/bin/env python3
"""Run and resume the frozen two-model IVEBench generation plan on eight GPUs."""

from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = ROOT / "artifacts/coinve_ablation_v1"
MAIN_JUDGE_TASKS = 1368
VS_BASE_JUDGE_TASKS = 912
TOTAL_JUDGE_TASKS = MAIN_JUDGE_TASKS + VS_BASE_JUDGE_TASKS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_all_judges(judge: Path, environment: dict[str, str]) -> int:
    specifications = (
        ("judge_tasks_pairwise.json", "gemini_pairwise_blind_v1"),
        ("judge_tasks_pairwise_vs_base.json", "gemini_pairwise_vs_base_blind_v1"),
    )
    for tasks_name, output_name in specifications:
        result = subprocess.run(
            [
                sys.executable, str(judge),
                "--tasks", str(PLAN_ROOT / tasks_name),
                "--output", str(PLAN_ROOT / output_name),
                "--api-key-env", "OPEN_ROUTER_QWEN", "--workers", "2", "--max-attempts", "4",
            ],
            env=environment,
        )
        if result.returncode:
            return result.returncode
    return 0


def render_and_audit() -> None:
    report = ROOT / "scripts/render_coinve_ablation_report.py"
    subprocess.run([sys.executable, str(report)], check=True)
    subprocess.run(
        [
            sys.executable, str(report),
            "--input", str(PLAN_ROOT / "gemini_pairwise_vs_base_blind_v1/summary.json"),
            "--output-json", str(PLAN_ROOT / "pairwise_vs_base_report.json"),
            "--output-markdown", str(PLAN_ROOT / "pairwise_vs_base_report.md"),
            "--expected-tasks", str(VS_BASE_JUDGE_TASKS),
            "--expected-groups", "456",
            "--title", "CoinVE 数据消融：相对 Base 的 IVEBench 两两盲评",
        ],
        check=True,
    )
    audit = ROOT / "scripts/audit_coinve_ablation_completion.py"
    subprocess.run([sys.executable, str(audit)], check=True)
    supplemental_audit = ROOT / "scripts/audit_coinve_ablation_vs_base.py"
    subprocess.run([sys.executable, str(supplemental_audit)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_ROOT / "generation_plan.json")
    parser.add_argument("--skip-judge", action="store_true", help="Generate locally and prepare blinded tasks without calling an external API.")
    parser.add_argument("--judge-only", action="store_true", help="Judge already prepared full pairwise tasks without running generation.")
    args = parser.parse_args()
    if args.skip_judge and args.judge_only:
        parser.error("--skip-judge and --judge-only are mutually exclusive")
    specification = json.loads(args.plan.read_text())
    if specification.get("schema_version") != "ivebench_coinve_ablation_generation_v1" or specification.get("generation_seed") != 42:
        raise ValueError("unexpected generation plan")
    status_path = PLAN_ROOT / "generation_status.json"
    judge_environment = dict(os.environ)
    if not judge_environment.get("OPEN_ROUTER_QWEN") and judge_environment.get("LIJUYANG_ICLR"):
        judge_environment["OPEN_ROUTER_QWEN"] = judge_environment["LIJUYANG_ICLR"]
    if args.judge_only:
        prepare = ROOT / "scripts/prepare_coinve_ablation_pairwise.py"
        judge = ROOT / "scripts/judge_gemini.py"
        subprocess.run([sys.executable, str(prepare), "--phase", "all"], check=True)
        write_json(status_path, {
            "status": "judging", "started_at": now(), "tasks": TOTAL_JUDGE_TASKS,
            "main_tasks": MAIN_JUDGE_TASKS, "vs_base_tasks": VS_BASE_JUDGE_TASKS,
        })
        if run_all_judges(judge, judge_environment):
            write_json(status_path, {"status": "generation_complete_judge_incomplete", "finished_at": now()})
            return 1
        render_and_audit()
        write_json(status_path, {
            "status": "complete", "finished_at": now(), "judge_tasks": TOTAL_JUDGE_TASKS,
            "main_judge_tasks": MAIN_JUDGE_TASKS, "vs_base_judge_tasks": VS_BASE_JUDGE_TASKS,
            "completion_audit": str(PLAN_ROOT / "completion_audit.json"),
            "vs_base_completion_audit": str(PLAN_ROOT / "completion_audit_vs_base.json"),
        })
        return 0
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"expected exactly eight visible GPUs, got {torch.cuda.device_count()}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = [str(index) for index in range(8)]
    if len(visible) != 8:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain eight GPU identifiers")

    processes: list[subprocess.Popen] = []
    background: list[subprocess.Popen] = []
    interrupted = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signum)
        for process in background:
            if process.poll() is None:
                os.killpg(process.pid, signum)

    def cleanup() -> None:
        terminate(signal.SIGTERM, None)
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for process in background:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    atexit.register(cleanup)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate)

    with ExitStack() as stack:
        lock = stack.enter_context((PLAN_ROOT / ".generation.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group_index, group in enumerate(specification["groups"]):
            status = {
                "status": "running", "group": group["name"], "group_index": group_index,
                "group_count": len(specification["groups"]), "started_at": now(),
            }
            write_json(status_path, status)
            processes = []
            for job_index, job in enumerate(group["jobs"]):
                config = json.loads(Path(job["config_path"]).read_text())
                if len(config) != job["sample_count"]:
                    raise ValueError(f"config count mismatch for {job['run_name']}")
                artifact = ROOT / "artifacts" / job["stage"]
                output = artifact / "outputs_raw" / job["run_name"]
                provenance = {
                    "benchmark": "IVEBench", "scope": job["stage"], "phase": job["phase"],
                    "model_kind": job["model_kind"], "seed": 42,
                    "base_model_sha256": specification["base_model_sha256"],
                    "adapter_sha256": job["adapter_sha256"],
                    "config_sha256": job["config_sha256"], "run_name": job["run_name"],
                }
                provenance_path = output / "generation_provenance.json"
                if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
                    raise ValueError(f"output provenance changed: {output}")
                if output.exists() and any(output.glob("*.mp4")) and not provenance_path.exists():
                    raise ValueError(f"existing outputs lack provenance: {output}")
                write_json(provenance_path, provenance)
                environment = dict(os.environ)
                for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LIJUYANG_ICLR", "OPEN_ROUTER_QWEN"):
                    environment.pop(name, None)
                environment.update({
                    "CUDA_VISIBLE_DEVICES": visible[int(job["gpu_slot"])],
                    "MAIN_PROCESS_PORT": str(30100 + group_index * 16 + job_index),
                    "NUM_MACHINES": "1", "MACHINE_RANK": "0", "MAIN_PROCESS_IP": "127.0.0.1",
                    "LANCE_MAX_DURATION": str(job["max_duration"]),
                })
                log_path = PLAN_ROOT / "logs" / f"{job['run_name']}.launcher.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                process = subprocess.Popen(job["command"], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                print(f"started group={group['name']} model={job['model_kind']} gpu={job['gpu_slot']} pid={process.pid} samples={job['sample_count']}", flush=True)
            status["pids"] = [process.pid for process in processes]
            write_json(status_path, status)
            while any(process.poll() is None for process in processes):
                if interrupted:
                    return 1
                if any(process.poll() not in (None, 0) for process in processes):
                    terminate(signal.SIGTERM, None)
                    for process in processes:
                        process.wait()
                    status.update(status="failed", finished_at=now(), returncodes=[process.returncode for process in processes])
                    write_json(status_path, status)
                    return 1
                time.sleep(5)
            if any(process.returncode for process in processes):
                status.update(status="failed", finished_at=now(), returncodes=[process.returncode for process in processes])
                write_json(status_path, status)
                return 1
            status.update(status="group_complete", finished_at=now(), returncodes=[0] * len(processes))
            write_json(status_path, status)
            print(f"group complete: {group['name']}", flush=True)
            if args.skip_judge:
                prepare = ROOT / "scripts/prepare_coinve_ablation_pairwise.py"
                prepare_log_path = PLAN_ROOT / "logs" / f"prepare_media_{group['stage']}.log"
                prepare_log = stack.enter_context(prepare_log_path.open("a"))
                prepare_process = subprocess.Popen(
                    [sys.executable, str(prepare), "--media-only-stage", group["stage"]],
                    stdout=prepare_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                background.append(prepare_process)
                print(
                    f"started local judge-media preparation stage={group['stage']} pid={prepare_process.pid}",
                    flush=True,
                )
        for process in background:
            process.wait()
        if any(process.returncode for process in background):
            write_json(status_path, {
                "status": "failed", "finished_at": now(),
                "background_returncodes": [process.returncode for process in background],
            })
            return 1
        prepare = ROOT / "scripts/prepare_coinve_ablation_pairwise.py"
        judge = ROOT / "scripts/judge_gemini.py"
        subprocess.run([sys.executable, str(prepare), "--phase", "all"], check=True)
        if args.skip_judge:
            write_json(status_path, {
                "status": "generation_complete_judge_pending",
                "finished_at": now(),
                "groups_completed": [group["name"] for group in specification["groups"]],
                "judge_tasks": MAIN_JUDGE_TASKS,
                "vs_base_judge_tasks": VS_BASE_JUDGE_TASKS,
            })
            return 0
        if run_all_judges(judge, judge_environment):
            write_json(status_path, {"status": "generation_complete_judge_incomplete", "finished_at": now()})
            return 1
        render_and_audit()
        write_json(status_path, {
            "status": "complete", "finished_at": now(),
            "groups_completed": [group["name"] for group in specification["groups"]],
            "judge_tasks": TOTAL_JUDGE_TASKS,
            "main_judge_tasks": MAIN_JUDGE_TASKS, "vs_base_judge_tasks": VS_BASE_JUDGE_TASKS,
            "completion_audit": str(PLAN_ROOT / "completion_audit.json"),
            "vs_base_completion_audit": str(PLAN_ROOT / "completion_audit_vs_base.json"),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
