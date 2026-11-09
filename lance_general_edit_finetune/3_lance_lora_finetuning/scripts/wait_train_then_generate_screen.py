#!/usr/bin/env python3
"""After this exact training process finishes, generate val48 locally; never call a judge API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/lance_4k_r64_seed42_e2_v1.json"
PROFILE_SHA = "2ede096470808848c83661647e99aa4cff527b9fb64d8cd621d1ef8b78b4ace3"
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
CANDIDATES = (451, 677, 902)


def verify_jobs(profile: dict) -> str:
    path = EVALUATION / "screen_jobs.json"
    spec = json.loads(path.read_text())
    if (spec["stage"], spec["generation_seed"], spec["generation_count"], spec["checkpoint_steps"]) != (
        "screen", 42, 144, list(CANDIDATES)
    ) or len(spec["jobs"]) != 12:
        raise ValueError("unexpected screen scope")
    checkpoints = Path(profile["run"]["output_root"]) / profile["run"]["run_name"] / "checkpoints"
    for job in spec["jobs"]:
        step, slot = job["checkpoint_step"], job["gpu_slot"]
        if step not in CANDIDATES or slot not in range(4) or job["sample_count"] != 12:
            raise ValueError("unexpected screen shard")
        name = f"screen_step{step:07d}_shard{slot:02d}"
        expected = ["bash", str(ROOT / "scripts/run_lora_video_edit.sh"), str(EVALUATION),
                    profile["source"]["runtime"], profile["model"]["model_path"],
                    str(checkpoints / f"{step:07d}"), str(EVALUATION / "configs" / f"{name}.json"), name]
        if job["command"] != expected:
            raise ValueError("only the fixed local Lance inference command is permitted")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_alive(pid: int, started: float) -> bool | None:
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - started) < 0.001 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return None  # An observation failure is not evidence that training ended.


def verify_checkpoints(profile: dict) -> None:
    root = Path(profile["run"]["output_root"]) / profile["run"]["run_name"] / "checkpoints"
    for step in CANDIDATES:
        checkpoint = root / f"{step:07d}"
        marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        if (marker["status"], marker["world_size"], marker["completed_steps"]) != ("complete", 4, step):
            raise ValueError(f"incomplete or incorrect checkpoint: {checkpoint}")
        for name, size in marker["required_artifact_sizes"].items():
            if (checkpoint / name).stat().st_size != size:
                raise ValueError(f"checkpoint artifact size mismatch: {checkpoint / name}")
        if len(list((checkpoint / "optimizer").iterdir())) != marker["optimizer_dcp_file_count"]:
            raise ValueError(f"incomplete optimizer: {checkpoint}")
        contract = json.loads((checkpoint / "datamake_online_training_contract.json").read_text())
        if contract["profile_sha256"] != PROFILE_SHA:
            raise ValueError(f"checkpoint came from a different profile: {checkpoint}")


def gpu_processes() -> list[int] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return sorted({int(line.strip()) for line in completed.stdout.splitlines() if line.strip()})
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--train-start-time", type=float, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.train_pid <= 0:
        parser.error("invalid PID or poll interval")
    if os.environ.get("SLURM_JOB_ID") != args.slurm_job_id:
        raise ValueError("handoff must stay inside the verified training allocation")
    if hashlib.sha256(PROFILE.read_bytes()).hexdigest() != PROFILE_SHA:
        raise ValueError("frozen training profile changed")
    profile = json.loads(PROFILE.read_text())
    jobs_hash = verify_jobs(profile)
    if args.check_only:
        print(json.dumps({"training_alive": training_alive(args.train_pid, args.train_start_time),
                          "slurm_job_id": args.slurm_job_id, "judge_api_calls": 0}))
        return 0

    state = EVALUATION / "local_handoff_status.json"

    def record(status: str, **fields: object) -> None:
        value = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "training_pid": args.train_pid, "training_create_time": args.train_start_time,
                 "slurm_job_id": args.slurm_job_id, "judge_api_calls": 0, **fields}
        temporary = state.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(state)

    child = None
    stopped = False

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True
        # Never signal training, or any GPU process we did not start.
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    with (EVALUATION / ".local_handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not stopped:
                alive = training_alive(args.train_pid, args.train_start_time)
                if alive is False:
                    break
                record("waiting_for_training" if alive else "training_observation_retry")
                time.sleep(args.poll_seconds)
            if stopped:
                record("interrupted_without_stopping_training")
                return 130
            verify_checkpoints(profile)
            while not stopped:
                busy = gpu_processes()
                if busy == []:
                    break
                record("waiting_for_gpu_release" if busy is not None else "gpu_observation_retry",
                       visible_gpu_processes=busy)
                time.sleep(args.poll_seconds)
            if stopped:
                record("interrupted_without_stopping_training")
                return 130
            if hashlib.sha256(PROFILE.read_bytes()).hexdigest() != PROFILE_SHA:
                raise ValueError("profile changed while waiting")
            if verify_jobs(profile) != jobs_hash:
                raise ValueError("screen jobs changed while waiting")
            environment = dict(os.environ)
            for name in ("WAN_DB", "WANDB_API_KEY", "OPEN_ROUTER_QWEN", "LIJUYANG_ICLR",
                         "GEMINI_API_KEY", "GOOGLE_API_KEY"):
                environment.pop(name, None)
            environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
            record("generating_screen_locally", expected_videos=144)
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/run_evaluation_jobs.py"),
                 "--jobs", str(EVALUATION / "screen_jobs.json")],
                cwd=ROOT, env=environment, start_new_session=True,
            )
            result = child.wait()
            record("local_screen_complete_api_not_called" if result == 0 else "local_generation_failed",
                   generation_exit_code=result)
            return result
        except BaseException as error:
            record("failed_without_calling_api", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
