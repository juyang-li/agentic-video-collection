#!/usr/bin/env python3
"""Run the frozen IVEBench Base/LoRA generation groups on four visible GPUs."""

from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29840)
    args = parser.parse_args()
    specification = json.loads(args.jobs.read_text())
    allowed_scopes = {"short_subject_all_91", "long_subject_all_43", "attribute_all_94"}
    if specification.get("scope") not in allowed_scopes or specification.get("generation_seed") != 42:
        raise ValueError("unexpected generation scope or seed")
    evaluation = args.jobs.resolve().parent
    groups = specification["groups"]

    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected exactly four visible GPUs, got {torch.cuda.device_count()}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = [str(index) for index in range(4)]
    if len(visible) != 4:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain four GPU identifiers")

    all_processes: list[subprocess.Popen] = []
    interrupted = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        for process in all_processes:
            if process.poll() is None:
                os.killpg(process.pid, signum)

    def cleanup() -> None:
        terminate(signal.SIGTERM, None)
        for process in all_processes:
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
        lock = stack.enter_context((evaluation / ".generation.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group_index, group in enumerate(groups):
            jobs = group["jobs"]
            slots = [int(job["gpu_slot"]) for job in jobs]
            if len(slots) != len(set(slots)) or any(slot not in range(4) for slot in slots):
                raise ValueError(f"invalid GPU slots in {group['name']}: {slots}")
            status = {
                "status": "running", "scope": specification["scope"], "group": group["name"],
                "group_index": group_index, "group_count": len(groups), "started_at": now(),
            }
            write_json(evaluation / "generation_status.json", status)
            all_processes = []
            for job_index, job in enumerate(jobs):
                config_path = Path(job["config_path"])
                config = json.loads(config_path.read_text())
                if len(config) != int(job["sample_count"]):
                    raise ValueError(f"sample count mismatch: {job['run_name']}")
                output = evaluation / "outputs_raw" / job["run_name"]
                provenance = {
                    "benchmark": specification["benchmark"], "scope": specification["scope"],
                    "model_kind": job["model_kind"], "seed": 42,
                    "base_model_sha256": specification["base_model_sha256"],
                    "adapter_sha256": specification["adapter_sha256"] if job["model_kind"] != "base" else None,
                    "config_sha256": sha256_file(config_path), "run_name": job["run_name"],
                }
                provenance_path = output / "generation_provenance.json"
                if provenance_path.exists():
                    if json.loads(provenance_path.read_text()) != provenance:
                        raise ValueError(f"output provenance changed: {output}")
                elif output.exists() and any(output.glob("*.mp4")):
                    raise ValueError(f"existing outputs lack provenance: {output}")
                write_json(provenance_path, provenance)

                environment = dict(os.environ)
                for name in (
                    "WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                    "LIJUYANG_ICLR", "OPEN_ROUTER_QWEN",
                ):
                    environment.pop(name, None)
                environment.update(
                    CUDA_VISIBLE_DEVICES=visible[int(job["gpu_slot"])],
                    MAIN_PROCESS_PORT=str(args.master_port + group_index * 10 + job_index),
                    NUM_MACHINES="1", MACHINE_RANK="0", MAIN_PROCESS_IP="127.0.0.1",
                )
                log_path = evaluation / "logs" / f"{job['run_name']}.launcher.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                process = subprocess.Popen(
                    job["command"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                all_processes.append(process)
                print(
                    f"started group={group['name']} model={job['model_kind']} "
                    f"gpu={job['gpu_slot']} pid={process.pid} samples={job['sample_count']}",
                    flush=True,
                )
            status["pids"] = [process.pid for process in all_processes]
            write_json(evaluation / "generation_status.json", status)
            while any(process.poll() is None for process in all_processes):
                if interrupted:
                    return 1
                failed = [process for process in all_processes if process.poll() not in (None, 0)]
                if failed:
                    terminate(signal.SIGTERM, None)
                    for process in all_processes:
                        process.wait()
                    status.update(status="failed", finished_at=now(), returncodes=[p.returncode for p in all_processes])
                    write_json(evaluation / "generation_status.json", status)
                    return 1
                time.sleep(5)
            if any(process.returncode for process in all_processes):
                status.update(status="failed", finished_at=now(), returncodes=[p.returncode for p in all_processes])
                write_json(evaluation / "generation_status.json", status)
                return 1
            status.update(status="group_complete", finished_at=now(), returncodes=[0] * len(all_processes))
            write_json(evaluation / "generation_status.json", status)
            print(f"group complete: {group['name']}", flush=True)
        write_json(evaluation / "generation_status.json", {
            "status": "complete", "scope": specification["scope"], "finished_at": now(),
            "groups_completed": [group["name"] for group in groups],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
