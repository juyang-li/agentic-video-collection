#!/usr/bin/env python3
"""Run prepared checkpoint groups on four GPUs, preserving output provenance on resume."""

from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from judge_spatial import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29742)
    args = parser.parse_args()
    specification = json.loads(args.jobs.read_text())
    evaluation = args.jobs.resolve().parent
    jobs = specification["jobs"]
    if specification["generation_seed"] != 42 or specification["shards_per_checkpoint"] != 4:
        raise ValueError("expected seed42 with four independent GPU shards")
    steps = specification["checkpoint_steps"]
    if specification["stage"] != "screen" and len(steps) != 1:
        raise ValueError("later evaluation stages may run only one winner")
    for step in steps:
        group = [job for job in jobs if job["checkpoint_step"] == step]
        if sorted(job["gpu_slot"] for job in group) != [0, 1, 2, 3]:
            raise ValueError("each checkpoint requires exactly one job per GPU")
        for job in group:
            checkpoint = Path(job["adapter_checkpoint"])
            marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
            if marker.get("status") != "complete" or marker["completed_steps"] != step:
                raise ValueError(f"incomplete or wrong checkpoint: {checkpoint}")
    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError("this runner requires exactly four visible GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = [str(i) for i in range(4)]
    if len(visible) != 4:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must list four GPU identifiers")
    processes: list[subprocess.Popen] = []
    interrupted = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        for process in processes:
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

    atexit.register(cleanup)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate)
    with ExitStack() as stack:
        lock = stack.enter_context((evaluation / ".inference.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for step in steps:
            if interrupted:
                return 1
            group = sorted((job for job in jobs if job["checkpoint_step"] == step), key=lambda j: j["gpu_slot"])
            adapter_hash = sha256_file(Path(group[0]["adapter_checkpoint"]) / "adapter.safetensors")
            processes = []
            for job in group:
                if interrupted:
                    return 1
                slot = job["gpu_slot"]
                output = evaluation / "outputs_raw" / job["run_name"]
                provenance = {"job": job, "adapter_sha256": adapter_hash,
                              "config_sha256": sha256_file(Path(job["config_path"])), "seed": 42}
                provenance_path = output / "generation_provenance.json"
                if provenance_path.exists():
                    if json.loads(provenance_path.read_text()) != provenance:
                        raise ValueError(f"output provenance changed: {output}")
                elif output.exists() and any(output.glob("*.mp4")):
                    raise ValueError(f"existing outputs lack provenance: {output}")
                write_json(provenance_path, provenance)
                environment = dict(os.environ)
                for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LIJUYANG_ICLR", "OPEN_ROUTER_QWEN"):
                    environment.pop(name, None)
                environment.update(CUDA_VISIBLE_DEVICES=visible[slot], MAIN_PROCESS_PORT=str(args.master_port + slot),
                                   NUM_MACHINES="1", MACHINE_RANK="0", MAIN_PROCESS_IP="127.0.0.1")
                log_path = evaluation / "logs" / f"{job['run_name']}.launcher.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                process = subprocess.Popen(job["command"], env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                print(f"started step={step} gpu={slot} pid={process.pid} samples={job['sample_count']}", flush=True)
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None, 0) for process in processes):
                    terminate(signal.SIGTERM, None)
                    for process in processes:
                        process.wait()
                    return 1
                time.sleep(2)
            if any(process.returncode for process in processes):
                return 1
            print(f"checkpoint {step} generation and full-decode verification complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
