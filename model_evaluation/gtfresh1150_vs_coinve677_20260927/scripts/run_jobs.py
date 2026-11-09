#!/usr/bin/env python3
"""Run independent single-GPU jobs; preserve healthy shards if another fails."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29840)
    args = parser.parse_args()
    spec = json.loads(args.jobs.read_text())
    root = Path(__file__).resolve().parents[1]
    status_path = root / "metadata" / f"{spec['kind']}_run_status.json"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        import torch
        visible = [str(index) for index in range(torch.cuda.device_count())]
    if len(visible) < 1:
        raise RuntimeError("no visible GPU")
    status = {"kind": spec["kind"], "status": "running", "groups": []}
    write_json(status_path, status)

    with ExitStack() as stack:
        for group_index, group in enumerate(spec["groups"]):
            processes = []
            group_status = {"model_id": group["model_id"], "jobs": []}
            status["groups"].append(group_status)
            for job in group["jobs"]:
                slot = int(job["gpu_slot"])
                if slot >= len(visible):
                    raise RuntimeError(f"GPU slot {slot} unavailable")
                output = root / "outputs_raw" / job["run_name"]
                output.mkdir(parents=True, exist_ok=True)
                provenance = {
                    "job": job,
                    "adapter_sha256": sha256(Path(job["checkpoint"]) / "adapter.safetensors"),
                    "config_sha256": sha256(Path(job["config"])),
                    "seed": 42,
                }
                provenance_path = output / "generation_provenance.json"
                if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
                    raise ValueError(f"provenance changed: {output}")
                write_json(provenance_path, provenance)
                environment = dict(os.environ)
                for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                             "LIJUYANG_ICLR", "OPEN_ROUTER_QWEN"):
                    environment.pop(name, None)
                environment.update(
                    CUDA_VISIBLE_DEVICES=visible[slot],
                    MAIN_PROCESS_PORT=str(args.master_port + group_index * 16 + slot),
                    NUM_MACHINES="1", MACHINE_RANK="0", MAIN_PROCESS_IP="127.0.0.1",
                )
                log_path = root / "logs" / f"{job['run_name']}.launcher.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open("a"))
                process = subprocess.Popen(job["command"], env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((job, process, log_path))
                group_status["jobs"].append({"run_name": job["run_name"], "pid": process.pid,
                                               "status": "running", "log": str(log_path)})
                print(f"started model={group['model_id']} gpu={slot} pid={process.pid} samples={job['sample_count']}", flush=True)
            write_json(status_path, status)
            while any(process.poll() is None for _, process, _ in processes):
                time.sleep(10)
            failed = False
            for job, process, _ in processes:
                entry = next(item for item in group_status["jobs"] if item["run_name"] == job["run_name"])
                entry["returncode"] = process.returncode
                entry["status"] = "complete" if process.returncode == 0 else "failed"
                failed |= process.returncode != 0
            write_json(status_path, status)
            if failed:
                status["status"] = "failed"
                write_json(status_path, status)
                return 1
            print(f"model={group['model_id']} complete", flush=True)
    status["status"] = "complete"
    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
