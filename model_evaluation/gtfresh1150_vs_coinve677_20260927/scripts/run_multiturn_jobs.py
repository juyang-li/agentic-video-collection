#!/usr/bin/env python3
"""Run the two multi-turn model jobs concurrently and retain independent status."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.jobs.read_text())
    root = Path(__file__).resolve().parents[1]
    status_path = root / "metadata/multiturn_run_status.json"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    processes = []
    status = {"status": "running", "jobs": []}
    for job in specification["jobs"]:
        slot = int(job["gpu_slot"])
        environment = dict(os.environ)
        for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                     "LIJUYANG_ICLR", "OPEN_ROUTER_QWEN"):
            environment.pop(name, None)
        environment.update(CUDA_VISIBLE_DEVICES=visible[slot], PYTHONUNBUFFERED="1")
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a")
        process = subprocess.Popen(job["command"], env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((job, process, log))
        status["jobs"].append({"model_id": job["model_id"], "pid": process.pid,
                               "status": "running", "log": str(log_path)})
        print(f"started model={job['model_id']} gpu={slot} pid={process.pid}", flush=True)
    write_json(status_path, status)
    while any(process.poll() is None for _, process, _ in processes):
        time.sleep(10)
    failed = False
    for job, process, log in processes:
        log.close()
        entry = next(item for item in status["jobs"] if item["model_id"] == job["model_id"])
        entry["returncode"] = process.returncode
        entry["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    status["status"] = "failed" if failed else "complete"
    write_json(status_path, status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
