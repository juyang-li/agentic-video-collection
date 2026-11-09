#!/usr/bin/env python3
"""Launch and supervise four benchmark shards, one process per GPU."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_benchmark_shard.py")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(visible) != 4:
        raise ValueError("exactly four visible GPUs are required")
    jobs = []
    for gpu_slot, (model, shard) in enumerate((
        ("lance_base", 0), ("lance_base", 1),
        ("gtfresh_step1150", 0), ("gtfresh_step1150", 1),
    )):
        output = ROOT / "outputs" / model / f"shard{shard:02d}"
        log_path = ROOT / "logs" / f"{model}_shard{shard:02d}.log"
        command = [str(PYTHON), str(RUNNER), "--model", model, "--shard-index", str(shard),
                   "--num-shards", "2", "--output-dir", str(output)]
        jobs.append({"model_id": model, "shard": shard, "gpu": visible[gpu_slot],
                     "output": str(output), "log": str(log_path), "command": command})

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "metadata/run_status.json"
    active = []
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "jobs": []}
    for job in jobs:
        environment = dict(os.environ)
        for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                     "LIJUYANG_ICLR", "LIJUYANG_ICLR_PERSONAL", "OPEN_ROUTER_QWEN"):
            environment.pop(name, None)
        environment.update(CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1")
        log = Path(job["log"]).open("a", encoding="utf-8")
        process = subprocess.Popen(job["command"], env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        active.append((job, process, log))
        status["jobs"].append({**job, "pid": process.pid, "status": "running"})
        print(f"started {job['model_id']} shard={job['shard']} gpu={job['gpu']} pid={process.pid}", flush=True)
    write_json(status_path, status)

    while any(process.poll() is None for _, process, _ in active):
        time.sleep(10)
    failed = False
    for job, process, log in active:
        log.close()
        item = next(row for row in status["jobs"]
                    if row["model_id"] == job["model_id"] and row["shard"] == job["shard"])
        item["returncode"] = process.returncode
        item["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    status.update(status="failed" if failed else "complete",
                  finished_at=datetime.now().astimezone().isoformat())
    write_json(status_path, status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
