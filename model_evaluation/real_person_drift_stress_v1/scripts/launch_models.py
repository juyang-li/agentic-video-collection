#!/usr/bin/env python3
"""Launch the three drift evaluations, one model per GPU, and wait for completion."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_model.py")
MODELS = ("lance_base", "coinve_real_step677", "gtfresh_step1150")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
    if len(visible) < 3:
        raise RuntimeError("three visible GPUs are required")
    jobs, active = [], []
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    for slot, model in enumerate(MODELS):
        output = ROOT / "outputs" / model
        log_path = ROOT / "logs" / f"{model}.log"
        command = [str(PYTHON), str(RUNNER), "--model", model, "--output-dir", str(output)]
        job = {"model_id": model, "gpu": visible[slot], "output": str(output),
               "log": str(log_path), "command": command}
        environment = dict(os.environ)
        for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                     "LIJUYANG_ICLR", "LIJUYANG_ICLR_PERSONAL", "OPEN_ROUTER_QWEN"):
            environment.pop(name, None)
        environment.update(CUDA_VISIBLE_DEVICES=visible[slot], PYTHONUNBUFFERED="1")
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        job.update(pid=process.pid, status="running")
        jobs.append(job); active.append((job, process, log))
        print(f"started model={model} gpu={visible[slot]} pid={process.pid}", flush=True)
    status_path = ROOT / "metadata/run_status.json"
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "jobs": jobs}
    write_json(status_path, status)
    while any(process.poll() is None for _, process, _ in active):
        time.sleep(10)
    failed = False
    for job, process, log in active:
        log.close(); job["returncode"] = process.returncode
        job["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    status.update(status="failed" if failed else "complete",
                  finished_at=datetime.now().astimezone().isoformat())
    write_json(status_path, status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
