#!/usr/bin/env python3
"""Launch two models × four conditions on four GPUs (two conditions per process)."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_ablation.py")
JOBS = (
    ("lance_base", ("normal_replay", "vit_zero_text_keep", "vit_zero_text_zero")),
    ("lance_base", ("vit_gaussian_text_keep", "vit_gaussian_text_zero")),
    ("gtfresh_step1150", ("normal_replay", "vit_zero_text_keep", "vit_zero_text_zero")),
    ("gtfresh_step1150", ("vit_gaussian_text_keep", "vit_gaussian_text_zero")),
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-checkpoints", type=int)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(visible) < 4:
        raise RuntimeError("four visible GPUs are required")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    active, jobs = [], []
    for slot, (model, conditions) in enumerate(JOBS):
        log_path = ROOT / "logs" / f"{model}_{conditions[0].replace('vit_', '').replace('_text_keep', '')}.log"
        command = [str(PYTHON), str(RUNNER), "--model", model, "--conditions", *conditions]
        if args.limit_checkpoints:
            command += ["--limit-checkpoints", str(args.limit_checkpoints)]
        environment = dict(os.environ)
        for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                     "LIJUYANG_ICLR", "LIJUYANG_ICLR_PERSONAL", "OPEN_ROUTER_QWEN"):
            environment.pop(name, None)
        environment.update(CUDA_VISIBLE_DEVICES=visible[slot], PYTHONUNBUFFERED="1")
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        job = {"model_id": model, "conditions": list(conditions), "gpu": visible[slot],
               "pid": process.pid, "log": str(log_path), "command": command, "status": "running"}
        active.append((job, process, log)); jobs.append(job)
        print(f"started gpu={visible[slot]} pid={process.pid} model={model} conditions={conditions}", flush=True)
    status_path = ROOT / "run_status.json"
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "jobs": jobs}
    write_json(status_path, status)
    failed = False
    for job, process, log in active:
        process.wait(); log.close(); job["returncode"] = process.returncode
        job["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    status.update(status="failed" if failed else "complete",
                  finished_at=datetime.now().astimezone().isoformat())
    write_json(status_path, status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

