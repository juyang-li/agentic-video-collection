#!/usr/bin/env python3
"""Launch Base/gtfresh normal inference as four model-by-shard workers."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_inference.py")
JOBS = tuple((model, shard) for model in ("lance_base", "gtfresh_step1150") for shard in range(2))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "LIJUYANG_ICLR", "LIJUYANG_ICLR_PERSONAL", "OPEN_ROUTER_QWEN",
    ):
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def main() -> int:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if item.strip()]
    if len(visible) < 4:
        raise RuntimeError("four visible GPUs are required for four normal workers")
    environment = clean_environment()
    for model, shard in JOBS:
        subprocess.run([
            str(PYTHON), str(RUNNER), "--model", model, "--mode", "normal",
            "--num-shards", "2", "--shard-index", str(shard), "--dry-run",
        ], env=environment, check=True)

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    active = []
    jobs = []
    for slot, (model, shard) in enumerate(JOBS):
        log_path = ROOT / "logs" / f"{model}_normal_shard{shard}.log"
        command = [
            str(PYTHON), str(RUNNER), "--model", model, "--mode", "normal",
            "--num-shards", "2", "--shard-index", str(shard),
        ]
        job_environment = dict(environment, CUDA_VISIBLE_DEVICES=visible[slot])
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, env=job_environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job = {
            "model_id": model, "mode": "normal", "shard_index": shard,
            "shard_count": 2, "gpu": visible[slot], "pid": process.pid,
            "log": str(log_path), "status": "running",
        }
        jobs.append(job)
        active.append((job, process, log))
        print(f"started gpu={visible[slot]} pid={process.pid} {model} shard={shard}/2", flush=True)

    status = {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "jobs": jobs,
    }
    write_json(ROOT / "normal_run_status.json", status)
    failed = False
    for job, process, log in active:
        process.wait()
        log.close()
        job["returncode"] = process.returncode
        job["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    if not failed:
        for model in ("lance_base", "gtfresh_step1150"):
            result = subprocess.run([
                str(PYTHON), str(RUNNER), "--model", model, "--mode", "normal",
                "--verify-only",
            ], env=environment)
            failed |= result.returncode != 0
    status.update(
        status="failed" if failed else "complete",
        finished_at=datetime.now().astimezone().isoformat(),
    )
    write_json(ROOT / "normal_run_status.json", status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
