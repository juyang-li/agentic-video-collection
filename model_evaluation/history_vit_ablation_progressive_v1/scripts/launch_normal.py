#!/usr/bin/env python3
"""Launch the two frozen no-camera normal-history runs on two GPUs."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_small_ablation.py")
MODELS = ("lance_base", "gtfresh_step1150")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if item.strip()]
    if len(visible) < 2:
        raise RuntimeError("two visible GPUs are required for parallel normal runs")
    clean_env = dict(os.environ)
    for name in ("WAN_DB", "WANDB_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "LIJUYANG_ICLR", "LIJUYANG_ICLR_PERSONAL", "OPEN_ROUTER_QWEN"):
        clean_env.pop(name, None)
    for model in MODELS:
        subprocess.run([str(PYTHON), str(RUNNER), "--model", model, "--mode", "normal", "--dry-run"],
                       env=clean_env, check=True)

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    active = []
    jobs = []
    for slot, model in enumerate(MODELS):
        log_path = ROOT / "logs" / f"{model}_normal.log"
        command = [str(PYTHON), str(RUNNER), "--model", model, "--mode", "normal"]
        env = dict(clean_env)
        env.update(CUDA_VISIBLE_DEVICES=visible[slot], PYTHONUNBUFFERED="1")
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        job = {"model_id": model, "mode": "normal", "gpu": visible[slot], "pid": process.pid,
               "log": str(log_path), "status": "running"}
        jobs.append(job)
        active.append((job, process, log))
        print(f"started gpu={visible[slot]} pid={process.pid} {model} normal", flush=True)

    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "jobs": jobs}
    write_json(ROOT / "normal_run_status.json", status)
    failed = False
    for job, process, log in active:
        process.wait()
        log.close()
        job["returncode"] = process.returncode
        job["status"] = "complete" if process.returncode == 0 else "failed"
        failed |= process.returncode != 0
    status.update(status="failed" if failed else "complete",
                  finished_at=datetime.now().astimezone().isoformat())
    write_json(ROOT / "normal_run_status.json", status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
