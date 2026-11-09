#!/usr/bin/env python3
"""Run Base then LoRA official IVEBench metric bundles across four GPUs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
SCRIPT = Path(__file__).resolve().parent / "run_metric_bundle.py"
BUNDLES = ("quality", "temporal", "semantic", "motion")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    global ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("short_subject", "long_subject", "attribute_all"), default="short_subject")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    ARTIFACTS = ROOT / "artifacts" / args.stage
    if not (ARTIFACTS / "metric_frame_preparation.json").exists():
        raise FileNotFoundError("metric frames have not been prepared")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = ["0", "1", "2", "3"]
    if len(visible) != 4:
        raise RuntimeError(f"expected exactly four visible GPUs, got {visible}")

    processes: list[subprocess.Popen] = []
    interrupted = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate)

    status_path = ARTIFACTS / "local_metrics_status.json"
    with ExitStack() as stack:
        lock = stack.enter_context((ARTIFACTS / ".local_metrics.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "running", "started_at": now(), "active": [], "completed": []}
        jobs: dict[int, tuple[subprocess.Popen, str, str, int]] = {}

        def start(model_kind: str, bundle: str, gpu_slot: int) -> None:
            environment = dict(os.environ)
            for name in ("OPEN_ROUTER_QWEN", "LIJUYANG_ICLR", "WAN_DB", "WANDB_API_KEY"):
                environment.pop(name, None)
            environment["CUDA_VISIBLE_DEVICES"] = visible[gpu_slot]
            log_path = ARTIFACTS / "logs" / f"metrics_{model_kind}_{bundle}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = stack.enter_context(log_path.open("a"))
            process = subprocess.Popen(
                [args.python, str(SCRIPT), "--stage", args.stage,
                 "--model-kind", model_kind, "--bundle", bundle],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(process)
            jobs[process.pid] = (process, model_kind, bundle, gpu_slot)
            print(f"started model={model_kind} bundle={bundle} gpu={gpu_slot} pid={process.pid}", flush=True)

        for gpu_slot, bundle in enumerate(BUNDLES):
            start("base", bundle, gpu_slot)
        while jobs:
            status["active"] = [
                {"pid": pid, "model_kind": model, "bundle": bundle, "gpu_slot": gpu}
                for pid, (_, model, bundle, gpu) in sorted(jobs.items())
            ]
            write_json(status_path, status)
            if interrupted:
                return 1
            finished = [(pid, job) for pid, job in jobs.items() if job[0].poll() is not None]
            if not finished:
                time.sleep(5)
                continue
            for pid, (process, model_kind, bundle, gpu_slot) in finished:
                jobs.pop(pid)
                status["completed"].append({
                    "pid": pid, "model_kind": model_kind, "bundle": bundle,
                    "gpu_slot": gpu_slot, "returncode": process.returncode, "finished_at": now(),
                })
                if process.returncode:
                    terminate(signal.SIGTERM, None)
                    for other, *_ in jobs.values():
                        other.wait()
                    write_json(status_path, {**status, "status": "failed", "finished_at": now()})
                    return 1
                if model_kind == "base":
                    start("lora_step677", bundle, gpu_slot)
        write_json(status_path, {**status, "status": "complete", "active": [], "finished_at": now(),
                                 "models": ["base", "lora_step677"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
