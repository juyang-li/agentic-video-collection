#!/usr/bin/env python3
"""Wait for both formal ablation runs, verify them, then launch IVEBench evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STAGE_ROOT = Path(__file__).resolve().parents[1]
VIDEO_COLLECTION = STAGE_ROOT.parents[1]
EVAL_ROOT = VIDEO_COLLECTION / "model_evaluation/ivebench_lance"
STATUS_PATH = STAGE_ROOT / "logs/coinve_ablation_v1/pipeline_status.json"
LOG_PATH = STAGE_ROOT / "logs/coinve_ablation_v1/evaluation_pipeline.log"
RUNS = (
    STAGE_ROOT / "runs/lance_coinve_4k_r64_seed42_e2_v1",
    STAGE_ROOT / "runs/lance_mixed_2k2k_r64_seed42_e2_v1",
)
REQUIRED_STEPS = (451, 677, 902)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status: str, **details: object) -> None:
    value = {"status": status, "updated_at": now(), **details}
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS_PATH)


def completed_checkpoint(run: Path, step: int) -> bool:
    checkpoint = run / f"checkpoints/{step:07d}"
    marker_path = checkpoint / "checkpoint_complete.json"
    adapter_path = checkpoint / "adapter.safetensors"
    if not marker_path.is_file() or not adapter_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    required_size = marker.get("required_artifact_sizes", {}).get("adapter.safetensors")
    return (
        marker.get("status") == "complete"
        and marker.get("completed_steps") == step
        and marker.get("world_size") == 4
        and required_size == adapter_path.stat().st_size
        and adapter_path.stat().st_size > 0
    )


def pid_matches(pid: int, expected_fragment: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "run_general_edit_lora.py" in command and expected_fragment in command


def run_logged(command: list[str], label: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as log:
        log.write(f"[{now()}] starting {label}: {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=VIDEO_COLLECTION, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{now()}] finished {label}: returncode={result.returncode}\n")
        log.flush()
    if result.returncode:
        raise RuntimeError(f"{label} failed with return code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pids", type=int, nargs=2, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("poll interval must be at least five seconds")

    expected = ("lance_coinve_4k_r64_seed42_e2_v1", "lance_mixed_2k2k_r64_seed42_e2_v1")
    write_status("waiting_for_training", training_pids=args.training_pids, required_steps=REQUIRED_STEPS)
    while True:
        complete = {
            run.name: [step for step in REQUIRED_STEPS if completed_checkpoint(run, step)]
            for run in RUNS
        }
        alive = [pid_matches(pid, fragment) for pid, fragment in zip(args.training_pids, expected, strict=True)]
        write_status(
            "waiting_for_training",
            training_pids=args.training_pids,
            training_process_alive=alive,
            completed_checkpoints=complete,
            required_steps=REQUIRED_STEPS,
        )
        if all(len(complete[run.name]) == len(REQUIRED_STEPS) for run in RUNS) and not any(alive):
            break
        for index, is_alive in enumerate(alive):
            if not is_alive and 902 not in complete[RUNS[index].name]:
                raise RuntimeError(f"training process exited before a complete step-902 checkpoint: {RUNS[index].name}")
        time.sleep(args.poll_seconds)

    write_status("preparing_evaluation", completed_checkpoints=complete)
    run_logged(
        [sys.executable, str(EVAL_ROOT / "scripts/prepare_coinve_ablation_eval.py")],
        "evaluation preparation",
    )
    write_status("running_evaluation")
    run_logged(
        [sys.executable, str(EVAL_ROOT / "scripts/run_coinve_ablation_eval_generation.py"), "--skip-judge"],
        "local generation and pairwise task preparation",
    )
    generation_status_path = EVAL_ROOT / "artifacts/coinve_ablation_v1/generation_status.json"
    generation_status = json.loads(generation_status_path.read_text())
    if generation_status.get("status") != "generation_complete_judge_pending":
        raise RuntimeError(f"local evaluation did not report the expected status: {generation_status}")
    write_status(
        "judge_pending",
        generation_status=generation_status,
        judge_tasks=str(EVAL_ROOT / "artifacts/coinve_ablation_v1/judge_tasks_pairwise.json"),
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        write_status("failed", error=f"{type(error).__name__}: {error}")
        raise
    raise SystemExit(exit_code)
