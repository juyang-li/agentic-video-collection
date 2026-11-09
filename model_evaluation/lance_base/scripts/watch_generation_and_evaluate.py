#!/usr/bin/env python3
"""Monitor the existing base run, recover stopped inference, then score all 240 MP4s."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT.parent / "lance_tools"
sys.path.insert(0, str(TOOLS))
from judge_spatial_openrouter import write_json
from verify_outputs import probe
from prepare_judge_tasks import prepare

RUN_NAME = "omni_edit_spatial_base_seed42_shard00"
OUTPUT = ROOT / "outputs_raw" / RUN_NAME
CONFIG = ROOT / "configs/spatial_full_seed42/shard_00.json"
SCORES = ROOT / "gemini38_low_vertex_v2"
STATE = ROOT / "pipeline_status.json"


def inference_pids() -> list[int]:
    """Only this exact output's inference process, not unrelated jobs under the same uid."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(Path(arg).name == "inference_lance.py" for arg in args) and str(OUTPUT) in args:
                found.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return found


def generation_pane_live(socket_name: str) -> bool:
    result = subprocess.run(["tmux", "-L", socket_name, "has-session", "-t", "generation"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return True
    terminal_messages = ("can't find session", "no server running", "No such file or directory", "no sessions")
    if result.returncode == 1 and any(message in result.stderr for message in terminal_messages):
        return False
    raise RuntimeError(f"cannot inspect tmux (exit {result.returncode}); do not interpret observation failure as terminal")


def snapshot(stage: str, **extra) -> None:
    data = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "files_present_not_yet_final_validated": len(list(OUTPUT.glob("*.mp4"))),
            "expected": 240, **extra}
    write_json(STATE, data)
    print(json.dumps(data), flush=True)


def quarantine_corrupt_after_terminal(config: dict) -> None:
    """Keep corrupt partial files recoverable, and never touch a still-running writer."""
    for key in config:
        path = OUTPUT / f"{int(key):06d}.mp4"
        if not path.exists():
            continue
        try:
            probe(path)
        except Exception as error:
            directory = ROOT / "quarantine_interrupted" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory.mkdir(parents=True)
            path.rename(directory / path.name)
            write_json(directory / "reason.json", {"original": str(path), "error": str(error)})
            print(f"quarantined interrupted output: {path.name} -> {directory}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmux-socket", default="lance_spatial_123662")
    parser.add_argument("--expected-job", default="123662")
    parser.add_argument("--max-generation-restarts", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != args.expected_job:
        raise RuntimeError("allocation changed; verify the new GPU allocation explicitly before recovery")
    config = json.loads(CONFIG.read_text())
    if len(config) != 240:
        raise ValueError("expected full Spatial 240")
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = None
        log = None
        restarts = 0
        while True:
            pids = inference_pids()
            pane = generation_pane_live(args.tmux_socket)
            child_live = child is not None and child.poll() is None
            if pids or pane or child_live:
                snapshot("generating", inference_pids=pids, original_pane_live=pane,
                         recovery_child_pid=child.pid if child_live else None, restarts=restarts)
                time.sleep(30)
                continue
            # Both the original pane and scoped inference process are confirmed gone.
            if log:
                log.close()
                log = None
            quarantine_corrupt_after_terminal(config)
            missing = [k for k in config if not (OUTPUT / f"{int(k):06d}.mp4").is_file()]
            if not missing:
                break
            if restarts >= args.max_generation_restarts:
                raise RuntimeError(f"generation restart limit reached; {len(missing)} outputs still missing")
            restarts += 1
            snapshot("restarting_generation", missing=len(missing), restarts=restarts)
            log = (ROOT / "logs" / f"pipeline_recovery_{time.time_ns()}.log").open("a")
            child = subprocess.Popen(["bash", str(ROOT / "scripts/run_spatial_single.sh")],
                                     stdout=log, stderr=subprocess.STDOUT)
        snapshot("validating_all_outputs")
        subprocess.run([sys.executable, str(TOOLS / "verify_outputs.py"), str(CONFIG), str(OUTPUT),
                        "--inventory", str(ROOT / "technical" / f"{RUN_NAME}.csv")], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_evaluator_inputs.py")], check=True)
        tasks = prepare()
        snapshot("evaluating_gemini", reasoning="low", model="google/gemini-3.8-flash", provider_route="google-vertex/global")
        # No API approval gate: the user explicitly authorized this complete evaluation.
        result = subprocess.run([sys.executable, str(TOOLS / "judge_spatial_openrouter.py"),
                                 "--tasks", str(tasks), "--output", str(SCORES), "--workers", "2", "--route", "vertex"])
        summary = json.loads((SCORES / "summary.json").read_text())
        if result.returncode or summary.get("tasks_succeeded") != 240 or summary.get("tasks_failed") != 0:
            raise RuntimeError("full scoring incomplete; inspect failed IDs and receipts before retrying")
        snapshot("complete", generated_verified=240, scored=240,
                 reported_cost_usd=summary["reported_cost_usd_known_requests"],
                 requests_without_reported_cost=summary["requests_without_reported_cost"],
                 mean_scores=summary["mean_scores_successes_only"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        snapshot("needs_attention", error=f"{type(error).__name__}: {error}")
        raise
