#!/usr/bin/env python3
"""Adopt existing test wrappers, then continue frozen generation and scoring."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from judge_spatial import write_json
from run_funnel_evaluation import (
    COUNTS, EVALUATION, ROOT, require_api_authorization, scores_root,
    stage_commands, verify_contract,
)


def process_state(pid: int) -> tuple[int, str] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), fields[0]
    except FileNotFoundError:
        return None


def original_alive(pid: int, start_ticks: int) -> bool:
    current = process_state(pid)
    return current is not None and current[0] == start_ticks and current[1] != "Z"


def identify_wrappers(pids: list[int]) -> list[dict]:
    if len(pids) != 4 or len(set(pids)) != 4:
        raise ValueError("exactly four distinct existing test-wrapper PIDs required")
    result = []
    for shard, pid in enumerate(pids):
        proc = Path(f"/proc/{pid}")
        argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")
        state = process_state(pid)
        name = f"test_step0000677_shard{shard:02d}"
        if (proc.stat().st_uid != os.getuid() or state is None or state[1] == "Z"
                or str(ROOT / "scripts/run_lora_video_edit.sh") not in argv
                or str(EVALUATION) not in argv or name not in argv):
            raise ValueError("existing PID does not match its frozen test shard")
        result.append({"pid": pid, "start_ticks": state[0], "run_name": name})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--test-wrapper-pids", type=int, nargs=4, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != args.expected_job:
        parser.error("allocation changed; re-audit resources before recovery")
    contract = verify_contract(EVALUATION)
    require_api_authorization(EVALUATION, contract)
    decision = json.loads((EVALUATION / "promotion_decision.json").read_text())
    if decision["promotion_steps"] != [677]:
        raise ValueError("this recovery is restricted to the frozen winner 677")
    wrappers = identify_wrappers(args.test_wrapper_pids)
    if args.validate_only:
        print(json.dumps({"validated_wrappers": wrappers,
                          "protocol_sha256": contract["canonical_protocol_sha256"]}))
        return 0
    state = {"controller_pid": os.getpid(), "job_id": args.expected_job,
             "adopted_test_wrappers": wrappers,
             "protocol_sha256": contract["canonical_protocol_sha256"],
             "started_at_utc": datetime.now(timezone.utc).isoformat()}

    def record(phase: str, **fields) -> None:
        state.update(phase=phase, updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(EVALUATION / "session_recovery_status.json", state)
        print(json.dumps(state), flush=True)

    def run(command: list[str]) -> int:
        return subprocess.run(command, cwd=ROOT).returncode

    def judge(stage: str) -> int:
        require_api_authorization(EVALUATION, verify_contract(EVALUATION))
        return run(stage_commands(EVALUATION, stage)[2])

    with ExitStack() as locks:
        for name in (".session_recovery.lock", ".funnel.lock", ".recovery_scoring.lock"):
            lock = locks.enter_context((EVALUATION / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # The old parent died, releasing this lock while its independent
            # wrappers remained alive. Reclaim it without signalling those workers.
            with (EVALUATION / ".inference.lock").open("a") as inference_lock:
                fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                record("waiting_for_existing_test_wrappers")
                deadline = time.monotonic() + 4 * 3600
                while any(original_alive(w["pid"], w["start_ticks"]) for w in wrappers):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test wrappers still alive; do not duplicate generation")
                    time.sleep(30)
                # Process disappearance alone does not prove the allocated GPUs
                # are free (e.g. orphaned descendants). Refuse to launch on them.
                gpu_pids = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    text=True, timeout=15).strip()
                if gpu_pids:
                    raise RuntimeError("GPU workers remain after wrapper exit; manual inspection required")
            # Existing wrappers normally have finished full-decode verification.
            # The standard runner rechecks cached outputs and generates only missing
            # files, now that all old wrappers and GPU descendants are confirmed gone.
            record("verifying_or_resuming_test_outputs")
            commands = stage_commands(EVALUATION, "test")
            for command in commands[:2]:
                if run(command):
                    raise RuntimeError("test output verification/recovery failed")
            with ThreadPoolExecutor(max_workers=1) as pool:
                test_score = pool.submit(judge, "test")
                record("generating_omni_while_scoring_test")
                for command in stage_commands(EVALUATION, "omni")[:2]:
                    if run(command):
                        raise RuntimeError("Omni generation or verification failed")
                test_code = test_score.result()
                record("scoring_omni", test_scoring_returncode=test_code)
                omni_code = judge("omni")
            incomplete = []
            for stage, count in COUNTS.items():
                summary = json.loads((scores_root(EVALUATION, stage) / "summary.json").read_text())
                if (summary.get("scope"), summary.get("tasks_succeeded"), summary.get("tasks_failed")) != (
                        "full_manifest", count, 0):
                    incomplete.append(stage)
            record("scores_complete" if not incomplete else "finished_with_score_gaps",
                   omni_scoring_returncode=omni_code, incomplete_stages=incomplete)
            return int(bool(incomplete))
        except BaseException as error:
            record("recovery_error", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
