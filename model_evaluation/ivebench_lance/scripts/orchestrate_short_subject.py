#!/usr/bin/env python3
"""Finish, score, validate, package, and upload Short Subject after generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
SCRIPTS = ROOT / "scripts"
PYTHON = Path("/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python")
REVIEW_PYTHON = ROOT / "runtime" / "review_browser" / "bin" / "python"
VERSION = "ivebench_short_subject91_base_vs_lora_step677_seed42_gemini38flash_temp0_v1"
REVIEW = ARTIFACTS / "review" / VERSION
STATUS = ARTIFACTS / "orchestration_status.json"
STAGE_NAME = "short_subject"
EXPECTED_SAMPLES = 91
OUTPUT_PREFIX = "short_subject"
STAGES = {
    "short_subject": (91, "short_subject", "ivebench_short_subject91_base_vs_lora_step677_seed42_gemini38flash_temp0_v1"),
    "long_subject": (43, "long_subject", "ivebench_long_subject43_base_vs_lora_step677_seed42_gemini38flash_temp0_v1"),
    "attribute_all": (94, "attribute_all", "ivebench_attribute94_base_vs_lora_step677_seed42_gemini38flash_temp0_v1"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def status(stage: str, **kwargs: object) -> None:
    value = {"status": "running", "stage": stage, "updated_at": now(), **kwargs}
    write_json(STATUS, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def log_path(name: str) -> Path:
    path = ARTIFACTS / "logs" / f"orchestrator_{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run(name: str, command: list[str], environment: dict[str, str] | None = None) -> None:
    status(name)
    with log_path(name).open("a") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"stage {name} failed with return code {process.returncode}; see {log_path(name)}")


def start(name: str, command: list[str], environment: dict[str, str] | None = None) -> tuple[subprocess.Popen, object]:
    log = log_path(name).open("a")
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status(name, pid=process.pid)
    return process, log


def wait_process(name: str, process: subprocess.Popen, log: object) -> None:
    returncode = process.wait()
    log.close()
    if returncode:
        raise RuntimeError(f"stage {name} failed with return code {returncode}; see {log_path(name)}")


def wait_for_generation() -> None:
    dead_polls = 0
    while True:
        state_path = ARTIFACTS / "generation_status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status") == "complete":
            break
        if state.get("status") == "failed":
            raise RuntimeError(f"generation failed: {state}")
        pids = [int(pid) for pid in state.get("pids", [])]
        alive = 0
        for pid in pids:
            try:
                os.kill(pid, 0); alive += 1
            except ProcessLookupError:
                pass
        dead_polls = dead_polls + 1 if pids and alive == 0 else 0
        if dead_polls >= 3:
            raise RuntimeError(f"generation state is running but all recorded launchers are dead: {state}")
        status("waiting_for_generation", generation_state=state.get("status", "missing"),
               generation_group=state.get("group"), alive_launchers=alive)
        time.sleep(60)
    counts = {
        kind: len(list((ARTIFACTS / "outputs_raw").glob(f"{OUTPUT_PREFIX}_{kind}_shard*/*.mp4")))
        for kind in ("base", "lora_step677")
    }
    if counts != {"base": EXPECTED_SAMPLES, "lora_step677": EXPECTED_SAMPLES}:
        raise ValueError(f"generation declared complete with wrong output counts: {counts}")
    status("generation_verified", output_counts=counts)


def openrouter_balance(key: str) -> float:
    session = requests.Session(); session.trust_env = False
    response = session.get(
        "https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + key}, timeout=(20, 60))
    response.raise_for_status()
    data = response.json().get("data", {})
    return float(data["total_credits"]) - float(data["total_usage"])


def judge_command(tasks: str, output: str, limit: int | None = None) -> list[str]:
    command = [str(PYTHON), str(SCRIPTS / "judge_gemini.py"), "--tasks", str(ARTIFACTS / tasks),
               "--output", str(ARTIFACTS / output), "--api-key-env", "OPEN_ROUTER_QWEN", "--workers", "2"]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def judge_progress(tasks_file: str, directory: str) -> dict:
    """Return manifest-aware progress and an observed per-completed-task cost.

    A resumed judge directory can already contain the full run even though the
    idempotent preflight is invoked with ``--limit 1``.  Its summary cost is then
    cumulative, not the cost of one preflight request, so treating it as a unit
    cost grossly overestimates the remaining budget.
    """
    total = len(json.loads((ARTIFACTS / tasks_file).read_text()))
    root = ARTIFACTS / directory
    completed = 0
    for path in root.glob("records/*/result.json"):
        try:
            completed += json.loads(path.read_text()).get("status") == "success"
        except (OSError, ValueError, TypeError):
            continue
    if completed > total:
        raise ValueError(f"judge output has more successes than manifest tasks: {directory}")
    receipts = []
    for path in root.glob("records/*/attempt_*.json"):
        try:
            receipts.append(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            continue
    costs = [receipt.get("usage", {}).get("cost") for receipt in receipts]
    reported = sum(value for value in costs if isinstance(value, (int, float)))
    return {
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "reported_cost_usd": reported,
        "average_cost_per_completed_task_usd": reported / completed if completed else None,
    }


def main() -> int:
    global ARTIFACTS, REVIEW, STATUS, VERSION, STAGE_NAME, EXPECTED_SAMPLES, OUTPUT_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="short_subject")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()
    STAGE_NAME = args.stage
    EXPECTED_SAMPLES, OUTPUT_PREFIX, VERSION = STAGES[STAGE_NAME]
    ARTIFACTS = ROOT / "artifacts" / STAGE_NAME
    REVIEW = ARTIFACTS / "review" / VERSION
    STATUS = ARTIFACTS / "orchestration_status.json"
    key = os.environ.get("OPEN_ROUTER_QWEN", "")
    if not key:
        raise RuntimeError("OPEN_ROUTER_QWEN is missing")
    lock = (ARTIFACTS / ".orchestration.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.skip_wait:
            wait_for_generation()
        run("prepare_judge_media", [str(PYTHON), str(SCRIPTS / "prepare_judge_media.py"),
                                    "--stage", STAGE_NAME, "--workers", "4"])

        metric_prep, metric_prep_log = start(
            "prepare_metric_frames", [str(PYTHON), str(SCRIPTS / "prepare_metric_frames.py"),
                                      "--stage", STAGE_NAME, "--workers", "4"])
        run("api_absolute_preflight", judge_command(
            "judge_tasks_absolute.json", "gemini_absolute_temp0_v1", 1))
        run("api_pairwise_preflight", judge_command(
            "judge_tasks_pairwise.json", "gemini_pairwise_temp0_v1", 1))
        absolute_progress = judge_progress("judge_tasks_absolute.json", "gemini_absolute_temp0_v1")
        pairwise_progress = judge_progress("judge_tasks_pairwise.json", "gemini_pairwise_temp0_v1")
        stability_progress = judge_progress("judge_tasks_stability.json", "gemini_stability_temp0_v1")
        absolute_unit = absolute_progress["average_cost_per_completed_task_usd"]
        pairwise_unit = pairwise_progress["average_cost_per_completed_task_usd"]
        if absolute_unit is None or pairwise_unit is None:
            raise RuntimeError("real-media preflights did not produce usable cost observations")
        available = openrouter_balance(key)
        remaining_estimate = (
            absolute_unit * (absolute_progress["remaining"] + stability_progress["remaining"])
            + pairwise_unit * pairwise_progress["remaining"]
        )
        budget = {
            "checked_at": now(), "balance_usd": available,
            "absolute_observed_unit_cost_usd": absolute_unit,
            "pairwise_observed_unit_cost_usd": pairwise_unit,
            "absolute_progress": absolute_progress,
            "pairwise_progress": pairwise_progress,
            "stability_progress": stability_progress,
            "remaining_point_estimate_usd": remaining_estimate, "required_with_25_percent_safety_usd": remaining_estimate * 1.25,
            "sufficient_with_safety": available >= remaining_estimate * 1.25,
        }
        write_json(ARTIFACTS / "api_budget_after_preflight.json", budget)
        status("api_budget_checked", **budget)
        if not budget["sufficient_with_safety"]:
            wait_process("prepare_metric_frames", metric_prep, metric_prep_log)
            run("local_metrics_budget_fallback", [str(PYTHON), str(SCRIPTS / "run_local_metrics.py"),
                                                  "--stage", STAGE_NAME, "--python", str(PYTHON)])
            raise RuntimeError(f"OpenRouter balance is insufficient after real-media preflights: {budget}")

        absolute, absolute_log = start(
            "api_absolute_full", judge_command("judge_tasks_absolute.json", "gemini_absolute_temp0_v1"))
        wait_process("prepare_metric_frames", metric_prep, metric_prep_log)
        local, local_log = start(
            "local_metrics", [str(PYTHON), str(SCRIPTS / "run_local_metrics.py"),
                              "--stage", STAGE_NAME, "--python", str(PYTHON)])
        wait_process("api_absolute_full", absolute, absolute_log)
        run("api_pairwise_full", judge_command("judge_tasks_pairwise.json", "gemini_pairwise_temp0_v1"))
        run("api_stability", judge_command("judge_tasks_stability.json", "gemini_stability_temp0_v1"))
        wait_process("local_metrics", local, local_log)

        run("aggregate", [str(PYTHON), str(SCRIPTS / "aggregate_short_subject.py"), "--stage", STAGE_NAME])
        run("build_review", [str(PYTHON), str(SCRIPTS / "build_short_subject_review.py"),
                             "--stage", STAGE_NAME, "--workers", "4"])
        browser_environment = dict(os.environ)
        library = ROOT / "runtime" / "debs" / "extracted" / "usr" / "lib" / "x86_64-linux-gnu"
        browser_environment["LD_LIBRARY_PATH"] = str(library) + (":" + browser_environment["LD_LIBRARY_PATH"] if browser_environment.get("LD_LIBRARY_PATH") else "")
        browser_environment["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / "runtime" / "ms-playwright")
        run("browser_acceptance", [str(REVIEW_PYTHON), str(SCRIPTS / "test_short_subject_review.py"), str(REVIEW)], browser_environment)
        run("package_review", [str(PYTHON), str(SCRIPTS / "build_short_subject_review.py"),
                               "--stage", STAGE_NAME, "--output", str(REVIEW), "--package-only"])

        archive = REVIEW.with_suffix(".zip")
        receipt = archive.with_suffix(".sha256.json")
        remote = "/apps/bypy/lance_review/" + archive.name
        upload = ROOT.parents[1] / "lance_general_edit_finetune" / "1_specialist_filtering" / "scripts" / "upload_baidu_xpan.py"
        verify = ROOT.parents[1] / "model_evaluation" / "lance_base" / "scripts" / "verify_baidu_review.py"
        run("upload_review", [str(PYTHON), str(upload), "--local", str(archive), "--remote", remote,
                              "--state", str(archive.with_suffix(".baidu_upload.json"))])
        run("verify_remote", [str(PYTHON), str(verify), str(archive)])
        run("upload_receipt", [str(PYTHON), str(upload), "--local", str(receipt),
                               "--remote", str(PurePosixPath(remote).with_suffix(".sha256.json")),
                               "--state", str(archive.with_suffix(".sha256.baidu_upload.json"))])
        final = {
            "status": "complete", "stage": STAGE_NAME + "_uploaded", "finished_at": now(),
            "archive": str(archive), "receipt": json.loads(receipt.read_text()),
        }
        write_json(STATUS, final)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "failed", "stage": "orchestration", "failed_at": now(),
                   "error_type": type(error).__name__, "error": str(error)}
        write_json(STATUS, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
