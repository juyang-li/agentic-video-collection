#!/usr/bin/env python3
"""Start Long Subject only after the Short Subject review is uploaded."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SHORT_STATUS = ROOT / "artifacts" / "short_subject" / "orchestration_status.json"
LONG = ROOT / "artifacts" / "long_subject"
PYTHON = "/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python"


def write_status(stage: str, state: str = "running", **kwargs: object) -> None:
    LONG.mkdir(parents=True, exist_ok=True)
    value = {"status": state, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    temporary = (LONG / "queue_status.json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(LONG / "queue_status.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(stage: str, command: list[str]) -> None:
    write_status(stage)
    log = LONG / f"queue_{stage}.log"
    with log.open("a") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{stage} failed with return code {result.returncode}; see {log}")


def main() -> int:
    LONG.mkdir(parents=True, exist_ok=True)
    lock = (LONG / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while True:
            short = json.loads(SHORT_STATUS.read_text()) if SHORT_STATUS.exists() else {}
            if short.get("status") == "complete" and short.get("stage") == "short_subject_uploaded":
                break
            if short.get("status") == "failed":
                raise RuntimeError(f"Short Subject did not reach upload completion: {short}")
            write_status("waiting_for_short_subject_upload", short_stage=short.get("stage"), short_status=short.get("status"))
            time.sleep(60)
        run("prefetch_next_inputs", [PYTHON, str(SCRIPTS / "prefetch_next_inputs.py"), "--workers", "8"])
        run("prepare_long_subject", [PYTHON, str(SCRIPTS / "prepare_next_stage.py"), "long_subject"])
        run("generate_long_subject", [PYTHON, str(SCRIPTS / "run_generation.py"),
                                      "--jobs", str(LONG / "generation_jobs.json"), "--master-port", "29940"])
        write_status("long_subject_generation_complete", "complete")
        return 0
    except Exception as error:
        write_status("queue_long_subject", "failed", error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
