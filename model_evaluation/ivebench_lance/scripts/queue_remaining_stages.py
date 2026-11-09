#!/usr/bin/env python3
"""Resume Long Subject, then run/evaluate/upload Attribute Editing.

This entry point is deliberately independent of an earlier allocation's queue
status.  A dependent Slurm job may start after that allocation times out, so it
revalidates preparation and resumes generation from the already verified video
outputs before invoking the resumable evaluation/upload orchestrator.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LONG = ROOT / "artifacts" / "long_subject"
ATTRIBUTE = ROOT / "artifacts" / "attribute_all"
STATUS = ROOT / "artifacts" / "remaining_stages_status.json"
PYTHON = "/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python"


def write_status(stage: str, state: str = "running", **kwargs: object) -> None:
    value = {"status": state, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(stage: str, command: list[str]) -> None:
    write_status(stage)
    log = STATUS.parent / f"remaining_{stage}.log"
    with log.open("a") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ))
    if result.returncode:
        raise RuntimeError(f"{stage} failed with return code {result.returncode}; see {log}")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def generation_complete(artifact: Path) -> bool:
    return read_json(artifact / "generation_status.json").get("status") == "complete"


def uploaded(artifact: Path, stage: str) -> bool:
    value = read_json(artifact / "orchestration_status.json")
    return value.get("status") == "complete" and value.get("stage") == stage + "_uploaded"


def preparation_complete(stage: str, artifact: Path) -> bool:
    expected_samples = {"long_subject": 43, "attribute_all": 94}
    value = read_json(artifact / "preparation.json")
    return (
        value.get("status") == "complete"
        and value.get("stage") == stage
        and value.get("samples") == expected_samples[stage]
        and (artifact / "manifest.csv").is_file()
        and (artifact / "manifest.csv").stat().st_size > 0
        and (artifact / "generation_jobs.json").is_file()
        and (artifact / "generation_jobs.json").stat().st_size > 0
    )


def prepare(stage: str, artifact: Path) -> None:
    # A successor allocation should resume generation immediately. Preparation
    # hashes tens of GB of immutable model inputs, so repeating it at every
    # allocation boundary only leaves the GPUs idle without strengthening the
    # already-recorded preparation gate.
    if preparation_complete(stage, artifact):
        write_status(stage + "_preparation_already_complete")
        return
    run("prepare_" + stage, [PYTHON, str(SCRIPTS / "prepare_next_stage.py"), stage])


def resume_generation(stage: str, artifact: Path, port: int) -> None:
    if generation_complete(artifact):
        write_status(stage + "_generation_already_complete")
        return
    run("resume_" + stage + "_generation", [
        PYTHON, str(SCRIPTS / "run_generation.py"),
        "--jobs", str(artifact / "generation_jobs.json"), "--master-port", str(port),
    ])
    if not generation_complete(artifact):
        raise RuntimeError(f"{stage} generation did not reach complete status")


def evaluate_upload(stage: str, artifact: Path) -> None:
    if uploaded(artifact, stage):
        write_status(stage + "_already_uploaded")
        return
    run("evaluate_upload_" + stage, [
        PYTHON, str(SCRIPTS / "orchestrate_short_subject.py"),
        "--stage", stage, "--skip-wait",
    ])
    value = read_json(artifact / "orchestration_status.json")
    if value.get("status") != "complete" or value.get("stage") != stage + "_uploaded":
        raise RuntimeError(f"{stage} upload gate not satisfied: {value}")


def main() -> int:
    lock = (STATUS.parent / ".remaining_stages.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        prepare("long_subject", LONG)
        resume_generation("long_subject", LONG, 29940)
        evaluate_upload("long_subject", LONG)
        prepare("attribute_all", ATTRIBUTE)
        resume_generation("attribute_all", ATTRIBUTE, 30040)
        evaluate_upload("attribute_all", ATTRIBUTE)
        write_status("all_agreed_stages_complete", "complete")
        return 0
    except Exception as error:
        write_status("remaining_stages", "failed", error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
