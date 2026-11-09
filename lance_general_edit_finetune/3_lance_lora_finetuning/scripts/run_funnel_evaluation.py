#!/usr/bin/env python3
"""Resume the fixed four-GPU evaluation funnel using the base's exact judge protocol."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from judge_spatial import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
BASE = ROOT.parents[1] / "model_evaluation/lance_base"
JUDGE = BASE.parent / "lance_tools/judge_spatial_openrouter.py"
STAGES = ("screen", "remaining", "test", "omni")
COUNTS = {"screen": 144, "remaining": 147, "test": 204, "omni": 240}


def verify_contract(evaluation: Path) -> dict:
    contract = json.loads((evaluation / "judge_contract.json").read_text())
    profile = ROOT / "configs/lance_4k_r64_seed42_e2_v1.json"
    if sha256_file(profile) != contract["training_profile_sha256"]:
        raise ValueError("training profile differs from the evaluation contract")
    spec = importlib.util.spec_from_file_location("shared_spatial_judge", JUDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if contract["judge_route"] != "vertex" or contract["provider_route"] != "google-vertex/global":
        raise ValueError("this funnel is fixed to the base's complete Vertex v2 comparison")
    module.configure_route(contract["judge_route"])
    metadata, _rules = module.protocol()
    if module.digest(metadata) != contract["canonical_protocol_sha256"]:
        raise ValueError("shared judge protocol changed")
    recorded = json.loads((BASE / "gemini38_low_vertex_v2/protocol.json").read_text())
    if metadata != recorded:
        raise ValueError("base and finetune judge protocols differ")
    return contract


def scores_root(evaluation: Path, stage: str) -> Path:
    return evaluation / f"{stage}_scores_vertex_v2"


def require_api_authorization(evaluation: Path, contract: dict) -> None:
    # The hold cannot be cleared by changing routes or deleting the local record.
    # The record must reflect actual user approval and normal tool review.
    hold_path = evaluation / "api_authorization_hold.json"
    if not hold_path.exists():
        raise PermissionError("missing video API authorization record after auto-review rejection")
    hold = json.loads(hold_path.read_text())
    if hold["status"] != "explicit_user_approval_obtained":
        raise PermissionError(
            "Video API upload was rejected by auto-review; explicit user approval for the "
            "source/generated MP4 payload and OpenRouter/Google Vertex destination is still needed."
        )
    if (hold.get("approved_provider_route"), hold.get("approved_model")) != (
        contract["provider_route"], contract["model"]
    ):
        raise PermissionError("video API approval does not cover this provider route and model")


def stage_commands(evaluation: Path, stage: str) -> list[list[str]]:
    python = sys.executable
    return [
        [python, str(ROOT / "scripts/prepare_evaluation_configs.py"),
         "--stage", stage, "--evaluation-root", str(evaluation)],
        [python, str(ROOT / "scripts/run_evaluation_jobs.py"),
         "--jobs", str(evaluation / f"{stage}_jobs.json")],
        [python, str(JUDGE), "--route", "vertex",
         "--tasks", str(evaluation / f"{stage}_generation_tasks.json"),
         "--output", str(scores_root(evaluation, stage)),
         "--api-key-env", "OPEN_ROUTER_QWEN", "--workers", "2"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--through", choices=STAGES, default="omni")
    parser.add_argument("--plan", action="store_true", help="read-only contract and command inspection")
    args = parser.parse_args()
    evaluation = args.evaluation_root.resolve()
    contract = verify_contract(evaluation)
    stages = STAGES[:STAGES.index(args.through) + 1]
    if args.plan:
        print(json.dumps({"protocol_sha256": contract["canonical_protocol_sha256"],
                          "stages": {s: stage_commands(evaluation, s) for s in stages}}, indent=2))
        return 0
    if not os.environ.get(contract["api_key_environment"]):
        raise RuntimeError(f"missing {contract['api_key_environment']}")

    child = None
    interrupted = False

    def terminate(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate)

    def run(command: list[str]) -> None:
        nonlocal child
        if interrupted:
            raise InterruptedError("evaluation interrupted")
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        try:
            returncode = child.wait()
        finally:
            if child.poll() is None:
                terminate(signal.SIGTERM, None)
                child.wait(timeout=45)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    status_path = evaluation / "pipeline_status.json"
    with (evaluation / ".funnel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage in stages:
                write_json(status_path, {"status": "running", "stage": stage,
                                         "protocol_sha256": contract["canonical_protocol_sha256"]})
                for command in stage_commands(evaluation, stage):
                    if command[1] == str(JUDGE):
                        require_api_authorization(evaluation, contract)
                    run(command)
                summary = json.loads((scores_root(evaluation, stage) / "summary.json").read_text())
                if (summary.get("scope"), summary["tasks_succeeded"], summary["tasks_failed"]) != (
                    "full_manifest", COUNTS[stage], 0
                ):
                    raise ValueError(f"{stage} scoring did not complete the full manifest")
                if stage == "screen":
                    decision = evaluation / "promotion_decision.json"
                    scores = scores_root(evaluation, "screen") / "scores.csv"
                    if decision.exists():
                        frozen = json.loads(decision.read_text())
                        if frozen["scores_sha256"] != sha256_file(scores):
                            raise ValueError("screen scores changed after winner was frozen")
                    else:
                        run([sys.executable, str(ROOT / "scripts/decide_funnel_promotion.py"),
                             "--scores", str(scores),
                             "--screen-manifest", str(evaluation / "validation_screen48.csv"),
                             "--remaining-manifest", str(evaluation / "validation_remaining147.csv"),
                             "--decision-output", str(decision),
                             "--tasks-output", str(evaluation / "promotion_tasks.csv")])
            # Packaging comes after the requested scoring stages, never as a gate.
            stage = "review_package"
            review = evaluation / "review_val48"
            archive_receipt = review.with_suffix(".zip.sha256.json")
            if not archive_receipt.exists():
                run([sys.executable, str(ROOT / "scripts/build_model_audit.py"),
                     "--scores", str(scores_root(evaluation, "screen") / "scores.csv"), "--output", str(review)])
            receipt = json.loads(archive_receipt.read_text())
            if receipt["sha256"] != sha256_file(review.with_suffix(".zip")):
                raise ValueError("review archive checksum changed")
            write_json(status_path, {"status": "complete_through_requested_stage",
                                     "through": args.through,
                                     "protocol_sha256": contract["canonical_protocol_sha256"]})
        except BaseException as error:
            write_json(status_path, {"status": "interrupted" if interrupted else "failed",
                                     "stage": stage, "error_type": type(error).__name__})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
