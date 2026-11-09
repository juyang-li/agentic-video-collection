#!/usr/bin/env python3
"""Launch the CoinVE-4K and Mixed-2K+2K profiles as two independent 4-GPU jobs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess


STAGE_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = STAGE_ROOT / "scripts/run_general_edit_lora.py"
PROFILES = (
    ("coinve4k", STAGE_ROOT / "configs/lance_coinve_4k_r64_seed42_e2_v1.json", "0,1,2,3", 29652),
    ("mixed2k2k", STAGE_ROOT / "configs/lance_mixed_2k2k_r64_seed42_e2_v1.json", "4,5,6,7", 29662),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-label", default="")
    parser.add_argument("--resume-from", default="none")
    parser.add_argument("--skip-base-hash", action="store_true")
    args = parser.parse_args()
    if bool(args.smoke_steps) != bool(args.smoke_label):
        parser.error("smoke steps and label must be supplied together")
    processes = []

    def terminate(signum: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, terminate)
    label = f"smoke_{args.smoke_label}" if args.smoke_steps else "formal"
    log_root = STAGE_ROOT / "logs/coinve_ablation_v1"
    log_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name, profile, devices, port in PROFILES:
            command = [
                str(PYTHON), str(RUNNER), "--profile", str(profile),
                "--resume-from", args.resume_from, "--master-port", str(port),
            ]
            if args.smoke_steps:
                command.extend(["--smoke-steps", str(args.smoke_steps), "--smoke-label", args.smoke_label])
            if args.skip_base_hash:
                command.append("--skip-base-hash")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = devices
            log = stack.enter_context((log_root / f"{name}_{label}.log").open("a"))
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            print(f"started {name} pid={process.pid} gpus={devices} log={log.name}", flush=True)
        returncodes = [process.wait() for process in processes]
    if any(returncodes):
        print(f"dual launch failed: returncodes={returncodes}", flush=True)
        return 1
    print(f"dual launch complete: {label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
