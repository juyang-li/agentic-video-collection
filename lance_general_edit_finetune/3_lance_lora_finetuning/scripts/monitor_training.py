#!/usr/bin/env python3
"""Convert Lance step logs into append-only CSV and TensorBoard scalars."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from torch.utils.tensorboard import SummaryWriter


STEP = re.compile(
    r"\(step=(?P<step>\d+)\) mse=(?P<mse>[\d.eE+-]+), "
    r"targets=(?P<targets>\d+), histories=(?P<histories>\d+), "
    r"grad_norm=(?P<grad_norm>[\d.eE+-]+), lr=(?P<lr>[\d.eE+-]+), .*"
    r"step_time=(?P<step_time>[\d.eE+-]+)s"
)
GPU_FIELDS = (
    "gpu_memory_used_mib_max", "gpu_utilization_mean", "gpu_temperature_max",
    "gpu_power_draw_w_sum",
)
FIELDS = (
    "step", "mse", "targets", "histories", "grad_norm", "lr", "step_time_sec",
    "mse_ma25", "mse_ma50", "grad_norm_ma25", "step_time_ma25_sec", "eta_hours",
    *GPU_FIELDS,
)


def gpu_metrics() -> dict[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        rows = [[float(value.strip()) for value in line.split(",")] for line in output.splitlines()]
    except Exception:
        return {key: float("nan") for key in GPU_FIELDS}
    return {
        "gpu_memory_used_mib_max": max(row[0] for row in rows),
        "gpu_utilization_mean": sum(row[1] for row in rows) / len(rows),
        "gpu_temperature_max": max(row[2] for row in rows),
        "gpu_power_draw_w_sum": sum(row[3] for row in rows),
    }


def parse(line: str) -> dict[str, float] | None:
    match = STEP.search(line)
    if not match:
        return None
    value = {key: float(raw) for key, raw in match.groupdict().items()}
    value["step"] = int(value["step"])
    value["targets"] = int(value["targets"])
    value["histories"] = int(value["histories"])
    value["step_time_sec"] = value.pop("step_time")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--total-steps", type=int, default=902)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be positive")
    existing: set[int] = set()
    pending_gpu_samples: list[dict[str, float]] = []
    recent_mse: deque[float] = deque(maxlen=50)
    recent_grad: deque[float] = deque(maxlen=25)
    recent_time: deque[float] = deque(maxlen=25)
    if args.csv.is_file():
        with args.csv.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"existing monitor CSV schema differs: {args.csv}")
            rows = list(reader)
        existing = {int(row["step"]) for row in rows}
        ordered_rows = sorted(rows, key=lambda value: int(value["step"]))
        for row in ordered_rows[-50:]:
            recent_mse.append(float(row["mse"]))
        for row in ordered_rows[-25:]:
            recent_grad.append(float(row["grad_norm"]))
            recent_time.append(float(row["step_time_sec"]))
    mode = "a" if args.csv.is_file() else "w"
    with args.log.open(encoding="utf-8", errors="replace") as source, args.csv.open(
        mode, encoding="utf-8", newline=""
    ) as output, SummaryWriter(str(args.tensorboard_dir)) as tensorboard:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        while True:
            line = source.readline()
            if not line:
                if not args.follow:
                    break
                pending_gpu_samples.append(gpu_metrics())
                time.sleep(args.poll_seconds)
                continue
            metrics = parse(line)
            if metrics is None or metrics["step"] in existing:
                continue
            recent_mse.append(metrics["mse"])
            recent_grad.append(metrics["grad_norm"])
            recent_time.append(metrics["step_time_sec"])
            metrics["mse_ma25"] = sum(list(recent_mse)[-25:]) / min(25, len(recent_mse))
            metrics["mse_ma50"] = sum(recent_mse) / len(recent_mse)
            metrics["grad_norm_ma25"] = sum(recent_grad) / len(recent_grad)
            metrics["step_time_ma25_sec"] = sum(recent_time) / len(recent_time)
            metrics["eta_hours"] = max(0, args.total_steps - metrics["step"]) * (
                metrics["step_time_ma25_sec"] / 3600.0
            )
            # Sample during computation: GPUs often appear idle at the step-end
            # barrier. Historical backfill has no observable GPU telemetry.
            for key in GPU_FIELDS:
                values = [sample[key] for sample in pending_gpu_samples if math.isfinite(sample[key])]
                metrics[key] = (
                    (max(values) if key.endswith("_max") else sum(values) / len(values))
                    if values else float("nan")
                )
            pending_gpu_samples.clear()
            writer.writerow(metrics)
            output.flush()
            step = int(metrics["step"])
            for key, value in metrics.items():
                if key != "step":
                    tensorboard.add_scalar(f"train/{key}", value, step)
            tensorboard.flush()
            existing.add(step)
            if args.status_json is not None:
                args.status_json.parent.mkdir(parents=True, exist_ok=True)
                status = {
                    "schema_version": "lance_training_live_status_v1",
                    "updated_at_unix": time.time(),
                    "step": step,
                    "total_steps": args.total_steps,
                    "progress_fraction": step / args.total_steps,
                    "metrics": {
                        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                        for key, value in metrics.items()
                        if key != "step"
                    },
                }
                temporary = args.status_json.with_suffix(args.status_json.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(temporary, args.status_json)


if __name__ == "__main__":
    main()
