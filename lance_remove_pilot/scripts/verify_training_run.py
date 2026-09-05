#!/usr/bin/env python3
"""Strictly verify a completed Lance removal training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT / "scripts"))

import plot_metrics  # noqa: E402


METRIC_FIELDS = {
    "total_loss",
    "mse_loss",
    "lr",
    "grad_norm",
    "total_samples",
    "total_mse_tokens",
    "max_memory_allocated_mb",
    "step_wall_time_seconds",
}
CHECKPOINT_FILES = {
    "model.safetensors",
    "optimizer.00000-of-00001.pt",
    "scheduler.pt",
    "generation_config.json",
    "llm_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
EXPECTED_DATASETS = {
    "x2v_interleave_local:insave_smoke",
    "x2v_interleave_local:goku_smoke",
}


def fail(message: str) -> None:
    raise SystemExit(f"verification failed: {message}")


def read_csv(path: Path, expected_steps: int) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_steps:
        fail(f"expected {expected_steps} CSV rows, found {len(rows)}")
    if not rows or not METRIC_FIELDS.issubset(rows[0]):
        fail(f"metrics.csv is missing fields: {sorted(METRIC_FIELDS - set(rows[0] if rows else []))}")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(expected_steps)):
        fail(f"CSV steps are not continuous from 0: {steps}")
    for row in rows:
        for field in METRIC_FIELDS:
            value = float(row[field])
            if not math.isfinite(value):
                fail(f"non-finite {field} at step {row['step']}")
        if float(row["mse_loss"]) <= 0:
            fail(f"non-positive MSE loss at step {row['step']}")
        if float(row["grad_norm"]) <= 0:
            fail(f"non-positive gradient norm at step {row['step']}")
        if float(row["total_mse_tokens"]) <= 0:
            fail(f"no MSE target tokens at step {row['step']}")
    return rows


def verify_tensorboard(path: Path, expected_steps: int) -> None:
    if not path.is_dir():
        fail(f"missing TensorBoard directory: {path}")
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    required = {f"train/{field}" for field in METRIC_FIELDS}
    if not required.issubset(tags):
        fail(f"TensorBoard is missing tags: {sorted(required - tags)}")
    for tag in required:
        events = accumulator.Scalars(tag)
        steps = [event.step for event in events]
        if steps != list(range(expected_steps)):
            fail(f"TensorBoard tag {tag} has unexpected steps: {steps}")
        if not all(math.isfinite(event.value) for event in events):
            fail(f"TensorBoard tag {tag} contains a non-finite value")


def verify_dataset_sampling(run_dir: Path, expected_steps: int) -> None:
    path = run_dir / "dataset_samples.csv"
    if not path.is_file():
        fail(f"missing {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = {"step", "dataset_name", "samples"}
    if not rows or not expected_fields.issubset(rows[0]):
        fail("dataset_samples.csv has an unexpected schema")
    for step in range(expected_steps):
        step_rows = [row for row in rows if int(row["step"]) == step]
        names = {row["dataset_name"] for row in step_rows}
        if names != EXPECTED_DATASETS:
            fail(f"step {step} sampled datasets {sorted(names)}, expected {sorted(EXPECTED_DATASETS)}")
        if any(int(row["samples"]) < 1 for row in step_rows):
            fail(f"step {step} contains a non-positive dataset sample count")


def verify_checkpoint(run_dir: Path, expected_steps: int) -> Path:
    checkpoint = run_dir / "checkpoints" / f"{expected_steps - 1:07d}"
    if not checkpoint.is_dir():
        fail(f"missing final checkpoint: {checkpoint}")
    missing = sorted(name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file())
    if missing:
        fail(f"checkpoint is missing files: {missing}")
    for name in CHECKPOINT_FILES:
        if (checkpoint / name).stat().st_size == 0:
            fail(f"empty checkpoint file: {checkpoint / name}")
    return checkpoint


def verify_run(run_dir: Path, expected_steps: int) -> None:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        fail(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("completed_steps") != expected_steps:
        fail(f"summary completed_steps={summary.get('completed_steps')}, expected {expected_steps}")
    if summary.get("exception_count") != 0:
        fail(f"training loop caught {summary.get('exception_count')} exception(s)")

    rows = read_csv(run_dir / "metrics.csv", expected_steps)
    verify_tensorboard(run_dir / "tensorboard", expected_steps)
    verify_dataset_sampling(run_dir, expected_steps)
    checkpoint = verify_checkpoint(run_dir, expected_steps)

    svg_path = run_dir / "loss_curve.svg"
    plot_metrics.render_svg(
        plot_metrics.read_metrics(run_dir / "metrics.csv"),
        svg_path,
        f"Lance removal training: {run_dir.name}",
    )
    svg = svg_path.read_text(encoding="utf-8")
    if "total_loss" not in svg or "mse_loss" not in svg or "<polyline" not in svg:
        fail("loss_curve.svg does not contain both required loss series")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "steps": expected_steps,
                "first_total_loss": float(rows[0]["total_loss"]),
                "last_total_loss": float(rows[-1]["total_loss"]),
                "checkpoint": str(checkpoint),
                "tensorboard": str(run_dir / "tensorboard"),
                "loss_curve": str(svg_path),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()
    if args.expected_steps < 1:
        fail("--expected-steps must be positive")
    verify_run(args.run_dir.resolve(), args.expected_steps)


if __name__ == "__main__":
    main()
