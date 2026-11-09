#!/usr/bin/env python3
"""Compare noisy training losses with rolling means and stable window statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]


def read_run(path: Path) -> tuple[list[float], list[float], dict[int, str]]:
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError(f"non-contiguous steps: {path}")
    losses = [float(row["total_loss"]) for row in rows]
    grads = [float(row["grad_norm"]) for row in rows]
    if not losses or not all(math.isfinite(value) for value in losses + grads):
        raise ValueError(f"empty or non-finite metrics: {path}")
    with (path / "dataset_samples.csv").open(encoding="utf-8", newline="") as handle:
        dataset_rows = list(csv.DictReader(handle))
    datasets = {}
    for row in dataset_rows:
        step = int(row["step"])
        if step in datasets:
            raise ValueError("Expected one real 3-second sample per packed batch")
        datasets[step] = row["dataset_name"]
    if set(datasets) != set(steps):
        raise ValueError(f"dataset sample steps do not match metrics: {path}")
    return losses, grads, datasets


def rolling(values: list[float], window: int) -> list[float]:
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def summarize(losses: list[float], grads: list[float], datasets: dict[int, str]) -> dict[str, object]:
    window = min(100, len(losses) // 2)
    first = statistics.fmean(losses[:window])
    last = statistics.fmean(losses[-window:])
    x_mean = (len(losses) - 1) / 2
    y_mean = statistics.fmean(losses)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(losses)) / sum(
        (x - x_mean) ** 2 for x in range(len(losses))
    )
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for step, loss in enumerate(losses):
        by_dataset[datasets[step]].append(loss)
    return {
        "steps": len(losses),
        "first_window_steps": window,
        "first_window_mean": first,
        "last_window_mean": last,
        "last_vs_first_percent": (last / first - 1) * 100,
        "raw_mean": y_mean,
        "raw_std": statistics.pstdev(losses),
        "raw_min": min(losses),
        "raw_max": max(losses),
        "linear_slope_per_step": slope,
        "grad_norm_min": min(grads),
        "grad_norm_max": max(grads),
        "grad_norm_mean": statistics.fmean(grads),
        "dataset_loss": {
            name: {"steps": len(values), "mean": statistics.fmean(values)}
            for name, values in sorted(by_dataset.items())
        },
    }


def render_svg(series: list[tuple[str, list[float]]], output: Path, window: int) -> None:
    width, height = 1080, 600
    left, right, top, bottom = 90, 38, 72, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = [value for _, values in series for value in values]
    y_min, y_max = min(all_values), max(all_values)
    padding = max((y_max - y_min) * 0.08, 0.005)
    y_min, y_max = y_min - padding, y_max + padding
    x_max = max(len(values) for _, values in series) - 1

    def sx(value: float) -> float:
        return left + value / x_max * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Lance removal loss: raw and {window}-step rolling mean</text>',
    ]
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-family="monospace" font-size="13">{value:.3f}</text>')
    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        value = ratio * x_max
        svg.append(f'<text x="{x:.2f}" y="{top+plot_height+28}" text-anchor="middle" font-family="monospace" font-size="13">{value:.0f}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#111827"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#111827"/>')
    for index, (label, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        raw_points = " ".join(f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(values))
        smooth = rolling(values, window)
        smooth_points = " ".join(f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(smooth))
        svg.append(f'<polyline points="{raw_points}" fill="none" stroke="{color}" stroke-opacity="0.18" stroke-width="1"/>')
        svg.append(f'<polyline points="{smooth_points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_y = 54
        legend_x = left + index * 260
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+36}" y="{legend_y+5}" font-family="sans-serif" font-size="14">{label}</text>')
    svg.append(f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="15">Optimizer step</text>')
    svg.append(f'<text x="22" y="{top+plot_height/2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 22 {top+plot_height/2})">MSE loss</text>')
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()

    plot_series = []
    report = {"rolling_window": args.window, "runs": {}}
    for item in args.run:
        label, path_text = item.split("=", 1)
        path = Path(path_text)
        losses, grads, datasets = read_run(path)
        report["runs"][label] = summarize(losses, grads, datasets)
        plot_series.append((label, losses))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render_svg(plot_series, args.svg, args.window)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
