#!/usr/bin/env python3
"""Render training loss curves from metrics.csv as a dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


COLORS = {"total_loss": "#2563eb", "mse_loss": "#dc2626"}


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"step", "total_loss", "mse_loss"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"metrics must contain {sorted(required)}")
    parsed: list[dict[str, float]] = []
    previous_step: int | None = None
    for row in rows:
        step = int(row["step"])
        values = {
            "step": float(step),
            "total_loss": float(row["total_loss"]),
            "mse_loss": float(row["mse_loss"]),
        }
        if previous_step is not None and step <= previous_step:
            raise ValueError("metrics steps must be strictly increasing")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite metric at step {step}")
        parsed.append(values)
        previous_step = step
    return parsed


def render_svg(rows: list[dict[str, float]], output: Path, title: str) -> None:
    width, height = 960, 540
    left, right, top, bottom = 92, 32, 62, 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    steps = [row["step"] for row in rows]
    values = [row[key] for row in rows for key in COLORS]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.5)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">{title}</text>',
    ]
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="monospace" font-size="13" fill="#374151">{value:.4g}</text>'
        )
    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        value = x_min + ratio * (x_max - x_min)
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="monospace" font-size="13" fill="#374151">{value:.4g}</text>'
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">Optimizer step</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 24 {top + plot_height / 2})">Loss</text>',
        ]
    )
    for series_index, (key, color) in enumerate(COLORS.items()):
        points = " ".join(f"{sx(row['step']):.2f},{sy(row[key]):.2f}" for row in rows)
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for row in rows:
            svg.append(
                f'<circle cx="{sx(row["step"]):.2f}" cy="{sy(row[key]):.2f}" r="3.2" fill="{color}"/>'
            )
        legend_x = left + 18 + series_index * 190
        legend_y = top + 20
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="sans-serif" font-size="14">{key}</text>'
        )
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Lance removal training loss")
    args = parser.parse_args()
    rows = read_metrics(args.metrics)
    render_svg(rows, args.output, args.title)
    print(f"wrote={args.output} points={len(rows)}")


if __name__ == "__main__":
    main()
