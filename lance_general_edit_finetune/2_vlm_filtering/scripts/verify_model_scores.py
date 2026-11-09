#!/usr/bin/env python3
"""Fail a job early when a model smoke run did not produce usable scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-2", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=10)
    args = parser.parse_args()
    robot = rows(args.stage1_2)
    qwen = rows(args.qwen)
    if len(robot) != args.expected or any(row.get("technical_status") != "pass" for row in robot):
        raise ValueError("Stage 1.2 smoke did not produce the expected real pass scores")
    if len(qwen) != args.expected or any(row.get("parse_status") != "pass" for row in qwen):
        raise ValueError("Qwen smoke did not produce the expected parsed judgments")
    if any(row.get("model_backend") == "mock" for row in robot + qwen):
        raise ValueError("mock scores cannot pass the production smoke verifier")
    print(f"verified stage1_2={len(robot)} qwen={len(qwen)}")


if __name__ == "__main__":
    main()

