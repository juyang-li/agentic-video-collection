#!/usr/bin/env python3
"""Run the frozen six-model OpenRouter bakeoff with resume and a cost guard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_openrouter_model_bakeoff.py"
EVALUATOR = ROOT / "scripts/evaluate_model_bakeoff.py"
DEFAULT_INPUT = ROOT / "artifacts/model_bakeoff_v1/current_scope_63.csv"
PUBLIC_INPUT = ROOT / "artifacts/model_bakeoff_v1/public_vefx_smoke.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/model_bakeoff_v1/openrouter_private_63"


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    max_tokens: int
    public_cost_usd: float
    provider: str | None = None
    reasoning_effort: str | None = None
    exclude_reasoning: bool = False
    no_temperature: bool = False


CANDIDATES = (
    Candidate(
        "qwen3_5_397b",
        "qwen/qwen3.5-397b-a17b",
        1024,
        0.00177348,
        provider="deepinfra",
        reasoning_effort="none",
        exclude_reasoning=True,
    ),
    Candidate(
        "qwen3vl_235b",
        "qwen/qwen3-vl-235b-a22b-instruct",
        1536,
        0.00142096,
        provider="deepinfra",
    ),
    Candidate(
        "gemini37_flash",
        "google/gemini-3.7-flash",
        1536,
        0.004893,
        reasoning_effort="low",
        exclude_reasoning=True,
    ),
    Candidate(
        "gpt4o_2024_11_20",
        "openai/gpt-4o-2024-11-20",
        1024,
        0.0148025,
    ),
    Candidate(
        "gpt56_sol",
        "openai/gpt-5.6-sol",
        1536,
        0.023101,
        reasoning_effort="low",
        exclude_reasoning=True,
        no_temperature=True,
    ),
    Candidate(
        "gemini31_pro",
        "google/gemini-3.1-pro-preview",
        1536,
        0.028018,
        reasoning_effort="low",
        exclude_reasoning=True,
    ),
)


def command_for(
    candidate: Candidate,
    *,
    input_path: Path,
    output_dir: Path,
    concurrency: int,
    limit: int | None,
    overwrite: bool,
) -> tuple[list[str], Path, Path]:
    score = output_dir / f"{candidate.name}_scores.csv"
    summary = output_dir / f"{candidate.name}_summary.json"
    command = [
        sys.executable,
        str(RUNNER),
        "--input", str(input_path),
        "--output", str(score),
        "--summary", str(summary),
        "--model", candidate.model,
        "--max-tokens", str(candidate.max_tokens),
        "--concurrency", str(concurrency),
    ]
    if score.exists() and not overwrite:
        command.append("--resume")
    if overwrite:
        command.append("--overwrite")
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if candidate.provider:
        command.extend(("--provider-order", candidate.provider))
    if candidate.reasoning_effort:
        command.extend(("--reasoning-effort", candidate.reasoning_effort))
    if candidate.exclude_reasoning:
        command.append("--exclude-reasoning")
    if candidate.no_temperature:
        command.append("--no-temperature")
    return command, score, summary


def summary_cost(path: Path) -> float:
    if not path.is_file():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("total_cost_usd", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", action="append", choices=[item.name for item in CANDIDATES])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency-per-model", type=int, default=4)
    parser.add_argument("--max-run-cost-usd", type=float, default=6.0)
    parser.add_argument("--confirm-project-frame-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    is_public = input_path == PUBLIC_INPUT.resolve()
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    if not is_public and not args.confirm_project_frame_export:
        parser.error(
            "project-data runs require --confirm-project-frame-export after explicit "
            "user authorization; this fails before reading the API key or calling OpenRouter"
        )
    if args.concurrency_per_model < 1:
        parser.error("--concurrency-per-model must be positive")
    if args.max_run_cost_usd <= 0:
        parser.error("--max-run-cost-usd must be positive")
    selected = [
        item for item in CANDIDATES if not args.model or item.name in set(args.model)
    ]
    row_count = args.limit
    if row_count is None:
        import csv
        with input_path.open(encoding="utf-8-sig", newline="") as file:
            row_count = sum(1 for _ in csv.DictReader(file))
    public_projection = sum(item.public_cost_usd * row_count for item in selected)
    plan = {
        "input": str(input_path),
        "public_input": is_public,
        "rows": row_count,
        "models": [item.model for item in selected],
        "public_smoke_linear_projection_usd": public_projection,
        "max_run_cost_usd": args.max_run_cost_usd,
        "project_frame_export_confirmed": bool(
            is_public or args.confirm_project_frame_export
        ),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if public_projection > args.max_run_cost_usd:
        parser.error(
            f"public-smoke projection ${public_projection:.4f} exceeds "
            f"--max-run-cost-usd ${args.max_run_cost_usd:.4f}"
        )
    if args.dry_run:
        return
    if not os.environ.get("OPEN_ROUTER_QWEN", "").strip():
        parser.error("OPEN_ROUTER_QWEN is not set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_paths: list[Path] = []
    total_cost = 0.0
    failures: list[str] = []
    for candidate in selected:
        command, score, summary = command_for(
            candidate,
            input_path=input_path,
            output_dir=args.output_dir,
            concurrency=args.concurrency_per_model,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(f"running={candidate.name} model={candidate.model}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append(candidate.name)
        if score.is_file():
            score_paths.append(score)
        total_cost = sum(summary_cost(path) for path in args.output_dir.glob("*_summary.json"))
        print(f"observed_total_cost_usd={total_cost:.6f}", flush=True)
        if total_cost > args.max_run_cost_usd:
            raise RuntimeError(
                f"observed cost guard exceeded after {candidate.name}: "
                f"${total_cost:.6f} > ${args.max_run_cost_usd:.6f}"
            )

    if score_paths:
        evaluation_command = [
            sys.executable,
            str(EVALUATOR),
            "--manifest", str(input_path),
            "--output", str(args.output_dir / "evaluation.json"),
        ]
        for score in score_paths:
            evaluation_command.extend(("--scores", str(score)))
        subprocess.run(evaluation_command, check=True)
    if failures:
        raise RuntimeError(f"one or more model runs failed: {failures}")


if __name__ == "__main__":
    main()
