#!/usr/bin/env python3
"""Summarize successful OpenRouter public-demo compatibility checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/model_bakeoff_v1"
OUTPUT = RUN / "public_api_smoke_summary.json"
SCORES = [
    RUN / "public_smoke_qwen3_5_397b_scores.csv",
    RUN / "public_smoke_qwen3vl_235b_deepinfra_scores.csv",
    RUN / "public_smoke_gemini37_flash_scores.csv",
    RUN / "public_smoke_gemini31_pro_scores.csv",
    RUN / "public_smoke_gpt56_sol_scores.csv",
    RUN / "public_smoke_gpt4o_scores.csv",
]


def main() -> None:
    models = []
    for path in SCORES:
        with path.open(encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file))
        if row["parse_status"] != "pass":
            raise ValueError(f"public smoke is not successful: {path}: {row['error']}")
        cost = float(row["cost_usd"])
        models.append({
            "model_id": row["model_id"],
            "provider": row["provider"],
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "reasoning_tokens": int(row["reasoning_tokens"] or 0),
            "api_elapsed_seconds": float(row["api_elapsed_seconds"]),
            "cost_usd_one_public_pair": cost,
            "linear_cost_projection_63_pairs_usd": cost * 63,
            "public_demo_decision_not_an_accuracy_label": row["model_decision"],
        })
    payload = {
        "public_input": "VEFX-Bench official object-removal demo",
        "layout": "four aligned time cards; each contains FULL pair + tight ROI pair",
        "privacy_scope": "Only upstream public demo frames and its public instruction were sent.",
        "successful_models": len(models),
        "models": models,
        "linear_total_projection_63_pairs_usd": sum(
            model["linear_cost_projection_63_pairs_usd"] for model in models
        ),
        "projection_warning": (
            "A one-pair linear estimate is planning evidence only; image tokenization and "
            "reasoning length vary by sample."
        ),
        "accuracy_warning": (
            "The public demo has no local human usable/unusable label, so its decisions "
            "must not be used to rank model accuracy."
        ),
        "failed_endpoint_diagnostics": [
            {
                "model_id": "qwen/qwen3.5-397b-a17b",
                "provider": "DeepInfra",
                "cause": "five-image request exceeded provider limit of four",
                "resolution": "use the common four-card API layout",
            },
            {
                "model_id": "qwen/qwen3.5-397b-a17b",
                "provider": "DeepInfra",
                "cause": "default reasoning consumed the 768-token completion budget",
                "resolution": "reasoning=none, exclude=true",
            },
            {
                "model_id": "qwen/qwen3-vl-235b-a22b-instruct",
                "provider": "Venice",
                "cause": "structured response contained whitespace instead of JSON",
                "resolution": "pin DeepInfra, which returned valid strict JSON",
            },
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
