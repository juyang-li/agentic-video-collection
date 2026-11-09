#!/usr/bin/env python3
"""Run the frozen provenance-aware v12 prompt with Qwen2.5-VL-72B."""

from __future__ import annotations

import sys
from pathlib import Path


STAGE2_SCRIPTS = Path(__file__).resolve().parents[1] / "2_vlm_filtering" / "scripts"
sys.path.insert(0, str(STAGE2_SCRIPTS))

import run_openrouter_qwen3_judge as judge  # noqa: E402


judge.DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
judge.DEFAULT_PROMPT = (
    Path(__file__).resolve().parents[1]
    / "2_vlm_filtering"
    / "judge_prompt_timecard_v12_provenance_aware.txt"
)
judge.JUDGE_VERSION = "qwen2_5_vl_72b_provenance_aware_pair_judge_prompt_v12"


if __name__ == "__main__":
    judge.main()
