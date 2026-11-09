#!/usr/bin/env python3
"""Run the shared pair-audit layout with OpenRouter Qwen2.5-VL-72B."""

from __future__ import annotations

import run_openrouter_qwen3_judge as judge


judge.DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
judge.JUDGE_VERSION = "qwen2_5_vl_72b_openrouter_reconstruction_aware_pair_judge_v6"


if __name__ == "__main__":
    judge.main()
