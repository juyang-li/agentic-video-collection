#!/usr/bin/env python3
"""Run Qwen2.5-VL-72B with scale-calibrated reconstruction checks."""

from __future__ import annotations

import run_openrouter_qwen3_judge as judge


judge.DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
judge.DEFAULT_PROMPT = judge.ROOT / "judge_prompt_timecard_v10.txt"
judge.JUDGE_VERSION = "qwen2_5_vl_72b_openrouter_scale_calibrated_pair_judge_v10"
judge.AUDIT_TASK_ONLY_PRESENTATION = True


if __name__ == "__main__":
    judge.main()
