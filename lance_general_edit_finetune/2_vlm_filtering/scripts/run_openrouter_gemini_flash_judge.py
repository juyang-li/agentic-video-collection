#!/usr/bin/env python3
"""Run the provenance-aware pair judge with frozen Gemini 3.8 Flash."""

from __future__ import annotations

import run_openrouter_qwen3_judge as judge


judge.DEFAULT_MODEL = "google/gemini-3.8-flash"
judge.JUDGE_VERSION = "gemini_3_8_flash_provenance_aware_pair_judge"


if __name__ == "__main__":
    judge.main()
