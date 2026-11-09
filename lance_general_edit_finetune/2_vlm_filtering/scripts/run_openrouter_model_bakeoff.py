#!/usr/bin/env python3
"""Run one OpenRouter model with the frozen v6 prompt and visual layout."""

from __future__ import annotations

import run_openrouter_qwen3_judge as judge


# Hide the immutable training-direction wording after provenance routing.  The
# model sees one explicit REFERENCE -> GENERATED audit task, which is the v6
# presentation that avoids contradictory add/remove verbs on inverse pairs.
judge.AUDIT_TASK_ONLY_PRESENTATION = True
judge.JUDGE_VERSION = "openrouter_model_bakeoff_task_only_timecard_v6"
judge.DEFAULT_CARD_COUNT = 4
judge.DEFAULT_PROMPT = judge.ROOT / "judge_prompt_timecard_v6_fourcard.txt"
judge.ROI_METHOD_VERSION = "paired_tight_change_roi_four_aligned_v1"
judge.VISUAL_LAYOUT_VERSION = "four_aligned_full_tight_roi_timecards_v1"


if __name__ == "__main__":
    judge.main()
