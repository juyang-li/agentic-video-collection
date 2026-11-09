#!/usr/bin/env python3
"""Freeze 30 new, artifact-prone remove/reverse-add pairs for Qwen72B review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STAGE1 = PROJECT_ROOT / "1_specialist_filtering"
RUN1 = STAGE1 / "artifacts/filter_run_v1"
RUN2 = STAGE1 / "artifacts/filter_run_v2"
OUTPUT_ROOT = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_CURRENT = RUN2 / "stage1_next_input.csv"
DEFAULT_EXCLUDED = RUN2 / "stage1_structural_exclusions.csv"
DEFAULT_SCORES = RUN1 / "stage1_quality_available.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "artifact_risk_30.csv"
DEFAULT_SUMMARY = OUTPUT_ROOT / "artifact_risk_30_summary.json"
VERSION = "qwen72b_artifact_risk_30_v1"
SEED = "qwen72b-artifact-risk-20260831-v1"

QUOTAS = {
    "goku_structural_garment": 7,
    "goku_remove_wearable": 8,
    "goku_reverse_add_wearable": 8,
    "reco_remove_wearable_or_person": 7,
}

# Avoid generic "top" because ReCo uses it in phrases such as "top of tower".
WEARABLE_RE = re.compile(
    r"\b(?:wearing|worn|shirt|jacket|sweater|cardigan|hoodie|sweatshirt|"
    r"dress|skirt|pants|trousers|shorts|coat|blouse|tuxedo|corset|outfit|"
    r"garment|jumpsuit|vest|uniform|sunglasses|glasses|cap|hat|headband|"
    r"earring|necklace|bracelet|ring|scarf|shoe|boot|headdress|pantsuit|"
    r"blazer|suspenders)\b",
    re.IGNORECASE,
)
NON_PHOTOREAL_RE = re.compile(
    r"\b(?:cartoon|animated|mascot|robot|puppet|anime|panda|video game|3d character)\b",
    re.IGNORECASE,
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def touched_sample_ids() -> set[str]:
    """Exclude every pair already human-reviewed or scored by an earlier VLM."""
    touched: set[str] = set()
    roots = [STAGE1 / "artifacts", ROOT / "artifacts"]
    evidence_fields = {
        "human_label",
        "model_decision",
        "overall_status",
        "target_status",
        "review_status",
    }
    for root in roots:
        for path in root.rglob("*.csv"):
            try:
                with path.open(encoding="utf-8-sig", newline="") as file:
                    reader = csv.DictReader(file)
                    fields = set(reader.fieldnames or [])
                    if "sample_id" not in fields or not fields & evidence_fields:
                        continue
                    touched.update(
                        row["sample_id"] for row in reader if row.get("sample_id")
                    )
            except (OSError, UnicodeError, csv.Error):
                continue
    return touched


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generated_role(row: dict[str, str]) -> str:
    checked = row.get("quality_checked_roles", "").strip().lower()
    if checked in {"source", "target"}:
        return checked
    if row.get("source_provenance") == "edited":
        return "source"
    return "target"


def category(row: dict[str, str]) -> str:
    instruction = row.get("instruction", "")
    if (
        row.get("dataset") == "goku"
        and "excluded_garment_add_remove" in row.get("structural_reasons", "")
    ):
        return "goku_structural_garment"
    if (
        row.get("dataset") == "goku"
        and row.get("operation") == "remove"
        and WEARABLE_RE.search(instruction)
    ):
        return "goku_remove_wearable"
    if (
        row.get("dataset") == "goku"
        and row.get("operation") == "add"
        and generated_role(row) == "source"
        and WEARABLE_RE.search(instruction)
    ):
        return "goku_reverse_add_wearable"
    if (
        row.get("dataset") == "reco"
        and row.get("operation") == "remove"
        and not NON_PHOTOREAL_RE.search(instruction)
        and (
            WEARABLE_RE.search(instruction)
            or row.get("target_scope") in {"person", "clothes", "accessory"}
        )
    ):
        return "reco_remove_wearable_or_person"
    return ""


def score_risk(
    row: dict[str, str], quality: dict[str, str]
) -> tuple[float, list[str]]:
    reasons = []
    value = 0.0
    if "excluded_garment_add_remove" in row.get("structural_reasons", ""):
        value += 20
        reasons.append("metadata_garment_remove_reversal")
    if row.get("operation") == "remove":
        value += 8
        reasons.append("generated_target_remove")
    elif generated_role(row) == "source":
        value += 8
        reasons.append("generated_source_reverse_add")
    if WEARABLE_RE.search(row.get("instruction", "")):
        value += 8
        reasons.append("wearable_or_body_surface_completion")
    if row.get("target_scope") in {"person", "clothes", "accessory"}:
        value += 3
        reasons.append(f"scope_{row.get('target_scope')}")
    if row.get("stage1_1_decision") == "uncertain":
        value += 6
        reasons.append("stage1_quality_boundary")

    musiq = as_float(quality.get("musiq_p10", ""), 60.0)
    if musiq < 40:
        value += 6
        reasons.append("low_musiq_p10")
    elif musiq < 45:
        value += 4
        reasons.append("borderline_musiq_p10")
    elif musiq < 50:
        value += 2
        reasons.append("soft_musiq_p10")
    dino_jump = as_float(quality.get("dino_max_feature_jump", ""))
    if dino_jump > 0.35:
        value += 4
        reasons.append("high_dino_temporal_jump")
    elif dino_jump > 0.25:
        value += 2
        reasons.append("borderline_dino_temporal_jump")
    freeze = as_float(quality.get("freeze_ratio", ""))
    if freeze > 0.20:
        value += 2
        reasons.append("elevated_freeze_ratio")
    return value, reasons


def tie_break(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    current, current_fields = read_csv(args.current)
    excluded, excluded_fields = read_csv(args.excluded)
    quality_rows, _ = read_csv(args.scores)
    quality_by_sample_role = {
        (row["sample_id"], row["role"]): row for row in quality_rows
    }
    touched = touched_sample_ids()
    candidates: dict[str, list[tuple[float, dict[str, str], dict[str, str], list[str]]]] = {
        key: [] for key in QUOTAS
    }
    for row in [*current, *excluded]:
        if row.get("sample_id") in touched:
            continue
        group = category(row)
        if not group:
            continue
        role = generated_role(row)
        quality = quality_by_sample_role.get((row["sample_id"], role))
        if quality is None:
            continue
        source = STAGE1 / row.get("source_local", "")
        target = STAGE1 / row.get("target_local", "")
        if not source.is_file() or not target.is_file():
            continue
        risk, reasons = score_risk(row, quality)
        candidates[group].append((risk, row, quality, reasons))

    selected: list[dict[str, str]] = []
    for group, quota in QUOTAS.items():
        ranked = sorted(
            candidates[group],
            key=lambda item: (-item[0], tie_break(item[1]["sample_id"])),
        )
        if len(ranked) < quota:
            raise ValueError(f"{group} has {len(ranked)} candidates, needs {quota}")
        for risk, row, quality, reasons in ranked[:quota]:
            selected.append(
                {
                    "survey_version": VERSION,
                    "risk_group": group,
                    "artifact_risk_score": f"{risk:.3f}",
                    "artifact_risk_reasons": "|".join(reasons),
                    "generated_role": generated_role(row),
                    "generated_musiq_p10": quality.get("musiq_p10", ""),
                    "generated_dino_max_feature_jump": quality.get(
                        "dino_max_feature_jump", ""
                    ),
                    "generated_freeze_ratio": quality.get("freeze_ratio", ""),
                    **row,
                }
            )

    if len(selected) != 30 or len({row["sample_id"] for row in selected}) != 30:
        raise ValueError("selection must contain 30 unique samples")
    selected.sort(
        key=lambda row: (-as_float(row["artifact_risk_score"]), tie_break(row["sample_id"]))
    )
    for rank, row in enumerate(selected, 1):
        row["sample_rank"] = str(rank)
    extra_fields = [
        "survey_version",
        "sample_rank",
        "risk_group",
        "artifact_risk_score",
        "artifact_risk_reasons",
        "generated_role",
        "generated_musiq_p10",
        "generated_dino_max_feature_jump",
        "generated_freeze_ratio",
    ]
    source_fields = list(dict.fromkeys([*current_fields, *excluded_fields]))
    write_csv(args.output, selected, [*extra_fields, *source_fields])

    summary = {
        "survey_version": VERSION,
        "selection_seed": SEED,
        "rows": len(selected),
        "excluded_previously_touched_ids": len(touched),
        "risk_groups": dict(Counter(row["risk_group"] for row in selected)),
        "datasets": dict(Counter(row["dataset"] for row in selected)),
        "operations": dict(Counter(row["operation"] for row in selected)),
        "generated_roles": dict(Counter(row["generated_role"] for row in selected)),
        "stage1_1_decisions": dict(
            Counter(row["stage1_1_decision"] for row in selected)
        ),
        "structurally_excluded_rows": sum(
            row.get("structural_decision") == "rejected" for row in selected
        ),
        "selection_policy": (
            "new/unreviewed and not previously VLM-scored; enriched for remove targets, "
            "reverse-add generated sources, garments/accessories/body completion, and "
            "low or boundary Stage1 quality signals"
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
