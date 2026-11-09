#!/usr/bin/env python3
"""Build a frozen, mostly non-Ego holdout for artifact-focused Qwen3 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RUN = ROOT / "artifacts/qwen3_openrouter_v1"
DEFAULT_HOLDOUT = RUN / "calibration_holdout.csv"
DEFAULT_REVIEW = (
    PROJECT_ROOT
    / "1_specialist_filtering/artifacts/review_results"
    / "specialist_error_survey_460_reco_v3_review.csv"
)
DEFAULT_OUTPUT = RUN / "artifact_focus_eval_38.csv"
DEFAULT_SUMMARY = RUN / "artifact_focus_eval_38_summary.json"
SEED = "qwen3-artifact-focus-20260831-v1"

# These are explicit human annotations of the failure modes this experiment targets.
# geometry_human is intentionally not included: it is useful, but is not evidence for
# texture breaks, masking/inpainting failures, ghosting, or temporal residue.
FOCUS_TAGS = {
    "local_blur",
    "local_flicker",
    "mask_seam",
    "repair_hole",
    "smear_ghost",
    "texture_break",
    "texture_discontinuity",
}
TOLERATED_ARTIFACT_IDS = {
    "goku_remove-12d6e601b5fe",
    "reco-add-adb63ba9145d824d87d7",
}
CONTROL_QUOTAS = {
    ("goku", "add"): 5,
    ("goku", "remove"): 5,
    ("reco", "add"): 2,
    ("reco", "remove"): 3,
    ("reco", "replace"): 3,
    ("egoedit", "replace"): 2,
}
REVIEW_FIELDS = [
    "target_status",
    "target_defects",
    "target_notes",
    "source_status",
    "source_defects",
    "source_notes",
    "pair_status",
    "pair_errors",
    "pair_notes",
]
CLEAN_EMPTY_FIELDS = [
    "target_defects",
    "target_notes",
    "source_defects",
    "source_notes",
    "pair_errors",
    "pair_notes",
]
EXPERIMENT_FIELDS = [
    "focus_group",
    "focus_defect_ground_truth",
    "focus_defect_tags",
    "focus_human_evidence",
    *[f"review_{field}" for field in REVIEW_FIELDS],
]


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


def defect_tags(review: dict[str, str]) -> set[str]:
    value = "|".join(
        (review.get("target_defects", ""), review.get("source_defects", ""))
    )
    return {tag.strip() for tag in value.split("|") if tag.strip()}


def evidence(review: dict[str, str]) -> str:
    values = []
    for field in (
        "target_defects",
        "target_notes",
        "source_defects",
        "source_notes",
        "pair_errors",
        "pair_notes",
    ):
        value = review.get(field, "").strip()
        if value:
            values.append(f"{field}={value}")
    return "; ".join(values)


def deterministic_order(row: dict[str, str]) -> str:
    return hashlib.sha256(f"{SEED}|{row['sample_id']}".encode()).hexdigest()


def is_clean_control(row: dict[str, str], review: dict[str, str]) -> bool:
    return (
        row.get("human_label") == "usable"
        and review.get("target_status") == "pass"
        and review.get("source_status") == "pass"
        and review.get("pair_status") == "pass"
        and not any(review.get(field, "").strip() for field in CLEAN_EMPTY_FIELDS)
    )


def annotate(
    row: dict[str, str], review: dict[str, str], group: str
) -> dict[str, str]:
    tags = sorted(defect_tags(review) & FOCUS_TAGS)
    output = dict(row)
    output.update(
        {
            "focus_group": group,
            "focus_defect_ground_truth": "1" if tags else "0",
            "focus_defect_tags": "|".join(tags),
            "focus_human_evidence": evidence(review),
        }
    )
    output.update({f"review_{field}": review.get(field, "") for field in REVIEW_FIELDS})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    holdout, holdout_fields = read_csv(args.holdout)
    reviews, _ = read_csv(args.review)
    reviews_by_id = {row["sample_id"]: row for row in reviews}
    if len(reviews_by_id) != len(reviews):
        raise ValueError("review CSV contains duplicate sample_id")
    missing_reviews = sorted(
        row["sample_id"] for row in holdout if row["sample_id"] not in reviews_by_id
    )
    if missing_reviews:
        raise ValueError(f"holdout rows missing human review: {missing_reviews}")

    hard = [row for row in holdout if row.get("human_label") == "unusable"]
    borderline_focus = [
        row
        for row in holdout
        if row.get("human_label") == "uncertain"
        and defect_tags(reviews_by_id[row["sample_id"]]) & FOCUS_TAGS
    ]
    tolerated = [row for row in holdout if row["sample_id"] in TOLERATED_ARTIFACT_IDS]
    missing_tolerated = TOLERATED_ARTIFACT_IDS - {row["sample_id"] for row in tolerated}
    if missing_tolerated:
        raise ValueError(f"tolerated artifact controls missing from holdout: {missing_tolerated}")

    controls: list[dict[str, str]] = []
    for key, quota in CONTROL_QUOTAS.items():
        eligible = sorted(
            (
                row
                for row in holdout
                if (row.get("dataset"), row.get("operation")) == key
                and row["sample_id"] not in TOLERATED_ARTIFACT_IDS
                and is_clean_control(row, reviews_by_id[row["sample_id"]])
            ),
            key=deterministic_order,
        )
        if len(eligible) < quota:
            raise ValueError(f"not enough clean controls for {key}: {len(eligible)} < {quota}")
        controls.extend(eligible[:quota])

    groups = [
        ("hard_failure", sorted(hard, key=deterministic_order)),
        ("borderline_focus_defect", sorted(borderline_focus, key=deterministic_order)),
        ("tolerated_artifact", sorted(tolerated, key=deterministic_order)),
        ("clean_control", controls),
    ]
    selected_ids = [row["sample_id"] for _, rows in groups for row in rows]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(selected_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"focused evaluation groups overlap: {duplicates}")

    output = [
        annotate(row, reviews_by_id[row["sample_id"]], group)
        for group, rows in groups
        for row in rows
    ]
    if len(output) != 38:
        raise ValueError(f"expected 38 focused evaluation rows, got {len(output)}")
    ego_count = sum(row.get("dataset") == "egoedit" for row in output)
    if ego_count > 4:
        raise ValueError(f"EgoEdit cap exceeded: {ego_count} > 4")
    focus_count = sum(row["focus_defect_ground_truth"] == "1" for row in output)
    if focus_count < 7:
        raise ValueError(f"too few explicit focus defects: {focus_count} < 7")

    write_csv(args.output, output, [*holdout_fields, *EXPERIMENT_FIELDS])
    summary = {
        "experiment_version": "qwen3_artifact_focus_holdout_v1",
        "selection_seed": SEED,
        "source_holdout_rows": len(holdout),
        "selected_rows": len(output),
        "groups": dict(Counter(row["focus_group"] for row in output)),
        "human_labels": dict(Counter(row["human_label"] for row in output)),
        "datasets": dict(Counter(row["dataset"] for row in output)),
        "dataset_operations": {
            f"{dataset}:{operation}": count
            for (dataset, operation), count in sorted(
                Counter((row["dataset"], row["operation"]) for row in output).items()
            )
        },
        "explicit_focus_defects": focus_count,
        "focus_tags": dict(
            Counter(
                tag
                for row in output
                for tag in row["focus_defect_tags"].split("|")
                if tag
            )
        ),
        "egoedit_rows": ego_count,
        "promotion_metrics_note": (
            "Visible-artifact recall uses explicit human defect tags; clean false-positive "
            "rates use only clean_control, while tolerated_artifact is reported separately."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
