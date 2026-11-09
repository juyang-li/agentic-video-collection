#!/usr/bin/env python3
"""Combine the stable 72B judgment with cheap remove-reconstruction risks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

import cv2
import numpy as np

import run_openrouter_qwen3_judge as judge


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_BASE = RUN / "artifact_risk_30_qwen72b_reconstruction_v6_scores.csv"
DEFAULT_OUTPUT = RUN / "artifact_risk_30_qwen72b_hybrid_v8_scores.csv"
DEFAULT_SUMMARY = RUN / "artifact_risk_30_qwen72b_hybrid_v8_summary.json"
BASE_JUDGE = "qwen2_5_vl_72b_openrouter_reconstruction_aware_pair_judge_v6"
HYBRID_JUDGE = "qwen2_5_vl_72b_plus_cpu_remove_reconstruction_filter_v8"
HEADWEAR_RE = re.compile(
    r"\b(cap|hat|headwear|headpiece|helmet|beanie|fedora)\b|帽子|头饰|头盔",
    re.IGNORECASE,
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def grayscale_features(image: object) -> tuple[float, float, float]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = float(np.mean(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)))
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return laplacian_variance, gradient, entropy


def roi_detail_metrics(row: dict[str, str]) -> tuple[float, float, float]:
    audit_view = judge.build_audit_view(row)
    pairs = judge.rect.matched_frames(
        judge.resolve_media(row[f"{audit_view.left_training_role}_local"]),
        judge.resolve_media(row[f"{audit_view.right_training_role}_local"]),
        9,
    )
    cards, _proposals, _selected = judge.prepare_time_cards(
        pairs, 5, 640, 640, 512, 0.12
    )
    laplacian_ratios: list[float] = []
    gradient_ratios: list[float] = []
    entropy_deltas: list[float] = []
    for card in cards:
        if card.roi_left is None or card.roi_right is None:
            continue
        left_laplacian, left_gradient, left_entropy = grayscale_features(card.roi_left)
        right_laplacian, right_gradient, right_entropy = grayscale_features(card.roi_right)
        laplacian_ratios.append(right_laplacian / max(left_laplacian, 1e-6))
        gradient_ratios.append(right_gradient / max(left_gradient, 1e-6))
        entropy_deltas.append(right_entropy - left_entropy)
    if not entropy_deltas:
        return 1.0, 1.0, 0.0
    return (
        float(median(laplacian_ratios)),
        float(median(gradient_ratios)),
        float(median(entropy_deltas)),
    )


def parse_json(value: str, fallback: object) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-scores", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--entropy-delta-max", type=float, default=-0.65)
    parser.add_argument("--gradient-ratio-max", type=float, default=0.85)
    args = parser.parse_args()

    manifest, _manifest_fields = read_csv(args.manifest)
    base_rows, base_fields = read_csv(args.base_scores)
    expected = {row["sample_id"] for row in manifest}
    base = {row["sample_id"]: row for row in base_rows}
    if len(expected) != len(manifest) or set(base) != expected:
        raise ValueError("manifest and base scores must contain identical unique sample IDs")
    for row in base_rows:
        if row.get("judge_version") != BASE_JUDGE or row.get("parse_status") != "pass":
            raise ValueError(f"invalid base score identity for {row.get('sample_id')}")

    output: list[dict[str, str]] = []
    for index, source in enumerate(manifest, 1):
        sample_id = source["sample_id"]
        score = dict(base[sample_id])
        audit_view = judge.build_audit_view(source)
        laplacian_ratio, gradient_ratio, entropy_delta = roi_detail_metrics(source)
        blur_risk = (
            audit_view.audit_operation == "remove"
            and audit_view.generated_position == "RIGHT"
            and entropy_delta <= args.entropy_delta_max
            and gradient_ratio <= args.gradient_ratio_max
        )
        headwear_risk = (
            audit_view.audit_operation == "remove"
            and source.get("dataset", "").lower() == "goku"
            and source.get("target_scope", "").lower() == "accessory"
            and bool(HEADWEAR_RE.search(source.get("instruction", "")))
        )
        specialist_reasons: list[str] = []
        if blur_risk:
            specialist_reasons.append("cpu_large_local_detail_drop")
        if headwear_risk:
            specialist_reasons.append("goku_headwear_remove_reconstruction_risk")

        reasons = parse_json(score.get("reason_codes_json", ""), [])
        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(value) for value in reasons]
        for reason in specialist_reasons:
            if reason not in reasons:
                reasons.append(reason)
        evidence = parse_json(score.get("evidence_json", ""), {})
        if not isinstance(evidence, dict):
            evidence = {}
        if specialist_reasons:
            diagnostic = (
                "CPU specialist: "
                f"median ROI entropy delta={entropy_delta:.3f}, "
                f"gradient ratio={gradient_ratio:.3f}, "
                f"laplacian ratio={laplacian_ratio:.3f}; "
                f"flags={','.join(specialist_reasons)}."
            )
            prior = str(evidence.get("visible_artifact", "")).strip()
            evidence["visible_artifact"] = f"{prior} {diagnostic}".strip()
            score["visible_artifact"] = "fail"
            score["model_decision"] = "rejected"

        score.update(
            {
                "judge_version": HYBRID_JUDGE,
                "model_backend": "openrouter_plus_cpu",
                "reason_codes_json": json.dumps(
                    reasons[:8], ensure_ascii=False, separators=(",", ":")
                ),
                "evidence_json": json.dumps(
                    evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "cpu_roi_laplacian_ratio_median": f"{laplacian_ratio:.6f}",
                "cpu_roi_gradient_ratio_median": f"{gradient_ratio:.6f}",
                "cpu_roi_entropy_delta_median": f"{entropy_delta:.6f}",
                "cpu_blur_risk": "1" if blur_risk else "0",
                "cpu_headwear_risk": "1" if headwear_risk else "0",
                "base_model_decision": base[sample_id]["model_decision"],
                "specialist_reason_codes_json": json.dumps(
                    specialist_reasons, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        output.append(score)
        print(
            f"specialist={index}/{len(manifest)} sample_id={sample_id} "
            f"base={score['base_model_decision']} hybrid={score['model_decision']} "
            f"blur={int(blur_risk)} headwear={int(headwear_risk)}",
            flush=True,
        )

    extra_fields = [
        "cpu_roi_laplacian_ratio_median",
        "cpu_roi_gradient_ratio_median",
        "cpu_roi_entropy_delta_median",
        "cpu_blur_risk",
        "cpu_headwear_risk",
        "base_model_decision",
        "specialist_reason_codes_json",
    ]
    fields = list(dict.fromkeys([*base_fields, *extra_fields]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)

    summary = {
        "judge_version": HYBRID_JUDGE,
        "base_judge_version": BASE_JUDGE,
        "rows": len(output),
        "thresholds": {
            "entropy_delta_max": args.entropy_delta_max,
            "gradient_ratio_max": args.gradient_ratio_max,
        },
        "base_decisions": dict(Counter(row["base_model_decision"] for row in output)),
        "hybrid_decisions": dict(Counter(row["model_decision"] for row in output)),
        "cpu_blur_risk": sum(row["cpu_blur_risk"] == "1" for row in output),
        "cpu_headwear_risk": sum(row["cpu_headwear_risk"] == "1" for row in output),
        "new_rejections": sum(
            row["base_model_decision"] != "rejected"
            and row["model_decision"] == "rejected"
            for row in output
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
