#!/usr/bin/env python3
"""Run and evaluate PAL4VST on the frozen 40+30 human-reviewed pairs.

The experiment is deliberately diagnostic: it never mutates a filtering
decision.  For every pair it routes the generated side from provenance, takes
five non-conditioning time points, and evaluates both the full frame and the
existing tight pair-difference ROI.  The matching reference crops are scored
as a control for naturally difficult content.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SPECIALIST_ROOT = PROJECT / "1_specialist_filtering"
VENDOR = ROOT / "vendor/PAL4VST"
RUNTIME = ROOT / "vendor/pal4vst_runtime"
MMSEG_ROOT = VENDOR / "mmsegmentation"
DEFAULT_CONFIG = MMSEG_ROOT / "configs/pal4vst/swin-large_upernet_unified_512x512.py"
DEFAULT_CHECKPOINT = ROOT / "artifacts/models/pal4vst_unified/unified.pth"
DEFAULT_OUTPUT = ROOT / "artifacts/pal4vst_v1"

HOLDOUT_MANIFEST = ROOT / "artifacts/v12_holdout40_v1/v12_holdout40.csv"
HOLDOUT_LABELS = ROOT / "artifacts/v12_holdout40_v1/postreview_policy_labels.csv"
RISK_MANIFEST = ROOT / "artifacts/qwen72b_artifact_risk_v1/artifact_risk_30.csv"
RISK_LABELS = (
    ROOT
    / "artifacts/qwen72b_artifact_risk_v1/qwen32b_vs_72b_artifact_risk_30_v1_review.csv"
)

MODEL_INPUT_SIZE = 512
MODEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
MODEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
CHECKPOINT_SHA256 = "78f5e507c7d785216cb0ac2ceff6aa8b146eaccfcdb690d618dab39ee47b473f"

sys.path.insert(0, str(ROOT / "scripts"))
import run_openrouter_qwen3_judge as judge  # noqa: E402


@dataclass(frozen=True)
class LabeledPair:
    cohort: str
    human_status: str
    human_notes: str
    human_label_basis: str
    row: dict[str, str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.json")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_experiment_pairs() -> list[LabeledPair]:
    holdout_labels = {row["sample_id"]: row for row in read_csv(HOLDOUT_LABELS)}
    risk_labels = {row["sample_id"]: row for row in read_csv(RISK_LABELS)}
    output: list[LabeledPair] = []
    for row in read_csv(HOLDOUT_MANIFEST):
        label = holdout_labels[row["sample_id"]]
        output.append(
            LabeledPair(
                cohort="unseen_holdout40",
                human_status=label["overall_status"].strip().lower(),
                human_notes=label.get("human_raw_notes", "").strip(),
                human_label_basis=label.get("policy_basis", "postreview_policy"),
                row=row,
            )
        )
    for row in read_csv(RISK_MANIFEST):
        label = risk_labels[row["sample_id"]]
        output.append(
            LabeledPair(
                cohort="artifact_risk30",
                human_status=label["overall_status"].strip().lower(),
                human_notes=label.get("notes", "").strip(),
                human_label_basis="direct_human_review",
                row=row,
            )
        )
    sample_ids = [item.row["sample_id"] for item in output]
    if len(output) != 70:
        raise ValueError(f"expected the frozen 40+30 experiment, got {len(output)} rows")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("experiment cohorts overlap by sample_id")
    invalid = sorted({item.human_status for item in output} - {"usable", "unusable", "uncertain"})
    if invalid:
        raise ValueError(f"unexpected human statuses: {invalid}")
    return output


def install_unused_mmcv_extension_stub() -> None:
    """Let mmseg import with mmcv-lite; PAL's model does not call custom ops."""
    if "mmcv._ext" in sys.modules:
        return
    module = types.ModuleType("mmcv._ext")
    module.__file__ = "<unused-mmcv-extension>"

    def missing(name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                f"PAL4VST unexpectedly called unavailable MMCV custom op {name!r}"
            )

        return unavailable

    module.__getattr__ = missing  # type: ignore[attr-defined]
    sys.modules["mmcv._ext"] = module


def load_model(config: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PAL4VST checkpoint missing: {checkpoint}")
    for path in (RUNTIME, MMSEG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    install_unused_mmcv_extension_stub()
    from mmseg.apis import init_model

    model = init_model(
        str(config),
        checkpoint=None,
        device="cpu",
        cfg_options={"model.backbone.init_cfg": None},
    )
    # The official mmseg-0.25 checkpoint contains NumPy scalar metadata. Keep
    # weights_only=True and allow only the exact safe NumPy numeric types it uses.
    safe_globals = [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_globals):
        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = value.get("state_dict") if isinstance(value, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("PAL4VST checkpoint has no state_dict")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "PAL4VST state mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def normalized_batch(images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
    arrays = []
    for image in images:
        resized = image.convert("RGB").resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR
        )
        arrays.append(np.asarray(resized, dtype=np.float32))
    value = (np.stack(arrays) - MODEL_MEAN[None, None, None, :]) / MODEL_STD[
        None, None, None, :
    ]
    return torch.from_numpy(value).permute(0, 3, 1, 2).to(device)


def predict_probabilities(
    model: torch.nn.Module,
    images: Sequence[Image.Image],
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    meta = {
        "img_shape": (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        "ori_shape": (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        "pad_shape": (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        "padding_size": [0, 0, 0, 0],
    }
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        inputs = normalized_batch(batch_images, device)
        with torch.inference_mode():
            logits = model.encode_decode(inputs, [dict(meta) for _ in batch_images])
            probabilities = logits.softmax(dim=1)[:, 1].float().cpu().numpy()
        output.extend(probabilities)
    return output


def maximum_component_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size)


def probability_metrics(probability: np.ndarray) -> dict[str, float]:
    return {
        "par": float(probability.mean()),
        "prob_p90": float(np.quantile(probability, 0.90)),
        "prob_p95": float(np.quantile(probability, 0.95)),
        "prob_p99": float(np.quantile(probability, 0.99)),
        "area25": float((probability >= 0.25).mean()),
        "area50": float((probability >= 0.50).mean()),
        "area75": float((probability >= 0.75).mean()),
        "largest_component50": maximum_component_fraction(probability >= 0.50),
    }


def overlay(image: Image.Image, probability: np.ndarray) -> Image.Image:
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(np.clip(probability, 0.0, 1.0) * 255)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    alpha = np.asarray(heat, dtype=np.float32)[..., None] / 255.0 * 0.62
    original = np.asarray(base, dtype=np.float32)
    pink = np.empty_like(original)
    pink[..., 0] = 255
    pink[..., 1] = 35
    pink[..., 2] = 210
    result = original * (1.0 - alpha) + pink * alpha
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def fit_thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
    value = image.convert("RGB")
    value.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (20, 24, 31))
    canvas.paste(value, ((width - value.width) // 2, (height - value.height) // 2))
    return canvas


def build_contact_sheet(
    item: LabeledPair,
    cards: Sequence[Any],
    probabilities: dict[tuple[int, str, str], np.ndarray],
    frame_rows: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    tile_width, tile_height, header = 245, 150, 31
    columns = ("REFERENCE FULL", "GENERATED FULL", "FULL PAL", "GENERATED ROI", "ROI PAL")
    canvas = Image.new(
        "RGB",
        (tile_width * len(columns), header + len(cards) * (tile_height + header)),
        (245, 247, 250),
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        left = column * tile_width
        draw.rectangle((left, 0, left + tile_width - 1, header - 1), fill=(30, 41, 59))
        draw.text((left + 8, 9), label, fill=(255, 255, 255))
    lookup = {(int(row["display_index"]), row["side"], row["view"]): row for row in frame_rows}
    for row_index, card in enumerate(cards):
        top = header + row_index * (tile_height + header)
        roi_right = card.roi_right or card.full_right
        full_probability = probabilities[(card.display_index, "generated", "full")]
        roi_probability = probabilities[(card.display_index, "generated", "roi")]
        images = (
            card.full_left,
            card.full_right,
            overlay(card.full_right, full_probability),
            roi_right,
            overlay(roi_right, roi_probability),
        )
        full_row = lookup[(card.display_index, "generated", "full")]
        roi_row = lookup[(card.display_index, "generated", "roi")]
        labels = (
            f"F{card.display_index} t={card.time_percent}%",
            f"PAR={float(full_row['par']):.4f}",
            f"area50={float(full_row['area50']):.4f}",
            f"PAR={float(roi_row['par']):.4f}",
            f"area50={float(roi_row['area50']):.4f}",
        )
        for column, (image, label) in enumerate(zip(images, labels)):
            left = column * tile_width
            canvas.paste(fit_thumbnail(image, tile_width, tile_height), (left, top))
            draw.rectangle(
                (left, top + tile_height, left + tile_width - 1, top + tile_height + header - 1),
                fill=(226, 232, 240),
            )
            draw.text((left + 7, top + tile_height + 9), label, fill=(15, 23, 42))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=88, optimize=True)


def run_one(
    model: torch.nn.Module,
    item: LabeledPair,
    output: Path,
    device: torch.device,
    batch_size: int,
    save_visualization: bool,
) -> dict[str, Any]:
    row = item.row
    view = judge.build_audit_view(row)
    pairs = judge.rect.matched_frames(
        judge.resolve_media(row[f"{view.left_training_role}_local"]),
        judge.resolve_media(row[f"{view.right_training_role}_local"]),
        9,
    )
    cards, _proposals, selected = judge.prepare_time_cards(
        pairs,
        card_count=5,
        full_max_long_edge=1024,
        roi_max_long_edge=1024,
        analysis_long_edge=512,
        minimum_roi_reliability=0.12,
    )
    if len(cards) != 5 or len(selected) != 5:
        raise ValueError(
            f"expected five full+ROI cards for {row['sample_id']}, "
            f"got cards={len(cards)} roi={len(selected)}"
        )
    images: list[Image.Image] = []
    identities: list[tuple[int, str, str]] = []
    selected_by_index = {proposal.index: proposal for proposal in selected}
    for card in cards:
        for side, full, roi in (
            ("reference", card.full_left, card.roi_left or card.full_left),
            ("generated", card.full_right, card.roi_right or card.full_right),
        ):
            for view_name, image in (("full", full), ("roi", roi)):
                images.append(image)
                identities.append((card.display_index, side, view_name))
    predicted = predict_probabilities(model, images, device, batch_size)
    probabilities = dict(zip(identities, predicted))
    frame_rows: list[dict[str, Any]] = []
    for card in cards:
        proposal = selected_by_index[card.frame_index]
        x1, y1, x2, y2 = proposal.expanded_box
        for side in ("reference", "generated"):
            for view_name in ("full", "roi"):
                frame_row: dict[str, Any] = {
                    "cohort": item.cohort,
                    "sample_id": row["sample_id"],
                    "dataset": row.get("dataset", ""),
                    "branch": row.get("branch", ""),
                    "operation": row.get("operation", ""),
                    "human_status": item.human_status,
                    "display_index": card.display_index,
                    "frame_index": card.frame_index,
                    "time_percent": card.time_percent,
                    "side": side,
                    "view": view_name,
                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                    "roi_area": (x2 - x1) * (y2 - y1),
                    "roi_reliability": proposal.reliability,
                    "roi_change_score": proposal.change_score,
                    "roi_fallback": int(proposal.fallback),
                }
                frame_row.update(
                    probability_metrics(probabilities[(card.display_index, side, view_name)])
                )
                frame_rows.append(frame_row)
    if save_visualization:
        build_contact_sheet(
            item,
            cards,
            probabilities,
            frame_rows,
            output / "visualizations" / f"{row['sample_id']}.jpg",
        )
    return {
        "experiment_version": "pal4vst_frozen_40plus30_v1",
        "cohort": item.cohort,
        "sample_id": row["sample_id"],
        "dataset": row.get("dataset", ""),
        "branch": row.get("branch", ""),
        "operation": row.get("operation", ""),
        "instruction": row.get("instruction", ""),
        "human_status": item.human_status,
        "human_notes": item.human_notes,
        "human_label_basis": item.human_label_basis,
        "source_provenance": row.get("source_provenance", ""),
        "target_provenance": row.get("target_provenance", ""),
        "audit_operation": view.audit_operation,
        "audit_relation": view.audit_relation,
        "reference_training_role": view.left_training_role,
        "generated_training_role": view.right_training_role,
        "generated_position": view.generated_position,
        "baseline_kind": (
            "generated_endpoint" if view.generated_position == "BOTH" else "original_reference"
        ),
        "frame_rows": frame_rows,
    }


def median(values: Iterable[float]) -> float:
    data = list(values)
    return float(statistics.median(data)) if data else math.nan


def aggregate_sample(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw["frame_rows"]
    output = {key: value for key, value in raw.items() if key != "frame_rows"}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["side"], row["view"])].append(row)
    for side in ("reference", "generated"):
        for view_name in ("full", "roi"):
            values = grouped[(side, view_name)]
            prefix = f"{side}_{view_name}"
            for metric in (
                "par",
                "prob_p95",
                "area25",
                "area50",
                "largest_component50",
            ):
                sequence = [float(row[metric]) for row in values]
                output[f"{prefix}_{metric}_median"] = median(sequence)
                output[f"{prefix}_{metric}_max"] = max(sequence)
    for view_name in ("full", "roi"):
        for metric in ("par", "area25", "area50", "largest_component50"):
            reference = {
                int(row["display_index"]): float(row[metric])
                for row in grouped[("reference", view_name)]
            }
            differences = [
                float(row[metric]) - reference[int(row["display_index"])]
                for row in grouped[("generated", view_name)]
            ]
            output[f"delta_{view_name}_{metric}_median"] = median(differences)
            output[f"delta_{view_name}_{metric}_max"] = max(differences)
    output["roi_fallback_count"] = sum(
        int(row["roi_fallback"])
        for row in grouped[("generated", "roi")]
    )
    output["roi_area_median"] = median(
        float(row["roi_area"]) for row in grouped[("generated", "roi")]
    )
    # A localizer occasionally misses the actual edit while the full-frame PAL
    # still responds (and vice versa). This fixed, scale-compatible OR score is
    # the only combined metric in the experiment; it is not a fitted model.
    output["generated_full_or_roi_par_median"] = max(
        float(output["generated_full_par_median"]),
        float(output["generated_roi_par_median"]),
    )
    return output


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return math.nan
    wins = 0.0
    for good_score in negative:
        for bad_score in positive:
            wins += float(bad_score > good_score) + 0.5 * float(bad_score == good_score)
    return wins / (len(positive) * len(negative))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        return math.nan
    ranked = sorted(zip(scores, labels), reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, 1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positive_count


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else math.nan
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def threshold_summaries(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    thresholds = sorted(set(scores), reverse=True)
    candidates = [confusion(labels, scores, threshold) for threshold in thresholds]
    best = max(
        candidates,
        key=lambda value: (
            value["balanced_accuracy"],
            -value["false_positive_rate"],
            value["recall"],
        ),
    )
    recall80 = [value for value in candidates if value["recall"] >= 0.80]
    high_recall = min(
        recall80,
        key=lambda value: (
            value["false_positive_rate"],
            -value["precision"],
            -value["threshold"],
        ),
    )
    return {"best_balanced": best, "minimum_fpr_at_recall80": high_recall}


def evaluate_metric(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    decisive = [row for row in rows if row["human_status"] in {"usable", "unusable"}]
    labels = [int(row["human_status"] == "unusable") for row in decisive]
    scores = [float(row[metric]) for row in decisive]
    return {
        "metric": metric,
        "rows": len(decisive),
        "usable": labels.count(0),
        "unusable": labels.count(1),
        "usable_mean": float(np.mean([score for label, score in zip(labels, scores) if not label])),
        "unusable_mean": float(np.mean([score for label, score in zip(labels, scores) if label])),
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        **threshold_summaries(labels, scores),
    }


def evaluate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "generated_full_or_roi_par_median",
        "generated_full_par_median",
        "generated_full_par_max",
        "generated_roi_par_median",
        "generated_roi_par_max",
        "generated_full_area50_max",
        "generated_roi_area50_max",
        "generated_roi_largest_component50_max",
        "delta_full_par_median",
        "delta_full_par_max",
        "delta_roi_par_median",
        "delta_roi_par_max",
        "delta_roi_area50_max",
    )
    combined = [evaluate_metric(rows, metric) for metric in metrics]
    cohorts = {
        cohort: [evaluate_metric([row for row in rows if row["cohort"] == cohort], metric) for metric in metrics]
        for cohort in sorted({row["cohort"] for row in rows})
    }
    ranked = sorted(combined, key=lambda value: (value["roc_auc"], value["average_precision"]), reverse=True)
    return {
        "experiment_version": "pal4vst_frozen_40plus30_v1",
        "sample_counts": dict(Counter(row["human_status"] for row in rows)),
        "cohort_counts": {
            cohort: dict(Counter(row["human_status"] for row in rows if row["cohort"] == cohort))
            for cohort in sorted({row["cohort"] for row in rows})
        },
        "combined_metrics": combined,
        "cohort_metrics": cohorts,
        "best_metric": ranked[0],
        "interpretation_warning": (
            "Best thresholds are diagnostic upper bounds chosen on the same 70 reviewed pairs; "
            "they are not approved filtering thresholds. Unusable labels include non-artifact failures."
        ),
    }


def build_html(rows: Sequence[dict[str, Any]], evaluation: dict[str, Any]) -> str:
    best = evaluation["best_metric"]
    threshold = float(best["best_balanced"]["threshold"])
    metric = str(best["metric"])
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
    cards = []
    for row in ordered:
        score = float(row[metric])
        prediction = "PAL reject" if score >= threshold else "PAL keep"
        correctness = "neutral"
        if row["human_status"] == "unusable":
            correctness = "correct" if score >= threshold else "wrong"
        elif row["human_status"] == "usable":
            correctness = "correct" if score < threshold else "wrong"
        notes = html.escape(str(row.get("human_notes", ""))) or "无"
        instruction = html.escape(str(row.get("instruction", "")))
        image_path = f"visualizations/{html.escape(str(row['sample_id']))}.jpg"
        cards.append(
            f"""
            <article class="{correctness}">
              <h2>{html.escape(str(row['sample_id']))}</h2>
              <div class="facts"><span>{html.escape(str(row['cohort']))}</span><span>{html.escape(str(row['branch']))}</span><span>人工 {html.escape(str(row['human_status']))}</span><span>{prediction}</span><span>{metric}={score:.6f}</span></div>
              <p><b>人工备注：</b>{notes}</p><p><b>指令：</b>{instruction}</p>
              <img loading="lazy" src="{image_path}">
            </article>"""
        )
    summary = best["best_balanced"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PAL4VST 40+30实验</title><style>
    :root{{font-family:system-ui,sans-serif;color:#172033;background:#eef2f7}}body{{margin:0}}main{{max-width:1320px;margin:auto;padding:18px}}header,article{{background:white;border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 2px 10px #0001}}article.correct{{border-left:7px solid #22c55e}}article.wrong{{border-left:7px solid #ef4444}}article.neutral{{border-left:7px solid #94a3b8}}.facts{{display:flex;gap:7px;flex-wrap:wrap}}.facts span{{background:#e2e8f0;border-radius:7px;padding:5px 8px}}img{{width:100%;height:auto;border:1px solid #cbd5e1}}code{{background:#e2e8f0;padding:2px 5px;border-radius:4px}}</style></head><body><main>
    <header><h1>PAL4VST 冻结 40+30 实验</h1><p>统一模型同时检查五个非条件帧的 generated 全图与 tight ROI；reference 同坐标得分用于差值对照。粉色是模型预测的感知伪影概率。</p><p>同集诊断最佳指标：<code>{html.escape(metric)}</code>；ROC-AUC={float(best['roc_auc']):.3f}，AP={float(best['average_precision']):.3f}。同集最佳阈值 {threshold:.6f} 下：坏样本召回 {float(summary['recall']):.1%}，好样本误杀 {float(summary['false_positive_rate']):.1%}。该阈值没有资格直接用于全量筛选。</p></header>
    {''.join(cards)}</main></body></html>"""


def build_markdown(evaluation: dict[str, Any], elapsed_seconds: float) -> str:
    ranked = sorted(
        evaluation["combined_metrics"],
        key=lambda value: (value["roc_auc"], value["average_precision"]),
        reverse=True,
    )
    lines = [
        "# PAL4VST 40+30 实验结果",
        "",
        "本实验使用作者官方 `unified.pth`，对冻结的 40 条未见随机审核样本和 30 条高风险富集样本运行五时刻全图与 tight ROI。模型只生成诊断分数，不修改任何现有筛选决策。",
        "",
        "## 样本与运行",
        "",
        f"- 人工标签：{evaluation['sample_counts']}。",
        f"- 总耗时：{elapsed_seconds:.1f} 秒。",
        f"- checkpoint SHA256：`{CHECKPOINT_SHA256}`。",
        "- 单侧生成 pair 固定以 provenance 路由为 `REFERENCE → GENERATED`；EgoEdit 两侧均为 generated，另以 `baseline_kind=generated_endpoint` 标记。",
        "",
        "## 诊断排名",
        "",
        "| 指标 | ROC-AUC | AP | 最佳平衡召回 | 最佳平衡误杀 |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in ranked:
        point = value["best_balanced"]
        lines.append(
            f"| `{value['metric']}` | {value['roc_auc']:.3f} | {value['average_precision']:.3f} | {point['recall']:.1%} | {point['false_positive_rate']:.1%} |"
        )
    best = ranked[0]
    high_recall = best["minimum_fpr_at_recall80"]
    cohort_values = {
        cohort: next(value for value in values if value["metric"] == best["metric"])
        for cohort, values in evaluation["cohort_metrics"].items()
    }
    random_holdout = cohort_values["unseen_holdout40"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 全图与 ROI 的固定 OR 指标 `{best['metric']}` 是表现最好的诊断量，但 ROC-AUC 只有 `{best['roc_auc']:.3f}`。同集最佳平衡点只能召回 `{best['best_balanced']['recall']:.1%}` 的坏样本，同时误杀 `{best['best_balanced']['false_positive_rate']:.1%}` 的好样本。",
            f"- 将召回提高到至少 80% 时，好样本误杀上升到 `{high_recall['false_positive_rate']:.1%}`；在更接近自然分布的随机 holdout40 上，该指标 ROC-AUC 仅 `{random_holdout['roc_auc']:.3f}`。",
            "- `pilot-goku_add-5a25b43e48589a91` 的大片人体模糊和 `pilot-goku_add-b6accfce9a157a30` 的身体未补全会产生清楚的局部响应；全图/ROI OR 也能覆盖本批若干遮罩与透明人体案例。",
            "- 但模型同样会把正常人脸、运动中的手、反光锅具、透明保鲜膜和复杂食物纹理标成伪影；tight ROI 偶尔还会偏离真正编辑对象。reference 差值指标最高 AUC 也只有约 0.53，未能消除这种领域偏差。",
            "- 因此 PAL4VST 不进入 hard reject，也不替代当前 72B。若后续使用，只能以非常高阈值产生少量人工复核候选，或先用本项目像素级标注做领域适配。",
            "",
            "## 解释边界",
            "",
            "所有阈值都在同一 70 条上回看选出，只代表 PAL4VST 是否有分离信号，不能直接推广；人工 `unusable` 还包含指令不跟随和内容保持问题，并不都是局部视觉伪影。HTML 热图用于核对模型究竟抓到实际遮罩/涂抹还是自然纹理。",
            "",
        ]
    )
    return "\n".join(lines)


def assemble_outputs(output: Path, elapsed_seconds: float) -> dict[str, Any]:
    raw_files = sorted((output / "raw").glob("*.json"))
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in raw_files]
    if len(raw) != 70:
        raise ValueError(f"expected 70 completed raw samples, found {len(raw)}")
    frame_rows = [row for sample in raw for row in sample["frame_rows"]]
    sample_rows = [aggregate_sample(sample) for sample in raw]
    write_csv(output / "pal4vst_frame_scores.csv", frame_rows)
    write_csv(output / "pal4vst_sample_scores.csv", sample_rows)
    evaluation = evaluate(sample_rows)
    evaluation.update(
        {
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config": str(DEFAULT_CONFIG),
            "elapsed_seconds": elapsed_seconds,
            "frames_per_pair": 5,
            "model_input_size": MODEL_INPUT_SIZE,
            "frame_score_rows": len(frame_rows),
        }
    )
    write_json(output / "evaluation.json", evaluation)
    (output / "REPORT.md").write_text(
        build_markdown(evaluation, elapsed_seconds), encoding="utf-8"
    )
    (output / "index.html").write_text(
        build_html(sample_rows, evaluation), encoding="utf-8"
    )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    pairs = build_experiment_pairs()
    if args.limit is not None:
        pairs = pairs[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for item in pairs:
        manifest_rows.append(
            {
                "cohort": item.cohort,
                "sample_id": item.row["sample_id"],
                "dataset": item.row.get("dataset", ""),
                "branch": item.row.get("branch", ""),
                "operation": item.row.get("operation", ""),
                "instruction": item.row.get("instruction", ""),
                "human_status": item.human_status,
                "human_notes": item.human_notes,
                "human_label_basis": item.human_label_basis,
                "source_provenance": item.row.get("source_provenance", ""),
                "target_provenance": item.row.get("target_provenance", ""),
            }
        )
    write_csv(args.output / "experiment_manifest.csv", manifest_rows)

    if args.assemble_only:
        if len(pairs) != 70:
            parser.error("--assemble-only requires the complete 70-pair experiment")
        previous = args.output / "evaluation.json"
        elapsed = 0.0
        if previous.is_file():
            elapsed = float(json.loads(previous.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
        evaluation = assemble_outputs(args.output, elapsed)
        print(json.dumps(evaluation["best_metric"], ensure_ascii=False), flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = load_model(args.config, args.checkpoint, device)
    started = time.monotonic()
    for index, item in enumerate(pairs, 1):
        raw_path = args.output / "raw" / f"{item.row['sample_id']}.json"
        if args.resume and raw_path.is_file():
            print(f"[{index}/{len(pairs)}] resume={item.row['sample_id']}", flush=True)
            continue
        sample_started = time.monotonic()
        result = run_one(
            model,
            item,
            args.output,
            device,
            args.batch_size,
            not args.no_visualizations,
        )
        write_json(raw_path, result)
        print(
            f"[{index}/{len(pairs)}] sample={item.row['sample_id']} "
            f"status={item.human_status} seconds={time.monotonic()-sample_started:.2f}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    if len(pairs) == 70:
        evaluation = assemble_outputs(args.output, elapsed)
        print(json.dumps(evaluation["best_metric"], ensure_ascii=False), flush=True)
    else:
        print(f"smoke_complete={len(pairs)} elapsed_seconds={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
