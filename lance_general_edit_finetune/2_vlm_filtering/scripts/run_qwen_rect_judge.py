#!/usr/bin/env python3
"""Run a resumable Qwen pair judge with expanded rectangular edit-region crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
DEFAULT_INPUT = SPECIALIST_ROOT / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/vlm_run_v4/qwen_rect_pair_scores.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/vlm_run_v4/qwen_rect_pair_score_summary.json"
DEFAULT_PROMPT = ROOT / "judge_prompt_rect_v4.txt"
JUDGE_VERSION = "qwen2_5_vl_7b_pair_rect_judge_v4"
ROI_METHOD_VERSION = "paired_change_rect_v1"
VISUAL_LAYOUT_VERSION = "labeled_source_target_pair_strip_v1"
VALID_DECISIONS = {"accepted", "rejected", "uncertain"}
VALID_CHECKS = {"pass", "fail", "uncertain"}
CHECK_FIELDS = ["instruction_following", "visible_artifact", "content_preservation"]
FIELDS = [
    "judge_version",
    "roi_method_version",
    "visual_layout_version",
    "model_id",
    "model_backend",
    "prompt_sha256",
    "sample_id",
    "dataset",
    "branch",
    "operation",
    "target_scope",
    "original_stage1_1_decision",
    "sampled_full_pairs",
    "sampled_roi_pairs",
    "roi_boxes_json",
    "roi_reliability_min",
    "roi_reliability_mean",
    "roi_fallback_count",
    "input_tokens",
    "peak_gpu_memory_mb",
    "model_reported_decision",
    "model_decision",
    *CHECK_FIELDS,
    "reason_codes_json",
    "model_confidence",
    "short_evidence",
    "parse_status",
    "raw_response",
    "error",
    "elapsed_seconds",
]


@dataclass
class RegionProposal:
    index: int
    raw_box: tuple[float, float, float, float]
    expanded_box: tuple[float, float, float, float]
    reliability: float
    change_score: float
    fallback: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SPECIALIST_ROOT / path


def resize_long_edge(frame: np.ndarray, maximum: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def matched_frames(source_path: Path, target_path: Path, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from decord import VideoReader, cpu

    source_reader = VideoReader(str(source_path), ctx=cpu(0), num_threads=1)
    target_reader = VideoReader(str(target_path), ctx=cpu(0), num_threads=1)
    if len(source_reader) < 1 or len(target_reader) < 1:
        raise ValueError(f"empty source/target video: {source_path} | {target_path}")
    positions = np.linspace(0.0, 1.0, min(count, len(source_reader), len(target_reader)))
    source_indices = np.unique(np.rint(positions * (len(source_reader) - 1)).astype(int))
    target_indices = np.unique(np.rint(positions * (len(target_reader) - 1)).astype(int))
    pair_count = min(len(source_indices), len(target_indices))
    source = source_reader.get_batch(source_indices[:pair_count]).asnumpy()
    target = target_reader.get_batch(target_indices[:pair_count]).asnumpy()
    return list(zip(source, target))


def _expanded_box(
    box: tuple[float, float, float, float],
    factor: float,
    minimum_fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, minimum_fraction)
    height = max(y2 - y1, minimum_fraction)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = min(1.0, width * factor)
    height = min(1.0, height * factor)
    x1 = max(0.0, min(1.0 - width, center_x - width / 2.0))
    y1 = max(0.0, min(1.0 - height, center_y - height / 2.0))
    return (x1, y1, x1 + width, y1 + height)


def estimate_change_region(
    source: np.ndarray,
    target: np.ndarray,
    index: int,
    analysis_long_edge: int = 512,
    expand_factor: float = 2.0,
    minimum_crop_fraction: float = 0.24,
) -> RegionProposal:
    """Estimate a robust normalized edit box from one aligned source/target frame pair."""
    import cv2

    source_small = resize_long_edge(source, analysis_long_edge)
    height, width = source_small.shape[:2]
    target_small = cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA)
    source_float = source_small.astype(np.float32) / 255.0
    target_float = target_small.astype(np.float32) / 255.0

    # Color change finds the edit itself. Gradient change reduces sensitivity to a
    # mild global brightness shift while retaining seams, remnants, and deformations.
    color_delta = np.mean(np.abs(source_float - target_float), axis=2)
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    source_grad_x = cv2.Sobel(source_gray, cv2.CV_32F, 1, 0, ksize=3)
    source_grad_y = cv2.Sobel(source_gray, cv2.CV_32F, 0, 1, ksize=3)
    target_grad_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    target_grad_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_delta = np.minimum(
        1.0,
        (np.abs(source_grad_x - target_grad_x) + np.abs(source_grad_y - target_grad_y)) / 4.0,
    )
    difference = 0.78 * color_delta + 0.22 * gradient_delta
    baseline = float(np.percentile(difference, 35.0))
    signal = np.maximum(0.0, difference - baseline)
    signal = cv2.GaussianBlur(signal, (0, 0), sigmaX=2.2, sigmaY=2.2)
    # Edits such as a microphone, earring, or handheld object can occupy well
    # below five percent of the frame. A p95 gate would incorrectly call those
    # pairs unchanged, so localization strength is measured at p99.5.
    change_score = float(np.percentile(signal, 99.5))

    if not np.isfinite(signal).all() or change_score < 0.022:
        full = (0.0, 0.0, 1.0, 1.0)
        return RegionProposal(index, full, full, 0.0, max(0.0, change_score), True)

    threshold = max(
        0.022,
        float(np.percentile(signal, 84.0)),
        float(signal.mean() + 0.65 * signal.std()),
    )
    binary = (signal >= threshold).astype(np.uint8)
    kernel_size = max(3, int(round(min(height, width) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[float, int, int, int, int]] = []
    minimum_area = max(10, round(height * width * 0.00035))
    for component in range(1, count):
        x, y, component_width, component_height, area = stats[component]
        if area < minimum_area:
            continue
        component_score = float(signal[labels == component].sum())
        components.append((component_score, x, y, component_width, component_height))

    if not components:
        full = (0.0, 0.0, 1.0, 1.0)
        return RegionProposal(index, full, full, 0.0, change_score, True)

    components.sort(reverse=True)
    strongest = components[0][0]
    kept = [component for component in components if component[0] >= strongest * 0.10][:8]
    x1 = min(component[1] for component in kept)
    y1 = min(component[2] for component in kept)
    x2 = max(component[1] + component[3] for component in kept)
    y2 = max(component[2] + component[4] for component in kept)
    raw_box = (x1 / width, y1 / height, x2 / width, y2 / height)

    area_fraction = max(1e-6, (x2 - x1) * (y2 - y1) / (width * height))
    total_energy = float(signal.sum()) + 1e-8
    inside_energy = float(signal[y1:y2, x1:x2].sum())
    energy_fraction = inside_energy / total_energy
    concentration = energy_fraction / area_fraction
    dynamic_strength = min(1.0, change_score / 0.12)
    spatial_strength = min(1.0, max(0.0, (concentration - 1.0) / 3.0))
    reliability = float(dynamic_strength * (0.35 + 0.65 * spatial_strength))
    fallback = bool(area_fraction > 0.82 or reliability < 0.10)
    if fallback and area_fraction > 0.82:
        raw_box = (0.0, 0.0, 1.0, 1.0)
        expanded = raw_box
    else:
        expanded = _expanded_box(raw_box, expand_factor, minimum_crop_fraction)
    return RegionProposal(index, raw_box, expanded, reliability, change_score, fallback)


def crop_normalized(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, math.floor(x1 * width)))
    top = max(0, min(height - 1, math.floor(y1 * height)))
    right = max(left + 1, min(width, math.ceil(x2 * width)))
    bottom = max(top + 1, min(height, math.ceil(y2 * height)))
    return frame[top:bottom, left:right]


def prepare_visual_inputs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    full_max_long_edge: int,
    roi_max_long_edge: int,
    roi_pair_count: int,
    analysis_long_edge: int,
    expand_factor: float,
) -> tuple[list[tuple[Any, Any]], list[tuple[int, Any, Any]], list[RegionProposal]]:
    from PIL import Image

    proposals = [
        estimate_change_region(
            source,
            target,
            index,
            analysis_long_edge=analysis_long_edge,
            expand_factor=expand_factor,
        )
        for index, (source, target) in enumerate(pairs)
    ]
    full_images = [
        (
            Image.fromarray(resize_long_edge(source, full_max_long_edge)),
            Image.fromarray(resize_long_edge(target, full_max_long_edge)),
        )
        for source, target in pairs
    ]
    selected = sorted(
        sorted(proposals, key=lambda proposal: (proposal.reliability, proposal.change_score), reverse=True)[
            : min(roi_pair_count, len(proposals))
        ],
        key=lambda proposal: proposal.index,
    )
    # A true no-op has no source-target change from which to localize the requested
    # object. Duplicating the low-resolution full frame would add no evidence, so
    # uncertain proposals scan overlapping quadrants instead. Four fallback crops
    # cover the full canvas while remaining high-resolution rectangular views.
    scan_boxes = [
        (0.0, 0.0, 0.58, 0.58),
        (0.42, 0.0, 1.0, 0.58),
        (0.0, 0.42, 0.58, 1.0),
        (0.42, 0.42, 1.0, 1.0),
    ]
    fallback_index = 0
    for proposal in selected:
        if proposal.fallback:
            proposal.expanded_box = scan_boxes[fallback_index % len(scan_boxes)]
            fallback_index += 1
    roi_images = []
    for proposal in selected:
        source, target = pairs[proposal.index]
        source_crop = resize_long_edge(crop_normalized(source, proposal.expanded_box), roi_max_long_edge)
        target_crop = resize_long_edge(crop_normalized(target, proposal.expanded_box), roi_max_long_edge)
        roi_images.append((proposal.index, Image.fromarray(source_crop), Image.fromarray(target_crop)))
    return full_images, roi_images, proposals


def labeled_pair_strip(source: Any, target: Any) -> Any:
    """Place SOURCE and TARGET in one image with persistent colored role labels."""
    from PIL import Image, ImageDraw

    header = 32
    gap = 4
    height = max(source.height, target.height)
    canvas = Image.new("RGB", (source.width + target.width + gap, height + header), (24, 24, 24))
    target_x = source.width + gap
    canvas.paste(source, (0, header + (height - source.height) // 2))
    canvas.paste(target, (target_x, header + (height - target.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, source.width - 1, header - 1), fill=(35, 105, 220))
    draw.rectangle(
        (target_x, 0, target_x + target.width - 1, header - 1), fill=(225, 120, 30)
    )
    draw.text((10, 8), "SOURCE", fill=(255, 255, 255))
    draw.text((target_x + 10, 8), "TARGET", fill=(255, 255, 255))
    return canvas


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I | re.S)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")


def normalize_judgment(value: dict[str, Any]) -> dict[str, Any]:
    reported = str(value.get("decision", "")).lower().strip()
    if reported and reported not in VALID_DECISIONS:
        reported = "uncertain"
    result: dict[str, Any] = {"reported_decision": reported}
    for field in CHECK_FIELDS:
        check = str(value.get(field, "uncertain")).lower().strip()
        result[field] = check if check in VALID_CHECKS else "uncertain"
    if any(result[field] == "fail" for field in CHECK_FIELDS):
        decision = "rejected"
    elif all(result[field] == "pass" for field in CHECK_FIELDS):
        decision = "accepted"
    else:
        decision = "uncertain"
    result["decision"] = decision
    raw_codes = value.get("reason_codes", [])
    if isinstance(raw_codes, str):
        raw_codes = [raw_codes]
    result["reason_codes"] = [
        re.sub(r"[^a-z0-9_]+", "_", str(code).lower()).strip("_")[:80]
        for code in raw_codes
        if str(code).strip()
    ][:8]
    if reported and reported != decision:
        result["reason_codes"].append("inconsistent_overall_decision")
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = min(1.0, max(0.0, confidence))
    result["short_evidence"] = str(value.get("short_evidence", "")).strip()[:500]
    return result


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class QwenRectPairJudge:
    def __init__(self, model_id: str, device: str, local_files_only: bool) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, attn_implementation="flash_attention_2", **kwargs
            )
        except (ImportError, ValueError):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, attn_implementation="sdpa", **kwargs
            )
        self.model.to(self.device).eval()

    def judge(
        self,
        full_images: list[tuple[Any, Any]],
        roi_images: list[tuple[int, Any, Any]],
        instruction: str,
        operation: str,
        rubric: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Time-matched FULL pairs follow. In every image, SOURCE is the blue-labeled "
                    "left half and TARGET is the orange-labeled right half."
                ),
            }
        ]
        images: list[Any] = []
        for index, (source, target) in enumerate(full_images, 1):
            pair_image = labeled_pair_strip(source, target)
            content.append({"type": "text", "text": f"FULL pair time {index}"})
            content.append({"type": "image", "image": pair_image})
            images.append(pair_image)
        content.append(
            {
                "type": "text",
                "text": (
                    "High-resolution expanded RECTANGULAR change-region crops follow. "
                    "They use the same coordinates in SOURCE and TARGET."
                ),
            }
        )
        for frame_index, source, target in roi_images:
            display_index = frame_index + 1
            pair_image = labeled_pair_strip(source, target)
            content.append({"type": "text", "text": f"ROI pair time {display_index}"})
            content.append({"type": "image", "image": pair_image})
            images.append(pair_image)
        content.append(
            {
                "type": "text",
                "text": f"Operation: {operation}\nInstruction: {instruction}\n\nRubric:\n{rubric}",
            }
        )
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        continuation = generated[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        peak_mb = (
            self.torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )
        return response, int(inputs["input_ids"].shape[1]), float(peak_mb)


def mock_response(row: dict[str, str]) -> str:
    digest = hashlib.sha256(row["sample_id"].encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    if unit < 0.2:
        check, reason = "fail", "mock_instruction_failure"
    elif unit < 0.4:
        check, reason = "uncertain", "mock_insufficient_evidence"
    else:
        check, reason = "pass", "mock_pass"
    decision = {"fail": "rejected", "uncertain": "uncertain", "pass": "accepted"}[check]
    return json.dumps(
        {
            "decision": decision,
            "instruction_following": check,
            "visible_artifact": check,
            "content_preservation": check,
            "reason_codes": [reason],
            "confidence": round(0.55 + 0.44 * abs(unit - 0.3), 3),
            "short_evidence": "Deterministic mock response for pipeline validation.",
        }
    )


def blank_result(row: dict[str, str], model_id: str, backend: str, rubric_hash: str) -> dict[str, str]:
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "judge_version": JUDGE_VERSION,
            "roi_method_version": ROI_METHOD_VERSION,
            "visual_layout_version": VISUAL_LAYOUT_VERSION,
            "model_id": model_id,
            "model_backend": backend,
            "prompt_sha256": rubric_hash,
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "branch": row["branch"],
            "operation": row["operation"],
            "target_scope": row.get("target_scope", ""),
            "original_stage1_1_decision": row.get("stage1_1_decision", ""),
        }
    )
    return result


def serialize_proposals(proposals: list[RegionProposal]) -> str:
    values = [
        {
            "frame_index": proposal.index,
            "raw_box_norm": [round(value, 6) for value in proposal.raw_box],
            "expanded_box_norm": [round(value, 6) for value in proposal.expanded_box],
            "reliability": round(proposal.reliability, 6),
            "change_score": round(proposal.change_score, 6),
            "fallback": proposal.fallback,
        }
        for proposal in proposals
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def score_row(
    row: dict[str, str],
    judge: QwenRectPairJudge | None,
    model_id: str,
    rubric: str,
    frame_samples: int,
    roi_pair_count: int,
    full_max_long_edge: int,
    roi_max_long_edge: int,
    analysis_long_edge: int,
    expand_factor: float,
    max_new_tokens: int,
    mock: bool,
) -> dict[str, str]:
    started = time.monotonic()
    backend = "mock" if mock else "transformers"
    result = blank_result(row, model_id, backend, prompt_hash(rubric))
    try:
        pairs = matched_frames(
            resolve_media(row["source_local"]), resolve_media(row["target_local"]), frame_samples
        )
        full_images, roi_images, proposals = prepare_visual_inputs(
            pairs,
            full_max_long_edge,
            roi_max_long_edge,
            roi_pair_count,
            analysis_long_edge,
            expand_factor,
        )
        reliabilities = [proposal.reliability for proposal in proposals]
        result.update(
            {
                "sampled_full_pairs": str(len(full_images)),
                "sampled_roi_pairs": str(len(roi_images)),
                "roi_boxes_json": serialize_proposals(proposals),
                "roi_reliability_min": f"{min(reliabilities, default=0.0):.6f}",
                "roi_reliability_mean": f"{np.mean(reliabilities) if reliabilities else 0.0:.6f}",
                "roi_fallback_count": str(sum(proposal.fallback for proposal in proposals)),
            }
        )
        if mock:
            raw = mock_response(row)
            result["input_tokens"] = "0"
            result["peak_gpu_memory_mb"] = "0.000"
        else:
            raw, input_tokens, peak_mb = judge.judge(  # type: ignore[union-attr]
                full_images,
                roi_images,
                row["instruction"],
                row["operation"],
                rubric,
                max_new_tokens,
            )
            result["input_tokens"] = str(input_tokens)
            result["peak_gpu_memory_mb"] = f"{peak_mb:.3f}"
        result["raw_response"] = raw
        normalized = normalize_judgment(extract_json_object(raw))
        result.update(
            {
                "model_reported_decision": normalized["reported_decision"],
                "model_decision": normalized["decision"],
                **{field: normalized[field] for field in CHECK_FIELDS},
                "reason_codes_json": json.dumps(normalized["reason_codes"], ensure_ascii=False),
                "model_confidence": f"{normalized['confidence']:.6f}",
                "short_evidence": normalized["short_evidence"],
                "parse_status": "pass",
            }
        )
    except Exception as error:
        result["model_decision"] = "uncertain"
        result["parse_status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-samples", type=int, default=8)
    parser.add_argument("--roi-pair-count", type=int, default=4)
    parser.add_argument("--full-max-long-edge", type=int, default=448)
    parser.add_argument("--roi-max-long-edge", type=int, default=672)
    parser.add_argument("--analysis-long-edge", type=int, default=512)
    parser.add_argument("--expand-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    rubric = args.prompt.read_text(encoding="utf-8").strip()
    rows = read_csv(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing: dict[str, dict[str, str]] = {}
    if args.resume and args.output.is_file():
        existing_rows = read_csv(args.output)
        incompatible = [
            row for row in existing_rows if row.get("judge_version") not in {"", JUDGE_VERSION}
        ]
        if incompatible:
            raise ValueError(f"resume output contains a different judge version: {args.output}")
        existing = {
            row["sample_id"]: row
            for row in existing_rows
            if row.get("parse_status") in {"pass", "error"}
        }
    pending = [row for row in rows if row["sample_id"] not in existing]
    judge = None if args.mock_model else QwenRectPairJudge(
        args.model, args.device, args.local_files_only
    )
    results = list(existing.values())
    run_started = time.monotonic()
    for index, row in enumerate(pending, 1):
        results.append(
            score_row(
                row,
                judge,
                args.model,
                rubric,
                args.frame_samples,
                args.roi_pair_count,
                args.full_max_long_edge,
                args.roi_max_long_edge,
                args.analysis_long_edge,
                args.expand_factor,
                args.max_new_tokens,
                args.mock_model,
            )
        )
        if index % args.checkpoint_every == 0 or index == len(pending):
            write_csv(args.output, results)
        print(f"qwen_rect_judged={index}/{len(pending)} total={len(results)}", flush=True)

    elapsed = time.monotonic() - run_started
    summary = {
        "judge_version": JUDGE_VERSION,
        "roi_method_version": ROI_METHOD_VERSION,
        "visual_layout_version": VISUAL_LAYOUT_VERSION,
        "model_id": args.model,
        "model_backend": "mock" if args.mock_model else "transformers",
        "prompt_sha256": prompt_hash(rubric),
        "input_rows": len(rows),
        "completed": len(results),
        "parse_errors": sum(row.get("parse_status") != "pass" for row in results),
        "decision_counts": {
            decision: sum(row.get("model_decision") == decision for row in results)
            for decision in sorted(VALID_DECISIONS)
        },
        "new_run_elapsed_seconds": elapsed,
        "seconds_per_new_pair": elapsed / len(pending) if pending else 0.0,
        "mean_input_tokens": (
            sum(int(row.get("input_tokens") or 0) for row in results) / len(results)
            if results
            else 0.0
        ),
        "peak_gpu_memory_mb": max(
            (float(row.get("peak_gpu_memory_mb") or 0) for row in results), default=0.0
        ),
        "mean_roi_reliability": (
            sum(float(row.get("roi_reliability_mean") or 0) for row in results) / len(results)
            if results
            else 0.0
        ),
        "pairs_with_any_roi_fallback": sum(
            int(row.get("roi_fallback_count") or 0) > 0 for row in results
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
