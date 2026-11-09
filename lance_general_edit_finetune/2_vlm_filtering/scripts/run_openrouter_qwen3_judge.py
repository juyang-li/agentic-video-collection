#!/usr/bin/env python3
"""Run the CPU-preprocessed Stage 2 pair judge through OpenRouter Qwen3-VL-32B."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

import run_qwen_rect_judge as rect


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
DEFAULT_INPUT = SPECIALIST_ROOT / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_RUN = ROOT / "artifacts/qwen3_openrouter_v1"
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen3_pair_scores.csv"
DEFAULT_SUMMARY = DEFAULT_RUN / "qwen3_pair_score_summary.json"
DEFAULT_GATES = DEFAULT_RUN / "qwen3_calibration_gates.json"
DEFAULT_PROMPT = ROOT / "judge_prompt_timecard_v12_provenance_aware.txt"
DEFAULT_CARD_COUNT = 5
AUDIT_TASK_ONLY_PRESENTATION = False
DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_VERSION = "qwen3_vl_openrouter_provenance_aware_pair_judge_v7"
ROI_METHOD_VERSION = "paired_tight_change_roi_five_aligned_v6"
VISUAL_LAYOUT_VERSION = "five_aligned_full_tight_roi_timecards_v7"
CHECK_FIELDS = ["instruction_following", "visible_artifact", "content_preservation"]
VALID_CHECKS = {"pass", "fail", "uncertain"}
VALID_DECISIONS = {"accepted", "rejected", "uncertain"}
VALID_FRAME_CHECKS = {"pass", "fail", "not_visible"}
TEMPORAL_CHECK_FIELDS = ["instruction_following", "visible_artifact"]
REMOVAL_AUDIT_CHECK_FIELDS = [
    "removed_region_quality",
    "large_blur_fill",
    "body_reconstruction",
    "hair_reconstruction",
]
VALID_REMOVAL_CHECKS = {"pass", "fail", "not_applicable"}
JSON_SCHEMA = {
    "name": "video_edit_pair_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": sorted(VALID_DECISIONS)},
            **{
                field: {"type": "string", "enum": sorted(VALID_CHECKS)}
                for field in CHECK_FIELDS
            },
            "reason_codes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {field: {"type": "string"} for field in CHECK_FIELDS},
                "required": CHECK_FIELDS,
            },
        },
        "required": [
            "decision",
            *CHECK_FIELDS,
            "reason_codes",
            "evidence",
        ],
    },
}
FIELDS = [
    "judge_version",
    "roi_method_version",
    "visual_layout_version",
    "model_id",
    "model_backend",
    "provider",
    "generation_id",
    "prompt_sha256",
    "sample_id",
    "dataset",
    "branch",
    "operation",
    "target_scope",
    "generated_side",
    "training_operation",
    "training_instruction",
    "audit_operation",
    "audit_relation",
    "audit_routing_transform",
    "audit_left_training_role",
    "audit_right_training_role",
    "audit_left_label",
    "audit_right_label",
    "audit_generated_position",
    "original_stage1_1_decision",
    "sampled_full_pairs",
    "sampled_roi_pairs",
    "roi_boxes_json",
    "roi_reliability_min",
    "roi_reliability_mean",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "reasoning_effort",
    "cost_usd",
    "cost_source",
    "api_attempts",
    "api_elapsed_seconds",
    "model_reported_decision",
    "model_decision",
    *CHECK_FIELDS,
    "reason_codes_json",
    "removal_audit_json",
    "frame_audit_json",
    "temporal_audit_json",
    "evidence_json",
    "parse_status",
    "raw_response",
    "error",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class AuditView:
    """A provenance-normalized audit view without rewriting training text."""

    left_training_role: str
    right_training_role: str
    left_label: str
    right_label: str
    audit_operation: str
    audit_relation: str
    routing_transform: str
    generated_position: str


@dataclass(frozen=True)
class TimeCard:
    """One aligned audit time containing full context and an optional tight ROI."""

    frame_index: int
    display_index: int
    time_percent: int
    full_left: Any
    full_right: Any
    roi_left: Any | None
    roi_right: Any | None


OPERATION_ALIASES = {
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "replacement": "replace",
    "change": "replace",
}
INVERSE_OPERATIONS = {"add": "remove", "remove": "add", "replace": "replace"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SPECIALIST_ROOT / path


def derive_generated_side(row: dict[str, str]) -> str:
    source = row.get("source_provenance", "").strip().lower()
    target = row.get("target_provenance", "").strip().lower()
    if source == "edited" and target == "original":
        return "SOURCE"
    if source == "original" and target == "edited":
        return "TARGET"
    if source == "edited" and target == "edited":
        return "BOTH"
    if (
        source == "generation_side_unresolved"
        and target == "generation_side_unresolved"
    ):
        return "UNRESOLVED"
    return "UNKNOWN"


def normalize_operation(value: str) -> str:
    operation = value.strip().lower()
    try:
        return OPERATION_ALIASES[operation]
    except KeyError as error:
        raise ValueError(f"unsupported training operation: {value}") from error


def target_description_from_instruction(
    training_instruction: str, training_operation: str
) -> str:
    """Strip only a leading add/remove command from dataset metadata.

    The VLM then receives a target description instead of a second command
    that can conflict with the routed audit operation. Arbitrary instruction
    semantics are intentionally not rewritten here.
    """
    description = training_instruction.strip()
    operation = normalize_operation(training_operation)
    if operation not in {"add", "remove"}:
        return description
    command = r"add|insert|place|put" if operation == "add" else r"remove|delete|erase"
    stripped = re.sub(
        rf"^\s*(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:{command})\b\s*",
        "",
        description,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return stripped or description


def audit_task_text(
    training_operation: str,
    training_instruction: str,
    audit_view: AuditView,
) -> str:
    """Build one unambiguous visual audit task for the VLM."""
    if audit_view.generated_position == "UNRESOLVED_CHECK_BOTH":
        return (
            "唯一审查任务：生成侧无法从发布 metadata 中可靠确定。"
            "instruction_following 与 content_preservation 按左侧 SRC_VIDEO -> "
            "右侧 TARGET_VIDEO 判断；visible_artifact 独立检查两侧视频。\n"
            f"原始训练指令：{training_instruction}"
        )
    description = target_description_from_instruction(
        training_instruction, training_operation
    )
    if audit_view.audit_operation == "remove":
        return (
            "唯一审查任务：确认右侧 GENERATED_VIDEO 是否从左侧 REFERENCE_VIDEO 中"
            "完整去除了下述目标，并真实、清晰、结构完整地重建了原先被遮挡的内容。\n"
            f"待移除目标（这是名词性描述，不是另一条编辑命令）：{description}"
        )
    if audit_view.audit_operation == "add":
        return (
            "唯一审查任务：确认右侧 GENERATED_VIDEO 是否在左侧 REFERENCE_VIDEO 的"
            "基础上自然、完整且稳定地加入了下述目标。\n"
            f"待添加目标（这是名词性描述，不是另一条编辑命令）：{description}"
        )
    return (
        "唯一审查任务：按照下述数据集编辑描述，比较左侧 REFERENCE_VIDEO 与右侧 "
        "GENERATED_VIDEO；当前结构化操作为 replacement。\n"
        f"编辑描述：{description}"
    )


def build_audit_view(row: dict[str, str]) -> AuditView:
    """Route media to stable audit roles without mutating TRAIN_INSTRUCTION.

    TRAIN_SOURCE/TRAIN_TARGET remain the fine-tuning pair. For one-sided
    generated data, the audit display is always REFERENCE_VIDEO on the left and
    GENERATED_VIDEO on the right. Only structured operation semantics are
    inverted; the free-form instruction remains immutable provenance.
    """
    generated_side = derive_generated_side(row)
    training_operation = normalize_operation(row.get("operation", ""))
    if generated_side == "UNRESOLVED":
        return AuditView(
            left_training_role="source",
            right_training_role="target",
            left_label="SRC_VIDEO",
            right_label="TARGET_VIDEO",
            audit_operation=training_operation,
            audit_relation="training_direction_generation_side_unresolved",
            routing_transform="keep_training_order_check_artifacts_on_both",
            generated_position="UNRESOLVED_CHECK_BOTH",
        )
    if generated_side == "SOURCE":
        audit_operation = INVERSE_OPERATIONS[training_operation]
        return AuditView(
            left_training_role="target",
            right_training_role="source",
            left_label="REFERENCE_VIDEO",
            right_label="GENERATED_VIDEO",
            audit_operation=audit_operation,
            audit_relation="inverse_replace" if training_operation == "replace" else "inverse",
            routing_transform="swap_media_and_invert_structured_operation",
            generated_position="RIGHT",
        )
    if generated_side == "TARGET":
        return AuditView(
            left_training_role="source",
            right_training_role="target",
            left_label="REFERENCE_VIDEO",
            right_label="GENERATED_VIDEO",
            audit_operation=training_operation,
            audit_relation="forward",
            routing_transform="identity",
            generated_position="RIGHT",
        )
    if generated_side != "BOTH":
        raise ValueError(
            "cannot route audit media from provenance: "
            f"source={row.get('source_provenance', '')!r}, "
            f"target={row.get('target_provenance', '')!r}"
        )
    # Some datasets contain two generated endpoints.  Their training direction
    # is still meaningful, but there is no single original/edited swap.
    return AuditView(
        left_training_role="source",
        right_training_role="target",
        left_label="TRAIN_SOURCE_GENERATED",
        right_label="TRAIN_TARGET_GENERATED",
        audit_operation=training_operation,
        audit_relation="training_direction_both_generated",
        routing_transform="keep_training_order_both_generated",
        generated_position="BOTH",
    )


def select_reliable_proposals(
    proposals: list[rect.RegionProposal], count: int, minimum_reliability: float
) -> list[rect.RegionProposal]:
    eligible = [
        proposal
        for proposal in proposals
        if not proposal.fallback and proposal.reliability >= minimum_reliability
    ]
    if not eligible or count <= 0:
        return []

    ranked = sorted(
        eligible,
        key=lambda proposal: (proposal.reliability, proposal.change_score),
        reverse=True,
    )
    anchor = ranked[0]
    maximum_index = max(proposal.index for proposal in proposals)
    if count == 1 or maximum_index == 0:
        desired_indices = [anchor.index]
    else:
        desired_indices = sorted(
            {
                round(position * maximum_index / (count - 1))
                for position in range(count)
            }
        )
    eligible_by_index = {proposal.index: proposal for proposal in eligible}
    selected: list[rect.RegionProposal] = []
    for index in desired_indices:
        proposal = eligible_by_index.get(index)
        if proposal is None:
            # A nearly unchanged early frame is exactly where a late-only edit
            # can hide. Reuse the strongest edit coordinates at that time
            # instead of omitting it or falling back to an unhelpful full frame.
            proposal = rect.RegionProposal(
                index=index,
                raw_box=anchor.raw_box,
                expanded_box=anchor.expanded_box,
                reliability=anchor.reliability,
                change_score=0.0,
                fallback=False,
            )
        selected.append(proposal)
    return selected


def prepare_visual_inputs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    full_pair_count: int,
    full_max_long_edge: int,
    roi_max_long_edge: int,
    roi_pair_count: int,
    analysis_long_edge: int,
    expand_factor: float,
    minimum_roi_reliability: float,
) -> tuple[
    list[tuple[Any, Any]],
    list[tuple[int, Any, Any]],
    list[rect.RegionProposal],
    list[rect.RegionProposal],
]:
    from PIL import Image

    proposals = [
        rect.estimate_change_region(
            source,
            target,
            index,
            analysis_long_edge=analysis_long_edge,
            expand_factor=expand_factor,
        )
        for index, (source, target) in enumerate(pairs)
    ]
    selected = select_reliable_proposals(proposals, roi_pair_count, minimum_roi_reliability)
    full_indices = np.unique(
        np.rint(np.linspace(0, len(pairs) - 1, min(full_pair_count, len(pairs)))).astype(int)
    )
    full_images = [
        (
            Image.fromarray(rect.resize_long_edge(pairs[index][0], full_max_long_edge)),
            Image.fromarray(rect.resize_long_edge(pairs[index][1], full_max_long_edge)),
        )
        for index in full_indices
    ]
    roi_images: list[tuple[int, Any, Any]] = []
    for proposal in selected:
        source, target = pairs[proposal.index]
        source_crop = rect.resize_long_edge(
            rect.crop_normalized(source, proposal.expanded_box), roi_max_long_edge
        )
        target_crop = rect.resize_long_edge(
            rect.crop_normalized(target, proposal.expanded_box), roi_max_long_edge
        )
        roi_images.append(
            (proposal.index, Image.fromarray(source_crop), Image.fromarray(target_crop))
        )
    return full_images, roi_images, proposals, selected


def tighten_box(
    box: tuple[float, float, float, float],
    *,
    expand_factor: float = 1.35,
    minimum_side_fraction: float = 0.14,
    maximum_area_fraction: float = 0.30,
) -> tuple[float, float, float, float]:
    """Make a compact context crop centered on a raw change box.

    A diffuse difference mask can cover most of a frame.  The v4 time-card
    layout intentionally caps the diagnostic crop rather than pretending that
    a near-full-frame rectangle provides local detail.
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = min(1.0, max(minimum_side_fraction, (x2 - x1) * expand_factor))
    height = min(1.0, max(minimum_side_fraction, (y2 - y1) * expand_factor))
    area = width * height
    if area > maximum_area_fraction:
        scale = (maximum_area_fraction / area) ** 0.5
        width *= scale
        height *= scale
    left = max(0.0, min(1.0 - width, center_x - width / 2.0))
    top = max(0.0, min(1.0 - height, center_y - height / 2.0))
    return (left, top, left + width, top + height)


def prepare_time_cards(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    card_count: int,
    full_max_long_edge: int,
    roi_max_long_edge: int,
    analysis_long_edge: int,
    minimum_roi_reliability: float,
) -> tuple[list[TimeCard], list[rect.RegionProposal], list[rect.RegionProposal]]:
    """Build five aligned FULL+ROI cards, skipping the conditioning frame."""
    from PIL import Image

    indices = np.unique(
        np.rint(np.linspace(0, len(pairs) - 1, min(card_count, len(pairs)))).astype(int)
    )
    if len(pairs) > card_count and len(indices) and indices[0] == 0:
        # These datasets edit the first frame and propagate it forward. The
        # conditioning frame is weak evidence of propagation quality, so use
        # the next decoded time point for F1 and leave F2-F5 unchanged.
        indices[0] = 1
    proposals = [
        rect.estimate_change_region(
            left,
            right,
            index,
            analysis_long_edge=analysis_long_edge,
            expand_factor=1.0,
        )
        for index, (left, right) in enumerate(pairs)
    ]
    eligible = [
        proposal
        for proposal in proposals
        if not proposal.fallback and proposal.reliability >= minimum_roi_reliability
    ]
    selected: list[rect.RegionProposal] = []
    if eligible:
        anchor = max(
            eligible,
            key=lambda proposal: (proposal.reliability, proposal.change_score),
        )
        eligible_by_index = {proposal.index: proposal for proposal in eligible}
        for index in indices:
            proposal = eligible_by_index.get(int(index))
            if proposal is None:
                proposal = rect.RegionProposal(
                    index=int(index),
                    raw_box=anchor.raw_box,
                    expanded_box=anchor.expanded_box,
                    reliability=anchor.reliability,
                    change_score=0.0,
                    fallback=False,
                )
            selected.append(proposal)
    if selected:
        tight_selected = [
            rect.RegionProposal(
                index=proposal.index,
                raw_box=proposal.raw_box,
                expanded_box=tighten_box(proposal.raw_box),
                reliability=proposal.reliability,
                change_score=proposal.change_score,
                fallback=False,
            )
            for proposal in selected
        ]
    else:
        # For extremely small edits the difference localizer can legitimately
        # return no region. Still keep the promised five-card structure with a
        # compact central crop; the prompt tells the judge to fall back to FULL
        # when this automatic crop misses the target.
        central_box = tighten_box((0.0, 0.0, 1.0, 1.0), expand_factor=1.0)
        tight_selected = [
            rect.RegionProposal(
                index=int(index),
                raw_box=(0.0, 0.0, 1.0, 1.0),
                expanded_box=central_box,
                reliability=0.0,
                change_score=proposals[int(index)].change_score,
                fallback=True,
            )
            for index in indices
        ]
    selected_by_index = {proposal.index: proposal for proposal in tight_selected}
    cards: list[TimeCard] = []
    denominator = max(1, len(pairs) - 1)
    for display_index, frame_index in enumerate(indices, 1):
        left, right = pairs[frame_index]
        full_left = Image.fromarray(rect.resize_long_edge(left, full_max_long_edge))
        full_right = Image.fromarray(rect.resize_long_edge(right, full_max_long_edge))
        proposal = selected_by_index.get(int(frame_index))
        if proposal is None:
            roi_left = None
            roi_right = None
        else:
            roi_left = Image.fromarray(
                rect.resize_long_edge(
                    rect.crop_normalized(left, proposal.expanded_box), roi_max_long_edge
                )
            )
            roi_right = Image.fromarray(
                rect.resize_long_edge(
                    rect.crop_normalized(right, proposal.expanded_box), roi_max_long_edge
                )
            )
        cards.append(
            TimeCard(
                frame_index=int(frame_index),
                display_index=display_index,
                time_percent=round(int(frame_index) * 100 / denominator),
                full_left=full_left,
                full_right=full_right,
                roi_left=roi_left,
                roi_right=roi_right,
            )
        )
    return cards, proposals, tight_selected


def labeled_pair_strip(
    left: Any,
    right: Any,
    label: str,
    time_percent: int,
    left_label: str,
    right_label: str,
) -> Any:
    from PIL import Image, ImageDraw

    header = 38
    gap = 4
    height = max(left.height, right.height)
    canvas = Image.new(
        "RGB", (left.width + right.width + gap, height + header), (24, 24, 24)
    )
    right_x = left.width + gap
    canvas.paste(left, (0, header + (height - left.height) // 2))
    canvas.paste(right, (right_x, header + (height - right.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, left.width - 1, header - 1), fill=(35, 105, 220))
    draw.rectangle((right_x, 0, canvas.width - 1, header - 1), fill=(225, 120, 30))
    draw.text(
        (9, 11),
        f"{left_label} | {label} | t={time_percent}%",
        fill=(255, 255, 255),
    )
    draw.text(
        (right_x + 9, 11),
        f"{right_label} | {label} | t={time_percent}%",
        fill=(255, 255, 255),
    )
    return canvas


def temporal_roi_sheet(
    rows: list[tuple[int, Any, Any]],
    sheet_index: int,
    denominator: int,
    left_label: str,
    right_label: str,
) -> Any:
    from PIL import Image

    strips = []
    for frame_index, source, target in rows:
        time_percent = round(frame_index * 100 / denominator)
        strips.append(
            labeled_pair_strip(
                source,
                target,
                f"R{sheet_index}-T{frame_index + 1}",
                time_percent,
                left_label,
                right_label,
            )
        )
    width = max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips)
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    top = 0
    for strip in strips:
        sheet.paste(strip, (0, top))
        top += strip.height
    return sheet


def time_card_image(card: TimeCard, audit_view: AuditView) -> Any:
    """Stack one full pair and its same-time tight ROI into one API image."""
    from PIL import Image, ImageDraw

    full_strip = labeled_pair_strip(
        card.full_left,
        card.full_right,
        f"F{card.display_index} | FULL",
        card.time_percent,
        audit_view.left_label,
        audit_view.right_label,
    )
    if card.roi_left is not None and card.roi_right is not None:
        roi_strip = labeled_pair_strip(
            card.roi_left,
            card.roi_right,
            f"F{card.display_index} | TIGHT_ROI",
            card.time_percent,
            audit_view.left_label,
            audit_view.right_label,
        )
    else:
        roi_strip = Image.new("RGB", (full_strip.width, 82), (30, 30, 30))
        draw = ImageDraw.Draw(roi_strip)
        draw.text(
            (12, 30),
            f"F{card.display_index} | TIGHT_ROI_UNAVAILABLE | use FULL only",
            fill=(255, 210, 80),
        )
    gap = 8
    width = max(full_strip.width, roi_strip.width)
    canvas = Image.new(
        "RGB", (width, full_strip.height + gap + roi_strip.height), (20, 20, 20)
    )
    canvas.paste(full_strip, (0, 0))
    canvas.paste(roi_strip, (0, full_strip.height + gap))
    return canvas


def image_data_url(image: Any, jpeg_quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_content(
    full_images: list[tuple[Any, Any]],
    roi_images: list[tuple[int, Any, Any]],
    training_operation: str,
    training_instruction: str,
    audit_view: AuditView,
    rubric: str,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "请按下列结构化审查视图判断，不要擅自改写训练指令。\n"
                f"- TRAIN_OPERATION：{training_operation}\n"
                "- TRAIN_INSTRUCTION（原文，只描述 TRAIN_SOURCE -> TRAIN_TARGET）："
                f"{training_instruction}\n"
                f"- AUDIT_OPERATION：{audit_view.audit_operation}\n"
                f"- AUDIT_RELATION：{audit_view.audit_relation}\n"
                f"- 左图 {audit_view.left_label}=TRAIN_{audit_view.left_training_role.upper()}\n"
                f"- 右图 {audit_view.right_label}=TRAIN_{audit_view.right_training_role.upper()}\n"
                f"- 路由方式：{audit_view.routing_transform}\n"
                f"- 生成位置：{audit_view.generated_position}\n\n"
                "下面先给出按时间顺序排列的全图对照。"
            ),
        }
    ]
    denominator = max(1, len(full_images) - 1)
    for index, (source, target) in enumerate(full_images):
        label = f"F{index + 1}"
        time_percent = round(index * 100 / denominator)
        strip = labeled_pair_strip(
            source,
            target,
            label,
            time_percent,
            audit_view.left_label,
            audit_view.right_label,
        )
        content.append({"type": "text", "text": f"{label}，归一化时间 {time_percent}%"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(strip, jpeg_quality)},
            }
        )
    if roi_images:
        content.append(
            {
                "type": "text",
                "text": (
                    "下面是两张高分辨率时序局部联系表；内部各行的 F 编号与全图对应。"
                    "请逐行检查，不要只看最后一行。"
                ),
            }
        )
        roi_denominator = max(1, max(frame_index for frame_index, _, _ in roi_images))
        split_at = (len(roi_images) + 1) // 2
        groups = [roi_images[:split_at], roi_images[split_at:]]
        for roi_index, group in enumerate((group for group in groups if group), 1):
            labels = ", ".join(f"T{frame_index + 1}" for frame_index, _, _ in group)
            sheet = temporal_roi_sheet(
                group,
                roi_index,
                roi_denominator,
                audit_view.left_label,
                audit_view.right_label,
            )
            content.append(
                {
                    "type": "text",
                    "text": f"R{roi_index} 时序局部联系表，包含 {labels}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(sheet, jpeg_quality)},
                }
            )
    else:
        content.append({"type": "text", "text": "没有可靠的局部区域，本次只根据全图判断。"})
    content.append(
        {
            "type": "text",
            "text": (
                f"再次确认：AUDIT_OPERATION={audit_view.audit_operation}；"
                f"AUDIT_RELATION={audit_view.audit_relation}；"
                f"TRAIN_INSTRUCTION 原文={training_instruction}\n\n{rubric}"
            ),
        }
    )
    return content


def build_timecard_content(
    cards: list[TimeCard],
    training_operation: str,
    training_instruction: str,
    audit_view: AuditView,
    rubric: str,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    card_times = ",".join(str(card.time_percent) for card in cards)
    card_range = f"F1-F{len(cards)}"
    layout_description = (
        f"下面有{len(cards)}张时间卡。每张上半部分是FULL全图pair，下半部分是同一时刻、"
        f"同坐标的TIGHT_ROI pair；{card_range}依次对应约{card_times}%。"
        f"请综合{len(cards)}张卡判断，不需要逐卡输出pass/fail。"
    )
    if AUDIT_TASK_ONLY_PRESENTATION:
        opening_text = (
            f"{audit_task_text(training_operation, training_instruction, audit_view)}\n"
            f"左图={audit_view.left_label}；右图={audit_view.right_label}。"
            "不要执行或复述原训练方向，只完成上面的唯一审查任务。\n"
            f"{layout_description}"
        )
        closing_text = (
            f"再次确认当前唯一操作：{audit_view.audit_operation.upper()}。\n"
            f"{audit_task_text(training_operation, training_instruction, audit_view)}\n\n"
            f"{rubric}"
        )
    else:
        opening_text = (
            "请按下列结构化审查视图判断，不要擅自改写训练指令。\n"
            f"- TRAIN_OPERATION：{training_operation}\n"
            "- TRAIN_INSTRUCTION（原文，只描述 TRAIN_SOURCE -> TRAIN_TARGET）："
            f"{training_instruction}\n"
            f"- AUDIT_OPERATION：{audit_view.audit_operation}\n"
            f"- AUDIT_RELATION：{audit_view.audit_relation}\n"
            f"- 左图 {audit_view.left_label}=TRAIN_{audit_view.left_training_role.upper()}\n"
            f"- 右图 {audit_view.right_label}=TRAIN_{audit_view.right_training_role.upper()}\n"
            f"- 路由方式：{audit_view.routing_transform}\n"
            f"{layout_description}"
        )
        closing_text = (
            f"再次确认：AUDIT_OPERATION={audit_view.audit_operation}；"
            f"AUDIT_RELATION={audit_view.audit_relation}；"
            f"TRAIN_INSTRUCTION原文={training_instruction}\n\n{rubric}"
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": opening_text,
        }
    ]
    for card in cards:
        label = f"F{card.display_index}"
        content.append(
            {
                "type": "text",
                "text": (
                    f"{label}，归一化时间{card.time_percent}%。"
                    "请把该卡的FULL和TIGHT_ROI作为整体判断证据，不需要单独下结论。"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(time_card_image(card, audit_view), jpeg_quality)
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": closing_text,
        }
    )
    return content


def build_payload(
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int,
    data_collection: str,
    reasoning_effort: str | None = None,
    exclude_reasoning: bool = False,
    temperature: float | None = None,
    provider_order: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
        "provider": {
            "data_collection": data_collection,
            "allow_fallbacks": False,
            "require_parameters": True,
            "sort": "price",
        },
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None or exclude_reasoning:
        reasoning: dict[str, Any] = {"exclude": exclude_reasoning}
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        payload["reasoning"] = reasoning
    if provider_order:
        payload["provider"]["order"] = provider_order
    return payload


def retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return min(60.0, max(0.5, float(header)))
        except ValueError:
            pass
    return min(30.0, 2.0 ** (attempt - 1))


def post_json(
    api_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_attempts: int,
) -> tuple[dict[str, Any], int, float]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "lance-general-edit-filter/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), attempt, time.monotonic() - started
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")[:1500]
            retryable = error.code in {408, 409, 425, 429} or error.code >= 500
            if not retryable or attempt >= max_attempts:
                raise RuntimeError(f"OpenRouter HTTP {error.code}: {response_body}") from error
            time.sleep(retry_delay(error, attempt))
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= max_attempts:
                raise RuntimeError(f"OpenRouter network error: {error}") from error
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise AssertionError("unreachable")


def extract_message(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        error_detail = json.dumps(
            data.get("error", data), ensure_ascii=False, separators=(",", ":")
        )[:1500]
        raise ValueError(f"OpenRouter response has no choices: {error_detail}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content)


def normalize_judgment(
    value: dict[str, Any], *, removal_expected: bool = False
) -> dict[str, Any]:
    reported = str(value.get("decision", "")).lower().strip()
    if reported not in VALID_DECISIONS:
        reported = "uncertain"
    checks: dict[str, str] = {}
    for field in CHECK_FIELDS:
        check = str(value.get(field, "")).lower().strip()
        checks[field] = check if check in VALID_CHECKS else "uncertain"
    raw_frame_audit = value.get("frame_audit", {})
    if not isinstance(raw_frame_audit, dict):
        raw_frame_audit = {}
    frame_audit: dict[str, dict[str, str]] = {}
    # v5 asks for one holistic decision instead of five independent verdicts.
    # Legacy frame_audit payloads remain readable for historical result files.
    if raw_frame_audit:
        for index in range(1, 6):
            frame = f"F{index}"
            raw_frame = raw_frame_audit.get(frame, {})
            if not isinstance(raw_frame, dict):
                raw_frame = {}
            frame_audit[frame] = {}
            for field in CHECK_FIELDS:
                frame_check = str(raw_frame.get(field, "")).lower().strip()
                normalized_frame_check = (
                    frame_check if frame_check in VALID_FRAME_CHECKS else "not_visible"
                )
                frame_audit[frame][field] = normalized_frame_check
                if normalized_frame_check == "fail":
                    checks[field] = "fail"
    raw_temporal_audit = value.get("temporal_audit", {})
    if not isinstance(raw_temporal_audit, dict):
        raw_temporal_audit = {}
    temporal_audit: dict[str, dict[str, str]] = {}
    # v5 has no detached T1-T9 sheet. Keep parsing the field only for backward
    # compatibility with historical responses and unit tests.
    if raw_temporal_audit:
        for index in range(1, 10):
            time_label = f"T{index}"
            raw_time = raw_temporal_audit.get(time_label, {})
            if not isinstance(raw_time, dict):
                raw_time = {}
            temporal_audit[time_label] = {}
            for field in TEMPORAL_CHECK_FIELDS:
                time_check = str(raw_time.get(field, "")).lower().strip()
                normalized_time_check = (
                    time_check if time_check in VALID_FRAME_CHECKS else "not_visible"
                )
                temporal_audit[time_label][field] = normalized_time_check
                if normalized_time_check == "fail":
                    checks[field] = "fail"
    raw_removal_audit = value.get("removal_audit", {})
    if not isinstance(raw_removal_audit, dict):
        raw_removal_audit = {}
    applicable = str(raw_removal_audit.get("applicable", "")).lower().strip()
    if applicable not in {"yes", "no"}:
        applicable = "no"
    removal_audit: dict[str, str] = {
        "applicable": applicable,
        "reference_context": str(raw_removal_audit.get("reference_context", ""))[
            :600
        ].strip(),
        "generated_fill": str(raw_removal_audit.get("generated_fill", ""))[
            :600
        ].strip(),
    }
    removal_failures: list[str] = []
    for field in REMOVAL_AUDIT_CHECK_FIELDS:
        raw_check = str(raw_removal_audit.get(field, "")).lower().strip()
        check = raw_check if raw_check in VALID_REMOVAL_CHECKS else "not_applicable"
        removal_audit[field] = check
        if check == "fail":
            removal_failures.append(field)
    if removal_expected and raw_removal_audit:
        if applicable != "yes":
            if checks["visible_artifact"] != "fail":
                checks["visible_artifact"] = "uncertain"
        else:
            for required_field in ("removed_region_quality", "large_blur_fill"):
                if removal_audit[required_field] == "not_applicable":
                    if checks["visible_artifact"] != "fail":
                        checks["visible_artifact"] = "uncertain"
            if removal_failures:
                checks["visible_artifact"] = "fail"
    if any(check == "fail" for check in checks.values()):
        decision = "rejected"
    elif all(check == "pass" for check in checks.values()):
        decision = "accepted"
    else:
        decision = "uncertain"

    raw_codes = value.get("reason_codes", [])
    if isinstance(raw_codes, str):
        raw_codes = [raw_codes]
    reason_codes = [
        re.sub(r"[^a-z0-9_]+", "_", str(code).lower()).strip("_")[:80]
        for code in raw_codes
        if str(code).strip()
    ][:8]
    for removal_failure in removal_failures:
        if removal_failure not in reason_codes and len(reason_codes) < 8:
            reason_codes.append(removal_failure)
    if reported != decision:
        reason_codes.append("inconsistent_overall_decision")

    raw_evidence = value.get("evidence", {})
    if not isinstance(raw_evidence, dict):
        raw_evidence = {}
    evidence = {
        field: str(raw_evidence.get(field, "")).strip()[:600] for field in CHECK_FIELDS
    }
    return {
        "reported_decision": reported,
        "decision": decision,
        **checks,
        "reason_codes": reason_codes,
        "removal_audit": removal_audit,
        "frame_audit": frame_audit,
        "temporal_audit": temporal_audit,
        "evidence": evidence,
    }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_api_key(environment_name: str, key_file: Path | None) -> str:
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    if key_file is None:
        return ""
    if not key_file.is_file():
        raise FileNotFoundError(f"API key file does not exist: {key_file}")
    mode = key_file.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(
            f"API key file must not be accessible by group/others (use chmod 600): {key_file}"
        )
    value = key_file.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"API key file is empty: {key_file}")
    return value


def serialize_selected_proposals(proposals: list[rect.RegionProposal]) -> str:
    values = [
        {
            "frame_index": proposal.index,
            "expanded_box_norm": [round(value, 6) for value in proposal.expanded_box],
            "reliability": round(proposal.reliability, 6),
            "change_score": round(proposal.change_score, 6),
        }
        for proposal in proposals
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def blank_result(row: dict[str, str], model: str, rubric_hash: str, backend: str) -> dict[str, str]:
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "judge_version": JUDGE_VERSION,
            "roi_method_version": ROI_METHOD_VERSION,
            "visual_layout_version": VISUAL_LAYOUT_VERSION,
            "model_id": model,
            "model_backend": backend,
            "prompt_sha256": rubric_hash,
            "sample_id": row["sample_id"],
            "dataset": row.get("dataset", ""),
            "branch": row.get("branch", ""),
            "operation": row.get("operation", ""),
            "target_scope": row.get("target_scope", ""),
            "generated_side": derive_generated_side(row),
            "training_operation": row.get("operation", ""),
            "training_instruction": row.get("instruction", ""),
            "original_stage1_1_decision": row.get("stage1_1_decision", ""),
        }
    )
    return result


def mock_response(row: dict[str, str]) -> dict[str, Any]:
    digest = hashlib.sha256(row["sample_id"].encode("utf-8")).digest()
    unit = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    if unit < 0.2:
        checks = {field: "fail" if field == "instruction_following" else "pass" for field in CHECK_FIELDS}
        decision = "rejected"
        reason = "mock_instruction_failure"
    elif unit < 0.3:
        checks = {field: "uncertain" if field == "visible_artifact" else "pass" for field in CHECK_FIELDS}
        decision = "uncertain"
        reason = "mock_insufficient_detail"
    else:
        checks = {field: "pass" for field in CHECK_FIELDS}
        decision = "accepted"
        reason = "mock_pass"
    return {
        "decision": decision,
        **checks,
        "reason_codes": [reason],
        "evidence": {field: "F1 mock evidence" for field in CHECK_FIELDS},
    }


def score_row(
    row: dict[str, str],
    *,
    model: str,
    rubric: str,
    api_key: str,
    api_url: str,
    frame_samples: int,
    full_pair_count: int,
    roi_pair_count: int,
    full_max_long_edge: int,
    roi_max_long_edge: int,
    analysis_long_edge: int,
    expand_factor: float,
    minimum_roi_reliability: float,
    jpeg_quality: int,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
    data_collection: str,
    reasoning_effort: str | None,
    exclude_reasoning: bool,
    temperature: float | None,
    provider_order: list[str] | None,
    mock: bool,
) -> dict[str, str]:
    started = time.monotonic()
    backend = "mock" if mock else "openrouter"
    result = blank_result(row, model, prompt_hash(rubric), backend)
    try:
        audit_view = build_audit_view(row)
        result.update(
            {
                "audit_operation": audit_view.audit_operation,
                "audit_relation": audit_view.audit_relation,
                "audit_routing_transform": audit_view.routing_transform,
                "audit_left_training_role": audit_view.left_training_role,
                "audit_right_training_role": audit_view.right_training_role,
                "audit_left_label": audit_view.left_label,
                "audit_right_label": audit_view.right_label,
                "audit_generated_position": audit_view.generated_position,
            }
        )
        pairs = rect.matched_frames(
            resolve_media(row[f"{audit_view.left_training_role}_local"]),
            resolve_media(row[f"{audit_view.right_training_role}_local"]),
            frame_samples,
        )
        cards, _proposals, selected = prepare_time_cards(
            pairs,
            full_pair_count,
            full_max_long_edge,
            roi_max_long_edge,
            analysis_long_edge,
            minimum_roi_reliability,
        )
        reliabilities = [proposal.reliability for proposal in selected]
        result.update(
            {
                "sampled_full_pairs": str(len(cards)),
                "sampled_roi_pairs": str(
                    sum(card.roi_left is not None for card in cards)
                ),
                "roi_boxes_json": serialize_selected_proposals(selected),
                "roi_reliability_min": f"{min(reliabilities, default=0.0):.6f}",
                "roi_reliability_mean": f"{mean(reliabilities) if reliabilities else 0.0:.6f}",
            }
        )
        content = build_timecard_content(
            cards,
            row.get("operation", ""),
            row.get("instruction", ""),
            audit_view,
            rubric,
            jpeg_quality,
        )
        if mock:
            value = mock_response(row)
            raw = json.dumps(value, ensure_ascii=False)
            api_elapsed = 0.0
            attempts = 0
        else:
            payload = build_payload(
                model,
                content,
                max_tokens,
                data_collection,
                reasoning_effort,
                exclude_reasoning,
                temperature,
                provider_order,
            )
            data, attempts, api_elapsed = post_json(
                api_url, api_key, payload, timeout, max_attempts
            )
            raw = extract_message(data)
            usage = data.get("usage") or {}
            result.update(
                {
                    "provider": str(data.get("provider", "")),
                    "generation_id": str(data.get("id", "")),
                    "model_id": str(data.get("model", model)),
                    "input_tokens": str(usage.get("prompt_tokens", "")),
                    "output_tokens": str(usage.get("completion_tokens", "")),
                    "total_tokens": str(usage.get("total_tokens", "")),
                    "reasoning_tokens": str(
                        (usage.get("completion_tokens_details") or {}).get(
                            "reasoning_tokens", ""
                        )
                    ),
                    "reasoning_effort": reasoning_effort or "default",
                }
            )
            reported_cost = usage.get("cost")
            if reported_cost is not None:
                result["cost_usd"] = f"{float(reported_cost):.9f}"
                result["cost_source"] = "openrouter_usage"
            elif result["input_tokens"] and result["output_tokens"]:
                # This runner now supports arbitrary OpenRouter models.  A
                # Qwen-specific hard-coded fallback would silently misstate
                # their cost, so retain authoritative token counts and mark
                # absent provider cost explicitly.
                result["cost_source"] = "openrouter_usage_cost_missing"
        result["api_attempts"] = str(attempts)
        result["api_elapsed_seconds"] = f"{api_elapsed:.6f}"
        result["raw_response"] = raw
        if mock:
            result.update(
                {
                    "input_tokens": "0",
                    "output_tokens": "0",
                    "total_tokens": "0",
                    "cost_usd": "0.000000000",
                    "cost_source": "mock",
                }
            )
        normalized = normalize_judgment(
            rect.extract_json_object(raw),
            removal_expected=audit_view.audit_operation == "remove",
        )
        result.update(
            {
                "model_reported_decision": normalized["reported_decision"],
                "model_decision": normalized["decision"],
                **{field: normalized[field] for field in CHECK_FIELDS},
                "reason_codes_json": json.dumps(
                    normalized["reason_codes"], ensure_ascii=False, separators=(",", ":")
                ),
                "removal_audit_json": json.dumps(
                    normalized["removal_audit"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "frame_audit_json": json.dumps(
                    normalized["frame_audit"], ensure_ascii=False, separators=(",", ":")
                ),
                "temporal_audit_json": json.dumps(
                    normalized["temporal_audit"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "evidence_json": json.dumps(
                    normalized["evidence"], ensure_ascii=False, separators=(",", ":")
                ),
                "parse_status": "pass",
            }
        )
    except Exception as error:
        result["model_decision"] = "uncertain"
        result["parse_status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"[:2000]
    result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
    return result


def append_result(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def write_ordered_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def existing_results(
    path: Path,
    resume: bool,
    overwrite: bool,
    *,
    expected_prompt_hash: str,
    expected_model: str,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    if overwrite:
        path.unlink()
        return {}
    if not resume:
        raise FileExistsError(f"output already exists; pass --resume or --overwrite: {path}")
    latest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("judge_version") not in {"", JUDGE_VERSION}:
            raise ValueError(f"resume output contains a different judge version: {path}")
        if row.get("prompt_sha256") not in {"", expected_prompt_hash}:
            raise ValueError(f"resume output contains a different prompt: {path}")
        if row.get("model_id") not in {"", expected_model}:
            raise ValueError(f"resume output contains a different model: {path}")
        if row.get("sample_id"):
            latest[row["sample_id"]] = row
    return latest


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, str]],
    model: str,
    rubric: str,
    input_rows: int,
    pending_rows: int,
    wall_seconds: float,
    concurrency: int,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("parse_status") == "pass"]
    summary: dict[str, Any] = {
        "judge_version": JUDGE_VERSION,
        "roi_method_version": ROI_METHOD_VERSION,
        "visual_layout_version": VISUAL_LAYOUT_VERSION,
        "model_id": model,
        "model_backend": "openrouter" if any(row.get("model_backend") == "openrouter" for row in rows) else "mock",
        "prompt_sha256": prompt_hash(rubric),
        "input_rows": input_rows,
        "completed": len(completed),
        "parse_errors": len(rows) - len(completed),
        "decision_counts": dict(Counter(row.get("model_decision", "") for row in rows)),
        "check_counts": {
            field: dict(Counter(row.get(field, "") for row in completed))
            for field in CHECK_FIELDS
        },
        "provider_counts": dict(Counter(row.get("provider", "") for row in completed)),
        "new_rows_attempted": pending_rows,
        "concurrency": concurrency,
        "new_run_wall_seconds": wall_seconds,
        "mean_pair_elapsed_seconds": mean(
            float(row.get("elapsed_seconds") or 0) for row in rows
        )
        if rows
        else 0,
        "mean_api_elapsed_seconds": mean(
            float(row.get("api_elapsed_seconds") or 0) for row in completed
        )
        if completed
        else 0,
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in completed),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in completed),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0) for row in completed),
        "mean_input_tokens": mean(int(row.get("input_tokens") or 0) for row in completed)
        if completed
        else 0,
        "mean_roi_pairs": mean(int(row.get("sampled_roi_pairs") or 0) for row in completed)
        if completed
        else 0,
        "rows_retried": sum(int(row.get("api_attempts") or 0) > 1 for row in completed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_QWEN")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Permission-600 fallback used only when --api-key-env is unset.",
    )
    parser.add_argument("--frame-samples", type=int, default=9)
    parser.add_argument("--full-pair-count", type=int, default=DEFAULT_CARD_COUNT)
    parser.add_argument(
        "--roi-pair-count",
        type=int,
        default=DEFAULT_CARD_COUNT,
        help="Must equal --full-pair-count; each card aligns one ROI with one full pair.",
    )
    parser.add_argument("--full-max-long-edge", type=int, default=640)
    parser.add_argument("--roi-max-long-edge", type=int, default=640)
    parser.add_argument("--analysis-long-edge", type=int, default=512)
    parser.add_argument("--expand-factor", type=float, default=2.0)
    parser.add_argument("--minimum-roi-reliability", type=float, default=0.12)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="OpenRouter normalized reasoning effort; omitted by default.",
    )
    parser.add_argument(
        "--exclude-reasoning",
        action="store_true",
        help="Ask OpenRouter not to return the model's private reasoning text.",
    )
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Omit temperature for models/providers that do not support it.",
    )
    parser.add_argument(
        "--provider-order",
        action="append",
        help=(
            "OpenRouter provider slug to try in order. Repeat for multiple providers; "
            "fallback outside this list remains disabled."
        ),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--data-collection", choices=("allow", "deny"), default="deny")
    parser.add_argument("--calibration-gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument(
        "--allow-uncalibrated-full",
        action="store_true",
        help="Explicitly bypass the gate for diagnostics; never produces formal decisions.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Restrict the run to one or more exact sample IDs (repeatable).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    if args.frame_samples < 2:
        parser.error("--frame-samples must be at least 2")
    if args.full_pair_count < 2 or args.full_pair_count > args.frame_samples:
        parser.error("--full-pair-count must be between 2 and --frame-samples")
    if args.full_pair_count not in {4, 5}:
        parser.error("the audited layouts support 4 or 5 aligned time cards")
    if args.roi_pair_count != args.full_pair_count:
        parser.error("--roi-pair-count must equal --full-pair-count")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    rubric = args.prompt.read_text(encoding="utf-8").strip()
    is_default_full_input = args.input.resolve() == DEFAULT_INPUT.resolve()
    if is_default_full_input and not args.mock_model and not args.allow_uncalibrated_full:
        if not args.calibration_gates.is_file():
            raise RuntimeError(
                f"full run is locked; calibration gates are missing: {args.calibration_gates}"
            )
        gates = json.loads(args.calibration_gates.read_text(encoding="utf-8"))
        if gates.get("calibration_status") != "enabled":
            raise RuntimeError(
                "full run is locked because Qwen3 calibration_status is not enabled"
            )
        expected_gate_identity = {
            "judge_version": JUDGE_VERSION,
            "prompt_sha256": prompt_hash(rubric),
            "model_id": args.model,
            "model_backend": "openrouter",
        }
        for field, expected in expected_gate_identity.items():
            if gates.get(field) != expected:
                raise RuntimeError(
                    f"full run is locked because calibration {field} does not match: "
                    f"expected {expected!r}, got {gates.get(field)!r}"
                )
    rows = read_csv(args.input)
    if args.sample_id:
        requested = set(args.sample_id)
        rows = [row for row in rows if row.get("sample_id") in requested]
        missing = sorted(requested - {row.get("sample_id", "") for row in rows})
        if missing:
            raise ValueError(f"requested sample IDs are missing from input: {missing}")
    if args.limit is not None:
        rows = rows[: args.limit]
    previous = existing_results(
        args.output,
        args.resume,
        args.overwrite,
        expected_prompt_hash=prompt_hash(rubric),
        expected_model=args.model,
    )
    succeeded = {
        sample_id: row for sample_id, row in previous.items() if row.get("parse_status") == "pass"
    }
    pending = [row for row in rows if row["sample_id"] not in succeeded]

    api_key = "" if args.mock_model else load_api_key(args.api_key_env, args.api_key_file)
    if not args.mock_model and not api_key:
        raise RuntimeError(
            f"API key is missing: set {args.api_key_env} or pass --api-key-file"
        )

    common = {
        "model": args.model,
        "rubric": rubric,
        "api_key": api_key,
        "api_url": args.api_url,
        "frame_samples": args.frame_samples,
        "full_pair_count": args.full_pair_count,
        "roi_pair_count": args.roi_pair_count,
        "full_max_long_edge": args.full_max_long_edge,
        "roi_max_long_edge": args.roi_max_long_edge,
        "analysis_long_edge": args.analysis_long_edge,
        "expand_factor": args.expand_factor,
        "minimum_roi_reliability": args.minimum_roi_reliability,
        "jpeg_quality": args.jpeg_quality,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "data_collection": args.data_collection,
        "reasoning_effort": args.reasoning_effort,
        "exclude_reasoning": args.exclude_reasoning,
        "temperature": None if args.no_temperature else 0.0,
        "provider_order": args.provider_order,
        "mock": args.mock_model,
    }
    latest = dict(previous)
    run_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_rows: dict[Future[dict[str, str]], dict[str, str]] = {
            executor.submit(score_row, row, **common): row for row in pending
        }
        for completed_index, future in enumerate(as_completed(future_rows), 1):
            result = future.result()
            append_result(args.output, result)
            latest[result["sample_id"]] = result
            print(
                f"qwen3_judged={completed_index}/{len(pending)} "
                f"sample_id={result['sample_id']} status={result['parse_status']} "
                f"decision={result['model_decision']} cost_usd={result['cost_usd'] or '0'}",
                flush=True,
            )

    ordered = [latest[row["sample_id"]] for row in rows if row["sample_id"] in latest]
    write_ordered_csv(args.output, ordered)
    summary = write_summary(
        args.summary,
        rows=ordered,
        model=args.model,
        rubric=rubric,
        input_rows=len(rows),
        pending_rows=len(pending),
        wall_seconds=time.monotonic() - run_started,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
