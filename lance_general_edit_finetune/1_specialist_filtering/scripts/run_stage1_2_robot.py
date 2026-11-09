#!/usr/bin/env python3
"""Score ReCo robot/CG pairs with cheap specialist models.

The script intentionally writes scores only. Hard decisions are learned from the
compact human calibration set by ``calibrate_stage1_2.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/filter_run_v2"
VENDOR_LIB = ROOT / "artifacts/vendor/build/lib"
SCORE_VERSION = "stage1_2_robot_scores_v1"
ROBOT_CLASSES = ["robot", "humanoid robot", "android", "cyborg", "cartoon character", "3D character"]
POSITIVE_PROMPTS = [
    "a detailed realistic robot or character integrated naturally into the scene",
    "a high quality coherent character with complete geometry and natural lighting",
]
NEGATIVE_PROMPTS = [
    "a crude low-poly 3D model or cheap video game character",
    "a flat sticker poorly pasted into the scene",
    "a broken distorted synthetic character with incomplete geometry",
]
FIELDS = [
    "score_version",
    "model_backend",
    "sample_id",
    "dataset",
    "branch",
    "operation",
    "target_scope",
    "source_provenance",
    "target_provenance",
    "scored_role",
    "sampled_frames",
    "source_detection_support",
    "target_detection_support",
    "source_detection_confidence_mean",
    "target_detection_confidence_mean",
    "source_detection_area_mean",
    "target_detection_area_mean",
    "crop_musiq_p10",
    "crop_musiq_mean",
    "siglip_positive_mean",
    "siglip_negative_mean",
    "siglip_quality_margin",
    "nonedit_dino_cosine_p10",
    "nonedit_dino_cosine_mean",
    "fallback_full_frame_ratio",
    "technical_status",
    "error",
    "elapsed_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), level))


def uniform_frames(path: Path, count: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) < 1:
        raise ValueError(f"empty video: {path}")
    indices = np.unique(np.linspace(0, len(reader) - 1, min(count, len(reader))).round().astype(int))
    return reader.get_batch(indices).asnumpy()


def edited_role(row: dict[str, str]) -> str:
    source = row.get("source_provenance")
    target = row.get("target_provenance")
    if source == "edited" and target != "edited":
        return "source"
    if target == "edited" and source != "edited":
        return "target"
    return "both"


def largest_crop(frame: np.ndarray, boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    if not boxes:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2, _ = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    pad_x = 0.08 * (x2 - x1)
    pad_y = 0.08 * (y2 - y1)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    return frame[top:bottom, left:right] if right > left and bottom > top else frame


def detection_summary(
    detections: list[list[tuple[float, float, float, float, float]]], frames: np.ndarray
) -> tuple[float, float, float]:
    present = [boxes for boxes in detections if boxes]
    confidence = [max(box[-1] for box in boxes) for boxes in present]
    areas = []
    for boxes, frame in zip(detections, frames):
        if boxes:
            box = max(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
            areas.append((box[2] - box[0]) * (box[3] - box[1]) / (frame.shape[0] * frame.shape[1]))
    return (
        len(present) / max(1, len(detections)),
        float(np.mean(confidence)) if confidence else 0.0,
        float(np.mean(areas)) if areas else 0.0,
    )


class RobotModels:
    def __init__(
        self,
        device: str,
        yolo_model: str,
        siglip_model: str,
        dino_model: str,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoProcessor, Dinov2Model, SiglipModel

        try:
            from ultralytics import YOLOWorld
        except ImportError as error:
            raise RuntimeError("ultralytics is required for Stage 1.2") from error
        if str(VENDOR_LIB) not in sys.path:
            sys.path.insert(0, str(VENDOR_LIB))
        import pyiqa

        self.torch = torch
        self.device = torch.device(device)
        self.yolo = YOLOWorld(yolo_model)
        self.yolo.set_classes(ROBOT_CLASSES)
        self.musiq = pyiqa.create_metric("musiq", device=self.device)
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model)
        # Use the concrete class: AutoModel enumerates unrelated timm wrappers and
        # trips over the old timm version pinned by the Lance environment.
        self.siglip = SiglipModel.from_pretrained(siglip_model).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model, use_fast=False)
        self.dino = Dinov2Model.from_pretrained(dino_model).to(self.device).eval()
        with torch.inference_mode():
            prompt_inputs = self.siglip_processor(
                text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS, padding="max_length", return_tensors="pt"
            )
            prompt_inputs = {key: value.to(self.device) for key, value in prompt_inputs.items()}
            features = self.siglip.get_text_features(**prompt_inputs)
            self.text_features = torch.nn.functional.normalize(features, dim=-1)

    def detections(self, frames: np.ndarray) -> list[list[tuple[float, float, float, float, float]]]:
        results = self.yolo.predict(list(frames), verbose=False, conf=0.10, imgsz=640, device=str(self.device))
        output: list[list[tuple[float, float, float, float, float]]] = []
        for result in results:
            boxes: list[tuple[float, float, float, float, float]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
                boxes = [tuple(map(float, [*coords, confidence])) for coords, confidence in zip(xyxy, confidences)]
            output.append(boxes)
        return output

    def crop_quality(self, crops: list[np.ndarray]) -> dict[str, float]:
        import cv2

        torch = self.torch
        resized = np.stack([cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA) for crop in crops])
        with torch.inference_mode():
            tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
            musiq = self.musiq(tensor).reshape(-1).detach().cpu().tolist()
            inputs = self.siglip_processor(images=list(resized), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            image_features = self.siglip.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            similarities = image_features @ self.text_features.T
            positive = similarities[:, : len(POSITIVE_PROMPTS)].mean(dim=1)
            negative = similarities[:, len(POSITIVE_PROMPTS) :].mean(dim=1)
        positive_values = positive.detach().cpu().tolist()
        negative_values = negative.detach().cpu().tolist()
        return {
            "crop_musiq_p10": percentile([float(value) for value in musiq], 10),
            "crop_musiq_mean": float(np.mean(musiq)),
            "siglip_positive_mean": float(np.mean(positive_values)),
            "siglip_negative_mean": float(np.mean(negative_values)),
            "siglip_quality_margin": float((positive - negative).mean().item()),
        }

    def dino_nonedit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_boxes: list[list[tuple[float, float, float, float, float]]],
        target_boxes: list[list[tuple[float, float, float, float, float]]],
    ) -> dict[str, float]:
        torch = self.torch
        similarities: list[float] = []
        with torch.inference_mode():
            for source_frame, target_frame, left_boxes, right_boxes in zip(
                source, target, source_boxes, target_boxes
            ):
                inputs = self.dino_processor(images=[source_frame, target_frame], return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                tokens = self.dino(**inputs).last_hidden_state[:, 1:]
                tokens = torch.nn.functional.normalize(tokens, dim=-1)
                patch_count = tokens.shape[1]
                side = int(round(math.sqrt(patch_count)))
                keep = torch.ones(patch_count, dtype=torch.bool, device=self.device)
                height, width = source_frame.shape[:2]
                for box in left_boxes + right_boxes:
                    x1, y1, x2, y2, _ = box
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                    for index in range(patch_count):
                        y, x = divmod(index, side)
                        center_x, center_y = (x + 0.5) / side, (y + 0.5) / side
                        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                            keep[index] = False
                values = (tokens[0] * tokens[1]).sum(dim=-1)
                if int(keep.sum()) >= max(8, patch_count // 5):
                    values = values[keep]
                similarities.extend(float(value) for value in values.detach().cpu().tolist())
        return {
            "nonedit_dino_cosine_p10": percentile(similarities, 10),
            "nonedit_dino_cosine_mean": float(np.mean(similarities)),
        }


def mock_score(row: dict[str, str], frame_samples: int) -> dict[str, str]:
    digest = hashlib.sha256(row["sample_id"].encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    return {
        "score_version": SCORE_VERSION,
        "model_backend": "mock",
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "branch": row["branch"],
        "operation": row["operation"],
        "target_scope": row.get("target_scope", ""),
        "source_provenance": row.get("source_provenance", ""),
        "target_provenance": row.get("target_provenance", ""),
        "scored_role": edited_role(row),
        "sampled_frames": str(frame_samples),
        "source_detection_support": f"{unit:.8f}",
        "target_detection_support": f"{1-unit:.8f}",
        "source_detection_confidence_mean": f"{unit:.8f}",
        "target_detection_confidence_mean": f"{1-unit:.8f}",
        "source_detection_area_mean": "0.10000000",
        "target_detection_area_mean": "0.10000000",
        "crop_musiq_p10": f"{30+40*unit:.8f}",
        "crop_musiq_mean": f"{35+40*unit:.8f}",
        "siglip_positive_mean": f"{unit:.8f}",
        "siglip_negative_mean": f"{1-unit:.8f}",
        "siglip_quality_margin": f"{2*unit-1:.8f}",
        "nonedit_dino_cosine_p10": f"{0.5+0.5*unit:.8f}",
        "nonedit_dino_cosine_mean": f"{0.6+0.4*unit:.8f}",
        "fallback_full_frame_ratio": "0.00000000",
        "technical_status": "pass",
        "error": "",
        "elapsed_seconds": "0.000000",
    }


def score_pair(
    row: dict[str, str], models: RobotModels, frame_samples: int
) -> dict[str, str]:
    started = time.monotonic()
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "score_version": SCORE_VERSION,
            "model_backend": "yoloworld+musiq+siglip+dinov2",
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "branch": row["branch"],
            "operation": row["operation"],
            "target_scope": row.get("target_scope", ""),
            "source_provenance": row.get("source_provenance", ""),
            "target_provenance": row.get("target_provenance", ""),
            "scored_role": edited_role(row),
        }
    )
    try:
        source = uniform_frames(resolve_media(row["source_local"]), frame_samples)
        target = uniform_frames(resolve_media(row["target_local"]), frame_samples)
        count = min(len(source), len(target))
        source, target = source[:count], target[:count]
        source_boxes = models.detections(source)
        target_boxes = models.detections(target)
        source_summary = detection_summary(source_boxes, source)
        target_summary = detection_summary(target_boxes, target)
        role = edited_role(row)
        selected_frames: list[np.ndarray] = []
        selected_boxes: list[list[tuple[float, float, float, float, float]]] = []
        if role in {"source", "both"}:
            selected_frames.extend(source)
            selected_boxes.extend(source_boxes)
        if role in {"target", "both"}:
            selected_frames.extend(target)
            selected_boxes.extend(target_boxes)
        crops = [largest_crop(frame, boxes) for frame, boxes in zip(selected_frames, selected_boxes)]
        fallback_ratio = sum(not boxes for boxes in selected_boxes) / max(1, len(selected_boxes))
        metrics = models.crop_quality(crops)
        metrics.update(models.dino_nonedit(source, target, source_boxes, target_boxes))
        result.update(
            {
                "sampled_frames": str(count),
                "source_detection_support": f"{source_summary[0]:.8f}",
                "target_detection_support": f"{target_summary[0]:.8f}",
                "source_detection_confidence_mean": f"{source_summary[1]:.8f}",
                "target_detection_confidence_mean": f"{target_summary[1]:.8f}",
                "source_detection_area_mean": f"{source_summary[2]:.8f}",
                "target_detection_area_mean": f"{target_summary[2]:.8f}",
                "fallback_full_frame_ratio": f"{fallback_ratio:.8f}",
                "technical_status": "pass",
                **{key: f"{value:.8f}" for key, value in metrics.items()},
            }
        )
    except Exception as error:
        result["technical_status"] = "error"
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
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "stage1_2_robot_queue.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "stage1_2_robot_scores.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN / "stage1_2_robot_score_summary.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-model", default="yolov8s-worldv2.pt")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--frame-samples", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock-models", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing: dict[str, dict[str, str]] = {}
    if args.resume and args.output.is_file():
        existing = {
            row["sample_id"]: row
            for row in read_csv(args.output)
            if row.get("technical_status") in {"pass", "error"}
        }
    pending = [row for row in rows if row["sample_id"] not in existing]
    models = None if args.mock_models else RobotModels(
        args.device, args.yolo_model, args.siglip_model, args.dino_model
    )
    results = list(existing.values())
    for index, row in enumerate(pending, 1):
        result = mock_score(row, args.frame_samples) if args.mock_models else score_pair(
            row, models, args.frame_samples  # type: ignore[arg-type]
        )
        results.append(result)
        write_csv(args.output, results)
        print(f"stage1_2_scored={index}/{len(pending)} total={len(results)}", flush=True)

    summary = {
        "score_version": SCORE_VERSION,
        "model_backend": "mock" if args.mock_models else "yoloworld+musiq+siglip+dinov2",
        "input_rows": len(rows),
        "completed": len(results),
        "passed": sum(row.get("technical_status") == "pass" for row in results),
        "errors": sum(row.get("technical_status") == "error" for row in results),
        "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
