#!/usr/bin/env python3
"""Create exact 81-frame VACE conditioning videos and auditable edit masks.

Local-object masks use Grounding DINO on frames 0/40/80, SAM2 image
segmentation at those keyframes, and SAM2 video propagation.  Full-scene edits
use a full white mask.  Object insertion and the explicitly requested wall
region use documented semantic ROIs because the edited content does not exist
in the source video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image


GLOBAL_CATEGORIES = {
    "Environment_change", "Global_style_change", "Global_visual_effects_adding",
    "Landscape_change", "Relighting", "Season_change", "Time_change",
    "Weather_change",
}

# target text, maximum detected instances, normalized fallback/semantic ROI,
# and whether the ROI must be unioned with the tracked object mask.
SPECS = {
    "Sp_Appearance_change_09": ("beard . face", 1, (0.25, 0.03, 0.70, 0.78), False),
    "Sp_Appearance_change_05": ("face", 1, (0.20, 0.02, 0.80, 0.65), False),
    "Sp_Clothes_change_10": ("coat . turtleneck", 1, (0.18, 0.08, 0.82, 0.98), False),
    "Sp_Clothes_change_09": ("outfit . jacket", 1, (0.18, 0.08, 0.82, 0.98), False),
    "Sp_Color_change_02": ("yellow flower", 1, (0.05, 0.05, 0.95, 0.95), False),
    "Sp_Color_change_10": ("rabbit", 1, (0.20, 0.05, 0.95, 0.95), False),
    "Sp_Material_change_02": ("bicycle", 4, (0.00, 0.42, 1.00, 0.98), False),
    "Sp_Material_change_05": ("glass cup", 1, (0.12, 0.02, 0.85, 0.98), False),
    "Sp_Object_State_change_10": ("plant", 4, (0.32, 0.18, 0.95, 0.98), False),
    "Sp_Object_State_change_07": ("candles", 12, (0.08, 0.22, 0.92, 0.88), False),
    "Sp_Object_adding_02": ("manual insertion ROI: tabletop", 0, (0.48, 0.48, 0.86, 0.86), False),
    "Sp_Object_adding_08": ("manual insertion ROI: table beside vase", 0, (0.02, 0.48, 0.62, 0.94), False),
    "Sp_Object_direction_change_04": ("man", 1, (0.28, 0.05, 0.68, 0.98), False),
    "Sp_Object_direction_change_09": ("gray sofa", 1, (0.00, 0.22, 0.92, 0.98), False),
    "Sp_Object_location_change_07": ("ostrich", 1, (0.20, 0.15, 0.80, 0.98), True),
    "Sp_Object_location_change_03": ("Eiffel Tower", 1, (0.34, 0.02, 0.66, 0.98), True),
    "Sp_Object_location_swap_08": ("elephants", 2, (0.03, 0.05, 0.97, 0.90), False),
    "Sp_Object_location_swap_09": ("woman . man", 2, (0.10, 0.02, 0.90, 0.98), False),
    "Sp_Object_number_change_06": ("camels", 4, (0.02, 0.10, 0.98, 0.96), True),
    "Sp_Object_number_change_04": ("sausages", 12, (0.10, 0.18, 0.92, 0.92), True),
    "Sp_Object_removal_06": ("plants and dried flowers", 4, (0.00, 0.02, 0.56, 0.98), False),
    "Sp_Object_removal_05": ("tractor and farming equipment", 3, (0.28, 0.18, 0.90, 0.92), False),
    "Sp_Object_replacement_02": ("white swan", 1, (0.08, 0.18, 0.88, 0.92), False),
    "Sp_Object_replacement_09": ("lamp", 1, (0.20, 0.02, 0.80, 0.82), False),
    "Sp_Object_style_change_03": ("wolf", 1, (0.00, 0.05, 0.86, 0.98), False),
    "Sp_Object_style_change_05": ("man", 1, (0.12, 0.02, 0.82, 0.98), False),
    "Sp_Object_visual_effects_adding_04": ("woman's hair", 1, (0.34, 0.01, 0.66, 0.48), False),
    "Sp_Object_visual_effects_adding_09": ("chimneys", 2, (0.36, 0.00, 0.74, 0.60), False),
    "Sp_Size_shape_change_02": ("ball", 1, (0.05, 0.38, 0.78, 0.94), False),
    "Sp_Size_shape_change_06": ("antelope horns", 1, (0.31, 0.00, 0.69, 0.58), False),
    "Sp_Texture_change_09": ("turtle shell", 1, (0.06, 0.06, 0.94, 0.94), False),
    "Sp_Texture_change_08": ("manual semantic ROI: right half of brick wall", 0, (0.48, 0.00, 1.00, 1.00), False),
}

ROI_ONLY = {"Sp_Object_adding_02", "Sp_Object_adding_08", "Sp_Texture_change_08"}
DESTINATION_ROIS = {
    "Sp_Object_location_change_07": (0.67, 0.08, 0.98, 0.55),
    "Sp_Object_location_change_03": (0.20, 0.00, 0.80, 1.00),
    "Sp_Object_number_change_06": (0.00, 0.22, 1.00, 0.98),
    "Sp_Object_number_change_04": (0.10, 0.18, 0.95, 0.92),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def output_shape(h: int, w: int, frame_count: int) -> tuple[int, int, int]:
    # VACE 480p processor: vae_stride*patch_size = (4,16,16), seq_len=32760.
    df, dh, dw, seq_len, area = 4, 16, 16, 32760, 480 * 832
    area_z = min(seq_len, area / (dh * dw), (h // dh) * (w // dw))
    latent_frames = min((frame_count - 1) // df + 1, int(seq_len / area_z))
    target_area_z = min(area_z, int(seq_len / latent_frames))
    oh = round(math.sqrt(target_area_z * h / w)) * dh
    ow = int(target_area_z / (oh / dh)) * dw
    frames = (latent_frames - 1) * df + 1
    return int(oh), int(ow), int(frames)


def resize_crop(frame: np.ndarray, oh: int, ow: int, nearest: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    if (h, w) == (oh, ow):
        return frame
    scale = max(ow / w, oh / h)
    nw, nh = round(scale * w), round(scale * h)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    frame = cv2.resize(frame, (nw, nh), interpolation=interp)
    x, y = (nw - ow) // 2, (nh - oh) // 2
    return frame[y:y + oh, x:x + ow]


def read_conditioning_frames(source: Path) -> tuple[list[np.ndarray], dict[str, object]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    oh, ow, count = output_shape(h, w, n)
    if count < 81:
        raise RuntimeError(f"{source} yields only {count} VACE frames; expected 81")
    ids = np.linspace(0, n - 1, count).round().astype(int).tolist()
    frames = []
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {idx} from {source}")
        frames.append(resize_crop(frame, oh, ow))
    cap.release()
    return frames, {"original_frames": n, "original_fps": fps, "sampled_frame_ids": ids,
                    "width": ow, "height": oh, "frames": count, "fps": 16.0}


def write_video(path: Path, frames: list[np.ndarray], fps: int = 16, crf: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    h, w = frames[0].shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)
    ], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    os.replace(tmp, path)


def roi_mask(h: int, w: int, box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = np.zeros((h, w), np.uint8)
    out[max(0, round(y1 * h)):min(h, round(y2 * h)),
        max(0, round(x1 * w)):min(w, round(x2 * w))] = 255
    return out


class MaskModels:
    def __init__(self, gdino_path: Path | None, sam2_checkpoint: Path,
                 manual_boxes: bool = False, gdino_config: Path | None = None,
                 gdino_checkpoint: Path | None = None):
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.processor = None
        self.detector = None
        self.local_detector = None
        if not manual_boxes:
            if gdino_config is not None and gdino_checkpoint is not None:
                from groundingdino.util.inference import Model as GroundingDinoModel
                self.local_detector = GroundingDinoModel(
                    str(gdino_config), str(gdino_checkpoint), device="cuda")
            else:
                if gdino_path is None:
                    raise ValueError("provide --gdino or both local GroundingDINO paths")
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(str(gdino_path), local_files_only=True)
                self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                    str(gdino_path), local_files_only=True,
                    torch_dtype=torch.float32).to("cuda").eval()
        image_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                                 str(sam2_checkpoint), device="cuda")
        self.image_predictor = SAM2ImagePredictor(image_model)
        self.video_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", str(sam2_checkpoint), device="cuda")

    @torch.inference_mode()
    def detect(self, frame: np.ndarray, phrase: str, max_objects: int) -> tuple[np.ndarray, list[float]]:
        if self.local_detector is not None:
            detections, _ = self.local_detector.predict_with_caption(
                image=frame, caption=phrase, box_threshold=0.22, text_threshold=0.18)
            boxes = np.asarray(detections.xyxy, dtype=np.float32)
            scores = np.asarray(detections.confidence, dtype=np.float32)
            order = np.argsort(-scores)[:max_objects]
            return boxes[order], scores[order].tolist()
        if self.detector is None or self.processor is None:
            return np.empty((0, 4), dtype=np.float32), []
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=phrase, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.detector(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=0.22, text_threshold=0.18,
            target_sizes=[image.size[::-1]])[0]
        boxes = results["boxes"].float().cpu().numpy()
        scores = results["scores"].float().cpu().numpy()
        order = np.argsort(-scores)[:max_objects]
        return boxes[order], scores[order].tolist()

    @torch.inference_mode()
    def segment_keyframe(self, frame: np.ndarray, boxes: np.ndarray,
                         fallback: tuple[float, float, float, float]) -> tuple[np.ndarray, str]:
        h, w = frame.shape[:2]
        method = "grounding_dino+sam2"
        if len(boxes) == 0:
            x1, y1, x2, y2 = fallback
            boxes = np.array([[x1 * w, y1 * h, x2 * w, y2 * h]], np.float32)
            method = "manual_fallback_box+sam2"
        self.image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, scores, _ = self.image_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return np.any(masks, axis=0).astype(np.uint8), method

    @torch.inference_mode()
    def track(self, frames: list[np.ndarray], key_masks: dict[int, np.ndarray]) -> list[np.ndarray]:
        # SAM2's video predictor accepts a JPEG-frame directory, not an MP4.
        with tempfile.TemporaryDirectory(prefix="vace_sam2_") as tmp:
            frame_dir = Path(tmp)
            for idx, frame in enumerate(frames):
                if not cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to stage SAM2 frame {idx}")
            state = self.video_predictor.init_state(str(frame_dir), offload_video_to_cpu=True)
            for idx, mask in sorted(key_masks.items()):
                self.video_predictor.add_new_mask(state, idx, 1, mask.astype(bool))
            result: dict[int, np.ndarray] = {}
            for idx, _, logits in self.video_predictor.propagate_in_video(state):
                result[int(idx)] = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
            self.video_predictor.reset_state(state)
        if set(result) != set(range(len(frames))):
            missing = sorted(set(range(len(frames))) - set(result))
            raise RuntimeError(f"SAM2 propagation omitted frames: {missing[:8]}")
        return [result[i] for i in range(len(frames))]


def process_row(row: dict[str, str], out_root: Path, models: MaskModels | None) -> dict[str, object]:
    started = time.time()
    eval_id = row["eval_id"]
    sample_dir = out_root / eval_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    source = Path(row["official_source_video"])
    frames, probe = read_conditioning_frames(source)
    proxy = sample_dir / "source_conditioning.mp4"
    write_video(proxy, frames)
    h, w = frames[0].shape[:2]
    target_phrase = "full frame"
    detections: dict[str, object] = {}

    if row["category"] in GLOBAL_CATEGORIES:
        masks = [np.full((h, w), 255, np.uint8) for _ in frames]
        method = "full_frame_global_edit"
    else:
        target_phrase, max_objects, fallback, union_roi = SPECS[eval_id]
        if eval_id in ROI_ONLY:
            masks = [roi_mask(h, w, fallback) for _ in frames]
            method = "manual_semantic_roi"
        else:
            if models is None:
                raise RuntimeError("models required for local masks")
            key_masks: dict[int, np.ndarray] = {}
            methods = []
            for idx in (0, len(frames) // 2, len(frames) - 1):
                boxes, scores = models.detect(frames[idx], target_phrase, max_objects)
                key_masks[idx], key_method = models.segment_keyframe(frames[idx], boxes, fallback)
                methods.append(key_method)
                detections[str(idx)] = {"boxes_xyxy": boxes.round(3).tolist(), "scores": scores,
                                        "method": key_method}
            masks = [m * 255 for m in models.track(frames, key_masks)]
            method = "+".join(sorted(set(methods))) + "+sam2_video_propagation"
            if union_roi:
                extra = roi_mask(h, w, DESTINATION_ROIS[eval_id])
                masks = [np.maximum(m, extra) for m in masks]
                method += "+destination_roi_union"

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    masks = [cv2.dilate(np.where(m > 127, 255, 0).astype(np.uint8), kernel) for m in masks]
    frames_dir = sample_dir / "mask_frames"
    frames_dir.mkdir(exist_ok=True)
    overlays = []
    areas = []
    temporal_ious = []
    previous_binary = None
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        cv2.imwrite(str(frames_dir / f"{i:05d}.png"), mask)
        binary = mask > 0
        areas.append(float(binary.mean()))
        if previous_binary is not None:
            union = np.logical_or(previous_binary, binary).sum()
            temporal_ious.append(float(np.logical_and(previous_binary, binary).sum() / max(1, union)))
        previous_binary = binary
        overlay = frame.copy()
        red = np.zeros_like(frame); red[:, :, 2] = 255
        alpha = (mask.astype(np.float32) / 255.0 * 0.48)[:, :, None]
        overlay = np.clip(overlay * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)
        overlays.append(overlay)
    mask_video = sample_dir / "mask.mp4"
    overlay_video = sample_dir / "mask_overlay.mp4"
    write_video(mask_video, [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks], crf=0)
    write_video(overlay_video, overlays, crf=15)
    record = {
        "eval_id": eval_id, "tier": row["tier"], "category": row["category"],
        "editing_instruction": row["editing_instruction"], "official_source_video": str(source),
        "official_source_sha256": row["official_source_sha256"],
        "conditioning_source": str(proxy), "conditioning_source_sha256": sha256(proxy),
        "mask_video": str(mask_video), "mask_sha256": sha256(mask_video),
        "mask_frames_dir": str(frames_dir), "overlay_video": str(overlay_video),
        "mask_method": method, "target_phrase_or_roi": target_phrase,
        "detections": detections, "mask_area_min": min(areas), "mask_area_mean": sum(areas) / len(areas),
        "mask_area_max": max(areas),
        "temporal_iou_min": min(temporal_ious) if temporal_ious else 1.0,
        "temporal_iou_mean": sum(temporal_ious) / len(temporal_ious) if temporal_ious else 1.0,
        "mask_qc": {
            "non_empty": min(areas) > 0,
            "local_area_in_range": (row["category"] in GLOBAL_CATEGORIES) or (0.002 <= min(areas) and max(areas) <= 0.75),
            "frame_count_ok": len(masks) == 81,
        },
        "probe": probe, "dilation_pixels": 16,
        "binary_truth_values": [0, 255], "elapsed_s": round(time.time() - started, 3), "status": "complete",
    }
    write_json(sample_dir / "mask_manifest.json", record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gdino", type=Path)
    ap.add_argument("--gdino-config", type=Path)
    ap.add_argument("--gdino-checkpoint", type=Path)
    ap.add_argument("--sam2-checkpoint", type=Path, required=True)
    ap.add_argument("--manual-boxes", action="store_true",
                    help="Use the curated fallback boxes as SAM2 prompts; skip Grounding DINO")
    ap.add_argument("--tier", choices=["A", "B", "all"], default="A")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if args.tier == "all" or r["tier"] == args.tier]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    local_needed = any(r["category"] not in GLOBAL_CATEGORIES and r["eval_id"] not in ROI_ONLY for r in rows)
    models = MaskModels(
        args.gdino, args.sam2_checkpoint, args.manual_boxes,
        args.gdino_config, args.gdino_checkpoint,
    ) if local_needed else None
    records = []
    for pos, row in enumerate(rows, 1):
        existing = args.output_root / row["eval_id"] / "mask_manifest.json"
        if existing.exists():
            record = json.loads(existing.read_text())
            if all(Path(record[k]).exists() for k in ("conditioning_source", "mask_video", "overlay_video")):
                records.append(record)
                print(f"[{pos}/{len(rows)}] reuse {row['eval_id']}", flush=True)
                continue
        print(f"[{pos}/{len(rows)}] build {row['eval_id']}", flush=True)
        records.append(process_row(row, args.output_root, models))
        with (args.output_root / f"mask_inventory_{args.tier}.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "tier": args.tier, "count": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
