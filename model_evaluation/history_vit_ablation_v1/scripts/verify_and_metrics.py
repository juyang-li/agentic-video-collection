#!/usr/bin/env python3
"""Audit all inference receipts and compare four ablations with same-run normal controls."""

from __future__ import annotations

import csv
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIFT = ROOT.parent / "real_person_drift_stress_v1"
MODELS = ("lance_base", "gtfresh_step1150")
CONDITIONS = (
    "vit_zero_text_keep",
    "vit_gaussian_text_keep",
    "vit_zero_text_zero",
    "vit_gaussian_text_zero",
)
ALL_CONDITIONS = ("normal_replay",) + CONDITIONS


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_checkpoints() -> list[dict]:
    with (DRIFT / "checkpoints.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 24:
        raise RuntimeError("expected 24 checkpoints")
    return rows


def video_path(model: str, condition: str, row: dict) -> Path:
    return ROOT / "outputs" / model / condition / row["drift_id"] / f"turn{int(row['round']) - 1:02d}.mp4"


def receipt_path(model: str, condition: str, row: dict) -> Path:
    return video_path(model, condition, row).with_suffix(".json")


def audit(checkpoints: list[dict]) -> dict:
    benchmark_hash = sha256(DRIFT / "benchmark.jsonl")
    videos = receipts = hook_receipts = 0
    for model in MODELS:
        for condition in ALL_CONDITIONS:
            manifest = json.loads((ROOT / "outputs" / model / condition / "manifest.json").read_text())
            if (manifest.get("status"), manifest.get("checkpoint_count"), manifest.get("benchmark_sha256")) != (
                    "generated", 24, benchmark_hash):
                raise RuntimeError(f"bad manifest: {model}/{condition}")
            if model == "gtfresh_step1150":
                check = manifest.get("adapter_check") or {}
                if (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B")) != (1016, 1016, 503):
                    raise RuntimeError(f"bad gtfresh adapter check: {condition}")
            for row in checkpoints:
                video = video_path(model, condition, row)
                receipt = receipt_path(model, condition, row)
                if not video.is_file() or video.stat().st_size <= 1000 or not receipt.is_file():
                    raise RuntimeError(f"missing output or receipt: {video}")
                data = json.loads(receipt.read_text())
                if data.get("status") != "generated" or data.get("output_sha256") != sha256(video):
                    raise RuntimeError(f"receipt mismatch: {receipt}")
                trace = data.get("trace") or {}
                if condition != "normal_replay":
                    if trace.get("vit_hook_calls") != 1 or trace.get("direct_source_unchanged") is not True:
                        raise RuntimeError(f"ViT hook audit failed: {receipt}")
                    if int(trace.get("history_vit_tokens", 0)) <= 0:
                        raise RuntimeError(f"empty history ViT span: {receipt}")
                    if condition.endswith("text_zero"):
                        if trace.get("text_hook_calls") != 1 or int(trace.get("history_text_tokens", 0)) <= 0:
                            raise RuntimeError(f"text hook audit failed: {receipt}")
                    hook_receipts += 1
                videos += 1; receipts += 1
    if (videos, receipts, hook_receipts) != (240, 240, 192):
        raise RuntimeError("coverage mismatch")
    return {"videos": videos, "receipts": receipts, "formal_ablation_hook_receipts": hook_receipts,
            "models": list(MODELS), "conditions": list(ALL_CONDITIONS), "checkpoints": 24}


def ssim_bgr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = []
    for channel in range(3):
        x, y = a[:, :, channel], b[:, :, channel]
        ux = cv2.GaussianBlur(x, (11, 11), 1.5)
        uy = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - ux * ux
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - uy * uy
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - ux * uy
        scores.append(float(np.mean(((2 * ux * uy + c1) * (2 * vxy + c2)) /
                                    ((ux * ux + uy * uy + c1) * (vx + vy + c2)))))
    return sum(scores) / 3


def compare_task(task: tuple[str, dict]) -> list[dict]:
    model, row = task
    normal_path = video_path(model, "normal_replay", row)
    normal = cv2.VideoCapture(str(normal_path))
    variants = {condition: cv2.VideoCapture(str(video_path(model, condition, row))) for condition in CONDITIONS}
    accum = {condition: {"mae": [], "ssim": [], "changed": []} for condition in CONDITIONS}
    frames = 0
    while True:
        ok, base = normal.read()
        reads = {condition: capture.read() for condition, capture in variants.items()}
        if not ok:
            if any(flag for flag, _ in reads.values()):
                raise RuntimeError(f"frame count mismatch: {model}/{row['drift_id']}/r{row['round']}")
            break
        if not all(flag for flag, _ in reads.values()):
            raise RuntimeError(f"short ablation video: {model}/{row['drift_id']}/r{row['round']}")
        for condition, (_, frame) in reads.items():
            if frame.shape != base.shape:
                raise RuntimeError("frame shape mismatch")
            difference = np.abs(frame.astype(np.int16) - base.astype(np.int16))
            accum[condition]["mae"].append(float(difference.mean()) / 255.0)
            accum[condition]["ssim"].append(ssim_bgr(frame, base))
            accum[condition]["changed"].append(float(np.mean(np.max(difference, axis=2) > 8)))
        frames += 1
    normal.release()
    for capture in variants.values():
        capture.release()
    if frames < 2:
        raise RuntimeError(f"empty or single-frame output: {normal_path}")
    result = []
    for condition in CONDITIONS:
        result.append({"model_id": model, "condition": condition, "drift_id": row["drift_id"],
                       "round": int(row["round"]), "pattern": row["pattern"],
                       "comparison_type": row["comparison_type"], "frames": frames,
                       "rgb_mae_01": f"{np.mean(accum[condition]['mae']):.9f}",
                       "mean_frame_ssim": f"{np.mean(accum[condition]['ssim']):.9f}",
                       "changed_pixel_ratio_gt_8_255": f"{np.mean(accum[condition]['changed']):.9f}",
                       "normal_sha256": sha256(normal_path),
                       "ablation_sha256": sha256(video_path(model, condition, row))})
    return result


def main() -> None:
    checkpoints = load_checkpoints()
    report = audit(checkpoints)
    tasks = [(model, row) for model in MODELS for row in checkpoints]
    with Pool(processes=8) as pool:
        grouped = pool.map(compare_task, tasks)
    rows = [row for group in grouped for row in group]
    if len(rows) != 192:
        raise RuntimeError("expected 192 metric rows")
    with (ROOT / "difference_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frame_counts = sorted({int(row["frames"]) for row in rows})
    report.update(status="passed", metric_rows=192, decoded_video_count=240,
                  decoded_frame_counts=frame_counts,
                  metric_reference="same-run normal_replay", metrics=["RGB MAE [0,1]", "mean frame SSIM",
                  "changed-pixel ratio (max channel delta > 8/255)"],
                  note="Metrics quantify output change only; human review determines semantic effect.")
    (ROOT / "inference_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

