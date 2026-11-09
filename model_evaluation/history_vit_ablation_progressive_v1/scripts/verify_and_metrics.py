#!/usr/bin/env python3
"""Audit the no-camera inference outputs and compute differences from normal."""

from __future__ import annotations

import csv
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lance_base", "gtfresh_step1150")
CONDITIONS = (
    "vit_zero_text_keep",
    "vit_gaussian_text_keep",
    "vit_zero_text_zero",
    "vit_gaussian_text_zero",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normal_path(model: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model / "normal" / chain_id / f"turn{round_no - 1:02d}.mp4"


def ablation_path(model: str, condition: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model / condition / chain_id / f"turn{round_no - 1:02d}.mp4"


def load_manifest(model: str, condition: str) -> dict:
    path = ROOT / "outputs" / model / condition / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def case_signature(cases: list[dict]) -> list[tuple]:
    return [
        (
            case["chain_id"],
            case["source_sha256"],
            tuple(item["instruction"] for item in case["outputs"]),
        )
        for case in cases
    ]


def load_cases() -> list[dict]:
    manifest = load_manifest("lance_base", "normal")
    cases = manifest.get("cases") or []
    if len(cases) != 3 or sum(len(case.get("outputs", [])) for case in cases) != 9:
        raise RuntimeError("expected three no-camera chains and nine normal turns")
    if sum(max(0, len(case["outputs"]) - 1) for case in cases) != 6:
        raise RuntimeError("expected six R2+ checkpoints")
    forbidden = ("switch the camera", "only change the camera", "change the camera viewpoint")
    for case in cases:
        if not Path(case["source"]).is_file():
            raise FileNotFoundError(case["source"])
        for item in case["outputs"]:
            if any(phrase in item["instruction"].lower() for phrase in forbidden):
                raise RuntimeError(f"camera-edit instruction remains: {case['chain_id']} R{item['round']}")
    return cases


def adapter_ok(model: str, manifest: dict) -> None:
    check = manifest.get("adapter_check")
    if model == "lance_base":
        if check is not None:
            raise RuntimeError("Lance Base unexpectedly reports an adapter")
        return
    check = check or {}
    values = (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B"))
    if values != (1016, 1016, 503):
        raise RuntimeError(f"gtfresh adapter verification missing: {values}")
    if check.get("missing_live_keys", 0) != 0:
        raise RuntimeError("gtfresh adapter has missing live keys")


def audit(cases: list[dict]) -> dict:
    signature = case_signature(cases)
    normal_videos = ablation_videos = receipts = hook_receipts = 0
    for model in MODELS:
        normal = load_manifest(model, "normal")
        expected = ("generated", "normal", 3, 9, 6)
        actual = (
            normal.get("status"), normal.get("condition"), normal.get("chain_count"),
            normal.get("total_turns"), normal.get("formal_checkpoints"),
        )
        if actual != expected or case_signature(normal.get("cases") or []) != signature:
            raise RuntimeError(f"bad normal manifest: {model}: {actual}")
        adapter_ok(model, normal)
        for case in cases:
            status_path = ROOT / "outputs" / model / "normal" / case["chain_id"] / "case_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "generated" or status.get("turns") != len(case["outputs"]):
                raise RuntimeError(f"bad normal case status: {status_path}")
            for round_no in range(1, len(case["outputs"]) + 1):
                video = normal_path(model, case["chain_id"], round_no)
                if not video.is_file() or video.stat().st_size <= 1000:
                    raise FileNotFoundError(video)
                normal_videos += 1

        for condition in CONDITIONS:
            manifest = load_manifest(model, condition)
            actual = (
                manifest.get("status"), manifest.get("condition"), manifest.get("chain_count"),
                manifest.get("total_turns"), manifest.get("formal_checkpoints"),
                manifest.get("checkpoint_count"), manifest.get("fixed_normal_history"),
            )
            expected = ("generated", condition, 3, 9, 6, 6, True)
            if actual != expected or case_signature(manifest.get("cases") or []) != signature:
                raise RuntimeError(f"bad ablation manifest: {model}/{condition}: {actual}")
            adapter_ok(model, manifest)
            for case in cases:
                for round_no in range(2, len(case["outputs"]) + 1):
                    video = ablation_path(model, condition, case["chain_id"], round_no)
                    receipt = video.with_suffix(".json")
                    if not video.is_file() or video.stat().st_size <= 1000 or not receipt.is_file():
                        raise FileNotFoundError(f"missing output or receipt: {video}")
                    data = json.loads(receipt.read_text(encoding="utf-8"))
                    expected_receipt = (
                        "generated", model, condition, case["chain_id"], round_no, sha256(video),
                    )
                    actual_receipt = (
                        data.get("status"), data.get("model_id"), data.get("condition"),
                        data.get("chain_id"), data.get("round"), data.get("output_sha256"),
                    )
                    if actual_receipt != expected_receipt:
                        raise RuntimeError(f"receipt mismatch: {receipt}")
                    trace = data.get("trace") or {}
                    if (trace.get("vit_hook_calls") != 1
                            or trace.get("direct_source_unchanged") is not True
                            or int(trace.get("history_vit_tokens", 0)) <= 0):
                        raise RuntimeError(f"ViT intervention audit failed: {receipt}")
                    if condition.endswith("text_zero") and (
                            trace.get("text_hook_calls") != 1
                            or int(trace.get("history_text_tokens", 0)) <= 0):
                        raise RuntimeError(f"text intervention audit failed: {receipt}")
                    ablation_videos += 1
                    receipts += 1
                    hook_receipts += 1
    if (normal_videos, ablation_videos, receipts, hook_receipts) != (18, 48, 48, 48):
        raise RuntimeError("inference coverage mismatch")
    return {
        "normal_videos": normal_videos,
        "formal_ablation_videos": ablation_videos,
        "receipts": receipts,
        "formal_ablation_hook_receipts": hook_receipts,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "chains": 3,
        "checkpoints": 6,
    }


def ssim_bgr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = []
    for channel in range(3):
        x, y = a[:, :, channel], b[:, :, channel]
        ux = cv2.GaussianBlur(x, (11, 11), 1.5)
        uy = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - ux * ux
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - uy * uy
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - ux * uy
        scores.append(float(np.mean(
            ((2 * ux * uy + c1) * (2 * vxy + c2))
            / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        )))
    return sum(scores) / 3


def compare_task(task: tuple[str, dict, int]) -> list[dict]:
    model, case, round_no = task
    baseline_path = normal_path(model, case["chain_id"], round_no)
    baseline = cv2.VideoCapture(str(baseline_path))
    variants = {
        condition: cv2.VideoCapture(str(ablation_path(model, condition, case["chain_id"], round_no)))
        for condition in CONDITIONS
    }
    accum = {condition: {"mae": [], "ssim": [], "changed": []} for condition in CONDITIONS}
    frames = 0
    while True:
        ok, base = baseline.read()
        reads = {condition: capture.read() for condition, capture in variants.items()}
        if not ok:
            if any(flag for flag, _ in reads.values()):
                raise RuntimeError(f"frame count mismatch: {model}/{case['chain_id']}/R{round_no}")
            break
        if not all(flag for flag, _ in reads.values()):
            raise RuntimeError(f"short ablation video: {model}/{case['chain_id']}/R{round_no}")
        for condition, (_, frame) in reads.items():
            if frame.shape != base.shape:
                raise RuntimeError("frame shape mismatch")
            difference = np.abs(frame.astype(np.int16) - base.astype(np.int16))
            accum[condition]["mae"].append(float(difference.mean()) / 255.0)
            accum[condition]["ssim"].append(ssim_bgr(frame, base))
            accum[condition]["changed"].append(float(np.mean(np.max(difference, axis=2) > 8)))
        frames += 1
    baseline.release()
    for capture in variants.values():
        capture.release()
    if frames < 2:
        raise RuntimeError(f"empty or single-frame output: {baseline_path}")
    rows = []
    instruction = case["outputs"][round_no - 1]["instruction"]
    for condition in CONDITIONS:
        rows.append({
            "model_id": model,
            "condition": condition,
            "chain_id": case["chain_id"],
            "round": round_no,
            "instruction": instruction,
            "frames": frames,
            "rgb_mae_01": f"{np.mean(accum[condition]['mae']):.9f}",
            "mean_frame_ssim": f"{np.mean(accum[condition]['ssim']):.9f}",
            "changed_pixel_ratio_gt_8_255": f"{np.mean(accum[condition]['changed']):.9f}",
            "normal_sha256": sha256(baseline_path),
            "ablation_sha256": sha256(ablation_path(model, condition, case["chain_id"], round_no)),
        })
    return rows


def main() -> None:
    cases = load_cases()
    report = audit(cases)
    tasks = [
        (model, case, round_no)
        for model in MODELS
        for case in cases
        for round_no in range(2, len(case["outputs"]) + 1)
    ]
    with Pool(processes=6) as pool:
        grouped = pool.map(compare_task, tasks)
    rows = [row for group in grouped for row in group]
    if len(rows) != 48 or len({(r["model_id"], r["condition"], r["chain_id"], r["round"])
                               for r in rows}) != 48:
        raise RuntimeError("expected 48 unique difference-metric rows")
    with (ROOT / "difference_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        status="passed",
        metric_rows=48,
        metric_decoded_video_count=60,
        decoded_frame_counts=sorted({int(row["frames"]) for row in rows}),
        metric_reference="same-model normal chain at the same round",
        metrics=["RGB MAE [0,1]", "mean frame SSIM",
                 "changed-pixel ratio (max channel delta > 8/255)"],
        note="Metrics quantify change only; human review determines semantic effect.",
    )
    (ROOT / "inference_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
