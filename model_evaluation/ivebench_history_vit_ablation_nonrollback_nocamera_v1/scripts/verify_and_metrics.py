#!/usr/bin/env python3
"""Audit IVEBench History-ViT outputs and measure each ablation against normal."""

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
CONDITIONS = ("vit_zero_text_keep", "vit_gaussian_text_keep")
EXPECTED = {"chains": 24, "turns": 65, "checkpoints": 41, "normal": 130, "ablation": 164}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases() -> list[dict]:
    payload = load_json(ROOT / "benchmark.json")
    cases = payload.get("cases") or []
    shape = (
        len(cases),
        sum(len(case.get("outputs", [])) for case in cases),
        sum(max(0, len(case.get("outputs", [])) - 1) for case in cases),
    )
    if shape != (EXPECTED["chains"], EXPECTED["turns"], EXPECTED["checkpoints"]):
        raise RuntimeError(f"benchmark shape mismatch: {shape}")
    axes: dict[str, int] = {}
    for case in cases:
        outputs = case.get("outputs") or []
        if case.get("chain_type") == "rollback" or any(
            item.get("history_reference_round") is not None for item in outputs
        ):
            raise RuntimeError(f"rollback/history-reference chain remains: {case.get('chain_id')}")
        if [item.get("round") for item in outputs] != list(range(1, len(outputs) + 1)):
            raise RuntimeError(f"non-contiguous rounds: {case.get('chain_id')}")
        source = Path(case["source"])
        if not source.is_file() or sha256(source) != case.get("source_sha256"):
            raise RuntimeError(f"source missing or hash mismatch: {source}")
        for item in outputs:
            axis = item.get("axis")
            axes[axis] = axes.get(axis, 0) + 1
            if axis == "camera":
                raise RuntimeError(f"camera-axis turn remains: {case['chain_id']} R{item['round']}")
    if axes != {"appearance": 23, "action": 20, "outfit": 22}:
        raise RuntimeError(f"unexpected axis distribution: {axes}")
    removed = payload.get("removed_camera_turns") or []
    rewrites = payload.get("camera_dependency_rewrites") or []
    if payload.get("removed_camera_turn_count") != 21 or len(removed) != 21:
        raise RuntimeError("camera-removal provenance must contain 21 turns")
    case_by_id = {case["chain_id"]: case for case in cases}
    for record in removed:
        if record.get("removed_turn", {}).get("axis") != "camera":
            raise RuntimeError(f"invalid removed-camera provenance: {record}")
        retained_source_rounds = {item.get("source_round") for item in case_by_id[record["chain_id"]]["outputs"]}
        if record.get("source_round") in retained_source_rounds:
            raise RuntimeError(f"removed camera source round is still retained: {record}")
    if payload.get("post_camera_rewrite_count") != 33 or len(rewrites) != 33:
        raise RuntimeError("camera-dependency provenance must contain 33 rewrites")
    for record in rewrites:
        case = case_by_id[record["chain_id"]]
        item = case["outputs"][int(record["derived_round"]) - 1]
        revised = record.get("revised") or {}
        for field in ("instruction", "preserve_requirements", "target_state"):
            if item.get(field) != revised.get(field):
                raise RuntimeError(f"camera-dependency rewrite mismatch: {record['chain_id']} R{record['derived_round']}")
        if not record.get("changed_fields") or record.get("original") == revised:
            raise RuntimeError(f"ineffective camera-dependency rewrite: {record}")
    return cases


def output_path(model: str, condition: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model / condition / chain_id / f"turn{round_no - 1:02d}.mp4"


def validate_adapter(model: str, manifest: dict) -> None:
    check = manifest.get("adapter_check")
    if model == "lance_base":
        if check not in (None, {}):
            raise RuntimeError("Lance Base unexpectedly reports an adapter")
        return
    check = check or {}
    values = (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B"))
    if values != (1016, 1016, 503) or check.get("missing_live_keys", 0) != 0:
        raise RuntimeError(f"gtfresh adapter contract failed: {values}, {check}")


def validate_ablation_manifest(model: str, condition: str, benchmark_sha: str) -> dict:
    manifest = load_json(ROOT / "outputs" / model / condition / "manifest.json")
    actual = (
        manifest.get("status"),
        manifest.get("condition"),
        manifest.get("chain_count"),
        manifest.get("total_turns"),
        manifest.get("formal_checkpoints"),
    )
    expected = ("generated", condition, EXPECTED["chains"], EXPECTED["turns"], EXPECTED["checkpoints"])
    if actual != expected or manifest.get("benchmark_sha256") != benchmark_sha:
        raise RuntimeError(f"manifest contract failed: {model}/{condition}: {actual}")
    if (
        manifest.get("checkpoint_count") != EXPECTED["checkpoints"]
        or manifest.get("fixed_normal_history") is not True
        or manifest.get("history_text_preserved") is not True
        or manifest.get("direct_source_vit_preserved") is not True
        or manifest.get("direct_source_vae_preserved") is not True
    ):
        raise RuntimeError(f"ablation manifest contract failed: {model}/{condition}")
    validate_adapter(model, manifest)
    return manifest


def full_decode(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = 0
    width = height = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"bad decoded frame: {path}")
        if frames == 0:
            height, width = frame.shape[:2]
        elif frame.shape[:2] != (height, width):
            raise RuntimeError(f"variable frame dimensions: {path}")
        frames += 1
    capture.release()
    if frames < 2:
        raise RuntimeError(f"empty or single-frame video: {path}")
    return frames, width, height


def audit(cases: list[dict]) -> dict:
    normal_count = ablation_count = receipt_count = 0
    decoded_frame_counts: set[int] = set()
    benchmark_sha = sha256(ROOT / "benchmark.json")
    for model in MODELS:
        shard_manifests = [
            load_json(ROOT / "outputs" / model / "normal" / "shards" / f"shard{index:02d}-of-02.json")
            for index in range(2)
        ]
        chain_ids = [chain_id for manifest in shard_manifests for chain_id in manifest.get("chain_ids", [])]
        if (
            any(manifest.get("status") != "generated" for manifest in shard_manifests)
            or any(manifest.get("model_id") != model for manifest in shard_manifests)
            or any(manifest.get("benchmark_sha256") != benchmark_sha for manifest in shard_manifests)
            or sum(int(manifest.get("turn_count", 0)) for manifest in shard_manifests) != 65
            or sorted(chain_ids) != sorted(case["chain_id"] for case in cases)
        ):
            raise RuntimeError(f"normal shard manifest contract failed: {model}")
        for manifest in shard_manifests:
            validate_adapter(model, manifest)
        for case in cases:
            case_status_path = ROOT / "outputs" / model / "normal" / case["chain_id"] / "case_status.json"
            status = load_json(case_status_path)
            if status.get("status") != "generated" or status.get("turn_count") != len(case["outputs"]):
                raise RuntimeError(f"normal case status failed: {case_status_path}")
            for item in case["outputs"]:
                video = output_path(model, "normal", case["chain_id"], item["round"])
                if not video.is_file() or video.stat().st_size <= 1000:
                    raise FileNotFoundError(video)
                frames, _, _ = full_decode(video)
                decoded_frame_counts.add(frames)
                normal_count += 1

        for condition in CONDITIONS:
            validate_ablation_manifest(model, condition, benchmark_sha)
            for case in cases:
                for item in case["outputs"][1:]:
                    round_no = item["round"]
                    video = output_path(model, condition, case["chain_id"], round_no)
                    receipt_path = video.with_suffix(".json")
                    if not video.is_file() or video.stat().st_size <= 1000 or not receipt_path.is_file():
                        raise FileNotFoundError(f"missing output or receipt: {video}")
                    receipt = load_json(receipt_path)
                    receipt_contract = (
                        receipt.get("status"), receipt.get("model_id"), receipt.get("condition"),
                        receipt.get("chain_id"), receipt.get("round"), receipt.get("output_sha256"),
                    )
                    expected_contract = (
                        "generated", model, condition, case["chain_id"], round_no, sha256(video),
                    )
                    if receipt_contract != expected_contract:
                        raise RuntimeError(f"receipt mismatch: {receipt_path}")
                    trace = receipt.get("trace") or {}
                    direct_ok = trace.get("direct_source_unchanged") is True or (
                        trace.get("direct_source_vit_unchanged") is True
                        and trace.get("direct_source_vae_unchanged") is True
                    )
                    if (
                        trace.get("vit_hook_calls") != 1
                        or int(trace.get("history_vit_tokens", 0)) <= 0
                        or not direct_ok
                        or trace.get("history_text_unchanged") is not True
                        or int(trace.get("text_hook_calls", 0)) != 0
                    ):
                        raise RuntimeError(f"causal intervention audit failed: {receipt_path}: {trace}")
                    frames, _, _ = full_decode(video)
                    decoded_frame_counts.add(frames)
                    ablation_count += 1
                    receipt_count += 1
    if (normal_count, ablation_count, receipt_count) != (
        EXPECTED["normal"], EXPECTED["ablation"], EXPECTED["ablation"]
    ):
        raise RuntimeError("inference coverage mismatch")
    return {
        "normal_videos": normal_count,
        "formal_ablation_videos": ablation_count,
        "receipts": receipt_count,
        "decoded_video_count": normal_count + ablation_count,
        "decoded_frame_counts": sorted(decoded_frame_counts),
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "chains": EXPECTED["chains"],
        "turns": EXPECTED["turns"],
        "checkpoints": EXPECTED["checkpoints"],
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


def compare_task(task: tuple[str, str, dict, int]) -> dict:
    model, condition, case, round_no = task
    baseline_path = output_path(model, "normal", case["chain_id"], round_no)
    variant_path = output_path(model, condition, case["chain_id"], round_no)
    baseline = cv2.VideoCapture(str(baseline_path))
    variant = cv2.VideoCapture(str(variant_path))
    maes: list[float] = []
    ssims: list[float] = []
    changed: list[float] = []
    while True:
        ok_base, base = baseline.read()
        ok_variant, frame = variant.read()
        if not ok_base:
            if ok_variant:
                raise RuntimeError(f"frame count mismatch: {variant_path}")
            break
        if not ok_variant or frame.shape != base.shape:
            raise RuntimeError(f"short or shape-mismatched ablation: {variant_path}")
        difference = np.abs(frame.astype(np.int16) - base.astype(np.int16))
        maes.append(float(difference.mean()) / 255.0)
        ssims.append(ssim_bgr(frame, base))
        changed.append(float(np.mean(np.max(difference, axis=2) > 8)))
    baseline.release()
    variant.release()
    if len(maes) < 2:
        raise RuntimeError(f"insufficient frames: {variant_path}")
    return {
        "model_id": model,
        "condition": condition,
        "chain_id": case["chain_id"],
        "round": round_no,
        "instruction": case["outputs"][round_no - 1]["instruction"],
        "frames": len(maes),
        "rgb_mae_01": f"{np.mean(maes):.9f}",
        "mean_frame_ssim": f"{np.mean(ssims):.9f}",
        "changed_pixel_ratio_gt_8_255": f"{np.mean(changed):.9f}",
        "normal_sha256": sha256(baseline_path),
        "ablation_sha256": sha256(variant_path),
    }


def main() -> None:
    cases = load_cases()
    report = audit(cases)
    tasks = [
        (model, condition, case, item["round"])
        for model in MODELS
        for condition in CONDITIONS
        for case in cases
        for item in case["outputs"][1:]
    ]
    with Pool(processes=min(8, len(tasks))) as pool:
        rows = pool.map(compare_task, tasks)
    keys = {(row["model_id"], row["condition"], row["chain_id"], row["round"]) for row in rows}
    if len(rows) != EXPECTED["ablation"] or len(keys) != EXPECTED["ablation"]:
        raise RuntimeError("expected 164 unique difference-metric rows")
    with (ROOT / "difference_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        status="passed",
        metric_rows=len(rows),
        metric_reference="same-model normal chain at the same round",
        metrics=["RGB MAE [0,1]", "mean frame SSIM", "changed-pixel ratio (max channel delta > 8/255)"],
        note="Metrics quantify change only; human review determines semantic effect.",
    )
    (ROOT / "inference_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
