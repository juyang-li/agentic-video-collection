#!/usr/bin/env python3
"""Score the edited side with technical, frame-diff, MUSIQ and DINOv2 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/survey_manifest_reco_v3.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/filter_run_v1/quality_calibration_scores.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/filter_run_v1/quality_calibration_summary.json"
VENDOR_LIB = ROOT / "artifacts/vendor/build/lib"

FIELDS = [
    "filter_version",
    "sample_id",
    "dataset",
    "branch",
    "pair_id",
    "video_id",
    "role",
    "local_path",
    "quality_checked_roles",
    "technical_status",
    "error",
    "frame_count",
    "fps",
    "duration_seconds",
    "sampled_frames",
    "frame_diff_mean",
    "frame_diff_p05",
    "frame_diff_p95",
    "freeze_ratio",
    "black_frame_ratio",
    "white_frame_ratio",
    "sharpness_p10",
    "sharpness_mean",
    "musiq_p10",
    "musiq_mean",
    "dino_adjacent_cosine_min",
    "dino_adjacent_cosine_mean",
    "dino_max_feature_jump",
    "elapsed_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def required_roles(row: dict[str, str]) -> tuple[str, ...]:
    value = row.get("quality_checked_roles", "both")
    if value == "source":
        return ("source",)
    if value == "target":
        return ("target",)
    return ("source", "target")


def video_jobs(rows: list[dict[str, str]], role_mode: str = "quality") -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        roles = ("source", "target") if role_mode == "both" else required_roles(row)
        for role in roles:
            video_id = row.get(f"{role}_video_id", f"{row['pair_id']}:{role}")
            if video_id in seen:
                continue
            seen.add(video_id)
            jobs.append(
                {
                    "filter_version": row.get("filter_version", "general_edit_stage1_pilot_v1"),
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "branch": row["branch"],
                    "pair_id": row["pair_id"],
                    "video_id": video_id,
                    "role": role,
                    "local_path": row.get(f"{role}_local", ""),
                    "quality_checked_roles": row.get("quality_checked_roles", "both"),
                }
            )
    return jobs


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), level)) if values else math.nan


def decode_frames(path: Path, frame_samples: int):
    import cv2
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    fps = float(reader.get_avg_fps())
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid video frame count or fps")
    count = min(frame_samples, frame_count)
    indices = np.linspace(0, frame_count - 1, count).round().astype(np.int64)
    indices = np.unique(indices)
    rgb = reader.get_batch(indices).asnumpy()
    small = np.stack(
        [cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA) for frame in rgb]
    )
    return rgb, small, frame_count, fps


def cheap_metrics(frames: np.ndarray) -> dict[str, float]:
    import cv2

    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    differences = [
        float(np.mean(np.abs(right.astype(np.float32) - left.astype(np.float32))) / 255.0)
        for left, right in zip(grays, grays[1:])
    ]
    brightness = [float(np.mean(frame) / 255.0) for frame in grays]
    sharpness = [float(cv2.Laplacian(frame, cv2.CV_32F).var()) for frame in grays]
    return {
        "frame_diff_mean": float(np.mean(differences)) if differences else 0.0,
        "frame_diff_p05": percentile(differences, 5),
        "frame_diff_p95": percentile(differences, 95),
        "freeze_ratio": float(np.mean(np.asarray(differences) < 0.002)) if differences else 1.0,
        "black_frame_ratio": float(np.mean(np.asarray(brightness) < 0.03)),
        "white_frame_ratio": float(np.mean(np.asarray(brightness) > 0.97)),
        "sharpness_p10": percentile(sharpness, 10),
        "sharpness_mean": float(np.mean(sharpness)),
    }


class FullModels:
    def __init__(self, device: str, dino_name: str) -> None:
        import torch
        from transformers import AutoImageProcessor, Dinov2Model

        if str(VENDOR_LIB) not in sys.path:
            sys.path.insert(0, str(VENDOR_LIB))
        import pyiqa

        self.torch = torch
        self.device = torch.device(device)
        self.musiq = pyiqa.create_metric("musiq", device=self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_name, local_files_only=True, use_fast=False
        )
        self.dino = Dinov2Model.from_pretrained(dino_name, local_files_only=True)
        self.dino.to(self.device).eval()

    def musiq_scores(self, frames: np.ndarray, batch_size: int) -> list[float]:
        torch = self.torch
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                batch = torch.from_numpy(frames[start : start + batch_size]).permute(0, 3, 1, 2)
                batch = batch.to(self.device, dtype=torch.float32) / 255.0
                values = self.musiq(batch).reshape(-1).detach().cpu().tolist()
                scores.extend(float(value) for value in values)
        return scores

    def dino_metrics(self, frames: np.ndarray, batch_size: int) -> dict[str, float]:
        torch = self.torch
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                batch_frames = [frame for frame in frames[start : start + batch_size]]
                inputs = self.dino_processor(images=batch_frames, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.dino(**inputs).last_hidden_state[:, 0]
                embeddings.append(torch.nn.functional.normalize(output, dim=-1).cpu())
        features = torch.cat(embeddings, dim=0)
        if len(features) < 2:
            similarities = torch.ones(1)
        else:
            similarities = (features[:-1] * features[1:]).sum(dim=-1)
        return {
            "dino_adjacent_cosine_min": float(similarities.min()),
            "dino_adjacent_cosine_mean": float(similarities.mean()),
            "dino_max_feature_jump": float((1.0 - similarities).max()),
        }


def blank_result(job: dict[str, str]) -> dict[str, str]:
    return {**job, **{field: "" for field in FIELDS if field not in job}}


def score_job(
    job: dict[str, str],
    frame_samples: int,
    models: FullModels | None,
    batch_size: int,
    minimum_duration: float,
) -> dict[str, str]:
    started = time.monotonic()
    result = blank_result(job)
    path = resolve_local_path(job["local_path"]) if job["local_path"] else Path("")
    try:
        if not job["local_path"] or not path.is_file():
            raise FileNotFoundError(f"missing local video: {path}")
        _, small_frames, frame_count, fps = decode_frames(path, frame_samples)
        duration = frame_count / fps
        result.update(
            {
                "technical_status": "pass" if duration >= minimum_duration else "fail_short",
                "frame_count": str(frame_count),
                "fps": f"{fps:.6f}",
                "duration_seconds": f"{duration:.6f}",
                "sampled_frames": str(len(small_frames)),
            }
        )
        if duration < minimum_duration:
            result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
            return result
        result.update({key: f"{value:.8f}" for key, value in cheap_metrics(small_frames).items()})
        if models is not None:
            musiq = models.musiq_scores(small_frames, batch_size)
            result["musiq_p10"] = f"{percentile(musiq, 10):.8f}"
            result["musiq_mean"] = f"{float(np.mean(musiq)):.8f}"
            result.update(
                {key: f"{value:.8f}" for key, value in models.dino_metrics(small_frames, batch_size).items()}
            )
    except Exception as error:  # keep one bad file from losing the complete run
        result["technical_status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
    return result


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
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
    parser.add_argument("--metric-set", choices=("technical", "full"), default="technical")
    parser.add_argument("--role-mode", choices=("quality", "both"), default="quality")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-small")
    parser.add_argument("--frame-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minimum-duration", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.frame_samples < 2 or args.batch_size < 1:
        raise ValueError("frame-samples must be >=2 and batch-size must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    jobs = video_jobs(read_csv(args.input), args.role_mode)
    jobs = [job for index, job in enumerate(jobs) if index % args.shard_count == args.shard_index]
    missing_jobs = sum(
        not job["local_path"] or not resolve_local_path(job["local_path"]).is_file()
        for job in jobs
    )
    if args.skip_missing:
        jobs = [
            job for job in jobs
            if job["local_path"] and resolve_local_path(job["local_path"]).is_file()
        ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    existing: dict[str, dict[str, str]] = {}
    if args.resume and args.output.is_file():
        for row in read_csv(args.output):
            technical_complete = row.get("technical_status") in {"pass", "fail_short"}
            full_complete = bool(row.get("musiq_p10") and row.get("dino_max_feature_jump"))
            if technical_complete and (
                args.metric_set == "technical"
                or full_complete
                or row.get("technical_status") == "fail_short"
            ):
                existing[row["video_id"]] = row
        jobs = [job for job in jobs if job["video_id"] not in existing]
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = FullModels(device, args.dinov2_model) if args.metric_set == "full" else None

    results: list[dict[str, str]] = list(existing.values())
    run_started = time.monotonic()
    for index, job in enumerate(jobs, 1):
        results.append(
            score_job(job, args.frame_samples, models, args.batch_size, args.minimum_duration)
        )
        if index % 25 == 0 or index == len(jobs):
            write_results(args.output, results)
            print(f"scored_new={index}/{len(jobs)} total={len(results)}", flush=True)

    summary = {
        "metric_set": args.metric_set,
        "device": device,
        "input": str(args.input),
        "videos": len(results),
        "resumed_videos": len(existing),
        "newly_scored_videos": len(jobs),
        "status_counts": dict(Counter(row["technical_status"] for row in results)),
        "elapsed_seconds": time.monotonic() - run_started,
        "frame_samples": args.frame_samples,
        "role_mode": args.role_mode,
        "missing_jobs_before_skip": missing_jobs,
        "dinov2_model": args.dinov2_model if models is not None else None,
        "musiq_model": "musiq-koniq" if models is not None else None,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
