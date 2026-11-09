#!/usr/bin/env python3
"""Run one isolated bundle of unmodified IVEBench metric functions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
UPSTREAM_METRICS = ROOT / "upstream" / "metrics"
INFO = ARTIFACTS / "short_subject_metric_info.json"
PATH_YML = ROOT / "metric_paths.yml"
FRAMES = ARTIFACTS / "metric_frames"
OUT = ARTIFACTS / "local_metrics"
CACHE = ROOT / "models" / "builtin"
STAGES = {
    "short_subject": (91, "short_subject_metric_info.json"),
    "long_subject": (43, "metric_info.json"),
    "attribute_all": (94, "metric_info.json"),
}

BUNDLES = {
    "quality": [
        ("quality", "subject_consistency"),
        ("quality", "background_consistency"),
        ("quality", "temporal_flickering"),
    ],
    "temporal": [
        ("quality", "motion_smoothness"),
        ("quality", "vtss"),
    ],
    "semantic": [
        ("compliance", "overall_semantic_consistency"),
        ("compliance", "phrase_semantic_consistency"),
        ("fidelity", "semantic_fidelity"),
    ],
    "motion": [("fidelity", "motion_fidelity")],
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reusable_metric_result(path: Path, metric: str, expected_samples: int) -> dict | None:
    """Load a previously completed full-manifest result, or reject it safely."""
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    details = result.get("details")
    if not (
        result.get("metric") == metric
        and result.get("status") == "complete"
        and result.get("records") == expected_samples
        and result.get("error_records") == 0
        and isinstance(details, list)
        and len(details) == expected_samples
        and result.get("valid_records", 0) > 0
    ):
        return None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="short_subject")
    parser.add_argument("--model-kind", choices=("base", "lora_step677"), required=True)
    parser.add_argument("--bundle", choices=tuple(BUNDLES), required=True)
    args = parser.parse_args()
    expected_samples, info_name = STAGES[args.stage]
    artifacts = ROOT / "artifacts" / args.stage
    info_path = artifacts / info_name
    frames = artifacts / "metric_frames"
    out = artifacts / "local_metrics"
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"bundle expects one visible GPU, got {torch.cuda.device_count()}")
    if not info_path.exists():
        raise FileNotFoundError(f"run prepare_metric_frames.py first: {info_path}")
    manifest = json.loads((CACHE / "prefetch_manifest.json").read_text())
    clip_paths = [Path(item["path"]) for item in manifest["clip"]]
    if len(clip_paths) != 1 or digest_file(clip_paths[0]) != manifest["clip"][0]["sha256"]:
        raise ValueError("cached CLIP model failed provenance validation")
    torch.hub.set_dir(manifest["torchhub_dir"])

    sys.path.insert(0, str(ROOT / "runtime" / "python_deps"))
    sys.path.insert(0, str(UPSTREAM_METRICS))
    os.chdir(UPSTREAM_METRICS)
    source = frames / "source"
    target = frames / args.model_kind
    output = out / args.model_kind / args.bundle
    output.mkdir(parents=True, exist_ok=True)
    full_info = json.loads(info_path.read_text())
    if len(full_info) != expected_samples:
        raise ValueError(f"{args.stage} metric info must contain {expected_samples} records")
    reusable = {
        metric: reusable_metric_result(output / f"{metric}.json", metric, expected_samples)
        for _, metric in BUNDLES[args.bundle]
    }
    if all(result is not None for result in reusable.values()):
        summary = {
            "status": "complete", "stage": args.stage, "model_kind": args.model_kind,
            "bundle": args.bundle, "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "official_upstream": str(UPSTREAM_METRICS), "failures": 0,
            "reused_complete_results": True,
            "metrics": {
                metric: {
                    key: value for key, value in result.items()
                    if key in ("status", "average", "records", "valid_records", "error_records")
                }
                for metric, result in reusable.items() if result is not None
            },
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    smoke_info = output / "_smoke_info.json"
    write_json(smoke_info, full_info[:1])
    summary = {
        "status": "running", "stage": args.stage, "model_kind": args.model_kind, "bundle": args.bundle,
        "started_at": datetime.now(timezone.utc).isoformat(), "official_upstream": str(UPSTREAM_METRICS),
        "metrics": {},
    }
    write_json(output / "summary.json", summary)
    failures = 0
    for folder, metric in BUNDLES[args.bundle]:
        started = datetime.now(timezone.utc).isoformat()
        try:
            module = importlib.import_module(f"{folder}.{metric}")
            function = getattr(module, f"compute_{metric}")
            smoke_average, smoke_details = function(
                json_dir=str(smoke_info), device="cuda", source_videos_path=str(source),
                target_videos_path=str(target), use_frames=True,
                path_yml=str(PATH_YML), clip_model_name=str(clip_paths[0]),
            )
            smoke_errors = sum(isinstance(item, dict) and "error" in item for item in smoke_details)
            smoke_valid = sum(isinstance(item, dict) and item.get("video_results") != -1 for item in smoke_details)
            smoke_ok = len(smoke_details) == 1 and smoke_errors == 0 and smoke_valid == 1
            smoke_result = {
                "metric": metric, "average": smoke_average, "details": smoke_details,
                "status": "complete" if smoke_ok else "failed",
            }
            write_json(output / f"{metric}_smoke.json", smoke_result)
            if not smoke_ok:
                raise RuntimeError(f"one-sample metric smoke failed: {smoke_result}")
            del smoke_details, smoke_result
            gc.collect(); torch.cuda.empty_cache()
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            average, details = function(
                json_dir=str(info_path), device="cuda", source_videos_path=str(source),
                target_videos_path=str(target), use_frames=True,
                path_yml=str(PATH_YML), clip_model_name=str(clip_paths[0]),
            )
            errors = sum(isinstance(item, dict) and "error" in item for item in details)
            valid = sum(isinstance(item, dict) and item.get("video_results") != -1 for item in details)
            ok = len(details) == expected_samples and errors == 0 and valid > 0
            result = {
                "metric": metric, "average": average, "details": details,
                "records": len(details), "valid_records": valid, "error_records": errors,
                "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete" if ok else "incomplete",
            }
            if not ok:
                failures += 1
        except Exception as error:
            failures += 1
            result = {
                "metric": metric, "status": "failed", "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
            }
        write_json(output / f"{metric}.json", result)
        summary["metrics"][metric] = {
            key: value for key, value in result.items()
            if key in ("status", "average", "records", "valid_records", "error_records", "error_type", "error")
        }
        write_json(output / "summary.json", summary)
        torch.cuda.empty_cache()
    summary.update(
        status="complete" if failures == 0 else "incomplete",
        failures=failures, finished_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
