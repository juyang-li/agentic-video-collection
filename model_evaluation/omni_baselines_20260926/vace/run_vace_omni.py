#!/usr/bin/env python3
"""Resumable Wan2.1-VACE-14B inference for the frozen OmniEdit subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"decodable": False}
    value = {
        "decodable": True,
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 6),
    }
    cap.release()
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--wan-repo", type=Path, required=True)
    ap.add_argument("--tier", choices=["A", "B", "all"], default="A")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--guide-scale", type=float, default=5.0)
    ap.add_argument("--offload-model", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("exactly one CUDA device must be visible")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if args.tier == "all" or r["tier"] == args.tier]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("empty task selection")

    sys.path.insert(0, str(args.wan_repo))
    from wan import WanVace  # type: ignore
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
    from wan.utils.utils import cache_video  # type: ignore

    args.output_root.mkdir(parents=True, exist_ok=True)
    config = WAN_CONFIGS["vace-14B"]
    run = {
        "schema_version": "omni.vace14b.mask.v1",
        "model_id": "Wan-AI/Wan2.1-VACE-14B",
        "checkpoint_dir": str(args.ckpt_dir),
        "wan_repo": str(args.wan_repo),
        "wan_commit": subprocess.check_output(
            ["git", "-C", str(args.wan_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tier": args.tier,
        "parameters": {
            "size": "832*480", "frames": 81, "fps": 16,
            "steps": args.steps, "solver": "unipc", "shift": args.shift,
            "guide_scale": args.guide_scale, "seed": args.seed,
            "prompt_enhancement": False, "offload_model": args.offload_model,
        },
        "cuda_visible_devices": visible,
        "torch": torch.__version__,
        "tasks": [],
        "status": "loading_model",
    }
    shard_suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index}-of-{args.num_shards}"
    manifest_path = args.output_root / f"generation_manifest_{args.tier}{shard_suffix}.json"
    write_json(manifest_path, run)
    pipeline = WanVace(
        config=config,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )
    run["status"] = "running"
    write_json(manifest_path, run)

    for position, row in enumerate(rows, 1):
        eval_id = row["eval_id"]
        mask_dir = args.mask_root / eval_id
        source = mask_dir / "source_conditioning.mp4"
        mask = mask_dir / "mask.mp4"
        mask_manifest = mask_dir / "mask_manifest.json"
        if not (source.is_file() and mask.is_file() and mask_manifest.is_file()):
            raise FileNotFoundError(f"missing audited condition for {eval_id}")
        output = args.output_root / "outputs" / f"{int(row['aligned_index']):03d}_{eval_id}.mp4"
        source_probe = probe(source)
        size_key = "832*480" if int(source_probe["width"]) >= int(source_probe["height"]) else "480*832"
        expected = {
            "decodable": True,
            # Wan's VACE preprocessing keeps the exact stride-aligned
            # conditioning dimensions (portrait OmniEdit is 464x848), which
            # can differ from the nominal SIZE_CONFIGS label (480x832).
            "width": int(source_probe["width"]),
            "height": int(source_probe["height"]),
            "frames": 81,
            "fps": 16.0,
        }
        if output.is_file() and probe(output) == expected:
            task = {"eval_id": eval_id, "status": "reused", "output": str(output),
                    "output_sha256": sha256(output), "probe": expected}
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            tmp = output.with_suffix(".tmp.mp4")
            if probe(tmp) != expected:
                print(f"[{position}/{len(rows)}] {eval_id}: prepare", flush=True)
                src_video, src_mask, src_refs = pipeline.prepare_source(
                    [str(source)], [str(mask)], [None], 81,
                    SIZE_CONFIGS[size_key], torch.device("cuda:0"),
                )
                print(f"[{position}/{len(rows)}] {eval_id}: generate", flush=True)
                video = pipeline.generate(
                    row["editing_instruction"], src_video, src_mask, src_refs,
                    size=SIZE_CONFIGS[size_key], frame_num=81,
                    shift=args.shift, sample_solver="unipc", sampling_steps=args.steps,
                    guide_scale=args.guide_scale, seed=args.seed,
                    offload_model=args.offload_model,
                )
                cache_video(tensor=video[None], save_file=str(tmp), fps=16, nrow=1,
                            normalize=True, value_range=(-1, 1))
            else:
                print(f"[{position}/{len(rows)}] {eval_id}: recover encoded temporary output",
                      flush=True)
            actual = probe(tmp)
            if actual != expected:
                raise RuntimeError(f"bad output for {eval_id}: {actual}")
            os.replace(tmp, output)
            task = {
                "eval_id": eval_id, "tier": row["tier"],
                "aligned_index": int(row["aligned_index"]), "category": row["category"],
                "prompt": row["editing_instruction"], "seed": args.seed,
                "size_key": size_key,
                "input_path": str(source), "input_sha256": sha256(source),
                "mask_path": str(mask), "mask_sha256": sha256(mask),
                "mask_manifest": str(mask_manifest), "output": str(output),
                "output_sha256": sha256(output), "probe": actual,
                "elapsed_s": round(time.time() - started, 3), "status": "generated",
            }
            print(f"[{position}/{len(rows)}] {eval_id}: complete {task['elapsed_s']}s", flush=True)
        run["tasks"] = [x for x in run["tasks"] if x["eval_id"] != eval_id] + [task]
        write_json(manifest_path, run)
    run["status"] = "complete"
    write_json(manifest_path, run)


if __name__ == "__main__":
    main()
