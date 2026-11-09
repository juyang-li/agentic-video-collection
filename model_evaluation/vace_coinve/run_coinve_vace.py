#!/usr/bin/env python3
"""Run Wan2.1-VACE over CoinVE instruction chains, round by round.

The previous round's output is always the next round's input.  The runner is
idempotent: a completed round with matching technical metadata is reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"decodable": False}
    result = {
        "decodable": True,
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 6),
    }
    cap.release()
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def read_rows(path: Path, max_chains: int, start: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["sample_rank"]))
    return rows[start : start + max_chains]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--max-chains", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--frames", type=int, default=81)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one GPU")

    sys.path.insert(0, str(args.wan_repo))
    from wan import WanVace  # type: ignore
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
    from wan.utils.utils import cache_video  # type: ignore

    rows = read_rows(args.manifest, args.max_chains, args.start)
    if not rows:
        raise RuntimeError("manifest selection is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": "coinve.vace_multiround.v1",
        "model": "Wan-AI/Wan2.1-VACE-1.3B",
        "manifest": str(args.manifest.resolve()),
        "media_root": str(args.media_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "checkpoint_dir": str(args.ckpt_dir.resolve()),
        "wan_repository": str(args.wan_repo.resolve()),
        "wan_commit": git_revision(args.wan_repo),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_visible_devices": visible,
        "config": {"size": "832*480", "frames": args.frames, "fps": 16, "steps": args.steps, "seed": args.seed},
        "chains": [],
    }

    pipeline = WanVace(
        config=WAN_CONFIGS["vace-1.3B"],
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    for row in rows:
        sample_id = row["sample_id"]
        instructions = json.loads(row["instructions_json"])
        chain_dir = args.output_root / sample_id
        chain_dir.mkdir(parents=True, exist_ok=True)
        chain = {"sample_id": sample_id, "sample_rank": int(row["sample_rank"]), "instructions": instructions, "rounds": []}
        current = args.media_root / f"{sample_id}_source.mp4"
        if not current.exists():
            raise FileNotFoundError(current)
        for round_index, instruction in enumerate(instructions, start=1):
            output = chain_dir / f"round_{round_index:02d}.mp4"
            expected = {"width": 832, "height": 480, "frames": args.frames, "fps": 16.0, "decodable": True}
            if output.exists() and probe(output) == expected:
                status = "reused"
            else:
                started = time.time()
                src_video, src_mask, src_refs = pipeline.prepare_source(
                    [str(current)], [None], [None], args.frames,
                    SIZE_CONFIGS["832*480"], torch.device("cuda:0"))
                video = pipeline.generate(
                    "Edit the video according to this instruction: " + instruction
                    + " Preserve all unrelated content, camera motion, identity, lighting, and temporal continuity.",
                    src_video, src_mask, src_refs,
                    size=SIZE_CONFIGS["832*480"], frame_num=args.frames,
                    shift=16, sample_solver="unipc", sampling_steps=args.steps,
                    guide_scale=5, seed=args.seed + int(row["sample_rank"]) * 100 + round_index,
                    offload_model=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                cache_video(tensor=video[None], save_file=str(output), fps=16, nrow=1,
                            normalize=True, value_range=(-1, 1))
                status = "generated"
                elapsed = time.time() - started
            info = {
                "round_index": round_index,
                "instruction": instruction,
                "input_video": str(current),
                "input_sha256": sha256(current),
                "output_video": str(output),
                "output_sha256": sha256(output),
                "output_probe": probe(output),
                "status": status,
            }
            if status == "generated":
                info["elapsed_s"] = round(time.time() - started, 3)
            chain["rounds"].append(info)
            write_json(chain_dir / "chain_manifest.json", chain)
            current = output
        run_manifest["chains"].append(chain)
        write_json(args.output_root / "run_manifest.json", run_manifest)

    run_manifest["status"] = "completed"
    write_json(args.output_root / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
