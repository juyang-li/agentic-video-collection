#!/usr/bin/env python3
"""Build aligned Bernini-R task files from the frozen OmniEdit subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

MV2V_CATEGORIES = {
    "Object_direction_change",
    "Object_location_change",
    "Object_location_swap",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    args = parser.parse_args()

    # Bernini resolves task paths relative to the checkout it runs from.  The
    # runtime checkout lives on node-local /tmp, so relative output paths can
    # silently place results there instead of in the benchmark workspace.
    # Keep the lexical absolute workspace path (without resolving a possible
    # runtime symlink used by recovery) in every generated task.
    args.output_dir = args.output_dir.absolute()
    args.task_dir = args.task_dir.absolute()
    args.stage_dir = args.stage_dir.absolute()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.task_dir.mkdir(parents=True, exist_ok=True)
    args.stage_dir.mkdir(parents=True, exist_ok=True)

    with args.aligned.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 48:
        raise ValueError(f"expected 48 aligned rows, got {len(rows)}")

    tasks = []
    records = []
    seen = set()
    for row in rows:
        eval_id = row["eval_id"]
        if eval_id in seen:
            raise ValueError(f"duplicate eval_id: {eval_id}")
        seen.add(eval_id)
        source = Path(row["official_source_video"])
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_source_hash = sha256(source)
        if actual_source_hash != row["official_source_sha256"]:
            raise ValueError(f"source SHA256 mismatch for {eval_id}")

        staged_source = args.stage_dir / f"{int(row['aligned_index']):03d}_{eval_id}.mp4"
        if not staged_source.exists() or sha256(staged_source) != actual_source_hash:
            shutil.copy2(source, staged_source)
        output = args.output_dir / f"{int(row['aligned_index']):03d}_{eval_id}.mp4"
        task_type = "mv2v" if row["category"] in MV2V_CATEGORIES else "v2v"
        task = {
            "task_type": task_type,
            "prompt": row["editing_instruction"],
            "video": str(staged_source),
            "output": str(output),
        }
        tasks.append(task)
        records.append({
            "eval_id": eval_id,
            "tier": row["tier"],
            "aligned_index": int(row["aligned_index"]),
            "video_id": row["video_id"],
            "category": row["category"],
            "model_id": "ByteDance/Bernini-R-Diffusers",
            "model_revision": "pending_download_resolution",
            "code_commit": "pending_runtime_resolution",
            "task_type": task_type,
            "guidance_mode": "v2v_apg",
            "prompt": row["editing_instruction"],
            "seed": 42,
            "num_frames": 81,
            "fps": 16,
            "height": 480,
            "width": 848,
            "num_inference_steps": 40,
            "flow_shift": 5.0,
            "input_path": str(source),
            "staged_input_path": str(staged_source),
            "input_sha256": actual_source_hash,
            "output_path": str(output),
            "status": "complete" if output.is_file() else "pending",
        })

    tier_a = [task for task, row in zip(tasks, rows) if row["tier"] == "A"]
    smoke = [
        next(task for task in tier_a if task["task_type"] == "v2v"),
        next(task for task in tier_a if task["task_type"] == "mv2v"),
    ]
    for name, content in (
        ("tasks_full.json", tasks),
        ("tasks_tier_a.json", tier_a),
        ("tasks_smoke.json", smoke),
    ):
        with (args.task_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    with (args.task_dir / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    provenance = {
        "aligned_manifest": str(args.aligned.resolve()),
        "aligned_manifest_sha256": sha256(args.aligned),
        "model_dir": str(args.model_dir),
        "code_dir": str(args.code_dir),
        "rows": len(rows),
        "tier_a_rows": len(tier_a),
        "mv2v_rows": sum(task["task_type"] == "mv2v" for task in tasks),
    }
    with (args.task_dir / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
