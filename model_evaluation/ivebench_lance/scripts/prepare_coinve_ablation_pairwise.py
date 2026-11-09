#!/usr/bin/env python3
"""Prepare blinded, bidirectional pairwise tasks for the three step-677 models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

from decord import VideoReader, cpu
import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/coinve_ablation_v1"
MODELS = ("single4k_step677", "coinve4k_step677", "mixed2k2k_step677")
PAIRS = ((MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2]))
BASE_PAIRS = (("base", MODELS[1]), ("base", MODELS[2]))
STAGES = {
    "short_subject": ("short_subject_manifest.csv", "official_id"),
    "long_subject": ("manifest.csv", "generation_id"),
    "attribute_all": ("manifest.csv", "generation_id"),
}


def sha256_file(path: Path) -> str:
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


def generated_path(artifact: Path, stage: str, model: str, output_id: int) -> Path:
    if model == "single4k_step677":
        pattern = f"{stage}_lora_step677_shard*/{output_id:06d}.mp4"
    else:
        pattern = f"{stage}_{model}_shard*/{output_id:06d}.mp4"
    matches = list((artifact / "outputs_raw").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {model}/{output_id} output, got {matches}")
    return matches[0].resolve()


def encode(destination: Path, frames: list[np.ndarray], size: tuple[int, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = imageio_ffmpeg.write_frames(
        str(temporary), size, fps=5, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart", "-threads", "2"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)))
    finally:
        writer.close()
    temporary.replace(destination)


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if not len(reader):
        raise ValueError(f"empty video: {path}")
    for start in range(0, len(reader), 16):
        reader.get_batch(list(range(start, min(start + 16, len(reader)))))
    first = reader[0]
    return {
        "path": str(path.resolve()), "frames": len(reader), "fps": round(float(reader.get_avg_fps()), 6),
        "width": int(first.shape[1]), "height": int(first.shape[0]), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_stage(stage: str, manifest_name: str, id_column: str, allowed_ids: set[str] | None) -> list[dict]:
    artifact = ROOT / "artifacts" / stage
    with (artifact / manifest_name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if allowed_ids is not None:
        rows = [row for row in rows if row["sample_id"] in allowed_ids]
    old_prepared = {row["sample_id"]: row for row in json.loads((artifact / "prepared_media.json").read_text())}
    media_root = artifact / "judge_media_coinve_ablation_v1"

    def one(row: dict[str, str]) -> dict:
        sample_id = row["sample_id"]
        output_id = int(row[id_column])
        old = old_prepared[sample_id]
        positions = old["judge_positions"]
        old_media = old["judge_media"]
        expected_frames = int(row["lance_sampled_frames"])
        result_media = {
            "source": old_media["source"],
            "base": old_media["base"],
            "single4k_step677": old_media["lora_step677"],
        }
        reference_size = (int(old_media["source"]["width"]), int(old_media["source"]["height"]))
        for model in MODELS[1:]:
            source = generated_path(artifact, stage, model, output_id)
            reader = VideoReader(str(source), ctx=cpu(0), num_threads=2)
            if len(reader) != expected_frames or abs(float(reader.get_avg_fps()) - 12.0) > 0.01:
                raise ValueError(f"unaligned generated output: {source}")
            destination = media_root / f"{sample_id}_{model}.mp4"
            if not destination.exists():
                encode(destination, [reader[position].asnumpy() for position in positions], reference_size)
            info = probe(destination)
            if (info["frames"], info["fps"], info["width"], info["height"]) != (
                old_media["source"]["frames"], old_media["source"]["fps"],
                old_media["source"]["width"], old_media["source"]["height"],
            ):
                raise ValueError(f"judge-media alignment failed: {destination}")
            result_media[model] = info
        return {
            "stage": stage, "sample_id": sample_id, "official_id": int(row["official_id"]),
            "subcategory": row["subcategory"], "instruction": row["edit_prompt"],
            "judge_media": result_media,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(one, rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("1", "all"), default="all")
    parser.add_argument("--media-only-stage", choices=tuple(STAGES))
    args = parser.parse_args()
    if args.media_only_stage:
        manifest, id_column = STAGES[args.media_only_stage]
        prepared = prepare_stage(args.media_only_stage, manifest, id_column, None)
        path = OUTPUT / f"prepared_pairwise_media_{args.media_only_stage}.json"
        write_json(path, prepared)
        print(json.dumps({"media_only_stage": args.media_only_stage, "samples": len(prepared)}, indent=2))
        return
    phase_manifest = json.loads((OUTPUT / "phase_manifest.json").read_text())["stages"]
    phase1_ids = {sample_id for key, value in phase_manifest.items() if key.endswith("phase1") for sample_id in value["sample_ids"]}
    allowed_ids = phase1_ids if args.phase == "1" else None
    prepared = []
    for stage, (manifest, id_column) in STAGES.items():
        prepared.extend(prepare_stage(stage, manifest, id_column, allowed_ids))
    prepared.sort(key=lambda row: (row["sample_id"] not in phase1_ids, list(STAGES).index(row["stage"]), row["official_id"]))
    tasks = []
    base_tasks = []
    for row in prepared:
        common = {
            "kind": "pairwise", "stage": row["stage"], "sample_id": row["sample_id"],
            "official_id": row["official_id"], "subcategory": row["subcategory"],
            "instruction": row["instruction"], "source_video": row["judge_media"]["source"]["path"],
            "source_sha256": row["judge_media"]["source"]["sha256"],
        }
        for left, right in PAIRS:
            for order, (candidate_a, candidate_b) in enumerate(((left, right), (right, left)), 1):
                tasks.append({
                    **common, "model_pair": "__vs__".join((left, right)), "order": order,
                    "candidate_a_model": candidate_a, "candidate_b_model": candidate_b,
                    "candidate_a_video": row["judge_media"][candidate_a]["path"],
                    "candidate_b_video": row["judge_media"][candidate_b]["path"],
                    "candidate_a_sha256": row["judge_media"][candidate_a]["sha256"],
                    "candidate_b_sha256": row["judge_media"][candidate_b]["sha256"],
                })
        for left, right in BASE_PAIRS:
            for order, (candidate_a, candidate_b) in enumerate(((left, right), (right, left)), 1):
                base_tasks.append({
                    **common, "model_pair": "__vs__".join((left, right)), "order": order,
                    "candidate_a_model": candidate_a, "candidate_b_model": candidate_b,
                    "candidate_a_video": row["judge_media"][candidate_a]["path"],
                    "candidate_b_video": row["judge_media"][candidate_b]["path"],
                    "candidate_a_sha256": row["judge_media"][candidate_a]["sha256"],
                    "candidate_b_sha256": row["judge_media"][candidate_b]["sha256"],
                })
    expected_samples = 114 if args.phase == "1" else 228
    expected_tasks = expected_samples * len(PAIRS) * 2
    expected_base_tasks = expected_samples * len(BASE_PAIRS) * 2
    phase1_tasks = 114 * len(PAIRS) * 2
    phase1_base_tasks = 114 * len(BASE_PAIRS) * 2
    if (
        len(prepared) != expected_samples
        or len(tasks) != expected_tasks
        or len(base_tasks) != expected_base_tasks
        or sum(row["sample_id"] in phase1_ids for row in prepared) != 114
    ):
        raise RuntimeError("unexpected IVEBench ablation task counts")
    suffix = "_phase1" if args.phase == "1" else ""
    prepared_path = OUTPUT / f"prepared_pairwise_media{suffix}.json"
    tasks_path = OUTPUT / f"judge_tasks_pairwise{suffix}.json"
    base_tasks_path = OUTPUT / f"judge_tasks_pairwise_vs_base{suffix}.json"
    summary_path = OUTPUT / f"pairwise_preparation{suffix}.json"
    base_summary_path = OUTPUT / f"pairwise_vs_base_preparation{suffix}.json"
    write_json(prepared_path, prepared)
    write_json(tasks_path, tasks)
    write_json(base_tasks_path, base_tasks)
    write_json(summary_path, {
        "status": "complete", "scope": "phase1" if args.phase == "1" else "full", "samples": expected_samples, "models": MODELS,
        "unordered_pairs": [list(pair) for pair in PAIRS], "orders_per_pair": 2,
        "phase1_samples": 114, "phase1_tasks": phase1_tasks, "full_tasks": len(tasks),
        "blinding": "API payload contains anonymous Candidate A/B labels; model identity is omitted from the prompt",
        "tasks_sha256": sha256_file(tasks_path),
    })
    write_json(base_summary_path, {
        "status": "complete", "scope": "phase1" if args.phase == "1" else "full", "samples": expected_samples,
        "models": ("base", MODELS[1], MODELS[2]),
        "unordered_pairs": [list(pair) for pair in BASE_PAIRS], "orders_per_pair": 2,
        "phase1_samples": 114, "phase1_tasks": phase1_base_tasks, "full_tasks": len(base_tasks),
        "blinding": "API payload contains anonymous Candidate A/B labels; model identity is omitted from the prompt",
        "tasks_sha256": sha256_file(base_tasks_path),
    })
    print(json.dumps({
        "scope": args.phase, "samples": len(prepared), "phase1_tasks": phase1_tasks,
        "tasks": len(tasks), "phase1_vs_base_tasks": phase1_base_tasks,
        "vs_base_tasks": len(base_tasks),
    }, indent=2))


if __name__ == "__main__":
    main()
