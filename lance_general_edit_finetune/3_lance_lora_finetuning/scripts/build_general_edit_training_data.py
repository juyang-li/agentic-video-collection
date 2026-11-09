#!/usr/bin/env python3
"""Freeze the 4K general-edit train split and its two-epoch seed-42 schedule."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import decord
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA = "lance_general_edit_single_turn_v1"
MANIFEST_SCHEMA = "datamake_lance_chain_dataset_manifest_v1"
SEED = 42
EPOCHS = 2
WORLD_SIZE = 4
ACCUMULATION = 2
CROP_SECONDS = 5.0
SAMPLE_FPS = 12
EXPECTED_TRAIN = 3601
EXPECTED_SPLITS = {"train": 3601, "validation": 195, "test": 204}
EXPECTED_BRANCHES = {
    "goku_remove": 548,
    "goku_add": 364,
    "egoedit_replace": 890,
    "reco_add": 543,
    "reco_remove": 537,
    "reco_replace": 719,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_unit(*parts: object) -> float:
    return stable_int(*parts) / float(2**64)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def probe_video(path: Path) -> dict[str, Any]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    frames = len(reader)
    height, width = map(int, reader[0].shape[:2])
    duration = frames / fps
    if fps <= 0 or frames <= 0 or duration < CROP_SECONDS:
        raise ValueError(
            f"invalid/short video: {path} fps={fps} frames={frames} duration={duration}"
        )
    return {
        "path": str(path),
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
    }


def epoch_groups(sample_ids: list[str], epoch_index: int) -> list[list[tuple[str, bool]]]:
    """Mirror the trainer: shuffle, rank-pad, shuffle groups, accumulation-pad."""

    rng = random.Random(SEED + 1_000_003 * epoch_index)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)
    groups: list[list[tuple[str, bool]]] = []
    full = len(shuffled) // WORLD_SIZE * WORLD_SIZE
    for start in range(0, full, WORLD_SIZE):
        groups.append([(sample_id, True) for sample_id in shuffled[start : start + WORLD_SIZE]])
    leftover = shuffled[full:]
    if leftover:
        group = [(sample_id, True) for sample_id in leftover]
        group.extend([(leftover[-1], False)] * (WORLD_SIZE - len(group)))
        groups.append(group)
    rng.shuffle(groups)
    padding_source = groups[-1][0][0]
    while len(groups) % ACCUMULATION:
        groups.append([(padding_source, False)] * WORLD_SIZE)
    return groups


def byte_shards(rows: list[dict[str, Any]], output: Path, rows_per_shard: int) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for shard_index, start in enumerate(range(0, len(rows), rows_per_shard)):
        values = []
        for row in rows[start : start + rows_per_shard]:
            values.append(
                {
                    "sample_id": row["sample_id"],
                    "caption": row["caption"],
                    "input_video_bytes": Path(row["input_video_path"]).read_bytes(),
                    "output_video_bytes": Path(row["output_video_path"]).read_bytes(),
                }
            )
        path = output / f"train-{shard_index:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path, compression="NONE", row_group_size=8)
        result.append(
            {
                "path": str(path.resolve()),
                "rows": len(values),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    stage_root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=stage_root / "artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=stage_root / "artifacts/lance_4k_r64_seed42_e2_v1/data",
    )
    parser.add_argument("--probe-workers", type=int, default=32)
    parser.add_argument("--embed-video-bytes", action="store_true")
    parser.add_argument("--rows-per-byte-shard", type=int, default=64)
    parser.add_argument(
        "--expected-branches-json",
        default=json.dumps(EXPECTED_BRANCHES, sort_keys=True),
        help="Exact train-split branch counts as JSON; defaults to the original Single-4K contract.",
    )
    parser.add_argument(
        "--relative-media-root",
        type=Path,
        default=stage_root.parent / "1_specialist_filtering",
        help="Root used only for relative source_local/target_local paths.",
    )
    parser.add_argument("--dataset-name", default="general_edit_4k_train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = args.pool.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    relative_media_root = args.relative_media_root.resolve()
    expected_branches = json.loads(args.expected_branches_json)

    with pool.open("r", encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    split_counts = Counter(row["split"] for row in all_rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"frozen split counts changed: {dict(split_counts)}")
    rows = [row for row in all_rows if row["split"] == "train"]
    if len(rows) != EXPECTED_TRAIN:
        raise ValueError(f"expected {EXPECTED_TRAIN} train rows, got {len(rows)}")
    branch_counts = Counter(row["branch"] for row in rows)
    if dict(branch_counts) != expected_branches:
        raise ValueError(f"frozen branch counts changed: {dict(branch_counts)}")
    if len({row["sample_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("sample_id is not globally unique")
    split_media: dict[str, set[str]] = {}
    for row in all_rows:
        media = split_media.setdefault(row["split"], set())
        media.update((row["source_local"], row["target_local"]))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_media[left] & split_media[right]
        if overlap:
            raise ValueError(f"media leakage between {left}/{right}: {sorted(overlap)[:5]}")

    resolved: list[tuple[dict[str, str], Path, Path]] = []
    media_paths: set[Path] = set()
    for row in rows:
        source_value = Path(row["source_local"])
        target_value = Path(row["target_local"])
        source = (source_value if source_value.is_absolute() else relative_media_root / source_value).resolve()
        target = (target_value if target_value.is_absolute() else relative_media_root / target_value).resolve()
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"missing pair for {row['sample_id']}: {source}, {target}")
        resolved.append((row, source, target))
        media_paths.update((source, target))

    cache_path = output / "video_probe.json"
    if cache_path.is_file():
        probe = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        with ThreadPoolExecutor(max_workers=args.probe_workers) as executor:
            values = executor.map(probe_video, sorted(media_paths))
            probe = {value["path"]: value for value in values}
        write_json(cache_path, probe)
    if set(probe) != {str(path) for path in media_paths}:
        raise ValueError("video probe cache does not match the frozen train media set")

    parquet_rows: list[dict[str, Any]] = []
    crop_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row, source, target in resolved:
        source_probe = probe[str(source)]
        target_probe = probe[str(target)]
        max_start = min(source_probe["duration"], target_probe["duration"]) - CROP_SECONDS
        if max_start < -1e-6:
            raise ValueError(f"pair shorter than {CROP_SECONDS}s: {row['sample_id']}")
        crop_starts = []
        for epoch in range(1, EPOCHS + 1):
            start = stable_unit(SEED, epoch, row["sample_id"], "crop") * max(0.0, max_start)
            start = round(start, 9)
            source_start = int(math.floor(start * source_probe["fps"] + 1e-9))
            target_start = int(math.floor(start * target_probe["fps"] + 1e-9))
            crop = {
                "start_seconds": start,
                "duration_seconds": CROP_SECONDS,
                "source_start_frame": source_start,
                "target_start_frame": target_start,
            }
            crop_starts.append(crop)
            crop_map[(row["sample_id"], epoch)] = crop
        parquet_rows.append(
            {
                "schema_version": SCHEMA,
                "sample_id": row["sample_id"],
                "chain_id": row["sample_id"],
                "source_longest_prefix_sample_id": row["sample_id"],
                "round_count": 1,
                "prompts": [row["instruction"]],
                "caption": row["instruction"],
                "video_paths": [str(source), str(target)],
                "input_video_path": str(source),
                "output_video_path": str(target),
                "turn_ids": [row["sample_id"]],
                "edit_axes": [row["branch"]],
                "branch": row["branch"],
                "dataset": row["dataset"],
                "operation": row["operation"],
                "split": "train",
                "crop_starts": crop_starts,
            }
        )

    index_path = output / "train_index.parquet"
    pq.write_table(pa.Table.from_pylist(parquet_rows), index_path, compression="zstd", row_group_size=128)

    sample_by_id = {row["sample_id"]: row for row in parquet_rows}
    schedule_path = output / "training_schedule.csv"
    schedule_fields = [
        "epoch", "global_step", "rank", "slot", "contributes_loss", "sample_id",
        "branch", "crop_start_seconds", "source_start_frame", "target_start_frame",
        "sample_seed", "timestep_seed", "noise_seed",
    ]
    global_step = 0
    real_per_epoch: Counter[int] = Counter()
    with schedule_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=schedule_fields)
        writer.writeheader()
        for epoch_index in range(EPOCHS):
            groups = epoch_groups(list(sample_by_id), epoch_index)
            for group_start in range(0, len(groups), ACCUMULATION):
                global_step += 1
                window = groups[group_start : group_start + ACCUMULATION]
                for slot, group in enumerate(window):
                    for rank, (sample_id, contributes) in enumerate(group):
                        crop = crop_map[(sample_id, epoch_index + 1)]
                        writer.writerow(
                            {
                                "epoch": epoch_index + 1,
                                "global_step": global_step,
                                "rank": rank,
                                "slot": slot,
                                "contributes_loss": int(contributes),
                                "sample_id": sample_id,
                                "branch": sample_by_id[sample_id]["branch"],
                                "crop_start_seconds": f"{crop['start_seconds']:.9f}",
                                "source_start_frame": crop["source_start_frame"],
                                "target_start_frame": crop["target_start_frame"],
                                "sample_seed": stable_int(SEED, epoch_index + 1, sample_id, "sample") % (2**31 - 1),
                                "timestep_seed": stable_int(SEED, epoch_index + 1, sample_id, "timestep") % (2**31 - 1),
                                "noise_seed": stable_int(SEED, epoch_index + 1, sample_id, "noise") % (2**31 - 1),
                            }
                        )
                        if contributes:
                            real_per_epoch[epoch_index + 1] += 1
    if real_per_epoch != Counter({1: EXPECTED_TRAIN, 2: EXPECTED_TRAIN}):
        raise RuntimeError(f"schedule exposure mismatch: {real_per_epoch}")

    yaml_path = output / "lance_dataset.yaml"
    dataset_yaml = {
        "D_general_edit": {
            "datamake_multiturn_v2v_local": {
                "is_mandatory": True,
                "data_type": "video",
                "dataset_names": {args.dataset_name: {"data_dir": [str(index_path)]}},
                "dataset_type": "datamake_multiturn_video2video",
                "datamake_record_level": "chain",
                "datamake_history_mode": "online_rollout_full",
                "datamake_chain_manifest": str(output / "manifest.json"),
                "media_root": "/",
                "is_image": False,
                "task_type": "tv2v",
                "sample_task": "edit",
                "target_modality": "video",
                "vision_cond_type": ["vit", "vae_Nsplit"],
                "text_template": True,
                "raw_bytes_input": False,
                "video_frame_sampler_args": {
                    "type": "multi_clips",
                    "params": {
                        "temporal": 4,
                        "sample_fps": SAMPLE_FPS,
                        "truncate": False,
                        "max_duration": CROP_SECONDS,
                        "length_type": "kn+1",
                        "assert_seconds": True,
                    },
                },
                "video_transform_args": {
                    "resolution": 480,
                    "mode": "bucket",
                    "divisible_crop_size": 16,
                    "stride_spatial": 16,
                    "stride_temporal": 4,
                    "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    "mean": 0.5,
                    "std": 0.5,
                },
                "vit_video_transform_args": {
                    "resolution": 448,
                    "mode": "bucket",
                    "divisible_crop_size": 28,
                    "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711],
                },
            }
        }
    }
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    embedded = []
    if args.embed_video_bytes:
        embedded = byte_shards(parquet_rows, output / "lance_bytes", args.rows_per_byte_shard)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_schema_version": SCHEMA,
        "split": "train",
        "history_mode": "online_rollout_full",
        "record_level": "chain",
        "parquet_file": str(index_path),
        "parquet_sha256": sha256_file(index_path),
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "source_pool": str(pool),
        "source_pool_sha256": sha256_file(pool),
        "chain_count": EXPECTED_TRAIN,
        "target_video_count": EXPECTED_TRAIN,
        "chain_length_count": {"1": EXPECTED_TRAIN},
        "branch_counts": dict(sorted(branch_counts.items())),
        "split_counts": EXPECTED_SPLITS,
        "seed": SEED,
        "epochs": EPOCHS,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": ACCUMULATION,
        "optimizer_steps_per_epoch": global_step // EPOCHS,
        "total_optimizer_steps": global_step,
        "real_sample_exposures": EXPECTED_TRAIN * EPOCHS,
        "crop_seconds": CROP_SECONDS,
        "sample_fps": SAMPLE_FPS,
        "schedule": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "embedded_byte_shards": embedded,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
