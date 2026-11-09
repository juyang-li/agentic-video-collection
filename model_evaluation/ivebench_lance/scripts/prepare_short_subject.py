#!/usr/bin/env python3
"""Download, verify, and freeze the IVEBench Short subject-editing generation jobs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
SOURCE = DATA / "short"
ARTIFACTS = ROOT / "artifacts" / "short_subject"
PROMPTS = DATA / "ivebenchdb_prompt_short.json"
TREE = DATA / "dataset_tree.json"

DATASET_ID = "Coraxor/IVEBench"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"
UPSTREAM_REVISION = "2dcf8900ee65197f43854c76f61c66a32bb8dbfc"
EXPECTED_COUNTS = {
    "add new subject": 28,
    "remove existing subject": 33,
    "replace existing subject": 30,
}

PROJECT = ROOT.parents[1]
FINETUNE = PROJECT / "lance_general_edit_finetune" / "3_lance_lora_finetuning"
RUNTIME = FINETUNE / "runtime" / "lance"
MODEL = RUNTIME / "downloads" / "Lance_3B_Video"
ADAPTER = FINETUNE / "runs" / "lance_4k_r64_seed42_e2_v1" / "checkpoints" / "0000677"
TOOLS = PROJECT / "model_evaluation" / "lance_tools"


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


def selected_rows() -> list[dict]:
    rows = [row for row in json.loads(PROMPTS.read_text()) if row["category"] == "subject_editing"]
    rows.sort(key=lambda row: int(row["id"]))
    counts = Counter(row["subcategory"] for row in rows)
    if len(rows) != 91 or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected Short subject scope: count={len(rows)} subcategories={dict(counts)}")
    if len({row["id"] for row in rows}) != len(rows) or len({row["src_video_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate Short subject id or video name")
    return rows


def tree_entries() -> dict[str, dict]:
    entries = {}
    for entry in json.loads(TREE.read_text()):
        if entry.get("type") == "file" and entry["path"].startswith("ivebenchdb_mp4_short/"):
            entries[Path(entry["path"]).name] = entry
    return entries


def validate_download(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["size"]):
        raise ValueError(f"size mismatch for {path}: {path.stat().st_size} != {entry['size']}")
    expected = entry["lfs"]["oid"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {path}: {actual} != {expected}")


def download_one(row: dict, entry: dict) -> str:
    SOURCE.mkdir(parents=True, exist_ok=True)
    destination = SOURCE / row["src_video_name"]
    if destination.exists():
        validate_download(destination, entry)
        return f"reused {destination.name}"

    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{entry['path']}"
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as handle:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                handle.write(block)
            if partial.stat().st_size != int(entry["size"]):
                raise ValueError(f"incomplete size {partial.stat().st_size}/{entry['size']}")
            if sha256_file(partial) != entry["lfs"]["oid"]:
                raise ValueError("downloaded sha256 does not match official LFS oid")
            partial.replace(destination)
            return f"downloaded {destination.name}"
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def probe_sources(rows: list[dict], entries: dict[str, dict]) -> list[dict]:
    from decord import VideoReader, cpu
    import numpy as np

    manifest = []
    for row in rows:
        path = SOURCE / row["src_video_name"]
        validate_download(path, entries[row["src_video_name"]])
        reader = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(reader)
        if not 32 <= total_frames <= 128:
            raise ValueError(f"{path}: expected official Short frame count, got {total_frames}")
        fps = float(reader.get_avg_fps())
        rounded_fps = int(round(fps)) if fps > 0 else 24
        duration = min(total_frames / rounded_fps, 6.0)
        sampled_frames = int(round(duration * 12))
        if sampled_frames % 4:
            sampled_frames = sampled_frames // 4 * 4 + 1
        else:
            sampled_frames = sampled_frames // 4 * 4 + 1 - 4
        indices = np.linspace(0, total_frames - 1, sampled_frames, dtype=int).tolist()
        first = reader[0]
        manifest.append({
            "sample_id": Path(row["src_video_name"]).stem,
            "official_id": int(row["id"]),
            "subset": "short",
            "category": row["category"],
            "subcategory": row["subcategory"],
            "source_prompt": row["source_prompt"],
            "edit_prompt": row["edit_prompt"],
            "target_prompt": row["target_prompt"],
            "target_phrase": row["target_phrase"],
            "target_span": row["target_span"],
            "source_path": str(path.resolve()),
            "source_bytes": path.stat().st_size,
            "source_sha256": entries[row["src_video_name"]]["lfs"]["oid"],
            "source_frames": total_frames,
            "source_fps": round(fps, 6),
            "source_height": int(first.shape[0]),
            "source_width": int(first.shape[1]),
            "lance_sampled_frames": sampled_frames,
            "lance_source_indices": json.dumps(indices, separators=(",", ":")),
        })
    return manifest


def write_manifest(rows: list[dict]) -> None:
    path = ARTIFACTS / "short_subject_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def config_item(row: dict) -> dict:
    return {
        "element_dtype_array": ["text", "video"],
        "interleave_array": [row["edit_prompt"], row["source_path"]],
        "istarget_in_interleave": [0, 0],
    }


def job(model_kind: str, slot: int, config_path: Path, run_name: str, sample_count: int) -> dict:
    if model_kind == "base":
        command = [
            "bash", str((TOOLS / "run_video_edit.sh").resolve()), str(ARTIFACTS.resolve()),
            str(RUNTIME.resolve()), str(MODEL.resolve()), str(config_path.resolve()), run_name,
        ]
    elif model_kind == "lora_step677":
        command = [
            "bash", str((FINETUNE / "scripts" / "run_lora_video_edit.sh").resolve()),
            str(ARTIFACTS.resolve()), str(RUNTIME.resolve()), str(MODEL.resolve()),
            str(ADAPTER.resolve()), str(config_path.resolve()), run_name,
        ]
    else:
        raise ValueError(model_kind)
    return {
        "model_kind": model_kind,
        "gpu_slot": slot,
        "config_path": str(config_path.resolve()),
        "run_name": run_name,
        "sample_count": sample_count,
        "command": command,
    }


def write_jobs(manifest: list[dict]) -> None:
    required = [
        RUNTIME / "inference_lance.sh", MODEL / "model.safetensors",
        ADAPTER / "adapter.safetensors", ADAPTER / "checkpoint_complete.json",
    ]
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)
    marker = json.loads((ADAPTER / "checkpoint_complete.json").read_text())
    if marker.get("status") != "complete" or marker.get("completed_steps") != 677:
        raise ValueError("step 677 checkpoint is not complete")

    configs = ARTIFACTS / "configs"
    groups = []
    smoke_jobs = []
    first = manifest[0]
    for slot, kind in enumerate(("base", "lora_step677")):
        name = f"smoke_{kind}"
        path = configs / f"{name}.json"
        write_json(path, {f"{first['official_id']:04d}": config_item(first)})
        smoke_jobs.append(job(kind, slot, path, name, 1))
    groups.append({"name": "matched_smoke", "jobs": smoke_jobs})

    for kind in ("base", "lora_step677"):
        model_jobs = []
        for slot in range(4):
            shard = manifest[slot::4]
            name = f"short_subject_{kind}_shard{slot:02d}"
            path = configs / f"{name}.json"
            write_json(path, {f"{row['official_id']:04d}": config_item(row) for row in shard})
            model_jobs.append(job(kind, slot, path, name, len(shard)))
        groups.append({"name": f"full_{kind}", "jobs": model_jobs})

    specification = {
        "benchmark": "IVEBench",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "upstream_code_revision": UPSTREAM_REVISION,
        "scope": "short_subject_all_91",
        "generation_seed": 42,
        "inference": {
            "resolution": "video_480p", "save_fps": 12, "steps": 30,
            "timestep_shift": 3.5, "cfg_text_scale": 4.0,
            "max_num_frames": 121, "max_duration_seconds": 6.0,
            "prompt_enhancement": False, "dtype": "bf16",
        },
        "base_model_path": str(MODEL.resolve()),
        "base_model_sha256": "7f0550e1d1511b29a4740a67c1e18e176302a4ecb3177c8a5850ff5fe6447c25",
        "adapter_path": str(ADAPTER.resolve()),
        "adapter_sha256": sha256_file(ADAPTER / "adapter.safetensors"),
        "manifest_sha256": sha256_file(ARTIFACTS / "short_subject_manifest.csv"),
        "groups": groups,
    }
    write_json(ARTIFACTS / "generation_jobs.json", specification)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    for path in (PROMPTS, TREE):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = selected_rows()
    entries = tree_entries()
    missing_entries = [row["src_video_name"] for row in rows if row["src_video_name"] not in entries]
    if missing_entries:
        raise ValueError(f"missing official tree entries: {missing_entries}")

    if args.download:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_one, row, entries[row["src_video_name"]]): row for row in rows}
            for future in as_completed(futures):
                print(future.result(), flush=True)

    manifest = probe_sources(rows, entries)
    write_manifest(manifest)
    write_jobs(manifest)
    preparation = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(manifest),
        "subcategory_counts": dict(Counter(row["subcategory"] for row in manifest)),
        "source_bytes": sum(row["source_bytes"] for row in manifest),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "upstream_code_revision": UPSTREAM_REVISION,
        "manifest_sha256": sha256_file(ARTIFACTS / "short_subject_manifest.csv"),
        "generation_jobs_sha256": sha256_file(ARTIFACTS / "generation_jobs.json"),
    }
    write_json(ARTIFACTS / "preparation.json", preparation)
    print(json.dumps(preparation, indent=2), flush=True)


if __name__ == "__main__":
    main()
