#!/usr/bin/env python3
"""Select, fetch, technically validate, and freeze the CoinVE ablation pools."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
import time
from typing import Any

import decord


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_ROOT.parents[1]
COINVE_ROOT = PROJECT_ROOT / "dataset_research/coinve_research"
METADATA = COINVE_ROOT / "metadata/metadata_coinve200k.jsonl"
OUTPUT = STAGE_ROOT / "artifacts/coinve_ablation_v1"
SINGLE_POOL = STAGE_ROOT / "artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv"
SINGLE_MEDIA_ROOT = STAGE_ROOT.parent / "1_specialist_filtering"
REPO_ID = "FireCRT/CoinVE-200K"
REVISION = "7989adca1088928414242c57dd62ce485f740889"
METADATA_SHA256 = "d1f0b643d2c5cc1cfe855351d2642fb1b6c0cb403f5301002111ac8c2246a5a6"
SELECTION_SEED = 20260904
TRAIN_SEED = 42
CROP_SECONDS = 5.0
PROBE_POLICY = "coinve_pair_full_decode_common5s_v1"
ANCHOR_QUOTAS = {"add": 1000, "remove": 1200, "replace": 1800}
CANDIDATE_FACTOR = 1.5
PURE_SPLITS = {
    "add": {"train": 900, "validation": 49, "test": 51},
    "remove": {"train": 1080, "validation": 59, "test": 61},
    "replace": {"train": 1621, "validation": 87, "test": 92},
}
MIXED_COIN_SPLITS = {
    "add": {"train": 450, "validation": 24, "test": 26},
    "remove": {"train": 540, "validation": 30, "test": 30},
    "replace": {"train": 811, "validation": 43, "test": 46},
}
MIXED_SINGLE_SPLITS = {
    "goku_remove": {"train": 274, "validation": 14, "test": 12},
    "goku_add": {"train": 182, "validation": 8, "test": 10},
    "egoedit_replace": {"train": 445, "validation": 25, "test": 30},
    "reco_add": {"train": 272, "validation": 14, "test": 14},
    "reco_remove": {"train": 269, "validation": 15, "test": 16},
    "reco_replace": {"train": 358, "validation": 22, "test": 20},
}
FRAME_RANGE = re.compile(r"_(\d+)to(\d+)(?:_[0-9a-f]+)?\.mp4$")
PRINT_LOCK = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def shard_and_member(dataset_path: str) -> tuple[str, str]:
    parts = Path(dataset_path).parts
    if len(parts) < 3:
        raise ValueError(f"unexpected CoinVE path: {dataset_path}")
    return str(Path(parts[0]) / f"{parts[1]}.tar"), parts[-1]


def first_nonstyle(operations: list[object]) -> str | None:
    for operation in operations:
        value = str(operation).strip().lower()
        if "styl" not in value:
            return value
    return None


def proxy_long_enough(source_path: str) -> tuple[bool, int]:
    match = FRAME_RANGE.search(source_path)
    if not match:
        return False, 0
    span = int(match.group(2)) - int(match.group(1)) + 1
    return span >= 151, span


def select_candidates() -> None:
    if sha256_file(METADATA) != METADATA_SHA256:
        raise ValueError("CoinVE metadata differs from the frozen official revision")
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with METADATA.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            anchor = first_nonstyle(item["instruction_operation"])
            long_enough, proxy_frames = proxy_long_enough(item["source_video_path"])
            if anchor not in ANCHOR_QUOTAS or not long_enough:
                continue
            by_anchor[anchor].append({"metadata_index": index, "item": item, "proxy_frames": proxy_frames})
    for anchor in by_anchor:
        by_anchor[anchor].sort(key=lambda row: stable_key(SELECTION_SEED, anchor, row["metadata_index"]))

    rows: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for anchor in ("add", "remove", "replace"):
        wanted = math.ceil(ANCHOR_QUOTAS[anchor] * CANDIDATE_FACTOR)
        chosen = []
        for value in by_anchor[anchor]:
            item = value["item"]
            source_path = str(item["source_video_path"])
            if source_path in used_sources:
                continue
            used_sources.add(source_path)
            chosen.append(value)
            if len(chosen) == wanted:
                break
        if len(chosen) != wanted:
            raise RuntimeError(f"not enough unique-source {anchor} candidates")
        for anchor_rank, value in enumerate(chosen, 1):
            index, item = value["metadata_index"], value["item"]
            instructions = list(item["instruction"])
            operations = list(item["instruction_operation"])
            objects = list(item["instruction_object"])
            if not (len(instructions) == len(operations) == len(objects)):
                raise ValueError(f"instruction arrays differ at metadata row {index}")
            sample_id = f"coinve-{index:06d}"
            source_shard, source_member = shard_and_member(str(item["source_video_path"]))
            target_shard, target_member = shard_and_member(str(item["edited_video_path"]))
            rows.append({
                "selection_version": "coinve_ablation_v1",
                "sample_id": sample_id,
                "metadata_index": index,
                "anchor_operation": anchor,
                "anchor_rank": anchor_rank,
                "instruction_count": len(instructions),
                "has_stylize": int(any("styl" in str(op).lower() for op in operations)),
                "instruction": "\n".join(f"{number}. {text}" for number, text in enumerate(instructions, 1)),
                "instructions_json": json.dumps(instructions, ensure_ascii=False),
                "operations_json": json.dumps(operations, ensure_ascii=False),
                "objects_json": json.dumps(objects, ensure_ascii=False),
                "proxy_source_frames": value["proxy_frames"],
                "repo_id": REPO_ID,
                "revision": REVISION,
                "source_dataset_path": item["source_video_path"],
                "target_dataset_path": item["edited_video_path"],
                "source_shard": source_shard,
                "target_shard": target_shard,
                "source_member": source_member,
                "target_member": target_member,
                "source_local": str((OUTPUT / "media" / f"{sample_id}_source.mp4").resolve()),
                "target_local": str((OUTPUT / "media" / f"{sample_id}_target.mp4").resolve()),
                "selection_seed": SELECTION_SEED,
            })
    rows.sort(key=lambda row: (list(ANCHOR_QUOTAS).index(row["anchor_operation"]), int(row["anchor_rank"])))
    manifest = OUTPUT / "candidate_manifest.csv"
    if manifest.exists():
        old = read_csv(manifest)
        if [(r["sample_id"], r["anchor_rank"]) for r in old] != [(r["sample_id"], str(r["anchor_rank"])) for r in rows]:
            raise ValueError("refuse to alter the frozen CoinVE candidate order")
    else:
        write_csv(manifest, rows)
    write_json(OUTPUT / "selection_summary.json", {
        "status": "complete", "repo_id": REPO_ID, "revision": REVISION,
        "metadata": str(METADATA), "metadata_sha256": METADATA_SHA256,
        "selection_seed": SELECTION_SEED, "candidate_factor": CANDIDATE_FACTOR,
        "anchor_definition": "first non-Stylize operation in original instruction order",
        "prompt_policy": "preserve all ordered instructions as one numbered caption",
        "technical_prefilter": "source filename frame span >=151",
        "unique_source_video": True,
        "candidate_counts": dict(Counter(row["anchor_operation"] for row in rows)),
        "instruction_counts": dict(Counter(str(row["instruction_count"]) for row in rows)),
        "stylize_candidates": sum(int(row["has_stylize"]) for row in rows),
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest),
    })
    print(f"selected candidate rows={len(rows)} manifest={manifest}", flush=True)


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def download_candidates(factor: float, workers: int) -> None:
    # Reuse the audited HTTP Range implementation, but locate each shard's median
    # target first so later binary searches run within progressively smaller bounds.
    import sys
    sys.path.insert(0, str(COINVE_ROOT))
    from download_random_pairs import (  # type: ignore
        BLOCK_SIZE, FINAL_SCAN_SIZE, MAX_PROBE_SIZE, PROBE_SIZE,
        RemoteTar, WantedMember, scan_regular_headers,
    )

    rows = read_csv(OUTPUT / "candidate_manifest.csv")
    selected = [row for row in rows if int(row["anchor_rank"]) <= math.ceil(ANCHOR_QUOTAS[row["anchor_operation"]] * factor)]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in selected:
        for role in ("source", "target"):
            grouped[row[f"{role}_shard"]].append(WantedMember(
                member=row[f"{role}_member"], destination=Path(row[f"{role}_local"]),
                sample_id=row["sample_id"], role=role,
            ))

    def extract(shard: str, wanted: list[Any]) -> dict[str, Any]:
        pending = [item for item in wanted if not item.destination.is_file()]
        if not pending:
            return {"extracted": 0, "preserved": len(wanted), "bytes": 0}
        # Find the lexical median first, then use its exact tar offset to bound
        # both child searches. With tens of selected files per shard, this avoids
        # repeating the upper levels of the same binary search for every file.
        ordered = sorted(pending, key=lambda item: item.member)
        remote = RemoteTar(shard)
        transferred = 0
        completed = 0
        started = time.monotonic()
        log(f"[{shard}] start targets={len(ordered)} preserved={len(wanted)-len(pending)}")

        def locate_bounded(target: str, low: int, high: int):
            nonlocal transferred
            for _ in range(48):
                span = high - low
                if span <= 0:
                    break
                if span <= FINAL_SCAN_SIZE:
                    data = remote.read_range(low, high - 1)
                    transferred += len(data)
                    for entry in scan_regular_headers(data, low):
                        if entry.name == target:
                            return entry
                    break
                midpoint = low + span // 2
                probe_size = PROBE_SIZE
                entries = []
                while probe_size <= MAX_PROBE_SIZE and not entries:
                    start = max(low, midpoint - probe_size // 2)
                    start = start // BLOCK_SIZE * BLOCK_SIZE
                    end = min(high, start + probe_size)
                    end = end // BLOCK_SIZE * BLOCK_SIZE
                    if end <= start:
                        break
                    data = remote.read_range(start, end - 1)
                    transferred += len(data)
                    entries = scan_regular_headers(data, start)
                    probe_size *= 2
                if not entries:
                    raise RuntimeError(f"no tar header near offset {midpoint}")
                for entry in entries:
                    if entry.name == target:
                        return entry
                lower = [entry for entry in entries if entry.name < target]
                upper = [entry for entry in entries if entry.name > target]
                old = (low, high)
                if lower:
                    low = max(low, lower[-1].next_offset)
                if upper:
                    high = min(high, upper[0].offset)
                if (low, high) == old:
                    break
            raise FileNotFoundError(f"member not found in {shard}: {target}")

        def extract_tree(values: list[Any], low: int, high: int) -> None:
            nonlocal completed, transferred
            if not values:
                return
            middle = len(values) // 2
            item = values[middle]
            entry = locate_bounded(item.member, low, high)
            payload = remote.read_range(entry.data_offset, entry.data_offset + entry.size - 1)
            transferred += len(payload)
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            part = item.destination.with_suffix(item.destination.suffix + ".part")
            part.write_bytes(payload)
            part.replace(item.destination)
            completed += 1
            if completed % 25 == 0 or completed == len(ordered):
                log(f"[{shard}] progress={completed}/{len(ordered)} transferred_gb={transferred/1e9:.2f}")
            extract_tree(values[:middle], low, entry.offset)
            extract_tree(values[middle + 1:], entry.next_offset, high)

        try:
            extract_tree(ordered, 0, remote.size)
        finally:
            remote.close()
        log(f"[{shard}] done files={len(ordered)} elapsed_s={time.monotonic()-started:.1f}")
        return {"extracted": len(ordered), "preserved": len(wanted)-len(pending), "bytes": transferred}

    status_path = OUTPUT / "download_status.json"
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as executor:
        futures = {executor.submit(extract, shard, values): shard for shard, values in grouped.items()}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                results[shard] = future.result()
            except Exception as error:
                errors[shard] = type(error).__name__
                log(f"[{shard}] failed error={type(error).__name__}")
            write_json(status_path, {
                "status": "running" if futures else "unknown", "factor": factor,
                "selected_samples": len(selected), "results": results, "errors": errors,
                "complete_role_files": sum(Path(row[f"{role}_local"]).is_file() for row in selected for role in ("source", "target")),
            })
    complete = sum(Path(row[f"{role}_local"]).is_file() for row in selected for role in ("source", "target"))
    write_json(status_path, {
        "status": "complete" if not errors and complete == 2 * len(selected) else "incomplete",
        "factor": factor, "selected_samples": len(selected), "results": results, "errors": errors,
        "complete_role_files": complete,
    })
    if errors or complete != 2 * len(selected):
        raise SystemExit(f"download incomplete: errors={len(errors)} role_files={complete}/{2*len(selected)}")


def decode_video(path: Path) -> dict[str, Any]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    frames = len(reader)
    if frames <= 0:
        raise ValueError("zero frames")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid FPS")
    first = reader[0]
    for start in range(0, frames, 64):
        reader.get_batch(list(range(start, min(start + 64, frames))))
    return {
        "frames": frames, "fps": round(fps, 9), "duration": frames / fps,
        "width": int(first.shape[1]), "height": int(first.shape[0]),
        "bytes": path.stat().st_size,
    }


def probe_candidates(workers: int) -> None:
    rows = read_csv(OUTPUT / "candidate_manifest.csv")
    status_path = OUTPUT / "technical_probe.jsonl"
    known: dict[str, dict[str, Any]] = {}
    if status_path.exists():
        with status_path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if value.get("probe_policy") == PROBE_POLICY:
                    known[value["sample_id"]] = value
    pending = [row for row in rows if row["sample_id"] not in known and Path(row["source_local"]).is_file() and Path(row["target_local"]).is_file()]

    def probe_pair(row: dict[str, str]) -> dict[str, Any]:
        try:
            source = decode_video(Path(row["source_local"]))
            target = decode_video(Path(row["target_local"]))
            reasons = []
            if min(source["duration"], target["duration"]) + 1e-6 < CROP_SECONDS:
                reasons.append("common_duration_under_5s")
            if (source["width"], source["height"]) != (target["width"], target["height"]):
                reasons.append("resolution_mismatch")
            if abs(source["fps"] - target["fps"]) > max(0.1, 0.01 * source["fps"]):
                reasons.append("fps_mismatch")
            return {"sample_id": row["sample_id"], "probe_policy": PROBE_POLICY, "status": "pass" if not reasons else "reject", "reasons": reasons, "source": source, "target": target}
        except Exception as error:
            return {"sample_id": row["sample_id"], "probe_policy": PROBE_POLICY, "status": "reject", "reasons": [f"decode_error:{type(error).__name__}"]}

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor, status_path.open("a", encoding="utf-8") as handle:
        futures = {executor.submit(probe_pair, row): row for row in pending}
        for future in as_completed(futures):
            value = future.result()
            known[value["sample_id"]] = value
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if completed % 100 == 0 or completed == len(pending):
                log(f"probe progress={completed}/{len(pending)} pass={sum(v['status']=='pass' for v in known.values())}")
    by_id = {row["sample_id"]: row for row in rows}
    summary = {
        "status": "complete_for_downloaded", "probed": len(known),
        "pass": sum(value["status"] == "pass" for value in known.values()),
        "reject": sum(value["status"] != "pass" for value in known.values()),
        "pass_by_anchor": dict(Counter(by_id[sample_id]["anchor_operation"] for sample_id, value in known.items() if value["status"] == "pass")),
        "reject_reasons": dict(Counter(reason for value in known.values() for reason in value.get("reasons", []))),
        "required_by_anchor": ANCHOR_QUOTAS,
    }
    write_json(OUTPUT / "technical_probe_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def allocate(rows: list[dict[str, Any]], counts: dict[str, int], seed_parts: tuple[object, ...]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: stable_key(*seed_parts, row["sample_id"]))
    result: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        result[split] = ordered[cursor:cursor + counts[split]]
        cursor += counts[split]
    if cursor != len(ordered):
        raise ValueError(f"allocation counts do not consume rows: {cursor}/{len(ordered)}")
    return result


def finalize() -> None:
    candidates = read_csv(OUTPUT / "candidate_manifest.csv")
    probes = {}
    with (OUTPUT / "technical_probe.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            probes[value["sample_id"]] = value
    selected: dict[str, list[dict[str, Any]]] = {}
    for anchor, quota in ANCHOR_QUOTAS.items():
        valid = [row for row in candidates if row["anchor_operation"] == anchor and probes.get(row["sample_id"], {}).get("status") == "pass"]
        valid.sort(key=lambda row: int(row["anchor_rank"]))
        if len(valid) < quota:
            raise RuntimeError(f"{anchor} needs {quota} valid samples, only {len(valid)} available; download/probe more candidates")
        selected[anchor] = valid[:quota]

    pure_rows: list[dict[str, Any]] = []
    pure_by_anchor_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for anchor, rows in selected.items():
        allocation = allocate(rows, PURE_SPLITS[anchor], (TRAIN_SEED, "pure-split", anchor))
        for split, values in allocation.items():
            pure_by_anchor_split[(anchor, split)] = values
            for row in values:
                pure_rows.append({
                    **row, "dataset": "CoinVE-200K", "branch": f"coinve_{anchor}",
                    "operation": anchor, "split": split, "technical_status": "pass",
                })
    pure_rows.sort(key=lambda row: ("train validation test".split().index(row["split"]), row["branch"], row["sample_id"]))

    mixed_coin: list[dict[str, Any]] = []
    for anchor in ANCHOR_QUOTAS:
        for split in ("train", "validation", "test"):
            available = pure_by_anchor_split[(anchor, split)]
            count = MIXED_COIN_SPLITS[anchor][split]
            values = sorted(available, key=lambda row: stable_key(TRAIN_SEED, "mixed-coin", anchor, split, row["sample_id"]))[:count]
            if len(values) != count:
                raise RuntimeError("CoinVE nested subset allocation failed")
            for row in values:
                mixed_coin.append({
                    **row, "dataset": "CoinVE-200K", "branch": f"coinve_{anchor}",
                    "operation": anchor, "split": split, "technical_status": "pass",
                })

    original = read_csv(SINGLE_POOL)
    mixed_single: list[dict[str, Any]] = []
    for branch, split_counts in MIXED_SINGLE_SPLITS.items():
        for split, count in split_counts.items():
            available = [row for row in original if row["branch"] == branch and row["split"] == split]
            values = sorted(available, key=lambda row: stable_key(TRAIN_SEED, "mixed-single", branch, split, row["sample_id"]))[:count]
            if len(values) != count:
                raise RuntimeError(f"Single subset allocation failed for {branch}/{split}")
            for row in values:
                source = (SINGLE_MEDIA_ROOT / row["source_local"]).resolve()
                target = (SINGLE_MEDIA_ROOT / row["target_local"]).resolve()
                mixed_single.append({
                    "selection_version": "mixed_2k2k_v1", "sample_id": row["sample_id"],
                    "dataset": row["dataset"], "branch": row["branch"],
                    "operation": row["operation"], "instruction": row["instruction"],
                    "source_local": str(source), "target_local": str(target), "split": split,
                    "source_family": "single4k", "metadata_index": "",
                    "instruction_count": 1, "has_stylize": 0,
                    "repo_id": row["repo_id"], "revision": row["revision"],
                    "selection_seed": SELECTION_SEED, "technical_status": row["technical_status"],
                })
    normalized_coin = [{
        "selection_version": "mixed_2k2k_v1", "sample_id": row["sample_id"],
        "dataset": row["dataset"], "branch": row["branch"], "operation": row["operation"],
        "instruction": row["instruction"], "source_local": row["source_local"],
        "target_local": row["target_local"], "split": row["split"],
        "source_family": "coinve", "metadata_index": row["metadata_index"],
        "instruction_count": row["instruction_count"], "has_stylize": row["has_stylize"],
        "repo_id": row["repo_id"], "revision": row["revision"],
        "selection_seed": SELECTION_SEED, "technical_status": "pass",
    } for row in mixed_coin]
    mixed_rows = mixed_single + normalized_coin
    mixed_rows.sort(key=lambda row: ("train validation test".split().index(row["split"]), row["source_family"], row["branch"], row["sample_id"]))

    pure_path = OUTPUT / "coinve_4k_pool.csv"
    mixed_path = OUTPUT / "mixed_2k2k_pool.csv"
    write_csv(pure_path, pure_rows)
    write_csv(mixed_path, mixed_rows)

    original_probe_path = STAGE_ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/data/video_probe.json"
    original_probe = json.loads(original_probe_path.read_text())

    def freeze_probe_cache(rows: list[dict[str, Any]], destination: Path) -> None:
        cache = {}
        for row in rows:
            if row["split"] != "train":
                continue
            for role in ("source", "target"):
                path = str(Path(row[f"{role}_local"]).resolve())
                if row.get("source_family", "coinve") == "single4k":
                    info = original_probe[path]
                else:
                    info = {"path": path, **probes[row["sample_id"]][role]}
                cache[path] = info
        write_json(destination, cache)

    freeze_probe_cache(
        pure_rows,
        STAGE_ROOT / "artifacts/lance_coinve_4k_r64_seed42_e2_v1/data/video_probe.json",
    )
    freeze_probe_cache(
        mixed_rows,
        STAGE_ROOT / "artifacts/lance_mixed_2k2k_r64_seed42_e2_v1/data/video_probe.json",
    )

    def check(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        split_counts = Counter(row["split"] for row in rows)
        if split_counts != Counter({"train": 3601, "validation": 195, "test": 204}):
            raise ValueError(f"{label} split mismatch: {split_counts}")
        if len(rows) != 4000 or len({row["sample_id"] for row in rows}) != 4000:
            raise ValueError(f"{label} identity count mismatch")
        return {
            "rows": len(rows), "splits": dict(split_counts),
            "datasets": dict(Counter(row["dataset"] for row in rows)),
            "branches": dict(Counter(row["branch"] for row in rows)),
            "operations": dict(Counter(row["operation"] for row in rows)),
        }
    summary = {
        "status": "complete", "selection_seed": SELECTION_SEED, "training_seed": TRAIN_SEED,
        "pure_coinve": check(pure_rows, "pure CoinVE"),
        "mixed_2k2k": check(mixed_rows, "mixed"),
        "coinve_nested_ids": len({row["sample_id"] for row in mixed_coin} & {row["sample_id"] for row in pure_rows}),
        "pure_pool": str(pure_path.resolve()), "pure_pool_sha256": sha256_file(pure_path),
        "mixed_pool": str(mixed_path.resolve()), "mixed_pool_sha256": sha256_file(mixed_path),
    }
    if summary["coinve_nested_ids"] != 2000:
        raise ValueError("mixed CoinVE half is not a nested subset of pure CoinVE")
    write_json(OUTPUT / "finalize_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "download", "probe", "finalize"))
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.phase == "select":
        select_candidates()
    elif args.phase == "download":
        if not 1.0 <= args.factor <= CANDIDATE_FACTOR:
            parser.error(f"--factor must be in [1.0, {CANDIDATE_FACTOR}]")
        download_candidates(args.factor, args.workers)
    elif args.phase == "probe":
        probe_candidates(args.workers)
    else:
        finalize()


if __name__ == "__main__":
    main()
