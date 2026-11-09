#!/usr/bin/env python3
"""Build a branch-covered human audit package from the frozen mixed train set."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from decord import VideoReader, cpu
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(
    "/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/model_experiments/lance/"
    "outputs/training_data/datamake_coinve4k_mixed_r5x3_seed20260911_v1"
)
MEDIA_ROOT = Path(
    "/scratch/prj0000000275/jcy/xks/DataMake/00_engine/outputs/delivery/"
    "first_batch_candidate_20260830_v002/dataset_v1"
)
DATAMAKE_SCRIPTS = Path(
    "/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/"
    "model_experiments/lance/scripts"
)
PACKAGE_NAME = "gtfresh_training_data_branch_audit_20261006"
PACKAGE = ROOT / "review" / PACKAGE_NAME

UE_CHAIN_IDS = (
    "c_063574b09768f6ba", "c_582439526d60e001", "c_63bc4d4189caa919",
    "c_9ebe12b3d2dd5d68", "c_b31a30e4ea5e5301", "c_f8e78ab42b9a51f9",
    "c_5f1eb50eb42c9a4d", "c_622d450bb4862a9c", "c_6d5d7c4a0d9b6849",
    "c_a656930bc092dbc6", "c_c07161dce94be3a3", "c_ccfcfd7d5ff5e6d4",
    "c_ed18909727386977", "c_0c24dba06ac9e4b1", "c_1db7f4c9b7adb50c",
    "c_32e5d67877be6d92", "c_40dd5211ee99d028", "c_4bb8ec92365d8f42",
    "c_67a8b355215f6ede", "c_781c75d4c62fa2bb",
)
COINVE_IDS = (
    "coinve-022004", "coinve-184242",  # anchor Add
    "coinve-081813", "coinve-193912",  # anchor Remove
    "coinve-007443", "coinve-107020",  # anchor Replace
)
FIELDS = ("person_relevance", "instruction_correctness", "logical_correctness",
          "content_preservation", "temporal_consistency", "visual_quality", "technical_quality")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def full_decode(path: Path) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    height, width = reader[0].shape[:2]
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path),
                             "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", result.stderr)
    if not match:
        raise RuntimeError(f"cannot probe {path}")
    prefix = path.read_bytes()[:2_000_000]
    return {
        "codec_name": match.group(1).strip().split()[0],
        "pix_fmt": match.group(2).split("(")[0], "width": width, "height": height,
        "fps": round(reader.get_avg_fps(), 6), "frames": len(reader),
        "faststart": prefix.find(b"moov") >= 0 and
                     (prefix.find(b"mdat") < 0 or prefix.find(b"moov") < prefix.find(b"mdat")),
    }


def inventory_item(source: Path, destination: Path, role: str, sample_id: str,
                   round_no: int | None, crop: dict | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    crop_evidence = None
    if crop is not None:
        sys.path.insert(0, str(DATAMAKE_SCRIPTS))
        from datamake_coinve_review_media import sampled_frames
        frames, crop_evidence = sampled_frames(source, crop)
        width, height = frames[0].size
        scale = min(960 / width, 540 / height, 1.0)
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        writer = imageio_ffmpeg.write_frames(
            str(temporary), size, fps=12, codec="libx264", pix_fmt_out="yuv420p",
            macro_block_size=2, output_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            from PIL import Image
            for frame in frames:
                writer.send(np.asarray(frame.resize(size, Image.Resampling.LANCZOS)))
        finally:
            writer.close()
    else:
        initial = probe(source)
        if initial["codec_name"] == "h264" and initial["pix_fmt"] == "yuv420p":
            command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
                       "-c:v", "copy", "-an", "-movflags", "+faststart", str(temporary)]
        else:
            command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                       "-an", "-movflags", "+faststart", str(temporary)]
        subprocess.run(command, check=True)
    os.replace(temporary, destination)
    full_decode(destination)
    result = probe(destination)
    if result["codec_name"] != "h264" or result["pix_fmt"] != "yuv420p" or not result["faststart"]:
        raise RuntimeError(f"nonportable media: {destination}: {result}")
    item = {
        "sample_id": sample_id, "role": role, "round": round_no,
        "path": destination.relative_to(PACKAGE).as_posix(), "source": str(source),
        "source_sha256": sha256(source), "sha256": sha256(destination),
        "bytes": destination.stat().st_size, "full_decode_ok": True, **result,
    }
    if crop_evidence is not None:
        item["training_crop"] = crop_evidence
    return item


def load_rows() -> tuple[dict[str, dict], dict[str, dict]]:
    rows = []
    for path in sorted((DATASET / "train").glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    ue = {}
    coinve = {}
    for row in rows:
        if row["dataset_origin"] == "ue_synthetic":
            previous = ue.get(row["chain_id"])
            if previous is None or (row["target_round"], -row["sampling_repeat_index"]) > (
                    previous["target_round"], -previous["sampling_repeat_index"]):
                ue[row["chain_id"]] = row
        else:
            coinve[row["sample_id"]] = row
    selected_ue = {key: ue[key] for key in UE_CHAIN_IDS}
    selected_coinve = {key: coinve[key] for key in COINVE_IDS}
    return selected_ue, selected_coinve


def load_provenance() -> dict[str, dict]:
    return {row["sample_id"]: row for row in (
        json.loads(line) for line in (DATASET / "coinve_provenance.jsonl").read_text().splitlines())}


def audit_row(sample_id: str, origin: str, task: str) -> dict:
    return {"hash": hashlib.sha256(sample_id.encode()).hexdigest()[:5], "sample_id": sample_id,
            "origin": origin, "task": task, **{field: "1" for field in FIELDS}, "notes": ""}


def main() -> None:
    if PACKAGE.exists() or PACKAGE.with_suffix(".zip").exists():
        raise FileExistsError(f"refusing to overwrite: {PACKAGE}")
    selected_ue, selected_coinve = load_rows()
    provenance = load_provenance()
    (PACKAGE / "media").mkdir(parents=True)
    inventory, audits, items = [], [], []
    axis_labels = {"identity_id": "身份", "outfit_id": "服装", "action_id": "动作", "camera_id": "相机"}

    for index, chain_id in enumerate(UE_CHAIN_IDS, 1):
        row = selected_ue[chain_id]
        facts = json.loads(row["semantic_facts_json"])
        paths = []
        for round_no, relative in enumerate(row["video_paths"]):
            source = MEDIA_ROOT / relative
            destination = PACKAGE / "media" / f"ue_{index:02d}_r{round_no:02d}.mp4"
            info = inventory_item(source, destination, "ue_source" if round_no == 0 else "ue_target",
                                  chain_id, round_no)
            inventory.append(info); paths.append(info["path"])
        task = " / ".join(axis_labels[axis] for axis in row["edit_axes"])
        audit = audit_row(chain_id, "ue_synthetic", task)
        audits.append(audit)
        items.append({
            "sample_id": chain_id, "origin": "UE 合成", "kind": "multiturn",
            "category": f"{row['target_round']}轮 · {row['chain_policy']}",
            "task": task, "prompts": row["prompts"], "axes": row["edit_axes"],
            "paths": paths, "facts": facts, "audit": audit,
        })

    for offset, sample_id in enumerate(COINVE_IDS, 1):
        row = selected_coinve[sample_id]
        prov = provenance[sample_id]
        crop = {"path": row["input_video_path"], "start_seconds": row["crop_start_seconds"],
                "duration_seconds": row["crop_duration_seconds"], "policy": row["crop_policy"]}
        paths = []
        for role, source_path in (("coinve_source", row["video_paths"][0]),
                                  ("coinve_target", row["video_paths"][1])):
            source = Path(source_path)
            destination = PACKAGE / "media" / f"coinve_{offset:02d}_{role.rsplit('_',1)[1]}.mp4"
            role_crop = {**crop, "path": str(source)}
            info = inventory_item(source, destination, role, sample_id,
                                  0 if role.endswith("source") else 1, role_crop)
            inventory.append(info); paths.append(info["path"])
        branch = prov["branch"]
        task = " / ".join(row["edit_axes"])
        audit = audit_row(sample_id, "coinve", task)
        audits.append(audit)
        items.append({
            "sample_id": sample_id, "origin": "CoinVE 真实", "kind": "single",
            "category": branch, "task": task, "prompts": row["prompts"],
            "axes": row["edit_axes"], "paths": paths,
            "facts": {"anchor_branch": branch, "crop_start_seconds": row["crop_start_seconds"],
                      "crop_duration_seconds": row["crop_duration_seconds"]},
            "audit": audit,
        })

    axis_counts = {}
    for axis in axis_labels:
        axis_counts[axis] = sum(r["edit_axes"].count(axis) for r in selected_ue.values())
    length_counts = {str(length): sum(r["target_round"] == length for r in selected_ue.values())
                     for length in (3, 4, 5)}
    policy_counts = {policy: sum(r["chain_policy"] == policy for r in selected_ue.values())
                     for policy in ("progressive_shortest", "freeform_walk")}
    fact_counts = {key: sum(json.loads(r["semantic_facts_json"])[key] for r in selected_ue.values())
                   for key in ("has_consecutive_same_axis", "has_repeated_axis", "has_rollback",
                               "is_semantic_shortest", "returns_to_start")}
    branch_counts = {branch: sum(provenance[sid]["branch"] == branch for sid in COINVE_IDS)
                     for branch in ("coinve_add", "coinve_remove", "coinve_replace")}
    coverage = {"ue_chains": 20, "ue_turns": sum(axis_counts.values()), "ue_axis_counts": axis_counts,
                "ue_length_counts": length_counts, "ue_policy_counts": policy_counts,
                "ue_structure_counts": fact_counts, "coinve_samples": 6,
                "coinve_branch_counts": branch_counts}
    if min(axis_counts.values()) < 19 or length_counts != {"3": 6, "4": 7, "5": 7}:
        raise RuntimeError(f"selection coverage changed: {coverage}")
    if policy_counts != {"progressive_shortest": 10, "freeform_walk": 10}:
        raise RuntimeError(f"selection policy coverage changed: {coverage}")
    if branch_counts != {"coinve_add": 2, "coinve_remove": 2, "coinve_replace": 2}:
        raise RuntimeError(f"CoinVE coverage changed: {coverage}")

    with (PACKAGE / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        rows = []
        for item in items:
            rows.append({"hash": item["audit"]["hash"], "sample_id": item["sample_id"],
                         "origin": item["origin"], "category": item["category"], "task": item["task"],
                         "rounds": len(item["prompts"]), "instructions": " | ".join(item["prompts"])})
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (PACKAGE / "audit.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0])); writer.writeheader(); writer.writerows(audits)
    write_json(PACKAGE / "media_inventory.json", inventory)
    write_json(PACKAGE / "selection_manifest.json", {
        "schema_version": "gtfresh_training_branch_audit_v1", "dataset": str(DATASET),
        "split": "train", "selection_seed": 20261006,
        "selection_policy": "20 fixed full UE chains stratified for four axes, lengths and structures; 2 person-relevant CoinVE samples per anchor branch",
        "ue_chain_ids": list(UE_CHAIN_IDS), "coinve_sample_ids": list(COINVE_IDS),
        "coverage": coverage,
        "source_manifests": {name: {"path": str(DATASET / name), "sha256": sha256(DATASET / name)}
                             for name in ("split_manifest.json", "sampling_manifest.json", "coinve_provenance.jsonl")},
    })
    summary = {"status": "built_pending_browser_validation", "cards": 26, "audit_rows": 26,
               "media_files": len(inventory), "synthetic_cards": 20, "coinve_cards": 6,
               "synthetic_turns": sum(axis_counts.values()), "coverage": coverage,
               "default_rating": 1, "automatic_scoring": False}
    write_json(PACKAGE / "summary.json", summary)
    payload = {"title": "师兄 gtfresh 训练数据抽样审核", "review_version": PACKAGE_NAME,
               "fields": list(FIELDS), "items": items, "audits": audits, "coverage": coverage}
    template = (Path(__file__).with_name("review_template.html")).read_text(encoding="utf-8")
    document = template.replace("__AUDIT_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", document, re.S)
    subprocess.run(["node", "--check"], input=scripts[-1], text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# 师兄 gtfresh 训练数据抽样审核包\n\n完整解压后，用 Chrome 或 Edge 打开 `index.html`。页面完全离线。\n\n"
        "前 20 张卡是 UE 合成完整 GT 多轮链，后 6 张是 CoinVE 真实 source/target 配对。CoinVE 视频严格按训练记录的"
        "同一起点 5 秒、12 fps 窗口展示。所有审核字段默认是 1；失败改成 0，存疑改成 2。支持 CSV/JSON 导出导入。\n\n"
        "`inventory.csv` 保存任务与指令；`audit.csv` 按项目规范不重复 instruction；`selection_manifest.json` 记录抽样覆盖和来源哈希。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
