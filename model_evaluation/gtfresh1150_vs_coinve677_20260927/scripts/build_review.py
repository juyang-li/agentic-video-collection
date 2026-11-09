#!/usr/bin/env python3
"""Build the score-free portable OmniEdit + multi-turn human review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
ALIGNED = WORKSPACE / "video_collection/model_evaluation/omni_baselines_20260926/aligned_subset.csv"
BASE_REVIEW = WORKSPACE / "video_collection/model_evaluation/omni_baselines_20260926/lance/omni48_lance_base_seed42_human_review_20260926"
PACKAGE_NAME = "omni48_multiturn_gtfresh1150_vs_coinve677_human_review_20260927"
PACKAGE = ROOT / "review" / PACKAGE_NAME
SUITE = Path("/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/model_experiments/lance/configs/datamake_mixed_complete_chain_review_20260911.json")
MODELS = (("coinve_real_step677", "仅真实数据 · CoinVE-4K step677"),
          ("gtfresh_step1150", "真实+合成数据 · gtfresh-v1 step1150"))
FIELDS = ("instruction_correctness", "logical_correctness", "content_preservation",
          "temporal_consistency", "visual_quality", "technical_quality", "overall_decision")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def full_decode(path: Path) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    height, width = reader[0].shape[:2]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True)
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", result.stderr)
    if not match:
        raise RuntimeError(f"cannot probe {path}")
    data = path.read_bytes()[:2_000_000]
    return {"codec_name": match.group(1).strip().split()[0], "pix_fmt": match.group(2).split("(")[0],
            "width": width, "height": height, "fps": round(reader.get_avg_fps(), 6), "frames": len(reader),
            "faststart": data.find(b"moov") >= 0 and (data.find(b"mdat") < 0 or data.find(b"moov") < data.find(b"mdat"))}


def materialize(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    initial = probe(source)
    temporary = destination.with_suffix(".tmp.mp4")
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
        raise RuntimeError(f"nonportable review media: {destination}: {result}")
    return {"path": destination.relative_to(PACKAGE).as_posix(), "source": str(source),
            "sha256": sha256(destination), "bytes": destination.stat().st_size, "full_decode_ok": True, **result}


def raw_single(model: str, index: int) -> Path:
    shard = (index - 1) % 4
    return ROOT / "outputs_raw" / f"omni48_{model}_shard{shard:02d}" / f"{index:06d}.mp4"


def review_row(review_id: str, section: str, eval_id: str, category: str, instruction: str,
               model_id: str, model_label: str) -> dict:
    return {"review_id": review_id, "section": section, "eval_id": eval_id, "category": category,
            "instruction": instruction, "model_id": model_id, "model_label": model_label,
            **{field: "1" for field in FIELDS}, "notes": ""}


def main() -> None:
    if PACKAGE.exists() or PACKAGE.with_suffix(".zip").exists():
        raise FileExistsError(f"refusing to overwrite review package: {PACKAGE}")
    (PACKAGE / "media").mkdir(parents=True)
    aligned = read_csv(ALIGNED)
    suite = json.loads(SUITE.read_text())
    inventory, reviews, items, generation = [], [], [], []

    for index, row in enumerate(aligned, 1):
        paths = []
        source_proxy = BASE_REVIEW / "media" / f"{index:03d}_source.mp4"
        source_dst = PACKAGE / "media" / f"single_{index:03d}_source.mp4"
        inventory.append({"role": "single_source", "eval_id": row["eval_id"], **materialize(source_proxy, source_dst)})
        paths.append({"label": "原始视频", "path": source_dst.relative_to(PACKAGE).as_posix()})
        item_reviews = []
        for model_id, model_label in MODELS:
            raw = raw_single(model_id, index)
            dst = PACKAGE / "media" / f"single_{index:03d}_{model_id}.mp4"
            inventory.append({"role": "single_output", "eval_id": row["eval_id"], "model_id": model_id,
                              **materialize(raw, dst)})
            paths.append({"label": model_label, "path": dst.relative_to(PACKAGE).as_posix()})
            review = review_row(f"single:{row['eval_id']}:{model_id}", "single", row["eval_id"],
                                row["category"], row["editing_instruction"], model_id, model_label)
            reviews.append(review); item_reviews.append(review)
            generation.append({"section": "single", "eval_id": row["eval_id"], "model_id": model_id,
                               "source": row["official_source_video"], "source_sha256": row["official_source_sha256"],
                               "output": str(raw), "output_sha256": sha256(raw), "seed": 42,
                               "steps": 30, "shift": 3.5, "cfg_text": 4.0})
        items.append({"section": "single", "eval_id": row["eval_id"], "category": row["category"],
                      "instruction": row["editing_instruction"], "panels": paths, "reviews": item_reviews})

    media_root = Path(suite["media_root"])
    manifests = {model: json.loads((ROOT / "multiturn" / model / "manifest.json").read_text()) for model, _ in MODELS}
    for chain_index, chain in enumerate(suite["chains"], 1):
        chain_id, turns = chain["chain_id"], chain["target_round"]
        prompts = chain["prompts"][:turns]
        rows = []
        gt_paths = []
        for round_index, relative in enumerate(chain["video_paths"][:turns + 1]):
            source = media_root / relative
            dst = PACKAGE / "media" / f"chain{chain_index}_gt_r{round_index:02d}.mp4"
            inventory.append({"role": "multiturn_gt", "eval_id": chain_id, "round": round_index,
                              **materialize(source, dst)})
            gt_paths.append(dst.relative_to(PACKAGE).as_posix())
        rows.append({"label": "真值参考", "paths": gt_paths})
        item_reviews = []
        for model_id, model_label in MODELS:
            manifest = manifests[model_id]
            case = next(case for case in manifest["cases"] if case["chain_id"] == chain_id and case["mode"] == "autoregressive")
            model_paths = [gt_paths[0]]
            for output in case["outputs"]:
                raw = ROOT / "multiturn" / model_id / output["output"]
                dst = PACKAGE / "media" / f"chain{chain_index}_{model_id}_r{output['round']:02d}.mp4"
                inventory.append({"role": "multiturn_output", "eval_id": chain_id, "round": output["round"],
                                  "model_id": model_id, **materialize(raw, dst)})
                model_paths.append(dst.relative_to(PACKAGE).as_posix())
                generation.append({"section": "multiturn", "eval_id": chain_id, "round": output["round"],
                                   "model_id": model_id, "output": str(raw), "output_sha256": sha256(raw),
                                   "seed": 42, "steps": 30, "shift": 3.5, "cfg_text": 1.0})
            rows.append({"label": model_label, "paths": model_paths})
            instruction = "\n".join(f"R{i + 1}: {prompt}" for i, prompt in enumerate(prompts))
            review = review_row(f"multiturn:{chain_id}:{model_id}", "multiturn", chain_id,
                                f"{turns}轮编辑链", instruction, model_id, model_label)
            reviews.append(review); item_reviews.append(review)
        items.append({"section": "multiturn", "eval_id": chain_id, "category": f"{turns}轮编辑链",
                      "instruction": "\n".join(f"R{i + 1}: {p}" for i, p in enumerate(prompts)),
                      "prompts": prompts, "rows": rows, "reviews": item_reviews})

    write_csv(PACKAGE / "review_manifest.csv", reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"status": "built_pending_browser_validation", "human_only": True,
               "gemini_included": False, "single_turn_samples": 48, "single_turn_outputs": 96,
               "multiturn_chains": 3, "multiturn_turns_per_model": 12,
               "review_units": len(reviews), "media_files": len(inventory),
               "comparison_limit": "practical comparison of existing models; not a controlled synthetic-data ablation"}
    write_json(PACKAGE / "summary.json", summary)
    payload = {"review_version": PACKAGE_NAME, "title": "Lance · 仅真实数据 vs 真实+合成数据 · 人工审核",
               "items": items, "reviews": reviews}
    template = (Path(__file__).with_name("review_template.html")).read_text(encoding="utf-8")
    document = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# Lance 人工审核包：仅真实数据 vs 真实+合成数据\n\n"
        "完整解压后，用 Chrome 或 Edge 打开 `index.html`。页面完全离线，不调用任何 API。\n\n"
        "单轮部分包含 OmniEdit Spatial 48 条（24类×2），依次展示原始视频、仅真实 CoinVE-4K step677、"
        "真实+合成 gtfresh-v1 step1150。OmniEdit Spatial 不提供 GT。多轮部分包含同一组3/4/5轮验证链，"
        "展示真值参考与两个模型的自回填结果。\n\n"
        "所有审核项默认是1（通过）；请把失败改为0、存疑改为2，并填写备注。支持 CSV/JSON 导出与导入。"
        "本包不含 Gemini 分数、API key、checkpoint 或 optimizer。\n\n"
        "注意：两个模型的训练配方除数据外也有差异，本包用于比较两个现有模型的实际效果，不构成严格的单变量合成数据消融。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
