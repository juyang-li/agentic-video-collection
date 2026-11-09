#!/usr/bin/env python3
"""Build the human-only, checkpoint-focused drift review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark.jsonl"
PACKAGE_NAME = "real_person_drift_stress_three_model_human_review_20261009"
PACKAGE = ROOT / "review" / PACKAGE_NAME
TEMPLATE = ROOT.parent / "gtfresh1150_vs_coinve677_20260927/scripts/review_template.html"
MODELS = (
    ("lance_base", "Lance 基模"),
    ("coinve_real_step677", "仅真实数据 · CoinVE step677"),
    ("gtfresh_step1150", "真实+合成数据 · gtfresh step1150"),
)
FIELDS = ("edit_applied", "history_state_restored", "no_residual",
          "intervening_edits_preserved", "identity_background_stable",
          "temporal_consistency", "visual_quality", "overall_drift")
KIND_LABELS = {
    "rollback_to_original": "回到原视频状态",
    "second_rollback_to_original": "第二次回到原视频状态",
    "replay_prior_edit": "重放早期编辑状态",
    "restore_intermediate_state": "恢复中间历史状态",
    "rollback_preserve_intervening": "回撤动作并保留中间编辑",
    "delayed_rollback_preserve_intervening": "延迟回撤并保留中间编辑",
}


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
    return {"codec_name": match.group(1).strip().split()[0], "pix_fmt": match.group(2).split("(")[0],
            "width": width, "height": height, "fps": round(float(reader.get_avg_fps()), 6),
            "frames": len(reader), "faststart": prefix.find(b"moov") >= 0 and
            (prefix.find(b"mdat") < 0 or prefix.find(b"moov") < prefix.find(b"mdat"))}


def materialize(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    initial = probe(source)
    temporary = destination.with_suffix(".tmp.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
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
    final = probe(destination)
    if final["codec_name"] != "h264" or final["pix_fmt"] != "yuv420p" or not final["faststart"]:
        raise RuntimeError(f"nonportable media: {destination}: {final}")
    return {"path": destination.relative_to(PACKAGE).as_posix(), "source": str(source),
            "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "full_decode_ok": True, **final}


def validate_runs(chains: list[dict]) -> None:
    expected_ids = [chain["drift_id"] for chain in chains]
    benchmark_hash = sha256(BENCHMARK)
    for model_id, _ in MODELS:
        manifest = json.loads((ROOT / "outputs" / model_id / "manifest.json").read_text())
        if manifest.get("status") != "generated" or manifest.get("benchmark_sha256") != benchmark_hash:
            raise RuntimeError(f"incomplete or mismatched run: {model_id}")
        if [case["drift_id"] for case in manifest["cases"]] != expected_ids:
            raise RuntimeError(f"coverage/order mismatch: {model_id}")
        if model_id == "coinve_real_step677":
            check = manifest.get("adapter_load_check") or {}
            if (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B")) != (504, 504, 252):
                raise RuntimeError("CoinVE adapter load verification missing")
        if model_id == "gtfresh_step1150":
            check = manifest.get("adapter_load_check") or {}
            if (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B")) != (1016, 1016, 503):
                raise RuntimeError("gtfresh adapter load verification missing")


def review_row(chain: dict, checkpoint: dict, model_id: str, model_label: str) -> dict:
    preserve = ",".join(map(str, checkpoint["preserve_rounds"])) or "none"
    return {
        "review_id": f"{chain['drift_id']}:{model_id}:r{checkpoint['round']}",
        "section": "multiturn", "eval_id": chain["drift_id"], "pattern": chain["pattern"],
        "checkpoint_round": checkpoint["round"], "reference_round": checkpoint["reference_round"],
        "preserve_rounds": preserve, "comparison_type": checkpoint["comparison_type"],
        "model_id": model_id,
        "model_label": f"{model_label} · 检查 R{checkpoint['round']} ↔ R{checkpoint['reference_round']} · {KIND_LABELS[checkpoint['comparison_type']]}",
        **{field: "1" for field in FIELDS}, "notes": "",
    }


def main() -> None:
    if PACKAGE.exists() or PACKAGE.with_suffix(".zip").exists():
        raise FileExistsError(f"refusing to overwrite: {PACKAGE}")
    chains = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(chains) != 12 or sum(len(chain["turns"]) for chain in chains) != 48 or sum(len(chain["checkpoints"]) for chain in chains) != 24:
        raise RuntimeError("benchmark shape mismatch")
    validate_runs(chains)
    (PACKAGE / "media").mkdir(parents=True)

    inventory, generation, reviews, items = [], [], [], []
    for chain_index, chain in enumerate(chains, 1):
        source = ROOT / chain["source"]["media"]
        source_dst = PACKAGE / "media" / f"chain{chain_index:02d}_source.mp4"
        source_info = materialize(source, source_dst)
        inventory.append({"role": "source", "eval_id": chain["drift_id"], **source_info})
        rows = [{"label": "原始视频（R0）", "paths": [source_info["path"]] * 5}]
        item_reviews = []
        for model_id, model_label in MODELS:
            paths = [source_info["path"]]
            for round_index, turn in enumerate(chain["turns"], 1):
                raw = ROOT / "outputs" / model_id / chain["drift_id"] / f"turn{round_index - 1:02d}.mp4"
                if not raw.is_file():
                    raise FileNotFoundError(raw)
                destination = PACKAGE / "media" / f"chain{chain_index:02d}_{model_id}_r{round_index}.mp4"
                info = materialize(raw, destination)
                inventory.append({"role": "output", "eval_id": chain["drift_id"], "round": round_index,
                                  "model_id": model_id, **info})
                paths.append(info["path"])
                generation.append({"eval_id": chain["drift_id"], "round": round_index, "axis": turn["axis"],
                                   "instruction": turn["instruction"], "model_id": model_id,
                                   "source_sha256": sha256(source), "output_sha256": sha256(raw),
                                   "autoregressive": True, "seed": 42, "steps": 30, "shift": 3.5,
                                   "cfg_text": 1.0, "cfg_vit": 1.0})
            rows.append({"label": model_label, "paths": paths})
            for cp in chain["checkpoints"]:
                entry = review_row(chain, cp, model_id, model_label)
                reviews.append(entry); item_reviews.append(entry)
        prompts = [turn["instruction"] for turn in chain["turns"]]
        checkpoint_text = "；".join(
            f"R{cp['round']}↔R{cp['reference_round']} {KIND_LABELS[cp['comparison_type']]}"
            + (f"（同时保留 R{','.join(map(str, cp['preserve_rounds']))}）" if cp["preserve_rounds"] else "")
            for cp in chain["checkpoints"])
        items.append({"section": "multiturn", "eval_id": chain["drift_id"],
                      "category": f"{chain['pattern']} · {chain['focus']}",
                      "instruction": "\n".join([*(f"R{i}: {p}" for i, p in enumerate(prompts, 1)), f"重点检查：{checkpoint_text}"]),
                      "prompts": prompts, "rows": rows, "reviews": item_reviews})

    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0])); writer.writeheader(); writer.writerows(reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"status": "built_pending_browser_validation", "human_only": True, "gemini_included": False,
               "models": [model for model, _ in MODELS], "chains": 12, "turns_per_model": 48,
               "generated_outputs": 144, "drift_checkpoints_per_model": 24, "review_units": 72,
               "media_files": 156,
               "comparison_limit": "The checkpoints differ in training data and recipe; this is a practical checkpoint comparison, not a controlled data-only ablation."}
    write_json(PACKAGE / "summary.json", summary)
    payload = {"review_version": PACKAGE_NAME, "title": "真实人物多轮编辑 · 漂移专项三模型人工审核",
               "items": items, "reviews": reviews}
    document = TEMPLATE.read_text(encoding="utf-8")
    document = document.replace("真实数据 vs 混合数据 Lance 人工审核", payload["title"])
    old_fields = "const fields=['instruction_correctness','logical_correctness','content_preservation','temporal_consistency','visual_quality','technical_quality','overall_decision'];"
    new_fields = "const fields=['edit_applied','history_state_restored','no_residual','intervening_edits_preserved','identity_background_stable','temporal_consistency','visual_quality','overall_drift'];"
    old_labels = "const labels={instruction_correctness:'指令完成',logical_correctness:'逻辑正确',content_preservation:'内容保持',temporal_consistency:'时序一致',visual_quality:'视觉质量',technical_quality:'技术质量',overall_decision:'总体结论'};"
    new_labels = "const labels={edit_applied:'本轮指令完成',history_state_restored:'历史状态恢复',no_residual:'无残留编辑',intervening_edits_preserved:'中间编辑保持',identity_background_stable:'人物背景稳定',temporal_consistency:'时序一致',visual_quality:'视觉质量',overall_drift:'总体无漂移'};"
    document = document.replace(old_fields, new_fields).replace(old_labels, new_labels)
    document = document.replace("r.overall_decision!=='1'", "r.overall_drift!=='1'")
    document = document.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    if old_fields in document or old_labels in document or "__REVIEW_DATA__" in document:
        raise RuntimeError("review template substitution failed")
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[-1]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# 真实人物多轮编辑漂移专项审核包\n\n完整解压后用 Chrome 或 Edge 打开 `index.html`；页面完全离线。\n\n"
        "共 12 条四轮链，依次展示 R0 原视频、Lance 基模、CoinVE-only step677、gtfresh step1150。"
        "重点审核 24 个回撤/重放检查点，每个检查点对三个模型分别标注，共 72 个审核单元。\n\n"
        "所有项默认 1（通过），0 为不通过，2 为存疑。`历史状态恢复` 看目标状态是否回到指定轮；"
        "`无残留编辑` 看被撤销属性是否仍残留；`中间编辑保持` 只在延迟回撤链中重点判断。"
        "支持 CSV/JSON 导出与导入。本包不含自动指标或 Gemini 评分。\n\n"
        "注意：三个 checkpoint 的训练配方不完全一致，因此结论是现有模型的实际横向比较，并非严格单变量数据消融。\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
