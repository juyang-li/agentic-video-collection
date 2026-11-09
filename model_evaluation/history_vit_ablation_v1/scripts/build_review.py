#!/usr/bin/env python3
"""Build the offline human-review package for the history-ViT ablation."""

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
DRIFT = ROOT.parent / "real_person_drift_stress_v1"
BENCHMARK = DRIFT / "benchmark.jsonl"
CHECKPOINTS = DRIFT / "checkpoints.csv"
PACKAGE_NAME = "real_person_history_vit_ablation_base_gtfresh_human_review_20261016"
PACKAGE = ROOT / "review" / PACKAGE_NAME
TEMPLATE = ROOT.parent / "gtfresh1150_vs_coinve677_20260927/scripts/review_template.html"
MODELS = (
    ("lance_base", "Lance Base"),
    ("gtfresh_step1150", "gtfresh step1150"),
)
CONDITIONS = (
    ("vit_zero_text_keep", "历史 ViT=0；历史文本保留"),
    ("vit_gaussian_text_keep", "历史 ViT=匹配高斯；历史文本保留"),
    ("vit_zero_text_zero", "历史 ViT=0；历史文本=0"),
    ("vit_gaussian_text_zero", "历史 ViT=匹配高斯；历史文本=0"),
)
ALL_CONDITIONS = ("normal_replay", *(condition for condition, _ in CONDITIONS))
FIELDS = (
    "edit_applied",
    "history_state_correct",
    "normal_equivalent",
    "identity_background_stable",
    "temporal_consistency",
    "visual_quality",
    "overall_pass",
)
KIND_LABELS = {
    "rollback_to_original": "回到原视频状态",
    "second_rollback_to_original": "再次回到原视频状态",
    "replay_prior_edit": "重放早期编辑状态",
    "restore_intermediate_state": "恢复中间历史状态",
    "rollback_preserve_intervening": "回撤并保留中间编辑",
    "delayed_rollback_preserve_intervening": "延迟回撤并保留中间编辑",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def full_decode(path: Path) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
         "-map", "0:v:0", "-f", "null", "-"],
        check=True,
    )


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    height, width = reader[0].shape[:2]
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", result.stderr)
    if not match:
        raise RuntimeError(f"cannot probe {path}")
    prefix = path.read_bytes()[:2_000_000]
    return {
        "codec_name": match.group(1).strip().split()[0],
        "pix_fmt": match.group(2).split("(")[0],
        "width": width,
        "height": height,
        "fps": round(float(reader.get_avg_fps()), 6),
        "frames": len(reader),
        "faststart": prefix.find(b"moov") >= 0
        and (prefix.find(b"mdat") < 0 or prefix.find(b"moov") < prefix.find(b"mdat")),
    }


def materialize(source: Path, destination: Path) -> dict:
    if not source.is_file() or source.stat().st_size <= 1000:
        raise FileNotFoundError(source)
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
    return {
        "path": destination.relative_to(PACKAGE).as_posix(),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "full_decode_ok": True,
        **final,
    }


def load_inputs() -> tuple[dict[str, dict], list[dict]]:
    records = {
        row["drift_id"]: row
        for row in (json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines()
                    if line.strip())
    }
    with CHECKPOINTS.open(encoding="utf-8-sig", newline="") as stream:
        checkpoints = list(csv.DictReader(stream))
    if len(records) != 12 or len(checkpoints) != 24:
        raise RuntimeError("expected frozen 12-chain / 24-checkpoint benchmark")
    if any(row["drift_id"] not in records for row in checkpoints):
        raise RuntimeError("checkpoint references an unknown drift chain")
    return records, checkpoints


def output_path(model_id: str, condition: str, checkpoint: dict) -> Path:
    return (ROOT / "outputs" / model_id / condition / checkpoint["drift_id"]
            / f"turn{int(checkpoint['round']) - 1:02d}.mp4")


def direct_source_path(model_id: str, checkpoint: dict) -> Path:
    return (DRIFT / "outputs" / model_id / checkpoint["drift_id"]
            / f"turn{int(checkpoint['round']) - 2:02d}.mp4")


def validate_runs(checkpoints: list[dict]) -> None:
    benchmark_hash = sha256(BENCHMARK)
    for model_id, _ in MODELS:
        old_manifest_path = DRIFT / "outputs" / model_id / "manifest.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("status") != "generated" or old_manifest.get("benchmark_sha256") != benchmark_hash:
            raise RuntimeError(f"invalid canonical normal-chain run: {model_id}")
        for condition in ALL_CONDITIONS:
            manifest_path = ROOT / "outputs" / model_id / condition / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("status"), manifest.get("checkpoint_count"),
                    manifest.get("benchmark_sha256")) != ("generated", 24, benchmark_hash):
                raise RuntimeError(f"incomplete or mismatched run: {model_id}/{condition}")
            if model_id == "lance_base" and manifest.get("adapter_check") is not None:
                raise RuntimeError(f"Base unexpectedly reports an adapter: {condition}")
            if model_id == "gtfresh_step1150":
                check = manifest.get("adapter_check") or {}
                values = (check.get("tensor_count"), check.get("matching_after_cast"),
                          check.get("nonzero_lora_B"), check.get("missing_live_keys"))
                if values != (1016, 1016, 503, 0):
                    raise RuntimeError(f"gtfresh adapter verification missing: {condition}: {values}")
            for checkpoint in checkpoints:
                video = output_path(model_id, condition, checkpoint)
                receipt = video.with_suffix(".json")
                if not video.is_file() or video.stat().st_size <= 1000 or not receipt.is_file():
                    raise FileNotFoundError(f"missing output or receipt: {video}")
                data = json.loads(receipt.read_text(encoding="utf-8"))
                expected = ("generated", model_id, condition, checkpoint["drift_id"],
                            int(checkpoint["round"]), sha256(video))
                actual = (data.get("status"), data.get("model_id"), data.get("condition"),
                          data.get("drift_id"), data.get("round"), data.get("output_sha256"))
                if actual != expected:
                    raise RuntimeError(f"receipt mismatch: {receipt}")
                if condition != "normal_replay":
                    trace = data.get("trace") or {}
                    if (trace.get("vit_hook_calls") != 1
                            or trace.get("direct_source_unchanged") is not True
                            or int(trace.get("history_vit_tokens", 0)) <= 0):
                        raise RuntimeError(f"ViT intervention audit failed: {receipt}")
                    if condition.endswith("text_zero") and (
                            trace.get("text_hook_calls") != 1
                            or int(trace.get("history_text_tokens", 0)) <= 0):
                        raise RuntimeError(f"text intervention audit failed: {receipt}")
                direct = direct_source_path(model_id, checkpoint)
                if not direct.is_file() or direct.stat().st_size <= 1000:
                    raise FileNotFoundError(direct)


def review_row(checkpoint: dict, card_id: str, model_id: str, model_label: str,
               condition: str, condition_label: str) -> dict:
    return {
        "review_id": f"{card_id}:{model_id}:{condition}",
        "card_id": card_id,
        "section": "multiturn",
        "eval_id": checkpoint["drift_id"],
        "pattern": checkpoint["pattern"],
        "checkpoint_round": checkpoint["round"],
        "reference_round": checkpoint["reference_round"],
        "preserve_rounds": checkpoint["preserve_rounds"] or "none",
        "comparison_type": checkpoint["comparison_type"],
        "model_id": model_id,
        "model_label": f"{model_label} · {condition_label}",
        "condition": condition,
        "condition_label": condition_label,
        **{field: "1" for field in FIELDS},
        "notes": "",
    }


def main() -> None:
    archive = PACKAGE.with_suffix(".zip")
    if PACKAGE.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite: {PACKAGE} or {archive}")
    records, checkpoints = load_inputs()
    validate_runs(checkpoints)
    (PACKAGE / "media").mkdir(parents=True)

    inventory: list[dict] = []
    generation: list[dict] = []
    reviews: list[dict] = []
    items: list[dict] = []
    source_media: dict[str, dict] = {}
    for chain_number, drift_id in enumerate(records, 1):
        source = DRIFT / records[drift_id]["source"]["media"]
        destination = PACKAGE / "media" / f"chain{chain_number:02d}_source.mp4"
        info = materialize(source, destination)
        inventory.append({"role": "source", "eval_id": drift_id, **info})
        source_media[drift_id] = info

    for card_number, checkpoint in enumerate(checkpoints, 1):
        drift_id = checkpoint["drift_id"]
        target_round = int(checkpoint["round"])
        record = records[drift_id]
        card_id = f"{drift_id}_r{target_round}"
        source_info = source_media[drift_id]
        column_labels = [
            "R0 source",
            f"最新 direct source（旧正常链 R{target_round - 1}）",
            f"当前 normal_replay（R{target_round}）",
            *(label for _, label in CONDITIONS),
        ]
        rows = [{"label": "Source", "paths": [source_info["path"]] * 7}]
        item_reviews: list[dict] = []
        for model_id, model_label in MODELS:
            direct = direct_source_path(model_id, checkpoint)
            direct_destination = PACKAGE / "media" / f"card{card_number:02d}_{model_id}_direct.mp4"
            direct_info = materialize(direct, direct_destination)
            inventory.append({"role": "direct_source", "card_id": card_id, "eval_id": drift_id,
                              "round": target_round - 1, "model_id": model_id, **direct_info})

            normal = output_path(model_id, "normal_replay", checkpoint)
            normal_destination = PACKAGE / "media" / f"card{card_number:02d}_{model_id}_normal.mp4"
            normal_info = materialize(normal, normal_destination)
            inventory.append({"role": "normal_replay", "card_id": card_id, "eval_id": drift_id,
                              "round": target_round, "model_id": model_id,
                              "condition": "normal_replay", **normal_info})
            paths = [source_info["path"], direct_info["path"], normal_info["path"]]
            generation.append({"card_id": card_id, "eval_id": drift_id, "round": target_round,
                               "model_id": model_id, "condition": "normal_replay",
                               "instruction": record["turns"][target_round - 1]["instruction"],
                               "output_sha256": sha256(normal), "fixed_normal_history": True,
                               "seed": 42, "steps": 30, "shift": 3.5,
                               "cfg_text": 1.0, "cfg_vit": 1.0})
            for condition, condition_label in CONDITIONS:
                raw = output_path(model_id, condition, checkpoint)
                destination = (PACKAGE / "media"
                               / f"card{card_number:02d}_{model_id}_{condition}.mp4")
                info = materialize(raw, destination)
                inventory.append({"role": "ablation", "card_id": card_id, "eval_id": drift_id,
                                  "round": target_round, "model_id": model_id,
                                  "condition": condition, **info})
                paths.append(info["path"])
                generation.append({"card_id": card_id, "eval_id": drift_id, "round": target_round,
                                   "model_id": model_id, "condition": condition,
                                   "instruction": record["turns"][target_round - 1]["instruction"],
                                   "output_sha256": sha256(raw), "fixed_normal_history": True,
                                   "seed": 42, "steps": 30, "shift": 3.5,
                                   "cfg_text": 1.0, "cfg_vit": 1.0})
                entry = review_row(checkpoint, card_id, model_id, model_label,
                                   condition, condition_label)
                reviews.append(entry)
                item_reviews.append(entry)
            if len(paths) != 7:
                raise AssertionError(f"expected seven paths for {card_id}/{model_id}")
            rows.append({"label": model_label, "paths": paths})

        turn_lines = [f"R{i}: {turn['instruction']}" for i, turn in enumerate(
            record["turns"][:target_round], 1)]
        preserve = checkpoint["preserve_rounds"] or "none"
        instruction = "\n".join([
            *turn_lines,
            f"检查 R{target_round} ↔ R{checkpoint['reference_round']}："
            f"{KIND_LABELS[checkpoint['comparison_type']]}；保留轮次 {preserve}",
        ])
        items.append({
            "section": "multiturn",
            "eval_id": card_id,
            "category": f"{checkpoint['pattern']} · {record['focus']}",
            "instruction": instruction,
            "prompts": column_labels[1:],
            "columns": column_labels[1:],
            "rows": rows,
            "reviews": item_reviews,
        })

    if len(items) != 24 or len(reviews) != 192 or len(inventory) != 300:
        raise RuntimeError(f"package shape mismatch: cards={len(items)}, reviews={len(reviews)}, "
                           f"media={len(inventory)}")
    paths = [row["path"] for row in inventory]
    if len(set(paths)) != 300:
        raise RuntimeError("media inventory paths must be unique")

    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    optional_reports = []
    for name in ("difference_metrics.csv", "inference_audit.json"):
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, PACKAGE / name)
            optional_reports.append(name)
    role_counts = {role: sum(item["role"] == role for item in inventory)
                   for role in ("source", "direct_source", "normal_replay", "ablation")}
    summary = {
        "status": "built_pending_browser_validation",
        "human_only": True,
        "gemini_included": False,
        "models": [model_id for model_id, _ in MODELS],
        "conditions": [condition for condition, _ in CONDITIONS],
        "cards": 24,
        "chains": 12,
        "review_units": 192,
        "media_files": 300,
        "media_role_counts": role_counts,
        "normal_replay_outputs": 48,
        "formal_ablation_outputs": 192,
        "optional_reports": optional_reports,
        "comparison_limit": "Human review determines semantic effect; difference metrics only quantify change from normal_replay.",
    }
    write_json(PACKAGE / "summary.json", summary)

    payload = {
        "review_version": PACKAGE_NAME,
        "title": "历史编辑链 ViT 消融 · Base vs gtfresh 人工审核",
        "items": items,
        "reviews": reviews,
    }
    document = TEMPLATE.read_text(encoding="utf-8")
    document = document.replace("真实数据 vs 混合数据 Lance 人工审核", payload["title"])
    old_fields = "const fields=['instruction_correctness','logical_correctness','content_preservation','temporal_consistency','visual_quality','technical_quality','overall_decision'];"
    new_fields = "const fields=['edit_applied','history_state_correct','normal_equivalent','identity_background_stable','temporal_consistency','visual_quality','overall_pass'];"
    old_labels = "const labels={instruction_correctness:'指令完成',logical_correctness:'逻辑正确',content_preservation:'内容保持',temporal_consistency:'时序一致',visual_quality:'视觉质量',technical_quality:'技术质量',overall_decision:'总体结论'};"
    new_labels = "const labels={edit_applied:'当前编辑完成',history_state_correct:'历史状态正确',normal_equivalent:'与正常推理等价',identity_background_stable:'人物背景稳定',temporal_consistency:'时序一致',visual_quality:'视觉质量',overall_pass:'总体通过'};"
    old_header = "x.prompts.map((p,i)=>`<td class=\"prompt\"><b>R${i+1}</b> · ${esc(p)}</td>`)"
    new_header = "(x.columns||x.prompts).map(p=>`<td class=\"prompt\">${esc(p)}</td>`)"
    document = document.replace(old_fields, new_fields).replace(old_labels, new_labels)
    document = document.replace("r.overall_decision!=='1'", "r.overall_pass!=='1'")
    document = document.replace(old_header, new_header)
    document = document.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    if (old_fields in document or old_labels in document or old_header in document
            or "__REVIEW_DATA__" in document):
        raise RuntimeError("review template substitution failed")
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[-1]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# 历史编辑链 ViT 消融人工审核包\n\n"
        "完整解压后用 Chrome 或 Edge 打开 `index.html`，页面完全离线。\n\n"
        "24 张 checkpoint 卡片；每卡依次展示 R0、最新 direct source、normal_replay 和四种消融。"
        "Base 与 gtfresh 各有四个消融审核单元，共 192 个审核单元。所有字段默认 1；"
        "0 表示不通过，2 表示存疑。支持 CSV/JSON 导入导出。\n\n"
        "本包不调用 Gemini。若包内含 `difference_metrics.csv`，它只描述相对 normal_replay 的像素变化，"
        "语义结论仍以人工审核为准。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
