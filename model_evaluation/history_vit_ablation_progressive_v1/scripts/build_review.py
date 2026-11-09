#!/usr/bin/env python3
"""Build the offline human-review package for the no-camera ablation."""

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
PACKAGE_NAME = "ue_ordinary_no_camera_history_vit_ablation_base_gtfresh_human_review_20261016"
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
FIELDS = (
    "edit_applied",
    "history_preserved",
    "normal_equivalent",
    "identity_background_stable",
    "temporal_consistency",
    "visual_quality",
    "overall_pass",
)


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


def normal_path(model_id: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model_id / "normal" / chain_id / f"turn{round_no - 1:02d}.mp4"


def ablation_path(model_id: str, condition: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model_id / condition / chain_id / f"turn{round_no - 1:02d}.mp4"


def case_signature(cases: list[dict]) -> list[tuple]:
    return [(case["chain_id"], case["source_sha256"],
             tuple(item["instruction"] for item in case["outputs"])) for case in cases]


def adapter_ok(model_id: str, manifest: dict) -> None:
    check = manifest.get("adapter_check")
    if model_id == "lance_base":
        if check is not None:
            raise RuntimeError("Lance Base unexpectedly reports an adapter")
        return
    check = check or {}
    values = (check.get("tensor_count"), check.get("matching_after_cast"), check.get("nonzero_lora_B"))
    if values != (1016, 1016, 503) or check.get("missing_live_keys", 0) != 0:
        raise RuntimeError(f"gtfresh adapter verification missing: {values}")


def load_and_validate() -> list[dict]:
    base_manifest = json.loads(
        (ROOT / "outputs/lance_base/normal/manifest.json").read_text(encoding="utf-8"))
    cases = base_manifest.get("cases") or []
    if (len(cases), sum(len(case.get("outputs", [])) for case in cases),
            sum(max(0, len(case.get("outputs", [])) - 1) for case in cases)) != (3, 9, 6):
        raise RuntimeError("expected three chains, nine turns, and six R2+ checkpoints")
    signature = case_signature(cases)
    for model_id, _ in MODELS:
        normal_manifest = json.loads(
            (ROOT / "outputs" / model_id / "normal" / "manifest.json").read_text(encoding="utf-8"))
        normal_contract = (
            normal_manifest.get("status"), normal_manifest.get("condition"),
            normal_manifest.get("chain_count"), normal_manifest.get("total_turns"),
            normal_manifest.get("formal_checkpoints"),
        )
        if normal_contract != ("generated", "normal", 3, 9, 6):
            raise RuntimeError(f"bad normal manifest: {model_id}: {normal_contract}")
        if case_signature(normal_manifest.get("cases") or []) != signature:
            raise RuntimeError(f"case mismatch: {model_id}/normal")
        adapter_ok(model_id, normal_manifest)
        for case in cases:
            for round_no in range(1, len(case["outputs"]) + 1):
                video = normal_path(model_id, case["chain_id"], round_no)
                if not video.is_file() or video.stat().st_size <= 1000:
                    raise FileNotFoundError(video)
        for condition, _ in CONDITIONS:
            manifest = json.loads(
                (ROOT / "outputs" / model_id / condition / "manifest.json").read_text(encoding="utf-8"))
            contract = (
                manifest.get("status"), manifest.get("condition"), manifest.get("chain_count"),
                manifest.get("total_turns"), manifest.get("formal_checkpoints"),
                manifest.get("checkpoint_count"), manifest.get("fixed_normal_history"),
            )
            if contract != ("generated", condition, 3, 9, 6, 6, True):
                raise RuntimeError(f"bad ablation manifest: {model_id}/{condition}: {contract}")
            if case_signature(manifest.get("cases") or []) != signature:
                raise RuntimeError(f"case mismatch: {model_id}/{condition}")
            adapter_ok(model_id, manifest)
            for case in cases:
                for round_no in range(2, len(case["outputs"]) + 1):
                    video = ablation_path(model_id, condition, case["chain_id"], round_no)
                    receipt = video.with_suffix(".json")
                    if not video.is_file() or video.stat().st_size <= 1000 or not receipt.is_file():
                        raise FileNotFoundError(f"missing output or receipt: {video}")
                    data = json.loads(receipt.read_text(encoding="utf-8"))
                    expected = ("generated", model_id, condition, case["chain_id"], round_no, sha256(video))
                    actual = (data.get("status"), data.get("model_id"), data.get("condition"),
                              data.get("chain_id"), data.get("round"), data.get("output_sha256"))
                    if actual != expected:
                        raise RuntimeError(f"receipt mismatch: {receipt}")
    audit = json.loads((ROOT / "inference_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("metric_rows") != 48:
        raise RuntimeError("inference audit has not passed")
    with (ROOT / "difference_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    if len(metrics) != 48 or len({(row["model_id"], row["condition"], row["chain_id"], row["round"])
                                  for row in metrics}) != 48:
        raise RuntimeError("difference_metrics.csv must contain 48 unique rows")
    return cases


def review_row(card_id: str, chain_id: str, round_no: int, model_id: str, model_label: str,
               condition: str, condition_label: str) -> dict:
    return {
        "review_id": f"{card_id}:{model_id}:{condition}",
        "card_id": card_id,
        "section": "multiturn",
        "chain_id": chain_id,
        "round": round_no,
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
    cases = load_and_validate()
    (PACKAGE / "media").mkdir(parents=True)

    inventory: list[dict] = []
    generation: list[dict] = []
    reviews: list[dict] = []
    items: list[dict] = []
    source_media: dict[str, dict] = {}
    for chain_number, case in enumerate(cases, 1):
        source = Path(case["source"])
        destination = PACKAGE / "media" / f"chain{chain_number:02d}_source.mp4"
        info = materialize(source, destination)
        inventory.append({"role": "source", "chain_id": case["chain_id"], **info})
        source_media[case["chain_id"]] = info

    checkpoints = [(case, round_no) for case in cases
                   for round_no in range(2, len(case["outputs"]) + 1)]
    for card_number, (case, target_round) in enumerate(checkpoints, 1):
        chain_id = case["chain_id"]
        card_id = f"{chain_id}_r{target_round}"
        source_info = source_media[chain_id]
        columns = [
            "R0 source",
            f"最新 direct source（正常链 R{target_round - 1}）",
            f"当前 normal（R{target_round}）",
            *(label for _, label in CONDITIONS),
        ]
        rows = [{"label": "Source", "paths": [source_info["path"]] * 7}]
        item_reviews = []
        for model_id, model_label in MODELS:
            direct = normal_path(model_id, chain_id, target_round - 1)
            direct_info = materialize(
                direct, PACKAGE / "media" / f"card{card_number:02d}_{model_id}_direct.mp4")
            inventory.append({"role": "direct_source", "card_id": card_id, "chain_id": chain_id,
                              "round": target_round - 1, "model_id": model_id, **direct_info})
            normal = normal_path(model_id, chain_id, target_round)
            normal_info = materialize(
                normal, PACKAGE / "media" / f"card{card_number:02d}_{model_id}_normal.mp4")
            inventory.append({"role": "normal", "card_id": card_id, "chain_id": chain_id,
                              "round": target_round, "model_id": model_id, **normal_info})
            paths = [source_info["path"], direct_info["path"], normal_info["path"]]
            generation.append({
                "card_id": card_id, "chain_id": chain_id, "round": target_round,
                "model_id": model_id, "condition": "normal",
                "instruction": case["outputs"][target_round - 1]["instruction"],
                "output_sha256": sha256(normal), "fixed_normal_history": True,
                "seed": 42, "steps": 30, "shift": 3.5, "cfg_text": 1.0, "cfg_vit": 1.0,
            })
            for condition, condition_label in CONDITIONS:
                raw = ablation_path(model_id, condition, chain_id, target_round)
                info = materialize(
                    raw, PACKAGE / "media" / f"card{card_number:02d}_{model_id}_{condition}.mp4")
                inventory.append({"role": "ablation", "card_id": card_id, "chain_id": chain_id,
                                  "round": target_round, "model_id": model_id,
                                  "condition": condition, **info})
                paths.append(info["path"])
                generation.append({
                    "card_id": card_id, "chain_id": chain_id, "round": target_round,
                    "model_id": model_id, "condition": condition,
                    "instruction": case["outputs"][target_round - 1]["instruction"],
                    "output_sha256": sha256(raw), "fixed_normal_history": True,
                    "seed": 42, "steps": 30, "shift": 3.5, "cfg_text": 1.0, "cfg_vit": 1.0,
                })
                entry = review_row(card_id, chain_id, target_round, model_id, model_label,
                                   condition, condition_label)
                reviews.append(entry)
                item_reviews.append(entry)
            if len(paths) != 7:
                raise AssertionError(f"expected seven paths for {card_id}/{model_id}")
            rows.append({"label": model_label, "paths": paths})

        turn_lines = [
            f"R{i}: {item['instruction']}" for i, item in enumerate(case["outputs"][:target_round], 1)
        ]
        items.append({
            "section": "multiturn",
            "eval_id": card_id,
            "category": "UE 普通连续编辑 · 无相机编辑",
            "instruction": "\n".join([*turn_lines, f"当前检查：R{target_round}"]),
            "prompts": columns[1:],
            "columns": columns[1:],
            "rows": rows,
            "reviews": item_reviews,
        })

    if len(items) != 6 or len(reviews) != 48 or len(inventory) != 75:
        raise RuntimeError(f"package shape mismatch: cards={len(items)}, reviews={len(reviews)}, "
                           f"media={len(inventory)}")
    if len({row["path"] for row in inventory}) != 75:
        raise RuntimeError("media inventory paths must be unique")

    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "difference_metrics.csv", PACKAGE / "difference_metrics.csv")
    shutil.copy2(ROOT / "inference_audit.json", PACKAGE / "inference_audit.json")

    role_counts = {role: sum(item["role"] == role for item in inventory)
                   for role in ("source", "direct_source", "normal", "ablation")}
    summary = {
        "status": "built_pending_browser_validation",
        "human_only": True,
        "gemini_included": False,
        "gt_videos": 0,
        "models": [model_id for model_id, _ in MODELS],
        "conditions": [condition for condition, _ in CONDITIONS],
        "chains": 3,
        "cards": 6,
        "review_units": 48,
        "media_files": 75,
        "media_role_counts": role_counts,
        "normal_comparison_outputs": 12,
        "formal_ablation_outputs": 48,
        "difference_metric_rows": 48,
        "comparison_limit": "Metrics quantify change from normal; semantic effect is determined by human review.",
    }
    write_json(PACKAGE / "summary.json", summary)

    payload = {
        "review_version": PACKAGE_NAME,
        "title": "UE 普通连续编辑（无相机编辑）· 历史 ViT 消融人工审核",
        "items": items,
        "reviews": reviews,
    }
    document = TEMPLATE.read_text(encoding="utf-8")
    document = document.replace("真实数据 vs 混合数据 Lance 人工审核", payload["title"])
    old_fields = "const fields=['instruction_correctness','logical_correctness','content_preservation','temporal_consistency','visual_quality','technical_quality','overall_decision'];"
    new_fields = "const fields=['edit_applied','history_preserved','normal_equivalent','identity_background_stable','temporal_consistency','visual_quality','overall_pass'];"
    old_labels = "const labels={instruction_correctness:'指令完成',logical_correctness:'逻辑正确',content_preservation:'内容保持',temporal_consistency:'时序一致',visual_quality:'视觉质量',technical_quality:'技术质量',overall_decision:'总体结论'};"
    new_labels = "const labels={edit_applied:'当前编辑完成',history_preserved:'历史编辑保留',normal_equivalent:'与正常推理等价',identity_background_stable:'人物背景稳定',temporal_consistency:'时序一致',visual_quality:'视觉质量',overall_pass:'总体通过'};"
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
        "# UE 普通连续编辑（无相机编辑）历史 ViT 消融审核包\n\n"
        "完整解压后用 Chrome 或 Edge 打开 `index.html`，页面完全离线。\n\n"
        "共 3 条链、6 个 R2+ checkpoint。每张卡展示 source、最新 direct source、normal 和四种历史消融；"
        "Base 与 gtfresh 各四个审核单元，共 48 个。所有字段默认 1，0 为不通过，2 为存疑。\n\n"
        "本包不含 GT 视频，不调用 Gemini。`difference_metrics.csv` 仅描述相对 normal 的像素变化，"
        "语义结论以人工审核为准。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
