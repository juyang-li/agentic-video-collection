#!/usr/bin/env python3
"""Build the offline IVEBench History-ViT human-review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ivebench24_nonrollback_nocamera_history_vit_ablation_base_gtfresh_human_review_20261017"
PACKAGE = ROOT / "review" / PACKAGE_NAME
TEMPLATE = ROOT.parent / "gtfresh1150_vs_coinve677_20260927/scripts/review_template.html"
MODELS = (("lance_base", "Lance Base"), ("gtfresh_step1150", "gtfresh step1150"))
CONDITIONS = (
    ("vit_zero_text_keep", "历史 ViT=0；历史文本保留"),
    ("vit_gaussian_text_keep", "历史 ViT=匹配高斯；历史文本保留"),
)
FIELDS = (
    "edit_applied", "history_preserved", "normal_equivalent",
    "identity_background_stable", "temporal_consistency", "visual_quality", "overall_pass",
)
EXPECTED = {"chains": 24, "turns": 65, "checkpoints": 41, "reviews": 164, "media": 318}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(model: str, condition: str, chain_id: str, round_no: int) -> Path:
    return ROOT / "outputs" / model / condition / chain_id / f"turn{round_no - 1:02d}.mp4"


def probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", result.stderr)
    if not match:
        raise RuntimeError(f"cannot probe codec: {path}")
    prefix = path.read_bytes()[:2_000_000]
    return {
        "codec_name": match.group(1).strip().split()[0],
        "pix_fmt": match.group(2).split("(")[0],
        "width": width, "height": height, "fps": round(fps, 6), "frames": frames,
        "faststart": prefix.find(b"moov") >= 0
        and (prefix.find(b"mdat") < 0 or prefix.find(b"moov") < prefix.find(b"mdat")),
    }


def full_decode(path: Path) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
         "-map", "0:v:0", "-f", "null", "-"],
        check=True,
    )


class MediaStore:
    """Materialize each distinct content hash once and retain all logical usages."""

    def __init__(self) -> None:
        self.by_hash: dict[str, dict] = {}

    def add(self, source: Path, role: str, usage: dict) -> str:
        if not source.is_file() or source.stat().st_size <= 1000:
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if source_hash in self.by_hash:
            self.by_hash[source_hash]["usages"].append({"role": role, **usage})
            return self.by_hash[source_hash]["path"]
        destination = PACKAGE / "media" / f"{source_hash}.mp4"
        temporary = destination.with_suffix(".tmp.mp4")
        initial = probe(source)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if initial["codec_name"] == "h264" and initial["pix_fmt"] == "yuv420p":
            command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
                       "-c:v", "copy", "-an", "-movflags", "+faststart", str(temporary)]
        else:
            command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temporary)]
        subprocess.run(command, check=True)
        os.replace(temporary, destination)
        full_decode(destination)
        final = probe(destination)
        if final["codec_name"] != "h264" or final["pix_fmt"] != "yuv420p" or not final["faststart"]:
            raise RuntimeError(f"nonportable media: {destination}: {final}")
        entry = {
            "path": destination.relative_to(PACKAGE).as_posix(),
            "source": str(source.resolve()),
            "source_sha256": source_hash,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "full_decode_ok": True,
            "usages": [{"role": role, **usage}],
            **final,
        }
        self.by_hash[source_hash] = entry
        return entry["path"]

    def inventory(self) -> list[dict]:
        return sorted(self.by_hash.values(), key=lambda item: item["path"])


def validate_input() -> list[dict]:
    payload = load_json(ROOT / "benchmark.json")
    cases = payload.get("cases") or []
    shape = (
        len(cases), sum(len(case.get("outputs", [])) for case in cases),
        sum(max(0, len(case.get("outputs", [])) - 1) for case in cases),
    )
    if shape != (EXPECTED["chains"], EXPECTED["turns"], EXPECTED["checkpoints"]):
        raise RuntimeError(f"benchmark shape mismatch: {shape}")
    benchmark_sha = sha256(ROOT / "benchmark.json")
    for model_id, _ in MODELS:
        shard_manifests = [
            load_json(ROOT / "outputs" / model_id / "normal" / "shards" / f"shard{index:02d}-of-02.json")
            for index in range(2)
        ]
        chain_ids = [chain_id for manifest in shard_manifests for chain_id in manifest.get("chain_ids", [])]
        if (
            any(manifest.get("status") != "generated" for manifest in shard_manifests)
            or any(manifest.get("benchmark_sha256") != benchmark_sha for manifest in shard_manifests)
            or sum(int(manifest.get("turn_count", 0)) for manifest in shard_manifests) != 65
            or sorted(chain_ids) != sorted(case["chain_id"] for case in cases)
        ):
            raise RuntimeError(f"normal shard manifest contract failed: {model_id}")
        for condition, _ in CONDITIONS:
            manifest = load_json(ROOT / "outputs" / model_id / condition / "manifest.json")
            contract = (
                manifest.get("status"), manifest.get("condition"), manifest.get("chain_count"),
                manifest.get("total_turns"), manifest.get("formal_checkpoints"),
            )
            if contract != ("generated", condition, 24, 65, 41):
                raise RuntimeError(f"manifest contract failed: {model_id}/{condition}: {contract}")
            if (
                manifest.get("benchmark_sha256") != benchmark_sha
                or manifest.get("checkpoint_count") != 41
                or manifest.get("fixed_normal_history") is not True
            ):
                raise RuntimeError(f"ablation manifest contract failed: {model_id}/{condition}")
    audit = load_json(ROOT / "inference_audit.json")
    if (audit.get("status"), audit.get("metric_rows"), audit.get("formal_ablation_videos")) != (
        "passed", 164, 164
    ):
        raise RuntimeError("inference audit has not passed")
    with (ROOT / "difference_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    metric_keys = {(row["model_id"], row["condition"], row["chain_id"], row["round"]) for row in metrics}
    if len(metrics) != 164 or len(metric_keys) != 164:
        raise RuntimeError("difference_metrics.csv must contain 164 unique rows")
    return cases


def review_row(card_id: str, chain_id: str, round_no: int, model_id: str, model_label: str,
               condition: str, condition_label: str) -> dict:
    return {
        "review_id": f"{card_id}:{model_id}:{condition}", "card_id": card_id,
        "section": "multiturn", "chain_id": chain_id, "round": round_no,
        "model_id": model_id, "model_label": f"{model_label} · {condition_label}",
        "condition": condition, "condition_label": condition_label,
        **{field: "1" for field in FIELDS}, "notes": "",
    }


def generation_row(model: str, condition: str, case: dict, item: dict, raw: Path) -> dict:
    row = {
        "chain_id": case["chain_id"], "round": item["round"], "model_id": model,
        "condition": condition, "instruction": item["instruction"], "output_sha256": sha256(raw),
        "seed": 42, "steps": 30, "shift": 3.5, "cfg_text": 1.0, "cfg_vit": 1.0,
    }
    if condition != "normal":
        receipt = load_json(raw.with_suffix(".json"))
        row.update(fixed_normal_history=True, intervention_trace=receipt.get("trace"))
    return row


def main() -> None:
    archive = PACKAGE.with_suffix(".zip")
    if PACKAGE.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite: {PACKAGE} or {archive}")
    cases = validate_input()
    (PACKAGE / "media").mkdir(parents=True)
    media = MediaStore()
    source_paths: dict[str, str] = {}
    normal_paths: dict[tuple[str, str, int], str] = {}
    ablation_paths: dict[tuple[str, str, str, int], str] = {}
    generation: list[dict] = []

    for case in cases:
        source_paths[case["chain_id"]] = media.add(
            Path(case["source"]), "source", {"chain_id": case["chain_id"]}
        )
    for model_id, _ in MODELS:
        for case in cases:
            for item in case["outputs"]:
                raw = output_path(model_id, "normal", case["chain_id"], item["round"])
                normal_paths[(model_id, case["chain_id"], item["round"])] = media.add(
                    raw, "normal", {"model_id": model_id, "chain_id": case["chain_id"], "round": item["round"]}
                )
                generation.append(generation_row(model_id, "normal", case, item, raw))
            for condition, _ in CONDITIONS:
                for item in case["outputs"][1:]:
                    raw = output_path(model_id, condition, case["chain_id"], item["round"])
                    key = (model_id, condition, case["chain_id"], item["round"])
                    ablation_paths[key] = media.add(
                        raw, "ablation", {"model_id": model_id, "condition": condition,
                                          "chain_id": case["chain_id"], "round": item["round"]}
                    )
                    generation.append(generation_row(model_id, condition, case, item, raw))

    reviews: list[dict] = []
    items: list[dict] = []
    checkpoints = [(case, item["round"]) for case in cases for item in case["outputs"][1:]]
    for case, target_round in checkpoints:
        chain_id = case["chain_id"]
        card_id = f"{chain_id}_r{target_round}"
        columns = [
            "R0 原始视频", f"最新 direct source（normal R{target_round - 1}）",
            f"当前 normal（R{target_round}）", *(label for _, label in CONDITIONS),
        ]
        source = source_paths[chain_id]
        rows = [{"label": "Source", "paths": [source] * len(columns)}]
        item_reviews = []
        for model_id, model_label in MODELS:
            paths = [
                source,
                normal_paths[(model_id, chain_id, target_round - 1)],
                normal_paths[(model_id, chain_id, target_round)],
                *(ablation_paths[(model_id, condition, chain_id, target_round)]
                  for condition, _ in CONDITIONS),
            ]
            rows.append({"label": model_label, "paths": paths})
            for condition, condition_label in CONDITIONS:
                entry = review_row(
                    card_id, chain_id, target_round, model_id, model_label, condition, condition_label
                )
                reviews.append(entry)
                item_reviews.append(entry)
        history = [
            f"R{item['round']}: {item['instruction']}" for item in case["outputs"][:target_round]
        ]
        items.append({
            "section": "multiturn", "eval_id": card_id,
            "category": "IVEBench 非回撤连续编辑 · 无 camera turn",
            "instruction": "\n".join([*history, f"当前检查：R{target_round}"]),
            "prompts": columns[1:], "columns": columns[1:], "rows": rows,
            "reviews": item_reviews,
        })

    inventory = media.inventory()
    if len(items) != 41 or len(reviews) != 164 or len(generation) != 294 or len(inventory) != 318:
        raise RuntimeError(
            f"package shape mismatch: cards={len(items)}, reviews={len(reviews)}, "
            f"generation={len(generation)}, physical_media={len(inventory)}"
        )
    role_counts = {
        role: sum(any(usage["role"] == role for usage in item["usages"]) for item in inventory)
        for role in ("source", "normal", "ablation")
    }
    if role_counts != {"source": 24, "normal": 130, "ablation": 164}:
        raise RuntimeError(f"unexpected physical media role counts: {role_counts}")

    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "benchmark.json", PACKAGE / "benchmark.json")
    shutil.copy2(ROOT / "difference_metrics.csv", PACKAGE / "difference_metrics.csv")
    shutil.copy2(ROOT / "inference_audit.json", PACKAGE / "inference_audit.json")
    summary = {
        "status": "built_pending_browser_validation", "human_only": True,
        "gemini_included": False, "gt_videos": 0,
        "models": [item[0] for item in MODELS], "conditions": [item[0] for item in CONDITIONS],
        "chains": 24, "turns": 65, "cards": 41, "review_units": 164,
        "media_files": 318, "media_role_counts": role_counts,
        "normal_outputs": 130, "formal_ablation_outputs": 164, "difference_metric_rows": 164,
        "comparison_limit": "Metrics quantify change from normal; semantic effect is determined by human review.",
    }
    write_json(PACKAGE / "summary.json", summary)

    payload = {
        "review_version": PACKAGE_NAME,
        "title": "IVEBench 非回撤无 camera 编辑 · 历史 ViT 消融人工审核",
        "items": items, "reviews": reviews,
    }
    document = TEMPLATE.read_text(encoding="utf-8")
    document = document.replace("真实数据 vs 混合数据 Lance 人工审核", payload["title"])
    old_fields = "const fields=['instruction_correctness','logical_correctness','content_preservation','temporal_consistency','visual_quality','technical_quality','overall_decision'];"
    new_fields = "const fields=['edit_applied','history_preserved','normal_equivalent','identity_background_stable','temporal_consistency','visual_quality','overall_pass'];"
    old_labels = "const labels={instruction_correctness:'指令完成',logical_correctness:'逻辑正确',content_preservation:'内容保持',temporal_consistency:'时序一致',visual_quality:'视觉质量',technical_quality:'技术质量',overall_decision:'总体结论'};"
    new_labels = "const labels={edit_applied:'当前编辑完成',history_preserved:'历史编辑保留',normal_equivalent:'与 normal 等价',identity_background_stable:'人物背景稳定',temporal_consistency:'时序一致',visual_quality:'视觉质量',overall_pass:'总体通过'};"
    old_header = "x.prompts.map((p,i)=>`<td class=\"prompt\"><b>R${i+1}</b> · ${esc(p)}</td>`)"
    new_header = "(x.columns||x.prompts).map(p=>`<td class=\"prompt\">${esc(p)}</td>`)"
    document = document.replace(old_fields, new_fields).replace(old_labels, new_labels)
    document = document.replace("r.overall_decision!=='1'", "r.overall_pass!=='1'")
    document = document.replace(old_header, new_header)
    document = document.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    if any(marker in document for marker in (old_fields, old_labels, old_header, "__REVIEW_DATA__")):
        raise RuntimeError("review template substitution failed")
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", document, re.S)
    subprocess.run(["node", "--check"], input=scripts[-1], text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# IVEBench 历史 ViT 消融人工审核包\n\n"
        "完整解压后用 Chrome 或 Edge 打开 `index.html`，页面完全离线。\n\n"
        "共 24 条非回撤链、65 turns、41 个 R2+ checkpoint；camera turn 已移除。每张卡按 Base、gtfresh 两行展示原视频、最新 direct source、normal、历史 ViT zero 和匹配高斯。共 164 个审核单元，字段默认 1；0 为不通过，2 为存疑。\n\n"
        "本包不含 GT，不调用 Gemini。两个模型只能分别观察自身消融效应；`difference_metrics.csv` 仅描述相对同模型 normal 的像素变化，语义结论以人工审核为准。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
