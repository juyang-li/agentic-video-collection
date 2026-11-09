#!/usr/bin/env python3
"""Build the complete offline IVEBench CoinVE ablation review package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from build_short_subject_review import codec, faststart, prepare_video, probe


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/coinve_ablation_v1"
VERSION = "ivebench_coinve_ablation228_base_single_coinve_mixed_step677_hq_v2"
EXPECTED_SAMPLES = 228
MEDIA_KEYS = ("source", "base", "single4k_step677", "coinve4k_step677", "mixed2k2k_step677")
MODEL_LABELS = {
    "base": "Base",
    "single4k_step677": "Single-4K",
    "coinve4k_step677": "CoinVE-4K",
    "mixed2k2k_step677": "Mixed-4K",
    "tie": "平局",
    "order_inconsistent": "顺序不一致",
}
PAIRS = (
    ("base_vs_single", "base", "single4k_step677"),
    ("base_vs_coinve", "base", "coinve4k_step677"),
    ("base_vs_mixed", "base", "mixed2k2k_step677"),
    ("single_vs_coinve", "single4k_step677", "coinve4k_step677"),
    ("single_vs_mixed", "single4k_step677", "mixed2k2k_step677"),
    ("coinve_vs_mixed", "coinve4k_step677", "mixed2k2k_step677"),
)
PRIOR_REVIEW = {
    "short_subject": "ivebench_short_subject91_base_vs_lora_step677_seed42_gemini38flash_temp0_v1",
    "long_subject": "ivebench_long_subject43_base_vs_lora_step677_seed42_gemini38flash_temp0_v1",
    "attribute_all": "ivebench_attribute94_base_vs_lora_step677_seed42_gemini38flash_temp0_v1",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def canonical_model(value: str) -> str:
    return "single4k_step677" if value == "lora_step677" else value


def quarantine(path: Path, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.parent / f"{path.name}.{reason}.{stamp}.{os.getpid()}"
    path.replace(destination)
    return destination


def load_judgments() -> tuple[dict[tuple[str, frozenset[str]], list[dict]], dict]:
    sources = [
        (ARTIFACT / "gemini_pairwise_blind_v1/scores.csv", None),
        (ARTIFACT / "gemini_pairwise_vs_base_blind_v1/scores.csv", None),
    ]
    for stage in ("short_subject", "long_subject", "attribute_all"):
        sources.append((ROOT / f"artifacts/{stage}/gemini_pairwise_temp0_v1/scores.csv", stage))
    grouped: dict[tuple[str, frozenset[str]], list[dict]] = defaultdict(list)
    raw_rows = 0
    for path, forced_stage in sources:
        for row in read_csv(path):
            raw_rows += 1
            normalized = dict(row)
            if forced_stage:
                normalized["stage"] = forced_stage
            for key in ("candidate_a_model", "candidate_b_model", "normalized_instruction_winner",
                        "normalized_fidelity_winner", "normalized_overall_winner"):
                normalized[key] = canonical_model(normalized[key])
            models = frozenset((normalized["candidate_a_model"], normalized["candidate_b_model"]))
            grouped[(normalized["sample_id"], models)].append(normalized)
    if raw_rows != EXPECTED_SAMPLES * len(PAIRS) * 2:
        raise ValueError(f"expected 2736 pairwise score rows, got {raw_rows}")
    malformed = []
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["order"]))
        if len(rows) != 2 or [int(row["order"]) for row in rows] != [1, 2]:
            malformed.append(str(key))
    if len(grouped) != EXPECTED_SAMPLES * len(PAIRS) or malformed:
        raise ValueError(f"malformed bidirectional groups: groups={len(grouped)} errors={malformed[:3]}")
    historic_cost = 0.0
    for stage in ("short_subject", "long_subject", "attribute_all"):
        historic_cost += json.loads(
            (ROOT / f"artifacts/{stage}/gemini_pairwise_temp0_v1/summary.json").read_text()
        )["reported_cost_usd"]
    current_cost = sum(
        json.loads((ARTIFACT / path).read_text())["reported_cost_usd"]
        for path in ("gemini_pairwise_blind_v1/summary.json", "gemini_pairwise_vs_base_blind_v1/summary.json")
    )
    return grouped, {
        "current_tasks": 2280,
        "current_cost_usd": current_cost,
        "historic_base_single_tasks": 456,
        "historic_base_single_cost_usd": historic_cost,
        "all_tasks": raw_rows,
        "all_cost_usd": current_cost + historic_cost,
    }


def pair_record(rows: list[dict], slug: str, left: str, right: str) -> dict:
    def consensus(field: str) -> str:
        values = [row[f"normalized_{field}_winner"] for row in rows]
        return values[0] if values[0] == values[1] else "order_inconsistent"

    orders = []
    for row in rows:
        orders.append({
            "order": int(row["order"]),
            "candidate_a": row["candidate_a_model"],
            "candidate_b": row["candidate_b_model"],
            "instruction_preference": row["instruction_preference"],
            "fidelity_preference": row["fidelity_preference"],
            "overall_preference": row["overall_preference"],
            "reason": row["reason"],
        })
    return {
        "slug": slug,
        "left": left,
        "right": right,
        "label": f"{MODEL_LABELS[left]} vs {MODEL_LABELS[right]}",
        "instruction": consensus("instruction"),
        "fidelity": consensus("fidelity"),
        "overall": consensus("overall"),
        "orders": orders,
    }


def copy_media(item: dict, output: Path) -> tuple[dict, dict]:
    sequence = int(item["sequence_index"])
    output_id = int(item["generation_id"])
    media_paths: dict[str, str] = {}
    inventory = []
    signatures = set()
    for key in MEDIA_KEYS:
        stage_artifact = ROOT / "artifacts" / item["stage"]
        if key in ("source", "base", "single4k_step677"):
            suffix = {"source": "source", "base": "base", "single4k_step677": "lora_step677"}[key]
            source = (
                stage_artifact / "review" / PRIOR_REVIEW[item["stage"]] / "media"
                / f"{output_id:04d}_{suffix}.mp4"
            )
            origin = "lance_actual_input_frames" if key == "source" else "prior_hq_review_bit_identical"
        else:
            matches = list(
                (stage_artifact / "outputs_raw").glob(
                    f"{item['stage']}_{key}_shard*/{output_id:06d}.mp4"
                )
            )
            if len(matches) != 1:
                raise ValueError(f"expected one raw output: {item['sample_id']} {key} {matches}")
            source = matches[0]
            origin = "raw_lance_inference_output"
        if not source.is_file():
            raise FileNotFoundError(source)
        original_sha = digest(source)
        destination = output / "media" / f"{sequence:04d}_{item['sample_id']}_{key}.mp4"
        if key in ("source", "base", "single4k_step677"):
            shutil.copyfile(source, destination)
            encoding = "bit_identical_copy_from_prior_hq_review"
        else:
            encoding = prepare_video(source, destination)
        info = probe(destination)
        copied_sha = info["sha256"]
        if codec(destination) != ("h264", "yuv420p") or not faststart(destination):
            raise ValueError(f"browser-incompatible review media: {destination}")
        relative = destination.relative_to(output).as_posix()
        media_paths[key] = relative
        signature = (info["frames"], float(info["fps"]), info["width"], info["height"])
        signatures.add(signature)
        inventory.append({
            **info, "sample_id": item["sample_id"], "role": key, "path": relative,
            "sha256": copied_sha, "origin": origin, "origin_path": str(source.resolve()),
            "origin_sha256": original_sha, "encoding": encoding,
            "full_decode_ok": True, "codec": "h264", "pixel_format": "yuv420p", "faststart": True,
        })
    if len(signatures) != 1:
        raise ValueError(f"five-way media is not aligned: {item['sample_id']} {signatures}")
    return media_paths, {"items": inventory, "signature": next(iter(signatures))}


def flatten_row(item: dict) -> dict:
    row = {
        "sequence_index": item["sequence_index"], "sample_id": item["sample_id"],
        "stage": item["stage"], "official_id": item["official_id"],
        "subcategory": item["subcategory"], "instruction": item["instruction"],
        "review_source_path": item["media"]["source"], "review_base_path": item["media"]["base"],
        "review_single4k_path": item["media"]["single4k_step677"],
        "review_coinve4k_path": item["media"]["coinve4k_step677"],
        "review_mixed4k_path": item["media"]["mixed2k2k_step677"],
        "frames": item["frames"], "fps": item["fps"], "width": item["width"], "height": item["height"],
        "human_judge_correctness": "1", "human_media_technical_quality": "1",
        "human_best_model": "", "human_notes": "",
    }
    for pair in item["pairs"]:
        prefix = pair["slug"]
        row[f"{prefix}_instruction_consensus"] = pair["instruction"]
        row[f"{prefix}_fidelity_consensus"] = pair["fidelity"]
        row[f"{prefix}_overall_consensus"] = pair["overall"]
        for order in pair["orders"]:
            number = order["order"]
            row[f"{prefix}_order{number}_assignment"] = (
                f"A={MODEL_LABELS[order['candidate_a']]}; B={MODEL_LABELS[order['candidate_b']]}"
            )
            row[f"{prefix}_order{number}_preferences"] = (
                f"instruction={order['instruction_preference']}; fidelity={order['fidelity_preference']}; "
                f"overall={order['overall_preference']}"
            )
            row[f"{prefix}_order{number}_reason"] = order["reason"]
    return row


def aggregate(items: list[dict], costs: dict) -> dict:
    pairs = {}
    for slug, left, right in PAIRS:
        counts = {dimension: Counter() for dimension in ("instruction", "fidelity", "overall")}
        for item in items:
            pair = next(value for value in item["pairs"] if value["slug"] == slug)
            for dimension in counts:
                counts[dimension][pair[dimension]] += 1
        pairs[slug] = {
            "left": left, "right": right, "label": f"{MODEL_LABELS[left]} vs {MODEL_LABELS[right]}",
            "counts": {dimension: dict(value) for dimension, value in counts.items()},
            "order_consistency": {
                dimension: 1 - value.get("order_inconsistent", 0) / len(items)
                for dimension, value in counts.items()
            },
        }
    return {"samples": len(items), "pairs": pairs, **costs}


def copy_evidence(output: Path) -> None:
    results = output / "results"
    results.mkdir()
    for name in (
        "pairwise_report.json", "pairwise_report.md", "pairwise_vs_base_report.json",
        "pairwise_vs_base_report.md", "completion_audit.json", "completion_audit_vs_base.json",
        "generation_plan.json", "generation_status.json", "pairwise_preparation.json",
        "pairwise_vs_base_preparation.json", "phase_manifest.json",
    ):
        shutil.copyfile(ARTIFACT / name, results / name)
    for label, source in (
        ("current_three_finetunes", ARTIFACT / "gemini_pairwise_blind_v1"),
        ("current_vs_base", ARTIFACT / "gemini_pairwise_vs_base_blind_v1"),
    ):
        destination = results / label
        destination.mkdir()
        for name in ("scores.csv", "summary.json", "protocol.json"):
            shutil.copyfile(source / name, destination / name)
    historic = results / "historical_base_vs_single"
    historic.mkdir()
    for stage in ("short_subject", "long_subject", "attribute_all"):
        destination = historic / stage
        destination.mkdir()
        source = ROOT / f"artifacts/{stage}/gemini_pairwise_temp0_v1"
        for name in ("scores.csv", "summary.json", "protocol.json"):
            shutil.copyfile(source / name, destination / name)


def build(output: Path, limit: int | None) -> None:
    media_manifest = json.loads((ARTIFACT / "prepared_pairwise_media.json").read_text())
    if len(media_manifest) != EXPECTED_SAMPLES or len({row["sample_id"] for row in media_manifest}) != EXPECTED_SAMPLES:
        raise ValueError("prepared media manifest is incomplete")
    groups, costs = load_judgments()
    generation_ids = {}
    for stage in PRIOR_REVIEW:
        for row in read_csv(ROOT / f"artifacts/{stage}/sample_results.csv"):
            generation_ids[(stage, row["sample_id"])] = int(row["generation_id"])
    chosen = media_manifest[:limit] if limit else media_manifest
    selected = [
        dict(item, sequence_index=index, generation_id=generation_ids[(item["stage"], item["sample_id"])])
        for index, item in enumerate(chosen, 1)
    ]
    if output.exists():
        quarantine(output, "interrupted")
    output.mkdir(parents=True)
    (output / "media").mkdir()
    items, inventory = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        copied = list(pool.map(lambda item: copy_media(item, output), selected))
    for index, (source, (paths, media_info)) in enumerate(zip(selected, copied), 1):
        pairs = []
        for slug, left, right in PAIRS:
            key = (source["sample_id"], frozenset((left, right)))
            pairs.append(pair_record(groups[key], slug, left, right))
        frames, fps, width, height = media_info["signature"]
        items.append({
            "sequence_index": index, "sample_id": source["sample_id"], "stage": source["stage"],
            "official_id": source["official_id"], "subcategory": source["subcategory"],
            "instruction": source["instruction"], "media": paths, "pairs": pairs,
            "frames": frames, "fps": fps, "width": width, "height": height,
        })
        inventory.extend(media_info["items"])
        print(f"prepared review {index}/{len(selected)}", flush=True)
    summary = aggregate(items, costs)
    survey = {
        "review_version": output.name, "title": "IVEBench CoinVE 数据消融 · 228 条",
        "model_labels": MODEL_LABELS, "pair_order": [pair[0] for pair in PAIRS],
        "summary": summary, "items": items,
    }
    payload = json.dumps(survey, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("coinve_ablation_review.html").read_text(encoding="utf-8")
    document = template.replace("__SURVEY_DATA__", payload)
    if "__SURVEY_DATA__" in document:
        raise ValueError("template substitution failed")
    for script in re.findall(r"<script(?: [^>]*)?>(.*?)</script>", document, re.S)[1:]:
        subprocess.run(["node", "--check"], input=script, text=True, check=True)
    (output / "index.html").write_text(document, encoding="utf-8")
    write_json(output / "review_data.json", survey)
    rows = [flatten_row(item) for item in items]
    with (output / "review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "media_inventory.json", inventory)
    copy_evidence(output)
    write_json(output / "review_validation.json", {
        "status": "passed", "samples": len(items), "videos": len(inventory),
        "pairwise_score_rows": len(items) * len(PAIRS) * 2,
        "all_package_media_full_decode_verified": True,
        "all_five_way_media_temporally_aligned": True,
        "all_media_h264_yuv420p_faststart": True,
        "source_is_lance_actual_frozen_input_frames": True,
        "candidate_media_is_848x480_12fps_raw_output_or_lossless_remux": True,
        "javascript_syntax_verified": True, "browser_test": "pending",
        "production_complete": len(items) == EXPECTED_SAMPLES,
    })
    (output / "README.md").write_text(f"""# IVEBench CoinVE 数据消融审核包

请先完整解压，再用 Chrome 或 Edge 打开 `index.html`。页面完全离线，可连续滚动查看 {len(items)} 条样本。每条从左到右依次为时间严格对齐的 Source、Base、Single-4K、CoinVE-4K、Mixed-4K，支持五路同步播放、暂停和拖动。

页面展示 Gemini 3.8 Flash（low、temperature=0、固定 Google AI Studio 路由）的六组完整两两比较。每组都包含指令完成度、内容保真度、综合偏好的双顺序共识，以及 A/B、B/A 两次完整理由。本轮新增 CoinVE/Mixed 相关任务 2,280 次，实付 ${costs['current_cost_usd']:.6f}；Base/Single 的 456 次结果来自先前同协议评测，连同后总计 2,736 次、${costs['all_cost_usd']:.6f}。

人工审核字段采用 1=通过、0=不通过、2=存疑，新记录默认均为 1；也可选择人工最佳模型并填写备注。浏览器会写入 localStorage，请在移动目录或换浏览器前导出 CSV。原始聚合报告、逐任务分数、协议、费用和完成审计位于 `results/`。

审核视频为 Lance 实际使用/产生的 848×480、12 fps 时间对齐序列。Source 是 Lance 实际读取的冻结输入帧；Base、Single-4K 复用此前已校验的高质量审核副本；CoinVE-4K、Mixed-4K 来自本轮原始推理输出，仅在需要时做无损视频流 remux。它们不是发送给 Gemini 的 426×240、5 fps 裁判代理。`media_inventory.json` 记录所有媒体身份；包内没有 API key、模型权重或账户配置，也不会发起网络请求。
""", encoding="utf-8")
    print(json.dumps({"status": "ready_for_browser_test", "output": str(output), "samples": len(items), "videos": len(inventory)}, ensure_ascii=False))


def validate_archive(archive: Path, package_root: str) -> None:
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        names = bundle.namelist()
        if len([name for name in names if name.endswith(".mp4")]) != EXPECTED_SAMPLES * len(MEDIA_KEYS):
            raise ValueError("ZIP contains the wrong number of videos")
        if {name.split("/")[0] for name in names} != {package_root}:
            raise ValueError("ZIP must contain exactly one package root")


def package(output: Path) -> None:
    rows = read_csv(output / "review_manifest.csv")
    inventory = json.loads((output / "media_inventory.json").read_text())
    browser = json.loads((output / "browser_validation.json").read_text())
    if len(rows) != EXPECTED_SAMPLES or len(inventory) != EXPECTED_SAMPLES * len(MEDIA_KEYS):
        raise ValueError("refusing to package an incomplete review")
    if browser.get("status") != "passed" or browser.get("sample_count") != EXPECTED_SAMPLES or browser.get("external_requests"):
        raise ValueError("real-browser acceptance has not passed")
    for item in inventory:
        if digest(output / item["path"]) != item["sha256"]:
            raise ValueError(f"media hash mismatch: {item['path']}")
    validation = json.loads((output / "review_validation.json").read_text())
    validation["browser_test"] = "passed; see browser_validation.json"
    write_json(output / "review_validation.json", validation)
    archive = output.with_suffix(".zip")
    receipt_path = archive.with_suffix(".sha256.json")
    if archive.exists():
        quarantine(archive, "superseded")
    if receipt_path.exists():
        quarantine(receipt_path, "superseded")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent).as_posix())
    validate_archive(archive, output.name)
    receipt = {
        "local_file": str(archive.resolve()), "filename": archive.name,
        "bytes": archive.stat().st_size, "sha256": digest(archive),
        "zip_crc_verified": True, "package_root": output.name,
        "samples": EXPECTED_SAMPLES, "videos": EXPECTED_SAMPLES * len(MEDIA_KEYS),
        "pairwise_tasks": EXPECTED_SAMPLES * len(PAIRS) * 2,
        "remote_path": "/apps/bypy/lance_review/" + archive.name,
        "upload_status": "not_uploaded",
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT / "review" / VERSION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.package_only:
        package(output)
    else:
        if args.limit is not None and not 1 <= args.limit < EXPECTED_SAMPLES:
            raise ValueError("--limit must be a smoke size between 1 and 227")
        build(output, args.limit)


if __name__ == "__main__":
    main()
