#!/usr/bin/env python3
"""Build a score-free offline review package for the two benchmark runs."""

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
BENCHMARK_ROOT = ROOT.parent / "real_person_multiturn_benchmark_v2"
BENCHMARK = BENCHMARK_ROOT / "benchmark.jsonl"
PACKAGE_NAME = "real_person_base_vs_gtfresh1150_human_review_20260929"
PACKAGE = ROOT / "review" / PACKAGE_NAME
TEMPLATE = ROOT.parent / "gtfresh1150_vs_coinve677_20260927/scripts/review_template.html"
MODELS = (
    ("lance_base", "Lance 基模"),
    ("gtfresh_step1150", "师兄 GT · gtfresh step1150"),
)
FIELDS = ("instruction_correctness", "logical_correctness", "content_preservation",
          "temporal_consistency", "visual_quality", "technical_quality", "overall_decision")


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
    result = probe(destination)
    if result["codec_name"] != "h264" or result["pix_fmt"] != "yuv420p" or not result["faststart"]:
        raise RuntimeError(f"nonportable review media: {destination}: {result}")
    return {"path": destination.relative_to(PACKAGE).as_posix(), "source": str(source),
            "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "full_decode_ok": True, **result}


def raw_output(model_id: str, chain_index: int, chain_id: str, round_index: int) -> Path:
    shard = (chain_index - 1) % 2
    return ROOT / "outputs" / model_id / f"shard{shard:02d}" / chain_id / f"turn{round_index - 1:02d}.mp4"


def review_row(chain: dict, model_id: str, model_label: str) -> dict:
    return {
        "review_id": f"{chain['chain_id']}:{model_id}", "section": "multiturn",
        "eval_id": chain["chain_id"], "category": f"{len(chain['turns'])}轮 · {chain['chain_type']}",
        "instruction": "\n".join(f"R{x['round']}: {x['instruction']}" for x in chain["turns"]),
        "model_id": model_id, "model_label": model_label,
        **{field: "1" for field in FIELDS}, "notes": "",
    }


def validate_manifests(chains: list[dict]) -> None:
    benchmark_hash = sha256(BENCHMARK)
    expected = {chain["chain_id"]: len(chain["turns"]) for chain in chains}
    for model_id, _ in MODELS:
        seen = {}
        for shard in range(2):
            manifest = json.loads((ROOT / "outputs" / model_id / f"shard{shard:02d}/manifest.json").read_text())
            if manifest["status"] != "generated" or manifest["benchmark_sha256"] != benchmark_hash:
                raise RuntimeError(f"incomplete or mismatched manifest: {model_id} shard {shard}")
            if model_id == "gtfresh_step1150":
                check = manifest.get("adapter_load_check", {})
                if check.get("tensor_count") != 1016 or check.get("matching_after_bf16_cast") != 1016:
                    raise RuntimeError(f"adapter load verification missing: shard {shard}")
            for case in manifest["cases"]:
                if case["chain_id"] in seen:
                    raise RuntimeError(f"duplicate generated chain: {case['chain_id']}")
                seen[case["chain_id"]] = len(case["outputs"])
        if seen != expected:
            raise RuntimeError(f"generated chain coverage mismatch: {model_id}")


def main() -> None:
    if PACKAGE.exists() or PACKAGE.with_suffix(".zip").exists():
        raise FileExistsError(f"refusing to overwrite: {PACKAGE}")
    chains = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(chains) != 30 or sum(len(chain["turns"]) for chain in chains) != 112:
        raise RuntimeError("benchmark is not the frozen 30-chain / 112-turn v2")
    validate_manifests(chains)
    (PACKAGE / "media").mkdir(parents=True)

    inventory, generation, reviews, items = [], [], [], []
    for chain_index, chain in enumerate(chains, 1):
        chain_id = chain["chain_id"]
        source = BENCHMARK_ROOT / chain["source"]["media"]
        source_dst = PACKAGE / "media" / f"chain{chain_index:03d}_source.mp4"
        inventory.append({"role": "source", "eval_id": chain_id, **materialize(source, source_dst)})
        rows = [{"label": "原始视频", "paths": [source_dst.relative_to(PACKAGE).as_posix()]}]
        item_reviews = []
        for model_id, model_label in MODELS:
            paths = [source_dst.relative_to(PACKAGE).as_posix()]
            for round_index, turn in enumerate(chain["turns"], 1):
                raw = raw_output(model_id, chain_index, chain_id, round_index)
                if not raw.is_file():
                    raise FileNotFoundError(raw)
                dst = PACKAGE / "media" / f"chain{chain_index:03d}_{model_id}_r{round_index:02d}.mp4"
                info = materialize(raw, dst)
                inventory.append({"role": "output", "eval_id": chain_id, "round": round_index,
                                  "model_id": model_id, **info})
                paths.append(info["path"])
                generation.append({
                    "eval_id": chain_id, "round": round_index, "axis": turn["axis"],
                    "instruction": turn["instruction"], "model_id": model_id,
                    "source": str(source), "source_sha256": sha256(source),
                    "output": str(raw), "output_sha256": sha256(raw),
                    "autoregressive": True, "seed": 42, "steps": 30,
                    "shift": 3.5, "cfg_text": 1.0,
                })
            rows.append({"label": model_label, "paths": paths})
            review = review_row(chain, model_id, model_label)
            reviews.append(review)
            item_reviews.append(review)
        prompts = [turn["instruction"] for turn in chain["turns"]]
        items.append({
            "section": "multiturn", "eval_id": chain_id,
            "category": f"{len(prompts)}轮 · {chain['chain_type']}",
            "instruction": "\n".join(f"R{i}: {prompt}" for i, prompt in enumerate(prompts, 1)),
            "prompts": prompts, "rows": rows, "reviews": item_reviews,
        })

    with (PACKAGE / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader(); writer.writerows(reviews)
    write_json(PACKAGE / "media_inventory.json", inventory)
    with (PACKAGE / "generation_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in generation:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "built_pending_browser_validation", "human_only": True,
        "gemini_included": False, "models": [model for model, _ in MODELS],
        "chains": 30, "turns_per_model": 112, "generated_outputs": 224,
        "review_units": 60, "media_files": 254,
        "comparison_limit": "The checkpoints differ by both training data and training recipe; this is a practical model comparison, not a controlled data-only ablation.",
    }
    write_json(PACKAGE / "summary.json", summary)
    payload = {
        "review_version": PACKAGE_NAME,
        "title": "真实人物多轮编辑 · Lance 基模 vs 师兄 GT checkpoint",
        "items": items, "reviews": reviews,
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    document = template.replace("真实数据 vs 混合数据 Lance 人工审核", payload["title"])
    document = document.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (PACKAGE / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[-1]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    (PACKAGE / "README.md").write_text(
        "# 真实人物多轮编辑人工审核包\n\n"
        "完整解压 ZIP 后，用 Chrome 或 Edge 打开 `index.html`。页面完全离线，不调用任何 API。\n\n"
        "共 30 条真实视频编辑链。每张卡依次展示原始视频、Lance 基模的自回填多轮结果、"
        "师兄 gtfresh step1150 checkpoint 的自回填多轮结果。每轮输出都会作为下一轮输入。\n\n"
        "审核项默认均为 1（通过）；失败改为 0，存疑改为 2，并可填写备注。支持 CSV/JSON 导出和导入。"
        "本包不含 Gemini 评分、API key、模型权重或训练状态。\n\n"
        "注意：两者除了训练数据外，LoRA 覆盖和训练配方也不同，因此这是现有模型的实际效果比较，"
        "不是严格的单变量数据消融。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

