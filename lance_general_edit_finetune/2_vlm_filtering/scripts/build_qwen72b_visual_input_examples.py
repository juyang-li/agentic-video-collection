#!/usr/bin/env python3
"""Export exact Qwen72B visual inputs for a few representative audit samples."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen72b_artifact_risk_v1"
SPECIALIST_ROOT = ROOT.parent / "1_specialist_filtering"
DEFAULT_MANIFEST = RUN / "artifact_risk_30.csv"
DEFAULT_SCORES = RUN / "artifact_risk_30_qwen72b_audit_v3_scores.csv"
DEFAULT_REVIEW = RUN / "qwen32b_vs_72b_artifact_risk_30_v1_review.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/qwen72b_visual_input_examples_v1"
DEFAULT_SAMPLES = [
    "pilot-goku_add-5a25b43e48589a91",
    "pilot-goku_add-7bcd3ee57eb2d560",
    "pilot-goku_remove-b3a5688eed684d00",
    "pilot-goku_remove-070d927c7ab25eb4",
    "pilot-reco_remove-cffc9be4b5a90e12",
]
SAMPLE_NOTES = {
    DEFAULT_SAMPLES[0]: "反向 GOKU add，生成侧为 TRAIN_SOURCE；新版漏放，用于检查局部修补是否可见。",
    DEFAULT_SAMPLES[1]: "反向 GOKU add，生成侧为 TRAIN_SOURCE；新版成功抓到编辑未完成。",
    DEFAULT_SAMPLES[2]: "正向 GOKU remove，小配饰金色耳环；新版抓到残留。",
    DEFAULT_SAMPLES[3]: "正向 GOKU remove，帽子移除；新版漏放。",
    DEFAULT_SAMPLES[4]: "正向 ReCo remove，人物/服装描述；新版成功抓到人物仍存在。",
}

sys.path.insert(0, str(ROOT / "scripts"))
import run_openrouter_qwen3_judge as judge  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=90, optimize=True)


def roi_box_map(
    pairs: list[tuple[object, object]],
    proposals: list[object],
    output: Path,
    right_label: str,
) -> None:
    tiles: list[Image.Image] = []
    for proposal in proposals:
        frame = pairs[proposal.index][1]
        image = Image.fromarray(frame).convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((420, 260), Image.Resampling.LANCZOS)
        scale_x = image.width / original_width
        scale_y = image.height / original_height
        x1, y1, x2, y2 = proposal.expanded_box
        draw = ImageDraw.Draw(image)
        box = (
            round(x1 * original_width * scale_x),
            round(y1 * original_height * scale_y),
            round(x2 * original_width * scale_x),
            round(y2 * original_height * scale_y),
        )
        draw.rectangle(box, outline=(255, 40, 40), width=4)
        area = (x2 - x1) * (y2 - y1) * 100
        draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
        draw.text(
            (7, 7),
            f"T{proposal.index + 1} | {right_label} | ROI area={area:.1f}%",
            fill=(255, 255, 255),
        )
        tiles.append(image)
    columns = 3
    tile_width = max(image.width for image in tiles)
    tile_height = max(image.height for image in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (28, 28, 28))
    for index, image in enumerate(tiles):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        sheet.paste(image, (left, top))
    save_jpeg(sheet, output)


def sample_section(item: dict[str, object]) -> str:
    full = "".join(
        f'<figure><img src="{path}"><figcaption>API image {index}: F{index}</figcaption></figure>'
        for index, path in enumerate(item["full_images"], 1)  # type: ignore[arg-type]
    )
    sheets = "".join(
        f'<figure><img src="{path}"><figcaption>API image {index + 5}: R{index} ROI长图</figcaption></figure>'
        for index, path in enumerate(item["roi_sheets"], 1)  # type: ignore[arg-type]
    )
    individual = "".join(
        f'<figure><img src="{entry["path"]}"><figcaption>T{entry["time_index"]}，框占原图 {entry["area_percent"]:.1f}%</figcaption></figure>'
        for entry in item["individual_rois"]  # type: ignore[union-attr]
    )
    return f"""
    <section id="{html.escape(str(item['sample_id']))}">
      <h2>{html.escape(str(item['sample_id']))}</h2>
      <p><strong>{html.escape(str(item['note']))}</strong></p>
      <div class="facts">
        <span>{html.escape(str(item['dataset']))} / {html.escape(str(item['operation']))}</span>
        <span>人工：{html.escape(str(item['human_status']))}</span>
        <span>72B：{html.escape(str(item['model_decision']))}</span>
        <span>route：{html.escape(str(item['audit_relation']))} / {html.escape(str(item['audit_operation']))}</span>
      </div>
      <pre>{html.escape(str(item['instruction']))}</pre>
      <p>实际媒体映射：左侧 <code>{html.escape(str(item['left_label']))}=TRAIN_{html.escape(str(item['left_training_role'])).upper()}</code>；右侧 <code>{html.escape(str(item['right_label']))}=TRAIN_{html.escape(str(item['right_training_role'])).upper()}</code>。</p>
      <h3>1. 实际发送的5张全图pair</h3><div class="grid full">{full}</div>
      <h3>2. 实际发送的2张ROI时序长图</h3><div class="grid sheets">{sheets}</div>
      <h3>3. ROI框在右侧审查视频中的覆盖范围</h3><figure class="wide"><img src="{item['roi_map']}"><figcaption>红框是当前 expanded ROI；面积越大，局部放大收益越低。</figcaption></figure>
      <h3>4. 将当前9个ROI拆开后的原始行</h3><p class="muted">这些单行图没有单独发送给API；这里只用于观察“堆成长图之前”每个局部pair有多清楚。</p><div class="grid individual">{individual}</div>
    </section>"""


def build_html(items: list[dict[str, object]]) -> str:
    nav = "".join(
        f'<a href="#{html.escape(str(item["sample_id"]))}">{index}. {html.escape(str(item["sample_id"]))}</a>'
        for index, item in enumerate(items, 1)
    )
    sections = "".join(sample_section(item) for item in items)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen72B图像组织实例</title><style>
    :root{{font-family:system-ui,sans-serif;color:#1f2937;background:#eef2f7}}body{{margin:0}}main{{max-width:1500px;margin:auto;padding:18px}}header,section{{background:white;border-radius:12px;padding:16px;margin-bottom:18px;box-shadow:0 2px 10px #0001}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a,.facts span{{background:#e2e8f0;padding:6px 9px;border-radius:7px;text-decoration:none;color:#1e3a8a}}.facts{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}pre{{white-space:pre-wrap;background:#f8fafc;padding:10px;border-radius:7px}}.grid{{display:grid;gap:10px}}.grid.full{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid.sheets{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid.individual{{grid-template-columns:repeat(2,minmax(0,1fr))}}figure{{margin:0;border:1px solid #dbe2ea;padding:7px;border-radius:8px;background:#fafafa}}img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;color:#475569;margin-top:5px}}.wide{{max-width:1200px}}.muted{{color:#64748b}}h2{{border-bottom:2px solid #dbe2ea;padding-bottom:7px}}@media(max-width:800px){{.grid.full,.grid.sheets,.grid.individual{{grid-template-columns:1fr}}}}
    </style></head><body><main><header><h1>Qwen2.5-VL-72B 实际图像输入组织</h1><p>每个样本实际向API发送7张图片：5张全图pair（F1-F5）和2张由9个ROI时刻堆叠的长图（R1-R2）。页面同时提供拆开的ROI行和红框覆盖范围，便于判断局部是否足够紧。</p><nav>{nav}</nav></header>{sections}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()

    manifest = {row["sample_id"]: row for row in read_csv(args.manifest)}
    scores = {row["sample_id"]: row for row in read_csv(args.scores)}
    reviews = {row["sample_id"]: row for row in read_csv(args.review)}
    sample_ids = args.sample_id or DEFAULT_SAMPLES
    missing = [sample_id for sample_id in sample_ids if sample_id not in manifest]
    if missing:
        raise ValueError(f"sample IDs missing from manifest: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    for sample_id in sample_ids:
        row = manifest[sample_id]
        score = scores[sample_id]
        view = judge.build_audit_view(row)
        pairs = judge.rect.matched_frames(
            judge.resolve_media(row[f"{view.left_training_role}_local"]),
            judge.resolve_media(row[f"{view.right_training_role}_local"]),
            9,
        )
        full_images, roi_images, _proposals, selected = judge.prepare_visual_inputs(
            pairs, 5, 640, 640, 9, 512, 2.0, 0.12
        )
        sample_dir = args.output / sample_id
        full_paths: list[str] = []
        denominator = max(1, len(full_images) - 1)
        for index, (left, right) in enumerate(full_images, 1):
            strip = judge.labeled_pair_strip(
                left,
                right,
                f"F{index}",
                round((index - 1) * 100 / denominator),
                view.left_label,
                view.right_label,
            )
            path = sample_dir / f"F{index}.jpg"
            save_jpeg(strip, path)
            full_paths.append(str(path.relative_to(args.output)))

        roi_denominator = max(1, max(frame_index for frame_index, _, _ in roi_images))
        split_at = (len(roi_images) + 1) // 2
        groups = [roi_images[:split_at], roi_images[split_at:]]
        sheet_paths: list[str] = []
        for sheet_index, group in enumerate((group for group in groups if group), 1):
            sheet = judge.temporal_roi_sheet(
                group, sheet_index, roi_denominator, view.left_label, view.right_label
            )
            path = sample_dir / f"R{sheet_index}_actual_api_sheet.jpg"
            save_jpeg(sheet, path)
            sheet_paths.append(str(path.relative_to(args.output)))

        individual: list[dict[str, object]] = []
        for (frame_index, left, right), proposal in zip(roi_images, selected):
            strip = judge.labeled_pair_strip(
                left,
                right,
                f"T{frame_index + 1}",
                round(frame_index * 100 / roi_denominator),
                view.left_label,
                view.right_label,
            )
            path = sample_dir / f"T{frame_index + 1}_individual.jpg"
            save_jpeg(strip, path)
            x1, y1, x2, y2 = proposal.expanded_box
            individual.append(
                {
                    "path": str(path.relative_to(args.output)),
                    "time_index": frame_index + 1,
                    "area_percent": (x2 - x1) * (y2 - y1) * 100,
                }
            )
        roi_map_path = sample_dir / "roi_box_map.jpg"
        roi_box_map(pairs, selected, roi_map_path, view.right_label)
        items.append(
            {
                "sample_id": sample_id,
                "note": SAMPLE_NOTES.get(sample_id, ""),
                "dataset": row.get("dataset", ""),
                "operation": row.get("operation", ""),
                "instruction": row.get("instruction", ""),
                "human_status": reviews[sample_id].get("overall_status", ""),
                "model_decision": score.get("model_decision", ""),
                "audit_operation": score.get("audit_operation", ""),
                "audit_relation": score.get("audit_relation", ""),
                "left_label": view.left_label,
                "right_label": view.right_label,
                "left_training_role": view.left_training_role,
                "right_training_role": view.right_training_role,
                "full_images": full_paths,
                "roi_sheets": sheet_paths,
                "roi_map": str(roi_map_path.relative_to(args.output)),
                "individual_rois": individual,
            }
        )
        print(f"exported={sample_id} full={len(full_paths)} roi={len(individual)}")

    (args.output / "index.html").write_text(build_html(items), encoding="utf-8")
    (args.output / "examples.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README.txt").write_text(
        "解压后打开 index.html。F1-F5 与 R1-R2 是实际发送给72B的JPEG；"
        "T1-T9 individual 和 roi_box_map 仅用于解释当前布局。\n",
        encoding="utf-8",
    )
    print(f"output={args.output} examples={len(items)}")


if __name__ == "__main__":
    main()
