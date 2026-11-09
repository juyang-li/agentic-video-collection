#!/usr/bin/env python3
"""Freeze aligned OmniEdit samples and build the score-free Lance review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT.parent
LANCE = EVAL / "lance_base"
OLD_REVIEW = LANCE / "review/lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906"
PACKAGE_NAME = "omni48_lance_base_seed42_human_review_20260926"
OUTPUT = ROOT / "lance" / PACKAGE_NAME
ARCHIVE = ROOT / "lance" / f"{PACKAGE_NAME}.zip"
FIELDS = (
    "instruction_correctness", "logical_correctness", "content_preservation",
    "temporal_consistency", "visual_quality", "technical_quality", "overall_decision", "notes",
)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selection_hash(category: str, video_id: str) -> str:
    return hashlib.sha256(f"42|{category}|{video_id}".encode()).hexdigest()


def freeze_manifests() -> tuple[list[dict], list[dict]]:
    source_rows = read_csv(LANCE / "omni_edit_spatial_240.csv")
    assert len(source_rows) == 240
    categories: dict[str, list[dict]] = {}
    for row in source_rows:
        categories.setdefault(row["category"], []).append(row)
    assert len(categories) == 24 and {len(v) for v in categories.values()} == {10}
    selected, reserves = [], []
    aligned_index = 0
    for category in sorted(categories):
        ordered = sorted(categories[category], key=lambda x: selection_hash(category, x["video_id"]))
        for rank, row in enumerate(ordered[:3], 1):
            common = dict(
                eval_id=row["video_id"],
                tier={1: "A", 2: "B", 3: "RESERVE"}[rank],
                aligned_index=(aligned_index + rank if rank <= 2 else ""),
                sample_index=int(row["sample_index"]),
                video_id=row["video_id"], category=category,
                editing_instruction=row["editing_instruction"],
                official_source_repo_path=row["official_source_repo_path"],
                official_source_video=row["official_source_video"],
                official_source_sha256=row["official_source_sha256"],
                dataset_revision=row["dataset_revision"],
                selection_hash=selection_hash(category, row["video_id"]),
            )
            (selected if rank <= 2 else reserves).append(common)
        aligned_index += 2
    assert len(selected) == 48 and len(reserves) == 24
    assert {x["tier"] for x in selected} == {"A", "B"}
    write_csv(ROOT / "aligned_subset.csv", selected)
    write_csv(ROOT / "aligned_reserves.csv", reserves)
    manifest_receipt = {
        "selection_rule": "per category, ascending SHA256('42|category|video_id'); ranks 1/2=A/B, rank 3=RESERVE",
        "seed": 42, "dataset": "OmniEdit-Bench/OmniEdit-Bench", "track": "Spatial",
        "dataset_revision": "7447b1a57b52585c3c2c5ac7f00feefdf89f9d61",
        "selected_rows": 48, "reserve_rows": 24, "category_count": 24,
        "aligned_subset_sha256": sha256(ROOT / "aligned_subset.csv"),
        "aligned_reserves_sha256": sha256(ROOT / "aligned_reserves.csv"),
    }
    write_json(ROOT / "aligned_manifest_receipt.json", manifest_receipt)
    return selected, reserves


def probe_video(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    assert len(reader) > 0
    height, width = reader[0].shape[:2]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"]
    p = subprocess.run(command, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", p.stderr)
    assert match, p.stderr
    return dict(codec_name=match.group(1).strip().split()[0], pix_fmt=match.group(2).strip().split("(")[0],
                width=width, height=height, avg_frame_rate=round(reader.get_avg_fps(), 6), nb_frames=len(reader))


def full_decode(path: Path) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build(selected: list[dict]) -> None:
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {OUTPUT} / {ARCHIVE}")
    (OUTPUT / "media").mkdir(parents=True)
    old_rows = {r["sample_id"]: r for r in read_csv(OLD_REVIEW / "review_manifest.csv")}
    old_media = {(r["sample_id"], r["role"]): r for r in json.loads((OLD_REVIEW / "media_inventory.json").read_text())}
    review_rows, generation_rows, media_inventory = [], [], []
    for n, item in enumerate(selected, 1):
        sid = item["video_id"]
        old = old_rows[sid]
        source_raw = Path(item["official_source_video"])
        generated_raw = Path(read_csv(LANCE / "omni_edit_spatial_240.csv")[int(item["sample_index"]) - 1]["generated_video"])
        assert source_raw.is_file() and generated_raw.is_file()
        assert sha256(source_raw) == item["official_source_sha256"]
        assert sha256(generated_raw) == old["generated_original_sha256"]
        src_rel = Path("media") / f"{n:03d}_source.mp4"
        gen_rel = Path("media") / f"{n:03d}_lance_base_output.mp4"
        for role, rel in (("source", src_rel), ("generated", gen_rel)):
            inv = old_media[(sid, role)]
            old_path = OLD_REVIEW / inv["path"]
            assert old_path.is_file() and sha256(old_path) == inv["sha256"]
            link_or_copy(old_path, OUTPUT / rel)
            assert sha256(OUTPUT / rel) == inv["sha256"]
            probe = probe_video(OUTPUT / rel)
            assert probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p"
            full_decode(OUTPUT / rel)
            media_inventory.append(dict(eval_id=sid, role=role, path=rel.as_posix(), sha256=inv["sha256"],
                                        bytes=(OUTPUT / rel).stat().st_size, full_decode_ok=True, **probe))
        base = dict(eval_id=sid, tier=item["tier"], aligned_index=int(item["aligned_index"]),
                    sample_index=int(item["sample_index"]), category=item["category"],
                    instruction=item["editing_instruction"], dataset="OmniEdit Spatial",
                    dataset_revision=item["dataset_revision"], dataset_target_status="not_provided_by_omni_edit",
                    model_id="Lance", model_variant="base", generation_seed=42,
                    lance_code_commit="9717d9160ed16fedfdad5ce39ccbc6c9d69695e0",
                    model_revision="7395315758865e6f56ab87ad06a88c7ac172f056",
                    source_original_sha256=item["official_source_sha256"],
                    generated_original_sha256=old["generated_original_sha256"],
                    review_source_path=src_rel.as_posix(), review_generated_path=gen_rel.as_posix(),
                    review_source_sha256=old_media[(sid, "source")]["sha256"],
                    review_generated_sha256=old_media[(sid, "generated")]["sha256"],
                    review_frames=old["review_frames"], review_fps=old["review_fps"],
                    review_duration_seconds=old["review_duration_seconds"])
        review_rows.append({**base, **{f: "" for f in FIELDS}})
        generation_rows.append(dict(
            eval_id=sid, model_id="Lance", model_revision=base["model_revision"], code_commit=base["lance_code_commit"],
            task_type="video-edit", prompt=item["editing_instruction"], seed=42,
            input_path=item["official_source_video"], input_sha256=item["official_source_sha256"],
            output_path=str(generated_raw), output_sha256=old["generated_original_sha256"],
            inference=dict(steps=30, shift=3.5, cfg=4.0, resolution="480p", fps=12, dtype="bf16",
                           kv_cache=True, prompt_enhancement=False), status="reused_verified_historical_output"))
    write_csv(OUTPUT / "review_manifest.csv", review_rows)
    with (OUTPUT / "generation_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in generation_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(OUTPUT / "media_inventory.json", media_inventory)
    write_json(OUTPUT / "summary.json", {
        "status": "complete", "review_type": "human_only_unlabeled", "sample_count": 48,
        "category_count": 24, "samples_per_category": 2, "tiers": {"A": 24, "B": 24},
        "automated_judge_included": False, "historical_gemini_scores_included": False,
        "lance_regenerated": False, "source": "existing verified OmniEdit Spatial 240 Lance outputs",
        "aligned_subset_sha256": sha256(ROOT / "aligned_subset.csv"),
    })
    template = (Path(__file__).with_name("human_review.html")).read_text(encoding="utf-8")
    payload = {"review_version": PACKAGE_NAME, "title": "Lance Base · OmniEdit Spatial 48 · 人工审核", "items": [
        dict(eval_id=r["eval_id"], tier=r["tier"], category=r["category"], instruction=r["instruction"],
             panels=[{"label": "SOURCE · OFFICIAL INPUT", "path": r["review_source_path"]},
                     {"label": "LANCE BASE · MODEL OUTPUT", "path": r["review_generated_path"]}])
        for r in review_rows]}
    document = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    assert "__REVIEW_DATA__" not in document
    (OUTPUT / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    embedded = json.loads(re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', document, re.S).group(1))
    assert len(embedded["items"]) == 48
    (OUTPUT / "README.md").write_text(
        "# Lance Base · OmniEdit Spatial 48 人工审核包\n\n"
        "完整解压后用 Chrome / Edge / Safari 打开 `index.html`。本包完全离线，不调用 API。\n\n"
        "左侧是 OmniEdit 官方 source，右侧是 Lance Base 既有 seed 42 模型输出；OmniEdit Spatial 不提供 GT。"
        "两侧是已验证的 H.264/yuv420p/faststart 审阅代理，支持同步播放和拖动。\n\n"
        "审核项初始均为空。1=通过，0=不通过，2=存疑；标注保存在浏览器 localStorage，请定期导出 CSV 或 JSON。"
        "本包不包含且不显示历史 Gemini 分数。\n", encoding="utf-8")
    # Bind hashes after all content exists, excluding the checksum file itself.
    checksum_lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def validate_and_package() -> None:
    rows = read_csv(OUTPUT / "review_manifest.csv")
    assert len(rows) == 48 and len({r["eval_id"] for r in rows}) == 48
    assert {r["tier"] for r in rows} == {"A", "B"}
    assert len({r["category"] for r in rows}) == 24
    assert {sum(r["category"] == c for r in rows) for c in {r["category"] for r in rows}} == {2}
    assert all(not r[f] for r in rows for f in FIELDS)
    for line in (OUTPUT / "SHA256SUMS").read_text().splitlines():
        expected, rel = line.split("  ", 1)
        assert sha256(OUTPUT / rel) == expected
    print(json.dumps({"status": "built_pending_browser_validation", "package": str(OUTPUT),
                      "samples": 48, "videos": 96}, ensure_ascii=False, indent=2))


def main() -> None:
    selected, _ = freeze_manifests()
    build(selected)
    validate_and_package()


if __name__ == "__main__":
    main()
