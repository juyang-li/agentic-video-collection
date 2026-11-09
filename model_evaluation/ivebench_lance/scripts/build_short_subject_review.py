#!/usr/bin/env python3
"""Build and package a complete offline IVEBench Base-vs-LoRA review."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import zipfile

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "short_subject"
FRAMES = ARTIFACTS / "metric_frames"
OUTPUTS = ARTIFACTS / "outputs_raw"
VERSION = "ivebench_short_subject91_base_vs_lora_step677_seed42_gemini38flash_temp0_v1"
EXPECTED_SAMPLES = 91
OUTPUT_PREFIX = "short_subject"
TITLE = "IVEBench Short · 主体编辑 91 条"
STAGE_NAME = "short_subject"
STAGES = {
    "short_subject": (91, "short_subject", "ivebench_short_subject91_base_vs_lora_step677_seed42_gemini38flash_temp0_v1", "IVEBench Short · 主体编辑 91 条"),
    "long_subject": (43, "long_subject", "ivebench_long_subject43_base_vs_lora_step677_seed42_gemini38flash_temp0_v1", "IVEBench Long · 主体编辑 43 条"),
    "attribute_all": (94, "attribute_all", "ivebench_attribute94_base_vs_lora_step677_seed42_gemini38flash_temp0_v1", "IVEBench Short+Long · 属性编辑 94 条"),
}
STAGE_REVIEW_META = {
    "short_subject": ("IVEBench Short", "subject_editing_short_all_91"),
    "long_subject": ("IVEBench Long", "subject_editing_long_all_43"),
    "attribute_all": ("IVEBench Short+Long", "attribute_editing_all_94"),
}
METRICS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "vtss", "overall_semantic_consistency",
    "phrase_semantic_consistency", "instruction_satisfaction", "semantic_fidelity",
    "motion_fidelity", "content_fidelity",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quarantine(path: Path, reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.parent / f"{path.name}.{reason}.{stamp}.{os.getpid()}"
    path.replace(destination)
    return destination


def complete_review_tree(output: Path) -> bool:
    try:
        rows = read_csv(output / "review_manifest.csv")
        inventory = json.loads((output / "media_inventory.json").read_text())
        validation = json.loads((output / "review_validation.json").read_text())
        return (
            (output / "index.html").is_file()
            and (output / "README.md").is_file()
            and len(rows) == EXPECTED_SAMPLES
            and len(inventory) == EXPECTED_SAMPLES * 3
            and validation.get("status") == "passed"
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return False


def validate_archive(archive: Path, package_root: str) -> None:
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        if len([name for name in bundle.namelist() if name.endswith(".mp4")]) != EXPECTED_SAMPLES * 3:
            raise ValueError("ZIP contains the wrong number of videos")
        if {name.split("/")[0] for name in bundle.namelist()} != {package_root}:
            raise ValueError("ZIP does not have exactly one package root")


def faststart(path: Path) -> bool:
    atoms = []
    size_total = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() < size_total:
            offset = handle.tell()
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError(f"truncated MP4: {path}")
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
            if size == 0:
                size = size_total - offset
            if size < 8:
                raise ValueError(f"bad MP4 atom in {path}")
            atoms.append(kind)
            handle.seek(offset + size)
    return b"moov" in atoms and b"mdat" in atoms and atoms.index(b"moov") < atoms.index(b"mdat")


def codec(path: Path) -> tuple[str, str]:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffprobe = ffmpeg.with_name("ffprobe")
    if not ffprobe.exists():
        system_ffprobe = shutil.which("ffprobe")
        if system_ffprobe is None:
            # imageio-ffmpeg commonly ships ffmpeg without ffprobe.  Its input
            # banner still reports the codec and pixel format without decoding
            # the whole file, so use that as the dependency-free fallback.
            result = subprocess.run(
                [str(ffmpeg), "-hide_banner", "-i", str(path)],
                check=False, text=True, capture_output=True,
            )
            for line in result.stderr.splitlines():
                if " Video: " not in line:
                    continue
                fields = line.split("Video:", 1)[1].split(",")
                if len(fields) >= 2:
                    codec_name = fields[0].strip().split()[0]
                    pixel_format = fields[1].strip().split("(", 1)[0]
                    return codec_name, pixel_format
            raise ValueError(f"ffmpeg could not identify the video codec: {path}")
        ffprobe = Path(system_ffprobe)
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(path)],
        check=True, text=True, capture_output=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return stream["codec_name"], stream["pix_fmt"]


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if not len(reader):
        raise ValueError(f"empty video: {path}")
    for offset in range(0, len(reader), 16):
        reader.get_batch(list(range(offset, min(offset + 16, len(reader)))))
    height, width = map(int, reader[0].shape[:2])
    return {
        "frames": len(reader), "fps": round(float(reader.get_avg_fps()), 6),
        "width": width, "height": height, "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def generated_path(model_kind: str, output_id: int) -> Path:
    matches = list(OUTPUTS.glob(f"{OUTPUT_PREFIX}_{model_kind}_shard*/{output_id:06d}.mp4"))
    if len(matches) != 1:
        raise ValueError(f"expected one {model_kind}/{output_id} output, got {matches}")
    return matches[0]


def prepare_video(source: Path, destination: Path, source_is_frames: bool = False) -> str:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if source_is_frames:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-framerate", "12", "-i", str(source / "frame_%05d.png"), "-an",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
                   "-movflags", "+faststart", "-threads", "2", str(destination)]
        encoding = "h264_crf18_from_exact_aligned_png"
    else:
        source_codec = codec(source)
        options = ["-c:v", "copy"] if source_codec == ("h264", "yuv420p") else [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"]
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source),
                   "-map", "0:v:0", "-an", *options, "-movflags", "+faststart", "-threads", "2", str(destination)]
        encoding = "lossless_video_stream_remux" if options == ["-c:v", "copy"] else "h264_crf18_proxy"
    subprocess.run(command, check=True)
    return encoding


def sample_total(row: dict[str, object], prefix: str) -> float:
    def value(metric: str) -> float:
        raw = float(row[f"{prefix}_{metric}"])
        if metric in ("instruction_satisfaction", "content_fidelity"):
            return (raw - 1) / 4
        if metric == "vtss":
            return (raw + .05) / .15
        return raw
    quality = (
        value("subject_consistency")
        + value("background_consistency")
        + value("temporal_flickering")
        + value("motion_smoothness")
        + 4 * value("vtss")
    ) / 8
    compliance = (value("overall_semantic_consistency") + value("phrase_semantic_consistency") + 3 * value("instruction_satisfaction")) / 5
    fidelity = (value("semantic_fidelity") + value("motion_fidelity") + 3 * value("content_fidelity")) / 5
    return (quality + compliance + fidelity) / 3


def prepare_row(input_row: dict[str, str], output: Path) -> tuple[dict, list[dict]]:
    row: dict[str, object] = dict(input_row)
    official_id = int(row["official_id"])
    output_id = int(row.get("generation_id") or official_id)
    sample_id = str(row["sample_id"])
    for metric in METRICS:
        for prefix in ("base", "lora", "delta"):
            row[f"{prefix}_{metric}"] = float(row[f"{prefix}_{metric}"])
    row["official_id"] = official_id
    row["generation_seed"] = int(row["generation_seed"])
    base_input = generated_path("base", output_id)
    lora_input = generated_path("lora_step677", output_id)
    paths = {
        "source": output / "media" / f"{output_id:04d}_source.mp4",
        "base": output / "media" / f"{output_id:04d}_base.mp4",
        "lora": output / "media" / f"{output_id:04d}_lora_step677.mp4",
    }
    encodings = {
        "source": prepare_video(FRAMES / "source" / sample_id, paths["source"], True),
        "base": prepare_video(base_input, paths["base"]),
        "lora": prepare_video(lora_input, paths["lora"]),
    }
    infos = {kind: probe(path) for kind, path in paths.items()}
    signatures = {(info["frames"], info["fps"], info["width"], info["height"]) for info in infos.values()}
    if len(signatures) != 1 or next(iter(signatures))[1] != 12.0:
        raise ValueError(f"unaligned review media for {sample_id}: {infos}")
    inventory = []
    for kind, path in paths.items():
        if codec(path) != ("h264", "yuv420p") or not faststart(path):
            raise ValueError(f"browser incompatible media: {path}")
        inventory.append({
            "sample_id": sample_id, "role": kind, "path": path.relative_to(output).as_posix(),
            **infos[kind], "encoding": encodings[kind], "full_decode_ok": True,
            "codec": "h264", "pixel_format": "yuv420p", "faststart": True,
        })
    raw_hashes = {"base": digest(base_input), "lora": digest(lora_input)}
    prepared = json.loads((ARTIFACTS / "prepared_media.json").read_text())
    provenance = {item["sample_id"]: item for item in prepared}[sample_id]
    if raw_hashes != {"base": provenance["base_generated_sha256"], "lora": provenance["lora_generated_sha256"]}:
        raise ValueError(f"raw output changed after judge preparation: {sample_id}")
    base_total, lora_total = sample_total(row, "base"), sample_total(row, "lora")
    dataset_label, review_group = STAGE_REVIEW_META[STAGE_NAME]
    row.update(
        dataset=dataset_label, review_group=review_group,
        base_checkpoint="official_base", lora_checkpoint="step_677", judge_model="google/gemini-3.8-flash",
        judge_reasoning="low", judge_temperature=0.0, judge_route="OpenRouter / Google AI Studio",
        review_source_path=inventory[0]["path"], review_base_path=inventory[1]["path"],
        review_lora_path=inventory[2]["path"], review_frames=infos["source"]["frames"],
        review_fps=infos["source"]["fps"], review_duration_seconds=infos["source"]["frames"] / infos["source"]["fps"],
        base_generated_original_sha256=raw_hashes["base"], lora_generated_original_sha256=raw_hashes["lora"],
        base_sample_total=base_total, lora_sample_total=lora_total, delta_sample_total=lora_total - base_total,
    )
    return row, inventory


def finish(output: Path, rows: list[dict], inventory: list[dict]) -> None:
    report = json.loads((ARTIFACTS / "evaluation_report.json").read_text())
    survey = {"review_version": output.name, "title": TITLE, "items": rows, "report": report}
    payload = json.dumps(survey, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("ivebench_review.html").read_text(encoding="utf-8")
    document = template.replace("__SURVEY_DATA__", payload)
    if "__SURVEY_DATA__" in document:
        raise ValueError("review template substitution failed")
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", document, re.S)
    for script in scripts[1:]:
        subprocess.run(["node", "--check"], input=script, text=True, check=True)
    embedded = json.loads(re.search(r'<script id="surveyData" type="application/json">(.*?)</script>', document, re.S)[1])
    if len(embedded["items"]) != EXPECTED_SAMPLES:
        raise ValueError("embedded review data is incomplete")
    (output / "index.html").write_text(document, encoding="utf-8")
    with (output / "review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "media_inventory.json", inventory)
    results = output / "results"
    results.mkdir()
    for name in (
        "evaluation_report.json",
        "sample_results.csv",
        "judge_preparation.json",
        "metric_frame_preparation.json",
        "manifest.csv",
        "preparation.json",
        "generation_jobs.json",
        "generation_status.json",
        "api_budget_after_preflight.json",
    ):
        shutil.copyfile(ARTIFACTS / name, results / name)
    for label, source in (
        ("gemini_absolute", ARTIFACTS / "gemini_absolute_temp0_v1"),
        ("gemini_pairwise", ARTIFACTS / "gemini_pairwise_temp0_v1"),
        ("gemini_stability", ARTIFACTS / "gemini_stability_temp0_v1"),
    ):
        destination = results / label
        destination.mkdir()
        for name in ("scores.csv", "summary.json", "protocol.json"):
            shutil.copyfile(source / name, destination / name)
    shutil.copytree(ARTIFACTS / "local_metrics", results / "local_metrics")
    # Preserve the evidence used by the generation gate, not only the derived
    # score tables.  These inventories prove full decode, exact source/output
    # frame count, FPS, duration, and hashes for every generated shard.
    shutil.copytree(ARTIFACTS / "technical", results / "generation_technical")
    provenance_dir = results / "generation_provenance"
    provenance_dir.mkdir()
    provenance_files = sorted((ARTIFACTS / "outputs_raw").glob("*/generation_provenance.json"))
    if len(provenance_files) != 10:  # 2 matched smoke + 4 Base + 4 LoRA shards
        raise ValueError(f"expected 10 generation provenance records, got {len(provenance_files)}")
    for source in provenance_files:
        shutil.copyfile(source, provenance_dir / f"{source.parent.name}.json")
    validation = {
        "status": "passed", "samples": EXPECTED_SAMPLES, "videos": EXPECTED_SAMPLES * 3,
        "all_media_full_decode_verified": True, "all_triplets_temporally_aligned": True,
        "all_media_h264_yuv420p_faststart": True, "embedded_records": EXPECTED_SAMPLES,
        "javascript_syntax_verified": True, "browser_test": "pending",
        "raw_score_report_sha256": digest(ARTIFACTS / "evaluation_report.json"),
    }
    write_json(output / "review_validation.json", validation)
    (output / "README.md").write_text(f"""# {TITLE} · Base vs LoRA step 677

请先完整解压，用 Chrome 或 Edge 打开 `index.html`。页面完全离线，可直接滚轮连续查看 {EXPECTED_SAMPLES} 条；每条从左到右是时间对齐 SOURCE、Lance Base、LoRA step 677，三段可同步播放、暂停和拖动。

页面逐条展示 Gemini 3.8 Flash（low、temperature=0、固定 Google AI Studio 路由）的 IS/CF 分数及完整评语、A/B 与 B/A 双盲对调结果，以及 9 个本地 IVEBench 指标。顶部聚合采用官方权重；本子集没有数量编辑，Quantity Accuracy 记为 N/A 并从分母排除。该分数用于 Base/LoRA 配对比较，不能替代 IVEBench 600 条完整榜单分数。

SOURCE 严格采用 Lance 实际输入的冻结采样帧，帧数、分辨率和 12 fps 与两个输出逐条一致。网页视频只是审核副本；原始输出哈希、完整指标、协议、费用和稳定性检查位于 `review_manifest.csv`、`media_inventory.json` 与 `results/`。

人工结论和备注保存在浏览器 localStorage，可导出/导入 CSV。移动目录或换浏览器前请先导出。包内没有 API key、模型权重或账户配置，也不会发起网络请求。
""", encoding="utf-8")


def package(output: Path) -> None:
    rows = read_csv(output / "review_manifest.csv")
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError("refusing to package an incomplete review")
    browser = json.loads((output / "browser_validation.json").read_text())
    if browser.get("status") != "passed" or browser.get("sample_count") != EXPECTED_SAMPLES or browser.get("external_requests"):
        raise ValueError("real-browser acceptance has not passed")
    inventory = json.loads((output / "media_inventory.json").read_text())
    if len(inventory) != EXPECTED_SAMPLES * 3:
        raise ValueError("media inventory is incomplete")
    for item in inventory:
        if digest(output / item["path"]) != item["sha256"]:
            raise ValueError(f"media hash mismatch: {item['path']}")
    validation = json.loads((output / "review_validation.json").read_text())
    validation["browser_test"] = "passed; see browser_validation.json"
    write_json(output / "review_validation.json", validation)
    archive = output.with_suffix(".zip")
    receipt_path = archive.with_suffix(".sha256.json")
    reuse_existing_archive = False
    if archive.exists():
        try:
            validate_archive(archive, output.name)
            archive_sha256 = digest(archive)
            if receipt_path.exists():
                existing = json.loads(receipt_path.read_text())
                if (
                    existing.get("bytes") == archive.stat().st_size
                    and existing.get("sha256") == archive_sha256
                    and existing.get("zip_crc_verified") is True
                ):
                    print(json.dumps(existing, ensure_ascii=False, indent=2))
                    return
            reuse_existing_archive = True
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
            quarantined = quarantine(archive, "incomplete")
            print(f"quarantined incomplete archive: {quarantined}", flush=True)
            if receipt_path.exists():
                quarantine(receipt_path, "incomplete")
    if not reuse_existing_archive:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output.parent).as_posix())
        validate_archive(archive, output.name)
    receipt = {
        "local_file": str(archive.resolve()), "filename": archive.name,
        "bytes": archive.stat().st_size, "sha256": digest(archive), "zip_crc_verified": True,
        "package_root": output.name, "samples": EXPECTED_SAMPLES, "videos": EXPECTED_SAMPLES * 3,
        "remote_path": "/apps/bypy/lance_review/" + archive.name, "upload_status": "not_uploaded",
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main() -> None:
    global ARTIFACTS, FRAMES, OUTPUTS, VERSION, EXPECTED_SAMPLES, OUTPUT_PREFIX, TITLE, STAGE_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="short_subject")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    EXPECTED_SAMPLES, OUTPUT_PREFIX, VERSION, TITLE = STAGES[args.stage]
    STAGE_NAME = args.stage
    ARTIFACTS = ROOT / "artifacts" / args.stage
    FRAMES = ARTIFACTS / "metric_frames"
    OUTPUTS = ARTIFACTS / "outputs_raw"
    if args.output is None:
        args.output = ARTIFACTS / "review" / VERSION
    output = args.output.resolve()
    if args.package_only:
        package(output); return
    rows = read_csv(ARTIFACTS / "sample_results.csv")
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"sample_results.csv must contain {EXPECTED_SAMPLES} rows")
    if args.limit:
        if args.limit < 1 or args.limit >= EXPECTED_SAMPLES or output.name == VERSION:
            raise ValueError(
                f"smoke limit requires 1..{EXPECTED_SAMPLES - 1} and a non-production output"
            )
        rows = rows[:args.limit]
    if output.exists():
        if complete_review_tree(output):
            print(json.dumps({"status": "review_already_complete", "samples": EXPECTED_SAMPLES,
                              "videos": EXPECTED_SAMPLES * 3, "output": str(output)}))
            return
        quarantined = quarantine(output, "interrupted")
        print(f"quarantined interrupted review tree: {quarantined}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    completed, inventory = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, media in pool.map(lambda item: prepare_row(item, output), rows):
            completed.append(row); inventory.extend(media)
            print(f"prepared review {len(completed)}/{len(rows)}", flush=True)
    completed.sort(key=lambda item: int(item["sequence_index"]))
    if len(completed) != EXPECTED_SAMPLES:
        print(json.dumps({"status": "smoke_complete", "samples": len(completed), "output": str(output)})); return
    finish(output, completed, inventory)
    print(json.dumps({"status": "ready_for_browser_test", "samples": EXPECTED_SAMPLES,
                      "videos": EXPECTED_SAMPLES * 3, "output": str(output)}))


if __name__ == "__main__":
    main()
