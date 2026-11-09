#!/usr/bin/env python3
"""Build the complete, offline, scrollable BASE Spatial review; no model/API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import zipfile

from decord import VideoReader, cpu
import imageio_ffmpeg

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / "lance_general_edit_finetune/3_lance_lora_finetuning/scripts"))
# Reuse the established audit's full-clip source sampling and browser codec check.
from build_model_audit import source_proxy, probe_video

METRICS = ("accuracy", "preservation", "realism", "consistency", "overall_score")
VERSION = "lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def probe(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if not len(reader):
        raise ValueError(f"empty video: {path}")
    for offset in range(0, len(reader), 16):
        reader.get_batch(list(range(offset, min(offset + 16, len(reader)))))
    height, width = reader[0].shape[:2]
    return dict(frames=len(reader), fps=round(reader.get_avg_fps(), 6),
                width=width, height=height, bytes=path.stat().st_size, sha256=digest(path))


def faststart(path: Path) -> bool:
    atoms = []
    with path.open("rb") as stream:
        while stream.tell() < path.stat().st_size:
            offset = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"truncated MP4 atom: {path}")
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
            if size == 0:
                size = path.stat().st_size - offset
            if size < 8:
                raise ValueError(f"invalid MP4 atom: {path}")
            atoms.append(kind)
            stream.seek(offset + size)
    return b"moov" in atoms and b"mdat" in atoms and atoms.index(b"moov") < atoms.index(b"mdat")


def load_inputs() -> tuple[list[tuple[dict, dict]], dict, dict, dict]:
    judged = BASE / "gemini38_low_vertex_v2"
    audit = json.loads((BASE / "completion_audit.json").read_text())
    for name, key in (("scores.csv", "scores_csv_sha256"), ("summary.json", "summary_sha256")):
        if digest(judged / name) != audit[key]:
            raise ValueError(f"formal audit hash mismatch: {name}")
    scores = read_csv(judged / "scores.csv")
    manifest = read_csv(BASE / "omni_edit_spatial_240.csv")
    summary = json.loads((judged / "summary.json").read_text())
    protocol = json.loads((judged / "protocol.json").read_text())
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == audit["formal_protocol_sha256"]
    by_id = {row["sample_id"]: row for row in scores}
    assert len(scores) == len(by_id) == len(manifest) == 240
    assert {row["video_id"] for row in manifest} == set(by_id)
    assert summary["status"] == "complete" and summary["tasks_succeeded"] == 240 and summary["tasks_failed"] == 0
    assert len(Counter(row["category"] for row in scores)) == 24
    assert set(Counter(row["category"] for row in scores).values()) == {10}
    pairs = []
    for item in sorted(manifest, key=lambda row: int(row["sample_index"])):
        score = by_id[item["video_id"]]
        for left, right in (("sample_index", "sample_index"), ("category", "category"),
                            ("editing_instruction", "instruction"), ("generated_video", "generated_video"),
                            ("official_source_video", "source_path"), ("official_source_sha256", "source_sha256"),
                            ("dataset_revision", "dataset_revision")):
            assert item[left] == score[right], (item["video_id"], left)
        assert score["checkpoint_step"] == "base" and item["generation_seed"] == score["generation_seed"] == "42"
        assert not item["dataset_target_video"] and item["dataset_target_status"] == "not_provided_by_omni_edit"
        for metric in METRICS:
            score[metric] = float(score[metric])
            assert 0 <= score[metric] <= 5
        a, p, r, c = (score[key] for key in METRICS[:4])
        assert abs(score["overall_score"] - round(.5*a + (.2*p + .15*r + .15*c)*a*.2, 4)) < 1e-8
        assert score["reason"].strip()
        pairs.append((item, score))
    for metric in METRICS:
        assert abs(sum(row[metric] for row in scores) / 240 - summary["mean_scores_successes_only"][metric]) < 1e-8
    return pairs, summary, protocol, audit


def prepare_pair(pair: tuple[dict, dict], output: Path) -> tuple[dict, list[dict]]:
    item, score = pair
    source, generated = Path(item["official_source_video"]), Path(item["generated_video"])
    info = probe(generated)
    assert info["fps"] == 12.0
    assert digest(source) == item["official_source_sha256"]
    index = int(item["sample_index"])
    source_dest = output / "media" / f"{index:03d}_source_review.mp4"
    generated_dest = output / "media" / f"{index:03d}_lance_base_generated.mp4"
    indices = source_proxy(source, source_dest, info["frames"], info["fps"])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    codec, pixels = probe_video(ffmpeg, generated)
    encoding = ["-c:v", "copy"] if (codec, pixels) == ("h264", "yuv420p") else [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"]
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(generated),
                    "-map", "0:v:0", "-an", *encoding, "-movflags", "+faststart", "-threads", "2",
                    str(generated_dest)], check=True)
    media = []
    for role, path in (("source", source_dest), ("generated", generated_dest)):
        review_info = probe(path)
        assert review_info["frames"] == info["frames"] and review_info["fps"] == info["fps"]
        assert probe_video(ffmpeg, path) == ("h264", "yuv420p") and faststart(path)
        media.append(dict(sample_id=item["video_id"], role=role, path=path.relative_to(output).as_posix(),
                          **review_info, full_decode_ok=True, codec="h264", pixel_format="yuv420p", faststart=True))
    row = dict(sample_id=item["video_id"], sample_index=index, dataset="OmniEdit Spatial",
               category=item["category"], instruction=item["editing_instruction"], checkpoint_step="base",
               generation_seed=42, dataset_revision=item["dataset_revision"],
               dataset_target_status=item["dataset_target_status"], source_provenance="official_model_input",
               generated_provenance="lance_base_model_output", judge_model="google/gemini-3.8-flash",
               judge_reasoning="low", judge_route="google-vertex/global", judge_status="scored",
               review_group="full_spatial_240", official_source_repo_path=item["official_source_repo_path"],
               source_original_sha256=item["official_source_sha256"], generated_original_sha256=info["sha256"],
               review_source_path=media[0]["path"], review_generated_path=media[1]["path"],
               source_frame_indices_json=json.dumps(indices), review_frames=info["frames"], review_fps=info["fps"],
               review_duration_seconds=info["frames"]/info["fps"],
               generated_review_encoding="lossless_video_stream_remux" if encoding == ["-c:v", "copy"] else "h264_crf18_review_proxy",
               **{key: score[key] for key in (*METRICS, "reason")})
    return row, media


def finish(output: Path, rows: list[dict], media: list[dict], summary: dict, protocol: dict, audit: dict) -> None:
    survey = dict(review_version=output.name, items=rows, summary=summary)
    payload = json.dumps(survey, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("scroll_review.html").read_text(encoding="utf-8")
    document = template.replace("__SURVEY_DATA__", payload)
    assert "__SURVEY_DATA__" not in document
    (output / "index.html").write_text(document, encoding="utf-8")
    # Syntax-check the exact embedded offline script, not a separate build input.
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    embedded = json.loads(re.search(r'<script id="surveyData" type="application/json">(.*?)</script>', document, re.S)[1])
    assert len(embedded["items"]) == len(rows)
    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The official result files are small, contain no credentials, and remain byte-identical.
    for name in ("scores.csv", "summary.json", "protocol.json"):
        shutil.copyfile(BASE / "gemini38_low_vertex_v2" / name, output / name)
    write_json(output / "media_inventory.json", media)
    write_json(output / "evaluation_audit.json", audit)
    write_json(output / "review_validation.json", dict(
        status="passed", sample_count=len(rows), media_count=len(media),
        complete_benchmark=len(rows) == 240, score_source="gemini38_low_vertex_v2 only",
        scores_sha256=digest(output / "scores.csv"), inline_json_count_verified=True,
        javascript_syntax_verified=True, all_media_full_decode_verified=True,
        all_pairs_frame_count_and_fps_matched=True, all_media_h264_yuv420p_faststart=True,
        browser_test="see browser_validation.json, added after real-browser acceptance",
        weight_revision=audit["weight_revision"]))
    (output / "README.md").write_text("""# Lance BASE · OmniEdit Spatial 滚动审阅包

先完整解压，再用 Chrome / Edge / Safari 打开本目录 `index.html`，离线可用。
这是 240 条全集（24 类 × 10），不是抽样；默认按原始编号连续排列，直接滚轮向下看，无需“下一条”。
顶部可以搜索指令 / ID、筛选类别、按 Gemini 总分排序；清空筛选恢复全集。

## 左右视频与评分

- 左：SOURCE，官方模型输入的审阅代理；右：LANCE BASE GENERATED_VIDEO，seed 42 基模输出。
- OmniEdit Spatial 没有提供 dataset target / GT。本包不把模型生成结果叫作数据集 target。
- Gemini 3.8 Flash（low），统一 Vertex v2：四项原始分数及完整英文评语逐条显示，未重新翻译或改写。
- 四维及总分均为 0–5。总分采用官方公式，不是四维算术平均：`0.5*A + (0.2*P+0.15*R+0.15*C)*(A/5)`。
- `scores.csv`、`summary.json`、`protocol.json` 与正式 v2 文件逐字节一致；CSV 中历史服务器绝对路径用于追溯，不是离线网页播放路径。可移植路径见 `review_manifest.csv`。
- Gemini 实际评分使用未经抽帧、重编码的完整官方 source MP4 和完整 generated MP4，**不是本包审阅代理**。

## 播放与人工记录

每条两个视频同步播放、暂停和拖动。点击“同步播放”，或对当前可见条目按空格。
仅滚动不会产生人工结论；初始全部未标注。1 = 接受，0 = 拒绝，2 = 不确定；可填写备注、清除单条结论。
标注按 sample_id 自动保存在当前浏览器，可导出 CSV / JSON 并导入续审。换浏览器或移动目录前请先导出备份；隐私模式可能无法持久保存。
视频随滚动懒加载，离开附近视区释放，避免一次解码 480 个视频。重新滚回来从片头播放，不影响已保存标注。

## 审阅媒体说明

仅修改审阅副本，不改动原始输入、生成视频或已完成评分。
左侧按与 Lance 单片段采样相同的 full-clip linspace 选帧，缩放至不超过 720×480，编码 H.264 / yuv420p；帧数与播放 FPS 对齐右侧。因此左侧是时间对齐审阅代理，不是原始帧率全视频。
右侧优先无损重封装原始 H.264 视频流；仅不兼容浏览器时才生成 H.264 CRF18 代理。逐条方式见 manifest。
两侧均静音、faststart，帧数、FPS、全帧解码已逐条检查。原视频 SHA256、源采样索引和审阅文件哈希分别见 manifest / media_inventory.json。
审阅包不含模型权重、API 凭据或账户配置；审阅过程不调用 API、不联网。
""", encoding="utf-8")


def package(output: Path) -> None:
    rows = read_csv(output / "review_manifest.csv")
    assert len(rows) == 240, "never publish a smoke subset as the complete benchmark"
    browser = json.loads((output / "browser_validation.json").read_text())
    assert browser["status"] == "passed" and browser["sample_count"] == 240
    assert not browser["javascript_errors"] and not browser["external_requests"]
    # Bind every displayed score back to the exact source/generated hashes judged by v2.
    records = {}
    for path in (BASE / "gemini38_low_vertex_v2/records").glob("*/result.json"):
        result = json.loads(path.read_text())
        sid = result["identity"]["task"]["sample_id"]
        assert sid not in records
        records[sid] = result
    assert len(records) == 240
    audit = json.loads((output / "evaluation_audit.json").read_text())
    for row in rows:
        result = records[row["sample_id"]]
        identity = result["identity"]
        assert result["status"] == "success" and result["returned_model"] == "google/gemini-3.8-flash"
        assert result["returned_provider"] == "Google"
        assert identity["protocol_sha256"] == audit["formal_protocol_sha256"]
        assert identity["source_sha256"] == row["source_original_sha256"]
        assert identity["generated_sha256"] == row["generated_original_sha256"]
        assert identity["task"]["instruction"] == row["instruction"]
        assert result["score"]["reason"] == row["reason"]
        for metric in METRICS:
            assert abs(float(row[metric]) - result["score"][metric]) < 1e-8
        for key in ("review_source_path", "review_generated_path"):
            assert (output / row[key]).is_file() and Path(row[key]).parts[0] == "media"
    for entry in json.loads((output / "media_inventory.json").read_text()):
        assert digest(output / entry["path"]) == entry["sha256"]
    validation = json.loads((output / "review_validation.json").read_text())
    validation.update(browser_test="passed; see browser_validation.json", score_media_binding_to_raw_judge_records_verified=240)
    write_json(output / "review_validation.json", validation)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert {name.split("/")[0] for name in names} == {output.name}
        assert sum(name.endswith(".mp4") for name in names) == 480
    receipt = dict(local_file=str(archive), filename=archive.name, bytes=archive.stat().st_size,
                   sha256=digest(archive), zip_crc_verified=True, package_root=output.name,
                   samples=240, videos=480, remote_path="/apps/bypy/lance_review/" + archive.name,
                   upload_status="not_uploaded")
    write_json(archive.with_suffix(".sha256.json"), receipt)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "review" / VERSION)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="smoke only, never zip/upload")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.package_only:
        package(args.output)
        return
    pairs, summary, protocol, audit = load_inputs()
    if args.limit:
        assert args.output.name != VERSION and args.limit > 0
        pairs = pairs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "media").mkdir()
    rows, media = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, entries in pool.map(lambda pair: prepare_pair(pair, args.output), pairs):
            rows.append(row)
            media.extend(entries)
            if len(rows) % 20 == 0 or len(rows) == len(pairs):
                print(f"prepared + verified {len(rows)}/{len(pairs)} pairs", flush=True)
    finish(args.output, rows, media, summary, protocol, audit)
    print(f"ready for browser verification: {args.output}", flush=True)


if __name__ == "__main__":
    main()
