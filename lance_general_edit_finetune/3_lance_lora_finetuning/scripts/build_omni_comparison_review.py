#!/usr/bin/env python3
"""Build a complete offline, scrollable Base-vs-LoRA OmniEdit review; no inference/API."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
EVALUATION = ROOT / "artifacts/lance_4k_r64_seed42_e2_v1/evaluation"
BASE = PROJECT / "model_evaluation/lance_base"
BASE_REVIEW = BASE / "review/lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906"
sys.path.insert(0, str(BASE / "scripts"))
from build_scroll_review import digest, faststart, probe, probe_video, read_csv, write_json

METRICS = ("accuracy", "preservation", "realism", "consistency", "overall_score")
VERSION = "lance_4k_r64_seed42_step677_vs_base_omni240_scroll_vertex_v2_20260907"


def link(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def successful_receipts(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted((root / "records").glob("*/result.json")):
        record = json.loads(path.read_text())
        sample_id = record["identity"]["task"]["sample_id"]
        if sample_id in result or record.get("status") != "success":
            raise ValueError("duplicate or invalid scoring receipt")
        result[sample_id] = record
    if len(result) != 240:
        raise ValueError("complete Omni review requires exactly 240 successful receipts")
    return result


def load_inputs() -> tuple[list[tuple[dict, dict, dict]], dict]:
    base_manifest = read_csv(BASE_REVIEW / "review_manifest.csv")
    base_scores = read_csv(BASE / "gemini38_low_vertex_v2/scores.csv")
    fine_scores = read_csv(EVALUATION / "omni_scores_vertex_v2/scores.csv")
    base_summary = json.loads((BASE / "gemini38_low_vertex_v2/summary.json").read_text())
    fine_summary = json.loads((EVALUATION / "omni_scores_vertex_v2/summary.json").read_text())
    base_protocol = json.loads((BASE / "gemini38_low_vertex_v2/protocol.json").read_text())
    fine_protocol = json.loads((EVALUATION / "omni_scores_vertex_v2/protocol.json").read_text())
    contract = json.loads((EVALUATION / "judge_contract.json").read_text())
    report = json.loads((EVALUATION / "evaluation_report.json").read_text())
    integrity = json.loads((EVALUATION / "output_integrity_audit.json").read_text())
    if base_protocol != fine_protocol:
        raise ValueError("Base and LoRA judge protocols differ")
    protocol_sha = hashlib.sha256(json.dumps(base_protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if protocol_sha != contract["canonical_protocol_sha256"]:
        raise ValueError("protocol differs from frozen comparison contract")
    if (base_summary.get("status"), base_summary.get("tasks_succeeded"), base_summary.get("tasks_failed")) != ("complete", 240, 0):
        raise ValueError("Base formal score set is incomplete")
    if (fine_summary.get("status"), fine_summary.get("tasks_succeeded"), fine_summary.get("tasks_failed")) != ("complete", 240, 0):
        raise ValueError("LoRA formal score set is incomplete")
    if integrity.get("output_integrity_status") != "passed" or integrity.get("stages", {}).get("omni", {}).get("verified_videos") != 240:
        raise ValueError("LoRA Omni outputs lack the frozen integrity audit")
    if report.get("omni_paired_comparison", {}).get("paired_samples") != 240:
        raise ValueError("paired comparison report is incomplete")
    mappings = []
    for rows, name in ((base_manifest, "base review"), (base_scores, "base scores"), (fine_scores, "LoRA scores")):
        by_id = {row["sample_id"]: row for row in rows}
        if len(rows) != len(by_id) or len(by_id) != 240:
            raise ValueError(f"{name} must contain 240 unique samples")
        mappings.append(by_id)
    base_review, base_by_id, fine_by_id = mappings
    base_receipts = successful_receipts(BASE / "gemini38_low_vertex_v2")
    fine_receipts = successful_receipts(EVALUATION / "omni_scores_vertex_v2")
    pairs = []
    for sample_id, base_row in sorted(base_review.items(), key=lambda item: int(item[1]["sample_index"])):
        base_score, fine_score = base_by_id[sample_id], fine_by_id[sample_id]
        br, fr = base_receipts[sample_id], fine_receipts[sample_id]
        if (base_row["instruction"] != base_score["instruction"] or base_row["instruction"] != fine_score["instruction"]
                or base_row["category"] != base_score["category"] or base_row["category"] != fine_score["branch"]
                or base_score["source_path"] != fine_score["source_path"]
                or base_score["generation_seed"] != fine_score["generation_seed"] or base_score["generation_seed"] != "42"
                or base_score["checkpoint_step"] != "base" or fine_score["checkpoint_step"] != "677"):
            raise ValueError(f"paired task mismatch: {sample_id}")
        for record, score, expected_generated in (
            (br, base_score, base_row["generated_original_sha256"]),
            (fr, fine_score, digest(Path(fine_score["generated_video"]))),
        ):
            identity = record["identity"]
            if (record["returned_model"] != "google/gemini-3.8-flash" or record["returned_provider"] != "Google"
                    or identity["protocol_sha256"] != protocol_sha
                    or identity["source_sha256"] != base_row["source_original_sha256"]
                    or identity["generated_sha256"] != expected_generated
                    or identity["task"]["instruction"] != score["instruction"]
                    or record["score"]["reason"] != score["reason"]):
                raise ValueError(f"score/media receipt mismatch: {sample_id}")
            for metric in METRICS:
                if abs(float(score[metric]) - float(record["score"][metric])) > 1e-8:
                    raise ValueError(f"score receipt value mismatch: {sample_id}/{metric}")
            score["_generated_original_sha256"] = identity["generated_sha256"]
        pairs.append((base_row, base_score, fine_score))
    if len(Counter(item[0]["category"] for item in pairs)) != 24 or set(Counter(item[0]["category"] for item in pairs).values()) != {10}:
        raise ValueError("expected 24 categories of 10 paired samples")
    summary = {
        "base_mean_scores": base_summary["mean_scores_successes_only"],
        "finetune_mean_scores": fine_summary["mean_scores_successes_only"],
        "mean_differences_finetune_minus_base": report["omni_paired_comparison"]["mean_differences"],
        "paired_overall_wins_ties_losses": report["omni_paired_comparison"]["overall_sample_wins_ties_losses"],
        "paired_overall_bootstrap_95_ci": report["omni_paired_comparison"]["paired_stratified_bootstrap_95_ci"]["overall_score"],
        "judge_protocol_sha256": protocol_sha,
    }
    return pairs, summary


def prepare_pair(pair: tuple[dict, dict, dict], output: Path) -> tuple[dict, list[dict]]:
    base_row, base_score, fine_score = pair
    index = int(base_row["sample_index"])
    source_input = BASE_REVIEW / base_row["review_source_path"]
    base_input = BASE_REVIEW / base_row["review_generated_path"]
    fine_input = Path(fine_score["generated_video"])
    source_output = output / "media" / f"{index:03d}_source_review.mp4"
    base_output = output / "media" / f"{index:03d}_lance_base_generated.mp4"
    fine_output = output / "media" / f"{index:03d}_lance_lora_step677_generated.mp4"
    link(source_input, source_output)
    link(base_input, base_output)
    fine_raw = probe(fine_input)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    codec, pixels = probe_video(ffmpeg, fine_input)
    encoding = ["-c:v", "copy"] if (codec, pixels) == ("h264", "yuv420p") else [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"]
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(fine_input),
                    "-map", "0:v:0", "-an", *encoding, "-movflags", "+faststart", "-threads", "2",
                    str(fine_output)], check=True)
    media = []
    infos = []
    for role, path in (("source", source_output), ("base", base_output), ("finetune", fine_output)):
        info = probe(path)
        if probe_video(ffmpeg, path) != ("h264", "yuv420p") or not faststart(path):
            raise ValueError(f"browser-incompatible review media: {path}")
        infos.append(info)
        media.append({"sample_id": base_row["sample_id"], "role": role,
                      "path": path.relative_to(output).as_posix(), **info,
                      "full_decode_ok": True, "codec": "h264", "pixel_format": "yuv420p", "faststart": True})
    if len({(info["frames"], info["fps"]) for info in infos}) != 1:
        raise ValueError(f"three-way review media not temporally aligned: {base_row['sample_id']}")
    if fine_raw["sha256"] != fine_score["_generated_original_sha256"]:
        raise ValueError("LoRA output changed after scoring")
    row = {
        "sample_id": base_row["sample_id"], "sample_index": index, "dataset": "OmniEdit Spatial",
        "category": base_row["category"], "instruction": base_row["instruction"],
        "base_checkpoint": "official_base", "finetune_checkpoint_step": 677, "generation_seed": 42,
        "judge_model": "google/gemini-3.8-flash", "judge_reasoning": "low",
        "judge_route": "google-vertex/global", "review_group": "complete_paired_spatial_240",
        "dataset_target_status": "not_provided_by_omni_edit",
        "source_original_sha256": base_row["source_original_sha256"],
        "base_generated_original_sha256": base_score["_generated_original_sha256"],
        "finetune_generated_original_sha256": fine_score["_generated_original_sha256"],
        "review_source_path": media[0]["path"], "review_base_path": media[1]["path"],
        "review_finetune_path": media[2]["path"], "source_frame_indices_json": base_row["source_frame_indices_json"],
        "review_frames": infos[0]["frames"], "review_fps": infos[0]["fps"],
        "review_duration_seconds": infos[0]["frames"] / infos[0]["fps"],
        "finetune_review_encoding": "lossless_video_stream_remux" if encoding == ["-c:v", "copy"] else "h264_crf18_review_proxy",
    }
    for metric in METRICS:
        row[f"base_{metric}"] = float(base_score[metric])
        row[f"finetune_{metric}"] = float(fine_score[metric])
        row[f"delta_{metric}"] = round(float(fine_score[metric]) - float(base_score[metric]), 6)
    row["base_reason"] = base_score["reason"]
    row["finetune_reason"] = fine_score["reason"]
    return row, media


def comparison_template() -> str:
    template = (BASE / "scripts/scroll_review.html").read_text(encoding="utf-8")
    replacements = {
        "<title>Lance BASE · OmniEdit Spatial · 滚动审阅</title>": "<title>Lance LoRA step 677 vs Base · OmniEdit Spatial · 滚动对比</title>",
        "main{max-width:1480px": "main{max-width:1780px",
        ".videos{display:grid;grid-template-columns:1fr 1fr;gap:14px}": ".videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}",
        "<strong>Lance BASE · Spatial 240</strong>": "<strong>Lance LoRA step 677 vs Base · Spatial 240</strong>",
        "<h1>单轮视频编辑 · 基模完整审阅</h1>": "<h1>单轮视频编辑 · Base 与微调模型全量对比</h1>",
        "<p>240 条 · 24 类 × 10 · Lance 默认配置 · seed 42 · <strong>Gemini 3.8 Flash / low</strong> · Vertex v2</p>": "<p>240 条 · 24 类 × 10 · 同输入 / 推理配置 / seed 42 · <strong>Gemini 3.8 Flash / low</strong> · Vertex v2</p>",
        "<p class=\"muted\">直接滚轮向下看；左为官方输入的时间对齐审阅代理，右为 Lance 基模生成视频。数据集没有提供 target / GT。</p>": "<p class=\"muted\">直接滚轮向下看；每条依次为官方输入、Base 输出、LoRA step 677 输出。数据集没有提供 target / GT。</p>",
        "<p class=\"muted\">评分基于两段未经抽帧、重编码的完整原始 MP4，非网页代理。总分采用官方 accuracy 惩罚公式，不是四项均分。Gemini 评语保留原文。</p>": "<p class=\"muted\">Base 与 LoRA 分别用同一官方输入及各自完整原始输出评分；网页视频仅供审阅。两组四维分数、总分、完整评语和 LoRA−Base 差值均逐条显示。</p>",
        "<option value=\"low\">Gemini 总分：低 → 高</option><option value=\"high\">Gemini 总分：高 → 低</option>": "<option value=\"delta_low\">总分差值：下降最多</option><option value=\"delta_high\">总分差值：提升最多</option><option value=\"finetune_low\">LoRA 总分：低 → 高</option><option value=\"finetune_high\">LoRA 总分：高 → 低</option><option value=\"base_low\">Base 总分：低 → 高</option>",
        "for(const [key,label] of metrics)$(\"totals\").append(scoreBox(key,label+\" · 全集均值\",survey.summary.mean_scores_successes_only[key],4));": "for(const [key,label] of metrics){$(\"totals\").append(scoreBox(key,label+\" · Base\",survey.summary.base_mean_scores[key],4));$(\"totals\").append(scoreBox(key,label+\" · LoRA\",survey.summary.finetune_mean_scores[key],4));const d=survey.summary.mean_differences_finetune_minus_base[key],box=el(\"div\",\"score delta-score\"+(key===\"overall_score\"?\" overall\":\"\"));box.append(el(\"small\",\"\",label+\" · LoRA−Base\"),el(\"b\",\"\",d>0?\"+\"+Number(d).toFixed(4):Number(d).toFixed(4)));$(\"totals\").append(box);}",
        "const videos=rec.slots.map((slot,i)=>{const v=document.createElement(\"video\");v.controls=true;v.muted=true;v.playsInline=true;v.preload=\"metadata\";v.src=rec.item[i===0?\"review_source_path\":\"review_generated_path\"];v.setAttribute(\"aria-label\",i===0?\"官方输入审阅代理\":\"Lance 基模生成视频\");slot.replaceChildren(v);return v;});": "const paths=[rec.item.review_source_path,rec.item.review_base_path,rec.item.review_finetune_path],labels=[\"官方输入审阅代理\",\"Lance Base 生成视频\",\"Lance LoRA step 677 生成视频\"];const videos=rec.slots.map((slot,i)=>{const v=document.createElement(\"video\");v.controls=true;v.muted=true;v.playsInline=true;v.preload=\"metadata\";v.src=paths[i];v.setAttribute(\"aria-label\",labels[i]);slot.replaceChildren(v);return v;});",
        "const other=videos[video===videos[0]?1:0];\n      video.addEventListener(\"play\",()=>{activate(rec);pauseOthers(rec);if(other.paused)other.play().catch(()=>mediaError(rec,\"请点击同步播放；若仍失败，请完整解压后用 Chrome / Edge 打开。\"));});\n      video.addEventListener(\"pause\",()=>{if(!other.paused)other.pause();});\n      video.addEventListener(\"seeking\",()=>{if(other.readyState&&Math.abs(video.currentTime-other.currentTime)>.08)other.currentTime=video.currentTime;});\n      video.addEventListener(\"ratechange\",()=>{if(other.playbackRate!==video.playbackRate)other.playbackRate=video.playbackRate;});\n      video.addEventListener(\"timeupdate\",()=>{if(video===videos[0]&&!video.paused&&!video.seeking&&!other.seeking&&other.readyState&&Math.abs(video.currentTime-other.currentTime)>.15)other.currentTime=video.currentTime;});": "const others=videos.filter(v=>v!==video);\n      video.addEventListener(\"play\",()=>{activate(rec);pauseOthers(rec);for(const other of others)if(other.paused)other.play().catch(()=>mediaError(rec,\"请点击同步播放；若仍失败，请完整解压后用 Chrome / Edge 打开。\"));});\n      video.addEventListener(\"pause\",()=>{for(const other of others)if(!other.paused)other.pause();});\n      video.addEventListener(\"seeking\",()=>{for(const other of others)if(other.readyState&&Math.abs(video.currentTime-other.currentTime)>.08)other.currentTime=video.currentTime;});\n      video.addEventListener(\"ratechange\",()=>{for(const other of others)if(other.playbackRate!==video.playbackRate)other.playbackRate=video.playbackRate;});\n      video.addEventListener(\"timeupdate\",()=>{if(video===videos[0]&&!video.paused&&!video.seeking)for(const other of others)if(!other.seeking&&other.readyState&&Math.abs(video.currentTime-other.currentTime)>.15)other.currentTime=video.currentTime;});",
        "rec.message.textContent=\"两侧同步 · \"": "rec.message.textContent=\"三段同步 · \"",
        "for(const [label,sub] of [[\"SOURCE · 官方输入\",\"按 Lance 采样帧对齐的审阅代理\"],[\"LANCE BASE · GENERATED_VIDEO\",\"模型生成输出；不是 dataset target / GT\"]])": "for(const [label,sub] of [[\"SOURCE · 官方输入\",\"按 Lance 采样帧对齐的审阅代理\"],[\"LANCE BASE · GENERATED_VIDEO\",\"相同配置与 seed 42 的基模输出\"],[\"LANCE LoRA step 677 · GENERATED_VIDEO\",\"微调模型输出；不是 dataset target / GT\"]])",
        "node.append(el(\"strong\",\"\",\"Gemini 3.8 Flash（low）· 正式 Vertex v2 评分\"));\n    const scores=el(\"div\",\"scores\");for(const [key,label] of metrics)scores.append(scoreBox(key,label,item[key],key===\"overall_score\"?2:0));node.append(scores,el(\"div\",\"reason\",item.reason));": "node.append(el(\"strong\",\"\",\"Base · Gemini 3.8 Flash（low）· 正式 Vertex v2\"));const baseScores=el(\"div\",\"scores\");for(const [key,label] of metrics)baseScores.append(scoreBox(key,label,item[\"base_\"+key],key===\"overall_score\"?2:0));const baseReason=el(\"div\",\"reason base-reason\",item.base_reason);node.append(baseScores,baseReason);node.append(el(\"strong\",\"\",\"LoRA step 677 · 同一 judge 协议\"));const fineScores=el(\"div\",\"scores\");for(const [key,label] of metrics)fineScores.append(scoreBox(key,label,item[\"finetune_\"+key],key===\"overall_score\"?2:0));const fineReason=el(\"div\",\"reason finetune-reason\",item.finetune_reason);node.append(fineScores,fineReason);const delta=el(\"div\",\"scores delta-row\");for(const [key,label] of metrics){const value=item[\"delta_\"+key],box=el(\"div\",\"score delta-score\"+(key===\"overall_score\"?\" overall\":\"\"));box.append(el(\"small\",\"\",label+\" · LoRA−Base\"),el(\"b\",\"\",value>0?\"+\"+Number(value).toFixed(key===\"overall_score\"?2:0):Number(value).toFixed(key===\"overall_score\"?2:0)));delta.append(box);}node.append(delta);",
        "filtered.sort((a,b)=>sort===\"index\"?a.sample_index-b.sample_index:(sort===\"low\"?a.overall_score-b.overall_score:b.overall_score-a.overall_score)||a.sample_index-b.sample_index);": "const fields={delta_low:[\"delta_overall_score\",1],delta_high:[\"delta_overall_score\",-1],finetune_low:[\"finetune_overall_score\",1],finetune_high:[\"finetune_overall_score\",-1],base_low:[\"base_overall_score\",1]};filtered.sort((a,b)=>sort===\"index\"?a.sample_index-b.sample_index:((a[fields[sort][0]]-b[fields[sort][0]])*fields[sort][1]||a.sample_index-b.sample_index));",
        "Gemini 分数保持不变。": "Base/LoRA Gemini 分数保持不变。",
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise ValueError("comparison template source changed; expected one replacement: " + old[:80])
        template = template.replace(old, new)
    template = template.replace("OmniEdit Spatial · 全集审阅 · BASE · seed 42 · scored · google-vertex/global · official input → generated video", "OmniEdit Spatial · 完整配对 · Base vs LoRA step 677 · seed 42 · google-vertex/global")
    template = template.replace("@media(max-width:760px)", ".delta-score{border-color:#665e3e;background:#252216}.base-reason{border-left-color:#768498}.finetune-reason{border-left-color:#9a72d6}@media(max-width:980px){.videos{grid-template-columns:1fr}}@media(max-width:760px)")
    return template


def finish(output: Path, rows: list[dict], media: list[dict], summary: dict) -> None:
    payload = json.dumps({"review_version": output.name, "items": rows, "summary": summary}, ensure_ascii=False).replace("<", "\\u003c")
    document = comparison_template().replace("__SURVEY_DATA__", payload)
    if "__SURVEY_DATA__" in document:
        raise ValueError("survey data was not embedded")
    (output / "index.html").write_text(document, encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", document, re.S)
    if len(scripts) != 1:
        raise ValueError("unexpected executable script count")
    subprocess.run(["node", "--check"], input=scripts[0], text=True, check=True)
    embedded = json.loads(re.search(r'<script id="surveyData" type="application/json">(.*?)</script>', document, re.S)[1])
    if len(embedded["items"]) != 240:
        raise ValueError("embedded review is not the full 240")
    with (output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "media_inventory.json", media)
    write_json(output / "summary.json", summary)
    for source, target in (
        (BASE / "gemini38_low_vertex_v2/scores.csv", output / "base_scores.csv"),
        (BASE / "gemini38_low_vertex_v2/summary.json", output / "base_summary.json"),
        (BASE / "gemini38_low_vertex_v2/protocol.json", output / "judge_protocol.json"),
        (EVALUATION / "omni_scores_vertex_v2/scores.csv", output / "finetune_scores.csv"),
        (EVALUATION / "omni_scores_vertex_v2/summary.json", output / "finetune_summary.json"),
        (EVALUATION / "evaluation_report.json", output / "evaluation_report.json"),
        (EVALUATION / "output_integrity_audit.json", output / "output_integrity_audit.json"),
        (EVALUATION / "RESULTS.md", output / "RESULTS.md"),
    ):
        shutil.copyfile(source, target)
    write_json(output / "review_validation.json", {
        "status": "passed_pending_real_browser_test", "sample_count": 240, "media_count": 720,
        "complete_paired_benchmark": True, "inline_json_count_verified": True,
        "javascript_syntax_verified": True, "all_media_full_decode_verified": True,
        "all_three_videos_frame_count_and_fps_matched": True,
        "all_media_h264_yuv420p_faststart": True,
        "base_and_finetune_scores_bound_to_raw_judge_receipts": 240,
        "browser_test": "pending",
    })
    (output / "README.md").write_text("""# Lance LoRA step 677 vs Base · OmniEdit Spatial 全量滚动审核包

完整解压 ZIP 后，使用 Chrome 或 Edge 打开本目录 `index.html`。网页完全离线，直接滚轮浏览 240 条，不需要点击“下一条”。

每条从左到右为：OmniEdit 官方输入的时间对齐审阅代理、Lance 官方 Base 输出、LoRA step 677 输出。三段同步播放、暂停和拖动；空格控制当前可见条目。页面显示同一 Gemini 3.8 Flash low / Vertex v2 协议对两种模型给出的四维分数、官方总分、完整英文评语，以及 `LoRA − Base` 差值。

顶部支持 ID/指令搜索、类别筛选，并可按总分下降、提升或模型总分排序。人工结论初始全部未标注；1/0/2 对应接受/拒绝/不确定，备注及结论自动存储在当前浏览器，并可导入/导出 CSV、JSON。

OmniEdit Spatial 不提供 dataset target/GT；Base 和 LoRA 都是模型生成输出。Gemini 正式评分使用完整原始 source/generated MP4，本包只为浏览器审阅制作或复用 H.264/yuv420p/faststart 代理。媒体逐条完整解码，三段帧数和 FPS 对齐；原始及审核文件 SHA256 见 manifest 和 media inventory。

`base_scores.csv`、`finetune_scores.csv`、两份 summary、共同 judge protocol、总体结果报告及输出完整性审计一并收录。审核包不含模型权重、API 密钥或账户配置，打开网页不会调用网络或产生 API 费用。
""", encoding="utf-8")


def package(output: Path) -> None:
    rows = read_csv(output / "review_manifest.csv")
    browser = json.loads((output / "browser_validation.json").read_text())
    validation = json.loads((output / "review_validation.json").read_text())
    if len(rows) != 240 or browser.get("status") != "passed" or browser.get("sample_count") != 240:
        raise ValueError("real-browser acceptance of all 240 entries is required")
    if browser.get("javascript_errors") or browser.get("external_requests"):
        raise ValueError("browser validation reported errors or external requests")
    inventory = json.loads((output / "media_inventory.json").read_text())
    if len(inventory) != 720 or Counter(item["role"] for item in inventory) != {"source": 240, "base": 240, "finetune": 240}:
        raise ValueError("media inventory must contain three videos per sample")
    for item in inventory:
        if digest(output / item["path"]) != item["sha256"]:
            raise ValueError("review media changed after validation")
    for row in rows:
        for key in ("review_source_path", "review_base_path", "review_finetune_path"):
            path = Path(row[key])
            if path.parts[0] != "media" or not (output / path).is_file():
                raise ValueError("invalid portable review media path")
    validation.update(status="passed", browser_test="passed; see browser_validation.json",
                      archive_score_media_binding_reverified=True)
    write_json(output / "review_validation.json", validation)
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or {name.split("/")[0] for name in bundle.namelist()} != {output.name}:
            raise ValueError("ZIP CRC or single-root layout failed")
        if sum(name.endswith(".mp4") for name in bundle.namelist()) != 720:
            raise ValueError("ZIP media count mismatch")
    receipt = {"local_file": str(archive.resolve()), "filename": archive.name,
               "bytes": archive.stat().st_size, "sha256": digest(archive),
               "zip_crc_verified": True, "package_root": output.name,
               "samples": 240, "videos": 720,
               "remote_path": "/apps/bypy/lance_review/" + archive.name,
               "upload_status": "not_uploaded"}
    write_json(archive.with_suffix(".sha256.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def repair_html(output: Path) -> None:
    """Apply the two exact renderer fixes found by real-browser acceptance."""
    path = output / "index.html"
    document = path.read_text(encoding="utf-8")
    replacements = {
        'box=el("div","score delta-score");box.append(el("small","",label+" · LoRA−Base"),el("b",d>0?':
            'box=el("div","score delta-score"+(key==="overall_score"?" overall":""));box.append(el("small","",label+" · LoRA−Base"),el("b","",d>0?',
        'box=el("div","score delta-score");box.append(el("small","",label+" · LoRA−Base"),el("b",value>0?':
            'box=el("div","score delta-score"+(key==="overall_score"?" overall":""));box.append(el("small","",label+" · LoRA−Base"),el("b","",value>0?',
    }
    for old, new in replacements.items():
        if document.count(old) != 1:
            raise ValueError("expected exactly one known pre-fix renderer fragment")
        document = document.replace(old, new)
    scripts = re.findall(r"<script>(.*?)</script>", document, re.S)
    subprocess.run(["node", "--check"], input=scripts[0], text=True, check=True)
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"repaired_html": str(path)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVALUATION / "review" / VERSION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="smoke build only; cannot be packaged")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--repair-html", action="store_true")
    args = parser.parse_args()
    if args.repair_html:
        repair_html(args.output.resolve()); return
    if args.package_only:
        package(args.output.resolve()); return
    pairs, summary = load_inputs()
    if args.limit:
        if args.limit < 1 or args.output.name == VERSION:
            raise ValueError("smoke limit requires a non-production output path")
        pairs = pairs[:args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    rows, media = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, entries in pool.map(lambda pair: prepare_pair(pair, output), pairs):
            rows.append(row); media.extend(entries)
            if len(rows) % 20 == 0 or len(rows) == len(pairs):
                print(f"prepared + verified {len(rows)}/{len(pairs)} samples", flush=True)
    if len(pairs) != 240:
        print(json.dumps({"smoke_samples": len(pairs), "output": str(output)})); return
    finish(output, rows, media, summary)
    print(json.dumps({"ready_for_browser_validation": str(output), "samples": 240, "videos": 720}))


if __name__ == "__main__":
    main()
